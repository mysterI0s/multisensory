"""
Source Separation U-Net.

Port of sourcesep.py — a U-Net operating on spectrograms, conditioned on
ShiftNet video features at multiple scales.

The encoder uses 9 conv layers (LeakyReLU + BN), and the decoder uses 8
transposed conv layers with skip connections. Two output heads produce
foreground and background separated spectrograms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from ..utils.audio import (
    stft, istft, normalize_spec, unnormalize_spec,
    normalize_phase, unnormalize_phase, db_from_amp
)


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

def _unet_bn_params():
    """BN parameters matching the TF unet_arg_scope (decay=0.9997, eps=1e-5)."""
    return dict(momentum=0.0003, eps=1e-5)


class UNetEncoder(nn.Module):
    """
    Encoder with 9 convolutional layers.

    Each layer: Conv2d → BN → LeakyReLU(0.2)
    Strides: conv1-2: [1,2]; conv3-9: 2.
    Activations are stored for skip connections. Video features are concatenated
    before conv4, conv5, and conv6 if net_style != 'no-im'.
    """

    def __init__(self, net_style='full'):
        super().__init__()
        self.net_style = net_style
        bn = _unet_bn_params()

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        v = (net_style != 'no-im')
        channels = [
            (2,   64,  4, (1, 2)),   # conv1
            (64,  128, 4, (1, 2)),   # conv2
            (128, 256, 4, 2),        # conv3
            (320 if v else 256, 512, 4, 2),  # conv4 (vid scale 0 added before this: +64)
            (640 if v else 512, 512, 4, 2),  # conv5 (vid scale 1 added before this: +128)
            (1024 if v else 512, 512, 4, 2), # conv6 (vid scale 2 added before this: +512)
            (512, 512, 4, 2),        # conv7
            (512, 512, 4, 2),        # conv8
            (512, 512, 4, 2),        # conv9
        ]

        for i, (c_in, c_out, k, s) in enumerate(channels):
            if isinstance(s, tuple):
                stride = s
            else:
                stride = (s, s)
            self.layers.append(
                nn.Conv2d(c_in, c_out, k, stride=stride, padding=0, bias=False)
            )
            self.bns.append(nn.BatchNorm2d(c_out, **bn))

        for layer in self.layers:
            nn.init.normal_(layer.weight, 0.0, 0.02)
        for bn_layer in self.bns:
            nn.init.normal_(bn_layer.weight, 1.0, 0.02)
            nn.init.zeros_(bn_layer.bias)

    def forward(self, x):
        activations = []
        strides_list = [
            (1, 2), (1, 2), (2, 2), (2, 2), (2, 2),
            (2, 2), (2, 2), (2, 2), (2, 2)
        ]

        for i, (conv, bn) in enumerate(zip(self.layers, self.bns)):
            stride = strides_list[i]
            k = 4
            ph = max(k - 1, 0)
            pw = max(k - 1, 0)
            pad_top = ph // 2
            pad_bottom = ph - pad_top
            pad_left = pw // 2
            pad_right = pw - pad_left
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

            x = conv(x)
            x = bn(x)
            activations.append(x)
            x = F.leaky_relu(x, 0.2)
        return x, activations


class UNetDecoder(nn.Module):
    """
    Decoder with 8 transposed convolutional layers.

    Each layer: ReLU(concat(x, skip)) → ConvTranspose2d → BN
    """

    def __init__(self, net_style='full'):
        super().__init__()
        bn = _unet_bn_params()

        self.deconvs = nn.ModuleList()
        self.bns = nn.ModuleList()

        v = (net_style != 'no-im')
        decoder_spec = [
            (512,       512, 4, 2),     # deconv1 (no skip)
            (512 + 512, 512, 4, 2),     # deconv2 (skip: conv8_out)
            (512 + 512, 512, 4, 2),     # deconv3 (skip: conv7_out)
            (512 + 512, 512, 4, 2),     # deconv4 (skip: conv6_out)
            (1536 if v else 1024, 512, 4, 2),     # deconv5 (skip: enc5_merged = 512 + 512)
            (1152 if v else 768, 256, 4, 2),      # deconv6 (skip: enc4_merged = 512 + 128)
            (576 if v else 384, 128, 4, (1, 2)),  # deconv7 (skip: enc3_merged = 256 + 64)
            (256 if v else 192,  64,  4, (1, 2)),  # deconv8 (skip: conv2_out = 128)
        ]

        for i, (c_in, c_out, k, s) in enumerate(decoder_spec):
            if isinstance(s, tuple):
                stride = s
            else:
                stride = (s, s)
            self.deconvs.append(
                nn.ConvTranspose2d(c_in, c_out, k, stride=stride, padding=0,
                                   bias=False)
            )
            self.bns.append(nn.BatchNorm2d(c_out, **bn))

        for layer in self.deconvs:
            nn.init.normal_(layer.weight, 0.0, 0.02)
        for bn_layer in self.bns:
            nn.init.normal_(bn_layer.weight, 1.0, 0.02)
            nn.init.zeros_(bn_layer.bias)

    def forward(self, x, activations):
        skips = list(activations[:-1])
        skips.reverse()  # conv8, conv7, conv6, conv5, conv4, conv3, conv2, conv1

        strides_list = [
            (2, 2), (2, 2), (2, 2), (2, 2), (2, 2),
            (2, 2), (1, 2), (1, 2)
        ]

        for i, (deconv, bn) in enumerate(zip(self.deconvs, self.bns)):
            if i > 0:
                skip = skips[i - 1]
                x = torch.cat([x, skip], dim=1)

            x = F.relu(x)

            x = deconv(x)
            stride = strides_list[i]
            k = 4
            crop_h = k - stride[0]
            crop_w = k - stride[1]
            if crop_h > 0:
                ch = crop_h // 2
                x = x[:, :, ch:x.shape[2] - (crop_h - ch), :]
            if crop_w > 0:
                cw = crop_w // 2
                x = x[:, :, :, cw:x.shape[3] - (crop_w - cw)]
            x = bn(x)

        last_skip = skips[-1]
        return x, last_skip


class OutputHead(nn.Module):
    """
    Output head: ConvTranspose2d producing 2-channel output (spec + phase).
    No normalization, no activation.
    """

    def __init__(self, in_channels, stride=(1, 2)):
        super().__init__()
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        self.deconv = nn.ConvTranspose2d(in_channels, 2, 4, stride=stride,
                                          padding=0, bias=True)
        nn.init.normal_(self.deconv.weight, 0.0, 0.02)
        nn.init.zeros_(self.deconv.bias)

    def forward(self, x, skip):
        x = F.relu(torch.cat([x, skip], dim=1))
        x = self.deconv(x)
        k = 4
        crop_h = k - self.stride[0]
        crop_w = k - self.stride[1]
        if crop_h > 0:
            ch = crop_h // 2
            x = x[:, :, ch:x.shape[2] - (crop_h - ch), :]
        if crop_w > 0:
            cw = crop_w // 2
            x = x[:, :, :, cw:x.shape[3] - (crop_w - cw)]
        return x


# ---------------------------------------------------------------------------
# Video-conditioned merge at different U-Net levels
# ---------------------------------------------------------------------------

class VideoConditioner(nn.Module):
    """
    Condition the U-Net encoder on video features from ShiftNet.

    At levels 3, 4, 5 of the encoder, concatenate video features
    (Global Average Pooled → resized → tiled).
    """

    def __init__(self):
        super().__init__()

    def merge(self, net, vid_feat, enc_activation):
        """
        Merge video features into encoder activations.

        Args:
            net: current encoder feature (B, C, T, F)
            vid_feat: ShiftNet scale feature (B, C_vid, D, H, W)
            enc_activation: list of activations (to be modified in-place)

        Returns:
            Updated net with video features concatenated
        """
        if vid_feat is None:
            return net

        # Global average pool spatial dims: (B, C_vid, D, H, W) → (B, C_vid, D, 1)
        vid = vid_feat.mean(dim=[3, 4])  # (B, C_vid, D)
        vid = vid.unsqueeze(3)  # (B, C_vid, D, 1)

        # Resize temporal dimension to match net
        if vid.shape[2] != net.shape[2]:
            vid = F.interpolate(vid, size=(net.shape[2], 1),
                                mode='bilinear', align_corners=False)

        # Tile across frequency dimension
        vid = vid.expand(-1, -1, -1, net.shape[3])  # (B, C_vid, T, F)

        # Concatenate
        net = torch.cat([net, vid], dim=1)
        return net


# ---------------------------------------------------------------------------
# SourceSep U-Net: Full model
# ---------------------------------------------------------------------------

class SourceSepUNet(nn.Module):
    """
    U-Net for audio-visual source separation.

    Processes spectrograms (magnitude + phase), optionally conditioned on
    video features from a pretrained ShiftNet.

    Input:  mixed audio spectrogram + video frames
    Output: separated foreground and background audio (spec + waveform)
    """

    def __init__(self, pr, shift_net=None, net_style='full'):
        """
        Args:
            pr: Params object
            shift_net: optional pretrained ShiftNet module for video conditioning
            net_style: 'full' (video conditioned), 'no-im' (audio only),
                       'static' (static video frame)
        """
        super().__init__()
        self.pr = pr
        self.shift_net = shift_net
        self.net_style = net_style

        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.conditioner = VideoConditioner() if net_style != 'no-im' else None

        # Output heads: fg and bg
        self.fg_head = OutputHead(64 + 64, stride=(1, 2))  # 64 decoder + 64 skip
        self.bg_head = OutputHead(64 + 64, stride=(1, 2))

    def _get_video_features(self, ims, samples_trunc):
        """
        Get multi-scale video features from ShiftNet.

        Args:
            ims: (B, C, D, H, W) video frames
            samples_trunc: (B, T, 2) audio samples

        Returns:
            scales: list of feature maps at different levels, or None
        """
        if self.net_style == 'no-im' or self.shift_net is None:
            return None

        with torch.no_grad():
            if self.net_style == 'static':
                B, C, D, H, W = ims.shape
                mid = D // 2
                ims_static = ims[:, :, mid:mid + 1].expand(-1, -1, D, -1, -1)
                _, _, _, _, scales, _ = self.shift_net(ims_static, samples_trunc)
            else:
                _, _, _, _, scales, _ = self.shift_net(ims, samples_trunc)

        return scales

    def _process_output(self, out, phase, num_freq):
        """
        Process raw network output into spectrogram and waveform.

        Args:
            out: (B, 2, T, F) network output (channel 0 = spec, channel 1 = phase)
            phase: (B, T, F) original phase
            num_freq: original frequency dimension

        Returns:
            pred_spec, pred_phase, pred_wav
        """
        pr = self.pr

        # Separate magnitude and phase predictions
        pred_spec = torch.tanh(out[:, 0])  # (B, T, F')
        pred_spec = unnormalize_spec(pred_spec, pr)

        pred_phase = torch.tanh(out[:, 1])
        pred_phase = unnormalize_phase(pred_phase, pr)

        # Pad frequency dimension back to original size
        freq_diff = num_freq - pred_spec.shape[-1]
        if freq_diff > 0:
            val = db_from_amp(torch.tensor(0.0)).item() if pr.log_spec else 0.0
            pred_spec = F.pad(pred_spec, (0, freq_diff), value=val)

        # Handle phase
        if pr.phase_type == 'pred':
            pred_phase = torch.cat([pred_phase, phase[..., -1:]], dim=-1)
        elif pr.phase_type == 'orig':
            pred_phase = phase
        else:
            raise RuntimeError(f"Unknown phase_type: {pr.phase_type}")

        # iSTFT
        pred_wav = istft(pred_spec, pred_phase, pr)

        return pred_spec, pred_phase, pred_wav

    def forward(self, ims, samples_trunc, spec_mix, phase_mix):
        """
        Forward pass.

        Args:
            ims:            (B, C=3, D, H, W) video frames (or None for no-im)
            samples_trunc:  (B, T, 2) truncated audio samples (for ShiftNet)
            spec_mix:       (B, T_spec, F) mixed spectrogram magnitude
            phase_mix:      (B, T_spec, F) mixed spectrogram phase

        Returns:
            pred_spec_fg, pred_wav_fg, pred_phase_fg,
            pred_spec_bg, pred_wav_bg, pred_phase_bg,
            vid_net (scales or None)
        """
        pr = self.pr
        num_freq = spec_mix.shape[-1]

        # Get video features
        vid_scales = self._get_video_features(ims, samples_trunc)

        # Prepare U-Net input: (B, 2, T, F)
        spec_norm = normalize_spec(spec_mix, pr).unsqueeze(1)    # (B, 1, T, F)
        phase_norm = normalize_phase(phase_mix, pr).unsqueeze(1)  # (B, 1, T, F)
        unet_input = torch.cat([spec_norm, phase_norm], dim=1)    # (B, 2, T, F)

        # Truncate frequency dimension
        unet_input = unet_input[:, :, :, :pr.freq_len]

        # Encode
        encoded, activations = self.encoder(unet_input)

        # Merge video features at levels 3, 4, 5 (after conv3, conv4, conv5)
        if vid_scales is not None and self.conditioner is not None:
            for level_idx, enc_idx in enumerate([3, 4, 5]):
                if level_idx < len(vid_scales):
                    activations[enc_idx] = self.conditioner.merge(
                        activations[enc_idx], vid_scales[level_idx], activations
                    )

        # Decode
        decoded, last_skip = self.decoder(encoded, activations)

        # Output heads
        out_fg = self.fg_head(decoded, last_skip)
        out_bg = self.bg_head(decoded, last_skip)

        # Process outputs
        pred_spec_fg, pred_phase_fg, pred_wav_fg = self._process_output(
            out_fg, phase_mix, num_freq
        )
        pred_spec_bg, pred_phase_bg, pred_wav_bg = self._process_output(
            out_bg, phase_mix, num_freq
        )

        return (
            pred_spec_fg, pred_wav_fg, pred_phase_fg,
            pred_spec_bg, pred_wav_bg, pred_phase_bg,
            vid_scales,
        )


# ---------------------------------------------------------------------------
# NetClf: Inference wrapper (replaces sourcesep.NetClf)
# ---------------------------------------------------------------------------

class SourceSepClassifier:
    """
    Inference wrapper for source separation.

    Usage:
        clf = SourceSepClassifier(pr, weights_path, device='cuda:0')
        result = clf.predict(ims, samples)
    """

    def __init__(self, pr, weights_path, shift_weights_path=None,
                 device='cpu', restore_only_shift=False):
        self.pr = pr
        self.device = torch.device(device)

        # Build ShiftNet if needed
        shift_net = None
        if pr.net_style != 'no-im' and shift_weights_path is not None:
            from .shift_net import ShiftNet
            shift_net = ShiftNet(pr).to(self.device)
            shift_state = torch.load(shift_weights_path, map_location=self.device,
                                     weights_only=True)
            if "model_state_dict" in shift_state:
                shift_state = shift_state["model_state_dict"]
            shift_net.load_state_dict(shift_state)
            shift_net.eval()

        # Build SourceSep
        self.model = SourceSepUNet(pr, shift_net=shift_net,
                                    net_style=pr.net_style).to(self.device)
        self.model.eval()

        if weights_path.endswith(".pt"):
            state_dict = torch.load(weights_path, map_location=self.device,
                                    weights_only=True)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict, strict=False)

        print(f"SourceSep loaded from {weights_path}")

    @torch.no_grad()
    def predict(self, ims, samples):
        """
        Run source separation.

        Args:
            ims:     (1, T, H, W, 3) uint8 numpy array
            samples: (1, N, 2) float32 numpy array

        Returns:
            dict with separated audio and spectrograms
        """
        import numpy as np
        pr = self.pr

        # Convert: NDHWC → NCDHW
        ims_t = torch.from_numpy(ims).permute(0, 4, 1, 2, 3).float().to(self.device)
        samples_t = torch.from_numpy(samples).to(self.device)

        # Truncate samples
        sample_len = getattr(pr, 'sample_len', None) or pr.num_samples
        samples_trunc = samples_t[:, :sample_len]

        # Compute STFT
        from ..utils.audio import stft as audio_stft
        spec_mix, phase_mix = audio_stft(samples_trunc[:, :, 0], pr)
        spec_mix = spec_mix[:, :pr.spec_len]
        phase_mix = phase_mix[:, :pr.spec_len]

        # Also compute spectrogram of original mix for return
        spec_mix_ret = spec_mix.clone()

        # Forward
        results = self.model(ims_t, samples_trunc, spec_mix, phase_mix)
        pred_spec_fg, pred_wav_fg, _, pred_spec_bg, pred_wav_bg, _, _ = results

        return dict(
            samples_pred_fg=pred_wav_fg.cpu().numpy(),
            samples_pred_bg=pred_wav_bg.cpu().numpy(),
            spec_pred_fg=pred_spec_fg.cpu().numpy(),
            spec_pred_bg=pred_spec_bg.cpu().numpy(),
            samples_mix=samples,
            spec_mix=spec_mix_ret.cpu().numpy(),
        )
