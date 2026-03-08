"""
ShiftNet: Audio-Visual Correspondence Network.

Port of shift_net.py — a two-stream 3D CNN (ResNet-18 variant) with a 2D
sound feature sub-network, merged via audio-visual fusion.

Original reference:
    Owens & Efros, "Audio-Visual Scene Analysis with Self-Supervised
    Multisensory Features", 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .blocks import Block2D, Block3D, Conv3dSame, Conv2dSame, _BN_MOMENTUM, _BN_EPS


# ---------------------------------------------------------------------------
# Helper: sound feature normalization (from shift_net.normalize_sfs)
# ---------------------------------------------------------------------------
def _normalize_sfs(sfs, scale=255.0):
    """Compress dynamic range: sign(x) * log(1 + scale*|x|) / log(1 + scale)."""
    return torch.sign(sfs) * (torch.log1p(scale * torch.abs(sfs)) / math.log(1.0 + scale))


def _normalize_ims(im):
    """Normalize uint8 images to [-1, 1]."""
    return -1.0 + (2.0 / 255.0) * im.float()


# ---------------------------------------------------------------------------
# Sound Feature Sub-Network (2D convolutions)
# ---------------------------------------------------------------------------
class SoundFeatureNet(nn.Module):
    """
    2D convolutional network for raw audio waveform features.

    Architecture:
        normalize → transpose → conv1(65×1, s=4) → pool(4×1) →
        block2(128, 15×1, s=4×1) → block3(128, 15×1, s=4×1) →
        block4(256, 15×1, s=4×1)
    """

    def __init__(self, bn_momentum=_BN_MOMENTUM, bn_eps=_BN_EPS):
        super().__init__()
        # Input: (batch, channels=2, time, 1) after transpose/unsqueeze
        self.conv1 = Conv2dSame(2, 64, (65, 1), stride=(4, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum, eps=bn_eps)
        self.pool1 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.block2 = Block2D(64, 128, (15, 1), stride=(4, 1),
                              bn_momentum=bn_momentum, bn_eps=bn_eps)
        self.block3 = Block2D(128, 128, (15, 1), stride=(4, 1),
                              bn_momentum=bn_momentum, bn_eps=bn_eps)
        self.block4 = Block2D(128, 256, (15, 1), stride=(4, 1),
                              bn_momentum=bn_momentum, bn_eps=bn_eps)

    def forward(self, sfs):
        """
        Args:
            sfs: (batch, time, channels) raw audio samples (stereo: channels=2)

        Returns:
            (batch, 256, time', 1) feature map
        """
        x = _normalize_sfs(sfs)
        # Reshape to (batch, channels, time, 1) for 2D conv
        x = x.transpose(1, 2)  # (B, 2, T)
        x = x.unsqueeze(3)     # (B, 2, T, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x  # (B, 256, T', 1)


# ---------------------------------------------------------------------------
# Image Feature Sub-Network (3D convolutions)
# ---------------------------------------------------------------------------
class ImageFeatureNet(nn.Module):
    """
    3D convolutional network for video frames (ResNet-18 variant).

    Architecture:
        normalize → conv1(5×7×7, s=2) → pool(1×3×3, s=1×2×2) →
        block2_1(64, 3³, s=1) → block2_2(64, 3³, s=2)
    """

    def __init__(self, bn_momentum=_BN_MOMENTUM, bn_eps=_BN_EPS):
        super().__init__()
        self.conv1 = Conv3dSame(3, 64, (5, 7, 7), stride=2, bias=False)
        self.bn1 = nn.BatchNorm3d(64, momentum=bn_momentum, eps=bn_eps)
        self.pool1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.block2_1 = Block3D(64, 64, (3, 3, 3), stride=1,
                                bn_momentum=bn_momentum, bn_eps=bn_eps)
        self.block2_2 = Block3D(64, 64, (3, 3, 3), stride=2,
                                bn_momentum=bn_momentum, bn_eps=bn_eps)

    def forward(self, ims):
        """
        Args:
            ims: (batch, channels, depth, height, width) or
                 (batch, depth, height, width, channels) uint8 video frames

        Returns:
            (batch, 64, D', H', W') feature maps
        """
        # Expect NCDHW input
        x = _normalize_ims(ims)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        return x


# ---------------------------------------------------------------------------
# Merge Module: fuse image and sound features
# ---------------------------------------------------------------------------
class MergeModule(nn.Module):
    """
    Fuse image and sound feature networks.

    Uses fractional max pooling to align temporal dimensions of the sound
    features to match the image features, followed by 1×1×1 convolutions
    with a residual connection.
    """

    def __init__(self, sf_channels=256, im_channels=64, use_sound=True,
                 bn_momentum=_BN_MOMENTUM, bn_eps=_BN_EPS):
        super().__init__()
        self.use_sound = use_sound

        # 2D conv to reduce sound features before merge
        self.sf_conv5 = Conv2dSame(sf_channels, 128, (3, 1), bias=False)
        self.sf_bn5 = nn.BatchNorm2d(128, momentum=bn_momentum, eps=bn_eps)

        # Merge convolutions (3D)
        merged_channels = im_channels + 128  # image + sound
        self.merge_conv1 = nn.Conv3d(merged_channels, 512, 1, bias=False)
        self.merge_bn1 = nn.BatchNorm3d(512, momentum=bn_momentum, eps=bn_eps)
        self.merge_conv2 = nn.Conv3d(512, 128, 1, bias=True)
        # Shortcut: take first 64 and last 64 channels
        self.merge_bn_out = nn.BatchNorm3d(128, momentum=bn_momentum, eps=bn_eps)

    def forward(self, sf_net, im_net, train=True):
        """
        Args:
            sf_net: (B, 256, T_sf, C_sf) sound features
            im_net: (B, 64, D, H, W) image features
            train: whether in training mode (affects fractional pooling)

        Returns:
            (B, 128, D, H, W) merged features
        """
        # Fractional max pool to align temporal dimension
        target_time = im_net.shape[2]
        ratio = float(sf_net.shape[2] - 1) / target_time

        if train:
            sf_net = F.fractional_max_pool2d(
                sf_net, (ratio, 1.0),
                output_size=(target_time, sf_net.shape[3])
            )[0]
        else:
            sf_net = F.adaptive_max_pool2d(
                sf_net, (target_time, sf_net.shape[3])
            )

        # 2D conv on sound features
        sf_w = sf_net.shape[3]
        sf_net = F.relu(self.sf_bn5(self.sf_conv5(sf_net)))

        # Take middle column and expand spatially
        sf_net = sf_net[:, :, :, sf_net.shape[3] // 2]  # (B, 128, T)
        sf_net = sf_net.unsqueeze(3).unsqueeze(4)  # (B, 128, T, 1, 1)
        sf_net = sf_net.expand(-1, -1, -1, im_net.shape[3], im_net.shape[4])

        if not self.use_sound:
            sf_net = torch.zeros_like(sf_net)

        # Concatenate image and sound features
        net = torch.cat([im_net, sf_net], dim=1)  # (B, 64+128, D, H, W)

        # Residual merge
        shortcut = torch.cat([net[:, :64], net[:, -64:]], dim=1)  # (B, 128, D, H, W)
        out = F.relu(self.merge_bn1(self.merge_conv1(net)))
        out = self.merge_conv2(out)
        out = F.relu(self.merge_bn_out(out + shortcut))
        return out


# ---------------------------------------------------------------------------
# ShiftNet: Full model
# ---------------------------------------------------------------------------
class ShiftNet(nn.Module):
    """
    Audio-Visual Correspondence Network (ShiftNet).

    Two-stream architecture:
        - Image stream: 3D ResNet-18 variant
        - Sound stream: 2D CNN on raw waveform
        - Merge: audio-visual fusion
        - Head: binary classification (synced vs shifted)

    Outputs:
        logits:    (B, 1) classification logits
        cam:       (B, 1, D, H, W) class activation map
        last_conv: (B, 512, D', H', W') last conv features
        im_net:    (B, 64, D'', H'', W'') image features (for reuse)
        scales:    list of intermediate feature maps
        im_scales: list of image-stream feature maps
    """

    def __init__(self, pr=None, use_sound=True, cam_mode=False):
        super().__init__()
        self.use_sound = use_sound
        self.cam_mode = cam_mode

        # Sub-networks
        self.sound_net = SoundFeatureNet()
        self.image_net = ImageFeatureNet()
        self.merge = MergeModule(use_sound=use_sound)

        # Post-merge blocks
        self.block3_1 = Block3D(128, 128, (3, 3, 3), stride=1)
        self.block3_2 = Block3D(128, 128, (3, 3, 3), stride=1)

        self.block4_1 = Block3D(128, 256, (3, 3, 3), stride=(2, 2, 2))
        self.block4_2 = Block3D(256, 256, (3, 3, 3), stride=1)

        s = 1 if cam_mode else 2
        self.block5_1 = Block3D(256, 512, (3, 3, 3), stride=(1, s, s))
        self.block5_2 = Block3D(512, 512, (3, 3, 3), stride=1)

        # Classification head
        self.logits_conv = nn.Conv3d(512, 1, 1, bias=True)

    def forward(self, ims, sfs, im_net=None):
        """
        Args:
            ims:    (B, C=3, D, H, W) video frames (NCDHW), or None if reusing im_net
            sfs:    (B, T, 2) raw audio samples
            im_net: optional precomputed image features for reuse

        Returns:
            logits, cam, last_conv, im_net, scales, im_scales
        """
        scales = []
        im_scales = []

        # Sound features
        sf_feat = self.sound_net(sfs)

        # Image features
        if im_net is None:
            im_feat = self.image_net(ims)
        else:
            im_feat = im_net

        scales.append(im_feat)
        im_scales.append(im_feat)

        # Merge
        net = self.merge(sf_feat, im_feat, train=self.training)

        # Remaining blocks
        net = self.block3_1(net)
        net = self.block3_2(net)
        scales.append(net)
        im_scales.append(net)

        net = self.block4_1(net)
        net = self.block4_2(net)
        im_scales.append(net)

        net = self.block5_1(net)
        net = self.block5_2(net)
        scales.append(net)
        im_scales.append(net)

        last_conv = net

        # Global average pooling → logits
        pooled = net.mean(dim=[2, 3, 4], keepdim=True)
        logits = self.logits_conv(pooled).squeeze(4).squeeze(3).squeeze(2)

        # Class activation map (reuse logits conv on full spatial features)
        cam = self.logits_conv(last_conv)

        return logits, cam, last_conv, im_feat, scales, im_scales


# ---------------------------------------------------------------------------
# NetClf: Inference wrapper (replaces shift_net.NetClf)
# ---------------------------------------------------------------------------
class ShiftNetClassifier:
    """
    Inference wrapper for ShiftNet — replaces the TF session-based NetClf.

    Usage:
        clf = ShiftNetClassifier(pr, weights_path, device='cuda:0')
        logits, cam = clf.predict(ims, samples)
    """

    def __init__(self, pr, weights_path, device='cpu'):
        self.pr = pr
        self.device = torch.device(device)
        self.model = ShiftNet(pr, cam_mode=pr.cam).to(self.device)
        self.model.eval()

        # Load weights
        if weights_path.endswith(".pt"):
            state_dict = torch.load(weights_path, map_location=self.device,
                                    weights_only=True)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict)
        print(f"ShiftNet loaded from {weights_path}")

    @torch.no_grad()
    def predict_cam(self, ims, samples):
        """
        Predict class activation map.

        Args:
            ims:     (1, T, H, W, 3) uint8 numpy array
            samples: (1, N, 2) float32 numpy array

        Returns:
            cam: numpy array
        """
        # Convert NDHWC → NCDHW
        ims_t = torch.from_numpy(ims).permute(0, 4, 1, 2, 3).to(self.device)
        sfs_t = torch.from_numpy(samples).to(self.device)

        _, cam, _, _, _, _ = self.model(ims_t, sfs_t)
        return cam.cpu().numpy()

    @torch.no_grad()
    def predict_cam_resize(self, ims, samples):
        """Predict CAM with input image resizing to crop_im_dim."""
        pr = self.pr
        ims_t = torch.from_numpy(ims).float().permute(0, 4, 1, 2, 3).to(self.device)
        # Resize each frame
        B, C, D, H, W = ims_t.shape
        ims_flat = ims_t.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        ims_resized = F.interpolate(ims_flat, size=(pr.crop_im_dim, pr.crop_im_dim),
                                     mode='bilinear', align_corners=False)
        ims_resized = ims_resized.reshape(B, D, C, pr.crop_im_dim, pr.crop_im_dim)
        ims_resized = ims_resized.permute(0, 2, 1, 3, 4)

        sfs_t = torch.from_numpy(samples).to(self.device)
        _, cam, _, _, _, _ = self.model(ims_resized, sfs_t)
        return cam.cpu().numpy()
