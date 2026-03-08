"""
Audio processing utilities for PyTorch.

Port of soundrep.py and related STFT/iSTFT functions from the TF codebase.
Replaces: tf.signal.stft, tf.signal.inverse_stft, soundrep.griffin_lim,
          soundrep.db_from_amp, soundrep.amp_from_db, tfutil.normalize_rms.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


# ---------------------------------------------------------------------------
# STFT parameter helpers (from soundrep.py)
# ---------------------------------------------------------------------------

def stft_frame_length(pr):
    """Frame length in samples for STFT."""
    return int(pr.frame_length_ms * pr.samp_sr * 0.001)


def stft_frame_step(pr):
    """Hop length in samples for STFT."""
    return int(pr.frame_step_ms * pr.samp_sr * 0.001)


def stft_num_fft(pr):
    """FFT size (next power of 2 >= frame_length)."""
    return int(2 ** np.ceil(np.log2(stft_frame_length(pr))))


# ---------------------------------------------------------------------------
# STFT / iSTFT  (replaces tf.signal.stft / tf.signal.inverse_stft)
# ---------------------------------------------------------------------------

def stft(samples, pr):
    """
    Compute STFT magnitude and phase.

    Args:
        samples: (batch, time) real tensor
        pr: Params with frame_length_ms, frame_step_ms, samp_sr, log_spec, pad_stft

    Returns:
        mag:   (batch, time_frames, freq_bins) — dB if pr.log_spec else linear
        phase: (batch, time_frames, freq_bins)
    """
    n_fft = stft_num_fft(pr)
    hop = stft_frame_step(pr)
    win_len = stft_frame_length(pr)
    window = torch.hann_window(win_len, device=samples.device, dtype=samples.dtype)

    # torch.stft returns (batch, freq, frames, 2) with return_complex=False
    # or (batch, freq, frames) with return_complex=True
    spec = torch.stft(
        samples,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_len,
        window=window,
        center=False,          # TF pad_end=False is similar to center=False
        pad_mode="constant",
        return_complex=True,
    )
    # spec shape: (batch, freq, frames) → transpose to (batch, frames, freq) like TF
    spec = spec.transpose(-1, -2)  # (batch, frames, freq)

    mag = spec.abs()
    phase = spec.angle()

    if pr.log_spec:
        mag = db_from_amp(mag)

    return mag, phase


def make_complex(mag, phase):
    """Rebuild complex spectrogram from magnitude and phase."""
    return mag * torch.complex(torch.cos(phase), torch.sin(phase))


def istft(mag, phase, pr):
    """
    Inverse STFT: magnitude + phase → waveform.

    Args:
        mag:   (batch, frames, freq) — dB if pr.log_spec else linear
        phase: (batch, frames, freq)
        pr:    Params

    Returns:
        samples: (batch, time)
    """
    if pr.log_spec:
        mag = amp_from_db(mag)

    spec = make_complex(mag, phase)
    # Transpose back to (batch, freq, frames) for torch.istft
    spec = spec.transpose(-1, -2)

    n_fft = stft_num_fft(pr)
    hop = stft_frame_step(pr)
    win_len = stft_frame_length(pr)
    window = torch.hann_window(win_len, device=spec.device, dtype=torch.float32)

    samples = torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win_len,
        window=window,
        center=False,
    )
    return samples


# ---------------------------------------------------------------------------
# Griffin-Lim (from soundrep.py)
# ---------------------------------------------------------------------------

def griffin_lim(spec_complex, frame_length, frame_step, num_fft, num_iters=1):
    """
    Griffin-Lim algorithm for phase reconstruction.

    Args:
        spec_complex: (batch, frames, freq) complex tensor
        frame_length, frame_step, num_fft: STFT parameters
        num_iters: number of iterations

    Returns:
        waveform: (batch, time)
    """
    window = torch.hann_window(frame_length, device=spec_complex.device)
    spec_mag = spec_complex.abs().to(torch.complex64)

    best = spec_complex.clone()
    for _ in range(num_iters):
        # iSTFT
        best_t = best.transpose(-1, -2)  # (batch, freq, frames)
        samples = torch.istft(best_t, num_fft, frame_step, frame_length, window, center=False)

        # STFT of reconstructed signal
        est = torch.stft(samples, num_fft, frame_step, frame_length, window,
                         center=False, return_complex=True)
        est = est.transpose(-1, -2)  # (batch, frames, freq)

        # Update phase
        phase = est / torch.clamp(est.abs().to(torch.complex64), min=1e-8)
        best = spec_mag * phase

    # Final iSTFT
    best_t = best.transpose(-1, -2)
    y = torch.istft(best_t, num_fft, frame_step, frame_length, window, center=False)
    return y.float()


# ---------------------------------------------------------------------------
# dB / amplitude conversions (from soundrep.py)
# ---------------------------------------------------------------------------

def db_from_amp(x):
    """Convert amplitude to decibels: 20 * log10(max(x, 1e-5))."""
    return 20.0 * torch.log10(torch.clamp(x, min=1e-5))


def amp_from_db(x):
    """Convert decibels to amplitude: 10^(x/20)."""
    return torch.pow(10.0, x / 20.0)


# ---------------------------------------------------------------------------
# RMS normalization (from tfutil.py)
# ---------------------------------------------------------------------------

def normalize_rms(samples, desired_rms=0.1, eps=1e-4):
    """
    Normalize audio to a target RMS.

    Args:
        samples: (batch, time) or (batch, time, channels)
        desired_rms: target RMS value
        eps: minimum RMS to avoid division by zero

    Returns:
        Normalized samples
    """
    rms = torch.sqrt(torch.mean(samples ** 2, dim=1, keepdim=True))
    rms = torch.clamp(rms, min=eps)
    return samples * (desired_rms / rms)


def normalize_rms_np(samples, desired_rms=0.1, eps=1e-4):
    """Numpy version of normalize_rms."""
    rms = np.maximum(eps, np.sqrt(np.mean(samples ** 2, axis=1, keepdims=True)))
    return samples * (desired_rms / rms)


# ---------------------------------------------------------------------------
# Image normalization (from tfutil.py)
# ---------------------------------------------------------------------------

def normalize_ims(im):
    """
    Normalize uint8 images to [-1, 1] range.
    TF: -1.0 + (2.0 / 255) * im

    Args:
        im: uint8 tensor or float tensor

    Returns:
        float tensor in [-1, 1]
    """
    im = im.float() if isinstance(im, torch.Tensor) else torch.tensor(im, dtype=torch.float32)
    return -1.0 + (2.0 / 255.0) * im


def unnormalize_ims(im):
    """Inverse of normalize_ims: [-1, 1] → [0, 255]."""
    return (im + 1.0) * (255.0 / 2.0)


# ---------------------------------------------------------------------------
# Sound feature normalization (from shift_net.py)
# ---------------------------------------------------------------------------

def normalize_sfs(sfs, scale=255.0):
    """
    Compress dynamic range of sound features (raw waveform samples).
    TF: sign(sfs) * (log(1 + scale * |sfs|) / log(1 + scale))

    Args:
        sfs: (batch, time, channels) or (batch, time)

    Returns:
        Normalized features
    """
    return torch.sign(sfs) * (torch.log1p(scale * torch.abs(sfs)) / math.log(1.0 + scale))


# ---------------------------------------------------------------------------
# Spectrogram normalization (from sourcesep.py)
# ---------------------------------------------------------------------------

def norm_range(x, min_val, max_val):
    """Normalize to [-1, 1] given a known min/max range."""
    return 2.0 * (x - min_val) / float(max_val - min_val) - 1.0


def unnorm_range(y, min_val, max_val):
    """Inverse of norm_range."""
    return 0.5 * float(max_val - min_val) * (y + 1.0) + min_val


def normalize_spec(spec, pr):
    """Normalize spectrogram magnitude to [-1, 1]."""
    return norm_range(spec, pr.spec_min, pr.spec_max)


def unnormalize_spec(spec, pr):
    """Denormalize spectrogram magnitude from [-1, 1]."""
    return unnorm_range(spec, pr.spec_min, pr.spec_max)


def normalize_phase(phase, pr=None):
    """Normalize phase to [-1, 1] (from [-pi, pi])."""
    return norm_range(phase, -np.pi, np.pi)


def unnormalize_phase(phase, pr=None):
    """Denormalize phase from [-1, 1] to [-pi, pi]."""
    return unnorm_range(phase, -np.pi, np.pi)


# ---------------------------------------------------------------------------
# Spectrogram packing for multi-track (from soundrep.py)
# ---------------------------------------------------------------------------

def pack_spec(spec_complex, pr):
    """
    Split complex spectrogram into magnitude (dB) and phase.
    Drops the last frequency bin (assumes odd number of bins).

    Args:
        spec_complex: (batch, frames, freq) complex tensor

    Returns:
        spec_complex (trimmed), spec_mag (dB), spec_phase
    """
    assert spec_complex.shape[-1] % 2 == 1
    spec_complex = spec_complex[..., :-1]
    spec_mag = db_from_amp(spec_complex.abs())
    spec_phase = spec_complex.angle()
    return spec_complex, spec_mag, spec_phase


def unpack_spec(spec_mag, spec_phase, pr):
    """
    Recombine magnitude (dB) and phase into a complex spectrogram.
    Pads the last frequency bin back.
    """
    mag = amp_from_db(spec_mag)
    spec = make_complex(mag, spec_phase)
    # Pad one frequency bin at the end
    spec = F.pad(spec, (0, 1), mode="constant", value=0.0)
    return spec


def stft_multi_track(samples, pr):
    """
    Multi-channel STFT: for each channel, compute mag and phase,
    concatenate along last dimension.

    Args:
        samples: (batch, time, channels)
        pr: Params

    Returns:
        (batch, frames, 2*channels) — alternating mag, phase per channel
    """
    n_fft = stft_num_fft(pr)
    hop = stft_frame_step(pr)
    win_len = stft_frame_length(pr)
    window = torch.hann_window(win_len, device=samples.device, dtype=samples.dtype)

    tracks = []
    for i in range(samples.shape[-1]):
        channel = samples[..., i]
        spec = torch.stft(channel, n_fft, hop, win_len, window,
                          center=False, return_complex=True)
        spec = spec.transpose(-1, -2)  # (batch, frames, freq)
        _, spec_mag, spec_phase = pack_spec(spec, pr)
        tracks.append(spec_mag.unsqueeze(-1))
        tracks.append(spec_phase.unsqueeze(-1))
    return torch.cat(tracks, dim=-1)
