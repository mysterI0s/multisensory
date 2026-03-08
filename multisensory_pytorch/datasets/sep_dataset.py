"""
Dataset for source separation training.

Port of sep_dset.py — pairs of videos whose audio is mixed for training the
source separation model.
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class SeparationDataset(Dataset):
    """
    Dataset for source separation training.

    Each sample returns:
      - ims: (3, sampled_frames, crop_dim, crop_dim) float32 — video frames
      - samples_mix: (num_samples, 2) float32 — mixed audio
      - spec_fg: foreground spectrogram components (computed in training loop)
      - spec_bg: background spectrogram components (computed in training loop)
      - samples_fg: (num_samples, 2) float32 — foreground audio
      - samples_bg: (num_samples, 2) float32 — background audio

    Directory structure:
        data_root/
            video_001/
                frames/  (000.jpg, 001.jpg, ...)
                audio.wav
            ...
    """

    def __init__(self, data_root, pr, train=True, both_in_batch=True):
        self.pr = pr
        self.train = train
        self.both_in_batch = both_in_batch

        # Find all video directories
        if os.path.isdir(data_root):
            self.video_dirs = sorted([
                os.path.join(data_root, d)
                for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d))
            ])
        elif data_root.endswith(".txt"):
            with open(data_root) as f:
                self.video_dirs = [l.strip() for l in f if l.strip()]
        else:
            self.video_dirs = []

        if len(self.video_dirs) == 0:
            print(f"WARNING: No video directories found in {data_root}")

    def __len__(self):
        return len(self.video_dirs)

    def _load_video(self, vdir):
        """Load frames and audio from a video directory."""
        pr = self.pr

        # Load frames
        frame_files = sorted(glob.glob(os.path.join(vdir, "frames", "*.jpg")))
        if not frame_files:
            frame_files = sorted(glob.glob(os.path.join(vdir, "frames", "*.png")))

        total_frames = len(frame_files)
        num_slice = pr.sampled_frames
        max_start = max(0, total_frames - num_slice)

        if self.train:
            start = random.randint(0, max_start)
        else:
            start = 0

        # Load frames
        ims = []
        for i in range(start, min(start + num_slice, total_frames)):
            im = np.array(Image.open(frame_files[i]))
            ims.append(im)
        while len(ims) < num_slice:
            ims.append(ims[-1].copy())
        ims = np.stack(ims)

        # Crop
        if self.train and pr.augment_ims:
            y = random.randint(0, max(0, ims.shape[1] - pr.crop_im_dim))
            x = random.randint(0, max(0, ims.shape[2] - pr.crop_im_dim))
        else:
            y = max(0, (ims.shape[1] - pr.crop_im_dim) // 2)
            x = max(0, (ims.shape[2] - pr.crop_im_dim) // 2)

        d = pr.crop_im_dim
        ims = ims[:, y:y + d, x:x + d]

        if self.train and pr.augment_ims and random.random() > 0.5:
            ims = ims[:, :, ::-1].copy()

        # Load audio
        num_audio = int(pr.samples_per_frame * num_slice)
        try:
            import soundfile as sf
            audio_path = os.path.join(vdir, "audio.wav")
            samples, sr = sf.read(audio_path, dtype='float32')
            if sr != pr.samp_sr:
                import librosa
                samples = librosa.resample(samples.T, orig_sr=sr,
                                           target_sr=int(pr.samp_sr)).T
        except (ImportError, FileNotFoundError):
            samples = np.zeros((pr.full_samples_len, 2), dtype=np.float32)

        if samples.ndim == 1:
            samples = np.stack([samples, samples], axis=1)

        audio_start = int(start * pr.samples_per_frame)
        audio = samples[audio_start:audio_start + num_audio]

        if audio.shape[0] < num_audio:
            pad = np.zeros((num_audio - audio.shape[0], 2), dtype=np.float32)
            audio = np.concatenate([audio, pad], axis=0)

        return ims, audio

    def __getitem__(self, idx):
        pr = self.pr

        # Load foreground video
        ims_fg, samples_fg = self._load_video(self.video_dirs[idx])

        # Load a random background video for mixing
        bg_idx = random.randint(0, len(self.video_dirs) - 1)
        while bg_idx == idx and len(self.video_dirs) > 1:
            bg_idx = random.randint(0, len(self.video_dirs) - 1)
        _, samples_bg = self._load_video(self.video_dirs[bg_idx])

        # Normalize RMS
        if pr.normalize_rms:
            from ..utils.audio import normalize_rms_np
            desired_rms = getattr(pr, 'input_rms', 0.1)
            samples_fg = normalize_rms_np(samples_fg[None], desired_rms)[0]
            samples_bg = normalize_rms_np(samples_bg[None], desired_rms)[0]

        # Mix audio
        samples_mix = samples_fg + samples_bg

        # Convert to tensors
        ims_t = torch.from_numpy(ims_fg.copy()).permute(3, 0, 1, 2).float()  # (3, T, H, W)
        mix_t = torch.from_numpy(samples_mix.copy()).float()
        fg_t = torch.from_numpy(samples_fg.copy()).float()
        bg_t = torch.from_numpy(samples_bg.copy()).float()

        return {
            'ims': ims_t,
            'samples_mix': mix_t,
            'samples_fg': fg_t,
            'samples_bg': bg_t,
        }
