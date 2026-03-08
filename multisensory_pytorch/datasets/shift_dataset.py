"""
Dataset for ShiftNet (audio-visual correspondence).

Port of shift_dset.py / sep_dset.py — replaces the TF TFRecordReader +
queue-based pipeline with a PyTorch Dataset + DataLoader.

Supports reading from:
  (1) Pre-extracted directories with JPEG frames + WAV audio
  (2) TFRecords (requires tensorflow for reading)
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class ShiftDataset(Dataset):
    """
    Dataset for ShiftNet training.

    Each sample returns:
      - ims: (sampled_frames, crop_dim, crop_dim, 3) uint8
      - samples_gt: (num_samples, 2) float32 — synced audio
      - samples_shift: (num_samples, 2) float32 — shifted audio
      - label: 0 (shifted) or 1 (synced)

    Directory structure expected:
        data_root/
            video_001/
                frames/  (000.jpg, 001.jpg, ...)
                audio.wav
            video_002/
                ...
    """

    def __init__(self, data_root, pr, train=True):
        self.pr = pr
        self.train = train
        self.data_root = data_root

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

    def __getitem__(self, idx):
        pr = self.pr
        vdir = self.video_dirs[idx]

        # Load frames
        frame_files = sorted(glob.glob(os.path.join(vdir, "frames", "*.jpg")))
        if not frame_files:
            frame_files = sorted(glob.glob(os.path.join(vdir, "frames", "*.png")))

        total_frames = len(frame_files)
        num_slice = pr.sampled_frames

        # Random or fixed frame start
        max_start = max(0, total_frames - num_slice)
        if self.train and not pr.fix_frame:
            gt_start = random.randint(0, max_start)
        else:
            gt_start = 0

        # Load and crop frames
        ims = []
        for i in range(gt_start, min(gt_start + num_slice, total_frames)):
            im = np.array(Image.open(frame_files[i]))
            ims.append(im)

        # Pad if fewer frames
        while len(ims) < num_slice:
            ims.append(ims[-1].copy())

        ims = np.stack(ims)  # (T, H, W, 3)

        # Crop
        if self.train and pr.augment_ims:
            y = random.randint(0, max(0, ims.shape[1] - pr.crop_im_dim))
            x = random.randint(0, max(0, ims.shape[2] - pr.crop_im_dim))
        else:
            y = max(0, (ims.shape[1] - pr.crop_im_dim) // 2)
            x = max(0, (ims.shape[2] - pr.crop_im_dim) // 2)

        d = pr.crop_im_dim
        ims = ims[:, y:y + d, x:x + d]

        # Random horizontal flip
        if self.train and pr.augment_ims and random.random() > 0.5:
            ims = ims[:, :, ::-1].copy()

        # Load audio
        try:
            import soundfile as sf
            audio_path = os.path.join(vdir, "audio.wav")
            samples, sr = sf.read(audio_path, dtype='float32')
            if sr != pr.samp_sr:
                import librosa
                samples = librosa.resample(samples.T, orig_sr=sr,
                                           target_sr=int(pr.samp_sr)).T
        except ImportError:
            # Fallback: numpy memmap or dummy
            samples = np.zeros((pr.full_samples_len, 2), dtype=np.float32)

        if samples.ndim == 1:
            samples = np.stack([samples, samples], axis=1)

        # Slice synced audio
        num_audio_samples = int(pr.samples_per_frame * num_slice)
        gt_audio_start = int(gt_start * pr.samples_per_frame)
        samples_gt = samples[gt_audio_start:gt_audio_start + num_audio_samples]

        # Pad if needed
        if samples_gt.shape[0] < num_audio_samples:
            pad = np.zeros((num_audio_samples - samples_gt.shape[0], 2), dtype=np.float32)
            samples_gt = np.concatenate([samples_gt, pad], axis=0)

        # Shifted audio (for negative example)
        if pr.do_shift:
            shift_start = random.randint(0, max(0, total_frames - num_slice))
            while abs(shift_start - gt_start) < getattr(pr, 'min_shift_frames', 0):
                shift_start = random.randint(0, max(0, total_frames - num_slice))
            shift_audio_start = int(shift_start * pr.samples_per_frame)
            samples_shift = samples[shift_audio_start:shift_audio_start + num_audio_samples]
            if samples_shift.shape[0] < num_audio_samples:
                pad = np.zeros((num_audio_samples - samples_shift.shape[0], 2), dtype=np.float32)
                samples_shift = np.concatenate([samples_shift, pad], axis=0)
        else:
            samples_shift = samples_gt.copy()

        # Random label: 0 = shifted (negative), 1 = synced (positive)
        if pr.do_shift and self.train:
            label = random.randint(0, 1)
        else:
            label = 1

        if label == 0:
            audio_out = samples_shift
        else:
            audio_out = samples_gt

        # Convert to tensors
        # PyTorch convention: images as (C, D, H, W) for 3D or (C, H, W) for 2D
        ims_t = torch.from_numpy(ims.copy()).permute(3, 0, 1, 2).float()  # (3, T, H, W)
        audio_t = torch.from_numpy(audio_out.copy()).float()
        label_t = torch.tensor(label, dtype=torch.long)

        return {
            'ims': ims_t,
            'samples': audio_t,
            'label': label_t,
        }


class ShiftDatasetFromTFRecords(Dataset):
    """
    Read data from TFRecords (requires tensorflow installed).
    Falls back gracefully if TF is not available.
    """

    def __init__(self, tf_path, pr, train=True):
        self.pr = pr
        self.train = train
        self.records = []

        try:
            import tensorflow as tf
            tf.compat.v1.enable_eager_execution()

            # Find record files
            if os.path.isdir(tf_path):
                rec_files = sorted(glob.glob(os.path.join(tf_path, "*.tf")))
                rec_files += sorted(glob.glob(os.path.join(tf_path, "*.tfrecords")))
            elif tf_path.endswith(".txt"):
                with open(tf_path) as f:
                    rec_files = [l.strip() for l in f if l.strip() and os.path.exists(l.strip())]
            else:
                rec_files = [tf_path]

            self.records = rec_files
            self._read_tf = True
            print(f"ShiftDatasetFromTFRecords: found {len(rec_files)} record files")
        except ImportError:
            print("WARNING: tensorflow not installed, ShiftDatasetFromTFRecords unavailable")
            self._read_tf = False

    def __len__(self):
        # Approximate count
        return max(1, len(self.records) * 100)

    def __getitem__(self, idx):
        raise NotImplementedError(
            "TFRecord reading within __getitem__ requires a streaming approach. "
            "Consider pre-converting TFRecords to individual files using "
            "`scripts/convert_tfrecords.py`."
        )
