"""
Video source separation inference.

Port of sep_video.py — processes a video file, performs on/off-screen
source separation, and outputs separated audio tracks as MP4 files.
"""

import os
import sys
import argparse
import subprocess
import tempfile
import glob
import numpy as np
import torch

from PIL import Image

from ..models.sourcesep import SourceSepUNet, SourceSepClassifier
from ..models.shift_net import ShiftNet
from ..utils.audio import stft, normalize_rms_np
from ..utils.params import sep_full, sep_unet_pit


def extract_video(vid_file, output_dir, pr, start_time, dur):
    """Extract frames and audio from video using FFmpeg."""
    # Small frames for model input
    subprocess.run([
        'ffmpeg', '-loglevel', 'error', '-ss', str(start_time),
        '-i', vid_file, '-t', str(dur), '-r', str(pr.fps),
        '-vf', f'scale=256:256',
        os.path.join(output_dir, 'small_%04d.png')
    ], check=True)

    # Full-res frames
    subprocess.run([
        'ffmpeg', '-loglevel', 'error', '-ss', str(start_time),
        '-i', vid_file, '-t', str(dur), '-r', str(pr.fps),
        os.path.join(output_dir, 'full_%04d.png')
    ], check=True)

    # Audio
    subprocess.run([
        'ffmpeg', '-loglevel', 'error', '-ss', str(start_time),
        '-i', vid_file, '-t', str(dur),
        '-ar', str(int(pr.samp_sr)), '-ac', '2',
        os.path.join(output_dir, 'sound.wav')
    ], check=True)


def load_frames(frame_dir, pattern, max_frames=None):
    """Load frames from directory."""
    files = sorted(glob.glob(os.path.join(frame_dir, pattern)))
    if max_frames:
        files = files[:max_frames]
    frames = [np.array(Image.open(f)) for f in files]
    return np.array(frames) if frames else None


def run_separation(vid_file, start_time, clip_dur, pr, device, out_dir=None):
    """
    Run source separation on a video clip.

    Args:
        vid_file: path to video file
        start_time: start time in seconds
        clip_dur: clip duration in seconds
        pr: Params
        device: torch device
        out_dir: optional output directory

    Returns:
        dict with separated audio and video frames
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract
        extract_video(vid_file, tmpdir, pr, start_time, clip_dur + 0.05)

        # Load frames
        ims = load_frames(tmpdir, 'small_*.png', pr.sampled_frames)
        if ims is None or len(ims) == 0:
            print("ERROR: No frames extracted")
            return None

        # Crop to 224x224
        d = 224
        y = x = ims.shape[1] // 2 - d // 2
        ims = ims[:, y:y + d, x:x + d]
        ims = ims[:pr.sampled_frames]

        # Load audio
        try:
            import soundfile as sf
            samples, sr = sf.read(os.path.join(tmpdir, 'sound.wav'), dtype='float32')
        except ImportError:
            import scipy.io.wavfile as wav
            sr, samples_int = wav.read(os.path.join(tmpdir, 'sound.wav'))
            samples = samples_int.astype(np.float32) / np.iinfo(samples_int.dtype).max

        if samples.ndim == 1:
            samples = np.stack([samples, samples], axis=1)
        samples = samples[:pr.num_samples]

        if samples.shape[0] < pr.num_samples:
            print("WARNING: Not enough audio samples")
            return None

        # Normalize
        input_rms = getattr(pr, 'input_rms', np.sqrt(0.1 ** 2 + 0.1 ** 2))
        samples = normalize_rms_np(samples[None], input_rms)[0]

        # Build model
        clf = SourceSepClassifier(
            pr,
            weights_path=pr.model_path,
            shift_weights_path=getattr(pr, 'shift_model_path', None),
            device=str(device),
        )

        # Predict
        result = clf.predict(ims[None], samples[None])

        samples_fg = result['samples_pred_fg'][0]
        samples_bg = result['samples_pred_bg'][0]

        # Save outputs
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            vid_name = os.path.basename(vid_file).split('.')[0]

            # Load full-res frames for visualization
            fulls = load_frames(tmpdir, 'full_*.png', pr.sampled_frames)
            if fulls is None:
                fulls = ims

            _save_video(fulls, samples_fg, pr, os.path.join(out_dir, f'fg_{vid_name}.mp4'))
            _save_video(fulls, samples_bg, pr, os.path.join(out_dir, f'bg_{vid_name}.mp4'))
            print(f"Saved to {out_dir}")

        return result


def _save_video(frames, audio, pr, output_path):
    """Save frames + audio as MP4 using FFmpeg."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save frames
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(tmpdir, f'{i:04d}.png'))

        # Save audio
        audio_clipped = np.clip(audio, -1.0, 1.0)
        if audio_clipped.ndim == 1:
            audio_clipped = audio_clipped[:, None]

        try:
            import soundfile as sf
            sf.write(os.path.join(tmpdir, 'audio.wav'), audio_clipped,
                     int(pr.samp_sr))
        except ImportError:
            import scipy.io.wavfile as wav
            wav.write(os.path.join(tmpdir, 'audio.wav'), int(pr.samp_sr),
                      (audio_clipped * 32767).astype(np.int16))

        # Combine with FFmpeg
        subprocess.run([
            'ffmpeg', '-loglevel', 'error', '-y',
            '-r', str(pr.fps),
            '-i', os.path.join(tmpdir, '%04d.png'),
            '-i', os.path.join(tmpdir, 'audio.wav'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-shortest',
            output_path
        ], check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Separate on- and off-screen audio from a video"
    )
    parser.add_argument("vid_file", type=str, help="Video file to process")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration in seconds")
    parser.add_argument("--model", type=str, default="full",
                        choices=["full", "unet-pit"],
                        help="Model variant")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to PyTorch weights (.pt)")
    parser.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    # Load params
    if args.model == "full":
        pr = sep_full()
    else:
        pr = sep_unet_pit()

    if args.duration is None:
        args.duration = pr.vid_dur + 0.01

    if args.weights:
        pr.model_path = args.weights
    else:
        pr.model_path = f"../results/nets/sep/{pr.name}/net-{pr.train_iters}.pt"

    pr.input_rms = np.sqrt(0.1 ** 2 + 0.1 ** 2)

    if not os.path.exists(args.vid_file):
        print(f"File not found: {args.vid_file}")
        sys.exit(1)

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")

    run_separation(
        args.vid_file,
        start_time=args.start,
        clip_dur=args.duration,
        pr=pr,
        device=device,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
