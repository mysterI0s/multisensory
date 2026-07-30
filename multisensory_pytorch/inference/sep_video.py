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
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            str(start_time),
            "-i",
            vid_file,
            "-t",
            str(dur),
            "-r",
            str(pr.fps),
            "-vf",
            f"scale=256:256",
            os.path.join(output_dir, "small_%04d.png"),
        ],
        check=True,
    )

    # Full-res frames
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            str(start_time),
            "-i",
            vid_file,
            "-t",
            str(dur),
            "-r",
            str(pr.fps),
            os.path.join(output_dir, "full_%04d.png"),
        ],
        check=True,
    )

    # Audio
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            str(start_time),
            "-i",
            vid_file,
            "-t",
            str(dur),
            "-ar",
            str(int(pr.samp_sr)),
            "-ac",
            "2",
            os.path.join(output_dir, "sound.wav"),
        ],
        check=True,
    )


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
    # Force single-chunk models to run in discrete 2.135s sliding windows
    chunk_dur = pr.vid_dur
    num_chunks = int(np.ceil(clip_dur / chunk_dur))
    padded_dur = num_chunks * chunk_dur

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract the full padded duration
        extract_video(vid_file, tmpdir, pr, start_time, padded_dur + 0.05)

        # Load frames
        total_frames = int(padded_dur * pr.fps)
        ims = load_frames(tmpdir, "small_*.png", total_frames)
        if ims is None or len(ims) == 0:
            print("ERROR: No frames extracted")
            return None

        # Crop to 224x224
        d = 224
        y = x = ims.shape[1] // 2 - d // 2
        ims = ims[:, y : y + d, x : x + d]

        # Load audio
        try:
            import soundfile as sf

            samples, sr = sf.read(os.path.join(tmpdir, "sound.wav"), dtype="float32")
        except ImportError:
            import scipy.io.wavfile as wav

            sr, samples_int = wav.read(os.path.join(tmpdir, "sound.wav"))
            samples = samples_int.astype(np.float32) / np.iinfo(samples_int.dtype).max

        if samples.ndim == 1:
            samples = np.stack([samples, samples], axis=1)

        # Build model once
        clf = SourceSepClassifier(
            pr,
            weights_path=pr.model_path,
            shift_weights_path=getattr(pr, "shift_model_path", None),
            device=str(device),
        )

        # Process chunks sequentially
        fg_chunks = []
        bg_chunks = []

        for i in range(num_chunks):
            print(f"Processing chunk {i+1} of {num_chunks}...")
            f_start = i * pr.sampled_frames
            f_end = (i + 1) * pr.sampled_frames

            s_start = i * pr.num_samples
            s_end = (i + 1) * pr.num_samples

            chunk_ims = ims[f_start:f_end]
            chunk_samples = samples[s_start:s_end]

            # Pad if the final chunk is slightly short (though padded_dur extraction should prevent this)
            if len(chunk_ims) < pr.sampled_frames:
                pad_f = pr.sampled_frames - len(chunk_ims)
                chunk_ims = np.pad(
                    chunk_ims, ((0, pad_f), (0, 0), (0, 0), (0, 0)), mode="edge"
                )

            if len(chunk_samples) < pr.num_samples:
                pad_s = pr.num_samples - len(chunk_samples)
                chunk_samples = np.pad(
                    chunk_samples, ((0, pad_s), (0, 0)), mode="constant"
                )

            # Normalize audio chunk
            input_rms = getattr(pr, "input_rms", np.sqrt(0.1**2 + 0.1**2))
            chunk_samples = normalize_rms_np(chunk_samples[None], input_rms)[0]

            # Predict
            result = clf.predict(chunk_ims[None], chunk_samples[None])
            fg_chunks.append(result["samples_pred_fg"][0])
            bg_chunks.append(result["samples_pred_bg"][0])

        # Concatenate all chunks back together
        samples_fg = np.concatenate(fg_chunks, axis=0)
        samples_bg = np.concatenate(bg_chunks, axis=0)

        # Trim to exact requested clip duration
        exact_samples = int(clip_dur * pr.samp_sr)
        exact_frames = int(clip_dur * pr.fps)

        samples_fg = samples_fg[:exact_samples]
        samples_bg = samples_bg[:exact_samples]

        # Save outputs
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            vid_name = os.path.basename(vid_file).split(".")[0]

            # Load full-res frames for visualization
            fulls = load_frames(tmpdir, "full_*.png", total_frames)
            if fulls is None:
                fulls = ims

            fulls = fulls[:exact_frames]

            _save_video(
                fulls, samples_fg, pr, os.path.join(out_dir, f"fg_{vid_name}.mp4")
            )
            _save_video(
                fulls, samples_bg, pr, os.path.join(out_dir, f"bg_{vid_name}.mp4")
            )
            print(f"Saved to {out_dir}")

        return dict(samples_pred_fg=samples_fg[None], samples_pred_bg=samples_bg[None])


def _save_video(frames, audio, pr, output_path):
    """Save frames + audio as MP4 using FFmpeg."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save frames
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(tmpdir, f"{i:04d}.png"))

        # Save audio
        audio_clipped = np.clip(audio, -1.0, 1.0)
        if audio_clipped.ndim == 1:
            audio_clipped = audio_clipped[:, None]

        try:
            import soundfile as sf

            sf.write(os.path.join(tmpdir, "audio.wav"), audio_clipped, int(pr.samp_sr))
        except ImportError:
            import scipy.io.wavfile as wav

            wav.write(
                os.path.join(tmpdir, "audio.wav"),
                int(pr.samp_sr),
                (audio_clipped * 32767).astype(np.int16),
            )

        # Combine with FFmpeg
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-r",
                str(pr.fps),
                "-i",
                os.path.join(tmpdir, "%04d.png"),
                "-i",
                os.path.join(tmpdir, "audio.wav"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                output_path,
            ],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Separate on- and off-screen audio from a video"
    )
    parser.add_argument("vid_file", type=str, help="Video file to process")
    parser.add_argument(
        "--start", type=float, default=0.0, help="Start time in seconds"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="Duration in seconds"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="full",
        choices=["full", "unet-pit"],
        help="Model variant",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to PyTorch weights (.pt) for the SourceSep model",
    )
    parser.add_argument(
        "--shift_weights",
        type=str,
        default=None,
        help="Path to PyTorch weights (.pt) for the ShiftNet (video) model",
    )
    parser.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
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

    if args.shift_weights:
        pr.shift_model_path = args.shift_weights
    else:
        # Default fallback
        pr.shift_model_path = f"../results/nets/shift/net-650000.pt"

    pr.input_rms = np.sqrt(0.1**2 + 0.1**2)

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
