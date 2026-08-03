#!/usr/bin/env python
"""
sep_cam_probe.py (v2) -- does the SEPARATION checkpoint also give a usable heatmap?

v1 established that it does: the CAM tensor exists in the separation graph and
comes out strongly non-flat (spatial contrast 0.63 at 7x7, 0.75 at 14x14,
against a 0.05 failure threshold).

v2 answers the three questions v1's output raised:

  1. Is this actually running on the GPU?  v1 reported 2469 ms for a 2.135 s
     window and printed no `Created device .../device:GPU:0` line.  v2 prints
     the device list explicitly and does warm, repeated timing so the number
     excludes graph warmup / cuDNN autotuning.

  2. Does pr.cam=True perturb separation?  v1 said yes: fg rms moved +0.7% but
     bg rms moved -15%.  v2 can write the raw arrays so you can diff two runs
     numerically instead of by ear (--save_npy).

  3. Is np.abs() the right reduction?  find_cam() uses it, but the raw CAM is
     signed (v1: -0.54 to +0.71), so abs conflates evidence AGAINST alignment
     with evidence FOR it.  v2 reports abs, positive-only, and raw side by
     side, and writes an overlay for each.

v2 also reports temporal consistency across the 8 CAM timesteps -- a heatmap
that jumps around frame to frame will flicker badly once it is driving a live
overlay at 8 fps.

Usage
-----
    python sep_cam_probe.py --vid_file ../data/translator.mp4 --out ../results/probe_lo
    python sep_cam_probe.py --vid_file ../data/translator.mp4 --cam_hires \\
        --out ../results/probe_hi --save_npy

    # steady-state latency, 10 timed iterations after a warmup
    python sep_cam_probe.py --vid_file ../data/translator.mp4 --warm 10
"""

import argparse
import os
import subprocess
import sys
import tempfile
import shutil
import time

import numpy as np

# Headless backend BEFORE anything drags matplotlib in. sep_video.py imports
# pylab at module scope (line 2), which tries to open a display on a server.
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402

from scipy.io import wavfile  # noqa: E402
from PIL import Image  # noqa: E402

import tfutil as mu  # noqa: E402
import sep_params  # noqa: E402
import sep_video  # noqa: E402


def report_devices():
    """Say plainly whether TensorFlow can see a GPU.

    v1's log had no 'Created device .../device:GPU:0' line, which is what TF
    prints when it actually binds one -- so the 2469 ms may have been CPU.
    """
    print("")
    print("---- devices ----")
    try:
        import tensorflow as tf_top

        gpus = tf_top.config.list_physical_devices("GPU")
        print("tf.__version__        :", tf_top.__version__)
        print("physical GPUs visible :", len(gpus))
        for g in gpus:
            print("   ", g)
        if not gpus:
            print("  >> NO GPU VISIBLE. Timings below are CPU and are not")
            print("     representative of your serving latency.")
            print("     Check: pip list | grep tensorflow   (need tensorflow-gpu")
            print("     or TF>=2.x with CUDA), and nvidia-smi.")
    except Exception as e:  # pragma: no cover
        print("could not query devices:", e)


def extract(vid_file, start, dur, fps, samp_sr, tmp):
    """Mirror the ffmpeg calls in sep_video.run(), minus the full-res one."""
    frame_pat = os.path.join(tmp, "small_%04d.png")
    wav_path = os.path.join(tmp, "sound.wav")

    subprocess.check_call(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            str(start),
            "-i",
            vid_file,
            "-t",
            str(dur),
            "-r",
            str(fps),
            "-vf",
            "scale=256:256",
            frame_pat,
        ]
    )
    subprocess.check_call(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            str(start),
            "-i",
            vid_file,
            "-t",
            str(dur),
            "-ar",
            str(int(samp_sr)),
            "-ac",
            "2",
            wav_path,
        ]
    )
    return wav_path


def load_frames(tmp, sampled_frames, crop_dim):
    files = sorted(
        os.path.join(tmp, f) for f in os.listdir(tmp) if f.startswith("small_")
    )
    if not files:
        sys.exit("ERROR: ffmpeg extracted no frames. Is ffmpeg installed?")

    ims = np.array([np.asarray(Image.open(f).convert("RGB")) for f in files])
    d = crop_dim
    y = x = ims.shape[1] // 2 - d // 2
    ims = ims[:, y : y + d, x : x + d]
    ims = ims[:sampled_frames]

    if ims.shape[0] < sampled_frames:
        sys.exit(
            "ERROR: got %d frames, need %d. Use a longer clip or an earlier "
            "--start." % (ims.shape[0], sampled_frames)
        )
    return ims.astype(np.uint8)


def load_audio(wav_path, num_samples, input_rms):
    sr, data = wavfile.read(wav_path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    if data.ndim == 1:  # mono -> duplicate; the net wants 2 channels
        data = np.stack([data, data], axis=1)

    data = data[:num_samples]
    if data.shape[0] < num_samples:
        sys.exit("ERROR: got %d audio samples, need %d." % (data.shape[0], num_samples))

    return mu.normalize_rms_np(data[None], input_rms)[0]


def save_wav(path, samples, sr):
    x = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    wavfile.write(path, int(sr), (x * 32767).astype(np.int16))


def write_overlay(spatial, ims, path):
    norm = spatial - spatial.min()
    norm = norm / (norm.max() + 1e-8)
    heat = (cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat).resize(
        (ims.shape[2], ims.shape[1]), Image.BILINEAR
    )
    base = Image.fromarray(ims[len(ims) // 2]).convert("RGB")
    Image.blend(base, heat_img.convert("RGB"), 0.5).save(path)


def describe(spatial, name):
    """Print contrast / halves / peak for one spatial reduction."""
    flatness = spatial.std() / (abs(spatial.mean()) + 1e-8)
    h, w = spatial.shape
    left = spatial[:, : w // 2].mean()
    right = spatial[:, w // 2 :].mean()
    peak = np.unravel_index(np.argmax(spatial), spatial.shape)
    print(
        "  %-9s contrast %.4f | L %.5f R %.5f (ratio %.3f) | peak %s "
        "-> rel (%.2f, %.2f)"
        % (
            name,
            flatness,
            left,
            right,
            left / (right + 1e-8),
            peak,
            peak[0] / float(h - 1) if h > 1 else 0.0,
            peak[1] / float(w - 1) if w > 1 else 0.0,
        )
    )
    if flatness < 0.05:
        print("     >> WARNING: near-flat. Head is probably not localizing.")
    return flatness


def cam_report(cam, out_dir, ims, tag, save_npy=False):
    print("")
    print("---- CAM report (%s) ----" % tag)
    print("shape (T', H', W')    :", cam.shape)
    print(
        "raw min / max / std   : %.5f / %.5f / %.5f" % (cam.min(), cam.max(), cam.std())
    )
    print("positive fraction     : %.3f" % float((cam > 0).mean()))

    # Three reductions. find_cam() uses abs, but the CAM is signed, so abs
    # conflates 'strong evidence against alignment' with 'strong evidence
    # for'. Positive-only is the one that should track the sounding object.
    reductions = {
        "abs": np.abs(cam).mean(axis=0),
        "positive": np.maximum(cam, 0.0).mean(axis=0),
        "raw": cam.mean(axis=0),
    }

    print("")
    print("reduction comparison (peak rel coords are (row, col), 0=top/left):")
    for name, spatial in reductions.items():
        describe(spatial, name)
        write_overlay(spatial, ims, os.path.join(out_dir, "overlay_%s.png" % name))

    # Backwards-compatible name for the v1 output.
    write_overlay(reductions["abs"], ims, os.path.join(out_dir, "overlay.png"))

    # Temporal consistency. The CAM has 8 timesteps; if the peak wanders
    # between them the live overlay will flicker at 8 fps.
    print("")
    print("temporal consistency (positive reduction, per-timestep):")
    pos_t = np.maximum(cam, 0.0)
    peaks = [
        np.unravel_index(np.argmax(pos_t[t]), pos_t[t].shape)
        for t in range(pos_t.shape[0])
    ]
    print("  per-timestep peaks  :", peaks)
    flat = pos_t.reshape(pos_t.shape[0], -1)
    if flat.shape[0] > 1:
        cors = []
        for t in range(flat.shape[0] - 1):
            a, b = flat[t], flat[t + 1]
            if a.std() > 1e-9 and b.std() > 1e-9:
                cors.append(float(np.corrcoef(a, b)[0, 1]))
        if cors:
            print(
                "  adjacent-frame corr : mean %.4f  min %.4f"
                % (float(np.mean(cors)), float(np.min(cors)))
            )
            if np.mean(cors) < 0.5:
                print(
                    "     >> Low. Expect visible flicker; consider EMA "
                    "smoothing across windows."
                )

    np.save(os.path.join(out_dir, "cam.npy"), cam)
    if save_npy:
        for name, spatial in reductions.items():
            np.save(os.path.join(out_dir, "spatial_%s.npy" % name), spatial)

    Image.fromarray(ims[len(ims) // 2]).save(os.path.join(out_dir, "center_frame.png"))


BASE_VID_DUR = 2.135  # sep_params.VidDur


def check_vid_dur(vid_dur, fps=29.97, samp_sr=21000.0):
    """Warn if vid_dur is not a power-of-two multiple of the base window.

    sep_params.base() computes

        pr.spec_len = 128 * int(2 ** np.round(np.log2(vid_dur / VidDur)))

    The np.round means spec_len can ONLY be 128 * a power of two. If you ask
    for a window between two of those steps, the spectrogram is allocated for
    the rounded size and the surplus audio is quietly truncated -- you pay for
    the extra decode and get nothing back.

    The u-net also halves the spectrogram time axis nine times (gen/conv1..9),
    so spec_len must stay divisible by 512 in practice; 128 * 2**k satisfies
    that for k >= 2 and degrades gracefully below.
    """
    ratio = vid_dur / BASE_VID_DUR
    exp = np.log2(ratio)
    nearest = int(np.round(exp))
    snapped = BASE_VID_DUR * (2.0**nearest)
    spec_len = 128 * int(2**nearest)
    frames = int(vid_dur * fps)
    samples = int(round((samp_sr / fps) * frames))

    print("")
    print("---- window check ----")
    print(
        "requested vid_dur   : %.4f s  (%.3fx the %.3f s base)"
        % (vid_dur, ratio, BASE_VID_DUR)
    )
    print("sampled_frames      : %d" % frames)
    print(
        "num_samples         : %d  (%.4f s at %g Hz)"
        % (samples, samples / samp_sr, samp_sr)
    )
    print("spec_len            : %d" % spec_len)

    if abs(exp - nearest) > 0.08:
        print("")
        print("  >> WARNING: not a power-of-two multiple.")
        print(
            "     spec_len rounds to the %gx step (%.3f s) while you feed"
            % (2.0**nearest, snapped)
        )
        print(
            "     %.3f s of audio. The surplus is TRUNCATED inside the net." % vid_dur
        )
        print("     Check the 'Raw spec length' vs 'Truncated spec length'")
        print("     lines below. If they differ, you are wasting the extra.")
        print(
            "     Safe values: %s"
            % ", ".join("%.3f" % (BASE_VID_DUR * 2.0**k) for k in (-1, 0, 1, 2))
        )
    else:
        print("  OK: clean %gx step, no spectrogram truncation." % (2.0**nearest))

    print("")
    print(
        "  NOTE: the checkpoint was TRAINED at %.3f s. Longer windows run"
        % BASE_VID_DUR
    )
    print("  the net off-distribution the same way --cam_hires did. Treat any")
    print("  result at a non-default window as unverified until you A/B it.")


def report_gpu_memory():
    """Report GPU memory footprint.

    The question this answers: with allow_growth on, how much VRAM does
    this model actually hold?  That determines whether the PyTorch model
    can stay resident alongside it on a 6 GB card, or whether the model
    switch has to unload one to load the other.
    """
    import subprocess

    print("")
    print("---- GPU memory ----")
    try:
        import tensorflow as _tf

        info = _tf.config.experimental.get_memory_info("GPU:0")
        for key in ("current", "peak"):
            if key in info:
                print("  tf %-18s: %8.1f MiB" % (key, info[key] / (1024.0**2)))
    except Exception as exc:
        print("  tf memory info unavailable: %s" % exc)
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
            .splitlines()[0]
        )
        used, total = [s.strip() for s in out.split(",")[:2]]
        free = int(total) - int(used)
        print("  whole card            : %s MiB used / %s MiB total" % (used, total))
        print("  free for another model: %d MiB" % free)
    except Exception as exc:
        print("  nvidia-smi unavailable: %s" % exc)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vid_file", required=True)
    p.add_argument("--out", default="../results/sep_cam_probe")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model", default="full")
    p.add_argument(
        "--mask",
        default=None,
        choices=[None, "l", "r"],
        help="Grey out half the frame. NOTE the mapping is "
        "inverted vs the flag name: mask='l' hides the LEFT "
        "half, so fg becomes the RIGHT speaker.",
    )
    p.add_argument(
        "--cam_hires",
        action="store_true",
        help="pr.cam=True -> 14x14 CAM instead of 7x7. Measured "
        "cost: bg rms shifted -15%% on translator.mp4.",
    )
    p.add_argument(
        "--warm",
        type=int,
        default=0,
        help="Run N extra timed iterations after a warmup and "
        "report median steady-state latency.",
    )
    p.add_argument(
        "--save_npy",
        action="store_true",
        help="Also save the spatial reductions, for diffing runs.",
    )
    p.add_argument(
        "--vid_dur",
        type=float,
        default=None,
        help="Analysis window in seconds. Default 2.135 (the "
        "training window). Only power-of-two multiples are "
        "clean: 1.068, 2.135, 4.270, 8.540. Other values get "
        "their spectrogram truncated.",
    )
    arg = p.parse_args()

    mu.set_gpus([arg.gpu])
    report_devices()

    if arg.vid_dur is None:
        pr = getattr(sep_params, arg.model)()
    else:
        check_vid_dur(arg.vid_dur)
        pr = getattr(sep_params, arg.model)(vid_dur=arg.vid_dur)
    # input_rms is set by sep_params itself now; the guard covers older
    # param sets. model_path honors MULTISENSORY_RESULTS like the rest.
    if not hasattr(pr, "input_rms"):
        pr.input_rms = np.sqrt(0.1**2 + 0.1**2)
    pr.model_path = os.path.join(
        sep_params.results_root(), "nets", "sep", pr.name, "net.tf-%d" % pr.train_iters
    )

    if arg.cam_hires:
        pr.cam = True

    print("")
    print("model_path    :", pr.model_path)
    print("pr.cam        :", pr.cam)
    print("sampled_frames:", pr.sampled_frames)
    print("num_samples   :", pr.num_samples, "@", pr.samp_sr, "Hz")
    print("window        : %.3f s" % (pr.num_samples / float(pr.samp_sr)))

    os.makedirs(arg.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="sep_cam_probe_")
    try:
        wav_path = extract(
            arg.vid_file, arg.start, pr.vid_dur + 0.05, pr.fps, pr.samp_sr, tmp
        )
        ims = load_frames(tmp, pr.sampled_frames, pr.crop_im_dim)
        samples = load_audio(wav_path, pr.num_samples, pr.input_rms)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if arg.mask == "l":
        ims[:, :, : ims.shape[2] // 2] = 128
    elif arg.mask == "r":
        ims[:, :, ims.shape[2] // 2 :] = 128

    net = sep_video.NetClf(pr, gpu="/gpu:%d" % arg.gpu)
    net.init()

    if net.cam_op is None:
        sys.exit(
            "ERROR: no CAM tensor (net_style=%r has no video trunk). "
            "Use net_style='full'." % pr.net_style
        )

    t0 = time.time()
    ret = net.predict_with_cam(ims[None], samples[None])
    first_ms = (time.time() - t0) * 1000.0
    print("")
    print("first predict_with_cam : %.1f ms (includes warmup/autotune)" % first_ms)

    if arg.warm > 0:
        times = []
        for _ in range(arg.warm):
            t0 = time.time()
            net.predict_with_cam(ims[None], samples[None])
            times.append((time.time() - t0) * 1000.0)
        times = np.array(times)
        window_ms = 1000.0 * pr.num_samples / float(pr.samp_sr)
        print(
            "steady-state over %d runs: median %.1f ms  min %.1f  max %.1f"
            % (
                arg.warm,
                float(np.median(times)),
                float(times.min()),
                float(times.max()),
            )
        )
        rt = float(np.median(times)) / window_ms
        print("real-time factor       : %.2fx  (window is %.0f ms)" % (rt, window_ms))
        if rt >= 1.0:
            print(
                "  >> Slower than real time. Cannot sustain a live stream "
                "at this hop without dropping windows."
            )

    report_gpu_memory()

    save_wav(os.path.join(arg.out, "mix.wav"), samples[:, 0], pr.samp_sr)
    save_wav(os.path.join(arg.out, "fg.wav"), ret["samples_pred_fg"][0], pr.samp_sr)
    save_wav(os.path.join(arg.out, "bg.wav"), ret["samples_pred_bg"][0], pr.samp_sr)

    fg = np.asarray(ret["samples_pred_fg"][0], dtype=np.float64)
    bg = np.asarray(ret["samples_pred_bg"][0], dtype=np.float64)
    print("")
    print(
        "fg rms / bg rms       : %.5f / %.5f"
        % (np.sqrt((fg**2).mean()), np.sqrt((bg**2).mean()))
    )
    if arg.save_npy:
        np.save(os.path.join(arg.out, "fg.npy"), fg)
        np.save(os.path.join(arg.out, "bg.npy"), bg)

    cam_report(ret["cam"], arg.out, ims, "pr.cam=%s" % pr.cam, save_npy=arg.save_npy)

    print("")
    print("wrote ->", os.path.abspath(arg.out))


if __name__ == "__main__":
    main()
