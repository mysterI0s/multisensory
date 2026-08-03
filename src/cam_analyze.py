#!/usr/bin/env python
"""
cam_analyze.py (v2) -- offline analysis of cam.npy / fg.npy / bg.npy.

No TensorFlow, no model load, no GPU. Reads what --save_npy wrote.

What changed in v2
------------------
v1 was written to compare pr.cam=False against pr.cam=True. That comparison
is now settled and the answer is: DO NOT USE --cam_hires. Measured on
translator.mp4, the two configurations produced heatmaps with a spatial
correlation of 0.0684 -- essentially unrelated. A stride change should give
you the same map at finer resolution; an uncorrelated map means the stride-1
configuration is off-distribution (the trunk and its BatchNorm statistics were
trained at stride 2), not that it is a better view of the same thing. It also
changed the audio: fg 13.2 dB below signal, bg 8.5 dB. Both audible.

So v2 is aimed at the comparison that actually matters: the SAME config at two
different timestamps, with a different person speaking in each.

For that test the interpretation of correlation INVERTS. Low correlation is
the good outcome -- it means the heatmap moved when the speaker changed. Pass
--expect moved so the tool reports it that way instead of telling you the
opposite.

v2 also adds left/right column analysis, which is the metric that decides the
SonicSight use case. Horizontal bands tell you face-vs-chyron; vertical bands
tell you left-speaker-vs-right-speaker.

Usage
-----
    # single run
    python cam_analyze.py --dir ../results/two_a

    # THE test: same config, two timestamps, different speaker in each
    python cam_analyze.py --dir ../results/two_a --compare ../results/two_b \\
        --expect moved
"""

import argparse
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None


def reduce_positive(cam):
    """Mean over time of the positive part.

    Positive-only gives the same argmax as abs (verified on both runs) but
    much higher contrast (1.41 vs 0.63 at 7x7), so it is the better display
    reduction.
    """
    return np.maximum(cam, 0.0).mean(axis=0)


def contrast(spatial):
    """std/mean, guarded.

    The raw signed CAM has a mean near zero, so this explodes on it (the probe
    once printed 43.16). Only compare contrast within the same reduction.
    """
    m = abs(float(spatial.mean()))
    if m < 1e-6:
        return float("nan")
    return float(spatial.std()) / m


def topk(spatial, k=5):
    flat = spatial.flatten()
    idx = np.argsort(flat)[::-1][:k]
    h, w = spatial.shape
    out = []
    for rank, i in enumerate(idx):
        r, c = np.unravel_index(i, spatial.shape)
        out.append(
            {
                "rank": rank + 1,
                "cell": (int(r), int(c)),
                "rel": (
                    r / float(h - 1) if h > 1 else 0.0,
                    c / float(w - 1) if w > 1 else 0.0,
                ),
                "y_px": (int(224.0 * r / h), int(224.0 * (r + 1) / h)),
                "x_px": (int(224.0 * c / w), int(224.0 * (c + 1) / w)),
                "value": float(flat[i]),
                "frac_of_max": float(flat[i] / (flat[idx[0]] + 1e-12)),
            }
        )
    return out


def print_topk(spatial, k=5, label=""):
    print("  top-%d cells %s" % (k, label))
    print("    rank  cell      rel(row,col)   y px      x px      value    %max")
    for e in topk(spatial, k):
        print(
            "    %-5d %-9s (%.2f, %.2f)   %3d-%3d   %3d-%3d   %.5f  %5.1f%%"
            % (
                e["rank"],
                str(e["cell"]),
                e["rel"][0],
                e["rel"][1],
                e["y_px"][0],
                e["y_px"][1],
                e["x_px"][0],
                e["x_px"][1],
                e["value"],
                100.0 * e["frac_of_max"],
            )
        )


def band_report(spatial, nbands=4, axis=0):
    """Energy per band. axis=0 -> horizontal bands (rows, face vs chyron).
    axis=1 -> vertical bands (columns, left speaker vs right speaker)."""
    n = spatial.shape[axis]
    total = spatial.sum() + 1e-12
    what = (
        "horizontal band (top -> bottom)"
        if axis == 0
        else "vertical band (left -> right)"
    )
    print("  energy by %s:" % what)
    edges = np.linspace(0, n, nbands + 1).astype(int)
    shares = []
    for b in range(nbands):
        lo_i, hi_i = edges[b], max(edges[b + 1], edges[b] + 1)
        sub = spatial[lo_i:hi_i] if axis == 0 else spatial[:, lo_i:hi_i]
        share = sub.sum() / total
        shares.append(share)
        p0 = int(224.0 * lo_i / n)
        p1 = int(224.0 * hi_i / n)
        tag = "y" if axis == 0 else "x"
        print(
            "    %s %3d-%3d: %5.1f%%  %s"
            % (tag, p0, p1, 100.0 * share, "#" * int(round(share * 50)))
        )
    return shares


def lr_mass(spatial):
    """Fraction of positive mass in the left vs right half of the frame."""
    w = spatial.shape[1]
    left = float(spatial[:, : w // 2].sum())
    right = float(spatial[:, (w + 1) // 2 :].sum())
    tot = left + right + 1e-12
    return left / tot, right / tot


def masked_peak(spatial, ignore_bottom):
    h, w = spatial.shape
    keep = max(int(round(h * (1.0 - ignore_bottom))), 1)
    sub = spatial[:keep]
    r, c = np.unravel_index(np.argmax(sub), sub.shape)
    return (
        (int(r), int(c)),
        (r / float(h - 1) if h > 1 else 0.0, c / float(w - 1) if w > 1 else 0.0),
        keep,
    )


def pool_to(spatial, target):
    h, w = spatial.shape
    if (h, w) == (target, target):
        return spatial
    if h % target or w % target:
        return None
    fh, fw = h // target, w // target
    return spatial.reshape(target, fh, target, fw).mean(axis=(1, 3))


def write_marked_overlay(spatial, frame_path, out_path, k=3):
    if Image is None or not os.path.exists(frame_path):
        return
    norm = spatial - spatial.min()
    norm = norm / (norm.max() + 1e-8)
    heat = (cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)
    base = Image.open(frame_path).convert("RGB")
    heat_img = Image.fromarray(heat).resize(base.size, Image.BILINEAR)
    blend = Image.blend(base, heat_img, 0.5)
    d = ImageDraw.Draw(blend)
    h, w = spatial.shape
    cw, ch = base.size[0] / float(w), base.size[1] / float(h)
    for e in topk(spatial, k):
        r, c = e["cell"]
        d.rectangle(
            [c * cw, r * ch, (c + 1) * cw, (r + 1) * ch],
            outline=(255, 255, 255),
            width=2,
        )
        d.text((c * cw + 3, r * ch + 2), str(e["rank"]), fill=(255, 255, 255))
    blend.save(out_path)
    print("  wrote", out_path)


def temporal_report(cam):
    pos = np.maximum(cam, 0.0)
    peaks = [
        np.unravel_index(np.argmax(pos[t]), pos[t].shape) for t in range(pos.shape[0])
    ]
    print("  per-timestep peaks  :", [(int(a), int(b)) for a, b in peaks])
    flat = pos.reshape(pos.shape[0], -1)
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


def analyze_dir(d, ignore_bottom, k, nbands):
    cam_path = os.path.join(d, "cam.npy")
    if not os.path.exists(cam_path):
        sys.exit("ERROR: no cam.npy in %s (re-run the probe with --save_npy)" % d)
    cam = np.load(cam_path)

    print("")
    print("=" * 70)
    print("DIR:", os.path.abspath(d))
    print("=" * 70)
    print("cam shape (T,H,W)   :", cam.shape)
    if cam.shape[0] != 8:
        print(
            "  (!) T=%d, not the default 8. This run used a non-default" % cam.shape[0]
        )
        print("      --vid_dur, so the net ran OFF-DISTRIBUTION relative to")
        print("      the 2.135 s training window. CAM magnitudes are not")
        print("      comparable to default-window runs, and any pass/fail")
        print("      verdict below is unreliable. Re-run at the default.")
    print(
        "raw min/max/std     : %.5f / %.5f / %.5f" % (cam.min(), cam.max(), cam.std())
    )
    pos_frac = float((cam > 0).mean())
    print("positive fraction   : %.3f" % pos_frac)
    if pos_frac < 0.10:
        print(
            "  (!) CAM has COLLAPSED: only %.1f%% of cells are positive."
            % (100.0 * pos_frac)
        )
        print("      joint/logits is an audio-video ALIGNMENT head, so a")
        print("      near-uniformly negative map means it judges this audio")
        print("      as not matching this video -- which is the correct")
        print("      answer when you have masked out the speaker. Treat the")
        print("      peak location here as meaningless: argmax over an")
        print("      all-negative map lands on noise, often cell (0, 0).")
        print("      Do NOT read localization out of this run.")

    spatial = reduce_positive(cam)
    print("positive contrast   : %.4f" % contrast(spatial))
    print("")
    print_topk(spatial, k, "(positive reduction)")
    print("")
    band_report(spatial, nbands, axis=0)
    print("")
    band_report(spatial, nbands, axis=1)

    l, r = lr_mass(spatial)
    print("")
    print("  LEFT half %5.1f%%   RIGHT half %5.1f%%" % (100 * l, 100 * r))

    print("")
    temporal_report(cam)

    if ignore_bottom > 0:
        cell, rel, keep = masked_peak(spatial, ignore_bottom)
        print("")
        print(
            "  peak excluding bottom %.0f%% (rows 0-%d): cell %s -> rel (%.2f, %.2f)"
            % (100.0 * ignore_bottom, keep - 1, str(cell), rel[0], rel[1])
        )

    write_marked_overlay(
        spatial,
        os.path.join(d, "center_frame.png"),
        os.path.join(d, "overlay_topk.png"),
        k=3,
    )
    return cam, spatial


def _corr(x, y):
    a, b = x.flatten(), y.flatten()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mask_cross_check(d_l, d_r):
    """The screen-mask consistency test, for two people speaking AT ONCE.

    This is the paper's Figure 1(c) experiment turned into a number.

    Grey out half the frame and the network should foreground whoever is still
    visible. Note sep_video's mapping is inverted relative to the flag name:
    --mask l greys the LEFT half, so the visible person is on the RIGHT and fg
    becomes the RIGHT speaker.

        run A (--mask l): fg = RIGHT speaker, bg = LEFT speaker
        run B (--mask r): fg = LEFT speaker,  bg = RIGHT speaker

    So if separation genuinely depends on the video, the two runs should agree
    CROSSWISE:

        fg(A) ~ bg(B)     both are the right speaker   -> HIGH
        bg(A) ~ fg(B)     both are the left speaker    -> HIGH
        fg(A) ~ fg(B)     different speakers           -> LOW
        bg(A) ~ bg(B)     different speakers           -> LOW

    Two independent runs reaching the same decomposition from opposite
    directions is hard to fake. If all four correlations are similar, the
    model is splitting the audio some fixed way and ignoring the pixels --
    the failure mode that matters, and the one you cannot detect by listening
    to a single run.
    """
    arrs = {}
    for tag, d in (("L", d_l), ("R", d_r)):
        for n in ("fg", "bg"):
            p = os.path.join(d, "%s.npy" % n)
            if not os.path.exists(p):
                print("")
                print(
                    "mask cross-check needs %s.npy in %s -- re-run with "
                    "--save_npy" % (n, d)
                )
                return
            arrs[tag + n] = np.load(p)

    shapes = set(v.shape for v in arrs.values())
    if len(shapes) != 1:
        print("")
        print("shape mismatch across runs: %s" % shapes)
        print("Both runs need the same --vid_dur and --start.")
        return

    c_cross_r = _corr(arrs["Lfg"], arrs["Rbg"])
    c_cross_l = _corr(arrs["Lbg"], arrs["Rfg"])
    c_dir_fg = _corr(arrs["Lfg"], arrs["Rfg"])
    c_dir_bg = _corr(arrs["Lbg"], arrs["Rbg"])

    print("")
    print("---- screen-mask cross-check ----")
    print("  A = --mask l run (left half greyed  -> fg should be RIGHT speaker)")
    print("  B = --mask r run (right half greyed -> fg should be LEFT speaker)")
    print("")
    print("  CROSS (should be HIGH -- same speaker reached two ways):")
    print("    fg(A) vs bg(B)   [right speaker] : %+.4f" % c_cross_r)
    print("    bg(A) vs fg(B)   [left speaker]  : %+.4f" % c_cross_l)
    print("")
    print("  DIRECT (should be LOW -- different speakers):")
    print("    fg(A) vs fg(B)                   : %+.4f" % c_dir_fg)
    print("    bg(A) vs bg(B)                   : %+.4f" % c_dir_bg)

    vals = [c_cross_r, c_cross_l, c_dir_fg, c_dir_bg]
    if any(v != v for v in vals):
        print("")
        print("  Some correlations are NaN (a channel is silent).")
        print("  Inconclusive -- check fg.wav / bg.wav are not empty.")
        return

    cross = 0.5 * (c_cross_r + c_cross_l)
    direct = 0.5 * (c_dir_fg + c_dir_bg)
    margin = cross - direct

    print("")
    print(
        "  cross mean %+.4f   direct mean %+.4f   margin %+.4f"
        % (cross, direct, margin)
    )
    print("")
    if margin > 0.30:
        print("  PASS. The two masked runs agree crosswise, so the visible")
        print("  half of the frame controls which voice lands in fg. The")
        print("  network is using the pixels. This is the result that")
        print("  justifies the single-model design.")
    elif margin > 0.10:
        print("  WEAK PASS. Cross beats direct but not decisively. Listen to")
        print("  fg.wav from both runs before committing.")
    elif margin > -0.10:
        print("  FAIL. Cross and direct are indistinguishable, so masking half")
        print("  the frame did not change which voice landed in fg. On this")
        print("  clip the model is splitting audio without regard to the")
        print("  video, and a heatmap from it would not be meaningful.")
    else:
        print("  INVERTED. Direct beats cross, so the fg/bg mapping is the")
        print("  opposite of what --mask naming implies. Still shows real")
        print("  video-dependence -- just relabel the channels.")

    print("")
    ratios = []
    for tag, fg, bg in (
        ("A", arrs["Lfg"], arrs["Lbg"]),
        ("B", arrs["Rfg"], arrs["Rbg"]),
    ):
        rf = float(np.sqrt((fg**2).mean()))
        rb = float(np.sqrt((bg**2).mean()))
        ratios.append(rf / (rb + 1e-12))
        print(
            "  %s: fg rms %.5f   bg rms %.5f   fg/bg %.3f" % (tag, rf, rb, ratios[-1])
        )

    if all(0.85 < r < 1.18 for r in ratios):
        print("  Both ratios sit near 1.0, so the energy split is even either")
        print("  way. Weak evidence the model is not committing to a speaker.")
    elif (ratios[0] - 1.0) * (ratios[1] - 1.0) < 0:
        print(
            "  The ratios STRADDLE 1.0 (%.3f vs %.3f). Flipping the mask"
            % (ratios[0], ratios[1])
        )
        print("  moved energy across the fg/bg boundary.")
        if margin > 0.30:
            print("  This CORROBORATES the pass above by an independent")
            print("  route: the correlations say the two runs swap which")
            print("  speaker is fg, and the raw energy agrees. Whichever")
            print("  speaker is louder in the mixture should give the")
            print("  higher fg/bg ratio when that speaker is the visible")
            print("  one -- check that against who you hear dominating.")
        else:
            print("  So the video IS changing the split even though the")
            print("  waveform correlations are muddy. Partly offsets the")
            print("  weak result above: pixels are read, split is unclean.")
    else:
        print(
            "  Both ratios fall on the same side of 1.0 (%.3f, %.3f), so"
            % (ratios[0], ratios[1])
        )
        print("  the mask did not move energy across the fg/bg boundary.")


def compare(d_a, spatial_a, d_b, spatial_b, expect):
    print("")
    print("=" * 70)
    print("COMPARISON  (--expect %s)" % expect)
    print("=" * 70)

    a, b = spatial_a, spatial_b
    if a.shape != b.shape:
        target = min(a.shape[0], b.shape[0])
        pa, pb = pool_to(a, target), pool_to(b, target)
        if pa is None or pb is None:
            print("shapes %s vs %s not poolable to a common grid" % (a.shape, b.shape))
            a = None
        else:
            print(
                "pooled both to %dx%d (NOTE: comparing different strides is"
                % (target, target)
            )
            print("only meaningful if you know both are in-distribution)")
            a, b = pa, pb

    if a is not None and a.shape == b.shape:
        fa, fb = a.flatten(), b.flatten()
        cor = (
            float(np.corrcoef(fa, fb)[0, 1])
            if fa.std() > 1e-9 and fb.std() > 1e-9
            else float("nan")
        )
        print("")
        print("spatial correlation : %.4f" % cor)
        print(
            "peak A %s   peak B %s"
            % (
                np.unravel_index(np.argmax(a), a.shape),
                np.unravel_index(np.argmax(b), b.shape),
            )
        )

        la, ra = lr_mass(a)
        lb, rb = lr_mass(b)
        print("")
        print("left/right mass  A: L %5.1f%% R %5.1f%%" % (100 * la, 100 * ra))
        print("left/right mass  B: L %5.1f%% R %5.1f%%" % (100 * lb, 100 * rb))
        shift = (lb - la) * 100.0
        print("LEFT-share shift A->B: %+.1f percentage points" % shift)

        print("")
        if expect == "moved":
            # Two timestamps, different speaker in each. The heatmap SHOULD
            # move, and specifically it should move horizontally toward
            # whoever is talking.
            if abs(shift) >= 15.0:
                print("  PASS-ish: mass shifted horizontally by %+.1f pts." % shift)
                print("  Check the sign matches who is speaking in each clip.")
            else:
                print(
                    "  CONCERN: mass barely moved (%+.1f pts). If a different" % shift
                )
                print("  person is speaking in each clip, the CAM is not")
                print("  tracking the speaker.")
            if cor == cor and cor > 0.9:
                print("  Also: correlation %.3f is very high -- the map is" % cor)
                print("  nearly identical across clips, i.e. static.")
        else:
            if cor == cor and cor < 0.3:
                print("  WARNING: correlation %.3f. These two configurations" % cor)
                print("  disagree about where the sound is. If the only")
                print("  difference is a conv stride, the changed one is")
                print("  probably off-distribution rather than sharper.")

    if expect == "mask":
        mask_cross_check(d_a, d_b)
        return

    for name in ("fg", "bg"):
        pa = os.path.join(d_a, "%s.npy" % name)
        pb = os.path.join(d_b, "%s.npy" % name)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        xa, xb = np.load(pa), np.load(pb)
        if xa.shape != xb.shape:
            print("%s: shape mismatch %s vs %s" % (name, xa.shape, xb.shape))
            continue
        rms_a = np.sqrt((xa**2).mean())
        rms_b = np.sqrt((xb**2).mean())
        rms_d = np.sqrt(((xa - xb) ** 2).mean())
        snr = 20.0 * np.log10((rms_a + 1e-12) / (rms_d + 1e-12))
        cor = float(np.corrcoef(xa.flatten(), xb.flatten())[0, 1])
        print("")
        print(
            "%s: rms A %.5f  rms B %.5f  (%+.1f%%)"
            % (name, rms_a, rms_b, 100.0 * (rms_b - rms_a) / (rms_a + 1e-12))
        )
        print("%s: diff is %.1f dB below signal, waveform corr %.5f" % (name, snr, cor))
        # rms alone is a poor proxy: two very different waveforms can share an
        # rms. Trust the dB figure and the correlation.
        if snr > 20:
            print("    -> inaudible in practice.")
        elif snr > 10:
            print("    -> audible on close listening.")
        else:
            print("    -> substantial difference.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--compare", default=None)
    p.add_argument(
        "--expect",
        default="same",
        choices=["same", "moved", "mask"],
        help="'moved': two timestamps, different speaker in each. "
        "'mask': --dir is the --mask l run and --compare is "
        "the --mask r run; runs the screen-mask cross-check, "
        "which is the right test when both people talk at "
        "once.",
    )
    p.add_argument("--ignore_bottom", type=float, default=0.25)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--bands", type=int, default=4)
    arg = p.parse_args()

    _, sp_a = analyze_dir(arg.dir, arg.ignore_bottom, arg.topk, arg.bands)
    if arg.compare:
        _, sp_b = analyze_dir(arg.compare, arg.ignore_bottom, arg.topk, arg.bands)
        compare(arg.dir, sp_a, arg.compare, sp_b, arg.expect)


if __name__ == "__main__":
    main()
