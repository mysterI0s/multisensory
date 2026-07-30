#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T2 - Pretext-task self-oracle.  THE decisive test.

Why this one matters: it needs NO reference implementation and NO ground-truth
separation. The model's own training objective is the oracle.

Owens & Efros Eq. 1 trained the net so that
    p(y=1 | I, A_aligned)  >  p(y=1 | I, A_shifted)
where A_shifted is the same audio displaced by 2.0-5.8 s. If the port is
correct, the alignment logit for aligned audio beats the shifted logit on a
clear majority of trials. If padding / BN / fusion-pooling is broken, the
feature map no longer means anything and the score collapses to chance.

This is a 2AFC (paired) test, which is easier than the paper's single-clip
binary decision at 59.9%. On clean talking-head footage a correct model should
land well above that. What you are looking for is the SIGN of the result:

    ~50%          -> broken. The weights are not being applied correctly.
    >65%          -> the graph is wired correctly and the checkpoint loaded.

With N paired trials, chance SE = 50/sqrt(N) percentage points.
  N=100 -> SE 5.0pp, need >60% for p<0.05
  N=200 -> SE 3.5pp, need >57% for p<0.05
  N=400 -> SE 2.5pp, need >55% for p<0.05

Runs unmodified under BOTH:
  - the original Python 2.7 / TF 1.8 stack
  - the migrated Python 3.13 / tf.compat.v1 stack
so you can run it in each and compare the two numbers directly.

Usage:
  python t2_pretext_oracle.py --video ../data/translator.mp4 --trials 200
  python t2_pretext_oracle.py --video_dir /path/to/clips --trials 400
"""
from __future__ import print_function

import argparse
import glob
import json
import math
import os
import random
import sys

import numpy as np


def add_src_to_path(repo_root):
    src = os.path.join(repo_root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def center_crop(ims, dim):
    """ims: [T, H, W, 3] -> [T, dim, dim, 3] center crop."""
    h, w = ims.shape[1], ims.shape[2]
    y0 = (h - dim) // 2          # // on purpose: Py2 and Py3 agree
    x0 = (w - dim) // 2
    return ims[:, y0:y0 + dim, x0:x0 + dim, :]


def as_stereo(samples):
    """[N] or [N,1] -> [N,2]. 39% of in-the-wild tracks are mono (paper 6.2)."""
    s = np.asarray(samples)
    if s.ndim == 1:
        s = s[:, None]
    if s.shape[1] == 1:
        s = np.concatenate([s, s], axis=1)
    return s[:, :2].astype(np.float32)


def load_window(path, pr, start_time, total_dur):
    """Load one long window: all frames + all audio, so we can slice both."""
    import aolib.util as ut
    import aolib.img as ig
    import aolib.sound as sound

    with ut.VidFrames(
        path, sound=True, start_time=start_time,
        end_time=start_time + total_dur, fps=pr.fps,
    ) as (im_files, snd_file):
        # NOTE: list comprehension, not map(). map() is an iterator in Py3 and
        # np.array(map(...)) silently produces a 0-d object array.
        ims = np.array([ig.load(f) for f in im_files])
        snd = sound.load_sound(snd_file).normalized()
        samples = as_stereo(snd.samples)
    return ims, samples


def build_graph(pr, tf, shift_net):
    ims_ph = tf.placeholder(
        tf.uint8, [1, pr.sampled_frames, pr.crop_im_dim, pr.crop_im_dim, 3]
    )
    samples_ph = tf.placeholder(tf.float32, [1, pr.num_samples, 2])
    net = shift_net.make_net(ims_ph, samples_ph, pr, reuse=False, train=False)
    return ims_ph, samples_ph, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--video", default=None, help="single video file")
    ap.add_argument("--video_dir", default=None, help="directory of videos")
    ap.add_argument("--model", default="results/nets/shift/net.tf-650000")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="t2_results.json")
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()

    add_src_to_path(args.repo_root)
    import tensorflow as tf
    if hasattr(tf, "compat") and hasattr(tf.compat, "v1") and not hasattr(tf, "placeholder"):
        tf = tf.compat.v1
        tf.disable_v2_behavior()

    import shift_params
    import shift_net

    pr = shift_params.shift_v1()
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(args.repo_root, model_path)

    videos = []
    if args.video:
        videos = [args.video]
    elif args.video_dir:
        for ext in ("*.mp4", "*.mkv", "*.webm", "*.avi"):
            videos.extend(sorted(glob.glob(os.path.join(args.video_dir, ext))))
    if not videos:
        print("ERROR: pass --video or --video_dir")
        sys.exit(2)

    total_dur = 10.1                     # matches shift_params total_dur
    shift_lo, shift_hi = 2.0, 5.8        # paper section 3
    rng = random.Random(args.seed)

    print("building graph ...")
    ims_ph, samples_ph, net = build_graph(pr, tf, shift_net)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("restoring %s" % model_path)
    tf.train.Saver().restore(sess, model_path)
    tf.get_default_graph().finalize()

    n_correct = 0
    n_done = 0
    records = []

    while n_done < args.trials:
        vid = rng.choice(videos)
        try:
            # pick a random start far enough from the end
            start = rng.uniform(0.0, 2.0)
            ims_full, samples_full = load_window(vid, pr, start, total_dur)
        except Exception as e:
            print("  skip %s (%s)" % (os.path.basename(vid), e))
            continue

        if ims_full.shape[0] < pr.sampled_frames:
            print("  skip %s (too few frames)" % os.path.basename(vid))
            continue

        need = int(shift_hi * pr.samp_sr) + pr.num_samples
        if samples_full.shape[0] < need:
            print("  skip %s (audio too short)" % os.path.basename(vid))
            continue

        ims = center_crop(ims_full[: pr.sampled_frames], pr.crop_im_dim)
        ims = ims[np.newaxis].astype(np.uint8)

        a_aligned = samples_full[: pr.num_samples][np.newaxis]

        off = int(rng.uniform(shift_lo, shift_hi) * pr.samp_sr)
        a_shifted = samples_full[off: off + pr.num_samples][np.newaxis]

        l_aligned = float(
            np.squeeze(sess.run(net.logits, {ims_ph: ims, samples_ph: a_aligned}))
        )
        l_shifted = float(
            np.squeeze(sess.run(net.logits, {ims_ph: ims, samples_ph: a_shifted}))
        )

        ok = l_aligned > l_shifted
        n_correct += int(ok)
        n_done += 1
        records.append(
            {
                "video": os.path.basename(vid),
                "start": round(start, 3),
                "shift_s": round(off / float(pr.samp_sr), 3),
                "logit_aligned": l_aligned,
                "logit_shifted": l_shifted,
                "correct": bool(ok),
            }
        )
        if n_done % 10 == 0:
            print(
                "  %4d/%d  running acc = %.1f%%"
                % (n_done, args.trials, 100.0 * n_correct / n_done)
            )

    acc = 100.0 * n_correct / max(n_done, 1)
    se = 50.0 / math.sqrt(max(n_done, 1))
    z = (acc - 50.0) / se if se > 0 else 0.0

    print("")
    print("=" * 66)
    print("T2 PRETEXT SELF-ORACLE")
    print("=" * 66)
    print("  trials              : %d" % n_done)
    print("  2AFC accuracy       : %.1f%%" % acc)
    print("  chance SE           : %.1f pp" % se)
    print("  z vs chance         : %.2f" % z)
    print("")
    if z < 2.0:
        print("  VERDICT: INDISTINGUISHABLE FROM CHANCE -> the model is NOT working.")
        print("           The checkpoint is loading but the graph is not")
        print("           reproducing the trained function. Look at padding,")
        print("           BN moving stats, and the fusion fractional-max-pool.")
    else:
        print("  VERDICT: significantly above chance -> alignment head is live.")
        print("           Proceed to T3 (layer-wise diff) to quantify drift.")
    print("=" * 66)

    with open(args.out, "w") as f:
        json.dump(
            {"accuracy": acc, "trials": n_done, "z": z, "records": records}, f, indent=2
        )
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
