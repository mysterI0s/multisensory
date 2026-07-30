#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T3 - Deterministic activation dump.

Run this ONCE inside the original Python 2.7 / TF 1.8 container (the gold
reference) and ONCE inside the migrated Python 3.13 stack, on the SAME video
with the SAME frame/sample slice. Then diff the two .npz files with
t4_compare_dumps.py.

Input is fixed and synthetic-free: we take a deterministic slice of a real
video (no random crop, no augmentation, train=False) so the only difference
between the two runs is the framework.

Also runs the forward pass TWICE and reports self-consistency. TF's eval-mode
fractional_max_pool is seeded and should be bit-identical across runs. If the
two passes differ, the fusion pooling is nondeterministic and every downstream
comparison is meaningless until you fix that first.

Usage:
  python t3_dump_activations.py --video ../data/translator.mp4 --out dump_tf18.npz
  python t3_dump_activations.py --video ../data/translator.mp4 --out dump_py313.npz
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np


def center_crop(ims, dim):
    h, w = ims.shape[1], ims.shape[2]
    y0 = (h - dim) // 2
    x0 = (w - dim) // 2
    return ims[:, y0:y0 + dim, x0:x0 + dim, :]


def as_stereo(samples):
    s = np.asarray(samples)
    if s.ndim == 1:
        s = s[:, None]
    if s.shape[1] == 1:
        s = np.concatenate([s, s], axis=1)
    return s[:, :2].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="results/nets/shift/net.tf-650000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_input", default=None,
                    help="write the exact ims/samples used, so the other "
                         "container can reuse byte-identical input")
    ap.add_argument("--load_input", default=None,
                    help="reuse an input .npz written by --save_input")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.repo_root, "src"))
    import tensorflow as tf
    if hasattr(tf, "compat") and hasattr(tf.compat, "v1") and not hasattr(tf, "placeholder"):
        tf = tf.compat.v1
        tf.disable_v2_behavior()

    import shift_params
    import shift_net

    pr = shift_params.shift_v1()

    # ---------------- input ----------------
    if args.load_input:
        d = np.load(args.load_input)
        ims, samples = d["ims"], d["samples"]
        print("loaded fixed input from %s" % args.load_input)
    else:
        import aolib.util as ut
        import aolib.img as ig
        import aolib.sound as sound

        dur = pr.vid_dur + 2.0 / 30
        with ut.VidFrames(args.video, sound=True, start_time=0.0,
                          end_time=dur, fps=pr.fps) as (im_files, snd_file):
            frames = np.array([ig.load(f) for f in im_files])
            snd = sound.load_sound(snd_file).normalized()
            samples = as_stereo(snd.samples)[: pr.num_samples][np.newaxis]
        ims = center_crop(frames[: pr.sampled_frames], pr.crop_im_dim)
        ims = ims[np.newaxis].astype(np.uint8)

    if args.save_input:
        np.savez_compressed(args.save_input, ims=ims, samples=samples)
        print("wrote fixed input to %s" % args.save_input)

    print("ims     %s %s" % (ims.shape, ims.dtype))
    print("samples %s %s" % (samples.shape, samples.dtype))

    # ---------------- graph ----------------
    ims_ph = tf.placeholder(
        tf.uint8, [1, pr.sampled_frames, pr.crop_im_dim, pr.crop_im_dim, 3]
    )
    samples_ph = tf.placeholder(tf.float32, [1, pr.num_samples, 2])
    net = shift_net.make_net(ims_ph, samples_ph, pr, reuse=False, train=False)

    fetches = {
        "logits": net.logits,
        "cam": net.cam,
        "last_conv": net.last_conv,
        "im_net": net.im_net,
    }
    for i, t in enumerate(net.scales):
        fetches["scale_%d" % i] = t
    for i, t in enumerate(net.im_scales):
        fetches["im_scale_%d" % i] = t

    # named intermediates, so a diff points at a specific layer
    g = tf.get_default_graph()
    wanted = [
        "sf/conv1_1", "sf/conv2_1_1", "sf/conv3_1_1", "sf/conv4_1_1", "sf/conv5_1",
        "im/conv1", "im/conv2_1_1", "im/conv2_2_1",
        "im/merge1", "im/merge2",
        "im/conv3_1_1", "im/conv4_1_1", "im/conv5_1_1",
    ]
    for scope in wanted:
        for suffix in ("/Relu:0", "/BiasAdd:0", "/convolution:0", "/Conv2D:0"):
            name = scope + suffix
            try:
                fetches["op:" + scope] = g.get_tensor_by_name(name)
                break
            except (KeyError, ValueError):
                continue

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(args.repo_root, model_path)
    print("restoring %s" % model_path)
    tf.train.Saver().restore(sess, model_path)
    g.finalize()

    feed = {ims_ph: ims, samples_ph: samples}
    run_a = sess.run(fetches, feed)
    run_b = sess.run(fetches, feed)

    # ---------------- self-consistency ----------------
    print("")
    print("determinism check (same input, two forward passes):")
    worst = 0.0
    for k in sorted(run_a):
        d = float(np.max(np.abs(np.asarray(run_a[k], np.float64)
                                - np.asarray(run_b[k], np.float64))))
        worst = max(worst, d)
        flag = "" if d == 0.0 else "   <-- NONDETERMINISTIC"
        print("  %-24s max|a-b| = %.3e%s" % (k, d, flag))
    if worst > 0.0:
        print("")
        print("  WARNING: forward pass is not deterministic in eval mode.")
        print("  Fix the fusion fractional_max_pool before trusting any diff.")

    out = {k: np.asarray(v) for k, v in run_a.items()}
    out["_logits_scalar"] = np.asarray(run_a["logits"]).reshape(-1)
    np.savez_compressed(args.out, **out)
    print("")
    print("wrote %s (%d tensors)" % (args.out, len(out)))
    print("logits = %s" % np.asarray(run_a["logits"]).reshape(-1))


if __name__ == "__main__":
    main()
