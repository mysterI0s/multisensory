#!/usr/bin/env python3
"""
T4 - Diff two activation dumps produced by t3_dump_activations.py.

The point is not just "do they match" but "WHERE do they first stop matching".
Tensors are printed in forward order, so the first row that blows up tells you
which layer introduced the error. Everything after it is downstream noise.

Usage:
  python3 t4_compare_dumps.py dump_tf18.npz dump_py313.npz
  python3 t4_compare_dumps.py dump_tf18.npz dump_pytorch.npz --tol 1e-4
"""
import argparse
import sys

import numpy as np


# forward order, so the first failure is the culprit
ORDER = [
    "op:sf/conv1_1", "op:sf/conv2_1_1", "op:sf/conv3_1_1", "op:sf/conv4_1_1",
    "op:im/conv1", "op:im/conv2_1_1", "op:im/conv2_2_1",
    "im_net", "im_scale_0", "scale_0",
    "op:sf/conv5_1", "op:im/merge1", "op:im/merge2",
    "op:im/conv3_1_1", "im_scale_1", "scale_1",
    "op:im/conv4_1_1", "im_scale_2",
    "op:im/conv5_1_1", "im_scale_3", "scale_2",
    "last_conv", "cam", "logits",
]


def rank(k):
    try:
        return ORDER.index(k)
    except ValueError:
        return len(ORDER) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="gold reference dump (TF 1.8 / Py2.7)")
    ap.add_argument("test", help="dump under test")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    a = np.load(args.ref)
    b = np.load(args.test)

    keys = sorted(set(a.files) & set(b.files), key=rank)
    only_a = sorted(set(a.files) - set(b.files))
    only_b = sorted(set(b.files) - set(a.files))

    print("=" * 96)
    print("%-26s %14s %12s %12s %12s  %s" %
          ("tensor", "shape", "max|d|", "mean|d|", "rel", ""))
    print("=" * 96)

    first_bad = None
    rows = 0
    for k in keys:
        x = np.asarray(a[k], np.float64)
        y = np.asarray(b[k], np.float64)

        if x.shape != y.shape:
            print("%-26s %14s %12s %12s %12s  SHAPE MISMATCH %s vs %s" %
                  (k[:26], "-", "-", "-", "-", x.shape, y.shape))
            if first_bad is None:
                first_bad = (k, "shape mismatch %s vs %s" % (x.shape, y.shape))
            rows += 1
            continue

        d = np.abs(x - y)
        mx = float(d.max()) if d.size else 0.0
        mn = float(d.mean()) if d.size else 0.0
        scale = float(np.abs(x).max()) or 1.0
        rel = mx / scale
        bad = rel > args.tol
        if bad and first_bad is None:
            first_bad = (k, "rel %.3e > tol %.1e" % (rel, args.tol))
        print("%-26s %14s %12.3e %12.3e %12.3e  %s" %
              (k[:26], "x".join(str(v) for v in x.shape)[:14], mx, mn, rel,
               "FAIL" if bad else "ok"))
        rows += 1

    print("=" * 96)
    if only_a:
        print("only in ref : %s" % ", ".join(only_a))
    if only_b:
        print("only in test: %s" % ", ".join(only_b))

    print("")
    if first_bad is None:
        print("VERDICT: all %d tensors agree within rel tol %.1e." % (rows, args.tol))
        print("         The two stacks compute the same function.")
        sys.exit(0)
    else:
        k, why = first_bad
        print("VERDICT: FIRST DIVERGENCE AT  ->  %s   (%s)" % (k, why))
        print("")
        print("         Everything listed after this row is downstream fallout.")
        print("         Fix this layer, re-dump, and re-run before reading further.")
        hint = {
            "op:sf/conv1_1": "audio front end: 65x1 stride-4 SAME padding, or mono/stereo handling",
            "op:im/conv1": "video front end: 5x7x7 stride-2 SAME padding (asymmetric in TF)",
            "op:im/conv2_2_1": "stride-2 3x3x3 block: check for DOUBLE padding",
            "op:sf/conv5_1": "fusion: fractional_max_pool temporal alignment",
            "op:im/merge1": "fusion: tile/concat order or the 64+64 residual slice",
            "logits": "global average pool axes, or the logits conv reuse",
        }.get(k)
        if hint:
            print("         likely cause: %s" % hint)
        sys.exit(1)


if __name__ == "__main__":
    main()
