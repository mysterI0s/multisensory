# Testing the "the Python 3 migration works" premise on a real video

The claim under test is *"the conversion from Python 2.7 works."* The author's
evidence is `py_compile`, which only parses. These five tests escalate from
"does it run" to "does it compute the same function as the released model."

Run them in order. **Stop at the first failure** — every later test assumes the
earlier ones passed.

| Test | Question | Needs model? | Needs TF 1.8 oracle? |
|---|---|---|---|
| T1 | Are there Py2 semantics that compile but break? | no | no |
| T0 | Does it run end-to-end on `translator.mp4`? | yes | no |
| T2 | **Is the model actually working?** | yes | **no** |
| T3/T4 | Where exactly does it diverge? | yes | yes |
| T5 | Does separation hit the paper's numbers? | yes | no |

---

## Setup

```bash
cd multisensory-master
./download_models.sh        # -> results/nets/{shift,cam,sep,unet_pit}/net.tf-*
./download_sample_data.sh   # -> data/translator.mp4, data/crossfire.mp4
cp -r /path/to/mstest/* .
```

`translator.mp4` is the right video to test on: it is the repo's own demo, the
README documents the expected result, and reference outputs are published on
YouTube so you have a qualitative check as well as a numeric one.

---

## T1 — static audit (30 seconds, no dependencies)

```bash
python3 t1_py3_audit.py src/
```

Finds the four break classes `py_compile` cannot see: removed builtins,
moved builtins, iterator-vs-list, and classic-vs-true division. Exits non-zero
on any certain failure, so you can drop it straight into CI.

---

## T0 — smoke test

```bash
./t0_smoke.sh .
```

Imports every module (which `py_compile` never did) and then runs all four
README commands. Writes `t0_smoke.log` with full tracebacks.

This distinguishes *crashes* from *runs*. It proves nothing about correctness —
a model with scrambled padding runs perfectly happily.

---

## T2 — the pretext self-oracle ⭐ the one that actually settles it

```bash
python t2_pretext_oracle.py --video data/translator.mp4 --trials 200
```

**This is the key test, because it needs no reference implementation.**

The model was trained (Eq. 1) so that aligned audio scores higher than audio
shifted by 2.0–5.8 s. That objective is a built-in oracle: feed the same frames
with aligned vs. shifted audio and check which gets the higher alignment logit.

- **~50%** → broken. The checkpoint loads, the code runs, and the network is
  computing garbage. This is exactly the failure mode that padding, BN, and
  fusion-pooling bugs produce.
- **well above chance** → the graph is wired correctly and the weights are
  being applied. Move on to T3.

The script reports a z-score against chance so you are not eyeballing it. With
200 paired trials, chance SE is 3.5 pp, so anything above ~57% is significant.

For a stronger result use a directory of clips rather than one video:

```bash
python t2_pretext_oracle.py --video_dir /path/to/audioset_clips --trials 400
```

Run T2 under **both** stacks. Two numbers, same protocol, directly comparable.

---

## T3 + T4 — layer-wise diff against the real TF 1.8 oracle

T2 tells you *whether* it works. This tells you *where* it breaks.

```bash
docker build -f Dockerfile.tf18 -t multisensory:tf18 .

# gold reference, and freeze the exact input tensors
docker run --rm -v "$PWD":/work -w /work multisensory:tf18 \
  python t3_dump_activations.py --repo_root /work \
      --video data/translator.mp4 \
      --save_input fixed_input.npz --out dump_tf18.npz

# migrated stack, byte-identical input
python3 t3_dump_activations.py --repo_root . \
      --load_input fixed_input.npz --out dump_py313.npz

python3 t4_compare_dumps.py dump_tf18.npz dump_py313.npz --tol 1e-4
```

`t3` also runs the forward pass twice and reports self-consistency. TF's
eval-mode `fractional_max_pool` is seeded and should be bit-identical; if it is
not, fix that before trusting any diff.

`t4` prints tensors in **forward order** and names the first divergence, so you
get "broke at `im/conv1`" rather than "the logits are wrong." It maps common
first-failure points to their likely cause.

The same two scripts validate the PyTorch port later — dump from the PyTorch
model into the same key names and diff against `dump_tf18.npz`.

---

## T5 — task-level numbers

If T2 and T4 pass, confirm the released model reproduces its published results:

- **Table 2**, on/off-screen separation: **11.4 dB on-screen / 7.0 dB off-screen**
- **Section 5**, pretext accuracy on held-out AudioSet: **59.9%**
- **Table 1**, UCF-101 split 1 fine-tuned: **82.1%**

Build synthetic mixtures from disjoint-speaker VoxCeleb pairs (the paper's
72/8/20 split), run `sep_video.py --model full`, and compute SDR. Anything near
0 dB means the separation head is not working regardless of what T0 said.

---

## Interpreting the outcome

| T0 | T2 | Meaning |
|---|---|---|
| fail | — | Migration incomplete. Fix the runtime errors T1 listed. |
| pass | ~50% | **Worst case.** Runs cleanly, computes nonsense. Only T2 catches this. |
| pass | above chance | Migration is real. Use T3/T4 to quantify residual drift. |

The middle row is the whole reason to do this. A migration that runs is not a
migration that works, and `py_compile` cannot tell the two apart.

---

# SonicSight server integration

This repo is also imported directly by the SonicSight backend
(`SonicSightBackend/src/engines/multisensory_engine.py` adds `src/` to its
path). What the server relies on:

- `sep_params.full()` — the LOCKED 2.135 s window (`num_samples` 44144 @ 21000 Hz,
  63 frames, `spec_len` 128) and `input_rms`, which now lives in the params.
- `sep_video.NetClf.predict_with_cam()` — separation + the 7×7 alignment CAM in
  one `sess.run`. The CAM postprocess on the server side reproduces
  `cam_analyze.reduce_positive` exactly so live output stays comparable to the
  probe tooling.
- `sep_video` module-level defaults: `TF_CUDNN_WORKSPACE_LIMIT_IN_MB=512`
  (do not remove — uncapped, the allocator holds ~2.9 GB for a ~0.5 GB working
  set) and `_session_config()` reading `MS_GPU_ALLOW_GROWTH` /
  `MS_GPU_MEM_FRACTION`, which is what lets TF share a 6 GB card with PyTorch.
- `MULTISENSORY_RESULTS` (default `../results`) points every checkpoint path;
  the server passes an absolute `model_path` explicitly.
- `sep_video` no longer imports `pylab`/`aolib.imtable` at module level, so the
  server does not drag in a plotting stack. The dead `predict_cam`/`find_cam`
  (cam_v1) paths were removed; use `sep_cam_probe.py` for offline CAM work.

**Hard-won rules — do not deviate** (evidence in the project handoff):
`vid_dur` stays 2.135 (4.27 flipped the decisive mask test from PASS +0.8248 to
a plausible-looking FAIL on the identical clip); `pr.cam` stays False (the
14×14 CAM correlates 0.0684 with the validated 7×7); the multisensory streaming
hop is 250 ms (inference is 142–148 ms; a 125 ms hop compounds lag).

The evidence base (`t0_smoke.sh`, `t1`–`t4`, `Dockerfile.tf18`,
`src/sep_cam_probe.py`, `src/cam_analyze.py`, `src/gpu_doctor.py`) is how we
prove the model still works — it is not clutter. `multisensory_pytorch/` is an
unvalidated port skeleton with no checkpoint; the server deliberately uses the
TF path that carries the measurement record.
