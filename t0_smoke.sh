#!/usr/bin/env bash
# T0 - Smoke test. Does the migrated code run AT ALL on a real video?
#
# No numerical claims here. This only separates "crashes" from "produces
# output", and captures the traceback if it crashes. Run it first: if T0
# fails there is no point running T2-T4.
#
# Usage:  ./t0_smoke.sh /path/to/multisensory-master

set -u

REPO="${1:-.}"
LOG="${REPO}/t0_smoke.log"
PY="${PY:-python3}"

cd "$REPO" || { echo "cannot cd to $REPO"; exit 2; }

: > "$LOG"

run_case () {
  local name="$1"; shift
  echo "" | tee -a "$LOG"
  echo "=================================================================" | tee -a "$LOG"
  echo "CASE: $name" | tee -a "$LOG"
  echo "CMD : $*" | tee -a "$LOG"
  echo "=================================================================" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then
    echo "RESULT: PASS  ($name)" | tee -a "$LOG"
    return 0
  else
    local rc=$?
    echo "RESULT: FAIL rc=$rc  ($name)" | tee -a "$LOG"
    echo "--- last 25 lines ---" | tee -a "$LOG"
    tail -25 "$LOG"
    return 1
  fi
}

echo "repo    : $(pwd)"        | tee -a "$LOG"
echo "python  : $($PY -V 2>&1)" | tee -a "$LOG"
echo "ffmpeg  : $(ffmpeg -version 2>/dev/null | head -1)" | tee -a "$LOG"

if [ ! -d results/nets ]; then
  echo ""
  echo "!! results/nets missing - run ./download_models.sh first"
fi
if [ ! -f data/translator.mp4 ]; then
  echo "!! data/translator.mp4 missing - run ./download_sample_data.sh first"
fi

FAILED=0

# 1. Can every module even be IMPORTED? py_compile never did this.
for m in tfutil shift_params shift_net sep_params sourcesep shift_dset sep_dset aolib.util aolib.img aolib.sound; do
  run_case "import $m" $PY -c "import sys; sys.path.insert(0,'src'); import $m; print('imported', '$m')" || FAILED=$((FAILED+1))
done

# 2. The repo's own flagship example.
( cd src && run_case "shift_example.py" $PY shift_example.py ) || FAILED=$((FAILED+1))

# 3. The README separation command.
( cd src && run_case "sep_video.py full" $PY sep_video.py ../data/translator.mp4 \
      --model full --duration_mult 4 --out ../results/ ) || FAILED=$((FAILED+1))

# 4. CAM path.
( cd src && run_case "sep_video.py --cam" $PY sep_video.py ../data/translator.mp4 \
      --model full --cam --out ../results/ ) || FAILED=$((FAILED+1))

# 5. Blind u-net PIT baseline.
( cd src && run_case "sep_video.py unet_pit" $PY sep_video.py ../data/translator.mp4 \
      --model unet_pit --duration_mult 4 --out ../results/ ) || FAILED=$((FAILED+1))

echo ""
echo "================================================================="
echo "T0 SUMMARY: $FAILED case(s) failed. Full log: $LOG"
echo "================================================================="
exit $(( FAILED > 0 ? 1 : 0 ))
