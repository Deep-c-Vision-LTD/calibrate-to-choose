#!/bin/bash
# Track A cost benchmark (same protocol as scripts/compile/run_cost.sh: batch 1, float32, 6 threads,
# 3 warm-ups, median of 30). Usage: [COST_TAG=_runN] run_cost_a.sh <cpu|mps|both> <variant> [<variant> ...]
# B448 (= existing 'ours' model) is re-measured in the same session as the contention reference.
# Output: results/cost_raw/trackA_<variant><TAG>_<dev>_p<k>.json (k = position in the list).
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
B=$K/scripts/trackA/cost_bench_a.py
OUT=$K/results/cost_raw
export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${TMPDIR:-/tmp}"
MODE=$1; shift
TAG=${COST_TAG:-}
run_dev() {
  dev=$1; shift; n=0
  for v in "$@"; do
    n=$((n+1)); f=$OUT/trackA_${v}${TAG}_${dev}_p${n}
    echo "[$(date)] $dev $v (pass $n); load: $(uptime)"
    $PY -I -W ignore $B --model ours_v --variant $v --device $dev --threads 6 --warmup 3 --runs 30 \
      --out $f.json > $f.stdout 2>&1 || echo "$dev $v FAILED rc=$?"
    grep -E '"median"|"GFLOPs"' $f.json | head -3
  done
}
if [ "$MODE" = cpu ] || [ "$MODE" = both ]; then run_dev cpu "$@"; fi
if [ "$MODE" = mps ] || [ "$MODE" = both ]; then
  LOCK=$K/.mps_lock
  echo "[$(date)] waiting for MPS lock"
  until mkdir $LOCK 2>/dev/null; do sleep 20; done
  trap 'rmdir $LOCK 2>/dev/null' EXIT INT TERM HUP
  echo "[$(date)] MPS lock acquired; load: $(uptime)"
  run_dev mps "$@"
  rmdir $LOCK 2>/dev/null; trap - EXIT INT TERM HUP
fi
echo "[$(date)] COST DONE"
