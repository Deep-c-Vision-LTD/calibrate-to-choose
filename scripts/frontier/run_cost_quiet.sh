#!/bin/bash
# Frontier compile: quiet-machine re-measurement of the Track A variants (copy of
# scripts/trackA/run_cost_a.sh; same bench script scripts/trackA/cost_bench_a.py UNCHANGED, same protocol:
# batch 1, float32, 6 threads, 3 warm-ups, median of 30). Only the output names differ so that no
# existing Track A raw file is overwritten:
#   results/cost_raw/frontier_quiet_<variant>_<dev>_p<k>.json  (k = position in the list)
# B448 (= the existing 'ours' model of cost_table.json, measured on a quiet machine) is measured first
# and last as the same-session reference. Usage: run_cost_quiet.sh <cpu|mps|both> <variant> ...
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
B=$K/scripts/trackA/cost_bench_a.py
OUT=$K/results/cost_raw
export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${TMPDIR:-/tmp}"
MODE=$1; shift
run_dev() {
  dev=$1; shift; n=0
  for v in "$@"; do
    n=$((n+1)); f=$OUT/frontier_quiet_${v}_${dev}_p${n}
    if [ -e $f.json ]; then echo "exists, skip: $f.json"; continue; fi
    echo "[$(date)] $dev $v (pass $n); load: $(uptime)"
    $PY -I -W ignore $B --model ours_v --variant $v --device $dev --threads 6 --warmup 3 --runs 30 \
      --out $f.json > $f.stdout 2>&1 || echo "$dev $v FAILED rc=$?"
    echo "[$(date)] done $dev $v; load: $(uptime)"
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
