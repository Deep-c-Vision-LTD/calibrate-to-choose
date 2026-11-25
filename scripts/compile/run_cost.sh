#!/bin/bash
# Measured cost benchmark: CPU passes (no lock), then MPS passes under the shared MPS lock.
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
B=$K/scripts/compile/cost_bench.py
OUT=$K/results/cost_raw
mkdir -p $OUT
export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${TMPDIR:-/tmp}"
# optional: EXTRA="--readout_ckpt <readout weights>" (weights do not affect cost)
runs_for() { if [ "$1" = "msdb" ]; then echo 20; else echo 30; fi; }
echo "[$(date)] CPU passes start; load: $(uptime)"
for m in ours iie sum msdb; do
  echo "[$(date)] cpu $m"
  $PY -I -W ignore $B --model $m --device cpu --threads 6 --warmup 3 --runs $(runs_for $m) --out $OUT/${m}_cpu.json $EXTRA \
    > $OUT/${m}_cpu.stdout 2>&1 || echo "cpu $m FAILED rc=$?"
done
echo "[$(date)] waiting for MPS lock"
LOCK=$K/.mps_lock
until mkdir $LOCK 2>/dev/null; do sleep 20; done
trap 'rmdir $LOCK 2>/dev/null' EXIT INT TERM
echo "[$(date)] MPS lock acquired; load: $(uptime)"
for m in ours iie sum msdb; do
  echo "[$(date)] mps $m"
  $PY -I -W ignore $B --model $m --device mps --threads 6 --warmup 3 --runs $(runs_for $m) --out $OUT/${m}_mps.json $EXTRA \
    > $OUT/${m}_mps.stdout 2>&1 || echo "mps $m FAILED rc=$?"
done
rmdir $LOCK 2>/dev/null
trap - EXIT INT TERM
echo "[$(date)] COST DONE"
