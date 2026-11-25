#!/bin/bash
# Track A MPS queue: each arg = "row|feats|seed|desc". Waits for the feature files, takes the shared
# MPS lock per row (released after each row, 30 s courtesy gap), skips rows whose summary exists.
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
S=$K/scripts/trackA/run_row_a.py
LOCK=$K/.mps_lock
export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${TMPDIR:-/tmp}"
HAVE=0
trap '[ $HAVE = 1 ] && rmdir $LOCK 2>/dev/null; exit 1' INT TERM HUP
for spec in "$@"; do
  IFS='|' read -r row feats seed desc <<< "$spec"
  if [ -e $K/results/${row}_summary.json ]; then echo "[$(date)] $row exists, skip"; continue; fi
  for f in ${feats//,/ }; do
    until [ -e $K/cache/$f ]; do sleep 30; done
  done
  echo "[$(date)] $row: waiting for MPS lock"
  until mkdir $LOCK 2>/dev/null; do sleep 20; done
  HAVE=1
  echo "[$(date)] $row: lock acquired; load: $(uptime)"
  $PY -I $S --row $row --feats $feats --seed $seed --desc "$desc" --device mps --save-heads > $K/logs/trackA_${row}.log 2>&1
  echo "[$(date)] $row: rc=$? $(tail -1 $K/logs/trackA_${row}.log)"
  rmdir $LOCK 2>/dev/null; HAVE=0
  sleep 30
done
echo "[$(date)] QUEUE DONE"
