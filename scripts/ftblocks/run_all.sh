#!/bin/bash
# ftblocks: the whole pre-registered pipeline (PLAN.json + PLAN_AMENDMENT_1.json), one stage after the other.
# Every stage is resumable (per-run files in cache/ftblocks/runs); the shared MPS lock (.mps_lock)
# is taken per run inside train_ftblocks.py and released after each run. Nothing here kills or renices anything.
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
S=$K/scripts/ftblocks
PY="${PY:-python3}"
export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${TMPDIR:-/tmp}" || exit 9
echo "[$(date)] run_all start (pid $$)"
$PY -I $S/train_ftblocks.py nested >> $K/logs/ftblocks_nested.log 2>&1 || { rc=$?; echo "[$(date)] nested failed rc=$rc"; exit 1; }
echo "[$(date)] nested done"
$PY -I $S/train_ftblocks.py global >> $K/logs/ftblocks_global.log 2>&1 || { rc=$?; echo "[$(date)] global failed rc=$rc"; exit 2; }
echo "[$(date)] global done"
D=$($PY -I $S/train_ftblocks.py pitt-depth 2>/dev/null | tail -1)
echo "[$(date)] optional row: pitt-depth = $D"
if [ "$D" = "8" ] || [ "$D" = "10" ]; then
  F=$K/cache/ftblocks/pitt_b$(printf %02d $D)_fp16.npy
  if [ ! -f "$F" ]; then
    $PY -I $S/cache_blocks.py --set pitt --depths $D --dtype fp16 >> $K/logs/ftblocks_cache_pitt.log 2>&1 || { rc=$?; echo "[$(date)] pitt cache failed rc=$rc"; exit 3; }
  fi
fi
$PY -I $S/train_ftblocks.py distill >> $K/logs/ftblocks_distill.log 2>&1 || { rc=$?; echo "[$(date)] distill failed rc=$rc"; exit 4; }
echo "[$(date)] distill stage done"
$PY -I $S/train_ftblocks.py assemble >> $K/logs/ftblocks_assemble.log 2>&1
$PY -I $S/stats_ftblocks.py >> $K/logs/ftblocks_stats.log 2>&1 || { rc=$?; echo "[$(date)] stats failed rc=$rc"; exit 5; }
$PY -I $S/overview_ftblocks.py >> $K/logs/ftblocks_overview.log 2>&1 || { rc=$?; echo "[$(date)] overview failed rc=$rc"; exit 6; }
echo "[$(date)] run_all done"
