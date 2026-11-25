#!/bin/bash
# Track B MPS chain: teachers (polite lock per fold), then the MPS Pitt feature worker (polite lock in chunks).
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
D=$K/scripts/distill
L=$K/logs; mkdir -p $L
export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${TMPDIR:-/tmp}"
$PY -I $D/train_teachers.py --dev mps --folds 0,1,2,3,4 >> $L/distill_teachers_mps.log 2>&1
$PY -I $D/pitt_msdb_worker.py --dev mps >> $L/distill_pitt_msdb_mps.log 2>&1
echo "[$(date)] MPS CHAIN DONE" >> $L/distill_pitt_msdb_mps.log
