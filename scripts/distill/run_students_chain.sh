#!/bin/bash
# Track B: after the Pitt pass -> label any straggler mixes, build Pitt targets, train the Pitt students on MPS.
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
D=$K/scripts/distill
L=$K/logs; mkdir -p $L
export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${TMPDIR:-/tmp}"
$PY -I $D/pitt_msdb_worker.py --finish --dev cpu --threads 4 >> $L/distill_pitt_finish2.log 2>&1
$PY -I $D/make_targets.py --part pitt > $L/distill_targets_pitt.log 2>&1 || { echo "targets failed"; exit 1; }
$PY -I $D/train_students.py --row distill_pitt --device mps > $L/distill_student_pitt.log 2>&1
$PY -I $D/train_students.py --row distill_pitt_only --device mps > $L/distill_student_pitt_only.log 2>&1
echo "[$(date)] STUDENTS CHAIN DONE" >> $L/distill_student_pitt_only.log
