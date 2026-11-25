#!/bin/bash
# CPU rows (no MPS lock); threads capped at 6.
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
L=$K/logs; mkdir -p $L
cd $K/scripts/msdb
export OMP_NUM_THREADS=6
$PY -I score_zeroshot.py > $L/msdb_zeroshot.log 2>&1
$PY -I train_msdb.py --row msdb_adapt_dsp --mode dsp --loss klcc --dev cpu > $L/msdb_adapt_dsp.log 2>&1
$PY -I train_msdb.py --row msdb_adapt_dsp_curves --mode dsp --loss klcc --curves --dev cpu > $L/msdb_adapt_dsp_curves.log 2>&1
echo CPU_QUEUE_DONE >> $L/msdb_adapt_dsp_curves.log
