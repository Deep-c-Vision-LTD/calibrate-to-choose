#!/bin/bash
# Sequential MPS rows; each python process acquires/releases the shared MPS lock itself.
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
L=$K/logs; mkdir -p $L
SP=$K/splits
cd $K/scripts/msdb
export OMP_NUM_THREADS=6
$PY -I train_msdb.py --row msdb_adapt --mode head --loss klcc > $L/msdb_adapt.log 2>&1
$PY -I train_msdb.py --row msdb_adapt_curves --mode head --loss klcc --curves > $L/msdb_adapt_curves.log 2>&1
$PY -I train_msdb.py --row msdb_adapt_grouped --mode head --loss klcc --split $SP/groups.json > $L/msdb_adapt_grouped.log 2>&1
$PY -I train_msdb.py --row msdb_adapt_pairsafe --mode head --loss klcc --split $SP/kfold_seed0_pairsafe.json > $L/msdb_adapt_pairsafe.log 2>&1
$PY -I train_msdb.py --row msdb_adapt_kl --mode head --loss kl > $L/msdb_adapt_kl.log 2>&1
echo MPS_QUEUE_DONE >> $L/msdb_adapt_kl.log
