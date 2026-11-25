#!/bin/bash
# MPS rows after msdb_adapt (restarted with a 2 s lock poll); each python process acquires and
# releases the shared MPS lock itself.
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
L=$K/logs; mkdir -p $L
SP=$K/splits
cd $K/scripts/msdb
export OMP_NUM_THREADS=6 MPS_LOCK_POLL=0.001
$PY -I train_msdb.py --row msdb_adapt_curves --mode head --loss klcc --curves > $L/msdb_adapt_curves.log 2>&1
$PY -I train_msdb.py --row msdb_adapt_grouped --mode head --loss klcc --split $SP/groups.json > $L/msdb_adapt_grouped.log 2>&1
$PY -I train_msdb.py --row msdb_adapt_pairsafe --mode head --loss klcc --split $SP/kfold_seed0_pairsafe.json > $L/msdb_adapt_pairsafe.log 2>&1
$PY -I train_msdb.py --row msdb_adapt_kl --mode head --loss kl > $L/msdb_adapt_kl_mps.log 2>&1
$PY -I train_msdb.py --row msdb_adapt --mode head --loss klcc --seeds 0,1,2 > $L/msdb_adapt_seeds.log 2>&1
echo MPS_QUEUE_B_DONE > $L/msdb_mps_queue_b.done
