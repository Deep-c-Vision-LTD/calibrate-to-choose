#!/bin/bash
# Second CPU stage: KL-only variant of the head adaptation on CPU (threads capped at 6), started
# after the first CPU queue. If the MPS queue reaches this row first, the row lock makes one exit.
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
L=$K/logs; mkdir -p $L
cd $K/scripts/msdb
export OMP_NUM_THREADS=6
while pgrep -f run_cpu_queue.sh >/dev/null; do sleep 30; done
$PY -I train_msdb.py --row msdb_adapt_kl --mode head --loss kl --dev cpu > $L/msdb_adapt_kl_cpu.log 2>&1
echo CPU_QUEUE2_DONE >> $L/msdb_adapt_kl_cpu.log
