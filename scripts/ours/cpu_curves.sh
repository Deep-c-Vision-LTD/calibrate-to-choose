#!/bin/zsh
# CPU part of the matched-budget curves (no MPS lock needed): n in {25,50,100,200,400}, both inits;
# n = 800 runs on the GPU queue (mps_queue.py). Threads capped at 6.
KB="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $KB/scripts/ours
for init in scratch warm; do
  echo "$(date '+%F %T') start cpu curves $init"
  $PY -I run_curves.py --init $init --sizes 25 50 100 200 400 --device cpu || echo "$(date '+%F %T') FAILED cpu curves $init"
done
echo "$(date '+%F %T') cpu curves done"
