#!/bin/bash
# Build-phase checks of the ftblocks track (no long run): MPS replay check of the block cache, the N=0
# reproduction smoke, 20-step N=4 / N=2 smokes, and the timing bench. Every Python step takes the shared
# MPS lock (mkdir .mps_lock) for its own duration and releases it (also on failure).
#   nohup bash run_smoke_bench.sh > ../../logs/ftblocks_smoke_bench.log 2>&1 &
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
S=$K/scripts/ftblocks
export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${TMPDIR:-/tmp}"
echo "[$(date)] mps replay check"; $PY -I $S/cache_blocks.py --set add --verify-only --mps || echo "FAILED mps check"
echo "[$(date)] smoke n0";  $PY -I $S/train_ftblocks.py smoke-n0 || echo "FAILED smoke n0"
echo "[$(date)] smoke ft N4"; $PY -I $S/train_ftblocks.py smoke-ft --n-ft 4 --lr-bb 3e-5 --steps 20 || echo "FAILED smoke N4"
echo "[$(date)] smoke ft N2"; $PY -I $S/train_ftblocks.py smoke-ft --n-ft 2 --lr-bb 3e-5 --steps 20 || echo "FAILED smoke N2"
echo "[$(date)] bench";     $PY -I $S/bench_ftblocks.py --steps 30 --bf16 || echo "FAILED bench"
echo "[$(date)] ALL DONE"
