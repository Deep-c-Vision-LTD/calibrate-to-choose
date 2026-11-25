#!/bin/bash
# Track A: CPU feature caches, one after another (6 threads each).
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
S=$K/scripts/trackA/cache_feats_a.py
export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${TMPDIR:-/tmp}"
for spec in "$@"; do
  set -- $spec
  echo "[$(date)] cache $spec; load: $(uptime)"
  $PY -I -W ignore $S --backbone $1 --size $2 --store $3 --threads 6 || echo "cache $spec FAILED rc=$?"
done
echo "[$(date)] CACHE CHAIN DONE"
