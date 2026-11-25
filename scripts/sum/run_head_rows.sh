#!/bin/bash
# Head-only SUM adaptation rows on CPU (no MPS lock needed): SUM loss, then KL + (1 - CC).
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
export OMP_NUM_THREADS=6
for L in sum klcc; do
  ROW=sum_ft_head; [ "$L" = "klcc" ] && ROW=sum_ft_head_klcc
  echo "[$(date)] start $ROW"
  $PY -I -W ignore $K/scripts/sum/ft_head.py --scripts $K/scripts/sum --code $K/third_party/SUM/code \
    --weights $K/third_party/SUM/weights --data ${ADD1000_DIR:-$K/data/ADD1000}/flat \
    --split $K/splits/kfold_seed0.json --dec_cache $K/cache/sum_dec --results $K/results --row $ROW \
    --loss $L --device cpu --threads 6 || echo "$ROW FAILED rc=$?"
done
echo "[$(date)] HEAD ROWS DONE"
