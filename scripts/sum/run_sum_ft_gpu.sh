#!/bin/bash
# SUM in-domain fine-tune on one CUDA GPU: rows sum_ft and sum_ft_klcc (five folds) and the learning-curve runs
# behind sum_ft_curve. Provider-neutral: it calls scripts/sum/ft_sum_folds.py directly with the arguments recorded in
# results/sum_ft_summary.json, results/sum_ft_klcc_summary.json and results/sum_ft_curve_summary.json. The paper's
# runs used one cloud NVIDIA A10G 24 GB GPU (PyTorch 2.3.0, CUDA 12.1, Python 3.11) and SUM's selective scan in pure
# PyTorch (sum_common.py), so no mamba-ssm build is needed.
#
#   scripts/sum/run_sum_ft_gpu.sh --tag sum_ft                    # SUM's own loss without the NSS term, folds 0-4
#   scripts/sum/run_sum_ft_gpu.sh --tag sum_ft_klcc --loss klcc   # KL + (1 - CC), folds 0-4
#   scripts/sum/run_sum_ft_gpu.sh --tag sum_curve_a --curve 25:0,1,2/50:0,1,2/100:0,1,2/200:0
#   scripts/sum/run_sum_ft_gpu.sh --tag sum_curve_b --curve 200:1,2/400:0
#
# Outputs go to runs/<tag>/ckpt/<tag>/. Then write the result files:
#   python -I scripts/sum/finalize_ft.py --scripts scripts/sum --work runs/sum_ft/ckpt/sum_ft --results results \
#       --split splits/kfold_seed0.json --tag sum_ft --command "scripts/sum/run_sum_ft_gpu.sh --tag sum_ft"
#   python -I scripts/sum/finalize_curve.py --runs runs/sum_curve_a/ckpt/sum_curve_a runs/sum_curve_b/ckpt/sum_curve_b \
#       --jobs sum_curve_a,sum_curve_b
# (finalize_curve.py takes the n = 800 point from fold 0 of sum_ft; it reads runs/sum_ft/ckpt/sum_ft/.)
set -euo pipefail
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
TAG=sum_ft; LOSS=sum; FOLDS=0,1,2,3,4; EPOCHS=12; BATCH=16; MICRO=8; LR=1e-4; CURVE=""; CURVE_FOLD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG=$2; shift 2;;
    --loss) LOSS=$2; shift 2;;
    --folds) FOLDS=$2; shift 2;;
    --epochs) EPOCHS=$2; shift 2;;
    --batch) BATCH=$2; shift 2;;
    --micro) MICRO=$2; shift 2;;
    --lr) LR=$2; shift 2;;
    --curve) CURVE=$2; shift 2;;
    --curve_fold) CURVE_FOLD=$2; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
DATA="${ADD1000_DIR:-$K/data/ADD1000}/flat"     # needs DATA/images/<id>.jpg and DATA/maps/<id>.png
OUT="$K/runs/$TAG/ckpt/$TAG"
mkdir -p "$OUT"
CMD=("$PY" -I -W ignore "$K/scripts/sum/ft_sum_folds.py"
     --scripts "$K/scripts/sum" --code "$K/third_party/SUM/code" --weights "$K/third_party/SUM/weights"
     --data "$DATA" --split "$K/splits/kfold_seed0.json" --out_dir "$OUT"
     --loss "$LOSS" --folds "$FOLDS" --epochs "$EPOCHS" --batch "$BATCH" --micro "$MICRO" --lr "$LR"
     --max_steps 0 --tag "$TAG" --token 2 --device cuda --threads 4)
if [ -n "$CURVE" ]; then
  CMD+=(--curve "$CURVE" --curve_fold "$CURVE_FOLD")
  if [ "$CURVE_FOLD" = 0 ]; then CMD+=(--curve_expect "$K/splits/curve_subsets_seed0.json"); fi
fi
echo "+ ${CMD[*]}"
"${CMD[@]}"
