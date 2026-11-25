#!/bin/zsh
# Sequential queue for the "ours" track. Rows run in chunks of about 25-30 min of MPS time
# (similar to the other tracks' per-task holds); each chunk takes the shared MPS lock for its own
# duration only (released by a subshell trap on exit/failure), so other tracks can interleave.
# Rows whose summary file already exists are skipped (resume-safe).
#   nohup zsh queue.sh > ../../logs/ours_queue.log 2>&1 &
KB="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
S=$KB/scripts/ours
R=$KB/results
LOCK=$KB/.mps_lock
export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

row() {   # row <name> <args...>: run one row unless its summary exists
  local name=$1; shift
  if [[ -f $R/${name}_summary.json ]]; then echo "skip $name"; return 0; fi
  echo "$(date '+%F %T') start $name"
  $PY -I $S/run_row.py --row $name "$@" || echo "$(date '+%F %T') FAILED $name"
}
curves() {   # curves <init>
  if [[ -f $R/curve_ours_$1_aggregate.json ]]; then echo "skip curves $1"; return 0; fi
  echo "$(date '+%F %T') start curves $1"
  $PY -I $S/run_curves.py --init $1 || echo "$(date '+%F %T') FAILED curves $1"
}
chunk() {   # chunk <check files...> -- <shell function calls as one string>
  local todo=0
  while [[ $1 != "--" ]]; do [[ -f $1 ]] || todo=1; shift; done; shift
  if (( ! todo )); then echo "skip chunk: $1"; return 0; fi
  until mkdir $LOCK 2>/dev/null; do sleep 20; done
  echo "$(date '+%F %T') lock acquired for: $1"
  ( trap "rmdir $LOCK 2>/dev/null" EXIT INT TERM; eval "$1" )
  echo "$(date '+%F %T') lock released"   # the subshell trap released it; never rmdir here
}

chunk $R/ours_scratch_s0_summary.json $R/ours_warm_s0_summary.json -- \
  "row ours_scratch_s0 --kind kfold --init scratch --seed 0; row ours_warm_s0 --kind kfold --init warm --seed 0"
chunk $R/ours_pairsafe_summary.json $R/ours_grouped_summary.json -- \
  "row ours_pairsafe --kind kfold --init scratch --seed 0 --folds-file $KB/splits/kfold_seed0_pairsafe.json; row ours_grouped --kind kfold --init scratch --seed 0 --folds-file $KB/splits/groups.json"
for s in 1 2; do
  chunk $R/ours_scratch_s${s}_summary.json $R/ours_warm_s${s}_summary.json -- \
    "row ours_scratch_s$s --kind kfold --init scratch --seed $s; row ours_warm_s$s --kind kfold --init warm --seed $s"
done
chunk $R/curve_ours_scratch_aggregate.json -- "curves scratch"
chunk $R/curve_ours_warm_aggregate.json -- "curves warm"
for m in no_aspp no_centerbias no_finalizer kl_only; do
  chunk $R/abl_${m}_s0_summary.json $R/abl_${m}_s1_summary.json -- \
    "row abl_${m}_s0 --kind abl --init warm --mode $m --seed 0; row abl_${m}_s1 --kind abl --init warm --mode $m --seed 1"
done
chunk $R/abl_no_aspp_s2_summary.json $R/abl_no_centerbias_s2_summary.json -- \
  "row abl_no_aspp_s2 --kind abl --init warm --mode no_aspp --seed 2; row abl_no_centerbias_s2 --kind abl --init warm --mode no_centerbias --seed 2"
chunk $R/abl_no_finalizer_s2_summary.json $R/abl_kl_only_s2_summary.json -- \
  "row abl_no_finalizer_s2 --kind abl --init warm --mode no_finalizer --seed 2; row abl_kl_only_s2 --kind abl --init warm --mode kl_only --seed 2"
echo "$(date '+%F %T') queue done"
