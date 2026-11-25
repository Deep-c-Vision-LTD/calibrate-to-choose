#!/bin/bash
# Re-run of the grouped-fold row with the singleton-batch skip (grouped fold 3 has 801 training ids).
KB="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOCK=$KB/.mps_lock
until mkdir "$LOCK" 2>/dev/null; do sleep 20; done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM HUP
echo "[$(date '+%F %T')] MPS lock acquired"
${PY:-python3} -I $KB/scripts/ours/run_row.py --row ours_grouped --kind kfold --init scratch --seed 0 \
  --folds-file $KB/splits/groups.json --skip-singleton-batch --device mps
echo "[$(date '+%F %T')] exit $?"
