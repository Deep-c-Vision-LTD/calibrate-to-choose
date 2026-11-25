#!/bin/bash
# Run one command under the shared MPS lock (mkdir .mps_lock), released on exit.
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOCK=$K/.mps_lock
echo "[$(date)] waiting for MPS lock: $*"
until mkdir $LOCK 2>/dev/null; do sleep 20; done
trap 'rmdir $LOCK 2>/dev/null' EXIT INT TERM HUP
echo "[$(date)] MPS lock acquired; load: $(uptime)"
export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${TMPDIR:-/tmp}"
"$@"
rc=$?
echo "[$(date)] done rc=$rc"
exit $rc
