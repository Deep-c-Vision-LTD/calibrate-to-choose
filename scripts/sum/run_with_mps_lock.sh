#!/bin/bash
# usage: run_with_mps_lock.sh <command...>   (acquires the shared MPS lock, releases on exit)
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOCK=$K/.mps_lock
until mkdir "$LOCK" 2>/dev/null; do sleep 0.5; done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM HUP
echo "[lock acquired $(date)]"
"$@"
rc=$?
echo "[done rc=$rc $(date)]"
exit $rc
