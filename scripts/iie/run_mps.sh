#!/bin/zsh
# Run a command while holding the shared MPS lock (released on exit or failure).
# usage: run_mps.sh <command ...>
K="${CBYC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOCK=$K/.mps_lock
echo "[$(date +%H:%M:%S)] waiting for MPS lock"
until mkdir $LOCK 2>/dev/null; do sleep 20; done
trap 'rmdir $LOCK 2>/dev/null; echo "[$(date +%H:%M:%S)] lock released"' EXIT INT TERM HUP
echo "[$(date +%H:%M:%S)] lock acquired"
"$@"
rc=$?
echo "[$(date +%H:%M:%S)] command exit $rc"
exit $rc
