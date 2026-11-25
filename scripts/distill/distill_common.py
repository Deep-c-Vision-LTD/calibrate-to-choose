"""Shared helpers of track B (distillation): paths, the polite MPS lock, Pitt pool bookkeeping."""
import sys
sys.dont_write_bytecode = True
import os, json, time, glob, atexit, signal, subprocess

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
CD = os.path.join(KB, "cache", "distill")
TD = os.path.join(CD, "teachers")
PM = os.path.join(CD, "pitt_msdb")
LOGS = os.path.join(KB, "logs")
RESULTS = os.path.join(KB, "results")
LOCK = os.path.join(KB, ".mps_lock")
PITT = os.environ.get("PITT_ADS_DIR", os.path.join(KB, "data", "pitt_ads", "imgs"))   # Pitt Ads images (not redistributed)


def pool():
    """-> list of Pitt file names in the pool (subset order, near-duplicates of ADD1000 removed)."""
    d = json.load(open(os.path.join(CD, "pitt_subset.json")))
    return [d["subset_files"][k] for k in d["pool_positions"]]


def tracka_waiting():
    """True while a track-A job is alive and its log says it is waiting for the MPS lock
    (track A has MPS priority: we do not compete for the lock while it waits)."""
    try:
        ps = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True, timeout=10).stdout
    except Exception:          # noqa
        return False
    if "scripts/trackA/" not in ps:
        return False
    for p in glob.glob(os.path.join(LOGS, "trackA_*.log")):
        try:
            with open(p, "rb") as fh:
                fh.seek(0, 2); n = fh.tell(); fh.seek(max(0, n - 600))
                last = fh.read().decode("utf8", "ignore").strip().splitlines()
        except OSError:
            continue
        if last and "waiting for MPS lock" in last[-1]:
            return True
        # track A's queue sleeps 30 s between rows before printing 'waiting' again
        if last and ": rc=" in last[-1] and time.time() - os.path.getmtime(p) < 75 and "QUEUE DONE" not in last[-1]:
            return True
    return False


class PoliteLock:
    """mkdir lock on .mps_lock (the shared protocol), acquired only while track A is not
    waiting for it; released on exit, failure or SIGTERM."""

    def __init__(self, poll=20):
        self.poll, self.held = poll, False
        atexit.register(self.release)
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, self._handler)

    def _handler(self, signum, frame):
        self.release(); raise SystemExit(128 + signum)

    def acquire(self, max_wait=None):
        t0 = time.time()
        while True:
            if not tracka_waiting():
                try:
                    os.mkdir(LOCK); self.held = True
                    print(f"[lock] acquired after {time.time() - t0:.0f}s", flush=True)
                    return True
                except FileExistsError:
                    pass
            if max_wait is not None and time.time() - t0 > max_wait:
                return False
            time.sleep(self.poll)

    def release(self):
        if self.held:
            try:
                os.rmdir(LOCK)
            except FileNotFoundError:
                pass
            self.held = False
            print("[lock] released", flush=True)
