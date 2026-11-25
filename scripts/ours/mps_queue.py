"""
GPU queue for the "ours" track. Runs result rows one at a time in priority order; each row
takes the shared MPS lock (mkdir of .mps_lock) for its own duration only and
releases it in a finally block once the hold exceeds HOLD_BUDGET (default 25 min, so a hold is
about two of our rows, similar to the other tracks' per-row holds) (also on SIGTERM/SIGINT/SIGHUP), then waits a short courtesy
gap before polling again so other waiting jobs can take the GPU. Rows whose summary file
already exists are skipped (resume-safe). Each row's output goes to logs/ours_<row>.log.

  nohup python -I mps_queue.py > ../../logs/ours_mps_queue.log 2>&1 &
"""
import sys
sys.dont_write_bytecode = True
import os, time, signal, subprocess

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
PY = sys.executable
S = os.path.join(KB, "scripts", "ours")
R = os.path.join(KB, "results")
LOGS = os.path.join(KB, "logs")
LOCK = os.path.join(KB, ".mps_lock")
POLL = float(os.environ.get("OURS_LOCK_POLL", "0.001"))
COURTESY = float(os.environ.get("OURS_COURTESY_S", "5"))
# keep the lock for further rows while the current hold is shorter than this (about one other-track row)
HOLD_BUDGET = float(os.environ.get("OURS_HOLD_BUDGET_S", "1500"))
SPL = os.path.join(KB, "splits")


def row(name, *args):
    return ([os.path.join(R, f"{name}_summary.json")], name,
            [PY, "-I", os.path.join(S, "run_row.py"), "--row", name, *args, "--device", "mps"])


def curve(init, sizes):
    files = [os.path.join(R, f"curve_ours_{init}_n{n}_r{r}_summary.json") for n in sizes for r in range(3)]
    return (files, f"curves_{init}_n{'-'.join(map(str, sizes))}",
            [PY, "-I", os.path.join(S, "run_curves.py"), "--init", init, "--sizes", *map(str, sizes), "--device", "mps"])


JOBS = [
    row("ours_scratch_s0", "--kind", "kfold", "--init", "scratch", "--seed", "0"),
    row("ours_warm_s0", "--kind", "kfold", "--init", "warm", "--seed", "0"),
    row("ours_pairsafe", "--kind", "kfold", "--init", "scratch", "--seed", "0", "--folds-file", os.path.join(SPL, "kfold_seed0_pairsafe.json")),
    row("ours_grouped", "--kind", "kfold", "--init", "scratch", "--seed", "0", "--folds-file", os.path.join(SPL, "groups.json")),
    curve("scratch", [800]),
    curve("warm", [800]),
    row("ours_scratch_s1", "--kind", "kfold", "--init", "scratch", "--seed", "1"),
    row("ours_warm_s1", "--kind", "kfold", "--init", "warm", "--seed", "1"),
    row("ours_scratch_s2", "--kind", "kfold", "--init", "scratch", "--seed", "2"),
    row("ours_warm_s2", "--kind", "kfold", "--init", "warm", "--seed", "2"),
]
for s in (0, 1, 2):
    for m in ("no_aspp", "no_centerbias", "no_finalizer", "kl_only"):
        JOBS.append(row(f"abl_{m}_s{s}", "--kind", "abl", "--init", "warm", "--mode", m, "--seed", str(s)))
# lowest priority: the Table II "full" reference under the ablation rng stream (ours_warm_s* is the
# same recipe up to the data-order draw, and is used as the ablation reference if these do not run)
for s in (0, 1, 2):
    JOBS.append(row(f"abl_full_s{s}", "--kind", "abl", "--init", "warm", "--mode", "full", "--seed", str(s)))

have_lock = False
child = None


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def release():
    global have_lock
    if have_lock:
        try:
            os.rmdir(LOCK)
        except FileNotFoundError:
            pass
        have_lock = False
        log("MPS lock released")


def on_signal(signum, frame):
    log(f"signal {signum}: stopping")
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(30)
        except subprocess.TimeoutExpired:
            child.kill()
    release()
    sys.exit(1)


def acquire():
    global have_lock
    t0 = time.time()
    while True:
        try:
            os.mkdir(LOCK)
            have_lock = True
            log(f"MPS lock acquired after {time.time() - t0:.0f}s wait")
            return
        except FileExistsError:
            time.sleep(POLL)


def main():
    global child
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)
    env = dict(os.environ, OMP_NUM_THREADS="6", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    log(f"queue start: {len(JOBS)} jobs, poll {POLL}s, courtesy {COURTESY}s, hold budget {HOLD_BUDGET}s")
    t_hold = None
    try:
        for files, name, cmd in JOBS:
            if all(os.path.exists(f) for f in files):
                log(f"skip {name} (done)")
                continue
            if have_lock and time.time() - t_hold > HOLD_BUDGET:
                release()
                time.sleep(COURTESY)
            if not have_lock:
                acquire()
                t_hold = time.time()
            log(f"start {name}: {' '.join(cmd[2:])}")
            t0 = time.time()
            with open(os.path.join(LOGS, f"ours_{name}.log"), "a") as fh:
                child = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=S)
                rc = child.wait()
            child = None
            log(f"{'done' if rc == 0 else f'FAILED rc={rc}'} {name} in {time.time() - t0:.0f}s")
    finally:
        release()
    log("queue done")


if __name__ == "__main__":
    main()
