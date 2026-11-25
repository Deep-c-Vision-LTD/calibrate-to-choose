"""
Task-pool worker for the adapted DeepGaze IIE runs. Several workers (CPU and one MPS) pull
(row, fold) or (curve, n, resample) tasks from a fixed priority list; a task is claimed by
atomically creating logs/iie_partial/<task>.claim, and its result is written to
logs/iie_partial/<task>.json in exactly the format ft_kfold.py / ft_curves.py aggregate.
Training settings per task are identical to those scripts (same shuffle streams, prior,
sigma rule). An MPS worker takes the shared MPS lock before claiming each task and releases
it after the task.

  python -I pool_worker.py --device cpu --threads 3
  python -I pool_worker.py --device mps
"""
import os, sys, json, time, argparse, datetime, signal, atexit, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import iie_common as C
import iie_train as T

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--only", default="", help="comma list of task-name prefixes to restrict to")
ap.add_argument("--max_tasks", type=int, default=1000)
a = ap.parse_args()
torch.set_num_threads(a.threads)
PART = os.path.join(C.KBS, "logs", "iie_partial"); os.makedirs(PART, exist_ok=True)
CKPT = os.path.join(C.CACHE, "ckpt"); os.makedirs(CKPT, exist_ok=True)
LOCK = os.path.join(C.KBS, ".mps_lock")
LOG = os.path.join(C.KBS, "logs", f"iie_pool_{a.device}_{os.getpid()}.log")
SPLITS = os.path.join(C.KBS, "splits")
have_lock = False


def log(s):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {s}"
    print(line, flush=True); open(LOG, "a").write(line + "\n")


def release():
    global have_lock
    if have_lock:
        try:
            os.rmdir(LOCK)
        except OSError:
            pass
        have_lock = False
        log("MPS lock released")


def _sig(signum, frame):
    release(); sys.exit(1)


atexit.register(release)
for s_ in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(s_, _sig)


def acquire():
    global have_lock
    while True:
        try:
            os.mkdir(LOCK); have_lock = True; log("MPS lock acquired"); return
        except FileExistsError:
            time.sleep(20)


# ---------------- task list (priority order, re-read before every claim) ----------------
# tasks.json: {"kfold_rows": {row: [split_file, loss, epochs]}, "curve_rows": {row: [loss, epochs]},
#              "tasks": [[row, fold] | ["curve:" + row, n, r], ...]}
TASKFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.json")


def load_tasks():
    tj = json.load(open(TASKFILE))
    kf = {r: (os.path.join(C.KBS, v[0]) if not os.path.isabs(v[0]) else v[0], v[1], int(v[2])) for r, v in tj["kfold_rows"].items()}
    cr = {r: (v[0], int(v[1])) for r, v in tj["curve_rows"].items()}
    return kf, cr, [tuple(t) for t in tj["tasks"]]


KF, CR, TASKS = load_tasks()


def tname(t):
    if t[0].startswith("curve:"):
        return f"{t[0][6:]}_n{t[1]}_r{t[2]}"
    return f"{t[0]}_fold{t[1]}"


def claim(t):
    name = tname(t)
    if os.path.exists(os.path.join(PART, name + ".json")):
        return False
    try:
        fd = os.open(os.path.join(PART, name + ".claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, json.dumps({"pid": os.getpid(), "device": a.device, "host": socket.gethostname(),
                             "start": datetime.datetime.now().isoformat()}).encode())
    os.close(fd)
    return True


D = T.Data()
_orders_cache = {}


def run_kfold(row, k):
    split, loss, epochs = KF[row]
    sp = json.load(open(split))
    folds = [[str(i) for i in f] for f in sp["folds"]]
    key = (row, epochs)
    if key not in _orders_cache:
        _orders_cache[key] = T.published_orders(folds, epochs, n_total=1000)
    te = folds[k]
    tr = [i for kk in range(len(folds)) if kk != k for i in folds[kk]]
    sel = [str(x) for x in np.random.default_rng(k).choice(tr, size=min(C.SIGMA_SELECT_N, len(tr)), replace=False)]
    res = T.train_eval(D, tr, te, _orders_cache[key][k], loss, 1e-3, 8, a.device, sel, log=log,
                       tag=f"{row} fold {k}", ckpt_path=os.path.join(CKPT, f"{row}_fold{k}.pt"))
    res.update({"fold": k, "sigma_select_ids": sel, "device": a.device, "worker_pid": os.getpid(),
                "fast_final": C.FAST_FINAL, "split_file": split})
    return res


def run_curve(row, n, r):
    idx = np.random.default_rng(0).permutation(len(D.ids))
    te = [D.ids[j] for j in idx[:200]]
    pool = [D.ids[j] for j in idx[200:]]
    tr = [str(x) for x in np.random.default_rng(1000 * n + r).choice(np.array(pool), size=n, replace=False)]
    loss, epochs = CR[row]
    orders = T.seeded_orders(tr, epochs, seed=r)
    res = T.train_eval(D, tr, te, orders, loss, 1e-3, min(8, n), a.device, tr, log=log, tag=f"{row} n={n} r={r}")
    res.update({"n": n, "resample": r, "train_ids": tr, "seed": r, "device": a.device,
                "worker_pid": os.getpid(), "fast_final": C.FAST_FINAL})
    return res


log(f"worker start device={a.device} threads={a.threads} fast_final={C.FAST_FINAL} only={a.only}")
done = 0
while done < a.max_tasks:
    if a.device == "mps":
        acquire()
    picked = None
    KF, CR, TASKS = load_tasks()
    for t in TASKS:
        if a.only and not any(tname(t).startswith(p) for p in a.only.split(",")):
            continue
        if not t[0].startswith("curve:") and not os.path.exists(KF[t[0]][0]):
            continue
        if claim(t):
            picked = t; break
    if picked is None:
        log("no unclaimed tasks left"); release(); break
    name = tname(picked)
    log(f"claimed {name}")
    t0 = time.time()
    res = run_curve(picked[0][6:], picked[1], picked[2]) if picked[0].startswith("curve:") else run_kfold(picked[0], picked[1])
    json.dump(res, open(os.path.join(PART, name + ".json"), "w"))
    log(f"finished {name} in {time.time() - t0:.0f}s")
    if a.device == "mps":
        release()
        time.sleep(60)   # courtesy gap so other tracks polling the lock (every 20 s) can take it
    done += 1
log("worker exit")
