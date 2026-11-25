"""
Keep the IIE task pool busy: every 60 s make sure N_CPU CPU workers and one MPS worker are
running while unclaimed tasks remain in tasks.json; lift the temporary reservations on the
recipe-(4) grouped / pair-safe tasks once the two old-list workers have exited; exit when the
pool is exhausted and no worker is running.

  python -I supervise.py --n_cpu 4 --old_pids 84929,84942
"""
import os, sys, json, time, subprocess, argparse, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
KBS = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
PART = os.path.join(KBS, "logs", "iie_partial")
PY = sys.executable
ap = argparse.ArgumentParser()
ap.add_argument("--n_cpu", type=int, default=4)
ap.add_argument("--old_pids", default="")
a = ap.parse_args()
old = [int(x) for x in a.old_pids.split(",") if x]
LOG = os.path.join(KBS, "logs", "iie_supervise.log")


def log(s):
    open(LOG, "a").write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {s}\n")


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def workers(dev):
    out = subprocess.run(["pgrep", "-f", f"pool_worker.py --device {dev}"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def tname(t):
    return f"{t[0][6:]}_n{t[1]}_r{t[2]}" if t[0].startswith("curve:") else f"{t[0]}_fold{t[1]}"


def unclaimed():
    tj = json.load(open(os.path.join(HERE, "tasks.json")))
    return [tname(t) for t in tj["tasks"]
            if not os.path.exists(os.path.join(PART, tname(t) + ".json")) and not os.path.exists(os.path.join(PART, tname(t) + ".claim"))]


def start(dev, k):
    env = dict(os.environ, IIE_FAST_FINAL="1", OMP_NUM_THREADS="4" if dev == "mps" else "3")
    out = open(os.path.join(KBS, "logs", f"iie_pool_{dev}_sup{k}.out"), "w")
    subprocess.Popen([PY, "-I", os.path.join(HERE, "pool_worker.py"), "--device", dev, "--threads", env["OMP_NUM_THREADS"]],
                     cwd=HERE, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    log(f"started {dev} worker #{k}")


k = 0
lifted = False
log(f"supervisor start n_cpu={a.n_cpu} old={old}")
while True:
    if not lifted and not any(alive(p) for p in old):
        for f in os.listdir(PART):
            if f.endswith(".claim") and (f.startswith("iie_ft_klcc_grouped") or f.startswith("iie_ft_klcc_pairsafe")):
                p = os.path.join(PART, f)
                if "reserved" in open(p).read():
                    os.remove(p); log(f"lifted reservation {f}")
        lifted = True
    left = unclaimed()
    cpu, mps = workers("cpu"), workers("mps")
    if not left and not cpu and not mps:
        log("pool exhausted; exit"); break
    if left:
        for _ in range(max(0, a.n_cpu - len(cpu))):
            k += 1; start("cpu", k); time.sleep(5)
        if not mps:
            k += 1; start("mps", k)
    time.sleep(60)
