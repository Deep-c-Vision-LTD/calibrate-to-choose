"""
Track B, step 2b: MSDB frozen features for the Pitt pool and the 5 teachers' pseudo-labels.

Same settings as scripts/msdb/cache_msdb.py: image -> RGB -> PIL bilinear 512x512, [0,255],
pixel_per_dva 21.7, the package's own per-scale processing (msdb_backbone.per_scale_features,
released weights, float64 normaliser buffers cast to float32), generalisation-mode scale mix,
mix stored / used as float16 with LayerNorm0 statistics (mean, biased variance) taken from the
float32 mix -- exactly the head input the teachers were trained on (cache/msdb/mix_fp16.npy).

Per pool image i:
  * if all 5 teachers exist (cache/distill/teachers/ALL_DONE.json): teacher k's 256x256 log density
    with fold k's own training prior is written to tlogd[k, i] (flag 3);
  * otherwise the float16 mix + stats are stored (flag 2) and `--finish` applies the teachers later.
Workers on different devices share the work through atomic claim files (O_EXCL).
MPS workers hold the shared lock in chunks (--chunk-s) and yield it in between; they never take it
while a track-A job is waiting (track A has MPS priority).

  python -I pitt_msdb_worker.py --dev cpu --reverse
  python -I pitt_msdb_worker.py --dev mps
  python -I pitt_msdb_worker.py --finish --dev cpu
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, argparse
import numpy as np, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import distill_common as DC                                  # noqa: E402
sys.path.insert(0, os.path.join(DC.KB, "scripts", "msdb"))
from msdb_common import S, RO, C_IN, build_sal_net_pretrained, MSDBHead   # noqa: E402
import train_msdb as TM                                      # noqa: E402

K = 5


def open_arrays(n):
    os.makedirs(DC.PM, exist_ok=True); os.makedirs(os.path.join(DC.PM, "claims"), exist_ok=True)
    spec = {"flags": ((n,), np.uint8), "mixstats": ((n, 2), np.float64),
            "mix_fp16": ((n, C_IN, RO, RO), np.float16), "tlogd": ((K, n, 256, 256), np.float32)}
    arrs = {}
    for k, (shape, dt) in spec.items():
        p = os.path.join(DC.PM, f"{k}.npy")
        if not os.path.exists(p):
            try:
                fd = os.open(p + ".creating", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
                a = np.lib.format.open_memmap(p + ".tmp.npy", mode="w+", dtype=dt, shape=shape); a.flush(); del a
                os.replace(p + ".tmp.npy", p); os.remove(p + ".creating")
            except FileExistsError:
                while not os.path.exists(p):
                    time.sleep(1)
        arrs[k] = np.load(p, mmap_mode="r+")
        assert arrs[k].shape == shape and arrs[k].dtype == dt, (k, arrs[k].shape)
    return arrs


def load_teachers(dev):
    heads, cbds = [], []
    for k in range(K):
        ck = torch.load(os.path.join(DC.TD, f"fold{k}.pt"), map_location="cpu", weights_only=False)
        net, s0, c0, _, _ = build_sal_net_pretrained()
        h = MSDBHead(net, s0, c0)
        h.load_state_dict(ck["state_dict"]); h.to(dev).eval()
        heads.append(h); cbds.append(ck["cbd256"].float().to(dev))
    return heads, cbds


@torch.no_grad()
def apply_teachers(heads, cbds, f16, mu, var, dev):
    """f16: (1,2560,64,64) float16 on dev; mu/var floats from the float32 mix -> (5,256,256) float32."""
    f = f16.float()
    m = torch.tensor([mu], dtype=torch.float32, device=dev)
    sd = torch.sqrt(torch.tensor([var], dtype=torch.float32, device=dev) + 1e-12)
    out = []
    for h, cb in zip(heads, cbds):
        out.append(h.finalize(TM.readout_head(h, f, m, sd), cb)[0].float().cpu().numpy())
    return np.stack(out)


def claim(i):
    try:
        fd = os.open(os.path.join(DC.PM, "claims", f"{i}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--finish", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--chunk-s", type=float, default=900)
    ap.add_argument("--yield-s", type=float, default=45)
    ap.add_argument("--start-frac", type=float, default=0.0, help="start position as a fraction of the pool (forward order, wraps)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    files = DC.pool(); n = len(files)
    arrs = open_arrays(n)
    ready = lambda: os.path.exists(os.path.join(DC.TD, "ALL_DONE.json"))   # noqa: E731
    t0 = time.time()
    if a.finish:
        assert ready(), "teachers not finished"
        heads, cbds = load_teachers(a.dev)
        todo = [i for i in range(n) if arrs["flags"][i] == 2]
        print(f"[finish] {len(todo)} stored mixes to label", flush=True)
        for j, i in enumerate(todo):
            f16 = torch.from_numpy(np.array(arrs["mix_fp16"][i]))[None].to(a.dev)
            mu, var = arrs["mixstats"][i]
            arrs["tlogd"][:, i] = apply_teachers(heads, cbds, f16, float(mu), float(var), a.dev)
            arrs["tlogd"].flush(); arrs["flags"][i] = 3; arrs["flags"].flush()
            if (j + 1) % 50 == 0:
                print(f"[finish] {j + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        print(f"[finish] done; flags: {np.bincount(np.asarray(arrs['flags']), minlength=4).tolist()}", flush=True)
        return

    from msdb_backbone import load_msdb, per_scale_features, mix_features
    lock = DC.PoliteLock() if a.dev == "mps" else None
    if lock:
        lock.acquire()
    model = load_msdb(a.dev, cast_fp32=True)
    _, _, _, scale_w, _ = build_sal_net_pretrained()
    sw = scale_w.to(a.dev)
    heads = cbds = None
    t_chunk = time.time()
    order = list(range(n))[::-1] if a.reverse else list(range(n))
    if a.start_frac:
        st = int(a.start_frac * n); order = order[st:] + order[:st]
    k_done = 0
    for i in order:
        if arrs["flags"][i] >= 2 or not claim(i):
            continue
        if lock and not lock.held:
            lock.acquire(); t_chunk = time.time()
        if heads is None and ready():
            heads, cbds = load_teachers(a.dev)
            print(f"[{a.dev}] teachers loaded; labelling online from now on", flush=True)
        ti = time.time()
        im = Image.open(os.path.join(DC.PITT, files[i])).convert("RGB").resize((S, S), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1).copy())[None].to(a.dev)
        with torch.no_grad():
            mix = mix_features(per_scale_features(model, x), sw)        # (1,2560,64,64) float32
            mu = float(mix.mean()); var = float(((mix - mix.mean()) ** 2).mean())
            f16 = mix.half()
        arrs["mixstats"][i] = [mu, var]; arrs["mixstats"].flush()
        if heads is not None:
            arrs["tlogd"][:, i] = apply_teachers(heads, cbds, f16, mu, var, a.dev); arrs["tlogd"].flush()
            arrs["flags"][i] = 3
        else:
            arrs["mix_fp16"][i] = f16[0].cpu().numpy(); arrs["mix_fp16"].flush()
            arrs["flags"][i] = 2
        arrs["flags"].flush()
        k_done += 1
        if k_done % 10 == 0 or k_done == 1:
            fl = np.bincount(np.asarray(arrs["flags"]), minlength=4)
            print(f"[{a.dev}] {k_done} new (pos {i}), {time.time() - ti:.2f}s this img, {(time.time() - t0) / k_done:.2f}s/img avg, "
                  f"flags todo/claimed-or-active {int(fl[0]) + int(fl[1])} mix-stored {int(fl[2])} labelled {int(fl[3])}", flush=True)
        if lock and (time.time() - t_chunk > a.chunk_s or (k_done % 5 == 0 and DC.tracka_waiting())):
            lock.release()
            if a.dev == "mps":
                torch.mps.synchronize()
            time.sleep(a.yield_s)
    if lock:
        lock.release()
    fl = np.bincount(np.asarray(arrs["flags"]), minlength=4)
    print(f"[{a.dev}] worker done: {k_done} images in {time.time() - t0:.0f}s; flags {fl.tolist()}", flush=True)


if __name__ == "__main__":
    main()
