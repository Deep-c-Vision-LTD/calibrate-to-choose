"""
One caching pass of DeepGazeMSDB's frozen multi-scale backbone over ADD1000 (512 x 512 input,
pixel_per_dva = 21.7, as eval_deepgaze_msdb.py). Per image (index = position in the
numerically sorted id list) it stores:

  mix_fp16.npy   (1000, 2560, 64, 64) float16  generalisation-mode scale mix (head input)
  mixstats.npy   (1000, 2) float64            mean and biased variance of the fp32 mix over
                                              (C,H,W) = LayerNorm0's per-image statistics
  readout_zs.npy (1000, 64, 64) float32       released saliency network applied to the fp32
                                              mix (zero-shot readout, before the finalizer)
  ss_proj.npy    (1000, 10, 8, 64, 64) float32 per-scale centred projections through the
                                              released LayerNorm0 weight and conv0
  ss_gram.npy    (1000, 10, 10) float64       per-scale centred Gram matrix over (C,H,W)
  done.npy       (1000,) uint8                0 = todo, 1 = claimed, 2 = finished

ss_proj + ss_gram are exact sufficient statistics for LayerNorm0 -> conv0 under ANY scale
weights a (sum a = 1): conv0(LN0(sum a_s f_s)) = (sum a_s P_s) / sqrt(a^T G a + eps) + W0 b0,
so MSDB's per-dataset parameters (scale weights, sigma, centre-bias weight, priority) can be
learned exactly with the released saliency network frozen.

Several workers may run at once (e.g. MPS forward from index 0, CPU backward from 999);
they share done.npy and skip claimed images.

  python -I cache_msdb.py --dev mps [--reverse] [--limit N]
"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msdb_common import *          # noqa
from msdb_backbone import load_msdb, per_scale_features, mix_features


def open_arrays(n):
    os.makedirs(CACHE, exist_ok=True)
    spec = {"mix_fp16": ((n, C_IN, RO, RO), np.float16), "mixstats": ((n, 2), np.float64),
            "readout_zs": ((n, RO, RO), np.float32), "ss_proj": ((n, 10, 8, RO, RO), np.float32),
            "ss_gram": ((n, 10, 10), np.float64), "done": ((n,), np.uint8)}
    arrs = {}
    for k, (shape, dt) in spec.items():
        p = os.path.join(CACHE, f"{k}.npy")
        if not os.path.exists(p):
            try:
                a = np.lib.format.open_memmap(p, mode="w+", dtype=dt, shape=shape); a.flush(); del a
            except FileExistsError:
                pass
        arrs[k] = np.load(p, mmap_mode="r+")
        assert arrs[k].shape == shape and arrs[k].dtype == dt, (k, arrs[k].shape)
    return arrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="mps")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-lock", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ids, folds, fold_of = load_split()
    n = len(ids)
    arrs = open_arrays(n)
    lock = None
    if a.dev == "mps" and not a.no_lock:
        lock = MPSLock().acquire()
    model = load_msdb(a.dev, cast_fp32=True)
    net, sigma0, cbw0, scale_w, sd = build_sal_net_pretrained()
    net = net.to(a.dev).eval()
    sw = scale_w.to(a.dev)
    W0 = net.conv0.weight[:, :, 0, 0].detach()
    Wp = W0 * net.layernorm0.weight.detach()[None]          # (8,2560) LN0 weight folded in
    order = list(range(n))[::-1] if a.reverse else list(range(n))
    t0, k_done, maxabs = time.time(), 0, 0.0
    for i in order:
        if arrs["done"][i] != 0:
            continue
        arrs["done"][i] = 1; arrs["done"].flush()
        x = torch.from_numpy(load_img(ids[i]))[None].to(a.dev)
        with torch.no_grad():
            feats = per_scale_features(model, x)
            mix = mix_features(feats, sw)                     # (1,2560,64,64) fp32
            mu = mix.mean(); var = ((mix - mu) ** 2).mean()
            r = net(mix)[0, 0]
            Fs = torch.cat(feats, 0)                           # (10,2560,64,64)
            mus = Fs.mean(dim=(1, 2, 3))
            Fc = Fs - mus.view(-1, 1, 1, 1)
            proj = torch.einsum("kc,schw->skhw", Wp, Fc)
            flat = Fc.reshape(10, -1)
            G = (flat @ flat.T) / flat.shape[1]
            maxabs = max(maxabs, float(mix.abs().max()))
        arrs["mix_fp16"][i] = mix[0].half().cpu().numpy()
        arrs["mixstats"][i] = [float(mu), float(var)]
        arrs["readout_zs"][i] = r.float().cpu().numpy()
        arrs["ss_proj"][i] = proj.float().cpu().numpy()
        arrs["ss_gram"][i] = G.float().cpu().numpy().astype(np.float64)
        for k in ["mix_fp16", "mixstats", "readout_zs", "ss_proj", "ss_gram"]:
            arrs[k].flush()
        arrs["done"][i] = 2; arrs["done"].flush()
        k_done += 1
        if k_done % 10 == 0 or k_done == 1:
            el = time.time() - t0
            left = int((arrs["done"] == 0).sum())
            print(f"[{a.dev}] {k_done} new (idx {i}), {el/k_done:.2f}s/img, todo {left}, "
                  f"max|mix| {maxabs:.1f}, finished total {int((arrs['done'] == 2).sum())}", flush=True)
        if a.limit and k_done >= a.limit:
            break
    print(f"[{a.dev}] worker done: {k_done} images in {time.time()-t0:.0f}s; "
          f"finished total {int((arrs['done'] == 2).sum())}/{n}", flush=True)
    if lock:
        lock.release()


if __name__ == "__main__":
    main()
