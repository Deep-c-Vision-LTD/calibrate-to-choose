"""
Step-2 smoke test: the NEW native code path fed 384 x 384 inputs reproduces the EXISTING 384 pipeline
(scripts/iie/iie_train.py train loop with iie_common.StackedIIE, IIE_FAST_FINAL=1 as used for every fold of
iie_ft_klcc_e20, batch 8), on fold 0, published data order, first NSTEPS optimiser steps.
  (i)   native caching code (cache_native.py's extraction call) on PIL-bilinear 384 images vs the cached
        384 features (scripts/iie/cache_features.py), first batch;
  (ii)  step-1 gradients, per-step batch losses, parameters after NSTEPS steps;
  (iii) predictions of both trained readouts on 24 fold-0 test images: reference iie_train.predict_module
        (384 path) vs native_common.predict_native with H = W = 384; sigma-selected metrics on those images.
Writes cache/iie_native/smoke384.json. CPU only.

  python -I smoke384.py [--threads 6] [--steps 10]
"""
import os, sys, json, time, argparse
os.environ["IIE_FAST_FINAL"] = "1"            # the 384 row's training path (custom FinalMix backward)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import native_common as N
C = N.C
sys.path.insert(0, os.path.join(C.KBS, "scripts", "iie"))
import iie_train as T
import deepgaze_pytorch
assert C.FAST_FINAL

ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--steps", type=int, default=10); a = ap.parse_args()
torch.set_num_threads(a.threads)
t00 = time.time()
D = T.Data()
ids, folds = C.load_split()
folds = [[str(i) for i in f] for f in folds]
orders = T.published_orders(folds, 20, n_total=1000)
k = 0
te = folds[k]
tr = [i for kk in range(5) if kk != k for i in folds[kk]]
order = orders[k][0]
prior = D.prior(tr)
rep = {"fold": k, "steps": a.steps, "threads": a.threads}

# (i) features: caching code path on 384 inputs vs cached 384 features
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
fb = order[:8]
mx, rel = 0.0, []
with torch.no_grad():
    for i in fb:
        x = torch.from_numpy(C.load_img_384(i))[None]
        f = torch.cat(C.iie_features(model, x), dim=1)[0].numpy()
        ref = np.asarray(D.feats[D.pos[i]])
        mx = max(mx, float(np.abs(f - ref).max())); rel.append(float(np.linalg.norm(f - ref) / np.linalg.norm(ref)))
rep["features_384"] = {"ids": fb, "max_abs": mx, "max_rel_norm": max(rel)}
print("features", rep["features_384"], flush=True)

# (ii) training: reference vs new path from the same initial weights, same batches
cbt = torch.from_numpy(prior.astype(np.float32))
m_ref = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
st_ref = C.StackedIIE(m_ref)
opt_ref = torch.optim.Adam(st_ref.parameters(), lr=1e-3)
m_new = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
st_new = N.make_native_stacked(m_new)
opt_new = torch.optim.Adam(st_new.parameters(), lr=1e-3)
geo = N.Geometry(C.S, C.S)
cbd = geo.dcb(N.native_centerbias(prior, C.S, C.S))
L_ref, L_new, t_ref, t_new = [], [], 0.0, 0.0
for s in range(a.steps):
    bt = order[s * 8:(s + 1) * 8]
    t0 = time.time()
    feats = D.batch_feats(bt, "cpu")
    l = T.loss_fn(st_ref, feats, cbt[None].expand(len(bt), -1, -1), bt, D, "klcc64", "cpu")
    opt_ref.zero_grad(); l.backward()
    if s == 0:
        g_ref = torch.cat([p.grad.detach().double().flatten() for p in st_ref.parameters()])
    opt_ref.step(); L_ref.append(float(l)); t_ref += time.time() - t0
    t0 = time.time()
    opt_new.zero_grad(); tot = 0.0
    for i in bt:
        fi = torch.from_numpy(np.array(D.feats[D.pos[i]]))[None]
        g64 = torch.from_numpy(np.array(D.ms64[D.pos[i]], dtype=np.float32))
        li = N.klcc64_native(st_new.logP(C.split_channels(fi), cbd, geo), geo, g64) / len(bt)
        li.backward(); tot += float(li)
    if s == 0:
        g_new = torch.cat([p.grad.detach().double().flatten() for p in st_new.parameters()])
    opt_new.step(); L_new.append(tot); t_new += time.time() - t0
    print(f"step {s}: loss ref {L_ref[-1]:.7f} new {L_new[-1]:.7f}", flush=True)
p_ref = torch.cat([p.detach().double().flatten() for p in st_ref.parameters()])
p_new = torch.cat([p.detach().double().flatten() for p in st_new.parameters()])
p_0 = torch.cat([p.detach().double().flatten() for p in C.StackedIIE(deepgaze_pytorch.DeepGazeIIE(pretrained=True)).parameters()])
rep["train"] = {"loss_ref": L_ref, "loss_new": L_new,
                "max_rel_loss_diff": float(max(abs(x - y) / abs(x) for x, y in zip(L_ref, L_new))),
                "step1_grad_global_rel_err": float((g_ref - g_new).norm() / g_ref.norm()),
                "params_global_rel_diff": float((p_ref - p_new).norm() / p_ref.norm()),
                "update_global_rel_diff": float((p_ref - p_new).norm() / (p_ref - p_0).norm()),
                "n_params": [int(p_ref.numel()), int(p_new.numel())],
                "seconds_ref": t_ref, "seconds_new": t_new}
print("train", {k_: v for k_, v in rep["train"].items() if not k_.startswith("loss_")}, flush=True)

# (iii) predictions through the evaluation paths
st_ref.write_back(m_ref); st_new.write_back(m_new)
ev = te[:24]
sel = [str(x) for x in np.random.default_rng(k).choice(tr, size=C.SIGMA_SELECT_N, replace=False)][:24]
pr = T.predict_module(m_ref, D, ev + sel, prior, "cpu")
pn = N.predict_native(m_new, lambda i: torch.from_numpy(np.array(D.feats[D.pos[i]]))[None], ev + sel, prior,
                      lambda i: (C.S, C.S))
rd = [float(np.abs(pr[i] - pn[i]).max() / pr[i].max()) for i in ev]
s_ref, _ = T.select_sigma({i: pr[i] for i in sel}, D, sel)
s_new, _ = T.select_sigma({i: pn[i] for i in sel}, D, sel)
from scipy.ndimage import gaussian_filter
def mets(p, s):
    out = {m: [] for m in ("CC", "SIM", "KLD")}
    for i in ev:
        q = gaussian_filter(p[i], s) if s > 0 else p[i]
        for m, v in C.metrics(q, D.raw[D.pos[i]].astype(np.float64)).items():
            out[m].append(v)
    return out
mr, mn = mets(pr, s_ref), mets(pn, s_new)
rep["predict"] = {"n_test_images": len(ev), "max_rel_density_diff": max(rd), "sigma_ref": s_ref, "sigma_new": s_new,
                  "mean_metrics_ref": {m: float(np.mean(v)) for m, v in mr.items()},
                  "mean_metrics_new": {m: float(np.mean(v)) for m, v in mn.items()},
                  "max_abs_metric_diff_per_image": {m: float(np.max(np.abs(np.array(mr[m]) - np.array(mn[m])))) for m in mr}}
print("predict", rep["predict"], flush=True)
rep["pass"] = bool(rep["features_384"]["max_rel_norm"] < 1e-5 and rep["train"]["max_rel_loss_diff"] < 1e-5
                   and rep["train"]["step1_grad_global_rel_err"] < 1e-4 and rep["predict"]["max_rel_density_diff"] < 1e-3
                   and max(rep["predict"]["max_abs_metric_diff_per_image"].values()) < 1e-4 and s_ref == s_new)
rep["seconds"] = time.time() - t00
json.dump(rep, open(os.path.join(N.NCACHE, "smoke384.json"), "w"), indent=1)
print("PASS" if rep["pass"] else "FAIL", f"{rep['seconds']:.0f}s")
