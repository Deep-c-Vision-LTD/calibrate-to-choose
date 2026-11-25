"""
Validation before the caching pass (10 images):
  (a) CPU, original float64 normaliser buffers, full model forward (the published path)
  (b) CPU, buffers cast to float32, full forward
  (c) MPS, buffers cast to float32, full forward
  (d) cached path: per-scale features -> generalisation scale mix -> fp16 round trip ->
      MSDBHead (our re-implementation of saliency network + finalizer) on CPU
  (e) sufficient-statistics path for LayerNorm0+conv0 (for the per-dataset-parameter row)
and (f) the 512-grid scores of (a) against the stored per-image results of the earlier zero-shot run (script not released)
(results/reference/msdb_zeroshot_reference_per_image.json, keyed by index in the numerically sorted id list).

  python -I validate_msdb.py
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msdb_common import *          # noqa
from msdb_backbone import load_msdb, per_scale_features, mix_features

torch.set_num_threads(6)
N = 10
ids, folds, fold_of = load_split()
ck = json.load(open(os.path.join(ROOT, "results", "reference", "msdb_zeroshot_reference_per_image.json")))
# published 2-fold prior: even indices (A) / odd indices (B) of the sorted id list
idx = list(range(len(ids)))
A, B = idx[0::2], idx[1::2]
test_idx = list(range(N))                     # 0..9 -> 5 in A, 5 in B
t0 = time.time()
cb_fit_A = fit_prior([ids[i] for i in A])     # used for B images
cb_fit_B = fit_prior([ids[i] for i in B])     # used for A images
print(f"priors fitted in {time.time()-t0:.0f}s", flush=True)

def prior_for(i):
    return cb_fit_A if i % 2 == 1 else cb_fit_B

net, sigma0, cbw0, scale_w, sd = build_sal_net_pretrained()
print("generalisation scale weights:", [round(float(w), 4) for w in scale_w], "sigma0", sigma0, "cbw0", cbw0)

def full_forward(model, dev, i):
    x = torch.from_numpy(load_img(ids[i]))[None].to(dev)
    cb = torch.from_numpy(prior_for(i).astype(np.float32))[None].to(dev)
    with torch.no_grad():
        logd = model(x, cb, pixel_per_dva=PPD, dataset=None)
    return logd[0].cpu().double().numpy()

res = {"ids": [ids[i] for i in test_idx]}
CPU_NPZ = os.path.join(CACHE, 'validation_cpu_part.npz')
if os.path.exists(CPU_NPZ):
    z = np.load(CPU_NPZ); a_out, b_out, d_out, e_err = list(z['a']), list(z['b']), list(z['d']), list(z['e']); res['cpu_f64buf_s_per_img'] = float(z['t'])
else:
  if True:
    m = load_msdb("cpu", cast_fp32=False)
    a_out, ta = [], time.time()
    for i in test_idx:
        a_out.append(full_forward(m, "cpu", i))
    res["cpu_f64buf_s_per_img"] = (time.time() - ta) / N
    print(f"(a) cpu original: {res['cpu_f64buf_s_per_img']:.1f}s/img", flush=True)
    # (b) CPU cast
    for mod in m.modules():
        for name, buf in list(mod._buffers.items()):
            if buf is not None and buf.dtype == torch.float64:
                mod._buffers[name] = buf.float()
    b_out = [full_forward(m, "cpu", i) for i in test_idx]
    # (d) cached path on CPU features
    head = MSDBHead(net, sigma0, cbw0).eval()
    d_out, e_err = [], []
    W0 = net.conv0.weight[:, :, 0, 0].double()            # (8,2560)
    lnw, lnb = net.layernorm0.weight.double(), net.layernorm0.bias.double()
    Wp = (W0 * lnw[None]).float()                          # LN weight folded into conv0
    for i in test_idx:
        x = torch.from_numpy(load_img(ids[i]))[None]
        feats = per_scale_features(m, x)
        mix = mix_features(feats, scale_w)
        mix16 = mix.half().float()
        with torch.no_grad():
            logd = head(mix16, cb_downscale(prior_for(i))[None])
        d_out.append(np.log(logd256_to_prob(logd, S)[0] + 1e-300))
        # (e) sufficient statistics: conv0(LN0(mix)) from per-scale centred projections + Gram
        with torch.no_grad():
            Fs = torch.cat(feats, 0).double()                         # (10,2560,64,64)
            mu = Fs.mean(dim=(1, 2, 3))
            Fc = (Fs - mu.view(-1, 1, 1, 1))
            Pk = torch.einsum("kc,schw->skhw", Wp.double(), Fc)         # (10,8,64,64)
            G = (Fc.reshape(10, -1) @ Fc.reshape(10, -1).T) / Fc[0].numel()
            a = scale_w.double()
            var = a @ G @ a
            y_ss = torch.einsum("s,skhw->khw", a, Pk) / torch.sqrt(var + 1e-12) + (W0 @ lnb).view(-1, 1, 1)
            y_dir = net.conv0(net.layernorm0(mix.double().float()))[0].double()
            e_err.append(float((y_ss - y_dir).abs().max() / y_dir.abs().max()))
        print(f"  cached/suffstat img {ids[i]} done", flush=True)
    del m
    np.savez(CPU_NPZ, a=np.array(a_out), b=np.array(b_out), d=np.array(d_out), e=np.array(e_err), t=res['cpu_f64buf_s_per_img'])
# (c) MPS
dev = "mps" if torch.backends.mps.is_available() else "cpu"
import contextlib
with contextlib.nullcontext():   # MPS part < 1 min (no lock needed); shares the GPU with our own cache pass
    mm = load_msdb(dev, cast_fp32=True)
    full_forward(mm, dev, 0)                                  # warm-up
    tc = time.time()
    c_out = [full_forward(mm, dev, i) for i in test_idx]
    torch.mps.synchronize() if dev == "mps" else None
    res["mps_s_per_img"] = (time.time() - tc) / N
    del mm
    if dev == "mps":
        torch.mps.empty_cache()
print(f"(c) {dev}: {res['mps_s_per_img']:.2f}s/img", flush=True)

def cmp(x, y):
    px, py = np.exp(x), np.exp(y)
    return {"max_abs_logd": float(np.abs(x - y).max()), "cc_density": cc(px, py)}

gt512 = {i: load_gt(ids[i], S) for i in test_idx}
rows = []
for k, i in enumerate(test_idx):
    pa = np.exp(a_out[k])
    lab = "fold_A" if i % 2 == 0 else "fold_B"
    pub = ck[lab][str(i)]
    mine = [cc(pa, gt512[i]), sim(pa, gt512[i]), kld(pa, gt512[i])]
    mine_d = [cc(np.exp(d_out[k]), gt512[i]), sim(np.exp(d_out[k]), gt512[i]), kld(np.exp(d_out[k]), gt512[i])]
    mine_c = [cc(np.exp(c_out[k]), gt512[i]), sim(np.exp(c_out[k]), gt512[i]), kld(np.exp(c_out[k]), gt512[i])]
    rows.append({"id": ids[i], "idx": i, "published_512": pub, "cpu_orig_512": mine,
                 "mps_512": mine_c, "cached_fp16_head_512": mine_d,
                 "a_vs_b": cmp(a_out[k], b_out[k]), "a_vs_c_mps": cmp(a_out[k], c_out[k]),
                 "a_vs_d_cached": cmp(a_out[k], d_out[k]), "suffstat_rel_err": e_err[k]})
    print(json.dumps(rows[-1]), flush=True)
res["rows"] = rows
res["max_abs_diff_published_vs_cpu_orig_CC"] = max(abs(r["published_512"][0] - r["cpu_orig_512"][0]) for r in rows)
res["max_abs_diff_CC_cpu_vs_mps"] = max(abs(r["mps_512"][0] - r["cpu_orig_512"][0]) for r in rows)
res["max_abs_diff_CC_cpu_vs_cached"] = max(abs(r["cached_fp16_head_512"][0] - r["cpu_orig_512"][0]) for r in rows)
res["max_suffstat_rel_err"] = max(e_err)
for k in ["max_abs_diff_published_vs_cpu_orig_CC", "max_abs_diff_CC_cpu_vs_mps", "max_abs_diff_CC_cpu_vs_cached", "max_suffstat_rel_err"]:
    print(k, res[k])
write_json(res, os.path.join(CACHE, "validation_10img.json"))
print("saved", os.path.join(CACHE, "validation_10img.json"))
