"""
Check that the cached-feature path reproduces the full DeepGaze IIE forward pass.
Runs on CPU on the first N images (ids_sorted order), with (a) uniform centre bias and
(b) an ADD1000 centre bias (mean of 100 other GT maps). Compares log densities and
CC/SIM/KLD for fp32 features and fp16-rounded features.

  python -I verify_cache.py [N]
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import iie_common as C

torch.set_num_threads(6)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
import deepgaze_pytorch
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
ids, folds = C.load_split()
test = ids[:N]
prior_ids = ids[500:600]
cb_ad = C.centerbias_from_maps(np.stack([C.to_prob(C.load_map_raw(i)) for i in prior_ids]))
cbs = {"uniform": np.zeros((C.S, C.S), np.float64), "adprior": cb_ad}
rep = {"N": N, "ids": test, "trainable_params": C.trainable_params(model), "cases": {}}
t0 = time.time()
with torch.no_grad():
    for name, cb in cbs.items():
        cbt = torch.from_numpy(cb.astype(np.float32))[None]
        rows = []
        for i in test:
            x = torch.from_numpy(C.load_img_384(i))[None]
            full = model(x, cbt)[0, 0].double()
            feats = C.iie_features(model, x)
            c32 = C.forward_cached(model, feats, cbt)[0, 0].double()
            c16 = C.forward_cached(model, [f.half().float() for f in feats], cbt)[0, 0].double()
            g = C.load_map_raw(i)
            mf = C.metrics(torch.exp(full).numpy(), g)
            m32 = C.metrics(torch.exp(c32).numpy(), g)
            m16 = C.metrics(torch.exp(c16).numpy(), g)
            rows.append({"id": i,
                         "maxabs_logd_fp32": float((full - c32).abs().max()),
                         "maxabs_logd_fp16": float((full - c16).abs().max()),
                         "full": mf, "cached_fp32": m32, "cached_fp16": m16})
            print(name, i, rows[-1]["maxabs_logd_fp32"], rows[-1]["maxabs_logd_fp16"],
                  {k: round(mf[k] - m16[k], 7) for k in mf}, flush=True)
        rep["cases"][name] = {
            "rows": rows,
            "max_abs_logd_fp32": max(r["maxabs_logd_fp32"] for r in rows),
            "max_abs_logd_fp16": max(r["maxabs_logd_fp16"] for r in rows),
            "max_abs_dCC_fp32": max(abs(r["full"]["CC"] - r["cached_fp32"]["CC"]) for r in rows),
            "max_abs_dCC_fp16": max(abs(r["full"]["CC"] - r["cached_fp16"]["CC"]) for r in rows),
            "max_abs_dSIM_fp16": max(abs(r["full"]["SIM"] - r["cached_fp16"]["SIM"]) for r in rows),
            "max_abs_dKLD_fp16": max(abs(r["full"]["KLD"] - r["cached_fp16"]["KLD"]) for r in rows),
        }
rep["seconds"] = time.time() - t0
json.dump(rep, open(os.path.join(C.KBS, "logs", "iie_verify_cache.json"), "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in rep["cases"].items()}, indent=1))
print("trainable params", rep["trainable_params"], "seconds", rep["seconds"])
