"""
End-to-end check of a TRAINED readout: load a saved adapted readout into a released
deepgaze_pytorch.DeepGazeIIE instance and run the full forward pass (image -> backbones ->
readouts, no feature cache) on a few test images of that fold; compare with the per-image
scores recorded by the cached pipeline (no-sigma variant).

  python -I verify_trained_full.py --row iie_ft_klcc_e20 --fold 0 --n 6
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import iie_common as C, iie_train as T
import deepgaze_pytorch

ap = argparse.ArgumentParser()
ap.add_argument("--row", default="iie_ft_klcc_e20")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--split", default=C.SPLIT)
ap.add_argument("--n", type=int, default=6)
a = ap.parse_args()
torch.set_num_threads(4)
t0 = time.time()
D = T.Data()
folds = [[str(i) for i in f] for f in json.load(open(a.split))["folds"]]
k = a.fold
te = folds[k][:a.n]
tr = [i for kk in range(len(folds)) if kk != k for i in folds[kk]]
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
sd = torch.load(os.path.join(C.CACHE, "ckpt", f"{a.row}_fold{k}.pt"), map_location="cpu")
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected and all(("features" in m) for m in missing)
cb = torch.from_numpy(D.prior(tr).astype(np.float32))[None]
rec = {r["id"]: r for r in json.load(open(os.path.join(C.RESULTS, f"{a.row}_nosigma_per_image.json")))
       if r["fold"] == k}
rows = []
with torch.no_grad():
    for i in te:
        x = torch.from_numpy(C.load_img_384(i))[None]
        full = model(x, cb)[0, 0].double()
        feats = [torch.from_numpy(np.ascontiguousarray(f)) for f in C.split_channels(D.feats[[D.pos[i]]])]
        cached = C.forward_cached(model, feats, cb)[0, 0].double()
        g = D.raw[D.pos[i]].astype(np.float64)
        m = C.metrics(torch.exp(full).numpy(), g)
        rows.append({"id": i, "maxabs_logd_full_vs_cached": float((full - cached).abs().max()),
                     "full": m, "recorded": {q: rec[i][q] for q in ("CC", "SIM", "KLD")},
                     "abs_dCC": abs(m["CC"] - rec[i]["CC"])})
        print(i, rows[-1]["maxabs_logd_full_vs_cached"], round(rows[-1]["abs_dCC"], 9), flush=True)
rep = {"row": a.row, "fold": k, "ids": te, "rows": rows,
       "max_abs_dCC_full_vs_recorded": max(r["abs_dCC"] for r in rows),
       "max_abs_logd_full_vs_cached": max(r["maxabs_logd_full_vs_cached"] for r in rows),
       "seconds": time.time() - t0}
json.dump(rep, open(os.path.join(C.KBS, "logs", f"iie_verify_trained_full_{a.row}_f{k}.json"), "w"), indent=1)
print({q: rep[q] for q in ("max_abs_dCC_full_vs_recorded", "max_abs_logd_full_vs_cached", "seconds")})
