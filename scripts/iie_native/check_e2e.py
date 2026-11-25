"""End-to-end re-score of a trained native fold WITHOUT the feature cache, the GT cache or the stacked code:
fresh released DeepGazeIIE + saved readout checkpoint, full model forward model(image, centrebias) on the raw
native image, density resized to 384 (torch bilinear), the fold's selected sigma, GT re-read from the PNG
(bilinear 384), iie_common.metrics. Compares with the stored per-image rows; also re-checks that test ids are
disjoint from the training ids used for the prior and the sigma selection.
  python -I check_e2e.py --fold 0 --n 20 [--threads 3]"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import native_common as N
C = N.C
ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, default=0); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--threads", type=int, default=3); a = ap.parse_args()
torch.set_num_threads(a.threads)
import deepgaze_pytorch
k = a.fold
part = json.load(open(os.path.join(N.NCACHE, "partial", f"{N.ROW}_fold{k}.json")))
sp = json.load(open(C.SPLIT)); folds = [[str(i) for i in f] for f in sp["folds"]]
te = folds[k]; tr = [i for kk in range(5) if kk != k for i in folds[kk]]
assert not set(te) & set(tr) and not set(part["sigma_select_ids"]) & set(te)
assert [r["id"] for r in part["rows_sigma"]] == te
acc = np.zeros((C.S, C.S))
for i in tr:                                  # prior from the PNGs directly (training ids only)
    m = C.load_map_raw(i); m = m - m.min(); s = m.sum(); acc += m / s if s > 0 else m + 1.0 / m.size
prior = np.log(C.to_prob(acc) + C.EPS)
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
sd = torch.load(os.path.join(N.NCACHE, "ckpt", f"{N.ROW}_fold{k}.pt"))
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected and all(("saliency_networks" not in x and "fixation_selection_networks" not in x and "finalizers" not in x) for x in missing)
stored = {r["id"]: r for r in part["rows_sigma"]}
sig = part["sigma"]
out = []
with torch.no_grad():
    for i in te[:a.n]:
        x = torch.from_numpy(N.load_img_native(i))[None]
        H, W = x.shape[2:]
        cb = N.native_centerbias(prior, H, W)
        d = torch.exp(model(x, cb)[0, 0].double())
        d384 = F.interpolate(d[None, None], size=(C.S, C.S), mode="bilinear", align_corners=False)[0, 0].numpy()
        p = gaussian_filter(d384, sig) if sig > 0 else d384
        mt = C.metrics(p, C.load_map_raw(i))
        out.append({"id": i, **{m: [mt[m], stored[i][m]] for m in ("CC", "SIM", "KLD")}})
mx = {m: max(abs(r[m][0] - r[m][1]) for r in out) for m in ("CC", "SIM", "KLD")}
rep = {"fold": k, "n": len(out), "sigma": sig, "max_abs_diff_vs_stored": mx,
       "mean_e2e": {m: float(np.mean([r[m][0] for r in out])) for m in mx},
       "mean_stored": {m: float(np.mean([r[m][1] for r in out])) for m in mx}, "pass": bool(max(mx.values()) < 1e-4)}
print(json.dumps(rep, indent=1))
json.dump({**rep, "rows": out}, open(os.path.join(N.NCACHE, f"check_e2e_fold{k}.json"), "w"), indent=1)
