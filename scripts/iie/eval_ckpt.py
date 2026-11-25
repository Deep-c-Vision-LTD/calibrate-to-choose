"""Recover a pool task whose training finished (checkpoint saved) but whose evaluation
crashed: load the saved readout into a released DeepGazeIIE instance and run the same
evaluation as iie_train.train_eval. Training-side fields are taken from the run log.

  python -I eval_ckpt.py --row iie_ft_klcc --fold 1 --split splits/kfold_seed0.json --t_train 1540 --steps 6000 --log logs/iie_pool_mps_<pid>.log
"""
import os, sys, json, re, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from scipy.ndimage import gaussian_filter
import iie_common as C, iie_train as T
import deepgaze_pytorch
ap = argparse.ArgumentParser()
ap.add_argument("--row", required=True); ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--split", required=True); ap.add_argument("--t_train", type=float, required=True)
ap.add_argument("--steps", type=int, required=True); ap.add_argument("--log", required=True)
ap.add_argument("--train_device", default="mps"); ap.add_argument("--device", default="cpu")
a = ap.parse_args()
torch.set_num_threads(3)
D = T.Data()
folds = [[str(i) for i in f] for f in json.load(open(a.split))["folds"]]
k = a.fold
te = folds[k]; tr = [i for kk in range(len(folds)) if kk != k for i in folds[kk]]
sel = [str(x) for x in np.random.default_rng(k).choice(tr, size=min(C.SIGMA_SELECT_N, len(tr)), replace=False)]
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
sd = torch.load(os.path.join(C.CACHE, "ckpt", f"{a.row}_fold{k}.pt"), map_location="cpu")
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected and all(("features" in m) for m in missing), (unexpected, missing[:5])
C.readout_to(model, a.device)
cb = D.prior(tr)
t1 = time.time()
pt = T.predict_module(model, D, te, cb, a.device)
ps = T.predict_module(model, D, sel, cb, a.device)
sigma, curve = T.select_sigma(ps, D, sel)
rows0, rows_s, pubm = [], [], []
for i in te:
    g = D.raw[D.pos[i]].astype(np.float64)
    rows0.append({"id": i, **C.metrics(pt[i], g)})
    p2 = gaussian_filter(pt[i], sigma) if sigma > 0 else pt[i]
    rows_s.append({"id": i, "sigma": sigma, **C.metrics(p2, g)})
    pubm.append(C.metrics(pt[i], D.minsub(i)))
pat = re.compile(rf"{a.row} fold {k} epoch (\d+) mean loss ([0-9.]+)")
ep_logged = {int(m.group(1)): float(m.group(2)) for m in (pat.search(l) for l in open(a.log)) if m}
res = {"rows_nosigma": rows0, "rows_sigma": rows_s, "sigma": sigma, "sigma_curve_train": curve,
       "published_scoring_nosigma": {m: float(np.mean([r[m] for r in pubm])) for m in ("CC", "SIM", "KLD")},
       "epoch_mean_loss": [ep_logged.get(e) for e in range(max(ep_logged) + 1)],
       "steps": a.steps, "n_train": len(tr), "n_test": len(te), "trainable_params": C.trainable_params(model),
       "t_train_s": a.t_train, "t_eval_s": time.time() - t1, "n_sigma_select": len(sel), "extra_eval": None,
       "fold": k, "sigma_select_ids": sel, "device": a.train_device, "fast_final": True, "split_file": a.split,
       "recovered_from_checkpoint": "training finished on %s; evaluation (identical code) re-run from the saved readout on %s after an MPS float64 error in the original evaluation" % (a.train_device, a.device)}
json.dump(res, open(os.path.join(C.KBS, "logs", "iie_partial", f"{a.row}_fold{k}.json"), "w"))
print(f"{a.row} fold {k}: CC={np.mean([r['CC'] for r in rows0]):.4f} (sigma=0) CC={np.mean([r['CC'] for r in rows_s]):.4f} (sigma={sigma})")
