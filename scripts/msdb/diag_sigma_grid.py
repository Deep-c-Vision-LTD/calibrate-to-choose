"""
Diagnostic (NOT the fixed protocol): the post-hoc sigma rule selected the grid maximum (12) in
every fold for zero-shot MSDB + fold prior. Re-select sigma on the same 200 training images per
fold from an extended grid [0..12 step 2, 14, 16, 20, 24, 28, 32] and score the test folds, to
show how much the grid cap matters. Row: msdb_zeroshot_foldprior_extgrid (diagnostic only).

  python -I diag_sigma_grid.py
"""
import os, sys, time
import numpy as np, torch
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msdb_common import *          # noqa
torch.set_num_threads(3)
t0 = time.time()
EXT = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
ids, folds, fold_of = load_split()
R = torch.from_numpy(np.load(os.path.join(CACHE, "readout_zs.npy")))[:, None]
net, sigma0, cbw0, scale_w, sd = build_sal_net_pretrained()
head = MSDBHead(net, sigma0, cbw0).eval()
pos = {i: k for k, i in enumerate(ids)}
G384 = {i: load_gt(i, EVAL) for i in ids}
per, logs = [], {}
for f in range(5):
    tr = [i for k in range(5) if k != f for i in folds[k]]
    cbd = cb_downscale(fit_prior(tr))
    sel = np.random.default_rng(f).choice(np.array(tr), size=SIGMA_SELECT_N, replace=False)
    with torch.no_grad():
        pred = lambda i: logd256_to_prob(head.finalize(R[pos[i]:pos[i] + 1], cbd[None]), EVAL)[0]
        P = {i: pred(i) for i in list(sel) + folds[f]}
    table = {s: float(np.mean([cc(gaussian_filter(P[i], s) if s else P[i], G384[i]) for i in sel])) for s in EXT}
    best = max(table, key=table.get)
    logs[f] = {"sigma": best, "train_table": table}
    for i in folds[f]:
        p = gaussian_filter(P[i], best) if best else P[i]
        per.append({"id": i, "fold": f, **metrics(p, G384[i])})
    print(f"fold {f}: sigma {best} test CC {np.mean([r['CC'] for r in per if r['fold'] == f]):.4f}", flush=True)
fm = [{"fold": f, **{m: float(np.mean([r[m] for r in per if r["fold"] == f])) for m in ["CC", "SIM", "KLD"]}} for f in range(5)]
out = {"row": "msdb_zeroshot_foldprior_extgrid", "status": "DIAGNOSTIC ONLY (sigma grid extended beyond the fixed protocol)",
       "grid": EXT, "per_fold": fm, "sigma_per_fold": logs,
       **{f"{m}_mean": float(np.mean([x[m] for x in fm])) for m in ["CC", "SIM", "KLD"]},
       "CC_sd_folds_ddof1": float(np.std([x["CC"] for x in fm], ddof=1)),
       "calibration_maps": "800 training maps per fold (prior); sigma on 200 of them", "training_maps": 0,
       "wall_clock_s": time.time() - t0, "command": "cd scripts/msdb && python -I diag_sigma_grid.py"}
write_json(per, os.path.join(RESULTS, "msdb_zeroshot_foldprior_extgrid_per_image.json"))
write_json(out, os.path.join(RESULTS, "msdb_zeroshot_foldprior_extgrid_summary.json"))
print(f"extgrid: CC {out['CC_mean']:.4f} SIM {out['SIM_mean']:.4f} KLD {out['KLD_mean']:.4f}")
