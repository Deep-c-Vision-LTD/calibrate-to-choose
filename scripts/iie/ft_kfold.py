"""
K-fold in-domain adaptation of DeepGaze IIE on ADD1000 (cached readout inputs).

  python -I ft_kfold.py --row iie_ft_kl   --loss kl384  --epochs 20 --primary nosigma
  python -I ft_kfold.py --row iie_ft_klcc --loss klcc64 --epochs 60 --primary sigma
  python -I ft_kfold.py --row iie_ft_klcc_grouped --split splits/groups_folds.json ...

Writes results/<row>_per_image.json / _summary.json for the primary variant and
results/<row>_<other>_per_image.json / _summary.json for the other (sigma / nosigma).
Per-fold partial results go to logs/iie_partial/ so an interrupted run resumes.
"""
import os, sys, json, time, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import iie_common as C
import iie_train as T

ap = argparse.ArgumentParser()
ap.add_argument("--row", required=True)
ap.add_argument("--split", default=C.SPLIT)
ap.add_argument("--loss", choices=["kl384", "klcc64"], required=True)
ap.add_argument("--epochs", type=int, required=True)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--device", default="mps")
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--folds", default="all")
ap.add_argument("--primary", choices=["sigma", "nosigma"], required=True)
ap.add_argument("--note", default="")
ap.add_argument("--aggregate_only", action="store_true")
a = ap.parse_args()
torch.set_num_threads(a.threads)
CMD = "python -I " + " ".join([os.path.abspath(__file__)] + sys.argv[1:])
LOG = os.path.join(C.KBS, "logs", f"{a.row}.log")
PART = os.path.join(C.KBS, "logs", "iie_partial"); os.makedirs(PART, exist_ok=True)
CKPT = os.path.join(C.CACHE, "ckpt"); os.makedirs(CKPT, exist_ok=True)


def log(s):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


D = T.Data()
sp = json.load(open(a.split))
folds = [[str(i) for i in f] for f in sp["folds"]]
all_ids = [i for f in folds for i in f]
assert len(all_ids) == len(set(all_ids)), "folds overlap"
assert set(all_ids) <= set(D.ids), "unknown ids in split"
fold_list = list(range(len(folds))) if a.folds == "all" else [int(x) for x in a.folds.split(",")]
orders = T.published_orders(folds, a.epochs, n_total=1000)
log(f"{'aggregate' if a.aggregate_only else 'start'} row={a.row} split={a.split} loss={a.loss} epochs={a.epochs} lr={a.lr} bs={a.bs} dev={a.device} "
    f"folds={fold_list} sizes={[len(f) for f in folds]}")
t_all = time.time()
for k in ([] if a.aggregate_only else fold_list):
    pp = os.path.join(PART, f"{a.row}_fold{k}.json")
    if os.path.exists(pp):
        log(f"fold {k}: partial exists, skipping"); continue
    te = folds[k]
    tr = [i for kk in range(len(folds)) if kk != k for i in folds[kk]]
    rng = np.random.default_rng(k)
    sel = [str(x) for x in rng.choice(tr, size=min(C.SIGMA_SELECT_N, len(tr)), replace=False)]
    res = T.train_eval(D, tr, te, orders[k], a.loss, a.lr, a.bs, a.device, sel, log=log, tag=f"fold {k}",
                       ckpt_path=os.path.join(CKPT, f"{a.row}_fold{k}.pt"))
    res["fold"] = k; res["sigma_select_ids"] = sel
    json.dump(res, open(pp, "w"))

# aggregate if all folds present
parts = {}
for k in range(len(folds)):
    pp = os.path.join(PART, f"{a.row}_fold{k}.json")
    if os.path.exists(pp):
        parts[k] = json.load(open(pp))
if len(parts) < len(folds):
    log(f"only {len(parts)}/{len(folds)} folds done; not aggregating"); sys.exit(0)

order = {i: n for n, i in enumerate(D.ids)}
def collect(key):
    rows = []
    for k, r in parts.items():
        for x in r[key]:
            rows.append({"id": x["id"], "fold": k, **{m: x[m] for m in ("CC", "SIM", "KLD")},
                         **({"sigma": x["sigma"]} if "sigma" in x else {})})
    rows.sort(key=lambda r: order[r["id"]])
    return rows

recipe = {"kl384": "published ft_deepgaze_iie_indomain.py recipe: KL(g||p) on 384 grid (g = min-subtracted "
                   "sum-normalised bilinear-384 GT), Adam lr %g, batch %d, %d epochs" % (a.lr, a.bs, a.epochs),
          "klcc64": "our kfold_train_eval.py loss KL(g64||p64) + (1 - CC(p64,g64)) (weights 1, 1) on the 64x64 loss grid; "
                    "p64 = IIE density exactly sum-pooled 6x6 from 384; Adam lr %g, batch %d, %d epochs" % (a.lr, a.bs, a.epochs)}[a.loss]
common = {
    "split_file": a.split, "fold_sizes": [len(f) for f in folds],
    "trainable_params_trained": parts[0]["trainable_params"],
    "trained_parameters": "all DeepGaze IIE readouts: 4 backbones x 30 components (saliency network + fixation-selection network + finalizer sigma and centre-bias weight); backbones frozen",
    "features": "released DeepGaze IIE backbones on PIL-bilinear 384x384 input, cached at the 12x12 readout grid (float32; bit-exact, logs/iie_verify_cache.json); training forward = stacked re-implementation of the released readouts (logs/iie_verify_stacked_*.json); evaluation = released module code path after writing back the trained parameters",
    "recipe": recipe,
    "centre_prior": "per fold, log(to_prob(sum of the training folds' min-subtracted 384 maps)) + 1e-8 (as ft_deepgaze_iie_indomain.py), DeepGaze's centre-bias input",
    "shuffle": "published stream: np.random.default_rng(0), permutation(1000) consumed, then in-place shuffle of the fold's training list each epoch, continuing across folds",
    "training_maps_per_fold": [parts[k]["n_train"] for k in sorted(parts)],
    "steps_per_fold": [parts[k]["steps"] for k in sorted(parts)],
    "t_train_s_per_fold": [parts[k]["t_train_s"] for k in sorted(parts)],
    "t_eval_s_per_fold": [parts[k]["t_eval_s"] for k in sorted(parts)],
    "wall_clock_s_total": float(sum(parts[k]["t_train_s"] + parts[k]["t_eval_s"] for k in parts)),
    "epoch_mean_loss_per_fold": {k: parts[k]["epoch_mean_loss"] for k in sorted(parts)},
    "device_per_fold": [parts[k].get("device", a.device) for k in sorted(parts)],
    "custom_finalmix_backward_per_fold": [parts[k].get("fast_final", False) for k in sorted(parts)],
    "command": CMD, "note": a.note,
    "pool_worker_note": "folds may have been trained by pool_worker.py workers (same settings); see device_per_fold",
    "native_resolution": "DeepGaze IIE predicts at its input resolution (384 x 384), so native and unified scores are identical",
}
variants = {"nosigma": ("rows_nosigma", "no extra post-hoc finalizer (IIE's own learned blur only)"),
            "sigma": ("rows_sigma", "plus Gaussian finalizer, sigma selected per fold from [0,2,...,12] by mean CC on 200 training images (np.random.default_rng(fold).choice), scipy gaussian_filter on the 384 density")}
for v, (key, desc) in variants.items():
    row = a.row if v == a.primary else f"{a.row}_{v}"
    rows = collect(key)
    calib = {"prior_maps": parts[0]["n_train"], "sigma_selection_maps": (parts[0]["n_sigma_select"] if v == "sigma" else 0),
             "note": "sigma-selection maps are a subset of the training maps"}
    summ = {"row": row, **C.fold_summary(rows), "n_images": len(rows), "variant": desc,
            "calibration_maps": calib, "training_maps": parts[0]["n_train"],
            "sigma_per_fold": [parts[k]["sigma"] for k in sorted(parts)] if v == "sigma" else None,
            "sigma_curve_train_per_fold": {k: parts[k]["sigma_curve_train"] for k in sorted(parts)} if v == "sigma" else None,
            **common}
    if v == "nosigma":
        pubf = [parts[k]["published_scoring_nosigma"] for k in sorted(parts)]
        summ["published_scoring_min_subtracted_gt"] = {
            "per_fold": pubf, "CC_mean": float(np.mean([p["CC"] for p in pubf])),
            "CC_sd": float(np.std([p["CC"] for p in pubf])),
            "SIM_mean": float(np.mean([p["SIM"] for p in pubf])), "KLD_mean": float(np.mean([p["KLD"] for p in pubf]))}
    json.dump(rows, open(os.path.join(C.RESULTS, f"{row}_per_image.json"), "w"), indent=0)
    json.dump(summ, open(os.path.join(C.RESULTS, f"{row}_summary.json"), "w"), indent=1)
    log(f"{row}: CC={summ['CC_mean']:.4f}+/-{summ['CC_sd_over_folds']:.4f} SIM={summ['SIM_mean']:.4f} KLD={summ['KLD_mean']:.4f}")
log(f"done ({time.time() - t_all:.0f}s this invocation)")
