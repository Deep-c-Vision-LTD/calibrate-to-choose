"""
Matched-budget learning curve for adapted DeepGaze IIE on the fold-0 test set.
Subset rule of sample_efficiency.py: idx = default_rng(0).permutation(1000); test = idx[:200]
(= fold 0 of splits/kfold_seed0.json); pool = idx[200:]; for n in SIZES and resample r,
train ids = default_rng(1000*n + r).choice(pool, n, replace=False); shuffle stream
default_rng(r) (fresh copy each epoch). Training policy at small n: FIXED EPOCHS (same as our
curve: our 60-epoch budget at every n, so fewer steps at small n), batch min(8, n).
Centre prior from the n training maps only; sigma finalizer selected on all n training maps
(sample_efficiency.py rule).

  python -I ft_curves.py --row iie_curve_klcc --loss klcc64 --epochs 60 [--sizes 25,50] [--device cpu]
"""
import os, sys, json, time, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import iie_common as C
import iie_train as T

ap = argparse.ArgumentParser()
ap.add_argument("--row", required=True)
ap.add_argument("--loss", choices=["kl384", "klcc64"], required=True)
ap.add_argument("--epochs", type=int, required=True)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--device", default="mps")
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--sizes", default="25,50,100,200,400,800")
ap.add_argument("--resamples", type=int, default=3)
ap.add_argument("--aggregate_only", action="store_true")
ap.add_argument("--note", default="")
a = ap.parse_args()
torch.set_num_threads(a.threads)
CMD = "python -I " + " ".join([os.path.abspath(__file__)] + sys.argv[1:])
LOG = os.path.join(C.KBS, "logs", f"{a.row}.log")
PART = os.path.join(C.KBS, "logs", "iie_partial"); os.makedirs(PART, exist_ok=True)
ALL_SIZES = [25, 50, 100, 200, 400, 800]


def log(s):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    open(LOG, "a").write(line + "\n")


D = T.Data()
idx = np.random.default_rng(0).permutation(len(D.ids))
te = [D.ids[j] for j in idx[:200]]
pool = [D.ids[j] for j in idx[200:]]
assert set(te) == set(D.folds[0]), "curve test set must equal fold 0"
sizes = [int(x) for x in a.sizes.split(",")]
if not a.aggregate_only:
    log(f"start row={a.row} loss={a.loss} epochs={a.epochs} sizes={sizes} dev={a.device}")
    for n in sizes:
        for r in range(a.resamples):
            pp = os.path.join(PART, f"{a.row}_n{n}_r{r}.json")
            if os.path.exists(pp):
                continue
            tr = [str(x) for x in np.random.default_rng(1000 * n + r).choice(np.array(pool), size=n, replace=False)]
            orders = T.seeded_orders(tr, a.epochs, seed=r)
            res = T.train_eval(D, tr, te, orders, a.loss, a.lr, min(a.bs, n), a.device, tr, log=log,
                               tag=f"n={n} r={r}")
            res.update({"n": n, "resample": r, "train_ids": tr, "seed": r})
            json.dump(res, open(pp, "w"))

# aggregate whatever is complete
rows_s, rows_0, per_n = [], [], []
for n in ALL_SIZES:
    runs = []
    for r in range(a.resamples):
        pp = os.path.join(PART, f"{a.row}_n{n}_r{r}.json")
        if os.path.exists(pp):
            runs.append(json.load(open(pp)))
    if not runs:
        continue
    for res in runs:
        for x in res["rows_sigma"]:
            rows_s.append({"id": x["id"], "fold": 0, "n": n, "resample": res["resample"], "seed": res["seed"],
                           "sigma": x["sigma"], **{m: x[m] for m in ("CC", "SIM", "KLD")}})
        for x in res["rows_nosigma"]:
            rows_0.append({"id": x["id"], "fold": 0, "n": n, "resample": res["resample"], "seed": res["seed"],
                           **{m: x[m] for m in ("CC", "SIM", "KLD")}})
    ent = {"n": n, "n_runs": len(runs), "calibration_maps": {"prior_maps": n, "sigma_selection_maps": n},
           "training_maps": n, "steps_per_run": [x["steps"] for x in runs],
           "t_train_s_per_run": [x["t_train_s"] for x in runs], "sigma_per_run": [x["sigma"] for x in runs]}
    for tag, key in (("sigma", "rows_sigma"), ("nosigma", "rows_nosigma")):
        for m in ("CC", "SIM", "KLD"):
            v = [float(np.mean([x[m] for x in res[key]])) for res in runs]
            ent[f"{tag}_{m}_runs"] = v
            ent[f"{tag}_{m}_mean"] = float(np.mean(v))
            ent[f"{tag}_{m}_sd"] = float(np.std(v))
    per_n.append(ent)
    log(f"n={n}: CC(sigma)={ent['sigma_CC_mean']:.4f}+/-{ent['sigma_CC_sd']:.4f} over {len(runs)} runs; "
        f"CC(nosigma)={ent['nosigma_CC_mean']:.4f}")

ref = {}
for rr in ("iie_zeroshot_uniform", "iie_zeroshot_foldprior", "iie_zeroshot_foldprior_sigma", "iie_zeroshot_adprior"):
    p = os.path.join(C.RESULTS, f"{rr}_per_image.json")
    if os.path.exists(p):
        rows = [x for x in json.load(open(p)) if x["fold"] == 0]
        ref[rr] = {m: float(np.mean([x[m] for x in rows])) for m in ("CC", "SIM", "KLD")}
summ = {"row": a.row, "test_set": "fold 0 of splits/kfold_seed0.json (= sample_efficiency.py held-out 200)",
        "per_n": per_n, "recipe_loss": a.loss, "epochs": a.epochs, "lr": a.lr, "batch": f"min({a.bs}, n)",
        "training_policy_small_n": "fixed epochs (%d at every n, as our sample_efficiency.py curve), so steps = epochs x ceil(n/batch); not matched steps" % a.epochs,
        "subset_rule": "sample_efficiency.py: default_rng(1000*n + r).choice(pool, n, replace=False), pool = default_rng(0).permutation(1000)[200:]",
        "shuffle": "default_rng(r), fresh copy of the training list shuffled every epoch",
        "centre_prior": "from the n training maps only (log(to_prob(sum of min-subtracted 384 maps)) + 1e-8)",
        "sigma_rule": "selected on all n training maps (sample_efficiency.py rule), grid [0,2,...,12]",
        "primary_metric_variant": "sigma (with finalizer); nosigma also stored",
        "n0_reference_fold0": ref,
        "trainable_params_trained": 3059160, "command": CMD, "note": a.note,
        "per_image_files": {"sigma": f"{a.row}_per_image.json", "nosigma": f"{a.row}_nosigma_per_image.json"}}
json.dump(rows_s, open(os.path.join(C.RESULTS, f"{a.row}_per_image.json"), "w"), indent=0)
json.dump(rows_0, open(os.path.join(C.RESULTS, f"{a.row}_nosigma_per_image.json"), "w"), indent=0)
json.dump(summ, open(os.path.join(C.RESULTS, f"{a.row}_summary.json"), "w"), indent=1)
log("aggregated")
