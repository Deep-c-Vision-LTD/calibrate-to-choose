"""Turn the per-run outputs of ft_sum_folds.py --curve (cloud GPU runs sum_curve_a / sum_curve_b; scripts/sum/run_sum_ft_gpu.sh
makes the same call, copied from their output folders) into results files that mirror the other learning curves:

  <prefix>_per_image.json          rows {id, fold, n, resample, seed, sigma, CC, SIM, KLD, ...}
                                   (same keys compile_results.multi_curve reads: id, n, resample, CC/SIM/KLD)
  <prefix>_nosigma_per_image.json  same rows with CC/SIM/KLD = the no-sigma scores (as iie_curve_*_nosigma)
  <prefix>_summary.json            header + "curve" (per n) + "runs" (per (n, r), with train_ids), as
                                   msdb_adapt_curves_summary.json

The n = 800 point is NOT a curve run: it is taken from the finished k-fold row sum_ft, fold 0
(results/sum_ft_per_image.json + results/sum_ft_summary.json): the same 800 training ids, but sigma
chosen on 200 of the 800 training maps (default_rng(0)), data order default_rng(0) over the pool in
fold order and torch seed 0. That is stated in the summary and in every n = 800 row.

Every curve run is validated before anything is written: meta parses; train_ids == the stored
subsets (splits/curve_subsets_seed0.json, = msdb/ours train_ids); test ids == fold 0 in order;
steps == steps_planned == 12 x ceil(n/16); no debug flags (max_steps, limit_eval); sigma chosen on
all n training maps and == argmax of its table; orders_match_expected; 200 finite rows; and (unless
--no_rescore) every per-image score re-computed from the saved 256 maps with ours_lib's scorer to 1e-9.
Writing into results/ additionally requires the complete plan (13 runs + n = 800) and a clean
validation; anything else (smoke tests, partial downloads) must use another --out_dir.

python -I scripts/sum/finalize_curve.py --runs runs/sum_curve_a/ckpt/sum_curve_a \
    runs/sum_curve_b/ckpt/sum_curve_b --jobs sum_curve_a,sum_curve_b [--out_dir results]
(ours_lib needs torch)
"""
import argparse, glob, json, os, re, sys

K = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="dirs holding <tag>_n{n}_r{r}_meta.json etc.")
ap.add_argument("--out_dir", default=os.path.join(K, "results"))
ap.add_argument("--prefix", default="sum_ft_curve")
ap.add_argument("--plan", default="25:0,1,2/50:0,1,2/100:0,1,2/200:0,1,2/400:0")
ap.add_argument("--split", default=os.path.join(K, "splits", "kfold_seed0.json"))
ap.add_argument("--expect", default=os.path.join(K, "splits", "curve_subsets_seed0.json"))
ap.add_argument("--n800_from", default=os.path.join(K, "results"), help="dir with sum_ft_per_image.json / sum_ft_summary.json")
ap.add_argument("--n800_maps", default=os.path.join(K, "runs", "sum_ft", "ckpt", "sum_ft"),
                help="dir with sum_ft_fold0_test_maps256.npy / _test_ids.json (re-score check of the n=800 rows)")
ap.add_argument("--no_n800", action="store_true", help="omit the n = 800 point (never allowed into results/)")
ap.add_argument("--jobs", default="", help="comma-separated run names, recorded in the summary")
ap.add_argument("--allow_partial", action="store_true", help="accept a subset of the plan (never into results/)")
ap.add_argument("--allow_debug", action="store_true", help="accept max_steps/limit_eval runs (never into results/)")
ap.add_argument("--no_rescore", action="store_true")
ap.add_argument("--note", default="")
args = ap.parse_args()

sys.path.insert(0, os.path.join(K, "scripts", "sum")); sys.path.insert(0, os.path.join(K, "scripts", "ours"))
import numpy as np                     # noqa: E402
from PIL import Image                  # noqa: E402
from scipy.ndimage import gaussian_filter   # noqa: E402
import sum_common as sc                # noqa: E402

TO_RESULTS = os.path.realpath(args.out_dir) == os.path.realpath(os.path.join(K, "results"))
folds = sc.load_split(args.split)
F0 = list(folds[0])
EXP = json.load(open(args.expect))
PLAN = sc.parse_curve(args.plan)
METRICS = ("CC", "SIM", "KLD")
problems, warnings = [], []
if TO_RESULTS and (args.allow_partial or args.allow_debug or args.no_n800 or args.no_rescore):
    sys.exit("refusing: --allow_partial/--allow_debug/--no_n800/--no_rescore cannot write into results/")

L = None
if not args.no_rescore:
    import ours_lib as L               # noqa: E402  (independent scorer: same protocol, separate code)
    GT = {i: L.load_gt_eval(os.path.join(L.FLAT, "maps", f"{i}.png")) for i in F0}


def rescore(maps, ids, rows, sigma, label):
    """re-score every row from the saved 256x256 maps with ours_lib (PIL bilinear to 384)"""
    worst = 0.0
    for k, (i, row) in enumerate(zip(ids, rows)):
        p = np.asarray(Image.fromarray(maps[k]).resize((384, 384), Image.BILINEAR), np.float64)
        ps = gaussian_filter(p, sigma) if sigma > 0 else p
        vals = {"CC": L.cc(ps, GT[i]), "SIM": L.sim(ps, GT[i]), "KLD": L.kld(ps, GT[i]),
                "CC_nosigma": L.cc(p, GT[i]), "SIM_nosigma": L.sim(p, GT[i]), "KLD_nosigma": L.kld(p, GT[i])}
        for key, v in vals.items():
            worst = max(worst, abs(v - row[key]))
    if worst > 1e-9:
        problems.append(f"{label}: re-scored per-image values differ by up to {worst:.3g}")
    return worst


# ------------------------------------------------------------------ collect curve runs
found, errors = {}, []
for d in args.runs:
    for mp in sorted(glob.glob(os.path.join(d, "*_n*_r*_meta.json"))):
        m_ = re.match(r"(.+)_n(\d+)_r(\d+)_meta\.json$", os.path.basename(mp))
        if not m_:
            continue
        key = (int(m_.group(2)), int(m_.group(3)))
        if key in found:
            problems.append(f"duplicate run n={key[0]} r={key[1]}: {found[key]['pre']} and {mp}")
            continue
        found[key] = {"pre": mp[:-len("_meta.json")], "tag": m_.group(1), "dir": d}
    errors += sorted(glob.glob(os.path.join(d, "*_n*_r*_error.json")))
for e in errors:
    m_ = re.match(r".+_n(\d+)_r(\d+)_error\.json$", os.path.basename(e))
    if m_ and (int(m_.group(1)), int(m_.group(2))) not in found:
        problems.append(f"run failed in the job and has no result: {e}")
    else:
        warnings.append(f"error marker present for a run that later completed: {e}")
missing = [k for k in PLAN if k not in found]
extra = [k for k in found if k not in PLAN]
if missing:
    (warnings if args.allow_partial else problems).append(f"planned runs missing: {missing}")
if extra:
    problems.append(f"runs outside the plan: {extra}")

runs, rows_all = [], []
for (n, r) in [k for k in PLAN if k in found]:
    pre = found[(n, r)]["pre"]; lab = f"n={n} r={r}"
    try:
        meta = json.load(open(pre + "_meta.json"))
        rows = json.load(open(pre + "_per_image.json"))
        te = json.load(open(pre + "_test_ids.json"))
        maps = np.load(pre + "_test_maps256.npy")
    except Exception as e:   # noqa: BLE001
        problems.append(f"{lab}: cannot read outputs ({e!r})"); continue
    chk = {
        "train_ids == stored subset": meta["train_ids"] == EXP["train_ids"].get(f"{n}:{r}"),
        "train_ids == rule": meta["train_ids"] == sc.curve_subset(folds, 0, n, r),
        "meta n/r": (meta["n"], meta["r"]) == (n, r),
        "test ids == fold 0 (ordered)": te == F0 and [x["id"] for x in rows] == F0,
        "maps 200x256x256 finite": maps.shape == (len(F0), 256, 256) and bool(np.isfinite(maps).all()),
        "rows finite": all(np.isfinite([x[k] for k in ("CC", "SIM", "KLD", "CC_nosigma")]).all() for x in rows),
        "sigma == argmax(table)": int(max(meta["sigma_table_trainCC"].items(), key=lambda kv: kv[1])[0]) == meta["sigma"],
        "rows carry the run's sigma": all(x["sigma"] == meta["sigma"] for x in rows),
        "orders_match_expected": meta.get("orders_match_expected") is True,
        "params trained == 57,501,196": meta.get("n_trainable_params") == 57501196,
    }
    dbg = {"steps == steps_planned == 12 x ceil(n/16)":
               meta["steps"] == meta["steps_planned"] == 12 * -(-n // 16) and meta["epochs"] == 12,
           "no max_steps / limit_eval": not meta.get("max_steps") and not meta.get("limit_eval"),
           "sigma chosen on all n training maps": meta["n_sigma_select"] == n,
           "200 test images": meta["n_test"] == len(F0) == 200}
    for name, ok in {**chk, **dbg}.items():
        if not ok:
            (warnings if (args.allow_debug and name in dbg) or
             (args.allow_debug and name in ("test ids == fold 0 (ordered)", "maps 200x256x256 finite")) else problems
             ).append(f"{lab}: check failed: {name}")
    worst = None
    if L is not None:
        ids_scored = te if len(te) < len(F0) else F0
        if all(i in GT for i in ids_scored):
            worst = rescore(maps, ids_scored, rows, meta["sigma"], lab)
    runs.append({"n": n, "resample": r, "train_ids": meta["train_ids"], "posthoc_sigma": meta["sigma"],
                 **{k: float(np.mean([x[k] for x in rows])) for k in METRICS},
                 **{f"{k}_nosigma": float(np.mean([x[f"{k}_nosigma"] for x in rows])) for k in METRICS},
                 **{f"{k}_native256": float(np.mean([x[f"{k}_native256"] for x in rows])) for k in METRICS},
                 "epoch_loss": [e["loss"] for e in meta["epoch_logs"]],
                 "epoch_lr": [e["lr"] for e in meta["epoch_logs"]],
                 "trainable_params": meta["n_trainable_params"], "sigma_table": meta["sigma_table_trainCC"],
                 "n_sigma_select": meta["n_sigma_select"], "train_s": meta["train_seconds"],
                 "eval_s": meta["eval_seconds"], "steps": meta["steps"], "steps_planned": meta["steps_planned"],
                 "batch": meta["batch"], "micro": meta["micro"], "orders_sha1": meta["orders_sha1"],
                 "orders_match_expected": meta["orders_match_expected"],
                 "max_mem_gb": max([e.get("max_mem_gb", 0) for e in meta["epoch_logs"]] or [0]) or None,
                 "rescore_max_abs_diff": worst, "source": os.path.relpath(pre, K) + "_*",
                 "device": meta["device"], "torch": meta["torch"], "numpy": meta["numpy"],
                 "max_steps": meta.get("max_steps"), "limit_eval": meta.get("limit_eval")})
    for x in rows:
        rows_all.append({"id": x["id"], "fold": 0, "n": n, "resample": r, "seed": r, "sigma": x["sigma"],
                         **{k: x[k] for k in METRICS},
                         **{f"{k}_nosigma": x[f"{k}_nosigma"] for k in METRICS},
                         **{f"{k}_native256": x[f"{k}_native256"] for k in METRICS},
                         "token": x["token"], "source": "curve run"})

# ------------------------------------------------------------------ n = 800 from sum_ft fold 0
N800_NOTE = ("n = 800 is the finished k-fold row sum_ft, fold 0 (not rerun): same 800 training ids (folds 1-4, "
             "= the n=800 draw as a set), same recipe (SUM loss, Adam 1e-4, StepLR(4, 0.1), batch 16, 12 epochs = "
             "600 steps; 800/16 has no partial batch). Differences from the curve runs: data order default_rng(0) "
             "shuffling the pool in fold order (not default_rng(r) over the draw order), torch seed 0 (not 800000), "
             "and sigma chosen on 200 of the 800 training maps (default_rng(0), k-fold rule) instead of all 800; the "
             "training-CC sigma table is flat (0.97387 at sigma 0, 0.97397 at sigma 2) and test CC is 0.9203 with "
             "sigma vs 0.9201 without. Sigma cannot be re-chosen on all 800 maps (no weights or training "
             "predictions were saved). One resample only.")
if not args.no_n800:
    pi = [x for x in json.load(open(os.path.join(args.n800_from, "sum_ft_per_image.json"))) if x["fold"] == 0]
    ss = json.load(open(os.path.join(args.n800_from, "sum_ft_summary.json")))
    fm0 = ss["fold_meta"]["0"]
    lab = "n=800 (sum_ft fold 0)"
    if [x["id"] for x in pi] != F0:
        problems.append(f"{lab}: per-image ids != fold 0 in order")
    if not (fm0["n_train_weights"] == 800 and fm0["n_trainable_params"] == 57501196):
        problems.append(f"{lab}: fold meta does not describe an 800-map full fine-tune")
    tr800 = sc.train_ids_for(folds, 0)
    if sorted(tr800) != sorted(sc.curve_subset(folds, 0, 800, 0)):
        problems.append(f"{lab}: training set != the n=800 draw as a set")
    worst = None
    if L is not None:
        mp = os.path.join(args.n800_maps, "sum_ft_fold0_test_maps256.npy")
        ip = os.path.join(args.n800_maps, "sum_ft_fold0_test_ids.json")
        if os.path.exists(mp) and json.load(open(ip)) == F0:
            worst = rescore(np.load(mp), F0, pi, fm0["sigma"], lab)
        else:
            problems.append(f"{lab}: saved fold-0 maps/ids not found or not fold 0 ({mp})")
    ep800 = [e for e in ss["train_log_per_epoch"] if e["fold"] == 0]
    runs.append({"n": 800, "resample": 0, "train_ids": tr800, "posthoc_sigma": fm0["sigma"],
                 **{k: float(np.mean([x[k] for x in pi])) for k in METRICS},
                 **{f"{k}_nosigma": float(np.mean([x[f"{k}_nosigma"] for x in pi])) for k in METRICS},
                 **{f"{k}_native256": float(np.mean([x[f"{k}_native256"] for x in pi])) for k in METRICS},
                 "epoch_loss": [e["loss"] for e in ep800], "epoch_lr": [e["lr"] for e in ep800],
                 "trainable_params": fm0["n_trainable_params"], "sigma_table": fm0["sigma_table_trainCC"],
                 "n_sigma_select": fm0["n_sigma_select"], "train_s": fm0["train_seconds"],
                 "steps": int(sum(e["steps"] for e in ep800)), "steps_planned": 600, "batch": 16, "micro": 8,
                 "orders_match_expected": None, "rescore_max_abs_diff": worst,
                 "train_ids_order": "pool (fold) order as used by the k-fold path; equals the n=800 draw as a set",
                 "source": "results/sum_ft_per_image.json + results/sum_ft_summary.json (fold 0)", "note": N800_NOTE})
    for x in pi:
        rows_all.append({"id": x["id"], "fold": 0, "n": 800, "resample": 0, "seed": x["seed"], "sigma": x["sigma"],
                         **{k: x[k] for k in METRICS},
                         **{f"{k}_nosigma": x[f"{k}_nosigma"] for k in METRICS},
                         **{f"{k}_native256": x[f"{k}_native256"] for k in METRICS},
                         "token": x["token"], "source": "sum_ft fold 0 (sigma on 200 of the 800 training maps)"})

# ------------------------------------------------------------------ guard, then write
if TO_RESULTS and (missing or args.no_n800):
    problems.append("results/ needs the complete plan plus the n = 800 point")
for w in warnings:
    print("WARNING:", w)
if problems:
    for p in problems:
        print("PROBLEM:", p)
    sys.exit(f"{len(problems)} problem(s); nothing written")

curve = []
for n in sorted({x["n"] for x in runs}):
    rr = sorted([x for x in runs if x["n"] == n], key=lambda x: x["resample"])
    c = {"n": n, "n_runs": len(rr), "resamples": [x["resample"] for x in rr]}
    for k in METRICS:
        v = [x[k] for x in rr]; vn = [x[f"{k}_nosigma"] for x in rr]
        c[f"mean_{k}"] = float(np.mean(v))
        c[f"sd_{k}"] = float(np.std(v))                                   # ddof 0, as msdb_adapt_curves_summary
        c[f"sd_{k}_ddof1"] = float(np.std(v, ddof=1)) if len(v) > 1 else None
        c[f"runs_{k}"] = v
        c[f"nosigma_mean_{k}"] = float(np.mean(vn)); c[f"nosigma_runs_{k}"] = vn
    c["sigma_per_run"] = [x["posthoc_sigma"] for x in rr]
    c["steps_per_run"] = [x["steps"] for x in rr]
    c["train_s_per_run"] = [x["train_s"] for x in rr]
    if n == 800:
        c["note"] = "single resample taken from sum_ft fold 0 (see n800_note)"
    curve.append(c)

os.makedirs(args.out_dir, exist_ok=True)
pf = f"{args.prefix}_per_image.json"; pfn = f"{args.prefix}_nosigma_per_image.json"
rows_all.sort(key=lambda x: (x["n"], x["resample"], F0.index(x["id"])))
summary = {
    "row": args.prefix, "mode": "full",
    "loss": "SUM's own objective -2*CC + 10*KL - SIM + 5*MSE (NSS dropped: ADD1000 has density maps only)",
    "split": "splits/kfold_seed0.json", "epochs": 12, "lr": 1e-4, "scheduler": "StepLR(step 4, gamma 0.1), per epoch",
    "batch": "min(16, n), last partial batch kept (steps = 12 x ceil(n/16)); micro-batches of 8 with accumulation "
             "weighted by micro size / batch size (exact batch mean)",
    "optimizer": "Adam", "device": "cuda (cloud NVIDIA A10G 24 GB GPU), chunked pure-PyTorch selective scan",
    "what_is_trained": "all 57,501,196 SUM parameters (released weights reloaded for every run), domain token 2, "
                       "stochastic depth as released",
    "init": "released SUM weights (hf safe-models/SUM); torch.manual_seed(1000*n + r)",
    "input": "256x256 stretch resize + ImageNet normalisation",
    "scoring": "384x384: prediction (256 sigmoid output) and GT PIL-bilinear to 384, clip at 0, divide by sum; "
               "KLD eps 1e-8; *_nosigma = no post-hoc blur; *_native256 = raw 256 output vs GT at 256, no sigma",
    "test_set": "fold 0 of the split (200 ids)",
    "subset_rule": "sample_efficiency.py: pool = folds 1-4 in order, numpy default_rng(1000*n + r).choice(pool, n, "
                   "replace=False); identical to msdb_adapt_curves / curve_ours_* train_ids for every (n, r)",
    "data_order": "default_rng(r), fresh copy of the training list shuffled every epoch (as ours / IIE curves)",
    "training_policy_small_n": "fixed epochs (SUM's own 12 at every n, so steps = 12 x ceil(n/16)); no early stopping",
    "calibration_maps": "per run: the same n training maps select the post-hoc sigma (grid 0,2,...,12, mean CC); "
                        "no centre prior",
    "training_maps": "per run: n",
    "resamples": "n = 25, 50, 100, 200: r = 0, 1, 2; n = 400: r = 0 only; n = 800: sum_ft fold 0 only (budget-driven)",
    "n800_note": N800_NOTE if not args.no_n800 else "omitted",
    "curve": curve, "runs": runs, "trainable_params": 57501196,
    "wall_clock_s_curve_runs": float(sum(x.get("train_s", 0) + x.get("eval_s", 0) for x in runs if x["n"] != 800)),
    "jobs": [j for j in args.jobs.split(",") if j],
    "command": "cloud GPU runs (the ft_sum_folds.py call of scripts/sum/run_sum_ft_gpu.sh) --tag sum_curve_a --curve 25:0,1,2/50:0,1,2/"
               "100:0,1,2/200:0 and --tag sum_curve_b --curve 200:1,2/400:0 (--loss sum --epochs 12 --micro 8 "
               ") -> ft_sum_folds.py --curve ... --curve_expect splits/curve_subsets_seed0.json; "
               "finalized by scripts/sum/finalize_curve.py",
    "validation": {"rescored_with": None if args.no_rescore else "scripts/ours/ours_lib.py cc/sim/kld (tolerance 1e-9)",
                   "warnings": warnings},
    "per_image_files": {"sigma": pf, "nosigma": pfn},
    "note": args.note}
json.dump(rows_all, open(os.path.join(args.out_dir, pf), "w"))
json.dump([{**x, "CC": x["CC_nosigma"], "SIM": x["SIM_nosigma"], "KLD": x["KLD_nosigma"], "sigma": 0}
           for x in rows_all], open(os.path.join(args.out_dir, pfn), "w"))
json.dump(summary, open(os.path.join(args.out_dir, f"{args.prefix}_summary.json"), "w"), indent=1)
print(f"wrote {args.out_dir}/{args.prefix}_{{summary,per_image,nosigma_per_image}}.json: "
      + "; ".join(f"n={c['n']} ({c['n_runs']} runs) CC {c['mean_CC']:.4f}" for c in curve))
