"""
Collect every DeepGaze MSDB row into results/msdb_track_overview.json, add the n = 0 point of
the curves (zero-shot, uniform prior, fold-0 test set) and paired per-image tests between rows
scored on the same images (Wilcoxon signed-rank and paired t on per-image CC at 384).

  python -I aggregate_msdb.py
"""
import json, os, sys, glob
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msdb_common import RESULTS, write_json, load_split

ROWS = ["msdb_zeroshot_adprior", "msdb_zeroshot_uniform", "msdb_zeroshot_foldprior", "msdb_adapt", "msdb_adapt_kl",
        "msdb_adapt_dsp", "msdb_adapt_grouped", "msdb_adapt_pairsafe"]
CURVES = ["msdb_adapt_curves", "msdb_adapt_dsp_curves"]
ids, folds, fold_of = load_split()


def load(row):
    s, p = os.path.join(RESULTS, f"{row}_summary.json"), os.path.join(RESULTS, f"{row}_per_image.json")
    if not (os.path.exists(s) and os.path.exists(p)):
        return None, None
    return json.load(open(s)), json.load(open(p))


out = {"rows": [], "curves": {}, "paired_tests": []}
per_all = {}
for r in ROWS:
    s, p = load(r)
    if s is None:
        out["rows"].append({"row": r, "status": "missing"}); continue
    if p and "seed" in p[0]:
        p0 = [x for x in p if x["seed"] == 0]
    else:
        p0 = p
    per_all[r] = {x["id"]: x for x in p0}
    out["rows"].append({"row": r, "CC_mean": s["CC_mean"], "CC_sd_folds_ddof1": s["CC_sd_folds_ddof1"],
                        "SIM_mean": s["SIM_mean"], "KLD_mean": s["KLD_mean"],
                        "CC_512_mean": s.get("CC_512_mean"), "SIM_512_mean": s.get("SIM_512_mean"),
                        "KLD_512_mean": s.get("KLD_512_mean"),
                        "per_fold_CC": [f["CC"] for f in s["per_fold"]],
                        "trainable_params": s.get("trainable_params", 0),
                        "calibration_maps": s.get("calibration_maps"), "training_maps": s.get("training_maps"),
                        "split": s.get("split", "splits/kfold_seed0.json"),
                        "seeds_CC": {k: v["CC_mean"] for k, v in s.items() if k.startswith("seed") and isinstance(v, dict)},
                        "summary_file": os.path.join(RESULTS, f"{r}_summary.json"),
                        "per_image_file": os.path.join(RESULTS, f"{r}_per_image.json")})

uni = per_all.get("msdb_zeroshot_uniform")
for c in CURVES:
    s, p = load(c)
    if s is None:
        out["curves"][c] = "missing"; continue
    pts = [{"n": 0, "mean_CC": float(np.mean([uni[i]["CC"] for i in folds[0]])) if uni else None,
            "note": "zero-shot released MSDB, uniform prior, no post-hoc sigma, fold-0 test set (no ADD1000 maps used)"}]
    pts += [{"n": x["n"], "mean_CC": x["mean_CC"], "sd_CC": x["sd_CC"], "runs_CC": x["runs_CC"],
             "mean_SIM": x["mean_SIM"], "mean_KLD": x["mean_KLD"]} for x in s["curve"]]
    out["curves"][c] = {"points": pts, "summary_file": os.path.join(RESULTS, f"{c}_summary.json"),
                        "per_image_file": os.path.join(RESULTS, f"{c}_per_image.json")}


def paired(a, b, metric="CC"):
    if a not in per_all or b not in per_all:
        return None
    common = sorted(set(per_all[a]) & set(per_all[b]), key=int)
    x = np.array([per_all[a][i][metric] for i in common]); y = np.array([per_all[b][i][metric] for i in common])
    d = x - y
    w = stats.wilcoxon(x, y)
    t = stats.ttest_rel(x, y)
    return {"a": a, "b": b, "metric": metric, "n": len(common), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "mean_diff": float(d.mean()), "frac_images_a_better": float((d > 0).mean()),
            "wilcoxon_p": float(w.pvalue), "paired_t_p": float(t.pvalue)}


for a, b in [("msdb_adapt", "msdb_zeroshot_foldprior"), ("msdb_adapt", "msdb_zeroshot_adprior"),
             ("msdb_adapt", "msdb_adapt_kl"), ("msdb_adapt", "msdb_adapt_dsp"),
             ("msdb_zeroshot_foldprior", "msdb_zeroshot_adprior"), ("msdb_zeroshot_adprior", "msdb_zeroshot_uniform"),
             ("msdb_adapt_dsp", "msdb_zeroshot_foldprior"), ("msdb_adapt", "msdb_adapt_grouped"),
             ("msdb_adapt", "msdb_adapt_pairsafe"), ("msdb_adapt_grouped", "msdb_zeroshot_adprior")]:
    for m in ["CC", "SIM", "KLD"]:
        r = paired(a, b, m)
        if r:
            out["paired_tests"].append(r)
cost_p = os.path.join(RESULTS, "msdb_costs.json")
if os.path.exists(cost_p):
    out["costs_file"] = cost_p
write_json(out, os.path.join(RESULTS, "msdb_track_overview.json"))
for r in out["rows"]:
    if "CC_mean" in r:
        print(f"{r['row']:28s} CC {r['CC_mean']:.4f} (sd {r['CC_sd_folds_ddof1']:.4f}) SIM {r['SIM_mean']:.4f} KLD {r['KLD_mean']:.4f}")
    else:
        print(f"{r['row']:28s} {r['status']}")
for c, v in out["curves"].items():
    if isinstance(v, dict):
        print(c, [(p["n"], round(p["mean_CC"], 4) if p["mean_CC"] is not None else None) for p in v["points"]])
for t in out["paired_tests"]:
    if t["metric"] == "CC":
        print(f"{t['a']} vs {t['b']}: dCC {t['mean_diff']:+.4f} wilcoxon p {t['wilcoxon_p']:.2e} (n {t['n']})")
