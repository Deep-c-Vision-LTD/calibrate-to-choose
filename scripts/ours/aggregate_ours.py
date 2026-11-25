"""
Collect every "ours" row into results/ours_track_overview.json:
 - per-row 5-fold mean CC/SIM/KLD (+ SD over folds) and file paths
 - per-init seed summaries (mean and SD over the 3 seed-level 5-fold means)
 - paired per-image comparisons on the same 1000 test images (seed-averaged per image):
   ablation rows vs the warm-start reference (the ablation protocol uses warm start),
   scratch vs warm, pair-safe / grouped vs scratch seed 0 (different test-fold membership,
   so only image-level paired deltas are reported); Wilcoxon signed-rank + paired bootstrap CI
 - validation against the published 0.898 (scratch) / 0.895 (warm) and Table II
 - learning-curve aggregates

  python -I aggregate_ours.py
"""
import sys
sys.dont_write_bytecode = True
import os, json, glob
import numpy as np
from scipy.stats import wilcoxon

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
R = os.path.join(KB, "results")
PUB = {"ours_scratch": {"CC": 0.8978, "SIM": 0.8054, "KLD": 0.1483},
       "ours_warm": {"CC": 0.8950, "SIM": 0.8035, "KLD": 0.1533},
       "abl_no_aspp": {"CC": 0.8924, "SIM": 0.7991, "KLD": 0.1599},
       "abl_no_centerbias": {"CC": 0.8948, "SIM": 0.8037, "KLD": 0.1532},
       "abl_no_finalizer": {"CC": 0.8916, "SIM": 0.8005, "KLD": 0.1593},
       "abl_kl_only": {"CC": 0.8900, "SIM": 0.7988, "KLD": 0.1567},
       "abl_full_tableII": {"CC": 0.8950, "SIM": 0.8038, "KLD": 0.1530}}
PUB_SD_CC = {"ours_scratch": 0.0032, "ours_warm": 0.0032}


def load(row):
    s = os.path.join(R, f"{row}_summary.json"); p = os.path.join(R, f"{row}_per_image.json")
    if not os.path.exists(s):
        return None, None
    return json.load(open(s)), json.load(open(p))


def per_image_vec(rows, metric):
    """Per-image metric averaged over the rows (seeds) that exist; None if none exists."""
    pi = []
    for r in rows:
        s, p = load(r)
        if s is None:
            continue
        pi.append({x["id"]: x[metric] for x in p})
    if not pi:
        return None
    ids = sorted(pi[0], key=int)
    return ids, np.array([[d[i] for i in ids] for d in pi]).mean(0)


def paired(a_rows, b_rows, metric):
    A = per_image_vec(a_rows, metric); B = per_image_vec(b_rows, metric)
    if A is None or B is None:
        return None
    assert A[0] == B[0]
    d = A[1] - B[1]
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = None
    return {"rows_A": [r for r in a_rows if load(r)[0] is not None], "rows_B": [r for r in b_rows if load(r)[0] is not None],
            "mean_delta_A_minus_B": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_images": int(len(d)), "frac_images_A_better": float(np.mean(d > 0) if metric != "KLD" else np.mean(d < 0)),
            "wilcoxon_p": p}


def main():
    out = {"rows": {}, "seed_summaries": {}, "paired": {}, "validation": {}, "curves": {}}
    for f in sorted(glob.glob(os.path.join(R, "*_summary.json"))):
        row = os.path.basename(f)[:-len("_summary.json")]
        if not (row.startswith("ours_") or row.startswith("abl_") or row.startswith("curve_ours_")):
            continue
        s = json.load(open(f))
        out["rows"][row] = {"mean_CC": s["mean_CC"], "sd_CC": s.get("sd_CC_ddof0"), "mean_SIM": s["mean_SIM"], "mean_KLD": s["mean_KLD"],
                            "per_fold_CC": [x["CC"] for x in s["per_fold"]],
                            "sigmas": [x.get("sigma") for x in s.get("folds_detail", [])] or s.get("sigma"),
                            "summary_file": f, "per_image_file": os.path.join(R, f"{row}_per_image.json"),
                            "calibration_and_training_data": s.get("calibration_and_training_data"),
                            "wall_clock_s": s.get("wall_clock_s")}
    fam = {"ours_scratch": [f"ours_scratch_s{s}" for s in range(3)], "ours_warm": [f"ours_warm_s{s}" for s in range(3)]}
    for m in ("no_aspp", "no_centerbias", "no_finalizer", "kl_only", "full"):
        fam[f"abl_{m}"] = [f"abl_{m}_s{s}" for s in range(3)]
    for k, rows in fam.items():
        have = [r for r in rows if r in out["rows"]]
        if not have:
            continue
        v = {mm: [out["rows"][r][f"mean_{mm}"] for r in have] for mm in ("CC", "SIM", "KLD")}
        out["seed_summaries"][k] = {"rows": have, **{f"mean_{mm}": float(np.mean(x)) for mm, x in v.items()},
                                   **{f"sd_over_seeds_{mm}": float(np.std(x, ddof=1)) if len(x) > 1 else None for mm, x in v.items()},
                                   "seed_means_CC": v["CC"], "published": PUB.get(k)}
    # ablation reference: the full recipe under the ablation rng stream (abl_full_s*) when all three seeds ran,
    # otherwise ours_warm_s* (same warm-start recipe; differs only in the data-order draw)
    ref = "abl_full" if all(r in out["rows"] for r in fam["abl_full"]) else "ours_warm"
    out["paired"]["ablation_reference"] = ref
    for m in ("no_aspp", "no_centerbias", "no_finalizer", "kl_only"):
        out["paired"][f"{ref}_vs_abl_{m}"] = {mm: paired(fam[ref], fam[f"abl_{m}"], mm) for mm in ("CC", "SIM", "KLD")}
    out["paired"]["ours_scratch_vs_ours_warm"] = {mm: paired(fam["ours_scratch"], fam["ours_warm"], mm) for mm in ("CC", "SIM", "KLD")}
    out["paired"]["ours_scratch_s0_vs_ours_pairsafe"] = {mm: paired(["ours_scratch_s0"], ["ours_pairsafe"], mm) for mm in ("CC", "SIM", "KLD")}
    out["paired"]["ours_scratch_s0_vs_ours_grouped"] = {mm: paired(["ours_scratch_s0"], ["ours_grouped"], mm) for mm in ("CC", "SIM", "KLD")}
    for k in ("ours_scratch", "ours_warm"):
        s0 = out["rows"].get(f"{k}_s0")
        if s0:
            out["validation"][f"{k}_s0"] = {"reproduced_CC": s0["mean_CC"], "published_CC": PUB[k]["CC"], "delta": s0["mean_CC"] - PUB[k]["CC"],
                                            "published_sd_over_folds": PUB_SD_CC[k], "reproduced_SIM": s0["mean_SIM"], "published_SIM": PUB[k]["SIM"],
                                            "reproduced_KLD": s0["mean_KLD"], "published_KLD": PUB[k]["KLD"]}
    for k in ("no_aspp", "no_centerbias", "no_finalizer", "kl_only", "full"):
        ss = out["seed_summaries"].get(f"abl_{k}")
        s0 = out["rows"].get(f"abl_{k}_s0")
        pk = "abl_full_tableII" if k == "full" else f"abl_{k}"
        if ss:
            out["validation"][f"abl_{k}"] = {"seed0_CC": s0["mean_CC"] if s0 else None, "seed_mean_CC": ss["mean_CC"],
                                             "n_seeds": len(ss["rows"]), "published_CC": PUB[pk]["CC"]}
    for init in ("scratch", "warm"):
        f = os.path.join(R, f"curve_ours_{init}_aggregate.json")
        if os.path.exists(f):
            out["curves"][init] = {str(b["n"]): {"mean_CC": b["mean_CC"], "sd_CC": b["sd_CC_ddof0"], "mean_SIM": b["mean_SIM"], "mean_KLD": b["mean_KLD"]}
                                   for b in json.load(open(f))["by_n"]}
            out["curves"][f"{init}_file"] = f
    json.dump(out, open(os.path.join(R, "ours_track_overview.json"), "w"), indent=1)
    for k, v in out["rows"].items():
        print(f"{k:32s} CC={v['mean_CC']:.4f} sd={v['sd_CC'] if v['sd_CC'] is None else round(v['sd_CC'], 4)} SIM={v['mean_SIM']:.4f} KLD={v['mean_KLD']:.4f}")
    print(json.dumps(out["seed_summaries"], indent=0)[:3000])
    print(json.dumps(out["validation"], indent=0))


if __name__ == "__main__":
    main()
