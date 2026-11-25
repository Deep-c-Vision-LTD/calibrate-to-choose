"""
Paired per-image comparison of the DeepGaze MSDB rows with the readout rows of the 'ours'
track (read-only use of their per-image files). Same images, same 384 scoring grid.
Writes results/msdb_vs_ours_paired.json.

  python -I compare_vs_ours.py
"""
import json, os, glob
import numpy as np
from scipy import stats

R = os.path.join(os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))), "results")


def per(name, seed=None):
    p = os.path.join(R, f"{name}_per_image.json")
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))
    if seed is not None and rows and "seed" in rows[0]:
        rows = [r for r in rows if r["seed"] == seed]
    return {str(r["id"]): r for r in rows}


def fold_means(d, m):
    f = {}
    for r in d.values():
        f.setdefault(r["fold"], []).append(r[m])
    v = [np.mean(f[k]) for k in sorted(f)]
    return float(np.mean(v)), float(np.std(v, ddof=1))


def paired(a, b, na, nb, m):
    ids = sorted(set(a) & set(b), key=int)
    x = np.array([a[i][m] for i in ids]); y = np.array([b[i][m] for i in ids]); d = x - y
    better = (d > 0) if m != "KLD" else (d < 0)
    return {"a": na, "b": nb, "metric": m, "n": len(ids), "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "mean_diff_a_minus_b": float(d.mean()), "frac_images_a_better": float(better.mean()),
            "wilcoxon_p": float(stats.wilcoxon(x, y).pvalue), "paired_t_p": float(stats.ttest_rel(x, y).pvalue)}


out = {"note": "per-image paired tests on the same ADD1000 images, CC/SIM/KLD at 384; MSDB rows seed 0 unless stated",
       "row_means": {}, "paired": []}
rows = {"msdb_adapt": per("msdb_adapt", 0), "msdb_adapt_kl": per("msdb_adapt_kl"),
        "msdb_adapt_grouped": per("msdb_adapt_grouped"), "msdb_adapt_pairsafe": per("msdb_adapt_pairsafe"),
        "msdb_zeroshot_adprior": per("msdb_zeroshot_adprior"), "msdb_zeroshot_foldprior": per("msdb_zeroshot_foldprior")}
for s in (0, 1, 2):
    for kind in ("warm", "scratch"):
        rows[f"ours_{kind}_s{s}"] = per(f"ours_{kind}_s{s}")
for extra in ("ours_pairsafe", "ours_grouped", "ours_grouped_s0", "ours_warm_grouped", "ours_scratch_grouped"):
    rows[extra] = per(extra)
rows = {k: v for k, v in rows.items() if v}
for k, v in rows.items():
    out["row_means"][k] = {m: fold_means(v, m) for m in ("CC", "SIM", "KLD")}
    out["row_means"][k]["n_images"] = len(v)
pairs = [("msdb_adapt", "ours_warm_s0"), ("msdb_adapt", "ours_scratch_s0"), ("msdb_adapt_kl", "ours_warm_s0"),
         ("msdb_zeroshot_foldprior", "ours_warm_s0"), ("msdb_zeroshot_adprior", "ours_warm_s0"),
         ("msdb_adapt_pairsafe", "ours_pairsafe"), ("msdb_adapt_grouped", "ours_grouped")]
for a, b in pairs:
    if a in rows and b in rows:
        for m in ("CC", "SIM", "KLD"):
            out["paired"].append(paired(rows[a], rows[b], a, b, m))
# matched-budget curves on the fold-0 test set
cur = {}
ms = json.load(open(os.path.join(R, "msdb_adapt_curves_summary.json")))
cur["msdb_adapt_curves"] = {c["n"]: c["mean_CC"] for c in ms["curve"]}
for kind in ("warm", "scratch"):
    p = os.path.join(R, f"curve_ours_{kind}_aggregate.json")
    if os.path.exists(p):
        cur[f"curve_ours_{kind}"] = json.load(open(p))
out["curves_raw"] = cur
json.dump(out, open(os.path.join(R, "msdb_vs_ours_paired.json"), "w"), indent=1)
for k, v in out["row_means"].items():
    print(f"{k:26s} CC {v['CC'][0]:.4f} (sd {v['CC'][1]:.4f}) SIM {v['SIM'][0]:.4f} KLD {v['KLD'][0]:.4f} n {v['n_images']}")
for t in out["paired"]:
    print(f"{t['a']} vs {t['b']} {t['metric']}: diff {t['mean_diff_a_minus_b']:+.4f} a-better {t['frac_images_a_better']:.3f} "
          f"wilcoxon p {t['wilcoxon_p']:.2e}")
