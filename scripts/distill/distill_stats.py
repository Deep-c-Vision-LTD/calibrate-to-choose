"""
Track B statistics: paired tests of every distillation row against ours_scratch_s0 and against
msdb_adapt (seed 0), with the statistics of scripts/compile/compile_results.py (copied unchanged:
Wilcoxon signed-rank on the 1000 out-of-fold per-image values, Nadeau-Bengio corrected resampled
t-test on the 5 fold differences, 5000-sample bootstrap CI of the mean per-image difference, dz,
rank-biserial r). Differences = distill row minus reference (for KLD negative = distill better).
Holm correction across the track-B family (all distill rows x both references), separately per
metric and test. Writes results/distill_track_overview.json and adds "paired_tests" to each
distill row summary (new rows only; no existing file is touched).
"""
import sys
sys.dont_write_bytecode = True
import os, json, math, glob
import numpy as np
from scipy import stats

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RES = os.path.join(KB, "results")
METRICS = ("CC", "SIM", "KLD")
HIGHER = {"CC": True, "SIM": True, "KLD": False}
FOLDS = json.load(open(os.path.join(KB, "splits", "kfold_seed0.json")))["folds"]
FOLD_OF = {str(i): f for f, ids in enumerate(FOLDS) for i in ids}
IDS = sorted(FOLD_OF, key=int)
B_BOOT = 5000
ROWS = ["distill_pitt", "distill_pitt_only", "distill_gt800_teacher800", "distill_pitt_L",
        "distill_pitt_only_L", "distill_gt800_teacher800_L"]
REFS = {"ours_scratch_s0": ("ours_scratch_s0_per_image.json", None), "msdb_adapt_s0": ("msdb_adapt_per_image.json", 0)}


def load_rows(fname, seed=None):
    out = {}
    for r in json.load(open(os.path.join(RES, fname))):
        if seed is not None and r.get("seed", 0) != seed:
            continue
        i = str(r["id"]); assert i not in out
        out[i] = {"fold": int(r["fold"]), **{m: float(r[m]) for m in METRICS}}
    assert len(out) == 1000 and all(FOLD_OF[i] == r["fold"] for i, r in out.items())
    return out


def vec(rows, m):
    return np.array([rows[i][m] for i in IDS])


def fold_means(rows, m, nf=5):
    acc = [[] for _ in range(nf)]
    for i, r in rows.items():
        acc[r["fold"]].append(r[m])
    return np.array([np.mean(a) for a in acc]), np.array([len(a) for a in acc])


def holm(pvals):
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); adj = np.empty(n); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, (n - rank) * p[idx])); adj[idx] = run
    return adj


def nb_test(dk, ratio):
    K = len(dk); m = dk.mean(); v = dk.var(ddof=1)
    if v == 0:
        return float(m), float("inf") if m != 0 else 0.0, 0.0 if m != 0 else 1.0
    t = m / math.sqrt((1.0 / K + ratio) * v)
    p = 2 * stats.t.sf(abs(t), K - 1)
    return float(m), float(t), float(p)


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def compare(ref, oth, seed=0):
    sizes = [len(f) for f in FOLDS]
    ratio = float(np.mean([s / (1000 - s) for s in sizes]))
    rng = np.random.default_rng(seed)
    o = {}
    for m in METRICS:
        a, b = vec(ref, m), vec(oth, m)
        d = a - b
        better = (d > 0) if HIGHER[m] else (d < 0)
        try:
            w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            wp, ws = float(w.pvalue), float(w.statistic)
        except ValueError:
            wp, ws = 1.0, float("nan")
        idx = rng.integers(0, len(d), size=(B_BOOT, len(d)))
        bm = d[idx].mean(1)
        fa, _ = fold_means(ref, m); fb, _ = fold_means(oth, m)
        dk = fa - fb
        nbm, nbt, nbp = nb_test(dk, ratio)
        won = int(((dk > 0) if HIGHER[m] else (dk < 0)).sum())
        o[m] = {"mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                "boot95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
                "rank_biserial": rank_biserial(d), "frac_images_ref_better": float(better.mean()),
                "wilcoxon_p": wp, "wilcoxon_W": ws,
                "fold_diffs": [float(x) for x in dk], "folds_ref_better": won, "n_folds": 5,
                "nb_t": nbt, "nb_p": nbp, "nb_ratio_ntest_ntrain": ratio}
    return o


def main():
    refs = {k: load_rows(f, s) for k, (f, s) in REFS.items()}
    rows = {r: load_rows(f"{r}_per_image.json") for r in ROWS if os.path.exists(os.path.join(RES, f"{r}_per_image.json"))}
    tests = {}
    keys = []
    for r in rows:
        for rf in refs:
            tests[(r, rf)] = compare(rows[r], refs[rf]); keys.append((r, rf))
    for m in METRICS:
        for key in ("wilcoxon_p", "nb_p"):
            adj = holm([tests[k][m][key] for k in keys])
            for k, pa in zip(keys, adj):
                tests[k][m][key + "_holm"] = float(pa)
    desc = {}
    for name, rr in list(rows.items()) + list(refs.items()):
        d = {}
        for m in METRICS:
            fm, _ = fold_means(rr, m)
            d[m] = {"mean": float(fm.mean()), "sd_folds_ddof1": float(fm.std(ddof=1)), "per_fold": [float(x) for x in fm]}
        desc[name] = d
    ov = {"description": "Track B: distillation of the adapted MSDB head into our fast readout through unlabeled Pitt ads. "
                         "Differences = distill row minus reference; Holm over the track-B family (rows x 2 references) per metric and test.",
          "rows": desc,
          "tests": {f"{r}__vs__{rf}": t for (r, rf), t in tests.items()},
          "family": [f"{r}__vs__{rf}" for (r, rf) in keys]}
    # secondary (attribution) comparisons, own Holm family; differences = first minus second
    SEC = [("distill_pitt", "distill_gt800_teacher800"), ("distill_pitt", "distill_pitt_only"),
           ("distill_pitt_only", "ours_scratch_s0"), ("distill_pitt_L", "ours_L448_s0"), ("distill_pitt_L", "distill_pitt"),
           ("distill_pitt_L", "msdb_adapt_s0")]
    extra = {"ours_L448_s0": "ours_L448_s0_per_image.json"}
    pool_rows = dict(rows); pool_rows.update(refs)
    for nm, fn in extra.items():
        if os.path.exists(os.path.join(RES, fn)):
            pool_rows[nm] = load_rows(fn)
    sec = {}
    for a_, b_ in SEC:
        if a_ in pool_rows and b_ in pool_rows and (a_, b_) not in tests:
            sec[(a_, b_)] = compare(pool_rows[a_], pool_rows[b_])
    sk = list(sec)
    for m in METRICS:
        for key in ("wilcoxon_p", "nb_p"):
            if sk:
                adj = holm([sec[k][m][key] for k in sk])
                for k, pa in zip(sk, adj):
                    sec[k][m][key + "_holm"] = float(pa)
    ov["secondary_tests"] = {f"{a_}__vs__{b_}": t for (a_, b_), t in sec.items()}
    ov["secondary_note"] = "attribution comparisons (first minus second); Holm within this secondary family only"
    for (a_, b_), t in sec.items():
        c = t["CC"]
        print(f"[sec] {a_:24s} vs {b_:24s} dCC {c['mean_diff']:+.4f} [{c['boot95'][0]:+.4f},{c['boot95'][1]:+.4f}] "
              f"folds {c['folds_ref_better']}/5 pW {c['wilcoxon_p_holm']:.2e} pNB {c['nb_p_holm']:.3g} | dSIM {t['SIM']['mean_diff']:+.4f} dKLD {t['KLD']['mean_diff']:+.4f}")
    for f in ("pitt_subset.json",):
        sub = json.load(open(os.path.join(KB, "cache", "distill", f)))
        ov["pitt_pool"] = {k: sub[k] for k in ("n_files", "subset_rule", "criteria_counts", "n_removed_union", "n_unreadable", "n_pool", "removed", "thresholds")}
    tchk = os.path.join(KB, "cache", "distill", "teachers", "ALL_DONE.json")
    ov["teacher_check"] = json.load(open(tchk)) if os.path.exists(tchk) else None
    json.dump(ov, open(os.path.join(RES, "distill_track_overview.json"), "w"), indent=1)
    for r in rows:
        p = os.path.join(RES, f"{r}_summary.json")
        s = json.load(open(p))
        s["paired_tests"] = {rf: tests[(r, rf)] for rf in refs}
        s["paired_tests_note"] = "differences = this row minus reference; *_holm = Holm over the track-B family (see distill_track_overview.json)"
        json.dump(s, open(p, "w"), indent=1)
    for (r, rf), t in tests.items():
        c = t["CC"]
        print(f"{r:28s} vs {rf:16s} dCC {c['mean_diff']:+.4f} [{c['boot95'][0]:+.4f},{c['boot95'][1]:+.4f}] "
              f"folds {c['folds_ref_better']}/5 imgs {c['frac_images_ref_better']:.3f} pW {c['wilcoxon_p_holm']:.2e} pNB {c['nb_p_holm']:.3g} | "
              f"dSIM {t['SIM']['mean_diff']:+.4f} dKLD {t['KLD']['mean_diff']:+.4f}")


if __name__ == "__main__":
    main()
