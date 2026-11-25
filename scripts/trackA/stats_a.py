"""
Track A paired tests: each new row vs ours_scratch_s0 and vs msdb_adapt (seed 0), plus
seed-averaged comparisons when a row has several seeds. Statistics copied from
scripts/compile/compile_results.py (Wilcoxon signed-rank two-sided zero_method='wilcox' on the
1000 out-of-fold images; Nadeau-Bengio corrected t on the 5 fold differences, df 4,
n_test/n_train = 0.25; 5000-sample bootstrap CI of the mean per-image difference; Holm across
the Track A family, per reference, metric and test). Differences = new row minus reference
(for KLD negative = new row better).

  python -I stats_a.py --rows ours_L448_s0 ours_ms_B_s0 ... [--groups ours_L448=ours_L448_s0,ours_L448_s1,...]
"""
import sys
sys.dont_write_bytecode = True
import os, json, math, argparse, time
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


def load(fname, seed=None):
    out = {}
    for r in json.load(open(os.path.join(RES, fname))):
        if seed is not None and r.get("seed", 0) != seed:
            continue
        i = str(r["id"]); assert i not in out
        out[i] = {"fold": int(r["fold"]), **{m: float(r[m]) for m in METRICS}}
    assert len(out) == 1000 and all(FOLD_OF[i] == r["fold"] for i, r in out.items()), fname
    return out


def seed_avg(lst):
    return {i: {"fold": lst[0][i]["fold"], **{m: float(np.mean([r[i][m] for r in lst])) for m in METRICS}} for i in lst[0]}


def vec(rows, m):
    return np.array([rows[i][m] for i in IDS])


def fold_means(rows, m):
    acc = [[] for _ in range(5)]
    for r in rows.values():
        acc[r["fold"]].append(r[m])
    return np.array([np.mean(a) for a in acc])


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, (n - rank) * p[idx])); adj[idx] = run
    return adj


def nb_test(dk, ratio=0.25):
    K = len(dk); m = dk.mean(); v = dk.var(ddof=1)
    if v == 0:
        return float(m), (float("inf") if m != 0 else 0.0), (0.0 if m != 0 else 1.0)
    t = m / math.sqrt((1.0 / K + ratio) * v)
    return float(m), float(t), float(2 * stats.t.sf(abs(t), K - 1))


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d)); wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def compare(a_rows, b_rows, seed=0):
    rng = np.random.default_rng(seed); o = {}
    for m in METRICS:
        a, b = vec(a_rows, m), vec(b_rows, m); d = a - b
        better = (d > 0) if HIGHER[m] else (d < 0)
        try:
            w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided"); wp = float(w.pvalue)
        except ValueError:
            wp = 1.0
        bm = d[rng.integers(0, len(d), size=(B_BOOT, len(d)))].mean(1)
        dk = fold_means(a_rows, m) - fold_means(b_rows, m)
        _, nbt, nbp = nb_test(dk)
        o[m] = {"mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                "boot95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
                "rank_biserial": rank_biserial(d), "frac_images_row_better": float(better.mean()),
                "wilcoxon_p": wp, "fold_diffs": [float(x) for x in dk],
                "folds_row_better": int(((dk > 0) if HIGHER[m] else (dk < 0)).sum()), "nb_t": nbt, "nb_p": nbp}
    return o


def describe(rows):
    o = {}
    for m in METRICS:
        fm = fold_means(rows, m)
        o[m] = {"mean": float(fm.mean()), "sd_folds_ddof1": float(fm.std(ddof=1)), "per_fold": [float(x) for x in fm]}
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--groups", nargs="*", default=[], help="name=row1,row2,... seed-averaged groups")
    ap.add_argument("--pairs", nargs="*", default=[], help="extra diagnostic pairs a:b among the new rows (no Holm)")
    ap.add_argument("--out", default=os.path.join(RES, "trackA_paired_tests.json"))
    a = ap.parse_args()
    refs = {"ours_scratch_s0": load("ours_scratch_s0_per_image.json"), "msdb_adapt_s0": load("msdb_adapt_per_image.json", seed=0)}
    refs_avg = {"ours_scratch_seedavg": seed_avg([load(f"ours_scratch_s{s}_per_image.json") for s in range(3)]),
                "msdb_adapt_seedavg": seed_avg([load("msdb_adapt_per_image.json", seed=s) for s in range(3)])}
    rows = {r: load(f"{r}_per_image.json") for r in a.rows}
    groups = {}
    for g in a.groups:
        name, members = g.split("=")
        groups[name] = seed_avg([rows[m] if m in rows else load(f"{m}_per_image.json") for m in members.split(",")])
    out = {"description": __doc__.strip(), "rows": {}, "groups": {}, "refs": {k: describe(v) for k, v in {**refs, **refs_avg}.items()}}
    for r, rr in rows.items():
        out["rows"][r] = {"describe": describe(rr), "vs": {k: compare(rr, v) for k, v in refs.items()}}
    for g, rr in groups.items():
        out["groups"][g] = {"describe": describe(rr), "members": [x for x in a.groups if x.startswith(g + "=")][0].split("=")[1].split(","),
                            "vs": {k: compare(rr, v) for k, v in refs_avg.items()}}
    # Holm within Track A family, per reference / metric / test
    for sect, refnames in (("rows", list(refs)), ("groups", list(refs_avg))):
        names = list(out[sect])
        for ref in refnames:
            for m in METRICS:
                for key in ("wilcoxon_p", "nb_p"):
                    if not names:
                        continue
                    adj = holm([out[sect][n]["vs"][ref][m][key] for n in names])
                    for n, pa in zip(names, adj):
                        out[sect][n]["vs"][ref][m][key + "_holm_trackA"] = float(pa)
    out["pairs"] = {}
    for pr in a.pairs:
        x, y = pr.split(":")
        out["pairs"][pr] = compare(rows[x] if x in rows else load(f"{x}_per_image.json"), rows[y] if y in rows else load(f"{y}_per_image.json"))
        c = out["pairs"][pr]["CC"]
        print(f"pair {pr}: dCC {c['mean_diff']:+.4f} [{c['boot95'][0]:+.4f},{c['boot95'][1]:+.4f}] pW {c['wilcoxon_p']:.1e} pNB {c['nb_p']:.3f} folds {c['folds_row_better']}/5")
    out["holm_family"] = {"rows": list(out["rows"]), "groups": list(out["groups"])}
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S"); out["command"] = "python -I " + " ".join(sys.argv)
    json.dump(out, open(a.out, "w"), indent=1)
    for sect in ("rows", "groups"):
        for n, o in out[sect].items():
            d = o["describe"]
            line = f"{n:28s} CC {d['CC']['mean']:.4f}±{d['CC']['sd_folds_ddof1']:.4f} SIM {d['SIM']['mean']:.4f} KLD {d['KLD']['mean']:.4f}"
            for ref, c in o["vs"].items():
                line += (f" | vs {ref}: dCC {c['CC']['mean_diff']:+.4f} [{c['CC']['boot95'][0]:+.4f},{c['CC']['boot95'][1]:+.4f}]"
                         f" pW {c['CC']['wilcoxon_p']:.1e} pNB {c['CC']['nb_p']:.3f} folds {c['CC']['folds_row_better']}/5 img {c['CC']['frac_images_row_better']:.1%}")
            print(line)


if __name__ == "__main__":
    main()
