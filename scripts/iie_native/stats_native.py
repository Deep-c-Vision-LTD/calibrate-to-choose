"""
Paired statistics for the native-resolution adapted IIE row (iie_native_adapt) with the statistics of
scripts/compile/compile_results.py: its functions (load_rows, check_folds, vec, fold_means, describe, holm,
nb_test, rank_biserial, compare) and constants are loaded from that file's SOURCE with ast and executed in a
private namespace, so the module's top-level pipeline (which rewrites results/main_table.json) is NOT run.
  * per image: Wilcoxon signed-rank (two-sided, zero_method wilcox), 1000 out-of-fold values paired by id;
  * per fold: Nadeau-Bengio corrected resampled t-test on the 5 fold-mean differences;
  * 5000-sample paired bootstrap 95% CI of the mean difference (numpy default_rng(0) per comparison);
  * Holm per metric and per test within each named family below.
Differences are "first minus second"; for KLD negative = first better.
Cross-check: compare(ours_scratch_s0, iie_ft_klcc_e20) must equal results/main_table.json (raw p, mean, CI).

Families
  iie_native_vs_384                 : iie_native_adapt - iie_ft_klcc_e20 (same recipe, native vs 384 input)
  iie_native_vs_384_nosigma         : the same without the post-hoc sigma (information)
  iie_native_vs_adapted3            : ours_scratch_s0, msdb_adapt_s0, sum_ft each minus iie_native_adapt
  kbs_adapted_pairwise6_iienative   : the paper's kbs_adapted_pairwise6 with iie_native_adapt in place of iie_ft_klcc_e20

  python -I stats_native.py   -> results/iie_native_adapt_paired_tests.json
"""
import ast, os, sys, json, math, time
import numpy as np
from scipy import stats

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
SRC = os.path.join(KB, "scripts", "compile", "compile_results.py")
WANT_F = ["jload", "load_rows", "check_folds", "vec", "fold_means", "describe", "holm", "nb_test", "rank_biserial", "compare"]
WANT_A = ["KB", "RES", "SPL", "METRICS", "HIGHER", "SPLITS", "FOLD_OF", "IDS", "B_BOOT"]
tree = ast.parse(open(SRC).read())
body = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT_F:
        body.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in WANT_A for t in node.targets):
        body.append(node)
NS = {"os": os, "json": json, "math": math, "np": np, "stats": stats, "time": time, "__file__": SRC}
exec(compile(ast.Module(body=body, type_ignores=[]), SRC, "exec"), NS)
missing = [n for n in WANT_F + WANT_A if n not in NS]
assert not missing, missing
compare, holm, describe, load_rows, check_folds = NS["compare"], NS["holm"], NS["describe"], NS["load_rows"], NS["check_folds"]
METRICS, HIGHER = NS["METRICS"], NS["HIGHER"]

R = {}
def add(name, fname, seed=None):
    rows = load_rows(fname, seed=seed)
    check_folds(rows, "random", name)
    R[name] = rows

add("iie_native_adapt", "iie_native_adapt_per_image.json")
add("iie_native_adapt_nosigma", "iie_native_adapt_nosigma_per_image.json")
add("iie_ft_klcc_e20", "iie_ft_klcc_e20_per_image.json")
add("iie_ft_klcc_e20_nosigma", "iie_ft_klcc_e20_nosigma_per_image.json")
add("ours_scratch_s0", "ours_scratch_s0_per_image.json")
add("msdb_adapt_s0", "msdb_adapt_per_image.json", seed=0)
add("sum_ft", "sum_ft_per_image.json")

# cross-check against the compiled main table
mt = json.load(open(os.path.join(NS["RES"], "main_table.json")))["tests_vs_ours"]["ours_scratch_s0"]["iie_ft_klcc_e20"]
xc = compare(R["ours_scratch_s0"], R["iie_ft_klcc_e20"])
xcheck = {m: {"mean_diff": [xc[m]["mean_diff"], mt[m]["mean_diff"]], "wilcoxon_p": [xc[m]["wilcoxon_p"], mt[m]["wilcoxon_p"]],
              "nb_p": [xc[m]["nb_p"], mt[m]["nb_p"]], "boot95": [xc[m]["boot95"], mt[m]["boot95"]]} for m in METRICS}
xcheck_ok = all(abs(v[0] - v[1]) < 1e-12 if not isinstance(v[0], list) else max(abs(p - q) for p, q in zip(*v)) < 1e-12
                for m in METRICS for v in xcheck[m].values())
print("cross-check vs main_table.json:", "OK" if xcheck_ok else "MISMATCH", flush=True)

PAIR6 = [("msdb_adapt_s0", "sum_ft"), ("msdb_adapt_s0", "ours_scratch_s0"), ("msdb_adapt_s0", "iie_native_adapt"),
         ("sum_ft", "ours_scratch_s0"), ("sum_ft", "iie_native_adapt"), ("ours_scratch_s0", "iie_native_adapt")]
FAM = {
    "iie_native_vs_384": ([("iie_native_adapt", "iie_ft_klcc_e20")],
                          "IIE adapted, selected recipe: native-resolution input minus 384 input (same folds, recipe, scorer)"),
    "iie_native_vs_384_nosigma": ([("iie_native_adapt_nosigma", "iie_ft_klcc_e20_nosigma")],
                                  "as iie_native_vs_384 without the post-hoc sigma (information)"),
    "iie_native_vs_adapted3": ([("ours_scratch_s0", "iie_native_adapt"), ("msdb_adapt_s0", "iie_native_adapt"),
                                ("sum_ft", "iie_native_adapt")],
                               "each other adapted model minus native-resolution adapted IIE"),
    "kbs_adapted_pairwise6_iienative": (PAIR6, "the paper's kbs_adapted_pairwise6 family with iie_native_adapt in place "
                                               "of iie_ft_klcc_e20"),
}
out_f = {}
for fname, (pairs, dsc) in FAM.items():
    res = {f"{x}_minus_{y}": compare(R[x], R[y]) for x, y in pairs}
    keys = list(res)
    for m in METRICS:
        for key in ("wilcoxon_p", "nb_p"):
            adj = holm([res[c][m][key] for c in keys])
            for c, pa in zip(keys, adj):
                res[c][m][key + "_holm"] = float(pa)
    out_f[fname] = {"description": dsc, "comparisons": keys,
                    "holm": "per metric and per test (Wilcoxon, NB) over the comparisons of this family",
                    "results": res}

desc = {n: describe(r) for n, r in R.items()}
ADAPTED_NATIVE = ["ours_scratch_s0", "msdb_adapt_s0", "sum_ft", "iie_native_adapt"]
ADAPTED_384 = ["ours_scratch_s0", "msdb_adapt_s0", "sum_ft", "iie_ft_klcc_e20"]
def ranking(rows):
    o = {}
    for m in METRICS:
        o[m] = sorted(rows, key=lambda n: desc[n][m]["mean"], reverse=HIGHER[m])
    return o
rk_nat, rk_384 = ranking(ADAPTED_NATIVE), ranking(ADAPTED_384)
f3 = out_f["iie_native_vs_adapted3"]["results"]
sig_all = {m: all(f3[c][m]["wilcoxon_p_holm"] < 0.05 and f3[c][m]["nb_p_holm"] < 0.05 and
                  ((f3[c][m]["mean_diff"] > 0) if HIGHER[m] else (f3[c][m]["mean_diff"] < 0)) for c in f3) for m in METRICS}
last = {m: rk_nat[m][-1] == "iie_native_adapt" for m in METRICS}
verdict = {"iie_last_by_mean": last, "every_other_adapted_model_better_holm_both_tests": sig_all,
           "iie_last_among_adapted_holds": bool(all(last.values()) and all(sig_all.values()))}
out = {"description": __doc__, "stats_source": os.path.relpath(SRC, KB) + " (functions executed from source via ast; B_BOOT = %d)" % NS["B_BOOT"],
       "cross_check_main_table": {"ok": xcheck_ok, "values_ours_scratch_s0_vs_iie_ft_klcc_e20": xcheck},
       "rows": desc, "ranking_adapted_native": rk_nat, "ranking_adapted_384": rk_384, "families": out_f,
       "verdict": verdict}
json.dump(out, open(os.path.join(NS["RES"], "iie_native_adapt_paired_tests.json"), "w"), indent=1)

# ---- text report
print("\nrow means (mean of 5 fold means; SD over folds, ddof 1)")
for n in ["iie_native_adapt", "iie_ft_klcc_e20", "iie_native_adapt_nosigma", "iie_ft_klcc_e20_nosigma",
          "ours_scratch_s0", "msdb_adapt_s0", "sum_ft"]:
    print(f"  {n:28s} " + "  ".join(f"{m} {desc[n][m]['mean']:.4f}+/-{desc[n][m]['sd_folds_ddof1']:.4f}" for m in METRICS))
for fname, f in out_f.items():
    print(f"\nfamily {fname}")
    for c in f["comparisons"]:
        for m in METRICS:
            r = f["results"][c][m]
            print(f"  {c:40s} {m:3s} diff {r['mean_diff']:+.4f} CI [{r['boot95'][0]:+.4f},{r['boot95'][1]:+.4f}] "
                  f"folds {r['folds_ref_better']}/5 img {r['frac_images_ref_better']:.3f} "
                  f"W p_holm {r['wilcoxon_p_holm']:.2e} NB p_holm {r['nb_p_holm']:.2e}")
print("\nranking (native IIE):", json.dumps(rk_nat))
print("verdict:", json.dumps(verdict))
