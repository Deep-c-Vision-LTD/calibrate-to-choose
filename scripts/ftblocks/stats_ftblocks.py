"""
ftblocks statistics, with the statistics code of scripts/compile/compile_results.py reused VERBATIM:
the functions holm, nb_test, rank_biserial, compare, vec and fold_means are read from that file's source
with the ast module and executed here (the module itself is never imported, because importing it would
run the whole compilation). They give: per-image Wilcoxon signed-rank (two-sided, zero_method 'wilcox')
on the 1000 out-of-fold values paired by id; fold-level Nadeau-Bengio corrected resampled t
(t = mean(d) / sqrt((1/K + n_test/n_train) var(d, ddof 1)), df = K - 1); 5000-sample paired bootstrap
95% CI of the mean per-image difference (rng default_rng(0)); dz; rank-biserial r; folds won.
Fold SD with ddof 1 (describe()). Differences = ftblocks row MINUS reference (KLD: negative = ftblocks better).

Families (pre-registered in PLAN.json "statistics"), Holm within each family separately per metric and test:
  primary   : ours_ftblocks_nested vs {ours_scratch_s0, ours_scratch_seedavg, distill_pitt, ours_L448_s0,
              ours_ms_B_s0, msdb_adapt_s0, sum_ft}
  secondary : each secondary row that exists (ours_ftblocks_global, ours_ftblocks_distill) vs the same references
Writes results/ftblocks_stats.json (new file; no existing file is touched).

  python -I stats_ftblocks.py
"""
import sys
sys.dont_write_bytecode = True
import os, json, ast, math, time
import numpy as np
from scipy import stats

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RES = os.path.join(KB, "results")
COMPILE = os.path.join(KB, "scripts", "compile", "compile_results.py")
METRICS = ("CC", "SIM", "KLD")
HIGHER = {"CC": True, "SIM": True, "KLD": False}
B_BOOT = 5000
SPLITS = {"random": json.load(open(os.path.join(KB, "splits", "kfold_seed0.json")))["folds"]}
FOLD_OF = {"random": {str(i): f for f, ids in enumerate(SPLITS["random"]) for i in ids}}
IDS = sorted(FOLD_OF["random"], key=int)
assert len(IDS) == 1000

REUSED = ("holm", "nb_test", "rank_biserial", "compare", "vec", "fold_means", "describe", "seed_avg")


def load_compile_functions():
    src = open(COMPILE).read()
    tree = ast.parse(src)
    ns = {"np": np, "stats": stats, "math": math, "METRICS": METRICS, "HIGHER": HIGHER, "B_BOOT": B_BOOT,
          "SPLITS": SPLITS, "IDS": IDS}
    got = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in REUSED:
            exec(compile(ast.Module(body=[node], type_ignores=[]), COMPILE, "exec"), ns)
            got.append(node.name)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "B_BOOT" for t in node.targets):
            assert ast.literal_eval(node.value) == B_BOOT
    assert sorted(got) == sorted(REUSED), got
    return ns


NS = load_compile_functions()


def load_rows(fname, seed=None):
    out = {}
    for r in json.load(open(os.path.join(RES, fname))):
        if seed is not None and r.get("seed", 0) != seed:
            continue
        i = str(r["id"]); assert i not in out, (fname, i)
        out[i] = {"fold": int(r["fold"]), **{m: float(r[m]) for m in METRICS}}
    assert len(out) == 1000 and all(FOLD_OF["random"][i] == r["fold"] for i, r in out.items()), fname
    return out


REFS = {
    "ours_scratch_s0": lambda: load_rows("ours_scratch_s0_per_image.json"),
    "ours_scratch_seedavg": lambda: NS["seed_avg"]([load_rows(f"ours_scratch_s{s}_per_image.json") for s in range(3)]),
    "distill_pitt": lambda: load_rows("distill_pitt_per_image.json"),
    "ours_L448_s0": lambda: load_rows("ours_L448_s0_per_image.json"),
    "ours_ms_B_s0": lambda: load_rows("ours_ms_B_s0_per_image.json"),
    "msdb_adapt_s0": lambda: load_rows("msdb_adapt_per_image.json", seed=0),
    "sum_ft": lambda: load_rows("sum_ft_per_image.json"),
}
PRIMARY = "ours_ftblocks_nested"
SECONDARY = ["ours_ftblocks_global", "ours_ftblocks_distill"]


def family(row_names, refs):
    """all (row, ref) pairs of the family; Holm over the pairs, separately per metric and test."""
    rows = {n: load_rows(f"{n}_per_image.json") for n in row_names}
    pairs = [(n, r) for n in row_names for r in refs]
    res = {f"{n} - {r}": NS["compare"](rows[n], refs[r], "random") for n, r in pairs}
    keys = list(res)
    for m in METRICS:
        for t in ("wilcoxon_p", "nb_p"):
            adj = NS["holm"]([res[k][m][t] for k in keys])
            for k, pa in zip(keys, adj):
                res[k][m][t + "_holm"] = float(pa)
    return {"pairs": keys, "family_size": len(keys), "tests": res,
            "describe": {n: NS["describe"](rows[n]) for n in row_names}}


def main():
    if not os.path.exists(os.path.join(RES, f"{PRIMARY}_per_image.json")):
        raise SystemExit(f"{PRIMARY} not assembled yet")
    refs = {n: f() for n, f in REFS.items()}
    out = {"note": "differences = ftblocks row minus reference; statistics code reused verbatim from "
                   "scripts/compile/compile_results.py (ast-extracted: " + ", ".join(REUSED) + ")",
           "references_describe": {n: NS["describe"](r) for n, r in refs.items()},
           "primary": family([PRIMARY], refs)}
    sec = [n for n in SECONDARY if os.path.exists(os.path.join(RES, f"{n}_per_image.json"))]
    if sec:
        out["secondary"] = family(sec, refs)
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    p = os.path.join(RES, "ftblocks_stats.json")
    if os.path.exists(p):
        p = os.path.join(RES, f"ftblocks_stats_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(out, open(p, "w"), indent=1)
    d = out["primary"]["tests"]
    for k, v in d.items():
        c = v["CC"]
        print(f"{k}: dCC {c['mean_diff']:+.4f} [{c['boot95'][0]:+.4f}, {c['boot95'][1]:+.4f}] "
              f"Wilcoxon p {c['wilcoxon_p']:.2e} (Holm {c['wilcoxon_p_holm']:.2e}); NB t {c['nb_t']:.2f} p {c['nb_p']:.3f} "
              f"(Holm {c['nb_p_holm']:.3f}); folds won {c['folds_ref_better']}/5")
    print("written", p)


if __name__ == "__main__":
    main()
