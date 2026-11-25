"""
Matched-budget learning curves with the full SUM fine-tune curve (added 24 Nov 2026).

  python -I compile_curves.py      # after compile_results.py; writes results/curves_overview.json + results/CURVES.md

Inputs (nothing is re-trained or re-scored here; every per-image file is read as stored):
  sum_ft_curve_per_image.json (+ _nosigma)        SUM, all 57,501,196 params, SUM loss, 12 epochs, sigma on the n maps
  msdb_adapt_curves_per_image.json                MSDB head (26,463 params), KL+CC, 60 epochs
  msdb_adapt_dsp_curves_per_image.json            MSDB 13 per-dataset params
  curve_ours_{scratch,warm}_n{n}_r{r}_per_image   ours (3,721,089 params), KL+CC, 60 epochs
  iie_curve_klcce20_per_image.json                DeepGaze IIE adapted, the recipe compile_results.py uses for its curve
                                                  tests (KL+CC, 20 epochs, + sigma; chosen on fold-0 training data,
                                                  iie_recipe_selection_summary.json)
  main_table.json                                 curve table cross-check and n = 0 references
Test set: fold 0 of splits/kfold_seed0.json (200 ids) for every family; training subset of (n, r) identical
across families (compile_results.py subset_check).

Statistics, for each n and metric, difference = SUM minus other:
  * per image: mean over the resamples each family has at that n (SUM: 3 at n <= 200, 1 at n = 400 (r = 0) and
    n = 800 (sum_ft fold 0); others: 3), then the difference;
  * paired bootstrap 95% CI of the mean per-image difference (5000 resamples of the 200 images, default_rng(0),
    the same image draws for CC, SIM and KLD);
  * Wilcoxon signed-rank (two-sided, zero_method 'wilcox') on the 200 per-image differences, raw and Holm-corrected
    within each comparison over 6 budgets x 3 metrics (the convention of compile_results.py curve tests);
  * resamples won: resample r of SUM vs resample r of the other family (same training subset), where both exist.
  Only one test fold is used, so these are per-image tests on 200 images, not fold-level tests (no NB test).
Crossing / budget to reach a CC level: piecewise-linear interpolation of the curve mean in log2(n) between the
two adjacent budgets that bracket the level (budgets double, so log2 is the natural axis); linear-in-n
interpolation reported as a sensitivity; no extrapolation beyond 25 or 800. Percentile 95% intervals from 5000
bootstraps of the 200 test images (test-image sampling only; training-resample variation and the single-run
n = 400 / 800 points are not covered by these intervals).
"""
import sys
sys.dont_write_bytecode = True
import os, json, math, time
import numpy as np
from scipy import stats

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RES = os.path.join(KB, "results")
METRICS = ("CC", "SIM", "KLD")
HIGHER = {"CC": True, "SIM": True, "KLD": False}
NS = [25, 50, 100, 200, 400, 800]
B_BOOT = 5000
T0 = time.time()
SPL = json.load(open(os.path.join(KB, "splits", "kfold_seed0.json")))["folds"]
F0 = sorted([str(i) for i in SPL[0]], key=int)
assert len(F0) == 200
MT = json.load(open(os.path.join(RES, "main_table.json")))


def jload(name):
    return json.load(open(os.path.join(RES, name)))


# ------------------------------------------------------------------ load: fam -> n -> r -> metric -> array(200)
def from_multi(fname):
    out = {}
    for x in jload(fname):
        out.setdefault(int(x["n"]), {}).setdefault(int(x["resample"]), {})[str(x["id"])] = x
    return out


def from_ours(init):
    out = {}
    for n in NS:
        for r in range(3):
            p = os.path.join(RES, f"curve_ours_{init}_n{n}_r{r}_per_image.json")
            if os.path.exists(p):
                out.setdefault(n, {})[r] = {str(x["id"]): x for x in json.load(open(p))}
    return out


def to_arrays(raw):
    out = {}
    for n, byr in raw.items():
        for r, rows in byr.items():
            assert sorted(rows, key=int) == F0, (n, r)
            out.setdefault(n, {})[r] = {m: np.array([float(rows[i][m]) for i in F0]) for m in METRICS}
    return {n: dict(sorted(v.items())) for n, v in sorted(out.items())}


FAM = {
    "sum_ft_full": to_arrays(from_multi("sum_ft_curve_per_image.json")),
    "sum_ft_full_nosigma": to_arrays(from_multi("sum_ft_curve_nosigma_per_image.json")),
    "msdb_adapt_head": to_arrays(from_multi("msdb_adapt_curves_per_image.json")),
    "msdb_adapt_dsp": to_arrays(from_multi("msdb_adapt_dsp_curves_per_image.json")),
    "ours_scratch": to_arrays(from_ours("scratch")),
    "ours_warm": to_arrays(from_ours("warm")),
    "iie_adapt_klcc_e20": to_arrays(from_multi("iie_curve_klcce20_per_image.json")),
}
LABEL = {"sum_ft_full": "SUM full fine-tune (57,501,196 params)",
         "sum_ft_full_nosigma": "SUM full fine-tune, no post-hoc sigma",
         "msdb_adapt_head": "DeepGaze MSDB adapted, head (26,463 params)",
         "msdb_adapt_dsp": "DeepGaze MSDB adapted, 13 per-dataset params",
         "ours_scratch": "Ours, from scratch (3,721,089 params)",
         "ours_warm": "Ours, warm start (3,721,089 params)",
         "iie_adapt_klcc_e20": "DeepGaze IIE adapted, KL+CC 20 ep + sigma (selected recipe; 3,059,160 params)"}

# cross-check: curve means equal the compiled table in main_table.json
xc_worst = 0.0
for fam, cur in FAM.items():
    for n, byr in cur.items():
        t = MT["curves"]["table"][fam][str(n)]
        assert t["n_resamples"] == len(byr), (fam, n)
        for m in METRICS:
            xc_worst = max(xc_worst, abs(float(np.mean([v[m].mean() for v in byr.values()])) - t[f"mean_{m}"]))
assert xc_worst < 1e-12, xc_worst


def per_image_mean(fam, n, m, rs=None):
    byr = FAM[fam][n]
    rs = list(byr) if rs is None else rs
    return np.mean([byr[r][m] for r in rs], axis=0)


def holm(pvals):          # identical to compile_results.holm
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); adj = np.empty(n); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, (n - rank) * p[idx])); adj[idx] = run
    return adj


def paired(a_by_m, b_by_m, seed=0):
    """a_by_m, b_by_m: metric -> per-image array (200). Difference a - b."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(F0), size=(B_BOOT, len(F0)))
    o = {}
    for m in METRICS:
        a, b = a_by_m[m], b_by_m[m]
        d = a - b
        bm = d[idx].mean(1)
        better = (d > 0) if HIGHER[m] else (d < 0)
        try:
            w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            wp, ws = float(w.pvalue), float(w.statistic)
        except ValueError:
            wp, ws = 1.0, float("nan")
        o[m] = {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()),
                "median_diff": float(np.median(d)),
                "boot95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
                "frac_images_sum_better": float(better.mean()), "wilcoxon_p": wp, "wilcoxon_W": ws}
    return o


# ------------------------------------------------------------------ matched-n comparisons (SUM minus other)
PRIMARY = ["msdb_adapt_head", "ours_scratch", "iie_adapt_klcc_e20"]
SECONDARY = ["ours_warm", "msdb_adapt_dsp"]
comparisons = {}
for oth in PRIMARY + SECONDARY:
    key = f"sum_ft_full_minus_{oth}"
    byn = {}
    for n in NS:
        a = {m: per_image_mean("sum_ft_full", n, m) for m in METRICS}
        b = {m: per_image_mean(oth, n, m) for m in METRICS}
        o = paired(a, b)
        common = [r for r in FAM["sum_ft_full"][n] if r in FAM[oth][n]]
        for m in METRICS:
            won = sum(((FAM["sum_ft_full"][n][r][m].mean() > FAM[oth][n][r][m].mean()) if HIGHER[m] else
                       (FAM["sum_ft_full"][n][r][m].mean() < FAM[oth][n][r][m].mean())) for r in common)
            o[m].update({"resamples_sum_better": int(won), "n_resamples_paired": len(common),
                         "n_resamples_sum": len(FAM["sum_ft_full"][n]), "n_resamples_other": len(FAM[oth][n])})
        byn[n] = o
    flat = [(n, m) for n in NS for m in METRICS]
    adj = holm([byn[n][m]["wilcoxon_p"] for n, m in flat])
    for (n, m), pa in zip(flat, adj):
        byn[n][m]["wilcoxon_p_holm"] = float(pa)
    comparisons[key] = {"primary": oth in PRIMARY, "other": oth, "other_label": LABEL[oth], "by_n": byn}

# consistency with main_table (ours minus SUM there): mean differences and Wilcoxon p must agree up to sign
cons_worst = 0.0
for refn in ("ours_scratch", "ours_warm"):
    mt = MT["curves"]["paired_tests"][f"{refn}_vs_sum_ft_full"]
    for n in NS:
        for m in METRICS:
            c = comparisons[f"sum_ft_full_minus_{refn}"]["by_n"][n][m]
            cons_worst = max(cons_worst, abs(c["mean_diff"] + mt[str(n)][m]["mean_diff"]),
                             abs(c["wilcoxon_p"] - mt[str(n)][m]["wilcoxon_p"]))
assert cons_worst < 1e-12, cons_worst

# sensitivity at the single-resample budgets
sens = {}
for oth in PRIMARY + SECONDARY:
    s = {}
    for n in (400, 800):
        if 0 in FAM[oth][n]:
            s[f"n{n}_r0_only"] = paired({m: FAM["sum_ft_full"][n][0][m] for m in METRICS},
                                        {m: FAM[oth][n][0][m] for m in METRICS})
    s["n800_sum_nosigma_vs_other"] = paired({m: per_image_mean("sum_ft_full_nosigma", 800, m) for m in METRICS},
                                            {m: per_image_mean(oth, 800, m) for m in METRICS})
    sens[f"sum_ft_full_minus_{oth}"] = s
sum_sigma_effect = {n: {m: float(per_image_mean("sum_ft_full", n, m).mean() - per_image_mean("sum_ft_full_nosigma", n, m).mean())
                        for m in METRICS} for n in NS}
# does any n = 800 conclusion (sign of the CC difference, and whether the 95% CI excludes 0) change without sigma?
n800_nosigma_same = all(
    (np.sign(C_["CC"]["mean_diff"]) == np.sign(comparisons[k]["by_n"][800]["CC"]["mean_diff"])) and
    ((C_["CC"]["boot95"][0] > 0 or C_["CC"]["boot95"][1] < 0) ==
     (comparisons[k]["by_n"][800]["CC"]["boot95"][0] > 0 or comparisons[k]["by_n"][800]["CC"]["boot95"][1] < 0))
    for k, C_ in ((k, s["n800_sum_nosigma_vs_other"]) for k, s in sens.items()))

# cross-budget: each model at budget n vs our scratch readout at n = 800 (same 200 test images, paired by image)
cross_budget = {}
for f in ("sum_ft_full", "msdb_adapt_head"):
    ref800 = {m: per_image_mean("ours_scratch", 800, m) for m in METRICS}
    cross_budget[f"{f}_at_n_minus_ours_scratch_at_800"] = {
        n: paired({m: per_image_mean(f, n, m) for m in METRICS}, ref800) for n in NS}


# ------------------------------------------------------------------ crossings and budget to reach a level
def interp_log2(n1, y1, n2, y2, target):
    x1, x2 = math.log2(n1), math.log2(n2)
    return 2 ** (x1 + (x2 - x1) * (target - y1) / (y2 - y1))


def interp_lin(n1, y1, n2, y2, target):
    return n1 + (n2 - n1) * (target - y1) / (y2 - y1)


def budget_to_reach(means, level, how="log2"):
    """means: list over NS (CC, higher better). First budget whose mean >= level, interpolated from the previous
    budget. Returns (value, status): status 'interpolated', 'at_or_below_25', 'not_reached_by_800'."""
    f = interp_log2 if how == "log2" else interp_lin
    for k, (n, y) in enumerate(zip(NS, means)):
        if y >= level:
            if k == 0:
                return 25.0, "at_or_below_25"
            return float(f(NS[k - 1], means[k - 1], n, y, level)), "interpolated"
    return float("inf"), "not_reached_by_800"


def crossings(diff, how="log2"):
    """diff: list over NS of (SUM - other) curve-mean difference; every sign change, interpolated."""
    f = interp_log2 if how == "log2" else interp_lin
    out = []
    for k in range(len(NS) - 1):
        d1, d2 = diff[k], diff[k + 1]
        if d1 == 0:
            out.append({"n": float(NS[k]), "between": [NS[k], NS[k]], "direction": "touch"})
        elif d1 * d2 < 0:
            out.append({"n": float(f(NS[k], d1, NS[k + 1], d2, 0.0)), "between": [NS[k], NS[k + 1]],
                        "direction": "SUM ahead -> SUM behind" if d1 > 0 else "SUM behind -> SUM ahead"})
    return out


def curve_means(fam, m, img_idx=None):
    vals = []
    for n in NS:
        v = per_image_mean(fam, n, m)
        vals.append(float(v.mean() if img_idx is None else v[img_idx].mean()))
    return vals


CURVE_FAMS = ["sum_ft_full", "msdb_adapt_head", "ours_scratch", "ours_warm", "iie_adapt_klcc_e20", "msdb_adapt_dsp",
              "sum_ft_full_nosigma"]
LEVELS = [0.90, 0.91]
rng = np.random.default_rng(1)
boot_idx = rng.integers(0, len(F0), size=(B_BOOT, len(F0)))
# per-image curve arrays once: fam -> metric -> (6, 200)
PI = {f: {m: np.stack([per_image_mean(f, n, m) for n in NS]) for m in METRICS} for f in CURVE_FAMS}
BOOT_MEANS = {f: {m: PI[f][m][:, boot_idx].mean(2).T for m in METRICS} for f in CURVE_FAMS}   # (B, 6)

budget = {}
for f in CURVE_FAMS:
    means = curve_means(f, "CC")
    o = {"curve_mean_CC": dict(zip(map(str, NS), means)),
         "monotone_increasing": all(b > a for a, b in zip(means, means[1:]))}
    for lev in LEVELS:
        v, st = budget_to_reach(means, lev, "log2")
        vl, _ = budget_to_reach(means, lev, "lin")
        bs = np.array([budget_to_reach(list(BOOT_MEANS[f]["CC"][b]), lev, "log2")[0] for b in range(B_BOOT)])
        frac_nr = float(np.isinf(bs).mean())
        # 'not reached by 800' = +inf; inverted-CDF percentiles pick sample values, so inf never meets arithmetic
        lo, hi = (float(np.percentile(bs, 2.5, method="inverted_cdf")),
                  float(np.percentile(bs, 97.5, method="inverted_cdf")))
        o[f"{lev:.2f}"] = {"n_log2_interp": v, "status": st, "n_linear_interp": vl,
                           "boot95_log2": [lo, hi], "boot_frac_not_reached_by_800": frac_nr}
    budget[f] = o

cross = {}
for oth in ["msdb_adapt_head", "ours_scratch", "iie_adapt_klcc_e20", "ours_warm", "msdb_adapt_dsp"]:
    o = {}
    for m in METRICS:
        diff = [a - b for a, b in zip(curve_means("sum_ft_full", m), curve_means(oth, m))]
        if not HIGHER[m]:
            diff = [-x for x in diff]          # positive = SUM better for every metric
        cl = crossings(diff, "log2")
        cn = crossings(diff, "lin")
        e = {"diff_sum_better_positive": dict(zip(map(str, NS), diff)), "crossings_log2": cl,
             "crossings_linear": cn}
        if m == "CC" or oth == "msdb_adapt_head":
            bd = BOOT_MEANS["sum_ft_full"][m] - BOOT_MEANS[oth][m]
            if not HIGHER[m]:
                bd = -bd
            first = []
            for b in range(B_BOOT):
                c = [x for x in crossings(list(bd[b]), "log2") if x["direction"] == "SUM ahead -> SUM behind"]
                first.append(c[0]["n"] if c else float("nan"))
            first = np.array(first)
            ok = ~np.isnan(first)
            e["boot_first_down_crossing"] = {
                "frac_with_crossing": float(ok.mean()),
                "boot95_log2": [float(np.percentile(first[ok], 2.5)), float(np.percentile(first[ok], 97.5))] if ok.any() else None,
                "n_sign_patterns": {"SUM better at every n": int((bd > 0).all(1).sum()),
                                    "SUM worse at every n": int((bd < 0).all(1).sum())}}
        o[m] = e
    cross[f"sum_ft_full_vs_{oth}"] = o

# n = 0 references on fold 0 (from main_table)
n0 = {k: MT["curves"]["n0_fold0"][k] for k in ("sum_zeroshot_t2", "sum_calibrated", "msdb_zeroshot_uniform",
                                               "msdb_zeroshot_foldprior", "iie_zeroshot_uniform")}

overview = {
    "description": "Matched-budget learning curves on the fold-0 test set (200 ids of splits/kfold_seed0.json), with "
                   "the full SUM fine-tune curve added 24 Nov 2026. Differences are SUM minus other (KLD: negative = "
                   "SUM better).",
    "test_set": "fold 0 of splits/kfold_seed0.json (200 ids); the same 200 images for every family and budget",
    "only_one_test_fold": "All tests are per-image tests on 200 images of ONE test fold; images are the unit, so they "
                          "measure test-image sampling only. They are not fold-level tests (no Nadeau-Bengio test is "
                          "possible), and training-subset variation enters only through the average over resamples.",
    "families": {f: {"label": LABEL[f], "resamples_per_n": {str(n): list(FAM[f][n]) for n in NS}} for f in FAM},
    "best_iie_curve": "iie_adapt_klcc_e20 (= iie_curve_klcce20: KL+CC, 20 epochs, + sigma), the recipe chosen on "
                      "fold-0 training data (iie_recipe_selection_summary.json) and the IIE curve compile_results.py "
                      "uses in its curve tests; the partial 60-epoch curve is worse at every n",
    "subset_check": MT["curves"]["subset_check"],
    "caveats": MT["curves"]["sum_ft_full_caveats"] + [
        "training policy: every family uses its own fixed recipe at every n (SUM 12 epochs, batch 16, steps = 12 x "
        "ceil(n/16); ours and MSDB 60 epochs, batch 16; IIE 20 epochs, batch 8); no family was tuned per budget",
        f"SUM's n = 800 sigma (chosen on 200 of the 800 maps) changes test CC by {sum_sigma_effect[800]['CC']:+.4f} over "
        "no sigma; with SUM's no-sigma n = 800 scores the sign and CI-excludes-zero status of every n = 800 CC "
        "comparison " + ("are unchanged" if n800_nosigma_same else "CHANGE for at least one comparison")
        + " (sensitivity_single_resample_budgets)"],
    "methods": __doc__,
    "matched_n_tests": comparisons,
    "sensitivity_single_resample_budgets": sens,
    "sum_sigma_effect_curve_mean": sum_sigma_effect,
    "n800_conclusions_same_without_sum_sigma": n800_nosigma_same,
    "cross_budget_vs_ours_scratch_800": cross_budget,
    "budget_to_reach_CC": budget,
    "crossings": cross,
    "n0_fold0_references": n0,
    "cross_checks": {"curve_means_vs_main_table_max_abs": xc_worst,
                     "ours_vs_sum_tests_vs_main_table_max_abs": cons_worst},
    "command": "python -I " + " ".join(sys.argv), "wall_clock_s": None,
}
overview["wall_clock_s"] = time.time() - T0


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    if isinstance(o, np.floating):
        return clean(float(o))
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(overview), open(os.path.join(RES, "curves_overview.json"), "w"), indent=1)

# ------------------------------------------------------------------ CURVES.md
L = []
w = L.append


def pf(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def nfmt(v, st):
    if st == "not_reached_by_800":
        return "not reached by 800"
    if st == "at_or_below_25":
        return "<= 25"
    return f"{v:.0f}"


def bfmt(iv, frac):
    lo, hi = iv
    hs = "> 800" if math.isinf(hi) else f"{hi:.0f}"
    ls = "> 800" if math.isinf(lo) else f"{lo:.0f}"
    pct = 100 * frac
    ptxt = f"{pct:.1f}" if 99.5 <= pct < 100 else f"{pct:.0f}"
    return f"{ls} to {hs}" + (f" ({ptxt}% of bootstraps not reached)" if frac > 0 else "")


C = comparisons
cm = {f: curve_means(f, "CC") for f in CURVE_FAMS}
short = {"sum_ft_full": "SUM full FT", "msdb_adapt_head": "MSDB head", "ours_scratch": "Ours scratch",
         "ours_warm": "Ours warm", "iie_adapt_klcc_e20": "IIE adapted (e20)", "msdb_adapt_dsp": "MSDB 13 params"}
mx = cross["sum_ft_full_vs_msdb_adapt_head"]["CC"]
xc = mx["crossings_log2"][0] if mx["crossings_log2"] else None
xcl = mx["crossings_linear"][0] if mx["crossings_linear"] else None
bcr = mx["boot_first_down_crossing"]
B = budget
cs = C["sum_ft_full_minus_ours_scratch"]["by_n"]
cmh = C["sum_ft_full_minus_msdb_adapt_head"]["by_n"]
ci = C["sum_ft_full_minus_iie_adapt_klcc_e20"]["by_n"]

w("# Matched-budget learning curves with full SUM fine-tuning (ADD1000, fold-0 test set)")
w("")
w("Compiled 24 Nov 2026 by scripts/compile/compile_curves.py from the per-image curve files (sources and all numbers: "
  "curves_overview.json; the curve table and ours-minus-SUM tests are also in main_table.json). "
  "Test set = fold 0 of splits/kfold_seed0.json (200 ids) for every model and budget; the training subset of each "
  "(n, r) is the same for every model (default_rng(1000n + r) draw from the 800-id pool; checked, 0 mismatches). "
  "SUM = all 57,501,196 parameters fine-tuned with SUM's own loss (12 epochs, token 2), post-hoc sigma chosen on "
  "the same n maps (n = 800: on 200 of the 800 maps, see caveats), trained on a cloud NVIDIA A10G.")
w("")
w("## Plain-language summary")
w("")
def sig(c, a=0.05):
    return c["wilcoxon_p_holm"] < a


CB = cross_budget["sum_ft_full_at_n_minus_ours_scratch_at_800"]
CBM = cross_budget["msdb_adapt_head_at_n_minus_ours_scratch_at_800"]
n_sum_match = next((n for n in NS if CB[n]["CC"]["mean_diff"] >= 0), None)
n_msdb_match = next((n for n in NS if CBM[n]["CC"]["mean_diff"] >= 0), None)
lead_n = [n for n in NS if cmh[n]["CC"]["mean_diff"] > 0]
trail_n = [n for n in NS if cmh[n]["CC"]["mean_diff"] < 0]
w(f"* Fully fine-tuned SUM is the most data-efficient model at small budgets: with 25 maps it reaches CC "
  f"{cm['sum_ft_full'][0]:.4f}, above adapted MSDB ({cm['msdb_adapt_head'][0]:.4f}), our readout trained from "
  f"scratch ({cm['ours_scratch'][0]:.4f}) and adapted DeepGaze IIE ({cm['iie_adapt_klcc_e20'][0]:.4f}). "
  + (f"SUM first matches our scratch readout trained on all 800 maps (CC {cm['ours_scratch'][5]:.4f}) at n = "
     f"{n_sum_match} (CC {cm['sum_ft_full'][NS.index(n_sum_match)]:.4f}; per-image difference "
     f"{CB[n_sum_match]['CC']['mean_diff']:+.4f} [{CB[n_sum_match]['CC']['boot95'][0]:+.4f}, "
     f"{CB[n_sum_match]['CC']['boot95'][1]:+.4f}], p {pf(CB[n_sum_match]['CC']['wilcoxon_p'])}, i.e. level, not "
     f"better); adapted MSDB at n = {n_sum_match} is level with it too ({CBM[n_sum_match]['CC']['mean_diff']:+.4f} "
     f"[{CBM[n_sum_match]['CC']['boot95'][0]:+.4f}, {CBM[n_sum_match]['CC']['boot95'][1]:+.4f}], p "
     f"{pf(CBM[n_sum_match]['CC']['wilcoxon_p'])}) and first exceeds its mean at n = {n_msdb_match}." if n_sum_match else ""))
w(f"* SUM and adapted MSDB cross between n = {xc['between'][0]} and {xc['between'][1]} (about {xc['n']:.0f} maps, "
  f"log2 interpolation; {xcl['n']:.0f} with linear interpolation). SUM leads at n = "
  f"{', '.join(map(str, lead_n))} (CC " + ", ".join(f"{cmh[n]['CC']['mean_diff']:+.4f}" for n in lead_n)
  + "; " + ", ".join(f"n = {n} Holm p {pf(cmh[n]['CC']['wilcoxon_p_holm'])}" for n in lead_n if n <= 50)
  + f"; 95% CI excludes 0 at n = {', '.join(str(n) for n in lead_n if cmh[n]['CC']['boot95'][0] > 0)}). MSDB leads "
  f"at n = {', '.join(map(str, trail_n))} by {min(-cmh[n]['CC']['mean_diff'] for n in trail_n):.4f} to "
  f"{max(-cmh[n]['CC']['mean_diff'] for n in trail_n):.4f} CC, but at none of these budgets is the difference "
  + ("significant" if any(sig(cmh[n]["CC"]) for n in trail_n) else "significant (Holm p = 1.000; raw p "
     + " to ".join(pf(f(cmh[n]["CC"]["wilcoxon_p"] for n in trail_n)) for f in (min, max))
     + "; every 95% CI includes 0)")
  + (", although MSDB wins every paired resample there" if all(
      cmh[n]["CC"]["resamples_sum_better"] == 0 for n in trail_n) else "") + f". In the {100*bcr['frac_with_crossing']:.0f}% of bootstraps over test images that show a crossing, the first "
  f"crossing lies at {bcr['boot95_log2'][0]:.0f} to {bcr['boot95_log2'][1]:.0f} maps (percentile 95% among those "
  f"bootstraps); the other {100*(1-bcr['frac_with_crossing']):.0f}% keep SUM ahead at every n. On SIM SUM stays ahead of MSDB at every n (Holm-significant at n = "
  + ", ".join(str(n) for n in NS if sig(cmh[n]["SIM"]) and cmh[n]["SIM"]["mean_diff"] > 0)
  + "). On KLD SUM is lower at every n, but significantly so only at n = "
  + ", ".join(str(n) for n in NS if sig(cmh[n]["KLD"]) and cmh[n]["KLD"]["mean_diff"] < 0)
  + f"; from n = 200 the KLD gap is at most {max(abs(cmh[n]['KLD']['mean_diff']) for n in NS if n >= 200):.4f} "
  f"and {100*cross['sum_ft_full_vs_msdb_adapt_head']['KLD']['boot_first_down_crossing']['frac_with_crossing']:.0f}% "
  "of bootstraps show a KLD crossing, so KLD is level from n = 200. Only the SIM lead is robust across budgets.")
reach = lambda lev: [short[f] for f in ("sum_ft_full", "msdb_adapt_head", "ours_scratch", "ours_warm",
                                        "iie_adapt_klcc_e20", "msdb_adapt_dsp") if B[f][lev]["status"] == "not_reached_by_800"]
ratio90 = B["ours_scratch"]["0.90"]["n_log2_interp"] / B["sum_ft_full"]["0.90"]["n_log2_interp"]
w(f"* To reach CC 0.90 SUM needs about {nfmt(B['sum_ft_full']['0.90']['n_log2_interp'], B['sum_ft_full']['0.90']['status'])} maps, "
  f"MSDB about {nfmt(B['msdb_adapt_head']['0.90']['n_log2_interp'], B['msdb_adapt_head']['0.90']['status'])}, our scratch readout about "
  f"{nfmt(B['ours_scratch']['0.90']['n_log2_interp'], B['ours_scratch']['0.90']['status'])} (about {ratio90:.0f}x SUM's); "
  f"not reached by 800: {', '.join(reach('0.90'))}. To reach CC 0.91 MSDB needs about "
  f"{nfmt(B['msdb_adapt_head']['0.91']['n_log2_interp'], B['msdb_adapt_head']['0.91']['status'])} maps and SUM about "
  f"{nfmt(B['sum_ft_full']['0.91']['n_log2_interp'], B['sum_ft_full']['0.91']['status'])} (this interpolation uses "
  f"SUM's single n = 400 run; bootstrap intervals overlap widely); not reached by 800: {', '.join(reach('0.91'))}.")
w(f"* Against our scratch readout SUM is better at every budget, by {cs[25]['CC']['mean_diff']:+.4f} CC at n = 25 "
  f"shrinking to {cs[800]['CC']['mean_diff']:+.4f} at n = 800 (Holm p <= "
  f"{pf(max(cs[n]['CC']['wilcoxon_p_holm'] for n in NS))}, every paired resample). Against adapted IIE it is better "
  f"by {min(ci[n]['CC']['mean_diff'] for n in NS):+.4f} to {max(ci[n]['CC']['mean_diff'] for n in NS):+.4f} CC "
  f"(Holm p <= {pf(max(ci[n]['CC']['wilcoxon_p_holm'] for n in NS))})."
  + (" Both hold on SIM and KLD at every n (Holm p < 0.05)." if all(
      sig(C[f"sum_ft_full_minus_{o}"]["by_n"][n][m]) and (C[f"sum_ft_full_minus_{o}"]["by_n"][n][m]["mean_diff"] > 0)
      == HIGHER[m] for o in ("ours_scratch", "iie_adapt_klcc_e20") for n in NS for m in ("SIM", "KLD")) else ""))
w("* All tests use one test fold of 200 images: they are per-image tests (images are the unit), not fold-level tests, "
  "and the bootstrap intervals cover test-image sampling only. n = 400 has one SUM resample and n = 800 is the "
  "k-fold sum_ft fold 0 (see caveats), so the two largest budgets rest on single SUM runs.")
w("")
w("## Curve means (CC; mean over resamples, SD over resamples ddof 0; 1 run = single resample)")
w("")
fams_md = ["sum_ft_full", "msdb_adapt_head", "ours_scratch", "ours_warm", "iie_adapt_klcc_e20", "msdb_adapt_dsp"]
w("| n | " + " | ".join(short[f] for f in fams_md) + " |")
w("|---" * (len(fams_md) + 1) + "|")
for k, n in enumerate(NS):
    cells = []
    for f in fams_md:
        runs = [FAM[f][n][r]["CC"].mean() for r in FAM[f][n]]
        cells.append(f"{np.mean(runs):.4f} ({np.std(runs):.4f})" if len(runs) > 1 else f"{runs[0]:.4f} (1 run)")
    w(f"| {n} | " + " | ".join(cells) + " |")
w("")
w("## Matched-n differences, SUM minus other (CC)")
w("")
w("Per image: mean over the resamples each model has at that n; then SUM minus other. CI = paired bootstrap 95% "
  "(5000 resamples of the 200 images). p = Wilcoxon signed-rank on the 200 per-image differences, raw / Holm "
  "(Holm within each comparison over 6 budgets x 3 metrics). img = share of images where SUM is better; res = "
  "resamples where SUM is better, paired by training subset. Resamples averaged (SUM / other) are 3 / 3 at "
  "n <= 200 and 1 / 3 at n = 400 and 800.")
w("")
w("| n | SUM - MSDB head [95% CI] | p raw / Holm | img | res | SUM - ours scratch [95% CI] | p raw / Holm | img | res | SUM - IIE adapted [95% CI] | p raw / Holm | img | res |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for n in NS:
    cells = []
    for oth in PRIMARY:
        c = C[f"sum_ft_full_minus_{oth}"]["by_n"][n]["CC"]
        cells += [f"{c['mean_diff']:+.4f} [{c['boot95'][0]:+.4f}, {c['boot95'][1]:+.4f}]",
                  f"{pf(c['wilcoxon_p'])} / {pf(c['wilcoxon_p_holm'])}", f"{100*c['frac_images_sum_better']:.0f}%",
                  f"{c['resamples_sum_better']}/{c['n_resamples_paired']}"]
    w(f"| {n} | " + " | ".join(cells) + " |")
w("")
w("Same comparisons for SIM and KLD (difference [95% CI], Holm p; KLD negative = SUM better):")
w("")
w("| n | SIM: SUM - MSDB | SIM: SUM - ours scratch | SIM: SUM - IIE | KLD: SUM - MSDB | KLD: SUM - ours scratch | KLD: SUM - IIE |")
w("|---|---|---|---|---|---|---|")
for n in NS:
    cells = []
    for m in ("SIM", "KLD"):
        for oth in PRIMARY:
            c = C[f"sum_ft_full_minus_{oth}"]["by_n"][n][m]
            cells.append(f"{c['mean_diff']:+.4f} [{c['boot95'][0]:+.4f}, {c['boot95'][1]:+.4f}] ({pf(c['wilcoxon_p_holm'])})")
    w(f"| {n} | " + " | ".join(cells) + " |")
w("")
w("Secondary (CC): SUM minus our warm start and SUM minus MSDB's 13-parameter adaptation:")
w("")
w("| n | SUM - ours warm [95% CI] | Holm p | res | SUM - MSDB 13 params [95% CI] | Holm p | res |")
w("|---|---|---|---|---|---|---|")
for n in NS:
    cells = []
    for oth in SECONDARY:
        c = C[f"sum_ft_full_minus_{oth}"]["by_n"][n]["CC"]
        cells += [f"{c['mean_diff']:+.4f} [{c['boot95'][0]:+.4f}, {c['boot95'][1]:+.4f}]", pf(c["wilcoxon_p_holm"]),
                  f"{c['resamples_sum_better']}/{c['n_resamples_paired']}"]
    w(f"| {n} | " + " | ".join(cells) + " |")
w("")
w("Cross-budget (CC): SUM and MSDB at budget n minus our scratch readout at n = 800 (same 200 images, per-image "
  "paired; bootstrap CI, raw Wilcoxon p):")
w("")
w("| n | SUM at n - ours scratch at 800 | MSDB head at n - ours scratch at 800 |")
w("|---|---|---|")
for n in NS:
    a_, b_ = CB[n]["CC"], CBM[n]["CC"]
    w(f"| {n} | {a_['mean_diff']:+.4f} [{a_['boot95'][0]:+.4f}, {a_['boot95'][1]:+.4f}], p {pf(a_['wilcoxon_p'])} | "
      f"{b_['mean_diff']:+.4f} [{b_['boot95'][0]:+.4f}, {b_['boot95'][1]:+.4f}], p {pf(b_['wilcoxon_p'])} |")
w("")
w("### Sensitivity at the single-resample budgets (CC, SUM minus other)")
w("")
w("r0 only = SUM r = 0 against the other model's r = 0 alone (same training subset); no sigma = SUM's n = 800 "
  "scores without post-hoc sigma (removes the sigma-rule difference of the n = 800 point). Raw Wilcoxon p.")
w("")
w("| Other | n = 400, r0 only | n = 800, r0 only | n = 800, SUM no sigma vs other (3 resamples) |")
w("|---|---|---|---|")
for oth in PRIMARY + SECONDARY:
    s = sens[f"sum_ft_full_minus_{oth}"]
    cells = []
    for k in ("n400_r0_only", "n800_r0_only", "n800_sum_nosigma_vs_other"):
        c = s[k]["CC"]
        cells.append(f"{c['mean_diff']:+.4f} [{c['boot95'][0]:+.4f}, {c['boot95'][1]:+.4f}], p {pf(c['wilcoxon_p'])}")
    w(f"| {short[oth]} | " + " | ".join(cells) + " |")
w("")
w("## Crossing of the SUM and MSDB curves")
w("")
w("Difference of curve means, SUM minus MSDB head (positive = SUM better; KLD sign flipped so that positive = SUM "
  "better for every metric). Crossing = zero of the difference, piecewise-linear in log2(n) between the bracketing "
  "budgets (linear-in-n in brackets). Bootstrap = first SUM-ahead to SUM-behind crossing in 5000 bootstraps of the "
  "200 test images; the 95% range is the percentile interval among the bootstraps that show such a crossing (their "
  "share in brackets), and every other bootstrap keeps SUM ahead at every n.")
w("")
w("| Metric | " + " | ".join(f"n={n}" for n in NS) + " | crossing (log2 / linear) | bootstrap 95% |")
w("|---" * (len(NS) + 3) + "|")
for m in METRICS:
    e = cross["sum_ft_full_vs_msdb_adapt_head"][m]
    diffs = " | ".join(f"{e['diff_sum_better_positive'][str(n)]:+.4f}" for n in NS)
    cl = "; ".join(f"{x['n']:.0f} ({y['n']:.0f}), {x['direction']}" for x, y in zip(e["crossings_log2"], e["crossings_linear"])) or "none"
    bb = e["boot_first_down_crossing"]
    bs = (f"{bb['boot95_log2'][0]:.0f} to {bb['boot95_log2'][1]:.0f} ({100*bb['frac_with_crossing']:.0f}% cross)"
          if bb["boot95_log2"] else "no crossing")
    w(f"| {m} | {diffs} | {cl} | {bs} |")
w("")
others_x = []
for oth in ["ours_scratch", "ours_warm", "iie_adapt_klcc_e20", "msdb_adapt_dsp"]:
    e = cross[f"sum_ft_full_vs_{oth}"]["CC"]
    others_x.append(f"{short[oth]}: " + ("no crossing, SUM ahead at every n" if not e["crossings_log2"] and
                                         all(v > 0 for v in e["diff_sum_better_positive"].values()) else
                                         "; ".join(f"{x['n']:.0f}" for x in e["crossings_log2"]) or "no crossing"))
w("Other CC curves vs SUM: " + "; ".join(others_x) + ".")
w("")
w("## Maps needed to reach CC 0.90 and 0.91")
w("")
w("First budget whose curve mean reaches the level, interpolated piecewise-linearly in log2(n) from the budget below "
  "(linear-in-n in brackets); no extrapolation beyond n = 800. Bootstrap = 5000 bootstraps of the 200 test images "
  "(percentile 95%). SUM's 0.91 value interpolates between n = 200 (3 resamples) and its single n = 400 run.")
w("")
w("| Model | CC 0.90: maps (linear) | bootstrap 95% | CC 0.91: maps (linear) | bootstrap 95% |")
w("|---|---|---|---|---|")
for f in ["sum_ft_full", "sum_ft_full_nosigma", "msdb_adapt_head", "ours_scratch", "ours_warm", "iie_adapt_klcc_e20",
          "msdb_adapt_dsp"]:
    cells = []
    for lev in ("0.90", "0.91"):
        o = B[f][lev]
        lin = "" if o["status"] != "interpolated" else f" ({o['n_linear_interp']:.0f})"
        cells += [nfmt(o["n_log2_interp"], o["status"]) + lin, bfmt(o["boot95_log2"], o["boot_frac_not_reached_by_800"])]
    w(f"| {short.get(f, 'SUM full FT, no sigma')} | " + " | ".join(cells) + " |")
w("")
w("## n = 0 references on the same 200 test images")
w("")
w("| Reference | CC | SIM | KLD |")
w("|---|---|---|---|")
for k, v in n0.items():
    w(f"| {k} ({v['note']}) | {v['CC']:.4f} | {v['SIM']:.4f} | {v['KLD']:.4f} |")
w("")
w("## Caveats")
w("")
for c in overview["caveats"]:
    w(f"* {c}")
w("* Only one test fold (fold 0, 200 images) exists for the curves, so no test here is a fold-level test; with "
  "images as the unit, the Wilcoxon p-values and bootstrap CIs are liberal about generalisation to new training "
  "subsets and say nothing about other folds.")
w("* The n = 800 SUM point and the k-fold sum_ft row are the same run (fold 0); MSDB, ours and IIE at n = 800 are "
  "curve runs (3 resamples of the full 800-id pool, differing only in data order and init), not their k-fold rows.")
w("")
w("## Files")
w("")
for f in ["curves_overview.json", "main_table.json (curves.table.sum_ft_full*, curves.paired_tests.*_vs_sum_ft_full)",
          "sum_ft_curve_summary.json", "sum_ft_curve_per_image.json", "sum_ft_curve_nosigma_per_image.json"]:
    w(f"* {os.path.join(RES, f)}")
w(f"* script: {os.path.join(KB, 'scripts', 'compile', 'compile_curves.py')}")
w("")
txt = "\n".join(L) + "\n"
assert "\u2014" not in txt and "\u2013" not in txt, "dash check"
open(os.path.join(RES, "CURVES.md"), "w").write(txt)
print("wrote curves_overview.json and CURVES.md", f"{time.time() - T0:.1f}s")
