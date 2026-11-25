"""
Unified results of the KBS letter: every row on the 384 x 384 protocol, paired tests of our readout
against every other row, ablations, learning curves, grouped / pair-safe folds, near-duplicates.

  python -I compile_results.py            # writes results/main_table.json (+ tests)

Statistics (all from the saved per-image files; nothing is re-trained here):
  * per-image test: Wilcoxon signed-rank (two-sided, zero_method='wilcox') on the 1000 out-of-fold
    values, paired by image id (both rows on the same split);
  * fold-level test: Nadeau-Bengio corrected resampled t-test on the 5 per-fold mean differences,
    t = mean(d) / sqrt((1/K + n_test/n_train) * var(d, ddof=1)), df = K - 1, two-sided;
  * Holm correction across the family of compared rows, separately for each reference (ours seed 0,
    ours seed-averaged), metric (CC, SIM, KLD) and test (Wilcoxon, NB);
  * effect sizes: mean and median per-image difference with a 5000-sample bootstrap 95% CI of the
    mean (seed 0), paired Cohen's dz = mean(d) / sd(d), matched-pairs rank-biserial r, share of
    images on which ours is better, folds won.
Differences are always "ours minus other"; for KLD a negative difference means ours is better.
24 Nov 2026: rows sum_ft and sum_ft_klcc (full SUM fine-tune, cloud A10G) added to the main family, which
grows by 2 (Holm-adjusted p of the other rows can only rise or stay; all other numbers are unchanged).
24 Nov 2026 (later): the full SUM fine-tune learning curve (results/sum_ft_curve_*, scripts/sum/finalize_curve.py)
added to curves: table rows sum_ft_full / sum_ft_full_nosigma and paired tests ours_{scratch,warm}_vs_sum_ft_full
(each comparison is its own Holm family over budgets x metrics, so no existing family grows). SUM has one resample
at n = 400 (r = 0) and n = 800 (= sum_ft fold 0: sigma on 200 of the 800 maps, other data order and torch seed).
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, glob, math
import numpy as np
from scipy import stats

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RES = os.path.join(KB, "results")
SPL = os.path.join(KB, "splits")
METRICS = ("CC", "SIM", "KLD")
HIGHER = {"CC": True, "SIM": True, "KLD": False}
T0 = time.time()

SPLITS = {
    "random": json.load(open(os.path.join(SPL, "kfold_seed0.json")))["folds"],
    "grouped": json.load(open(os.path.join(SPL, "groups.json")))["folds"],
    "pairsafe": json.load(open(os.path.join(SPL, "kfold_seed0_pairsafe.json")))["folds"],
}
FOLD_OF = {k: {str(i): f for f, ids in enumerate(v) for i in ids} for k, v in SPLITS.items()}
IDS = sorted(FOLD_OF["random"], key=int)
assert len(IDS) == 1000


def jload(name):
    return json.load(open(os.path.join(RES, name)))


# ------------------------------------------------------------------------------------ loading
def load_rows(fname, split="random", seed=None, fields=None, flt=None):
    """-> dict id -> {fold, CC, SIM, KLD}; fields maps CC/SIM/KLD to other keys (e.g. CC_nosigma)."""
    d = jload(fname)
    fields = fields or {m: m for m in METRICS}
    out = {}
    for r in d:
        if seed is not None and r.get("seed", 0) != seed:
            continue
        if flt is not None and not flt(r):
            continue
        i = str(r["id"])
        assert i not in out, (fname, i)
        out[i] = {"fold": int(r["fold"]), **{m: float(r[fields[m]]) for m in METRICS}}
    return out


def check_folds(rows, split, name):
    bad = [i for i, r in rows.items() if FOLD_OF[split][i] != r["fold"]]
    assert len(rows) == 1000, (name, len(rows))
    assert not bad, (name, "fold mismatch", bad[:5])


def seed_avg(list_rows):
    out = {}
    for i in list_rows[0]:
        out[i] = {"fold": list_rows[0][i]["fold"],
                  **{m: float(np.mean([r[i][m] for r in list_rows])) for m in METRICS}}
    return out


def vec(rows, m, ids=None):
    ids = ids or IDS
    return np.array([rows[i][m] for i in ids])


def fold_means(rows, m, nf=5):
    acc = [[] for _ in range(nf)]
    for i, r in rows.items():
        acc[r["fold"]].append(r[m])
    return np.array([np.mean(a) for a in acc]), np.array([len(a) for a in acc])


def describe(rows):
    o = {}
    for m in METRICS:
        fm, fn = fold_means(rows, m)
        o[m] = {"mean": float(fm.mean()), "sd_folds_ddof1": float(fm.std(ddof=1)), "sd_folds_ddof0": float(fm.std()),
                "per_fold": [float(x) for x in fm], "mean_over_images": float(vec(rows, m).mean())}
    o["fold_sizes"] = [int(x) for x in fn]
    return o


# ------------------------------------------------------------------------------------ catalogue
R = {}          # name -> dict(rows=..., meta=...)


def add(name, rows, split="random", **meta):
    check_folds(rows, split, name)
    R[name] = {"rows": rows, "split": split, **meta}




def native_means(fname, seed=None, suffix="_512"):
    d = jload(fname)
    vals = {m: [] for m in METRICS}
    for r in d:
        if seed is not None and r.get("seed", 0) != seed:
            continue
        for m in METRICS:
            k = m + suffix
            if k in r:
                vals[m].append(r[k])
    return {m: float(np.mean(v)) for m, v in vals.items() if v}


# ---- ours (random folds, scratch and warm, 3 seeds each)
for init in ("scratch", "warm"):
    for s in range(3):
        nm = f"ours_{init}_s{s}"
        sm = jload(f"{nm}_summary.json")
        add(nm, load_rows(f"{nm}_per_image.json"), group="ours", family="ours", seed=s,
            calib="200 maps per fold for sigma (subset of the training maps); centre bias learned jointly, no fitted prior"
                  + ("; warm-start readout trained on UEyes (0 ADD1000 maps)" if init == "warm" else ""),
            calib_n=200, train_n=800, trainable=sm.get("trainable_params", 3721089),
            wall_s=sm.get("wall_clock_s"), device=sm.get("device"),
            label=f"Ours, {'from scratch' if init == 'scratch' else 'warm start'}, seed {s}")
R["ours_scratch_seedavg"] = {"rows": seed_avg([R[f"ours_scratch_s{s}"]["rows"] for s in range(3)]), "split": "random",
                             "group": "ours", "family": "ours", "label": "Ours, from scratch, mean of 3 seeds per image",
                             "calib": R["ours_scratch_s0"]["calib"], "calib_n": 200, "train_n": 800, "trainable": 3721089}
R["ours_warm_seedavg"] = {"rows": seed_avg([R[f"ours_warm_s{s}"]["rows"] for s in range(3)]), "split": "random",
                          "group": "ours", "family": "ours", "label": "Ours, warm start, mean of 3 seeds per image",
                          "calib": R["ours_warm_s0"]["calib"], "calib_n": 200, "train_n": 800, "trainable": 3721089}

# ---- ablations (warm-start protocol of ablation_add1000.py)
ABL = ["abl_full", "abl_no_aspp", "abl_no_centerbias", "abl_no_finalizer", "abl_kl_only"]
for a in ABL:
    for s in range(3):
        nm = f"{a}_s{s}"
        sm = jload(f"{nm}_summary.json")
        add(nm, load_rows(f"{nm}_per_image.json"), group="ablation", family="ours", seed=s,
            calib_n=0 if a == "abl_no_finalizer" else 200, train_n=800, trainable=sm.get("trainable_params"),
            wall_s=sm.get("wall_clock_s"), label=f"{a} seed {s}")
    R[f"{a}_seedavg"] = {"rows": seed_avg([R[f"{a}_s{s}"]["rows"] for s in range(3)]), "split": "random",
                         "group": "ablation", "family": "ours", "label": f"{a}, mean of 3 seeds per image"}

# ---- DeepGaze IIE
IIE = [
    ("iie_zeroshot_uniform", "zeroshot", 0, 0, "none (uniform log prior)", "DeepGaze IIE, zero-shot, uniform prior"),
    ("iie_zeroshot_adprior", "zeroshot", 500, 0, "published 2-fold ADD1000 centre prior: 500 maps of the other half "
     "(not fold-safe under the 5-fold protocol; all 1000 maps used across the two halves)",
     "DeepGaze IIE, zero-shot, published 2-fold ad prior"),
    ("iie_zeroshot_foldprior", "calibrated", 800, 0, "centre prior from the fold's 800 training maps",
     "DeepGaze IIE, zero-shot weights, fold prior"),
    ("iie_zeroshot_foldprior_sigma", "calibrated", 800, 0, "fold prior (800 training maps) + sigma on 200 of them",
     "DeepGaze IIE, zero-shot weights, fold prior + sigma"),
    ("iie_ft_kl", "adapted", 800, 800, "fold prior (800 training maps); no post-hoc sigma",
     "DeepGaze IIE adapted, published recipe (KL, 20 ep), no sigma"),
    ("iie_ft_kl_sigma", "adapted", 800, 800, "fold prior (800) + sigma on 200 of them",
     "DeepGaze IIE adapted, published recipe (KL, 20 ep) + sigma"),
    ("iie_ft_klcc_nosigma", "adapted", 800, 800, "fold prior (800); no post-hoc sigma",
     "DeepGaze IIE adapted, our loss (KL+CC, 60 ep), no sigma"),
    ("iie_ft_klcc", "adapted", 800, 800, "fold prior (800) + sigma on 200 of them",
     "DeepGaze IIE adapted, our loss (KL+CC, 60 ep) + sigma"),
    ("iie_ft_klcc_e20_nosigma", "adapted", 800, 800, "fold prior (800); no post-hoc sigma",
     "DeepGaze IIE adapted, our loss (KL+CC, 20 ep), no sigma"),
    ("iie_ft_klcc_e20", "adapted", 800, 800, "fold prior (800) + sigma on 200 of them",
     "DeepGaze IIE adapted, our loss (KL+CC, 20 ep) + sigma [selected recipe]"),
]
for nm, kind, cn, tn, calib, label in IIE:
    sm = jload(f"{nm}_summary.json")
    add(nm, load_rows(f"{nm}_per_image.json"), group=kind, family="iie", calib=calib, calib_n=cn, train_n=tn,
        trainable=sm.get("trainable_params_trained", 0) if tn else 0, label=label,
        train_s_per_fold=sm.get("t_train_s_per_fold"), device=sm.get("device_per_fold", sm.get("device")))

# ---- DeepGaze MSDB
MS = [
    ("msdb_zeroshot_uniform", "zeroshot", 0, 0, "none (uniform log prior)", "DeepGaze MSDB, zero-shot, uniform prior", None),
    ("msdb_zeroshot_adprior", "zeroshot", 500, 0, "published 2-fold ADD1000 centre prior (500 maps of the other half; "
     "not fold-safe)", "DeepGaze MSDB, zero-shot, published 2-fold ad prior", None),
    ("msdb_zeroshot_foldprior", "calibrated", 800, 0, "fold prior (800 training maps) + sigma on 200 of them "
     "(sigma = 12, the grid edge, in every fold)", "DeepGaze MSDB, zero-shot weights, fold prior + sigma", None),
    ("msdb_zeroshot_foldprior_extgrid", "diagnostic", 800, 0, "as foldprior with the sigma grid extended past 12 "
     "(DIAGNOSTIC ONLY: grid chosen after seeing the edge effect)", "DeepGaze MSDB, fold prior + sigma, extended grid "
     "(diagnostic)", None),
    ("msdb_adapt_dsp", "adapted", 800, 800, "fold prior (800) + sigma on 200 of them",
     "DeepGaze MSDB adapted, 13 per-dataset params (KL+CC)", 0),
    ("msdb_adapt_kl", "adapted", 800, 800, "fold prior (800) + sigma on 200 of them",
     "DeepGaze MSDB adapted, head 26,463 params, KL only", 0),
]
for nm, kind, cn, tn, calib, label, seed in MS:
    sm = jload(f"{nm}_summary.json")
    add(nm, load_rows(f"{nm}_per_image.json", seed=seed), group=kind, family="msdb", calib=calib, calib_n=cn,
        train_n=tn, trainable=(13 if "dsp" in nm else 26463) if tn else 0, label=label,
        native=native_means(f"{nm}_per_image.json", seed=seed) if nm != "msdb_zeroshot_foldprior_extgrid" else None)
for s in range(3):
    add(f"msdb_adapt_s{s}", load_rows("msdb_adapt_per_image.json", seed=s), group="adapted", family="msdb", seed=s,
        calib="fold prior (800 training maps) + sigma on 200 of them", calib_n=800, train_n=800, trainable=26463,
        label=f"DeepGaze MSDB adapted, head 26,463 params, our loss (KL+CC), seed {s}",
        native=native_means("msdb_adapt_per_image.json", seed=s))
R["msdb_adapt_seedavg"] = {"rows": seed_avg([R[f"msdb_adapt_s{s}"]["rows"] for s in range(3)]), "split": "random",
                           "group": "adapted", "family": "msdb", "calib_n": 800, "train_n": 800, "trainable": 26463,
                           "calib": R["msdb_adapt_s0"]["calib"],
                           "label": "DeepGaze MSDB adapted, head, KL+CC, mean of 3 seeds per image"}

# ---- SUM
for t in range(4):
    nm = f"sum_zeroshot_t{t}"
    add(nm, load_rows(f"{nm}_per_image.json"), group="zeroshot", family="sum", calib="none", calib_n=0, train_n=0,
        trainable=0, label=f"SUM, zero-shot, token {t}" + (" (e-commerce, published)" if t == 2 else ""),
        native=native_means(f"{nm}_per_image.json", suffix="_native256"))
add("sum_calibrated", load_rows("sum_calibrated_per_image.json"), group="calibrated", family="sum",
    calib="token chosen on the fold's 800 training maps; sigma on 200 of them; no prior", calib_n=800, train_n=0,
    trainable=0, label="SUM, token + sigma calibrated on training folds",
    native=native_means("sum_calibrated_per_image.json", suffix="_native256"))
add("sum_calibrated_cb", load_rows("sum_calibrated_cb_per_image.json"), group="calibrated", family="sum",
    calib="token + centre prior from the fold's 800 training maps; sigma and mix weight on 200 of them", calib_n=800,
    train_n=0, trainable=0, label="SUM, token + sigma + centre-prior mix calibrated on training folds",
    native=native_means("sum_calibrated_cb_per_image.json", suffix="_native256"))
for nm, lab in (("sum_ft_head", "SUM loss"), ("sum_ft_head_klcc", "our loss KL+CC")):
    add(nm, load_rows(f"{nm}_per_image.json"), group="adapted", family="sum",
        calib="token chosen on training folds (800); sigma on 200 training maps", calib_n=800, train_n=800,
        trainable=36937, label=f"SUM adapted, output head only (36,937 params), {lab} + sigma [lower bound, not full FT]")
    add(nm + "_nosigma", load_rows(f"{nm}_per_image.json", fields={m: m + "_nosigma" for m in METRICS}),
        group="adapted", family="sum", calib="token chosen on training folds (800); no sigma", calib_n=800,
        train_n=800, trainable=36937, label=f"SUM adapted, output head only, {lab}, no sigma",
        native=native_means(f"{nm}_per_image.json", suffix="_native256"),
        native_note="native 256 scores stored by the SUM track are for the no-sigma variant")
# SUM full fine-tune (added 24 Nov 2026): all 57,501,196 parameters trained per fold on a cloud A10G
# (scripts/sum/ft_sum_folds.py, chunked pure-PyTorch scan), finalized by scripts/sum/finalize_ft.py.
# Only the with-sigma rows enter the main family (no _nosigma rows, to keep the family growth to 2);
# their no-sigma 384 and native-256 means are attached to the row entry as extra information.
for nm, lab in (("sum_ft", "SUM loss (-2CC + 10KL - SIM + 5MSE)"), ("sum_ft_klcc", "our loss KL+CC")):
    sm = jload(f"{nm}_summary.json")
    assert sm.get("complete") and sm.get("mode") == "full", nm
    add(nm, load_rows(f"{nm}_per_image.json"), group="adapted", family="sum",
        calib="token 2 (chosen on the fold's 800 training maps in every fold, sum_calibrated); sigma on 200 of "
              "them; no prior", calib_n=800, train_n=800, trainable=int(sm["n_trainable_params"]),
        label=f"SUM adapted, full fine-tune (all 57,501,196 params), {lab} + sigma",
        extra_means={"no_sigma_384": native_means(f"{nm}_per_image.json", suffix="_nosigma"),
                     "native256_no_sigma": native_means(f"{nm}_per_image.json", suffix="_native256"),
                     "note": "means over the 1000 images; native256 = raw 256 output vs GT resized to 256, no sigma"},
        train_info={"device": "cuda: NVIDIA A10G 24 GB (cloud GPU), chunked pure-PyTorch "
                              "selective scan",
                    "train_s_per_fold": [sm["fold_meta"][str(k)]["train_seconds"] for k in range(5)],
                    "fold_s_per_fold": [sm["fold_meta"][str(k)]["fold_seconds"] for k in range(5)],
                    "sigma_per_fold": [sm["fold_meta"][str(k)]["sigma"] for k in range(5)],
                    "covers": "train_seconds: 12-epoch end-to-end training from images (no feature cache); "
                              "fold_seconds adds model loading, sigma selection and test scoring"})

# ---- grouped and pair-safe splits
for nm, split, fam, seed in [("ours_grouped", "grouped", "ours", None), ("ours_pairsafe", "pairsafe", "ours", None),
                             ("iie_ft_klcc_grouped", "grouped", "iie", None),
                             ("iie_ft_klcc_grouped_nosigma", "grouped", "iie", None),
                             ("iie_ft_klcce20_grouped", "grouped", "iie", None),
                             ("iie_ft_klcce20_grouped_nosigma", "grouped", "iie", None),
                             ("iie_ft_klcc_pairsafe", "pairsafe", "iie", None),
                             ("iie_ft_klcc_pairsafe_nosigma", "pairsafe", "iie", None),
                             ("iie_ft_klcce20_pairsafe", "pairsafe", "iie", None),
                             ("iie_ft_klcce20_pairsafe_nosigma", "pairsafe", "iie", None),
                             ("msdb_adapt_grouped", "grouped", "msdb", 0), ("msdb_adapt_pairsafe", "pairsafe", "msdb", 0)]:
    add(nm, load_rows(f"{nm}_per_image.json", split=split, seed=seed), split=split, group=split, family=fam, label=nm)

# ------------------------------------------------------------------------------------ statistics
B_BOOT = 5000


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


def compare(ref, oth, split="random", seed=0):
    """ref, oth: row dicts on the same split, paired by id. Differences = ref - oth."""
    nf = len(SPLITS[split])
    sizes = [len(f) for f in SPLITS[split]]
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
        fa, _ = fold_means(ref, m, nf); fb, _ = fold_means(oth, m, nf)
        dk = fa - fb
        nbm, nbt, nbp = nb_test(dk, ratio)
        won = int(((dk > 0) if HIGHER[m] else (dk < 0)).sum())
        o[m] = {"mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                "boot95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0,
                "rank_biserial": rank_biserial(d), "frac_images_ref_better": float(better.mean()),
                "wilcoxon_p": wp, "wilcoxon_W": ws,
                "fold_diffs": [float(x) for x in dk], "folds_ref_better": won, "n_folds": nf,
                "nb_t": nbt, "nb_p": nbp, "nb_ratio_ntest_ntrain": ratio}
    return o


def family_tests(ref_name, others, split="random"):
    ref = R[ref_name]["rows"]
    res = {o: compare(ref, R[o]["rows"], split) for o in others}
    for m in METRICS:
        for key in ("wilcoxon_p", "nb_p"):
            adj = holm([res[o][m][key] for o in others])
            for o, pa in zip(others, adj):
                res[o][m][key + "_holm"] = float(pa)
    return res


# ---- main family: every non-ours row on the random folds (plus our other initialisation)
main_others = [n for n, r in R.items() if r["split"] == "random" and r.get("group") in
               ("zeroshot", "calibrated", "diagnostic", "adapted") and not n.endswith("seedavg")
               and not (n.startswith("msdb_adapt_s") and n != "msdb_adapt_s0")]
main_others_s0 = main_others + ["ours_warm_s0"]
main_others_avg = [("msdb_adapt_seedavg" if n == "msdb_adapt_s0" else n) for n in main_others] + ["ours_warm_seedavg"]
TESTS = {"ours_scratch_s0": family_tests("ours_scratch_s0", main_others_s0),
         "ours_scratch_seedavg": family_tests("ours_scratch_seedavg", main_others_avg)}
# warm-start reference as well (the published headline row was warm start)
TESTS["ours_warm_s0"] = family_tests("ours_warm_s0", main_others)

# ---- ablations: variant minus full, seed-averaged per image, Holm over the 4 variants
abl_res = {}
for key in ["seedavg", "s0", "s1", "s2"]:
    full = R[f"abl_full_{key}"]["rows"]
    var = [f"{a}_{key}" for a in ABL[1:]]
    res = {}
    for v in var:
        c = compare(R[v]["rows"], full)          # variant - full
        res[v] = c
    for m in METRICS:
        for k2 in ("wilcoxon_p", "nb_p"):
            adj = holm([res[v][m][k2] for v in var])
            for v, pa in zip(var, adj):
                res[v][m][k2 + "_holm"] = float(pa)
    abl_res[key] = res

# ---- grouped / pair-safe
gp = {}
for split, pairs in {
    "grouped": [("ours_grouped", "ours_scratch_s0"), ("iie_ft_klcc_grouped", "iie_ft_klcc"),
                ("iie_ft_klcce20_grouped", "iie_ft_klcc_e20"), ("msdb_adapt_grouped", "msdb_adapt_s0")],
    "pairsafe": [("ours_pairsafe", "ours_scratch_s0"), ("iie_ft_klcc_pairsafe", "iie_ft_klcc"),
                 ("iie_ft_klcce20_pairsafe", "iie_ft_klcc_e20"), ("msdb_adapt_pairsafe", "msdb_adapt_s0")]}.items():
    out = {}
    for new, base in pairs:
        a, b = R[new]["rows"], R[base]["rows"]
        o = {"row": new, "random_fold_row": base, "split_desc": describe(a), "random_desc": describe(b)}
        for m in METRICS:
            d = vec(a, m) - vec(b, m)
            w = stats.wilcoxon(vec(a, m), vec(b, m))
            o[m] = {"split_mean": float(vec(a, m).mean()), "random_mean": float(vec(b, m).mean()),
                    "diff_split_minus_random": float(d.mean()), "wilcoxon_p_per_image": float(w.pvalue)}
        out[new] = o
    # ours vs baselines inside the same split (same folds -> full paired tests)
    ref = "ours_grouped" if split == "grouped" else "ours_pairsafe"
    oth = [n for n in R if R[n]["split"] == split and n != ref]
    out["ours_vs_baselines_same_split"] = family_tests(ref, oth, split)
    gp[split] = out

# ------------------------------------------------------------------------------------ curves
NS = [25, 50, 100, 200, 400, 800]
F0 = [i for i in IDS if FOLD_OF["random"][i] == 0]


def curve_from_files(pattern_fn):
    """pattern_fn(n, r) -> dict id -> metrics (fold-0 ids)."""
    out = {}
    for n in NS:
        runs = []
        for r in range(3):
            rows = pattern_fn(n, r)
            if rows is None:
                continue
            assert sorted(rows, key=int) == sorted(F0, key=int), (n, r)
            runs.append(rows)
        if runs:
            out[n] = runs
    return out


def ours_curve(init):
    def fn(n, r):
        p = os.path.join(RES, f"curve_ours_{init}_n{n}_r{r}_per_image.json")
        if not os.path.exists(p):
            return None
        return {str(x["id"]): {m: x[m] for m in METRICS} for x in json.load(open(p))}
    return curve_from_files(fn)


def multi_curve(fname, rkey="resample"):
    d = jload(fname)
    def fn(n, r):
        rows = {str(x["id"]): {m: x[m] for m in METRICS} for x in d if x["n"] == n and x[rkey] == r}
        return rows or None
    return curve_from_files(fn)


CURVES = {
    "ours_scratch": ours_curve("scratch"),
    "ours_warm": ours_curve("warm"),
    "iie_adapt_klcc_e20": multi_curve("iie_curve_klcce20_per_image.json"),
    "iie_adapt_klcc_e20_nosigma": multi_curve("iie_curve_klcce20_nosigma_per_image.json"),
    "iie_adapt_klcc_e60_partial": multi_curve("iie_curve_klcc_per_image.json"),
    "msdb_adapt_head": multi_curve("msdb_adapt_curves_per_image.json"),
    "msdb_adapt_dsp": multi_curve("msdb_adapt_dsp_curves_per_image.json"),
    # full SUM fine-tune curve (added 24 Nov 2026; cloud A10G, scripts/sum/ft_sum_folds.py --curve, finalized by
    # scripts/sum/finalize_curve.py): r = 0, 1, 2 at n <= 200, r = 0 only at n = 400, n = 800 = sum_ft fold 0
    "sum_ft_full": multi_curve("sum_ft_curve_per_image.json"),
    "sum_ft_full_nosigma": multi_curve("sum_ft_curve_nosigma_per_image.json"),
}
CURVE_TRAINABLE = {"ours_scratch": 3721089, "ours_warm": 3721089, "iie_adapt_klcc_e20": 3059160,
                   "iie_adapt_klcc_e20_nosigma": 3059160, "iie_adapt_klcc_e60_partial": 3059160,
                   "msdb_adapt_head": 26463, "msdb_adapt_dsp": 13,
                   "sum_ft_full": 57501196, "sum_ft_full_nosigma": 57501196}
# resample ids actually present per n (curve_from_files keeps runs in r order and drops missing r)
SUM_CURVE_R = {}
for _x in jload("sum_ft_curve_per_image.json"):
    SUM_CURVE_R.setdefault(_x["n"], set()).add(_x["resample"])
SUM_CURVE_R = {n: sorted(v) for n, v in sorted(SUM_CURVE_R.items())}
for _n, _rs in SUM_CURVE_R.items():
    assert _rs == list(range(len(_rs))), ("SUM curve resamples must be a prefix 0..k-1", _n, _rs)
    assert len(CURVES["sum_ft_full"][_n]) == len(_rs) == len(CURVES["sum_ft_full_nosigma"][_n])

# check: identical training subsets across families (ours/msdb store train ids; IIE uses the same rule)
subset_check = {}
idx = np.random.default_rng(0).permutation(1000)
pool = [IDS[j] for j in idx[200:]]
assert sorted([IDS[j] for j in idx[:200]], key=int) == sorted(F0, key=int)
ms_runs = {(r["n"], r["resample"]): [str(x) for x in r["train_ids"]] for r in jload("msdb_adapt_curves_summary.json")["runs"]}
mismatch = 0
for n in NS:
    for r in range(3):
        rule = [str(x) for x in np.random.default_rng(1000 * n + r).choice(np.array(pool), size=n, replace=False)]
        for init in ("scratch", "warm"):
            ti = [str(x) for x in jload(f"curve_ours_{init}_n{n}_r{r}_summary.json")["train_ids"]]
            mismatch += int(ti != rule)
        mismatch += int(ms_runs[(n, r)] != rule)
subset_check = {"rule": "default_rng(1000*n + r).choice(default_rng(0).permutation(1000)[200:] mapped to sorted ids, n)",
                "ours_and_msdb_train_ids_equal_rule_for_all_n_r": mismatch == 0, "n_mismatches": mismatch,
                "iie": "ft_curves.py applies the same rule in code (train ids not stored)"}
# SUM curve (added 24 Nov 2026): stored train_ids of every run vs the same rule (n = 800: as a set, because the
# n = 800 point is the k-fold sum_ft fold 0, which lists the 800 pool ids in fold order)
_sum_mis, _sum_set_eq_800 = 0, None
for _run in jload("sum_ft_curve_summary.json")["runs"]:
    _rule = [str(x) for x in np.random.default_rng(1000 * _run["n"] + _run["resample"]).choice(
        np.array(pool), size=_run["n"], replace=False)]
    _ti = [str(x) for x in _run["train_ids"]]
    if _run["n"] == 800:
        _sum_set_eq_800 = sorted(_ti, key=int) == sorted(_rule, key=int)
        _sum_mis += int(not _sum_set_eq_800)
    else:
        _sum_mis += int(_ti != _rule)
subset_check["sum_ft_full_train_ids_equal_rule"] = _sum_mis == 0
subset_check["sum_ft_full_n_mismatches"] = _sum_mis
subset_check["sum_ft_full_n800_equal_as_set"] = _sum_set_eq_800

curve_tab = {}
for fam, cur in CURVES.items():
    t = {}
    for n, runs in cur.items():
        per_run = {m: [float(np.mean([rr[i][m] for i in F0])) for rr in runs] for m in METRICS}
        t[n] = {"n_resamples": len(runs), "trainable": CURVE_TRAINABLE[fam],
                **{f"mean_{m}": float(np.mean(per_run[m])) for m in METRICS},
                **{f"sd_{m}_ddof0": float(np.std(per_run[m])) for m in METRICS},
                **{f"sd_{m}_ddof1": float(np.std(per_run[m], ddof=1)) if len(runs) > 1 else None for m in METRICS},
                "runs_CC": per_run["CC"]}
    curve_tab[fam] = t

# n = 0 references on the SAME fold-0 test set
n0 = {}
for nm, note in [("iie_zeroshot_uniform", "0 maps"), ("iie_zeroshot_adprior", "2-fold prior: 500 maps of the other half "
                 "(includes other fold-0 test images, never the scored image)"),
                 ("iie_zeroshot_foldprior", "prior from the 800 pool maps"),
                 ("iie_zeroshot_foldprior_sigma", "prior from 800 pool maps + sigma on 200"),
                 ("msdb_zeroshot_uniform", "0 maps"), ("msdb_zeroshot_adprior", "2-fold prior (500 maps, see IIE)"),
                 ("msdb_zeroshot_foldprior", "prior from 800 pool maps + sigma on 200"),
                 ("sum_zeroshot_t2", "0 maps (published token)"), ("sum_zeroshot_t0", "0 maps"),
                 ("sum_calibrated", "token + sigma on the 800 pool maps"),
                 ("sum_calibrated_cb", "token + prior + sigma on the 800 pool maps")]:
    rows = R[nm]["rows"]
    n0[nm] = {"note": note, **{m: float(np.mean([rows[i][m] for i in F0])) for m in METRICS}}
ws = jload("ours_warm_s0_summary.json")["folds_detail"][0]
n0["ours_warm_not_finetuned"] = {"note": "warm-start readout before ADD1000 training, sigma chosen on 200 pool maps "
                                         "(fold-level only, from ours_warm_s0 summary)",
                                 "CC": ws["start_CC"], "SIM": ws["start_SIM"], "KLD": ws["start_KLD"]}

# paired tests at each n (fold-0 test images; per-image values averaged over the 3 resamples; the same
# training subset is used by every family for a given (n, r), so resamples are paired too)
curve_tests = {}
for refn in ("ours_scratch", "ours_warm"):
    for othn in ("iie_adapt_klcc_e20", "msdb_adapt_head", "msdb_adapt_dsp"):
        key = f"{refn}_vs_{othn}"
        curve_tests[key] = {}
        for n in NS:
            if n not in CURVES[othn] or n not in CURVES[refn]:
                continue
            o = {}
            for m in METRICS:
                a = np.array([np.mean([rr[i][m] for rr in CURVES[refn][n]]) for i in F0])
                b = np.array([np.mean([rr[i][m] for rr in CURVES[othn][n]]) for i in F0])
                d = a - b
                ra = [np.mean([rr[i][m] for i in F0]) for rr in CURVES[refn][n]]
                rb = [np.mean([rr[i][m] for i in F0]) for rr in CURVES[othn][n]]
                won = sum(((x > y) if HIGHER[m] else (x < y)) for x, y in zip(ra, rb))
                o[m] = {"mean_diff": float(d.mean()), "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue),
                        "frac_images_ref_better": float(((d > 0) if HIGHER[m] else (d < 0)).mean()),
                        "resamples_ref_better": int(won), "n_resamples": len(ra)}
            curve_tests[key][n] = o
# ours minus full SUM fine-tune (added 24 Nov 2026), same construction as above. SUM has fewer resamples at
# n = 400 (r = 0) and n = 800 (sum_ft fold 0), so per-image values average the resamples each family has, and
# 'resamples_ref_better' pairs resample r of ours with resample r of SUM (same training subset) where both exist.
for refn in ("ours_scratch", "ours_warm"):
    key = f"{refn}_vs_sum_ft_full"
    curve_tests[key] = {}
    for n in NS:
        if n not in CURVES["sum_ft_full"] or n not in CURVES[refn]:
            continue
        rs = SUM_CURVE_R[n]
        o = {}
        for m in METRICS:
            a = np.array([np.mean([rr[i][m] for rr in CURVES[refn][n]]) for i in F0])
            b = np.array([np.mean([rr[i][m] for rr in CURVES["sum_ft_full"][n]]) for i in F0])
            d = a - b
            ra = [np.mean([CURVES[refn][n][r][i][m] for i in F0]) for r in rs]
            rb = [np.mean([CURVES["sum_ft_full"][n][r][i][m] for i in F0]) for r in rs]
            won = sum(((x > y) if HIGHER[m] else (x < y)) for x, y in zip(ra, rb))
            o[m] = {"mean_diff": float(d.mean()), "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue),
                    "frac_images_ref_better": float(((d > 0) if HIGHER[m] else (d < 0)).mean()),
                    "resamples_ref_better": int(won), "n_resamples": len(rs),
                    "n_resamples_ref_averaged": len(CURVES[refn][n]), "n_resamples_sum_averaged": len(rs)}
        curve_tests[key][n] = o
# Holm within each comparison across the 6 budgets and 3 metrics
for key, byn in curve_tests.items():
    flat = [(n, m) for n in byn for m in METRICS]
    adj = holm([byn[n][m]["wilcoxon_p"] for n, m in flat])
    for (n, m), pa in zip(flat, adj):
        byn[n][m]["wilcoxon_p_holm"] = float(pa)

# ------------------------------------------------------------------------------------ main table
def row_entry(name):
    r = R[name]
    e = {"row": name, "label": r.get("label", name), "group": r.get("group"), "family": r.get("family"),
         "split": r["split"], "calibration_maps": r.get("calib_n"), "calibration_note": r.get("calib"),
         "training_maps": r.get("train_n"), "trainable_params": r.get("trainable"),
         "per_image_file": None}
    e.update({"stats_384": describe(r["rows"])})
    if r.get("native"):
        e["native_resolution_means"] = r["native"]
    if r.get("native_note"):
        e["native_resolution_note"] = r["native_note"]
    if r.get("extra_means"):          # only rows added after 23 Nov 2026 carry these keys
        e["extra_means"] = r["extra_means"]
    if r.get("train_info"):
        e["training_wall_clock"] = r["train_info"]
    return e


table = []
order = (["iie_zeroshot_uniform", "iie_zeroshot_adprior", "msdb_zeroshot_uniform", "msdb_zeroshot_adprior"]
         + [f"sum_zeroshot_t{t}" for t in range(4)]
         + ["iie_zeroshot_foldprior", "iie_zeroshot_foldprior_sigma", "msdb_zeroshot_foldprior",
            "msdb_zeroshot_foldprior_extgrid", "sum_calibrated", "sum_calibrated_cb"]
         + ["iie_ft_kl", "iie_ft_kl_sigma", "iie_ft_klcc_nosigma", "iie_ft_klcc", "iie_ft_klcc_e20_nosigma",
            "iie_ft_klcc_e20", "sum_ft_head_nosigma", "sum_ft_head", "sum_ft_head_klcc_nosigma", "sum_ft_head_klcc",
            "sum_ft", "sum_ft_klcc", "msdb_adapt_dsp", "msdb_adapt_kl", "msdb_adapt_s0", "msdb_adapt_s1", "msdb_adapt_s2", "msdb_adapt_seedavg"]
         + [f"ours_warm_s{s}" for s in range(3)] + ["ours_warm_seedavg"]
         + [f"ours_scratch_s{s}" for s in range(3)] + ["ours_scratch_seedavg"]
         + [f"{a}_s{s}" for a in ABL for s in range(3)] + [f"{a}_seedavg" for a in ABL]
         + [n for n in R if R[n]["split"] in ("grouped", "pairsafe")])
assert len(order) == len(set(order)) and set(order) == set(R), set(R) ^ set(order)
for n in order:
    e = row_entry(n)
    base = n.replace("_seedavg", "")
    for cand in (f"{n}_per_image.json", f"{base}_per_image.json", f"{n.rsplit('_s', 1)[0]}_per_image.json",
                 f"{n.replace('_nosigma', '')}_per_image.json"):
        if os.path.exists(os.path.join(RES, cand)):
            e["per_image_file"] = os.path.join(RES, cand); break
    if e["per_image_file"] is None and n.endswith("_seedavg"):
        e["per_image_file"] = [os.path.join(RES, f"{base}_s{s}_per_image.json") for s in range(3)]
        e["derived"] = "per-image mean over seeds 0, 1, 2"
    table.append(e)

# multi-seed aggregates (mean +/- SD over seed means)
seed_groups = {"ours_scratch": [f"ours_scratch_s{s}" for s in range(3)],
               "ours_warm": [f"ours_warm_s{s}" for s in range(3)],
               "msdb_adapt": [f"msdb_adapt_s{s}" for s in range(3)],
               **{a: [f"{a}_s{s}" for s in range(3)] for a in ABL}}
seed_agg = {}
for g, names in seed_groups.items():
    o = {}
    for m in METRICS:
        means = [vec(R[n]["rows"], m).mean() for n in names]
        o[m] = {"seed_means": [float(x) for x in means], "mean_over_seeds": float(np.mean(means)),
                "sd_over_seeds_ddof1": float(np.std(means, ddof=1)), "sd_over_seeds_ddof0": float(np.std(means))}
    seed_agg[g] = o

# cross-checks against the track-reported numbers
xcheck = {}
for n in ["ours_scratch_s0", "ours_warm_s0", "iie_ft_kl", "iie_ft_klcc_e20", "iie_zeroshot_adprior", "msdb_adapt_s0",
          "msdb_zeroshot_adprior", "sum_zeroshot_t2", "sum_ft_head", "ours_grouped", "sum_ft", "sum_ft_klcc"]:
    xcheck[n] = {m: round(float(vec(R[n]["rows"], m).mean()), 5) for m in METRICS}

# fold-level-only reference rows (no per-image file exists)
fold_only = {}
wsm = jload("ours_warm_s0_summary.json")["folds_detail"]
fold_only["ours_warm_not_finetuned"] = {
    "label": "Ours, warm-start readout (UEyes), not trained on ADD1000; sigma chosen on 200 training maps",
    "calibration_maps": 200, "training_maps": 0,
    "sigma_per_fold": [f["start_sigma"] for f in wsm],
    **{m: {"per_fold": [f[f"start_{m}"] for f in wsm], "mean": float(np.mean([f[f"start_{m}"] for f in wsm])),
           "sd_folds_ddof1": float(np.std([f[f"start_{m}"] for f in wsm], ddof=1))} for m in METRICS},
    "source": "ours_warm_s0_summary.json folds_detail start_* (no per-image values, so no paired tests)"}

# near-duplicate summary
nd = json.load(open(os.path.join(SPL, "near_dup_pairs.json")))
near_dup = {
    "primary_definition": nd["primary"]["definition"],
    "primary_n_pairs": nd["primary"]["n_pairs"], "primary_n_cross_fold_seed0": nd["primary"]["n_cross_fold_seed0"],
    "primary_pairs": [{k: p.get(k) for k in ("a", "b", "meanpatch_cos", "phash256_bits", "fold_a", "fold_b", "visual")}
                      for p in nd["primary"]["pairs"]],
    "extended_definition": nd["extended"]["definition"], "extended_n_pairs": nd["extended"]["n_pairs"],
    "extended_n_cross_fold_seed0": nd["extended"]["n_cross_fold_seed0"],
    "phash_reproduction": {"n_pairs_phash256_le26": nd["hash_reproduction_attempt"]["n_pairs"],
                           "reproduced_earlier_5_pairs": nd["hash_reproduction_attempt"].get("reproduced_earlier_count"),
                           "scan_counts_at_26_bits": nd["hash_reproduction_attempt"].get("scan_counts_at_26_bits")},
    "pairsafe_split": {"components_constrained": json.load(open(os.path.join(SPL, "kfold_seed0_pairsafe.json")))["components_constrained"],
                       "n_ids_changed_fold": json.load(open(os.path.join(SPL, "kfold_seed0_pairsafe.json")))["n_ids_changed_fold"]},
    "effect_of_pairsafe_folds_CC": {k: v["CC"]["diff_split_minus_random"] for k, v in gp["pairsafe"].items()
                                    if k != "ours_vs_baselines_same_split"},
    "source": os.path.join(SPL, "near_dup_pairs.json"),
}

out = {"description": "Unified ADD1000 results (384 x 384 scoring; GT bilinear 384; clip at 0 and divide by sum). "
                      "SD over folds with ddof=1 unless stated; per-image files are the sources.",
       "protocol": {"folds": os.path.join(SPL, "kfold_seed0.json"), "grouped_folds": os.path.join(SPL, "groups.json"),
                    "pairsafe_folds": os.path.join(SPL, "kfold_seed0_pairsafe.json"),
                    "metrics": "CC, SIM, KLD (eps 1e-8) at 384", "n_images": 1000},
       "rows": table, "fold_level_only_rows": fold_only, "seed_aggregates": seed_agg, "cross_check_means": xcheck,
       "near_duplicates": near_dup,
       "tests_vs_ours": TESTS, "ablations": abl_res, "grouped_pairsafe": gp,
       "curves": {"table": curve_tab, "n0_fold0": n0, "paired_tests": curve_tests, "subset_check": subset_check,
                  "test_set": "fold 0 of kfold_seed0.json (200 ids) = sample_efficiency.py held-out set",
                  "sum_adapted_curve": "full fine-tune of all 57,501,196 SUM parameters (SUM loss, 12 epochs, token 2, "
                                       "sigma on the same n maps), table rows sum_ft_full / sum_ft_full_nosigma; "
                                       "source results/sum_ft_curve_summary.json (added 24 Nov 2026)",
                  "sum_ft_full_resamples": {str(n): rs for n, rs in SUM_CURVE_R.items()},
                  "sum_ft_full_caveats": [
                      "n = 400: one resample (r = 0) only; budget-driven",
                      "n = 800: not a curve run but the k-fold row sum_ft, fold 0: same 800 training ids, but sigma "
                      "chosen on 200 of the 800 training maps (curve runs: all n), data order default_rng(0) over "
                      "the pool and torch seed 0; one resample; test CC 0.9203 with sigma vs 0.9201 without",
                      "trained on a cloud NVIDIA A10G; every other curve was trained locally on an Apple M4 Pro (CPU or MPS: MSDB head on MPS, MSDB 13 params on CPU, ours 30 of 36 runs on CPU and 6 on MPS, IIE by CPU/MPS pool workers)"],
                  "sum_ft_full_n800_note": jload("sum_ft_curve_summary.json")["n800_note"]},
       "stats_notes": __doc__,
       "wall_clock_s": None, "command": "python -I " + " ".join(sys.argv)}
out["wall_clock_s"] = time.time() - T0


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


json.dump(clean(out), open(os.path.join(RES, "main_table.json"), "w"), indent=1)
print("wrote main_table.json", len(table), "rows", f"{time.time() - T0:.1f}s")
print(json.dumps(xcheck, indent=0))
print(json.dumps(subset_check))
