"""
Accuracy-cost frontier of every ADD1000-adapted model (KBS letter), compiled from saved files only.

  python -I compile_frontier.py     # writes results/frontier.json, results/FRONTIER.md, results/frontier.png

Nothing is trained or re-scored here. Accuracy comes from the per-image files of each row (same folds
splits/kfold_seed0.json, same 384 x 384 clip-and-sum scoring, mean +/- SD over the 5 folds, ddof = 1).
Costs come from:
  * results/cost_table.json (quiet machine): ours B448, DeepGaze IIE, DeepGaze MSDB, SUM;
  * results/cost_raw/frontier_quiet_<variant>_<dev>_p<k>.json: quiet-machine re-measurement of the Track A
    backbones (L448, msB, B644, Lms; B448 measured first and last as the same-session check), made with
    scripts/trackA/cost_bench_a.py unchanged through scripts/frontier/run_cost_quiet.sh (same protocol:
    batch 1, float32, 6 threads, 3 warm-ups, median of 30). These replace Track A's contended estimates;
    the estimates and the contended raw numbers are kept in frontier.json for comparison;
  * results/trackA_cost.json: GFLOPs and parameter counts of the Track A backbones (deterministic).
Distilled students have exactly the inference graph of the matching non-distilled model (the teacher is
used only to make training targets), so they take that model's cost.

Pareto frontier: on (CPU latency, mean CC), over one entry per configuration (seed 0 or single run; seed
averages and extra seeds excluded). A model is on the frontier if no other model is at most as slow and at
least as accurate, with one of the two strict. The same rule is applied on (MPS latency, CC) and (GFLOPs, CC).
24 Nov 2026: the SUM full fine-tune rows (sum_ft = SUM's own loss, primary; sum_ft_klcc = KL+CC loss,
secondary) replace the former placeholder; both are frontier candidates with SUM's cost from cost_table.json
(inference cost does not change with fine-tuning). Only quiet-machine CPU latencies are used or quoted;
Track A's contended estimates are kept in frontier.json for audit only.

"Fast" (declared here before looking at the paired tests): CPU latency <= 1/10 of adapted MSDB's
(>= 10x faster). The best fast model is the fast model with the highest mean CC.

Paired tests: copied from scripts/compile/compile_results.py (Wilcoxon signed-rank, two-sided,
zero_method='wilcox', on the 1000 out-of-fold images; Nadeau-Bengio corrected t on the 5 fold differences,
df 4, n_test/n_train = 0.25; 5000-sample bootstrap 95% CI of the mean per-image difference, rng seed 0).
Differences = fast model minus reference (for KLD negative = fast model better). Holm family (primary):
{best fast model, best model at our base inference cost} x {msdb_adapt_s0, ours_scratch_s0} = 4 comparisons,
corrected separately per metric and per test. Secondary family: the same two models against the seed
averages (msdb_adapt seeds 0-2, ours_scratch seeds 0-2).

SUM full fine-tune tests (added 24 Nov 2026; differences = sum_ft minus reference): primary Holm family =
sum_ft vs {msdb_adapt_s0, ours_scratch_s0, distill_pitt, distill_pitt_L, sum_ft_head, sum_ft_klcc} = 6
comparisons, Holm per metric (CC, SIM, KLD) and per test (Wilcoxon, NB). Secondary: sum_ft vs
msdb_adapt_seedavg, a single comparison (no correction). Same Wilcoxon and NB as above; the bootstrap CI of
these tests uses the letter's stated convention, 10,000 resamples with a fresh numpy default_rng(20261121)
per comparison (drawn in metric order CC, SIM, KLD); the 5,000-resample rng-0 CI of the older sections is
also stored for comparison.

Training cost: per-fold training wall-clock with the hardware it ran on, read from the row summaries (or
cost_table.json training_adaptation where it already lists the row). Hardware differs between rows (Apple
M4 Pro CPU, Apple M4 Pro GPU via MPS, cloud NVIDIA A10G), so these times are not comparable across hardware.

Zero-shot to adapted gains: read from results/main_table.json (rows already compiled there).
"""
import sys
sys.dont_write_bytecode = True
import os, json, glob, math, time
import numpy as np
from scipy import stats

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RES = os.path.join(KB, "results")
RAW = os.path.join(RES, "cost_raw")
METRICS = ("CC", "SIM", "KLD")
HIGHER = {"CC": True, "SIM": True, "KLD": False}
FOLDS = json.load(open(os.path.join(KB, "splits", "kfold_seed0.json")))["folds"]
FOLD_OF = {str(i): f for f, ids in enumerate(FOLDS) for i in ids}
IDS = sorted(FOLD_OF, key=int)
assert len(IDS) == 1000
B_BOOT = 5000
T0 = time.time()
GB = 1024 ** 3


# ------------------------------------------------------------------------------------ accuracy
def load(fname, seed=None, fields=None):
    fields = fields or {m: m for m in METRICS}
    out = {}
    for r in json.load(open(os.path.join(RES, fname))):
        if seed is not None and r.get("seed", 0) != seed:
            continue
        i = str(r["id"]); assert i not in out, (fname, i)
        out[i] = {"fold": int(r["fold"]), **{m: float(r[fields[m]]) for m in METRICS}}
    assert len(out) == 1000, (fname, len(out))
    bad = [i for i, r in out.items() if FOLD_OF[i] != r["fold"]]
    assert not bad, (fname, "fold mismatch", bad[:5])
    return out


def seed_avg(lst):
    return {i: {"fold": lst[0][i]["fold"], **{m: float(np.mean([r[i][m] for r in lst])) for m in METRICS}}
            for i in lst[0]}


def vec(rows, m):
    return np.array([rows[i][m] for i in IDS])


def fold_means(rows, m):
    acc = [[] for _ in range(5)]
    for r in rows.values():
        acc[r["fold"]].append(r[m])
    return np.array([np.mean(a) for a in acc])


def describe(rows):
    o = {}
    for m in METRICS:
        fm = fold_means(rows, m)
        o[m] = {"mean": float(fm.mean()), "sd_folds_ddof1": float(fm.std(ddof=1)), "per_fold": [float(x) for x in fm]}
    return o


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
    """a - b, paired by image id."""
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
                "rank_biserial": rank_biserial(d), "frac_images_model_better": float(better.mean()),
                "wilcoxon_p": wp, "fold_diffs": [float(x) for x in dk],
                "folds_model_better": int(((dk > 0) if HIGHER[m] else (dk < 0)).sum()), "nb_t": nbt, "nb_p": nbp}
    return o


# ------------------------------------------------------------------------------------ costs
CT = json.load(open(os.path.join(RES, "cost_table.json")))["inference"]
TA = json.load(open(os.path.join(RES, "trackA_cost.json")))


def quiet(variant, dev):
    """All quiet-session passes of a Track A variant on a device."""
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, f"frontier_quiet_{variant}_{dev}_p*.json"))):
        d = json.load(open(f))
        e = {"file": os.path.basename(f), "median_s": d["latency_s"]["median"],
             "iqr_s": [d["latency_s"]["q25"], d["latency_s"]["q75"]], "runs": d["runs"], "threads": d["threads"],
             "latency_incl_finalizer_s": d.get("latency_incl_finalizer_s_median"), "GFLOPs": d.get("GFLOPs"),
             "params_total": d.get("params_total")}
        if dev == "cpu":
            e["peak_rss_GB"] = d["peak_rss_during_inference_GB"]; e["rss_after_load_GB"] = d["rss_after_load_GB"]
        else:
            e["mps_driver_allocated_max_GB"] = d.get("mps_driver_allocated_max_GB")
        out.append(e)
    return out


B448_CT = CT["ours"]
B448_Q = {dev: quiet("B448", dev) for dev in ("cpu", "mps")}
for dev in ("cpu", "mps"):
    assert B448_Q[dev], f"quiet B448 {dev} measurement missing"
B448_Q_CPU = float(np.mean([e["median_s"] for e in B448_Q["cpu"]]))
B448_Q_MPS = float(np.mean([e["median_s"] for e in B448_Q["mps"]]))
B448_Q_RSS = float(np.mean([e["peak_rss_GB"] for e in B448_Q["cpu"]]))
# the Track A bench loads the backbone through a different path than cost_bench.py's 'ours' branch: its
# loaded-model RSS is lower by a constant amount (same activation increment). Align to cost_table.json.
RSS_OFFSET = B448_CT["cpu_peak_rss_GB"] - B448_Q_RSS
SESSION = {"cpu_B448_quiet_median_s": B448_Q_CPU, "cpu_B448_cost_table_s": B448_CT["cpu_latency_median_s"],
           "cpu_session_factor": B448_Q_CPU / B448_CT["cpu_latency_median_s"],
           "mps_B448_quiet_median_s": B448_Q_MPS, "mps_B448_cost_table_s": B448_CT["mps_latency_median_s"],
           "mps_session_factor": B448_Q_MPS / B448_CT["mps_latency_median_s"],
           "cpu_rss_B448_trackA_bench_GB": B448_Q_RSS, "cpu_rss_B448_cost_table_GB": B448_CT["cpu_peak_rss_GB"],
           "cpu_rss_alignment_offset_GB": RSS_OFFSET,
           "B448_passes": B448_Q}


def cost_ct(key, trainable, note):
    c = CT[key]
    return {"source": "cost_table.json (quiet machine)", "GFLOPs": c["GFLOPs"], "params_total": c["params_total"],
            "params_frozen": 0 if trainable == c["params_total"] else c["params_backbone_frozen"],
            "params_trainable": trainable,
            "cpu_latency_s": c["cpu_latency_median_s"], "cpu_latency_iqr_s": c["cpu_latency_iqr_s"],
            "cpu_latency_kind": "measured, quiet machine",
            "mps_latency_s": c["mps_latency_median_s"], "mps_latency_iqr_s": c["mps_latency_iqr_s"],
            "mps_latency_kind": "measured",
            "cpu_peak_rss_GB": c["cpu_peak_rss_GB"], "cpu_peak_rss_kind": "measured (cost_bench.py)",
            "mps_driver_mem_GB": c["mps_driver_allocated_max_GB"], "note": note}


def cost_ta(variant, trainable, note):
    v = TA["variants"][variant]
    qc, qm = quiet(variant, "cpu"), quiet(variant, "mps")
    assert qc and qm, f"quiet measurement missing for {variant}"
    cpu = float(np.median([e["median_s"] for e in qc])); mps = float(np.median([e["median_s"] for e in qm]))
    rss_raw = float(np.max([e["peak_rss_GB"] for e in qc]))
    contended = {k: [e["median_s"] for e in v["latency"].get(k, [])] for k in ("cpu_run1", "cpu_run2", "mps_run1")}
    assert abs(qc[0]["GFLOPs"] - v["GFLOPs"]) < 1e-6 and qc[0]["params_total"] == v["params_total"]
    return {"source": "GFLOPs/params: trackA_cost.json; latency/memory: quiet re-measurement "
                      "(results/cost_raw/frontier_quiet_*)",
            "GFLOPs": v["GFLOPs"], "params_total": v["params_total"], "params_frozen": v["params_backbone_frozen"],
            "params_trainable": trainable,
            "cpu_latency_s": cpu, "cpu_latency_iqr_s": qc[0]["iqr_s"], "cpu_latency_kind": "measured, quiet machine",
            "mps_latency_s": mps, "mps_latency_iqr_s": qm[0]["iqr_s"], "mps_latency_kind": "measured",
            "cpu_peak_rss_GB": rss_raw + RSS_OFFSET,
            "cpu_peak_rss_kind": f"Track A bench {rss_raw:.2f} GB + {RSS_OFFSET:.2f} GB loading-path offset "
                                 f"(B448: {B448_Q_RSS:.2f} GB in this bench vs {B448_CT['cpu_peak_rss_GB']:.2f} GB "
                                 "in cost_table.json)",
            "cpu_peak_rss_GB_trackA_bench_raw": rss_raw,
            "mps_driver_mem_GB": float(np.max([e["mps_driver_allocated_max_GB"] for e in qm])),
            "quiet_passes": {"cpu": qc, "mps": qm},
            "trackA_contended": {"cpu_latency_normalised_estimate_s": v["cpu_latency_normalised_estimate_s"],
                                 "mps_latency_normalised_estimate_s": v["mps_latency_normalised_estimate_s"],
                                 "raw_contended_medians_s": contended},
            "note": note}


def cost_b448(trainable, note):
    c = cost_ct("ours", trainable, note)
    c["quiet_session_check_cpu_s"] = B448_Q_CPU; c["quiet_session_check_mps_s"] = B448_Q_MPS
    return c


TRACKA_CONTENDED_B448 = {"cpu_latency_normalised_estimate_s": B448_CT["cpu_latency_median_s"],
                         "mps_latency_normalised_estimate_s": B448_CT["mps_latency_median_s"],
                         "raw_contended_medians_s": {k: [e["median_s"] for e in TA["variants"]["B448"]["latency"][k]]
                                                     for k in ("cpu_run1", "cpu_run2", "mps_run1")}}

# ------------------------------------------------------------------------------------ catalogue
M_OURS = "800 (train 800; sigma on 200 of them; centre bias learned jointly, no fitted prior)"
M_MSDB = "800 (train 800; centre prior from the 800; sigma on 200 of them)"
M_PITT = ("800 (student GT 800; teacher = adapted MSDB trained on the same 800; sigma on 200 of them) "
          "+ 2,499 unlabeled Pitt ads with teacher maps")
CAT = [
    # name, label, family, kind, loader, cost, maps_n, maps_desc, extra_unlabeled, primary(frontier candidate)
    ("ours_scratch_s0", "Ours B448 (DINOv2-B 448 + readout), scratch, seed 0", "ours", "ours",
     lambda: load("ours_scratch_s0_per_image.json"), lambda: cost_b448(3721089, "reference fast model"),
     800, M_OURS, 0, True),
    ("ours_scratch_seedavg", "Ours B448, scratch, per-image mean of seeds 0-2", "ours", "ours",
     lambda: seed_avg([load(f"ours_scratch_s{s}_per_image.json") for s in range(3)]),
     lambda: cost_b448(3721089, "seed average (3 trainings, one model's cost)"), 800, M_OURS, 0, False),
    ("ours_warm_s0", "Ours B448, warm start (UEyes readout), seed 0", "ours", "ours",
     lambda: load("ours_warm_s0_per_image.json"), lambda: cost_b448(3721089, ""), 800, M_OURS, 0, True),
    ("ours_B644_s0", "Ours B644 (DINOv2-B at 644)", "ours", "ours",
     lambda: load("ours_B644_s0_per_image.json"), lambda: cost_ta("B644", 3721089, "single scale, 46x46 tokens"),
     800, M_OURS, 0, True),
    ("ours_L448_s0", "Ours L448 (DINOv2-L 448 + readout), seed 0", "ours", "ours",
     lambda: load("ours_L448_s0_per_image.json"), lambda: cost_ta("L448", 4933505, ""), 800, M_OURS, 0, True),
    ("ours_L448_s1", "Ours L448, seed 1", "ours", "ours",
     lambda: load("ours_L448_s1_per_image.json"), lambda: cost_ta("L448", 4933505, ""), 800, M_OURS, 0, False),
    ("ours_L448_s2", "Ours L448, seed 2", "ours", "ours",
     lambda: load("ours_L448_s2_per_image.json"), lambda: cost_ta("L448", 4933505, ""), 800, M_OURS, 0, False),
    ("ours_L448_seedavg", "Ours L448, per-image mean of seeds 0-2", "ours", "ours",
     lambda: seed_avg([load(f"ours_L448_s{s}_per_image.json") for s in range(3)]),
     lambda: cost_ta("L448", 4933505, "seed average"), 800, M_OURS, 0, False),
    ("ours_ms_B_s0", "Ours msB (DINOv2-B at 448 + 672, concatenated)", "ours", "ours",
     lambda: load("ours_ms_B_s0_per_image.json"), lambda: cost_ta("msB", 7358337, "two backbone passes"),
     800, M_OURS, 0, True),
    ("ours_L_ms_s0", "Ours L-ms (DINOv2-L at 448 + 672, concatenated)", "ours", "ours",
     lambda: load("ours_L_ms_s0_per_image.json"), lambda: cost_ta("Lms", 9783169, "two backbone passes"),
     800, M_OURS, 0, True),
    ("distill_gt800_teacher800", "Distilled B448: GT + teacher maps on the same 800 images", "ours", "distill",
     lambda: load("distill_gt800_teacher800_per_image.json"), lambda: cost_b448(3721089, "inference = ours B448"),
     800, "800 (GT + teacher map of each of the same 800 training images; sigma on 200 of them)", 0, True),
    ("distill_pitt_only", "Distilled B448: teacher maps on Pitt only (no ADD1000 GT in the student)", "ours",
     "distill", lambda: load("distill_pitt_only_per_image.json"), lambda: cost_b448(3721089, "inference = ours B448"),
     800, "0 direct (800 only through the teacher); sigma on 200 Pitt images vs teacher maps", 2499, True),
    ("distill_pitt", "Distilled B448: 800 GT + 2,499 Pitt teacher maps", "ours", "distill",
     lambda: load("distill_pitt_per_image.json"), lambda: cost_b448(3721089, "inference = ours B448"),
     800, M_PITT, 2499, True),
    ("distill_pitt_L", "Distilled L448: 800 GT + 2,499 Pitt teacher maps", "ours", "distill",
     lambda: load("distill_pitt_L_per_image.json"), lambda: cost_ta("L448", 4933505, "inference = ours L448"),
     800, M_PITT, 2499, True),
    ("msdb_adapt_s0", "DeepGaze MSDB adapted, head 26,463 params (KL+CC), seed 0", "msdb", "baseline",
     lambda: load("msdb_adapt_per_image.json", seed=0), lambda: cost_ct("msdb", 26463, ""), 800, M_MSDB, 0, True),
    ("msdb_adapt_seedavg", "DeepGaze MSDB adapted, head, per-image mean of seeds 0-2", "msdb", "baseline",
     lambda: seed_avg([load("msdb_adapt_per_image.json", seed=s) for s in range(3)]),
     lambda: cost_ct("msdb", 26463, "seed average"), 800, M_MSDB, 0, False),
    ("msdb_adapt_kl", "DeepGaze MSDB adapted, head, KL only", "msdb", "baseline",
     lambda: load("msdb_adapt_kl_per_image.json", seed=0), lambda: cost_ct("msdb", 26463, ""), 800, M_MSDB, 0, True),
    ("msdb_adapt_dsp", "DeepGaze MSDB adapted, 13 per-dataset params", "msdb", "baseline",
     lambda: load("msdb_adapt_dsp_per_image.json", seed=0), lambda: cost_ct("msdb", 13, ""), 800, M_MSDB, 0, True),
    ("iie_ft_kl_sigma", "DeepGaze IIE adapted, published recipe (KL, 20 ep) + sigma [best IIE CC]", "iie", "baseline",
     lambda: load("iie_ft_kl_sigma_per_image.json"), lambda: cost_ct("iie", 3059160, ""), 800, M_MSDB, 0, True),
    ("iie_ft_klcc_e20", "DeepGaze IIE adapted, KL+CC 20 ep + sigma [selected recipe]", "iie", "baseline",
     lambda: load("iie_ft_klcc_e20_per_image.json"), lambda: cost_ct("iie", 3059160, ""), 800, M_MSDB, 0, True),
    ("sum_ft_head", "SUM adapted, output head only (36,937 params), SUM loss + sigma", "sum", "baseline",
     lambda: load("sum_ft_head_per_image.json"),
     lambda: cost_ct("sum", 36937, "pure-PyTorch selective scan; overstates latency vs the CUDA kernel"),
     800, "800 (train 800; token chosen on the 800; sigma on 200 of them)", 0, True),
    ("sum_ft_head_klcc", "SUM adapted, output head only, KL+CC + sigma", "sum", "baseline",
     lambda: load("sum_ft_head_klcc_per_image.json"),
     lambda: cost_ct("sum", 36937, "pure-PyTorch selective scan; overstates latency vs the CUDA kernel"),
     800, "800 (train 800; token chosen on the 800; sigma on 200 of them)", 0, True),
    ("sum_ft", "SUM adapted, full fine-tune (all 57,501,196 params), SUM loss + sigma", "sum", "baseline",
     lambda: load("sum_ft_per_image.json"),
     lambda: cost_ct("sum", 57501196, "pure-PyTorch selective scan; overstates latency vs the CUDA kernel"),
     800, "800 (train 800; token 2 chosen on the 800; sigma on 200 of them)", 0, True),
    ("sum_ft_klcc", "SUM adapted, full fine-tune (all 57,501,196 params), KL+CC + sigma [secondary loss]", "sum",
     "baseline", lambda: load("sum_ft_klcc_per_image.json"),
     lambda: cost_ct("sum", 57501196, "pure-PyTorch selective scan; overstates latency vs the CUDA kernel"),
     800, "800 (train 800; token 2 chosen on the 800; sigma on 200 of them)", 0, True),
]

# ---- SUM full fine-tune record (replaces the former sum_ft_full placeholder)
SUM_FULL = {}
for _t in ("sum_ft", "sum_ft_klcc"):
    _s = json.load(open(os.path.join(RES, f"{_t}_summary.json")))
    assert _s["complete"] and _s["mode"] == "full" and _s["folds_completed"] == [0, 1, 2, 3, 4], _t
    SUM_FULL[_t] = {
        "status": "available (5/5 folds)", "summary_file": os.path.join(RES, f"{_t}_summary.json"),
        "per_image_file": os.path.join(RES, f"{_t}_per_image.json"),
        "raw_job_output": os.path.join(KB, "runs", _t, "ckpt", _t),
        "loss": _s["loss"], "init": _s["init"], "n_trainable_params": _s["n_trainable_params"],
        "means": {m: _s[f"{m}_mean"] for m in METRICS},
        "sd_folds_ddof1": {m: _s[f"{m}_sd_folds_ddof1"] for m in METRICS},
        "sd_folds_ddof0": {m: _s[f"{m}_sd_folds"] for m in METRICS},
        "no_sigma_384_means": {m: _s["no_sigma_at384"][f"{m}_mean"] for m in METRICS},
        "sigma_per_fold": [_s["fold_meta"][str(k)]["sigma"] for k in range(5)],
        "train_seconds_per_fold": [_s["fold_meta"][str(k)]["train_seconds"] for k in range(5)],
        "hardware": "NVIDIA A10G 24 GB (cloud GPU), chunked pure-PyTorch selective scan",
        "command": _s["command"]}

# ------------------------------------------------------------------------------------ training cost
CT_TRAIN = {t["row"]: t for t in json.load(open(os.path.join(RES, "cost_table.json")))["training_adaptation"]}
HW = {"mps": "Apple M4 Pro GPU (MPS)", "cpu": "Apple M4 Pro CPU",
      "a10g": "NVIDIA A10G 24 GB (cloud, CUDA)", "mixed": "Apple M4 Pro, CPU or MPS by fold (contended)"}


def _tc(per_fold, hw, covers, source, one_off="", note=""):
    per_fold = [float(x) for x in per_fold]
    return {"hardware": hw, "train_s_per_fold": per_fold, "train_s_per_fold_median": float(np.median(per_fold)),
            "train_s_total_5_folds": float(np.sum(per_fold)), "covers": covers, "one_off_not_included": one_off,
            "source": source, "note": note}


def train_cost(name):
    js = lambda f: json.load(open(os.path.join(RES, f)))   # noqa: E731
    if name in ("ours_scratch_s0", "ours_warm_s0", "msdb_adapt_s0", "msdb_adapt_kl", "msdb_adapt_dsp",
                "sum_ft_head", "sum_ft_head_klcc"):
        t = CT_TRAIN[name]; dev = t["device"]
        one = {"ours": "DINOv2-B 448 feature cache, 129 s for 1000 images (CPU, 6 threads)",
               "msdb": "MSDB 10-scale feature cache, 3,561 s for 1000 images (MPS)",
               "sum": "SUM decoder-feature cache, 1,988 s for 1000 images (CPU, 6 threads)"}[t["model"]]
        return _tc(t["train_s_per_fold"], HW[dev], t["what_the_time_covers"], "cost_table.json training_adaptation",
                   one, t.get("note", ""))
    if name in ("ours_scratch_seedavg", "ours_L448_seedavg", "msdb_adapt_seedavg"):
        return None
    if name.startswith(("ours_B644", "ours_L448", "ours_ms_B", "ours_L_ms")):
        s = js(f"{name}_summary.json")
        return _tc([f["wall_s"] for f in s["folds_detail"]], HW[s["device"]],
                   "per fold: 60-epoch readout training on cached features + sigma selection + test scoring",
                   f"{name}_summary.json folds_detail wall_s", "backbone feature cache (time not recorded in the summary)")
    if name.startswith("distill_"):
        s = js(f"{name}_summary.json"); devs = {f["device"] for f in s["folds_detail"]}
        assert len(devs) == 1
        return _tc([f["wall_s"] for f in s["folds_detail"]], HW[devs.pop()],
                   "per fold: student training (3000 steps) + sigma selection + test scoring",
                   f"{name}_summary.json folds_detail wall_s",
                   "adapted-MSDB teacher (head training per fold) and its maps for the 2,499 Pitt images (MSDB "
                   "feature pass 3.6 s/img on MPS); backbone features of ADD1000 and Pitt" if "pitt" in name else
                   "adapted-MSDB teacher maps of the 800 training images; backbone feature cache")
    if name.startswith("iie_"):
        s = js(f"{name}_summary.json")
        dev = s.get("device_per_fold", s.get("device"))
        hw = HW["mixed"] if isinstance(dev, list) and len(set(dev)) > 1 else HW[dev if isinstance(dev, str) else dev[0]]
        return _tc(s["t_train_s_per_fold"], hw, "per fold: readout training only (evaluation excluded)",
                   f"{name}_summary.json t_train_s_per_fold", "IIE 4-backbone feature cache, 2,300 s for 1000 images (CPU)",
                   "measured under heavy CPU contention" + (f"; devices by fold {dev}" if isinstance(dev, list) else ""))
    if name in ("sum_ft", "sum_ft_klcc"):
        return _tc(SUM_FULL[name]["train_seconds_per_fold"], HW["a10g"],
                   "per fold: 12-epoch end-to-end fine-tune of all 57.5M parameters from images (chunked pure-PyTorch "
                   "selective scan on the GPU); excludes model loading, sigma selection and test scoring",
                   f"results/{name}_summary.json fold_meta train_seconds (identical to runs/{name}/ckpt/{name}/{name}_fold_meta.json)", "none (no feature cache)")
    raise KeyError(name)

# ------------------------------------------------------------------------------------ build
ROWS = {}
ENTRIES = []
for name, label, fam, kind, loader, costf, maps_n, maps_desc, extra, primary in CAT:
    rows = loader(); ROWS[name] = rows
    e = {"row": name, "label": label, "family": fam, "kind": kind, "frontier_candidate": primary,
         "accuracy": describe(rows), "cost": costf(), "add1000_maps_per_fold": maps_n,
         "add1000_maps_desc": maps_desc, "extra_unlabeled_images": extra}
    if e["cost"].get("cpu_latency_kind") and name.startswith(("ours_scratch", "ours_warm", "distill_pitt", "distill_gt")) \
            and "trackA_contended" not in e["cost"]:
        e["cost"]["trackA_contended"] = TRACKA_CONTENDED_B448
    e["training_cost"] = train_cost(name)
    ENTRIES.append(e)
E = {e["row"]: e for e in ENTRIES}

MSDB = E["msdb_adapt_s0"]; OURS = E["ours_scratch_s0"]
gap0 = MSDB["accuracy"]["CC"]["mean"] - OURS["accuracy"]["CC"]["mean"]
for e in ENTRIES:
    c = e["cost"]; a = e["accuracy"]["CC"]["mean"]
    e["vs_msdb_adapt"] = {"dCC_mean": a - MSDB["accuracy"]["CC"]["mean"],
                          "share_of_ours_to_msdb_gap_closed": (a - OURS["accuracy"]["CC"]["mean"]) / gap0,
                          "GFLOPs_fewer_x": MSDB["cost"]["GFLOPs"] / c["GFLOPs"],
                          "cpu_faster_x": MSDB["cost"]["cpu_latency_s"] / c["cpu_latency_s"],
                          "mps_faster_x": MSDB["cost"]["mps_latency_s"] / c["mps_latency_s"],
                          "cpu_rss_lower_x": MSDB["cost"]["cpu_peak_rss_GB"] / c["cpu_peak_rss_GB"]}
    e["throughput_img_per_s"] = {"cpu_6_threads": 1.0 / c["cpu_latency_s"], "mps": 1.0 / c["mps_latency_s"]}


def pareto(entries, xkey):
    pts = [(e["row"], xkey(e), e["accuracy"]["CC"]["mean"]) for e in entries]
    on = []
    for n, x, y in pts:
        dom = any((x2 <= x and y2 >= y) and (x2 < x or y2 > y) for n2, x2, y2 in pts if n2 != n)
        if not dom:
            on.append(n)
    return sorted(on, key=lambda n: xkey(E[n]))


CANDS = [e for e in ENTRIES if e["frontier_candidate"]]
FRONT = pareto(CANDS, lambda e: e["cost"]["cpu_latency_s"])
for e in ENTRIES:
    e["on_pareto_frontier_cc_vs_cpu"] = e["row"] in FRONT


def est_cpu(e):
    return e["cost"].get("trackA_contended", {}).get("cpu_latency_normalised_estimate_s", e["cost"]["cpu_latency_s"])


def raw_cpu(e):
    t = e["cost"].get("trackA_contended")
    if not t:
        return e["cost"]["cpu_latency_s"]
    v = [x for k in ("cpu_run1", "cpu_run2") for x in t["raw_contended_medians_s"][k]]
    return float(min(v)) if e["row"] not in ("ours_scratch_s0", "ours_warm_s0", "distill_pitt", "distill_gt800_teacher800",
                                             "distill_pitt_only") else e["cost"]["cpu_latency_s"]


ROBUST = {
    "cpu_latency_quiet (primary)": FRONT,
    "cpu_latency_trackA_normalised_estimates": pareto(CANDS, est_cpu),
    "cpu_latency_trackA_raw_contended_min (B448 rows at cost_table)": pareto(CANDS, raw_cpu),
    "mps_latency": pareto(CANDS, lambda e: e["cost"]["mps_latency_s"]),
    "GFLOPs": pareto(CANDS, lambda e: e["cost"]["GFLOPs"]),
}

# ---- best fast model and paired tests
FAST_MAX = MSDB["cost"]["cpu_latency_s"] / 10.0
fast = [e for e in CANDS if e["cost"]["cpu_latency_s"] <= FAST_MAX]
best_fast = max(fast, key=lambda e: e["accuracy"]["CC"]["mean"])
base_cost = [e for e in CANDS if e["cost"]["GFLOPs"] == OURS["cost"]["GFLOPs"]]
best_base = max(base_cost, key=lambda e: e["accuracy"]["CC"]["mean"])
models = [best_fast["row"]] + ([best_base["row"]] if best_base["row"] != best_fast["row"] else [])


def family(refs):
    res = {mdl: {ref: compare(ROWS[mdl], ROWS[ref]) for ref in refs} for mdl in models}
    keys = [(mdl, ref) for mdl in models for ref in refs]
    for m in METRICS:
        for k in ("wilcoxon_p", "nb_p"):
            adj = holm([res[a][b][m][k] for a, b in keys])
            for (a, b), pa in zip(keys, adj):
                res[a][b][m][k + "_holm"] = float(pa)
    return {"members": [f"{a} vs {b}" for a, b in keys], "results": res}


TESTS = {"primary": family(["msdb_adapt_s0", "ours_scratch_s0"]),
         "secondary_seedavg": family(["msdb_adapt_seedavg", "ours_scratch_seedavg"])}

# cross-check against Track B's stored tests (same functions, same rng seed -> identical numbers expected)
xchk = {}
ov = json.load(open(os.path.join(RES, "distill_track_overview.json")))["tests"]
for mdl in models:
    for ref in ("ours_scratch_s0", "msdb_adapt_s0"):
        k = f"{mdl}__vs__{ref}"
        if k in ov:
            mine = TESTS["primary"]["results"][mdl][ref]["CC"]
            theirs = ov[k]["CC"]
            xchk[k] = {"mean_diff_here": mine["mean_diff"], "mean_diff_trackB": theirs["mean_diff"],
                       "wilcoxon_p_here": mine["wilcoxon_p"], "wilcoxon_p_trackB": theirs["wilcoxon_p"],
                       "nb_p_here": mine["nb_p"], "nb_p_trackB": theirs["nb_p"],
                       "boot95_here": mine["boot95"], "boot95_trackB": theirs["boot95"]}

reaches = [e["row"] for e in fast if e["accuracy"]["CC"]["mean"] >= MSDB["accuracy"]["CC"]["mean"]]
within_0002 = [e["row"] for e in CANDS if e["row"] != MSDB["row"]
               and e["accuracy"]["CC"]["mean"] >= MSDB["accuracy"]["CC"]["mean"] - 0.002]

# ---- SUM full fine-tune paired tests (letter convention: 10,000 bootstrap resamples, rng 20261121)
B_BOOT_SUM, SEED_SUM = 10000, 20261121


def compare_boot(a_rows, b_rows, n_boot, seed):
    rng = np.random.default_rng(seed); o = {}
    for m in METRICS:
        d = vec(a_rows, m) - vec(b_rows, m)
        bm = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
        o[m] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
    return o


SUM_REFS = ["msdb_adapt_s0", "ours_scratch_s0", "distill_pitt", "distill_pitt_L", "sum_ft_head", "sum_ft_klcc"]


def sum_family(refs):
    res = {}
    for ref in refs:
        c = compare(ROWS["sum_ft"], ROWS[ref])               # Wilcoxon, NB, effect sizes, 5000/rng-0 CI
        bb = compare_boot(ROWS["sum_ft"], ROWS[ref], B_BOOT_SUM, SEED_SUM)
        for m in METRICS:
            c[m]["boot95_5000_seed0"] = c[m].pop("boot95")
            c[m]["boot95"] = bb[m]
        res[ref] = c
    for m in METRICS:
        for k in ("wilcoxon_p", "nb_p"):
            adj = holm([res[r][m][k] for r in refs])
            for r, pa in zip(refs, adj):
                res[r][m][k + "_holm"] = float(pa)
    return {"members": [f"sum_ft vs {r}" for r in refs], "results": res}


TESTS_SUM = {"primary": sum_family(SUM_REFS), "secondary_seedavg": sum_family(["msdb_adapt_seedavg"]),
             "bootstrap": f"{B_BOOT_SUM} resamples, fresh numpy default_rng({SEED_SUM}) per comparison, metric order "
                          "CC, SIM, KLD; boot95_5000_seed0 = the convention of the older sections",
             "holm": "primary: 6 comparisons, per metric and per test; secondary: 1 comparison (no correction)",
             "differences": "sum_ft minus reference (KLD: negative = sum_ft better)"}
# cross-check vs main_table.json (ours_scratch_s0 minus sum_ft there; same Wilcoxon / NB, sign flipped)
MT = json.load(open(os.path.join(RES, "main_table.json")))
_mt = MT["tests_vs_ours"]["ours_scratch_s0"].get("sum_ft")
xchk_sum = None
if _mt:
    _h = TESTS_SUM["primary"]["results"]["ours_scratch_s0"]
    xchk_sum = {m: {"mean_diff_here": _h[m]["mean_diff"], "minus_mean_diff_main_table": -_mt[m]["mean_diff"],
                    "wilcoxon_p_here": _h[m]["wilcoxon_p"], "wilcoxon_p_main_table": _mt[m]["wilcoxon_p"],
                    "nb_p_here": _h[m]["nb_p"], "nb_p_main_table": _mt[m]["nb_p"]} for m in METRICS}

# ---- zero-shot -> adapted gain per model family (rows of main_table.json)
MTR = {r["row"]: r for r in MT["rows"]}
GAIN_DEF = [
    ("SUM", "sum_zeroshot_t2", None, "sum_calibrated", "sum_ft"),
    ("DeepGaze MSDB", "msdb_zeroshot_uniform", "msdb_zeroshot_adprior", "msdb_zeroshot_foldprior", "msdb_adapt_s0"),
    ("DeepGaze IIE", "iie_zeroshot_uniform", "iie_zeroshot_adprior", "iie_zeroshot_foldprior_sigma", "iie_ft_kl_sigma"),
]


def mt_stats(n):
    s = MTR[n]["stats_384"]
    return {m: {"mean": s[m]["mean"], "sd_folds_ddof1": s[m]["sd_folds_ddof1"]} for m in METRICS}


GAINS = {}
for fam_, zs, zs_pub, cal, ad in GAIN_DEF:
    g = {"zero_shot_row": zs, "zero_shot_published_prior_row": zs_pub, "calibrated_zero_shot_row": cal, "adapted_row": ad,
         "zero_shot": mt_stats(zs), "adapted": mt_stats(ad), "calibrated_zero_shot": mt_stats(cal),
         "gain_from_zero_shot": {m: mt_stats(ad)[m]["mean"] - mt_stats(zs)[m]["mean"] for m in METRICS},
         "gain_from_calibrated_zero_shot": {m: mt_stats(ad)[m]["mean"] - mt_stats(cal)[m]["mean"] for m in METRICS}}
    if zs_pub:
        g["zero_shot_published_prior"] = mt_stats(zs_pub)
        g["gain_from_published_prior_zero_shot"] = {m: mt_stats(ad)[m]["mean"] - mt_stats(zs_pub)[m]["mean"] for m in METRICS}
    GAINS[fam_] = g
GAINS["ours"] = {"note": "n/a: our readout has no zero-shot version on ADD1000 (the warm-start readout before "
                         "ADD1000 training exists only at fold level, main_table.json fold_level_only_rows)"}
RANK = {k: sorted([f for f, *_ in GAIN_DEF], key=lambda f: -GAINS[f][key]["CC"]["mean"])
        for k, key in (("zero_shot_no_add1000_maps", "zero_shot"), ("calibrated_zero_shot_fold_safe", "calibrated_zero_shot"),
                       ("adapted", "adapted"))}
RANK["zero_shot_published_2fold_prior (MSDB, IIE; SUM t2 has no prior)"] = sorted(
    [f for f, *_ in GAIN_DEF], key=lambda f: -(GAINS[f].get("zero_shot_published_prior", GAINS[f]["zero_shot"])["CC"]["mean"]))

OUT = {
    "description": __doc__.strip(),
    "protocol": {"folds": "splits/kfold_seed0.json (5 x 200)", "scoring": "384 x 384, clip at 0 and divide by sum",
                 "sd": "over the 5 fold means, ddof = 1", "fast_definition": f"CPU latency <= MSDB-adapted / 10 = {FAST_MAX:.3f} s",
                 "frontier": "Pareto on (CPU latency, mean CC) over frontier_candidate rows",
                 "latency": "batch 1, float32, preprocessed input in memory, model forward only (our Gaussian "
                            "finalizer adds 1.4 ms on CPU and is not included, as in cost_table.json ratios); "
                            "CPU 6 threads; MPS = Apple M4 Pro GPU"},
    "quiet_session": SESSION,
    "models": ENTRIES,
    "sum_full_finetune": SUM_FULL,
    "pareto_frontier_cc_vs_cpu": FRONT,
    "pareto_frontier_cc_vs_mps": ROBUST["mps_latency"],
    "pareto_frontier_cc_vs_gflops": ROBUST["GFLOPs"],
    "pareto_frontier_robustness": ROBUST,
    "pareto_frontier_robustness_note": "the two trackA_* axes use Track A's superseded contended CPU estimates; audit "
                                       "only, never quoted",
    "best_fast_model": best_fast["row"], "best_model_at_our_base_cost": best_base["row"],
    "fast_models_reaching_msdb_adapted_cc": reaches,
    "candidates_within_0.002_CC_of_msdb_adapted": within_0002,
    "paired_tests": TESTS, "paired_tests_crosscheck_trackB": xchk,
    "paired_tests_sum_ft": TESTS_SUM, "paired_tests_sum_ft_crosscheck_main_table": xchk_sum,
    "zero_shot_to_adapted_gain": GAINS, "zero_shot_vs_adapted_ranking_by_CC": RANK,
    "training_cost_note": "per-fold training wall-clock as recorded by each row; hardware differs between rows "
                          "(Apple M4 Pro CPU, Apple M4 Pro GPU via MPS, cloud NVIDIA A10G), so times are not "
                          "comparable across hardware; ours/MSDB/distill on MPS held the shared MPS lock while the "
                          "CPU was shared with other tracks",
    "created": time.strftime("%Y-%m-%d %H:%M:%S"), "command": "python -I " + " ".join(sys.argv),
}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
        return str(o)
    return o


json.dump(clean(OUT), open(os.path.join(RES, "frontier.json"), "w"), indent=1)

# ------------------------------------------------------------------------------------ figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_OURS, C_DIST, C_BASE = "#2a78d6", "#1baf7a", "#eb6834"   # categorical slots 1, 3, 2 (validated all-pairs set)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FIG_ROWS = {  # row: (short label, dx pt, dy pt, ha, va)
    "ours_scratch_s0": ("Ours B448", -8, 0, "right", "center"),
    "ours_warm_s0": ("Ours B448, warm start", -8, 0, "right", "center"),
    "distill_gt800_teacher800": ("Distilled B448 (same 800)", -8, 0, "right", "center"),
    "distill_pitt_only": ("Distilled B448 (Pitt only)", -8, 0, "right", "center"),
    "distill_pitt": ("Distilled B448 (800 + Pitt)", -8, 4, "right", "bottom"),
    "ours_B644_s0": ("Ours B644", 0, -11, "center", "center"),
    "ours_L448_s0": ("Ours L448", 6, -10, "left", "center"),
    "ours_ms_B_s0": ("Ours msB", 8, 0, "left", "center"),
    "ours_L_ms_s0": ("Ours L-ms", 8, 0, "left", "center"),
    "distill_pitt_L": ("Distilled L448\n(800 + Pitt)", 8, 6, "left", "bottom"),
    "msdb_adapt_s0": ("MSDB adapted (head)", -9, 5, "right", "bottom"),
    "msdb_adapt_dsp": ("MSDB adapted (13 params)", -6, -10, "right", "center"),
    "iie_ft_kl_sigma": ("DeepGaze IIE adapted", 8, 0, "left", "center"),
    "sum_ft_head": ("SUM head-only", 8, 0, "left", "center"),
    "sum_ft": ("SUM full fine-tune", 8, -5, "left", "top"),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=200)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(INK2); ax.spines[s].set_linewidth(0.8)
ax.grid(True, which="major", color=GRID, linewidth=0.7); ax.set_axisbelow(True)
ax.set_xscale("log")

# frontier as the best CC attainable within a CPU-latency budget (step function)
fx = [E[n]["cost"]["cpu_latency_s"] for n in FRONT]; fy = [E[n]["accuracy"]["CC"]["mean"] for n in FRONT]
sx, sy = [], []
for i in range(len(fx)):
    if i:
        sx.append(fx[i]); sy.append(fy[i - 1])
    sx.append(fx[i]); sy.append(fy[i])
sx.append(14.0); sy.append(fy[-1])
ax.plot(sx, sy, color=INK2, lw=1.2, ls=(0, (4, 2)), zorder=1, label="Pareto frontier (CC vs CPU latency)")
ax.axhline(MSDB["accuracy"]["CC"]["mean"], color=C_BASE, lw=0.8, ls=":", zorder=0)
ax.text(0.021, MSDB["accuracy"]["CC"]["mean"] + 0.0008, f"MSDB adapted CC {MSDB['accuracy']['CC']['mean']:.3f}",
        color=INK2, fontsize=7.5, va="bottom")

for n, (lab, dx, dy, ha, va) in FIG_ROWS.items():
    e = E[n]; x = e["cost"]["cpu_latency_s"]; y = e["accuracy"]["CC"]["mean"]; sd = e["accuracy"]["CC"]["sd_folds_ddof1"]
    col, mk = (C_DIST, "D") if e["kind"] == "distill" else ((C_OURS, "o") if e["family"] == "ours" else (C_BASE, "s"))
    on = e["on_pareto_frontier_cc_vs_cpu"]
    ax.errorbar(x, y, yerr=sd, fmt="none", ecolor=col, elinewidth=0.9, capsize=0, alpha=0.55, zorder=2)
    ax.scatter([x], [y], s=46 if on else 34, marker=mk, color=col, edgecolors="#fcfcfb" if not on else INK,
               linewidths=1.6 if not on else 1.1, zorder=4)
    ax.annotate(lab + (f"\nCC {y:.3f}" if on else ""), (x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=7.4, linespacing=1.15, color=INK if on else INK2,
                fontweight="bold" if on else "normal", zorder=5)

ax.set_xlim(0.02, 14); ax.set_ylim(0.848, 0.928)
ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]); ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10"])
ax.set_xlabel("CPU latency per image, seconds (log scale; batch 1, 6 threads, Apple M4 Pro)", color=INK)
ax.set_ylabel("CC on ADD1000 (mean over 5 folds; bars = fold SD)", color=INK)
ax.tick_params(colors=INK2)
from matplotlib.lines import Line2D
hand = [Line2D([], [], marker="o", ls="", color=C_OURS, markersize=6, label="Ours (frozen DINOv2 + readout)"),
        Line2D([], [], marker="D", ls="", color=C_DIST, markersize=5.5, label="Ours, distilled from adapted MSDB"),
        Line2D([], [], marker="s", ls="", color=C_BASE, markersize=5.5, label="Adapted baselines"),
        Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 2)), label="Pareto frontier (outlined, bold labels)")]
ax.legend(handles=hand, loc="lower left", frameon=False, fontsize=7.4)
ax.set_title("ADD1000: accuracy vs CPU cost of every adapted model (800 training maps per fold)",
             fontsize=9.5, color=INK, loc="left")
fig.text(0.01, 0.005, "SUM rows: latency with a pure-PyTorch selective scan (94% of SUM's CPU time; zero-cost scan "
         "would give 0.073 s). Not an architectural cost.", fontsize=6.6, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(RES, "frontier.png"), facecolor=fig.get_facecolor())

# ------------------------------------------------------------------------------------ markdown
def f4(x):
    return f"{x:.4f}"


def ms(e, m):
    a = e["accuracy"][m]; return f"{a['mean']:.4f} ± {a['sd_folds_ddof1']:.4f}"


def pf(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def lat(e, key):
    c = e["cost"]; return f"{c[key]:.3f}"


def ptxt(n):
    return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n:,}"


bf, bb = E[best_fast["row"]], E[best_base["row"]]
T = TESTS["primary"]["results"]
tf_m, tf_o = T[bf["row"]]["msdb_adapt_s0"], T[bf["row"]]["ours_scratch_s0"]
tb_m, tb_o = T[bb["row"]]["msdb_adapt_s0"], T[bb["row"]]["ours_scratch_s0"]
L = []
L.append("# ADD1000 accuracy-cost frontier of every adapted model\n")
L.append(f"Compiled {time.strftime('%d %b %Y')} by scripts/frontier/compile_frontier.py from saved per-image files and "
         "cost measurements only (nothing re-trained or re-scored). Same folds (splits/kfold_seed0.json), same "
         "384 x 384 clip-and-sum scoring as main_table.json; mean ± SD over the 5 folds (ddof 1). Differences are "
         "model minus reference (for KLD, negative = model better). Machine: Apple M4 Pro, 6 CPU threads, MPS = "
         "its GPU; no CUDA GPU.\n")
SF = E["sum_ft"]; SK = E["sum_ft_klcc"]
TS = TESTS_SUM["primary"]["results"]; TS2 = TESTS_SUM["secondary_seedavg"]["results"]["msdb_adapt_seedavg"]


def sig(c):
    w, n = c["wilcoxon_p_holm"] < 0.05, c["nb_p_holm"] < 0.05
    return ("significant on both tests" if w and n else "significant per image (Wilcoxon) but not at fold level (NB)"
            if w else "significant at fold level (NB) but not per image (Wilcoxon)" if n else "not significant on either test")


def cif(x):
    # 4 decimals, but never print a signed zero for a bound that is not zero
    return f"{x:+.4f}" if abs(x) >= 5e-5 else f"{x:+.5f}"


def dtxt(c, m, holm=True):
    k = "_holm" if holm else ""
    return (f"Δ{m} {c[m]['mean_diff']:+.4f} [95% CI {cif(c[m]['boot95'][0])}, {cif(c[m]['boot95'][1])}], "
            f"{c[m]['folds_model_better']}/5 folds, {c[m]['frac_images_model_better']:.1%} of images, "
            f"{'Holm ' if holm else ''}Wilcoxon p {pf(c[m]['wilcoxon_p' + k])}, {'Holm ' if holm else ''}NB p "
            f"{pf(c[m]['nb_p' + k])}")


_sm = TS["msdb_adapt_s0"]
_ours_rows = [n for n in ("ours_scratch_s0", "distill_pitt", "distill_pitt_L")]
_x = lambda a, b, k: E[a]["cost"][k] / E[b]["cost"][k]   # noqa: E731
L.append("## Answer\n")
L.append(f"**With SUM fully fine-tuned, a {SF['cost']['GFLOPs']:.0f}-GFLOP model comes within "
         f"{-SF['vs_msdb_adapt']['dCC_mean']:.4f} CC of adapted MSDB, and the CC difference is not significant.** "
         f"Adapted MSDB: CC {ms(MSDB,'CC')}, SIM {ms(MSDB,'SIM')}, KLD {ms(MSDB,'KLD')}. Fully fine-tuned SUM (sum_ft: all "
         f"57,501,196 parameters trained on each fold's 800 maps with SUM's own loss, token 2, sigma on 200 training "
         f"maps; trained on a cloud NVIDIA A10G): CC {ms(SF,'CC')}, SIM {ms(SF,'SIM')}, KLD {ms(SF,'KLD')}, at "
         f"{SF['cost']['GFLOPs']:.1f} GFLOPs ({SF['vs_msdb_adapt']['GFLOPs_fewer_x']:.0f}x fewer than MSDB). sum_ft minus "
         f"MSDB adapted (seed 0): {dtxt(_sm, 'CC')} ({sig(_sm['CC'])}); {dtxt(_sm, 'SIM')} ({sig(_sm['SIM'])}); "
         f"{dtxt(_sm, 'KLD')} ({sig(_sm['KLD'])}). Against the MSDB seed average the picture is the same (ΔCC "
         f"{TS2['CC']['mean_diff']:+.4f}, NB p {pf(TS2['CC']['nb_p'])}; ΔSIM {TS2['SIM']['mean_diff']:+.4f}, NB p "
         f"{pf(TS2['SIM']['nb_p'])}; ΔKLD {TS2['KLD']['mean_diff']:+.4f}, NB p {pf(TS2['KLD']['nb_p'])}). So on this "
         f"benchmark full SUM fine-tuning is not significantly different from adapted MSDB on CC "
         f"({-_sm['CC']['mean_diff']:.4f} lower, lower in {5 - _sm['CC']['folds_model_better']} of 5 folds; "
         f"not significant is not the same as equivalent) and is slightly better on SIM. The KL+CC variant "
         f"(sum_ft_klcc) gives the same CC ({SK['accuracy']['CC']['mean']:.4f}; sum_ft minus sum_ft_klcc ΔCC "
         f"{TS['sum_ft_klcc']['CC']['mean_diff']:+.4f}, Holm NB p {pf(TS['sum_ft_klcc']['CC']['nb_p_holm'])}). "
         + ("" if not reaches else f"(Unexpected: {reaches} reach MSDB's CC under the CPU definition of fast.) "))
L.append(f"* Under the CPU-latency definition of fast declared before the tests (at least 10x faster than MSDB on CPU, "
         f"<= {FAST_MAX:.3f} s), SUM is not fast: {SF['cost']['cpu_latency_s']:.3f} s on CPU "
         f"({SF['vs_msdb_adapt']['cpu_faster_x']:.1f}x faster than MSDB) with the pure-PyTorch chunked selective scan, "
         f"{SF['cost']['mps_latency_s']:.3f} s on MPS ({SF['vs_msdb_adapt']['mps_faster_x']:.1f}x faster). Under that "
         f"definition no fast model reaches adapted MSDB's CC, as before. The candidates within 0.002 CC of MSDB are "
         + ", ".join(within_0002) + "; of these only "
         + ", ".join(n for n in within_0002 if E[n]["cost"]["GFLOPs"] < MSDB["cost"]["GFLOPs"] / 10)
         + f" has far lower compute than MSDB (sum_ft_klcc is {-SK['vs_msdb_adapt']['dCC_mean']:.4f} CC below MSDB).")
# guard the prose above against silent changes of the underlying numbers
assert sig(_sm["CC"]) == "not significant on either test" and _sm["SIM"]["mean_diff"] > 0 \
    and _sm["SIM"]["nb_p_holm"] < 0.05 and _sm["SIM"]["wilcoxon_p_holm"] < 0.05, "update the Answer prose"
L.append(f"* Best fast model (>= 10x faster than MSDB on CPU, highest CC): **{bf['row']}** (frozen DINOv2-large 448 + "
         f"readout, trained on the 800 maps plus 2,499 unlabeled Pitt ads labelled by the adapted MSDB teacher). CC "
         f"{ms(bf,'CC')}, SIM {ms(bf,'SIM')}, KLD {ms(bf,'KLD')}. Against MSDB adapted: ΔCC "
         f"{tf_m['CC']['mean_diff']:+.4f} [95% CI {tf_m['CC']['boot95'][0]:+.4f}, {tf_m['CC']['boot95'][1]:+.4f}], "
         f"better in {tf_m['CC']['folds_model_better']}/5 folds and on {tf_m['CC']['frac_images_model_better']:.1%} of "
         f"images, Holm Wilcoxon p {pf(tf_m['CC']['wilcoxon_p_holm'])}, Holm NB p {pf(tf_m['CC']['nb_p_holm'])}. It closes "
         f"{bf['vs_msdb_adapt']['share_of_ours_to_msdb_gap_closed']:.0%} of the {gap0:.4f} CC gap between our base model and MSDB. "
         f"Cost: {bf['cost']['GFLOPs']:.0f} GFLOPs ({bf['vs_msdb_adapt']['GFLOPs_fewer_x']:.1f}x fewer than MSDB), "
         f"{lat(bf,'cpu_latency_s')} s CPU ({bf['vs_msdb_adapt']['cpu_faster_x']:.1f}x faster, measured on a quiet machine), "
         f"{lat(bf,'mps_latency_s')} s MPS ({bf['vs_msdb_adapt']['mps_faster_x']:.1f}x faster), "
         f"{bf['cost']['cpu_peak_rss_GB']:.1f} GB CPU peak memory vs {MSDB['cost']['cpu_peak_rss_GB']:.1f} GB.")
L.append(f"* Best model at our base inference cost: **{bb['row']}** (same network and cost as ours B448; distillation "
         f"changes training only). CC {ms(bb,'CC')}; vs MSDB ΔCC {tb_m['CC']['mean_diff']:+.4f} "
         f"[{tb_m['CC']['boot95'][0]:+.4f}, {tb_m['CC']['boot95'][1]:+.4f}], {tb_m['CC']['folds_model_better']}/5 folds, "
         f"{tb_m['CC']['frac_images_model_better']:.1%} of images, Holm NB p {pf(tb_m['CC']['nb_p_holm'])}; vs ours B448 "
         f"ΔCC {tb_o['CC']['mean_diff']:+.4f} [{tb_o['CC']['boot95'][0]:+.4f}, {tb_o['CC']['boot95'][1]:+.4f}], "
         f"{tb_o['CC']['folds_model_better']}/5 folds, Holm NB p {pf(tb_o['CC']['nb_p_holm'])}. Cost: "
         f"{bb['vs_msdb_adapt']['GFLOPs_fewer_x']:.1f}x fewer GFLOPs, {bb['vs_msdb_adapt']['cpu_faster_x']:.1f}x faster on CPU, "
         f"{bb['vs_msdb_adapt']['mps_faster_x']:.1f}x faster on MPS, {bb['vs_msdb_adapt']['cpu_rss_lower_x']:.1f}x less CPU memory; "
         f"it closes {bb['vs_msdb_adapt']['share_of_ours_to_msdb_gap_closed']:.0%} of the gap.")
_so = {n: TS[n] for n in _ours_rows}
L.append(f"* sum_ft beats every model of ours on all three metrics: vs distill_pitt_L ΔCC "
         f"{_so['distill_pitt_L']['CC']['mean_diff']:+.4f}, vs distill_pitt {_so['distill_pitt']['CC']['mean_diff']:+.4f}, "
         f"vs ours B448 {_so['ours_scratch_s0']['CC']['mean_diff']:+.4f} (each 5/5 folds on CC, SIM and KLD; largest Holm "
         f"NB p over these 3 x 3 tests {pf(max(_so[n][m]['nb_p_holm'] for n in _so for m in METRICS))}, largest Holm "
         f"Wilcoxon p {pf(max(_so[n][m]['wilcoxon_p_holm'] for n in _so for m in METRICS))}). Against ours B448 it needs "
         f"{1/_x('sum_ft','ours_scratch_s0','GFLOPs'):.1f}x fewer GFLOPs ({SF['cost']['GFLOPs']:.0f} vs "
         f"{OURS['cost']['GFLOPs']:.0f}) and fewer total parameters ({ptxt(SF['cost']['params_total'])} vs "
         f"{ptxt(OURS['cost']['params_total'])}), but in our measurement it is {_x('sum_ft','ours_scratch_s0','cpu_latency_s'):.1f}x "
         f"slower on CPU, {_x('sum_ft','ours_scratch_s0','mps_latency_s'):.1f}x slower on MPS and uses "
         f"{_x('sum_ft','ours_scratch_s0','cpu_peak_rss_GB'):.1f}x the CPU peak memory ({SF['cost']['cpu_peak_rss_GB']:.2f} vs "
         f"{OURS['cost']['cpu_peak_rss_GB']:.2f} GB). Against distill_pitt_L: {1/_x('sum_ft','distill_pitt_L','GFLOPs'):.1f}x "
         f"fewer GFLOPs, {_x('sum_ft','distill_pitt_L','cpu_latency_s'):.1f}x slower on CPU, "
         f"{_x('sum_ft','distill_pitt_L','mps_latency_s'):.1f}x slower on MPS.")
_FM, _FG = ROBUST["mps_latency"], ROBUST["GFLOPs"]
_ours_fam = [e["row"] for e in CANDS if e["family"] == "ours"]
L.append(f"* Pareto frontiers (same candidates, one entry per configuration). CC vs quiet CPU latency, {len(FRONT)} points: "
         + ", ".join(f"{n} ({E[n]['cost']['cpu_latency_s']:.3f} s, CC {E[n]['accuracy']['CC']['mean']:.4f})" for n in FRONT)
         + f". CC vs MPS latency, {len(_FM)} points: "
         + ", ".join(f"{n} ({E[n]['cost']['mps_latency_s']:.3f} s)" for n in _FM)
         + f". CC vs GFLOPs, {len(_FG)} points: "
         + ", ".join(f"{n} ({E[n]['cost']['GFLOPs']:.1f})" for n in _FG)
         + (". No model of ours is on the GFLOPs frontier any more: sum_ft has fewer FLOPs and higher CC than every one "
            "of them." if not set(_FG) & set(_ours_fam) else ".")
         + " Every non-distilled variant of ours (bigger backbone, higher resolution, multi-scale) is dominated on CPU "
           "and MPS by a distilled student that is at least as fast. Adapted DeepGaze IIE, SUM head-only and the "
           "13-parameter MSDB are each slower on CPU and MPS than our base model and at least 0.03 CC below it; on the "
           "GFLOPs axis IIE (24 GFLOPs) stays on the frontier as the cheapest model, while SUM head-only is now "
           "dominated by sum_ft (same network, higher CC).")
assert set(FRONT) == set(_FM), "CPU and MPS frontiers differ; update the prose"
_fr = [E[n] for n in FRONT if n not in ("msdb_adapt_s0", "sum_ft")]
_g = [e["vs_msdb_adapt"] for e in _fr]
L.append(f"* So the honest position has two parts. Against adapted MSDB, ours remains an accuracy-cost trade-off, not "
         f"parity: {min(-e['dCC_mean'] for e in _g):.3f} to "
         f"{max(-e['dCC_mean'] for e in _g):.3f} CC below adapted MSDB for {min(e['GFLOPs_fewer_x'] for e in _g):.1f}x to "
         f"{max(e['GFLOPs_fewer_x'] for e in _g):.1f}x fewer FLOPs, {min(e['cpu_faster_x'] for e in _g):.1f}x to "
         f"{max(e['cpu_faster_x'] for e in _g):.1f}x lower CPU latency, {min(e['mps_faster_x'] for e in _g):.1f}x to "
         f"{max(e['mps_faster_x'] for e in _g):.1f}x lower MPS latency and {min(e['cpu_rss_lower_x'] for e in _g):.1f}x to "
         f"{max(e['cpu_rss_lower_x'] for e in _g):.1f}x lower CPU memory (throughput on 6 CPU threads: "
         + ", ".join(f"{e['row']} {e['throughput_img_per_s']['cpu_6_threads']:.1f} img/s" for e in _fr)
         + f" vs MSDB {MSDB['throughput_img_per_s']['cpu_6_threads']:.2f} img/s). MSDB stays better on "
         + " and ".join(f"{1 - T[n]['msdb_adapt_s0']['CC']['frac_images_model_better']:.0%}" for n in models)
         + " of images (CC, vs " + " and ".join(models) + "). The distilled students need the adapted MSDB teacher once at training time (one pass over the 2,499 "
         "Pitt images), not at inference.")
_tsf, _to = SF["training_cost"], OURS["training_cost"]
PROF = json.load(open(os.path.join(RES, "sum_cpu_profile.json")))
_pc, _pm = PROF["headline_cpu"], PROF["headline_mps"]
L.append(f"* SUM's latency here is mostly its selective scan, not its architecture (results/sum_cpu_profile.json, quiet "
         f"machine): {_pc['scan_fraction_timed_wrapper']:.0%} of SUM's CPU forward is the pure-PyTorch chunked scan "
         f"(one torch.cumprod call alone is {_pc['torch_cumprod_share_of_forward']:.0%}). With a zero-cost scan (a lower "
         f"bound, not a real kernel) SUM's forward takes {_pc['forward_s_median_identity_scan']:.3f} s on CPU "
         f"({_pc['ratio_sum_identity_scan_over_ours_B448']:.2f}x our B448) and {_pm['forward_s_median_identity_scan']:.3f} s "
         f"on MPS; a numerically equivalent rewrite of the cumprod already gives {_pc['forward_s_median_mulloop_diagnostic']:.3f} s "
         f"on CPU, and mamba-ssm's sequential reference scan gives {_pc['forward_s_median_ref_mamba_selective_scan_ref']:.2f} s. "
         f"So 'SUM is not fast' and our latency advantage over SUM hold only for the scan implementations available "
         f"on this machine; SUM is {_pc['ratio_sum_identity_scan_over_ours_B448']:.1f}x to "
         f"{_pc['ratio_sum_ref_mamba_over_ours_B448']:.1f}x our CPU latency depending on the scan.")
L.append(f"* Against fully fine-tuned SUM, ours is {_so['distill_pitt_L']['CC']['mean_diff']:.3f} to "
         f"{_so['ours_scratch_s0']['CC']['mean_diff']:.3f} CC below, and our only inference-cost advantage is latency on "
         "this machine (CPU and MPS), and that advantage comes from SUM's pure-PyTorch selective scan, not from the "
         "architecture (previous bullet). In FLOPs and total parameters SUM is the cheaper "
         "model. Training cost points the other way: SUM's full fine-tune took a median "
         f"{_tsf['train_s_per_fold_median']:,.0f} s per fold on a cloud NVIDIA A10G, ours B448 "
         f"{_to['train_s_per_fold_median']:,.0f} s per fold on the Apple M4 Pro GPU (MPS) plus a one-off 129 s feature "
         "cache; the hardware differs and the A10G run also used the chunked pure-PyTorch scan, so this is not a "
         "like-for-like ratio (section 1b). The SUM vs MSDB comparison is "
         "also asymmetric: SUM trains all 57.5M parameters, MSDB only its 26,463-parameter head.\n")

L.append("## 1. Every adapted model\n")
L.append("Maps = ADD1000 maps per fold used in any way (training, prior, sigma; never the scored image). Latency = model "
         "forward, batch 1, float32, input already in memory; our post-hoc finalizer adds 1.4 ms on CPU. "
         "CPU peak memory = peak process RSS during inference incl. weights; MPS memory = maximum driver allocation. "
         "F = on the Pareto frontier (CC vs CPU latency). All CPU latencies are quiet-machine measurements "
         "(see section 4). Gap closed = (CC - ours B448) / (MSDB adapted - ours B448); negative = below ours B448. "
         "Train s/fold = median per-fold training wall-clock and the hardware it ran on (details and what each time "
         "covers in section 1b; times on different hardware are not comparable).\n")
L.append("| Row | F | CC | SIM | KLD | Trainable / total params | GFLOPs | CPU s | MPS s | CPU mem GB | MPS mem GB | ADD1000 maps (+ unlabeled) | ΔCC vs MSDB | gap closed | Train s/fold (hardware) |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
HW_SHORT = {HW["mps"]: "MPS", HW["cpu"]: "CPU", HW["a10g"]: "A10G", HW["mixed"]: "CPU/MPS"}
order = ["ours_scratch_s0", "ours_scratch_seedavg", "ours_warm_s0", "ours_B644_s0", "ours_L448_s0", "ours_L448_s1",
         "ours_L448_s2", "ours_L448_seedavg", "ours_ms_B_s0", "ours_L_ms_s0", "distill_gt800_teacher800",
         "distill_pitt_only", "distill_pitt", "distill_pitt_L", "msdb_adapt_s0", "msdb_adapt_seedavg", "msdb_adapt_kl",
         "msdb_adapt_dsp", "iie_ft_kl_sigma", "iie_ft_klcc_e20", "sum_ft_head", "sum_ft_head_klcc", "sum_ft", "sum_ft_klcc"]
for n in order:
    e = E[n]; c = e["cost"]; tc = e["training_cost"]
    extra = f" + {e['extra_unlabeled_images']:,}" if e["extra_unlabeled_images"] else ""
    maps = ("0 direct (800 via teacher)" if n == "distill_pitt_only" else str(e["add1000_maps_per_fold"])) + extra
    tr = (f"{tc['train_s_per_fold_median']:,.0f} ({HW_SHORT[tc['hardware']]})" if tc else
          "3 trainings (seed 0: msdb_adapt_s0 row)" if n == "msdb_adapt_seedavg" else "3 trainings, see seed rows")
    L.append(f"| {n} | {'F' if e['on_pareto_frontier_cc_vs_cpu'] else ''} | {ms(e,'CC')} | {ms(e,'SIM')} | {ms(e,'KLD')} | "
             f"{ptxt(c['params_trainable'])} / {ptxt(c['params_total'])} | {c['GFLOPs']:.0f} | {c['cpu_latency_s']:.3f} | "
             f"{c['mps_latency_s']:.3f} | {c['cpu_peak_rss_GB']:.2f} | {c['mps_driver_mem_GB']:.2f} | {maps} | "
             f"{e['vs_msdb_adapt']['dCC_mean']:+.4f} | {e['vs_msdb_adapt']['share_of_ours_to_msdb_gap_closed']:.0%} | {tr} |")
L.append("")
L.append("Row notes: seed-average rows (per-image mean of 3 trainings) and extra seeds are shown for stability but are "
         "not frontier candidates. Track A rows and distill_pitt_L are single trainings except L448 (3 seeds, SD of seed "
         "means 0.0002). The distilled rows are single seeds. SUM latency uses a pure-PyTorch selective scan and "
         "overstates SUM's latency relative to the CUDA kernel. sum_ft and sum_ft_klcc are single full fine-tunes per "
         "fold (seed 0) of all 57,501,196 SUM parameters on a cloud NVIDIA A10G (12 epochs, Adam 1e-4, StepLR(4, 0.1), "
         "effective batch 16, input 256, token 2); their post-hoc sigma per fold was "
         + " / ".join(f"{t} {SUM_FULL[t]['sigma_per_fold']}" for t in SUM_FULL)
         + ". Their raw 256 x 256 test predictions (runs mirror) were re-scored locally with the unified scorer and "
           "reproduce the per-image values to within 1.4e-10.\n")

L.append("## 1b. Training cost (per fold, as recorded by each row)\n")
L.append("Times are wall-clock seconds per fold from the row summaries (cost_table.json training_adaptation where it "
         "already lists the row). They were taken on different hardware and, on the Apple M4 Pro, while other tracks "
         "shared the CPU (MPS jobs held the shared MPS lock), so they are indicative and must not be compared across "
         "hardware without saying so. One-off costs listed in the last column are not included in the per-fold times.\n")
L.append("| Row | Hardware | median s/fold | total s (5 folds) | what the time covers | one-off cost not included |")
L.append("|---|---|---|---|---|---|")
for n in order:
    tc = E[n]["training_cost"]
    if not tc:
        continue
    L.append(f"| {n} | {tc['hardware']} | {tc['train_s_per_fold_median']:,.0f} | {tc['train_s_total_5_folds']:,.0f} | "
             f"{tc['covers']} | {tc['one_off_not_included'] or 'none recorded'} |")
L.append("")
_mps_rows = [n for n in order if E[n]["training_cost"] and E[n]["training_cost"]["hardware"] == HW["mps"]]
_mps_med = [E[n]["training_cost"]["train_s_per_fold_median"] for n in _mps_rows]
_ep = [x["seconds"] for x in json.load(open(os.path.join(RES, "sum_ft_summary.json")))["train_log_per_epoch"]]
L.append(f"Reading: on the same hardware (Apple M4 Pro GPU via MPS), every MPS row ({', '.join(_mps_rows)}) trains in "
         f"{min(_mps_med):,.0f} to {max(_mps_med):,.0f} s per fold (median) once features are cached; MSDB additionally "
         "needs its one-off feature cache (3,561 s for 1000 images on MPS; ours took 129 s on CPU, a different device). "
         "The SUM full fine-tune ran on a different machine (cloud NVIDIA A10G) and trains end to end from images, so its "
         f"time is not directly comparable to the Apple M4 Pro rows; its epochs took {min(_ep):.0f} to {max(_ep):.0f} s "
         f"each on the A10G ({len(_ep)} epochs over 5 folds).\n")

L.append("## 2. Pareto frontiers\n")
L.append("CC vs quiet CPU latency (primary):\n")
L.append("| Frontier point | CPU s | CC | what it buys over the previous point |")
L.append("|---|---|---|---|")
prev = None
for n in FRONT:
    e = E[n]
    if prev is None:
        what = f"fastest model class; +{e['accuracy']['CC']['mean'] - OURS['accuracy']['CC']['mean']:.4f} CC over ours B448 at identical cost"
    else:
        what = (f"+{e['accuracy']['CC']['mean'] - prev['accuracy']['CC']['mean']:.4f} CC for "
                f"{e['cost']['cpu_latency_s'] / prev['cost']['cpu_latency_s']:.1f}x the CPU latency")
    L.append(f"| {n} | {e['cost']['cpu_latency_s']:.3f} | {e['accuracy']['CC']['mean']:.4f} | {what} |")
    prev = e
L.append("")
L.append("Frontier membership on each cost axis (same candidates):\n")
L.append(f"* CC vs quiet CPU latency: {', '.join(FRONT)}")
L.append(f"* CC vs MPS latency: {', '.join(ROBUST['mps_latency'])}")
L.append(f"* CC vs GFLOPs: {', '.join(ROBUST['GFLOPs'])}")
_same = (ROBUST["cpu_latency_trackA_normalised_estimates"] == FRONT
         and ROBUST["cpu_latency_trackA_raw_contended_min (B448 rows at cost_table)"] == FRONT)
L.append("")
L.append("Replacing the quiet CPU latencies by Track A's superseded contended estimates (kept in frontier.json for audit "
         "only, not quoted) " + ("leaves the CPU frontier unchanged." if _same else "changes the CPU frontier (see frontier.json)."))
L.append("")

L.append("## 3. Paired tests of the best fast models (Holm)\n")
L.append("Family (primary): {" + ", ".join(models) + "} x {msdb_adapt_s0, ours_scratch_s0} = "
         f"{len(TESTS['primary']['members'])} comparisons, Holm per metric and per test. Wilcoxon = per-image signed rank on "
         "1000 out-of-fold images; NB = Nadeau-Bengio corrected t on the 5 fold differences (df 4, ratio 0.25). "
         "CI = 5000-sample bootstrap of the mean per-image difference (rng seed 0, the convention of this section since it "
         "was first compiled). img = share of images where the model is better.\n")
L.append("| Model | Reference | Metric | Δ (model - ref) [95% CI] | folds model better | img | p Wilcoxon (Holm) | p NB (Holm) | dz |")
L.append("|---|---|---|---|---|---|---|---|---|")
for fam_key in ("primary", "secondary_seedavg"):
    res = TESTS[fam_key]["results"]
    for mdl in models:
        for ref, r in res[mdl].items():
            for m in METRICS:
                c = r[m]
                L.append(f"| {mdl} | {ref} | {m} | {c['mean_diff']:+.4f} [{cif(c['boot95'][0])}, {cif(c['boot95'][1])}] | "
                         f"{c['folds_model_better']}/5 | {c['frac_images_model_better']:.1%} | {pf(c['wilcoxon_p'])} "
                         f"({pf(c['wilcoxon_p_holm'])}) | {pf(c['nb_p'])} ({pf(c['nb_p_holm'])}) | {c['dz']:+.2f} |")
    if fam_key == "primary":
        L.append("| | | | **secondary family: seed-averaged references (Holm within these 4)** | | | | | |")
L.append("")
if xchk:
    L.append("Cross-check: recomputed tests match Track B's stored tests for "
             + ", ".join(f"{k} (ΔCC {v['mean_diff_here']:+.5f} vs {v['mean_diff_trackB']:+.5f}; NB p {v['nb_p_here']:.4f} vs {v['nb_p_trackB']:.4f})" for k, v in xchk.items())
             + ".\n")

L.append("## 3b. Paired tests of the full SUM fine-tune (Holm)\n")
L.append("Differences = sum_ft minus reference (for KLD, negative = sum_ft better). Primary Holm family: sum_ft vs {"
         + ", ".join(SUM_REFS) + f"}} = {len(SUM_REFS)} comparisons, Holm per metric and per test (Wilcoxon, NB as in "
         "section 3). Secondary: sum_ft vs msdb_adapt_seedavg (per-image mean of MSDB seeds 0 to 2), one comparison, no "
         f"correction. CI = {B_BOOT_SUM:,}-sample paired bootstrap of the mean per-image difference, fresh numpy "
         f"default_rng({SEED_SUM}) per comparison (the 5,000-sample rng-0 CI of section 3 is stored in frontier.json as "
         "boot95_5000_seed0). Fold SD uses ddof 1.\n")
L.append("| Reference | Metric | Δ (sum_ft - ref) [95% CI] | folds sum_ft better | img | p Wilcoxon (Holm) | p NB (Holm) | dz |")
L.append("|---|---|---|---|---|---|---|---|")
for fam_key in ("primary", "secondary_seedavg"):
    for ref, r in TESTS_SUM[fam_key]["results"].items():
        for m in METRICS:
            c = r[m]
            L.append(f"| {ref} | {m} | {c['mean_diff']:+.4f} [{cif(c['boot95'][0])}, {cif(c['boot95'][1])}] | "
                     f"{c['folds_model_better']}/5 | {c['frac_images_model_better']:.1%} | {pf(c['wilcoxon_p'])} "
                     f"({pf(c['wilcoxon_p_holm'])}) | {pf(c['nb_p'])} ({pf(c['nb_p_holm'])}) | {c['dz']:+.2f} |")
    if fam_key == "primary":
        L.append("| **secondary: seed-averaged MSDB (single comparison, no correction)** | | | | | | | |")
L.append("")
if xchk_sum:
    _ok = all(abs(v["mean_diff_here"] - v["minus_mean_diff_main_table"]) < 1e-12
              and abs(v["wilcoxon_p_here"] - v["wilcoxon_p_main_table"]) <= 1e-12 * max(1.0, v["wilcoxon_p_main_table"])
              and abs(v["nb_p_here"] - v["nb_p_main_table"]) < 1e-12 for v in xchk_sum.values())
    L.append("Cross-check: sum_ft vs ours_scratch_s0 here " + ("matches" if _ok else "DOES NOT match")
             + " main_table.json (ours minus sum_ft there, same Wilcoxon and NB p): "
             + "; ".join(f"{m} Δ {v['mean_diff_here']:+.5f} vs {v['minus_mean_diff_main_table']:+.5f}, NB p "
                         f"{v['nb_p_here']:.5f} vs {v['nb_p_main_table']:.5f}" for m, v in xchk_sum.items())
             + ". Holm-adjusted p differ by design (different families).\n")

L.append("## 3c. Zero-shot to adapted gain per model family\n")
L.append("All rows from main_table.json (same folds and scoring). Zero-shot = released weights with no ADD1000 map used "
         "(SUM token 2; MSDB and IIE with a uniform prior). Published-prior zero-shot = MSDB and IIE with the published "
         "2-fold ADD1000 centre prior (500 maps of the other half; not fold-safe). Calibrated zero-shot = released weights "
         "plus fold-safe post-hoc calibration on the fold's training maps (SUM token + sigma; MSDB / IIE fold prior + "
         "sigma). Adapted = the family's adapted row on the 800 training maps per fold. CC mean ± fold SD (ddof 1).\n")
L.append("| Family | zero-shot CC | published-prior zero-shot CC | calibrated zero-shot CC | adapted row | adapted CC | gain from zero-shot (CC / SIM / KLD) | gain from published-prior zero-shot (CC) | gain from calibrated (CC) |")
L.append("|---|---|---|---|---|---|---|---|---|")
for fam_, *_ in GAIN_DEF:
    g = GAINS[fam_]
    zp = g.get("zero_shot_published_prior")
    L.append(f"| {fam_} | {g['zero_shot']['CC']['mean']:.4f} ± {g['zero_shot']['CC']['sd_folds_ddof1']:.4f} ({g['zero_shot_row']}) | "
             + (f"{zp['CC']['mean']:.4f} ± {zp['CC']['sd_folds_ddof1']:.4f} ({g['zero_shot_published_prior_row']})" if zp else "n/a (no prior in SUM)")
             + f" | {g['calibrated_zero_shot']['CC']['mean']:.4f} ({g['calibrated_zero_shot_row']}) | {g['adapted_row']} | "
             f"{g['adapted']['CC']['mean']:.4f} ± {g['adapted']['CC']['sd_folds_ddof1']:.4f} | "
             f"{g['gain_from_zero_shot']['CC']:+.4f} / {g['gain_from_zero_shot']['SIM']:+.4f} / {g['gain_from_zero_shot']['KLD']:+.4f} | "
             + (f"{g['gain_from_published_prior_zero_shot']['CC']:+.4f}" if zp else "n/a")
             + f" | {g['gain_from_calibrated_zero_shot']['CC']:+.4f} |")
L.append("| Ours | n/a | n/a | n/a | ours_scratch_s0 | "
         f"{OURS['accuracy']['CC']['mean']:.4f} ± {OURS['accuracy']['CC']['sd_folds_ddof1']:.4f} | n/a (no zero-shot "
         "version of our readout on ADD1000) | n/a | n/a |")
L.append("")
_rk = lambda k: " > ".join(f"{f} {GAINS[f][k]['CC']['mean']:.4f}" for f in RANK[{"zero_shot": "zero_shot_no_add1000_maps", "calibrated_zero_shot": "calibrated_zero_shot_fold_safe", "adapted": "adapted"}[k]])  # noqa: E731
L.append(f"Ranking by CC. Zero-shot (no ADD1000 maps): {_rk('zero_shot')}. Calibrated zero-shot (fold-safe): "
         f"{_rk('calibrated_zero_shot')}. Adapted: {_rk('adapted')}; ours B448 ({OURS['accuracy']['CC']['mean']:.4f}) "
         f"and our best fast model {bf['row']} ({bf['accuracy']['CC']['mean']:.4f}) sit between SUM and IIE.")
_g_ms, _g_sum, _g_iie = GAINS["DeepGaze MSDB"], GAINS["SUM"], GAINS["DeepGaze IIE"]
L.append(f"Reading: zero-shot rank does not predict adapted rank. MSDB is last zero-shot (uniform prior) and first once "
         f"adapted (gain {_g_ms['gain_from_zero_shot']['CC']:+.4f} CC, or {_g_ms['gain_from_published_prior_zero_shot']['CC']:+.4f} "
         f"from the published-prior number); SUM is first zero-shot and second adapted (gain {_g_sum['gain_from_zero_shot']['CC']:+.4f}); "
         f"IIE is second zero-shot and last adapted (gain {_g_iie['gain_from_zero_shot']['CC']:+.4f}). Against fold-safe "
         f"calibrated zero-shot the gains are closer ({_g_sum['gain_from_calibrated_zero_shot']['CC']:+.4f} SUM, "
         f"{_g_ms['gain_from_calibrated_zero_shot']['CC']:+.4f} MSDB, {_g_iie['gain_from_calibrated_zero_shot']['CC']:+.4f} IIE). "
         "Note the adapted rows differ in what is trained: all SUM parameters, the MSDB head only (26,463 parameters), "
         "all 120 IIE readouts (3.06M parameters, frozen backbones).\n")
assert RANK["zero_shot_no_add1000_maps"] == ["SUM", "DeepGaze IIE", "DeepGaze MSDB"] and \
    RANK["adapted"] == ["DeepGaze MSDB", "SUM", "DeepGaze IIE"], "update the ranking prose"

L.append("## 4. Cost measurements and caveats\n")
L.append(f"* Quiet-machine re-measurement ({time.strftime('%d %b %Y')}, no other experiment running; load average 1.5 before the "
         "bench, up to 4.2 during it from the bench's own 6 threads): Track A could only measure CPU latency under "
         "contention and reported normalised estimates, which are superseded and not quoted here (they remain in "
         "frontier.json for audit). The Track A variants were re-measured with the unchanged scripts/trackA/cost_bench_a.py "
         f"(same protocol). Same-session check: B448 {B448_Q_CPU:.4f} s CPU vs {B448_CT['cpu_latency_median_s']:.4f} s in "
         f"cost_table.json (factor {SESSION['cpu_session_factor']:.3f}), MPS {B448_Q_MPS:.4f} s vs "
         f"{B448_CT['mps_latency_median_s']:.4f} s, so no normalisation is applied.")
for v, n in (("L448", "ours_L448_s0"), ("msB", "ours_ms_B_s0"), ("B644", "ours_B644_s0"), ("Lms", "ours_L_ms_s0")):
    c = E[n]["cost"]
    L.append(f"  * {v}: quiet CPU {c['cpu_latency_s']:.3f} s (IQR {c['cpu_latency_iqr_s'][0]:.3f} to "
             f"{c['cpu_latency_iqr_s'][1]:.3f}); MPS {c['mps_latency_s']:.3f} s.")
L.append(f"* CPU memory of the Track A backbones: their bench loads the backbone through a different path whose loaded-model "
         f"RSS is lower by a constant {RSS_OFFSET:.2f} GB (B448: {B448_Q_RSS:.2f} GB there vs {B448_CT['cpu_peak_rss_GB']:.2f} GB "
         "in cost_table.json, with the same activation increment), so their values are aligned by adding that offset; raw "
         "values are in frontier.json. MPS memory needs no alignment (B448 matches cost_table.json).")
L.append("* GFLOPs and parameter counts are deterministic (FlopCounterMode convention of cost_table.json); the quiet "
         f"bench reproduced every Track A GFLOP count exactly. SUM's {CT['sum']['GFLOPs']:.1f} GFLOPs exclude its selective "
         f"scan (estimated at {CT['sum']['selective_scan_GFLOPs_estimate_not_in_total']:.1f} GFLOPs more in cost_table.json); "
         f"with the scan estimate SUM is {CT['sum']['GFLOPs_incl_scan_estimate']:.1f} GFLOPs, still "
         f"{OURS['cost']['GFLOPs'] / CT['sum']['GFLOPs_incl_scan_estimate']:.1f}x fewer than ours B448.")
L.append("* Distilled students: inference cost equals the matching non-distilled network. Training needs the adapted MSDB "
         "teacher once over the 2,499 Pitt images (Track B: MSDB feature pass 3.6 s/img on MPS, 15 to 35 s/img on a "
         "contended CPU), a one-off training cost.")
L.append("* SUM inference cost is the released SUM network's (cost_table.json), unchanged by fine-tuning: 26.4 GFLOPs, "
         f"{CT['sum']['cpu_latency_median_s']:.3f} s CPU and {CT['sum']['mps_latency_median_s']:.3f} s MPS with the pure-PyTorch "
         "chunked selective scan, 2.29 GB CPU peak memory. The latency overstates SUM's speed penalty relative to a "
         "compiled CUDA scan kernel; no CUDA GPU was available for inference timing.")
_sdm = MT["seed_aggregates"]["msdb_adapt"]["CC"]["sd_over_seeds_ddof1"]
_sdo = MT["seed_aggregates"]["ours_scratch"]["CC"]["sd_over_seeds_ddof1"]
L.append("* Single training runs: distilled rows, the multi-scale/644 rows and both SUM full fine-tunes are seed 0 only. "
         f"The SD of seed means over 3 seeds is 0.0002 for L448, {_sdo:.4f} for ours_scratch and {_sdm:.4f} for MSDB "
         "adapted (ddof 1), far below the gaps between model families and below the "
         f"{-SF['vs_msdb_adapt']['dCC_mean']:.4f} CC difference between sum_ft and MSDB adapted; a single SUM training "
         "cannot show SUM's own seed spread, and that CC difference is not significant in any case.")
L.append("* MSDB 'adapted' trains only its 26,463-parameter head on cached frozen features, while sum_ft trains all of "
         "SUM; a fuller MSDB fine-tune could move MSDB up. Both SUM full fine-tune rows keep SUM's released recipe "
         "choices (optimiser, schedule, input size) with a fixed epoch count; no SUM-specific search was run on ADD1000.\n")
L.append("## Files\n")
for f in ("results/frontier.json", "results/FRONTIER.md", "results/frontier.png", "scripts/frontier/compile_frontier.py",
          "scripts/frontier/run_cost_quiet.sh", "results/cost_raw/frontier_quiet_<variant>_<dev>_p<k>.json (+ .stdout)",
          "results/sum_ft_summary.json", "results/sum_ft_klcc_summary.json", "runs/sum_ft/ckpt/sum_ft/",
          "runs/sum_ft_klcc/ckpt/sum_ft_klcc/", "results/archive_pre_sumft/ (versions before the SUM rows)",
          "logs/frontier_cost_quiet.log", "logs/frontier_compile.log"):
    L.append(f"* {os.path.join(KB, f)}")
open(os.path.join(RES, "FRONTIER.md"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------------------------ console
print(f"frontier (CC vs CPU): {FRONT}")
for k, v in ROBUST.items():
    print(f"  robustness {k}: {v}")
print(f"best fast: {best_fast['row']}  best at base cost: {best_base['row']}  reaching MSDB: {reaches}")
for n in order:
    e = E[n]; c = e["cost"]
    print(f"{n:26s} CC {ms(e,'CC')} SIM {e['accuracy']['SIM']['mean']:.4f} KLD {e['accuracy']['KLD']['mean']:.4f} "
          f"GF {c['GFLOPs']:7.1f} CPU {c['cpu_latency_s']:.3f} MPS {c['mps_latency_s']:.3f} RSS {c['cpu_peak_rss_GB']:.2f} "
          f"{'F' if e['on_pareto_frontier_cc_vs_cpu'] else ''}")
for fam_key in TESTS:
    for mdl, rr in TESTS[fam_key]["results"].items():
        for ref, r in rr.items():
            c = r["CC"]
            print(f"[{fam_key}] {mdl} vs {ref}: dCC {c['mean_diff']:+.4f} [{c['boot95'][0]:+.4f},{c['boot95'][1]:+.4f}] "
                  f"folds {c['folds_model_better']}/5 img {c['frac_images_model_better']:.1%} pW {c['wilcoxon_p']:.1e} "
                  f"(Holm {c['wilcoxon_p_holm']:.1e}) pNB {c['nb_p']:.4f} (Holm {c['nb_p_holm']:.4f})")
print("crosscheck:", json.dumps(xchk, indent=0))
print(f"done in {time.time() - T0:.1f} s")
print(f"frontier (CC vs MPS): {ROBUST['mps_latency']}  (CC vs GFLOPs): {ROBUST['GFLOPs']}  within 0.002 of MSDB: {within_0002}")
for fam_key in ("primary", "secondary_seedavg"):
    for ref, r in TESTS_SUM[fam_key]["results"].items():
        for m in METRICS:
            c = r[m]
            print(f"[sum_ft {fam_key}] vs {ref} {m}: d {c['mean_diff']:+.4f} [{c['boot95'][0]:+.4f},{c['boot95'][1]:+.4f}] "
                  f"folds {c['folds_model_better']}/5 img {c['frac_images_model_better']:.1%} pW {c['wilcoxon_p']:.2e} "
                  f"(Holm {c['wilcoxon_p_holm']:.2e}) pNB {c['nb_p']:.4f} (Holm {c['nb_p_holm']:.4f})")
print("crosscheck sum_ft vs main_table:", json.dumps(xchk_sum))
print("ranking:", json.dumps(RANK))
