"""
Facts pack for the KBS short communication (ADD1000: adapt before you compare).

Builds facts.json and FACTS.md in facts/ from the released results files only. Every fact carries
its value, unit, source file and JSON path (or 'computed here' with its inputs) and the definition of
the number. Means, fold SDs and the key paired tests are recomputed from the per-image files and checked
against the compiled JSONs; mismatches are listed under "checks" and "warnings".

  python -I scripts/stats/build_facts.py      # writes facts/facts.json

Nothing is trained, scored or downloaded; inputs are read-only.
"""
import sys
sys.dont_write_bytecode = True
import json, os, math, re, datetime, hashlib
import numpy as np
from scipy import stats

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RES = os.path.join(KB, "results")
SPL = os.path.join(KB, "splits")
OUT = os.path.join(KB, "facts")
ADD_PDF = "Liang et al. (2021), J. Vis. Commun. Image Represent. 81, 103356 (dataset paper)"
ADD_README = "github.com/lscumt/ADD1000 README.md"
AUDIT = os.path.join(KB, "facts", "inputs", "add1000_release_audit.json")
METRICS = ("CC", "SIM", "KLD")
HIGHER = {"CC": True, "SIM": True, "KLD": False}

_cache = {}


def J(path):
    p = path if os.path.isabs(path) else os.path.join(RES, path)
    if p not in _cache:
        with open(p) as f:
            _cache[p] = json.load(f)
    return _cache[p]


def rel(p):
    for base, tag in ((RES, "results/"), (SPL, "splits/"), (KB, "")):
        if p.startswith(base + "/"):
            return tag + p[len(base) + 1:]
    return p


def resolve(obj, path):
    """path tokens separated by '/'; 'key[field=value]' selects a list element, 'key[i]' an index."""
    cur = obj
    for tok in path.split("/"):
        m = re.match(r"^(.*?)((?:\[[^\]]+\])*)$", tok)
        key, sels = m.group(1), re.findall(r"\[([^\]]+)\]", m.group(2))
        if key:
            cur = cur[key]
        for s in sels:
            if "=" in s:
                f, v = s.split("=", 1)
                hits = [x for x in cur if str(x.get(f)) == v]
                assert len(hits) == 1, (path, s, len(hits))
                cur = hits[0]
            else:
                cur = cur[int(s)]
    return cur


FACTS, WARN, CHECKS = [], [], []
_ids = set()


def fact(fid, section, value=None, unit="", src=None, path=None, definition="", sd_ddof=None, note=None,
         recomputed=None, tol=None, display=None, inputs=None):
    """src+path: value is READ from the file (value arg, if given, must agree). src='computed': value given,
    inputs lists the source files/paths. recomputed: independent recomputation from per-image files."""
    assert fid not in _ids, fid
    _ids.add(fid)
    if src and src != "computed" and path is not None:
        v = resolve(J(src), path)
        if value is not None and not _close(value, v, 1e-12):
            WARN.append(f"{fid}: given value {value} != file value {v} ({rel(src)}::{path})")
        value = v
    f = {"id": fid, "section": section, "value": value, "unit": unit,
         "source": (rel(src if os.path.isabs(src) else os.path.join(RES, src)) if src and src != "computed" else src),
         "json_path": path, "definition": definition}
    if inputs:
        f["inputs"] = inputs
    if sd_ddof is not None:
        f["sd_ddof"] = sd_ddof
    if display is not None:
        f["display"] = display
    if note:
        f["note"] = note
    if recomputed is not None:
        t = tol if tol is not None else 1e-9
        ok = _close(recomputed, value, t)
        f["recomputed"] = recomputed
        f["recompute_match"] = ok
        CHECKS.append({"id": fid, "file_value": value, "recomputed": recomputed, "tol": t, "match": ok})
        if not ok:
            WARN.append(f"MISMATCH {fid}: file {value} vs recomputed {recomputed} (tol {t})")
    FACTS.append(f)
    return value


def _close(a, b, tol):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float, np.floating, np.integer)) and isinstance(b, (int, float, np.floating, np.integer)):
        a, b = float(a), float(b)
        if math.isnan(a) and math.isnan(b):
            return True
        if abs(b) < 1e-300 or abs(a) < 1e-300:
            return abs(a - b) <= tol
        # p-values span many decades: accept relative agreement as well
        return abs(a - b) <= tol or abs(a - b) / max(abs(a), abs(b)) <= 1e-6
    return a == b


def d4(x):
    return f"{x:.4f}"


def pfmt(p):
    if p is None:
        return "n/a"
    if p >= 0.001:
        return f"{p:.3f}"
    return f"{p:.1e}"


# ------------------------------------------------------------------------------------------ splits
KF = J(os.path.join(SPL, "kfold_seed0.json"))
GR = J(os.path.join(SPL, "groups.json"))
PS = J(os.path.join(SPL, "kfold_seed0_pairsafe.json"))
ND = J(os.path.join(SPL, "near_dup_pairs.json"))
FOLD_OF = {}
for name, sp in (("random", KF), ("grouped", GR), ("pairsafe", PS)):
    FOLD_OF[name] = {i: k for k, f in enumerate(sp["folds"]) for i in f}
SIZES = {name: [len(f) for f in sp["folds"]] for name, sp in (("random", KF), ("grouped", GR), ("pairsafe", PS))}


# ------------------------------------------------------------------------------------------ per-image
def load_pi(fname, seed=None, fields=None, flt=None, split="random", n_expect=1000):
    d = J(fname)
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
    assert len(out) == n_expect, (fname, len(out))
    if n_expect == 1000:
        bad = [i for i, r in out.items() if FOLD_OF[split][i] != r["fold"]]
        assert not bad, (fname, "fold mismatch", bad[:3])
    return out


def avg_rows(rowsets):
    ids = sorted(rowsets[0], key=int)
    out = {}
    for i in ids:
        out[i] = {"fold": rowsets[0][i]["fold"], **{m: float(np.mean([rs[i][m] for rs in rowsets])) for m in METRICS}}
    return out


def vec(rows, m, ids=None):
    ids = ids or sorted(rows, key=int)
    return np.array([rows[i][m] for i in ids])


def fold_means(rows, m, nf=5):
    fm = []
    for k in range(nf):
        v = [r[m] for r in rows.values() if r["fold"] == k]
        fm.append(float(np.mean(v)))
    return np.array(fm)


def desc(rows, m, nf=5):
    fm = fold_means(rows, m, nf)
    return {"mean_of_fold_means": float(fm.mean()), "mean_over_images": float(vec(rows, m).mean()),
            "sd_ddof1": float(fm.std(ddof=1)), "sd_ddof0": float(fm.std(ddof=0)), "per_fold": fm.tolist()}


def holm(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min(1.0, (n - rank) * p[idx]))
        adj[idx] = run
    return adj


def nb(dk, ratio):
    K = len(dk)
    m, v = dk.mean(), dk.var(ddof=1)
    if v == 0:
        return float("inf"), 0.0
    t = m / math.sqrt((1.0 / K + ratio) * v)
    return float(t), float(2 * stats.t.sf(abs(t), K - 1))


def rank_biserial(d):
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def compare(a_rows, b_rows, split="random", B=5000, seed=0, rng=None):
    """differences = a - b, paired by image id; same protocol as compile_results.compare."""
    nf = len(SIZES[split])
    ratio = float(np.mean([s / (1000 - s) for s in SIZES[split]]))
    rng = rng if rng is not None else np.random.default_rng(seed)
    ids = sorted(a_rows, key=int)
    assert sorted(b_rows, key=int) == ids
    o = {}
    for m in METRICS:
        a, b = vec(a_rows, m, ids), vec(b_rows, m, ids)
        d = a - b
        better = (d > 0) if HIGHER[m] else (d < 0)
        w = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        idx = rng.integers(0, len(d), size=(B, len(d)))
        bm = d[idx].mean(1)
        dk = fold_means(a_rows, m, nf) - fold_means(b_rows, m, nf)
        t, p = nb(dk, ratio)
        o[m] = {"mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                "boot95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                "dz": float(d.mean() / d.std(ddof=1)), "rank_biserial": rank_biserial(d),
                "frac_images_first_better": float(better.mean()), "wilcoxon_p": float(w.pvalue),
                "fold_diffs": dk.tolist(), "folds_first_better": int(((dk > 0) if HIGHER[m] else (dk < 0)).sum()),
                "nb_t": t, "nb_p": p, "nb_ratio": ratio}
    return o


# ------------------------------------------------------------------------------------------ rows
MT = J("main_table.json")
FR = J("frontier.json")
CT = J("cost_table.json")
ROWINFO = {r["row"]: r for r in MT["rows"]}


def rows_for(name):
    """per-image rows of a main-table row (or frontier/distill row), built exactly from the files."""
    if name in ("ours_scratch_seedavg", "ours_warm_seedavg") or (name.startswith("abl_") and name.endswith("_seedavg")):
        base = name[:-len("_seedavg")]
        return avg_rows([load_pi(f"{base}_s{s}_per_image.json") for s in range(3)])
    if name == "msdb_adapt_seedavg":
        return avg_rows([load_pi("msdb_adapt_per_image.json", seed=s) for s in range(3)])
    if name in ("msdb_adapt_s0", "msdb_adapt_s1", "msdb_adapt_s2"):
        return load_pi("msdb_adapt_per_image.json", seed=int(name[-1]))
    if name in ("sum_ft_head_nosigma", "sum_ft_head_klcc_nosigma"):
        return load_pi(name.replace("_nosigma", "") + "_per_image.json", fields={m: m + "_nosigma" for m in METRICS})
    if name in ("msdb_adapt_grouped", "msdb_adapt_pairsafe"):
        return load_pi(f"{name}_per_image.json", seed=0, split=name.split("_")[-1])
    if name.endswith("_grouped") or name.endswith("_grouped_nosigma"):
        return load_pi(f"{name}_per_image.json", split="grouped")
    if name.endswith("_pairsafe") or name.endswith("_pairsafe_nosigma"):
        return load_pi(f"{name}_per_image.json", split="pairsafe")
    if name == "ours_L448_seedavg":
        return avg_rows([load_pi(f"ours_L448_s{s}_per_image.json") for s in range(3)])
    return load_pi(f"{FN(name)}_per_image.json")


# Revision rows that depend on the MIT1003 centre-bias template (24 Nov 2026, author decision): read from the run with
# DeepGaze's distributed template (cache/revision/deepgaze_template/centerbias_mit1003.npy, sha256
# 051645b7...e9), files with suffix _offtpl. The logical row names (and so every fact id) are unchanged. The calibrated
# DeepGaze rows do not use the template; they come from the same _offtpl run.
REVFILE = {"msdb_zeroshot_mit1003": "msdb_zeroshot_mit1003_offtpl", "iie_zeroshot_mit1003": "iie_zeroshot_mit1003_offtpl",
           "iie_zeroshot_mit1003_native": "iie_zeroshot_mit1003_native_offtpl",
           "msdb_calib_wide": "msdb_calib_wide_offtpl", "iie_calib_wide": "iie_calib_wide_offtpl"}
REVCURVE = {"msdb": "calib_curve_msdb_offtpl.json", "iie": "calib_curve_iie_offtpl.json", "sum": "calib_curve_sum.json"}


def FN(name):
    """results file stem of a logical row name"""
    return REVFILE.get(name, name)


ROWS = {}


def R(name):
    if name not in ROWS:
        ROWS[name] = rows_for(name)
    return ROWS[name]


def split_of(name):
    if "grouped" in name:
        return "grouped"
    if "pairsafe" in name:
        return "pairsafe"
    return "random"


# ========================================================================================== 1 dataset
SEC = "dataset"
ADDREF = ("Liang, Liu, Qian, 'Fixation prediction for advertising images: Dataset and benchmark', "
          "J. Vis. Commun. Image Represent. 81 (2021) 103356, doi 10.1016/j.jvcir.2021.103356; " + ADD_PDF)


def dfact(fid, value, unit, where, definition, note=None):
    f = {"id": fid, "section": SEC, "value": value, "unit": unit, "source": ADD_PDF, "json_path": None,
         "location": where, "definition": definition, "reference": ADDREF}
    if note:
        f["note"] = note
    assert fid not in _ids
    _ids.add(fid)
    FACTS.append(f)


dfact("dataset.n_images", 1000, "images", "abstract; Sec. 3.1", "advertising images in ADD1000",
      note="1000 images and 1000 maps in the release (counted when ADD1000_DIR is present)")
_flat = os.path.join(os.environ.get("ADD1000_DIR", os.path.join(KB, "data", "ADD1000")), "flat")
if os.path.isdir(_flat):   # optional check of a local ADD1000 copy (not redistributed)
    nimg = len(os.listdir(os.path.join(_flat, "images")))
    nmap = len(os.listdir(os.path.join(_flat, "maps")))
    assert nimg == 1000 and nmap == 1000, (nimg, nmap)
dfact("dataset.n_observers_total", 57, "observers", "Sec. 3.2 (also abstract and repo README)",
      "observers recruited in total (24 male, 33 female)")
dfact("dataset.observers_male", 24, "observers", "Sec. 3.2", "male observers")
dfact("dataset.observers_female", 33, "observers", "Sec. 3.2", "female observers")
dfact("dataset.observer_age_mean", 21.7, "years", "Sec. 3.2", "mean age (min 18, max 30, SD 2.88, median 21)")
dfact("dataset.observer_age_range", [18, 30], "years", "Sec. 3.2", "age range")
dfact("dataset.observer_background", "34 undergraduate, 20 graduate, 3 doctoral students, China University of Mining "
      "and Technology; normal or corrected-to-normal vision; naive to the purpose", "", "Sec. 3.2", "observer pool")
dfact("dataset.every_observer_viewed_all_images", True, "", "Sec. 3.2",
      "the paper states that all 1000 images were viewed by every observer (8 cohorts of 125 images, each observer "
      "4 sessions at least 3 days apart, 2 cohorts per session)")
dfact("dataset.valid_individual_records", 56978, "observer-image recordings", "Sec. 3.5",
      "valid individual eye-movement recordings over the 1000 images (57 observers x 1000 images = 57,000 possible)",
      note="57,000 - 56,978 = 22 recordings were invalid, so not every map aggregates exactly 57 observers; the "
           "paper gives no per-image observer count. Safe wording: '57 observers, each of whom viewed all 1000 "
           "images (56,978 valid recordings)'. Do NOT write '57 observers per image'.")
dfact("dataset.invalid_records_derived", 57 * 1000 - 56978, "recordings", "derived from Sec. 3.2 and 3.5",
      "57 x 1000 minus valid recordings")
dfact("dataset.n_fixations_total", 1025617, "fixations", "Sec. 3.5", "fixations in total")
dfact("dataset.fixations_per_record_mean", 18, "fixations", "Sec. 3.5",
      "the paper's 'on average 18 fixations per image'; arithmetically this is per individual recording "
      "(1,025,617 / 56,978 = 18.0), not per aggregated map", note="phrase as per viewing, not per image")
dfact("dataset.total_viewing_s", 254176, "s", "Sec. 3.5", "total viewing duration")
dfact("dataset.mean_viewing_s", 4.461, "s", "Sec. 3.5",
      "average viewing time per image per viewing (254,176 s / 56,978 = 4.461 s); free, self-paced viewing (no "
      "fixed presentation time, Sec. 3.3)")
dfact("dataset.eye_tracker", "EyeLink 1000 Plus, monocular (dominant eye) 500 Hz, remote mode with chin support",
      "", "Sec. 3.3", "recording device")
dfact("dataset.display", "19-inch LCD, 1280 x 1024, 60 Hz; images at original resolution on grey (125) background; "
      "nine-point calibration at least twice per subject", "", "Sec. 3.3", "display")
dfact("dataset.image_size_range", "279 x 411 (min) to 1024 x 843 (max)", "pixels", "Sec. 3.1", "variable image sizes")
dfact("dataset.content_categories", 12, "categories", "Sec. 3.1, Fig. 3",
      "content categories (alcohol, tea, home appliance, vehicle, food, toiletry, beverage, tourism, electronic "
      "product, real estate, clothing, others); no per-image labels released")
dfact("dataset.attribute_dimensions", "4 bipolar dimensions, 8 attributes, about 250 images per pair", "",
      "Sec. 3.1", "(calming, exciting), (retro, fashional), (aesthetic, utilitarian), (emotional, rational)")
dfact("dataset.market_language", "mostly Chinese-market ads with Chinese or English text; 20 images with a celebrity "
      "spokesperson; no prices", "", "Sec. 3.1", "content notes")
dfact("dataset.gt_gaussian_sigma", 50, "pixels (as stated)", "Sec. 3.4",
      "ground truth = average over observers of 2-D Gaussians placed on fixations, sigma 50, height by fixation "
      "duration; the first fixation is deleted only if shorter than 90 ms")
dfact("dataset.paper_protocol", "5-fold CV; per fold 800 train, 100 validation, 100 test; 5 runs per fold, the "
      "'moderate' run reported", "", "Sec. 4.4", "the dataset paper's own protocol (ours: 800 train / 200 test, "
      "no validation split; folds differ)")
dfact("dataset.telsp_reported_CC", 0.825, "CC", "Table 1", "TEL-SP (the dataset paper's ad-specific model) on its "
      "own 5-fold protocol", note="NOT comparable: different folds, 100-image test sets, different GT "
      "normalisation and scorer; no code, weights or predicted maps are public (see dataset.telsp_availability)")
dfact("dataset.telsp_reported_SIM", 0.812, "SIM", "Table 1", "TEL-SP SIM on its own protocol (not comparable)")
dfact("dataset.telsp_params_gflops", "1.1M parameters, 6.4 GFLOPs", "", "Sec. 4.3", "TEL-SP size as stated")
FACTS.append({"id": "dataset.telsp_availability", "section": SEC,
              "value": resolve(J(AUDIT), "liang_group_repos/TEL-SP"), "unit": "", "source": AUDIT,
              "json_path": "liang_group_repos/TEL-SP",
              "definition": "availability of the dataset paper's own predictor (searched 23 Nov 2026)",
              "note": "the dataset paper says 'dataset and code' are at github.com/lscumt/ADD1000; the repo holds "
                      "data zips + README only (" + J(AUDIT)["liang_group_repos"]["https://github.com/lscumt/ADD1000"] + ")"})
_ids.add("dataset.telsp_availability")
FACTS.append({"id": "dataset.released_content", "section": SEC, "value": resolve(J(AUDIT), "add1000_metadata"),
              "unit": "", "source": AUDIT, "json_path": "add1000_metadata",
              "definition": "what the public release contains (density maps only; no labels)"})
_ids.add("dataset.released_content")
FACTS.append({"id": "dataset.readme_observer_statement", "section": SEC,
              "value": "README: eye-movement data of 57 subjects on 1000 ad images", "unit": "",
              "source": ADD_README, "json_path": None, "definition": "repository README statement (total subjects)"})
_ids.add("dataset.readme_observer_statement")

# ========================================================================================== 2 protocol
SEC = "protocol"
fact("protocol.folds_file_description", SEC, src=os.path.join(SPL, "kfold_seed0.json"), path="description",
     definition="seed-0 split used by every main row")
perm = np.random.default_rng(0).permutation(1000)
ids_sorted = KF["ids_sorted"]
rebuilt = [[ids_sorted[j] for j in perm[200 * k:200 * (k + 1)]] for k in range(5)]
ok_perm = rebuilt == KF["folds"]
fact("protocol.n_folds", SEC, 5, "folds", "computed", definition="number of folds in splits/kfold_seed0.json",
     inputs=["splits/kfold_seed0.json folds"], recomputed=len(KF["folds"]))
fact("protocol.test_per_fold", SEC, 200, "images", "computed", definition="test images per fold (all 5 folds)",
     inputs=["splits/kfold_seed0.json folds"], recomputed=int(min(SIZES["random"])))
fact("protocol.train_per_fold", SEC, 800, "images", "computed", definition="training images per fold (other 4 folds)",
     recomputed=1000 - int(max(SIZES["random"])))
allids = [i for f in KF["folds"] for i in f]
assert len(set(allids)) == 1000 and sorted(allids, key=int) == ids_sorted
fact("protocol.split_rule_reproduced", SEC, True, "", "computed",
     definition="folds = numpy default_rng(0).permutation(1000) over numerically sorted ids, 5 consecutive chunks "
                "of 200; reproduced here with numpy " + np.__version__, recomputed=bool(ok_perm))
fact("protocol.scoring", SEC, src="main_table.json", path="description",
     definition="common scoring grid and normalisation for every row")
fact("protocol.metrics", SEC, src="main_table.json", path="protocol/metrics", definition="metrics and KLD epsilon")
fact("protocol.statistics", SEC, src="main_table.json", path="stats_notes",
     definition="test definitions (Wilcoxon, Nadeau-Bengio, Holm, effect sizes) used by main_table.json")
NATIVE = [
    ("protocol.native.ours", "ours_scratch_s0_summary.json", "protocol/backbone",
     "ours: frozen DINOv2-base at 448x448 (PIL bilinear stretch, ImageNet mean/std); 32x32 patch grid"),
    ("protocol.native.ours_output", "ours_scratch_s0_summary.json", "protocol/native_resolution",
     "ours: output resolution (logits upsampled to 384 inside the model)"),
    ("protocol.native.iie", "iie_zeroshot_uniform_summary.json", "protocol",
     "DeepGaze IIE input 384x384 PIL bilinear; predicts at 384"),
    ("protocol.native.iie_backbone_input", "cost_table.json", "inference/iie/note",
     "DeepGaze IIE backbones see the image downsampled by 2 (192x192)"),
    ("protocol.native.msdb", "msdb_adapt_summary.json", "input",
     "DeepGaze MSDB input 512x512 PIL bilinear, pixel_per_dva 21.7, generalisation mode"),
    ("protocol.native.msdb_scoring", "msdb_zeroshot_uniform_summary.json", "scoring",
     "MSDB 512 density (nearest 2x upsample of its 256 map) to 384 by torch bilinear"),
    ("protocol.native.sum", "sum_ft_summary.json", "protocol",
     "SUM input 256x256 stretch resize + ImageNet normalisation, token 2; 256 output PIL-bilinear to 384"),
    ("protocol.native.sum_zeroshot", "sum_zeroshot_t2_summary.json", "protocol", "SUM zero-shot scoring protocol"),
]
for fid, src, pth, dfn in NATIVE:
    fact(fid, SEC, src=src, path=pth, definition=dfn)
fact("protocol.input_px.ours", SEC, 448, "px (square)", "computed", definition="native input side, ours (DINOv2-B/14)",
     inputs=["results/cost_table.json inference/ours/input"], recomputed=int(CT["inference"]["ours"]["input"].split("x")[0]))
fact("protocol.input_px.iie", SEC, 384, "px (square)", "computed", definition="native input side, DeepGaze IIE",
     inputs=["results/cost_table.json inference/iie/input"], recomputed=int(CT["inference"]["iie"]["input"].split("x")[0]))
fact("protocol.input_px.msdb", SEC, 512, "px (square)", "computed", definition="native input side, DeepGaze MSDB",
     inputs=["results/cost_table.json inference/msdb/input"], recomputed=int(CT["inference"]["msdb"]["input"].split("x")[0]))
fact("protocol.input_px.sum", SEC, 256, "px (square)", "computed", definition="native input side, SUM",
     inputs=["results/cost_table.json inference/sum/input"], recomputed=int(CT["inference"]["sum"]["input"].split("x")[0]))
fact("protocol.scoring_grid_px", SEC, 384, "px (square)", "computed",
     definition="common scoring grid: GT PIL-bilinear to 384x384; predictions bilinear to 384 when native differs; "
                "clip at 0 and divide by sum; KLD eps 1e-8", inputs=["results/main_table.json description"])
fact("protocol.sigma_rule.kfold", SEC, src="ours_scratch_s0_summary.json", path="protocol/finalizer",
     definition="post-hoc Gaussian finalizer: sigma grid {0,2,...,12} px at 384, chosen by mean CC on 200 random "
                "training maps of the fold (k-fold rows; different random draws per track; never test maps)",
     note="SUM/MSDB tracks draw with numpy default_rng(fold); curves choose sigma on all n training maps")
fact("protocol.sigma_rule.curves", SEC, src="iie_curve_klcce20_summary.json", path="sigma_rule",
     definition="learning curves: sigma chosen on all n training maps")
fact("protocol.sigma.msdb_foldprior_per_fold", SEC, src="msdb_zeroshot_foldprior_summary.json",
     path="posthoc_sigma_per_fold", definition="calibrated zero-shot MSDB: sigma per fold (12 = grid edge in all folds)")
fact("protocol.sigma.sum_ft_per_fold", SEC, src="frontier.json", path="sum_full_finetune/sum_ft/sigma_per_fold",
     definition="sum_ft post-hoc sigma per fold")
fact("protocol.sigma.iie_klcc_e20_per_fold", SEC, src="iie_ft_klcc_e20_summary.json", path="sigma_per_fold",
     definition="iie_ft_klcc_e20 post-hoc sigma per fold")
fact("protocol.sigma.iie_kl_sigma_per_fold", SEC, src="iie_ft_kl_sigma_summary.json", path="sigma_per_fold",
     definition="iie_ft_kl_sigma post-hoc sigma per fold")
fact("protocol.sigma.ours_scratch_per_fold", SEC, [x["sigma"] for x in J("ours_scratch_s0_summary.json")["folds_detail"]],
     "px", "computed", definition="ours_scratch_s0 post-hoc sigma per fold",
     inputs=["results/ours_scratch_s0_summary.json folds_detail[*]/sigma"])
fact("protocol.recipe.ours", SEC, src="ours_scratch_s0_summary.json", path="protocol",
     definition="ours: readout recipe (60 epochs, AdamW 3e-4 wd 1e-4, batch 16, KL+(1-CC) at 64x64)")
fact("protocol.recipe.ours_centre_prior", SEC, src="ours_scratch_s0_summary.json",
     path="calibration_and_training_data/centre_prior", definition="ours: centre prior is learned with the readout")
fact("protocol.recipe.msdb_head", SEC, src="msdb_adapt_summary.json", path="what_is_trained",
     definition="MSDB adapted: what is trained (head 26,460 + 3 finalizer scalars; backbone and scale mix frozen)")
fact("protocol.recipe.msdb_head_loss", SEC, src="msdb_adapt_summary.json", path="loss", definition="MSDB adapted loss")
fact("protocol.recipe.msdb_head_epochs", SEC, src="msdb_adapt_summary.json", path="epochs", unit="epochs",
     definition="MSDB adapted epochs (our recipe copied; no MSDB-specific search)")
fact("protocol.recipe.msdb_head_lr", SEC, src="msdb_adapt_summary.json", path="lr", definition="MSDB adapted AdamW lr")
fact("protocol.recipe.msdb_calibration", SEC, src="msdb_adapt_summary.json", path="calibration_maps",
     definition="MSDB adapted: prior from the fold's training maps, sigma on 200 of them")
fact("protocol.recipe.iie_selected", SEC, src="iie_ft_klcc_e20_summary.json", path="recipe",
     definition="IIE adapted, selected recipe (iie_ft_klcc_e20)")
fact("protocol.recipe.iie_published", SEC, src="iie_ft_kl_sigma_summary.json", path="recipe",
     definition="IIE adapted, published recipe (iie_ft_kl / iie_ft_kl_sigma)")
fact("protocol.recipe.iie_trained", SEC, src="iie_ft_klcc_e20_summary.json", path="trained_parameters",
     definition="IIE adapted: all 120 readout components trained, 4 backbones frozen")
fact("protocol.recipe.iie_selection_rule", SEC, src="iie_recipe_selection_summary.json", path="rule",
     definition="how the IIE recipe was chosen (fold-0 training ids only, inner 600/200 split)",
     note="fold-0 training ids are the test ids of folds 1-4; the choice was among 3 recipes and the published "
          "recipe gives the same CC (0.8551 vs 0.8549), so the selection has no material effect; say so if asked")
fact("protocol.recipe.iie_selection_chosen", SEC, src="iie_recipe_selection_summary.json", path="chosen",
     definition="chosen IIE recipe")
fact("protocol.recipe.sum_ft", SEC, src="sum_ft_summary.json", path="loss", definition="SUM full fine-tune loss")
fact("protocol.recipe.sum_ft_init", SEC, src="sum_ft_summary.json", path="init",
     definition="SUM full fine-tune initialisation (released weights; all parameters trained)")
fact("protocol.recipe.sum_ft_epochs", SEC, 12, "epochs", "computed", definition="SUM full fine-tune epochs per fold",
     inputs=["results/sum_ft_summary.json train_log_per_epoch (12 per fold)"],
     recomputed=len([e for e in J("sum_ft_summary.json")["train_log_per_epoch"] if e["fold"] == 0]))
fact("protocol.recipe.sum_ft_token", SEC, src="sum_zeroshot_t2_summary.json", path="token_meaning",
     definition="SUM domain token 2 meaning (used by sum_ft, sum_zeroshot_t2)")
fact("protocol.recipe.sum_token_choice", SEC, src="sum_calibrated_summary.json", path="calibration_maps_used_per_fold",
     definition="calibrated SUM chose its token on each fold's 800 training maps (token 2 in every fold, per the "
                "main_table row note), which supports token 2 without test data")
tok = {}
for x in J("sum_calibrated_per_image.json"):
    tok.setdefault(int(x["fold"]), set()).add(int(x["token"]))
fact("protocol.recipe.sum_token_per_fold", SEC, {str(k): sorted(v_) for k, v_ in sorted(tok.items())}, "token", "computed",
     definition="SUM token selected on each fold's 800 training maps by calibrated SUM (sum_calibrated): token 2 "
                "(e-commerce) in every fold, the token used by sum_zeroshot_t2 and sum_ft",
     inputs=["results/sum_calibrated_per_image.json [*]/token by fold"])
fact("protocol.recipe.sum_scan_validation", SEC, src="sum_validate_t2.json", path="max_abs_dCC",
     definition="pure-PyTorch selective scan vs stored kernel outputs: max |dCC| over 20 images")
fact("protocol.recipe.sum_ft_rescore", SEC, "sum_ft raw 256x256 test predictions re-scored locally reproduce every "
     "per-image value to within 1.4e-10", "", "computed", definition="cloud-run integrity check",
     inputs=["results/FRONTIER.md section 1 row notes"])
fact("protocol.recipe.sum_ft_seed", SEC, "seed 0, one full fine-tune per fold (single seed)", "", "computed",
     definition="sum_ft replication", inputs=["results/sum_ft_summary.json command (--seed 0)"])
# calibration vs training maps per row
for r in MT["rows"]:
    if r.get("group") in ("ablation",) or r.get("split") != "random":
        continue
    fact(f"protocol.maps.{r['row']}", SEC,
         {"calibration_maps": r.get("calibration_maps"), "training_maps": r.get("training_maps"),
          "trainable_params": r.get("trainable_params"), "note": r.get("calibration_note")},
         "maps per fold", "computed", definition="ADD1000 maps per fold used for calibration (prior/token/sigma) vs "
         "for training weights, and parameters updated on ADD1000",
         inputs=[f"results/main_table.json rows[row={r['row']}]/calibration_maps|training_maps|trainable_params|calibration_note"])

# ========================================================================================== 3 main rows
SEC = "main_table"
for r in MT["rows"]:
    name = r["row"]
    try:
        rows = R(name)
    except Exception as e:  # pragma: no cover
        WARN.append(f"could not rebuild per-image rows of {name}: {e}")
        rows = None
    sp = split_of(name)
    for m in METRICS:
        base = f"rows[row={name}]/stats_384/{m}"
        dd = desc(rows, m) if rows is not None else None
        fact(f"main.{name}.{m}.mean", SEC, src="main_table.json", path=base + "/mean", unit=m,
             definition=f"{r['label']}: {m} mean over the 5 fold means ({sp} folds; = mean over the 1000 images for "
                        f"equal folds)", recomputed=dd["mean_of_fold_means"] if dd else None, display=None)
        fact(f"main.{name}.{m}.sd_folds", SEC, src="main_table.json", path=base + "/sd_folds_ddof1", unit=m,
             definition=f"{r['label']}: SD of the 5 fold means", sd_ddof=1,
             recomputed=dd["sd_ddof1"] if dd else None)
        if sp != "random":
            fact(f"main.{name}.{m}.mean_over_images", SEC, src="main_table.json", path=base + "/mean_over_images",
                 unit=m, definition=f"{r['label']}: mean over the 1000 images (folds unequal, differs from the mean "
                                    f"of fold means)", recomputed=dd["mean_over_images"] if dd else None)
    fact(f"main.{name}.label", SEC, src="main_table.json", path=f"rows[row={name}]/label", definition="row label")
# seed aggregates (SD over 3 seed means)
for agg in ("ours_scratch", "ours_warm", "msdb_adapt"):
    for m in METRICS:
        sm = []
        for s in range(3):
            rr = R(f"{agg}_s{s}")
            sm.append(float(vec(rr, m).mean()))
        fact(f"seeds.{agg}.{m}.mean_over_seeds", "seeds", src="main_table.json",
             path=f"seed_aggregates/{agg}/{m}/mean_over_seeds", unit=m,
             definition=f"{agg}: mean of the 3 seed means (init seeds 0, 1, 2)", recomputed=float(np.mean(sm)))
        fact(f"seeds.{agg}.{m}.sd_over_seeds", "seeds", src="main_table.json",
             path=f"seed_aggregates/{agg}/{m}/sd_over_seeds_ddof1", unit=m, sd_ddof=1,
             definition=f"{agg}: SD over the 3 seed means", recomputed=float(np.std(sm, ddof=1)))
fact("main.ours_warm_not_finetuned.CC.mean", SEC, src="main_table.json", path="fold_level_only_rows/ours_warm_not_finetuned/CC/mean",
     unit="CC", definition="warm-start readout (UEyes) before any ADD1000 training, sigma on 200 maps; fold level only")
fact("main.ours_warm_not_finetuned.CC.sd_folds", SEC, src="main_table.json",
     path="fold_level_only_rows/ours_warm_not_finetuned/CC/sd_folds_ddof1", unit="CC", sd_ddof=1,
     definition="SD of fold means, warm start not trained on ADD1000")

# ========================================================================================== 4 paired tests
SEC = "tests"
# 4a main family (reference ours_scratch_s0; differences ours minus row), recomputed and Holm re-applied
fam = list(MT["tests_vs_ours"]["ours_scratch_s0"].keys())
ref = R("ours_scratch_s0")
mine = {o: compare(ref, R(o)) for o in fam}
for m in METRICS:
    for key in ("wilcoxon_p", "nb_p"):
        adj = holm([mine[o][m][key] for o in fam])
        for o, a in zip(fam, adj):
            mine[o][m][key + "_holm"] = float(a)
fact("tests.family.main_ours_scratch_s0", SEC, fam, "rows", "computed",
     definition=f"Holm family 'main' (reference ours_scratch_s0): the {len(fam)} rows compared with ours seed 0 in "
                "main_table.json tests_vs_ours/ours_scratch_s0; Holm per metric and per test",
     inputs=["results/main_table.json tests_vs_ours/ours_scratch_s0"])
for o in fam:
    for m in METRICS:
        base = f"tests_vs_ours/ours_scratch_s0/{o}/{m}"
        mm = mine[o][m]
        pre = f"tests.main.ours_minus_{o}.{m}"
        fact(pre + ".mean_diff", SEC, src="main_table.json", path=base + "/mean_diff", unit=m,
             definition=f"ours_scratch_s0 minus {o}: mean per-image difference over 1000 out-of-fold images "
                        f"({'negative' if m == 'KLD' else 'positive'} = ours better)", recomputed=mm["mean_diff"])
        fact(pre + ".boot95", SEC, src="main_table.json", path=base + "/boot95", unit=m,
             definition="5000-sample bootstrap 95% CI of the mean per-image difference (numpy default_rng(0) per "
                        "comparison, metric order CC, SIM, KLD)", recomputed=mm["boot95"], tol=1e-12)
        fact(pre + ".folds_ours_better", SEC, src="main_table.json", path=base + "/folds_ref_better", unit="of 5 folds",
             definition="folds in which ours (reference) is better", recomputed=mm["folds_first_better"])
        fact(pre + ".frac_images_ours_better", SEC, src="main_table.json", path=base + "/frac_images_ref_better",
             unit="fraction of 1000 images", definition="share of images on which ours is better",
             recomputed=mm["frac_images_first_better"])
        fact(pre + ".wilcoxon_p", SEC, src="main_table.json", path=base + "/wilcoxon_p", unit="p",
             definition="two-sided Wilcoxon signed-rank (zero_method wilcox), raw", recomputed=mm["wilcoxon_p"], tol=0)
        fact(pre + ".wilcoxon_p_holm", SEC, src="main_table.json", path=base + "/wilcoxon_p_holm", unit="p",
             definition="Wilcoxon p, Holm over the main family (30 rows)", recomputed=mm["wilcoxon_p_holm"], tol=0)
        fact(pre + ".nb_p", SEC, src="main_table.json", path=base + "/nb_p", unit="p",
             definition="Nadeau-Bengio corrected t on 5 fold differences (df 4, ratio 0.25), raw",
             recomputed=mm["nb_p"], tol=0)
        fact(pre + ".nb_p_holm", SEC, src="main_table.json", path=base + "/nb_p_holm", unit="p",
             definition="NB p, Holm over the main family (30 rows)", recomputed=mm["nb_p_holm"], tol=0)
        fact(pre + ".dz", SEC, src="main_table.json", path=base + "/dz", unit="", definition="paired Cohen's dz",
             recomputed=mm["dz"])

# 4b frontier family: sum_ft vs 6 references (differences sum_ft minus reference)
SF = FR["paired_tests_sum_ft"]
fam_sf = list(SF["primary"]["results"].keys())
a = R("sum_ft")
mine_sf = {}
for o in fam_sf:
    mine_sf[o] = compare(a, R(o), B=10000, rng=np.random.default_rng(20261121))
for m in METRICS:
    for key in ("wilcoxon_p", "nb_p"):
        adj = holm([mine_sf[o][m][key] for o in fam_sf])
        for o, x in zip(fam_sf, adj):
            mine_sf[o][m][key + "_holm"] = float(x)
fact("tests.family.frontier_sum_ft", SEC, fam_sf, "references", "computed",
     definition="Holm family 'frontier sum_ft' (FRONTIER.md 3b): sum_ft vs 6 references, Holm per metric and test; "
                "CI = 10,000-sample bootstrap, numpy default_rng(20261121) per comparison",
     inputs=["results/frontier.json paired_tests_sum_ft/primary"])
for o in fam_sf:
    for m in METRICS:
        base = f"paired_tests_sum_ft/primary/results/{o}/{m}"
        mm = mine_sf[o][m]
        pre = f"tests.frontier.sum_ft_minus_{o}.{m}"
        fact(pre + ".mean_diff", SEC, src="frontier.json", path=base + "/mean_diff", unit=m,
             definition=f"sum_ft minus {o}: mean per-image difference", recomputed=mm["mean_diff"])
        fact(pre + ".boot95", SEC, src="frontier.json", path=base + "/boot95", unit=m,
             definition="10,000-sample bootstrap 95% CI (rng 20261121)", recomputed=mm["boot95"], tol=1e-12)
        fact(pre + ".folds_sum_ft_better", SEC, src="frontier.json", path=base + "/folds_model_better",
             unit="of 5 folds", definition="folds where sum_ft is better", recomputed=mm["folds_first_better"])
        fact(pre + ".frac_images_sum_ft_better", SEC, src="frontier.json", path=base + "/frac_images_model_better",
             unit="fraction", definition="share of images where sum_ft is better", recomputed=mm["frac_images_first_better"])
        for key in ("wilcoxon_p", "wilcoxon_p_holm", "nb_p", "nb_p_holm"):
            fact(pre + "." + key, SEC, src="frontier.json", path=base + "/" + key, unit="p",
                 definition=f"{key} (Holm over the 6-comparison frontier sum_ft family)" if "holm" in key else key,
                 recomputed=mm[key], tol=0)
for m in METRICS:
    base = f"paired_tests_sum_ft/secondary_seedavg/results/msdb_adapt_seedavg/{m}"
    fact(f"tests.frontier.sum_ft_minus_msdb_adapt_seedavg.{m}.mean_diff", SEC, src="frontier.json",
         path=base + "/mean_diff", unit=m, definition="sum_ft minus MSDB seed average (secondary, uncorrected)")
    fact(f"tests.frontier.sum_ft_minus_msdb_adapt_seedavg.{m}.nb_p", SEC, src="frontier.json", path=base + "/nb_p",
         unit="p", definition="NB p, single comparison (no correction)")

# 4c families computed here for the KBS paper
KEY4 = [("msdb_adapt_s0", "sum_ft"), ("sum_ft", "ours_scratch_s0"), ("ours_scratch_s0", "iie_ft_klcc_e20"),
        ("msdb_adapt_s0", "ours_scratch_s0")]
PAIR6 = [("msdb_adapt_s0", "sum_ft"), ("msdb_adapt_s0", "ours_scratch_s0"), ("msdb_adapt_s0", "iie_ft_klcc_e20"),
         ("sum_ft", "ours_scratch_s0"), ("sum_ft", "iie_ft_klcc_e20"), ("ours_scratch_s0", "iie_ft_klcc_e20")]
ZS3 = [("sum_zeroshot_t2", "iie_zeroshot_uniform"), ("iie_zeroshot_uniform", "msdb_zeroshot_uniform"),
       ("sum_zeroshot_t2", "msdb_zeroshot_uniform")]
GAIN3 = [("sum_ft", "sum_zeroshot_t2"), ("msdb_adapt_s0", "msdb_zeroshot_uniform"),
         ("iie_ft_klcc_e20", "iie_zeroshot_uniform")]
GAIN3C = [("sum_ft", "sum_calibrated"), ("msdb_adapt_s0", "msdb_zeroshot_foldprior"),
          ("iie_ft_klcc_e20", "iie_zeroshot_foldprior_sigma")]
NEWFAM = {"kbs_key4": (KEY4, "the 4 key comparisons of the KBS paper (MSDB vs SUM, SUM vs ours, ours vs IIE, MSDB vs ours)"),
          "kbs_adapted_pairwise6": (PAIR6, "all 6 pairs of the 4 adapted models (MSDB head, SUM full FT, ours, IIE selected recipe)"),
          "kbs_zeroshot_pairwise3": (ZS3, "the 3 pairs of the zero-shot rows with no ADD1000 map (SUM token 2, IIE and MSDB uniform prior)"),
          "kbs_gain_from_zeroshot3": (GAIN3, "adapted minus zero-shot (no ADD1000 map) per family"),
          "kbs_gain_from_calibrated3": (GAIN3C, "adapted minus fold-safe calibrated zero-shot per family")}
NEWRES = {}
for fname, (pairs, dsc) in NEWFAM.items():
    res = {}
    for a_, b_ in pairs:
        res[(a_, b_)] = compare(R(a_), R(b_), B=10000, rng=np.random.default_rng(20261124))
    for m in METRICS:
        for key in ("wilcoxon_p", "nb_p"):
            adj = holm([res[p][m][key] for p in pairs])
            for p, x in zip(pairs, adj):
                res[p][m][key + "_holm"] = float(x)
    NEWRES[fname] = res
    fact(f"tests.family.{fname}", SEC, [f"{a_} minus {b_}" for a_, b_ in pairs], "comparisons", "computed",
         definition=f"Holm family '{fname}' (computed here, not in the compiled JSONs): {dsc}; Holm per metric and "
                    "per test over these comparisons; CI = 10,000-sample paired bootstrap, numpy "
                    "default_rng(20261124) per comparison, metric order CC, SIM, KLD",
         inputs=[f"results/{ROWINFO[x]['per_image_file'].split('/')[-1] if x in ROWINFO else x}" for p in pairs for x in p])
    for (a_, b_) in pairs:
        for m in METRICS:
            mm = res[(a_, b_)][m]
            pre = f"tests.{fname}.{a_}_minus_{b_}.{m}"
            ins = [f"per-image rows of {a_} and {b_} (paired by id)"]
            fact(pre + ".mean_diff", SEC, mm["mean_diff"], m, "computed", inputs=ins,
                 definition=f"{a_} minus {b_}: mean per-image difference (for KLD negative = first better)",
                 display=f"{mm['mean_diff']:+.4f}")
            fact(pre + ".boot95", SEC, mm["boot95"], m, "computed", inputs=ins, definition="bootstrap 95% CI")
            fact(pre + ".folds_first_better", SEC, mm["folds_first_better"], "of 5 folds", "computed", inputs=ins,
                 definition=f"folds where {a_} is better")
            fact(pre + ".fold_diffs", SEC, mm["fold_diffs"], m, "computed", inputs=ins,
                 definition="per-fold mean differences (input of the NB test)")
            fact(pre + ".frac_images_first_better", SEC, mm["frac_images_first_better"], "fraction", "computed",
                 inputs=ins, definition=f"share of the 1000 images where {a_} is better")
            for key in ("wilcoxon_p", "wilcoxon_p_holm", "nb_p", "nb_p_holm"):
                fact(pre + "." + key, SEC, mm[key], "p", "computed", inputs=ins,
                     definition=(key + f" (Holm over family {fname})") if "holm" in key else key, display=pfmt(mm[key]))
            fact(pre + ".dz", SEC, mm["dz"], "", "computed", inputs=ins, definition="paired Cohen's dz")
# consistency of the new key4 raw tests with the compiled ones
cmp_map = {("msdb_adapt_s0", "sum_ft"): ("frontier", "msdb_adapt_s0", -1),
           ("sum_ft", "ours_scratch_s0"): ("frontier", "ours_scratch_s0", 1),
           ("ours_scratch_s0", "iie_ft_klcc_e20"): ("main", "iie_ft_klcc_e20", 1),
           ("msdb_adapt_s0", "ours_scratch_s0"): ("main", "msdb_adapt_s0", -1)}
for p, (where, o, sgn) in cmp_map.items():
    for m in METRICS:
        mine_v = NEWRES["kbs_key4"][p][m]
        if where == "frontier":
            comp = FR["paired_tests_sum_ft"]["primary"]["results"][o][m]
        else:
            comp = MT["tests_vs_ours"]["ours_scratch_s0"][o][m]
        okd = _close(sgn * mine_v["mean_diff"], comp["mean_diff"], 1e-12)
        okw = _close(mine_v["wilcoxon_p"], comp["wilcoxon_p"], 0)
        okn = _close(mine_v["nb_p"], comp["nb_p"], 0)
        CHECKS.append({"id": f"kbs_key4 {p[0]}-{p[1]} {m} vs compiled {where}", "match": okd and okw and okn,
                       "mean_diff_here": mine_v["mean_diff"], "compiled_signed": sgn * comp["mean_diff"]})
        if not (okd and okw and okn):
            WARN.append(f"MISMATCH kbs_key4 {p} {m} vs compiled {where}")

# ========================================================================================== 5 gains / ranking
SEC = "gains"
ZG = FR["zero_shot_to_adapted_gain"]
for famname, key in (("SUM", "sum"), ("DeepGaze MSDB", "msdb"), ("DeepGaze IIE", "iie")):
    g = ZG[famname]
    for m in METRICS:
        zs = float(vec(R(g["zero_shot_row"]), m).mean())
        ad = float(vec(R(g["adapted_row"]), m).mean())
        cz = float(vec(R(g["calibrated_zero_shot_row"]), m).mean())
        fact(f"gain.{key}.from_zeroshot.{m}", SEC, src="frontier.json",
             path=f"zero_shot_to_adapted_gain/{famname}/gain_from_zero_shot/{m}", unit=m,
             definition=f"{famname}: {g['adapted_row']} minus {g['zero_shot_row']} (no ADD1000 map in the zero-shot row)",
             recomputed=ad - zs)
        fact(f"gain.{key}.from_calibrated.{m}", SEC, src="frontier.json",
             path=f"zero_shot_to_adapted_gain/{famname}/gain_from_calibrated_zero_shot/{m}", unit=m,
             definition=f"{famname}: {g['adapted_row']} minus {g['calibrated_zero_shot_row']} (fold-safe calibrated)",
             recomputed=ad - cz)
        if "gain_from_published_prior_zero_shot" in g and m in g["gain_from_published_prior_zero_shot"]:
            pz = float(vec(R(g["zero_shot_published_prior_row"]), m).mean())
            fact(f"gain.{key}.from_published_prior.{m}", SEC, src="frontier.json",
                 path=f"zero_shot_to_adapted_gain/{famname}/gain_from_published_prior_zero_shot/{m}", unit=m,
                 definition=f"{famname}: adapted minus published 2-fold-prior zero-shot (500 maps; not fold-safe)",
                 recomputed=ad - pz)
    fact(f"gain.{key}.rows", SEC, {k: g[k] for k in ("zero_shot_row", "calibrated_zero_shot_row", "adapted_row")},
         "", "computed", definition=f"rows used for the {famname} gain in frontier.json",
         inputs=[f"results/frontier.json zero_shot_to_adapted_gain/{famname}"])
# IIE gain with the selected recipe (frontier uses iie_ft_kl_sigma)
gi = float(vec(R("iie_ft_klcc_e20"), "CC").mean() - vec(R("iie_zeroshot_uniform"), "CC").mean())
fact("gain.iie.from_zeroshot.CC.selected_recipe", SEC, gi, "CC", "computed",
     definition="IIE gain with the selected recipe: iie_ft_klcc_e20 minus iie_zeroshot_uniform",
     inputs=["results/iie_ft_klcc_e20_per_image.json", "results/iie_zeroshot_uniform_per_image.json"])
zk = {n: {m: float(vec(R(n), m).mean()) for m in METRICS} for n in ("sum_zeroshot_t2", "iie_zeroshot_uniform", "msdb_zeroshot_uniform")}
for k, v in FR["zero_shot_vs_adapted_ranking_by_CC"].items():
    fact(f"ranking.{k}", SEC, src="frontier.json", path=f"zero_shot_vs_adapted_ranking_by_CC/{k}",
         definition=f"CC ranking: {k}")
# between-model gaps vs gains (check the brief's 'gains larger than any between-model gap')
ccm = {n: float(vec(R(n), "CC").mean()) for n in ("sum_zeroshot_t2", "iie_zeroshot_uniform", "msdb_zeroshot_uniform",
                                                     "sum_ft", "msdb_adapt_s0", "iie_ft_klcc_e20", "iie_ft_kl_sigma",
                                                     "ours_scratch_s0", "sum_calibrated", "msdb_zeroshot_foldprior",
                                                     "iie_zeroshot_foldprior_sigma")}
zs_gaps = {"SUM-IIE": ccm["sum_zeroshot_t2"] - ccm["iie_zeroshot_uniform"],
           "IIE-MSDB": ccm["iie_zeroshot_uniform"] - ccm["msdb_zeroshot_uniform"],
           "SUM-MSDB": ccm["sum_zeroshot_t2"] - ccm["msdb_zeroshot_uniform"]}
ad_set = {"MSDB": ccm["msdb_adapt_s0"], "SUM": ccm["sum_ft"], "ours": ccm["ours_scratch_s0"], "IIE": ccm["iie_ft_klcc_e20"]}
ad_gaps = {f"{x}-{y}": ad_set[x] - ad_set[y] for x in ad_set for y in ad_set if ad_set[x] > ad_set[y]}
gains = {"SUM": ccm["sum_ft"] - ccm["sum_zeroshot_t2"], "MSDB": ccm["msdb_adapt_s0"] - ccm["msdb_zeroshot_uniform"],
         "IIE": ccm["iie_ft_klcc_e20"] - ccm["iie_zeroshot_uniform"]}
cal_gaps = {"SUM-MSDB": ccm["sum_calibrated"] - ccm["msdb_zeroshot_foldprior"],
            "MSDB-IIE": ccm["msdb_zeroshot_foldprior"] - ccm["iie_zeroshot_foldprior_sigma"],
            "SUM-IIE": ccm["sum_calibrated"] - ccm["iie_zeroshot_foldprior_sigma"]}
fact("gap.zeroshot_CC", SEC, zs_gaps, "CC", "computed", definition="between-model CC gaps among the zero-shot rows "
     "(no ADD1000 map)", inputs=["per-image files of sum_zeroshot_t2, iie_zeroshot_uniform, msdb_zeroshot_uniform"])
fact("gap.calibrated_CC", SEC, cal_gaps, "CC", "computed", definition="between-model CC gaps among calibrated "
     "zero-shot rows", inputs=["sum_calibrated, msdb_zeroshot_foldprior, iie_zeroshot_foldprior_sigma"])
fact("gap.adapted_CC", SEC, ad_gaps, "CC", "computed", definition="between-model CC gaps among the 4 adapted rows "
     "(MSDB head, SUM full FT, ours, IIE selected recipe)", inputs=["per-image files of the 4 adapted rows"])
fact("gap.adapted_max_CC", SEC, max(ad_gaps.values()), "CC", "computed",
     definition="largest CC gap between adapted models (MSDB minus IIE)")
fact("gap.zeroshot_max_CC", SEC, max(zs_gaps.values()), "CC", "computed",
     definition="largest CC gap between zero-shot models (SUM minus MSDB)")
fact("gain.min_CC", SEC, min(gains.values()), "CC", "computed",
     definition="smallest zero-shot-to-adapted CC gain (SUM)")
fact("gain.max_CC", SEC, max(gains.values()), "CC", "computed",
     definition="largest zero-shot-to-adapted CC gain (MSDB)")
holds_ad = min(gains.values()) > max(ad_gaps.values())
holds_all = min(gains.values()) > max(list(ad_gaps.values()) + list(zs_gaps.values()))
fact("gain.exceeds_every_adapted_gap", SEC, holds_ad, "", "computed",
     definition="the smallest gain (SUM +0.0807) exceeds every gap between adapted models (max MSDB-IIE)")
fact("gain.exceeds_every_gap_incl_zeroshot", SEC, holds_all, "", "computed",
     definition="the smallest gain also exceeds every zero-shot gap (FALSE: zero-shot SUM-MSDB 0.127 and SUM-IIE "
                "0.087 exceed SUM's gain 0.081)")
if not holds_all:
    WARN.append(f"SCOPE: every zero-shot-to-adapted CC gain (+{min(gains.values()):.3f} SUM to "
                f"+{max(gains.values()):.3f} MSDB) exceeds every gap between the three released models as adapted (SUM, "
                f"MSDB, IIE at 384 px; max "
                f"{max(ad_gaps.values()):.4f}, MSDB vs IIE) but not every zero-shot gap: SUM-MSDB "
                f"{zs_gaps['SUM-MSDB']:.4f} and SUM-IIE {zs_gaps['SUM-IIE']:.4f} exceed SUM's gain.")

# ========================================================================================== 6 curves
SEC = "curves"
F0 = KF["folds"][0]
NS = [25, 50, 100, 200, 400, 800]


def curve_from_rows(rows, n_list, key_r="resample"):
    """rows: list of per-image dicts with n, resample (or r). -> {n: {r: {metric: mean}}}"""
    out = {}
    for x in rows:
        n, r = int(x["n"]), int(x.get(key_r, x.get("r")))
        out.setdefault(n, {}).setdefault(r, {m: [] for m in METRICS})
        for m in METRICS:
            out[n][r][m].append(float(x[m]))
    res = {}
    for n in sorted(out):
        res[n] = {}
        for r in sorted(out[n]):
            assert len(out[n][r]["CC"]) == 200, (n, r)
            res[n][r] = {m: float(np.mean(out[n][r][m])) for m in METRICS}
    return res


CURVES = {}
# ours (scratch, warm)
for init in ("scratch", "warm"):
    rows = []
    for n in NS:
        for r in range(3):
            d = J(f"curve_ours_{init}_n{n}_r{r}_per_image.json")
            assert [str(x["id"]) for x in d] == F0
            rows += [{**x, "resample": r} for x in d]
    CURVES[f"ours_{init}"] = (curve_from_rows(rows, NS), f"curve_ours_{init}_n<n>_r<r>_per_image.json",
                              f"ours_{init}")
CURVES["msdb_adapt_head"] = (curve_from_rows(J("msdb_adapt_curves_per_image.json"), NS), "msdb_adapt_curves_per_image.json",
                             "msdb_adapt_head")
CURVES["msdb_adapt_dsp"] = (curve_from_rows(J("msdb_adapt_dsp_curves_per_image.json"), NS),
                            "msdb_adapt_dsp_curves_per_image.json", "msdb_adapt_dsp")
CURVES["iie_adapt_klcc_e20"] = (curve_from_rows(J("iie_curve_klcce20_per_image.json"), NS), "iie_curve_klcce20_per_image.json",
                                "iie_adapt_klcc_e20")
CURVES["iie_adapt_klcc_e60_partial"] = (curve_from_rows(J("iie_curve_klcc_per_image.json"), NS),
                                        "iie_curve_klcc_per_image.json", "iie_adapt_klcc_e60_partial")
CT_TAB = MT["curves"]["table"]
for cname, (cv, srcf, tabkey) in CURVES.items():
    for n, byr in cv.items():
        for m in METRICS:
            runs = [byr[r][m] for r in sorted(byr)]
            mean_, sd0, sd1 = float(np.mean(runs)), float(np.std(runs)), float(np.std(runs, ddof=1))
            tab = CT_TAB.get(tabkey, {}).get(str(n))
            pre = f"curve.{cname}.n{n}.{m}"
            if tab is not None and f"mean_{m}" in tab:
                fact(pre + ".mean", SEC, src="main_table.json", path=f"curves/table/{tabkey}/{n}/mean_{m}", unit=m,
                     definition=f"{cname}, n = {n} training maps (fold-0 test set, 200 images): mean over "
                                f"{len(runs)} resamples of the per-resample mean", recomputed=mean_)
                fact(pre + ".sd_resamples_ddof0", SEC, src="main_table.json", path=f"curves/table/{tabkey}/{n}/sd_{m}_ddof0",
                     unit=m, sd_ddof=0, definition=f"SD over the {len(runs)} resample means (ddof 0)",
                     recomputed=sd0)
                if f"sd_{m}_ddof1" in tab:
                    fact(pre + ".sd_resamples_ddof1", SEC, src="main_table.json",
                         path=f"curves/table/{tabkey}/{n}/sd_{m}_ddof1", unit=m, sd_ddof=1,
                         definition=f"SD over the {len(runs)} resample means (ddof 1)", recomputed=sd1)
            else:
                fact(pre + ".mean", SEC, mean_, m, "computed", inputs=[f"results/{srcf}"],
                     definition=f"{cname}, n = {n}: mean over {len(runs)} resamples")
                fact(pre + ".sd_resamples_ddof0", SEC, sd0, m, "computed", inputs=[f"results/{srcf}"], sd_ddof=0,
                     definition="SD over resample means (ddof 0)")
            fact(pre + ".runs", SEC, runs, m, "computed", inputs=[f"results/{srcf}"],
                 definition="per-resample means (resample 0, 1, 2)")
# cross-check the per-family summaries
for n_i, c in enumerate(J("msdb_adapt_curves_summary.json")["curve"]):
    fact(f"curve.msdb_adapt_head.n{c['n']}.CC.summary_mean", SEC, src="msdb_adapt_curves_summary.json",
         path=f"curve[{n_i}]/mean_CC", unit="CC", definition="MSDB head curve, family summary file",
         recomputed=float(np.mean([CURVES['msdb_adapt_head'][0][c['n']][r]['CC'] for r in range(3)])))
for n_i, c in enumerate(J("curve_ours_scratch_aggregate.json")["by_n"]):
    fact(f"curve.ours_scratch.n{c['n']}.CC.aggregate_mean", SEC, src="curve_ours_scratch_aggregate.json",
         path=f"by_n[{n_i}]/mean_CC", unit="CC", definition="ours scratch curve, aggregate file",
         recomputed=float(np.mean([CURVES['ours_scratch'][0][c['n']][r]['CC'] for r in range(3)])))
# subset identity across models
msdb_runs = {(x["n"], x["resample"]): x["train_ids"] for x in J("msdb_adapt_curves_summary.json")["runs"]}
mism = 0
for n in NS:
    for r in range(3):
        for init in ("scratch", "warm"):
            s = J(f"curve_ours_{init}_n{n}_r{r}_summary.json")
            tid = s.get("train_ids")
            if tid is not None and list(map(str, tid)) != list(map(str, msdb_runs[(n, r)])):
                mism += 1
        pool = [KF["ids_sorted"][j] for j in perm[200:]]
        rule = [str(x) for x in np.random.default_rng(1000 * n + r).choice(np.array(pool), size=n, replace=False)]
        if rule != list(map(str, msdb_runs[(n, r)])):
            mism += 1
fact("curve.subset_identity_mismatches", SEC, 0, "mismatches", "computed",
     definition="training subsets: default_rng(1000n + r).choice(pool of folds 1-4, n) identical for ours, MSDB "
                "(stored ids) and the rule IIE applies in code; mismatches found here", recomputed=mism,
     inputs=["results/msdb_adapt_curves_summary.json runs[*]/train_ids", "results/curve_ours_*_summary.json train_ids"])
fact("curve.subset_check_compiled", SEC, src="main_table.json", path="curves/subset_check/n_mismatches",
     definition="compiled subset check (main_table.json)")
fact("curve.test_set", SEC, src="main_table.json", path="curves/test_set", definition="learning-curve test set")
fact("curve.sum_status_compiled", SEC, src="main_table.json", path="curves/sum_adapted_curve",
     definition="status of the SUM curve in the compiled main_table.json (written before the SUM curve job)")
# n = 0 references on fold 0
for k, v in MT["curves"]["n0_fold0"].items():
    if k == "ours_warm_not_finetuned":
        fact(f"curve.n0.{k}.CC", SEC, src="main_table.json", path=f"curves/n0_fold0/{k}/CC", unit="CC",
             definition=f"n = 0 reference on the fold-0 test set: {v.get('note')}")
        continue
    rr = R(k)
    f0 = {i: x for i, x in rr.items() if x["fold"] == 0}
    for m in METRICS:
        fact(f"curve.n0.{k}.{m}", SEC, src="main_table.json", path=f"curves/n0_fold0/{k}/{m}", unit=m,
             definition=f"n = 0 reference on the fold-0 test set (200 images): {v.get('note')}",
             recomputed=float(np.mean([x[m] for x in f0.values()])))
# per-budget paired tests from main_table (ours vs adapted baselines)
for cmpname, byn in MT["curves"]["paired_tests"].items():
    for n, mm in byn.items():
        for m in ("CC",):
            base = f"curves/paired_tests/{cmpname}/{n}/{m}"
            fact(f"curve.test.{cmpname}.n{n}.{m}.mean_diff", SEC, src="main_table.json", path=base + "/mean_diff",
                 unit=m, definition=f"{cmpname} at n = {n}: mean per-image difference (first minus second; per-image "
                                    f"values averaged over 3 resamples)")
            fact(f"curve.test.{cmpname}.n{n}.{m}.wilcoxon_p_holm", SEC, src="main_table.json",
                 path=base + "/wilcoxon_p_holm", unit="p",
                 definition="Wilcoxon p, Holm over 6 budgets x 3 metrics within this comparison")
            fact(f"curve.test.{cmpname}.n{n}.{m}.resamples_first_better", SEC, src="main_table.json",
                 path=base + "/resamples_ref_better", unit="of 3", definition="resamples won by the first model")
# MSDB-head vs IIE / ours at n = 25 (headline 0.873)
fact("curve.headline.msdb_head_n25_CC", SEC, src="main_table.json", path="curves/table/msdb_adapt_head/25/mean_CC",
     unit="CC", definition="MSDB head adapted on 25 maps (3 resamples), fold-0 test set",
     recomputed=float(np.mean([CURVES['msdb_adapt_head'][0][25][r]['CC'] for r in range(3)])))
# SUM curve placeholder
sum_curve = os.path.join(RES, "sum_ft_curve_summary.json")
sum_curve_pi = os.path.join(RES, "sum_ft_curve_per_image.json")
present = os.path.exists(sum_curve) and os.path.exists(sum_curve_pi)
FACTS.append({"id": "curve.sum_ft.status", "section": SEC, "value": "present" if present else "ABSENT",
              "unit": "", "source": "results/sum_ft_curve_summary.json", "json_path": None,
              "definition": "SUM full fine-tune learning curve (n in 25, 50, 100, 200 x 3 resamples, n = 400 x 1, "
                            "n = 800 = sum_ft fold 0); format = msdb_adapt_curves_summary.json ('curve' list with "
                            "n, mean_CC, sd_CC (ddof 0), sd_CC_ddof1, runs_CC; per-image rows id, n, resample, CC, "
                            "SIM, KLD) per scripts/sum/finalize_curve.py",
              "note": "the n = 800 point is sum_ft fold 0 (sigma on 200 of the 800 maps, data order default_rng(0), "
                      "torch seed 0), not a curve run; n = 400 has one resample (no SD)"})
_ids.add("curve.sum_ft.status")
if present:
    sc = J(sum_curve)
    pi = J(sum_curve_pi)
    cv = curve_from_rows(pi, NS)
    for c_i, c in enumerate(sc["curve"]):
        n = c["n"]
        runs = [cv[n][r]["CC"] for r in sorted(cv[n])]
        fact(f"curve.sum_ft.n{n}.CC.mean", SEC, src="sum_ft_curve_summary.json", path=f"curve[{c_i}]/mean_CC",
             unit="CC", definition=f"SUM full FT curve, n = {n}", recomputed=float(np.mean(runs)))
        fact(f"curve.sum_ft.n{n}.CC.sd_resamples_ddof0", SEC, src="sum_ft_curve_summary.json",
             path=f"curve[{c_i}]/sd_CC", unit="CC", sd_ddof=0, definition="SD over resamples (ddof 0)",
             recomputed=float(np.std(runs)))
else:
    for n in NS:
        FACTS.append({"id": f"curve.sum_ft.n{n}.CC.mean", "section": SEC, "value": None, "unit": "CC",
                      "source": "results/sum_ft_curve_summary.json (pending)", "json_path": "curve[n=%d]/mean_CC" % n,
                      "definition": f"SUM full FT curve point n = {n}: PENDING (job running); mark every sentence "
                                    f"that uses it with \\TODO{{}}"})
        _ids.add(f"curve.sum_ft.n{n}.CC.mean")
    WARN.append("SUM CURVE: results/sum_ft_curve_summary.json and sum_ft_curve_per_image.json do not exist yet; every "
                "curve.sum_ft.* fact is a placeholder (value null). Rerun this script when they land.")

# ========================================================================================== 7 grouped / pair-safe
SEC = "robustness"
for split in ("grouped", "pairsafe"):
    gp = MT["grouped_pairsafe"][split]
    for key, entry in gp.items():
        if key == "ours_vs_baselines_same_split":
            continue
        rnd = entry["random_fold_row"]
        for m in METRICS:
            a_ = R(key)
            b_ = R(rnd)
            ids = sorted(a_, key=int)
            dvec = vec(a_, m, ids) - vec(b_, m, ids)
            wp = float(stats.wilcoxon(vec(a_, m, ids), vec(b_, m, ids)).pvalue)
            fact(f"{split}.{key}.{m}.split_mean_over_images", SEC, src="main_table.json",
                 path=f"grouped_pairsafe/{split}/{key}/{m}/split_mean", unit=m,
                 definition=f"{key}: mean over the 1000 images on {split} folds", recomputed=float(vec(a_, m).mean()))
            fact(f"{split}.{key}.{m}.diff_vs_random", SEC, src="main_table.json",
                 path=f"grouped_pairsafe/{split}/{key}/{m}/diff_split_minus_random", unit=m,
                 definition=f"{key} minus {rnd} (same images, {split} vs random fold assignment)",
                 recomputed=float(dvec.mean()))
            fact(f"{split}.{key}.{m}.wilcoxon_p_vs_random", SEC, src="main_table.json",
                 path=f"grouped_pairsafe/{split}/{key}/{m}/wilcoxon_p_per_image", unit="p",
                 definition="per-image Wilcoxon, split vs random (uncorrected)", recomputed=wp, tol=0)
            fact(f"{split}.{key}.{m}.sd_folds", SEC, src="main_table.json",
                 path=f"grouped_pairsafe/{split}/{key}/split_desc/{m}/sd_folds_ddof1", unit=m, sd_ddof=1,
                 definition=f"{key}: SD of the 5 {split} fold means", recomputed=desc(a_, m)["sd_ddof1"])
    for o, tt in gp["ours_vs_baselines_same_split"].items():
        refname = f"ours_{split}"
        mm = compare(R(refname), R(o), split=split)
        for m in METRICS:
            base = f"grouped_pairsafe/{split}/ours_vs_baselines_same_split/{o}/{m}"
            pre = f"{split}.test.ours_minus_{o}.{m}"
            fact(pre + ".mean_diff", SEC, src="main_table.json", path=base + "/mean_diff", unit=m,
                 definition=f"ours_{split} minus {o} on the same {split} folds", recomputed=mm[m]["mean_diff"])
            fact(pre + ".folds_ours_better", SEC, src="main_table.json", path=base + "/folds_ref_better",
                 unit="of 5", definition="folds where ours is better", recomputed=mm[m]["folds_first_better"])
            fact(pre + ".wilcoxon_p_holm", SEC, src="main_table.json", path=base + "/wilcoxon_p_holm", unit="p",
                 definition=f"Wilcoxon p, Holm within the {split} family (5 rows)")
            fact(pre + ".nb_p", SEC, src="main_table.json", path=base + "/nb_p", unit="p",
                 definition=f"NB p raw ({split}; ratio = mean of fold test/train ratios)", recomputed=mm[m]["nb_p"], tol=0)
            fact(pre + ".nb_p_holm", SEC, src="main_table.json", path=base + "/nb_p_holm", unit="p",
                 definition=f"NB p, Holm within the {split} family")
fact("grouped.definition", SEC, src=os.path.join(SPL, "groups.json"), path="description",
     definition="grouped folds: brand/template proxy groups (no brand labels exist)")
fact("grouped.n_groups", SEC, src=os.path.join(SPL, "groups.json"), path="clustering/n_groups_after_near_dup_merge",
     unit="groups", definition="groups after merging near-duplicate components",
     recomputed=len({x["group"] for x in GR["per_image"]}))
fact("grouped.group_size_min_median_max", SEC, src=os.path.join(SPL, "groups.json"),
     path="clustering/group_size_min_median_max", unit="images", definition="group sizes")
fact("grouped.fold_sizes", SEC, src=os.path.join(SPL, "groups.json"), path="fold_sizes", unit="images",
     definition="grouped fold sizes", recomputed=SIZES["grouped"])
fact("grouped.mean_test_to_nearest_train_cls_cos", SEC, src=os.path.join(SPL, "groups.json"),
     path="mean_test_to_nearest_train_cls_cos", definition="mean cosine of each test image's DINOv2 CLS to its "
     "nearest training image: random vs grouped folds (grouping lowers train-test similarity)")
fact("pairsafe.definition", SEC, src=os.path.join(SPL, "kfold_seed0_pairsafe.json"), path="description",
     definition="pair-safe folds")
fact("pairsafe.n_ids_changed_fold", SEC, src=os.path.join(SPL, "kfold_seed0_pairsafe.json"), path="n_ids_changed_fold",
     unit="ids", definition="ids that changed fold relative to the seed-0 split",
     recomputed=sum(1 for i in ids_sorted if FOLD_OF["pairsafe"][i] != FOLD_OF["random"][i]))
fact("pairsafe.n_components", SEC, len(PS["components_constrained"]), "components", "computed",
     definition="extended near-duplicate components kept within one fold",
     inputs=["splits/kfold_seed0_pairsafe.json components_constrained"])

# ========================================================================================== 8 near-duplicates
SEC = "near_duplicates"
fact("neardup.primary_definition", SEC, src=os.path.join(SPL, "near_dup_pairs.json"), path="primary/definition",
     definition="primary near-duplicate definition used in the paper")
fact("neardup.primary_n_pairs", SEC, src=os.path.join(SPL, "near_dup_pairs.json"), path="primary/n_pairs", unit="pairs",
     definition="pairs above the primary threshold (all visually inspected)",
     recomputed=len(ND["primary"]["pairs"]))
xf = sum(1 for p in ND["primary"]["pairs"] if FOLD_OF["random"][p["a"]] != FOLD_OF["random"][p["b"]])
fact("neardup.primary_n_cross_fold", SEC, src=os.path.join(SPL, "near_dup_pairs.json"), path="primary/n_cross_fold_seed0",
     unit="pairs", definition="primary pairs split across seed-0 folds (recomputed from kfold_seed0.json)", recomputed=xf)
fact("neardup.primary_pairs", SEC, [{"a": p["a"], "b": p["b"], "cos": p["meanpatch_cos"], "folds": [p["fold_a"], p["fold_b"]],
                                     "visual": p["visual"]} for p in ND["primary"]["pairs"]], "", "computed",
     definition="primary pairs (mean-patch cosine, seed-0 folds, visual verdict)",
     inputs=["splits/near_dup_pairs.json primary/pairs"])
fact("neardup.min_primary_cos", SEC, min(p["meanpatch_cos"] for p in ND["primary"]["pairs"]), "cosine", "computed",
     definition="lowest cosine among primary pairs (threshold 0.94; gap 0.945 -> 0.930 in the ranking)",
     inputs=["splits/near_dup_pairs.json primary/pairs[*]/meanpatch_cos"])
fact("neardup.extended_definition", SEC, src=os.path.join(SPL, "near_dup_pairs.json"), path="extended/definition",
     definition="extended set used for pair-safe folds")
fact("neardup.extended_n_pairs", SEC, src=os.path.join(SPL, "near_dup_pairs.json"), path="extended/n_pairs", unit="pairs",
     definition="extended pairs", recomputed=len(ND["extended"]["pairs"]))
xfe = sum(1 for p in ND["extended"]["pairs"] if FOLD_OF["random"][p["a"]] != FOLD_OF["random"][p["b"]])
fact("neardup.extended_n_cross_fold", SEC, src=os.path.join(SPL, "near_dup_pairs.json"),
     path="extended/n_cross_fold_seed0", unit="pairs", definition="extended pairs across seed-0 folds", recomputed=xfe)
fact("neardup.phash256_le26_pairs", SEC, src=os.path.join(SPL, "near_dup_pairs.json"), path="hash_reproduction_attempt/n_pairs",
     unit="pairs", definition="256-bit pHash at <= 26 bits finds only the pixel-identical pair 182/761")
fact("neardup.phash_scan_counts", SEC, src=os.path.join(SPL, "near_dup_pairs.json"),
     path="hash_reproduction_attempt/scan_counts_at_26_bits",
     definition="pairs found by 64-bit (8x8) and 256-bit (16x16) p/d/aHash at 26 bits")
fact("neardup.old_record_reproduced", SEC, src="main_table.json", path="near_duplicates/phash_reproduction/reproduced_earlier_5_pairs",
     definition="the earlier '5 pairs / 4 cross-fold at 26 bits (pHash)' record could not be reproduced")
fact("neardup.n_rejected_after_visual_check", SEC, len(ND["rejected_after_visual_check"]), "pairs", "computed",
     definition="high-cosine candidates rejected after visual inspection",
     inputs=["splits/near_dup_pairs.json rejected_after_visual_check"])

# ========================================================================================== 9 ablations
SEC = "ablations"
full = R("abl_full_seedavg")
for v in ("no_aspp", "no_centerbias", "no_finalizer", "kl_only"):
    var = R(f"abl_{v}_seedavg")
    mm = compare(var, full)
    for m in METRICS:
        base = f"ablations/seedavg/abl_{v}_seedavg/{m}"
        fact(f"ablation.{v}.{m}.mean_diff", SEC, src="main_table.json", path=base + "/mean_diff", unit=m,
             definition=f"variant {v} minus full readout (warm-start ablation protocol, per-image values averaged "
                        f"over 3 seeds)", recomputed=mm[m]["mean_diff"])
        fact(f"ablation.{v}.{m}.nb_p_holm", SEC, src="main_table.json", path=base + "/nb_p_holm", unit="p",
             definition="NB p, Holm over the 4 ablation variants")
        fact(f"ablation.{v}.{m}.wilcoxon_p_holm", SEC, src="main_table.json", path=base + "/wilcoxon_p_holm", unit="p",
             definition="Wilcoxon p, Holm over the 4 ablation variants")
    fact(f"ablation.{v}.trainable_params", SEC, src="main_table.json", path=f"rows[row=abl_{v}_s0]/trainable_params",
         unit="parameters", definition=f"trainable parameters of variant {v}")
for m in METRICS:
    fact(f"ablation.full.{m}.mean_3seeds", SEC, src="main_table.json", path=f"seed_aggregates/abl_full/{m}/mean_over_seeds",
         unit=m, definition="full readout under the warm-start ablation protocol, mean of 3 seed means")

# ========================================================================================== 10 distillation
SEC = "distillation"
for row in ("distill_gt800_teacher800", "distill_pitt_only", "distill_pitt", "distill_pitt_L"):
    rr = R(row)
    for m in METRICS:
        fact(f"distill.{row}.{m}.mean", SEC, src=f"{row}_summary.json", path=f"mean_{m}", unit=m,
             definition=f"{row}: mean over 5 folds", recomputed=desc(rr, m)["mean_of_fold_means"])
        fact(f"distill.{row}.{m}.sd_folds", SEC, src=f"{row}_summary.json", path=f"sd_{m}_ddof1", unit=m, sd_ddof=1,
             definition=f"{row}: SD of fold means", recomputed=desc(rr, m)["sd_ddof1"])
    fact(f"distill.{row}.trainable_params", SEC, src=f"{row}_summary.json", path="trainable_params", unit="parameters",
         definition=f"{row}: trainable parameters of the student")
fact("distill.pitt_pool", SEC, src="distill_pitt_summary.json", path="calibration_and_training_data/pitt_pool",
     definition="unlabeled Pitt ads pool (2,500 drawn, 1 removed as a near-duplicate of ADD1000, 2,499 used)")
fact("distill.teacher", SEC, src="distill_pitt_summary.json", path="calibration_and_training_data/teacher",
     definition="teacher = MSDB head adapted per fold on the fold's 800 training maps only")
fact("distill.teacher_check_CC", SEC, src="distill_pitt_summary.json",
     path="calibration_and_training_data/teacher_check_mean_CC_on_test_folds", unit="CC",
     definition="retrained teachers reproduce msdb_adapt_s0 (0.91897)")
for o in ("msdb_adapt_s0", "ours_scratch_s0"):
    for mod in ("distill_pitt", "distill_pitt_L"):
        for m in METRICS:
            base = f"paired_tests/primary/results/{mod}/{o}/{m}"
            mm = compare(R(mod), R(o))
            fact(f"distill.test.{mod}_minus_{o}.{m}.mean_diff", SEC, src="frontier.json", path=base + "/mean_diff",
                 unit=m, definition=f"{mod} minus {o}", recomputed=mm[m]["mean_diff"])
            fact(f"distill.test.{mod}_minus_{o}.{m}.nb_p_holm", SEC, src="frontier.json", path=base + "/nb_p_holm",
                 unit="p", definition="NB p, Holm over the 4-comparison distillation family ({distill_pitt, "
                                      "distill_pitt_L} x {msdb_adapt_s0, ours_scratch_s0})")

# ========================================================================================== 11 track A (backbone / resolution)
SEC = "readout_variants"
for row in ("ours_B644_s0", "ours_L448_s0", "ours_L448_s1", "ours_L448_s2", "ours_ms_B_s0", "ours_L_ms_s0"):
    rr = R(row)
    for m in METRICS:
        fact(f"variant.{row}.{m}.mean", SEC, src=f"{row}_summary.json", path=f"mean_{m}", unit=m,
             definition=f"{row}: mean over 5 folds", recomputed=desc(rr, m)["mean_of_fold_means"])
        fact(f"variant.{row}.{m}.sd_folds", SEC, src=f"{row}_summary.json", path=f"sd_{m}_ddof1", unit=m, sd_ddof=1,
             definition=f"{row}: SD of fold means", recomputed=desc(rr, m)["sd_ddof1"])
    fact(f"variant.{row}.trainable_params", SEC, src=f"{row}_summary.json", path="trainable_params", unit="parameters",
         definition=f"{row}: trainable readout parameters")
    fact(f"variant.{row}.backbone", SEC, src=f"{row}_summary.json", path="protocol/backbone", definition="backbone/features")

for m in METRICS:
    fact(f"variant.ours_L448_seedavg.{m}.mean", SEC, src="frontier.json", path=f"models[row=ours_L448_seedavg]/accuracy/{m}/mean",
         unit=m, definition="DINOv2-L 448 readout, per-image mean over seeds 0-2", recomputed=desc(R("ours_L448_seedavg"), m)["mean_of_fold_means"])
    fact(f"variant.ours_L448_seedavg.{m}.sd_folds", SEC, src="frontier.json",
         path=f"models[row=ours_L448_seedavg]/accuracy/{m}/sd_folds_ddof1", unit=m, sd_ddof=1,
         definition="SD of fold means (seed-averaged per image)", recomputed=desc(R("ours_L448_seedavg"), m)["sd_ddof1"])

# ========================================================================================== 12 costs
SEC = "cost"
fact("cost.machine", SEC, src="cost_table.json", path="machine", definition="inference-cost machine")
for k in ("GFLOPs", "latency", "memory", "our_finalizer"):
    fact(f"cost.convention.{k}", SEC, src="cost_table.json", path=f"conventions/{k}", definition=f"{k} convention")
for mdl in ("ours", "iie", "msdb", "sum"):
    base = f"inference/{mdl}"
    fact(f"cost.{mdl}.name", SEC, src="cost_table.json", path=base + "/name", definition="model")
    fact(f"cost.{mdl}.input", SEC, src="cost_table.json", path=base + "/input", definition="input size for cost")
    fact(f"cost.{mdl}.params_total", SEC, src="cost_table.json", path=base + "/params_total", unit="parameters",
         definition="parameters instantiated")
    fact(f"cost.{mdl}.params_frozen", SEC, src="cost_table.json", path=base + "/params_backbone_frozen",
         unit="parameters", definition="frozen parameters in our adapted row (for sum_ft: 0, all trained)")
    fact(f"cost.{mdl}.params_trainable_adapted_row", SEC, src="cost_table.json",
         path=base + "/params_trainable_in_our_adaptation_row", unit="parameters",
         definition="trainable parameters in the cost-table adapted row (SUM: head-only row; sum_ft trains all 57,501,196)")
    fact(f"cost.{mdl}.GFLOPs", SEC, src="cost_table.json", path=base + "/GFLOPs", unit="GFLOPs",
         definition="2 x MACs of conv/matmul ops, batch 1 at the protocol input")
    fact(f"cost.{mdl}.cpu_latency_s", SEC, src="cost_table.json", path=base + "/cpu_latency_median_s", unit="s",
         definition="median batch-1 forward latency, CPU 6 threads, quiet machine")
    fact(f"cost.{mdl}.cpu_latency_iqr_s", SEC, src="cost_table.json", path=base + "/cpu_latency_iqr_s", unit="s",
         definition="CPU latency IQR")
    fact(f"cost.{mdl}.mps_latency_s", SEC, src="cost_table.json", path=base + "/mps_latency_median_s", unit="s",
         definition="median batch-1 forward latency, Apple M4 Pro GPU (MPS)")
    fact(f"cost.{mdl}.cpu_peak_rss_GB", SEC, src="cost_table.json", path=base + "/cpu_peak_rss_GB", unit="GB",
         definition="peak process RSS during CPU inference (includes weights)")
    fact(f"cost.{mdl}.mps_driver_mem_GB", SEC, src="cost_table.json", path=base + "/mps_driver_allocated_max_GB",
         unit="GB", definition="max MPS driver allocation")
fact("cost.ours.GFLOPs_backbone", SEC, src="cost_table.json", path="inference/ours/GFLOPs_backbone", unit="GFLOPs",
     definition="ours: DINOv2-B 448 backbone share")
fact("cost.ours.GFLOPs_readout", SEC, src="cost_table.json", path="inference/ours/GFLOPs_readout", unit="GFLOPs",
     definition="ours: readout share")
fact("cost.ours.finalizer_s", SEC, src="cost_table.json", path="inference/ours/finalizer_sigma4_scipy_cpu_s", unit="s",
     definition="our post-hoc Gaussian finalizer on CPU (not in the forward latency)")
fact("cost.sum.scan_GFLOPs_estimate", SEC, src="cost_table.json", path="inference/sum/selective_scan_GFLOPs_estimate_not_in_total",
     unit="GFLOPs", definition="SUM selective-scan estimate excluded from 26.4")
fact("cost.sum.GFLOPs_incl_scan", SEC, src="cost_table.json", path="inference/sum/GFLOPs_incl_scan_estimate",
     unit="GFLOPs", definition="SUM incl. scan estimate")
fact("cost.msdb.params_executed", SEC, src="cost_table.json", path="inference/msdb/params_executed_estimate",
     unit="parameters", definition="MSDB parameters actually executed (unexecuted backbone layers are instantiated)")
ci = CT["inference"]
ratios = {
    "msdb_over_sum_GFLOPs": ci["msdb"]["GFLOPs"] / ci["sum"]["GFLOPs"],
    "msdb_over_ours_GFLOPs": ci["msdb"]["GFLOPs"] / ci["ours"]["GFLOPs"],
    "ours_over_sum_GFLOPs": ci["ours"]["GFLOPs"] / ci["sum"]["GFLOPs"],
    "ours_over_iie_GFLOPs": ci["ours"]["GFLOPs"] / ci["iie"]["GFLOPs"],
    "msdb_over_ours_cpu": ci["msdb"]["cpu_latency_median_s"] / ci["ours"]["cpu_latency_median_s"],
    "msdb_over_ours_mps": ci["msdb"]["mps_latency_median_s"] / ci["ours"]["mps_latency_median_s"],
    "msdb_over_sum_cpu": ci["msdb"]["cpu_latency_median_s"] / ci["sum"]["cpu_latency_median_s"],
    "msdb_over_sum_mps": ci["msdb"]["mps_latency_median_s"] / ci["sum"]["mps_latency_median_s"],
    "msdb_over_ours_rss": ci["msdb"]["cpu_peak_rss_GB"] / ci["ours"]["cpu_peak_rss_GB"],
    "msdb_over_sum_rss": ci["msdb"]["cpu_peak_rss_GB"] / ci["sum"]["cpu_peak_rss_GB"],
    "msdb_over_sum_params_total": ci["msdb"]["params_total"] / ci["sum"]["params_total"],
    "sum_trainable_over_msdb_trainable": ci["sum"]["params_total"] / ci["msdb"]["params_trainable_in_our_adaptation_row"],
}
for k, v in ratios.items():
    fact(f"cost.ratio.{k}", SEC, v, "x", "computed", definition=f"ratio {k} from cost_table.json inference entries",
         inputs=["results/cost_table.json inference/*"])
fact("cost.ratio.msdb_over_sum_GFLOPs_frontier", SEC, src="frontier.json",
     path="models[row=sum_ft]/vs_msdb_adapt/GFLOPs_fewer_x", unit="x", definition="frontier.json: MSDB/SUM GFLOPs",
     recomputed=ratios["msdb_over_sum_GFLOPs"])
fact("cost.ratio.msdb_over_ours_GFLOPs_compiled", SEC, src="cost_table.json", path="ratios_vs_ours/msdb/GFLOPs_vs_ours",
     unit="x", definition="compiled ratio", recomputed=ratios["msdb_over_ours_GFLOPs"])
# frontier rows cost (track A + distill) and throughput
for m_ in FR["models"]:
    row = m_["row"]
    c = m_["cost"]
    for k, unit in (("GFLOPs", "GFLOPs"), ("params_total", "parameters"), ("params_trainable", "parameters"),
                    ("cpu_latency_s", "s"), ("mps_latency_s", "s"), ("cpu_peak_rss_GB", "GB")):
        if c.get(k) is not None:
            fact(f"cost.frontier.{row}.{k}", SEC, src="frontier.json", path=f"models[row={row}]/cost/{k}", unit=unit,
                 definition=f"{row}: {k} (frontier.json; quiet-machine latency)")
    fact(f"cost.frontier.{row}.on_cpu_frontier", SEC, src="frontier.json", path=f"models[row={row}]/on_pareto_frontier_cc_vs_cpu",
         definition=f"{row} on the CC vs quiet CPU latency Pareto frontier")
for k in ("pareto_frontier_cc_vs_cpu", "pareto_frontier_cc_vs_mps", "pareto_frontier_cc_vs_gflops", "best_fast_model",
          "best_model_at_our_base_cost"):
    fact(f"cost.{k}", SEC, src="frontier.json", path=k, definition=k)
fact("cost.fast_definition", SEC, src="frontier.json", path="protocol/fast_definition", definition="'fast' threshold")

# training cost
SEC = "training_cost"
for t in CT["training_adaptation"]:
    row = t["row"]
    fact(f"train.{row}.device", SEC, src="cost_table.json", path=f"training_adaptation[row={row}]/device",
         definition=f"{row}: training device")
    fact(f"train.{row}.median_s_per_fold", SEC, float(np.median(t["train_s_per_fold"])), "s", "computed",
         definition=f"{row}: median per-fold training wall-clock", inputs=[f"results/cost_table.json training_adaptation[row={row}]/train_s_per_fold"])
    fact(f"train.{row}.s_per_fold", SEC, src="cost_table.json", path=f"training_adaptation[row={row}]/train_s_per_fold",
         unit="s", definition=f"{row}: per-fold training wall-clock")
for m_ in FR["models"]:
    tc = m_.get("training_cost") or {}
    if tc.get("train_s_per_fold_median") is not None:
        fact(f"train.frontier.{m_['row']}.median_s_per_fold", SEC, src="frontier.json",
             path=f"models[row={m_['row']}]/training_cost/train_s_per_fold_median", unit="s",
             definition=f"{m_['row']}: median s per fold ({tc.get('hardware')}); covers: {tc.get('covers')}")
        fact(f"train.frontier.{m_['row']}.hardware", SEC, src="frontier.json",
             path=f"models[row={m_['row']}]/training_cost/hardware", definition="training hardware")
for k, v in CT["one_off_feature_cache"].items():
    fact(f"train.cache.{k}.s_total", SEC, src="cost_table.json", path=f"one_off_feature_cache/{k}/s_total", unit="s",
         definition=f"one-off cache: {v['what']}; device {v['device']}; 1000 images")
fact("train.caveat", SEC, src="cost_table.json", path="training_caveat", definition="training-time caveat")
sft = J("sum_ft_summary.json")
fact("train.sum_ft.epoch_s_range", SEC, [min(e["seconds"] for e in sft["train_log_per_epoch"]),
                                         max(e["seconds"] for e in sft["train_log_per_epoch"])], "s", "computed",
     definition="sum_ft seconds per epoch on the A10G (60 epochs over 5 folds)", inputs=["results/sum_ft_summary.json train_log_per_epoch[*]/seconds"])
fact("train.sum_ft.max_mem_GB", SEC, max(e.get("max_mem_gb", 0) for e in sft["train_log_per_epoch"]), "GB", "computed",
     definition="sum_ft peak CUDA memory during training", inputs=["results/sum_ft_summary.json train_log_per_epoch[*]/max_mem_gb"])

# ========================================================================================== 13 SUM scan profile
SEC = "sum_profile"
SP = "sum_cpu_profile.json"
for k, dfn, unit in (
        ("headline_cpu/scan_fraction_timed_wrapper", "share of SUM's CPU forward spent in the pure-PyTorch chunked "
         "selective scan (timed wrapper; '94%')", "fraction"),
        ("headline_cpu/scan_fraction_profiler_self_time", "same share by profiler self time", "fraction"),
        ("headline_cpu/forward_s_median_chunked_scan", "SUM CPU forward, chunked scan, profiling session", "s"),
        ("headline_cpu/scan_s_per_forward_median", "seconds in the scan per CPU forward", "s"),
        ("headline_cpu/non_scan_s_per_forward_median", "seconds outside the scan per CPU forward", "s"),
        ("headline_cpu/forward_s_median_identity_scan", "SUM CPU forward with a zero-cost scan (lower bound, not "
         "SUM's output)", "s"),
        ("headline_cpu/ratio_sum_identity_scan_over_ours_B448", "zero-cost-scan SUM / ours CPU latency", "x"),
        ("headline_cpu/torch_cumprod_share_of_forward", "share of the CPU forward in one torch.cumprod call", "fraction"),
        ("headline_cpu/forward_s_median_mulloop_diagnostic", "numerically equivalent cumprod rewrite, CPU forward", "s"),
        ("headline_cpu/forward_s_median_ref_mamba_selective_scan_ref", "mamba-ssm sequential reference scan, CPU forward", "s"),
        ("headline_cpu/ratio_sum_ref_mamba_over_ours_B448", "reference-scan SUM / ours CPU latency", "x"),
        ("headline_cpu/ratio_sum_over_ours_B448_cost_table", "cost-table SUM / ours CPU latency", "x"),
        ("headline_mps/scan_fraction_timed_wrapper", "scan share of the MPS forward (synced wrapper)", "fraction"),
        ("headline_mps/forward_s_median_identity_scan", "SUM MPS forward with a zero-cost scan", "s"),
        ("headline_mps/forward_s_median_chunked_scan", "SUM MPS forward, chunked scan", "s"),
        ("same_session_cost_bench_repro/ratio_to_cost_table", "same-session repro of the cost-table SUM CPU latency", "x"),
        ("run_notes/reference_scan_availability", "why no CUDA kernel / mamba-ssm on this machine", "")):
    fact("sumprof." + k.replace("/", "."), SEC, src=SP, path=k, unit=unit, definition=dfn)
fact("sumprof.n_scan_calls_per_forward", SEC, src=SP, path="cpu/modes/chunked_timed/n_scan_calls_per_forward",
     unit="calls", definition="selective-scan calls per SUM forward")

# ========================================================================================== 14 native resolution
SEC = "native_resolution"
nat = []
for name, suffix, desc_ in (("msdb_zeroshot_uniform", "_512", "512"), ("msdb_zeroshot_adprior", "_512", "512"),
                            ("msdb_adapt_s0", "_512", "512"), ("sum_zeroshot_t2", "_native256", "256"),
                            ("sum_calibrated_cb", "_native256", "256"), ("sum_ft_head_nosigma", "_native256", "256"),
                            ("sum_ft", "_native256", "256"), ("sum_ft_klcc", "_native256", "256")):
    if name == "msdb_adapt_s0":
        d = [x for x in J("msdb_adapt_per_image.json") if x.get("seed", 0) == 0]
    elif name == "sum_ft_head_nosigma":
        d = J("sum_ft_head_per_image.json")
    else:
        d = J(f"{name}_per_image.json")
    ref_field = "CC_nosigma" if name in ("sum_ft", "sum_ft_klcc", "sum_ft_head_nosigma") else "CC"
    c384 = float(np.mean([x[ref_field] for x in d]))
    cnat = float(np.mean([x["CC" + suffix] for x in d]))
    nat.append(abs(c384 - cnat))
    fact(f"native.{name}.CC_384", SEC, c384, "CC", "computed", inputs=[f"per-image {ref_field}"],
         definition=f"{name}: CC at the common 384 grid" + (" (no sigma)" if ref_field != "CC" else ""))
    fact(f"native.{name}.CC_native", SEC, cnat, "CC", "computed", inputs=[f"per-image CC{suffix}"],
         definition=f"{name}: CC at the native {desc_} grid (GT resized to {desc_})")
fact("native.max_abs_dCC_checked_here", SEC, max(nat), "CC", "computed",
     definition="largest |CC(384) - CC(native)| over the rows checked here (<= 0.0006 over all 16 rows "
                "with a native score)")

# ========================================================================================== 15 parameter counts (re-derived)
SEC = "params"
PC = [("params.ours_readout_trainable", 3721089, "cache/distill/students/distill_pitt_fold4_head.pt and "
       "cache/trackA/heads/ours_B644_s0_f0.pt (same readout architecture as ours_scratch_s0)",
       "readout parameters of ours (DINOv2-B input): 3,722,631 tensor elements minus 1,542 BatchNorm buffer elements",
       "ours_scratch_s0 heads were not saved; counted from same-architecture saved heads (torch.load, sum of numel)"),
      ("params.ours_L_readout_trainable", 4933505, "cache/trackA/heads/ours_L448_s0_f0.pt, "
       "cache/distill/students/distill_pitt_L_fold4_head.pt", "readout parameters with DINOv2-L input (1024 ch)", None),
      ("params.msdb_head_trainable", 26463, "cache/distill/teachers/fold0.pt (per-fold adapted MSDB head)",
       "MSDB adapted head: saliency network 26,460 + 3 finalizer scalars (sigma, centre-bias weight, log priority)",
       None),
      ("params.msdb_released_head_file", 26525, "~/.cache/torch/hub/checkpoints/deepgazemsdb.pth",
       "released MSDB head file: saliency network 26,460 + 50 per-dataset scale weights (2 x 5x5) + 15 per-dataset "
       "finalizer parameters (5 training datasets x 3)", "do not quote 26,525 as the trained head (audit note "
       "'MSDB_readout 26525' predates this check)"),
      ("params.iie_readouts_trainable", 3059160, "cache/iie/ckpt/iie_ft_klcc_e20_fold0.pt",
       "DeepGaze IIE: all 120 readout components (4 backbones x 30)", None)]
for fid, v, srcf, dfn, note in PC:
    FACTS.append({"id": fid, "section": SEC, "value": v, "unit": "parameters", "source": srcf, "json_path": None,
                  "definition": dfn, "method": "re-derived 24 Nov 2026 from the checkpoint with torch.load, sum of "
                                               "tensor numel (BatchNorm running stats excluded)", **({"note": note} if note else {})})
    _ids.add(fid)
fact("params.sum_total", SEC, src="sum_param_count.json", path="total", unit="parameters",
     definition="SUM parameters counted on the instantiated released model (all trained in sum_ft)")
fact("params.sum_encoder", SEC, src="sum_param_count.json", path="encoder(patch_embed+layers)", unit="parameters",
     definition="SUM encoder")
fact("params.sum_decoder", SEC, src="sum_param_count.json", path="decoder(layers_up)", unit="parameters",
     definition="SUM decoder")
fact("params.sum_head", SEC, src="sum_param_count.json", path="head(final_up+final_conv)", unit="parameters",
     definition="SUM output head (sum_ft_head rows train only this)")
fact("params.msdb_breakdown", SEC, src="msdb_costs.json", path="saliency_network_params", unit="parameters",
     definition="MSDB saliency network (msdb_costs.json)")
fact("params.sum_ft_trainable_summary", SEC, src="sum_ft_summary.json", path="n_trainable_params", unit="parameters",
     definition="sum_ft trainable parameters as recorded by the run")

# ========================================================================================== 17 revision (24 Nov 2026)
# Rows and families added in the revision (scripts/revision/*.py).
SEC = "revision"
REVROWS = ["msdb_zeroshot_mit1003", "iie_zeroshot_mit1003", "msdb_calib_wide", "iie_calib_wide", "sum_calib_wide",
           "iie_zeroshot_mit1003_native"]
HAVE = {r: os.path.exists(os.path.join(RES, f"{FN(r)}_per_image.json")) for r in REVROWS}
for r in REVROWS:
    fr = FN(r)
    fact(f"rev.{r}.available", SEC, HAVE[r], "", "computed", definition=f"results/{fr}_per_image.json exists",
         inputs=[f"results/{fr}_per_image.json"])
    if not HAVE[r]:
        WARN.append(f"REVISION ROW MISSING: results/{fr}_per_image.json (facts rev.{r}.* absent)")
        continue
    S_ = J(f"{fr}_summary.json")
    for m in METRICS:
        dd = desc(R(r), m)
        fact(f"rev.{r}.{m}.mean", SEC, src=f"{fr}_summary.json", path=f"{m}_mean", unit=m,
             definition=f"{r}: mean of the 5 fold means (file {fr})", recomputed=dd["mean_of_fold_means"])
        sdkey = f"{m}_sd_folds_ddof1" if f"{m}_sd_folds_ddof1" in S_ else f"{m}_sd_over_folds"
        fact(f"rev.{r}.{m}.sd_folds", SEC, dd["sd_ddof1"], m, "computed", sd_ddof=1,
             definition=f"{r}: SD of the 5 fold means (ddof 1)", inputs=[f"results/{fr}_per_image.json"])
    fact(f"rev.{r}.protocol", SEC, src=f"{fr}_summary.json", path="protocol", definition=f"{r}: protocol string")
    if "selection" in S_:
        fact(f"rev.{r}.sigma_per_fold", SEC, [x.get("sigma_wide", x.get("sigma")) for x in S_["selection"]], "px at 384",
             "computed", definition=f"{r}: blur sigma chosen per fold on the wide grid 0..32",
             inputs=[f"results/{fr}_summary.json selection"])
        fact(f"rev.{r}.validation_max_abs_diff", SEC, src=f"{fr}_summary.json", path="validation_max_abs_diff",
             definition=f"{r}: restricted to the protocol grid 0..12 the same code reproduces "
                        f"{S_.get('validation_protocol_grid_vs')} per image to this max |diff|")
if HAVE["iie_zeroshot_mit1003_native"]:
    fact("rev.iie_zeroshot_mit1003_native.image_size_range_WH", SEC, src=f"{FN('iie_zeroshot_mit1003_native')}_summary.json",
         path="image_size_range_WH", definition="smallest and largest ADD1000 image (W, H) on disk")

# centre-bias template: the official DeepGaze file
TPL_OFF = os.path.join(KB, "cache", "revision", "deepgaze_template", "centerbias_mit1003.npy")
if not os.path.exists(TPL_OFF):
    raise SystemExit("download DeepGaze's centerbias_mit1003.npy first (README, step 0): " + TPL_OFF)
_sha = hashlib.sha256(open(TPL_OFF, "rb").read()).hexdigest()
fact("rev.template.file", SEC, "cache/revision/deepgaze_template/centerbias_mit1003.npy", "", "computed",
     definition="centre-bias template used by every MIT1003-template row: DeepGaze's distributed centerbias_mit1003.npy "
                "(DeepGaze v1.0.0 release; downloaded 24 Nov 2026 with the author's approval), 1024 x 1024 float64 log "
                "density, rescaled to the grid with nearest interpolation (zoom order 0) and logsumexp-renormalised as in "
                "the DeepGaze README", inputs=[TPL_OFF])
fact("rev.template.sha256", SEC, _sha, "", "computed", definition="sha256 of the official template file", inputs=[TPL_OFF])
fact("rev.template.sha256_matches_recorded", SEC,
     _sha == "051645b7da47884c138e81f28a4afbadd99b6c3262eaf086434a9626e3578be9", "", "computed",
     definition="sha256 equals the value recorded at download", inputs=[TPL_OFF])


# families
REVFAM = {}
if all(HAVE[r] for r in ("msdb_zeroshot_mit1003", "iie_zeroshot_mit1003")):
    REVFAM["kbs_zeroshot_doc3"] = ([("sum_zeroshot_t2", "msdb_zeroshot_mit1003"), ("sum_zeroshot_t2", "iie_zeroshot_mit1003"),
                                    ("msdb_zeroshot_mit1003", "iie_zeroshot_mit1003")],
                                   "zero-shot rows with documented defaults (SUM token 2; IIE and MSDB with the official DeepGaze MIT1003 template)")
if all(HAVE[r] for r in ("msdb_calib_wide", "iie_calib_wide", "sum_calib_wide")):
    REVFAM["kbs_calibrated3"] = ([("msdb_calib_wide", "sum_calib_wide"), ("msdb_calib_wide", "iie_calib_wide"),
                                  ("sum_calib_wide", "iie_calib_wide")],
                                 "calibrated rows (prior or token and blur from the fold's training maps; wide blur grid)")
    REVFAM["kbs_step_training3"] = ([("sum_ft", "sum_calib_wide"), ("msdb_adapt_s0", "msdb_calib_wide"),
                                     ("iie_ft_klcc_e20", "iie_calib_wide")], "adapted minus calibrated (wide grid), per family")
if all(HAVE[r] for r in ("msdb_calib_wide", "iie_calib_wide", "sum_calib_wide", "msdb_zeroshot_mit1003", "iie_zeroshot_mit1003")):
    REVFAM["kbs_step_calibration3"] = ([("sum_calib_wide", "sum_zeroshot_t2"), ("msdb_calib_wide", "msdb_zeroshot_mit1003"),
                                        ("iie_calib_wide", "iie_zeroshot_mit1003")],
                                       "calibrated (wide grid) minus zero-shot with documented defaults, per family")
if HAVE["iie_zeroshot_mit1003_native"] and HAVE["iie_zeroshot_mit1003"]:
    REVFAM["kbs_iie_input1"] = ([("iie_zeroshot_mit1003_native", "iie_zeroshot_mit1003")],
                                "IIE zero-shot (official MIT1003 template): native-resolution input minus 384 input")
    REVFAM["kbs_zeroshot_doc3_iienative"] = ([("sum_zeroshot_t2", "iie_zeroshot_mit1003_native"),
                                              ("msdb_zeroshot_mit1003", "iie_zeroshot_mit1003_native")],
                                             "zero-shot pairs with IIE at native resolution (official MIT1003 template)")
REVFAM_SPLIT = {"kbs_grouped_msdb_iie": ([("msdb_adapt_grouped", "iie_ft_klcce20_grouped")], "grouped",
                                         "adapted MSDB head minus adapted IIE readouts (selected recipe) on the grouped folds"),
                "kbs_pairsafe_msdb_iie": ([("msdb_adapt_pairsafe", "iie_ft_klcce20_pairsafe")], "pairsafe",
                                          "adapted MSDB head minus adapted IIE readouts (selected recipe) on the pair-safe folds")}
for fname, spec in list(REVFAM.items()) + list(REVFAM_SPLIT.items()):
    if len(spec) == 2:
        pairs, dsc = spec; split = "random"
    else:
        pairs, split, dsc = spec
    res = {}
    for a_, b_ in pairs:
        res[(a_, b_)] = compare(R(a_), R(b_), split=split, B=10000, rng=np.random.default_rng(20261124))
    for m in METRICS:
        for key in ("wilcoxon_p", "nb_p"):
            adj = holm([res[p][m][key] for p in pairs])
            for p, x in zip(pairs, adj):
                res[p][m][key + "_holm"] = float(x)
    NEWRES[fname] = res
    fact(f"tests.family.{fname}", SEC, [f"{a_} minus {b_}" for a_, b_ in pairs], "comparisons", "computed",
         definition=f"Holm family '{fname}' (revision; computed here): {dsc}; split {split}; Holm per metric and per test "
                    "over these comparisons; CI = 10,000-sample paired bootstrap, numpy default_rng(20261124) per comparison",
         inputs=[f"per-image rows of {x} (results/{FN(x)}_per_image.json)" for p in pairs for x in p])
    for (a_, b_) in pairs:
        for m in METRICS:
            mm = res[(a_, b_)][m]
            pre = f"tests.{fname}.{a_}_minus_{b_}.{m}"
            ins = [f"per-image rows of {a_} and {b_} (paired by id)"]
            fact(pre + ".mean_diff", SEC, mm["mean_diff"], m, "computed", inputs=ins,
                 definition=f"{a_} minus {b_}: mean per-image difference (for KLD negative = first better)",
                 display=f"{mm['mean_diff']:+.4f}")
            fact(pre + ".boot95", SEC, mm["boot95"], m, "computed", inputs=ins, definition="bootstrap 95% CI")
            fact(pre + ".folds_first_better", SEC, mm["folds_first_better"], "of 5 folds", "computed", inputs=ins,
                 definition=f"folds where {a_} is better")
            fact(pre + ".fold_diffs", SEC, mm["fold_diffs"], m, "computed", inputs=ins, definition="per-fold mean differences")
            fact(pre + ".frac_images_first_better", SEC, mm["frac_images_first_better"], "fraction", "computed",
                 inputs=ins, definition=f"share of the 1000 images where {a_} is better")
            for key in ("wilcoxon_p", "wilcoxon_p_holm", "nb_p", "nb_p_holm"):
                fact(pre + "." + key, SEC, mm[key], "p", "computed", inputs=ins,
                     definition=(key + f" (Holm over family {fname})") if "holm" in key else key, display=pfmt(mm[key]))
            fact(pre + ".dz", SEC, mm["dz"], "", "computed", inputs=ins, definition="paired Cohen's dz")

# gaps and two-step decomposition (CC)
if "kbs_calibrated3" in REVFAM and "kbs_step_calibration3" in REVFAM:
    cm = {n: float(vec(R(n), "CC").mean()) for n in ("sum_zeroshot_t2", "msdb_zeroshot_mit1003", "iie_zeroshot_mit1003",
                                                       "sum_calib_wide", "msdb_calib_wide", "iie_calib_wide",
                                                       "sum_ft", "msdb_adapt_s0", "iie_ft_klcc_e20", "ours_scratch_s0")}
    step1 = {"SUM": cm["sum_calib_wide"] - cm["sum_zeroshot_t2"], "MSDB": cm["msdb_calib_wide"] - cm["msdb_zeroshot_mit1003"],
             "IIE": cm["iie_calib_wide"] - cm["iie_zeroshot_mit1003"]}
    step2 = {"SUM": cm["sum_ft"] - cm["sum_calib_wide"], "MSDB": cm["msdb_adapt_s0"] - cm["msdb_calib_wide"],
             "IIE": cm["iie_ft_klcc_e20"] - cm["iie_calib_wide"]}
    zsd = {"SUM-MSDB": cm["sum_zeroshot_t2"] - cm["msdb_zeroshot_mit1003"], "SUM-IIE": cm["sum_zeroshot_t2"] - cm["iie_zeroshot_mit1003"],
           "MSDB-IIE": cm["msdb_zeroshot_mit1003"] - cm["iie_zeroshot_mit1003"]}
    cal = {"MSDB-SUM": cm["msdb_calib_wide"] - cm["sum_calib_wide"], "MSDB-IIE": cm["msdb_calib_wide"] - cm["iie_calib_wide"],
           "SUM-IIE": cm["sum_calib_wide"] - cm["iie_calib_wide"]}
    ad3 = {"MSDB-SUM": cm["msdb_adapt_s0"] - cm["sum_ft"], "MSDB-IIE": cm["msdb_adapt_s0"] - cm["iie_ft_klcc_e20"],
           "SUM-IIE": cm["sum_ft"] - cm["iie_ft_klcc_e20"]}
    ins = ["per-image files of the zero-shot (documented defaults, official template: *_offtpl), calibrated (wide grid) and adapted rows"]
    fact("rev.gain.step_calibration_CC", SEC, step1, "CC", "computed", inputs=ins,
         definition="step 1: calibrated (wide grid) minus zero-shot with documented defaults, per family")
    fact("rev.gain.step_training_CC", SEC, step2, "CC", "computed", inputs=ins,
         definition="step 2: adapted minus calibrated (wide grid), per family")
    fact("rev.gain.step_training_min_CC", SEC, min(step2.values()), "CC", "computed", inputs=ins, definition="smallest step-2 gain")
    fact("rev.gain.step_training_max_CC", SEC, max(step2.values()), "CC", "computed", inputs=ins, definition="largest step-2 gain")
    fact("rev.gain.step_calibration_min_CC", SEC, min(step1.values()), "CC", "computed", inputs=ins, definition="smallest step-1 gain")
    fact("rev.gain.step_calibration_max_CC", SEC, max(step1.values()), "CC", "computed", inputs=ins, definition="largest step-1 gain")
    fact("rev.gap.zeroshot_doc_CC", SEC, zsd, "CC", "computed", inputs=ins, definition="CC gaps among zero-shot rows (documented defaults)")
    fact("rev.gap.calibrated_CC", SEC, cal, "CC", "computed", inputs=ins, definition="CC gaps among calibrated rows (wide grid)")
    fact("rev.gap.adapted3_CC", SEC, ad3, "CC", "computed", inputs=ins, definition="CC gaps among the three adapted models")
    fact("rev.gain.step_training_exceeds_every_adapted_gap", SEC,
         min(step2.values()) > max(abs(x) for x in list(ad_gaps.values())), "", "computed", inputs=ins,
         definition="the smallest step-2 gain exceeds every gap between the four adapted rows (incl. the readout)")
    fact("rev.ranking.zeroshot_doc", SEC, " > ".join(sorted(["SUM", "MSDB", "IIE"], key=lambda k: -{"SUM": cm["sum_zeroshot_t2"],
         "MSDB": cm["msdb_zeroshot_mit1003"], "IIE": cm["iie_zeroshot_mit1003"]}[k])), "", "computed", inputs=ins,
         definition="CC order of the zero-shot rows with documented defaults")
    fact("rev.ranking.calibrated", SEC, " > ".join(sorted(["SUM", "MSDB", "IIE"], key=lambda k: -{"SUM": cm["sum_calib_wide"],
         "MSDB": cm["msdb_calib_wide"], "IIE": cm["iie_calib_wide"]}[k])), "", "computed", inputs=ins,
         definition="CC order of the calibrated rows (wide grid)")

# calibration-only learning curves (fold 0)
for part in ("msdb", "iie", "sum"):
    fn = os.path.join(RES, REVCURVE[part])
    if not os.path.exists(fn):
        WARN.append(f"REVISION CURVE MISSING: results/{REVCURVE[part]}")
        continue
    cc_ = J(fn)
    for n, e in cc_["by_n"].items():
        for m in METRICS:
            fact(f"calib.{part}.n{n}.{m}", SEC, src=fn, path=f"by_n/{n}/{m}", unit=m,
                 definition=f"calibration only, {part}, n = {n} fold-0 training maps (mean over resamples; n = 0 zero-shot"
                            + (", official MIT1003 template" if part != "sum" else ", token 2") + f"; file {REVCURVE[part]})")
        if n != "0":
            fact(f"calib.{part}.n{n}.CC.sd_resamples_ddof1", SEC, src=fn, path=f"by_n/{n}/CC_sd_resamples_ddof1", unit="CC",
                 sd_ddof=1, definition=f"calibration only, {part}, n = {n}: SD over resample means (ddof 1; null for one run)")
            fact(f"calib.{part}.n{n}.sigmas", SEC, [x["sigma"] for x in e["runs"]], "px at 384", "computed",
                 definition="blur chosen per resample", inputs=[f"results/{REVCURVE[part]} by_n/{n}/runs"])
            if part == "sum":
                fact(f"calib.sum.n{n}.tokens", SEC, [x["token"] for x in e["runs"]], "", "computed",
                     definition="SUM token chosen per resample", inputs=["results/calib_curve_sum.json"])

# calibration saturation: largest gap between 50-map and 800-map calibration over the three models (fold 0)
_cc = {}
for part in ("msdb", "iie", "sum"):
    fn = os.path.join(RES, REVCURVE[part])
    if os.path.exists(fn):
        bn = J(fn)["by_n"]
        if "50" in bn and "800" in bn:
            _cc[part] = bn["800"]["CC"] - bn["50"]["CC"]
if len(_cc) == 3:
    fact("rev.calib.gap_n50_to_n800_CC", SEC, _cc, "CC", "computed",
         definition="calibration only, fold 0: CC at n = 800 minus CC at n = 50, per model",
         inputs=["results/calib_curve_{msdb,iie}_offtpl.json and calib_curve_sum.json by_n"])
    fact("rev.calib.max_gap_n50_to_n800_CC", SEC, max(_cc.values()), "CC", "computed",
         definition="largest |CC(800) - CC(50)| of calibration over the three models (fold 0)",
         inputs=["results/calib_curve_{msdb,iie}_offtpl.json and calib_curve_sum.json by_n"])

# map accounting (ADD1000 releases one density map per image)
rec = 56978 / 1000.0
for n in (25, 50, 100, 200, 800):
    fact(f"rev.accounting.n{n}.recordings", SEC, n * rec, "recordings", "computed",
         definition=f"approximate valid individual recordings behind {n} maps (n x 56,978 / 1000; average, not exact)",
         inputs=["dataset.valid_individual_records", "dataset.n_images"])
    fact(f"rev.accounting.n{n}.viewing_hours", SEC, n / 1000.0 * 254176 / 3600.0, "h", "computed",
         definition=f"approximate recorded viewing time behind {n} maps (n/1000 x 254,176 s / 3600)",
         inputs=["dataset.total_viewing_s", "dataset.n_images"])
fact("rev.accounting.recordings_per_map_mean", SEC, rec, "recordings", "computed",
     definition="mean valid recordings per map (56,978 / 1000)", inputs=["dataset.valid_individual_records"])

# near-duplicate bound for SUM (no pair-safe SUM run): images of the cross-fold extended pairs
xids = sorted({x for p in ND["extended"]["pairs"] if FOLD_OF["random"][p["a"]] != FOLD_OF["random"][p["b"]] for x in (p["a"], p["b"])}, key=int)
rest = [i for i in sorted(R("sum_ft"), key=int) if i not in set(xids)]
for a_, b_ in (("sum_ft", "msdb_adapt_s0"), ("msdb_adapt_s0", "iie_ft_klcc_e20")):
    dx = vec(R(a_), "CC", xids) - vec(R(b_), "CC", xids)
    dr = vec(R(a_), "CC", rest) - vec(R(b_), "CC", rest)
    fact(f"rev.neardup.{a_}_minus_{b_}.CC.on_pairs", SEC, float(dx.mean()), "CC", "computed",
         definition=f"{a_} minus {b_}, mean CC difference on the {len(xids)} images of the cross-fold extended near-duplicate pairs",
         inputs=["splits/near_dup_pairs.json extended", "per-image rows"])
    fact(f"rev.neardup.{a_}_minus_{b_}.CC.elsewhere", SEC, float(dr.mean()), "CC", "computed",
         definition=f"{a_} minus {b_} on the other {len(rest)} images", inputs=["per-image rows"])
fact("rev.neardup.n_images_cross_fold_pairs", SEC, len(xids), "images", "computed",
     definition="images in the cross-fold extended near-duplicate pairs", inputs=["splits/near_dup_pairs.json extended"])
fact("rev.neardup.sum_ft.CC.on_pairs", SEC, float(vec(R("sum_ft"), "CC", xids).mean()), "CC", "computed",
     definition="sum_ft mean CC on the cross-fold near-duplicate images", inputs=["results/sum_ft_per_image.json"])

# UEyes overlap with the paper's near-duplicate detector
if os.path.exists(os.path.join(RES, "ueyes_overlap.json")):
    for k in ("n_ueyes", "max_cos", "n_pairs_ge_0.94", "n_pairs_ge_0.925"):
        fact(f"rev.ueyes.{k}", SEC, src="ueyes_overlap.json", path=k, definition=f"ADD1000 vs UEyes: {k} (mean DINOv2-B patch cosine)")

# cost: GFLOPs ratio with SUM's scan estimate included
_g_msdb = [f for f in FACTS if f["id"] == "cost.msdb.GFLOPs"][0]["value"]
_g_sum_incl = [f for f in FACTS if f["id"] == "cost.sum.GFLOPs_incl_scan"][0]["value"]
fact("cost.ratio.msdb_over_sum_GFLOPs_incl_scan", SEC, _g_msdb / _g_sum_incl, "x", "computed",
     definition="MSDB GFLOPs over SUM GFLOPs including the scan estimate (12,256.2 / 31.0)",
     inputs=["cost.msdb.GFLOPs", "cost.sum.GFLOPs_incl_scan"])

# native-resolution check over every row with a native score
dn = {m: [] for m in METRICS}
for r_ in MT["rows"]:
    if "native_resolution_means" in r_:
        for m in METRICS:
            dn[m].append(abs(r_["native_resolution_means"][m] - r_["stats_384"][m]["mean"]))
for m in METRICS:
    fact(f"rev.native.max_abs_d{m}_all_rows", SEC, max(dn[m]), m, "computed",
         definition=f"largest |{m}(native grid) - {m}(384)| over the {len(dn[m])} main-table rows with a native score",
         inputs=["results/main_table.json rows[*]/native_resolution_means, stats_384"])
fact("rev.native.n_rows", SEC, len(dn["CC"]), "rows", "computed", definition="main-table rows with a native score",
     inputs=["results/main_table.json"])

# pretraining fixation data (documentation, not results)
fact("rev.pretrain.sum", SEC, "SALICON, MIT1003, CAT2000, SALECI (e-commerce), UEYE (interfaces), FIWI (webpages)", "", "computed",
     definition="SUM training sets as listed in its README", inputs=["third_party/SUM/code/README.md line 173"])
fact("rev.pretrain.msdb", SEC, "MIT1003, CAT2000, COCO-Freeview, DAEMONS, FIGRIM", "", "computed",
     definition="DeepGaze MSDB training datasets (deepgaze_pytorch MSDBDataset enum)",
     inputs=["deepgaze_pytorch/deepgazemsdb.py MSDBDataset"])
fact("rev.pretrain.iie", SEC, "SALICON (pretraining) and MIT1003", "", "computed",
     definition="DeepGaze IIE training data per Linardos et al. (2021)", inputs=["linardos2021iie"])

# SUM full fine-tune learning curve (landed 24 Nov 2026 09:04): SIM/KLD, ddof-1 SDs, per-budget tests vs MSDB head
if os.path.exists(os.path.join(RES, "sum_ft_curve_per_image.json")):
    scpi = J("sum_ft_curve_per_image.json")
    cvs = curve_from_rows(scpi, NS)
    for n in sorted(cvs):
        runs = sorted(cvs[n])
        for m in METRICS:
            vals = [cvs[n][r][m] for r in runs]
            fact(f"curve.sum_ft.n{n}.{m}.mean_rev", SEC, float(np.mean(vals)), m, "computed",
                 definition=f"SUM full FT curve, n = {n}, {m}: mean over {len(vals)} run(s) (recomputed from per-image rows)",
                 inputs=["results/sum_ft_curve_per_image.json"])
            fact(f"curve.sum_ft.n{n}.{m}.sd_resamples_ddof1", SEC, float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                 m, "computed", sd_ddof=1, definition=f"SD over resample means (ddof 1); null for one run",
                 inputs=["results/sum_ft_curve_per_image.json"])
        fact(f"curve.sum_ft.n{n}.runs", SEC, len(runs), "runs", "computed", definition="runs at this n",
             inputs=["results/sum_ft_curve_per_image.json"])
    msdbpi = J("msdb_adapt_curves_per_image.json")

    def avg_by_id(rows, n):
        d = {}
        for x in rows:
            if int(x["n"]) == n:
                d.setdefault(str(x["id"]), []).append(x)
        return {i: {m: float(np.mean([y[m] for y in v])) for m in METRICS} for i, v in d.items()}

    tests = {}
    for n in NS:
        a_, b_ = avg_by_id(scpi, n), avg_by_id(msdbpi, n)
        ids = sorted(a_, key=int)
        assert sorted(b_, key=int) == ids and len(ids) == 200
        for m in METRICS:
            av, bv = np.array([a_[i][m] for i in ids]), np.array([b_[i][m] for i in ids])
            tests[(n, m)] = {"mean_diff": float((av - bv).mean()), "wilcoxon_p": float(stats.wilcoxon(av, bv).pvalue),
                             "frac_first_better": float(((av - bv) > 0).mean() if HIGHER[m] else ((av - bv) < 0).mean())}
    keys = list(tests)
    adj = holm([tests[k]["wilcoxon_p"] for k in keys])
    for k, x in zip(keys, adj):
        tests[k]["wilcoxon_p_holm"] = float(x)
    for (n, m), t in tests.items():
        pre = f"curve.test.sum_ft_vs_msdb_adapt_head.n{n}.{m}"
        fact(pre + ".mean_diff", SEC, t["mean_diff"], m, "computed", display=f"{t['mean_diff']:+.4f}",
             definition=f"SUM full FT minus MSDB head at n = {n} (fold 0; per-image values averaged over the runs at this n)",
             inputs=["results/sum_ft_curve_per_image.json", "results/msdb_adapt_curves_per_image.json"])
        fact(pre + ".wilcoxon_p_holm", SEC, t["wilcoxon_p_holm"], "p", "computed", display=pfmt(t["wilcoxon_p_holm"]),
             definition="Wilcoxon p, Holm over 6 budgets x 3 metrics within this comparison", inputs=["as above"])
        fact(pre + ".frac_first_better", SEC, t["frac_first_better"], "fraction", "computed",
             definition="share of the 200 test images where SUM is better", inputs=["as above"])
    WARN.append("SUM CURVE landed: n = 400 is one run and n = 800 is sum_ft fold 0 (sigma on 200 of the 800 maps, data "
                "order default_rng(0)); 12 epochs at every n, so n = 25 has 24 optimiser steps.")

# ========================================================================================== 18 ftblocks and native IIE (24 Nov 2026, evening)
# (A) ours_ftblocks_nested: xSoyelNet with the last N DINOv2-B blocks fine-tuned with the readout, N chosen per fold by
#     nested inner validation from the pre-registered grid (scripts/ftblocks/PLAN.json, frozen 08:34:23 UTC, Amendment 1);
#     independently re-scored and leakage-checked on the CPU before use (all checks passed).
# (B) iie_native_adapt: DeepGaze IIE readouts adapted with native-resolution input, recipe of iie_ft_klcc_e20.
# The distillation row of the ftblocks plan (ours_ftblocks_distill) is NOT read here.
SEC = "ftblocks"
FTDIR = os.path.join(KB, "results", "ftblocks_runs")   # run records (copied from cache/ftblocks/runs)
PLANF = os.path.join(KB, "scripts", "ftblocks", "PLAN.json")
PLAN = J(PLANF)
FT = "ours_ftblocks_nested"
FTS = J(f"{FT}_summary.json")
FTC = J("ftblocks_cost.json")
CFGS = [("N0", "N0"), ("N2", "N2_lr3e-05"), ("N4", "N4_lr3e-05")]
_ftrows = R(FT)
for m in METRICS:
    dd = desc(_ftrows, m)
    fact(f"ft.nested.{m}.mean", SEC, src=f"{FT}_summary.json", path=f"mean_{m}", unit=m,
         definition=f"xSoyelNet, last 4 DINOv2-B blocks fine-tuned (nested selection): {m} mean of the 5 fold means",
         recomputed=dd["mean_of_fold_means"])
    fact(f"ft.nested.{m}.sd_folds", SEC, src=f"{FT}_summary.json", path=f"sd_{m}_ddof1", unit=m, sd_ddof=1,
         definition=f"fine-tuned xSoyelNet: SD of the 5 fold means ({m})", recomputed=dd["sd_ddof1"])
    fact(f"ft.nested.{m}.per_fold", SEC, [x[m] for x in FTS["per_fold"]], m, "computed",
         definition=f"fine-tuned xSoyelNet: {m} per outer fold (folds 0 to 4)", inputs=[f"results/{FT}_summary.json per_fold"],
         recomputed=dd["per_fold"])
fact("ft.nested.config_per_fold", SEC, src=f"{FT}_summary.json", path="config_used_per_fold",
     definition="config used in the final run of each outer fold (selected by the inner rule; no fallback needed)")
fact("ft.nested.sigma_per_fold", SEC, [x["sigma"] for x in FTS["folds_detail"]], "px at 384", "computed",
     definition="post-hoc blur sigma per fold (grid 0..12, chosen on 200 of the fold's 800 training maps)",
     inputs=[f"results/{FT}_summary.json folds_detail[*]/sigma"])
fact("ft.nested.trained_blocks", SEC, src=f"{FT}_summary.json", path="folds_detail[fold=0]/config/trained_blocks",
     definition="DINOv2-B transformer blocks fine-tuned (with the final LayerNorm and the readout); blocks 1 to 8 frozen")
fact("ft.nested.lr_backbone", SEC, src=f"{FT}_summary.json", path="folds_detail[fold=0]/config/lr_bb",
     definition="learning rate of the fine-tuned blocks and final norm (AdamW, wd 0.05, 5% linear warm-up, then constant); "
                "readout lr 3e-4, wd 1e-4")
_blk = 3 * 768 * 768 + 3 * 768 + 768 * 768 + 768 + 768 * 3072 + 3072 + 3072 * 768 + 768 + 4 * 768 + 2 * 768
fact("ft.nested.trainable_params", SEC, src=os.path.join(FTDIR, "final_k0_N4_lr3e-05.json"), path="trainable_params",
     unit="parameters", definition="parameters trained in the fine-tuned row: 4 DINOv2-B blocks + final LayerNorm + "
     "readout (recomputed: 4 x 7,089,408 + 1,536 + 3,721,089)", recomputed=4 * _blk + 2 * 768 + 3721089)
fact("ft.nested.params_total", SEC, src="ftblocks_cost.json", path="params_total", unit="parameters",
     definition="parameters of the fine-tuned model at inference (re-counted from the fold-0 checkpoint)",
     recomputed=CT["inference"]["ours"]["params_total"])
fact("ft.nested.GFLOPs", SEC, src="ftblocks_cost.json", path="GFLOPs", unit="GFLOPs",
     definition="GFLOPs per image of the fine-tuned model, re-measured from the fold-0 checkpoint (2 x MACs, batch 1, "
                "448 x 448); equals the frozen xSoyelNet graph", recomputed=CT["inference"]["ours"]["GFLOPs"])
fact("ft.nested.cpu_latency_note", SEC, src="ftblocks_cost.json", path="cpu_latency_s",
     definition="CPU latency of the fine-tuned checkpoint was not re-timed (machine not quiet); the graph is the frozen "
                "xSoyelNet graph, whose quiet-machine latency is cost.ours.cpu_latency_s")
fact("ft.nested.end_to_end_max_abs_diff", SEC, max(FTC["end_to_end_check"]["max_abs_diff_sigmoid_384"]), "", "computed",
     definition="full pixel pipeline vs the cached training path on 4 images (fold-0 checkpoint, CPU)",
     inputs=["results/ftblocks_cost.json end_to_end_check"])
fact("ft.cost.ratio.msdb_over_ft_GFLOPs", SEC, CT["inference"]["msdb"]["GFLOPs"] / FTC["GFLOPs"], "x", "computed",
     definition="MSDB GFLOPs over the fine-tuned xSoyelNet's GFLOPs", inputs=["results/cost_table.json inference/msdb/GFLOPs",
                                                                              "results/ftblocks_cost.json GFLOPs"])
# pre-registration and nested selection
fact("ft.plan.frozen_utc", SEC, src=PLANF, path="frozen_utc", definition="time PLAN.json was frozen (before any grid run)")
fact("ft.plan.sha256", SEC, hashlib.sha256(open(PLANF, "rb").read()).hexdigest(), "", "computed",
     definition="sha256 of PLAN.json (equals the plan_sha256 stored in every run file)", inputs=[PLANF])
fact("ft.plan.sha256_matches_runs", SEC, hashlib.sha256(open(PLANF, "rb").read()).hexdigest() == FTS["plan_sha256"], "",
     "computed", definition="PLAN.json unchanged since the runs", inputs=[PLANF, f"results/{FT}_summary.json plan_sha256"])
fact("ft.plan.full_grid", SEC, [g["name"] for g in PLAN["full_grid"]], "", "computed",
     definition="pre-registered full grid", inputs=[PLANF + " full_grid"])
fact("ft.plan.active_grid", SEC, [g["name"] for g in PLAN["active_grid"]], "", "computed",
     definition="grid actually run: the pre-registered fallback (lr 1e-5 dropped by the pre-registered time rule, "
                "full-grid estimate above 10 h, decided before any run)", inputs=[PLANF + " active_grid, fallback"])
fact("ft.plan.mode", SEC, src=PLANF, path="mode", definition="grid mode applied")
fact("ft.plan.tie_cc", SEC, src=PLANF, path="selection/tie_cc", unit="CC",
     definition="selection tie margin: configs within this inner-val CC of the best are tied; the simplest tied one wins")
fact("ft.plan.inner_rule", SEC, src=PLANF, path="inner_split/rule", definition="inner split rule (per outer fold k)")
fact("ft.plan.recipe_N_gt_0", SEC, src=PLANF, path="protocol/N_gt_0",
     definition="fine-tuning recipe for N > 0 (optimiser groups, weight decay 0.05 and 5% warm-up of the blocks, steps)")
fact("ft.plan.seeds", SEC, src=PLANF, path="protocol/seeds", definition="seeds of the ftblocks runs")
fact("ft.plan.precision", SEC, src=PLANF, path="protocol/precision", definition="numerical precision of the ftblocks runs")
fact("ft.plan.augmentation", SEC, src=PLANF, path="protocol/augmentation", definition="augmentation of the ftblocks runs")
fact("ft.plan.fallback_rule", SEC, src=PLANF, path="fallback/rule",
     definition="pre-registered time rule that reduced the grid from five to three configurations before any run")
fact("ft.plan.amendment1_sha256", SEC, src=f"{FT}_summary.json", path="amendment_sha256",
     definition="sha256 of PLAN_AMENDMENT_1.json (written before any grid run) stored in every run file")
_isp = J(os.path.join(KB, "scripts", "ftblocks", "inner_splits.json"))
_inner = {}
for k in range(5):
    S_ = J(os.path.join(FTDIR, f"selection_k{k}.json"))
    for short, cfg in CFGS:
        rj = J(os.path.join(FTDIR, f"inner_k{k}_{cfg}.json"))
        assert rj["n_train"] == 640 and rj["n_eval"] == 160 and rj["status"] == "ok", (k, cfg)
        vids = {str(x["id"]) for x in rj["rows"]}
        assert not (vids & {str(i) for i in KF["folds"][k]}), ("inner-val ids overlap the outer test fold", k, cfg)
        assert vids == {str(i) for i in _isp["folds"][k]["inner_val_ids"]}, ("inner-val ids differ from inner_splits.json", k, cfg)
        rec = float(np.mean([x["CC"] for x in rj["rows"]]))
        _inner[(k, short)] = rec
        fact(f"ft.inner.k{k}.{short}.CC", SEC, src=os.path.join(FTDIR, f"selection_k{k}.json"), path=f"inner_val_CC/{cfg}",
             unit="CC", definition=f"outer fold {k}: inner-validation CC of config {cfg} (trained on 640 of the fold's 800 "
             f"training maps, scored on the other 160; no outer test image)", recomputed=rec)
        fact(f"ft.inner.k{k}.{short}.wall_s", SEC, src=os.path.join(FTDIR, f"selection_k{k}.json"), path=f"inner_wall_s/{cfg}",
             unit="s", definition=f"outer fold {k}: wall clock of the inner run {cfg} (Apple M4 Pro GPU via MPS, float32; "
             "machine shared with other jobs, indicative)")
    fact(f"ft.inner.k{k}.selected", SEC, src=os.path.join(FTDIR, f"selection_k{k}.json"), path="selected",
         definition=f"outer fold {k}: config selected by the pre-registered rule")
    best = max(_inner[(k, s)] for s, _ in CFGS)
    sel = [s for s, _ in CFGS if _inner[(k, s)] >= best - PLAN["selection"]["tie_cc"]][0]
    assert dict(CFGS)[sel] == S_["selected"], (k, sel, S_["selected"])
    fact(f"ft.inner.k{k}.margin_N4_over_N2_CC", SEC, _inner[(k, "N4")] - _inner[(k, "N2")], "CC", "computed",
         definition=f"outer fold {k}: inner-val CC margin of the selected N4 over the runner-up N2",
         inputs=[f"cache/ftblocks/runs/selection_k{k}.json"])
fact("ft.inner.n_train", SEC, 640, "maps", "computed", definition="inner-train maps per outer fold (of 800)",
     inputs=["cache/ftblocks/runs/inner_k*_*.json n_train"])
fact("ft.inner.n_val", SEC, 160, "maps", "computed", definition="inner-validation maps per outer fold (of 800)",
     inputs=["cache/ftblocks/runs/inner_k*_*.json n_eval"])
_m = [_inner[(k, "N4")] - _inner[(k, "N2")] for k in range(5)]
fact("ft.inner.margin_N4_over_N2_range_CC", SEC, [min(_m), max(_m)], "CC", "computed",
     definition="range over the 5 outer folds of the inner-val margin of N4 over N2", inputs=["selection_k*.json"])
for short, cfg in CFGS:
    fact(f"ft.inner.mean_over_folds.{short}.CC", SEC, float(np.mean([_inner[(k, short)] for k in range(5)])), "CC",
         "computed", definition=f"inner-val CC of {cfg} averaged over the 5 outer folds (descriptive; the selection is per fold)",
         inputs=["selection_k*.json"])
fact("ft.inner.n_folds_N4_selected", SEC, sum(1 for k in range(5) if J(os.path.join(FTDIR, f"selection_k{k}.json"))["selected"] == "N4_lr3e-05"),
     "of 5 folds", "computed", definition="outer folds in which the nested rule selected N4_lr3e-05", inputs=["selection_k*.json"])
_gl = J("ours_ftblocks_global_per_image.json")
fact("ft.global.identical_to_nested", SEC, _gl == J(f"{FT}_per_image.json"), "", "computed",
     definition="the secondary, NOT nested global row (one config from the inner-val CC pooled over folds; also N4) has "
                "per-image rows identical to the nested row (same final runs)", inputs=["results/ours_ftblocks_global_per_image.json",
                                                                                         f"results/{FT}_per_image.json"])
# training cost (final runs; hardware labelled)
_fw = [x["wall_s"] for x in FTS["folds_detail"]]
fact("ft.train.final.wall_s_per_fold", SEC, _fw, "s", "computed",
     definition="final-run wall clock per outer fold (3000 steps, batch 16, float32; Apple M4 Pro GPU via MPS; machine "
                "shared with a CPU job, indicative)", inputs=[f"results/{FT}_summary.json folds_detail[*]/wall_s"])
fact("ft.train.final.median_s_per_fold", SEC, float(np.median(_fw)), "s", "computed", definition="median of the above",
     inputs=[f"results/{FT}_summary.json"])
fact("ft.train.final.s_per_step_range", SEC, [min(x["s_per_step"] for x in FTS["folds_detail"]),
                                              max(x["s_per_step"] for x in FTS["folds_detail"])], "s", "computed",
     definition="final runs: seconds per optimiser step (batch 16), range over folds", inputs=[f"results/{FT}_summary.json"])
fact("ft.train.final.steps", SEC, src=f"{FT}_summary.json", path="folds_detail[fold=0]/steps", unit="steps",
     definition="optimiser steps per final run (60 epochs of 800 maps at batch 16)")
fact("ft.train.device", SEC, src=f"{FT}_summary.json", path="folds_detail[fold=0]/device",
     definition="training device ('mps' = Apple M4 Pro GPU through Metal Performance Shaders)")
_iw = sum(J(os.path.join(FTDIR, f"selection_k{k}.json"))["inner_wall_s"][cfg] for k in range(5) for _, cfg in CFGS)
fact("ft.train.inner.wall_s_total", SEC, _iw, "s", "computed", definition="wall clock of all 15 inner runs (selection)",
     inputs=["selection_k*.json inner_wall_s"])
fact("ft.train.protocol_total_h", SEC, (_iw + sum(_fw)) / 3600.0, "h", "computed",
     definition="wall clock of the whole nested protocol: 15 inner runs + 5 final runs (MPS; indicative)",
     inputs=["selection_k*.json", f"results/{FT}_summary.json"])
fact("ft.nested.protocol_maps", SEC, {"calibration_maps": 200, "training_maps": 800, "selection_maps": "640/160 inner split of the 800",
                                       "trainable_params": FTS["folds_detail"][0]["trainable_params"]}, "maps per fold", "computed",
     definition="ADD1000 maps per fold: blur from 200 of the 800 training maps; config selected on an inner 640/160 split "
                "of the 800; the 200 test maps scored once", inputs=[f"results/{FT}_summary.json"])

# (B) DeepGaze IIE adapted with native-resolution input
IN = "iie_native_adapt"
INS = J(f"{IN}_summary.json")
_inrows = R(IN)
for m in METRICS:
    dd = desc(_inrows, m)
    fact(f"iie_native.{m}.mean", SEC, src=f"{IN}_summary.json", path=f"{m}_mean", unit=m,
         definition=f"DeepGaze IIE readouts adapted with native-resolution input: {m} mean of the 5 fold means",
         recomputed=dd["mean_of_fold_means"])
    fact(f"iie_native.{m}.sd_folds", SEC, src=f"{IN}_summary.json", path=f"{m}_sd_over_folds_ddof1", unit=m, sd_ddof=1,
         definition=f"native-input adapted IIE: SD of the 5 fold means ({m})", recomputed=dd["sd_ddof1"])
    fact(f"iie_native.{m}.per_fold", SEC, [x[m] for x in INS["per_fold"]], m, "computed",
         definition=f"native-input adapted IIE: {m} per fold", inputs=[f"results/{IN}_summary.json per_fold"],
         recomputed=dd["per_fold"])
for m in METRICS:
    fact(f"iie384.{m}.per_fold", SEC, desc(R("iie_ft_klcc_e20"), m)["per_fold"], m, "computed",
         definition=f"adapted IIE readouts with 384 input (iie_ft_klcc_e20): {m} per fold (folds 0 to 4)",
         inputs=["results/iie_ft_klcc_e20_per_image.json"])
fact("iie384.sigma_per_fold", SEC, [x["sigma"] for x in J("iie_ft_klcc_e20_summary.json")["per_fold"]]
     if "sigma" in J("iie_ft_klcc_e20_summary.json")["per_fold"][0] else J("iie_ft_klcc_e20_summary.json").get("sigma_per_fold"),
     "px at 384", "computed", definition="adapted IIE at 384: blur sigma per fold", inputs=["results/iie_ft_klcc_e20_summary.json"])
fact("iie_native.sigma_per_fold", SEC, src=f"{IN}_summary.json", path="sigma_per_fold", unit="px at 384",
     definition="post-hoc blur sigma per fold (grid 0..12 on 200 training maps)")
fact("iie_native.trainable_params", SEC, src=f"{IN}_summary.json", path="trainable_params_trained", unit="parameters",
     definition="IIE readout parameters trained (backbones frozen), as in the 384 row", recomputed=3059160)
fact("iie_native.image_size_range_WH", SEC, src=f"{IN}_summary.json", path="image_size_range_WH",
     definition="smallest and largest ADD1000 image (W, H) fed at native resolution")
fact("iie_native.train_s_per_fold", SEC, src=f"{IN}_summary.json", path="t_train_s_per_fold", unit="s",
     definition="training wall clock per fold (Apple M4 Pro CPU, 3 threads, two folds in parallel; indicative)")
fact("iie_native.steps_per_fold", SEC, src=f"{IN}_summary.json", path="steps_per_fold", unit="steps",
     definition="optimiser steps per fold (20 epochs, batch 8)")

# families
FTFAM = {
    "kbs_adapted_pairwise10": ([("ours_ftblocks_nested", "msdb_adapt_s0"), ("ours_ftblocks_nested", "sum_ft"),
                                ("ours_ftblocks_nested", "ours_scratch_s0"), ("ours_ftblocks_nested", "iie_native_adapt"),
                                ("msdb_adapt_s0", "sum_ft"), ("msdb_adapt_s0", "ours_scratch_s0"),
                                ("msdb_adapt_s0", "iie_native_adapt"), ("sum_ft", "ours_scratch_s0"),
                                ("sum_ft", "iie_native_adapt"), ("ours_scratch_s0", "iie_native_adapt")],
                               "all 10 pairs of the 5 adapted rows of Table 1 (fine-tuned xSoyelNet, MSDB head, SUM full "
                               "fine-tune, frozen xSoyelNet, IIE readouts with native input)"),
    "kbs_iie_input_adapted1": ([("iie_native_adapt", "iie_ft_klcc_e20")],
                               "adapted IIE readouts (same recipe): native-resolution input minus 384 input"),
    "kbs_ftblocks_prereg7": ([("ours_ftblocks_nested", x) for x in ("ours_scratch_s0", "ours_scratch_seedavg", "distill_pitt",
                                                                     "ours_L448_s0", "ours_ms_B_s0", "msdb_adapt_s0", "sum_ft")],
                             "the pre-registered primary family of PLAN.json: fine-tuned xSoyelNet minus 7 references"),
}
for fname, (pairs, dsc) in FTFAM.items():
    res = {}
    for a_, b_ in pairs:
        res[(a_, b_)] = compare(R(a_), R(b_), B=10000, rng=np.random.default_rng(20261124))
    for m in METRICS:
        for key in ("wilcoxon_p", "nb_p"):
            adj = holm([res[p][m][key] for p in pairs])
            for p, x in zip(pairs, adj):
                res[p][m][key + "_holm"] = float(x)
    NEWRES[fname] = res
    fact(f"tests.family.{fname}", SEC, [f"{a_} minus {b_}" for a_, b_ in pairs], "comparisons", "computed",
         definition=f"Holm family '{fname}' (computed here): {dsc}; split random; Holm per metric and per test over these "
                    "comparisons; CI = 10,000-sample paired bootstrap, numpy default_rng(20261124) per comparison",
         inputs=[f"per-image rows of {x} (results/{FN(x)}_per_image.json)" for p in pairs for x in p])
    for (a_, b_) in pairs:
        for m in METRICS:
            mm = res[(a_, b_)][m]
            pre = f"tests.{fname}.{a_}_minus_{b_}.{m}"
            ins = [f"per-image rows of {a_} and {b_} (paired by id)"]
            fact(pre + ".mean_diff", SEC, mm["mean_diff"], m, "computed", inputs=ins,
                 definition=f"{a_} minus {b_}: mean per-image difference (for KLD negative = first better)",
                 display=f"{mm['mean_diff']:+.4f}")
            fact(pre + ".boot95", SEC, mm["boot95"], m, "computed", inputs=ins, definition="bootstrap 95% CI")
            fact(pre + ".folds_first_better", SEC, mm["folds_first_better"], "of 5 folds", "computed", inputs=ins,
                 definition=f"folds where {a_} is better")
            fact(pre + ".fold_diffs", SEC, mm["fold_diffs"], m, "computed", inputs=ins, definition="per-fold mean differences")
            fact(pre + ".frac_images_first_better", SEC, mm["frac_images_first_better"], "fraction", "computed",
                 inputs=ins, definition=f"share of the 1000 images where {a_} is better")
            fact(pre + ".nb_t", SEC, mm["nb_t"], "", "computed", inputs=ins, definition="Nadeau-Bengio corrected t (df 4)")
            for key in ("wilcoxon_p", "wilcoxon_p_holm", "nb_p", "nb_p_holm"):
                fact(pre + "." + key, SEC, mm[key], "p", "computed", inputs=ins,
                     definition=(key + f" (Holm over family {fname})") if "holm" in key else key, display=pfmt(mm[key]))
            fact(pre + ".dz", SEC, mm["dz"], "", "computed", inputs=ins, definition="paired Cohen's dz")
# cross-check with the paired tests written by the native-IIE run (compile_results functions executed from source)
_IPT = J("iie_native_adapt_paired_tests.json")["families"]
for fam_src, mine_fam, pairs in (("kbs_adapted_pairwise6_iienative", "kbs_adapted_pairwise10",
                                  [("msdb_adapt_s0", "sum_ft"), ("msdb_adapt_s0", "ours_scratch_s0"), ("msdb_adapt_s0", "iie_native_adapt"),
                                   ("sum_ft", "ours_scratch_s0"), ("sum_ft", "iie_native_adapt"), ("ours_scratch_s0", "iie_native_adapt")]),
                                 ("iie_native_vs_384", "kbs_iie_input_adapted1", [("iie_native_adapt", "iie_ft_klcc_e20")])):
    for a_, b_ in pairs:
        for m in METRICS:
            comp = _IPT[fam_src]["results"][f"{a_}_minus_{b_}"][m]
            mine_v = NEWRES[mine_fam][(a_, b_)][m]
            ok = (_close(mine_v["mean_diff"], comp["mean_diff"], 1e-12) and _close(mine_v["wilcoxon_p"], comp["wilcoxon_p"], 0)
                  and _close(mine_v["nb_p"], comp["nb_p"], 0) and mine_v["folds_first_better"] == comp["folds_ref_better"])
            CHECKS.append({"id": f"{mine_fam} {a_}-{b_} {m} vs iie_native_adapt_paired_tests.json {fam_src} (raw)", "match": ok,
                           "mean_diff_here": mine_v["mean_diff"], "compiled": comp["mean_diff"]})
            if not ok:
                WARN.append(f"MISMATCH {mine_fam} {a_}-{b_} {m} vs iie_native_adapt_paired_tests.json {fam_src}")
# summary facts for the fine-tuned row (claims of Sec. 3)
_P10 = NEWRES["kbs_adapted_pairwise10"]
_refs = ["msdb_adapt_s0", "sum_ft", "ours_scratch_s0", "iie_native_adapt"]
_ok5 = all(_P10[("ours_ftblocks_nested", r)][m]["folds_first_better"] == 5 for r in _refs for m in METRICS)
_okt = all(_P10[("ours_ftblocks_nested", r)][m][k] < 0.05 for r in _refs for m in METRICS for k in ("wilcoxon_p_holm", "nb_p_holm"))
_okd = all((_P10[("ours_ftblocks_nested", r)][m]["mean_diff"] > 0) == HIGHER[m] for r in _refs for m in METRICS)
fact("ft.claim.leads_every_adapted_row_5of5_every_metric", SEC, _ok5 and _okd, "", "computed",
     definition="the fine-tuned xSoyelNet is better than each other adapted row of Table 1 on CC, SIM and KLD in every one "
                "of the 5 outer folds", inputs=["tests.kbs_adapted_pairwise10.*"])
fact("ft.claim.both_tests_reject_every_comparison", SEC, _okt, "", "computed",
     definition="every comparison of the fine-tuned xSoyelNet with another adapted row rejects at p < 0.05 on both tests "
                "(Holm within kbs_adapted_pairwise10), on CC, SIM and KLD", inputs=["tests.kbs_adapted_pairwise10.*"])
for m in METRICS:
    md = {r: _P10[("ours_ftblocks_nested", r)][m]["mean_diff"] for r in _refs}
    rr = min(md, key=lambda r: abs(md[r]))
    fact(f"ft.claim.min_margin.{m}", SEC, md[rr], m, "computed",
         definition=f"smallest margin of the fine-tuned xSoyelNet over another adapted row on {m} (against {rr})",
         inputs=["tests.kbs_adapted_pairwise10.*"])
    fact(f"ft.claim.min_margin.{m}.against", SEC, rr, "", "computed", definition="row giving the smallest margin",
         inputs=["tests.kbs_adapted_pairwise10.*"])
fact("ft.claim.max_nb_p_holm", SEC, max(_P10[("ours_ftblocks_nested", r)][m]["nb_p_holm"] for r in _refs for m in METRICS), "p",
     "computed", definition="largest Holm-adjusted fold-level p over the 12 comparisons of the fine-tuned row (family "
                            "kbs_adapted_pairwise10)", inputs=["tests.kbs_adapted_pairwise10.*"])
fact("ft.claim.max_wilcoxon_p_holm", SEC, max(_P10[("ours_ftblocks_nested", r)][m]["wilcoxon_p_holm"] for r in _refs for m in METRICS),
     "p", "computed", definition="largest Holm-adjusted per-image p over the same 12 comparisons", inputs=["tests.kbs_adapted_pairwise10.*"])
fact("ft.claim.min_frac_images_better", SEC, min(_P10[("ours_ftblocks_nested", r)][m]["frac_images_first_better"] for r in _refs
                                                 for m in METRICS), "fraction", "computed",
     definition="smallest share of the 1000 images on which the fine-tuned xSoyelNet beats another adapted row (any metric)",
     inputs=["tests.kbs_adapted_pairwise10.*"])
# adapted gaps in Table 1 (native IIE) vs the training step of the three released models
_cc5 = {n: float(vec(R(n), "CC").mean()) for n in ("ours_ftblocks_nested", "msdb_adapt_s0", "sum_ft", "ours_scratch_s0",
                                                   "iie_native_adapt", "iie_ft_klcc_e20")}
_rel3 = {"MSDB-SUM": _cc5["msdb_adapt_s0"] - _cc5["sum_ft"], "MSDB-IIEnative": _cc5["msdb_adapt_s0"] - _cc5["iie_native_adapt"],
         "SUM-IIEnative": _cc5["sum_ft"] - _cc5["iie_native_adapt"]}
fact("ft.gap.released3_native_CC", SEC, _rel3, "CC", "computed",
     definition="CC gaps among the three released models after adaptation, IIE with native input", inputs=["per-image rows"])
fact("ft.gap.ft_minus_iie384_CC", SEC, _cc5["ours_ftblocks_nested"] - _cc5["iie_ft_klcc_e20"], "CC", "computed",
     definition="fine-tuned xSoyelNet minus adapted IIE at 384 input (supplement row)", inputs=["per-image rows"])
_st = [f for f in FACTS if f["id"] == "rev.gain.step_training_min_CC"]
if _st:
    fact("ft.gap.step_training_min_exceeds_released3_native_gaps", SEC, _st[0]["value"] > max(abs(x) for x in _rel3.values()),
         "", "computed", definition="the smallest training step of the three released models (IIE at 384) exceeds every "
         "CC gap among them after adaptation with IIE at native input", inputs=["rev.gain.step_training_min_CC", "ft.gap.released3_native_CC"])
_vr = [r for r in ("msdb_adapt_s0", "sum_ft", "iie_native_adapt", "iie_ft_klcc_e20")]
fact("ft.ranking.adapted_released_native", SEC, " > ".join(sorted(["MSDB", "SUM", "IIE"], key=lambda k: -{"MSDB": _cc5["msdb_adapt_s0"],
     "SUM": _cc5["sum_ft"], "IIE": _cc5["iie_native_adapt"]}[k])), "", "computed",
     definition="CC order of the three released models after adaptation (IIE native input)", inputs=["per-image rows"])
# ========================================================================================== 19 final-review checks (draft 6)
# Facts asked for by the final review of draft 5 (24 Nov 2026): the calibrated ranking on the original blur grid (0 to
# 12; only MSDB's row differs from the wide grid), SUM calibrated with a centre-prior mix against the other calibrated
# rows, fold-level margins of the fine-tuned row, its margins without the near-duplicate images, the largest zero-shot or
# calibrated CC on fold 0 (Fig. 1, Table S8), SUM's GFLOPs ratio with its scan estimate, the earlier xSoyelNet
# configurations scored on the same outer folds, and the verdict of the pre-registered family.
SEC = "review6"
for a_, b_ in (("sum_calib_wide", "sum_calibrated"), ("iie_calib_wide", "iie_zeroshot_foldprior_sigma")):
    _d = max(abs(R(a_)[i][m] - R(b_)[i][m]) for i in R(a_) for m in METRICS)
    fact(f"review6.grid12.{a_}_vs_{b_}.max_abs_diff", SEC, _d, "", "computed", recomputed=0.0, tol=1e-9,
         definition=f"largest per-image |difference| (CC, SIM, KLD) between {a_} (blur grid 0 to 32) and {b_} (grid 0 to "
                    "12): the wide grid changes only MSDB's calibrated row", inputs=[f"per-image rows of {a_} and {b_}"])
R6FAM = {
    "kbs_calibrated3_grid12": ([("sum_calib_wide", "msdb_zeroshot_foldprior"), ("msdb_zeroshot_foldprior", "iie_calib_wide"),
                                ("sum_calib_wide", "iie_calib_wide")],
                               "calibrated rows on the original blur grid (0 to 12): MSDB with prior and blur from the main "
                               "table (msdb_zeroshot_foldprior); SUM and IIE choose the same blur on both grids"),
    "kbs_calibrated_sumcb2": ([("msdb_calib_wide", "sum_calibrated_cb"), ("sum_calibrated_cb", "iie_calib_wide")],
                              "SUM calibrated with token, blur and a centre-prior mix (sum_calibrated_cb) against the "
                              "calibrated MSDB and IIE rows of Table 1 (wide grid)"),
}
for fname, (pairs, dsc) in R6FAM.items():
    res = {}
    for a_, b_ in pairs:
        res[(a_, b_)] = compare(R(a_), R(b_), B=10000, rng=np.random.default_rng(20261124))
    for m in METRICS:
        for key in ("wilcoxon_p", "nb_p"):
            adj = holm([res[p][m][key] for p in pairs])
            for p, x in zip(pairs, adj):
                res[p][m][key + "_holm"] = float(x)
    NEWRES[fname] = res
    fact(f"tests.family.{fname}", SEC, [f"{a_} minus {b_}" for a_, b_ in pairs], "comparisons", "computed",
         definition=f"Holm family '{fname}' (computed here): {dsc}; split random; Holm per metric and per test over these "
                    "comparisons; CI = 10,000-sample paired bootstrap, numpy default_rng(20261124) per comparison",
         inputs=[f"per-image rows of {x} (results/{FN(x)}_per_image.json)" for p in pairs for x in p])
    for (a_, b_) in pairs:
        for m in METRICS:
            mm = res[(a_, b_)][m]
            pre = f"tests.{fname}.{a_}_minus_{b_}.{m}"
            ins = [f"per-image rows of {a_} and {b_} (paired by id)"]
            fact(pre + ".mean_diff", SEC, mm["mean_diff"], m, "computed", inputs=ins,
                 definition=f"{a_} minus {b_}: mean per-image difference (for KLD negative = first better)",
                 display=f"{mm['mean_diff']:+.4f}")
            fact(pre + ".boot95", SEC, mm["boot95"], m, "computed", inputs=ins, definition="bootstrap 95% CI")
            fact(pre + ".folds_first_better", SEC, mm["folds_first_better"], "of 5 folds", "computed", inputs=ins,
                 definition=f"folds where {a_} is better")
            fact(pre + ".fold_diffs", SEC, mm["fold_diffs"], m, "computed", inputs=ins, definition="per-fold mean differences")
            fact(pre + ".frac_images_first_better", SEC, mm["frac_images_first_better"], "fraction", "computed",
                 inputs=ins, definition=f"share of the 1000 images where {a_} is better")
            for key in ("wilcoxon_p", "wilcoxon_p_holm", "nb_p", "nb_p_holm"):
                fact(pre + "." + key, SEC, mm[key], "p", "computed", inputs=ins,
                     definition=(key + f" (Holm over family {fname})") if "holm" in key else key, display=pfmt(mm[key]))
            fact(pre + ".dz", SEC, mm["dz"], "", "computed", inputs=ins, definition="paired Cohen's dz")
# fold-level margins of the fine-tuned row over the two strongest adapted rows
_fm = {r: _P10[("ours_ftblocks_nested", r)]["CC"]["fold_diffs"] for r in ("msdb_adapt_s0", "sum_ft")}
fact("ft.claim.min_fold_margin_CC.msdb_sum", SEC, min(min(x) for x in _fm.values()), "CC", "computed",
     definition="smallest per-fold CC margin of the fine-tuned xSoyelNet over the adapted MSDB head or the SUM full "
                "fine-tune (5 folds each)", inputs=["tests.kbs_adapted_pairwise10.ours_ftblocks_nested_minus_{msdb_adapt_s0,sum_ft}.CC.fold_diffs"])
fact("ft.claim.min_fold_margin_CC.msdb", SEC, min(_fm["msdb_adapt_s0"]), "CC", "computed",
     definition="smallest per-fold CC margin over the adapted MSDB head", inputs=["tests.kbs_adapted_pairwise10.*"])
fact("ft.claim.min_fold_margin_CC.sum", SEC, min(_fm["sum_ft"]), "CC", "computed",
     definition="smallest per-fold CC margin over the SUM full fine-tune", inputs=["tests.kbs_adapted_pairwise10.*"])
# near-duplicate images (cross-fold extended pairs, as for SUM above): margins of the fine-tuned row with and without them
_xs = set(xids)
_rest = [i for i in sorted(R("ours_ftblocks_nested"), key=int) if i not in _xs]
_chg = []
for r in ("msdb_adapt_s0", "sum_ft", "ours_scratch_s0", "iie_native_adapt"):
    _all = float((vec(R("ours_ftblocks_nested"), "CC") - vec(R(r), "CC")).mean())
    _wo = float((vec(R("ours_ftblocks_nested"), "CC", _rest) - vec(R(r), "CC", _rest)).mean())
    _on = float((vec(R("ours_ftblocks_nested"), "CC", xids) - vec(R(r), "CC", xids)).mean())
    _chg.append(abs(_wo - _all))
    fact(f"ft.neardup.minus_{r}.CC.without_pairs", SEC, _wo, "CC", "computed",
         definition=f"fine-tuned xSoyelNet minus {r}, mean CC difference without the {len(xids)} images of the cross-fold "
                    "extended near-duplicate pairs", inputs=["splits/near_dup_pairs.json extended", "per-image rows"])
    fact(f"ft.neardup.minus_{r}.CC.on_pairs", SEC, _on, "CC", "computed",
         definition=f"fine-tuned xSoyelNet minus {r} on those {len(xids)} images only", inputs=["per-image rows"])
fact("ft.neardup.max_abs_change_CC", SEC, max(_chg), "CC", "computed",
     definition="largest change of the fine-tuned row's mean CC margin (over MSDB head, SUM full fine-tune, frozen "
                "xSoyelNet, native-input IIE) when the near-duplicate images are dropped", inputs=["ft.neardup.*"])
fact("ft.neardup.n_images", SEC, len(xids), "images", "computed", definition="images dropped (cross-fold extended pairs)",
     inputs=["splits/near_dup_pairs.json extended"])
# largest zero-shot or calibrated CC on fold 0 (the dashed lines and n = 0 points of Fig. 1, Table S8)
_cal = {f["id"]: f["value"] for f in FACTS if re.match(r"^calib\.(msdb|iie|sum)\.n\d+\.CC$", f["id"])}
_kmax = max(_cal, key=_cal.get)
fact("review6.fold0.max_zeroshot_or_calibrated_CC", SEC, _cal[_kmax], "CC", "computed",
     definition=f"largest CC of any zero-shot or calibration-only point on fold 0 (calib.*.n*.CC, {len(_cal)} points; "
                f"attained by {_kmax})", inputs=sorted(_cal))
# SUM needs fewer GFLOPs than xSoyelNet: ratio with SUM's scan estimate
_g_ours = [f for f in FACTS if f["id"] == "cost.ours.GFLOPs"][0]["value"]
fact("cost.ratio.ours_over_sum_GFLOPs_incl_scan", SEC, _g_ours / _g_sum_incl, "x", "computed", definition="xSoyelNet GFLOPs over SUM GFLOPs including the scan estimate (221.2 / 31.0)",
     inputs=["cost.ours.GFLOPs", "cost.sum.GFLOPs_incl_scan"])
# earlier xSoyelNet configurations scored on the same seed-0 outer folds (supplement Tables S2 and S5)
_cfg = ["ours_scratch_s0", "ours_warm_s0", "ours_B644_s0", "ours_L448_s0", "ours_ms_B_s0", "ours_L_ms_s0",
        "distill_gt800_teacher800", "distill_pitt_only", "distill_pitt", "distill_pitt_L",
        "abl_no_aspp_s0", "abl_no_centerbias_s0", "abl_no_finalizer_s0", "abl_kl_only_s0"]
assert all(os.path.exists(os.path.join(RES, f"{c}_per_image.json")) for c in _cfg), "missing configuration file"
fact("review6.n_ours_configs_before_ft", SEC, len(_cfg), "configurations", "computed",
     definition="xSoyelNet configurations scored on the seed-0 outer folds before the fine-tuned row (seeds not counted): "
                "frozen readout from scratch and warm start, four backbone or resolution variants, four distilled students, "
                "four ablations: " + ", ".join(_cfg), inputs=[f"results/{c}_per_image.json" for c in _cfg])
# verdict of the pre-registered primary family (PLAN.json) on the comparisons the paper reports
_PR = NEWRES["kbs_ftblocks_prereg7"]
fact("ft.prereg7.same_verdict_msdb_sum_frozen", SEC,
     all(_PR[("ours_ftblocks_nested", r)][m][k] < 0.05 and _PR[("ours_ftblocks_nested", r)][m]["folds_first_better"] == 5
         for r in ("msdb_adapt_s0", "sum_ft", "ours_scratch_s0") for m in METRICS for k in ("wilcoxon_p_holm", "nb_p_holm")),
     "", "computed", definition="in the pre-registered 7-member family the fine-tuned xSoyelNet also beats the MSDB head, "
     "the SUM full fine-tune and the frozen readout on CC, SIM and KLD in 5 of 5 folds with both tests rejecting",
     inputs=["tests.kbs_ftblocks_prereg7.*"])


# ========================================================================================== write
for f in FACTS:
    for k, v in list(f.items()):
        if isinstance(v, (np.floating,)):
            f[k] = float(v)
        if isinstance(v, (np.integer,)):
            f[k] = int(v)
        if isinstance(v, np.bool_):
            f[k] = bool(v)

n_rec = sum(1 for c in CHECKS)
n_bad = sum(1 for c in CHECKS if not c["match"])
meta = {
    "title": "Facts pack: Calibrate to Choose, Fine-Tune to Deploy (ADD1000)",
    "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "script": "scripts/stats/build_facts.py",
    "command": "python -I scripts/stats/build_facts.py",
    "numpy": np.__version__, "scipy": __import__("scipy").__version__,
    "inputs_root": "repository root",
    "conventions": {
        "value_origin": "facts with source+json_path were READ from that file (the resolver asserts the path); "
                        "facts with source 'computed' were computed here from the listed inputs",
        "differences": "'X_minus_Y' = X minus Y per image; for KLD negative = X better; main-family facts are ours "
                       "minus row, frontier facts sum_ft minus reference",
        "sd": "every SD fact names its ddof in sd_ddof",
        "holm": "every Holm-adjusted p names its family in the definition (families are listed under tests.family.*)",
    },
    "n_facts": len(FACTS), "n_recompute_checks": n_rec, "n_recompute_mismatches": n_bad,
}
out = {"meta": meta, "warnings": WARN, "facts": FACTS, "checks": CHECKS}


def _repo_relative(o):   # paths inside the repository are written relative to its root
    if isinstance(o, dict):
        return {k: _repo_relative(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_repo_relative(v) for v in o]
    if isinstance(o, str):
        return o.replace(KB + os.sep, "")
    return o


out = _repo_relative(out)
with open(os.path.join(OUT, "facts.json"), "w") as f:
    json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(f"{len(FACTS)} facts, {n_rec} recompute checks, {n_bad} mismatches, {len(WARN)} warnings")
for w in WARN:
    if w.startswith("MISMATCH"):
        print(w)
