"""
Revision rows (24 Nov 2026). CPU only, no training.
Run once per model (--part msdb | iie | sum); the parts are independent and can run in parallel.

(1) Zero-shot with a MIT1003 centre-bias template (DeepGaze's documented usage; no ADD1000 data):
      msdb_zeroshot_mit1003, iie_zeroshot_mit1003
    Template = DeepGaze's distributed centerbias_mit1003.npy (see below), rescaled to the model's input
    grid with nearest interpolation and logsumexp-renormalised, as in the DeepGaze README. Inputs as in the uniform-prior rows (MSDB 512, IIE 384).
(2) Calibrated rows on ONE wider blur grid for all three models (WIDE = 0, 2, ..., 32), otherwise identical to
    msdb_zeroshot_foldprior, iie_zeroshot_foldprior_sigma and sum_calibrated (prior or token from the fold's
    800 training maps; blur chosen by mean CC on the same 200 training maps, numpy default_rng(fold)):
      msdb_calib_wide, iie_calib_wide, sum_calib_wide
    Validation: the same code restricted to the protocol grid (0..12) reproduces the existing rows per image
    (results/calib_rows_validation_<part>.json).
(3) Calibration-only learning curve on fold 0 (results/calib_curve_<part>.json, calib_curve_<part>_per_image.json):
    the curve subsets of splits/curve_subsets_seed0.json (n = 25..400 x 3 resamples, n = 800 once); prior (MSDB,
    IIE) or token (SUM) from the n maps; blur from WIDE chosen by mean CC on the first min(n, 200) of the n maps
    (subset order); n = 800 is the whole pool (folds 1-4 in fold order), one run; scored on the 200 fold-0 test images. n = 0: the zero-shot rows above (MIT1003 template for
    MSDB and IIE; SUM token 2).

  python -I calib_rows.py --part msdb|iie|sum [--threads 2]

Template: DeepGaze's distributed MIT1003 centre-bias
template, cache/revision/deepgaze_template/centerbias_mit1003.npy (1024 x 1024 float64 log density, sha256
051645b7...e9, loaded with allow_pickle=False), same rescale/renormalise procedure; every output of the msdb and iie
parts then carries the suffix _offtpl (msdb_zeroshot_mit1003_offtpl, msdb_calib_wide_offtpl, calib_curve_msdb_offtpl,
calib_rows_validation_msdb_offtpl, ...). The calibrated rows and the curve points n > 0 do not use the template; only
the zero-shot rows and the curve's n = 0 point do. The sum part never uses the template (no suffix).
"""
import argparse, hashlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True, choices=["msdb", "iie", "sum"])
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--tpl", default="official", choices=["official"]); a = ap.parse_args()
import numpy as np, torch
torch.set_num_threads(a.threads)
from scipy.ndimage import gaussian_filter, zoom
from scipy.special import logsumexp

KBS = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
RES = os.path.join(KBS, "results")
TPL_OFFICIAL = os.path.join(KBS, "cache", "revision", "deepgaze_template", "centerbias_mit1003.npy")
TPL_OFFICIAL_SHA256 = "051645b7da47884c138e81f28a4afbadd99b6c3262eaf086434a9626e3578be9"
assert hashlib.sha256(open(TPL_OFFICIAL, "rb").read()).hexdigest() == TPL_OFFICIAL_SHA256
TPL_PATH, TPL_DESC = TPL_OFFICIAL, "official DeepGaze centerbias_mit1003.npy (v1.0.0 release)"
TPL = np.load(TPL_PATH, allow_pickle=False).astype(np.float64)
SUF = "_offtpl" if (a.tpl == "official" and a.part in ("msdb", "iie")) else ""
SPLIT = json.load(open(os.path.join(KBS, "splits", "kfold_seed0.json")))
IDS = [str(i) for i in SPLIT["ids_sorted"]]
FOLDS = [[str(i) for i in f] for f in SPLIT["folds"]]
FOLD_OF = {i: k for k, f in enumerate(FOLDS) for i in f}
POS = {i: k for k, i in enumerate(IDS)}
CS = json.load(open(os.path.join(KBS, "splits", "curve_subsets_seed0.json")))
WIDE = list(range(0, 33, 2))
PROTO = [0, 2, 4, 6, 8, 10, 12]
SELMAX = 200
T0 = time.time()
CMD = f"python -I scripts/revision/calib_rows.py --part {a.part} --threads {a.threads} --tpl {a.tpl}"
TEST0 = FOLDS[0]
SUBSETS = [(int(k.split(":")[0]), int(k.split(":")[1]), [str(x) for x in v]) for k, v in CS["train_ids"].items()]
SUBSETS = [s for s in SUBSETS if not (s[0] == 800 and s[1] > 0)]
if not any(s[0] == 800 for s in SUBSETS):   # n = 800 = the whole pool (folds 1-4 in fold order): one deterministic run
    SUBSETS.append((800, 0, [i for j, f in enumerate(FOLDS) if j != 0 for i in f]))


def tpl_at(size):
    cb = zoom(TPL, (size / TPL.shape[0], size / TPL.shape[1]), order=0, mode="nearest")
    return cb - logsumexp(cb)


def train_of(k):
    return [i for j, f in enumerate(FOLDS) if j != k for i in f]


def blur(p, s):
    return gaussian_filter(p, s) if s > 0 else p


def sigma_table(preds, gts, ccf):
    return {s: float(np.mean([ccf(blur(preds[i], s), gts[i]) for i in preds])) for s in WIDE}


def argmax_first(tab, grid):
    best, bcc = 0, -2
    for s in grid:                       # strict '>' = smallest sigma on ties (the rule of every existing row)
        if tab[s] > bcc:
            best, bcc = s, tab[s]
    return best


def summarize(row, per, notes, extra=None):
    fm = {m: [float(np.mean([r[m] for r in per if r["fold"] == k])) for k in range(5)] for m in ("CC", "SIM", "KLD")}
    out = {"row": row, "n_images": len(per), "per_fold": [{m: fm[m][k] for m in fm} | {"fold": k} for k in range(5)]}
    for m in fm:
        out[f"{m}_mean"] = float(np.mean(fm[m])); out[f"{m}_sd_folds_ddof1"] = float(np.std(fm[m], ddof=1))
        out[f"{m}_mean_over_images"] = float(np.mean([r[m] for r in per]))
    out.update({"protocol": notes, "command": CMD, "wall_s_script_so_far": time.time() - T0,
                "scoring": "CC/SIM/KLD on 384x384 vs GT PIL-bilinear to 384, clip at 0 and divide by sum",
                "template_file": TPL_PATH, "template": TPL_DESC, "file_suffix": SUF})
    if extra:
        out.update(extra)
    per = sorted(per, key=lambda r: POS[r["id"]])
    json.dump(per, open(os.path.join(RES, f"{row}{SUF}_per_image.json"), "w"), indent=0)
    json.dump(out, open(os.path.join(RES, f"{row}{SUF}_summary.json"), "w"), indent=1)
    print(f"{row}: CC {out['CC_mean']:.4f} (sd {out['CC_sd_folds_ddof1']:.4f}) SIM {out['SIM_mean']:.4f} "
          f"KLD {out['KLD_mean']:.4f}  [{time.time() - T0:.0f}s]", flush=True)
    return out


def check_same(new_rows, ref_file):
    ref = {str(r["id"]): r for r in json.load(open(os.path.join(RES, ref_file)))}
    d = max(abs(r[m] - ref[r["id"]][m]) for r in new_rows for m in ("CC", "SIM", "KLD"))
    print(f"  validation vs {ref_file}: max |diff| {d:.2e}", flush=True)
    return d


def kfold_calibrated(predict, prior_from, gts, sel_gts, ccf, metr, ref_file, row, notes, token_of=None):
    per_w, per_p, info = [], [], []
    for k in range(5):
        tr = train_of(k)
        prior = prior_from(tr) if prior_from else None
        tok = token_of(tr) if token_of else None
        sel = [str(x) for x in np.random.default_rng(k).choice(np.array(tr), size=SELMAX, replace=False)]
        ps = predict(sel, prior, tok)
        tab = sigma_table(ps, sel_gts, ccf)
        sw, sp = argmax_first(tab, WIDE), argmax_first(tab, PROTO)
        pt = predict(FOLDS[k], prior, tok)
        for i in FOLDS[k]:
            extra = {"token": tok} if tok is not None else {}
            per_w.append({"id": i, "fold": k, "sigma": sw, **extra, **metr(blur(pt[i], sw), gts[i])})
            per_p.append({"id": i, "fold": k, "sigma": sp, **extra, **metr(blur(pt[i], sp), gts[i])})
        info.append({"fold": k, "sigma_wide": sw, "sigma_protocol_grid": sp, "token": tok, "train_cc_by_sigma": tab})
        print(f"  fold {k}: sigma wide {sw}, protocol grid {sp}" + (f", token {tok}" if tok is not None else "")
              + f"  [{time.time() - T0:.0f}s]", flush=True)
    val = check_same(per_p, ref_file)
    summarize(row, per_w, notes, {"calibration_maps": 800, "training_maps": 0, "grid": WIDE, "selection": info,
                                  "validation_protocol_grid_vs": ref_file, "validation_max_abs_diff": val})
    return val


def calib_curve(part, predict, prior_from, gts, ccf, metr, zs_preds, token_of=None):
    pi = []
    res = {"0": {m: float(np.mean([metr(zs_preds[i], gts[i])[m] for i in TEST0])) for m in ("CC", "SIM", "KLD")}}
    runs = {}
    for n, r, tr in SUBSETS:
        prior = prior_from(tr) if prior_from else None
        tok = token_of(tr) if token_of else None
        selids = tr[:SELMAX]
        ps = predict(selids, prior, tok)
        s = argmax_first(sigma_table(ps, gts, ccf), WIDE)
        pt = predict(TEST0, prior, tok)
        vals = []
        for i in TEST0:
            mm = metr(blur(pt[i], s), gts[i]); vals.append(mm)
            pi.append({"model": part, "n": n, "resample": r, "id": i, **mm, "sigma": s, "token": tok})
        runs.setdefault(n, []).append({"resample": r, "sigma": s, "token": tok,
                                       **{m: float(np.mean([v[m] for v in vals])) for m in ("CC", "SIM", "KLD")}})
        print(f"  curve n{n} r{r}: CC {runs[n][-1]['CC']:.4f} sigma {s}  [{time.time() - T0:.0f}s]", flush=True)
    for n, rr in runs.items():
        res[str(n)] = {m: float(np.mean([x[m] for x in rr])) for m in ("CC", "SIM", "KLD")}
        res[str(n)]["CC_sd_resamples_ddof1"] = float(np.std([x["CC"] for x in rr], ddof=1)) if len(rr) > 1 else None
        res[str(n)]["runs"] = rr
    out = {"model": part, "rule": "fold-0 test (200 images); subsets = splits/curve_subsets_seed0.json train_ids (n = 800 "
           "once: the pool); prior (MSDB, IIE) or token (SUM) from the n maps; blur from the grid 0..32 (step 2) chosen "
           "by mean CC on the first min(n, 200) of the n maps; no weights trained; n = 0: zero-shot (MIT1003 template "
           "for MSDB and IIE, token 2 for SUM)", "grid": WIDE, "by_n": res, "command": CMD, "wall_s": time.time() - T0,
           "template_file": TPL_PATH, "template": TPL_DESC, "file_suffix": SUF}
    json.dump(out, open(os.path.join(RES, f"calib_curve_{part}{SUF}.json"), "w"), indent=1)
    json.dump(pi, open(os.path.join(RES, f"calib_curve_{part}{SUF}_per_image.json"), "w"))
    print(f"curve {part}: " + ", ".join(f"n{n} {res[n]['CC']:.4f}" for n in res), flush=True)


# =============================================================================== MSDB
if a.part == "msdb":
    sys.path.insert(0, os.path.join(HERE, "..", "msdb"))
    import msdb_common as M
    R = torch.from_numpy(np.load(os.path.join(M.CACHE, "readout_zs.npy")))[:, None]
    net, sigma0, cbw0, scale_w, sd = M.build_sal_net_pretrained()
    head = M.MSDBHead(net, sigma0, cbw0).eval()
    G = {i: M.load_gt(i, M.EVAL) for i in IDS}

    @torch.no_grad()
    def predict(id_list, cbd, tok=None):
        return {i: M.logd256_to_prob(head.finalize(R[POS[i]:POS[i] + 1], cbd[None]), M.EVAL)[0] for i in id_list}

    def prior_from(tr):
        return M.cb_downscale(M.fit_prior(tr))

    P_mit = predict(IDS, M.cb_downscale(tpl_at(M.S)))
    summarize("msdb_zeroshot_mit1003", [{"id": i, "fold": FOLD_OF[i], **M.metrics(P_mit[i], G[i])} for i in IDS],
              "DeepGazeMSDB released weights, generalisation mode (dataset=None), pixel_per_dva 21.7, 512x512 input; MIT1003 "
              f"centre-bias template ({TPL_DESC}) zoomed (order 0) to 512, logsumexp-renormalised, "
              "downscaled by the model's own nearest rule; no post-hoc blur; no ADD1000 data",
              {"calibration_maps": 0, "training_maps": 0})
    val = kfold_calibrated(predict, prior_from, G, G, M.cc, M.metrics, "msdb_zeroshot_foldprior_per_image.json", "msdb_calib_wide",
                           "as msdb_zeroshot_foldprior (released weights; centre prior = log mean density of the fold's 800 "
                           "training maps at 512), post-hoc blur from the wider grid 0..32 (step 2) chosen by mean CC on 200 "
                           "training maps (numpy default_rng(fold)); no weights trained")
    calib_curve("msdb", predict, prior_from, G, M.cc, M.metrics, P_mit)

# =============================================================================== IIE
if a.part == "iie":
    sys.path.insert(0, os.path.join(HERE, "..", "iie"))
    import iie_common as C
    import deepgaze_pytorch
    iie = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
    C.readout_to(iie, "cpu")
    raw = C.gt_cache()["raw384"]
    feats = C.load_feature_cache("float32")
    G = {i: raw[POS[i]].astype(np.float64) for i in IDS}

    def minsub(i):
        m = G[i] - G[i].min(); s = m.sum()
        return m / s if s > 0 else m + 1.0 / m.size

    GM = {i: minsub(i) for i in IDS}

    @torch.no_grad()
    def predict(id_list, cb, tok=None):
        out = {}
        cbt = torch.from_numpy(cb.astype(np.float32))
        uniq = list(dict.fromkeys(id_list))
        for k in range(0, len(uniq), 8):
            bt = uniq[k:k + 8]
            fx = torch.from_numpy(np.ascontiguousarray(feats[[POS[i] for i in bt]]))
            logd = C.forward_cached(iie, C.split_channels(fx), cbt[None].expand(len(bt), -1, -1))
            d = torch.exp(logd[:, 0].double()).numpy()
            for j, i in enumerate(bt):
                out[i] = d[j]
        return out

    def prior_from(tr):
        acc = np.sum([GM[i] for i in tr], axis=0)
        return np.log(C.to_prob(acc) + C.EPS)

    P_mit = predict(IDS, tpl_at(C.S))
    summarize("iie_zeroshot_mit1003", [{"id": i, "fold": FOLD_OF[i], **C.metrics(P_mit[i], G[i])} for i in IDS],
              "zero-shot DeepGaze IIE (released weights), input PIL bilinear 384x384 (backbones see 192x192), MIT1003 centre-bias "
              f"template ({TPL_DESC}) zoomed (order 0) to 384 and logsumexp-renormalised; cached features (forward_cached, "
              "exact); no post-hoc blur; no ADD1000 data", {"calibration_maps": 0, "training_maps": 0})
    val = kfold_calibrated(predict, prior_from, G, GM, C.cc, C.metrics, "iie_zeroshot_foldprior_sigma_per_image.json",
                           "iie_calib_wide", "as iie_zeroshot_foldprior_sigma (released weights, 384 input; prior = log(to_prob(sum "
                           "of min-subtracted 384 training maps)) + 1e-8 from the fold's 800 training maps), post-hoc blur from the "
                           "wider grid 0..32 (step 2) chosen by mean CC on 200 training maps (numpy default_rng(fold)); no weights trained")
    calib_curve("iie", predict, prior_from, G, C.cc, C.metrics, P_mit)

# =============================================================================== SUM
if a.part == "sum":
    sys.path.insert(0, os.path.join(HERE, "..", "sum"))
    import sum_common as sc
    maps = [np.load(os.path.join(KBS, "cache", "sum_maps", f"sum_zeroshot_t{t}_256.npy"), mmap_mode="r") for t in range(4)]
    assert bool(np.load(os.path.join(KBS, "cache", "sum_maps", "sum_zeroshot_done.npy")).all())
    ADD = os.path.join(os.environ.get("ADD1000_DIR", os.path.join(KBS, "data", "ADD1000")), "flat")
    G = {i: sc.load_gt(os.path.join(ADD, "maps", f"{i}.png"), 384) for i in IDS}
    ZCC = {t: {i: sc.cc(sc.resize_map(maps[t][POS[i]], 384), G[i]) for i in IDS} for t in range(4)}

    def token_of(tr):
        tok = {t: float(np.mean([ZCC[t][i] for i in tr])) for t in range(4)}
        return max(tok, key=tok.get)

    need = sorted({token_of(train_of(k)) for k in range(5)} | {token_of(s[2]) for s in SUBSETS} | {2})
    print("  SUM tokens needed:", need, flush=True)
    P = {t: {i: sc.resize_map(maps[t][POS[i]], 384) for i in IDS} for t in need}
    BT = {}   # cache of blurred-CC per (token, id, sigma): SUM predictions do not depend on the subset

    def predict(id_list, prior=None, tok=2):
        return {i: P[tok][i] for i in id_list}

    def sigma_table_cached(ids_, tok):
        for i in ids_:
            if (tok, i) not in BT:
                BT[(tok, i)] = {s: sc.cc(blur(P[tok][i], s), G[i]) for s in WIDE}
        return {s: float(np.mean([BT[(tok, i)][s] for i in ids_])) for s in WIDE}

    # k-fold (token on 800 maps; sigma on 200 maps; SUM's existing tie rule = smallest sigma among maxima)
    per_w, per_p, info = [], [], []
    for k in range(5):
        tr = train_of(k); tb = token_of(tr)
        sel = [str(x) for x in np.random.default_rng(k).choice(tr, size=sc.SIGMA_SELECT_N, replace=False)]
        tab = sigma_table_cached(sel, tb)
        sw = max(WIDE, key=lambda x: (tab[x], -x)); sp = max(PROTO, key=lambda x: (tab[x], -x))
        for i in FOLDS[k]:
            per_w.append({"id": i, "fold": k, "sigma": sw, "token": tb, **sc.metrics(blur(P[tb][i], sw), G[i])})
            per_p.append({"id": i, "fold": k, "sigma": sp, "token": tb, **sc.metrics(blur(P[tb][i], sp), G[i])})
        info.append({"fold": k, "token": tb, "sigma_wide": sw, "sigma_protocol_grid": sp, "train_cc_by_sigma": tab})
        print(f"  fold {k}: token {tb}, sigma wide {sw}, protocol grid {sp}  [{time.time() - T0:.0f}s]", flush=True)
    val = check_same(per_p, "sum_calibrated_per_image.json")
    summarize("sum_calib_wide", per_w, "as sum_calibrated (released weights, 256 input; token with the highest mean CC on the "
              "fold's 800 training maps), post-hoc blur from the wider grid 0..32 (step 2) chosen by mean CC on 200 training "
              "maps (numpy default_rng(fold)); no prior; no weights trained",
              {"calibration_maps": 800, "training_maps": 0, "grid": WIDE, "selection": info,
               "validation_protocol_grid_vs": "sum_calibrated_per_image.json", "validation_max_abs_diff": val})
    # curve
    pi, runs = [], {}
    res = {"0": {m: float(np.mean([sc.metrics(P[2][i], G[i])[m] for i in TEST0])) for m in ("CC", "SIM", "KLD")}}
    for n, r, tr in SUBSETS:
        tb = token_of(tr)
        tab = sigma_table_cached(tr[:SELMAX], tb)
        s = argmax_first(tab, WIDE)
        vals = []
        for i in TEST0:
            mm = sc.metrics(blur(P[tb][i], s), G[i]); vals.append(mm)
            pi.append({"model": "sum", "n": n, "resample": r, "id": i, **mm, "sigma": s, "token": tb})
        runs.setdefault(n, []).append({"resample": r, "sigma": s, "token": tb,
                                       **{m: float(np.mean([v[m] for v in vals])) for m in ("CC", "SIM", "KLD")}})
        print(f"  curve n{n} r{r}: CC {runs[n][-1]['CC']:.4f} token {tb} sigma {s}  [{time.time() - T0:.0f}s]", flush=True)
    for n, rr in runs.items():
        res[str(n)] = {m: float(np.mean([x[m] for x in rr])) for m in ("CC", "SIM", "KLD")}
        res[str(n)]["CC_sd_resamples_ddof1"] = float(np.std([x["CC"] for x in rr], ddof=1)) if len(rr) > 1 else None
        res[str(n)]["runs"] = rr
    out = {"model": "sum", "rule": "fold-0 test (200 images); subsets = splits/curve_subsets_seed0.json train_ids (n = 800 once: "
           "the pool); token with the highest mean CC on the n maps; blur from the grid 0..32 (step 2) chosen by mean CC on the "
           "first min(n, 200) of the n maps; no weights trained; n = 0: token 2 zero-shot", "grid": WIDE, "by_n": res,
           "command": CMD, "wall_s": time.time() - T0}
    json.dump(out, open(os.path.join(RES, "calib_curve_sum.json"), "w"), indent=1)
    json.dump(pi, open(os.path.join(RES, "calib_curve_sum_per_image.json"), "w"))
    print("curve sum: " + ", ".join(f"n{n} {res[n]['CC']:.4f}" for n in res), flush=True)

json.dump({"part": a.part, "validation_protocol_grid_max_abs_diff": val, "command": CMD, "wall_s": time.time() - T0,
           "template_file": TPL_PATH, "template": TPL_DESC},
          open(os.path.join(RES, f"calib_rows_validation_{a.part}{SUF}.json"), "w"), indent=1)
print("done", a.part, f"{time.time() - T0:.0f}s", flush=True)
