"""Rows (2) and (3): score cached SUM zero-shot maps.

sum_zeroshot_t{0..3}: raw SUM maps (sigmoid output, 256x256) resized bilinearly to 384
  (PIL mode F, as in eval_sum_add1000.py) and scored against the GT map resized bilinearly
  to 384; native-resolution scores (GT resized to 256) are logged alongside.
sum_calibrated: per fold, the token with the highest mean CC on that fold's 800 TRAINING
  images is chosen, then the Gaussian finalizer sigma is chosen from {0,2,...,12} by mean
  CC on 200 training images (numpy default_rng(fold), as for every adapted row), applied
  to the 384 map. No weights change.
sum_calibrated_cb: as sum_calibrated, plus a centre-prior mixture
  p = (1 - w) * prob(blur_sigma(pred)) + w * prior, prior = mean probability map of the
  fold's 800 training GT maps, (sigma, w) chosen jointly by mean CC on the same 200
  training images, w in {0, .05, .1, .15, .2, .25, .3, .4, .5}.
Diagnostics in the calibrated summaries (not rows): the same selection run with KLD as
the criterion, and a normalisation-only control (min-subtract before sum-normalising).

python -I score_zeroshot_calibrated.py --scripts DIR --data DIR --split FILE \
    --maps cache/sum_maps --results results [--infer_meta cache/sum_maps/sum_zeroshot_infer_meta.json]
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
for a in ("scripts", "data", "split", "maps", "results"):
    ap.add_argument(f"--{a}", required=True)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--allow_partial", action="store_true", help="debug only: score the images inferred so far")
args = ap.parse_args()
sys.path.insert(0, args.scripts)

import numpy as np
from scipy.ndimage import gaussian_filter
import sum_common as sc

t0 = time.time()
CMD = "python -I " + " ".join(sys.argv)
split = json.load(open(args.split))
ids = split["ids_sorted"]; folds = sc.load_split(args.split)
fold_of = {i: k for k, f in enumerate(folds) for i in f}
maps = [np.load(os.path.join(args.maps, f"sum_zeroshot_t{t}_256.npy"), mmap_mode="r") for t in range(4)]
done_mask = np.load(os.path.join(args.maps, "sum_zeroshot_done.npy"))
if args.allow_partial:
    keep = {i for i, d in zip(ids, done_mask) if d}
    folds = [[i for i in f if i in keep] for f in folds]
    ids_all = ids; ids = [i for i in ids if i in keep]
    meta = {"params": 0, "seconds_this_run": 0, "s_per_image_4_tokens": 0, "device": "partial"}
else:
    assert bool(done_mask.all()), "inference incomplete"
    ids_all = ids
    meta = json.load(open(os.path.join(args.maps, "sum_zeroshot_infer_meta.json")))
idx = {i: n for n, i in enumerate(ids_all)}

G384 = {i: sc.load_gt(os.path.join(args.data, "maps", f"{i}.png"), 384) for i in ids}
G256 = {i: sc.load_gt(os.path.join(args.data, "maps", f"{i}.png"), 256) for i in ids}
P384 = {t: {} for t in range(4)}
for t in range(4):
    for i in ids:
        P384[t][i] = sc.resize_map(maps[t][idx[i]], 384)
print(f"loaded maps + GT in {time.time() - t0:.1f}s", flush=True)

PROTO = ("ADD1000 seed-0 5-fold split (splits/kfold_seed0.json); CC/SIM/KLD on 384x384; "
         "GT = PNG map resized bilinearly (PIL) to 384; prediction resized bilinearly (PIL mode F) "
         "from SUM's native 256x256 output; probability maps = clip at 0 and divide by sum; EPS 1e-8")
published = {0: (0.8161051056732648, 0.7151296767296452, 0.4112185396594381),
             1: (0.7937065455405521, 0.6807948822632357, 0.4019097675344661),
             2: (0.8363667303665698, 0.6928166860779225, 0.5144052507932709),
             3: (0.7110342309253882, 0.6579669582333466, 0.43464271507806784)}


def save(row, per_image, summary):
    json.dump(per_image, open(os.path.join(args.results, f"{row}_per_image.json"), "w"))
    json.dump(summary, open(os.path.join(args.results, f"{row}_summary.json"), "w"), indent=1)


# ------------------------------------------------------------------ row 2: zero-shot tokens
zs, zsd = {}, {}
for t in range(4):
    rows = []
    for i in ids:
        m = sc.metrics(P384[t][i], G384[i]); mn = sc.metrics(np.asarray(maps[t][idx[i]], np.float64), G256[i])
        rows.append({"id": i, "fold": fold_of[i], "CC": m["CC"], "SIM": m["SIM"], "KLD": m["KLD"], "token": t,
                     "CC_native256": mn["CC"], "SIM_native256": mn["SIM"], "KLD_native256": mn["KLD"]})
    s = sc.fold_summary(rows, folds)
    sn = sc.fold_summary([{**r, "CC": r["CC_native256"], "SIM": r["SIM_native256"], "KLD": r["KLD_native256"]}
                          for r in rows], folds)
    pub = published[t]
    s.update({"row": f"sum_zeroshot_t{t}", "token": t,
              "token_meaning": ["natural scenes (SALICON mouse)", "natural scenes (eye tracking)",
                                "e-commerce", "user interface"][t],
              "native256": sn,
              "published_SPL_overall_mean": {"CC": pub[0], "SIM": pub[1], "KLD": pub[2]},
              "reproduction_abs_diff_overall": {"CC": abs(s["CC_mean_over_images"] - pub[0]),
                                                "SIM": abs(s["SIM_mean_over_images"] - pub[1]),
                                                "KLD": abs(s["KLD_mean_over_images"] - pub[2])},
              "calibration_maps_used": 0, "training_maps_used": 0,
              "protocol": PROTO + "; zero-shot: released weights, no ADD1000 data used",
              "params_total": meta["params"],
              "inference_wallclock_s_all_tokens_1000_imgs": meta["seconds_this_run"],
              "inference_s_per_image_4_tokens": meta["s_per_image_4_tokens"],
              "inference_device": meta["device"],
              "scan_impl": "pure-PyTorch chunked selective scan (verified vs CUDA-kernel stored outputs, |dCC|<3e-7)",
              "command_inference": "python -I scripts/sum/infer_zeroshot.py --scripts scripts/sum "
                                   "--code third_party/SUM/code --weights third_party/SUM/weights "
                                   "--data ADD1000/flat --split splits/kfold_seed0.json --out_dir cache/sum_maps "
                                   "--device cpu --threads 6",
              "command_scoring": CMD})
    save(f"sum_zeroshot_t{t}", rows, s)
    zs[t] = rows
    zsd[t] = {r["id"]: r for r in rows}
    print(f"t{t}: CC {s['CC_mean']:.4f}+/-{s['CC_sd_folds']:.4f} SIM {s['SIM_mean']:.4f} KLD {s['KLD_mean']:.4f} "
          f"(overall {s['CC_mean_over_images']:.4f}/{s['SIM_mean_over_images']:.4f}/{s['KLD_mean_over_images']:.4f}; "
          f"published {pub[0]:.4f}/{pub[1]:.4f}/{pub[2]:.4f}) native256 CC {sn['CC_mean']:.4f}", flush=True)

# ------------------------------------------------------------------ row 3: calibrated
WGRID = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
SIG_EXT = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32]
cal_rows = {"sum_calibrated": [], "sum_calibrated_cb": []}
diag = {"kldsel_sigma": [], "kldsel_cb": [], "minsub": [], "extgrid_sigma": [], "extgrid_cb": []}
fold_info = []
for k in range(len(folds)):
    te = folds[k]; tr = sc.train_ids_for(folds, k)
    tok_cc = {t: float(np.mean([zsd[t][i]["CC"] for i in tr])) for t in range(4)}
    tbest = max(tok_cc, key=tok_cc.get)
    sel = [str(s) for s in np.random.default_rng(k).choice(tr, size=sc.SIGMA_SELECT_N if not args.allow_partial else 40, replace=False)]
    prior = np.mean([sc.to_prob(G384[i]) for i in tr], 0)
    P = P384[tbest]
    blurred = {s: {i: (gaussian_filter(P[i], s) if s > 0 else P[i]) for i in sel} for s in sc.SIGMAS}
    # sigma only (CC criterion, the rule used for every adapted row)
    tab_s = {s: float(np.mean([sc.cc(blurred[s][i], G384[i]) for i in sel])) for s in sc.SIGMAS}
    sig = max(sc.SIGMAS, key=lambda s: (tab_s[s], -s))
    # sigma + prior mixture, joint (CC criterion)
    tab_cb = {}
    tab_cb_kld = {}
    for s in sc.SIGMAS:
        pb = {i: sc.to_prob(blurred[s][i]) for i in sel}
        for w in WGRID:
            mix = {i: (1 - w) * pb[i] + w * prior for i in sel}
            tab_cb[(s, w)] = float(np.mean([sc.cc(mix[i], G384[i]) for i in sel]))
            tab_cb_kld[(s, w)] = float(np.mean([sc.kld(mix[i], G384[i]) for i in sel]))
    s_cb, w_cb = max(tab_cb, key=lambda kw: (tab_cb[kw], -kw[1], -kw[0]))
    s_kld, w_kld = min(tab_cb_kld, key=lambda kw: (tab_cb_kld[kw], kw[1], kw[0]))
    tab_s_kld = {s: tab_cb_kld[(s, 0)] for s in sc.SIGMAS}
    sig_kld = min(sc.SIGMAS, key=lambda s: (tab_s_kld[s], s))
    # diagnostic: extended sigma grid (is the protocol grid's upper end binding?)
    tab_ext, tab_ext_cb = {}, {}
    for s in SIG_EXT:
        bl = blurred[s] if s in blurred else {i: gaussian_filter(P[i], s) for i in sel}
        tab_ext[s] = float(np.mean([sc.cc(bl[i], G384[i]) for i in sel]))
        pb = {i: sc.to_prob(bl[i]) for i in sel}
        for w in WGRID:
            tab_ext_cb[(s, w)] = float(np.mean([sc.cc((1 - w) * pb[i] + w * prior, G384[i]) for i in sel]))
    sig_ext = max(SIG_EXT, key=lambda s: (tab_ext[s], -s))
    s_ext_cb, w_ext_cb = max(tab_ext_cb, key=lambda kw: (tab_ext_cb[kw], -kw[1], -kw[0]))
    fold_info.append({"fold": k, "train_token_CC": tok_cc, "token": tbest, "sigma": sig,
                      "sigma_table_trainCC": {str(s): v for s, v in tab_s.items()},
                      "cb_sigma": s_cb, "cb_w": w_cb, "cb_trainCC": tab_cb[(s_cb, w_cb)],
                      "kldsel_sigma_only": sig_kld, "kldsel_cb_sigma": s_kld, "kldsel_cb_w": w_kld,
                      "extgrid_sigma_only": sig_ext, "extgrid_sigma_table_trainCC": {str(s): v for s, v in tab_ext.items()},
                      "extgrid_cb_sigma": s_ext_cb, "extgrid_cb_w": w_ext_cb,
                      "n_prior_maps": len(tr), "n_select_maps": len(sel)})
    prior256 = np.mean([sc.to_prob(G256[i]) for i in tr], 0)
    for i in te:
        raw = P[i]; raw256 = np.asarray(maps[tbest][idx[i]], np.float64)
        p1 = gaussian_filter(raw, sig) if sig > 0 else raw
        m1 = sc.metrics(p1, G384[i])
        n1 = sc.metrics(gaussian_filter(raw256, sig * 256 / 384) if sig > 0 else raw256, G256[i])
        cal_rows["sum_calibrated"].append({"id": i, "fold": k, **m1, "token": tbest, "sigma": sig,
                                           "CC_native256": n1["CC"], "SIM_native256": n1["SIM"],
                                           "KLD_native256": n1["KLD"]})
        p2 = (1 - w_cb) * sc.to_prob(gaussian_filter(raw, s_cb) if s_cb > 0 else raw) + w_cb * prior
        m2 = sc.metrics(p2, G384[i])
        q2 = (1 - w_cb) * sc.to_prob(gaussian_filter(raw256, s_cb * 256 / 384) if s_cb > 0 else raw256) + w_cb * prior256
        n2 = sc.metrics(q2, G256[i])
        cal_rows["sum_calibrated_cb"].append({"id": i, "fold": k, **m2, "token": tbest, "sigma": s_cb, "w": w_cb,
                                              "CC_native256": n2["CC"], "SIM_native256": n2["SIM"],
                                              "KLD_native256": n2["KLD"]})
        p3 = gaussian_filter(raw, sig_kld) if sig_kld > 0 else raw
        diag["kldsel_sigma"].append({"fold": k, **sc.metrics(p3, G384[i])})
        p4 = (1 - w_kld) * sc.to_prob(gaussian_filter(raw, s_kld) if s_kld > 0 else raw) + w_kld * prior
        diag["kldsel_cb"].append({"fold": k, **sc.metrics(p4, G384[i])})
        diag["minsub"].append({"fold": k, **sc.metrics(raw - raw.min(), G384[i])})
        diag["extgrid_sigma"].append({"fold": k, **sc.metrics(gaussian_filter(raw, sig_ext) if sig_ext > 0 else raw, G384[i])})
        p6 = (1 - w_ext_cb) * sc.to_prob(gaussian_filter(raw, s_ext_cb) if s_ext_cb > 0 else raw) + w_ext_cb * prior
        diag["extgrid_cb"].append({"fold": k, **sc.metrics(p6, G384[i])})
    print(f"fold {k}: token {tbest} sigma {sig} | cb sigma {s_cb} w {w_cb} | kld-sel sigma {sig_kld}, cb ({s_kld},{w_kld})",
          flush=True)

diag_summ = {name: sc.fold_summary(r, folds) for name, r in diag.items()}
for row, rows in cal_rows.items():
    s = sc.fold_summary(rows, folds)
    sn = sc.fold_summary([{**r, "CC": r["CC_native256"], "SIM": r["SIM_native256"], "KLD": r["KLD_native256"]}
                          for r in rows], folds)
    s.update({"row": row, "native256": sn, "fold_selection": fold_info,
              "calibration_maps_used_per_fold": {"token_choice": 800, "sigma_selection": 200,
                                                 "prior_fit": 800 if row.endswith("_cb") else 0,
                                                 "note": "all from the fold's training images; sigma/w "
                                                         "selection subset is drawn from the same 800"},
              "training_maps_used": 0,
              "protocol": PROTO + ("; post-hoc only: token chosen on training CC, Gaussian finalizer sigma "
                                   "from {0,2,4,6,8,10,12} by mean training CC on 200 training images "
                                   "(numpy default_rng(fold)), scipy gaussian_filter at 384")
                          + ("; plus centre-prior mixture (1-w)*prob(blur)+w*prior, prior = mean prob map of the "
                             "fold's 800 training GT maps, (sigma,w) jointly by training CC, w grid "
                             + str(WGRID) if row.endswith("_cb") else "")
                          + "; native256 uses sigma*256/384 and a 256 prior",
              "diagnostics_not_rows": {
                  "kld_criterion_sigma_only": {m: diag_summ["kldsel_sigma"][f"{m}_mean"] for m in ("CC", "SIM", "KLD")},
                  "kld_criterion_sigma_and_prior": {m: diag_summ["kldsel_cb"][f"{m}_mean"] for m in ("CC", "SIM", "KLD")},
                  "normalisation_only_min_subtract_raw": {m: diag_summ["minsub"][f"{m}_mean"] for m in ("CC", "SIM", "KLD")},
                  "extended_sigma_grid_" + str(SIG_EXT) + "_sigma_only": {m: diag_summ["extgrid_sigma"][f"{m}_mean"] for m in ("CC", "SIM", "KLD")},
                  "extended_sigma_grid_sigma_and_prior": {m: diag_summ["extgrid_cb"][f"{m}_mean"] for m in ("CC", "SIM", "KLD")},
                  "zero_shot_token2_same_folds": {m: sc.fold_summary(zs[2], folds)[f"{m}_mean"] for m in ("CC", "SIM", "KLD")}},
              "wallclock_s_scoring_script": time.time() - t0,
              "command": CMD})
    save(row, rows, s)
    print(f"{row}: CC {s['CC_mean']:.4f}+/-{s['CC_sd_folds']:.4f} SIM {s['SIM_mean']:.4f} KLD {s['KLD_mean']:.4f}",
          flush=True)
print("diagnostics:", json.dumps({n: {m: round(d[f'{m}_mean'], 4) for m in ('CC', 'SIM', 'KLD')}
                                  for n, d in diag_summ.items()}), flush=True)
print(f"done in {time.time() - t0:.1f}s")
