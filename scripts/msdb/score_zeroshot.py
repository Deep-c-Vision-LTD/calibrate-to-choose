"""
Zero-shot DeepGazeMSDB rows from the cached zero-shot readout (released head, generalisation
mode = dataset=None, pixel_per_dva 21.7, 512 x 512 input):

  msdb_zeroshot_adprior   : published protocol, ADD1000 2-fold log-density prior (fit on the
                            even-index half, score the odd half and vice versa, at 512);
                            scored at 384 (protocol) and 512 (reference / validation)
  msdb_zeroshot_uniform   : uniform prior (no ADD1000 data at all), scored at 384 and 512
  msdb_zeroshot_foldprior : prior fitted on each shared fold's 800 training maps plus the
                            post-hoc Gaussian sigma rule (selected on 200 training images per
                            fold, grid of kfold_train_eval.py); no weights trained. This is the
                            calibration-only starting point of the adapted rows.

  python -I score_zeroshot.py
"""
import json, os, sys, time
import numpy as np, torch
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msdb_common import *          # noqa

CMD = "cd scripts/msdb && python -I score_zeroshot.py"
t_start = time.time()
ids, folds, fold_of = load_split()
n = len(ids)
done = np.load(os.path.join(CACHE, "done.npy"))
assert (done == 2).all(), f"cache incomplete: {(done == 2).sum()}/{n}"
R = torch.from_numpy(np.load(os.path.join(CACHE, "readout_zs.npy")))[:, None]   # (n,1,64,64)
net, sigma0, cbw0, scale_w, sd = build_sal_net_pretrained()
head = MSDBHead(net, sigma0, cbw0).eval()
ck = json.load(open(os.path.join(ROOT, "results", "reference", "msdb_zeroshot_reference_per_image.json")))

print("loading GT ...", flush=True)
G384 = {i: load_gt(i, EVAL) for i in ids}
G512 = {i: load_gt(i, S) for i in ids}


@torch.no_grad()
def predict(k, cbd):
    logd = head.finalize(R[k:k + 1], cbd[None])
    return logd_to_both(logd)


def logd_to_both(logd):
    return logd256_to_prob(logd, EVAL)[0], logd256_to_prob(logd, S)[0]


def summarize(row, per, notes, calib, train, extra=None):
    folds_m = []
    for k in range(5):
        sel = [r for r in per if r["fold"] == k]
        folds_m.append({m: float(np.mean([r[m] for r in sel])) for m in ["CC", "SIM", "KLD", "CC_512", "SIM_512", "KLD_512"]}
                       | {"fold": k, "n": len(sel)})
    out = {"row": row, "n_images": len(per), "per_fold": folds_m}
    for m in ["CC", "SIM", "KLD", "CC_512", "SIM_512", "KLD_512"]:
        v = [f[m] for f in folds_m]
        out[f"{m}_mean"] = float(np.mean(v)); out[f"{m}_sd_folds_ddof1"] = float(np.std(v, ddof=1))
        out[f"{m}_sd_folds_ddof0"] = float(np.std(v))
        out[f"{m}_mean_images"] = float(np.mean([r[m] for r in per]))
    out.update({"calibration_maps": calib, "training_maps": train, "trainable_params": 0,
                "protocol": notes, "wall_clock_s": time.time() - t_start, "command": CMD,
                "scoring": "CC/SIM/KLD on 384x384 (GT PIL-bilinear to 384, clip+sum); prediction = MSDB 512 density "
                           "(nearest 2x upsample of its 256 map, as the package) resized to 384 with torch bilinear "
                           "(align_corners=False); *_512 = native 512 grid vs GT PIL-bilinear to 512 (published protocol)"})
    if extra:
        out.update(extra)
    write_json(per, os.path.join(RESULTS, f"{row}_per_image.json"))
    write_json(out, os.path.join(RESULTS, f"{row}_summary.json"))
    print(f"{row}: CC {out['CC_mean']:.4f} (sd {out['CC_sd_folds_ddof1']:.4f}) SIM {out['SIM_mean']:.4f} "
          f"KLD {out['KLD_mean']:.4f} | @512 CC {out['CC_512_mean']:.4f} SIM {out['SIM_512_mean']:.4f} "
          f"KLD {out['KLD_512_mean']:.4f}", flush=True)
    return out


def score_row(prior_of, sigma_of=None):
    per = []
    for k, i in enumerate(ids):
        cbd = prior_of(k)
        p384, p512 = predict(k, cbd)
        s = sigma_of(fold_of[i]) if sigma_of else 0
        if s > 0:
            p384 = gaussian_filter(p384, s); p512 = gaussian_filter(p512, s * S / EVAL)
        m = metrics(p384, G384[i]); m5 = metrics(p512, G512[i])
        per.append({"id": i, "fold": fold_of[i], **m, "CC_512": m5["CC"], "SIM_512": m5["SIM"], "KLD_512": m5["KLD"]})
    return per


# ---------------- (1) published 2-fold ad prior
idx = list(range(n)); A, B = idx[0::2], idx[1::2]
cbA = cb_downscale(fit_prior([ids[k] for k in A]))      # fit on A, used for B
cbB = cb_downscale(fit_prior([ids[k] for k in B]))
per1 = score_row(lambda k: cbA if k % 2 == 1 else cbB)
diffs = []
for k, r in enumerate(per1):
    lab = "fold_A" if k % 2 == 0 else "fold_B"
    pub = ck[lab][str(k)]
    r["prior_half"] = "A" if k % 2 == 0 else "B"
    r["published_512"] = {"CC": pub[0], "SIM": pub[1], "KLD": pub[2]}
    diffs.append([r["CC_512"] - pub[0], r["SIM_512"] - pub[1], r["KLD_512"] - pub[2]])
diffs = np.array(diffs)
pub_all = np.array([ck["fold_A" if k % 2 == 0 else "fold_B"][str(k)] for k in range(n)])
val = {"published_summary_512": {"CC": 0.7772111953758443, "SIM": 0.7105107455132272, "KLD": 0.3037882953509874},
       "stored_per_image_mean_512": dict(zip(["CC", "SIM", "KLD"], pub_all.mean(0).tolist())),
       "ours_mean_512": {m: float(np.mean([r[f"{m}_512"] for r in per1])) for m in ["CC", "SIM", "KLD"]},
       "per_image_abs_diff_vs_stored_512": {m: {"mean": float(np.abs(diffs[:, j]).mean()), "max": float(np.abs(diffs[:, j]).max())}
                                            for j, m in enumerate(["CC", "SIM", "KLD"])},
       "note": "stored per-image values come from the earlier zero-shot CPU run, script not released (float64 normaliser buffers, full model). "
               "Ours: frozen features computed once (cache pass), fp32 scale mix -> released saliency network in fp32 -> "
               "re-implemented finalizer; differences reflect MPS vs CPU arithmetic and buffer cast only."}
print(json.dumps(val, indent=1), flush=True)
s1 = summarize("msdb_zeroshot_adprior", per1,
               "DeepGazeMSDB released weights, generalisation mode (dataset=None), pixel_per_dva 21.7, 512x512 input; "
               "ADD1000 2-fold log-density centre prior (fit on even-index half of the numerically sorted ids at 512, "
               "applied to the odd half, and vice versa) exactly as eval_deepgaze_msdb.py. No post-hoc sigma. "
               "per_fold = grouping of the same per-image scores by the shared 5-fold split (the prior halves are the "
               "published A/B halves, not the 5 folds).",
               calib="500 ADD1000 maps per prior half (the complementary half); none from the test image",
               train=0, extra={"validation_vs_published": val})

# ---------------- (2) uniform prior
zero = torch.zeros(SAL, SAL)
per2 = score_row(lambda k: zero)
s2 = summarize("msdb_zeroshot_uniform", per2,
               "DeepGazeMSDB released weights, generalisation mode, pixel_per_dva 21.7, 512x512 input, uniform centre "
               "prior (log-density constant, equivalent to zeros), no post-hoc sigma. Uses no ADD1000 data.",
               calib=0, train=0)

# ---------------- (3) shared-fold prior + post-hoc sigma rule (calibration only)
cb_fold, sig_fold, sig_log = {}, {}, {}
for f in range(5):
    tr = [i for k2 in range(5) if k2 != f for i in folds[k2]]
    cb_fold[f] = cb_downscale(fit_prior(tr))
    rng = np.random.default_rng(f)
    sel = rng.choice(np.array(tr), size=SIGMA_SELECT_N, replace=False)
    preds = [predict(ids.index(i), cb_fold[f])[0] for i in sel]
    best, bcc, table = 0, -2, {}
    for s in SIGMAS:
        m = float(np.mean([cc(gaussian_filter(p, s) if s > 0 else p, G384[i]) for p, i in zip(preds, sel)]))
        table[s] = m
        if m > bcc:
            bcc, best = m, s
    sig_fold[f] = best; sig_log[f] = table
    print(f"fold {f}: sigma {best} table {table}", flush=True)
per3 = score_row(lambda k: cb_fold[fold_of[ids[k]]], sigma_of=lambda f: sig_fold[f])
sf = summarize("msdb_zeroshot_foldprior", per3,
               "DeepGazeMSDB released weights (no training), generalisation mode, 512x512 input; centre prior fitted "
               "on each shared fold's 800 training maps (log of mean training density at 512, downscaled by the model's "
               "own nearest rule); post-hoc Gaussian finalizer sigma selected per fold on 200 training images "
               "(numpy default_rng(fold)) from grid [0,2,...,12] at 384 by mean CC (sigma scaled by 512/384 for the "
               "native-512 column).",
               calib="800 training maps per fold (prior); sigma selected on 200 of them", train=0,
               extra={"posthoc_sigma_per_fold": sig_fold, "sigma_selection_cc_tables": sig_log})
print(f"done in {time.time()-t_start:.0f}s")
