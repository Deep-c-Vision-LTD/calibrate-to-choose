"""
Zero-shot / calibration-only DeepGaze IIE rows on ADD1000, per image, unified 384 scoring.
Uses the cached readout inputs and the released readout modules (forward_cached; bit-exact
w.r.t. the full forward pass, see verify_cache.py).

Rows written (results/<row>_per_image.json + <row>_summary.json):
  iie_zeroshot_adprior       published 2-fold ADD1000 centre prior (eval_deepgaze_iie.py):
                             prior = log(mean of prob-normalised GT maps of the complementary
                             half; halves = even / odd positions of the numerically sorted ids)
  iie_zeroshot_uniform       uniform centre prior (zeros), truly zero-shot
  iie_zeroshot_foldprior     prior from each 5-fold training split (800 maps), same rule as
                             the fine-tuned rows (log(to_prob(sum of min-subtracted maps)) + eps)
  iie_zeroshot_foldprior_sigma  + Gaussian finalizer sigma selected per fold on 200 training
                             images (grid of kfold_train_eval.py), i.e. calibration-only
                             adaptation under the same post-hoc rule as ours

  python -I zeroshot_iie.py [--device cpu]
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from scipy.ndimage import gaussian_filter
import iie_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()
torch.set_num_threads(a.threads)
dev = a.device
CMD = "python -I " + " ".join([os.path.abspath(__file__)] + sys.argv[1:])

import deepgaze_pytorch
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
C.readout_to(model, dev)
ids, folds = C.load_split()
fold_of = C.fold_of(folds)
pos = {i: k for k, i in enumerate(ids)}
gt = C.gt_cache()
raw = gt["raw384"]                         # exact uint8-valued bilinear 384 maps
feats_mm = C.load_feature_cache("float32")
t_start = time.time()


@torch.no_grad()
def predict(id_list, cb):
    """returns dict id -> float64 density (S,S)"""
    out = {}
    cbt = torch.from_numpy(cb.astype(np.float32)).to(dev)
    for k in range(0, len(id_list), 8):
        bt = id_list[k:k + 8]
        ix = [pos[i] for i in bt]
        fx = torch.from_numpy(np.ascontiguousarray(feats_mm[ix])).to(dev)
        logd = C.forward_cached(model, C.split_channels(fx), cbt[None].expand(len(bt), -1, -1))
        d = torch.exp(logd[:, 0].double()).cpu().numpy()
        for j, i in enumerate(bt):
            out[i] = d[j]
    return out


def prob(i):
    return C.to_prob(raw[pos[i]].astype(np.float64))

def minsub(i):
    m = raw[pos[i]].astype(np.float64); m = m - m.min(); s = m.sum()
    return m / s if s > 0 else m + 1.0 / m.size

def score_rows(preds, extra=None):
    rows = []
    for i in ids:
        g = raw[pos[i]].astype(np.float64)
        r = {"id": i, "fold": fold_of[i], **C.metrics(preds[i], g)}
        if extra: r.update(extra(i))
        rows.append(r)
    return rows

def save(row, rows, notes, calib, extra=None):
    summ = {"row": row, **C.fold_summary(rows),
            "n_images": len(rows), "protocol": notes,
            "calibration_maps": calib, "training_maps": 0,
            "trainable_params_trained": 0,
            "native_resolution": "DeepGaze IIE predicts at its input resolution (384 x 384 here), so the native and unified scores are identical",
            "wall_clock_s_total_script": time.time() - t_start, "command": CMD}
    if extra: summ.update(extra)
    json.dump(rows, open(os.path.join(C.RESULTS, f"{row}_per_image.json"), "w"), indent=0)
    json.dump(summ, open(os.path.join(C.RESULTS, f"{row}_summary.json"), "w"), indent=1)
    print(f"{row}: CC={summ['CC_mean']:.4f}+/-{summ['CC_sd_over_folds']:.4f} "
          f"SIM={summ['SIM_mean']:.4f} KLD={summ['KLD_mean']:.4f} "
          f"(over images CC={summ['CC_mean_over_images']:.4f} SIM={summ['SIM_mean_over_images']:.4f} "
          f"KLD={summ['KLD_mean_over_images']:.4f})", flush=True)
    return summ


# ---- (2) uniform prior ----
t0 = time.time()
pu = predict(ids, np.zeros((C.S, C.S)))
rows_u = score_rows(pu)
save("iie_zeroshot_uniform", rows_u,
     "zero-shot DeepGaze IIE (released weights, no ADD1000 data), uniform centre prior (zeros), "
     "input PIL bilinear 384x384, CC/SIM/KLD on 384 grid vs bilinear-384 GT (clip+sum normalisation)",
     calib=0, extra={"wall_clock_s_row": time.time() - t0,
                     "reference_previous_uniform_number": {"CC": 0.750, "SIM": 0.705, "KLD": 0.320,
                                                           "source": "earlier zero-shot baseline run, script not released (rounded)"}})

# ---- (1) published 2-fold ad prior ----
t0 = time.time()
half_a = ids[0::2]; half_b = ids[1::2]
def fit_cb(lst):
    return C.centerbias_from_maps(np.stack([prob(i) for i in lst]))
cb_a, cb_b = fit_cb(half_a), fit_cb(half_b)
pa = predict(half_b, cb_a); pa.update(predict(half_a, cb_b))
half = {i: ("A" if k % 2 == 0 else "B") for k, i in enumerate(ids)}
rows_a = score_rows(pa, extra=lambda i: {"prior_half": half[i], "prior_fit_on": "B" if half[i] == "A" else "A"})
mA = {m: float(np.mean([r[m] for r in rows_a if r["prior_half"] == "A"])) for m in ("CC", "SIM", "KLD")}
mB = {m: float(np.mean([r[m] for r in rows_a if r["prior_half"] == "B"])) for m in ("CC", "SIM", "KLD")}
# published zero-shot IIE scores of the earlier public release (github.com/Deep-c-Vision-LTD/dont-scale-fine-tune,
# results/deepgaze_iie_zeroshot_add1000.json); optional (PUBLISHED_IIE_JSON overrides the path), used only for the
# reproduction check recorded in the summary; without it the "published" and "reproduction_abs_diff" fields are skipped
PUB_PATH = os.environ.get("PUBLISHED_IIE_JSON", os.path.join(C.KBS, "third_party", "dont-scale-fine-tune", "results",
                                                             "deepgaze_iie_zeroshot_add1000.json"))
pub_check = {}
if os.path.exists(PUB_PATH):
    pub = json.load(open(PUB_PATH))
    pub_check = {"published": {"CC": pub["CC"], "SIM": pub["SIM"], "KLD": pub["KLD"], "per_fold": pub["per_fold"]},
                 "reproduction_abs_diff": {m: abs(float(np.mean([r[m] for r in rows_a])) - pub[m])
                                           for m in ("CC", "SIM", "KLD")}}
else:
    print(f"note: {PUB_PATH} not found (optional earlier release); iie_zeroshot_adprior is written without the "
          "'published' and 'reproduction_abs_diff' fields", flush=True)
save("iie_zeroshot_adprior", rows_a,
     "zero-shot DeepGaze IIE + published 2-fold ADD1000 centre prior (eval_deepgaze_iie.py): prior for an image "
     "is fit on the 500 GT maps of the other half (halves = even/odd positions in numerically sorted ids); "
     "'fold' in per-image rows is the 5-fold id (for pairing), 'prior_half' the 2-fold half",
     calib=500, extra={"wall_clock_s_row": time.time() - t0,
                       "two_fold_halves": {"A": mA, "B": mB}, **pub_check})

# ---- fold prior (800 training maps) +/- sigma finalizer ----
t0 = time.time()
pf, sig_info = {}, []
rows_f, rows_fs = [], []
for k, te in enumerate(folds):
    tr = [i for kk, f in enumerate(folds) if kk != k for i in f]
    acc = np.sum([minsub(i) for i in tr], axis=0)
    cb = np.log(C.to_prob(acc) + C.EPS)
    pt = predict(te, cb)
    rng = np.random.default_rng(k)
    sel = list(rng.choice(tr, size=C.SIGMA_SELECT_N, replace=False))
    ps = predict(sel, cb)
    best, best_cc, curve = 0, -2, {}
    for s in C.SIGMAS:
        m = float(np.mean([C.cc(gaussian_filter(ps[i], s) if s > 0 else ps[i], minsub(i)) for i in sel]))
        curve[s] = m
        if m > best_cc:
            best, best_cc = s, m
    sig_info.append({"fold": k, "sigma": best, "train_cc_by_sigma": curve, "sigma_select_ids": sel})
    for i in te:
        g = raw[pos[i]].astype(np.float64)
        rows_f.append({"id": i, "fold": k, **C.metrics(pt[i], g)})
        p2 = gaussian_filter(pt[i], best) if best > 0 else pt[i]
        rows_fs.append({"id": i, "fold": k, "sigma": best, **C.metrics(p2, g)})
    print(f"fold {k}: sigma={best}", flush=True)
order = {i: n for n, i in enumerate(ids)}
rows_f.sort(key=lambda r: order[r["id"]]); rows_fs.sort(key=lambda r: order[r["id"]])
save("iie_zeroshot_foldprior", rows_f,
     "zero-shot DeepGaze IIE readout + ADD1000 centre prior fit on each 5-fold training split (800 maps; "
     "log(to_prob(sum of min-subtracted 384 maps)) + 1e-8, the prior of ft_deepgaze_iie_indomain.py)",
     calib=800, extra={"wall_clock_s_row": time.time() - t0})
save("iie_zeroshot_foldprior_sigma", rows_fs,
     "as iie_zeroshot_foldprior, plus Gaussian finalizer on the 384 density with sigma selected per fold from "
     "[0,2,4,6,8,10,12] by mean CC on 200 training images (np.random.default_rng(fold).choice), scipy gaussian_filter",
     calib=800, extra={"wall_clock_s_row": time.time() - t0,
                       "calibration_detail": "800 maps for the prior; 200 of those same training maps for sigma selection",
                       "sigma_selection": sig_info})
print(f"total {time.time() - t_start:.0f}s", flush=True)
