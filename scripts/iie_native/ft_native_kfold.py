"""
K-fold in-domain adaptation of DeepGaze IIE on ADD1000 with NATIVE-resolution inputs: the recipe of row
iie_ft_klcc_e20 (scripts/iie/ft_kfold.py --loss klcc64 --epochs 20 --primary sigma; iie_train.train_eval)
with only the input size changed (see PLAN.md). CPU only.

  python -I ft_native_kfold.py --worker --threads 3     # claim folds until none is left (run 2 workers)
  python -I ft_native_kfold.py --aggregate              # write results/iie_native_adapt{,_nosigma}_*.json

Per fold: epoch checkpoints every 5 epochs (resume), readout checkpoint + partial JSON at the end
(cache/iie_native/ckpt/, cache/iie_native/partial/), so a crash loses at most part of one fold.
"""
import os, sys, json, time, argparse, datetime, socket
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from scipy.ndimage import gaussian_filter
import native_common as N
C = N.C
sys.path.insert(0, os.path.join(C.KBS, "scripts", "iie"))
import iie_train as T

ap = argparse.ArgumentParser()
ap.add_argument("--worker", action="store_true")
ap.add_argument("--aggregate", action="store_true")
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--folds", default="0,1,2,3,4")
a = ap.parse_args()
torch.set_num_threads(a.threads)
ROW = N.ROW
EPOCHS, LR, BS, LOSS = 20, 1e-3, 8, "klcc64"
PART = os.path.join(N.NCACHE, "partial"); os.makedirs(PART, exist_ok=True)
CKPT = os.path.join(N.NCACHE, "ckpt"); os.makedirs(CKPT, exist_ok=True)
LOG = os.path.join(C.KBS, "logs", f"{ROW}.log")
CMD = "python -I " + " ".join([os.path.abspath(__file__)] + sys.argv[1:])


def log(s):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [pid {os.getpid()}] {s}"
    print(line, flush=True); open(LOG, "a").write(line + "\n")


sp = json.load(open(C.SPLIT))
folds = [[str(i) for i in f] for f in sp["folds"]]
assert len({i for f in folds for i in f}) == 1000


def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def claim(k):
    pp = os.path.join(PART, f"{ROW}_fold{k}.json")
    cp = os.path.join(PART, f"{ROW}_fold{k}.claim")
    if os.path.exists(pp):
        return False
    try:
        fd = os.open(cp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        old = json.load(open(cp))
        if old.get("host") == socket.gethostname() and not alive(int(old["pid"])):
            log(f"fold {k}: stale claim of dead pid {old['pid']}, taking over")
            os.replace(cp, cp + f".stale{old['pid']}")
            return claim(k)
        return False
    os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "threads": a.threads,
                             "start": datetime.datetime.now().isoformat()}).encode())
    os.close(fd)
    return True


def run_fold(k, D, NF, orders):
    import deepgaze_pytorch
    te = folds[k]
    tr = [i for kk in range(len(folds)) if kk != k for i in folds[kk]]
    sel = [str(x) for x in np.random.default_rng(k).choice(tr, size=min(C.SIGMA_SELECT_N, len(tr)), replace=False)]
    t0 = time.time()
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
    st = N.make_native_stacked(model)
    n_params = int(sum(p.numel() for p in st.parameters()))
    opt = torch.optim.Adam(st.parameters(), lr=LR)
    prior = D.prior(tr)
    geos, cbds = {}, {}

    def prep(i):
        H, W = NF.hw(i)
        if (H, W) not in geos:
            geos[(H, W)] = N.Geometry(H, W)
            cbds[(H, W)] = geos[(H, W)].dcb(N.native_centerbias(prior, H, W))
        return geos[(H, W)], cbds[(H, W)]

    ep_ck = os.path.join(CKPT, f"{ROW}_fold{k}_epochs.pt")
    ep_losses, steps, start_ep, t_prev = [], 0, 0, 0.0
    if os.path.exists(ep_ck):
        ck = torch.load(ep_ck, weights_only=False)
        st.load_state_dict(ck["st"]); opt.load_state_dict(ck["opt"])
        ep_losses, steps, start_ep, t_prev = ck["ep_losses"], ck["steps"], ck["epoch"] + 1, ck["t_train_s"]
        log(f"fold {k}: resumed after epoch {ck['epoch']} ({steps} steps)")
    t_run = time.time()
    for ep in range(start_ep, EPOCHS):
        order = orders[k][ep]
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), BS):
            bt = order[b0:b0 + BS]
            opt.zero_grad()
            lb = 0.0
            for i in bt:                       # batch of 8 = 8 accumulated single-image passes (loss_i / 8)
                geo, cbd = prep(i)
                g64 = torch.from_numpy(np.array(D.ms64[D.pos[i]], dtype=np.float32))
                li = N.klcc64_native(st.logP(C.split_channels(NF.get(i)), cbd, geo), geo, g64) / len(bt)
                li.backward()
                lb += float(li.detach())
            opt.step()
            tot += lb; nb += 1; steps += 1
        ep_losses.append(tot / nb)
        t_train = t_prev + time.time() - t_run
        if ep % 5 == 0 or ep == EPOCHS - 1:
            log(f"fold {k} epoch {ep} mean loss {tot / nb:.4f}  ({t_train:.0f}s, {steps} steps)")
        if ep % 5 == 4 and ep < EPOCHS - 1:
            torch.save({"st": st.state_dict(), "opt": opt.state_dict(), "ep_losses": ep_losses, "steps": steps,
                        "epoch": ep, "t_train_s": t_train}, ep_ck + ".tmp")
            os.replace(ep_ck + ".tmp", ep_ck)
    t_train = t_prev + time.time() - t_run
    st.write_back(model)
    torch.save(C.readout_state_dict(model), os.path.join(CKPT, f"{ROW}_fold{k}.pt"))
    t1 = time.time()
    pt = N.predict_native(model, NF.get, te, prior, NF.hw)
    ps = N.predict_native(model, NF.get, sel, prior, NF.hw)
    sigma, curve = T.select_sigma(ps, D, sel)
    rows0, rows_s, pubm = [], [], []
    for i in te:
        g = D.raw[D.pos[i]].astype(np.float64)
        rows0.append({"id": i, **C.metrics(pt[i], g)})
        p2 = gaussian_filter(pt[i], sigma) if sigma > 0 else pt[i]
        rows_s.append({"id": i, "sigma": sigma, **C.metrics(p2, g)})
        pubm.append(C.metrics(pt[i], D.minsub(i)))
    res = {"rows_nosigma": rows0, "rows_sigma": rows_s, "sigma": sigma, "sigma_curve_train": curve,
           "published_scoring_nosigma": {m: float(np.mean([r[m] for r in pubm])) for m in ("CC", "SIM", "KLD")},
           "epoch_mean_loss": ep_losses, "steps": steps, "n_train": len(tr), "n_test": len(te),
           "trainable_params": n_params, "t_train_s": t_train, "t_eval_s": time.time() - t1,
           "n_sigma_select": len(sel), "fold": k, "sigma_select_ids": sel, "device": "cpu", "threads": a.threads,
           "worker_pid": os.getpid(), "resumed_from_epoch": start_ep if start_ep else None,
           "n_distinct_sizes_seen": len(geos)}
    log(f"fold {k} RESULT CC={np.mean([r['CC'] for r in rows0]):.4f} (sigma=0)  "
        f"CC={np.mean([r['CC'] for r in rows_s]):.4f} (sigma={sigma})  train {t_train:.0f}s eval {res['t_eval_s']:.0f}s")
    return res


if a.worker:
    D = T.Data()
    NF = N.NativeFeats()
    orders = T.published_orders(folds, EPOCHS, n_total=1000)
    log(f"worker start threads={a.threads} folds={a.folds}")
    for k in [int(x) for x in a.folds.split(",")]:
        if not claim(k):
            continue
        log(f"claimed fold {k}")
        res = run_fold(k, D, NF, orders)
        pp = os.path.join(PART, f"{ROW}_fold{k}.json")
        json.dump(res, open(pp + ".tmp", "w")); os.replace(pp + ".tmp", pp)
    log("worker exit: no unclaimed folds left")

if a.aggregate:
    parts = {}
    for k in range(len(folds)):
        pp = os.path.join(PART, f"{ROW}_fold{k}.json")
        if os.path.exists(pp):
            parts[k] = json.load(open(pp))
    if len(parts) < len(folds):
        log(f"aggregate: only {len(parts)}/{len(folds)} folds done"); sys.exit(1)
    ids, _ = C.load_split()
    order = {i: n for n, i in enumerate(ids)}

    def collect(key):
        rows = []
        for k, r in parts.items():
            for x in r[key]:
                rows.append({"id": x["id"], "fold": k, **{m: x[m] for m in ("CC", "SIM", "KLD")},
                             **({"sigma": x["sigma"]} if "sigma" in x else {})})
        rows.sort(key=lambda r: order[r["id"]])
        return rows

    idx = json.load(open(N.INDEX))
    Ws = [r["W"] for r in idx["images"]]; Hs = [r["H"] for r in idx["images"]]
    common = {
        "split_file": C.SPLIT, "fold_sizes": [len(f) for f in folds],
        "replicates_row": "iie_ft_klcc_e20 (selected recipe; the kbs_* statistics families use it)",
        "trainable_params_trained": parts[0]["trainable_params"],
        "trained_parameters": "all DeepGaze IIE readouts: 4 backbones x 30 components (saliency network + fixation-selection network + finalizer sigma and centre-bias weight); backbones frozen",
        "recipe": "our kfold_train_eval.py loss KL(g64||p64) + (1 - CC(p64,g64)) (weights 1, 1) on the 64x64 loss grid; "
                  "p64 = 6x6 block sums of the native density after torch bilinear resize to 384 (align_corners=False), "
                  "renormalised (= exact 6x6 sum-pool from 384 at 384 input); Adam lr %g, batch %d, %d epochs" % (LR, BS, EPOCHS),
        "input": "NATIVE resolution: each image at its original size (PIL RGB, no resize, 0..255), DeepGaze IIE's own "
                 "nearest 1/2 downsampling before the backbones, readout grid ceil(H/32) x ceil(W/32), output log density "
                 "at H x W (as scripts/revision/iie_native_zeroshot.py)",
        "image_size_range_WH": [[min(Ws), min(Hs)], [max(Ws), max(Hs)]],
        "features": "released DeepGaze IIE backbones on the native image, cached at the native readout grid (float32, "
                    "cache/iie_native/native_feats_float32.bin); training forward = size-general stacked re-implementation of the "
                    "released readouts (cache/iie_native/verify_native.json); evaluation = released module code path "
                    "(iie_common.forward_cached) after writing back the trained parameters",
        "batching": "each batch of 8 (published order) processed as 8 single-image forward/backward passes with gradient "
                    "accumulation of loss_i / 8 and one Adam step (identical to the batch-8 mean loss; per-image normalisations only)",
        "centre_prior": "per fold, log(to_prob(sum of the training folds' min-subtracted 384 maps)) + 1e-8 (as iie_ft_klcc_e20), "
                        "zoomed to H x W with scipy zoom order 0 (mode nearest) and logsumexp-renormalised (DeepGaze README rule)",
        "scoring": "native density resized to 384 x 384 with torch bilinear (align_corners=False, float64) as in "
                   "iie_native_zeroshot.py; CC/SIM/KLD on the 384 grid vs bilinear-384 GT, clip + sum, KLD eps 1e-8",
        "shuffle": "published stream: np.random.default_rng(0), permutation(1000) consumed, then in-place shuffle of the fold's training list each epoch, continuing across folds",
        "training_maps_per_fold": [parts[k]["n_train"] for k in sorted(parts)],
        "steps_per_fold": [parts[k]["steps"] for k in sorted(parts)],
        "t_train_s_per_fold": [parts[k]["t_train_s"] for k in sorted(parts)],
        "t_eval_s_per_fold": [parts[k]["t_eval_s"] for k in sorted(parts)],
        "wall_clock_s_total": float(sum(parts[k]["t_train_s"] + parts[k]["t_eval_s"] for k in parts)),
        "epoch_mean_loss_per_fold": {k: parts[k]["epoch_mean_loss"] for k in sorted(parts)},
        "device_per_fold": [parts[k]["device"] for k in sorted(parts)],
        "threads_per_fold": [parts[k]["threads"] for k in sorted(parts)],
        "resumed_from_epoch_per_fold": [parts[k].get("resumed_from_epoch") for k in sorted(parts)],
        "plan": "scripts/iie_native/PLAN.md", "smoke_test_384": "cache/iie_native/smoke384.json",
        "command": CMD, "worker_command": "python -I scripts/iie_native/ft_native_kfold.py --worker --threads 3 (2 workers)",
    }
    variants = {"nosigma": ("rows_nosigma", "no extra post-hoc finalizer (IIE's own learned blur only)"),
                "sigma": ("rows_sigma", "plus Gaussian finalizer, sigma selected per fold from [0,2,...,12] by mean CC on 200 training images (np.random.default_rng(fold).choice), scipy gaussian_filter on the 384-resized density")}
    for v, (key, desc) in variants.items():
        row = ROW if v == "sigma" else f"{ROW}_{v}"
        rows = collect(key)
        calib = {"prior_maps": parts[0]["n_train"], "sigma_selection_maps": (parts[0]["n_sigma_select"] if v == "sigma" else 0),
                 "note": "sigma-selection maps are a subset of the training maps"}
        summ = {"row": row, **C.fold_summary(rows), "n_images": len(rows), "variant": desc,
                "calibration_maps": calib, "training_maps": parts[0]["n_train"],
                "sigma_per_fold": [parts[k]["sigma"] for k in sorted(parts)] if v == "sigma" else None,
                "sigma_curve_train_per_fold": {k: parts[k]["sigma_curve_train"] for k in sorted(parts)} if v == "sigma" else None,
                **common}
        if v == "nosigma":
            pubf = [parts[k]["published_scoring_nosigma"] for k in sorted(parts)]
            summ["published_scoring_min_subtracted_gt"] = {
                "per_fold": pubf, "CC_mean": float(np.mean([p["CC"] for p in pubf])),
                "CC_sd": float(np.std([p["CC"] for p in pubf])),
                "SIM_mean": float(np.mean([p["SIM"] for p in pubf])), "KLD_mean": float(np.mean([p["KLD"] for p in pubf]))}
        json.dump(rows, open(os.path.join(C.RESULTS, f"{row}_per_image.json"), "w"), indent=0)
        json.dump(summ, open(os.path.join(C.RESULTS, f"{row}_summary.json"), "w"), indent=1)
        log(f"{row}: CC={summ['CC_mean']:.4f}+/-{summ['CC_sd_over_folds']:.4f} SIM={summ['SIM_mean']:.4f} KLD={summ['KLD_mean']:.4f}")
