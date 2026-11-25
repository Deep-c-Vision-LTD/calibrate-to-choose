"""
In-domain adaptation of DeepGazeMSDB on ADD1000 from cached frozen features (exact: the
backbone and, in the generalisation mode, the scale-mix weights are fixed, so the 64x64x2560
scale mix is the head's full input).

Modes
  head : train MSDB's whole saliency network (LayerNorm0/conv0 ... conv4, 26,460 params) plus
         the single-dataset finalizer scalars (Gaussian sigma, centre-bias weight, log priority
         scaling; 3 params) = 26,463 trainable params. Scale-mix weights stay at the released
         generalisation average. Initialised from the released weights.
  dsp  : MSDB's own per-dataset parameters only (10 scale-mix log weights, sigma, centre-bias
         weight, log priority = 13 params), released saliency network frozen. Uses the cached
         per-scale sufficient statistics (exact for any scale weights).

Protocol (same as our readout, kfold_train_eval.py / sample_efficiency.py):
  * folds from a split JSON ({"folds": [[ids]...]}); prior = log mean training density at 512
    from the fold's training maps only; 60 epochs, batch 16, AdamW (wd 1e-4); loss KL + (1 - CC)
    on a 64 x 64 grid (or KL only with --loss kl);
  * post-hoc Gaussian finalizer: sigma selected on training images only (k-fold: 200 drawn with
    default_rng(fold); curves: all n training images) from [0,2,...,12] at 384 by mean CC;
  * scoring at 384 (MSDB 512 density resized bilinearly), native 512 logged.

  python -I train_msdb.py --row msdb_adapt --mode head --loss klcc
  python -I train_msdb.py --row msdb_adapt_curves --mode head --curves
"""
import argparse, json, os, sys, time, threading, queue
import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msdb_common import *          # noqa

SAL_LAYERS = ["bias0", "softplus0", "layernorm1", "conv1", "bias1", "softplus1", "layernorm2", "conv2", "bias2",
              "softplus2", "layernorm3", "conv3", "bias3", "softplus3", "layernorm4", "conv4", "bias4", "softplus4"]


def rest_of_network(net, z):
    for name in SAL_LAYERS:
        z = net._modules[name](z)
    return z


def readout_head(head, f, mu, sd_):
    """conv0(LayerNorm0(f)) without materialising the normalised tensor. LayerNorm0 normalises
    over (C,H,W) jointly with per-channel affine (w,b), so with per-image mean mu and std sd_:
    conv0(LN0(f)) = (W0 diag(w) f - mu * W0 w) / sd_ + W0 b. Exact; gradients reach W0, w, b."""
    net = head.saliency_network
    W0 = net.conv0.weight[:, :, 0, 0]
    Wp = W0 * net.layernorm0.weight[None]
    z = F.conv2d(f, Wp[:, :, None, None])
    z = (z - mu.view(-1, 1, 1, 1) * Wp.sum(1).view(1, -1, 1, 1)) / sd_.view(-1, 1, 1, 1)
    z = z + (W0 @ net.layernorm0.bias).view(1, -1, 1, 1)
    return rest_of_network(net, z)


class DSP(torch.nn.Module):
    """MSDB per-dataset parameters for one new dataset on top of the frozen released head."""

    def __init__(self, head, scale_w):
        super().__init__()
        self.head = head
        for p in head.saliency_network.parameters():
            p.requires_grad_(False)
        self.theta = torch.nn.Parameter(torch.log(scale_w.clone().float()))
        net = head.saliency_network
        self.register_buffer("const0", (net.conv0.weight[:, :, 0, 0] @ net.layernorm0.bias).detach().clone())

    def readout(self, P, Gm):
        a = torch.softmax(self.theta, 0)
        var = torch.einsum("s,bst,t->b", a, Gm, a)
        z = torch.einsum("s,bskhw->bkhw", a, P) / torch.sqrt(var + 1e-12).view(-1, 1, 1, 1)
        z = z + self.const0.view(1, -1, 1, 1)
        return rest_of_network(self.head.saliency_network, z)


def log_p64(logd256):
    B = logd256.shape[0]
    return logd256.view(B, 64, 4, 64, 4).logsumexp(dim=(2, 4))


def loss_fn(logd256, g64, kind):
    lp = log_p64(logd256)
    kl = (g64 * (torch.log(g64 + 1e-12) - lp)).sum((1, 2)).mean()
    if kind == "kl":
        return kl
    p = torch.exp(lp)
    pc = p - p.mean((1, 2), keepdim=True); gc = g64 - g64.mean((1, 2), keepdim=True)
    ccb = (pc * gc).sum((1, 2)) / (torch.sqrt((pc ** 2).sum((1, 2)) * (gc ** 2).sum((1, 2))) + 1e-8)
    return kl + (1 - ccb.mean())


class RowReader:
    """Reads whole rows of a large .npy with pread (and F_NOCACHE on macOS): ~8x faster than
    page-faulting through a memmap under memory pressure, and it does not evict other jobs'
    page cache."""

    def __init__(self, path):
        import fcntl
        with open(path, "rb") as fh:
            v = np.lib.format.read_magic(fh)
            rd = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
            self.shape, _, self.dtype = rd(fh)
            self.offset = fh.tell()
        self.rb = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        self.fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.fcntl(self.fd, fcntl.F_NOCACHE, 1)
        except (AttributeError, OSError):
            pass

    def read(self, idx):
        out = np.empty((len(idx),) + tuple(self.shape[1:]), self.dtype)
        mv = memoryview(out).cast("B"); rb = self.rb
        for j, i in enumerate(idx):
            got = 0
            while got < rb:
                n = os.preadv(self.fd, [mv[j * rb + got:(j + 1) * rb]], self.offset + int(i) * rb + got)
                if n <= 0:
                    raise IOError("short read")
                got += n
        return out


class Data:
    def __init__(self, mode, dev):
        self.ids, _, _ = load_split()                   # cache order = kfold_seed0 ids_sorted
        self.pos = {i: k for k, i in enumerate(self.ids)}
        done = np.load(os.path.join(CACHE, "done.npy"))
        assert (done == 2).all(), f"cache incomplete {(done == 2).sum()}"
        self.mode, self.dev = mode, dev
        st = np.load(os.path.join(CACHE, "mixstats.npy"))
        self.mu = torch.from_numpy(st[:, 0]).float()
        self.sd = torch.from_numpy(np.sqrt(st[:, 1] + 1e-12)).float()
        if mode == "head":
            self.mix = RowReader(os.path.join(CACHE, "mix_fp16.npy"))
        else:
            self.P = torch.from_numpy(np.load(os.path.join(CACHE, "ss_proj.npy")))
            self.G = torch.from_numpy(np.load(os.path.join(CACHE, "ss_gram.npy"))).float()
        t = time.time()
        self.g64 = torch.stack([torch.from_numpy(to_prob(load_gt(i, 64))).float() for i in self.ids])
        self.g384 = np.stack([load_gt(i, EVAL).astype(np.uint8) for i in self.ids])
        self.g512 = np.stack([load_gt(i, S).astype(np.uint8) for i in self.ids])
        self.prior_cache = {}
        print(f"GT loaded in {time.time()-t:.0f}s", flush=True)

    def batch(self, idx):
        idx = np.asarray(idx)
        if self.mode == "head":
            f = torch.from_numpy(self.mix.read(idx))
            return (f, self.mu[idx], self.sd[idx])
        return (self.P[idx], self.G[idx])

    def prior(self, tr_ids):
        key = hash(tuple(sorted(tr_ids)))
        if key not in self.prior_cache:
            self.prior_cache[key] = cb_downscale(fit_prior(list(tr_ids)))
        return self.prior_cache[key]


def to_dev(b, dev):
    out = []
    for t in b:
        t = t.to(dev, non_blocking=False)
        out.append(t.float() if t.dtype == torch.float16 else t)
    return out


def prefetch(data, batches):
    q = queue.Queue(maxsize=2)

    def work():
        for bidx in batches:
            q.put((bidx, data.batch(bidx)))
        q.put(None)
    th = threading.Thread(target=work, daemon=True); th.start()
    while True:
        it = q.get()
        if it is None:
            break
        yield it


def forward(model, mode, b, cbd):
    if mode == "head":
        f, mu, sd_ = b
        r = readout_head(model, f, mu, sd_)
        return model.finalize(r, cbd)
    P, Gm = b
    r = model.readout(P, Gm)
    return model.head.finalize(r, cbd)


@torch.no_grad()
def predict_probs(model, mode, data, pos_list, cbd, dev, bs=16):
    out384, out512 = {}, {}
    for j in range(0, len(pos_list), bs):
        idx = pos_list[j:j + bs]
        b = to_dev(data.batch(idx), dev)
        logd = forward(model, mode, b, cbd)
        p384 = logd256_to_prob(logd, EVAL); p512 = logd256_to_prob(logd, S)
        for k, ii in enumerate(idx):
            out384[ii] = p384[k]; out512[ii] = p512[k]
    return out384, out512


def select_sigma(preds384, data, pos_list):
    best, bcc, table = 0, -2, {}
    for s in SIGMAS:
        m = float(np.mean([cc(gaussian_filter(preds384[k], s) if s > 0 else preds384[k], data.g384[k].astype(np.float64))
                           for k in pos_list]))
        table[s] = m
        if m > bcc:
            bcc, best = m, s
    return best, table


def evaluate(p384, p512, data, pos_list, sigma):
    rows = []
    for k in pos_list:
        a, b = p384[k], p512[k]
        if sigma > 0:
            a = gaussian_filter(a, sigma); b = gaussian_filter(b, sigma * S / EVAL)
        m = metrics(a, data.g384[k].astype(np.float64)); m5 = metrics(b, data.g512[k].astype(np.float64))
        rows.append({"id": data.ids[k], **m, "CC_512": m5["CC"], "SIM_512": m5["SIM"], "KLD_512": m5["KLD"]})
    return rows


def make_model(mode, dev):
    net, sigma0, cbw0, scale_w, sd = build_sal_net_pretrained()
    head = MSDBHead(net, sigma0, cbw0)
    if mode == "head":
        model = head
    else:
        model = DSP(head, scale_w)
    model = model.to(dev)
    model.eval()
    return model


def train_one(mode, data, tr_pos, cbd, dev, a, seed, sigma_pos):
    """Train from the released init on tr_pos; returns model, sigma, logs."""
    model = make_model(mode, dev)
    params = [p for p in model.parameters() if p.requires_grad]
    n_par = int(sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    rng = np.random.default_rng(seed)
    bs = min(a.bs, len(tr_pos))
    cbd_dev = cbd.to(dev)
    ep_loss, t0 = [], time.time()
    for ep in range(a.epochs):
        model.train()
        shuf = list(tr_pos); rng.shuffle(shuf)
        batches = [shuf[j:j + bs] for j in range(0, len(shuf), bs)]
        tot, nb = 0.0, 0
        for bidx, b in prefetch(data, batches):
            b = to_dev(b, dev)
            g = data.g64[np.asarray(bidx)].to(dev)
            logd = forward(model, mode, b, cbd_dev)
            loss = loss_fn(logd, g, a.loss)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        ep_loss.append(tot / nb)
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"    ep {ep} loss {ep_loss[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    p384, _ = predict_probs(model, mode, data, list(sigma_pos), cbd_dev, dev)
    sigma, table = select_sigma(p384, data, list(sigma_pos))
    fin = {"sigma_finalizer": float(model.sigma if mode == "head" else model.head.sigma),
           "cb_weight": float(model.cbw if mode == "head" else model.head.cbw),
           "priority": float(torch.exp(model.log_priority if mode == "head" else model.head.log_priority))}
    if mode == "dsp":
        fin["scale_weights"] = torch.softmax(model.theta, 0).tolist()
    return model, sigma, {"epoch_loss": ep_loss, "trainable_params": n_par, "sigma_table": table,
                          "train_s": time.time() - t0, "learned": fin}


def summarize_folds(per, nf):
    out = {"per_fold": []}
    for k in range(nf):
        sel = [r for r in per if r["fold"] == k]
        out["per_fold"].append({"fold": k, "n": len(sel), **{m: float(np.mean([r[m] for r in sel]))
                                for m in ["CC", "SIM", "KLD", "CC_512", "SIM_512", "KLD_512"]}})
    for m in ["CC", "SIM", "KLD", "CC_512", "SIM_512", "KLD_512"]:
        v = [f[m] for f in out["per_fold"]]
        out[f"{m}_mean"] = float(np.mean(v)); out[f"{m}_sd_folds_ddof1"] = float(np.std(v, ddof=1))
        out[f"{m}_sd_folds_ddof0"] = float(np.std(v)); out[f"{m}_mean_images"] = float(np.mean([r[m] for r in per]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--mode", choices=["head", "dsp"], default="head")
    ap.add_argument("--loss", choices=["klcc", "kl"], default="klcc")
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--sizes", default="25,50,100,200,400,800")
    ap.add_argument("--resamples", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dev", default="mps")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-lock", action="store_true")
    a = ap.parse_args()
    if a.lr is None:
        a.lr = 3e-4 if a.mode == "head" else 1e-2
    torch.set_num_threads(a.threads)
    cmd = "cd scripts/msdb && python -I train_msdb.py " + " ".join(sys.argv[1:])
    t_start = time.time()
    import signal
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda s, f: sys.exit(128 + s))      # so atexit cleanup runs
    # one process per row: a second invocation of the same row (e.g. CPU and MPS queues) exits
    row_lock = os.path.join(RESULTS, f".running_{a.row}")
    try:
        os.mkdir(row_lock)
    except FileExistsError:
        print(f"[{a.row}] already running in another process ({row_lock}); exiting", flush=True)
        return
    import atexit
    atexit.register(lambda: os.path.isdir(row_lock) and os.rmdir(row_lock))
    lock = MPSLock().acquire() if (a.dev == "mps" and not a.no_lock) else None
    torch.manual_seed(0)
    data = Data(a.mode, a.dev)
    sp = json.load(open(a.split))
    folds = [list(map(str, f)) for f in sp["folds"]]
    nf = len(folds)
    fold_of = {i: k for k, f in enumerate(folds) for i in f}
    assert len(fold_of) == len(data.ids)
    per_path = os.path.join(RESULTS, f"{a.row}_per_image.json")
    sum_path = os.path.join(RESULTS, f"{a.row}_summary.json")
    seeds = [int(s) for s in a.seeds.split(",")]

    # sanity: re-implemented readout equals the released saliency network on 2 cached images
    if a.mode == "head":
        m0 = make_model("head", a.dev)
        b = to_dev(data.batch([0, 1]), a.dev)
        with torch.no_grad():
            r1 = readout_head(m0, *b); r2 = m0.saliency_network(b[0])
        rel = float((r1 - r2).abs().max() / r2.abs().max())
        print(f"readout identity check: max rel err {rel:.2e}", flush=True)
        assert rel < 1e-3
        del m0
    else:
        m0 = make_model("dsp", a.dev)
        b = to_dev(data.batch([0, 1]), a.dev)
        rz = torch.from_numpy(np.load(os.path.join(CACHE, "readout_zs.npy"), mmap_mode="r")[:2].copy()).to(a.dev)
        with torch.no_grad():
            r1 = m0.readout(*b)[:, 0]
        rel = float((r1 - rz).abs().max() / rz.abs().max())
        print(f"sufficient-statistics identity check vs cached zero-shot readout: max rel err {rel:.2e}", flush=True)
        assert rel < 1e-3
        del m0

    common = {"row": a.row, "mode": a.mode, "loss": ("KL + (1 - CC)" if a.loss == "klcc" else "KL"),
              "split": os.path.relpath(a.split, KBS), "epochs": a.epochs, "lr": a.lr, "weight_decay": a.wd,
              "batch": a.bs, "optimizer": "AdamW", "device": a.dev, "command": cmd,
              "what_is_trained": ("MSDB saliency network (all 26,460 params) + single-dataset finalizer sigma, "
                                  "centre-bias weight and log priority scaling (3); frozen CLIP-RN50x64 + DINOv2-B/14 "
                                  "multi-scale backbone; scale-mix weights fixed at the released generalisation average"
                                  if a.mode == "head" else
                                  "MSDB per-dataset parameters only: 10 scale-mix log weights, finalizer sigma, "
                                  "centre-bias weight, log priority (13); released saliency network and backbone frozen"),
              "init": "released DeepGazeMSDB weights, generalisation-mode values (dataset=None)",
              "input": "512x512 image (PIL bilinear), pixel_per_dva 21.7, cached frozen features (fp16 scale mix)"
                       if a.mode == "head" else "512x512 image, cached per-scale sufficient statistics (fp32)",
              "scoring": "384x384 (MSDB 512 density -> torch bilinear 384; GT PIL bilinear 384, clip+sum); *_512 native"}

    if not a.curves:
        fl = list(range(nf)) if a.folds == "all" else [int(x) for x in a.folds.split(",")]
        per, fold_logs = [], []
        if os.path.exists(per_path) and os.path.exists(sum_path + ".partial"):
            per = json.load(open(per_path)); fold_logs = json.load(open(sum_path + ".partial"))
        elif os.path.exists(per_path) and os.path.exists(sum_path):     # extend a finished row with more seeds
            per = json.load(open(per_path)); fold_logs = json.load(open(sum_path))["fold_logs"]
        done_keys = {(r["fold"], r["seed"]) for r in fold_logs}
        if os.path.exists(sum_path) and all((k, s) in done_keys for s in seeds for k in fl):
            print(f"[{a.row}] all requested folds/seeds already finished; nothing to do", flush=True)
            if lock:
                lock.release()
            return
        seeds = sorted(set(seeds) | {r["seed"] for r in fold_logs})
        for seed in seeds:
            for k in fl:
                if (k, seed) in done_keys:
                    continue
                te = folds[k]; tr = [i for j in range(nf) if j != k for i in folds[j]]
                te_pos = [data.pos[i] for i in te]; tr_pos = [data.pos[i] for i in tr]
                cbd = data.prior(tr)
                rng = np.random.default_rng(k)
                sig_pos = rng.choice(np.array(tr_pos), size=min(SIGMA_SELECT_N, len(tr_pos)), replace=False)
                print(f"[{a.row}] fold {k} seed {seed}: train {len(tr)} test {len(te)}", flush=True)
                # starting point (released head + fold prior + sigma rule)
                m_start = make_model(a.mode, a.dev)
                p384s, p512s = predict_probs(m_start, a.mode, data, list(sig_pos), cbd.to(a.dev), a.dev)
                s_start, _ = select_sigma(p384s, data, list(sig_pos))
                p384t, p512t = predict_probs(m_start, a.mode, data, te_pos, cbd.to(a.dev), a.dev)
                start_rows = evaluate(p384t, p512t, data, te_pos, s_start)
                start_cc = float(np.mean([r["CC"] for r in start_rows]))
                del m_start
                model, sigma, lg = train_one(a.mode, data, tr_pos, cbd, a.dev, a, seed, sig_pos)
                p384t, p512t = predict_probs(model, a.mode, data, te_pos, cbd.to(a.dev), a.dev)
                rows = evaluate(p384t, p512t, data, te_pos, sigma)
                for r in rows:
                    r.update({"fold": k, "seed": seed})
                per.extend(rows)
                fl_log = {"fold": k, "seed": seed, "n_train": len(tr), "n_test": len(te), "posthoc_sigma": sigma,
                          "start_posthoc_sigma": s_start, "start_CC": start_cc,
                          "CC": float(np.mean([r["CC"] for r in rows])), "SIM": float(np.mean([r["SIM"] for r in rows])),
                          "KLD": float(np.mean([r["KLD"] for r in rows])), **lg}
                fold_logs.append(fl_log)
                print(f"[{a.row}] fold {k} seed {seed}: start CC {start_cc:.4f} (sigma {s_start}) -> trained CC "
                      f"{fl_log['CC']:.4f} SIM {fl_log['SIM']:.4f} KLD {fl_log['KLD']:.4f} (sigma {sigma}, "
                      f"{lg['train_s']:.0f}s) learned {lg['learned']}", flush=True)
                write_json(per, per_path); write_json(fold_logs, sum_path + ".partial")
                del model
                if a.dev == "mps":
                    torch.mps.empty_cache()
        summ = dict(common)
        for seed in seeds:
            ps = [r for r in per if r["seed"] == seed]
            summ[f"seed{seed}"] = summarize_folds(ps, nf)
        s0 = summ[f"seed{seeds[0]}"]
        for kk in ["CC_mean", "CC_sd_folds_ddof1", "SIM_mean", "KLD_mean", "CC_512_mean", "SIM_512_mean", "KLD_512_mean"]:
            summ[kk] = s0[kk]
        summ["per_fold"] = s0["per_fold"]
        if len(seeds) > 1:
            summ["CC_mean_over_seeds"] = float(np.mean([summ[f"seed{s}"]["CC_mean"] for s in seeds]))
            summ["CC_sd_over_seeds"] = float(np.std([summ[f"seed{s}"]["CC_mean"] for s in seeds], ddof=1))
        summ["fold_logs"] = fold_logs
        summ["trainable_params"] = fold_logs[0]["trainable_params"]
        summ["calibration_maps"] = "per fold: the fold's training maps fit the centre prior; post-hoc sigma selected on 200 of them"
        summ["training_maps"] = "per fold: all training-fold maps (" + ",".join(str(fl_["n_train"]) for fl_ in fold_logs[:nf]) + ")"
        summ["wall_clock_s"] = time.time() - t_start
        write_json(summ, sum_path)
        if os.path.exists(sum_path + ".partial"):
            os.remove(sum_path + ".partial")
        print(f"[{a.row}] CC {summ['CC_mean']:.4f} sd {summ['CC_sd_folds_ddof1']:.4f} SIM {summ['SIM_mean']:.4f} "
              f"KLD {summ['KLD_mean']:.4f} in {summ['wall_clock_s']:.0f}s", flush=True)
    else:
        sizes = [int(x) for x in a.sizes.split(",")]
        te = folds[0]; pool = np.array([i for j in range(1, nf) for i in folds[j]])
        te_pos = [data.pos[i] for i in te]
        per, runs = [], []
        if os.path.exists(per_path) and os.path.exists(sum_path + ".partial"):
            per = json.load(open(per_path)); runs = json.load(open(sum_path + ".partial"))
        done_keys = {(r["n"], r["resample"]) for r in runs}
        for n in sizes:
            for r_ in range(a.resamples):
                if (n, r_) in done_keys:
                    continue
                rr = np.random.default_rng(1000 * n + r_)
                tr = list(rr.choice(pool, size=n, replace=False))      # sample_efficiency.py subset rule
                tr_pos = [data.pos[i] for i in tr]
                cbd = data.prior(tr)                                   # prior from the same n maps only
                model, sigma, lg = train_one(a.mode, data, tr_pos, cbd, a.dev, a, r_, tr_pos)
                p384t, p512t = predict_probs(model, a.mode, data, te_pos, cbd.to(a.dev), a.dev)
                rows = evaluate(p384t, p512t, data, te_pos, sigma)
                for r in rows:
                    r.update({"fold": 0, "n": n, "resample": r_, "seed": r_})
                per.extend(rows)
                run = {"n": n, "resample": r_, "train_ids": [str(x) for x in tr], "posthoc_sigma": sigma,
                       "CC": float(np.mean([r["CC"] for r in rows])), "SIM": float(np.mean([r["SIM"] for r in rows])),
                       "KLD": float(np.mean([r["KLD"] for r in rows])), **lg}
                runs.append(run)
                print(f"[{a.row}] n={n} r={r_}: CC {run['CC']:.4f} SIM {run['SIM']:.4f} KLD {run['KLD']:.4f} "
                      f"(sigma {sigma}, {lg['train_s']:.0f}s)", flush=True)
                write_json(per, per_path); write_json(runs, sum_path + ".partial")
                del model
                if a.dev == "mps":
                    torch.mps.empty_cache()
        curve = []
        for n in sizes:
            rs = [r for r in runs if r["n"] == n]
            curve.append({"n": n, "mean_CC": float(np.mean([r["CC"] for r in rs])), "sd_CC": float(np.std([r["CC"] for r in rs])),
                          "mean_SIM": float(np.mean([r["SIM"] for r in rs])), "mean_KLD": float(np.mean([r["KLD"] for r in rs])),
                          "runs_CC": [r["CC"] for r in rs]})
        summ = dict(common)
        summ.update({"curve": curve, "test_set": "fold 0 of the split (200 ids)",
                     "subset_rule": "sample_efficiency.py: pool = folds 1-4 in order, numpy default_rng(1000*n + r).choice(pool, n, replace=False)",
                     "calibration_maps": "per run: the same n training maps fit the centre prior and select the post-hoc sigma",
                     "training_maps": "per run: n", "runs": runs, "trainable_params": runs[0]["trainable_params"],
                     "wall_clock_s": time.time() - t_start})
        write_json(summ, sum_path)
        if os.path.exists(sum_path + ".partial"):
            os.remove(sum_path + ".partial")
        for c in curve:
            print(f"[{a.row}] n={c['n']}: CC {c['mean_CC']:.4f} +/- {c['sd_CC']:.4f}", flush=True)
    if lock:
        lock.release()


if __name__ == "__main__":
    main()
