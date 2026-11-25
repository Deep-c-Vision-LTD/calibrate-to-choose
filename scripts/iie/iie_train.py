"""
Training / evaluation core for the adapted DeepGaze IIE rows (cached readout inputs).

Recipes
  kl384   : ft_deepgaze_iie_indomain.py (published): KL(g || p) on the 384 grid, g = min-subtracted,
            sum-normalised bilinear-384 GT, p = log_softmax of the IIE log density; Adam lr 1e-3,
            batch 8, 20 epochs, all 4 x 30 readouts + finalizers trainable (3,059,160 params),
            centre prior = log(to_prob(sum of the training maps)) + 1e-8.
  klcc64  : our loss from kfold_train_eval.py: KL(g64 || p64) + (1 - CC(p64, g64)), weights 1 and 1,
            on the 64 x 64 loss grid with g64 = min-subtracted, sum-normalised bilinear-64 GT. p64 is
            the IIE density sum-pooled exactly to 64 x 64 (6 x 6 blocks of the 384 grid); the CC term
            uses this density (our readout applies it to sigmoid(logits); IIE outputs a log density,
            and sigmoid(x) ~ exp(x) for these values anyway). Optimiser / batch / prior as kl384.
Evaluation: trained parameters are written back into a deepgaze_pytorch.DeepGazeIIE instance and
predictions come from the released module code path (forward_cached). Gaussian finalizer sigma is
selected on training images only (grid [0..12], mean CC), as in kfold_train_eval.py.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import iie_common as C

PUB_EPOCHS = 20


class Data:
    def __init__(self):
        self.ids, self.folds = C.load_split()
        self.pos = {i: k for k, i in enumerate(self.ids)}
        if all(os.path.exists(os.path.join(C.CACHE, f"gt_{k}.npy")) for k in ("raw384", "ms384", "ms64")):
            # memory-mapped copies of gt_cache.npz (shared page cache across worker processes)
            self.raw, self.ms384, self.ms64 = [np.load(os.path.join(C.CACHE, f"gt_{k}.npy"), mmap_mode="r")
                                               for k in ("raw384", "ms384", "ms64")]
        else:
            g = C.gt_cache()
            self.raw = g["raw384"]
            self.ms384 = g["ms384"]
            self.ms64 = g["ms64"]
        self.feats = C.load_feature_cache("float32")

    def minsub(self, i):
        m = self.raw[self.pos[i]].astype(np.float64); m = m - m.min(); s = m.sum()
        return m / s if s > 0 else m + 1.0 / m.size

    def prior(self, tr):
        acc = np.zeros((C.S, C.S), np.float64)
        for i in tr:
            acc += self.minsub(i)
        return np.log(C.to_prob(acc) + C.EPS)

    def batch_feats(self, bt, dev):
        ix = [self.pos[i] for i in bt]
        srt = np.argsort(ix)
        arr = np.empty((len(ix), sum(C.BACKBONE_CH), C.G, C.G), np.float32)
        arr[srt] = self.feats[np.asarray(ix)[srt]]
        return C.split_channels(torch.from_numpy(arr).to(dev))


def published_orders(folds, epochs, n_total=1000):
    """Replicates the shuffle stream of ft_deepgaze_iie_indomain.py: one default_rng(0) consumed by
    permutation(n_total) and then by an in-place shuffle of the fold's training list every epoch,
    continuing across folds. Returns {fold: [order_epoch0, ...]}."""
    rng = np.random.default_rng(0)
    rng.permutation(n_total)
    out = {}
    for f in range(len(folds)):
        tr = [i for k in range(len(folds)) if k != f for i in folds[k]]
        eps = []
        for _ in range(epochs):
            rng.shuffle(tr)
            eps.append(list(tr))
        out[f] = eps
    return out


def seeded_orders(tr_ids, epochs, seed):
    """sample_efficiency.py rule: fresh copy of the training list shuffled by default_rng(seed) every epoch."""
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(epochs):
        t = list(tr_ids); rng.shuffle(t); eps.append(t)
    return eps


def loss_fn(st, feats, cbb, ids, D, loss, dev):
    B = len(ids)
    if loss == "kl384":
        logd = st(feats, cbb)[:, 0]
        logp = F.log_softmax(logd.reshape(B, -1), 1).reshape(B, C.S, C.S)
        g = torch.from_numpy(np.ascontiguousarray(D.ms384[[D.pos[i] for i in ids]])).to(dev)
        return (g * (torch.log(g + C.EPS) - logp)).sum(dim=(1, 2)).mean()
    if loss == "klcc64":
        p192 = st.logd192(feats, cbb)                                   # per-384-pixel log density
        d64 = F.avg_pool2d(torch.exp(p192)[:, None], 3)[:, 0] * 36.0      # exact 6x6 sum-pool of the 384 density
        g = torch.from_numpy(np.ascontiguousarray(D.ms64[[D.pos[i] for i in ids]])).to(dev)
        kl = (g * (torch.log(g + 1e-12) - torch.log(d64.clamp_min(1e-30)))).sum((1, 2)).mean()
        pc = d64 - d64.mean((1, 2), keepdim=True); gc = g - g.mean((1, 2), keepdim=True)
        ccb = (pc * gc).sum((1, 2)) / (torch.sqrt((pc ** 2).sum((1, 2)) * (gc ** 2).sum((1, 2))) + 1e-8)
        return kl + (1 - ccb.mean())
    raise ValueError(loss)


@torch.no_grad()
def predict_module(model, D, id_list, cb, dev, bs=8):
    out = {}
    cbt = torch.from_numpy(cb.astype(np.float32)).to(dev)
    for k in range(0, len(id_list), bs):
        bt = id_list[k:k + bs]
        logd = C.forward_cached(model, D.batch_feats(bt, dev), cbt[None].expand(len(bt), -1, -1))
        d = torch.exp(logd[:, 0].float().cpu().double()).numpy()   # MPS has no float64
        for j, i in enumerate(bt):
            out[i] = d[j]
    return out


def select_sigma(preds, D, sel):
    best, best_cc, curve = 0, -2.0, {}
    for s in C.SIGMAS:
        m = float(np.mean([C.cc(gaussian_filter(preds[i], s) if s > 0 else preds[i], D.minsub(i)) for i in sel]))
        curve[str(s)] = m
        if m > best_cc:
            best, best_cc = s, m
    return best, curve


def train_eval(D, tr, te, orders, loss, lr, bs, dev, sigma_sel_ids, log=print, tag="", ckpt_path=None,
               extra_eval_ids=None):
    """Train a fresh released DeepGaze IIE readout on tr (feature cache), evaluate on te.
    orders: list (epochs) of id lists. Returns dict with per-image metrics (sigma 0 and selected)."""
    import deepgaze_pytorch
    t0 = time.time()
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
    st = C.StackedIIE(model).to(dev)
    n_params = int(sum(p.numel() for p in st.parameters()))
    opt = torch.optim.Adam(st.parameters(), lr=lr)
    cb = D.prior(tr)
    cbt = torch.from_numpy(cb.astype(np.float32)).to(dev)
    ep_losses = []
    steps = 0
    for ep, order in enumerate(orders):
        tot, nb = 0.0, 0
        for k in range(0, len(order), bs):
            bt = order[k:k + bs]
            feats = D.batch_feats(bt, dev)
            l = loss_fn(st, feats, cbt[None].expand(len(bt), -1, -1), bt, D, loss, dev)
            opt.zero_grad(); l.backward(); opt.step()
            tot += float(l); nb += 1; steps += 1
        ep_losses.append(tot / nb)
        if ep % 5 == 0 or ep == len(orders) - 1:
            log(f"{tag} epoch {ep} mean loss {tot / nb:.4f}  ({time.time() - t0:.0f}s, {steps} steps)")
    t_train = time.time() - t0
    st.write_back(model)
    C.readout_to(model, dev)
    if ckpt_path:
        torch.save(C.readout_state_dict(model), ckpt_path)
    t1 = time.time()
    pt = predict_module(model, D, te, cb, dev)
    ps = predict_module(model, D, sigma_sel_ids, cb, dev)
    sigma, curve = select_sigma(ps, D, sigma_sel_ids)
    rows0, rows_s = [], []
    pubm = []
    for i in te:
        g = D.raw[D.pos[i]].astype(np.float64)
        rows0.append({"id": i, **C.metrics(pt[i], g)})
        p2 = gaussian_filter(pt[i], sigma) if sigma > 0 else pt[i]
        rows_s.append({"id": i, "sigma": sigma, **C.metrics(p2, g)})
        pubm.append(C.metrics(pt[i], D.minsub(i)))           # published scoring (min-subtracted GT)
    extra = None
    if extra_eval_ids:
        pe = predict_module(model, D, extra_eval_ids, cb, dev)
        extra = {i: C.metrics(pe[i], D.raw[D.pos[i]].astype(np.float64)) for i in extra_eval_ids}
    res = {"rows_nosigma": rows0, "rows_sigma": rows_s, "sigma": sigma, "sigma_curve_train": curve,
           "published_scoring_nosigma": {m: float(np.mean([r[m] for r in pubm])) for m in ("CC", "SIM", "KLD")},
           "epoch_mean_loss": ep_losses, "steps": steps, "n_train": len(tr), "n_test": len(te),
           "trainable_params": n_params, "t_train_s": t_train, "t_eval_s": time.time() - t1,
           "n_sigma_select": len(sigma_sel_ids), "extra_eval": extra}
    log(f"{tag} RESULT CC={np.mean([r['CC'] for r in rows0]):.4f} (sigma=0)  "
        f"CC={np.mean([r['CC'] for r in rows_s]):.4f} (sigma={sigma})  train {t_train:.0f}s")
    del st, opt, model
    if dev == "mps":
        torch.mps.empty_cache()
    return res
