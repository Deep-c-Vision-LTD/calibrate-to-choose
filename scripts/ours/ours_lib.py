"""
Shared code for the "ours" track: the frozen-DINOv2-base + ASPP readout trained on ADD1000
exactly as kfold_train_eval.py / ablation_add1000.py / sample_efficiency.py (v3), with
per-image metrics, explicit init seeds and the unified scorer.

Readout = ReadoutHead (state-dict keys aspp.*, head.*, center_bias), the same module as the
readout part of model_aspp.ASPPSaliency; features are pre-cached by cache_features.py.

Unified scorer (fixed protocol of the letter): 384 x 384 grid; GT = map resized bilinearly
to 384; prediction and GT turned into probability maps by clipping at 0 and dividing by the
sum; CC on the raw maps, SIM / KLD (eps 1e-8) on the probability maps.
Training targets keep the published load_gt (64 x 64, min subtracted, normalised).
"""
import sys
sys.dont_write_bytecode = True
import os, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_aspp import ASPPModule  # noqa: E402

ROOT = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root (CBYC_ROOT overrides)
ADD1000_DIR = os.environ.get("ADD1000_DIR", os.path.join(ROOT, "data", "ADD1000"))   # holds flat/{images,maps}
KB = ROOT
FLAT = os.path.join(ADD1000_DIR, "flat")
CACHE = os.path.join(KB, "cache", "ours")
RESULTS = os.environ.get("OURS_RESULTS_DIR", os.path.join(KB, "results"))   # override only for smoke tests
WARM_CKPT = os.environ.get("WARM_CKPT", "")   # warm-start readout checkpoint: NOT released; only the warm-start readout rows of the supplement need it
EVAL = 384; LOSSG = 64
SIGMAS = [0, 2, 4, 6, 8, 10, 12]
SIGMA_SELECT_N = 200
EPS = 1e-8


class PlainReadout(nn.Module):
    """No-ASPP control from ablation_add1000.py: plain 2-conv readout, same width."""
    def __init__(self, in_ch=768, ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.GELU(),
            nn.Dropout2d(0.1))

    def forward(self, x):
        return self.net(x)


class ReadoutHead(nn.Module):
    def __init__(self, mode="full"):
        super().__init__()
        self.aspp = PlainReadout(768, 128) if mode == "no_aspp" else ASPPModule(768, 128)
        self.head = nn.Conv2d(128, 1, 1)
        self.use_cb = mode != "no_centerbias"
        if self.use_cb:
            self.center_bias = nn.Parameter(torch.zeros(1, 1, 16, 16))

    def forward(self, f):
        s = self.head(self.aspp(f))
        if self.use_cb:
            s = s + F.interpolate(self.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return s


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def warm_state():
    sd = torch.load(WARM_CKPT, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    return {k: v for k, v in sd.items() if k.startswith(("aspp.", "head.", "center_bias"))}


def load_ids():
    return json.load(open(os.path.join(KB, "splits", "kfold_seed0.json")))["ids_sorted"]


def load_gt_train(path, g):          # published load_gt (training target)
    m = np.asarray(Image.open(path).convert("L").resize((g, g), Image.BILINEAR), np.float64)
    m = m - m.min(); s = m.sum()
    return m / s if s > 0 else m + 1.0 / m.size


def load_gt_eval(path):              # unified protocol GT (raw resized map; to_prob applied in metrics)
    return np.asarray(Image.open(path).convert("L").resize((EVAL, EVAL), Image.BILINEAR), np.float64)


def to_prob(x):
    x = np.clip(x, 0, None); s = x.sum()
    return x / s if s > 0 else np.full_like(x, 1.0 / x.size)


def cc(p, g):
    a, b = p - p.mean(), g - g.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def sim(p, g):
    return float(np.minimum(to_prob(p), to_prob(g)).sum())


def kld(p, g):
    p, g = to_prob(p), to_prob(g)
    return float((g * (np.log(g + EPS) - np.log(p + EPS))).sum())


class Data:
    """All features on the training device (fp16, 1.5 GB) + GT maps, indexed by position in ids_sorted."""
    def __init__(self, dev):
        self.ids = load_ids()
        self.pos = {i: n for n, i in enumerate(self.ids)}
        f = np.load(os.path.join(CACHE, "feats_dinov2b_448_fp16.npy"), mmap_mode="r")
        assert f.shape == (1000, 768, 32, 32)
        self.feats = torch.from_numpy(np.array(f)).to(dev)
        self.gtl = torch.stack([torch.tensor(load_gt_train(os.path.join(FLAT, "maps", f"{i}.png"), LOSSG), dtype=torch.float32)
                                for i in self.ids]).to(dev)
        self.gte = [load_gt_eval(os.path.join(FLAT, "maps", f"{i}.png")) for i in self.ids]
        self.dev = dev

    def batch(self, idx):
        ix = torch.as_tensor(np.asarray(idx), device=self.dev)
        return self.feats[ix].float(), self.gtl[ix]


@torch.no_grad()
def raw_preds(head, data, idx, bs=50):
    """sigmoid(bilinear-upsampled readout) at 384, float64 numpy, one per index (as raw_pred in the scripts)."""
    head.eval(); out = []
    idx = list(idx)
    for j in range(0, len(idx), bs):
        f, _ = data.batch(idx[j:j + bs])
        s = head(f)
        pr = torch.sigmoid(F.interpolate(s, size=(EVAL, EVAL), mode="bilinear", align_corners=False))[:, 0]
        out += [p.astype(np.float32) for p in pr.cpu().numpy()]
    return out   # float32, as raw_pred in the published scripts


def choose_sigma(preds, gts):
    best_sigma, best_cc = 0, -2
    for s in SIGMAS:
        m = float(np.mean([cc(gaussian_filter(p, s) if s > 0 else p, g) for p, g in zip(preds, gts)]))
        if m > best_cc:
            best_cc, best_sigma = m, s
    return best_sigma


def select_sigma_subset(head, data, tr, rng, allow=True):
    """kfold_train_eval.select_sigma: random 200-image subset of the training fold (same rng draw)."""
    if not allow:
        return 0
    sel = tr if len(tr) <= SIGMA_SELECT_N else rng.choice(tr, size=SIGMA_SELECT_N, replace=False)
    return choose_sigma(raw_preds(head, data, sel), [data.gte[i] for i in sel])


def score(head, data, te, sigma):
    rows = []
    for i, p in zip(te, raw_preds(head, data, te)):
        p = p.astype(np.float64)          # evaluate() in the published scripts works in float64
        if sigma > 0:
            p = gaussian_filter(p, sigma)
        g = data.gte[i]
        rows.append({"id": data.ids[i], "CC": cc(p, g), "SIM": sim(p, g), "KLD": kld(p, g)})
    return rows


def train(head, data, tr, rng, epochs=60, lr=3e-4, bs=16, use_cc=True, skip_singleton=False):
    """skip_singleton: skip a trailing batch of exactly 1 image (BatchNorm cannot train on it).
    Only needed when len(tr) % bs == 1 (grouped fold 3: 801 training ids); never triggers for
    the 800-id seed-0 / pair-safe folds, so those rows are unaffected."""
    params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    tr = np.asarray(tr)
    for ep in range(epochs):
        head.train()
        tr_shuf = tr.copy(); rng.shuffle(tr_shuf)
        for j in range(0, len(tr_shuf), bs):
            ids = tr_shuf[j:j + bs]
            if skip_singleton and len(ids) == 1:
                continue
            f, g = data.batch(ids)
            s = head(f)
            s = F.interpolate(s, size=(LOSSG, LOSSG), mode="bilinear", align_corners=False)[:, 0]
            logp = F.log_softmax(s.reshape(len(ids), -1), 1).reshape(len(ids), LOSSG, LOSSG)
            kl = (g * (torch.log(g + 1e-12) - logp)).sum((1, 2)).mean()
            loss = kl
            if use_cc:
                ps = torch.sigmoid(s)
                pc = ps - ps.mean((1, 2), keepdim=True); gc = g - g.mean((1, 2), keepdim=True)
                ccb = (pc * gc).sum((1, 2)) / (torch.sqrt((pc ** 2).sum((1, 2)) * (gc ** 2).sum((1, 2))) + 1e-8)
                loss = kl + (1 - ccb.mean())
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()


def fold_stats(per_image, folds_n):
    out = {"per_fold": []}
    for k in range(folds_n):
        r = [x for x in per_image if x["fold"] == k]
        out["per_fold"].append({"fold": k, "n": len(r), **{m: float(np.mean([x[m] for x in r])) for m in ("CC", "SIM", "KLD")}})
    for m in ("CC", "SIM", "KLD"):
        v = [f[m] for f in out["per_fold"]]
        out[f"mean_{m}"] = float(np.mean(v))
        out[f"sd_{m}_ddof0"] = float(np.std(v))
        out[f"sd_{m}_ddof1"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        out[f"pooled_image_mean_{m}"] = float(np.mean([x[m] for x in per_image]))
    return out
