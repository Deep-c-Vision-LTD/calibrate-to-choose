"""
Shared helpers for the DeepGaze IIE rows of the KBS letter.

* Fixed ADD1000 split (splits/kfold_seed0.json), unified 384 x 384 scoring
  (GT = map resized bilinearly to 384; probability maps by clip at 0 and divide by sum,
  exactly as eval_deepgaze_iie.py).
* Cached-feature forward pass for DeepGaze IIE. DeepGazeIIIMixture interpolates every
  backbone feature map to the readout grid (12 x 12 for a 384 x 384 input) BEFORE the
  readouts, so caching the concatenated, already-interpolated features per backbone is
  exact. forward_cached() re-uses the model's own readout / finalizer modules, so the
  trainable parameters and their state dict are those of deepgaze_pytorch.DeepGazeIIE.
"""
import json, math, os
import numpy as np
from PIL import Image

ROOT = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root (CBYC_ROOT overrides)
ADD1000_DIR = os.environ.get("ADD1000_DIR", os.path.join(ROOT, "data", "ADD1000"))   # holds flat/{images,maps}
KBS = ROOT
ADD = os.path.join(ADD1000_DIR, "flat")
SPLIT = os.path.join(KBS, "splits", "kfold_seed0.json")
CACHE = os.path.join(KBS, "cache", "iie")
RESULTS = os.path.join(KBS, "results")

S = 384                       # unified scoring grid and IIE input size
G = 12                        # IIE readout grid at S = 384 (ceil(384 / 2 / 16))
LOSSG = 64                    # loss grid of our recipe (kfold_train_eval.py)
EPS = 1e-8
SIGMAS = [0, 2, 4, 6, 8, 10, 12]   # finalizer grid of kfold_train_eval.py
SIGMA_SELECT_N = 200
BACKBONE_CH = [2048, 2416, 2048, 2560]


def load_split(path=SPLIT):
    d = json.load(open(path))
    return d["ids_sorted"], [list(f) for f in d["folds"]]


def fold_of(folds):
    return {i: k for k, f in enumerate(folds) for i in f}


# ---------------- metrics (identical to eval_deepgaze_iie.py) ----------------
def to_prob(x):
    x = np.clip(x, 0, None); s = x.sum()
    return x / s if s > 0 else np.full_like(x, 1.0 / x.size)

def cc(p, g):
    a, b = p - p.mean(), g - g.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum()); return float((a * b).sum() / d) if d > 0 else 0.0

def sim(p, g):
    p, g = to_prob(p), to_prob(g); return float(np.minimum(p, g).sum())

def kld(p, g):
    p, g = to_prob(p), to_prob(g); return float((g * np.log(g + EPS) - g * np.log(p + EPS)).sum())

def metrics(p, g):
    return {"CC": cc(p, g), "SIM": sim(p, g), "KLD": kld(p, g)}


# ---------------- data ----------------
def img_path(i):
    return os.path.join(ADD, "images", f"{i}.jpg")

def map_path(i):
    return os.path.join(ADD, "maps", f"{i}.png")

def load_img_384(i):
    im = Image.open(img_path(i)).convert("RGB").resize((S, S), Image.BILINEAR)
    return np.asarray(im, np.float32).transpose(2, 0, 1)          # (3,S,S), 0..255

def load_map_raw(i, g=S):
    """GT resized bilinearly to g x g, float64, no other processing."""
    return np.asarray(Image.open(map_path(i)).convert("L").resize((g, g), Image.BILINEAR), np.float64)

def load_map_minsub(i, g=S):
    """Training target used by ft_deepgaze_iie_indomain.py and kfold_train_eval.py."""
    m = load_map_raw(i, g); m = m - m.min(); s = m.sum()
    return m / s if s > 0 else m + 1.0 / m.size


def gt_cache():
    """Return dict with eval GT (clip-only prob at 384), min-sub targets at 384 and 64."""
    path = os.path.join(CACHE, "gt_cache.npz")
    ids, _ = load_split()
    if os.path.exists(path):
        z = np.load(path)
        assert list(z["ids"]) == ids
        return {"ids": ids, "raw384": z["raw384"], "ms384": z["ms384"], "ms64": z["ms64"]}
    raw = np.stack([load_map_raw(i) for i in ids]).astype(np.float32)
    ms = np.stack([load_map_minsub(i) for i in ids]).astype(np.float32)
    ms64 = np.stack([load_map_minsub(i, LOSSG) for i in ids]).astype(np.float32)
    np.savez(path, ids=np.array(ids), raw384=raw, ms384=ms, ms64=ms64)
    return {"ids": ids, "raw384": raw, "ms384": ms, "ms64": ms64}


# ---------------- IIE cached forward ----------------
def iie_features(model, x):
    """x: (B,3,S,S) float32 0..255 -> list of 4 tensors (B,C_k,G,G) = readout_input per backbone.
    Feature step of DeepGazeIIIMixture.forward; adapted from deepgaze_pytorch (github.com/matthias-k/DeepGaze)."""
    import torch, torch.nn.functional as F
    out = []
    for m in model.models:
        h = F.interpolate(x, scale_factor=1 / m.downsample, recompute_scale_factor=False)
        feats = m.features(h)
        rs = [math.ceil(x.shape[2] / m.downsample / m.readout_factor),
              math.ceil(x.shape[3] / m.downsample / m.readout_factor)]
        feats = [F.interpolate(f, rs) for f in feats]
        out.append(torch.cat(feats, dim=1))
    return out


def forward_cached(model, feats, centerbias):
    """Exact replica of MixtureModel(DeepGazeIIIMixture) forward after the feature step; adapted from
    deepgaze_pytorch (github.com/matthias-k/DeepGaze).
    feats: list of 4 (B,C_k,G,G); centerbias: (B,S,S) log-density. Returns (B,1,S,S) log density."""
    import torch
    preds = []
    for m, x in zip(model.models, feats):
        comp = []
        for sal, fsn, fin in zip(m.saliency_networks, m.fixation_selection_networks, m.finalizers):
            h = sal(x)
            h = fsn((h, None))
            h = fin(h, centerbias)
            comp.append(h[:, None])
        p = torch.cat(comp, dim=1) - np.log(len(m.saliency_networks))
        preds.append(p.logsumexp(dim=1, keepdim=True))
    preds = torch.cat(preds, dim=1) - np.log(len(model.models))
    return preds.logsumexp(dim=1, keepdim=True)


def load_feature_cache(dtype="float16"):
    """Memory-mapped (N, 9072, G, G) cache in ids_sorted order."""
    path = os.path.join(CACHE, f"iie384_feats_{dtype}.npy")
    return np.load(path, mmap_mode="r")


def split_channels(arr):
    """(B,9072,G,G) -> list of 4 per-backbone arrays/tensors."""
    out, c0 = [], 0
    for c in BACKBONE_CH:
        out.append(arr[:, c0:c0 + c]); c0 += c
    return out


def centerbias_from_maps(maps_prob):
    """Mean of probability maps -> log density (eval_deepgaze_iie.py fit_centerbias form)."""
    acc = np.mean(maps_prob, axis=0, dtype=np.float64)
    return np.log(acc + EPS)


def trainable_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def fold_summary(per_image, folds_used=None, keys=("CC", "SIM", "KLD")):
    by = {}
    for r in per_image:
        by.setdefault(r["fold"], []).append(r)
    pf = []
    for k in sorted(by):
        rows = by[k]
        pf.append({"fold": k, "n": len(rows), **{m: float(np.mean([r[m] for r in rows])) for m in keys}})
    out = {"per_fold": pf}
    for m in keys:
        v = [r[m] for r in pf]
        out[f"{m}_mean"] = float(np.mean(v))
        out[f"{m}_sd_over_folds"] = float(np.std(v, ddof=0))
        out[f"{m}_sd_over_folds_ddof1"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        out[f"{m}_mean_over_images"] = float(np.mean([r[m] for r in per_image]))
    return out


# ---------------- fast batched readout (same parameters, same maths) ----------------
class FastIIE:
    """Batched re-implementation (adapted from deepgaze_pytorch, github.com/matthias-k/DeepGaze) of the
    DeepGaze IIE readouts + finalizers + mixture that
    operates on the model's OWN parameter tensors (stacked at every call, so autograd and
    the optimiser still act on the original nn.Parameters). Mathematically identical to
    forward_cached(); only floating-point summation order differs (checked by
    verify_fast.py). Speed-ups:
      * the 30 component readouts of a backbone run as one batched einsum;
      * the finalizer blur of a 16x nearest-upsampled 12x12 readout is written as
        M R M^T with M = (replicate-padded Gaussian, truncate=3) x (nearest upsample);
      * the final 2x nearest upsample commutes with normalisation / mixture, so the
        mixture is computed on the 192 grid and upsampled once (minus log 4)."""

    def __init__(self, model, device):
        import torch
        self.model = model
        self.dev = device
        self.K = [len(m.saliency_networks) for m in model.models]
        H = S // 2
        self.H = H
        self.up = 16   # 192 / 12

    @staticmethod
    def _ln(x, eps=1e-12):
        # layer norm over (ch,H,W) per leading index; x (..., ch, H, W)
        m = x.mean(dim=(-3, -2, -1), keepdim=True)
        v = ((x - m) ** 2).mean(dim=(-3, -2, -1), keepdim=True)
        return (x - m) / (v + eps).sqrt()

    def _stack(self, mods, getter):
        import torch
        return torch.stack([getter(m) for m in mods], 0)

    def _blur_matrix(self, sigma):
        """sigma (K,) -> M (K, 192, 12)."""
        import torch
        H, up = self.H, self.up
        h = torch.ceil(3.0 * sigma.detach()).to(torch.int64)          # truncate = 3
        hmax = int(h.max().item())
        t = torch.arange(-hmax, hmax + 1, device=sigma.device, dtype=sigma.dtype)   # (T,)
        mask = (t.abs()[None, :] <= h[:, None].to(sigma.dtype))
        ker = torch.exp(-0.5 * (t[None, :] / sigma[:, None]) ** 2) * mask
        ker = ker / ker.sum(1, keepdim=True)                            # (K,T)
        i = torch.arange(H, device=sigma.device)
        idx = torch.clamp(i[:, None] + torch.arange(-hmax, hmax + 1, device=sigma.device)[None, :], 0, H - 1) // up  # (H,T)
        K = sigma.shape[0]
        M = torch.zeros(K, H, H // up, device=sigma.device, dtype=sigma.dtype)
        M = M.scatter_add(2, idx[None].expand(K, H, -1), ker[:, None, :].expand(K, H, -1))
        return M

    def backbone_logd192(self, m, x, cbd):
        """x (B,C,G,G); cbd (B,192,192) -> (B,192,192) log density (normalised on 192 grid)."""
        import torch, torch.nn.functional as F
        sal, fsn, fin = m.saliency_networks, m.fixation_selection_networks, m.finalizers
        K = len(sal)
        # saliency network
        xh = self._ln(x)                                                    # (B,C,G,G)
        w0 = self._stack(sal, lambda s: s.layernorm0.weight)                # (K,C)
        b0 = self._stack(sal, lambda s: s.layernorm0.bias)                  # (K,C)
        W0 = self._stack(sal, lambda s: s.conv0.weight[:, :, 0, 0])         # (K,8,C)
        cb0 = self._stack(sal, lambda s: s.bias0.bias)                      # (K,8)
        z = torch.einsum("koc,bchw->bkohw", W0 * w0[:, None, :], xh)
        z = z + (torch.einsum("koc,kc->ko", W0, b0) + cb0)[None, :, :, None, None]
        z = F.softplus(z)
        for ln, conv, bias in (("layernorm1", "conv1", "bias1"), ("layernorm2", "conv2", "bias2")):
            lw = self._stack(sal, lambda s: getattr(s, ln).weight)          # (K,ci)
            lb = self._stack(sal, lambda s: getattr(s, ln).bias)
            Wc = self._stack(sal, lambda s: getattr(s, conv).weight[:, :, 0, 0])   # (K,co,ci)
            bc = self._stack(sal, lambda s: getattr(s, bias).bias)          # (K,co)
            z = self._ln(z) * lw[None, :, :, None, None] + lb[None, :, :, None, None]
            z = torch.einsum("koc,bkchw->bkohw", Wc, z) + bc[None, :, :, None, None]
            z = F.softplus(z)                                               # softplus1 / softplus3
        # fixation selection network
        lw = self._stack(fsn, lambda s: s.layernorm0.layernorm_part0.weight)    # (K,1)
        lb = self._stack(fsn, lambda s: s.layernorm0.layernorm_part0.bias)
        z = self._ln(z) * lw[None, :, :, None, None] + lb[None, :, :, None, None]
        Wc = self._stack(fsn, lambda s: s.conv0.conv_part0.weight[:, :, 0, 0])  # (K,128,1)
        bc = self._stack(fsn, lambda s: s.bias0.bias)
        z = F.softplus(torch.einsum("koc,bkchw->bkohw", Wc, z) + bc[None, :, :, None, None])
        lw = self._stack(fsn, lambda s: s.layernorm1.weight)
        lb = self._stack(fsn, lambda s: s.layernorm1.bias)
        z = self._ln(z) * lw[None, :, :, None, None] + lb[None, :, :, None, None]
        Wc = self._stack(fsn, lambda s: s.conv1.weight[:, :, 0, 0])
        bc = self._stack(fsn, lambda s: s.bias1.bias)
        z = F.softplus(torch.einsum("koc,bkchw->bkohw", Wc, z) + bc[None, :, :, None, None])
        Wc = self._stack(fsn, lambda s: s.conv2.weight[:, :, 0, 0])         # (K,1,16)
        r = torch.einsum("koc,bkchw->bkohw", Wc, z)[:, :, 0]                 # (B,K,G,G)
        # finalizer
        sigma = self._stack(fin, lambda f: f.gauss.sigma)                   # (K,)
        cw = self._stack(fin, lambda f: f.center_bias_weight[0])            # (K,)
        M = self._blur_matrix(sigma)                                        # (K,192,12)
        out = torch.matmul(M, torch.matmul(r, M.transpose(1, 2)))              # (B,K,192,192) = M r M^T
        out = out + cw[None, :, None, None] * cbd[:, None]
        out = out - out.logsumexp(dim=(2, 3), keepdim=True)                  # normalised on 192
        return (out - np.log(K)).logsumexp(dim=1)                            # (B,192,192)

    def __call__(self, feats, centerbias):
        """feats: list of 4 (B,C_k,G,G); centerbias (B,S,S). Returns (B,1,S,S) log density."""
        import torch
        cbd = centerbias[:, ::2, ::2]                                       # nearest, scale 1/2
        preds = torch.stack([self.backbone_logd192(m, x, cbd) for m, x in zip(self.model.models, feats)], 1)
        p192 = (preds - np.log(len(self.model.models))).logsumexp(dim=1)    # (B,192,192)
        p384 = p192.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2) - np.log(4.0)
        return p384[:, None]


def readout_to(model, dev):
    for m in model.models:
        for mod in (m.saliency_networks, m.fixation_selection_networks, m.finalizers):
            mod.to(dev)
    return model


def readout_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if ("saliency_networks" in k or "fixation_selection_networks" in k or "finalizers" in k)}


# ---------------- stacked-parameter readout (training) ----------------
# (name, getter for one component, kind) ; kind: s=saliency net, f=fixation selection net, z=finalizer
_SPEC = [
    ("ln0_w", lambda s, f, z: s.layernorm0.weight), ("ln0_b", lambda s, f, z: s.layernorm0.bias),
    ("W0", lambda s, f, z: s.conv0.weight), ("b0", lambda s, f, z: s.bias0.bias),
    ("ln1_w", lambda s, f, z: s.layernorm1.weight), ("ln1_b", lambda s, f, z: s.layernorm1.bias),
    ("W1", lambda s, f, z: s.conv1.weight), ("b1", lambda s, f, z: s.bias1.bias),
    ("ln2_w", lambda s, f, z: s.layernorm2.weight), ("ln2_b", lambda s, f, z: s.layernorm2.bias),
    ("W2", lambda s, f, z: s.conv2.weight), ("b2", lambda s, f, z: s.bias2.bias),
    ("fln0_w", lambda s, f, z: f.layernorm0.layernorm_part0.weight),
    ("fln0_b", lambda s, f, z: f.layernorm0.layernorm_part0.bias),
    ("fW0", lambda s, f, z: f.conv0.conv_part0.weight), ("fb0", lambda s, f, z: f.bias0.bias),
    ("fln1_w", lambda s, f, z: f.layernorm1.weight), ("fln1_b", lambda s, f, z: f.layernorm1.bias),
    ("fW1", lambda s, f, z: f.conv1.weight), ("fb1", lambda s, f, z: f.bias1.bias),
    ("fW2", lambda s, f, z: f.conv2.weight),
    ("sigma", lambda s, f, z: z.gauss.sigma), ("cbw", lambda s, f, z: z.center_bias_weight),
]


def _make_stacked_class():
    import torch, torch.nn as nn

    class StackedIIE(nn.Module):
        """DeepGaze IIE readouts (4 backbones x 30 components; adapted from deepgaze_pytorch,
        github.com/matthias-k/DeepGaze) held as stacked leaf
        parameters (one tensor per layer per backbone). Same parameters and maths as the
        released modules; write_back() copies them into a DeepGazeIIE instance, so the
        trained readout can be saved / evaluated with the original code path."""

        def __init__(self, model):
            super().__init__()
            self.n_back = len(model.models)
            self.K = [len(m.saliency_networks) for m in model.models]
            for b, m in enumerate(model.models):
                comps = list(zip(m.saliency_networks, m.fixation_selection_networks, m.finalizers))
                for name, get in _SPEC:
                    t = torch.stack([get(s, f, z).detach().clone().float() for s, f, z in comps], 0)
                    self.register_parameter(f"b{b}_{name}", nn.Parameter(t))

        def p(self, b, name):
            return getattr(self, f"b{b}_{name}")

        @torch.no_grad()
        def write_back(self, model):
            for b, m in enumerate(model.models):
                comps = list(zip(m.saliency_networks, m.fixation_selection_networks, m.finalizers))
                for name, get in _SPEC:
                    t = self.p(b, name)
                    for k, (s, f, z) in enumerate(comps):
                        dst = get(s, f, z)
                        dst.copy_(t[k].to(dst.device, dst.dtype).reshape(dst.shape))

        @staticmethod
        def _ln(x, eps=1e-12):
            m = x.mean(dim=(-3, -2, -1), keepdim=True)
            v = ((x - m) ** 2).mean(dim=(-3, -2, -1), keepdim=True)
            return (x - m) / (v + eps).sqrt()

        def _blur_matrix(self, sigma):
            H, up = S // 2, 16
            h = torch.ceil(3.0 * sigma.detach()).to(torch.int64)
            hmax = int(h.max().item())
            t = torch.arange(-hmax, hmax + 1, device=sigma.device, dtype=sigma.dtype)
            mask = (t.abs()[None, :] <= h[:, None].to(sigma.dtype))
            ker = torch.exp(-0.5 * (t[None, :] / sigma[:, None]) ** 2) * mask
            ker = ker / ker.sum(1, keepdim=True)
            i = torch.arange(H, device=sigma.device)
            idx = torch.clamp(i[:, None] + torch.arange(-hmax, hmax + 1, device=sigma.device)[None, :], 0, H - 1) // up
            K = sigma.shape[0]
            M = torch.zeros(K, H, H // up, device=sigma.device, dtype=sigma.dtype)
            return M.scatter_add(2, idx[None].expand(K, H, -1), ker[:, None, :].expand(K, H, -1))

        def backbone_logd192(self, b, x, cbd):
            import torch.nn.functional as F
            P = lambda n: self.p(b, n)
            K = self.K[b]
            E = lambda v: v[None, :, :, None, None]
            def conv(W, z):                       # W (K,co,ci,1,1), z (B,K,ci,h,w)
                return torch.einsum("koc,bkchw->bkohw", W[:, :, :, 0, 0], z)
            xh = self._ln(x)
            W0 = P("W0")[:, :, :, 0, 0]
            z = torch.einsum("koc,bchw->bkohw", W0 * P("ln0_w")[:, None, :], xh)
            z = F.softplus(z + E(torch.einsum("koc,kc->ko", W0, P("ln0_b")) + P("b0")))
            z = self._ln(z) * E(P("ln1_w")) + E(P("ln1_b"))
            z = F.softplus(conv(P("W1"), z) + E(P("b1")))
            z = self._ln(z) * E(P("ln2_w")) + E(P("ln2_b"))
            z = F.softplus(conv(P("W2"), z) + E(P("b2")))
            z = self._ln(z) * E(P("fln0_w")) + E(P("fln0_b"))
            z = F.softplus(conv(P("fW0"), z) + E(P("fb0")))
            z = self._ln(z) * E(P("fln1_w")) + E(P("fln1_b"))
            z = F.softplus(conv(P("fW1"), z) + E(P("fb1")))
            r = conv(P("fW2"), z)[:, :, 0]                                  # (B,K,G,G)
            M = self._blur_matrix(P("sigma"))
            out = torch.matmul(M, torch.matmul(r, M.transpose(1, 2)))         # (B,K,192,192)
            if FAST_FINAL:
                return _finalmix()(out, P("cbw")[:, 0], cbd)
            out = out + P("cbw")[None, :, 0, None, None] * cbd[:, None]
            out = out - out.logsumexp(dim=(2, 3), keepdim=True)
            return (out - np.log(K)).logsumexp(dim=1)

        def logd192(self, feats, centerbias):
            cbd = centerbias[:, ::2, ::2]
            preds = torch.stack([self.backbone_logd192(b, x, cbd) for b, x in enumerate(feats)], 1)
            return (preds - np.log(self.n_back)).logsumexp(dim=1) - np.log(4.0)   # per-384-pixel log density on 192 grid

        def forward(self, feats, centerbias):
            p = self.logd192(feats, centerbias)
            return p.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)[:, None]   # (B,1,S,S)

    return StackedIIE


FAST_FINAL = os.environ.get("IIE_FAST_FINAL", "0") == "1"
_FM = None


def _finalmix():
    """Custom autograd function for the finalizer tail + component mixture of one backbone:
    out = log( (1/K) sum_k softmax_ij(blur_k + w_k * cbd) ). Same maths as the logsumexp
    formulation; hand-written backward with far fewer passes over the (B,K,192,192) tensor:
    with A_k = softmax_ij(o_k), P = mean_k A_k and dP = g / P:
    do_k = A_k * (dP / K - <dP / K, A_k>),  dw_k = sum_{b,i,j} do_k * cbd."""
    global _FM
    if _FM is not None:
        return _FM
    import torch

    class FinalMix(torch.autograd.Function):
        @staticmethod
        def forward(ctx, blur, w, cbd):
            K = blur.shape[1]
            o = blur + w[None, :, None, None] * cbd[:, None]
            o = o - o.amax(dim=(2, 3), keepdim=True)
            A = torch.exp_(o)
            A = A.div_(A.sum(dim=(2, 3), keepdim=True))
            Pm = A.mean(dim=1).clamp_min(1e-38)
            ctx.save_for_backward(A, Pm, cbd)
            return torch.log(Pm)

        @staticmethod
        def backward(ctx, g):
            A, Pm, cbd = ctx.saved_tensors
            K = A.shape[1]
            dP = g / Pm / K                                             # (B,H,W)
            c = torch.einsum("bkij,bij->bk", A, dP)                     # (B,K)
            do = A * (dP[:, None] - c[:, :, None, None])
            dw = torch.einsum("bkij,bij->k", do, cbd)
            return do, dw, None

    _FM = FinalMix.apply
    return _FM


_STACKED = None
def StackedIIE(model):
    global _STACKED
    if _STACKED is None:
        _STACKED = _make_stacked_class()
    return _STACKED(model)
