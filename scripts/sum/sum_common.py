"""
Shared helpers for the SUM (Hosseini et al., WACV 2025) baseline rows on ADD1000.

* load_sum(): builds SUM from the authors' code directory and the released weights
  (config.json + model.safetensors) without installing the package, and replaces the
  CUDA-only selective-scan kernel with a pure-PyTorch implementation so the model runs
  on CPU / Apple MPS / any CUDA GPU without compiling mamba-ssm.
* The pure-PyTorch scan is a chunked two-level linear recurrence (sequential within
  chunks of length T ~ sqrt(L), sequential carry across chunks) wrapped in a custom
  autograd Function whose backward recomputes the states and runs the reverse
  recurrence, so training memory stays bounded. It is checked against the sequential
  reference recurrence (mamba-ssm's selective_scan_ref semantics) in test_scan.py.
* Metrics follow eval_deepgaze_iie.py / eval_sum_add1000.py exactly: CC, SIM, KLD on a
  common grid, probability maps by clipping at 0 and dividing by the sum, EPS = 1e-8.
"""
import importlib.util
import json
import math
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

EPS = 1e-8
S_EVAL = 384
S_IN = 256
IM_M = np.array([0.485, 0.456, 0.406], np.float32)
IM_S = np.array([0.229, 0.224, 0.225], np.float32)
SIGMAS = [0, 2, 4, 6, 8, 10, 12]        # identical to kfold_train_eval.py
SIGMA_SELECT_N = 200                     # identical to kfold_train_eval.py


# --------------------------------------------------------------------------------------
# selective scan in pure PyTorch
# --------------------------------------------------------------------------------------
def selective_scan_seq_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                           delta_softplus=False, return_last_state=False):
    """Sequential reference with mamba-ssm selective_scan_ref semantics (grouped B/C)."""
    u = u.float(); delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    b, d, L = u.shape
    g = B.shape[1]
    Bf = B.float().repeat_interleave(d // g, dim=1)      # (b, d, n, L)
    Cf = C.float().repeat_interleave(d // g, dim=1)
    dA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A.float())).unbind(2)
    dBu = torch.einsum("bdl,bdnl,bdl->bdln", delta, Bf, u).unbind(2)
    Cs = Cf.unbind(3)
    x = torch.zeros(b, d, A.shape[1], device=u.device, dtype=torch.float32)
    ys = []
    for i in range(L):
        x = dA[i] * x + dBu[i]
        ys.append((x * Cs[i]).sum(-1))
    y = torch.stack(ys, -1)
    if D is not None:
        y = y + u * D.float()[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    return y


def _chunk_len(L):
    return max(1, 2 ** int(round(math.log2(math.sqrt(L)))))


def _to_layout(x, T):
    """(b, K, M, L) -> (T, b, K, M, nc) contiguous, with L index = c * T + t."""
    b, K, M, L = x.shape
    return x.reshape(b, K, M, L // T, T).permute(4, 0, 1, 2, 3).contiguous()


def _from_layout(x):
    T, b, K, M, nc = x.shape
    return x.permute(1, 2, 3, 4, 0).reshape(b, K, M, nc * T)


def _scan_layout(a, bb):
    """x_t = a_t * x_{t-1} + b_t along flattened L (index c*T+t), x_{-1}=0.
    a, bb: (T, ..., nc). Returns X of the same shape. No autograd."""
    T = a.shape[0]
    x = bb[0]
    xs = [x]
    for t in range(1, T):
        x = torch.addcmul(bb[t], a[t], x)
        xs.append(x)
    X = torch.stack(xs, 0)
    del xs
    nc = a.shape[-1]
    if nc > 1:
        P = torch.cumprod(a.reshape(T, -1), 0).reshape(a.shape)   # 2-D: MPS scan supports <= 4 dims
        Pend = P[-1].unbind(-1)
        Xend = X[-1].unbind(-1)
        c = torch.zeros_like(Xend[0])
        cs = [c]
        for k in range(1, nc):
            c = torch.addcmul(Xend[k - 1], Pend[k - 1], c)
            cs.append(c)
        X.addcmul_(P, torch.stack(cs, -1).unsqueeze(0))
        del P
    return X


class _ChunkScan(torch.autograd.Function):
    """y_t[d] = sum_n C_t[g(d),n] x_t[d,n],  x_t = exp(dt_t A) x_{t-1} + dt_t u_t B_t[g(d)]."""

    @staticmethod
    def forward(ctx, u, dt, A, B, C):
        b, D, L = u.shape
        K, N = B.shape[1], B.shape[2]
        h = D // K
        T = _chunk_len(L)
        pad = (-L) % T
        ctx.save_for_backward(u, dt, A, B, C)
        ctx.shape = (b, D, L, K, N, h, T, pad)
        with torch.no_grad():
            X, _, _, CL, *_ = _ChunkScan._states(u, dt, A, B, C, ctx.shape)
            y = torch.einsum("tbkhnc,tbknc->tbkhc", X, CL[:, :, :, 0])
            y = _from_layout(y).reshape(b, D, L + pad)[..., :L]
        return y

    @staticmethod
    def _states(u, dt, A, B, C, shape):
        b, D, L, K, N, h, T, pad = shape
        if pad:
            u = F.pad(u, (0, pad)); dt = F.pad(dt, (0, pad))
            B = F.pad(B, (0, pad)); C = F.pad(C, (0, pad))
        dtL = _to_layout(dt.reshape(b, K, h, -1), T)            # (T,b,K,h,nc)
        dtuL = _to_layout((dt * u).reshape(b, K, h, -1), T)
        BL = _to_layout(B, T).unsqueeze(3)                       # (T,b,K,1,N,nc)
        CL = _to_layout(C, T).unsqueeze(3)
        AL = A.reshape(1, 1, K, h, N, 1)
        a = torch.exp(dtL.unsqueeze(4) * AL)                     # (T,b,K,h,N,nc)
        bb = dtuL.unsqueeze(4) * BL
        X = _scan_layout(a, bb)
        del bb
        return X, a, BL, CL, dtL, dtuL, AL

    @staticmethod
    def backward(ctx, gy):
        u, dt, A, B, C = ctx.saved_tensors
        b, D, L, K, N, h, T, pad = ctx.shape
        with torch.no_grad():
            X, a, BL, CL, dtL, dtuL, AL = _ChunkScan._states(u, dt, A, B, C, ctx.shape)
            gyp = F.pad(gy, (0, pad)) if pad else gy
            Lp = L + pad
            gyK = gyp.reshape(b, K, h, Lp)
            # reverse recurrence G_t = C_t*gy_t + a_{t+1} G_{t+1}, done as a forward scan in
            # reversed time: a'_s = a_{L-s} (next-step decay), r'_s = r_{L-1-s}
            dtp = F.pad(dt, (0, pad)) if pad else dt
            dt_next_rev = torch.flip(F.pad(dtp[..., 1:], (0, 1)), dims=[-1])
            a_rev = torch.exp(_to_layout(dt_next_rev.reshape(b, K, h, Lp), T).unsqueeze(4) * AL)
            gy_rev = _to_layout(torch.flip(gyK, dims=[-1]), T).unsqueeze(4)
            C_rev = _to_layout(torch.flip(C if not pad else F.pad(C, (0, pad)), dims=[-1]), T).unsqueeze(3)
            r_rev = gy_rev * C_rev
            del gy_rev, C_rev
            G = _scan_layout(a_rev, r_rev)
            del a_rev, r_rev
            G = torch.flip(G, dims=[0, G.dim() - 1]).contiguous()  # back to natural order
            # previous state x_{t-1}
            Xp = torch.empty_like(X)
            Xp[1:] = X[:-1]
            Xp[0, ..., 1:] = X[-1, ..., :-1]
            Xp[0, ..., 0] = 0
            gyL = _to_layout(gyK, T)                                # (T,b,K,h,nc)
            gC = torch.einsum("tbkhc,tbkhnc->tbknc", gyL, X)
            del X
            ga = G * Xp * a                                          # dL/d(dt*A)
            del Xp, a
            g_dt = torch.einsum("tbkhnc,khn->tbkhc", ga, AL.reshape(K, h, N))
            gA = torch.einsum("tbkhnc,tbkhc->khn", ga, dtL).reshape(D, N)
            del ga
            g_dtu = torch.einsum("tbkhnc,tbknc->tbkhc", G, BL[:, :, :, 0])
            gB = torch.einsum("tbkhnc,tbkhc->tbknc", G, dtuL)
            del G
            uL = _to_layout((F.pad(u, (0, pad)) if pad else u).reshape(b, K, h, Lp), T)
            g_dt = g_dt + g_dtu * uL
            g_u = g_dtu * dtL
            g_u = _from_layout(g_u).reshape(b, D, Lp)[..., :L]
            g_dt = _from_layout(g_dt).reshape(b, D, Lp)[..., :L]
            gB = _from_layout(gB)[..., :L]
            gC = _from_layout(gC)[..., :L]
        return g_u, g_dt, gA, gB, gC


def selective_scan_chunked(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                           delta_softplus=False, return_last_state=False):
    """Drop-in replacement for mamba_ssm selective_scan_fn (grouped B/C, z=None path)."""
    assert not return_last_state
    u = u.float(); delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    if B.dim() == 3:
        B = B.unsqueeze(1); C = C.unsqueeze(1)
    y = _ChunkScan.apply(u, delta, A.float(), B.float().contiguous(), C.float().contiguous())
    if D is not None:
        y = y + u * D.float()[None, :, None]
    if z is not None:
        y = y * F.silu(z.float())
    return y


# --------------------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------------------
def load_sum(code_dir, weights_dir, device="cpu", scan="chunked"):
    """Import SUM's model files from code_dir/net/models under a private package name
    (avoids running the package __init__ and name clashes), patch the scan, load weights."""
    models_dir = os.path.join(code_dir, "net", "models")
    pkg_name = "sum_thirdparty_models"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [models_dir]
        sys.modules[pkg_name] = pkg
        for mod in ("vmamba", "SUM"):
            spec = importlib.util.spec_from_file_location(f"{pkg_name}.{mod}",
                                                          os.path.join(models_dir, f"{mod}.py"))
            m = importlib.util.module_from_spec(spec)
            sys.modules[f"{pkg_name}.{mod}"] = m
            spec.loader.exec_module(m)
    vm = sys.modules[f"{pkg_name}.vmamba"]
    vm.selective_scan_fn = selective_scan_chunked if scan == "chunked" else selective_scan_seq_ref
    SUMcls = sys.modules[f"{pkg_name}.SUM"].SUM
    cfg = json.load(open(os.path.join(weights_dir, "config.json")))
    model = SUMcls(**cfg)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(weights_dir, "model.safetensors"))
    model.load_state_dict(sd, strict=True)
    return model.to(device), cfg


def set_scan(scan):
    vm = sys.modules["sum_thirdparty_models.vmamba"]
    vm.selective_scan_fn = selective_scan_chunked if scan == "chunked" else selective_scan_seq_ref


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_split(path):
    d = json.load(open(path))
    folds = [list(map(str, f)) for f in d["folds"]]
    return folds


def train_ids_for(folds, k):
    return [i for j, f in enumerate(folds) if j != k for i in f]


def load_image_tensor(path):
    """Same as SUM's load_and_preprocess_image: torchvision Resize((256,256)) (PIL bilinear
    with antialias), ToTensor, ImageNet normalisation."""
    import torchvision.transforms as tvt
    tf = tvt.Compose([tvt.Resize((S_IN, S_IN)), tvt.ToTensor(),
                      tvt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return tf(Image.open(path).convert("RGB"))


def load_gt(path, size):
    """GT density resized bilinearly (PIL) to size x size, float64, NOT renormalised
    (metrics normalise internally), exactly as eval_sum_add1000.py."""
    return np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), np.float64)


def resize_map(pred, size):
    """PIL mode-F bilinear resize, exactly as eval_sum_add1000.py."""
    return np.asarray(Image.fromarray(np.asarray(pred, np.float32)).resize((size, size), Image.BILINEAR),
                      np.float64)


# --------------------------------------------------------------------------------------
# metrics (eval_deepgaze_iie.py / eval_sum_add1000.py)
# --------------------------------------------------------------------------------------
def to_prob(x):
    x = np.clip(x, 0, None); s = x.sum()
    return x / s if s > 0 else np.full_like(x, 1.0 / x.size)


def cc(p, g):
    a, b = p - p.mean(), g - g.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def sim(p, g):
    p, g = to_prob(p), to_prob(g)
    return float(np.minimum(p, g).sum())


def kld(p, g):
    p, g = to_prob(p), to_prob(g)
    return float((g * np.log(g + EPS) - g * np.log(p + EPS)).sum())


def metrics(p, g):
    return {"CC": cc(p, g), "SIM": sim(p, g), "KLD": kld(p, g)}


def fold_summary(per_image, folds, keys=("CC", "SIM", "KLD")):
    """per-fold means, then mean and SD (population SD over the 5 fold means, as in
    kfold_train_eval.py's np.std) plus sample SD for reference."""
    out = {"per_fold": []}
    for k in range(len(folds)):
        rows = [r for r in per_image if r["fold"] == k]
        out["per_fold"].append({"fold": k, "n": len(rows),
                                **{m: float(np.mean([r[m] for r in rows])) for m in keys}})
    for m in keys:
        v = [f[m] for f in out["per_fold"]]
        out[f"{m}_mean"] = float(np.mean(v))
        out[f"{m}_sd_folds"] = float(np.std(v))
        out[f"{m}_sd_folds_ddof1"] = float(np.std(v, ddof=1))
        out[f"{m}_mean_over_images"] = float(np.mean([r[m] for r in per_image]))
    return out


# --------------------------------------------------------------------------------------
# learning-curve subsets (rule shared with ours/run_curves.py, msdb/train_msdb.py --curves
# and iie/ft_curves.py, i.e. sample_efficiency.py). Used only by ft_sum_folds.py --curve.
# --------------------------------------------------------------------------------------
def parse_curve(spec):
    """'25:0,1,2/50:0,1,2/400:0' (groups 'n:r,r,...' separated by '/' or ';') ->
    [(25, 0), (25, 1), (25, 2), (50, 0), ..., (400, 0)] in the given order. '/' is the
    separator to use in cloud hyperparameters (no shell metacharacters)."""
    runs = []
    for grp in spec.replace(";", "/").split("/"):
        grp = grp.strip()
        if not grp:
            continue
        n, rs = grp.split(":")
        runs += [(int(n), int(r)) for r in rs.split(",") if r.strip()]
    if not runs or len(set(runs)) != len(runs):
        raise ValueError(f"bad curve spec {spec!r}")
    return runs


def curve_subset(folds, test_fold, n, r):
    """pool = the other folds concatenated in fold order (for test fold 0 this equals
    default_rng(0).permutation(1000)[200:] mapped through ids_sorted); training ids =
    default_rng(1000 n + r).choice(pool, n, replace=False), kept in draw order."""
    pool = train_ids_for(folds, test_fold)
    return [str(x) for x in np.random.default_rng(1000 * n + r).choice(np.array(pool), size=n, replace=False)]


def curve_orders(train_ids, epochs, r):
    """Data order: default_rng(r); a fresh copy of the training list (draw order) is shuffled
    every epoch (run_curves.py / ft_curves.py seeded_orders / train_msdb.py)."""
    rng = np.random.default_rng(r)
    out = []
    for _ in range(epochs):
        o = list(train_ids)
        rng.shuffle(o)
        out.append(o)
    return out


def orders_sha1(orders):
    import hashlib
    return hashlib.sha1(json.dumps(orders).encode()).hexdigest()
