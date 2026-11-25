"""
Native-resolution variant of the adapted DeepGaze IIE row (revision, 24 Nov 2026).

Everything is the selected 384 recipe (row iie_ft_klcc_e20, scripts/iie/iie_train.py + ft_kfold.py)
except the INPUT SIZE: each ADD1000 image enters DeepGaze IIE at its original pixel size (PIL RGB, no
resize; the model applies its own nearest 1/2 downsampling before the backbones, exactly as in
scripts/revision/iie_native_zeroshot.py and the released DeepGazeIIIMixture.forward).

Pieces
  * NativeFeats      : frozen-backbone readout inputs per image (4 backbones, 9072 channels on the
                       ceil(H/32) x ceil(W/32) readout grid, float32), one flat memmap + index.
  * native_centerbias: the row's 384 fold prior zoomed to H x W with nearest (order 0) interpolation and
                       logsumexp-renormalised (the DeepGaze README rule used by iie_native_zeroshot.py).
  * Geometry         : exact index maps of the released Finalizer for an H x W input (nearest readout
                       upsample to floor(H/2) x floor(W/2), nearest centre-bias 1/2 downsample, nearest
                       upsample back to H x W), plus the linear map from the H x W density to the 64 x 64
                       loss grid = torch bilinear resize to 384 (align_corners=False, the scorer's resize)
                       followed by the exact 6 x 6 sum-pool of the 384 recipe.
  * NativeStackedIIE : the 384 row's StackedIIE (same 23 stacked parameter tensors per backbone,
                       3,059,160 trainable values) with a size-general finaliser: blur as M_y r M_x^T,
                       per-component softmax normalised over the H x W map via row/column multiplicities
                       of the final nearest upsample, and a custom backward (WFinalMix) that reduces to
                       the 384 row's FinalMix when all multiplicities are equal.
  * klcc64_native    : the 384 recipe's loss KL(g64||p64) + (1 - CC(p64, g64)) for one image.
  * predict_native   : evaluation through the RELEASED module code path (iie_common.forward_cached on the
                       native readout inputs and native centre bias), density resized to 384 x 384 with torch
                       bilinear (align_corners=False, float64) exactly as iie_native_zeroshot.py.

Geometry and NativeStackedIIE are adapted from deepgaze_pytorch (github.com/matthias-k/DeepGaze).
"""
import os, sys, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "iie"))
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
import iie_common as C

NCACHE = os.path.join(C.KBS, "cache", "iie_native")
FEAT_BIN = os.path.join(NCACHE, "native_feats_float32.bin")
INDEX = os.path.join(NCACHE, "native_feats_index.json")
CTOT = sum(C.BACKBONE_CH)          # 9072
ROW = "iie_native_adapt"


# ------------------------------------------------------------------ inputs
def image_size(i):
    with Image.open(C.img_path(i)) as im:
        return im.size                         # (W, H)


def load_img_native(i):
    """(3,H,W) float32 0..255, PIL RGB, no resize (as iie_native_zeroshot.py)."""
    im = Image.open(C.img_path(i)).convert("RGB")
    return np.asarray(im, np.float32).transpose(2, 0, 1)


def readout_hw(H, W):
    return math.ceil(H / 2 / 16), math.ceil(W / 2 / 16)


class NativeFeats:
    """Read-only access to the flat native readout-input cache (written by cache_native.py)."""

    def __init__(self, require_complete=True):
        idx = json.load(open(INDEX))
        self.meta = {r["id"]: r for r in idx["images"]}
        if require_complete:
            assert idx.get("complete"), "native feature cache incomplete"
        self.arr = np.memmap(FEAT_BIN, dtype=np.float32, mode="r", shape=(idx["n_floats"],))

    def hw(self, i):
        r = self.meta[i]
        return r["H"], r["W"]

    def get(self, i):
        r = self.meta[i]
        n = CTOT * r["h"] * r["w"]
        x = np.array(self.arr[r["offset"]:r["offset"] + n]).reshape(1, CTOT, r["h"], r["w"])
        return torch.from_numpy(x)


def native_centerbias(prior384, H, W):
    """384 log prior -> H x W log density: scipy zoom order 0 (mode nearest) + logsumexp renormalisation,
    the rule iie_native_zeroshot.py applies to the MIT1003 template. Returns float32 tensor (1,H,W)."""
    cb = zoom(prior384, (H / prior384.shape[0], W / prior384.shape[1]), order=0, mode="nearest")
    assert cb.shape == (H, W), (cb.shape, H, W)
    cb = cb - logsumexp(cb)
    return torch.from_numpy(cb.astype(np.float32))[None]


# ------------------------------------------------------------------ geometry of the released finaliser
def nearest_src(n_in, n_out):
    """Index that torch's nearest F.interpolate(size=...) reads for every output index (taken from torch itself)."""
    t = torch.arange(n_in, dtype=torch.float32).view(1, 1, n_in, 1)
    return F.interpolate(t, size=(n_out, 1)).view(-1).round().long()


def bilinear_matrix(n_in, n_out=C.S):
    """(n_out, n_in) matrix of torch bilinear resize, align_corners=False, no antialias (taken from torch)."""
    eye = torch.eye(n_in, dtype=torch.float64)[None, None]
    return F.interpolate(eye, size=(n_out, n_in), mode="bilinear", align_corners=False)[0, 0]


class Geometry:
    def __init__(self, H, W, dtype=torch.float32):
        self.H, self.W = H, W
        self.h, self.w = readout_hw(H, W)
        probe = F.interpolate(torch.zeros(1, 1, H, W), scale_factor=0.5, recompute_scale_factor=False)
        self.H2, self.W2 = int(probe.shape[2]), int(probe.shape[3])
        self.sy, self.sx = nearest_src(self.h, self.H2), nearest_src(self.w, self.W2)      # readout -> finaliser grid
        self.uy, self.ux = nearest_src(self.H2, H), nearest_src(self.W2, W)                # finaliser grid -> H x W
        self.cy = torch.bincount(self.uy, minlength=self.H2).to(dtype)
        self.cx = torch.bincount(self.ux, minlength=self.W2).to(dtype)
        assert int(self.cy.min()) >= 1 and int(self.cx.min()) >= 1
        self.wt = (self.cy[:, None] * self.cx[None, :]).contiguous()                        # multiplicity per cell
        self.Ly = self._loss_map(H, self.uy, self.H2).to(dtype)
        self.Lx = self._loss_map(W, self.ux, self.W2).to(dtype)

    @staticmethod
    def _loss_map(n, u, n2):
        """(64, n2): 6-row block sums (64 grid) of the bilinear-384 resize of the nearest-expanded density."""
        R = bilinear_matrix(n)                                   # (384, n)
        E = torch.zeros(n, n2, dtype=torch.float64); E[torch.arange(n), u] = 1.0
        Q = torch.zeros(C.LOSSG, C.S, dtype=torch.float64)
        f = C.S // C.LOSSG
        for a in range(C.LOSSG):
            Q[a, a * f:(a + 1) * f] = 1.0
        return Q @ R @ E

    def dcb(self, cb):
        """released Finalizer's centre-bias downsampling (nearest, scale 1/2) -> (1,H2,W2)."""
        return F.interpolate(cb.view(1, 1, self.H, self.W), scale_factor=0.5, recompute_scale_factor=False)[:, 0]


# ------------------------------------------------------------------ size-general stacked readout
_WFM = None


def wfinalmix():
    """log( (1/K) sum_k A_k ), A_k = exp(o_k) / sum_ij wt_ij exp(o_k)_ij, o_k = blur_k + w_k * cbd.
    A_k is the per-pixel density of component k on the H x W map (each finaliser cell repeated wt_ij times).
    Backward: with G = g / (K P), do_k = A_k * (G - wt * <G, A_k>), dw_k = sum do_k * cbd. With wt == const
    this is exactly iie_common's FinalMix (up to the constant, which cancels)."""
    global _WFM
    if _WFM is not None:
        return _WFM

    class WFinalMix(torch.autograd.Function):
        @staticmethod
        def forward(ctx, blur, w, cbd, wt):
            o = blur + w[None, :, None, None] * cbd[:, None]
            o = o - o.amax(dim=(2, 3), keepdim=True)
            A = torch.exp_(o)
            A = A.div_((A * wt).sum(dim=(2, 3), keepdim=True))
            Pm = A.mean(dim=1).clamp_min(1e-38)
            ctx.save_for_backward(A, Pm, cbd, wt)
            return torch.log(Pm)

        @staticmethod
        def backward(ctx, g):
            A, Pm, cbd, wt = ctx.saved_tensors
            K = A.shape[1]
            G = g / Pm / K
            c = torch.einsum("bkij,bij->bk", A, G)
            do = A * (G[:, None] - wt[None, None] * c[:, :, None, None])
            dw = torch.einsum("bkij,bij->k", do, cbd)
            return do, dw, None, None

    _WFM = WFinalMix.apply
    return _WFM


def _base_class():
    if C._STACKED is None:
        C._STACKED = C._make_stacked_class()
    return C._STACKED


def make_native_stacked(model, plain_autograd=False):
    Base = _base_class()

    class NativeStackedIIE(Base):
        """Same parameters as iie_common.StackedIIE; forward for one image of any size."""

        def _readout(self, b, x):
            """verbatim copy of StackedIIE.backbone_logd192 up to the readout r (B,K,h,w)."""
            P = lambda n: self.p(b, n)
            E = lambda v: v[None, :, :, None, None]
            def conv(W, z):
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
            return conv(P("fW2"), z)[:, :, 0]

        @staticmethod
        def _blur_native(sigma, n2, src, n_r):
            """(K, n2, n_r): replicate-padded Gaussian (truncate 3, as GaussianFilterNd) applied after the
            nearest upsample readout -> finaliser grid given by src (n2,). Same construction as
            StackedIIE._blur_matrix with //16 replaced by the exact torch index map."""
            h = torch.ceil(3.0 * sigma.detach()).to(torch.int64)
            hmax = int(h.max().item())
            t = torch.arange(-hmax, hmax + 1, device=sigma.device, dtype=sigma.dtype)
            mask = (t.abs()[None, :] <= h[:, None].to(sigma.dtype))
            ker = torch.exp(-0.5 * (t[None, :] / sigma[:, None]) ** 2) * mask
            ker = ker / ker.sum(1, keepdim=True)
            i = torch.arange(n2, device=sigma.device)
            idx = src[torch.clamp(i[:, None] + torch.arange(-hmax, hmax + 1, device=sigma.device)[None, :], 0, n2 - 1)]
            K = sigma.shape[0]
            M = torch.zeros(K, n2, n_r, device=sigma.device, dtype=sigma.dtype)
            return M.scatter_add(2, idx[None].expand(K, n2, -1), ker[:, None, :].expand(K, n2, -1))

        def backbone_logP(self, b, x, cbd, geo):
            r = self._readout(b, x)                                          # (1,K,h,w)
            sig = self.p(b, "sigma")
            My = self._blur_native(sig, geo.H2, geo.sy, geo.h)               # (K,H2,h)
            Mx = self._blur_native(sig, geo.W2, geo.sx, geo.w)               # (K,W2,w)
            out = torch.matmul(My, torch.matmul(r, Mx.transpose(1, 2)))      # (1,K,H2,W2)
            cbw = self.p(b, "cbw")[:, 0]
            if plain_autograd:
                o = out + cbw[None, :, None, None] * cbd[:, None]
                o = o - (o + torch.log(geo.wt)[None, None]).logsumexp(dim=(2, 3), keepdim=True)
                return (o - np.log(o.shape[1])).logsumexp(dim=1)
            return wfinalmix()(out, cbw, cbd, geo.wt)

        def logP(self, feats, cbd, geo):
            """per-pixel log density of the H x W prediction, on the finaliser grid (1,H2,W2);
            full map = logP[uy][:, ux]."""
            preds = torch.stack([self.backbone_logP(b, x, cbd, geo) for b, x in enumerate(feats)], 1)
            return (preds - np.log(self.n_back)).logsumexp(dim=1)

        def full_logd(self, feats, cbd, geo):
            lp = self.logP(feats, cbd, geo)
            return lp[:, geo.uy][:, :, geo.ux]                              # (1,H,W)

    return NativeStackedIIE(model)


def klcc64_native(logP, geo, g64):
    """Loss of the 384 recipe for one image. p64 = 6x6 block sums of the bilinear-384 resize of the H x W
    density, renormalised (identical to the 384 recipe when H = W = 384, where the resize is the identity)."""
    q = geo.Ly @ torch.exp(logP[0]) @ geo.Lx.T
    q = q / q.sum()
    kl = (g64 * (torch.log(g64 + 1e-12) - torch.log(q.clamp_min(1e-30)))).sum()
    pc = q - q.mean(); gc = g64 - g64.mean()
    ccv = (pc * gc).sum() / (torch.sqrt((pc ** 2).sum() * (gc ** 2).sum()) + 1e-8)
    return kl + (1 - ccv)


# ------------------------------------------------------------------ evaluation (released module code path)
@torch.no_grad()
def predict_native(model, feats_of, ids, prior384, hw_of):
    """-> {id: 384 x 384 float64 density}. feats_of(i) -> (1,9072,h,w); hw_of(i) -> (H, W)."""
    out = {}
    for i in ids:
        H, W = hw_of(i)
        cb = native_centerbias(prior384, H, W)
        logd = C.forward_cached(model, C.split_channels(feats_of(i)), cb)        # (1,1,H,W)
        d = torch.exp(logd[:, 0].double())
        out[i] = F.interpolate(d[None], size=(C.S, C.S), mode="bilinear", align_corners=False)[0, 0].numpy()
    return out
