"""
Shared helpers for the DeepGaze MSDB rows of the KBS short communication.

Protocol (fixed for every row):
  * folds  : splits/kfold_seed0.json (5 x 200 ids, numpy default_rng(0) permutation of the
             numerically sorted ids; identical to kfold_train_eval.py).
  * model input: image resized to 512 x 512 (PIL bilinear), pixel_per_dva = 21.7, exactly as
             eval_deepgaze_msdb.py (the published zero-shot protocol).
  * scoring: CC / SIM / KLD on a 384 x 384 grid. GT = map resized bilinearly (PIL) to 384,
             probability maps by clip at 0 and divide by sum (eval_deepgaze_iie.py).
             MSDB's native output is 512 x 512, so the predicted density is resized to 384
             (torch bilinear, align_corners=False) before scoring; the native 512 score is
             logged as well.

MSDB head on cached features. DeepGazeMSDB = frozen multi-scale CLIP-RN50x64 + DINOv2-B/14
backbone -> scale mix (10 scales, softmax-normalised exp weights) -> saliency network
(LayerNorm/1x1-conv/Softplus stack, 26,460 params) -> finalizer (Gaussian blur with learned
sigma, priority scaling, centre-bias weight, log-normalisation). Because the backbone is
frozen and the scale-mix weights are fixed in the generalisation mode, the 64 x 64 x 2560
scale mix is a sufficient input for the whole trainable head; we cache it once (fp16) and
re-implement the finalizer exactly (the 512 output is a nearest 2x upsample of the 256 map).
"""
import json, math, os, time, signal, atexit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root (CBYC_ROOT overrides)
ADD1000_DIR = os.environ.get("ADD1000_DIR", os.path.join(ROOT, "data", "ADD1000"))   # holds flat/{images,maps}
KBS = ROOT
ADD = os.path.join(ADD1000_DIR, "flat")
SPLIT = os.path.join(KBS, "splits", "kfold_seed0.json")
CACHE = os.path.join(KBS, "cache", "msdb")
RESULTS = os.path.join(KBS, "results")
LOGS = os.path.join(KBS, "logs")
LOCK = os.path.join(KBS, ".mps_lock")

EPS = 1e-8
S = 512                 # MSDB input size (published protocol)
EVAL = 384              # common scoring grid
PPD = 21.7              # pixel_per_dva, package default for unknown geometry
SAL = 256               # MSDB saliency-map resolution for a 512 input (S / 2)
RO = 64                 # MSDB readout grid for a 512 input (S / 8)
C_IN = 2560
SIGMA_SCALE = PPD / 2.0  # finalizer sigma scaling factor (pixel_per_dva / saliency factor 2)
SIGMAS = [0, 2, 4, 6, 8, 10, 12]   # post-hoc finalizer grid, kfold_train_eval.py
SIGMA_SELECT_N = 200


# ----------------------------------------------------------------------------- data
def load_split(path=SPLIT):
    d = json.load(open(path))
    ids = list(d["ids_sorted"])
    folds = [list(map(str, f)) for f in d["folds"]]
    fold_of = {}
    for k, f in enumerate(folds):
        for i in f:
            fold_of[i] = k
    return ids, folds, fold_of


def load_img(i, size=S):
    im = Image.open(os.path.join(ADD, "images", f"{i}.jpg")).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, np.float32).transpose(2, 0, 1).copy()       # (3,S,S) in [0,255]


def load_gt(i, size):
    return np.asarray(Image.open(os.path.join(ADD, "maps", f"{i}.png")).convert("L")
                      .resize((size, size), Image.BILINEAR), np.float64)


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


def fit_prior(ids, size=S):
    """Log-density centre prior = log(mean of to_prob(GT) + EPS), as eval_deepgaze_msdb.py."""
    acc = np.zeros((size, size), np.float64)
    for i in ids:
        acc += to_prob(load_gt(i, size))
    acc /= len(ids)
    return np.log(acc + EPS)


# ----------------------------------------------------------------------------- head
def gaussian_filter_1d(tensor, dim, sigma, truncate=3):
    """Same maths as deepgaze_pytorch.layers.gaussian_filter_1d (replicate padding); adapted from deepgaze_pytorch
    (github.com/matthias-k/DeepGaze)."""
    sigma = torch.as_tensor(sigma, device=tensor.device, dtype=tensor.dtype)
    ks = int(2 * math.ceil(float(truncate * sigma.detach().cpu())) + 1)
    mean = (ks - 1) / 2
    grid = (torch.arange(ks, device=tensor.device, dtype=tensor.dtype) - mean).view(1, 1, ks)
    src = tensor.shape
    t = torch.movedim(tensor, dim, len(src) - 1)
    last = t.shape
    t = t.reshape(-1, 1, src[dim])
    pad = (math.ceil((ks - 1) / 2), math.ceil((ks - 1) / 2))
    t = F.pad(t, pad, "replicate")
    k = torch.exp(-0.5 * (grid / sigma) ** 2); k = k / k.sum()
    t = F.conv1d(t, k).view(last)
    return torch.movedim(t, len(src) - 1, dim)


class MSDBHead(nn.Module):
    """Trainable part of DeepGazeMSDB for ONE (new) dataset, fed with the cached scale mix.

    Initialised from the pretrained generalisation-mode values: saliency network as released,
    sigma = mean(dataset_sigmas), centre-bias weight = mean(dataset_center_bias_weights),
    priority scaling = 1 (log 0), exactly the dataset=None path of the package.
    """

    def __init__(self, sal_net, sigma0, cbw0):
        super().__init__()
        self.saliency_network = sal_net
        self.sigma = nn.Parameter(torch.tensor(float(sigma0)))
        self.cbw = nn.Parameter(torch.tensor(float(cbw0)))
        self.log_priority = nn.Parameter(torch.tensor(0.0))

    def readout(self, f):                      # f: (B,2560,64,64) -> (B,1,64,64)
        return self.saliency_network(f)

    def finalize(self, r, cb_down):
        """r: (B,1,64,64) readout; cb_down: (B,256,256) or (256,256) nearest-downscaled log prior.
        Follows deepgaze_pytorch.modules.Finalizer; adapted from deepgaze_pytorch (github.com/matthias-k/DeepGaze).
        Returns log density on the 256 grid (B,256,256), normalised."""
        x = F.interpolate(r, [SAL, SAL], mode="bilinear")
        x = gaussian_filter_1d(x, 2, self.sigma * SIGMA_SCALE)
        x = gaussian_filter_1d(x, 3, self.sigma * SIGMA_SCALE)[:, 0]
        x = x * torch.exp(self.log_priority)
        x = x + self.cbw * cb_down
        return x - x.logsumexp(dim=(1, 2), keepdim=True)

    def forward(self, f, cb_down):
        return self.finalize(self.readout(f), cb_down)


def cb_downscale(cb512):
    """Model's own centre-bias downscaling (F.interpolate default = nearest) to the 256 grid."""
    t = torch.as_tensor(cb512, dtype=torch.float32)
    return F.interpolate(t.view(1, 1, S, S), size=(SAL, SAL))[0, 0]


def logd256_to_prob(logd256, size=EVAL):
    """256-grid log density -> probability map at `size` (numpy float64).
    512 native = nearest 2x upsample of the 256 map (exactly the package's last step);
    other sizes = bilinear resize (align_corners=False) of the 512 density."""
    p = torch.exp(logd256.detach().float().cpu())
    if p.dim() == 2:
        p = p[None]
    p512 = F.interpolate(p[:, None], size=(S, S), mode="nearest")[:, 0] / 4.0
    if size == S:
        out = p512
    else:
        out = F.interpolate(p512[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    return out.double().numpy()


def build_sal_net_pretrained():
    """Saliency network + generalisation-mode scalars from the released checkpoint (no backbone)."""
    from deepgaze_pytorch.deepgazemsdb import _build_saliency_network
    sd = torch.load(os.path.expanduser("~/.cache/torch/hub/checkpoints/deepgazemsdb.pth"), map_location="cpu")
    net = _build_saliency_network(C_IN)
    net.load_state_dict({k[len("saliency_network."):]: v for k, v in sd.items() if k.startswith("saliency_network.")})
    sigma0 = float(sd["finalizer.gauss.dataset_sigmas"].mean())
    cbw0 = float(sd["finalizer.dataset_center_bias_weights"].mean())
    pw = torch.exp(sd["features.pixel_per_dva_weights"]).mean(1)
    sw = torch.exp(sd["features.size_weights"]).mean(1)
    tot = pw.sum() + sw.sum()
    scale_w = torch.cat([pw / tot, sw / tot])          # 10 generalisation-mode mixing weights
    return net, sigma0, cbw0, scale_w, sd


def count_params(m):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


# ----------------------------------------------------------------------------- MPS lock
class MPSLock:
    """mkdir-based lock shared by all tracks; released on exit, failure or SIGTERM."""

    def __init__(self, path=LOCK, poll=None):
        poll = float(os.environ.get("MPS_LOCK_POLL", "20")) if poll is None else poll
        self.path, self.poll, self.held = path, poll, False

    def acquire(self):
        t0 = time.time()
        while True:
            try:
                os.mkdir(self.path); self.held = True; break
            except FileExistsError:
                time.sleep(self.poll)
        atexit.register(self.release)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, self._handler)
        print(f"[lock] acquired after {time.time() - t0:.0f}s", flush=True)
        return self

    def _handler(self, signum, frame):
        self.release(); raise SystemExit(128 + signum)

    def release(self):
        if self.held:
            try:
                os.rmdir(self.path)
            except FileNotFoundError:
                pass
            self.held = False
            print("[lock] released", flush=True)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *a):
        self.release()


def write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)
