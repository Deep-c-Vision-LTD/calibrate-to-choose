"""
Shared code of the ftblocks track: fine-tune the LAST N transformer blocks of DINOv2-base (+ its final
LayerNorm) together with our ASPP readout, from cached frozen block outputs.

What the base readout consumes (read from scripts/ours/cache_features.py, model_aspp.py, ours_lib.py):
  * checkpoint   facebook/dinov2-base (HF transformers AutoModel, attn_implementation 'sdpa', no registers)
  * input        PIL RGB, bilinear resize to 448 x 448, /255, ImageNet mean/std, interpolate_pos_encoding=True
  * tokens       last_hidden_state = final LayerNorm(output of block 12); CLS (token 0) dropped; the 1024
                 patch tokens reshaped to 768 x 32 x 32 (tok.transpose(1, 2).reshape(B, 768, 32, 32));
                 cached as float16 in cache/ours/feats_dinov2b_448_fp16.npy (ids_sorted order)
  * readout      ours_lib.ReadoutHead('full'): ASPP(768 -> 128, dilations 1/3/6/12 + GAP) + 1x1 head +
                 learned 16 x 16 centre-bias logit map; 3,721,089 params

Fine-tuned model with N > 0 (same architecture, same inference cost as ours):
  cached output of block 12-N (all 1025 tokens: CLS + 1024 patches; dinov2-base has no registers)
    -> blocks 12-N+1 .. 12 (trainable)  -> final LayerNorm (trainable)  -> drop CLS -> 768x32x32
    -> ReadoutHead (trainable)
N = 0 is the unchanged base recipe (ours_lib.train on the fp16 feature cache).
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, copy, math, hashlib, signal, atexit
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.insert(0, os.path.join(KB, "scripts", "ours"))
import ours_lib as L  # noqa: E402

CFT = os.path.join(KB, "cache", "ftblocks")
RUNS = os.path.join(CFT, "runs")
MODELS = os.path.join(CFT, "models")
SMOKE = os.path.join(CFT, "smoke")
LOCK = os.path.join(KB, ".mps_lock")
DRIVER_LOCK = os.path.join(CFT, ".driver_lock")
PLAN_JSON = os.path.join(HERE, "PLAN.json")
INNER_SPLITS = os.path.join(HERE, "inner_splits.json")
RESULTS = os.path.join(KB, "results")
PITT_DIR = os.environ.get("PITT_ADS_DIR", os.path.join(KB, "data", "pitt_ads", "imgs"))   # Pitt Ads images (not redistributed; only the optional distillation row of the plan reads them)
BACKBONE = "facebook/dinov2-base"
SIZE = 448
NTOK = 1025          # CLS + 32 x 32 patches
C = 768
IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)
# block caches: depth d = output of block d (1-indexed) = HF hidden_states[d]
ADD_CACHE = {8: os.path.join(CFT, "add_b08_fp32.npy"), 10: os.path.join(CFT, "add_b10_fp32.npy")}
PITT_CACHE = {8: os.path.join(CFT, "pitt_b08_fp16.npy"), 10: os.path.join(CFT, "pitt_b10_fp16.npy")}
PITT_FINAL16 = os.path.join(KB, "cache", "distill", "pitt_feats_dinov2b_448_fp16.npy")
PITT_SUBSET = os.path.join(KB, "cache", "distill", "pitt_subset.json")
PITT_T64 = os.path.join(KB, "cache", "distill", "targets", "pitt_t64.npy")

READOUT_LR, READOUT_WD = 3e-4, 1e-4
BB_WD = 0.05
WARMUP_FRAC = 0.05
EPOCHS = 60
BS = 16
W_PSEUDO = 1.0


def set_env():
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "6")


def load_backbone():
    set_env()
    from transformers import AutoModel
    m = AutoModel.from_pretrained(BACKBONE).eval()
    for p in m.parameters():
        p.requires_grad = False
    assert m.config._attn_implementation == "sdpa", m.config._attn_implementation
    assert m.config.drop_path_rate == 0.0 and m.config.hidden_dropout_prob == 0.0 and m.config.attention_probs_dropout_prob == 0.0
    assert len(m.encoder.layer) == 12 and m.config.hidden_size == C
    return m


def preprocess(path):
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    return torch.from_numpy(((np.asarray(im, np.float32) / 255 - IM_M) / IM_S).transpose(2, 0, 1))[None]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------- folds
def outer_folds(ids_pos):
    Fd = json.load(open(os.path.join(KB, "splits", "kfold_seed0.json")))
    folds = [[ids_pos[i] for i in f] for f in Fd["folds"]]
    assert sorted(sum(folds, [])) == list(range(1000))
    return folds


def outer_split(folds, k):
    """exactly run_row.py: test = fold k, train = concatenation of the other folds in fold order."""
    te = np.array(folds[k])
    tr = np.concatenate([np.array(folds[j]) for j in range(len(folds)) if j != k])
    return tr, te


def inner_split(tr, k):
    """pre-registered: perm = numpy default_rng(100 + k).permutation(800); inner-train = tr[perm[:640]],
    inner-val = tr[perm[640:]] (tr in run_row order)."""
    assert len(tr) == 800
    perm = np.random.default_rng(100 + k).permutation(len(tr))
    return tr[perm[:640]], tr[perm[640:]]


# ----------------------------------------------------------------------------------------- data
class _View:
    """Data-like object accepted by ours_lib.train / raw_preds / select_sigma_subset / score."""
    def __init__(self, parent, getx):
        self.p, self.getx = parent, getx
        self.ids, self.pos, self.gte, self.dev = parent.ids, parent.pos, parent.gte, parent.dev

    def batch(self, idx):
        idx = np.asarray(idx)
        ix = torch.as_tensor(idx, device=self.dev)
        return self.getx(idx), self.p.gtl[ix]


class FTData:
    """GT maps on the device + lazily opened feature sources:
       'final16'  the base fp16 readout features (cache/ours), loaded to the device as in ours_lib.Data
       depth d    the cached fp32 output of block d (memmap; batches gathered on the CPU, moved to the device)"""
    def __init__(self, dev):
        self.ids = L.load_ids()
        self.pos = {i: n for n, i in enumerate(self.ids)}
        self.dev = dev
        self.gtl = torch.stack([torch.tensor(L.load_gt_train(os.path.join(L.FLAT, "maps", f"{i}.png"), L.LOSSG), dtype=torch.float32)
                                for i in self.ids]).to(dev)
        self.gte = [L.load_gt_eval(os.path.join(L.FLAT, "maps", f"{i}.png")) for i in self.ids]
        self._f16 = None
        self._mm = {}

    def view(self, src):
        if src == "final16":
            if self._f16 is None:
                f = np.load(os.path.join(L.CACHE, "feats_dinov2b_448_fp16.npy"), mmap_mode="r")
                assert f.shape == (1000, 768, 32, 32)
                self._f16 = torch.from_numpy(np.array(f)).to(self.dev)
            return _View(self, lambda idx: self._f16[torch.as_tensor(idx, device=self.dev)].float())
        d = int(src)
        if d not in self._mm:
            mm = np.load(ADD_CACHE[d], mmap_mode="r")
            assert mm.shape == (1000, NTOK, C) and mm.dtype == np.float32, (mm.shape, mm.dtype)
            self._mm[d] = mm
        mm = self._mm[d]
        return _View(self, lambda idx: torch.from_numpy(np.ascontiguousarray(mm[idx])).to(self.dev).float())


class PittData:
    """Pitt pool (2,499 images, pool order) for the optional distillation row: features + teacher targets."""
    def __init__(self, dev, src):
        sub = json.load(open(PITT_SUBSET))
        self.pool_pos = np.array(sub["pool_positions"]); self.P = len(self.pool_pos)
        self.dev = dev
        if src == "final16":
            pf = np.load(PITT_FINAL16, mmap_mode="r")
            pf = pf[self.pool_pos] if pf.shape[0] == len(sub["subset_files"]) else np.array(pf)
            self._f = torch.from_numpy(np.ascontiguousarray(pf)).to(dev)
            self.getx = lambda idx: self._f[torch.as_tensor(idx, device=dev)].float()
        else:
            mm = np.load(PITT_CACHE[int(src)], mmap_mode="r")
            assert mm.shape == (self.P, NTOK, C), mm.shape
            self.getx = lambda idx: torch.from_numpy(np.ascontiguousarray(mm[np.asarray(idx)])).to(dev).float()
        self.pt64 = torch.from_numpy(np.load(PITT_T64)).to(dev)          # (5, P, 64, 64)
        assert self.pt64.shape[1] == self.P

    def batch(self, idx, k):
        return self.getx(idx), self.pt64[k][torch.as_tensor(np.asarray(idx), device=self.dev)]


class Cycler:          # identical to scripts/distill/train_students.py
    def __init__(self, n, rng):
        self.n, self.rng, self.buf = n, rng, []

    def take(self, b):
        out = []
        while len(out) < b:
            if not self.buf:
                self.buf = list(self.rng.permutation(self.n))
            out.append(self.buf.pop(0))
        return out


# ----------------------------------------------------------------------------------------- model
class FTModel(nn.Module):
    """[frozen blocks src+1..12-N (no grad, only if the source cache is shallower)] -> trainable blocks
    12-N+1..12 -> trainable final LayerNorm -> patch tokens 768x32x32 -> ReadoutHead."""
    def __init__(self, head, backbone, n_ft, src_depth):
        super().__init__()
        assert n_ft > 0 and src_depth <= 12 - n_ft
        self.n_ft, self.src_depth = n_ft, src_depth
        self.head = head
        self.blocks = nn.ModuleList([copy.deepcopy(b) for b in backbone.encoder.layer[12 - n_ft:]])
        self.norm = copy.deepcopy(backbone.layernorm)
        self.frozen = nn.ModuleList([copy.deepcopy(b) for b in backbone.encoder.layer[src_depth:12 - n_ft]])
        for p in self.frozen.parameters():
            p.requires_grad = False
        for p in list(self.blocks.parameters()) + list(self.norm.parameters()):
            p.requires_grad = True

    def backbone_params(self):
        return list(self.blocks.parameters()) + list(self.norm.parameters())

    def features(self, h):
        if len(self.frozen):
            with torch.no_grad():
                for b in self.frozen:
                    h = b(h)
        for b in self.blocks:
            h = b(h)
        h = self.norm(h)
        B = h.shape[0]
        return h[:, 1:, :].transpose(1, 2).reshape(B, C, 32, 32)

    def forward(self, h):
        return self.head(self.features(h))


def kl_cc(s, g):
    """base-recipe loss (identical to ours_lib.train / distill kl_cc): logits (B,1,h,w) -> 64x64."""
    B = s.shape[0]
    s = F.interpolate(s, size=(L.LOSSG, L.LOSSG), mode="bilinear", align_corners=False)[:, 0]
    logp = F.log_softmax(s.reshape(B, -1), 1).reshape(B, L.LOSSG, L.LOSSG)
    kl = (g * (torch.log(g + 1e-12) - logp)).sum((1, 2)).mean()
    ps = torch.sigmoid(s)
    pc = ps - ps.mean((1, 2), keepdim=True); gc = g - g.mean((1, 2), keepdim=True)
    ccb = (pc * gc).sum((1, 2)) / (torch.sqrt((pc ** 2).sum((1, 2)) * (gc ** 2).sum((1, 2))) + 1e-8)
    return kl + (1 - ccb.mean())


def make_optimizer(model, n_ft, lr_bb, total_steps):
    """N = 0: the base optimiser (AdamW lr 3e-4 wd 1e-4 on the readout, constant lr).
    N > 0: one AdamW with two groups: readout lr 3e-4 wd 1e-4 (constant, as the base), backbone blocks +
    final norm lr_bb wd 0.05 with linear warmup over the first 5% of steps (lr_bb * (s+1)/W for s < W),
    then constant (= the readout's schedule)."""
    if n_ft == 0:
        return torch.optim.AdamW(list(model.parameters()), lr=READOUT_LR, weight_decay=READOUT_WD), None, 0
    warm = max(1, int(round(WARMUP_FRAC * total_steps)))
    opt = torch.optim.AdamW([{"params": list(model.head.parameters()), "lr": READOUT_LR, "weight_decay": READOUT_WD},
                             {"params": model.backbone_params(), "lr": lr_bb, "weight_decay": BB_WD}])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lambda s: 1.0, lambda s: min(1.0, (s + 1) / warm)])
    return opt, sched, warm


def train_loop(model, view, tr, rng, n_ft, lr_bb, epochs=EPOCHS, bs=BS, pitt=None, k=None, cyc=None,
               max_steps=None, log=print, tag=""):
    """The body of ours_lib.train (same shuffling, batching, loss) with the ftblocks optimiser; optional
    Pitt stream (distill_pitt recipe: 16 GT + 16 Pitt per step, one forward, L(GT) + w L(Pitt))."""
    tr = np.asarray(tr)
    spe = int(math.ceil(len(tr) / bs))
    total = epochs * spe
    opt, sched, warm = make_optimizer(model, n_ft, lr_bb, total)
    steps, loss_log, t0 = 0, [], time.time()
    for ep in range(epochs):
        model.train()
        tr_shuf = tr.copy(); rng.shuffle(tr_shuf)
        tot, nb = 0.0, 0
        for j in range(0, len(tr_shuf), bs):
            ids = tr_shuf[j:j + bs]
            x, g = view.batch(ids)
            if pitt is not None:
                xp, tp = pitt.batch(cyc.take(bs), k)
                s = model(torch.cat([x, xp], 0))
                loss = kl_cc(s[:len(ids)], g) + W_PSEUDO * kl_cc(s[len(ids):], tp)
            else:
                loss = kl_cc(model(x), g)
            opt.zero_grad(); loss.backward(); opt.step()
            if sched is not None:
                sched.step()
            lv = float(loss.detach())
            if not math.isfinite(lv):
                raise FloatingPointError(f"{tag}: non-finite loss {lv} at step {steps}")
            steps += 1; tot += lv; nb += 1
            if max_steps is not None and steps >= max_steps:
                break
        loss_log.append(tot / max(nb, 1))
        if ep % 10 == 0 or ep == epochs - 1 or (max_steps is not None and steps >= max_steps):
            log(f"  [{tag}] ep {ep} loss {loss_log[-1]:.4f} step {steps}/{total} ({time.time() - t0:.0f}s)")
        if max_steps is not None and steps >= max_steps:
            break
    model.eval()
    return {"steps": steps, "planned_steps": total, "warmup_steps": warm, "epoch_loss": loss_log,
            "train_s": time.time() - t0, "s_per_step": (time.time() - t0) / max(steps, 1)}


def n_trainable(n_ft, backbone=None):
    head = L.n_params(L.ReadoutHead("full"))
    if n_ft == 0:
        return head
    per_block = sum(p.numel() for p in backbone.encoder.layer[0].parameters())
    return head + n_ft * per_block + sum(p.numel() for p in backbone.layernorm.parameters())


def cfg_name(n_ft, lr_bb):
    return "N0" if n_ft == 0 else f"N{n_ft}_lr{lr_bb:.0e}"


def src_depth_for(n_ft, available=(8, 10)):
    """deepest cached block output that is <= 12 - N (N=4 -> block 8, N=2 -> block 10)."""
    ok = [d for d in available if d <= 12 - n_ft]
    return max(ok)


# ----------------------------------------------------------------------------------------- MPS lock
def tracka_waiting():
    try:
        sys.path.insert(0, os.path.join(KB, "scripts", "distill"))
        import distill_common as DC
        return DC.tracka_waiting()
    except Exception:          # noqa
        return False


class MPSLock:
    """The shared protocol: mkdir .mps_lock while using the MPS GPU; released after every run,
    on exit, failure or SIGTERM/SIGINT/SIGHUP. Never touches a lock it does not hold."""
    def __init__(self, poll=20, log=print):
        self.poll, self.held, self.log = poll, False, log
        atexit.register(self.release)
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, self._handler)

    def _handler(self, signum, frame):
        self.release(); raise SystemExit(128 + signum)

    def acquire(self):
        t0 = time.time(); said = False
        while True:
            if not tracka_waiting():
                try:
                    os.mkdir(LOCK); self.held = True          # bare mkdir, exactly the shared convention
                    self.log(f"[lock] acquired after {time.time() - t0:.0f}s")
                    return
                except FileExistsError:
                    pass
            if not said:
                self.log("[lock] waiting for the shared MPS lock"); said = True
            time.sleep(self.poll)

    def release(self):
        if self.held:
            try:
                os.rmdir(LOCK)
            except FileNotFoundError:
                pass
            except OSError as e:
                self.log(f"[lock] could not remove lock dir: {e}")
            self.held = False
            self.log("[lock] released")


class DriverLock:
    """one ftblocks driver at a time (mkdir cache/ftblocks/.driver_lock)."""
    def __init__(self):
        os.makedirs(CFT, exist_ok=True)
        try:
            os.mkdir(DRIVER_LOCK)
        except FileExistsError:
            raise SystemExit(f"another ftblocks driver holds {DRIVER_LOCK} (remove it only if no driver is running)")
        with open(os.path.join(DRIVER_LOCK, "pid"), "w") as fh:
            fh.write(str(os.getpid()))
        atexit.register(self.release)

    def release(self):
        try:
            os.remove(os.path.join(DRIVER_LOCK, "pid")); os.rmdir(DRIVER_LOCK)
        except OSError:
            pass
