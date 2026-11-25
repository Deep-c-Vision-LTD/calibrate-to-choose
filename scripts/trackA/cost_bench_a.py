"""
Track A copy of scripts/compile/cost_bench.py (same protocol and timing code): adds our readout on
other frozen DINOv2 backbones / resolutions. Only the model construction branch is new.

  --model ours_v --variant {B448,L448,msB,B644,Lms}
     B448 : DINOv2-base 448 (= the existing 'ours' row; re-measured as a same-session reference)
     L448 : DINOv2-large 448 -> readout (1024 ch, 32x32)
     msB  : DINOv2-base at 448 and at 672; 672 tokens resized to 32x32 (bilinear, antialias) and
            concatenated -> readout (1536 ch). Two pre-resized input tensors (448 and 672), since
            image decoding / resizing is excluded from latency by the protocol.
     B644 : DINOv2-base 644 -> readout (768 ch, 46x46)
     Lms  : DINOv2-large at 448 and 672, as msB -> readout (2048 ch)

Original docstring follows.
Measured inference cost of the four model families used in the KBS letter (one model per process).

  python -I cost_bench.py --model {ours,iie,msdb,sum} --device {cpu,mps} --threads 6 \
      --warmup 3 --runs 20 --out results/cost_raw/<model>_<device>.json

What is measured (batch 1, float32, a fixed random input of the model's native input size):
  * parameters: total instantiated, backbone (frozen), trainable in our in-domain adaptation rows;
  * GFLOPs per image = 2 x multiply-accumulates of convolution and matrix-multiply ops only
    (torch.utils.flop_counter.FlopCounterMode; attention run through the SDPA math backend so its
    two batched matmuls are counted). Element-wise ops, normalisation, interpolation, pooling,
    the post-hoc Gaussian finalizer and SUM's selective scan are NOT in the total (the scan is
    reported separately as an estimate, using VMamba's own formula, converted to 2 x MACs);
  * CPU latency: median (and IQR) of `runs` timed forward passes after `warmup` passes, with
    torch.set_num_threads(threads); peak process RSS sampled every 2 ms during the timed passes;
  * MPS latency: same, with torch.mps.synchronize() around each pass; MPS memory = allocated
    tensors after loading (weights) and the driver allocation after the passes (the allocator keeps
    freed blocks, so it approximates the peak working set).

Inputs and post-processing follow each model's protocol in the letter:
  ours : 448x448 ImageNet-normalised image -> frozen DINOv2-base -> ASPP readout -> 32x32 logits ->
         bilinear 384 + sigmoid; then the Gaussian finalizer (sigma 4, scipy) timed separately.
  iie  : 384x384 image in [0,255] + 384x384 log centre prior (DeepGazeIIE, released weights).
  msdb : 512x512 image in [0,255] + 512x512 log centre prior, pixel_per_dva 21.7, dataset=None
         (generalisation parameters). Its two float64 normaliser buffers are cast to float32 on
         both devices (MPS cannot hold float64; the CPU number is then a float32 number too).
  sum  : 256x256 ImageNet-normalised image + one-hot domain token 2; pure-PyTorch chunked
         selective scan (the compiled CUDA kernel is unavailable here).
"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import argparse, json, time, threading, platform, statistics, warnings
warnings.filterwarnings("ignore")

KB = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
OURS_DIR = os.path.join(KB, "scripts", "ours")
SUM_DIR = os.path.join(KB, "scripts", "sum")
SUM_CODE = os.path.join(KB, "third_party", "SUM", "code")
SUM_W = os.path.join(KB, "third_party", "SUM", "weights")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["ours", "iie", "msdb", "sum", "ours_v"])
ap.add_argument("--variant", default="B448", choices=["B448", "L448", "msB", "B644", "Lms"])
ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--runs", type=int, default=20)
ap.add_argument("--no_flops", action="store_true")
ap.add_argument("--readout_ckpt", default=None, help="optional readout weights for ours (do not affect cost)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(a.threads)
PROC = psutil.Process()
GB = 1024 ** 3


def rss():
    return PROC.memory_info().rss


def npar(m):
    return int(sum(p.numel() for p in m.parameters()))


class PeakRSS:
    def __init__(self, dt=0.002):
        self.dt, self.peak, self._stop = dt, rss(), False

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start(); return self

    def _run(self):
        while not self._stop:
            r = rss()
            if r > self.peak:
                self.peak = r
            time.sleep(self.dt)

    def __exit__(self, *e):
        self._stop = True; self._t.join()


def cast_f64_buffers(model):
    n = 0
    for m in model.modules():
        for name, buf in list(m._buffers.items()):
            if buf is not None and buf.dtype == torch.float64:
                m._buffers[name] = buf.float(); n += 1
    return n


out = {"model": a.model, "device": a.device, "threads": a.threads, "warmup": a.warmup, "runs": a.runs,
       "torch": torch.__version__, "machine": platform.platform(), "processor": platform.processor(),
       "command": "python -I " + " ".join(sys.argv)}
rss_import = rss()
t_load = time.time()

# ------------------------------------------------------------------ model construction
scan_calls = []
if a.model == "ours_v":
    sys.path.insert(0, os.path.join(KB, "scripts", "trackA"))
    from transformers import AutoModel
    from run_row_a import ReadoutHeadA
    VAR = {"B448": ("facebook/dinov2-base", [448]), "L448": ("facebook/dinov2-large", [448]),
           "msB": ("facebook/dinov2-base", [448, 672]), "B644": ("facebook/dinov2-base", [644]),
           "Lms": ("facebook/dinov2-large", [448, 672])}
    bname, sizes = VAR[a.variant]
    backbone = AutoModel.from_pretrained(bname).eval()
    for p_ in backbone.parameters():
        p_.requires_grad = False
    C = backbone.config.hidden_size
    readout = ReadoutHeadA(C * len(sizes)).eval()
    out["variant"] = a.variant; out["backbone_name"] = bname; out["input_sizes"] = sizes
    out["readout_weights"] = "random init (weights do not affect cost)"
    bb = npar(backbone)
    out["params_total"] = bb + npar(readout)
    out["params_backbone_frozen"] = bb
    out["params_trainable_in_adaptation"] = npar(readout)
    out["input"] = " + ".join(f"1x3x{s_}x{s_}" for s_ in sizes) + " ImageNet-normalised"

    class OursVPipe(nn.Module):
        def __init__(self, bb_, ro, multi):
            super().__init__(); self.bb = bb_; self.ro = ro; self.multi = multi

        def tok(self, x):
            out_ = self.bb(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
            t = out_[:, 1:, :]
            B, N, C_ = t.shape; h = int(round(N ** 0.5))
            return t.transpose(1, 2).reshape(B, C_, h, h)

        def forward(self, *xs):
            fs = [self.tok(x) for x in xs]
            if self.multi:
                g0 = fs[0].shape[-2:]
                fs = [fs[0]] + [F.interpolate(f_, size=g0, mode="bilinear", antialias=True, align_corners=False) for f_ in fs[1:]]
            f = fs[0] if len(fs) == 1 else torch.cat(fs, 1)
            s_ = self.ro(f)
            return torch.sigmoid(F.interpolate(s_, size=(384, 384), mode="bilinear", align_corners=False))

    net = OursVPipe(backbone, readout, len(sizes) > 1).eval()
    model = net
    g = torch.Generator().manual_seed(0)
    args = tuple(torch.randn(1, 3, s_, s_, generator=g) for s_ in sizes); kwargs = {}

elif a.model == "ours":
    sys.path.insert(0, OURS_DIR)
    from model_aspp import ASPPSaliency
    model = ASPPSaliency(pretrained=True).eval()
    ck = {}
    if a.readout_ckpt and os.path.exists(a.readout_ckpt):
        sd = torch.load(a.readout_ckpt, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        ck = {k: v for k, v in sd.items() if k.startswith(("aspp.", "head.", "center_bias"))}
        model.load_state_dict(ck, strict=False)
    out["readout_weights"] = "warm-start readout (weights do not affect cost)" if ck else "random init"
    bb = npar(model.backbone)
    out["params_total"] = npar(model)
    out["params_backbone_frozen"] = bb
    out["params_trainable_in_adaptation"] = npar(model) - bb
    out["input"] = "1x3x448x448 ImageNet-normalised"

    class OursPipe(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, x):
            out_ = self.m.backbone(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
            tok = out_[:, 1:, :]
            B, N, C = tok.shape; h = int(round(N ** 0.5))
            f = tok.transpose(1, 2).reshape(B, C, h, h)
            s = self.m.head(self.m.aspp(f))
            s = s + F.interpolate(self.m.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)
            return torch.sigmoid(F.interpolate(s, size=(384, 384), mode="bilinear", align_corners=False))

    net = OursPipe(model).eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3, 448, 448, generator=g)
    args = (x,); kwargs = {}

elif a.model == "iie":
    import deepgaze_pytorch
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
    bb = int(sum(p.numel() for n, p in model.named_parameters() if ".features." in n))
    out["params_total"] = npar(model)
    out["params_backbone_frozen"] = bb
    out["params_trainable_in_adaptation"] = npar(model) - bb
    out["n_float64_buffers_cast"] = cast_f64_buffers(model) if a.device == "mps" else 0
    out["input"] = "1x3x384x384 in [0,255] + 1x384x384 log centre prior (uniform)"
    net = model
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 3, 384, 384, generator=g) * 255
    cb = torch.full((1, 384, 384), -float(np.log(384 * 384)))
    args = (x, cb); kwargs = {}

elif a.model == "msdb":
    import deepgaze_pytorch
    model = deepgaze_pytorch.DeepGazeMSDB(pretrained=True)
    model.eval()
    out["n_float64_buffers_cast"] = cast_f64_buffers(model)
    model.eval()
    out["params_total"] = npar(model)
    out["params_saliency_network"] = npar(model.saliency_network)
    out["params_finalizer_all_datasets"] = npar(model.finalizer)
    out["params_backbone_frozen"] = npar(model.features)
    out["params_trainable_in_adaptation"] = 26463
    out["params_trainable_in_adaptation_note"] = ("msdb_adapt: saliency network 26,460 + 3 single-dataset finalizer "
                                                  "scalars; msdb_adapt_dsp: 13")
    out["input"] = "1x3x512x512 in [0,255] + 1x512x512 log centre prior (uniform), pixel_per_dva 21.7, dataset=None"
    net = model
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 3, 512, 512, generator=g) * 255
    cb = torch.full((1, 512, 512), -float(np.log(512 * 512)))
    args = (x, cb); kwargs = {"pixel_per_dva": 21.7, "dataset": None}

elif a.model == "sum":
    sys.path.insert(0, SUM_DIR)
    import sum_common as sc
    model, cfg = sc.load_sum(SUM_CODE, SUM_W, "cpu", scan="chunked")
    model.eval()
    vm = sys.modules["sum_thirdparty_models.vmamba"]
    tot = npar(model)
    head = int(sum(p.numel() for n, p in model.named_parameters() if n.split("salu_mamba.", 1)[-1].startswith("final")))
    out["params_total"] = tot
    out["params_backbone_frozen"] = tot - head
    out["params_trainable_in_adaptation"] = head
    out["params_trainable_in_adaptation_note"] = "sum_ft_head: SUM's own output head (final_up + final_conv) only"
    out["input"] = "1x3x256x256 ImageNet-normalised + one-hot domain token 2"
    out["scan"] = "pure-PyTorch chunked selective scan (scripts/sum/sum_common.py), not the CUDA kernel"
    net = model
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3, 256, 256, generator=g)
    cond = torch.zeros(1, 4); cond[0, 2] = 1
    args = (x, cond); kwargs = {}

out["load_s"] = time.time() - t_load
rss_loaded = rss()
out["rss_after_import_GB"] = rss_import / GB
out["rss_after_load_GB"] = rss_loaded / GB


def move(t, dev):
    return t.to(dev) if torch.is_tensor(t) else t


# ------------------------------------------------------------------ latency
dev = torch.device(a.device)
if a.device == "mps":
    if a.model == "iie":
        cast_f64_buffers(model)
    net.to(dev)
    torch.mps.synchronize()
    out["mps_allocated_after_load_GB"] = torch.mps.current_allocated_memory() / GB
args_d = tuple(move(t, dev) for t in args)


def sync():
    if a.device == "mps":
        torch.mps.synchronize()


def fwd():
    with torch.inference_mode():
        y = net(*args_d, **kwargs)
    sync()
    return y


times, warm_times = [], []
drv_max = 0
with PeakRSS() as pk:
    for i in range(a.warmup):
        t0 = time.perf_counter(); y = fwd(); warm_times.append(time.perf_counter() - t0)
        if a.device == "mps":
            drv_max = max(drv_max, torch.mps.driver_allocated_memory())
    for i in range(a.runs):
        t0 = time.perf_counter(); y = fwd(); times.append(time.perf_counter() - t0)
        if a.device == "mps":
            drv_max = max(drv_max, torch.mps.driver_allocated_memory())
y_cpu = y.float().cpu() if torch.is_tensor(y) else y
out["output_shape"] = list(y_cpu.shape)
q = np.percentile(times, [25, 50, 75])
out["latency_s"] = {"median": float(q[1]), "q25": float(q[0]), "q75": float(q[2]), "mean": float(np.mean(times)),
                    "min": float(np.min(times)), "max": float(np.max(times)), "all": [float(t) for t in times],
                    "warmup": [float(t) for t in warm_times]}
out["peak_rss_during_inference_GB"] = pk.peak / GB
out["peak_rss_increment_over_loaded_GB"] = (pk.peak - rss_loaded) / GB
if a.device == "mps":
    out["mps_driver_allocated_max_GB"] = drv_max / GB

# post-hoc Gaussian finalizer of our pipeline (scipy, CPU), timed separately
if a.model in ("ours", "ours_v"):
    from scipy.ndimage import gaussian_filter
    p = y_cpu[0, 0].double().numpy()
    ft = []
    for i in range(a.warmup + a.runs):
        t0 = time.perf_counter(); gaussian_filter(p, 4); ft.append(time.perf_counter() - t0)
    ft = ft[a.warmup:]
    out["finalizer_sigma4_scipy_cpu_s_median"] = float(np.median(ft))
    out["latency_incl_finalizer_s_median"] = float(q[1] + np.median(ft))

# ------------------------------------------------------------------ FLOPs (CPU, one pass)
if not a.no_flops and a.device == "cpu":
    from torch.utils.flop_counter import FlopCounterMode
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if a.model == "sum":
        orig = vm.selective_scan_fn

        def rec(u, delta, A, B, C, *r, **k):
            scan_calls.append({"u": list(u.shape), "A": list(A.shape), "B": list(B.shape)})
            return orig(u, delta, A, B, C, *r, **k)
        vm.selective_scan_fn = rec
    net.requires_grad_(False)     # module tracker of FlopCounterMode hooks inputs that require grad
    fc = FlopCounterMode(display=False, depth=3)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), fc:
        net(*args, **kwargs)
    total = int(fc.get_total_flops())
    counts = fc.get_flop_counts()
    by_mod = {}
    for mod, d in counts.items():
        by_mod[mod] = {str(k): int(v) for k, v in d.items()}
    out["flops_convention"] = ("FLOPs = 2 x MACs of convolution and matmul ops (aten.convolution, mm, addmm, bmm, "
                               "baddbmm; attention via SDPA math backend), torch.utils.flop_counter.FlopCounterMode, "
                               "one batch-1 forward pass; element-wise ops, norms, interpolation, pooling and the "
                               "selective scan are excluded")
    out["GFLOPs"] = total / 1e9
    out["flops_by_module_depth3"] = {k: sum(v.values()) / 1e9 for k, v in by_mod.items()}
    out["flops_by_op_global"] = {k: v / 1e9 for k, v in by_mod.get("Global", {}).items()}
    if a.model == "sum":
        vm.selective_scan_fn = orig
        sf = 0.0
        for c in scan_calls:
            Bb, KD, L = c["u"]; Dn, N = c["A"]
            sf += vm.flops_selective_scan_ref(B=Bb, L=L, D=KD, N=N, with_D=True, with_Z=False, with_Group=True)
        out["selective_scan_calls"] = len(scan_calls)
        out["selective_scan_shapes_first"] = scan_calls[:2]
        out["selective_scan_GFLOPs_estimate"] = 2 * sf / 1e9
        out["selective_scan_note"] = ("estimate from VMamba's flops_selective_scan_ref (MACs, fvcore convention) x 2; "
                                      "not part of GFLOPs above")
        out["GFLOPs_incl_scan_estimate"] = out["GFLOPs"] + out["selective_scan_GFLOPs_estimate"]

out["rss_at_end_GB"] = PROC.memory_info().rss / GB
import resource
out["ru_maxrss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GB   # bytes on macOS
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("flops_by_module_depth3",)}, indent=1)[:4000])
