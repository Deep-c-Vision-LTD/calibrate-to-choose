"""Check FastIIE against forward_cached (the original module code path): outputs and
parameter gradients, on CPU, with trained-model weights. Uses real cached features if the
cache exists, else random features.

  python -I verify_fast.py [dev]
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import iie_common as C
import deepgaze_pytorch

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
torch.set_num_threads(6)
torch.manual_seed(0)
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
C.readout_to(model, dev)
fast = C.FastIIE(model, dev)
B = 4
path = os.path.join(C.CACHE, "iie384_feats_float32.npy")
if os.path.exists(path):
    arr = np.load(path, mmap_mode="r")
    fx = torch.from_numpy(np.ascontiguousarray(arr[:B])).to(dev); src = "cache"
else:
    fx = torch.randn(B, sum(C.BACKBONE_CH), C.G, C.G, device=dev) * 2; src = "random"
feats = C.split_channels(fx)
gt = C.gt_cache()
g = torch.from_numpy(gt["ms384"][:B]).to(dev)
cb_np = C.centerbias_from_maps(np.stack([C.to_prob(gt["raw384"][k].astype(np.float64)) for k in range(100, 300)]))
cb = torch.from_numpy(cb_np.astype(np.float32))[None].expand(B, -1, -1).contiguous().to(dev)
params = [p for p in model.parameters() if p.requires_grad]
rep = {"device": dev, "features": src}

def run(fn):
    for p in params: p.grad = None
    t = time.time()
    logd = fn(feats, cb)[:, 0]
    logp = F.log_softmax(logd.reshape(B, -1), 1).reshape(B, C.S, C.S)
    loss = (g * (torch.log(g + 1e-8) - logp)).sum((1, 2)).mean()
    loss.backward()
    if dev == "mps": torch.mps.synchronize()
    return logd.detach().double().cpu(), [p.grad.detach().double().cpu().clone() for p in params], float(loss), time.time() - t

a, ga, la, ta = run(lambda f, c: C.forward_cached(model, f, c))
b, gb, lb, tb = run(fast)
rel = [float((x - y).norm() / (x.norm() + 1e-30)) for x, y in zip(ga, gb)]
rep.update({"max_abs_logd": float((a - b).abs().max()), "loss_orig": la, "loss_fast": lb,
            "grad_rel_err_max": max(rel), "grad_rel_err_median": float(np.median(rel)),
            "t_orig_s": ta, "t_fast_s": tb})
pa, pb = torch.exp(a).numpy(), torch.exp(b).numpy()
rep["max_abs_dCC"] = max(abs(C.cc(pa[k], gt["raw384"][k]) - C.cc(pb[k], gt["raw384"][k])) for k in range(B))
print(json.dumps(rep, indent=1))
json.dump(rep, open(os.path.join(C.KBS, "logs", f"iie_verify_fast_{dev}.json"), "w"), indent=1)
