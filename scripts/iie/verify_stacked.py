"""Check StackedIIE (training implementation) against forward_cached (the released module
code path): outputs and parameter gradients, plus write_back round trip.

  python -I verify_stacked.py [dev]
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import iie_common as C
import deepgaze_pytorch

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
torch.set_num_threads(6)
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
st = C.StackedIIE(model).to(dev)
rep = {"device": dev, "trainable_params_model": C.trainable_params(model),
       "trainable_params_stacked": int(sum(p.numel() for p in st.parameters()))}
B = 4
path = os.path.join(C.CACHE, "iie384_feats_float32.npy")
if os.path.exists(path):
    arr = np.load(path, mmap_mode="r"); fx = torch.from_numpy(np.ascontiguousarray(arr[:B])); src = "cache"
else:
    torch.manual_seed(0); fx = torch.randn(B, sum(C.BACKBONE_CH), C.G, C.G) * 2; src = "random"
rep["features"] = src
gt = C.gt_cache()
g = torch.from_numpy(gt["ms384"][:B])
cb_np = C.centerbias_from_maps(np.stack([C.to_prob(gt["raw384"][k].astype(np.float64)) for k in range(100, 300)]))
cb = torch.from_numpy(cb_np.astype(np.float32))[None].expand(B, -1, -1).contiguous()

def lossf(logd, gg):
    logp = F.log_softmax(logd.reshape(B, -1), 1).reshape(B, C.S, C.S)
    return (gg * (torch.log(gg + 1e-8) - logp)).sum((1, 2)).mean()

# reference on CPU with module path
params = [p for p in model.parameters() if p.requires_grad]
la_logd = C.forward_cached(model, C.split_channels(fx), cb)[:, 0]
la = lossf(la_logd, g); la.backward()
# stacked
lb_logd = st(C.split_channels(fx.to(dev)), cb.to(dev))[:, 0]
lb = lossf(lb_logd, g.to(dev)); lb.backward()
rep["max_abs_logd"] = float((la_logd.detach().double() - lb_logd.detach().cpu().double()).abs().max())
rep["loss"] = [float(la), float(lb)]
# gradient comparison: global relative error over all parameters
ga, gb = [], []
for b, m in enumerate(model.models):
    comps = list(zip(m.saliency_networks, m.fixation_selection_networks, m.finalizers))
    for name, get in C._SPEC:
        tg = st.p(b, name).grad.detach().cpu().double()
        for k, (s, f, z) in enumerate(comps):
            ga.append(get(s, f, z).grad.detach().double().flatten()); gb.append(tg[k].flatten())
GA, GB = torch.cat(ga), torch.cat(gb)
rep["grad_global_rel_err"] = float((GA - GB).norm() / GA.norm())
rep["n_grad_elems"] = int(GA.numel())
# write_back round trip: perturb stacked params, write back, compare module forward
with torch.no_grad():
    for p_ in st.parameters():
        p_.mul_(1.01)
m2 = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
st.write_back(m2)
with torch.no_grad():
    a2 = C.forward_cached(m2, C.split_channels(fx), cb)[:, 0].double()
    b2 = st(C.split_channels(fx.to(dev)), cb.to(dev))[:, 0].cpu().double()
rep["write_back_max_abs_logd"] = float((a2 - b2).abs().max())
print(json.dumps(rep, indent=1))
json.dump(rep, open(os.path.join(C.KBS, "logs", f"iie_verify_stacked_{dev}_{src}.json"), "w"), indent=1)
