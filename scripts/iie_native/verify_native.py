"""
Check the size-general training forward (NativeStackedIIE + WFinalMix + klcc64_native) against the RELEASED
DeepGaze IIE module code path (iie_common.forward_cached, i.e. the released saliency / fixation-selection
networks and Finalizer) on native-size inputs, including odd heights / widths:
  * log density on the full H x W map (max abs difference),
  * loss value and gradients of every trainable readout parameter (global relative error), where the
    released-path loss is computed from its full H x W density with torch's own bilinear resize to 384,
    6 x 6 sum-pool, renormalisation, KL + (1 - CC),
  * WFinalMix custom backward vs plain autograd (logsumexp formulation),
  * write-back round trip (perturbed stacked parameters -> released modules).
Features are computed on the fly from the images (no cache needed). CPU only.

  python -I verify_native.py [--threads 6]
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import native_common as N
C = N.C

ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=6); a = ap.parse_args()
torch.set_num_threads(a.threads)
import deepgaze_pytorch

t0 = time.time()
ids, folds = C.load_split()
# images: smallest odd x odd, odd H / even W, even H / odd W, and one 384-resized image (square path)
sizes = {i: N.image_size(i) for i in ids}                                  # (W,H)
def pick(cond):
    c = [i for i in ids if cond(*sizes[i]) and sizes[i][0] * sizes[i][1] < 420 * 420]
    return min(c, key=lambda i: sizes[i][0] * sizes[i][1]) if c else None
cases = [x for x in (pick(lambda W, H: H % 2 == 1 and W % 2 == 1), pick(lambda W, H: H % 2 == 1 and W % 2 == 0),
                     pick(lambda W, H: H % 2 == 0 and W % 2 == 1), pick(lambda W, H: H % 2 == 0 and W % 2 == 0)) if x]
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
D_raw = np.load(os.path.join(C.CACHE, "gt_raw384.npy"), mmap_mode="r")
D_ms64 = np.load(os.path.join(C.CACHE, "gt_ms64.npy"), mmap_mode="r")
pos = {i: k for k, i in enumerate(ids)}
# a centre prior of the same form as the row's (training maps of fold 0 -> min-sub 384 -> sum -> log(prob)+eps)
acc = np.zeros((C.S, C.S))
for i in folds[1][:200]:
    m = D_raw[pos[i]].astype(np.float64); m = m - m.min(); acc += m / m.sum()
prior = np.log(C.to_prob(acc) + C.EPS)


def released_loss(logd_full, g64):
    d = torch.exp(logd_full)                                                  # (1,H,W)
    d384 = F.interpolate(d[None], size=(C.S, C.S), mode="bilinear", align_corners=False)[0]
    q = F.avg_pool2d(d384[None], C.S // C.LOSSG)[0, 0] * (C.S // C.LOSSG) ** 2
    q = q / q.sum()
    kl = (g64 * (torch.log(g64 + 1e-12) - torch.log(q.clamp_min(1e-30)))).sum()
    pc = q - q.mean(); gc = g64 - g64.mean()
    return kl + (1 - (pc * gc).sum() / (torch.sqrt((pc ** 2).sum() * (gc ** 2).sum()) + 1e-8))


def grads_module(m):
    out = []
    for mm in m.models:
        for s, f, z in zip(mm.saliency_networks, mm.fixation_selection_networks, mm.finalizers):
            for name, get in C._SPEC:
                p = get(s, f, z)
                out.append((p.grad if p.grad is not None else torch.zeros_like(p)).detach().double().flatten())
    return torch.cat(out)


def grads_stacked(st):
    out = []
    for b, mm in enumerate(st_model.models):
        for k in range(len(mm.saliency_networks)):
            for name, get in C._SPEC:
                g = st.p(b, name).grad
                out.append(g[k].detach().double().flatten())
    return torch.cat(out)


# every parameter the stacked class trains must be trainable in the released modules for this check
for p in model.parameters():
    p.requires_grad_(False)
for mm in model.models:
    for s, f, z in zip(mm.saliency_networks, mm.fixation_selection_networks, mm.finalizers):
        for name, get in C._SPEC:
            get(s, f, z).requires_grad_(True)
st_model = model
rep = {"cases": []}
for tag, i in [("native", c) for c in cases] + [("square384", cases[0])]:
    if tag == "native":
        x = torch.from_numpy(N.load_img_native(i))[None]
    else:
        x = torch.from_numpy(C.load_img_384(i))[None]
    H, W = x.shape[2], x.shape[3]
    with torch.no_grad():
        feats = C.iie_features(model, x)
    cb = N.native_centerbias(prior, H, W)
    geo = N.Geometry(H, W)
    g64 = torch.from_numpy(np.ascontiguousarray(D_ms64[pos[i]])).float()
    # released module path
    model.zero_grad(set_to_none=True)
    logd_rel = C.forward_cached(model, feats, cb)[:, 0]                       # (1,H,W)
    l_rel = released_loss(logd_rel, g64); l_rel.backward()
    g_rel = grads_module(model)
    # stacked native path (custom backward) and plain autograd
    res = {}
    for plain in (False, True):
        st = N.make_native_stacked(model, plain_autograd=plain)
        cbd = geo.dcb(cb)
        lp = st.logP(feats, cbd, geo)
        l = N.klcc64_native(lp, geo, g64); l.backward()
        full = lp[:, geo.uy][:, :, geo.ux]
        res[plain] = {"loss": float(l), "grad": grads_stacked(st), "logd": full.detach().double()}
    e = {"id": i, "kind": tag, "H": H, "W": W, "readout_hw": [geo.h, geo.w], "finaliser_hw": [geo.H2, geo.W2],
         "multiplicity_values": sorted(set(geo.wt.flatten().tolist())),
         "max_abs_logd_stacked_vs_released": float((res[False]["logd"] - logd_rel.detach().double()).abs().max()),
         "loss_released": float(l_rel), "loss_stacked": res[False]["loss"], "loss_stacked_plain": res[True]["loss"],
         "grad_rel_err_stacked_vs_released": float((res[False]["grad"] - g_rel).norm() / g_rel.norm()),
         "grad_rel_err_customBW_vs_plain": float((res[False]["grad"] - res[True]["grad"]).norm() / res[True]["grad"].norm()),
         "mass_full_map": float(torch.exp(res[False]["logd"]).sum())}
    print(json.dumps(e), flush=True)
    rep["cases"].append(e)

# write-back round trip on the last native case
i = cases[0]
x = torch.from_numpy(N.load_img_native(i))[None]; H, W = x.shape[2:]
with torch.no_grad():
    feats = C.iie_features(model, x)
cb = N.native_centerbias(prior, H, W); geo = N.Geometry(H, W)
st = N.make_native_stacked(model)
with torch.no_grad():
    for p_ in st.parameters():
        p_.mul_(1.01)
    m2 = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
    st.write_back(m2)
    a2 = C.forward_cached(m2, feats, cb)[:, 0].double()
    b2 = st.full_logd(feats, geo.dcb(cb), geo).double()
rep["write_back_max_abs_logd"] = float((a2 - b2).abs().max())
rep["trainable_params_stacked"] = int(sum(p.numel() for p in st.parameters()))
rep["seconds"] = time.time() - t0
rep["pass"] = bool(all(c["max_abs_logd_stacked_vs_released"] < 1e-3 and c["grad_rel_err_stacked_vs_released"] < 1e-4
                       and abs(c["loss_released"] - c["loss_stacked"]) < 1e-5 for c in rep["cases"])
                   and rep["write_back_max_abs_logd"] < 1e-3 and rep["trainable_params_stacked"] == 3059160)
print(json.dumps({k: v for k, v in rep.items() if k != "cases"}, indent=1))
json.dump(rep, open(os.path.join(N.NCACHE, "verify_native.json"), "w"), indent=1)
