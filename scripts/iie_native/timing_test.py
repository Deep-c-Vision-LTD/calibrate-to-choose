"""
Timing test for PLAN.md (before any training): fold 0, first training batch of the published shuffle stream
(8 images). Measures, at 3 and at 6 CPU threads:
  * native feature extraction (frozen backbones) per image,
  * one training step = 8 single-image forward/backward passes (gradient accumulation) + one Adam step,
  * evaluation per image through the released module path (forward_cached + resize to 384).
Writes cache/iie_native/timing_test.json. Nothing is trained beyond 3 throw-away steps; nothing is cached.

  python -I timing_test.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import native_common as N
C = N.C
sys.path.insert(0, os.path.join(C.KBS, "scripts", "iie"))
import iie_train as T
import deepgaze_pytorch

ids, folds = C.load_split()
orders = T.published_orders([[str(i) for i in f] for f in folds], 20, n_total=1000)
batch = orders[0][0][:8]
sizes = {i: N.image_size(i) for i in ids}
px_all = float(np.mean([w * h for w, h in sizes.values()]))
px_b = float(np.mean([sizes[i][0] * sizes[i][1] for i in batch]))
tr = [i for k in range(1, 5) for i in folds[k]]
D = T.Data()
prior = D.prior(tr)
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
rep = {"batch_ids": batch, "batch_sizes_WH": [sizes[i] for i in batch], "mean_px_batch": px_b,
       "mean_px_all1000": px_all, "px_ratio_all_over_batch": px_all / px_b, "by_threads": {}}
for th in (6, 3):
    torch.set_num_threads(th)
    r = {}
    feats = {}
    t = []
    with torch.no_grad():
        for i in batch:
            t0 = time.time()
            x = torch.from_numpy(N.load_img_native(i))[None]
            feats[i] = torch.cat(C.iie_features(model, x), dim=1)
            t.append(time.time() - t0)
    r["feat_s_per_img"] = float(np.mean(t[1:]))
    st = N.make_native_stacked(model)
    opt = torch.optim.Adam(st.parameters(), lr=1e-3)
    geos = {}
    ts = []
    for step in range(3):
        t0 = time.time()
        opt.zero_grad()
        tot = 0.0
        for i in batch:
            H, W = sizes[i][1], sizes[i][0]
            if (H, W) not in geos:
                geos[(H, W)] = N.Geometry(H, W)
            geo = geos[(H, W)]
            cbd = geo.dcb(N.native_centerbias(prior, H, W))
            g64 = torch.from_numpy(np.array(D.ms64[D.pos[i]], dtype=np.float32))
            lp = st.logP(C.split_channels(feats[i]), cbd, geo)
            l = N.klcc64_native(lp, geo, g64) / len(batch)
            l.backward(); tot += float(l)
        opt.step()
        ts.append(time.time() - t0)
    r["train_step_s"] = float(np.mean(ts[1:]))
    r["train_step_s_all"] = ts
    st.write_back(model)
    t0 = time.time()
    N.predict_native(model, lambda i: feats[i], batch, prior, lambda i: (sizes[i][1], sizes[i][0]))
    r["eval_s_per_img"] = (time.time() - t0) / len(batch)
    # restore pretrained readouts for the next thread setting
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
    ratio = px_all / px_b
    r["extrapolated"] = {
        "cache_1000_s": 1000 * r["feat_s_per_img"] * ratio,
        "train_per_fold_s": 2000 * r["train_step_s"] * ratio,
        "eval_per_fold_s": 400 * r["eval_s_per_img"] * ratio}
    rep["by_threads"][th] = r
    print(th, json.dumps(r), flush=True)
json.dump(rep, open(os.path.join(N.NCACHE, "timing_test.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
