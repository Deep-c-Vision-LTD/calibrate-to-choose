"""Spot-check the native feature cache: recompute 6 images (first/last of each shard order and the largest /
smallest image) with the same call and compare with the cached arrays; check no image block is all zeros.
Writes cache/iie_native/check_cache.json.   python -I check_cache.py [--threads 6]"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import native_common as N
C = N.C
ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=6); a = ap.parse_args()
torch.set_num_threads(a.threads)
import deepgaze_pytorch
NF = N.NativeFeats()
ids = list(NF.meta)
px = {i: NF.meta[i]["H"] * NF.meta[i]["W"] for i in ids}
pick = list(dict.fromkeys([ids[0], ids[1], ids[-1], ids[-2], max(ids, key=px.get), min(ids, key=px.get)]))
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
rep = {"checked": []}
with torch.no_grad():
    for i in pick:
        f = torch.cat(C.iie_features(model, torch.from_numpy(N.load_img_native(i))[None]), dim=1).numpy()
        c = NF.get(i).numpy()
        rep["checked"].append({"id": i, "HW": NF.hw(i), "max_abs": float(np.abs(f - c).max()),
                               "rel_norm": float(np.linalg.norm(f - c) / np.linalg.norm(c))})
zero = [i for i in ids if not np.any(NF.arr[NF.meta[i]["offset"]:NF.meta[i]["offset"] + 9072])]
rep["blocks_with_zero_first_channel_row"] = zero
rep["pass"] = bool(all(r["rel_norm"] < 1e-5 for r in rep["checked"]) and not zero)
print(json.dumps(rep, indent=1))
json.dump(rep, open(os.path.join(N.NCACHE, "check_cache.json"), "w"), indent=1)
