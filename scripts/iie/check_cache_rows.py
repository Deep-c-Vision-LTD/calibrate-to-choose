"""Spot-check stored cache rows against a fresh feature pass (ids at positions 0, 499, 999)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import iie_common as C, deepgaze_pytorch
torch.set_num_threads(6)
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
ids, _ = C.load_split(); arr = C.load_feature_cache("float32")
rep = {}
with torch.no_grad():
    for p in (0, 499, 999):
        f = torch.cat(C.iie_features(model, torch.from_numpy(C.load_img_384(ids[p]))[None]), 1)[0].numpy()
        rep[ids[p]] = float(np.abs(f - arr[p]).max())
print(rep)
json.dump(rep, open(os.path.join(C.KBS, "logs", "iie_check_cache_rows.json"), "w"))
