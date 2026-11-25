"""
Cache DeepGaze IIE readout inputs for all 1000 ADD1000 images (ids_sorted order) at
384 x 384 input: 4 backbones, 2048+2416+2048+2560 = 9072 channels on the 12 x 12 readout
grid, float32 (verify_cache.py showed the fp32 cached path is bit-exact w.r.t. the full
forward pass). Also caches the GT maps (gt_cache.npz). CPU only.

  python -I cache_features.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from numpy.lib.format import open_memmap
import iie_common as C

torch.set_num_threads(int(os.environ.get("NTHREADS", "6")))
import deepgaze_pytorch

ids, _ = C.load_split()
t0 = time.time()
gt = C.gt_cache()
print(f"GT cached ({time.time() - t0:.1f}s)", flush=True)
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
path = os.path.join(C.CACHE, "iie384_feats_float32.npy")
tmp = path + ".partial.npy"
arr = open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(ids), sum(C.BACKBONE_CH), C.G, C.G))
B = 4
with torch.no_grad():
    for k in range(0, len(ids), B):
        bt = ids[k:k + B]
        x = torch.from_numpy(np.stack([C.load_img_384(i) for i in bt]))
        f = torch.cat(C.iie_features(model, x), dim=1).numpy()
        arr[k:k + len(bt)] = f
        if (k // B) % 25 == 0:
            el = time.time() - t0
            print(f"{k + len(bt)}/{len(ids)}  {el:.0f}s", flush=True)
arr.flush(); del arr
os.replace(tmp, path)
json.dump({"ids": ids, "shape": [len(ids), sum(C.BACKBONE_CH), C.G, C.G], "dtype": "float32",
           "input": "PIL RGB resize to 384x384 bilinear, 0..255 float32 (as eval_deepgaze_iie.py)",
           "seconds": time.time() - t0},
          open(os.path.join(C.CACHE, "iie384_feats_meta.json"), "w"), indent=1)
print(f"done {time.time() - t0:.0f}s -> {path}", flush=True)
