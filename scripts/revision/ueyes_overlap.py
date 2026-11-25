"""
KBS revision (24 Nov 2026). Overlap check between ADD1000 and UEyes (one of SUM's training sets) with the
paper's own near-duplicate detector: cosine similarity of L2-normalised mean DINOv2-base patch tokens
(448 x 448, last_hidden_state, CLS dropped) >= 0.94, exactly as cache/ours/meanpatch_dinov2b_448.npy was
made (scripts/ours/cache_features.py). Writes results/ueyes_overlap.json (top pairs, count >= 0.94 and
>= 0.925, max cosine).
  python -I ueyes_overlap.py [--threads 4]
"""
import argparse, glob, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ours"))
import numpy as np, torch
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=4); a = ap.parse_args()
torch.set_num_threads(a.threads)
from model_aspp import ASPPSaliency
ROOT = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root (CBYC_ROOT overrides)
ADD1000_DIR = os.environ.get("ADD1000_DIR", os.path.join(ROOT, "data", "ADD1000"))   # holds flat/{images,maps}
UE = os.environ.get("UEYES_DIR", os.path.join(ROOT, "data", "UEyes", "images"))   # UEyes images (not redistributed)
OUTC = os.path.join(ROOT, "cache", "revision", "ueyes_meanpatch_dinov2b_448.npy")
SIZE = 448
IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)
files = sorted(glob.glob(os.path.join(UE, "*")))
files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
t0 = time.time()
if os.path.exists(OUTC):
    U = np.load(OUTC)
else:
    model = ASPPSaliency().eval()
    U = np.zeros((len(files), 768), np.float32)
    with torch.no_grad():
        for n, f in enumerate(files):
            im = Image.open(f).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            x = torch.from_numpy(((np.asarray(im, np.float32) / 255 - IM_M) / IM_S).transpose(2, 0, 1))[None]
            out = model.backbone(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
            U[n] = out[0, 1:, :].mean(0).numpy()
            if (n + 1) % 200 == 0:
                print(f"{n + 1}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    np.save(OUTC, U)
A = np.load(os.path.join(ROOT, "cache", "ours", "meanpatch_dinov2b_448.npy"))
ids = json.load(open(os.path.join(ROOT, "splits", "kfold_seed0.json")))["ids_sorted"]
unit = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
S = unit(A) @ unit(U).T
flat = np.argsort(S.ravel())[::-1][:30]
top = [{"add1000_id": ids[k // S.shape[1]], "ueyes_file": os.path.basename(files[k % S.shape[1]]),
        "cos": round(float(S.ravel()[k]), 4)} for k in flat]
res = {"n_add1000": len(ids), "n_ueyes": len(files), "detector": "cosine of L2-normalised mean DINOv2-base patch tokens at 448",
       "max_cos": float(S.max()), "n_pairs_ge_0.94": int((S >= 0.94).sum()), "n_pairs_ge_0.925": int((S >= 0.925).sum()),
       "n_add1000_with_any_ge_0.94": int((S.max(1) >= 0.94).sum()),
       "reference_within_add1000_threshold": 0.94, "top30": top, "wall_s": time.time() - t0,
       "command": "python -I scripts/revision/ueyes_overlap.py --threads %d" % a.threads}
json.dump(res, open(os.path.join(ROOT, "results", "ueyes_overlap.json"), "w"), indent=1)
print({k: v for k, v in res.items() if k != "top30"}, top[:5], flush=True)
