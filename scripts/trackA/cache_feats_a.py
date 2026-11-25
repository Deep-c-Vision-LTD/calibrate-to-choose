"""
Track A feature caches (backbone / resolution sweep for our readout).

Copy of scripts/ours/cache_features.py, generalised to a DINOv2 size and input resolution.
Identical preprocessing: RGB, PIL bilinear resize to SIZE x SIZE, ImageNet mean/std,
last_hidden_state patch tokens (CLS dropped), CPU, float32 forward, stored float16,
order = ids_sorted of splits/kfold_seed0.json.

  --store raw      : tokens on their native grid (SIZE/14 x SIZE/14)
  --store to32     : tokens resized to the 32 x 32 grid of the 448 input (multi-scale rows);
                     resize rule fixed a priori: F.interpolate(mode="bilinear", antialias=True,
                     align_corners=False) on the float32 tokens, then cast to float16
  --store both     : both files

  python -I cache_feats_a.py --backbone large --size 448 --store raw
  python -I cache_feats_a.py --backbone base --size 672 --store to32
  python -I cache_feats_a.py --check-base448 8      # bit-identity check vs cache/ours
"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import json, time, argparse
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ROOT = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root (CBYC_ROOT overrides)
ADD1000_DIR = os.environ.get("ADD1000_DIR", os.path.join(ROOT, "data", "ADD1000"))   # holds flat/{images,maps}
KB = ROOT
FLAT = os.path.join(ADD1000_DIR, "flat")
SPLITS = os.path.join(KB, "splits", "kfold_seed0.json")
OUT = os.path.join(KB, "cache", "trackA")
IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)
NAMES = {"base": "facebook/dinov2-base", "large": "facebook/dinov2-large", "small": "facebook/dinov2-small"}
TAG = {"base": "b", "large": "l", "small": "s"}


def prep(i, size):
    im = Image.open(os.path.join(FLAT, "images", f"{i}.jpg")).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(((np.asarray(im, np.float32) / 255 - IM_M) / IM_S).transpose(2, 0, 1))[None]


def tokens(model, x):
    out = model(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
    tok = out[:, 1:, :]
    B, N, C = tok.shape
    h = int(round(N ** 0.5)); assert h * h == N
    return tok.transpose(1, 2).reshape(B, C, h, h)


def to32(f):
    return F.interpolate(f, size=(32, 32), mode="bilinear", antialias=True, align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(NAMES), default="base")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--store", choices=["raw", "to32", "both"], default="raw")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--check-base448", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModel
    ids = json.load(open(SPLITS))["ids_sorted"]
    assert len(ids) == 1000
    t0 = time.time()
    if a.check_base448:
        model = AutoModel.from_pretrained(NAMES["base"]).eval()
        ref = np.load(os.path.join(KB, "cache", "ours", "feats_dinov2b_448_fp16.npy"), mmap_mode="r")
        with torch.no_grad():
            for n in range(a.check_base448):
                f = tokens(model, prep(ids[n], 448))[0].half().numpy()
                d = np.abs(f.astype(np.float32) - ref[n].astype(np.float32))
                print(f"id {ids[n]}: max|diff|={d.max():.3g} n_diff={int((d > 0).sum())} identical={np.array_equal(f, ref[n])}", flush=True)
        return
    model = AutoModel.from_pretrained(NAMES[a.backbone]).eval()
    C = model.config.hidden_size
    g = a.size // 14; assert g * 14 == a.size
    os.makedirs(OUT, exist_ok=True)
    base = f"feats_dinov2{TAG[a.backbone]}_{a.size}"
    files = {}
    if a.store in ("raw", "both"):
        files["raw"] = (os.path.join(OUT, f"{base}_fp16.npy"), (len(ids), C, g, g))
    if a.store in ("to32", "both"):
        files["to32"] = (os.path.join(OUT, f"{base}to32_fp16.npy"), (len(ids), C, 32, 32))
    mm = {k: np.lib.format.open_memmap(p + ".part", mode="w+", dtype=np.float16, shape=s) for k, (p, s) in files.items()}
    with torch.no_grad():
        for n, i in enumerate(ids):
            f = tokens(model, prep(i, a.size))
            if "raw" in mm:
                mm["raw"][n] = f[0].half().numpy()
            if "to32" in mm:
                mm["to32"][n] = to32(f)[0].half().numpy()
            if (n + 1) % 100 == 0:
                print(f"  {n + 1}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
    for k, (p, s) in files.items():
        mm[k].flush(); del mm[k]
        os.replace(p + ".part", p)
    meta = {"backbone": NAMES[a.backbone], "size": a.size, "grid": g, "channels": C, "store": a.store,
            "files": {k: os.path.basename(p) for k, (p, s) in files.items()},
            "to32_rule": "F.interpolate(bilinear, antialias=True, align_corners=False) on float32 tokens, then float16",
            "preprocess": "PIL RGB bilinear resize to size x size, ImageNet mean/std, last_hidden_state[:,1:], CPU float32 forward, stored float16",
            "device": "cpu", "threads": a.threads, "order": "ids_sorted of splits/kfold_seed0.json",
            "wall_s": time.time() - t0, "torch": torch.__version__, "cmd": " ".join(sys.argv)}
    json.dump(meta, open(os.path.join(OUT, f"{base}_{a.store}_meta.json"), "w"), indent=1)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
