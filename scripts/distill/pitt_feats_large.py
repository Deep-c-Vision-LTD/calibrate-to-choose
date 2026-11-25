"""
DINOv2-large (448) patch tokens for the Pitt pool, for the optional large student (distill_pitt_L).
Same preprocessing as track A's cache (scripts/trackA/cache_feats_a.py --backbone large --size 448
--store raw) and scripts/ours/cache_features.py: RGB, PIL bilinear 448x448, ImageNet mean/std,
last_hidden_state[:, 1:], CPU float32 forward, float16, 32x32x1024. Order = pool order.

  python -I pitt_feats_large.py
"""
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import time, json, argparse
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_common as DC                     # noqa: E402

IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-large").eval()
    files = DC.pool(); n = len(files)
    out = os.path.join(DC.CD, "pitt_feats_dinov2l_448_fp16.npy")
    mm = np.lib.format.open_memmap(out + ".part.npy", mode="w+", dtype=np.float16, shape=(n, 1024, 32, 32))
    t0 = time.time()
    with torch.no_grad():
        for k, f in enumerate(files):
            im = Image.open(os.path.join(DC.PITT, f)).convert("RGB").resize((448, 448), Image.BILINEAR)
            x = torch.from_numpy(((np.asarray(im, np.float32) / 255 - IM_M) / IM_S).transpose(2, 0, 1))[None]
            tok = model(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, 1:, :]
            mm[k] = tok.transpose(1, 2).reshape(1, 1024, 32, 32)[0].half().numpy()
            if (k + 1) % 250 == 0:
                print(f"  {k + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
    mm.flush(); del mm
    os.replace(out + ".part.npy", out)
    json.dump({"n": n, "order": "pool order (cache/distill/pitt_subset.json pool_positions)", "wall_s": time.time() - t0,
               "preprocess": "as scripts/trackA/cache_feats_a.py large 448 raw"}, open(out.replace(".npy", "_meta.json"), "w"), indent=1)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
