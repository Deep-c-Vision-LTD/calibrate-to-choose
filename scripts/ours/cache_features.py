"""
Cache frozen DINOv2-base features for all 1000 ADD1000 images once (CPU), exactly as
kfold_train_eval.py does in memory: RGB, bilinear resize to 448x448, ImageNet mean/std,
last_hidden_state patch tokens (32x32x768) stored as float16. Also stores the CLS token
and the mean patch token (float32) for the grouping analysis.

Order = ids sorted numerically (identical to splits/kfold_seed0.json "ids_sorted").
Also checks that the warm-start checkpoint carries no backbone weights that differ from
the public DINOv2-base weights (the internal scripts loaded it into the feature model).

  python -I cache_features.py --ckpt <warm_start.pt>
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch
from PIL import Image

ROOT = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root (CBYC_ROOT overrides)
ADD1000_DIR = os.environ.get("ADD1000_DIR", os.path.join(ROOT, "data", "ADD1000"))   # holds flat/{images,maps}
FLAT = os.path.join(ADD1000_DIR, "flat")
SPLITS = os.path.join(ROOT, "splits", "kfold_seed0.json")
OUT = os.path.join(ROOT, "cache", "ours")
SIZE = 448
IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from model_aspp import ASPPSaliency
    t0 = time.time()
    ids = json.load(open(SPLITS))["ids_sorted"]
    assert len(ids) == 1000
    model = ASPPSaliency().eval()
    report = {}
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        own = model.state_dict()
        bb = [k for k in sd if k.startswith("backbone.")]
        diffs = [float((sd[k].float() - own[k].float()).abs().max()) for k in bb if k in own]
        report["ckpt_backbone_keys"] = len(bb)
        report["ckpt_backbone_max_abs_diff_vs_public"] = max(diffs) if diffs else None
        report["ckpt_readout_keys"] = sorted(k for k in sd if not k.startswith("backbone."))
        print("ckpt backbone keys:", len(bb), "max abs diff vs public DINOv2-base:",
              report["ckpt_backbone_max_abs_diff_vs_public"], flush=True)
    os.makedirs(OUT, exist_ok=True)
    feats = np.lib.format.open_memmap(os.path.join(OUT, "feats_dinov2b_448_fp16.npy"), mode="w+",
                                      dtype=np.float16, shape=(len(ids), 768, 32, 32))
    cls = np.zeros((len(ids), 768), np.float32)
    meanp = np.zeros((len(ids), 768), np.float32)
    with torch.no_grad():
        for n, i in enumerate(ids):
            im = Image.open(os.path.join(FLAT, "images", f"{i}.jpg")).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            x = torch.from_numpy(((np.asarray(im, np.float32) / 255 - IM_M) / IM_S).transpose(2, 0, 1))[None]
            out = model.backbone(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
            tok = out[:, 1:, :]
            feats[n] = tok.transpose(1, 2).reshape(1, 768, 32, 32)[0].half().numpy()
            cls[n] = out[0, 0].numpy()
            meanp[n] = tok[0].mean(0).numpy()
            if (n + 1) % 100 == 0:
                print(f"  {n + 1}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
    feats.flush()
    np.save(os.path.join(OUT, "cls_dinov2b_448.npy"), cls)
    np.save(os.path.join(OUT, "meanpatch_dinov2b_448.npy"), meanp)
    report.update({"ids": ids, "size": SIZE, "device": "cpu", "dtype_feats": "float16",
                   "wall_s": time.time() - t0, "cmd": " ".join(sys.argv)})
    json.dump(report, open(os.path.join(OUT, "cache_meta.json"), "w"), indent=1)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
