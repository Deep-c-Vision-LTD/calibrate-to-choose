"""Cache SUM's (condition-free) encoder outputs for all ADD1000 images, for the
decoder-only adaptation row (encoder frozen in eval mode, so its outputs are fixed).
Saves float32 memmaps: enc_skip{0..3}.npy (B,H,W,C layouts as produced) and enc_x.npy.

python -I cache_encoder.py --scripts DIR --code DIR --weights DIR --data DIR --split FILE \
    --out_dir cache/sum_enc [--device cpu]
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
for a in ("scripts", "code", "weights", "data", "split", "out_dir"):
    ap.add_argument(f"--{a}", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=6)
args = ap.parse_args()
sys.path.insert(0, args.scripts)
import numpy as np, torch
import sum_common as sc

torch.set_num_threads(args.threads)
dev = args.device
model, _ = sc.load_sum(args.code, args.weights, dev)
model.eval(); net = model.salu_mamba
ids = json.load(open(args.split))["ids_sorted"]; n = len(ids)
os.makedirs(args.out_dir, exist_ok=True)
shapes = {"enc_skip0": (64, 64, 96), "enc_skip1": (32, 32, 192), "enc_skip2": (16, 16, 384),
          "enc_skip3": (8, 8, 768), "enc_x": (8, 8, 768)}
done_p = os.path.join(args.out_dir, "enc_done.npy")
mode = "r+" if os.path.exists(done_p) else "w+"
mm = {k: np.lib.format.open_memmap(os.path.join(args.out_dir, f"{k}.npy"), mode=mode, dtype=np.float32,
                                   shape=(n,) + s) for k, s in shapes.items()}
done = np.load(done_p) if mode == "r+" else np.zeros(n, bool)
json.dump({"ids": ids, "shapes": shapes}, open(os.path.join(args.out_dir, "enc_index.json"), "w"))
c = torch.zeros(1, 4, device=dev); c[0, 2] = 1
t0 = time.time(); k_done = 0
for k, iid in enumerate(ids):
    if done[k]:
        continue
    x = sc.load_image_tensor(os.path.join(args.data, "images", f"{iid}.jpg"))[None].to(dev)
    with torch.no_grad():
        feats, skips = net.forward_features(x, c)
    for j, s in enumerate(skips):
        mm[f"enc_skip{j}"][k] = s[0].float().cpu().numpy()
    mm["enc_x"][k] = feats[0].float().cpu().numpy()
    done[k] = True; k_done += 1
    if k_done % 50 == 0:
        for m in mm.values():
            m.flush()
        np.save(done_p, done)
        el = time.time() - t0
        print(f"{k + 1}/{n} {el / k_done:.2f} s/img eta {(n - done.sum()) * el / k_done / 60:.1f} min", flush=True)
for m in mm.values():
    m.flush()
np.save(done_p, done)
print(f"DONE {int(done.sum())} images, {time.time() - t0:.0f}s this run", flush=True)
