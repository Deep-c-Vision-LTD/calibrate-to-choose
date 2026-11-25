"""Row (2) inference: SUM zero-shot maps for all ADD1000 images and all four domain
tokens (0 natural/mouse, 1 natural/eye, 2 e-commerce, 3 UI), saved raw at SUM's native
256 x 256 output (sigmoid output, float32) so every scoring/calibration row reuses them.

SUM's encoder is condition-free (only the decoder's C-VSS blocks read the token), so the
encoder runs once per image and the decoder once per token; --check verifies on a few
images that this equals the model's own forward() for every token.

python -I infer_zeroshot.py --scripts DIR --code DIR --weights DIR --data DIR \
    --split splits/kfold_seed0.json --out_dir cache/sum_maps [--device cpu] [--check 3]
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
for a in ("scripts", "code", "weights", "data", "split", "out_dir"):
    ap.add_argument(f"--{a}", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--check", type=int, default=3)
args = ap.parse_args()
sys.path.insert(0, args.scripts)

import numpy as np, torch
import sum_common as sc

torch.set_num_threads(args.threads)
dev = args.device
model, cfg = sc.load_sum(args.code, args.weights, dev, scan="chunked")
model.eval()
net = model.salu_mamba
ids = json.load(open(args.split))["ids_sorted"]
n = len(ids)
os.makedirs(args.out_dir, exist_ok=True)

paths = [os.path.join(args.out_dir, f"sum_zeroshot_t{t}_256.npy") for t in range(4)]
done_path = os.path.join(args.out_dir, "sum_zeroshot_done.npy")
mode = "r+" if all(os.path.exists(p) for p in paths) and os.path.exists(done_path) else "w+"
maps = [np.lib.format.open_memmap(p, mode=mode, dtype=np.float32, shape=(n, 256, 256)) for p in paths]
done = np.load(done_path) if mode == "r+" else np.zeros(n, bool)
json.dump({"ids": ids, "tokens": [0, 1, 2, 3], "shape": [n, 256, 256], "dtype": "float32",
           "content": "SUM sigmoid output at native 256x256 input/output, released weights "
                      "(hf safe-models/SUM), pure-PyTorch chunked selective scan"},
          open(os.path.join(args.out_dir, "sum_zeroshot_index.json"), "w"), indent=1)

onehot = torch.eye(4, device=dev)


@torch.no_grad()
def four_tokens(x):
    feats, skips = net.forward_features(x, onehot[:1])
    f4 = feats.expand(4, *feats.shape[1:])
    s4 = [s.expand(4, *s.shape[1:]) for s in skips]
    y = net.forward_features_up(f4, s4, onehot)
    return torch.sigmoid(net.forward_final(y))[:, 0].float().cpu().numpy()


check = []
t0 = time.time(); k_done = 0
for k, iid in enumerate(ids):
    if done[k]:
        continue
    x = sc.load_image_tensor(os.path.join(args.data, "images", f"{iid}.jpg"))[None].to(dev)
    p4 = four_tokens(x)
    if len(check) < args.check:
        with torch.no_grad():
            ref = model(x.expand(4, -1, -1, -1), onehot)[:, 0].float().cpu().numpy()
        check.append(float(np.abs(ref - p4).max()))
        print(f"shared-encoder check {iid}: max abs diff {check[-1]:.2e}", flush=True)
    for t in range(4):
        maps[t][k] = p4[t]
    done[k] = True; k_done += 1
    if k_done % 50 == 0:
        for m in maps:
            m.flush()
        np.save(done_path, done)
        el = time.time() - t0
        print(f"{k + 1}/{n}  {el / k_done:.2f} s/img  eta {(n - done.sum()) * el / k_done / 60:.1f} min",
              flush=True)
for m in maps:
    m.flush()
np.save(done_path, done)
el = time.time() - t0
meta = {"n_done": int(done.sum()), "seconds_this_run": el, "s_per_image_4_tokens": el / max(k_done, 1),
        "device": dev, "threads": args.threads, "shared_encoder_check_max_abs_diff": check,
        "params": sc.n_params(model)}
json.dump(meta, open(os.path.join(args.out_dir, "sum_zeroshot_infer_meta.json"), "w"), indent=1)
print("DONE", json.dumps(meta), flush=True)
