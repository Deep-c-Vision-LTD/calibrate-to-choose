"""Cache SUM's last decoder-stage output (input to the output head final_up + final_conv) for all
ADD1000 images, domain token 2, from the cached encoder outputs (cache_encoder.py). Encoder and decoder
in eval mode with the released weights, so these features are fixed. Used by ft_head.py (head-only
in-domain adaptation row). Also checks that head(features) reproduces the stored zero-shot token-2 maps.

python -I cache_decoder.py --scripts DIR --code DIR --weights DIR --enc_cache cache/sum_enc \
    --zs_maps cache/sum_maps/sum_zeroshot_t2_256.npy --out_dir cache/sum_dec [--threads 6]
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
for a in ("scripts", "code", "weights", "enc_cache", "zs_maps", "out_dir"):
    ap.add_argument(f"--{a}", required=True)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--token", type=int, default=2)
args = ap.parse_args()
sys.path.insert(0, args.scripts)
import numpy as np, torch
import sum_common as sc

torch.set_num_threads(args.threads)
model, _ = sc.load_sum(args.code, args.weights, "cpu")
model.eval(); net = model.salu_mamba
ei = json.load(open(os.path.join(args.enc_cache, "enc_index.json")))
ids = ei["ids"]; n = len(ids)
assert bool(np.load(os.path.join(args.enc_cache, "enc_done.npy")).all())
E = {k: np.load(os.path.join(args.enc_cache, f"{k}.npy"), mmap_mode="r")
     for k in ("enc_skip0", "enc_skip1", "enc_skip2", "enc_skip3", "enc_x")}
os.makedirs(args.out_dir, exist_ok=True)
done_p = os.path.join(args.out_dir, "dec_done.npy")
mode = "r+" if os.path.exists(done_p) else "w+"
F = np.lib.format.open_memmap(os.path.join(args.out_dir, "dec_feat.npy"), mode=mode, dtype=np.float32,
                              shape=(n, 64, 64, 96))
done = np.load(done_p) if mode == "r+" else np.zeros(n, bool)
json.dump({"ids": ids, "shape": [64, 64, 96], "token": args.token,
           "content": "SUM layers_up output (B,H,W,C) before final_up, eval mode, released weights"},
          open(os.path.join(args.out_dir, "dec_index.json"), "w"))
t0 = time.time(); k_done = 0
for j in range(0, n, args.bs):
    ix = [k for k in range(j, min(n, j + args.bs)) if not done[k]]
    if not ix:
        continue
    xe = torch.from_numpy(np.ascontiguousarray(E["enc_x"][ix]))
    sk = [torch.from_numpy(np.ascontiguousarray(E[f"enc_skip{q}"][ix])) for q in range(4)]
    c = torch.zeros(len(ix), 4); c[:, args.token] = 1
    with torch.no_grad():
        f = net.forward_features_up(xe, sk, c)
    F[ix] = f.float().numpy()
    done[ix] = True; k_done += len(ix)
    if k_done % 100 < args.bs:
        F.flush(); np.save(done_p, done)
        el = time.time() - t0
        print(f"{int(done.sum())}/{n} {el / k_done:.2f} s/img eta {(n - done.sum()) * el / k_done / 60:.1f} min",
              flush=True)
F.flush(); np.save(done_p, done)
# reproduction check against the stored zero-shot token maps (256x256 sigmoid outputs)
zs = np.load(args.zs_maps, mmap_mode="r")
chk = list(range(0, n, 97))[:11]
with torch.no_grad():
    p = torch.sigmoid(net.forward_final(torch.from_numpy(np.ascontiguousarray(F[chk])))[:, 0]).numpy()
diff = float(np.abs(p - zs[chk]).max())
meta = {"n": n, "seconds_this_run": time.time() - t0, "threads": args.threads, "check_ids": [ids[k] for k in chk],
        "max_abs_diff_vs_zeroshot_maps": diff,
        "head_params": int(sum(q.numel() for q in list(net.final_up.parameters()) + list(net.final_conv.parameters())))}
json.dump(meta, open(os.path.join(args.out_dir, "dec_meta.json"), "w"), indent=1)
print("DONE", json.dumps(meta), flush=True)
