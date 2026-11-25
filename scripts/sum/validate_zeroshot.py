"""Row (1): reproduce the stored per-image zero-shot SUM CC (token 2, e-commerce) on a
subset of ADD1000 images, using the pure-PyTorch scan, and check the chunked scan
against the sequential reference on the full model.

python -I validate_zeroshot.py --scripts DIR --code DIR --weights DIR --data DIR \
    --stored results/sum_add1000_per_image.json --n 20 --device cpu --out FILE
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
for a in ("scripts", "code", "weights", "data", "stored", "out"):
    ap.add_argument(f"--{a}", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=6)
args = ap.parse_args()
sys.path.insert(0, args.scripts)

import numpy as np, torch
import sum_common as sc

torch.set_num_threads(args.threads)
dev = args.device
model, cfg = sc.load_sum(args.code, args.weights, dev, scan="chunked")
model.eval()
npar = sc.n_params(model)
print(f"SUM params: {npar:,}  config: {cfg}", flush=True)

stored = json.load(open(args.stored))
ids = sorted(stored.keys(), key=int)
rng = np.random.default_rng(123)
sel = sorted(rng.choice(ids, size=args.n, replace=False).tolist(), key=int)

rows = []
t0 = time.time()
for k, iid in enumerate(sel):
    x = sc.load_image_tensor(os.path.join(args.data, "images", f"{iid}.jpg"))[None].to(dev)
    cond = torch.zeros(1, 4, device=dev); cond[0, 2] = 1
    with torch.no_grad():
        p = model(x, cond)[0, 0].float().cpu().numpy()
    p384 = sc.resize_map(p, sc.S_EVAL)
    g384 = sc.load_gt(os.path.join(args.data, "maps", f"{iid}.png"), sc.S_EVAL)
    m = sc.metrics(p384, g384)
    ref = stored[iid]["2"]
    row = {"id": iid, "CC": m["CC"], "SIM": m["SIM"], "KLD": m["KLD"],
           "CC_stored": ref[0], "SIM_stored": ref[1], "KLD_stored": ref[2],
           "dCC": m["CC"] - ref[0], "dSIM": m["SIM"] - ref[1], "dKLD": m["KLD"] - ref[2]}
    if k < 2:   # full-model check of the chunked scan against the sequential reference
        sc.set_scan("seq")
        with torch.no_grad():
            pr = model(x, cond)[0, 0].float().cpu().numpy()
        sc.set_scan("chunked")
        row["max_abs_diff_chunk_vs_seq"] = float(np.abs(pr - p).max())
    rows.append(row)
    print(f"{iid}: CC {m['CC']:.5f} stored {ref[0]:.5f} d={row['dCC']:+.5f}  "
          f"SIM d={row['dSIM']:+.5f} KLD d={row['dKLD']:+.5f}", flush=True)

d = np.array([r["dCC"] for r in rows])
out = {"row": "sum_validate_t2", "n": len(rows), "device": dev, "params": npar,
       "max_abs_dCC": float(np.abs(d).max()), "mean_abs_dCC": float(np.abs(d).mean()),
       "max_abs_dSIM": float(np.abs([r["dSIM"] for r in rows]).max()),
       "max_abs_dKLD": float(np.abs([r["dKLD"] for r in rows]).max()),
       "mean_CC_ours": float(np.mean([r["CC"] for r in rows])),
       "mean_CC_stored": float(np.mean([r["CC_stored"] for r in rows])),
       "seconds": time.time() - t0, "rows": rows,
       "note": "stored values: published run with the authors' CUDA selective-scan kernel; "
               "here: pure-PyTorch chunked scan, same preprocessing/resizing/metrics."}
json.dump(out, open(args.out, "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
