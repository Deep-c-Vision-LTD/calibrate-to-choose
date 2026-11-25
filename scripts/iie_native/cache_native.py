"""
Cache the frozen DeepGaze IIE backbone outputs (readout inputs) for all 1000 ADD1000 images at NATIVE
resolution: image at its original size (PIL RGB, 0..255 float32, no resize) -> the model's own nearest 1/2
downsampling -> 4 frozen backbones -> features nearest-resized to the readout grid ceil(H/32) x ceil(W/32)
(iie_common.iie_features, identical to the released DeepGazeIIIMixture.forward). 9072 channels, float32,
one flat file (cache/iie_native/native_feats_float32.bin) with an index of offsets and sizes. No labels
are read. CPU only. Resumable (done list per shard, flushed every 10 images).

  python -I cache_native.py --threads 6                       # single process
  python -I cache_native.py --threads 3 --shard 0 --nshards 2 # two processes on disjoint ids
  python -I cache_native.py --finalize                        # mark complete once every shard is done
"""
import os, sys, json, time, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import native_common as N
C = N.C

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--finalize", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="stop after this many new images (timing tests)")
a = ap.parse_args()
torch.set_num_threads(a.threads)
os.makedirs(N.NCACHE, exist_ok=True)
LOG = os.path.join(C.KBS, "logs", "iie_native_cache.log")


def log(s):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [shard {a.shard}/{a.nshards}] {s}"
    print(line, flush=True); open(LOG, "a").write(line + "\n")


ids, _ = C.load_split()
# ---- index (deterministic from the image sizes; written once, re-checked on every start)
recs, off = [], 0
for i in ids:
    W, H = N.image_size(i)
    h, w = N.readout_hw(H, W)
    recs.append({"id": i, "H": H, "W": W, "h": h, "w": w, "offset": off})
    off += N.CTOT * h * w
n_floats = off
if os.path.exists(N.INDEX):
    old = json.load(open(N.INDEX))
    assert old["images"] == recs and old["n_floats"] == n_floats, "index mismatch"
    index = old
else:
    index = {"ids": ids, "n_floats": n_floats, "channels": N.CTOT, "dtype": "float32", "images": recs,
             "complete": False,
             "input": "native image (PIL RGB, no resize, 0..255 float32); DeepGaze IIE's own nearest 1/2 "
                      "downsampling (F.interpolate scale_factor 0.5, recompute_scale_factor=False) before the "
                      "backbones; features nearest-resized to ceil(H/32) x ceil(W/32); iie_common.iie_features",
             "layout": "image k occupies floats [offset, offset + 9072*h*w) as a C-contiguous (9072, h, w) array"}
    json.dump(index, open(N.INDEX, "w"))

done_files = [os.path.join(N.NCACHE, f"native_feats_done_shard{s}of{a.nshards}.json") for s in range(a.nshards)]
if a.finalize:
    done = set()
    for f in sorted(os.listdir(N.NCACHE)):
        if f.startswith("native_feats_done_shard"):
            done |= set(json.load(open(os.path.join(N.NCACHE, f)))["done"])
    missing = [i for i in ids if i not in done]
    if missing:
        log(f"finalize: {len(missing)} images missing, e.g. {missing[:5]}"); sys.exit(1)
    index["complete"] = True
    index["finalized"] = datetime.datetime.now().isoformat()
    json.dump(index, open(N.INDEX, "w"))
    log(f"finalize: complete, {n_floats * 4 / 1e9:.2f} GB"); sys.exit(0)

mode = "r+" if os.path.exists(N.FEAT_BIN) else "w+"
arr = np.memmap(N.FEAT_BIN, dtype=np.float32, mode=mode, shape=(n_floats,))
dfile = done_files[a.shard]
dd = json.load(open(dfile)) if os.path.exists(dfile) else {"done": [], "seconds": {}}
done = set(dd["done"])
mine = [r for k, r in enumerate(recs) if k % a.nshards == a.shard and r["id"] not in done]
log(f"start: {len(mine)} images to do in this shard, threads {a.threads}, file mode {mode}")
import deepgaze_pytorch
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
t0 = time.time(); n_new = 0
with torch.no_grad():
    for r in mine:
        t1 = time.time()
        x = torch.from_numpy(N.load_img_native(r["id"]))[None]
        assert x.shape[2] == r["H"] and x.shape[3] == r["W"]
        f = torch.cat(C.iie_features(model, x), dim=1)[0].numpy()
        assert f.shape == (N.CTOT, r["h"], r["w"]), f.shape
        arr[r["offset"]:r["offset"] + f.size] = f.reshape(-1)
        dd["done"].append(r["id"]); dd["seconds"][r["id"]] = time.time() - t1
        n_new += 1
        if n_new % 10 == 0 or n_new == len(mine) or (a.limit and n_new >= a.limit):
            arr.flush()
            json.dump(dd, open(dfile + ".tmp", "w")); os.replace(dfile + ".tmp", dfile)
            log(f"{len(dd['done'])} done in shard ({n_new} new, {time.time() - t0:.0f}s, "
                f"{(time.time() - t0) / n_new:.2f} s/img)")
        if a.limit and n_new >= a.limit:
            break
arr.flush(); del arr
log(f"exit: {n_new} new images in {time.time() - t0:.0f}s")
