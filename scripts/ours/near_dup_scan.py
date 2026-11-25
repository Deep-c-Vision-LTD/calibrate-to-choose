"""
Near-duplicate scan over the 1000 ADD1000 images with perceptual hashes (no external
hashing package: the imagehash reference algorithms are re-implemented here with
PIL + scipy so the scan has no extra dependency).

  phash  : grey, LANCZOS resize to (4h x 4h), 2-D DCT-II, top-left h x h block, bit = coef > median
  dhash  : grey, LANCZOS resize to (h+1) x h, bit = pixel > right neighbour
  ahash  : grey, LANCZOS resize to h x h, bit = pixel > mean
with h = 8 (64-bit) and h = 16 (256-bit). For each hash it reports the number of pairs at
Hamming distance <= T for a range of T and how many of those pairs straddle a test-fold
boundary of the fixed seed-0 folds. Writes a scan table; the pair list is written by
make_splits.py once the hash/threshold that reproduces the earlier record is identified.

  python -I near_dup_scan.py
"""
import sys
sys.dont_write_bytecode = True
import os, json, time
import numpy as np
from PIL import Image
from scipy.fftpack import dct

ROOT = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root (CBYC_ROOT overrides)
ADD1000_DIR = os.environ.get("ADD1000_DIR", os.path.join(ROOT, "data", "ADD1000"))   # holds flat/{images,maps}
FLAT = os.path.join(ADD1000_DIR, "flat")
SPL = json.load(open(os.path.join(ROOT, "splits", "kfold_seed0.json")))
OUT = os.path.join(ROOT, "cache", "ours", "near_dup_scan.json")


def grey(p):
    return Image.open(p).convert("L")


def phash(im, h):
    a = np.asarray(im.resize((4 * h, 4 * h), Image.LANCZOS), np.float64)
    d = dct(dct(a, axis=0), axis=1)[:h, :h]
    return (d > np.median(d)).ravel()


def dhash(im, h):
    a = np.asarray(im.resize((h + 1, h), Image.LANCZOS), np.float64)
    return (a[:, 1:] > a[:, :-1]).ravel()


def ahash(im, h):
    a = np.asarray(im.resize((h, h), Image.LANCZOS), np.float64)
    return (a > a.mean()).ravel()


def main():
    t0 = time.time()
    ids = SPL["ids_sorted"]
    fold_of = {i: k for k, f in enumerate(SPL["folds"]) for i in f}
    H = {f"{name}{h}": [] for name in ("phash", "dhash", "ahash") for h in (8, 16)}
    for i in ids:
        im = grey(os.path.join(FLAT, "images", f"{i}.jpg"))
        for h in (8, 16):
            H[f"phash{h}"].append(phash(im, h)); H[f"dhash{h}"].append(dhash(im, h)); H[f"ahash{h}"].append(ahash(im, h))
    res = {}
    iu = np.triu_indices(len(ids), 1)
    for k, v in H.items():
        B = np.array(v, np.uint8)
        D = (B[:, None, :] != B[None, :, :]).sum(-1)
        np.save(os.path.join(ROOT, "cache", "ours", f"hamming_{k}.npy"), D.astype(np.int16))
        d = D[iu]
        tab = {}
        for T in list(range(0, 41)):
            m = d <= T
            a, b = iu[0][m], iu[1][m]
            cross = int(sum(fold_of[ids[x]] != fold_of[ids[y]] for x, y in zip(a, b)))
            tab[T] = {"pairs": int(m.sum()), "cross_fold": cross}
        res[k] = {"bits": int(B.shape[1]), "table": tab}
        print(k, B.shape[1], {T: (tab[T]["pairs"], tab[T]["cross_fold"]) for T in (5, 10, 15, 20, 24, 25, 26, 27, 30)}, flush=True)
    json.dump(res, open(OUT, "w"), indent=0)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
