"""
Track B, step 0 + 1: the unlabeled Pitt Ads pool and the leakage guard.

  * pool  : files of data/pitt_ads/imgs (PITT_ADS_DIR) sorted by name; a fixed random subset
            = numpy default_rng(2026).permutation(n_files)[:N] (N = 2500 by default).
  * feats : frozen DINOv2-base (448 px) patch tokens for every subset image, computed exactly as
            scripts/ours/cache_features.py (RGB, PIL bilinear 448x448, ImageNet mean/std,
            last_hidden_state patch tokens 32x32x768 stored as float16, batch 1, CPU, 6 threads),
            plus CLS and mean patch token (float32).
  * guard : a subset image is REMOVED if, against ANY of the 1000 ADD1000 images,
              (a) cosine of L2-normalised mean DINOv2-base patch tokens >= 0.94 (the primary
                  ADD1000 near-duplicate threshold of splits/near_dup_pairs.json), or
              (b) cosine of L2-normalised DINOv2-base CLS tokens >= 0.94 (same threshold), or
              (c) mean-patch cosine >= 0.925 (the extended threshold of near_dup_pairs.json used for
                  the pair-safe folds; extra conservative margin), or
              (d) pHash-256 Hamming distance <= 26 bits (pixel-level check; same hash and threshold
                  as the hash_reproduction_attempt of near_dup_pairs.json), or
              (e) identical file MD5 or identical MD5 of the decoded 64x64 RGB thumbnail.
            Counts per criterion and the union are logged; unreadable files are dropped and logged.

Outputs (cache/distill/): pitt_subset.json, pitt_feats_dinov2b_448_fp16.npy (subset order),
pitt_cls_dinov2b_448.npy, pitt_meanpatch_dinov2b_448.npy, pitt_guard_top_pairs.json.

  python -I pitt_prep.py [--n 2500]
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, argparse, hashlib
import numpy as np, torch
from PIL import Image
from scipy.fftpack import dct

ROOT = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root (CBYC_ROOT overrides)
ADD1000_DIR = os.environ.get("ADD1000_DIR", os.path.join(ROOT, "data", "ADD1000"))   # holds flat/{images,maps}
KB = ROOT
FLAT = os.path.join(ADD1000_DIR, "flat")
PITT = os.environ.get("PITT_ADS_DIR", os.path.join(KB, "data", "pitt_ads", "imgs"))   # Pitt Ads images (not redistributed)
OUT = os.path.join(KB, "cache", "distill")
OURS_CACHE = os.path.join(KB, "cache", "ours")
sys.path.insert(0, os.path.join(KB, "scripts", "ours"))
SIZE = 448
IM_M = np.array([0.485, 0.456, 0.406], np.float32); IM_S = np.array([0.229, 0.224, 0.225], np.float32)
T_MP, T_CLS, T_MP_EXT, T_PHASH = 0.94, 0.94, 0.925, 26


def phash256(im_grey):          # identical to near_dup_scan.phash with h = 16
    a = np.asarray(im_grey.resize((64, 64), Image.LANCZOS), np.float64)
    d = dct(dct(a, axis=0), axis=1)[:16, :16]
    return (d > np.median(d)).ravel()


def thumb_md5(im_rgb):
    return hashlib.md5(np.asarray(im_rgb.resize((64, 64), Image.BILINEAR), np.uint8).tobytes()).hexdigest()


def file_md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def l2n(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    files = sorted(os.listdir(PITT))
    files = [f for f in files if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")]
    perm = np.random.default_rng(2026).permutation(len(files))
    subset = [files[i] for i in perm[:a.n]]
    print(f"{len(files)} Pitt files; subset {len(subset)} (default_rng(2026).permutation[:{a.n}])", flush=True)

    # ---- ADD1000 reference hashes (pHash-256, file MD5, thumbnail MD5)
    ids = json.load(open(os.path.join(KB, "splits", "kfold_seed0.json")))["ids_sorted"]
    add_ph, add_fmd5, add_tmd5 = [], {}, {}
    for i in ids:
        p = os.path.join(FLAT, "images", f"{i}.jpg")
        im = Image.open(p)
        add_ph.append(phash256(im.convert("L")))
        add_fmd5.setdefault(file_md5(p), []).append(i)
        add_tmd5.setdefault(thumb_md5(im.convert("RGB")), []).append(i)
    add_ph = np.array(add_ph, np.uint8)
    add_cls = l2n(np.load(os.path.join(OURS_CACHE, "cls_dinov2b_448.npy")))
    add_mp = l2n(np.load(os.path.join(OURS_CACHE, "meanpatch_dinov2b_448.npy")))
    print(f"ADD1000 hashes in {time.time() - t0:.0f}s", flush=True)

    # ---- DINOv2-base features of the subset (same code path as scripts/ours/cache_features.py)
    from model_aspp import ASPPSaliency
    model = ASPPSaliency().eval()
    n = len(subset)
    fpath = os.path.join(OUT, f"pitt_feats_dinov2b_448_fp16.npy")
    feats = np.lib.format.open_memmap(fpath, mode="w+", dtype=np.float16, shape=(n, 768, 32, 32))
    cls = np.zeros((n, 768), np.float32); meanp = np.zeros((n, 768), np.float32)
    ph = np.zeros((n, 256), np.uint8); fmd5, tmd5, unreadable, sizes = [], [], [], []
    with torch.no_grad():
        for k, f in enumerate(subset):
            p = os.path.join(PITT, f)
            try:
                im0 = Image.open(p); im0.load()
                rgb = im0.convert("RGB")
            except Exception as e:          # noqa
                unreadable.append({"file": f, "error": repr(e)}); fmd5.append(""); tmd5.append(""); sizes.append(None)
                continue
            sizes.append(list(rgb.size))
            ph[k] = phash256(im0.convert("L"))
            fmd5.append(file_md5(p)); tmd5.append(thumb_md5(rgb))
            im = rgb.resize((SIZE, SIZE), Image.BILINEAR)
            x = torch.from_numpy(((np.asarray(im, np.float32) / 255 - IM_M) / IM_S).transpose(2, 0, 1))[None]
            out = model.backbone(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
            tok = out[:, 1:, :]
            feats[k] = tok.transpose(1, 2).reshape(1, 768, 32, 32)[0].half().numpy()
            cls[k] = out[0, 0].numpy(); meanp[k] = tok[0].mean(0).numpy()
            if (k + 1) % 250 == 0:
                print(f"  {k + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
    feats.flush()
    np.save(os.path.join(OUT, "pitt_cls_dinov2b_448.npy"), cls)
    np.save(os.path.join(OUT, "pitt_meanpatch_dinov2b_448.npy"), meanp)

    # ---- leakage guard
    bad = {u["file"] for u in unreadable}
    ok = np.array([f not in bad for f in subset])
    S_mp = l2n(np.where(ok[:, None], meanp, 1.0)) @ add_mp.T          # (n,1000)
    S_cls = l2n(np.where(ok[:, None], cls, 1.0)) @ add_cls.T
    H = (ph[:, None, :] != add_ph[None, :, :]).sum(-1)                  # (n,1000) Hamming
    crit = {
        "a_meanpatch_cos_ge_0.94": (S_mp >= T_MP).any(1) & ok,
        "b_cls_cos_ge_0.94": (S_cls >= T_CLS).any(1) & ok,
        "c_meanpatch_cos_ge_0.925_extended": (S_mp >= T_MP_EXT).any(1) & ok,
        "d_phash256_le_26_bits": (H <= T_PHASH).any(1) & ok,
        "e_md5_file_or_thumbnail": np.array([(fmd5[k] in add_fmd5) or (tmd5[k] in add_tmd5) if ok[k] else False
                                             for k in range(n)]),
    }
    removed = np.zeros(n, bool)
    for v in crit.values():
        removed |= v
    keep = ok & ~removed
    rec_removed = []
    for k in np.where(removed)[0]:
        j_mp, j_cls, j_h = int(S_mp[k].argmax()), int(S_cls[k].argmax()), int(H[k].argmin())
        rec_removed.append({"pos": int(k), "file": subset[k], "criteria": [c for c, v in crit.items() if v[k]],
                            "max_meanpatch_cos": float(S_mp[k, j_mp]), "argmax_meanpatch_add_id": ids[j_mp],
                            "max_cls_cos": float(S_cls[k, j_cls]), "argmax_cls_add_id": ids[j_cls],
                            "min_phash_bits": int(H[k, j_h]), "argmin_phash_add_id": ids[j_h]})
    # top pairs for a visual check (by mean-patch and by CLS), whether removed or not
    top = []
    for name, S in (("meanpatch", S_mp), ("cls", S_cls)):
        mx = np.where(ok, S.max(1), -1)
        for k in np.argsort(-mx)[:15]:
            j = int(S[k].argmax())
            top.append({"by": name, "pos": int(k), "file": subset[k], "add_id": ids[j], "cos": float(S[k, j]),
                        "meanpatch_cos": float(S_mp[k, j]), "cls_cos": float(S_cls[k, j]), "phash_bits": int(H[k, j]),
                        "removed": bool(removed[k])})
    hmin = np.where(ok, H.min(1), 999)
    ph_tab = {int(T): int((hmin <= T).sum()) for T in (20, 26, 40, 60, 80)}
    rec = {"source": PITT, "n_files": len(files), "subset_rule": f"sorted file names; numpy default_rng(2026).permutation(n_files)[:{a.n}]",
           "subset_files": subset, "subset_image_sizes": sizes, "unreadable": unreadable,
           "criteria_counts": {c: int(v.sum()) for c, v in crit.items()},
           "n_removed_union": int(removed.sum()), "n_unreadable": len(unreadable), "n_pool": int(keep.sum()),
           "pool_positions": [int(k) for k in np.where(keep)[0]], "removed": rec_removed,
           "distribution": {"max_meanpatch_cos_quantiles": {q: float(np.quantile(S_mp.max(1)[ok], q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
                            "max_cls_cos_quantiles": {q: float(np.quantile(S_cls.max(1)[ok], q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
                            "n_with_min_phash_bits_le_T": ph_tab},
           "thresholds": {"meanpatch": T_MP, "cls": T_CLS, "meanpatch_extended": T_MP_EXT, "phash256_bits": T_PHASH},
           "features": {"file": fpath, "backbone": "facebook/dinov2-base (cached), 448x448, fp16 patch tokens, same code as scripts/ours/cache_features.py"},
           "wall_s": time.time() - t0, "command": " ".join(["python", "-I"] + sys.argv)}
    json.dump(rec, open(os.path.join(OUT, "pitt_subset.json"), "w"), indent=1)
    json.dump(top, open(os.path.join(OUT, "pitt_guard_top_pairs.json"), "w"), indent=1)
    print("criteria:", rec["criteria_counts"], "removed (union):", rec["n_removed_union"], "unreadable:", len(unreadable),
          "pool:", rec["n_pool"], f"({time.time() - t0:.0f}s)", flush=True)
    print("max-cos quantiles:", rec["distribution"], flush=True)


if __name__ == "__main__":
    main()
