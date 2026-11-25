"""
Step 1 split files for the KBS letter (ours track).

(a) near-duplicate pairs + pair-safe version of the fixed seed-0 folds
(b) folder-structure test (are the 8 release folders of 125 ids attribute categories?)
    + brand/template proxy groups + size-balanced 5-fold GroupKFold assignment

Inputs: cache/ours/{cls,meanpatch}_dinov2b_448.npy (cache_features.py),
        cache/ours/hamming_*.npy and near_dup_scan.json (near_dup_scan.py),
        splits/kfold_seed0.json (fixed folds; read only).
Visual verification of candidate pairs was done by inspecting side-by-side contact sheets
(recorded in VISUAL below).

  python -I make_splits.py
"""
import sys
sys.dont_write_bytecode = True
import os, json, time
import numpy as np
from collections import Counter, defaultdict

ROOT = os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root (CBYC_ROOT overrides)
ADD1000_DIR = os.environ.get("ADD1000_DIR", os.path.join(ROOT, "data", "ADD1000"))   # holds flat/{images,maps}
KB = ROOT
C = os.path.join(KB, "cache", "ours")
SPL = json.load(open(os.path.join(KB, "splits", "kfold_seed0.json")))
ids = SPL["ids_sorted"]
pos = {i: n for n, i in enumerate(ids)}
fold_of = {i: k for k, f in enumerate(SPL["folds"]) for i in f}
FOLDER = {}   # release folder of each id (ADD1000/ext)
for d in sorted(os.listdir(os.path.join(ADD1000_DIR, "ext"))):
    if d.startswith("ADD1000_"):
        for f in os.listdir(os.path.join(ADD1000_DIR, "ext", d)):
            FOLDER[os.path.splitext(f)[0]] = d
TOP = {i: (int(i) - 1) // 125 for i in ids}   # 8 top-level groups of 125 ids

MP_PRIMARY = 0.94     # mean-patch cosine threshold: largest gap in the top of the ranking (0.945 -> 0.930)
MP_EXTENDED = 0.925
# Visual verdicts from contact sheets (side-by-side thumbnails), 23 Nov 2026.
VISUAL = {
    ("182", "761"): "same creative, pixel-identical (7up)",
    ("391", "694"): "same template, different wine brand text",
    ("642", "910"): "same creative re-cropped (Tsingtao)",
    ("218", "226"): "same template, different bottle colour (Mizone)",
    ("330", "343"): "same campaign series and layout (coffee x classical poets)",
    ("312", "330"): "same campaign series and layout (coffee x classical poets)",
    ("312", "343"): "same campaign series and layout (coffee x classical poets)",
    ("351", "698"): "same creative, different aspect-ratio crop (black sesame paste)",
    ("256", "530"): "same campaign series and layout (vivo phone)",
}
REJECTED = {
    ("495", "945"): "same product category (canned drink), different brand/design",
    ("335", "340"): "two phone ads with similar white layout, different brands",
    ("378", "608"): "two shoe ads, different brands",
    ("4", "969"): "same brand (Tsingtao), different creatives; brand-level, not near-duplicate",
    ("190", "335"): "hash collision (pHash-64 = 6 bits): unrelated layouts",
    ("523", "535"): "hash collision (pHash-64 = 8 bits): unrelated",
    ("383", "778"): "hash collision (aHash-256 = 23 bits): unrelated",
    ("226", "782"): "hash collision (dHash-64 = 7 bits): unrelated",
    ("457", "807"): "hash collision (pHash-64 = 12 bits): two tourism posters, different designs",
}


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def components(nodes, edges):
    par = {n: n for n in nodes}

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in edges:
        par[f(a)] = f(b)
    comp = defaultdict(list)
    for n in nodes:
        comp[f(n)].append(n)
    return [sorted(v, key=lambda s: int(s.lstrip("c"))) for v in comp.values()]


def pair_safe_folds(comps, seed=0):
    """Move each multi-fold component into one fold (the fold holding most of its members;
    ties -> fold of the lowest id) and swap the same number of unconstrained images back,
    so every fold keeps 200 ids. Returns new folds and the list of moves."""
    rng = np.random.default_rng(seed)
    folds = [list(f) for f in SPL["folds"]]
    constrained = {i for c in comps for i in c}
    moves = []
    for c in comps:
        fs = [fold_of[i] for i in c]
        if len(set(fs)) == 1:
            continue
        cnt = Counter(fs)
        best = max(cnt.values())
        target = min((fold_of[i] for i in c if cnt[fold_of[i]] == best), key=lambda k: min(int(i) for i in c if fold_of[i] == k))
        for i in c:
            src = next(k for k, f in enumerate(folds) if i in f)
            if src == target:
                continue
            cand = sorted((j for j in folds[target] if j not in constrained), key=int)
            j = cand[int(rng.integers(len(cand)))]
            folds[src].remove(i); folds[target].append(i)
            folds[target].remove(j); folds[src].append(j)
            moves.append({"moved": i, "from": src, "to": target, "swapped_back": j})
    return [sorted(f, key=int) for f in folds], moves


def folder_probe(X):
    """Is the 8-folder partition predictable from image content beyond what id order alone
    gives? Linear probe (5-fold CV, multinomial logistic regression on L2-normalised CLS)
    for the true folder boundaries and for the same 125-id chunking shifted by s ids."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    out = {}
    for s in (0, 31, 62, 93):
        y = np.array([((int(i) - 1 + s) % 1000) // 125 for i in ids])
        clf = LogisticRegression(max_iter=3000, C=1.0)
        acc = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        out[f"shift_{s}"] = round(float(acc), 4)
    rng = np.random.default_rng(0)
    y = np.array([TOP[i] for i in ids]); perm = []
    for r in range(5):
        yp = rng.permutation(y)
        perm.append(float(cross_val_score(LogisticRegression(max_iter=3000), X, yp, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()))
    out["permuted_labels_mean"] = round(float(np.mean(perm)), 4)
    out["chance"] = 0.125
    return out


def main():
    t0 = time.time()
    X = unit(np.load(os.path.join(C, "cls_dinov2b_448.npy")))
    M = unit(np.load(os.path.join(C, "meanpatch_dinov2b_448.npy")))
    scan = json.load(open(os.path.join(C, "near_dup_scan.json")))
    Hp16 = np.load(os.path.join(C, "hamming_phash16.npy"))
    iu = np.triu_indices(len(ids), 1)
    mp = (M @ M.T)[iu]; cl = (X @ X.T)[iu]

    def pair_info(a, b):
        x, y = pos[a], pos[b]
        return {"a": a, "b": b, "meanpatch_cos": round(float(M[x] @ M[y]), 4), "cls_cos": round(float(X[x] @ X[y]), 4),
                "phash256_bits": int(Hp16[x, y]), "fold_a": fold_of[a], "fold_b": fold_of[b],
                "cross_fold": fold_of[a] != fold_of[b], "release_folder_a": FOLDER[a], "release_folder_b": FOLDER[b],
                "visual": VISUAL.get((a, b), VISUAL.get((b, a), "not inspected"))}

    prim = [(ids[iu[0][j]], ids[iu[1][j]]) for j in np.nonzero(mp >= MP_PRIMARY)[0]]
    ext = [(ids[iu[0][j]], ids[iu[1][j]]) for j in np.nonzero(mp >= MP_EXTENDED)[0]]
    ext = sorted(set(ext) | {k for k in VISUAL}, key=lambda p: (int(p[0]), int(p[1])))
    for p in ext:
        assert p in VISUAL or (p[1], p[0]) in VISUAL, f"uninspected pair above threshold: {p}"
    ph26 = [(ids[iu[0][j]], ids[iu[1][j]]) for j in np.nonzero(Hp16[iu] <= 26)[0]]

    comps_ext = [c for c in components(ids, ext) if len(c) > 1]
    new_folds, moves = pair_safe_folds(comps_ext)
    assert all(len(f) == 200 for f in new_folds) and sorted(sum(new_folds, []), key=int) == ids
    nf = {i: k for k, f in enumerate(new_folds) for i in f}
    assert all(len({nf[i] for i in c}) == 1 for c in comps_ext)

    nd = {
        "description": "ADD1000 near-duplicate pairs. The earlier record (5 pairs / 4 cross-fold at 26 bits) did not store "
                       "its hash type or pair list and could not be reproduced with standard perceptual hashes: imagehash-style "
                       "pHash/dHash/aHash at 64 and 256 bits (and 144/576/1024-bit variants, several resize filters) give either "
                       "1 pair (pHash-256 <= 26 bits: 182/761) or tens of thousands of pairs (64-bit hashes at 26 bits). Low-bit "
                       "hash hits other than 182/761 were visually unrelated layouts (hash collisions). Near-duplicates are "
                       "therefore defined on frozen DINOv2-base mean patch-token embeddings (448 px), which capture layout and "
                       "content jointly, and every pair above threshold was visually inspected.",
        "hash_reproduction_attempt": {
            "hash": "pHash, imagehash reference algorithm re-implemented (grey, LANCZOS 64x64, 2-D DCT-II, top-left 16x16, bit = coef > median) = 256 bits",
            "threshold_bits": 26,
            "pairs": [pair_info(a, b) for a, b in ph26],
            "n_pairs": len(ph26),
            "scan_counts_at_26_bits": {k: v["table"]["26"] for k, v in scan.items()},
            "reproduced_earlier_count": False,
        },
        "primary": {
            "definition": f"cosine similarity of L2-normalised mean DINOv2-base patch tokens (448 px) >= {MP_PRIMARY}; "
                          "threshold at the largest gap at the top of the ranking (0.945 -> 0.930)",
            "n_pairs": len(prim),
            "n_cross_fold_seed0": int(sum(fold_of[a] != fold_of[b] for a, b in prim)),
            "pairs": [pair_info(a, b) for a, b in prim],
            "note": "count and cross-fold count coincide with the earlier record (5 pairs, 4 cross-fold); all 5 visually confirmed as the same creative or template",
        },
        "extended": {
            "definition": f"primary plus pairs with mean-patch cosine >= {MP_EXTENDED} and visually confirmed same-campaign variants "
                          "found among the top-30 mean-patch and top-20 CLS pairs; used for the pair-safe folds (strictly more conservative)",
            "n_pairs": len(ext),
            "n_cross_fold_seed0": int(sum(fold_of[a] != fold_of[b] for a, b in ext)),
            "pairs": [pair_info(a, b) for a, b in ext],
            "components": comps_ext,
        },
        "rejected_after_visual_check": [dict(pair_info(a, b), verdict=v) for (a, b), v in REJECTED.items()],
        "release_folders_of_pairs": "pixel-identical pair 182/761 sits in release folders ADD1000_2 and ADD1000_7; see groups.json",
        "script": os.path.abspath(__file__),
    }
    json.dump(nd, open(os.path.join(KB, "splits", "near_dup_pairs.json"), "w"), indent=1)

    changed = sorted({m["moved"] for m in moves} | {m["swapped_back"] for m in moves}, key=int)
    json.dump({
        "description": "Seed-0 5-fold split with every extended near-duplicate component (splits/near_dup_pairs.json) placed in one "
                       "fold. Each cross-fold member is moved to the fold holding most of its component (tie: fold of the lowest "
                       "id) and an unconstrained id from that fold (numpy default_rng(0) draw) is swapped back, so folds stay at 200.",
        "base": "splits/kfold_seed0.json", "ids_sorted": ids, "folds": new_folds,
        "moves": moves, "n_ids_changed_fold": len(changed), "ids_changed_fold": changed,
        "components_constrained": comps_ext, "script": os.path.abspath(__file__),
    }, open(os.path.join(KB, "splits", "kfold_seed0_pairsafe.json"), "w"), indent=1)
    print("near-dup: primary", len(prim), "extended", len(ext), "moves", len(moves), flush=True)

    # ---------------- (b) folder-structure test ----------------
    probe = folder_probe(X)
    print("folder probe", probe, flush=True)
    # colour statistics per top-level group (calming/exciting etc. would shift saturation/brightness)
    from PIL import Image
    stats = []
    for i in ids:
        im = Image.open(os.path.join(ADD1000_DIR, "flat", "images", f"{i}.jpg")).convert("RGB").resize((128, 128))
        hsv = np.asarray(im.convert("HSV"), np.float64) / 255
        stats.append([hsv[..., 1].mean(), hsv[..., 2].mean()])
    stats = np.array(stats)
    from scipy.stats import kruskal
    grp = np.array([TOP[i] for i in ids])
    kw = {name: float(kruskal(*[stats[grp == g, k] for g in range(8)]).pvalue) for k, name in enumerate(["saturation", "value"])}
    kw_shift = {}
    for sh in (31, 62, 93):
        gs = np.array([((int(i) - 1 + sh) % 1000) // 125 for i in ids])
        kw_shift[f"shift_{sh}"] = {name: round(float(kruskal(*[stats[gs == g, k] for g in range(8)]).pvalue), 4) for k, name in enumerate(["saturation", "value"])}
    group_means = {f"group{g + 1}": {"saturation": round(float(stats[grp == g, 0].mean()), 4), "value": round(float(stats[grp == g, 1].mean()), 4)} for g in range(8)}
    sizes_mb = {}
    for d in sorted(os.listdir(os.path.join(ADD1000_DIR, "ext"))):
        p = os.path.join(ADD1000_DIR, "ext", d)
        sizes_mb[d] = round(sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)) / 2 ** 20, 1)

    # ---------------- brand/template proxy groups ----------------
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    sil = {}
    for k in (40, 50, 60):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        sil[k] = round(float(silhouette_score(X, lab, metric="cosine")), 4)
    K = 50
    lab = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X)
    # merge clusters linked by a near-duplicate component
    edges = [(f"c{lab[pos[a]]}", f"c{lab[pos[b]]}") for c in comps_ext for a, b in zip(c[:-1], c[1:])]
    cc = components([f"c{k}" for k in range(K)], edges)
    cmap = {}
    for gid, comp in enumerate(sorted(cc, key=lambda v: min(int(x[1:]) for x in v))):
        for x in comp:
            cmap[int(x[1:])] = gid
    group = np.array([cmap[l] for l in lab])
    n_groups = int(group.max()) + 1
    gsize = Counter(group.tolist())
    # size-balanced GroupKFold (greedy: largest group first into the currently smallest fold;
    # the rule of sklearn GroupKFold without shuffling; tie-breaking may differ across sklearn versions)
    order = sorted(gsize, key=lambda g: (-gsize[g], g))
    fsize = [0] * 5; g2f = {}
    for g in order:
        k = int(np.argmin(fsize)); g2f[g] = k; fsize[k] += gsize[g]
    gfolds = [sorted([i for i in ids if g2f[group[pos[i]]] == k], key=int) for k in range(5)]
    from sklearn.model_selection import GroupKFold
    sk = [sorted((ids[j] for j in te), key=int) for _, te in GroupKFold(5).split(np.zeros(1000), groups=group)]
    same_as_sklearn = sorted(map(tuple, sk)) == sorted(map(tuple, gfolds))
    # cross-check: how much more similar are test images to their nearest training image?
    def nn_sim(folds):
        S = X @ X.T; v = []
        for f in folds:
            te = [pos[i] for i in f]; tr = [pos[i] for i in ids if i not in set(f)]
            v.append(float(S[np.ix_(te, tr)].max(1).mean()))
        return round(float(np.mean(v)), 4)
    groups_out = {
        "description": "Brand/template proxy groups for ADD1000 and a 5-fold GroupKFold assignment. ADD1000 ships no brand, "
                       "campaign or category labels (see folder_structure_test). Groups = Ward agglomerative clustering of "
                       f"L2-normalised frozen DINOv2-base CLS embeddings (448 px input) into {K} clusters, then clusters joined "
                       "by an extended near-duplicate component (splits/near_dup_pairs.json) are merged (connected components). "
                       "Folds: size-balanced greedy GroupKFold (largest group into the smallest fold), no group spans two folds.",
        "folder_structure_test": {
            "verdict": "The 8 release folders of 125 consecutive ids are NOT attribute categories; they are upload chunks of the id "
                       "sequence. A leave-one-category-out split is therefore not written.",
            "evidence": [
                "README names 4 bipolar dimensions (8 attributes) but ships no per-image labels; under a 4-dimension design each "
                "image carries one pole per dimension, so 8 exclusive groups of 125 cannot be attribute classes.",
                "Folders 5-8 are split A/B (60/65 ids) and exactly these are the folders whose files exceed 25 MB in total "
                "(the GitHub web-upload file limit); folders 1-4 are 9.8-19 MB and unsplit: " + json.dumps(sizes_mb),
                "Pixel-identical pair 182/761 sits in folders ADD1000_2 and ADD1000_7; 391/694 (same template) in 4 and 6; "
                "642/910 (same creative) in 6 and 8.",
                "Linear probe on DINOv2 CLS (5-fold CV) predicting the folder of an image: accuracy at the true boundaries vs "
                "the same chunking shifted by 31/62/93 ids: " + json.dumps(probe) + ". Accuracy that is not peaked at shift 0 "
                "means content varies smoothly with id order, not by folder.",
                "Kruskal-Wallis across the 8 groups on mean HSV saturation / value: p = " + json.dumps({k: round(v, 4) for k, v in kw.items()}) + "; same test on the 125-id chunking shifted by 31/62/93 ids: " + json.dumps(kw_shift) + ". The colour difference is small and is a monotone drift with id (mean saturation 0.36-0.38 in folders 1-5, rising to 0.45 in folder 8; file sizes also grow with id), not the alternating pattern that paired attribute poles would produce; it is weaker for shifted chunks, so it is the one observation that does not argue against folders, but it carries no category information the content probe can pick up.",
            ],
            "group_colour_means": group_means,
            "folder_sizes_mb": sizes_mb,
            "probe": probe,
        },
        "visual_check": "Contact sheet of 9 images from each of the 6 largest, 2 mid-sized and 2 smallest groups (23 Nov 2026): groups are "
                        "product-category / visual-style families (wine bottles; shoes; green tea and herbal products; sunglasses; "
                        "pocket watches; ink-wash style real-estate and tourism posters), i.e. a plausible brand/category/template proxy; "
                        "the largest group (82 ids) is a mixed bag of conceptual creative ads. Groups are a proxy, not ground-truth brands.",
        "embedding": "facebook/dinov2-base CLS token (last_hidden_state[:,0], post-layernorm), 448x448 bilinear input, ImageNet norm",
        "clustering": {"method": "sklearn AgglomerativeClustering(linkage='ward') on L2-normalised CLS", "n_clusters": K,
                       "silhouette_cosine": sil, "n_groups_after_near_dup_merge": n_groups,
                       "group_size_min_median_max": [int(min(gsize.values())), float(np.median(list(gsize.values()))), int(max(gsize.values()))]},
        "fold_sizes": [len(f) for f in gfolds],
        "same_as_sklearn_GroupKFold": bool(same_as_sklearn),
        "mean_test_to_nearest_train_cls_cos": {"seed0_random_folds": nn_sim(SPL["folds"]), "grouped_folds": nn_sim(gfolds)},
        "ids_sorted": ids,
        "per_image": [{"id": i, "cluster": int(lab[pos[i]]), "group": int(group[pos[i]]), "fold": int(g2f[group[pos[i]]]),
                        "release_folder": FOLDER[i], "seed0_fold": fold_of[i]} for i in ids],
        "folds": gfolds,
        "script": os.path.abspath(__file__),
    }
    json.dump(groups_out, open(os.path.join(KB, "splits", "groups.json"), "w"), indent=1)
    print("groups:", n_groups, "fold sizes", [len(f) for f in gfolds], "sil", sil, groups_out["mean_test_to_nearest_train_cls_cos"],
          "kw", kw, "same_as_sklearn", same_as_sklearn, f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
