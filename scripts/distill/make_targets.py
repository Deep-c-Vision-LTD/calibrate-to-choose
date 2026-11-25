"""
Track B, step 2c: teacher pseudo-label targets for the students.

Teacher k's deployed prediction = its 256x256 log density (fold-k prior) -> MSDB's own 512 density
(nearest 2x) -> bilinear 384 (logd256_to_prob, as scored in msdb_adapt) -> its post-hoc Gaussian
sigma (chosen on fold-k training images). The student target is built from that 384 map exactly
like a GT map is turned into a training target by ours_lib.load_gt_train: PIL bilinear resize to
64x64 (float 'F' image), subtract the minimum, divide by the sum.

  --part add  : teacher k on all 1000 ADD1000 images -> cache/distill/targets/add_t64.npy (5,1000,64,64)
                (only fold k's 800 training images are ever used for student k)
  --part pitt : teacher k on every Pitt pool image   -> cache/distill/targets/pitt_t64.npy (5,P,64,64)
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, argparse
import numpy as np, torch
from PIL import Image
from scipy.ndimage import gaussian_filter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import distill_common as DC                                   # noqa: E402
sys.path.insert(0, os.path.join(DC.KB, "scripts", "msdb"))
sys.path.insert(0, os.path.join(DC.KB, "scripts", "ours"))
from msdb_common import logd256_to_prob                        # noqa: E402
import ours_lib as L                                           # noqa: E402

OUT = os.path.join(DC.CD, "targets")


def teacher_final384(logd, sigma):
    """logd: (B,256,256) float32 numpy -> list of (384,384) float64 deployed teacher maps."""
    p = logd256_to_prob(torch.from_numpy(np.ascontiguousarray(logd)), 384)
    return [gaussian_filter(x, sigma) if sigma > 0 else x for x in p]


def to_t64(p384):
    m = np.asarray(Image.fromarray(p384.astype(np.float32)).resize((L.LOSSG, L.LOSSG), Image.BILINEAR), np.float64)
    m = m - m.min(); s = m.sum()
    return m / s if s > 0 else m + 1.0 / m.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["add", "pitt"], required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    sig = [torch.load(os.path.join(DC.TD, f"fold{k}.pt"), map_location="cpu", weights_only=False)["posthoc_sigma"] for k in range(5)]
    info = {"teacher_posthoc_sigma": sig}
    if a.part == "add":
        out = np.zeros((5, 1000, L.LOSSG, L.LOSSG), np.float32)
        ids = L.load_ids()
        gtl = np.stack([L.load_gt_train(os.path.join(L.FLAT, "maps", f"{i}.png"), L.LOSSG) for i in ids])
        for k in range(5):
            logd = np.load(os.path.join(DC.TD, f"teacher_add1000_logd256_fold{k}.npy"), mmap_mode="r")
            for j in range(0, 1000, 50):
                for jj, p in enumerate(teacher_final384(logd[j:j + 50], sig[k])):
                    out[k, j + jj] = to_t64(p)
            # sanity only: agreement of teacher targets with the GT targets at 64x64
            cc = [L.cc(out[k, i].astype(np.float64), gtl[i]) for i in range(1000)]
            info[f"fold{k}_mean_cc_teacher_vs_gt_t64_all1000"] = float(np.mean(cc))
            print(f"fold {k}: done ({time.time() - t0:.0f}s), mean CC(teacher t64, GT t64) = {np.mean(cc):.4f}", flush=True)
        np.save(os.path.join(OUT, "add_t64.npy"), out)
    else:
        files = DC.pool(); n = len(files)
        flags = np.load(os.path.join(DC.PM, "flags.npy"))
        assert (flags == 3).all(), f"pitt pass incomplete: {np.bincount(flags, minlength=4).tolist()}"
        tl = np.load(os.path.join(DC.PM, "tlogd.npy"), mmap_mode="r")
        out = np.zeros((5, n, L.LOSSG, L.LOSSG), np.float32)
        for k in range(5):
            for j in range(0, n, 50):
                for jj, p in enumerate(teacher_final384(np.array(tl[k, j:j + 50]), sig[k])):
                    out[k, j + jj] = to_t64(p)
            print(f"fold {k}: done ({time.time() - t0:.0f}s)", flush=True)
        assert np.isfinite(out).all()
        np.save(os.path.join(OUT, "pitt_t64.npy"), out)
        # agreement between teachers on the pool (how different are the 5 fold teachers?)
        info["mean_cc_between_teacher0_and_teacher_k_on_pool"] = [float(np.mean([L.cc(out[0, i].astype(np.float64), out[k, i].astype(np.float64))
                                                                              for i in range(n)])) for k in range(5)]
        info["n_pool"] = n
    info["wall_s"] = time.time() - t0
    json.dump(info, open(os.path.join(OUT, f"{a.part}_targets_info.json"), "w"), indent=1)
    print(info, flush=True)


if __name__ == "__main__":
    main()
