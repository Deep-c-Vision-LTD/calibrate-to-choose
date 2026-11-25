"""
KBS revision (24 Nov 2026). Zero-shot DeepGaze IIE in its documented usage: the image at its NATIVE
resolution (the model downsamples by 2 internally) and a MIT1003 centre-bias template rescaled to the
image with nearest interpolation and logsumexp-renormalised. No ADD1000 data.
Template: DeepGaze's distributed
cache/revision/deepgaze_template/centerbias_mit1003.npy (1024 x 1024 float64 log density, sha256 051645b7...e9,
allow_pickle=False); row iie_zeroshot_mit1003_native_offtpl, resume file ..._offtpl_partial.json. Prediction density
resized to 384 x 384 (torch bilinear, align_corners=False) and scored against the same 384 GT as every
other row (iie gt_cache raw384; clip + sum normalisation).

Row: iie_zeroshot_mit1003_native_offtpl (results/<row>_per_image.json, _summary.json). Resumable.
  python -I iie_native_zeroshot.py [--threads 4]
"""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "iie"))
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
import iie_common as C

ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--tpl", default="official", choices=["official"]); a = ap.parse_args()
torch.set_num_threads(a.threads)
SUF = "_offtpl" if a.tpl == "official" else ""
ROW = "iie_zeroshot_mit1003_native" + SUF
PART = os.path.join(C.KBS, "cache", "revision", ROW + "_partial.json")
if a.tpl == "official":
    TPL_PATH = os.path.join(C.KBS, "cache", "revision", "deepgaze_template", "centerbias_mit1003.npy")
    assert hashlib.sha256(open(TPL_PATH, "rb").read()).hexdigest() == \
        "051645b7da47884c138e81f28a4afbadd99b6c3262eaf086434a9626e3578be9"
    TPL_DESC = "official DeepGaze centerbias_mit1003.npy (v1.0.0 release), cache/revision/deepgaze_template/centerbias_mit1003.npy"
TPL = np.load(TPL_PATH, allow_pickle=False).astype(np.float64)
import deepgaze_pytorch
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).eval()
ids, folds = C.load_split(); fold_of = C.fold_of(folds); pos = {i: k for k, i in enumerate(ids)}
raw = C.gt_cache()["raw384"]
done = json.load(open(PART)) if os.path.exists(PART) else {}
t0 = time.time(); sizes = {}
for n, i in enumerate(ids):
    if i in done:
        continue
    im = Image.open(C.img_path(i)).convert("RGB")
    W, H = im.size
    x = torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1))[None]
    cb = zoom(TPL, (H / TPL.shape[0], W / TPL.shape[1]), order=0, mode="nearest")
    cb = cb - logsumexp(cb)
    with torch.no_grad():
        logd = model(x, torch.from_numpy(cb.astype(np.float32))[None])
    d = torch.exp(logd.double())
    d384 = F.interpolate(d.view(1, 1, H, W), size=(C.S, C.S), mode="bilinear", align_corners=False)[0, 0].numpy()
    m = C.metrics(d384, raw[pos[i]].astype(np.float64))
    done[i] = {"id": i, "fold": fold_of[i], **m, "W": W, "H": H}
    if len(done) % 25 == 0:
        json.dump(done, open(PART, "w"))
        print(f"{len(done)}/1000 {time.time() - t0:.0f}s", flush=True)
json.dump(done, open(PART, "w"))
rows = [done[i] for i in ids]
summ = {"row": ROW, **C.fold_summary(rows), "n_images": len(rows), "calibration_maps": 0, "training_maps": 0,
        "protocol": "zero-shot DeepGaze IIE, released weights, native image resolution (no resize), MIT1003 centre-bias "
                    f"template ({TPL_DESC}) zoomed (order 0) to the image and "
                    "logsumexp-renormalised as in the DeepGaze README; density resized to 384 (torch bilinear) and scored "
                    "on the 384 grid against bilinear-384 GT; no post-hoc blur; no ADD1000 data",
        "image_size_range_WH": [[min(r["W"] for r in rows), min(r["H"] for r in rows)],
                                [max(r["W"] for r in rows), max(r["H"] for r in rows)]],
        "threads": a.threads, "wall_s_this_run": time.time() - t0, "template_file": TPL_PATH, "file_suffix": SUF,
        "command": "python -I scripts/revision/iie_native_zeroshot.py --threads %d --tpl %s" % (a.threads, a.tpl)}
json.dump(rows, open(os.path.join(C.RESULTS, f"{ROW}_per_image.json"), "w"), indent=0)
json.dump(summ, open(os.path.join(C.RESULTS, f"{ROW}_summary.json"), "w"), indent=1)
print(ROW, {k: summ[k] for k in summ if k.endswith("_mean")}, flush=True)
