"""
Track B, step 3: students = our readout (frozen DINOv2 448 + ASPP readout, scratch init) trained per
fold with MSDB-teacher pseudo-labels. Everything not listed below is the base recipe of
scripts/ours (kfold v3, ours_scratch_s0), imported unchanged from ours_lib: readout module,
init seed torch.manual_seed(1000 s + k), data-order rng default_rng(k + 1000 s) including the
start-of-training 200-image sigma draw, AdamW lr 3e-4 wd 1e-4, batch 16 ADD1000 images per step,
60 epochs x 50 steps = 3000 steps, loss L = KL + (1 - CC) at 64x64, post-hoc Gaussian sigma from
{0,...,12} by mean CC on a random 200-image subset of the training fold, unified 384 scorer.

Pseudo-label loss weight w = 1.0 and the 3000-step budget were fixed a priori (no tuning on any
test fold; no tuning at all).

rows
  distill_pitt            each step: the base recipe's 16 ADD1000 training images with GT targets
                          + 16 Pitt pool images with teacher-k targets (one forward of 32);
                          loss = L(GT batch) + w L(Pitt batch). The GT stream (order, steps) is
                          identical to ours_scratch_s0; the Pitt stream cycles through the pool in
                          permutations drawn by default_rng(100000 + k + 1000 s). Sigma: base rule
                          (200 training ADD1000 images vs GT).
  distill_pitt_only       each step: 16 Pitt images with teacher-k targets only (no ADD1000 GT in
                          training); 3000 steps. Sigma: same rule but on its own training data
                          (200 Pitt pool images drawn by the Pitt rng, teacher-k deployed maps as
                          reference), because no ADD1000 GT is used by this row.
  distill_gt800_teacher800 each step: the base recipe's 16 ADD1000 training images, each with both
                          targets; loss = L(s, GT) + w L(s, teacher-k map of that training image).
                          No extra images. Sigma: base rule.
  --backbone large        same rows on DINOv2-large features (row suffix _L); ADD1000 large features
                          from track A's cache, Pitt large features from cache/distill.

Teacher k (MSDB head adapted on fold k's 800 training maps only) labels the pool for student k;
student k is scored on fold k's 200 test images. Inference cost of every student = our base model
(same backbone and readout; the teacher is not used at test time).

Fold results go to cache/distill/students/<row>_fold<k>.json (folds may be split across processes
via mkdir claims); once all 5 exist, results/<row>_per_image.json and _summary.json are written.

  python -I train_students.py --row distill_pitt --device mps
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, argparse, platform
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import distill_common as DC                                   # noqa: E402
sys.path.insert(0, os.path.join(DC.KB, "scripts", "ours"))
sys.path.insert(0, os.path.join(DC.KB, "scripts", "msdb"))
import ours_lib as L                                           # noqa: E402
from model_aspp import ASPPModule                              # noqa: E402
from msdb_common import logd256_to_prob                        # noqa: E402
from scipy.ndimage import gaussian_filter                      # noqa: E402

W_PSEUDO = 1.0
STEPS_NOTE = "60 epochs x 50 steps of 16 ADD1000 training ids = 3000 optimiser steps (= base recipe)"
SMOKE = os.environ.get("DISTILL_SMOKE_DIR")          # smoke tests only: redirect all outputs, fake targets
SD = os.path.join(SMOKE, "students") if SMOKE else os.path.join(DC.CD, "students")
if SMOKE:
    DC.RESULTS = os.path.join(SMOKE, "results"); os.makedirs(DC.RESULTS, exist_ok=True)
FEATS = {"base": ("ours/feats_dinov2b_448_fp16.npy", "distill/pitt_feats_dinov2b_448_fp16.npy", 768),
         "large": ("trackA/feats_dinov2l_448_fp16.npy", "distill/pitt_feats_dinov2l_448_fp16.npy", 1024)}


class ReadoutHeadC(nn.Module):
    """ours_lib.ReadoutHead (mode full) with configurable input channels (same parameter order)."""
    def __init__(self, in_ch):
        super().__init__()
        self.aspp = ASPPModule(in_ch, 128)
        self.head = nn.Conv2d(128, 1, 1)
        self.use_cb = True
        self.center_bias = nn.Parameter(torch.zeros(1, 1, 16, 16))

    def forward(self, f):
        s = self.head(self.aspp(f))
        return s + F.interpolate(self.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)


def kl_cc(s, g):
    """Base-recipe loss (ours_lib.train): s logits (B,h,w) -> 64x64; g (B,64,64) normalised targets."""
    B = s.shape[0]
    s = F.interpolate(s, size=(L.LOSSG, L.LOSSG), mode="bilinear", align_corners=False)[:, 0]
    logp = F.log_softmax(s.reshape(B, -1), 1).reshape(B, L.LOSSG, L.LOSSG)
    kl = (g * (torch.log(g + 1e-12) - logp)).sum((1, 2)).mean()
    ps = torch.sigmoid(s)
    pc = ps - ps.mean((1, 2), keepdim=True); gc = g - g.mean((1, 2), keepdim=True)
    ccb = (pc * gc).sum((1, 2)) / (torch.sqrt((pc ** 2).sum((1, 2)) * (gc ** 2).sum((1, 2))) + 1e-8)
    return kl + (1 - ccb.mean())


class Data:
    def __init__(self, dev, backbone):
        fa, fp, self.in_ch = FEATS[backbone]
        self.ids = L.load_ids(); self.pos = {i: n for n, i in enumerate(self.ids)}
        f = np.load(os.path.join(DC.KB, "cache", fa), mmap_mode="r")
        assert f.shape == (1000, self.in_ch, 32, 32) and f.dtype == np.float16, f.shape
        self.feats = torch.from_numpy(np.array(f)).to(dev)
        self.gtl = torch.stack([torch.tensor(L.load_gt_train(os.path.join(L.FLAT, "maps", f"{i}.png"), L.LOSSG), dtype=torch.float32)
                                for i in self.ids]).to(dev)
        self.gte = [L.load_gt_eval(os.path.join(L.FLAT, "maps", f"{i}.png")) for i in self.ids]
        sub = json.load(open(os.path.join(DC.CD, "pitt_subset.json")))
        self.pool_pos = np.array(sub["pool_positions"]); self.P = len(self.pool_pos)
        pf = np.load(os.path.join(DC.KB, "cache", fp), mmap_mode="r")
        assert pf.shape[1:] == (self.in_ch, 32, 32) and pf.shape[0] in (len(sub["subset_files"]), self.P), pf.shape
        pf = pf[self.pool_pos] if pf.shape[0] == len(sub["subset_files"]) else np.array(pf)
        self.pfeats = torch.from_numpy(np.ascontiguousarray(pf)).to(dev)
        if SMOKE:
            self.at64 = self.gtl[None].repeat(5, 1, 1, 1).flip(-1)          # fake teacher targets (mirrored GT)
            yy, xx = torch.meshgrid(torch.arange(L.LOSSG), torch.arange(L.LOSSG), indexing="ij")
            blob = torch.exp(-((yy - 32.0) ** 2 + (xx - 32.0) ** 2) / (2 * 8.0 ** 2)); blob = blob / blob.sum()
            self.pt64 = blob.float().to(dev)[None, None].repeat(5, self.P, 1, 1)      # fake: centred Gaussian
        else:
            self.at64 = torch.from_numpy(np.load(os.path.join(DC.CD, "targets", "add_t64.npy"))).to(dev)       # (5,1000,64,64)
            pt = os.path.join(DC.CD, "targets", "pitt_t64.npy")
            self.pt64 = torch.from_numpy(np.load(pt)).to(dev) if os.path.exists(pt) else None              # (5,P,64,64)
        self.dev = dev

    def batch(self, idx):
        ix = torch.as_tensor(np.asarray(idx), device=self.dev)
        return self.feats[ix].float(), self.gtl[ix]

    def pbatch(self, pidx, k):
        ix = torch.as_tensor(np.asarray(pidx), device=self.dev)
        return self.pfeats[ix].float(), self.pt64[k][ix]


class Cycler:
    def __init__(self, n, rng):
        self.n, self.rng, self.buf = n, rng, []

    def take(self, b):
        out = []
        while len(out) < b:
            if not self.buf:
                self.buf = list(self.rng.permutation(self.n))
            out.append(self.buf.pop(0))
        return out


@torch.no_grad()
def raw_preds_feats(head, feats, bs=50):
    head.eval(); out = []
    for j in range(0, feats.shape[0], bs):
        s = head(feats[j:j + bs].float())
        out += [p.astype(np.float32) for p in torch.sigmoid(F.interpolate(s, size=(L.EVAL, L.EVAL), mode="bilinear", align_corners=False))[:, 0].cpu().numpy()]
    return out


def run_fold(row, kind, k, data, folds, seed, dev, backbone, epochs=60):
    t0 = time.time()
    te = np.array(folds[k]); tr = np.concatenate([np.array(folds[j]) for j in range(len(folds)) if j != k])
    torch.manual_seed(1000 * seed + k)
    head = (L.ReadoutHead("full") if backbone == "base" else ReadoutHeadC(data.in_ch)).to(dev)
    rng = np.random.default_rng(k + 1000 * seed)
    rng_p = np.random.default_rng(100000 + k + 1000 * seed)
    _ = rng.choice(tr, size=L.SIGMA_SELECT_N, replace=False)      # base recipe's start-of-training sigma draw (keeps the rng stream)
    cyc = Cycler(data.P, rng_p)
    opt = torch.optim.AdamW(list(head.parameters()), lr=3e-4, weight_decay=1e-4)
    steps, loss_log = 0, []
    for ep in range(epochs):
        head.train()
        tr_shuf = tr.copy(); rng.shuffle(tr_shuf)
        tot = 0.0
        for j in range(0, len(tr_shuf), 16):
            ids = tr_shuf[j:j + 16]
            if kind == "pitt":
                f, g = data.batch(ids)
                fp, tp = data.pbatch(cyc.take(16), k)
                s = head(torch.cat([f, fp], 0))
                loss = kl_cc(s[:len(ids)], g) + W_PSEUDO * kl_cc(s[len(ids):], tp)
            elif kind == "pitt_only":
                fp, tp = data.pbatch(cyc.take(16), k)
                loss = W_PSEUDO * kl_cc(head(fp), tp)
            else:   # gt800_teacher800
                f, g = data.batch(ids)
                t = data.at64[k][torch.as_tensor(np.asarray(ids), device=dev)]
                s = head(f)
                loss = kl_cc(s, g) + W_PSEUDO * kl_cc(s, t)
            opt.zero_grad(); loss.backward(); opt.step()
            steps += 1; tot += float(loss.detach())
        loss_log.append(tot / 50)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  [{row}] fold {k} ep {ep} loss {loss_log[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    head.eval()
    if kind == "pitt_only":
        sel = rng_p.choice(data.P, size=L.SIGMA_SELECT_N, replace=False)
        preds = raw_preds_feats(head, data.pfeats[torch.as_tensor(sel, device=dev)])
        sk = 6 if SMOKE else torch.load(os.path.join(DC.TD, f"fold{k}.pt"), map_location="cpu", weights_only=False)["posthoc_sigma"]
        tl = np.load(os.path.join(DC.PM, "tlogd.npy"), mmap_mode="r")
        refs = [gaussian_filter(p, sk) if sk > 0 else p
                for p in logd256_to_prob(torch.from_numpy(np.array(tl[k, np.sort(sel)])), L.EVAL)]
        order = {int(v): n for n, v in enumerate(np.sort(sel))}
        refs = [refs[order[int(v)]] for v in sel]
        sig = L.choose_sigma(preds, refs)
        sig_note = "200 Pitt pool images (Pitt rng), teacher-k deployed maps as reference"
    else:
        sig = L.select_sigma_subset(head, data, tr, rng)
        sig_note = "200 ADD1000 training images (base rule, base rng stream), GT reference"
    rows = L.score(head, data, te, sig)
    for x in rows:
        x.update({"fold": k, "seed": seed})
    fi = {"fold": k, "n_train_add1000_gt": 0 if kind == "pitt_only" else int(len(tr)),
          "n_pitt_pool": int(data.P) if kind in ("pitt", "pitt_only") else 0, "steps": steps, "sigma": int(sig),
          "sigma_selection": sig_note, "epoch_loss": loss_log, "device": dev, "wall_s": time.time() - t0,
          "CC": float(np.mean([x["CC"] for x in rows])), "SIM": float(np.mean([x["SIM"] for x in rows])),
          "KLD": float(np.mean([x["KLD"] for x in rows]))}
    print(f"[{row}] fold {k}: sigma={sig} CC={fi['CC']:.4f} SIM={fi['SIM']:.4f} KLD={fi['KLD']:.4f} ({fi['wall_s']:.0f}s on {dev})", flush=True)
    return rows, fi, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, choices=["distill_pitt", "distill_pitt_only", "distill_gt800_teacher800"])
    ap.add_argument("--backbone", default="base", choices=["base", "large"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--yield-s", type=float, default=45)
    ap.add_argument("--epochs", type=int, default=60, help="smoke tests only; every reported row uses 60")
    a = ap.parse_args()
    assert a.epochs == 60 or SMOKE, "non-default epochs only in smoke mode"
    kind = {"distill_pitt": "pitt", "distill_pitt_only": "pitt_only", "distill_gt800_teacher800": "gt800_teacher800"}[a.row]
    row = a.row + ("_L" if a.backbone == "large" else "") + (f"_s{a.seed}" if a.seed else "")
    os.makedirs(SD, exist_ok=True)
    out_sum = os.path.join(DC.RESULTS, f"{row}_summary.json"); out_pi = os.path.join(DC.RESULTS, f"{row}_per_image.json")
    if os.path.exists(out_sum) or os.path.exists(out_pi):
        print("exists, refusing to overwrite", out_sum); return
    torch.set_num_threads(6)
    t_start = time.time()
    dev = a.device
    lock = DC.PoliteLock() if dev == "mps" else None
    data = None
    Fd = json.load(open(os.path.join(DC.KB, "splits", "kfold_seed0.json")))
    for k in [int(x) for x in a.folds.split(",")]:
        fpath = os.path.join(SD, f"{row}_fold{k}.json")
        if os.path.exists(fpath):
            continue
        try:
            os.mkdir(os.path.join(SD, f".claim_{row}_fold{k}"))
        except FileExistsError:
            print(f"fold {k} claimed elsewhere", flush=True); continue
        if lock:
            lock.acquire()
        if data is None:
            data = Data(dev, a.backbone)
            if kind != "gt800_teacher800":
                assert data.pt64 is not None, "Pitt targets missing"
            folds = [[data.pos[i] for i in f] for f in Fd["folds"]]
            assert sorted(sum(folds, [])) == list(range(1000))
        rows, fi, head = run_fold(row, kind, k, data, folds, a.seed, dev, a.backbone, a.epochs)
        if lock:
            lock.release()
        json.dump({"rows": rows, "fold_info": fi}, open(fpath + ".tmp", "w")); os.replace(fpath + ".tmp", fpath)
        torch.save({kk: v.detach().cpu() for kk, v in head.state_dict().items()}, os.path.join(SD, f"{row}_fold{k}_head.pt"))
        del head
        if dev == "mps":
            torch.mps.empty_cache(); time.sleep(a.yield_s)
    # assemble the row when all folds exist
    fps = [os.path.join(SD, f"{row}_fold{k}.json") for k in range(5)]
    if SMOKE or not all(os.path.exists(p) for p in fps) or os.path.exists(out_sum):
        print("row not complete yet (or already assembled)", flush=True); return
    per_image, folds_info = [], []
    for p in fps:
        d = json.load(open(p)); per_image += d["rows"]; folds_info.append(d["fold_info"])
    ids = {r["id"] for r in per_image}
    assert len(per_image) == 1000 and len(ids) == 1000
    json.dump(per_image, open(out_pi, "w"))
    st = L.fold_stats(per_image, 5)
    subset = json.load(open(os.path.join(DC.CD, "pitt_subset.json")))
    teachers = json.load(open(os.path.join(DC.TD, "ALL_DONE.json")))
    summ = {"row": row, "track": "B (distillation from adapted MSDB through unlabeled Pitt ads)", "kind": kind,
            "backbone": a.backbone, "seed": a.seed, **st, "folds_detail": folds_info,
            "trainable_params": L.n_params(L.ReadoutHead("full") if a.backbone == "base" else ReadoutHeadC(FEATS[a.backbone][2])),
            "calibration_and_training_data": {
                "train_weight_maps_per_fold_add1000_gt": 0 if kind == "pitt_only" else 800,
                "teacher_targets": {"pitt": "teacher-k maps on the Pitt pool", "pitt_only": "teacher-k maps on the Pitt pool",
                                    "gt800_teacher800": "teacher-k maps on fold k's 800 training images"}[kind],
                "teacher": "DeepGaze MSDB head (26,463 params) adapted with the msdb_adapt recipe on fold k's 800 training maps only (prior and post-hoc sigma from the same training maps); retrained here because the msdb track did not save its heads; reproduction check vs msdb_adapt seed 0 in cache/distill/teachers/ALL_DONE.json",
                "teacher_check_mean_CC_on_test_folds": teachers["mean_CC"], "msdb_adapt_s0_mean_CC": teachers["msdb_adapt_s0_mean_CC"],
                "pitt_pool": {"n_subset": len(subset["subset_files"]), "n_removed_near_dup_of_add1000": subset["n_removed_union"],
                              "n_unreadable": subset["n_unreadable"], "n_pool": subset["n_pool"], "criteria_counts": subset["criteria_counts"]},
                "sigma_selection": folds_info[0]["sigma_selection"],
                "centre_prior": "learned 16x16 centre-bias logit map trained jointly (base recipe); teacher priors from fold-k training maps only",
                "test_maps_used_for_anything_but_scoring": 0},
            "protocol": {"backbone": f"frozen facebook/dinov2-{a.backbone}, 448x448, patch tokens cached fp16",
                         "steps": STEPS_NOTE, "pseudo_label_weight": W_PSEUDO, "batch": "16 ADD1000 ids per step" + (
                             " + 16 Pitt images" if kind == "pitt" else "") if kind != "pitt_only" else "16 Pitt images per step",
                         "optimizer": "AdamW lr 3e-4 wd 1e-4", "loss": "KL + (1 - CC) at 64x64 for every target",
                         "finalizer": "Gaussian sigma from {0,2,...,12} by mean CC on 200 training images",
                         "scoring": "unified: 384x384, GT bilinear to 384, clip at 0 and divide by sum; CC/SIM/KLD (eps 1e-8)",
                         "a_priori_choices": "w = 1.0 and 3000 steps fixed before any run; nothing tuned"},
            "inference_cost": "identical to ours (frozen DINOv2-base 448 + 3.72M readout): 221 GFLOPs, 0.122 s CPU (cost_table.json); the teacher is used only to make training targets" if a.backbone == "base" else "DINOv2-large 448 + readout (see track A cost rows)",
            "wall_clock_s_last_process": time.time() - t_start, "torch": torch.__version__, "machine": platform.platform(),
            "command": " ".join(["python", "-I"] + sys.argv)}
    json.dump(summ, open(out_sum, "w"), indent=1)
    print(f"[{row}] mean CC={st['mean_CC']:.4f} sd={st['sd_CC_ddof1']:.4f} SIM={st['mean_SIM']:.4f} KLD={st['mean_KLD']:.4f}", flush=True)


if __name__ == "__main__":
    main()
