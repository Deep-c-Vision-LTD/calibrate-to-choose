"""
Track A rows: our readout recipe (scripts/ours, kfold v3, scratch init) on other frozen
DINOv2 feature sets. Copy of scripts/ours/run_row.py (kind=kfold only); the ONLY change is
the input features (and hence the readout's input channels / token grid). Training, loss,
epochs, batch, optimiser, init-seed rule, data-order rng, start-of-training sigma draw,
finalizer rule (sigma from {0,...,12} by mean CC on a random 200-image subset of the training
fold) and the unified 384 scorer are imported unchanged from scripts/ours/ours_lib.py.

  --feats  comma-separated cache files (relative to cache), concatenated along
           channels in the given order; all must share the token grid.

  python -I run_row_a.py --row ours_L448_s0 --feats trackA/feats_dinov2l_448_fp16.npy --seed 0 --device mps
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, argparse, platform
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
OURS = os.path.join(os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))), "scripts/ours")
sys.path.insert(0, OURS)
import ours_lib as L  # noqa: E402
from model_aspp import ASPPModule  # noqa: E402


class ReadoutHeadA(nn.Module):
    """ours_lib.ReadoutHead (mode full) with a configurable number of input channels."""
    def __init__(self, in_ch):
        super().__init__()
        self.aspp = ASPPModule(in_ch, 128)
        self.head = nn.Conv2d(128, 1, 1)
        self.use_cb = True
        self.center_bias = nn.Parameter(torch.zeros(1, 1, 16, 16))

    def forward(self, f):
        s = self.head(self.aspp(f))
        return s + F.interpolate(self.center_bias, size=s.shape[-2:], mode="bilinear", align_corners=False)


class DataA:
    """ours_lib.Data with a configurable (concatenated) feature cache."""
    def __init__(self, dev, feat_files):
        self.ids = L.load_ids()
        self.pos = {i: n for n, i in enumerate(self.ids)}
        parts, self.shapes = [], []
        for fn in feat_files:
            f = np.load(os.path.join(L.KB, "cache", fn), mmap_mode="r")
            assert f.shape[0] == 1000 and f.dtype == np.float16, (fn, f.shape, f.dtype)
            self.shapes.append(list(f.shape))
            parts.append(torch.from_numpy(np.array(f)))
        assert len({tuple(s[2:]) for s in self.shapes}) == 1, self.shapes
        self.feats = (parts[0] if len(parts) == 1 else torch.cat(parts, 1)).to(dev)
        del parts
        self.in_ch = int(self.feats.shape[1]); self.grid = list(self.feats.shape[2:])
        self.gtl = torch.stack([torch.tensor(L.load_gt_train(os.path.join(L.FLAT, "maps", f"{i}.png"), L.LOSSG), dtype=torch.float32)
                                for i in self.ids]).to(dev)
        self.gte = [L.load_gt_eval(os.path.join(L.FLAT, "maps", f"{i}.png")) for i in self.ids]
        self.dev = dev

    def batch(self, idx):
        ix = torch.as_tensor(np.asarray(idx), device=self.dev)
        return self.feats[ix].float(), self.gtl[ix]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds-file", default=os.path.join(L.KB, "splits", "kfold_seed0.json"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-folds", type=int, default=0, help="smoke test only: stop after this many folds (use with OURS_RESULTS_DIR)")
    ap.add_argument("--save-heads", action="store_true", help="save the 5 fold readouts to cache/trackA/heads/<row>_f<k>.pt")
    a = ap.parse_args()
    out_sum = os.path.join(L.RESULTS, f"{a.row}_summary.json")
    out_pi = os.path.join(L.RESULTS, f"{a.row}_per_image.json")
    if os.path.exists(out_sum) or os.path.exists(out_pi):
        print("exists, refusing to overwrite", out_sum); return
    dev = a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    torch.set_num_threads(6)
    t0 = time.time()
    files = [s for s in a.feats.split(",") if s]
    data = DataA(dev, files)
    t_load = time.time() - t0
    nparam = L.n_params(ReadoutHeadA(data.in_ch))
    print(f"[{a.row}] feats {files} -> in_ch {data.in_ch} grid {data.grid}; readout params {nparam}; load {t_load:.0f}s", flush=True)
    per_image, folds_info = [], []
    Fj = json.load(open(a.folds_file))
    folds = [[data.pos[i] for i in f] for f in Fj["folds"]]
    assert sorted(sum(folds, [])) == list(range(1000))
    nrun = a.max_folds or len(folds)
    if a.max_folds:
        assert os.path.abspath(L.RESULTS) != os.path.join(L.KB, "results"), "partial runs must not write to results/"
    for k in range(nrun):
        tk = time.time()
        te = np.array(folds[k])
        tr = np.concatenate([np.array(folds[j]) for j in range(len(folds)) if j != k])
        torch.manual_seed(1000 * a.seed + k)
        head = ReadoutHeadA(data.in_ch).to(dev)
        rng = np.random.default_rng(k + 1000 * a.seed)
        fi = {"fold": k, "n_train": int(len(tr)), "n_test": int(len(te))}
        s0 = L.select_sigma_subset(head, data, tr, rng)          # start-of-training draw, kept for the rng stream
        st = L.score(head, data, te, s0)
        fi.update({"start_sigma": s0, "start_CC": float(np.mean([x["CC"] for x in st]))})
        L.train(head, data, tr, rng, epochs=a.epochs, lr=a.lr, use_cc=True)
        sig = L.select_sigma_subset(head, data, tr, rng)
        rows = L.score(head, data, te, sig)
        for x in rows:
            x.update({"fold": k, "seed": a.seed})
        per_image += rows
        fi.update({"sigma": sig, "CC": float(np.mean([x["CC"] for x in rows])), "SIM": float(np.mean([x["SIM"] for x in rows])),
                   "KLD": float(np.mean([x["KLD"] for x in rows])), "wall_s": time.time() - tk})
        if a.save_heads:
            hd = os.path.join(L.KB, "cache", "trackA", "heads"); os.makedirs(hd, exist_ok=True)
            torch.save({k2: v.cpu() for k2, v in head.state_dict().items()}, os.path.join(hd, f"{a.row}_f{k}.pt"))
        folds_info.append(fi)
        print(f"[{a.row}] fold {k}: sigma={sig} CC={fi['CC']:.4f} SIM={fi['SIM']:.4f} KLD={fi['KLD']:.4f} ({fi['wall_s']:.0f}s)", flush=True)
    json.dump(per_image, open(out_pi, "w"))
    st = L.fold_stats(per_image, nrun)
    summ = {"row": a.row, "kind": "kfold", "init": "scratch", "seed": a.seed, "mode": "full",
            "folds_file": a.folds_file, **st, "folds_detail": folds_info,
            "calibration_and_training_data": {
                "sigma_selection_maps_per_fold": L.SIGMA_SELECT_N, "sigma_selection_maps_are_subset_of_training_maps": True,
                "train_weight_maps_per_fold": int(len(tr)),
                "centre_prior": "learned 16x16 centre-bias logit map, trained jointly with the readout on the same training maps",
                "warm_start_source": "random init"},
            "trainable_params": nparam,
            "features": {"files": files, "shapes": data.shapes, "in_ch": data.in_ch, "grid": data.grid, "description": a.desc},
            "protocol": {"backbone": a.desc or ",".join(files),
                         "recipe": "scripts/ours kfold v3 (ours_lib.train/select_sigma_subset/score unchanged); only the input features differ",
                         "epochs": a.epochs, "lr": a.lr, "batch": 16, "optimizer": "AdamW wd 1e-4", "loss": "KL + (1 - CC) at 64x64",
                         "finalizer": "Gaussian sigma from {0,2,4,6,8,10,12} chosen by mean CC on a random 200-image subset of the training fold",
                         "scoring": "unified: 384x384, GT bilinear to 384, clip at 0 and divide by sum; CC/SIM/KLD (eps 1e-8)",
                         "init_seed_rule": "torch.manual_seed(1000*seed + fold); data-order rng default_rng(fold + 1000*seed)"},
            "wall_clock_s": time.time() - t0, "feature_load_s": t_load, "device": dev,
            "torch": torch.__version__, "machine": platform.platform(),
            "command": " ".join([os.path.basename(sys.executable), "-I"] + sys.argv)}
    json.dump(summ, open(out_sum, "w"), indent=1)
    print(f"[{a.row}] mean CC={st['mean_CC']:.4f} sd1={st['sd_CC_ddof1']:.4f} SIM={st['mean_SIM']:.4f} KLD={st['mean_KLD']:.4f} wall={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
