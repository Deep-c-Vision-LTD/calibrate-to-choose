"""
Run one result row of the "ours" track and write
  results/<row>_per_image.json  (list of {id, fold, CC, SIM, KLD, seed, ...})
  results/<row>_summary.json

kinds
  kfold : kfold_train_eval.py v3 protocol (start-of-training sigma selection kept so the rng
          stream matches the published script; 60 epochs, bs 16, AdamW 3e-4 / wd 1e-4,
          KL + (1 - CC) at 64x64, Gaussian finalizer sigma chosen on a random 200-image subset
          of the training fold from {0,2,...,12}); folds from --folds-file
  abl   : ablation_add1000.py protocol (warm start, no start-of-training selection), --mode one of
          no_aspp | no_centerbias | no_finalizer | kl_only | full
  curve : sample_efficiency.py v3 protocol on the fixed fold-0 test set (200 ids): n training
          maps drawn by default_rng(1000 n + r) from the 800-id pool, shuffling rng = default_rng(r),
          bs = min(16, n), sigma chosen on all n training images

Seeds: init seed s -> torch.manual_seed(1000 s + fold) before building the readout (kfold/abl;
curves use 1000 n + r) and data-order rng default_rng(fold + 1000 s); s = 0 gives exactly the
published data order / sigma-subset draws (the published scripts did not seed the init).
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, argparse, platform
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ours_lib as L  # noqa: E402


def build_head(mode, init, seed_torch, dev):
    torch.manual_seed(seed_torch)
    head = L.ReadoutHead(mode)
    info = {}
    if init == "warm":
        res = head.load_state_dict(L.warm_state(), strict=False)
        info = {"missing_keys": list(res.missing_keys), "n_unexpected_keys": len(res.unexpected_keys)}
    return head.to(dev), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--kind", choices=["kfold", "abl", "curve"], required=True)
    ap.add_argument("--init", choices=["scratch", "warm"], default="scratch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--folds-file", default=os.path.join(L.KB, "splits", "kfold_seed0.json"))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--r", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-singleton-batch", action="store_true",
                    help="skip a trailing 1-image batch (needed only when n_train %% 16 == 1, e.g. grouped fold 3)")
    a = ap.parse_args()
    out_sum = os.path.join(L.RESULTS, f"{a.row}_summary.json")
    out_pi = os.path.join(L.RESULTS, f"{a.row}_per_image.json")
    if os.path.exists(out_sum) and not a.force:
        print("exists, skipping", out_sum); return
    dev = a.device if (a.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    torch.set_num_threads(6)
    t0 = time.time()
    data = L.Data(dev)
    t_load = time.time() - t0
    per_image, folds_info = [], []
    mode = a.mode if a.kind == "abl" else "full"
    nparam = L.n_params(L.ReadoutHead(mode))

    if a.kind in ("kfold", "abl"):
        F = json.load(open(a.folds_file))
        folds = [[data.pos[i] for i in f] for f in F["folds"]]
        assert sorted(sum(folds, [])) == list(range(1000))
        for k in range(len(folds)):
            tk = time.time()
            te = np.array(folds[k])
            tr = np.concatenate([np.array(folds[j]) for j in range(len(folds)) if j != k])
            head, winfo = build_head(mode, a.init, 1000 * a.seed + k, dev)
            rng = np.random.default_rng(k + 1000 * a.seed)
            fi = {"fold": k, "n_train": int(len(tr)), "n_test": int(len(te)), "warm_load": winfo}
            if a.kind == "kfold":
                s0 = L.select_sigma_subset(head, data, tr, rng)
                st = L.score(head, data, te, s0)
                fi.update({"start_sigma": s0, "start_CC": float(np.mean([x["CC"] for x in st])),
                           "start_SIM": float(np.mean([x["SIM"] for x in st])), "start_KLD": float(np.mean([x["KLD"] for x in st]))})
            L.train(head, data, tr, rng, epochs=a.epochs, lr=a.lr, use_cc=(mode != "kl_only"), skip_singleton=a.skip_singleton_batch)
            if a.skip_singleton_batch and len(tr) % 16 == 1:
                fi["singleton_batch_skipped"] = "yes: n_train % 16 == 1, so 1 shuffled image per epoch (a different one each epoch) is not used for that epoch's update"
            sig = L.select_sigma_subset(head, data, tr, rng, allow=(mode != "no_finalizer"))
            rows = L.score(head, data, te, sig)
            for x in rows:
                x.update({"fold": k, "seed": a.seed})
            per_image += rows
            fi.update({"sigma": sig, "CC": float(np.mean([x["CC"] for x in rows])), "SIM": float(np.mean([x["SIM"] for x in rows])),
                       "KLD": float(np.mean([x["KLD"] for x in rows])), "wall_s": time.time() - tk})
            folds_info.append(fi)
            print(f"[{a.row}] fold {k}: sigma={sig} CC={fi['CC']:.4f} SIM={fi['SIM']:.4f} KLD={fi['KLD']:.4f} ({fi['wall_s']:.0f}s)", flush=True)
        nfold = len(folds)
        calib = {"sigma_selection_maps_per_fold": (0 if mode == "no_finalizer" else min(L.SIGMA_SELECT_N, int(len(tr)))),
                 "sigma_selection_maps_are_subset_of_training_maps": True,
                 "train_weight_maps_per_fold": int(len(tr)),
                 "centre_prior": ("none (centre-bias term removed)" if mode == "no_centerbias" else
                                  "learned 16x16 centre-bias logit map, trained jointly with the readout on the same training maps (no separately fitted prior)"),
                 "warm_start_source": ("readout initialised from a UEyes-trained checkpoint (0 ADD1000 maps)" if a.init == "warm" else "random init")}
    else:
        F = json.load(open(os.path.join(L.KB, "splits", "kfold_seed0.json")))
        idx = np.random.default_rng(0).permutation(1000)
        te, pool = idx[:200], idx[200:]
        assert [data.ids[j] for j in te] == F["folds"][0]
        n, r = a.n, a.r
        rr = np.random.default_rng(1000 * n + r)
        tr = rr.choice(pool, size=n, replace=False)
        tk = time.time()
        head, winfo = build_head("full", a.init, 1000 * n + r, dev)
        rng = np.random.default_rng(r)
        L.train(head, data, tr, rng, epochs=a.epochs, lr=a.lr, bs=min(16, n))
        preds = [p.astype(np.float64) for p in L.raw_preds(head, data, tr)]   # sample_efficiency.py selects on float64 preds
        sig = L.choose_sigma(preds, [data.gte[i] for i in tr])
        rows = L.score(head, data, te, sig)
        for x in rows:
            x.update({"fold": 0, "seed": r, "n": n, "r": r})
        per_image = rows
        folds_info = [{"fold": 0, "n_train": n, "n_test": 200, "sigma": sig, "warm_load": winfo,
                       "train_ids": [data.ids[j] for j in tr],
                       "CC": float(np.mean([x["CC"] for x in rows])), "SIM": float(np.mean([x["SIM"] for x in rows])),
                       "KLD": float(np.mean([x["KLD"] for x in rows])), "wall_s": time.time() - tk}]
        print(f"[{a.row}] n={n} r={r}: sigma={sig} CC={folds_info[0]['CC']:.4f} SIM={folds_info[0]['SIM']:.4f} KLD={folds_info[0]['KLD']:.4f}", flush=True)
        nfold = 1
        calib = {"sigma_selection_maps": n, "sigma_selection_maps_are_subset_of_training_maps": True,
                 "train_weight_maps": n,
                 "centre_prior": "learned 16x16 centre-bias logit map, trained jointly on the same n maps",
                 "warm_start_source": ("readout initialised from a UEyes-trained checkpoint (0 ADD1000 maps)" if a.init == "warm" else "random init")}

    json.dump(per_image, open(out_pi, "w"))
    st = L.fold_stats(per_image, nfold)
    summ = {"row": a.row, "kind": a.kind, "init": a.init, "seed": a.seed, "mode": mode, "n": a.n, "r": a.r,
            "folds_file": a.folds_file if a.kind != "curve" else "fixed fold-0 test set of splits/kfold_seed0.json",
            **{k: v for k, v in st.items()}, "folds_detail": folds_info,
            "calibration_and_training_data": calib,
            "trainable_params": nparam,
            "protocol": {"backbone": "frozen facebook/dinov2-base, 448x448 input, patch tokens cached fp16 (cache/ours)",
                         "epochs": a.epochs, "lr": a.lr, "batch": 16 if a.kind != "curve" else min(16, a.n), "optimizer": "AdamW wd 1e-4",
                         "loss": "KL" if mode == "kl_only" else "KL + (1 - CC) at 64x64",
                         "finalizer": "none (sigma forced 0)" if mode == "no_finalizer" else "Gaussian sigma from {0,2,4,6,8,10,12} chosen by mean CC on training images",
                         "scoring": "unified: 384x384, GT bilinear to 384, clip at 0 and divide by sum; CC/SIM/KLD (eps 1e-8)",
                         "native_resolution": "the readout's 32x32 logits are bilinearly upsampled to 384 inside the model, so the native score is the 384 score",
                         "init_seed_rule": "torch.manual_seed(1000*seed + fold) [curves: 1000*n + r]; data-order rng default_rng(fold + 1000*seed) [curves: default_rng(r)]",
                         "skip_singleton_batch": bool(a.skip_singleton_batch),
                         "gt_note": "training targets use the published load_gt (min subtracted); evaluation GT uses the unified clip/sum rule (differs from the published scorer only for SIM/KLD on the 1 map whose resized minimum is above 0)"},
            "wall_clock_s": time.time() - t0, "feature_load_s": t_load, "device": dev,
            "torch": torch.__version__, "machine": platform.platform(),
            "command": " ".join([os.path.basename(sys.executable), "-I"] + sys.argv)}
    json.dump(summ, open(out_sum, "w"), indent=1)
    print(f"[{a.row}] mean CC={st['mean_CC']:.4f} sd={st['sd_CC_ddof0']:.4f} SIM={st['mean_SIM']:.4f} KLD={st['mean_KLD']:.4f} wall={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
