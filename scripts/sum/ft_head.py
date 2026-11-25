"""Head-only in-domain adaptation of SUM on ADD1000 (rows sum_ft_head / sum_ft_head_klcc).

SUM (released weights, token 2 = e-commerce) is frozen up to its last decoder stage; the cached
64x64x96 decoder output (cache_decoder.py) feeds SUM's own output head (final_up: Linear 96->384 +
pixel-shuffle x4 + LayerNorm(24); final_conv: 1x1 conv 24->1), which is initialised from the
released weights and trained on the fold's 800 training maps. Training recipe follows SUM's
train.py (Adam lr 1e-4, StepLR(step 4, gamma 0.1), batch 16) with a fixed epoch count (no early
stopping, test fold never consulted). Post-hoc: the same Gaussian-finalizer rule as our model
(sigma from {0,2,...,12} by mean CC on 200 training images, numpy default_rng(fold)).

Losses
  --loss sum   SUM's objective -2*CC + 10*KL - SIM + 5*MSE on sigmoid(logits) vs map/255
               (NSS dropped: ADD1000 has density maps only)
  --loss klcc  KL(gt || softmax(logits)) + (1 - CC(sigmoid(logits), gt))

python -I ft_head.py --scripts DIR --code DIR --weights DIR --data ADD1000/flat --split kfold_seed0.json \
   --dec_cache cache/sum_dec --results results --row sum_ft_head --loss sum
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
for a in ("scripts", "code", "weights", "data", "split", "dec_cache", "results", "row"):
    ap.add_argument(f"--{a}", required=True)
ap.add_argument("--loss", choices=["sum", "klcc"], default="sum")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--step", type=int, default=4)
ap.add_argument("--gamma", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--device", default="cpu")
ap.add_argument("--diagnostic", action="store_true", help="mark the summary as a diagnostic, not a table row")
args = ap.parse_args()
sys.path.insert(0, args.scripts)
import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import sum_common as sc

T_START = time.time()
torch.set_num_threads(args.threads)
dev = args.device
EPS_SUM = 2.2204e-16
folds = sc.load_split(args.split)
di = json.load(open(os.path.join(args.dec_cache, "dec_index.json")))
assert bool(np.load(os.path.join(args.dec_cache, "dec_done.npy")).all()), "decoder cache incomplete"
assert di["token"] == 2
pos = {i: n for n, i in enumerate(di["ids"])}
FEAT = torch.from_numpy(np.load(os.path.join(args.dec_cache, "dec_feat.npy")))      # (n,64,64,96)
all_ids = [i for f in folds for i in f]
G256 = {i: sc.load_gt(os.path.join(args.data, "maps", f"{i}.png"), 256) for i in all_ids}
G384 = {i: sc.load_gt(os.path.join(args.data, "maps", f"{i}.png"), sc.S_EVAL) for i in all_ids}
G256t = torch.from_numpy(np.stack([G256[i] for i in di["ids"]]).astype(np.float32) / 255.0)  # by cache pos

model, _ = sc.load_sum(args.code, args.weights, "cpu")
net = model.salu_mamba
head0 = {"final_up": {k: v.clone() for k, v in net.final_up.state_dict().items()},
         "final_conv": {k: v.clone() for k, v in net.final_conv.state_dict().items()}}
head = torch.nn.ModuleDict({"final_up": net.final_up, "final_conv": net.final_conv}).to(dev)
n_head = int(sum(p.numel() for p in head.parameters()))


def head_fwd(x):
    x = head["final_up"](x)
    return head["final_conv"](x.permute(0, 3, 1, 2))


def sum_loss(pred, gt):
    """Adapted from SUM utils/loss_function.py (github.com/Arhosseini77/SUM, MIT); NSS omitted."""
    B = pred.shape[0]
    p = pred.reshape(B, -1); g = gt.reshape(B, -1)
    pn = p / p.sum(1, keepdim=True); gn = g / g.sum(1, keepdim=True)
    kl = (gn * torch.log(EPS_SUM + gn / (pn + EPS_SUM))).sum(1).mean()
    pc = p - p.mean(1, keepdim=True); gc = g - g.mean(1, keepdim=True)
    ccv = ((pc * gc).sum(1) / torch.sqrt((gc * gc).sum(1) * (pc * pc).sum(1))).mean()
    gm = (g - g.min(1, keepdim=True)[0]) / (g.max(1, keepdim=True)[0] - g.min(1, keepdim=True)[0])
    gm = gm / gm.sum(1, keepdim=True)
    pm = (p - p.min(1, keepdim=True)[0]) / (p.max(1, keepdim=True)[0] - p.min(1, keepdim=True)[0])
    pm = pm / pm.sum(1, keepdim=True)
    simv = torch.minimum(gm, pm).sum(1).mean()
    mse = F.mse_loss(pred, gt)
    return -2 * ccv + 10 * kl - 1 * simv + 5 * mse, {"kl": float(kl), "cc": float(ccv), "sim": float(simv),
                                                       "mse": float(mse)}


def klcc_loss(logits, gt):
    B = logits.shape[0]
    s = logits.reshape(B, -1)
    g = gt.reshape(B, -1); g = g / (g.sum(1, keepdim=True) + 1e-12)
    logp = F.log_softmax(s, 1)
    kl = (g * (torch.log(g + 1e-12) - logp)).sum(1).mean()
    ps = torch.sigmoid(s)
    pc = ps - ps.mean(1, keepdim=True); gc = g - g.mean(1, keepdim=True)
    ccb = (pc * gc).sum(1) / (torch.sqrt((pc ** 2).sum(1) * (gc ** 2).sum(1)) + 1e-8)
    return kl + (1 - ccb.mean()), {"kl": float(kl), "cc": float(ccb.mean())}


@torch.no_grad()
def predict(ids, bs=16):
    head.eval(); out = {}
    for j in range(0, len(ids), bs):
        b = ids[j:j + bs]
        ix = torch.tensor([pos[i] for i in b])
        p = torch.sigmoid(head_fwd(FEAT[ix].to(dev)))[:, 0].float().cpu().numpy()
        out.update(zip(b, p))
    return out


per_image, fold_meta, trainlog = [], {}, []
for k in range(len(folds)):
    t_fold = time.time()
    te, tr = folds[k], sc.train_ids_for(folds, k)
    for name in ("final_up", "final_conv"):
        head[name].load_state_dict(head0[name])
    torch.manual_seed(args.seed * 100 + k)
    rng = np.random.default_rng(args.seed * 100 + k)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step, gamma=args.gamma)
    for ep in range(args.epochs):
        head.train()
        order = list(tr); rng.shuffle(order)
        losses, comps, t_ep = [], [], time.time()
        for j in range(0, len(order) - args.batch + 1, args.batch):
            b = order[j:j + args.batch]
            ix = torch.tensor([pos[i] for i in b])
            x, g = FEAT[ix].to(dev), G256t[ix].to(dev)[:, None]
            logits = head_fwd(x)
            loss, comp = sum_loss(torch.sigmoid(logits), g) if args.loss == "sum" else klcc_loss(logits, g)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if not np.isfinite(float(loss)):
                raise RuntimeError(f"non-finite loss fold {k} epoch {ep}")
            losses.append(float(loss)); comps.append(comp)
        sched.step()
        rec = {"fold": k, "epoch": ep, "lr_next": opt.param_groups[0]["lr"], "loss": float(np.mean(losses)),
               "steps": len(losses), "seconds": time.time() - t_ep,
               **{key: float(np.mean([c[key] for c in comps])) for key in comps[0]}}
        trainlog.append(rec); print(json.dumps(rec), flush=True)
    t_train = time.time() - t_fold
    sel = [str(s) for s in np.random.default_rng(k).choice(tr, size=sc.SIGMA_SELECT_N, replace=False)]
    preds = predict(sel + te)
    p384 = {i: sc.resize_map(preds[i], sc.S_EVAL) for i in preds}
    best, best_cc, table = 0, -2.0, {}
    for s in sc.SIGMAS:
        m = float(np.mean([sc.cc(gaussian_filter(p384[i], s) if s > 0 else p384[i], G384[i]) for i in sel]))
        table[str(s)] = m
        if m > best_cc:
            best, best_cc = s, m
    for i in te:
        p = gaussian_filter(p384[i], best) if best > 0 else p384[i]
        m, mr, mn = sc.metrics(p, G384[i]), sc.metrics(p384[i], G384[i]), sc.metrics(preds[i], G256[i])
        per_image.append({"id": i, "fold": k, "CC": m["CC"], "SIM": m["SIM"], "KLD": m["KLD"], "sigma": best,
                          "seed": args.seed, "token": 2, "n_train": len(tr),
                          "CC_nosigma": mr["CC"], "SIM_nosigma": mr["SIM"], "KLD_nosigma": mr["KLD"],
                          "CC_native256": mn["CC"], "SIM_native256": mn["SIM"], "KLD_native256": mn["KLD"]})
    rows = [r for r in per_image if r["fold"] == k]
    fold_meta[str(k)] = {"sigma": best, "sigma_table_trainCC": table, "train_seconds": t_train,
                         "fold_seconds": time.time() - t_fold, "n_train_weights": len(tr), "n_sigma_select": len(sel),
                         "test_CC": float(np.mean([r["CC"] for r in rows])),
                         "test_CC_nosigma": float(np.mean([r["CC_nosigma"] for r in rows]))}
    print(f"fold {k}: sigma {best} CC {fold_meta[str(k)]['test_CC']:.4f} "
          f"(no sigma {fold_meta[str(k)]['test_CC_nosigma']:.4f}) in {time.time() - t_fold:.0f}s", flush=True)

summ = sc.fold_summary(per_image, folds)
alt = lambda keys: sc.fold_summary([{**r, "CC": r[keys[0]], "SIM": r[keys[1]], "KLD": r[keys[2]]}
                                    for r in per_image], folds)
out = {"row": args.row, **summ,
       "no_sigma_at384": alt(("CC_nosigma", "SIM_nosigma", "KLD_nosigma")),
       "native256_no_sigma": alt(("CC_native256", "SIM_native256", "KLD_native256")),
       "diagnostic_only": bool(args.diagnostic),
       "mode": "head-only: SUM frozen through its last decoder stage (released weights, token 2, eval mode, "
               "features cached once); SUM's own output head (final_up + final_conv) trained from its released init",
       "n_trainable_params": n_head, "params_total": 57501196,
       "loss": ("SUM's objective -2*CC + 10*KL - SIM + 5*MSE (NSS dropped: density maps only)" if args.loss == "sum"
                else "KL(gt || softmax(logits)) + (1 - CC(sigmoid(logits), gt)) at 256x256"),
       "recipe": {"optimizer": "Adam", "lr": args.lr, "scheduler": f"StepLR({args.step}, {args.gamma})",
                  "batch": args.batch, "epochs": args.epochs, "early_stopping": False, "seed": args.seed},
       "calibration_maps_used_per_fold": {"sigma_selection": 200, "prior_fit": 0, "token_choice": 800,
                                          "note": "token 2 chosen on training folds (sum_calibrated); sigma subset "
                                                  "drawn from the fold's 800 training images"},
       "training_maps_used_per_fold": 800,
       "fold_meta": fold_meta, "train_log_per_epoch": trainlog,
       "protocol": ("ADD1000 seed-0 5-fold split (splits/kfold_seed0.json); CC/SIM/KLD on 384x384 (PIL "
                    "bilinear resize of SUM's 256x256 sigmoid output; GT resized bilinearly to 384; clip at 0 and divide "
                    "by sum; EPS 1e-8); post-hoc Gaussian finalizer sigma from {0,2,4,6,8,10,12} by mean CC on 200 "
                    "training images; native256 = raw 256 output vs GT resized to 256, no sigma"),
       "device": dev, "threads": args.threads,
       "wallclock_s_total_incl_loading": time.time() - T_START,
       "wallclock_s_training_only": float(sum(v["train_seconds"] for v in fold_meta.values())),
       "feature_cache_cost": "one SUM forward pass per image through the decoder (see cache/sum_enc + cache/sum_dec meta)",
       "command": "python -I " + " ".join(sys.argv)}
os.makedirs(args.results, exist_ok=True)
json.dump([{k_: r[k_] for k_ in r} for r in per_image],
          open(os.path.join(args.results, f"{args.row}_per_image.json"), "w"))
json.dump(out, open(os.path.join(args.results, f"{args.row}_summary.json"), "w"), indent=1)
print(f"{args.row}: CC {summ['CC_mean']:.4f} +/- {summ['CC_sd_folds']:.4f} SIM {summ['SIM_mean']:.4f} "
      f"KLD {summ['KLD_mean']:.4f} | no-sigma CC {out['no_sigma_at384']['CC_mean']:.4f} | "
      f"native256 CC {out['native256_no_sigma']['CC_mean']:.4f} | {time.time() - T_START:.0f}s", flush=True)
