"""Row (4): in-domain fine-tuning of SUM (released weights) on ADD1000 under the shared
seed-0 5-fold split, followed by the same post-hoc Gaussian-finalizer rule as our model
(sigma chosen per fold on 200 training images from {0,2,...,12} by mean CC; never on test).

Losses
  --loss sum   SUM's own training objective, -2*CC + 10*KL - SIM - NSS + 5*MSE on the
               sigmoid output vs the density map in [0,1] (map/255), with the NSS term
               dropped because ADD1000 releases density maps only (no fixation points).
  --loss klcc  our objective, KL(gt || softmax(logits)) + (1 - CC(sigmoid(logits), gt)).
Optimiser follows SUM's train.py: Adam lr 1e-4, StepLR(step 4, gamma 0.1), batch 16
(gradient accumulation over --micro images; the model has LayerNorm only, so this is exact
up to stochastic depth), stochastic depth 0.3 as released, input 256x256 stretch resize.
Fixed epoch count (no early stopping) so the test fold is never consulted.

The selective scan is the pure-PyTorch chunked scan from sum_common.py unless --scan cuda
is given and mamba-ssm's kernel imports.

python -I ft_sum_folds.py --scripts DIR --code DIR --weights DIR --data DIR \
   --split kfold_seed0.json --out_dir OUT --loss sum --folds 0,1,2,3,4 --epochs 12

Learning-curve mode (--curve; the k-fold path above is untouched when it is not given):
test = fold --curve_fold (200 ids), pool = the other 800 in fold order, training ids for
(n, r) = default_rng(1000 n + r).choice(pool, n, replace=False) (the rule of
ours/run_curves.py, msdb/train_msdb.py --curves, iie/ft_curves.py); every run restarts from
the released weights; data order default_rng(r) (fresh copy shuffled every epoch); --epochs at
every n; batch min(--batch, n) INCLUDING the last partial batch, so steps = epochs x ceil(n/16);
sigma chosen on all n training maps; per-run files written as each run finishes.
python -I ft_sum_folds.py ... --curve 25:0,1,2/50:0,1,2/400:0 --curve_expect curve_subsets_seed0.json
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
for a in ("scripts", "code", "weights", "data", "split", "out_dir"):
    ap.add_argument(f"--{a}", required=True)
ap.add_argument("--loss", choices=["sum", "klcc"], default="sum")
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--micro", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--step", type=int, default=4)
ap.add_argument("--gamma", type=float, default=0.1)
ap.add_argument("--token", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="auto")
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--scan", choices=["chunked", "cuda"], default="chunked")
ap.add_argument("--max_steps", type=int, default=0, help="debug: stop each fold after N optimiser steps")
ap.add_argument("--tag", default=None)
ap.add_argument("--limit_eval", type=int, default=0, help="debug: evaluate only N test and N sigma-selection images")
ap.add_argument("--mode", choices=["full", "decoder"], default="full",
                help="full: all 57.5M parameters trained; decoder: encoder frozen (eval mode, outputs read from "
                     "--enc_cache), only the conditional decoder (13.85M) trained")
ap.add_argument("--enc_cache", default=None, help="dir with enc_skip{0..3}.npy, enc_x.npy, enc_index.json")
# curve flags use SUPPRESS so that, when absent, vars(args) (stored in the k-fold summary) is unchanged
ap.add_argument("--curve", default=argparse.SUPPRESS,
                help="learning-curve mode instead of k-fold: groups 'n:r,r,...' separated by '/' (or ';'), e.g. "
                     "25:0,1,2/50:0,1,2/400:0. Absent (default) = the k-fold path")
ap.add_argument("--curve_fold", type=int, default=argparse.SUPPRESS,
                help="curve mode: test fold (default 0); pool = the other folds in fold order")
ap.add_argument("--curve_expect", default=argparse.SUPPRESS,
                help="curve mode: JSON with the expected train_ids per 'n:r' (stored by the other curves); "
                     "abort before training on any mismatch")
args = ap.parse_args()
CURVE = getattr(args, "curve", ""); CURVE_FOLD = getattr(args, "curve_fold", 0)
CURVE_EXPECT_PATH = getattr(args, "curve_expect", None)
sys.path.insert(0, args.scripts)

import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import sum_common as sc

torch.set_num_threads(args.threads)
if args.device == "auto":
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
else:
    dev = args.device
tag = args.tag or f"sum_ft{'' if args.loss == 'sum' else '_klcc'}"
os.makedirs(args.out_dir, exist_ok=True)
folds = sc.load_split(args.split)
run_folds = [int(f) for f in args.folds.split(",")]
log_path = os.path.join(args.out_dir, f"{tag}_trainlog.jsonl")
EPS_SUM = 2.2204e-16


def log(rec):
    print(json.dumps(rec), flush=True)
    with open(log_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------- data (cached once)
all_ids = [i for f in folds for i in f]
if CURVE:   # learning-curve mode: plan, subset guard, and cache only the ids it needs
    assert args.mode == "full", "curve mode is defined for the full fine-tune only"
    CURVE_RUNS = [(n, r, sc.curve_subset(folds, CURVE_FOLD, n, r)) for n, r in sc.parse_curve(CURVE)]
    CURVE_EXPECT = json.load(open(CURVE_EXPECT_PATH)) if CURVE_EXPECT_PATH else None
    if CURVE_EXPECT is not None:
        for n, r, s in CURVE_RUNS:
            if CURVE_EXPECT["train_ids"].get(f"{n}:{r}") != s:
                sys.exit(f"ABORT: train_ids for n={n} r={r} differ from {CURVE_EXPECT_PATH}")
        print(f"curve subsets match {CURVE_EXPECT_PATH} for all {len(CURVE_RUNS)} runs", flush=True)
    CURVE_TE = folds[CURVE_FOLD][:args.limit_eval] if args.limit_eval else folds[CURVE_FOLD]
    all_ids = list(dict.fromkeys(CURVE_TE + [i for _, _, s in CURVE_RUNS for i in s]))
t_load = time.time()
IMG = {i: sc.load_image_tensor(os.path.join(args.data, "images", f"{i}.jpg")) for i in all_ids}
G256 = {i: torch.from_numpy((sc.load_gt(os.path.join(args.data, "maps", f"{i}.png"), 256) / 255.0)
                            .astype(np.float32)) for i in all_ids}
G384 = {i: sc.load_gt(os.path.join(args.data, "maps", f"{i}.png"), sc.S_EVAL) for i in all_ids}
G256np = {i: sc.load_gt(os.path.join(args.data, "maps", f"{i}.png"), 256) for i in all_ids}
ENC = None
if args.mode == "decoder":
    ei = json.load(open(os.path.join(args.enc_cache, "enc_index.json")))["ids"]
    assert bool(np.load(os.path.join(args.enc_cache, "enc_done.npy")).all()), "encoder cache incomplete"
    arrs = {k: np.load(os.path.join(args.enc_cache, f"{k}.npy"))
            for k in ("enc_skip0", "enc_skip1", "enc_skip2", "enc_skip3", "enc_x")}
    pos = {i: n for n, i in enumerate(ei)}
    ENC = {k: torch.from_numpy(v) for k, v in arrs.items()}
    ENC["pos"] = pos
log({"event": "data_cached", "n": len(all_ids), "seconds": time.time() - t_load, "device": dev, "mode": args.mode})


def enc_batch(b):
    ix = torch.tensor([ENC["pos"][i] for i in b])
    skips = [ENC[f"enc_skip{j}"][ix].to(dev) for j in range(4)]
    return ENC["enc_x"][ix].to(dev), skips


def forward_logits(net, b, c):
    if args.mode == "decoder":
        xe, skips = enc_batch(b)
        return net.forward_final(net.forward_features_up(xe, skips, c))
    x = torch.stack([IMG[i] for i in b]).to(dev)
    return net(x, c)


# ---------------------------------------------------------------- losses
def sum_loss(pred, gt):
    """Adapted from SUM utils/loss_function.py (github.com/Arhosseini77/SUM, MIT): per-image, then batch
    mean; NSS omitted."""
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
    loss = -2 * ccv + 10 * kl - 1 * simv + 5 * mse
    return loss, {"kl": float(kl), "cc": float(ccv), "sim": float(simv), "mse": float(mse)}


def klcc_loss(logits, gt):
    B = logits.shape[0]
    s = logits.reshape(B, -1)
    g = gt.reshape(B, -1); g = g / (g.sum(1, keepdim=True) + 1e-12)
    logp = F.log_softmax(s, 1)
    kl = (g * (torch.log(g + 1e-12) - logp)).sum(1).mean()
    ps = torch.sigmoid(s)
    pc = ps - ps.mean(1, keepdim=True); gc = g - g.mean(1, keepdim=True)
    ccb = (pc * gc).sum(1) / (torch.sqrt((pc ** 2).sum(1) * (gc ** 2).sum(1)) + 1e-8)
    loss = kl + (1 - ccb.mean())
    return loss, {"kl": float(kl), "cc": float(ccb.mean())}


# ---------------------------------------------------------------- per fold
@torch.no_grad()
def predict(model, ids, bs=8):
    model.eval()
    out = {}
    for j in range(0, len(ids), bs):
        b = ids[j:j + bs]
        c = torch.zeros(len(b), 4, device=dev); c[:, args.token] = 1
        p = torch.sigmoid(forward_logits(model.salu_mamba, b, c))[:, 0].float().cpu().numpy()
        for i, pi in zip(b, p):
            out[i] = pi
    return out


def select_sigma(pred384, sel_ids):
    best, best_cc, table = 0, -2.0, {}
    for s in sc.SIGMAS:
        m = float(np.mean([sc.cc(gaussian_filter(pred384[i], s) if s > 0 else pred384[i], G384[i])
                           for i in sel_ids]))
        table[str(s)] = m
        if m > best_cc:
            best, best_cc = s, m
    return best, table


# ---------------------------------------------------------------- learning-curve mode (--curve only)
CURVE_PROTOCOL = {
    "test_set": "fold --curve_fold of the split (fold 0 = 200 ids)",
    "pool": "the other folds concatenated in fold order (= default_rng(0).permutation(1000)[200:] via ids_sorted)",
    "subset_rule": "default_rng(1000*n + r).choice(pool, n, replace=False) (sample_efficiency.py)",
    "init": "released SUM weights reloaded for every run; torch.manual_seed(1000*n + r) (stochastic depth)",
    "data_order": "default_rng(r), fresh copy of the training list shuffled every epoch",
    "epochs": "fixed at every n (SUM's own 12), no early stopping",
    "batch": "min(batch, n) incl. the last partial batch (steps = epochs x ceil(n/batch)); micro-batch "
             "accumulation weighted by micro size / batch size (exact batch mean)",
    "optimiser": "Adam lr, StepLR(step, gamma) stepped once per epoch, as the k-fold path",
    "finalizer": "Gaussian sigma from {0,2,...,12} by mean CC at 384 on all n training maps; no centre prior",
    "scoring": "384x384: prediction and GT bilinear to 384, clip at 0, divide by sum; KLD eps 1e-8 (sum_common.metrics)"}


def _dump_json(obj, path, indent=None):
    # written in place at the final name (no temp file + rename): out_dir may be a checkpoint folder that a
    # cloud job syncs while it runs, and in-place writes are what such a sync uploads (k-fold job markers)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=indent)


def _meta_done(path):
    """a run counts as complete only if its _meta.json (written last) exists and parses"""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def curve_train(n, r, tr):
    torch.manual_seed(1000 * n + r); np.random.seed(1000 * n + r)
    orders = sc.curve_orders(tr, args.epochs, r)
    model, _ = sc.load_sum(args.code, args.weights, dev, scan="chunked")
    if args.scan == "cuda":
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _k
            sys.modules["sum_thirdparty_models.vmamba"].selective_scan_fn = _k
            log({"event": "scan", "impl": "mamba_ssm cuda kernel"})
        except Exception as e:
            log({"event": "scan", "impl": "chunked torch (cuda kernel import failed)", "err": repr(e)[:300]})
    net = model.salu_mamba
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    n_trainable = int(sum(p_.numel() for p_ in trainable))
    assert n_trainable == 57501196, f"expected all 57,501,196 SUM parameters trainable, got {n_trainable}"
    opt = torch.optim.Adam(trainable, lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step, gamma=args.gamma)
    bs = min(args.batch, n); micro = min(args.micro, bs)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step, ep_logs = 0, []
    for ep, order in enumerate(orders):
        model.train()
        t_ep = time.time(); losses = []; comps = []
        for j in range(0, len(order), bs):                   # keeps the last partial batch
            batch = order[j:j + bs]
            opt.zero_grad(set_to_none=True)
            bl = 0.0
            for m in range(0, len(batch), micro):
                b = batch[m:m + micro]
                w = len(b) / len(batch)                      # = 1/acc for a full batch
                g = torch.stack([G256[i] for i in b]).to(dev)[:, None]
                c = torch.zeros(len(b), 4, device=dev); c[:, args.token] = 1
                logits = forward_logits(net, b, c)
                if args.loss == "sum":
                    loss, comp = sum_loss(torch.sigmoid(logits), g)
                else:
                    loss, comp = klcc_loss(logits, g)
                (loss * w).backward()
                bl += float(loss) * w; comps.append(comp)
            opt.step(); step += 1
            losses.append(bl)
            if not np.isfinite(bl):
                raise RuntimeError(f"non-finite loss at n={n} r={r} epoch {ep} step {step}")
            if args.max_steps and step >= args.max_steps:
                break
        sched.step(); sync()
        rec = {"event": "epoch", "curve_n": n, "curve_r": r, "epoch": ep, "lr": opt.param_groups[0]["lr"],
               "loss": float(np.mean(losses)), "seconds": time.time() - t_ep, "steps": len(losses)}
        for key in comps[0]:
            rec[key] = float(np.mean([c_[key] for c_ in comps]))
        if dev == "cuda":
            rec["max_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        elif dev == "mps":
            rec["driver_mem_gb"] = torch.mps.driver_allocated_memory() / 2 ** 30
        log(rec); ep_logs.append(rec)
        if args.max_steps and step >= args.max_steps:
            break
    return model, n_trainable, step, ep_logs, sc.orders_sha1(orders), bs, micro


def curve_run(n, r, tr):
    pre = os.path.join(args.out_dir, f"{tag}_n{n}_r{r}")
    done = _meta_done(pre + "_meta.json")
    if done is not None:
        print(f"curve n={n} r={r} already done, skipping", flush=True)
        return done
    t_run = time.time()
    log({"event": "curve_run_start", "n": n, "r": r, "steps_planned": args.epochs * -(-n // min(args.batch, n))})
    model, n_trainable, steps, ep_logs, osha, bs, micro = curve_train(n, r, tr)
    t_train = time.time() - t_run
    sel = list(tr[:args.limit_eval]) if args.limit_eval else list(tr)       # sigma on all n training maps
    te = list(CURVE_TE)
    t_ev = time.time()
    preds = predict(model, sel + te)
    bad = [i for i in preds if not np.isfinite(preds[i]).all()]
    if bad:   # a NaN map would otherwise score silently as CC 0 (sc.cc returns 0 when d is NaN)
        raise RuntimeError(f"non-finite predictions at n={n} r={r} for {len(bad)} images, e.g. {bad[:3]}")
    p384 = {i: sc.resize_map(preds[i], sc.S_EVAL) for i in preds}
    sigma, table = select_sigma(p384, sel)
    np.save(pre + "_test_maps256.npy", np.stack([preds[i] for i in te]).astype(np.float32))   # in place
    _dump_json(te, pre + "_test_ids.json")
    rows = []
    for i in te:
        p = gaussian_filter(p384[i], sigma) if sigma > 0 else p384[i]
        m = sc.metrics(p, G384[i]); mr = sc.metrics(p384[i], G384[i]); mn = sc.metrics(preds[i], G256np[i])
        rows.append({"id": i, "fold": CURVE_FOLD, "n": n, "r": r, "resample": r, "seed": r,
                     "CC": m["CC"], "SIM": m["SIM"], "KLD": m["KLD"], "sigma": sigma, "token": args.token,
                     "CC_nosigma": mr["CC"], "SIM_nosigma": mr["SIM"], "KLD_nosigma": mr["KLD"],
                     "CC_native256": mn["CC"], "SIM_native256": mn["SIM"], "KLD_native256": mn["KLD"]})
    _dump_json(rows, pre + "_per_image.json")
    t_eval = time.time() - t_ev
    exp_sha = (CURVE_EXPECT or {}).get("orders_sha1", {}).get(f"{n}:{r}")
    meta = {"row": tag, "kind": "curve", "n": n, "r": r, "resample": r, "train_ids": list(tr),
            "test_fold": CURVE_FOLD, "n_test": len(te), "n_sigma_select": len(sel),
            "sigma": sigma, "sigma_table_trainCC": table,
            **{f"test_{k}": float(np.mean([x[k] for x in rows])) for k in
               ("CC", "SIM", "KLD", "CC_nosigma", "SIM_nosigma", "KLD_nosigma",
                "CC_native256", "SIM_native256", "KLD_native256")},
            "steps": steps, "steps_planned": args.epochs * -(-n // bs), "epochs": args.epochs,
            "batch": bs, "micro": micro, "lr": args.lr, "step_size": args.step, "gamma": args.gamma,
            "loss": args.loss, "token": args.token, "n_trainable_params": n_trainable,
            "train_seconds": t_train, "eval_seconds": t_eval, "run_seconds": time.time() - t_run,
            "seconds_per_step": t_train / max(steps, 1),
            "orders_sha1": osha, "orders_sha1_expected": exp_sha,
            "orders_match_expected": (osha == exp_sha) if exp_sha else None,
            "subset_checked_against": CURVE_EXPECT_PATH,
            "max_steps": args.max_steps, "limit_eval": args.limit_eval, "epoch_logs": ep_logs,
            "protocol": CURVE_PROTOCOL, "device": dev, "torch": torch.__version__, "numpy": np.__version__}
    _dump_json(meta, pre + "_meta.json", indent=1)                      # written last = run complete
    log({"event": "curve_run_done", **{k: meta[k] for k in
         ("n", "r", "sigma", "test_CC", "test_SIM", "test_KLD", "test_CC_nosigma", "steps",
          "train_seconds", "eval_seconds", "orders_match_expected")}})
    del model
    _free()
    return meta


def _free():
    import gc
    gc.collect()   # the optimizer/scheduler pair forms a reference cycle; free it before the next run
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev == "cuda":
        torch.cuda.empty_cache()


def run_curve():
    """Runs every planned (n, r). A failing run (non-finite loss/predictions, OOM, ...) writes
    <tag>_n{n}_r{r}_error.json, is listed under 'failed' in the summary, and the job continues with
    the next run; the process exits non-zero at the end if any run failed. Returns that exit code."""
    import traceback
    runs, failed = [], []
    for n, r, tr in CURVE_RUNS:
        meta = None
        try:
            meta = curve_run(n, r, tr)
        except Exception as e:   # noqa: BLE001 - recorded and re-signalled through the exit code
            err = {"n": n, "r": r, "error": repr(e)[:2000], "traceback": traceback.format_exc()[-6000:]}
            log({"event": "curve_run_failed", "n": n, "r": r, "error": err["error"]})
            _dump_json(err, os.path.join(args.out_dir, f"{tag}_n{n}_r{r}_error.json"), indent=1)
            failed.append({"n": n, "r": r, "error": err["error"]})
        if meta is None:
            _free()
        else:
            runs.append({k: meta[k] for k in ("n", "r", "sigma", "test_CC", "test_SIM", "test_KLD",
                                              "test_CC_nosigma", "test_SIM_nosigma", "test_KLD_nosigma",
                                              "steps", "steps_planned", "train_seconds", "eval_seconds",
                                              "n_sigma_select", "n_test", "orders_match_expected")})
        _dump_json({"row": tag, "kind": "curve", "curve": CURVE, "curve_fold": CURVE_FOLD,
                    "plan": [[n_, r_] for n_, r_, _ in CURVE_RUNS],
                    "complete": len(runs) == len(CURVE_RUNS), "runs": runs, "failed": failed,
                    "protocol": CURVE_PROTOCOL, "args": vars(args), "device": dev, "params_trained": 57501196},
                   os.path.join(args.out_dir, f"{tag}_curve_summary.json"), indent=1)
    print(f"ALL CURVE RUNS DONE {tag}: " + "; ".join(f"n={x['n']} r={x['r']} CC {x['test_CC']:.4f}" for x in runs)
          + (f" | FAILED: {[(x['n'], x['r']) for x in failed]}" if failed else ""), flush=True)
    return 1 if failed else 0


if CURVE:
    sys.exit(run_curve())

per_image_path = os.path.join(args.out_dir, f"{tag}_per_image.json")
per_image = json.load(open(per_image_path)) if os.path.exists(per_image_path) else []
fold_meta_path = os.path.join(args.out_dir, f"{tag}_fold_meta.json")
fold_meta = json.load(open(fold_meta_path)) if os.path.exists(fold_meta_path) else {}

for k in run_folds:
    if str(k) in fold_meta:
        print(f"fold {k} already done, skipping", flush=True)
        continue
    t_fold = time.time()
    te = folds[k]
    tr = sc.train_ids_for(folds, k)
    torch.manual_seed(args.seed * 100 + k); np.random.seed(args.seed * 100 + k)
    rng = np.random.default_rng(args.seed * 100 + k)
    model, _ = sc.load_sum(args.code, args.weights, dev, scan="chunked")
    if args.scan == "cuda":
        try:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _k
            sys.modules["sum_thirdparty_models.vmamba"].selective_scan_fn = _k
            log({"event": "scan", "impl": "mamba_ssm cuda kernel"})
        except Exception as e:
            log({"event": "scan", "impl": "chunked torch (cuda kernel import failed)", "err": repr(e)[:300]})
    net = model.salu_mamba
    if args.mode == "decoder":
        for n_, p_ in net.named_parameters():
            p_.requires_grad_(n_.startswith("layers_up") or n_.startswith("final"))
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    n_trainable = int(sum(p_.numel() for p_ in trainable))
    log({"event": "trainable", "fold": k, "n_trainable": n_trainable, "mode": args.mode})
    opt = torch.optim.Adam(trainable, lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step, gamma=args.gamma)
    acc = args.batch // args.micro
    step = 0
    for ep in range(args.epochs):
        model.train()
        order = list(tr); rng.shuffle(order)
        t_ep = time.time(); losses = []; comps = []
        for j in range(0, len(order) - args.batch + 1, args.batch):   # drop last partial batch
            opt.zero_grad(set_to_none=True)
            bl = 0.0
            for m in range(acc):
                b = order[j + m * args.micro: j + (m + 1) * args.micro]
                g = torch.stack([G256[i] for i in b]).to(dev)[:, None]
                c = torch.zeros(len(b), 4, device=dev); c[:, args.token] = 1
                logits = forward_logits(net, b, c)
                if args.loss == "sum":
                    loss, comp = sum_loss(torch.sigmoid(logits), g)
                else:
                    loss, comp = klcc_loss(logits, g)
                (loss / acc).backward()
                bl += float(loss) / acc; comps.append(comp)
            opt.step(); step += 1
            losses.append(bl)
            if not np.isfinite(bl):
                raise RuntimeError(f"non-finite loss at fold {k} epoch {ep} step {step}")
            if args.max_steps and step >= args.max_steps:
                break
        sched.step(); sync()
        rec = {"event": "epoch", "fold": k, "epoch": ep, "lr": opt.param_groups[0]["lr"],
               "loss": float(np.mean(losses)), "seconds": time.time() - t_ep, "steps": len(losses)}
        for key in comps[0]:
            rec[key] = float(np.mean([c[key] for c in comps]))
        if dev == "cuda":
            rec["max_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        elif dev == "mps":
            rec["driver_mem_gb"] = torch.mps.driver_allocated_memory() / 2 ** 30
        log(rec)
        if args.max_steps and step >= args.max_steps:
            break
    t_train = time.time() - t_fold
    # ---- sigma finalizer on training images only (same rule as kfold_train_eval.py)
    sel = [str(s) for s in np.random.default_rng(k).choice(tr, size=sc.SIGMA_SELECT_N, replace=False)]
    if args.limit_eval:
        sel, te = sel[:args.limit_eval], te[:args.limit_eval]
    preds = predict(model, sel + te)
    p384 = {i: sc.resize_map(preds[i], sc.S_EVAL) for i in preds}
    sigma, table = select_sigma(p384, sel)
    np.save(os.path.join(args.out_dir, f"{tag}_fold{k}_test_maps256.npy"),
            np.stack([preds[i] for i in te]).astype(np.float32))
    json.dump(te, open(os.path.join(args.out_dir, f"{tag}_fold{k}_test_ids.json"), "w"))
    for i in te:
        p = gaussian_filter(p384[i], sigma) if sigma > 0 else p384[i]
        m = sc.metrics(p, G384[i]); mr = sc.metrics(p384[i], G384[i]); mn = sc.metrics(preds[i], G256np[i])
        per_image.append({"id": i, "fold": k, "CC": m["CC"], "SIM": m["SIM"], "KLD": m["KLD"],
                          "sigma": sigma, "seed": args.seed, "token": args.token,
                          "CC_nosigma": mr["CC"], "SIM_nosigma": mr["SIM"], "KLD_nosigma": mr["KLD"],
                          "CC_native256": mn["CC"], "SIM_native256": mn["SIM"], "KLD_native256": mn["KLD"]})
    fm = {"fold": k, "sigma": sigma, "sigma_table_trainCC": table, "train_seconds": t_train,
          "fold_seconds": time.time() - t_fold, "n_train_weights": len(tr), "n_sigma_select": len(sel),
          "n_trainable_params": n_trainable, "mode": args.mode,
          "test_CC": float(np.mean([r["CC"] for r in per_image if r["fold"] == k])),
          "test_CC_nosigma": float(np.mean([r["CC_nosigma"] for r in per_image if r["fold"] == k]))}
    fold_meta[str(k)] = fm
    log({"event": "fold_done", **fm})
    json.dump(per_image, open(per_image_path, "w"))
    json.dump(fold_meta, open(fold_meta_path, "w"), indent=1)
    del model, opt
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev == "cuda":
        torch.cuda.empty_cache()

if len(fold_meta) == len(folds):
    summ = sc.fold_summary(per_image, folds)
    summ_nos = sc.fold_summary([{**r, "CC": r["CC_nosigma"], "SIM": r["SIM_nosigma"], "KLD": r["KLD_nosigma"]}
                                for r in per_image], folds)
    summ_nat = sc.fold_summary([{**r, "CC": r["CC_native256"], "SIM": r["SIM_native256"],
                                 "KLD": r["KLD_native256"]} for r in per_image], folds)
    out = {"row": tag, **summ, "no_sigma_at384": summ_nos, "native256_no_sigma": summ_nat,
           "fold_meta": fold_meta, "args": vars(args), "device": dev,
           "params_trained": 57501196}
    json.dump(out, open(os.path.join(args.out_dir, f"{tag}_summary_raw.json"), "w"), indent=1)
    print(f"ALL FOLDS DONE {tag}: CC {summ['CC_mean']:.4f} +/- {summ['CC_sd_folds']:.4f} "
          f"SIM {summ['SIM_mean']:.4f} KLD {summ['KLD_mean']:.4f}", flush=True)
