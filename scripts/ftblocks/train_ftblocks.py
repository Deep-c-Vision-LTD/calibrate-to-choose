"""
ftblocks: does fine-tuning the LAST N transformer blocks of DINOv2-base (+ final LayerNorm + our readout)
on each fold's 800 maps close the gap to ~0.92 at unchanged inference cost? Nested selection only:
no choice uses an outer test fold. The protocol (grid, inner split, selection rule, final retrain,
secondary rows, fallback) is pre-registered in scripts/ftblocks/PLAN.json / PLAN.md, amended BEFORE any
grid run by PLAN_AMENDMENT_1.json (pre-run review fixes M1-M4, see PLAN.md / PLAN_DEVIATIONS.md); this
driver reads the grid and rules from PLAN.json and records the sha256 of PLAN.json, of the amendment and
of its own code in every run. The frozen code version is kept in frozen_20261124T083423Z/.

Per outer fold k (seed-0 5-fold split; train = the other 4 folds in run_row.py order):
  inner split  perm = numpy default_rng(100 + k).permutation(800); inner-train = tr[perm[:640]],
               inner-val = tr[perm[640:]]  (ids recorded in scripts/ftblocks/inner_splits.json)
  grid         PLAN.json "active_grid" in simplicity order: N0 (the unchanged base recipe), N2/N4 x lr_bb
  inner run    train on inner-train (60 epochs x 40 steps = 2400 steps, batch 16), sigma from {0..12} on a
               random 200-image subset of inner-train (base rule, base rng stream), score inner-val (160)
  selection    highest mean inner-val CC; every config within 0.001 of the best counts as tied and the
               first of them in simplicity order (smaller N, then lower lr) is selected
  final run    selected config retrained on all 800 training ids (60 x 50 = 3000 steps), sigma on a
               random 200-image subset of them, scored ONCE on the 200 test ids -> ours_ftblocks_nested
Seeds (all runs of fold k, = ours_scratch_s0): torch.manual_seed(k) right before the readout is built,
data-order rng numpy default_rng(k) incl. the base recipe's start-of-training 200-image sigma draw.

Amendment 1 (pre-run review, applied before any grid run):
  M2 order     ALL inner runs of ALL folds first (config-major: N0 x 5 folds, then N2 x 5, then N4 x 5),
               then the 5 selections + the pooled choice + the distill trigger are written, and only then
               the 5 final runs. No outer-test score exists before every selection is fixed.
  M3 diverge   a non-finite training loss (or non-finite metrics) does not kill the driver: the run is
               written with status "diverged" (lock released). A diverged inner run counts as CC = -inf
               (cannot be selected for that fold). A diverged final run falls back to the next config by
               the inner rule (the rule re-applied to the remaining configs); the chain is recorded.
  M1 global    ours_ftblocks_global is NOT nested (the pooled choice uses other folds' inner-val sets, which
               contain ~160 of each fold's 200 test ids); it is labelled so in its summary, never a test result.

Secondary (labelled as such): ours_ftblocks_global = one config chosen by the same rule on the inner-val
CC pooled over the 5 folds, run on every outer fold (final runs shared with the nested row when the
config coincides). Optional ours_ftblocks_distill (only if the nested selection beats the N0 control on
inner validation in >= 4/5 folds): fold-k nested config + the distill_pitt recipe.

Stages
  python -I train_ftblocks.py make-inner-splits         # writes inner_splits.json (once, before the PLAN)
  python -I train_ftblocks.py smoke-n0 [--smoke-dir D]  # N=0 final protocol on fold 0 vs ours_scratch_s0
  python -I train_ftblocks.py smoke-ft --n-ft 4 --steps 20 [--smoke-dir D]
  python -I train_ftblocks.py nested                    # the long run (resumable)
  python -I train_ftblocks.py global                    # secondary row (after nested)
  python -I train_ftblocks.py pitt-depth                # prints the Pitt cache depth the distill row needs, or none
  python -I train_ftblocks.py distill                   # optional row; refuses unless triggered
  python -I train_ftblocks.py assemble                  # (re)write the result files that are complete
Every run writes cache/ftblocks/runs/<run>.json atomically and is skipped when that file exists, so a
crash loses at most the run in progress. The shared MPS lock is taken per run and released after it.
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, argparse, platform, gc
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch
import ft_common as FC  # noqa: E402
import ours_lib as L    # noqa: E402

SEED = 0
CODE_FILES = ["ft_common.py", "train_ftblocks.py", "cache_blocks.py"]
AMEND_JSON = os.path.join(HERE, "PLAN_AMENDMENT_1.json")
NINF = float("-inf")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def code_hashes():
    return {f: FC.sha256_file(os.path.join(HERE, f)) for f in CODE_FILES}


def amend_sha():
    return FC.sha256_file(AMEND_JSON) if os.path.exists(AMEND_JSON) else None


def load_plan(required=True):
    if not os.path.exists(FC.PLAN_JSON):
        if required:
            raise SystemExit("PLAN.json missing: the long run starts only from a frozen PLAN")
        return None, None
    plan = json.load(open(FC.PLAN_JSON))
    assert plan.get("frozen_utc"), "PLAN.json has no frozen_utc timestamp"
    return plan, FC.sha256_file(FC.PLAN_JSON)


def active_grid(plan):
    g = plan["active_grid"]
    return [(c["name"], int(c["N"]), (None if c["lr_bb"] is None else float(c["lr_bb"]))) for c in g]


def cfg_of(grid, name):
    return [c for c in grid if c[0] == name][0]


def cc_value(r):
    """mean CC of a run; a diverged run (or non-finite CC) counts as -inf (amendment M3)."""
    v = r.get("mean_CC")
    if r.get("status", "ok") == "diverged" or v is None or not np.isfinite(v):
        return NINF
    return float(v)


def jnum(v):
    return None if (v is None or not np.isfinite(v)) else float(v)


def select_config(vals, grid, tie):
    """vals: name -> mean inner-val CC (-inf = diverged). Highest wins; all configs >= best - tie are tied;
    the first tied config in simplicity order (the grid order: smaller N, then lower lr) is selected."""
    best = max(vals[n] for n, _, _ in grid)
    if not np.isfinite(best):
        raise RuntimeError("every config of this selection diverged")
    for n, _, _ in grid:
        if vals[n] >= best - tie - 1e-12:
            return n, best
    raise AssertionError


def rule_order(vals, grid, tie):
    """the fallback chain of amendment M3: the selection rule applied repeatedly to the remaining
    (non-diverged) configs: [selected, next by the rule, ...]."""
    rem = [g for g in grid if np.isfinite(vals[g[0]])]
    out = []
    while rem:
        n, _ = select_config(vals, rem, tie)
        out.append(n)
        rem = [g for g in rem if g[0] != n]
    return out


def resolve_chain(prefix, k, order, out_dir=None):
    """walk the chain of run files <prefix>_k<k>_<config>.json in `order`.
    returns (run or None, chain of existing runs, name of the next config to run or None)."""
    out_dir = out_dir or FC.RUNS
    chain = []
    for n in order:
        p = os.path.join(out_dir, f"{prefix}_k{k}_{n}.json")
        if not os.path.exists(p):
            return None, chain, n
        r = json.load(open(p))
        chain.append({"config": n, "run": r["run"], "status": r.get("status", "ok")})
        if r.get("status", "ok") != "diverged":
            return r, chain, None
    return None, chain, None


class Runner:
    def __init__(self, dev, plan=None, plan_sha=None, out_dir=None, yield_s=30):
        out_dir = out_dir or FC.RUNS
        self.dev, self.plan, self.plan_sha, self.out_dir, self.yield_s = dev, plan, plan_sha, out_dir, yield_s
        os.makedirs(out_dir, exist_ok=True); os.makedirs(FC.MODELS, exist_ok=True)
        torch.set_num_threads(6)
        self.data = FC.FTData(dev)
        self.folds = FC.outer_folds(self.data.pos)
        self.bb = FC.load_backbone()
        self.lock = FC.MPSLock(log=log) if dev == "mps" else None
        self.code = code_hashes()
        self.amend = amend_sha()
        if os.path.exists(AMEND_JSON):
            want = json.load(open(AMEND_JSON)).get("code_sha256")
            if want and want != self.code:
                log("WARNING: code differs from the hashes frozen in PLAN_AMENDMENT_1.json (recorded in every run)")
        elif plan is not None and plan.get("code_sha256") and plan["code_sha256"] != self.code:
            log("WARNING: code differs from the hashes frozen in PLAN.json (recorded in every run)")

    def splits(self, k):
        tr, te = FC.outer_split(self.folds, k)
        itr, iva = FC.inner_split(tr, k)
        if os.path.exists(FC.INNER_SPLITS):
            rec = json.load(open(FC.INNER_SPLITS))["folds"][k]
            assert [self.data.ids[i] for i in itr] == rec["inner_train_ids"] and [self.data.ids[i] for i in iva] == rec["inner_val_ids"]
            assert [self.data.ids[i] for i in te] == rec["test_ids"]
        return tr, te, itr, iva

    def _src(self, n_ft, pitt):
        if n_ft == 0:
            return "final16"
        if pitt:            # Pitt may only have a block-8 cache: ADD then uses the same source depth
            return FC.src_depth_for(n_ft, available=[d for d, p in FC.PITT_CACHE.items() if os.path.exists(p)])
        return FC.src_depth_for(n_ft)

    def _common(self, name, stage, k, n_ft, lr_bb, src, train_pos, eval_pos, pitt, max_steps):
        return {"run": name, "stage": stage, "fold": k,
                "config": {"name": FC.cfg_name(n_ft, lr_bb), "N": n_ft, "lr_bb": lr_bb, "source": src,
                           "trained_blocks": [] if n_ft == 0 else list(range(13 - n_ft, 13))},
                "n_train": int(len(train_pos)), "n_eval": int(len(eval_pos)), "device": self.dev,
                "pitt": bool(pitt), "max_steps": max_steps, "trainable_params": FC.n_trainable(n_ft, self.bb),
                "plan_sha256": self.plan_sha, "amendment_sha256": self.amend, "code_sha256": self.code,
                "torch": torch.__version__}

    def run(self, name, stage, k, n_ft, lr_bb, train_pos, eval_pos, save_model=False, pitt=False, max_steps=None):
        path = os.path.join(self.out_dir, f"{name}.json")
        if os.path.exists(path):
            return json.load(open(path))
        if self.lock:
            self.lock.acquire()
        t0 = time.time()
        try:
            try:
                res = self._run(name, stage, k, n_ft, lr_bb, train_pos, eval_pos, save_model, pitt, max_steps)
            except FloatingPointError as e:          # amendment M3: a diverged run is recorded, not fatal
                res = {**self._common(name, stage, k, n_ft, lr_bb, self._src(n_ft, pitt), train_pos, eval_pos, pitt, max_steps),
                       "status": "diverged", "error": str(e), "sigma": None, "rows": [],
                       "mean_CC": None, "mean_SIM": None, "mean_KLD": None, "train": {"steps": None, "s_per_step": None},
                       "eval_s": None, "wall_s": time.time() - t0, "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
        finally:
            gc.collect()
            if self.dev == "mps":
                torch.mps.empty_cache()
            if self.lock:
                self.lock.release()
        FC.jdump(res, path)
        if res["status"] == "diverged":
            log(f"[{name}] DIVERGED: {res['error']} ({res['wall_s']:.0f}s); recorded as status 'diverged' (CC = -inf)")
        else:
            log(f"[{name}] sigma={res['sigma']} CC={res['mean_CC']:.4f} SIM={res['mean_SIM']:.4f} KLD={res['mean_KLD']:.4f} "
                f"({res['wall_s']:.0f}s, {res['train']['s_per_step']:.3f} s/step)")
        if self.lock and self.yield_s:
            time.sleep(self.yield_s)          # courtesy gap so other tracks can take the lock
        return res

    def _run(self, name, stage, k, n_ft, lr_bb, train_pos, eval_pos, save_model, pitt, max_steps):
        t0 = time.time()
        dev = self.dev
        pdata, cyc = None, None
        src = self._src(n_ft, pitt)
        torch.manual_seed(1000 * SEED + k)
        head = L.ReadoutHead("full")
        model = head if n_ft == 0 else FC.FTModel(head, self.bb, n_ft, src)
        model = model.to(dev)
        view = self.data.view(src)
        rng = np.random.default_rng(k + 1000 * SEED)
        if pitt:
            rng_p = np.random.default_rng(100000 + k + 1000 * SEED)
            pdata = FC.PittData(dev, src)
            cyc = FC.Cycler(pdata.P, rng_p)
        train_pos = np.asarray(train_pos)
        if len(train_pos) > L.SIGMA_SELECT_N:
            _ = rng.choice(train_pos, size=L.SIGMA_SELECT_N, replace=False)   # base recipe's start-of-training sigma draw
        if n_ft == 0 and not pitt and max_steps is None:
            tt = time.time()
            L.train(model, view, train_pos, rng, epochs=FC.EPOCHS, lr=FC.READOUT_LR)      # the unchanged base recipe
            spe = int(np.ceil(len(train_pos) / FC.BS))
            tinfo = {"steps": FC.EPOCHS * spe, "planned_steps": FC.EPOCHS * spe, "warmup_steps": 0, "epoch_loss": None,
                     "train_s": time.time() - tt, "s_per_step": (time.time() - tt) / (FC.EPOCHS * spe),
                     "trainer": "ours_lib.train (unchanged base recipe)"}
            # ours_lib.train has no loss check: a non-finite model shows up as non-finite metrics below (M3)
        else:
            tinfo = FC.train_loop(model, view, train_pos, rng, n_ft, lr_bb, pitt=pdata, k=k, cyc=cyc,
                                  max_steps=max_steps, log=log, tag=name)
            tinfo["trainer"] = "ft_common.train_loop" + (" + Pitt stream (distill_pitt recipe)" if pitt else "")
        te0 = time.time()
        sig = L.select_sigma_subset(model, view, train_pos, rng)
        rows = L.score(model, view, eval_pos, sig)
        t_eval = time.time() - te0
        if not all(np.isfinite([x[m] for x in rows for m in ("CC", "SIM", "KLD")])):
            raise FloatingPointError(f"{name}: non-finite metrics after training (sigma {sig})")
        for x in rows:
            x.update({"fold": k, "seed": SEED})
        res = {**self._common(name, stage, k, n_ft, lr_bb, src, train_pos, eval_pos, pitt, max_steps),
               "status": "ok", "sigma": int(sig), "rows": rows,
               **{f"mean_{m}": float(np.mean([x[m] for x in rows])) for m in ("CC", "SIM", "KLD")},
               "train": tinfo, "eval_s": t_eval, "wall_s": time.time() - t0,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
        if save_model:
            sd = {kk: v.detach().cpu() for kk, v in model.state_dict().items()}
            torch.save({"config": res["config"], "state_dict": sd, "sigma": int(sig)}, os.path.join(FC.MODELS, f"{name}.pt"))
            res["model_file"] = os.path.join(FC.MODELS, f"{name}.pt")
        del model
        return res


# ----------------------------------------------------------------------------------------- selection
def selection(k, grid, tie, out_dir=None):
    out_dir = out_dir or FC.RUNS
    p = os.path.join(out_dir, f"selection_k{k}.json")
    runs = {n: os.path.join(out_dir, f"inner_k{k}_{n}.json") for n, _, _ in grid}
    if not all(os.path.exists(r) for r in runs.values()):
        return None
    R = {n: json.load(open(r)) for n, r in runs.items()}
    vals = {n: cc_value(R[n]) for n in R}
    sel, best = select_config(vals, grid, tie)
    c0 = grid[0][0]
    out = {"fold": k, "rule": f"highest mean inner-val CC; configs within {tie} of the best are tied; first tied config in simplicity order; "
                              "a diverged inner run counts as CC = -inf (amendment M3)",
           "simplicity_order": [n for n, _, _ in grid], "inner_val_CC": {n: jnum(v) for n, v in vals.items()},
           "inner_val_SIM": {n: R[n]["mean_SIM"] for n in R}, "inner_val_KLD": {n: R[n]["mean_KLD"] for n in R},
           "inner_sigma": {n: R[n]["sigma"] for n in R}, "inner_status": {n: R[n].get("status", "ok") for n in R},
           "inner_wall_s": {n: R[n]["wall_s"] for n in R}, "best_CC": best, "selected": sel,
           "fallback_order": rule_order(vals, grid, tie),
           "selected_beats_control": bool(sel != c0 and vals[sel] > vals[c0]),
           "selected_minus_control_CC": jnum(vals[sel] - vals[c0]) if np.isfinite(vals[c0]) else None}
    if os.path.exists(p):
        old = json.load(open(p))
        assert old["selected"] == out["selected"], "selection changed: inner run files were modified"
        return old
    FC.jdump(out, p)
    return out


def sel_vals(s, grid):
    return {n: (NINF if s["inner_val_CC"][n] is None else float(s["inner_val_CC"][n])) for n, _, _ in grid}


def pooled_selection(grid, tie, out_dir=None):
    S = [selection(k, grid, tie, out_dir) for k in range(5)]
    if any(s is None for s in S):
        return None
    pooled = {n: float(np.mean([sel_vals(s, grid)[n] for s in S])) for n, _, _ in grid}
    g, best = select_config(pooled, grid, tie)
    return {"pooled_inner_val_CC": {n: jnum(v) for n, v in pooled.items()}, "best_CC": best, "selected": g,
            "fallback_order": rule_order(pooled, grid, tie),
            "rule": "same rule on the inner-val CC averaged over the 5 folds (equal 160-image inner-val sets = pooled image mean); "
                    "a config that diverged in any fold's inner run has pooled CC = -inf",
            "nesting": "NOT nested (amendment M1): the pooled choice uses every fold's inner-val set, and those lie in the "
                       "other folds' test sets"}


def distill_trigger(S, plan):
    n_beat = sum(s["selected_beats_control"] for s in S)
    need = int(plan["optional_distill"]["trigger_min_folds"])
    return {"n_folds_selected_beats_control_on_inner_val": int(n_beat), "required": need, "triggered": bool(n_beat >= need),
            "per_fold": [{"fold": s["fold"], "selected": s["selected"], "selected_minus_control_CC": s["selected_minus_control_CC"]}
                         for s in S]}


def write_once(obj, path, key):
    """write a decision file once; on a restart assert that the decision is unchanged."""
    if os.path.exists(path):
        old = json.load(open(path))
        assert old[key] == obj[key], f"{path}: decision changed"
        return old
    obj = {**obj, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    FC.jdump(obj, path)
    return obj


def test_overlap_with_other_inner_val():
    """M1: per outer fold k, how many of its 200 test ids lie in the inner-val sets of the other folds
    (distinct images, and counted with multiplicity over the 4 other folds)."""
    rec = json.load(open(FC.INNER_SPLITS))["folds"]
    distinct, mult = [], []
    for k in range(5):
        te = set(rec[k]["test_ids"])
        others = [set(rec[j]["inner_val_ids"]) for j in range(5) if j != k]
        assert not (te & set(rec[k]["inner_val_ids"])) and not (te & set(rec[k]["inner_train_ids"]))
        distinct.append(len(te & set().union(*others)))
        mult.append(sum(len(te & o) for o in others))
    return distinct, mult


def final_order_nested(k, grid, tie):
    s = selection(k, grid, tie)
    return None if s is None else s["fallback_order"]


def distill_order(k, grid, tie):
    """the config used in fold k's nested row first (after any M3 fallback of its final run), then the inner rule."""
    order = final_order_nested(k, grid, tie)
    r, _, _ = resolve_chain("final", k, order)
    used = r["config"]["name"] if r is not None else order[0]
    return [used] + [n for n in order if n != used]


# ----------------------------------------------------------------------------------------- assembly
def write_row(row, fold_runs, extra, folds_n=5, results_dir=None, overwrite=False):
    results_dir = results_dir or FC.RESULTS
    out_pi = os.path.join(results_dir, f"{row}_per_image.json")
    out_sum = os.path.join(results_dir, f"{row}_summary.json")
    if (os.path.exists(out_pi) or os.path.exists(out_sum)) and not overwrite:
        log(f"{row}: result files exist, not overwriting"); return False
    per_image, det = [], []
    for k, r in enumerate(fold_runs):
        assert r["fold"] == k and r.get("status", "ok") == "ok"
        per_image += [{"id": x["id"], "CC": x["CC"], "SIM": x["SIM"], "KLD": x["KLD"], "fold": k, "seed": SEED} for x in r["rows"]]
        det.append({"fold": k, "config": r["config"], "run": r["run"], "n_train": r["n_train"], "n_test": r["n_eval"],
                    "sigma": r["sigma"], "CC": r["mean_CC"], "SIM": r["mean_SIM"], "KLD": r["mean_KLD"],
                    "steps": r["train"]["steps"], "s_per_step": r["train"]["s_per_step"], "wall_s": r["wall_s"],
                    "trainable_params": r["trainable_params"], "device": r["device"], "model_file": r.get("model_file")})
    if folds_n == 5:
        assert len(per_image) == 1000 and len({x["id"] for x in per_image}) == 1000
        rec = json.load(open(FC.INNER_SPLITS))["folds"]
        assert all(x["id"] in set(rec[x["fold"]]["test_ids"]) for x in per_image)
    st = L.fold_stats(per_image, folds_n)
    summ = {"row": row, "kind": "ftblocks", "init": "scratch readout + pretrained DINOv2-base blocks", "seed": SEED, "mode": "full",
            "folds_file": os.path.join(FC.KB, "splits", "kfold_seed0.json"), **st, "folds_detail": det, **extra,
            "protocol": {"backbone": "facebook/dinov2-base, 448x448; blocks 1..12-N frozen (outputs cached, cache/ftblocks), "
                                     "blocks 12-N+1..12 + final LayerNorm fine-tuned with the readout; N = 0 is the base recipe "
                                     "(frozen backbone, fp16 feature cache)",
                         "readout": "ours_lib.ReadoutHead('full'), 3,721,089 params, scratch init torch.manual_seed(k)",
                         "epochs": 60, "batch": 16, "steps_outer": 3000, "steps_inner": 2400,
                         "optimizer": "AdamW; readout lr 3e-4 wd 1e-4 constant; backbone blocks + final norm lr_bb wd 0.05, "
                                      "linear warmup over the first 5% of steps then constant (N = 0: the base single-group AdamW)",
                         "precision": "fp32", "loss": "KL + (1 - CC) at 64x64",
                         "finalizer": "Gaussian sigma from {0,2,4,6,8,10,12} by mean CC on a random 200-image subset of the training ids",
                         "scoring": "unified: 384x384, GT bilinear to 384, clip at 0 and divide by sum; CC/SIM/KLD (eps 1e-8)",
                         "data_order_rng": "default_rng(k), incl. the base start-of-training 200-image draw",
                         "inference_cost": "identical to ours: the same DINOv2-base 448 + readout graph (221 GFLOPs); only weights of "
                                           "the last N blocks, the final norm and the readout differ"},
            "torch": torch.__version__, "machine": platform.platform(),
            "command": " ".join(["python", "-I"] + sys.argv), "assembled": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    json.dump(per_image, open(out_pi, "w"))
    FC.jdump(summ, out_sum)
    log(f"[{row}] mean CC={st['mean_CC']:.4f} sd1={st['sd_CC_ddof1']:.4f} SIM={st['mean_SIM']:.4f} KLD={st['mean_KLD']:.4f}")
    return True


def assemble(plan, plan_sha):
    grid, tie = active_grid(plan), float(plan["selection"]["tie_cc"])
    S = [selection(k, grid, tie) for k in range(5)]
    if not all(s is not None for s in S):
        log("assemble: inner selections incomplete"); return
    common = {"plan_file": FC.PLAN_JSON, "plan_sha256": plan_sha, "plan_mode": plan["mode"],
              "amendment_file": AMEND_JSON, "amendment_sha256": amend_sha()}
    fin = [resolve_chain("final", k, S[k]["fallback_order"]) for k in range(5)]
    if all(r is not None for r, _, _ in fin):
        n_beat = sum(s["selected_beats_control"] for s in S)
        write_row("ours_ftblocks_nested", [r for r, _, _ in fin],
                  {"status": "PRIMARY row of the ftblocks experiment (nested selection, outer test folds used once)",
                   "selection_per_fold": S, "final_chain_per_fold": [c for _, c, _ in fin],
                   "config_used_per_fold": [r["config"]["name"] for r, _, _ in fin],
                   "n_folds_selected_beats_control_on_inner_val": int(n_beat), **common})
    pooled = pooled_selection(grid, tie)
    gfin = [resolve_chain("final", k, pooled["fallback_order"]) for k in range(5)]
    if all(r is not None for r, _, _ in gfin):
        ov, ovm = test_overlap_with_other_inner_val()
        write_row("ours_ftblocks_global", [r for r, _, _ in gfin],
                  {"status": "SECONDARY, NOT NESTED (transparency only): one config for all folds, chosen by the nested rule on "
                             "the inner-val CC pooled over the 5 folds. The pooled choice uses each fold's own test images through "
                             f"the other folds' inner-val sets: {'/'.join(map(str, ov))} distinct test images of 200 for folds 0-4 "
                             f"({'/'.join(map(str, ovm))} counted with multiplicity). This is NOT a held-out result and must never "
                             "be cited as a test result.",
                   "nested": False, "test_ids_in_other_folds_inner_val_distinct": ov,
                   "test_ids_in_other_folds_inner_val_with_multiplicity": ovm, "pooled_selection": pooled,
                   "final_chain_per_fold": [c for _, c, _ in gfin], **common})
    dfin = [resolve_chain("distill", k, distill_order(k, grid, tie)) for k in range(5)]
    if all(r is not None for r, _, _ in dfin):
        write_row("ours_ftblocks_distill", [r for r, _, _ in dfin],
                  {"status": "OPTIONAL SECONDARY: fold-k nested config + the distill_pitt recipe (16 GT + 16 Pitt teacher-k "
                             "targets per step, w = 1); run only because the pre-registered trigger held", "selection_per_fold": S,
                   "distill_chain_per_fold": [c for _, c, _ in dfin],
                   "config_used_per_fold": [r["config"]["name"] for r, _, _ in dfin], **common})


# ----------------------------------------------------------------------------------------- stages
def stage_make_inner_splits():
    if os.path.exists(FC.INNER_SPLITS):
        raise SystemExit(f"{FC.INNER_SPLITS} exists")
    ids = L.load_ids(); pos = {i: n for n, i in enumerate(ids)}
    folds = FC.outer_folds(pos)
    out = {"rule": "outer train ids of fold k in run_row.py order (concatenation of the other folds of splits/kfold_seed0.json); "
                   "perm = numpy.random.default_rng(100 + k).permutation(800); inner_train = tr[perm[:640]], inner_val = tr[perm[640:]]",
           "numpy": np.__version__, "folds": []}
    for k in range(5):
        tr, te = FC.outer_split(folds, k)
        itr, iva = FC.inner_split(tr, k)
        assert len(set(itr) | set(iva)) == 800 and not (set(itr) | set(iva)) & set(te)
        out["folds"].append({"fold": k, "inner_train_ids": [ids[i] for i in itr], "inner_val_ids": [ids[i] for i in iva],
                             "test_ids": [ids[i] for i in te]})
    FC.jdump(out, FC.INNER_SPLITS)
    print("written", FC.INNER_SPLITS)


def stage_nested(a):
    plan, sha = load_plan()
    grid, tie = active_grid(plan), float(plan["selection"]["tie_cc"])
    R = Runner(a.device, plan, sha, yield_s=a.yield_s)
    sp = {k: R.splits(k) for k in range(5)}
    # phase 1 (M2): every inner run of every fold, config-major, before any final run
    for n, N, lr in grid:
        for k in range(5):
            _, _, itr, iva = sp[k]
            R.run(f"inner_k{k}_{n}", "inner", k, N, lr, itr, iva)
    # phase 2 (M2): the 5 selections, the pooled choice and the distill trigger, fixed before any final run
    S = [selection(k, grid, tie) for k in range(5)]
    for s in S:
        log(f"fold {s['fold']}: inner-val CC " + ", ".join(f"{n} {v if v is None else round(v, 4)}" for n, v in s["inner_val_CC"].items())
            + f" -> selected {s['selected']} (fallback order {s['fallback_order']})")
    ps = write_once(pooled_selection(grid, tie), os.path.join(FC.RUNS, "pooled_selection.json"), "selected")
    log(f"pooled inner-val CC {ps['pooled_inner_val_CC']} -> global config {ps['selected']} (secondary, NOT nested)")
    trig = write_once(distill_trigger(S, plan), os.path.join(FC.RUNS, "distill_trigger.json"), "triggered")
    log(f"optional distill trigger: {trig['n_folds_selected_beats_control_on_inner_val']}/5 folds (need {trig['required']}) "
        f"-> triggered={trig['triggered']}")
    # phase 3: final runs, one outer-test scoring per fold (M3 fallback chain only if a final run diverges)
    for k in range(5):
        tr, te, _, _ = sp[k]
        while True:
            r, chain, nxt = resolve_chain("final", k, S[k]["fallback_order"])
            if r is not None or nxt is None:
                break
            c = cfg_of(grid, nxt)
            R.run(f"final_k{k}_{nxt}", "final", k, c[1], c[2], tr, te, save_model=True)
        if r is None:
            log(f"fold {k}: every config diverged in the final run (chain {chain})")
    assemble(plan, sha)


def stage_global(a):
    plan, sha = load_plan()
    grid, tie = active_grid(plan), float(plan["selection"]["tie_cc"])
    pp = os.path.join(FC.RUNS, "pooled_selection.json")
    if not os.path.exists(pp):
        raise SystemExit("global row needs the pooled choice written by the nested stage (phase 2)")
    ps = write_once(pooled_selection(grid, tie), pp, "selected")
    log(f"pooled inner-val CC {ps['pooled_inner_val_CC']} -> global config {ps['selected']} (secondary, NOT nested)")
    R = Runner(a.device, plan, sha, yield_s=a.yield_s)
    for k in range(5):
        tr, te, _, _ = R.splits(k)
        while True:
            r, chain, nxt = resolve_chain("final", k, ps["fallback_order"])
            if r is not None or nxt is None:
                break
            c = cfg_of(grid, nxt)
            R.run(f"final_k{k}_{nxt}", "final", k, c[1], c[2], tr, te, save_model=False)
    assemble(plan, sha)


def distill_state(plan):
    grid, tie = active_grid(plan), float(plan["selection"]["tie_cc"])
    S = [selection(k, grid, tie) for k in range(5)]
    if any(s is None for s in S):
        raise SystemExit("distill row needs all 5 inner selections")
    tp = os.path.join(FC.RUNS, "distill_trigger.json")
    if not os.path.exists(tp):
        raise SystemExit("distill trigger not written yet (nested stage phase 2)")
    trig = write_once(distill_trigger(S, plan), tp, "triggered")
    return grid, tie, S, trig


def stage_pitt_depth(a):
    """print the Pitt block-cache depth the optional row needs ('none' if not triggered or only N0 folds)."""
    plan, _ = load_plan()
    grid, tie, S, trig = distill_state(plan)
    if not trig["triggered"]:
        print("none"); return
    ns = sorted({cfg_of(grid, distill_order(k, grid, tie)[0])[1] for k in range(5)} - {0})
    print("none" if not ns else str(12 - max(ns)))       # both N=2 and N=4 -> block 8 only (PLAN)


def stage_distill(a):
    plan, sha = load_plan()
    grid, tie, S, trig = distill_state(plan)
    if not trig["triggered"]:
        log(f"optional distill row NOT triggered ({trig['n_folds_selected_beats_control_on_inner_val']}/5 < {trig['required']}); nothing run")
        return
    orders = [distill_order(k, grid, tie) for k in range(5)]
    for N in sorted({cfg_of(grid, o[0])[1] for o in orders} - {0}):
        d = FC.src_depth_for(N, available=[dd for dd, p in FC.PITT_CACHE.items() if os.path.exists(p)] or [0])
        if d == 0:
            raise SystemExit("Pitt block cache missing: run cache_blocks.py --set pitt (see PLAN) first")
    R = Runner(a.device, plan, sha, yield_s=a.yield_s)
    for k in range(5):
        tr, te, _, _ = R.splits(k)
        while True:
            r, chain, nxt = resolve_chain("distill", k, orders[k])
            if r is not None or nxt is None:
                break
            c = cfg_of(grid, nxt)
            R.run(f"distill_k{k}_{nxt}", "distill", k, c[1], c[2], tr, te, pitt=True)
    assemble(plan, sha)


def stage_smoke_n0(a):
    sd = a.smoke_dir
    R = Runner(a.device, out_dir=sd, yield_s=0)
    tr, te, _, _ = R.splits(0)
    r = R.run("smoke_n0_final_k0", "smoke", 0, 0, None, tr, te)
    ref = {x["id"]: x for x in json.load(open(os.path.join(FC.RESULTS, "ours_scratch_s0_per_image.json"))) if x["fold"] == 0}
    refsum = json.load(open(os.path.join(FC.RESULTS, "ours_scratch_s0_summary.json")))["folds_detail"][0]
    d = {m: np.array([x[m] - ref[x["id"]][m] for x in r["rows"]]) for m in ("CC", "SIM", "KLD")}
    rep = {"ours_scratch_s0_fold0": {"sigma": refsum["sigma"], "CC": refsum["CC"], "SIM": refsum["SIM"], "KLD": refsum["KLD"]},
           "smoke_fold0": {"sigma": r["sigma"], "CC": r["mean_CC"], "SIM": r["mean_SIM"], "KLD": r["mean_KLD"]},
           **{f"max_abs_per_image_diff_{m}": float(np.abs(v).max()) for m, v in d.items()},
           **{f"mean_diff_{m}": float(v.mean()) for m, v in d.items()},
           "n_images_identical_CC": int((d["CC"] == 0).sum()), "wall_s": r["wall_s"], "code_sha256": r["code_sha256"]}
    FC.jdump(rep, os.path.join(sd, "smoke_n0_vs_ours_scratch_s0.json"))
    log(f"smoke N=0 fold 0 vs ours_scratch_s0: {rep}")


def stage_smoke_ft(a):
    sd = a.smoke_dir
    R = Runner(a.device, out_dir=sd, yield_s=0)
    tr, te, itr, iva = R.splits(0)
    name = f"smoke_ft_N{a.n_ft}_steps{a.steps}_k0"
    r = R.run(name, "smoke", 0, a.n_ft, a.lr_bb, itr, iva, max_steps=a.steps, save_model=False)
    if r.get("status") == "diverged":
        log(f"smoke ft N={a.n_ft}: DIVERGED ({r['error']})"); return
    ok = all(np.isfinite([x["CC"] for x in r["rows"]])) and all(np.isfinite(r["train"]["epoch_loss"]))
    # exercise the standard writer on this single fold (smoke results dir, never results/)
    rd = os.path.join(sd, "results"); os.makedirs(rd, exist_ok=True)
    write_row(f"ours_ftblocks_smoke_N{a.n_ft}", [r], {"status": "SMOKE ONLY (20-step run, fold 0 inner split)"}, folds_n=1,
              results_dir=rd, overwrite=True)
    log(f"smoke ft N={a.n_ft}: steps {r['train']['steps']}, finite={ok}, loss {r['train']['epoch_loss']}, "
        f"{r['train']['s_per_step']:.3f} s/step, inner-val CC {r['mean_CC']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["make-inner-splits", "smoke-n0", "smoke-ft", "nested", "global", "pitt-depth", "distill", "assemble"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--yield-s", type=float, default=30)
    ap.add_argument("--n-ft", type=int, default=4)
    ap.add_argument("--lr-bb", type=float, default=3e-5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--smoke-dir", default=FC.SMOKE)
    a = ap.parse_args()
    if a.stage == "make-inner-splits":
        return stage_make_inner_splits()
    if a.stage == "assemble":
        plan, sha = load_plan()
        return assemble(plan, sha)
    if a.stage == "pitt-depth":
        return stage_pitt_depth(a)
    if a.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS not available")
    _dl = FC.DriverLock()          # noqa: F841  one driver at a time
    {"smoke-n0": stage_smoke_n0, "smoke-ft": stage_smoke_ft, "nested": stage_nested, "global": stage_global,
     "distill": stage_distill}[a.stage](a)


if __name__ == "__main__":
    main()
