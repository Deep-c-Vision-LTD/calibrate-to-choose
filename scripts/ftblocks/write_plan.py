"""
Write the pre-registration PLAN.json + PLAN.md of the ftblocks track (once; refuses to overwrite).
Reads the measured timing (cache/ftblocks/bench.json), the smoke results and the block-cache
verification, computes the time estimate for the full nested plan, and freezes the grid that will run:
the full grid if the estimate is <= BUDGET_H, otherwise the pre-registered fallback grid.

  python -I write_plan.py
"""
import sys
sys.dont_write_bytecode = True
import os, json, time, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ft_common as FC  # noqa: E402

BUDGET_H = 10.0
YIELD_S = 30
FULL_GRID = [{"name": "N0", "N": 0, "lr_bb": None},
             {"name": "N2_lr1e-05", "N": 2, "lr_bb": 1e-5}, {"name": "N2_lr3e-05", "N": 2, "lr_bb": 3e-5},
             {"name": "N4_lr1e-05", "N": 4, "lr_bb": 1e-5}, {"name": "N4_lr3e-05", "N": 4, "lr_bb": 3e-5}]
FALLBACK_GRID = [FULL_GRID[0], FULL_GRID[2], FULL_GRID[4]]          # drop lr_bb = 1e-5


def est_hours(grid, sps, eval_run_s, n0_inner_s, n0_final_s, overhead_s=20):
    """inner: 5 folds x configs on 640 (2400 steps); final: 5 runs on 800 (3000 steps) of the selected config.
    eval_run_s[N] = measured sigma selection (200 training images, CPU filtering) + scoring of 160 images in the
    20-step smoke run (finals score 200: x 400/360). Returns (inner_h, finals_worst_h, finals_best_h): worst = every
    fold selects the most expensive config, best = every fold selects the cheapest fine-tuning config."""
    inner = 0.0
    for c in grid:
        if c["N"] == 0:
            inner += n0_inner_s + YIELD_S
        else:
            inner += 2400 * sps[c["N"]] + eval_run_s[c["N"]] + overhead_s + YIELD_S
    inner *= 5
    ns = [c["N"] for c in grid if c["N"] > 0]
    fin = lambda n: 5 * (3000 * sps[n] + eval_run_s[n] * 400 / 360 + overhead_s + YIELD_S)
    return inner / 3600, fin(max(ns)) / 3600, fin(min(ns)) / 3600


def main():
    if os.path.exists(FC.PLAN_JSON):
        raise SystemExit("PLAN.json exists (frozen); not overwriting")
    bench = json.load(open(os.path.join(FC.CFT, "bench.json")))
    b = {r["N"]: r for r in bench["results"] if r["amp"] == "fp32"}
    bf = {r["N"]: r for r in bench["results"] if r["amp"] != "fp32"}
    sps = {n: b[n]["s_per_step_mean"] for n in b}
    ev = {n: b[n]["eval_s_per_image"] for n in b}
    smoke0 = json.load(open(os.path.join(FC.SMOKE, "smoke_n0_vs_ours_scratch_s0.json")))
    s0run = json.load(open(os.path.join(FC.SMOKE, "smoke_n0_final_k0.json")))
    n0_final_s = s0run["wall_s"]
    n0_inner_s = n0_final_s * 0.8
    smk = {n: json.load(open(os.path.join(FC.SMOKE, f"smoke_ft_N{n}_steps20_k0.json"))) for n in (4, 2)}
    evr = {n: smk[n]["eval_s"] for n in smk}
    cmeta = json.load(open(os.path.join(FC.CFT, "add_blocks_meta.json")))
    i_full, f_full, fb_full = est_hours(FULL_GRID, sps, evr, n0_inner_s, n0_final_s)
    i_fb, f_fb, fb_fb = est_hours(FALLBACK_GRID, sps, evr, n0_inner_s, n0_final_s)
    full_total = i_full + f_full
    mode = "full" if full_total <= BUDGET_H else "fallback_drop_lr1e-5"
    grid = FULL_GRID if mode == "full" else FALLBACK_GRID
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    PY = "python"
    S = os.path.join(FC.KB, "scripts", "ftblocks")
    plan = {
        "title": "ftblocks: fine-tune the last N DINOv2-base blocks + readout per fold (nested selection)",
        "frozen_utc": now, "frozen_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "status": "frozen BEFORE any grid run; only smoke tests (fold-0 N=0 reproduction, 20-step N=4/N=2 runs scored on "
                  "the fold-0 INNER-VAL set, never on a test fold) and the timing bench have been run",
        "question": "Does fine-tuning the last N transformer blocks of frozen-DINOv2-base (+ final LayerNorm + our ASPP readout) "
                    "on each fold's 800 maps close the gap from ours_scratch_s0 (CC 0.8975) towards msdb_adapt / sum_ft "
                    "(~0.92) at unchanged inference cost (same graph, 221 GFLOPs)?",
        "base_readout_inputs": {
            "checkpoint": "facebook/dinov2-base (HF transformers AutoModel, sdpa attention, no register tokens)",
            "input": "PIL RGB, bilinear resize 448x448, /255, ImageNet mean (0.485,0.456,0.406) / std (0.229,0.224,0.225), "
                     "interpolate_pos_encoding=True",
            "tokens": "last_hidden_state = final LayerNorm(output of block 12); CLS dropped; 1024 patch tokens -> 768x32x32",
            "stored": "cache/ours/feats_dinov2b_448_fp16.npy (fp16, ids_sorted order)",
            "readout": "ours_lib.ReadoutHead('full'): ASPP 768->128 (dilations 1,3,6,12 + GAP) + 1x1 head + learned 16x16 centre bias; 3,721,089 params"},
        "block_cache": {
            "files": {"8": FC.ADD_CACHE[8], "10": FC.ADD_CACHE[10]}, "dtype": "float32", "shape": [1000, 1025, 768],
            "why_two_depths": "N=4 starts from block 8, N=2 from block 10; caching both avoids a frozen forward of blocks 9-10 in "
                              "every N=2 step (6.3 GB total)",
            "verification": {"all_1000_images_cpu": cmeta["verification_all_images"], "reread_check": cmeta.get("reread_check"),
                             "summary": "CPU replay of the remaining frozen blocks + final norm on the cached block outputs reproduces "
                                        "last_hidden_state bit-for-bit (max abs diff 0.0) and the fp16 cast equals the existing "
                                        "readout cache in every element for all 1000 images at both depths; on MPS the replay "
                                        "deviates from the CPU by <= 2.1e-4 absolute (features up to |19.6|, i.e. ~1e-5 relative)"}},
        "protocol": {
            "outer_split": "splits/kfold_seed0.json, 5 folds x 200 test ids; train = the other 4 folds in run_row.py order",
            "seeds": "fold k: torch.manual_seed(k) immediately before the readout is built (all configs: same readout init); "
                     "data-order rng numpy default_rng(k), including the base recipe's start-of-training 200-image draw",
            "N0": "the unchanged base recipe: ours_lib.train on the fp16 feature cache (AdamW lr 3e-4 wd 1e-4 constant, 60 epochs, "
                  "batch 16, KL + (1 - CC) at 64x64)",
            "N_gt_0": "cached frozen block-(12-N) output -> trainable blocks 12-N+1..12 -> trainable final LayerNorm -> readout. "
                      "One AdamW: readout lr 3e-4 wd 1e-4 constant; backbone blocks + final norm lr_bb, wd 0.05 (all their "
                      "parameters), linear warmup lr_bb*(s+1)/W for s < W = round(0.05 * total steps), then constant (the readout's "
                      "schedule). Same loss, batch 16, data order, 60 epochs (inner 2400 steps, outer 3000 steps)",
            "precision": "fp32 everywhere (no autocast; bf16 was timed for information only and is NOT used)",
            "augmentation": "none (horizontal flip is NOT in the pre-registered grid and is therefore not used)",
            "finalizer": "Gaussian sigma from {0,2,4,6,8,10,12} by mean CC on a random 200-image subset of the run's training ids "
                         "(inner: of inner-train; final: of the 800 outer training ids), drawn from the data-order rng after training",
            "scoring": "ours_lib.score (unified 384 protocol: CC / SIM / KLD)",
            "per_run_checkpoint": "cache/ftblocks/runs/<run>.json written atomically after each run; existing runs are skipped "
                                  "(a crash loses at most the run in progress); nested final models saved to cache/ftblocks/models/"},
        "inner_split": {"rule": "perm = numpy default_rng(100 + k).permutation(800) over the outer training ids in run_row order; "
                                "inner_train = tr[perm[:640]], inner_val = tr[perm[640:]]",
                        "file": FC.INNER_SPLITS, "sha256": FC.sha256_file(FC.INNER_SPLITS)},
        "full_grid": FULL_GRID, "fallback_grid": FALLBACK_GRID, "mode": mode, "active_grid": grid,
        "selection": {"metric": "mean inner-val CC (160 images) after the inner-train sigma", "tie_cc": 0.001,
                      "rule": "best = max mean inner-val CC; every config with CC >= best - 0.001 is tied; select the first tied config "
                              "in simplicity order = grid order (smaller N first, then lower lr_bb)",
                      "final": "retrain the selected config on all 800 outer training ids (3000 steps), sigma on 200 of them, score "
                               "the 200 outer test ids ONCE"},
        "rows": {
            "ours_ftblocks_nested": "PRIMARY. Per-fold selected config (recorded per fold).",
            "ours_ftblocks_global": "SECONDARY, transparency only: one config chosen by the same rule on the inner-val CC averaged over the "
                                    "5 folds; run on all 5 outer folds; final runs are shared with the nested row where the config "
                                    "coincides (identical code, seeds and data).",
            "ours_ftblocks_distill": "OPTIONAL SECONDARY, see optional_distill."},
        "optional_distill": {
            "trigger_min_folds": 4,
            "trigger": "run ONLY if in >= 4 of 5 folds the nested selection beats the N0 control on inner validation, i.e. the "
                       "selected config is not N0 (by the tie rule this implies its inner-val CC exceeds N0's by more than 0.001)",
            "recipe": "fold-k nested config + the distill_pitt recipe of scripts/distill/train_students.py: each step the base 16 "
                      "ADD1000 training ids with GT + 16 Pitt pool images (2,499) with teacher-k targets (cache/distill/targets/"
                      "pitt_t64.npy, MSDB head adapted on fold k's 800 training maps), one forward of 32, loss = L(GT) + 1.0 L(Pitt); "
                      "Pitt stream Cycler(default_rng(100000 + k)); 3000 steps; sigma by the base rule on ADD1000 training images",
            "pitt_cache": "frozen block outputs of the 2,499 Pitt pool images computed exactly like the ADD1000 cache "
                          "(cache_blocks.py --set pitt) in fp16 (3.9 GB per depth) at depth 12-N of the selected N; if folds select "
                          "both N=2 and N=4 only depth 8 is cached and N=2 folds apply frozen blocks 9-10 on the fly for BOTH "
                          "streams (ADD1000 then also starts from block 8; mathematically identical model)",
            "no_selection": "no further tuning; the row is scored once per test fold"},
        "statistics": {
            "code": "scripts/ftblocks/stats_ftblocks.py; holm, nb_test, rank_biserial, compare, vec, fold_means, describe, seed_avg "
                    "executed verbatim from scripts/compile/compile_results.py (ast extraction)",
            "tests": "paired bootstrap 95% CI (5000) of the mean per-image difference, per-image Wilcoxon signed-rank, fold-level "
                     "Nadeau-Bengio corrected t (df 4), fold SD ddof 1; differences = ftblocks row minus reference",
            "primary_family": ["ours_ftblocks_nested - " + r for r in ("ours_scratch_s0", "ours_scratch_seedavg", "distill_pitt",
                                                                       "ours_L448_s0", "ours_ms_B_s0", "msdb_adapt_s0", "sum_ft")],
            "secondary_family": "each existing secondary row (global, distill) vs the same 7 references; Holm within the family",
            "holm": "within each family, separately per metric (CC, SIM, KLD) and test (Wilcoxon, NB)",
            "headline": "ours_ftblocks_nested - ours_scratch_s0 on CC (primary comparison)"},
        "timing": {
            "bench_file": os.path.join(FC.CFT, "bench.json"), "s_per_step_fp32_batch16_mps": sps,
            "eval_s_per_image": ev, "bf16_s_per_step_info_only": {n: r.get("s_per_step_mean", r.get("error")) for n, r in bf.items()},
            "n0_run_wall_s_fold0_final_smoke": n0_final_s,
            "eval_s_per_run_measured_smoke": evr,
            "estimate_full_grid_h": {"inner": round(i_full, 2), "finals_worst_case": round(f_full, 2), "finals_best_case": round(fb_full, 2),
                                     "total": round(full_total, 2), "total_best_case": round(i_full + fb_full, 2)},
            "estimate_fallback_grid_h": {"inner": round(i_fb, 2), "finals_worst_case": round(f_fb, 2), "finals_best_case": round(fb_fb, 2),
                                         "total": round(i_fb + f_fb, 2), "total_best_case": round(i_fb + fb_fb, 2)},
            "budget_h": BUDGET_H, "courtesy_gap_s_per_run": YIELD_S,
            "global_row_extra_h_worst": round(5 * (3000 * max(sps.values()) + 400 * max(ev.values())) / 3600, 2),
            "optional_distill_h_approx": {"all_folds_N2": round(5 * 3000 * 2 * sps[2] / 3600, 1), "all_folds_N4": round(5 * 3000 * 2 * sps[4] / 3600, 1),
                                          "note": "not measured: assumes a 32-image step costs 2x the measured 16-image step"},
            "mps_memory_gib_driver_alloc": {n: round(b[n]["driver_alloc_gib"], 1) for n in b},
            "note": "estimates exclude time spent waiting for the shared MPS lock"},
        "fallback": {
            "rule": f"if the full-grid estimate exceeds {BUDGET_H} h, run the fallback grid (N0, N2_lr3e-05, N4_lr3e-05): lr_bb = 1e-5 is "
                    "dropped; everything else (nesting, seeds, selection rule, final retrain, rows, statistics) is unchanged",
            "why_not_select_on_fold0_only": "selecting once on fold 0's inner split and applying the config to folds 1-4 would use "
                                            "images of folds 1-4's test sets (they are fold 0's training ids) for a choice made "
                                            "for those folds, violating the mandatory nesting; rejected",
            "why_drop_lr1e-5": "keeps both depths N=2 and N=4 (the question) and strict nesting; at 2400-3000 steps lr 1e-5 moves "
                               "the blocks least, so it is the least likely to change the answer",
            "applied": mode != "full"},
        "smoke": {"n0_fold0_vs_ours_scratch_s0": smoke0,
                  "ft_20_steps": {str(n): {"steps": r["train"]["steps"], "epoch_loss": r["train"]["epoch_loss"],
                                           "finite": True, "inner_val_CC_after_20_steps": r["mean_CC"],
                                           "s_per_step_incl_first_steps": r["train"]["s_per_step"]} for n, r in smk.items()}},
        "commands": {
            "env": "export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; cd /private/tmp",
            "nested (PRIMARY, the long run; resumable, per-run MPS lock)":
                f"nohup {PY} -I {S}/train_ftblocks.py nested > {FC.KB}/logs/ftblocks_nested.log 2>&1 &",
            "global (SECONDARY, after nested)": f"nohup {PY} -I {S}/train_ftblocks.py global > {FC.KB}/logs/ftblocks_global.log 2>&1 &",
            "optional distill: trigger check + run (refuses unless triggered; Pitt cache first)":
                f"{PY} -I {S}/cache_blocks.py --set pitt --depths <8 if any fold selected N=4 else 10> --dtype fp16 ; "
                f"nohup {PY} -I {S}/train_ftblocks.py distill > {FC.KB}/logs/ftblocks_distill.log 2>&1 &",
            "assemble result files": f"{PY} -I {S}/train_ftblocks.py assemble",
            "statistics": f"{PY} -I {S}/stats_ftblocks.py"},
        "outputs": {"runs": FC.RUNS, "models": FC.MODELS,
                    "results": ["results/ours_ftblocks_nested_per_image.json", "results/ours_ftblocks_nested_summary.json",
                                "results/ours_ftblocks_global_*", "results/ours_ftblocks_distill_* (only if triggered)",
                                "results/ftblocks_stats.json"]},
        "code_sha256": {f: FC.sha256_file(os.path.join(HERE, f)) for f in ("ft_common.py", "train_ftblocks.py", "cache_blocks.py")},
        "aux_code_sha256": {f: FC.sha256_file(os.path.join(HERE, f)) for f in ("stats_ftblocks.py", "bench_ftblocks.py", "write_plan.py")},
        "deviations_log": "any deviation from this plan must be appended to PLAN_DEVIATIONS.md with time and reason; PLAN.json is not edited"}
    FC.jdump(plan, FC.PLAN_JSON)
    write_md(plan)
    print(json.dumps(plan["timing"], indent=1))
    print("mode:", mode)


def write_md(p):
    t = p["timing"]
    g = "\n".join(f"| {i + 1} | {c['name']} | {c['N']} | {c['lr_bb'] if c['lr_bb'] else '- (frozen backbone)'} |" for i, c in enumerate(p["active_grid"]))
    sm = p["smoke"]["n0_fold0_vs_ours_scratch_s0"]
    md = f"""# ftblocks pre-registration (PLAN)

Frozen: **{p['frozen_utc']}** ({p['frozen_local']}). Machine-readable twin: `PLAN.json` (the driver reads the grid
and the rules from it and stores its sha256 in every run). Status: {p['status']}.

## Question
{p['question']}

## What the base readout consumes (read from the ours code)
- checkpoint: {p['base_readout_inputs']['checkpoint']}
- input: {p['base_readout_inputs']['input']}
- tokens: {p['base_readout_inputs']['tokens']}
- readout: {p['base_readout_inputs']['readout']}

## Block cache
Frozen outputs of block 8 and block 10 (all 1025 tokens: CLS + 32x32 patches), fp32, `cache/ftblocks/add_b08_fp32.npy`,
`add_b10_fp32.npy` (3.15 GB each). Two depths so that an N=2 step never pays for a frozen forward of blocks 9-10.
Verification: {p['block_cache']['verification']['summary']}.

## Protocol (per outer fold k of `splits/kfold_seed0.json`)
- Inner split: {p['inner_split']['rule']} (ids in `inner_splits.json`, sha256 `{p['inner_split']['sha256'][:16]}...`).
- Seeds: {p['protocol']['seeds']}.
- N0 = {p['protocol']['N0']}.
- N>0: {p['protocol']['N_gt_0']}.
- Precision: {p['protocol']['precision']}. Augmentation: {p['protocol']['augmentation']}.
- Finalizer: {p['protocol']['finalizer']}.

## Grid (mode: **{p['mode']}**; simplicity order = tie-break order)
| order | config | N | backbone lr |
|---|---|---|---|
{g}

Full grid (pre-registered): N0 + N in {{2, 4}} x lr_bb in {{1e-5, 3e-5}}. Fallback grid: N0, N2_lr3e-05, N4_lr3e-05.

## Selection rule
{p['selection']['rule']}. Then: {p['selection']['final']}.

## Rows
- `ours_ftblocks_nested`: {p['rows']['ours_ftblocks_nested']}
- `ours_ftblocks_global`: {p['rows']['ours_ftblocks_global']}
- `ours_ftblocks_distill`: trigger: {p['optional_distill']['trigger']}. Recipe: {p['optional_distill']['recipe']}.
  Pitt cache: {p['optional_distill']['pitt_cache']}.

## Statistics
{p['statistics']['code']}. {p['statistics']['tests']}. Primary family (Holm {p['statistics']['holm']}):
{', '.join(p['statistics']['primary_family'])}. Secondary family: {p['statistics']['secondary_family']}.
Headline: {p['statistics']['headline']}.

## Timing and budget
Measured on MPS, batch 16, fp32 (s/step incl. batch gather): {', '.join(f"N={n}: {v:.3f}" for n, v in t['s_per_step_fp32_batch16_mps'].items())}.
N0 fold run (smoke): {t['n0_run_wall_s_fold0_final_smoke']:.0f} s.
Estimate full grid: inner {t['estimate_full_grid_h']['inner']} h + finals (worst case) {t['estimate_full_grid_h']['finals_worst_case']} h
= **{t['estimate_full_grid_h']['total']} h** (best case {t['estimate_full_grid_h']['total_best_case']} h); fallback grid: inner
{t['estimate_fallback_grid_h']['inner']} h + finals {t['estimate_fallback_grid_h']['finals_best_case']}-{t['estimate_fallback_grid_h']['finals_worst_case']} h
= **{t['estimate_fallback_grid_h']['total_best_case']}-{t['estimate_fallback_grid_h']['total']} h** (budget ~{t['budget_h']} h; lock waits excluded).
Global row extra (worst case, only configs not already run): {t['global_row_extra_h_worst']} h. Optional distill row (if
triggered; batch 32 assumed 2x): ~{t['optional_distill_h_approx']['all_folds_N2']} h (all folds N=2) to
~{t['optional_distill_h_approx']['all_folds_N4']} h (all folds N=4). MPS driver allocation during N>0 runs: up to
~{max(t['mps_memory_gib_driver_alloc'].values())} GiB (unified memory, 48 GB machine).

Fallback rule: {p['fallback']['rule']}. Applied: **{p['fallback']['applied']}**.
Why not select on fold 0 only: {p['fallback']['why_not_select_on_fold0_only']}.

## Smoke results (no test fold used except the N0 reproduction of an existing row)
- N=0 on fold 0 (final protocol) vs ours_scratch_s0 fold 0: sigma {sm['smoke_fold0']['sigma']} vs {sm['ours_scratch_s0_fold0']['sigma']},
  CC {sm['smoke_fold0']['CC']:.6f} vs {sm['ours_scratch_s0_fold0']['CC']:.6f}; max per-image |dCC| = {sm['max_abs_per_image_diff_CC']:.2e},
  identical CC on {sm['n_images_identical_CC']}/200 images.
- 20-step N=4 and N=2 runs (fold-0 inner split): finite losses, outputs written (cache/ftblocks/smoke).

## Commands
```
{p['commands']['env']}
{chr(10).join(v for k, v in p['commands'].items() if k != 'env')}
```

## Rules
No choice uses an outer test fold. PLAN.json is never edited; deviations go to `PLAN_DEVIATIONS.md` with time and reason.
Existing result files and scripts are not modified; new results are named `ours_ftblocks_*`.
"""
    open(os.path.join(HERE, "PLAN.md"), "w").write(md)


if __name__ == "__main__":
    main()
