# ftblocks pre-registration (PLAN)

Frozen: **2026-11-24T08:34:23Z** (2026-11-24 08:34:23 GMT). Machine-readable twin: `PLAN.json` (the driver reads the grid
and the rules from it and stores its sha256 in every run). Status: frozen BEFORE any grid run; only smoke tests (fold-0 N=0 reproduction, 20-step N=4/N=2 runs scored on the fold-0 INNER-VAL set, never on a test fold) and the timing bench have been run.

## Question
Does fine-tuning the last N transformer blocks of frozen-DINOv2-base (+ final LayerNorm + our ASPP readout) on each fold's 800 maps close the gap from ours_scratch_s0 (CC 0.8975) towards msdb_adapt / sum_ft (~0.92) at unchanged inference cost (same graph, 221 GFLOPs)?

## What the base readout consumes (read from the ours code)
- checkpoint: facebook/dinov2-base (HF transformers AutoModel, sdpa attention, no register tokens)
- input: PIL RGB, bilinear resize 448x448, /255, ImageNet mean (0.485,0.456,0.406) / std (0.229,0.224,0.225), interpolate_pos_encoding=True
- tokens: last_hidden_state = final LayerNorm(output of block 12); CLS dropped; 1024 patch tokens -> 768x32x32
- readout: ours_lib.ReadoutHead('full'): ASPP 768->128 (dilations 1,3,6,12 + GAP) + 1x1 head + learned 16x16 centre bias; 3,721,089 params

## Block cache
Frozen outputs of block 8 and block 10 (all 1025 tokens: CLS + 32x32 patches), fp32, `cache/ftblocks/add_b08_fp32.npy`,
`add_b10_fp32.npy` (3.15 GB each). Two depths so that an N=2 step never pays for a frozen forward of blocks 9-10.
Verification: CPU replay of the remaining frozen blocks + final norm on the cached block outputs reproduces last_hidden_state bit-for-bit (max abs diff 0.0) and the fp16 cast equals the existing readout cache in every element for all 1000 images at both depths; on MPS the replay deviates from the CPU by <= 2.1e-4 absolute (features up to |19.6|, i.e. ~1e-5 relative).

## Protocol (per outer fold k of `splits/kfold_seed0.json`)
- Inner split: perm = numpy default_rng(100 + k).permutation(800) over the outer training ids in run_row order; inner_train = tr[perm[:640]], inner_val = tr[perm[640:]] (ids in `inner_splits.json`, sha256 `a5131be46ed3e6a3...`).
- Seeds: fold k: torch.manual_seed(k) immediately before the readout is built (all configs: same readout init); data-order rng numpy default_rng(k), including the base recipe's start-of-training 200-image draw.
- N0 = the unchanged base recipe: ours_lib.train on the fp16 feature cache (AdamW lr 3e-4 wd 1e-4 constant, 60 epochs, batch 16, KL + (1 - CC) at 64x64).
- N>0: cached frozen block-(12-N) output -> trainable blocks 12-N+1..12 -> trainable final LayerNorm -> readout. One AdamW: readout lr 3e-4 wd 1e-4 constant; backbone blocks + final norm lr_bb, wd 0.05 (all their parameters), linear warmup lr_bb*(s+1)/W for s < W = round(0.05 * total steps), then constant (the readout's schedule). Same loss, batch 16, data order, 60 epochs (inner 2400 steps, outer 3000 steps).
- Precision: fp32 everywhere (no autocast; bf16 was timed for information only and is NOT used). Augmentation: none (horizontal flip is NOT in the pre-registered grid and is therefore not used).
- Finalizer: Gaussian sigma from {0,2,4,6,8,10,12} by mean CC on a random 200-image subset of the run's training ids (inner: of inner-train; final: of the 800 outer training ids), drawn from the data-order rng after training.

## Grid (mode: **fallback_drop_lr1e-5**; simplicity order = tie-break order)
| order | config | N | backbone lr |
|---|---|---|---|
| 1 | N0 | 0 | - (frozen backbone) |
| 2 | N2_lr3e-05 | 2 | 3e-05 |
| 3 | N4_lr3e-05 | 4 | 3e-05 |

Full grid (pre-registered): N0 + N in {2, 4} x lr_bb in {1e-5, 3e-5}. Fallback grid: N0, N2_lr3e-05, N4_lr3e-05.

## Selection rule
best = max mean inner-val CC; every config with CC >= best - 0.001 is tied; select the first tied config in simplicity order = grid order (smaller N first, then lower lr_bb). Then: retrain the selected config on all 800 outer training ids (3000 steps), sigma on 200 of them, score the 200 outer test ids ONCE.

## Rows
- `ours_ftblocks_nested`: PRIMARY. Per-fold selected config (recorded per fold).
- `ours_ftblocks_global`: SECONDARY, transparency only: one config chosen by the same rule on the inner-val CC averaged over the 5 folds; run on all 5 outer folds; final runs are shared with the nested row where the config coincides (identical code, seeds and data).
- `ours_ftblocks_distill`: trigger: run ONLY if in >= 4 of 5 folds the nested selection beats the N0 control on inner validation, i.e. the selected config is not N0 (by the tie rule this implies its inner-val CC exceeds N0's by more than 0.001). Recipe: fold-k nested config + the distill_pitt recipe of scripts/distill/train_students.py: each step the base 16 ADD1000 training ids with GT + 16 Pitt pool images (2,499) with teacher-k targets (cache/distill/targets/pitt_t64.npy, MSDB head adapted on fold k's 800 training maps), one forward of 32, loss = L(GT) + 1.0 L(Pitt); Pitt stream Cycler(default_rng(100000 + k)); 3000 steps; sigma by the base rule on ADD1000 training images.
  Pitt cache: frozen block outputs of the 2,499 Pitt pool images computed exactly like the ADD1000 cache (cache_blocks.py --set pitt) in fp16 (3.9 GB per depth) at depth 12-N of the selected N; if folds select both N=2 and N=4 only depth 8 is cached and N=2 folds apply frozen blocks 9-10 on the fly for BOTH streams (ADD1000 then also starts from block 8; mathematically identical model).

## Statistics
scripts/ftblocks/stats_ftblocks.py; holm, nb_test, rank_biserial, compare, vec, fold_means, describe, seed_avg executed verbatim from scripts/compile/compile_results.py (ast extraction). paired bootstrap 95% CI (5000) of the mean per-image difference, per-image Wilcoxon signed-rank, fold-level Nadeau-Bengio corrected t (df 4), fold SD ddof 1; differences = ftblocks row minus reference. Primary family (Holm within each family, separately per metric (CC, SIM, KLD) and test (Wilcoxon, NB)):
ours_ftblocks_nested - ours_scratch_s0, ours_ftblocks_nested - ours_scratch_seedavg, ours_ftblocks_nested - distill_pitt, ours_ftblocks_nested - ours_L448_s0, ours_ftblocks_nested - ours_ms_B_s0, ours_ftblocks_nested - msdb_adapt_s0, ours_ftblocks_nested - sum_ft. Secondary family: each existing secondary row (global, distill) vs the same 7 references; Holm within the family.
Headline: ours_ftblocks_nested - ours_scratch_s0 on CC (primary comparison).

## Timing and budget
Measured on MPS, batch 16, fp32 (s/step incl. batch gather): N=0: 0.048, N=2: 0.610, N=4: 1.132.
N0 fold run (smoke): 150 s.
Estimate full grid: inner 12.16 h + finals (worst case) 4.81 h
= **16.97 h** (best case 14.79 h); fallback grid: inner
6.18 h + finals 2.63-4.81 h
= **8.81-10.99 h** (budget ~10.0 h; lock waits excluded).
Global row extra (worst case, only configs not already run): 4.73 h. Optional distill row (if
triggered; batch 32 assumed 2x): ~5.1 h (all folds N=2) to
~9.4 h (all folds N=4). MPS driver allocation during N>0 runs: up to
~18.3 GiB (unified memory, 48 GB machine).

Fallback rule: if the full-grid estimate exceeds 10.0 h, run the fallback grid (N0, N2_lr3e-05, N4_lr3e-05): lr_bb = 1e-5 is dropped; everything else (nesting, seeds, selection rule, final retrain, rows, statistics) is unchanged. Applied: **True**.
Why not select on fold 0 only: selecting once on fold 0's inner split and applying the config to folds 1-4 would use images of folds 1-4's test sets (they are fold 0's training ids) for a choice made for those folds, violating the mandatory nesting; rejected.

## Smoke results (no test fold used except the N0 reproduction of an existing row)
- N=0 on fold 0 (final protocol) vs ours_scratch_s0 fold 0: sigma 4 vs 4,
  CC 0.901052 vs 0.901052; max per-image |dCC| = 0.00e+00,
  identical CC on 200/200 images.
- 20-step N=4 and N=2 runs (fold-0 inner split): finite losses, outputs written (cache/ftblocks/smoke).

## Commands
```
export OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # from the repository root
nohup python -I scripts/ftblocks/train_ftblocks.py nested > logs/ftblocks_nested.log 2>&1 &
nohup python -I scripts/ftblocks/train_ftblocks.py global > logs/ftblocks_global.log 2>&1 &
python -I scripts/ftblocks/cache_blocks.py --set pitt --depths <8 if any fold selected N=4 else 10> --dtype fp16 ; nohup python -I scripts/ftblocks/train_ftblocks.py distill > logs/ftblocks_distill.log 2>&1 &
python -I scripts/ftblocks/train_ftblocks.py assemble
python -I scripts/ftblocks/stats_ftblocks.py
```

## Rules
No choice uses an outer test fold. PLAN.json is never edited; deviations go to `PLAN_DEVIATIONS.md` with time and reason.
Existing result files and scripts are not modified; new results are named `ours_ftblocks_*`.

## Amendment 1 (appended 2026-11-24T08:55:56Z, before any grid run; machine-readable: `PLAN_AMENDMENT_1.json`, sha256 778a2aea17f9fae2..., written 2026-11-24T08:51:23Z)
Source: adversarial pre-run review (must-fix M1-M4). Written before any grid run and before any outer-test score of an ftblocks config existed (`cache/ftblocks/runs` did not exist). PLAN.json is not edited. Grid, tie order, inner splits, seeds, recipe, finalizer, selection rule, final retrain, trigger, statistics code and the primary Holm family (nested vs the 7 references; kept as frozen, a superset of the 4 headline references) are unchanged.
- **M1, global row is NOT nested.** The pooled choice averages every fold's inner-val CC; fold j's inner-val ids are test ids of the other folds. Per outer fold k, 117/124/115/124/122 distinct test images of 200 (folds 0-4; 155/160/157/165/163 counted with multiplicity) enter the pooled choice. `ours_ftblocks_global` stays as secondary transparency only, its summary status says "NOT NESTED ... must never be cited as a test result", and its statistics are descriptive only.
- **M2, order.** All 15 inner runs first (config-major: N0 for folds 0-4, then N2_lr3e-05 for folds 0-4, then N4_lr3e-05 for folds 0-4). Then the 5 selections, the pooled choice (`runs/pooled_selection.json`) and the distill trigger (`runs/distill_trigger.json`) are written. Only then the 5 final runs. Any further cut can be decided before any outer-test score exists.
- **M3, divergence rule.** A non-finite training loss or non-finite metrics: the run is written with status "diverged" (lock released, driver continues, restart skips it). Diverged inner run: CC = -inf (cannot be selected; pooled CC -inf if it diverged in any fold). Diverged final run: fall back to the next config by the inner rule (the rule re-applied to the remaining configs, `selection_k*.json` "fallback_order"); the chain is recorded. The distill row starts from the config actually used in the fold's nested row. Tested on MPS with a forced NaN loss.
- **M4, smoke re-verified with the amended code** (train_ftblocks.py sha256 e1be47ac...; ft_common.py and cache_blocks.py unchanged; frozen copies in `frozen_20261124T083423Z/`). Fold-0 N=0 vs ours_scratch_s0 fold 0: sigma 4 vs 4, CC 0.9010515516988788 vs 0.9010515516988788, max per-image |dCC| = |dSIM| = |dKLD| = 0.0, 200/200 identical (153 s). 20-step N=4 / N=2 (fold-0 inner split): losses 0.6337 / 0.6362, inner-val CC 0.7679 / 0.7649 (identical to the first smoke), 1.133 / 0.683 s/step. Files: `cache/ftblocks/smoke_amend1/`, log `logs/ftblocks_smoke_amend1.log`.
- Run command: `nohup scripts/ftblocks/run_all.sh > logs/ftblocks_run_all.log 2>&1 &` (nested, global, optional distill incl. the Pitt cache if triggered, assemble, stats, overview).
