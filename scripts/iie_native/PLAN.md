# Adapted DeepGaze IIE at native input resolution (row `iie_native_adapt`)

Written 24 Nov 2026, before any training of this row. Purpose: test whether IIE is
handicapped by the 384 x 384 input (zero-shot IIE rises from 0.732 to 0.769 CC with the official MIT1003
template when fed native-resolution images, `results/iie_zeroshot_mit1003{,_native}_offtpl_summary.json`).
We re-run the paper's adapted IIE row with native-resolution inputs and change nothing else.

## 1. Which row is replicated

`facts/facts.json` (`protocol.recipe.iie_selected`, `main.iie_ft_klcc_e20.*`)
names two candidate rows for "adapted IIE 0.855": `iie_ft_klcc_e20` (selected recipe, CC 0.8549; the row the
kbs_* statistics families and the curves use) and `iie_ft_kl_sigma` (published recipe + sigma, 0.8551). We
replicate **`iie_ft_klcc_e20`**, the row the paper's statistics are built on.

## 2. Recipe copied from `iie_ft_klcc_e20` (cite: file / args)

Source: `results/iie_ft_klcc_e20_summary.json` (command
`scripts/iie/ft_kfold.py --row iie_ft_klcc_e20 --loss klcc64 --epochs 20 --primary sigma`, folds trained by
`scripts/iie/pool_worker.py` with the same settings, `scripts/iie/tasks.json` row `["splits/kfold_seed0.json",
"klcc64", 20]`), code in `scripts/iie/iie_train.py` (`train_eval`, `loss_fn`, `published_orders`,
`select_sigma`, `Data.prior`) and `scripts/iie/iie_common.py`.

| item | value (identical in the native row) |
|---|---|
| folds | `splits/kfold_seed0.json`, 5 x 200, train = the other 4 folds in fold order |
| model | `deepgaze_pytorch.DeepGazeIIE(pretrained=True)`, fresh per fold; 4 backbones frozen |
| trained parameters | all 4 x 30 readout components (saliency net, fixation-selection net, finalizer sigma and centre-bias weight) as the 23 stacked tensors per backbone of `iie_common.StackedIIE`: 3,059,160 values |
| loss | `klcc64`: KL(g64 \|\| p64) + (1 - CC(p64, g64)), weights 1 and 1, on the 64 x 64 grid; g64 = min-subtracted, sum-normalised bilinear-64 GT (`cache/iie/gt_ms64.npy`); KL eps 1e-12 / clamp 1e-30, CC eps 1e-8 |
| optimiser | Adam, lr 1e-3, default betas, no weight decay, no schedule |
| batch | 8 images per optimiser step, 100 steps per epoch, 20 epochs = 2000 steps per fold |
| data order | published stream: `np.random.default_rng(0)`, `permutation(1000)` consumed, then in-place shuffle of the fold's training list every epoch, continuing across folds (`iie_train.published_orders`) |
| centre prior | per fold, log(to_prob(sum of the 800 training maps, min-subtracted, bilinear 384)) + 1e-8 (`Data.prior`), DeepGaze's centre-bias input |
| post-hoc sigma | Gaussian (scipy `gaussian_filter`) on the 384 density, sigma from [0,2,...,12] chosen by mean CC vs min-subtracted 384 GT on 200 training images `np.random.default_rng(fold).choice(train, 200, replace=False)`; primary row with sigma, `_nosigma` row without |
| scorer | prediction on the common 384 x 384 grid; GT = map resized bilinearly to 384 (`cache/iie/gt_raw384.npy`); clip at 0 + divide by sum; CC, SIM, KLD with eps 1e-8 (`iie_common.metrics`) |
| evaluation code path | trained parameters written back into the released `DeepGazeIIE` modules; predictions via `iie_common.forward_cached` (released saliency / fixation-selection networks and `Finalizer`) |

## 3. What changes: the input size (native rule)

* Input = each image at its original pixel size (PIL RGB, no resize, 0..255 float32), exactly the input path of
  `scripts/revision/iie_native_zeroshot.py`. DeepGaze IIE then applies its own convention: nearest 1/2
  downsampling (`F.interpolate(scale_factor=0.5, recompute_scale_factor=False)`) before the backbones, features
  nearest-resized to the readout grid ceil(H/32) x ceil(W/32), finalizer on floor(H/2) x floor(W/2), output
  log density at H x W. Image sizes 260-1418 px wide, 263-987 px high (739 distinct sizes; mean 571k px vs
  147k px at 384, readout cells mean 577 vs 144).
* Native features (frozen backbones only, no labels) are cached as float32 in `cache/iie_native/`
  (`cache_native.py`; 21.0 GB) exactly like the 384 cache (`scripts/iie/cache_features.py`).
* Things that must follow the input size, each with the rule used (all reduce to the 384 recipe when H = W = 384):
  1. **Centre prior at H x W**: the row's 384 fold prior zoomed to H x W with scipy `zoom(order=0, mode="nearest")`
     and logsumexp-renormalised, the DeepGaze README rule that `iie_native_zeroshot.py` applies to the MIT1003
     template (renormalisation is a constant shift that the finalizer's own normalisation cancels).
  2. **Loss grid**: p64 = 6 x 6 block sums of the H x W density after the scorer's resize to 384 (torch bilinear,
     align_corners=False), renormalised to sum 1. At 384 the resize is the identity and this is exactly the 384
     recipe's "density sum-pooled 6 x 6 from 384".
  3. **Scoring and post-hoc sigma**: the H x W density is resized to 384 x 384 with torch bilinear
     (align_corners=False, float64), exactly as `iie_native_zeroshot.py`; sigma selection and metrics then run on
     the 384 grid with the unchanged code (`iie_train.select_sigma`, `iie_common.metrics`).
* **Training forward**: `native_common.NativeStackedIIE`, the 384 row's `StackedIIE` (same parameter tensors)
  with a size-general finalizer: blur as M_y r M_x^T with torch's own nearest index maps, per-component softmax
  normalised over the full H x W map via the row/column multiplicities of the final nearest upsample, custom
  backward `WFinalMix` (reduces to the 384 row's `FinalMix`). Verified against the released module path on
  native sizes incl. odd H/W (`verify_native.py` -> `cache/iie_native/verify_native.json`: log density max abs
  diff <= 3.8e-6, loss diff <= 1.2e-7, gradient global relative error <= 1.6e-6, custom vs plain autograd
  <= 5.2e-7, write-back 2.9e-6, 3,059,160 trainable values).

## 4. Batching method

Images differ in size, so one optimiser step processes its 8 images (the same 8, in the same order, from the
published stream) as **8 single-image forward/backward passes with gradient accumulation of loss_i / 8, then one
Adam step**. The 384 batch loss is mean_i KL_i + 1 - mean_i CC_i = mean_i (KL_i + 1 - CC_i) and every
normalisation in the readout is per image (layer norm over channels x space, per-image softmax), so the
accumulated gradient equals the batch-8 gradient up to float summation order: same batches, same 2000 Adam steps
per fold, same effective batch size. No grouping by image size (it would change batch composition / order).

## 5. Identical vs different

Identical: folds, training / test ids, data order, model and initial weights, trained parameter set and count,
loss definition and grid, optimiser and lr, effective batch 8, steps, centre-prior maps and formula, sigma grid,
sigma-selection ids and criterion, GT, scorer (384 grid, clip + sum, KLD eps 1e-8), evaluation code path,
output format (`results/iie_native_adapt{,_nosigma}_{per_image,summary}.json`).

Different: (a) input size (native instead of PIL-bilinear 384 x 384), hence backbone features, readout grid,
finalizer grid and output size; (b) the three size adapters of section 3 (prior zoom, loss-grid map, scorer
resize), which are identities at 384; (c) batch of 8 as 8 accumulated single-image passes instead of one
batched pass (mathematically identical); (d) device: CPU only (the 384 row ran folds 0, 3, 4 on CPU and 1, 2 on
MPS); (e) the custom backward is the weighted generalisation `WFinalMix` of `FinalMix`.

## 6. Smoke test (step 2, before the native run)

`smoke384.py`: the new code path fed 384 x 384 inputs on fold 0 for a few steps vs the existing pipeline
(`iie_train.loss_fn` + `StackedIIE` with `IIE_FAST_FINAL=1`, batch 8): (i) native caching code on PIL-bilinear
384 images vs the cached 384 features, (ii) per-step loss values and final parameters, (iii) predictions of the
two models through the evaluation path. Pass = losses within 1e-5 relative, step-1 gradients within 1e-4,
384-grid densities within 1e-3 relative, per-image metrics within 1e-4, same sigma.

Result (`cache/iie_native/smoke384.json`, 10 steps, PASS): features max abs diff 4.1e-6 (relative norm 1.2e-6);
per-step batch losses equal to 1.3e-7 relative (step 0: 0.5925175 vs 0.5925174; step 9: 0.4655399 vs
0.4655398); step-1 gradient global relative error 1.2e-6; parameters after 10 steps global relative difference
3.2e-8 (2.7e-6 of the update); 3,059,160 parameters in both; densities on 24 test images max relative
difference 1.9e-6; selected sigma 2 in both (24-image check); per-image metric differences <= 1.1e-8.

## 7. Statistics (step 4)

`stats_native.py` with the statistics functions of `scripts/compile/compile_results.py` (`compare`, `holm`,
`nb_test`, `rank_biserial`; loaded from that file's source without running it, so `main_table.json` is not
touched): 5000-sample paired bootstrap CI (rng 0), Wilcoxon signed-rank (two-sided, zero_method wilcox),
Nadeau-Bengio corrected t-test on the 5 fold mean differences, Holm per metric and test within a named family.
Families: `iie_native_vs_384` (native minus 384, same recipe); `iie_native_vs_adapted3` (ours_scratch_s0,
msdb_adapt seed 0, sum_ft each minus IIE native); `kbs_adapted_pairwise6_iienative` (the paper's 6 adapted
pairs with IIE native in place of iie_ft_klcc_e20). Question answered: is IIE still last among the adapted
models at native resolution?

## 8. Runtime (timing test, `timing_test.py` -> `cache/iie_native/timing_test.json`)

Fold 0, first training batch of the published stream (8 images, mean 736k px; dataset mean 571k px, so
per-image costs are scaled by 0.776 assuming time ~ pixels), CPU, measured 24 Nov 2026 with the MPS job running:

| | 6 threads | 3 threads |
|---|---|---|
| native feature extraction | 0.75 s / image | 0.54 s / image |
| training step (8 accumulated passes + Adam) | 0.99 s | 1.13 s |
| evaluation (released path + resize) | 0.76 s / image | 0.76 s / image |
| extrapolated: cache 1000 images | 585 s | 419 s |
| extrapolated: train one fold (2000 steps) | 1531 s | 1749 s |
| extrapolated: evaluate one fold (200 test + 200 sigma images) | 236 s | 236 s |

Threads scale poorly (6 threads only 1.14x faster than 3), so the run uses **two worker processes x 3 threads**
(6 threads in total) that claim folds atomically: about 2000 s (33 min) per fold per worker, 3 rounds
(folds 0+1, 2+3, 4) = about 1.7 h wall clock, plus about 5-7 min of feature caching (2 shards x 3 threads).
For comparison the 384 row took 1222-1238 s of training per CPU fold at 3 threads.

## 9. Resources and safety

CPU only, at most 6 threads in total (torch.set_num_threads plus OMP/vecLib env), no MPS, `.mps_lock` not
touched; no downloads (released weights from the local torch hub cache); new code only in
`scripts/iie_native/`, caches in `cache/iie_native/`, results `results/iie_native_adapt*`. Per-fold partial
results + readout checkpoints (`cache/iie_native/partial/`, `cache/iie_native/ckpt/`) and epoch checkpoints
every 5 epochs, so a crash loses at most part of one fold.

## 10. Outcome (appended 24 Nov 2026 after the run; sections 1-9 unchanged except the timing / smoke fills)

Run: cache 2 shards x 3 threads, 430 s wall (`check_cache.py`: recomputed features equal the cache, rel <= 3.7e-7);
training 2 workers x 3 threads, 18:14-20:59 (2 h 44 min wall; folds 0-3 in parallel pairs, fold 4 alone).
Per fold train / eval s: 3194/458, 3177/459, 3311/402, 3332/400, 2187/306 (slower than the timing-test
extrapolation because the two workers and the MPS job compete for memory bandwidth). No resumes, 2000 steps per
fold, 3,059,160 trained values, sigma 6 in every fold (384 row: 4). End-to-end re-score without any cache
(`check_e2e.py`, folds 0 and 3, 20 test images each, full released model on the raw image, GT from the PNGs):
identical to the stored rows (max abs diff 0).

| row | CC | SIM | KLD |
|---|---|---|---|
| iie_native_adapt (native input, + sigma) | 0.8946 +/- 0.0043 | 0.8081 +/- 0.0021 | 0.1428 +/- 0.0024 |
| iie_ft_klcc_e20 (384 input, same recipe) | 0.8549 +/- 0.0043 | 0.7761 +/- 0.0034 | 0.1932 +/- 0.0049 |
| ours_scratch_s0 | 0.8975 | 0.8052 | 0.1489 |
| msdb_adapt_s0 | 0.9190 | 0.8250 | 0.1197 |
| sum_ft | 0.9171 | 0.8285 | 0.1174 |

Native minus 384 (same recipe): CC +0.0396 [+0.0363, +0.0429], SIM +0.0320, KLD -0.0504, 5/5 folds, all Holm p
< 1e-3 on both tests. "IIE last among adapted models" does NOT hold at native resolution: MSDB and SUM stay
significantly ahead (5/5 folds, both tests), but IIE native and ours_scratch_s0 are not separable at the fold
level: ours - IIE CC +0.0029 [+0.0001, +0.0058] (Wilcoxon Holm 0.022, NB Holm 0.19), SIM -0.0029 (IIE better;
W 0.087, NB 0.064), KLD +0.0061 (IIE better; W 0.0088, NB 0.059). Full numbers:
`results/iie_native_adapt_paired_tests.json`.
