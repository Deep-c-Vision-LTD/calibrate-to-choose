# Calibrate to Choose, Fine-Tune to Deploy

Splits, per-image scores, scripts, statistics and cost records for:

> Hamit Soyel. **Calibrate to Choose, Fine-Tune to Deploy: Calibration Gives the Adapted CC Ranking of Three Saliency
> Models on Advertising Images, and Fine-Tuning Four Backbone Blocks Adds 0.043 CC to xSoyelNet.** Knowledge-Based
> Systems, short communication (submitted).

Repository: https://github.com/Deep-c-Vision-LTD/calibrate-to-choose

```bibtex
@article{soyel2026calibrate,
  author  = {Soyel, Hamit},
  title   = {Calibrate to Choose, Fine-Tune to Deploy: Calibration Gives the Adapted CC Ranking of Three Saliency
             Models on Advertising Images, and Fine-Tuning Four Backbone Blocks Adds 0.043 CC to xSoyelNet},
  journal = {Knowledge-Based Systems},
  note    = {Submitted},
  year    = {2026}
}
```

## The result in one paragraph

On the 1000 advertisements of ADD1000 we compare **SUM**, **DeepGaze MSDB** and **DeepGaze IIE** at three levels on
one seed-0 five-fold split: zero-shot (released weights, documented defaults, no ADD1000 map), calibrated (a centre
prior, or SUM's domain token, fitted on each fold's 800 training maps and a Gaussian blur chosen on 200 of them; no
weight is changed) and adapted (trained on the same 800 maps). Zero-shot, SUM leads (CC 0.836), and the order of the
two DeepGaze models rests on a choice that uses no target data: with a uniform centre prior IIE (0.750) leads MSDB
(0.710), with DeepGaze's MIT1003 template MSDB (0.790) leads IIE (0.732). Calibration closes SUM's lead over MSDB:
**MSDB (CC 0.842)** and **SUM (0.839)** are not separated and both lead **IIE (0.768)**. That is already the adapted
CC order: **MSDB with its head trained (CC 0.919, 26,463 parameters trained)** and **SUM fully fine-tuned (CC 0.917,
57,501,196 parameters)** end 0.0019 apart (Holm-adjusted p = 0.17 per image, 0.38 per fold), and IIE with all
readouts trained and native-resolution input is third (CC 0.895; 0.855 at 384 pixels). With IIE at 384 pixels,
training adds **0.077 to 0.087 CC** to every calibrated model. The ranking claim concerns CC: on SIM and KLD
calibrated MSDB leads SUM, adapted SUM leads MSDB on SIM and ties it on KLD. On one held-out fold, **50 maps
calibrate every model to within 0.007 CC of 800 maps**, and 25 maps train SUM to 0.888 and MSDB's head to 0.873,
above every zero-shot or calibrated model on that fold (at most 0.845). The practical rule: record a few dozen
fixation maps of the target material, calibrate every candidate on them and rank on the calibrated CC.

**xSoyelNet**, the author's ASPP readout (3,721,089 trainable parameters, trained from scratch on frozen DINOv2-B
features), trails MSDB's head by 0.022 CC with frozen backbones. **Fine-tuning DINOv2-B's last four blocks with the
readout adds 0.043 CC (to 0.940)**, the pre-registered primary comparison; the depth N in {0, 2, 4} was chosen in
every fold by nested inner validation (N = 4 in 5 of 5 folds; plan in `scripts/ftblocks/`). The fine-tuned model is
ahead of every adapted model on CC, SIM and KLD in all five folds (over MSDB 0.021 CC, over SUM 0.023), but at
unequal adaptation depth: it trains 32,080,257 parameters, MSDB, which also reads DINOv2-B, only its head, and no
depth-matched MSDB fine-tune was run. Its inference graph is the frozen model's: 221 GFLOPs per image against 12,256
for MSDB and about 31 for SUM.

## Reproduction

Python 3.12; every command runs from the repository root. CPU and Apple-MPS runs used an Apple M4 Pro; the SUM
fine-tune used one cloud NVIDIA A10G 24 GB GPU. Each summary file in `results/` records the command that wrote it
(`command`; `command_inference` and `command_scoring` for the SUM zero-shot rows). Each per-image file holds one
record per scored test image and run: `id`, `fold`, `CC`, `SIM`, `KLD`, and where they apply the chosen blur
`sigma`, the `seed`, and for learning-curve rows the number of training maps `n` and the resample.

**0. Setup.**

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# ADD1000 (not redistributed): data/ADD1000/flat/images/<id>.jpg and data/ADD1000/flat/maps/<id>.png (data/README.md)

# SUM code and released weights (not redistributed; third_party/README.md)
git clone https://github.com/Arhosseini77/SUM.git third_party/SUM/code
git -C third_party/SUM/code checkout 03b6eb88f35a71e9106e895014bdd8ff526d947e
python -c "from huggingface_hub import hf_hub_download as d; [d('safe-models/SUM', f, local_dir='third_party/SUM/weights') for f in ('config.json', 'model.safetensors')]"

# DeepGaze's MIT1003 centre-bias template (zero-shot DeepGaze rows)
mkdir -p cache/revision/deepgaze_template
curl -L -o cache/revision/deepgaze_template/centerbias_mit1003.npy \
  https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
shasum -a 256 cache/revision/deepgaze_template/centerbias_mit1003.npy
# 051645b7da47884c138e81f28a4afbadd99b6c3262eaf086434a9626e3578be9
```

DeepGaze IIE and MSDB weights and DINOv2 are fetched by `deepgaze_pytorch` and `transformers` on first use.

**1. Every number of the paper from the stored per-image scores** (no ADD1000 data and no GPU; `build_facts.py`
only checks the hash of the template downloaded in step 0):

```bash
python -I scripts/stats/build_facts.py          # facts/facts.json: each number with its source file, JSON path and recomputation
python -I scripts/figures/make_fig_curves.py    # figures/fig_curves.pdf (Fig. 1)
python -I scripts/figures/make_fig_cost.py      # figures/figS1_cost.pdf (Fig. S1)
```

`build_facts.py` recomputes the fold means, fold SDs (ddof 1) and the paired tests from the per-image files: Wilcoxon
signed-rank over the 1000 images and the Nadeau and Bengio corrected resampled t-test over the five folds, with Holm
correction within each family (three zero-shot pairs, three calibrated pairs, the ten pairs of the five adapted rows
of Table 1, and the families of the supplement); a difference counts only if both tests reject at p < 0.05. The compiled tables can be rebuilt the same way:

```bash
python -I scripts/compile/compile_results.py    # results/main_table.json (every row and its tests)
python -I scripts/compile/compile_curves.py     # results/curves_overview.json
python -I scripts/compile/cost_table.py         # results/cost_table.json from results/cost_raw/
python -I scripts/frontier/compile_frontier.py  # results/frontier.json
```

Rebuilt from the shipped files, these reproduce every number of the shipped tables, except two one-off cache
timings in `cost_table.json` that are read from the feature caches' metadata and come back as null without them.

**2. Zero-shot and calibrated rows** (CPU; needs ADD1000):

```bash
# DeepGaze IIE: readout inputs of the released backbones at 384 x 384, then the uniform-prior and fold-prior rows
python -I scripts/iie/cache_features.py
python -I scripts/iie/zeroshot_iie.py --device cpu --threads 4
# DeepGaze MSDB: frozen multi-scale features at 512 x 512, then its zero-shot rows
(cd scripts/msdb && python -I cache_msdb.py --dev cpu --no-lock --threads 4 && python -I score_zeroshot.py)
# SUM: raw maps for the four domain tokens, then the zero-shot and token-calibrated rows
python -I scripts/sum/infer_zeroshot.py --scripts scripts/sum --code third_party/SUM/code \
  --weights third_party/SUM/weights --data data/ADD1000/flat --split splits/kfold_seed0.json \
  --out_dir cache/sum_maps --device cpu --threads 4
python -I scripts/sum/score_zeroshot_calibrated.py --scripts scripts/sum --data data/ADD1000/flat \
  --split splits/kfold_seed0.json --maps cache/sum_maps --results results
# Table 1: MIT1003-template zero-shot rows, calibration on the blur grid 0 to 32, calibration-only curve
python -I scripts/revision/calib_rows.py --part msdb
python -I scripts/revision/calib_rows.py --part iie
python -I scripts/revision/calib_rows.py --part sum
python -I scripts/revision/iie_native_zeroshot.py   # IIE fed native-resolution images
```

**3. Adapted rows** (five folds, the fold's 800 training maps):

```bash
# DeepGaze MSDB: saliency network + three finalizer scalars, backbone and scale mix frozen (three seeds)
(cd scripts/msdb && python -I train_msdb.py --row msdb_adapt --mode head --loss klcc --seeds 0,1,2 --dev cpu)
# SUM: all parameters, SUM's loss without its NSS term, 12 epochs, on one CUDA GPU
scripts/sum/run_sum_ft_gpu.sh --tag sum_ft
python -I scripts/sum/finalize_ft.py --scripts scripts/sum --work runs/sum_ft/ckpt/sum_ft --results results \
  --split splits/kfold_seed0.json --tag sum_ft --command "scripts/sum/run_sum_ft_gpu.sh --tag sum_ft"
# DeepGaze IIE: all 120 readouts, KL + (1 - CC), 20 epochs (recipe chosen on fold-0 training images by select_recipe.py)
python -I scripts/iie/ft_kfold.py --row iie_ft_klcc_e20 --loss klcc64 --epochs 20 --primary sigma --device cpu --threads 4
# xSoyelNet, the frozen DINOv2-B readout
python -I scripts/ours/cache_features.py
python -I scripts/ours/run_row.py --row ours_scratch_s0 --kind kfold --init scratch --seed 0 --device cpu
# DeepGaze IIE readouts with native-resolution input (same recipe as iie_ft_klcc_e20; CPU)
python -I scripts/iie_native/cache_native.py --threads 6
python -I scripts/iie_native/ft_native_kfold.py --worker --threads 3    # run two workers; each claims folds
python -I scripts/iie_native/ft_native_kfold.py --aggregate             # results/iie_native_adapt_*.json
python -I scripts/iie_native/stats_native.py                            # results/iie_native_adapt_paired_tests.json
```

**3b. xSoyelNet with its last N DINOv2-B blocks fine-tuned** (nested selection, Apple MPS or CUDA). The grid, the
inner splits, the selection rule and the final retrain were fixed in `scripts/ftblocks/PLAN.json` before any run
(`PLAN.md` explains it; `PLAN_AMENDMENT_1.json` and `PLAN_DEVIATIONS.md` record the one amendment, also made before
any run). `PLAN.json`, `PLAN_AMENDMENT_1.json` and `inner_splits.json` are byte-identical to the frozen originals, so
their SHA-256 equals the hash stored in every run record (`results/ftblocks_runs/`); `build_facts.py` checks this.
The two plan files therefore keep the absolute paths of the machine they were written on. The run records and
`PLAN_AMENDMENT_1.json` also store the SHA-256 of the code as it ran. Of those files, `train_ftblocks.py`,
`cache_blocks.py` and `bench_ftblocks.py` are byte-identical here (`train_ftblocks.py` as amended; `PLAN.json` holds
the hash of its version before the amendment). `ft_common.py`, `stats_ftblocks.py`, `overview_ftblocks.py`,
`write_plan.py` and `run_all.sh` differ from the hashed versions only in lines that hold local paths (repository root,
Python interpreter, Pitt Ads folder, working directory, and the lock folder named in a comment); no logic changed. So
`train_ftblocks.py nested` prints a warning that the code differs from the frozen hashes; the warning is expected. The
frozen copies named in `PLAN.md` and `PLAN_DEVIATIONS.md` (`frozen_20261124T083423Z/`) are not released.

```bash
python -I scripts/ftblocks/cache_blocks.py --set add          # frozen outputs of DINOv2-B blocks 8 and 10 (fp32)
python -I scripts/ftblocks/train_ftblocks.py nested           # 15 inner runs, 5 selections, then 5 final runs
python -I scripts/ftblocks/train_ftblocks.py assemble         # results/ours_ftblocks_nested_*.json
python -I scripts/ftblocks/stats_ftblocks.py                  # paired tests of the pre-registered family
python -I scripts/ftblocks/cost_ftblocks.py --model-file cache/ftblocks/models/final_k0_N4_lr3e-05.pt
```

`results/ours_ftblocks_global_*` is a secondary row of the plan (one configuration chosen on the inner-validation CC
pooled over folds); it is not nested and is reported only for transparency (here it is identical to the nested row).
The fine-tuned checkpoints are not released.

**4. Learning curves on fold 0** (subsets of folds 1 to 4 in `splits/curve_subsets_seed0.json`, the same for every
model):

```bash
(cd scripts/msdb && python -I train_msdb.py --row msdb_adapt_curves --mode head --loss klcc --curves --dev cpu)
python -I scripts/iie/ft_curves.py --row iie_curve_klcce20 --loss klcc64 --epochs 20 --device cpu --threads 4
scripts/sum/run_sum_ft_gpu.sh --tag sum_curve_a --curve 25:0,1,2/50:0,1,2/100:0,1,2/200:0
scripts/sum/run_sum_ft_gpu.sh --tag sum_curve_b --curve 200:1,2/400:0
python -I scripts/sum/finalize_curve.py --runs runs/sum_curve_a/ckpt/sum_curve_a runs/sum_curve_b/ckpt/sum_curve_b \
  --jobs sum_curve_a,sum_curve_b
```

The calibration-only curve comes from step 2 (`calib_rows.py`). Grouped and pair-safe folds: pass
`--split splits/groups.json` or `--split splits/kfold_seed0_pairsafe.json` (MSDB, IIE) or `--folds-file` (xSoyelNet).
Inference cost: `scripts/compile/run_cost.sh`, then `scripts/compile/cost_table.py`; SUM's scan profile:
`python -I scripts/cost/profile_sum_scan.py --device cpu --out results/cost_raw/sum_profile_cpu.json`. Then rerun
step 1.

## Baseline conversion: details and pitfalls

- **One scorer for every model.** CC, SIM and KLD on a 384 x 384 grid: ground-truth map and prediction resized
  bilinearly, clipped at 0 and divided by their sum (KLD epsilon 1e-8). Scoring on each model's own output grid
  changes CC by at most 0.00063 over the rows that have a native score.
- **DeepGaze centre-bias template.** The zero-shot DeepGaze rows use DeepGaze's distributed MIT1003 template
  (`centerbias_mit1003.npy`, a 1024 x 1024 float64 log density): `np.load(path, allow_pickle=False)`, then
  `scipy.ndimage.zoom(t, (h / 1024, w / 1024), order=0, mode="nearest")` to the model's input grid, then subtract its
  `logsumexp`. Result files with `_offtpl` in their name use it (an earlier run with a template re-derived from the
  MIT1003 fixation maps is superseded and not included). The template matters: with a uniform prior the
  zero-shot order of IIE and MSDB is reversed, so always state which prior a zero-shot DeepGaze score used.
- **DeepGaze IIE input size.** IIE is fed 384 x 384 images (its backbones see 192 x 192) and its readout inputs are
  cached once; `verify_cache.py` shows the cached path reproduces the full forward pass (`checks/`).
  `iie_native_zeroshot.py` scores the zero-shot model on native-resolution images (a limitation the paper states):
  CC 0.769 against 0.732 at 384 x 384, both with the MIT1003 template. The adapted row was rerun at
  native resolution (`scripts/iie_native/`, CC 0.895 against 0.855); only IIE was given native input.
- **DeepGaze MSDB.** Generalisation mode (`dataset=None`), `pixel_per_dva = 21.7`, 512 x 512 input. Upstream
  `DeepGazeMSDB.train()` does not return `self`, so call `model.eval()` as its own statement; its two input
  normalisers keep float64 buffers, which Apple MPS cannot hold (`msdb_backbone.py` casts them). The adapted head
  starts from the released weights; `validate_msdb.py` checks the cached-feature head against the full model.
- **SUM's domain token.** SUM is conditioned on a one-hot token (0 natural scenes, mouse; 1 natural scenes, eye
  tracking; 2 e-commerce; 3 user interfaces). Token 2 was fixed before scoring; choosing the token on each fold's
  training maps picks the same token (the `sum_calibrated` rows record the choice per fold). Input: 256 x 256 stretch
  resize and ImageNet normalisation; output: a sigmoid map at 256 x 256.
- **SUM's selective scan in pure PyTorch.** The released code needs the CUDA `mamba-ssm` kernel.
  `sum_common.load_sum` replaces it with a chunked pure-PyTorch scan, checked against a sequential reference for
  outputs and gradients (`test_scan.py`) and against the CUDA kernel's stored per-image scores
  (`validate_zeroshot.py`, `results/sum_validate_t2.json`), so SUM runs on CPU, Apple MPS or any CUDA GPU. The scan is
  slow on CPU: it takes 94% of SUM's CPU latency (`results/sum_cpu_profile.json`), so CPU latency is not an
  architectural cost; compare GFLOPs instead (12,256 per image for MSDB, about 31 for SUM including a
  scan estimate).
- **Training maps only.** Every centre prior is the log mean density of training maps, every blur sigma is chosen by
  mean CC on 200 of the training maps (for adapted rows, the same maps that trained the weights), and a fold's 200
  test maps never calibrate or train its models. IIE's adaptation recipe was chosen on fold-0 training images
  (`scripts/iie/select_recipe.py`), which are test images of folds 1 to 4.
- **Warm-start rows.** The warm-start xSoyelNet rows of the supplement start from an earlier checkpoint that is not
  released (`WARM_CKPT`); their per-image scores are included. The xSoyelNet rows of Table 1 (frozen readout and
  four fine-tuned blocks) start from DINOv2-B and a randomly initialised readout and need no checkpoint.
- **Changes in this copy of the JSON files** (`results/`, `checks/`, `splits/`). String fields that held local or
  container file paths, or named the cloud provider, its instance type or its container image, were rewritten
  (repository-relative paths; "cloud NVIDIA A10G 24 GB GPU"), and cloud job names were replaced by their run tags.
  No number was changed: every numeric value was compared before and after.

## Repository layout

```
README.md, LICENSE, requirements.txt
splits/
  kfold_seed0.json             seed-0 five-fold split (numpy default_rng(0) permutation of the sorted ids)
  kfold_seed0_pairsafe.json    the same folds with near-duplicate creatives kept inside one fold
  groups.json                  49 proxy groups (DINOv2 class-token clusters merged through near-duplicates), 5 folds
  near_dup_pairs.json          near-duplicate pairs and how they were found and checked
  curve_subsets_seed0.json     training subsets of the fold-0 learning curves (n = 25 to 800, three resamples)
results/                       <row>_per_image.json (one record per test image and run) and <row>_summary.json;
                               main_table.json, curves_overview.json, frontier.json, cost_table.json (compiled)
  cost_raw/                    inference-cost and SUM scan-profile measurements
  ftblocks_runs/               one record per inner, final and selection run of the fine-tuned xSoyelNet
  reference/                   stored per-image zero-shot MSDB scores of an earlier run (CC, SIM, KLD at 512, keyed by
                               position in the sorted id list); scripts/msdb/score_zeroshot.py and validate_msdb.py
                               compare against them
checks/                        verification outputs of the cached DeepGaze IIE pipeline (scripts/iie/verify_*.py)
facts/
  facts.json                   every number of the paper: value, source file, JSON path, definition, recomputation
  inputs/                      what the public ADD1000 release contains (read by build_facts.py)
figures/                       Fig. 1 and Fig. S1 as rendered by scripts/figures/
scripts/
  stats/build_facts.py         fold statistics, paired tests and Holm families quoted in the paper
  figures/                     figure scripts
  revision/                    MIT1003-template zero-shot rows (also IIE at native resolution), wide-grid calibration,
                               calibration curve, UEyes check
  msdb/  iie/  sum/            DeepGaze MSDB, DeepGaze IIE and SUM: feature caches, zero-shot, calibration, adaptation
  ours/                        xSoyelNet, the frozen DINOv2-B readout, splits and near-duplicate scan
  ftblocks/                    xSoyelNet with the last N DINOv2-B blocks fine-tuned: plan (PLAN.json, PLAN.md,
                               amendment, deviations), inner splits, block cache, nested driver, statistics, cost
  iie_native/                  DeepGaze IIE readouts adapted with native-resolution input
  compile/  frontier/  cost/   compiled tables, paired tests, inference cost, SUM scan profile
  trackA/  distill/            xSoyelNet variants and distilled students (supplement)
data/README.md                 where to put ADD1000 (and the optional Pitt Ads and UEyes images)
third_party/README.md          where to get SUM, the DeepGaze template and the other third-party pieces
```

Scripts write feature caches to `cache/`, logs to `logs/` and GPU runs to `runs/` (all ignored by git). Environment
overrides: `CBYC_ROOT` (repository root), `ADD1000_DIR`, `PITT_ADS_DIR`, `UEYES_DIR`, `WARM_CKPT`, `PY`,
`PUBLISHED_IIE_JSON` (optional: the earlier release's zero-shot IIE scores, default
`third_party/dont-scale-fine-tune/results/deepgaze_iie_zeroshot_add1000.json`; `scripts/iie/zeroshot_iie.py` uses them
only for a reproduction check and skips it when the file is absent).

## License and data

- Code: MIT (see `LICENSE`). The released non-code files (`results/`, `splits/`, `facts/`, `checks/`, `figures/`) are
  under the same MIT licence; they contain no ADD1000 image or map.
- **ADD1000 images and ground-truth maps are not redistributed.** The ADD1000 release
  (https://github.com/lscumt/ADD1000) states no licence: obtain it from the dataset authors and observe any terms they
  set. This repository holds image ids, split assignments, per-image scores and statistics derived from them; it holds
  no image, no map, no predicted map and no trained weights (neither the adapted heads of the third-party models nor
  the xSoyelNet checkpoints).
- Third-party code and weights are fetched from their official sources and are not redistributed here. SUM (MIT),
  CLIP (MIT) and DINOv2 (Apache-2.0) are used under their licences. The DeepGaze IIE/MSDB code, weights and MIT1003
  centre-bias template (github.com/matthias-k/DeepGaze) carry no stated licence; users obtain them from upstream, and
  this repository contains none of them (see `third_party/README.md`), apart from functions and classes in
  `scripts/iie/iie_common.py`, `scripts/iie_native/native_common.py` and `scripts/msdb/msdb_common.py` (about 400
  lines) that re-implement DeepGaze's readouts, finalizers and mixture so that its released weights run on cached
  features; they are adapted from the DeepGaze code and each is marked by an attribution comment (the SUM loss,
  adapted from SUM, is marked the same way). Pitt
  Ads and UEyes images, used only by supplementary checks, are not redistributed either.
