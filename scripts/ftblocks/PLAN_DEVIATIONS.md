# ftblocks: deviations and amendments to PLAN.json (frozen 2026-11-24T08:34:23Z)

PLAN.json is never edited. Every change is listed here with its UTC time and reason.

## 2026-11-24T08:55:56Z: Amendment 1 (before any grid run; no outer-test score of any ftblocks config existed)
Reason: adversarial pre-run review, must-fix items M1-M4. Details: `PLAN_AMENDMENT_1.json` and the "Amendment 1" section of `PLAN.md`.
1. M1: `ours_ftblocks_global` is labelled NOT NESTED (the pooled choice sees 115-124 distinct test images of each fold through the other folds' inner-val sets; 155-165 with multiplicity). Secondary, descriptive only, never a test result.
2. M2: all inner runs of all folds run before any final run; selections, pooled choice and distill trigger are written before the first final run.
3. M3: diverged runs are recorded (status "diverged", CC = -inf for selection); a diverged final run falls back to the next config by the inner rule.
4. M4: smoke checks re-run with the amended code: N=0 again reproduces ours_scratch_s0 fold 0 exactly (200/200 identical), the 20-step N=4/N=2 runs give the same losses and inner-val CC as before.
Code: train_ftblocks.py changed (sha256 e1be47ac...); ft_common.py, cache_blocks.py, stats_ftblocks.py unchanged; new overview_ftblocks.py and run_all.sh. Frozen versions kept in `frozen_20261124T083423Z/`.
