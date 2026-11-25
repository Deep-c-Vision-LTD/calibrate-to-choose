"""Write results/<tag>_per_image.json and results/<tag>_summary.json for a SUM fine-tune row
from the working directory produced by ft_sum_folds.py.

python -I finalize_ft.py --scripts DIR --work cache/sum_ft --results results --split FILE --tag sum_ft \
    --command "<exact training command>" [--note "..."]
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
for a in ("scripts", "work", "results", "split", "tag", "command"):
    ap.add_argument(f"--{a}", required=True)
ap.add_argument("--note", default="")
args = ap.parse_args()
sys.path.insert(0, args.scripts)
import numpy as np
import sum_common as sc

folds = sc.load_split(args.split)
per = json.load(open(os.path.join(args.work, f"{args.tag}_per_image.json")))
fm = json.load(open(os.path.join(args.work, f"{args.tag}_fold_meta.json")))
logs = [json.loads(l) for l in open(os.path.join(args.work, f"{args.tag}_trainlog.jsonl"))]
done_folds = sorted(int(k) for k in fm)
fsub = [folds[k] if k in done_folds else [] for k in range(len(folds))]
use = [r for r in per if r["fold"] in done_folds]


def summ(rows, keymap):
    return sc.fold_summary([{**r, "CC": r[keymap[0]], "SIM": r[keymap[1]], "KLD": r[keymap[2]]} for r in rows],
                           [folds[k] for k in done_folds]) if rows else None


# fold_summary indexes folds by position, so remap fold ids to positions for partial runs
pos = {k: n for n, k in enumerate(done_folds)}
rows_pos = [{**r, "fold": pos[r["fold"]]} for r in use]
main = summ(rows_pos, ("CC", "SIM", "KLD"))
nos = summ(rows_pos, ("CC_nosigma", "SIM_nosigma", "KLD_nosigma"))
nat = summ(rows_pos, ("CC_native256", "SIM_native256", "KLD_native256"))
for s in (main, nos, nat):
    for f in s["per_fold"]:
        f["fold"] = done_folds[f["fold"]]
epochs = [l for l in logs if l.get("event") == "epoch"]
MODE = next(iter(fm.values())).get("mode", "full")
N_TRAIN = int(next(iter(fm.values())).get("n_trainable_params", 57501196))
args_run = None
loss_desc = {"sum_ft": "SUM's own objective -2*CC + 10*KL - SIM + 5*MSE (NSS term dropped: ADD1000 has density maps only)",
             "sum_ft_klcc": "our objective KL(gt || softmax(logits)) + (1 - CC(sigmoid(logits), gt)) at 256x256"}
out = {"row": args.tag, "folds_completed": done_folds, "complete": len(done_folds) == len(folds),
       **main, "no_sigma_at384": nos, "native256_no_sigma": nat,
       "fold_meta": fm,
       "train_log_per_epoch": [{k: e[k] for k in e if k != "event"} for e in epochs],
       "loss": loss_desc.get(args.tag.split("_seed")[0].replace("_dec", ""), args.tag),
       "mode": MODE,
       "n_trainable_params": N_TRAIN,
       "params_total": 57501196,
       "calibration_maps_used_per_fold": {"sigma_selection": 200, "prior_fit": 0,
                                          "note": "sigma subset drawn from the fold's 800 training images"},
       "training_maps_used_per_fold": 800,
       "init": (f"released SUM weights (hf safe-models/SUM, 57,501,196 parameters); "
                + ("all parameters trained" if MODE == "full" else
                   f"encoder frozen in eval mode (outputs cached once), only the conditional decoder "
                   f"(layers_up + final, {N_TRAIN:,} parameters) trained")),
       "protocol": ("ADD1000 seed-0 5-fold split (splits/kfold_seed0.json); "
                    + ("full fine-tune of SUM" if MODE == "full" else
                       "decoder-only adaptation of SUM (frozen VMamba encoder, trained conditional decoder)")
                    + " on the "
                    "fold's 800 training images, domain token 2 (e-commerce; best on training folds), input 256x256 "
                    "stretch resize + ImageNet normalisation, Adam lr 1e-4 with StepLR(4, 0.1), effective batch 16 "
                    "(gradient accumulation), stochastic depth 0.3 as released, fixed epoch count, no early stopping, "
                    "test fold never consulted; post-hoc Gaussian finalizer sigma from {0,2,4,6,8,10,12} by mean CC on "
                    "200 training images (numpy default_rng(fold)); CC/SIM/KLD on 384x384 (PIL bilinear resize of the "
                    "256 output; GT resized bilinearly to 384; clip at 0 and divide by sum); native256 = raw 256 output "
                    "vs GT resized to 256, no sigma"),
       "wallclock_s_total": float(sum(v["fold_seconds"] for v in fm.values())),
       "wallclock_s_training_only": float(sum(v["train_seconds"] for v in fm.values())),
       "command": args.command, "note": args.note}
json.dump(use, open(os.path.join(args.results, f"{args.tag}_per_image.json"), "w"))
json.dump(out, open(os.path.join(args.results, f"{args.tag}_summary.json"), "w"), indent=1)
print(f"{args.tag}: folds {done_folds} CC {main['CC_mean']:.4f} +/- {main['CC_sd_folds']:.4f} "
      f"SIM {main['SIM_mean']:.4f} KLD {main['KLD_mean']:.4f} | no-sigma CC {nos['CC_mean']:.4f} | "
      f"native256 CC {nat['CC_mean']:.4f}")
