"""
Choose the adapted-IIE recipe for the learning curves using fold-0 TRAINING data only:
fold-0's 800 training ids are split (default_rng(2026).permutation) into 600 inner-train and
200 inner-validation ids; each recipe (published kl384/20 epochs; our klcc64/60 epochs) is
trained on the 600 with its own prior from those 600 maps and sigma selected on 200 of them,
and scored on the 200 inner-validation ids (with sigma, the variant reported in the curves).
The fold-0 test images are never touched.

  python -I select_recipe.py [--device cpu]
"""
import os, sys, json, time, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import iie_common as C
import iie_train as T

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--recipes", default="kl384_e20,klcc64_e60,klcc64_e20")
a = ap.parse_args()
torch.set_num_threads(a.threads)
LOG = os.path.join(C.KBS, "logs", "iie_select_recipe.log")
PART = os.path.join(C.KBS, "logs", "iie_partial"); os.makedirs(PART, exist_ok=True)


def log(s):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {s}"
    print(line, flush=True); open(LOG, "a").write(line + "\n")


D = T.Data()
tr800 = [i for k in range(1, 5) for i in D.folds[k]]
perm = np.random.default_rng(2026).permutation(len(tr800))
inner_tr = [tr800[j] for j in perm[:600]]
inner_va = [tr800[j] for j in perm[600:]]
assert not set(inner_va) & set(D.folds[0])
sel = [str(x) for x in np.random.default_rng(0).choice(inner_tr, size=200, replace=False)]
recipes = {"kl384_e20": ("kl384", 20), "klcc64_e60": ("klcc64", 60), "klcc64_e20": ("klcc64", 20)}
out = {}
for name, (loss, ep) in recipes.items():
    if name not in a.recipes.split(",") and not os.path.exists(os.path.join(PART, f"iie_select_{name}.json")):
        continue
    pp = os.path.join(PART, f"iie_select_{name}.json")
    if os.path.exists(pp):
        out[name] = json.load(open(pp)); continue
    orders = T.seeded_orders(inner_tr, ep, seed=0)
    res = T.train_eval(D, inner_tr, inner_va, orders, loss, 1e-3, 8, a.device, sel, log=log, tag=name)
    res = {"inner_val_CC_sigma": float(np.mean([r["CC"] for r in res["rows_sigma"]])),
           "inner_val_CC_nosigma": float(np.mean([r["CC"] for r in res["rows_nosigma"]])),
           "inner_val_SIM_sigma": float(np.mean([r["SIM"] for r in res["rows_sigma"]])),
           "inner_val_KLD_sigma": float(np.mean([r["KLD"] for r in res["rows_sigma"]])),
           "sigma": res["sigma"], "steps": res["steps"], "t_train_s": res["t_train_s"]}
    json.dump(res, open(pp, "w")); out[name] = res
    log(f"{name}: inner-val CC(sigma)={res['inner_val_CC_sigma']:.4f} CC(nosigma)={res['inner_val_CC_nosigma']:.4f}")
if len(out) < len(recipes):
    log(f"only {list(out)} done; not choosing yet"); sys.exit(0)
best = max(out, key=lambda k: out[k]["inner_val_CC_sigma"])
rep = {"recipes": out, "chosen": best, "inner_train_n": 600, "inner_val_n": 200,
       "rule": "fold-0 training ids only; choose by inner-validation CC with the sigma finalizer",
       "command": "python -I " + " ".join([os.path.abspath(__file__)] + sys.argv[1:])}
json.dump(rep, open(os.path.join(C.RESULTS, "iie_recipe_selection_summary.json"), "w"), indent=1)
log(f"chosen: {best}")
