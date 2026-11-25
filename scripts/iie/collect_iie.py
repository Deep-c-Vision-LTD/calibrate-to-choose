"""Print a compact table of every IIE row summary and the reproduction checks."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import iie_common as C

R = C.RESULTS
rows = ["iie_zeroshot_uniform", "iie_zeroshot_adprior", "iie_zeroshot_foldprior", "iie_zeroshot_foldprior_sigma",
        "iie_ft_kl", "iie_ft_kl_sigma", "iie_ft_klcc", "iie_ft_klcc_nosigma",
        "iie_ft_klcc_e20", "iie_ft_klcc_e20_nosigma",
        "iie_ft_klcce20_grouped", "iie_ft_klcce20_grouped_nosigma", "iie_ft_klcce20_pairsafe", "iie_ft_klcce20_pairsafe_nosigma",
        "iie_ft_klcc_grouped", "iie_ft_klcc_grouped_nosigma", "iie_ft_klcc_pairsafe", "iie_ft_klcc_pairsafe_nosigma"]
out = {}
for r in rows:
    p = os.path.join(R, f"{r}_summary.json")
    if not os.path.exists(p):
        print(f"{r:32s} MISSING"); continue
    s = json.load(open(p))
    out[r] = {k: s.get(k) for k in ("CC_mean", "CC_sd_over_folds", "SIM_mean", "SIM_sd_over_folds", "KLD_mean",
                                    "KLD_sd_over_folds", "calibration_maps", "training_maps", "sigma_per_fold",
                                    "wall_clock_s_total", "published_scoring_min_subtracted_gt")}
    print(f"{r:32s} CC={s['CC_mean']:.4f}+/-{s['CC_sd_over_folds']:.4f} SIM={s['SIM_mean']:.4f}+/-{s['SIM_sd_over_folds']:.4f} "
          f"KLD={s['KLD_mean']:.4f}+/-{s['KLD_sd_over_folds']:.4f}  folds CC={[round(x['CC'], 4) for x in s['per_fold']]}"
          f"  calib={s.get('calibration_maps')} train={s.get('training_maps')} sigma={s.get('sigma_per_fold')}")
for cname in ("iie_curve_klcce20", "iie_curve_klcc"):
  p = os.path.join(R, f"{cname}_summary.json")
  print(cname)
  if os.path.exists(p):
    s = json.load(open(p))
    out[cname] = s["per_n"]
    for e in s["per_n"]:
        print(f"curve n={e['n']:4d} runs={e['n_runs']} CC(sigma)={e['sigma_CC_mean']:.4f}+/-{e['sigma_CC_sd']:.4f} "
              f"SIM={e['sigma_SIM_mean']:.4f} KLD={e['sigma_KLD_mean']:.4f} CC(nosigma)={e['nosigma_CC_mean']:.4f} "
              f"runs={[round(x, 4) for x in e['sigma_CC_runs']]} steps={e['steps_per_run']}")
    print("n=0 refs (fold 0):", s["n0_reference_fold0"])
json.dump(out, open(os.path.join(C.KBS, "logs", "iie_collect.json"), "w"), indent=1)
