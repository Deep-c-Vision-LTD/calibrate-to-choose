"""Fig. S1 of the supplement (revision, 24 Nov 2026; Fig. 2 of the first draft): CC of the models adapted on 800
maps per fold against GFLOPs per image (architecture-level cost), not CPU latency: SUM's CPU latency is dominated
by a pure-PyTorch selective scan (results/sum_cpu_profile.json) and is not an architectural cost.

Reads every number from the results folder (no hard-coded values):
  frontier.json models[row]   CC mean and fold SD (ddof 1); GFLOPs (2 x MACs of convolutions and matrix products)
  cost_table.json             SUM's selective-scan estimate (inference/sum/GFLOPs_incl_scan_estimate): SUM is drawn
                              at the estimate including the scan, with a line back to the counted 26.4 GFLOPs
  ours_ftblocks_nested_summary.json, ftblocks_cost.json
                              draft 5: the fine-tuned xSoyelNet (last 4 DINOv2-B blocks + readout, nested selection):
                              CC mean and fold SD (ddof 1) from the summary, GFLOPs re-measured from its fold-0
                              checkpoint (same graph as the frozen xSoyelNet). IIE is drawn with 384-pixel input (its
                              native-input row has no single per-image cost) and labelled so.

  python3 -I make_fig_cost.py [--out figures/figS1_cost.pdf]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import ENC, INK, INK2, RES, setup  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures", "figS1_cost.pdf"))
args = ap.parse_args()

F = json.load(open(os.path.join(RES, "frontier.json")))
M = {e["row"]: e for e in F["models"]}
CT = json.load(open(os.path.join(RES, "cost_table.json")))
sum_incl = CT["inference"]["sum"]["GFLOPs_incl_scan_estimate"]

ROWS = [  # row, entity, hollow, label, (dx, dy), ha, va
    ("msdb_adapt_s0", "msdb", False, "MSDB, head", (-7, 0), "right", "center"),
    ("sum_ft", "sum", False, "SUM, full fine-tune", (7, 0), "left", "center"),
    ("ours_scratch_s0", "readout", False, "xSoyelNet", (7, 0), "left", "center"),
    ("iie_ft_klcc_e20", "iie", False, "IIE, readouts (384 px)", (7, 0), "left", "center"),
    ("sum_ft_head", "sum", True, "SUM, output head", (7, 0), "left", "center"),
    ("msdb_adapt_dsp", "msdb", True, "MSDB, 13 parameters", (-7, 0), "right", "center"),
]
FTS = json.load(open(os.path.join(RES, "ours_ftblocks_nested_summary.json")))
FTC = json.load(open(os.path.join(RES, "ftblocks_cost.json")))
M["ours_ftblocks_nested"] = {"cost": {"GFLOPs": FTC["GFLOPs"]},
                             "accuracy": {"CC": {"mean": FTS["mean_CC"], "sd_folds_ddof1": FTS["sd_CC_ddof1"]}}}
ROWS.insert(0, ("ours_ftblocks_nested", "readout", False, "xSoyelNet, 4 blocks fine-tuned", (7, 0), "left", "center"))
ROWS[[r[0] for r in ROWS].index("ours_scratch_s0")] = ("ours_scratch_s0", "readout", False, "xSoyelNet, readout", (7, 0), "left", "center")
fig, ax = setup(5.3, 2.7)
ax.set_xscale("log")
for row, key, hollow, lab, (dx, dy), ha, va in ROWS:
    e, c = ENC[key], M[row]
    g = c["cost"]["GFLOPs"]
    x = sum_incl if row.startswith("sum") else g
    y, sd = c["accuracy"]["CC"]["mean"], c["accuracy"]["CC"]["sd_folds_ddof1"]
    if row.startswith("sum"):
        ax.plot([g, x], [y, y], color=e["color"], lw=1.0, zorder=2)
    ax.errorbar([x], [y], yerr=[sd], fmt="none", ecolor=e["color"], elinewidth=0.8, capsize=1.8, zorder=2)
    ax.plot([x], [y], ls="none", marker=e["marker"], ms=5.4 if e["marker"] != "D" else 4.8,
            mfc="white" if hollow else e["color"], mec=e["color"] if hollow else "white", mew=1.0 if hollow else 0.6, zorder=4)
    ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=8.0, color=INK, zorder=5)
ax.set_xlim(12, 40000)
ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
ax.set_xticklabels(["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
ax.minorticks_off()
ax.set_ylim(0.846, 0.948)
ax.set_xlabel("GFLOPs per image (log scale)", color=INK)
ax.set_ylabel("CC, 5 folds (bars: fold SD)", color=INK)
fig.tight_layout(pad=0.3)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fig.savefig(args.out)
print(f"wrote {os.path.abspath(args.out)}; SUM drawn at {sum_incl:.1f} GFLOPs incl. scan estimate")
