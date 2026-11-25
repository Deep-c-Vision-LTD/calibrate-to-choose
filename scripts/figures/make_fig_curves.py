"""Fig. 1 of the KBS short communication (revision, 24 Nov 2026): CC on ADD1000 fold 0 against the number of
fold-0 training maps, for two levels of using those maps.

  solid lines, filled markers  : adapted (weights trained on the n maps; prior and blur from the same maps)
      msdb_adapt_curves_summary.json   DeepGaze MSDB, head adapted (26,463 parameters)
      iie_curve_klcce20_summary.json   DeepGaze IIE, all readouts adapted (selected recipe, with blur)
      sum_ft_curve_summary.json        SUM full fine-tune: drawn only if the file exists (run pending on 24 Nov 2026);
                                       otherwise a placeholder note is drawn
      curve_ours_scratch_aggregate.json  xSoyelNet, frozen DINOv2-B readout trained from scratch (draft 6: added after
                                       the final review; the fine-tuned xSoyelNet has no curve)
  dashed lines, hollow markers : calibration only (released weights; prior or token and blur from the n maps)
      calib_curve_{msdb,iie}_offtpl.json, calib_curve_sum.json  (scripts/revision/calib_rows.py; MSDB and
      IIE from the run with DeepGaze's distributed MIT1003 template, 24 Nov 2026)
  n = 0 column                 : zero-shot with documented defaults (by_n/0 of the same files)
Points: mean over resamples of the per-resample mean CC; bars: SD over resamples (ddof 1), none for a single run.
No number is typed by hand.

  python3 -I make_fig_curves.py [--out figures/fig_curves.pdf]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from figstyle import ENC, INK, INK2, RES, setup  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--res", default=RES)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures", "fig_curves.pdf"))
args = ap.parse_args()


def load(name):
    return json.load(open(os.path.join(args.res, name)))


def series_from(points):
    points = sorted(points)
    n = np.array([p[0] for p in points], float)
    m = np.array([np.mean(p[1]) for p in points])
    s = np.array([np.std(p[1], ddof=1) if len(p[1]) > 1 else np.nan for p in points])
    return n, m, s


adapted = []  # (key, label, n, mean, sd)
msdb = load("msdb_adapt_curves_summary.json")["curve"]
adapted.append(("msdb", "MSDB, head adapted", *series_from([(c["n"], c["runs_CC"]) for c in msdb])))
SUM_FILE = os.path.join(args.res, "sum_ft_curve_summary.json")
have_sum = os.path.exists(SUM_FILE)
if have_sum:
    sc = json.load(open(SUM_FILE))["curve"]
    adapted.append(("sum", "SUM, full fine-tune", *series_from([(c["n"], c["runs_CC"]) for c in sc])))
iie = load("iie_curve_klcce20_summary.json")["per_n"]
adapted.append(("iie", "IIE, readouts adapted", *series_from([(c["n"], c["sigma_CC_runs"]) for c in iie])))
ro = load("curve_ours_scratch_aggregate.json")["by_n"]
adapted.append(("readout", "xSoyelNet, readout", *series_from([(c["n"], [r["CC"] for r in c["runs"]]) for c in ro])))

calib, zs = [], {}
missing = []
CURVE = {"msdb": "calib_curve_msdb_offtpl.json", "iie": "calib_curve_iie_offtpl.json", "sum": "calib_curve_sum.json"}
for key, lab in (("msdb", "MSDB"), ("sum", "SUM"), ("iie", "IIE")):
    if not os.path.exists(os.path.join(args.res, CURVE[key])):
        missing.append(lab)
        continue
    d = load(CURVE[key])["by_n"]
    zs[key] = d["0"]["CC"]
    pts = [(int(n), [r["CC"] for r in e["runs"]]) for n, e in d.items() if n != "0"]
    calib.append((key, f"{lab}, calibration only", *series_from(pts)))

X0 = 12.0
fig, ax = setup(5.3, 1.55)
ax.set_xscale("log")
handles = []
LEG = {"msdb": "MSDB", "sum": "SUM", "iie": "IIE", "readout": "xSoyelNet"}
for key, lab, n, m, s in adapted:
    e = ENC[key]
    ok = ~np.isnan(s)
    ax.errorbar(n[ok], m[ok], yerr=s[ok], fmt="none", ecolor=e["color"], elinewidth=0.8, capsize=1.8, zorder=3)
    h, = ax.plot(n, m, color=e["color"], ls="-", lw=1.6, marker=e["marker"], ms=5.0 if e["marker"] != "D" else 4.4,
                 mfc=e["color"], mec="white", mew=0.6, zorder=4, label=LEG[key])
    handles.append(h)
for key, lab, n, m, s in calib:
    e = ENC[key]
    ok = ~np.isnan(s)
    ax.errorbar(n[ok], m[ok], yerr=s[ok], fmt="none", ecolor=e["color"], elinewidth=0.7, capsize=1.6, zorder=3)
    h, = ax.plot(n, m, color=e["color"], ls=(0, (4, 2)), lw=1.2, marker=e["marker"], ms=4.6 if e["marker"] != "D" else 4.0,
                 mfc="white", mec=e["color"], mew=1.0, zorder=4, label=lab)
for key in [k for k in ("msdb", "sum", "iie") if k in zs]:
    e = ENC[key]
    ax.plot([X0], [zs[key]], ls="none", marker=e["marker"], ms=5.0 if e["marker"] != "D" else 4.4, mfc="white",
            mec=e["color"], mew=1.0, zorder=4)
ax.axvline(19.0, color=INK2, lw=0.5, ls=(0, (2, 2)), zorder=1)
# label above the plotting area (draft 6): inside it, the label crossed the y axis and the top grid line
ax.text(X0, 1.02, "zero-shot", ha="center", va="bottom", fontsize=8.5, color=INK2,
        transform=ax.get_xaxis_transform(), clip_on=False)
if missing:
    ax.text(0.99, 0.10, "calibration curve pending: " + ", ".join(missing), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8.0, color="#c0392b", style="italic")
if not have_sum:
    ax.text(0.99, 0.02, "SUM full fine-tune curve pending", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.0, color="#c0392b", style="italic")
ax.set_xlim(10, 1000)
ax.set_xticks([X0, 25, 50, 100, 200, 400, 800])
ax.set_xticklabels(["0", "25", "50", "100", "200", "400", "800"])
ax.minorticks_off()
lo = min(min(zs.values()), min(c[3].min() for c in calib), min(a[3].min() for a in adapted))
ax.set_ylim(np.floor((lo - 0.01) * 50) / 50, 0.935)
ax.set_xlabel("ADD1000 maps used (log scale)", color=INK)
ax.set_ylabel("CC (fold 0)", color=INK)
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, handlelength=2.4,
          borderaxespad=0.3, labelspacing=0.55, title="solid: adapted\ndashed: calibration", title_fontsize=8.0,
          alignment="left")
fig.tight_layout(pad=0.3)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fig.savefig(args.out)
print(f"wrote {os.path.abspath(args.out)}; SUM curve {'included' if have_sum else 'ABSENT (placeholder note drawn)'}")
