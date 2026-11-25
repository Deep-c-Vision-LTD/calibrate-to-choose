"""Shared style for the KBS figures (make_fig_curves.py, make_fig_cost.py).

One entity, one encoding, in both figures. Colours are slots 1 to 3 of the validated categorical palette
(colour-blind-safe palette; an all-pairs contrast check passes for these three) plus a neutral ink for the xSoyelNet
reference (frozen DINOv2-B readout);
every entity also has its own marker, so the figures read in grayscale and in print. In the learning-curve
figure the line style carries the level (solid: adapted; dashed, hollow markers: calibration only).
Revision (24 Nov 2026): figures are drawn at their placed width with every font at 8 pt or more.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = os.path.join(os.environ.get("CBYC_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))), "results")

INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
ENC = {  # entity: colour, marker, short label
    "msdb": dict(color="#2a78d6", marker="s", label="DeepGaze MSDB"),
    "sum": dict(color="#eb6834", marker="^", label="SUM"),
    "iie": dict(color="#1baf7a", marker="D", label="DeepGaze IIE"),
    "readout": dict(color="#3a3936", marker="o", label="xSoyelNet"),
}


def setup(width_in=5.3, height_in=1.75):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0, "legend.fontsize": 8.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, which="both")
    return fig, ax
