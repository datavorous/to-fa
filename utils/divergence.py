"""
uv run python utils/divergence.py

Grouped bar chart: aggregate throughput + mean SLO compliance per config.
Saves to results/awq/<latest>/divergence.png
"""

import pathlib
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DM Sans"
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BG = "#fff5f7"
BG_PANEL = "#fdf0f4"
GRID_COL = "#f0cad8"
TEXT = "#1a0f14"
SUBTEXT = "#9e8490"
BORDER = "#f0cad8"

C_TPUT = "#1a0f14"
C_SLO = "#d94f7a"

CONFIGS = [
    "baseline",
    "prefix\ncache",
    "chunked\nprefill",
    "awq",
    "stack",
    "admission\ncontrol",
    "prefix +\nchunked",
]

THROUGHPUT = [144, 661, 251, 189, 197, 334, 664]
SLO_RATE = [11.25, 60.25, 20.75, 1.75, 6.5, 24.75, 60.5]

# AWQ and stack are indices 3 and 4
DIVERGE_START = 3
DIVERGE_END = 4


def main():
    out_dir = sorted(pathlib.Path("results/awq").iterdir())[-1]

    n = len(CONFIGS)
    x = np.arange(n)
    w = 0.32
    gap = 0.04

    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor(BG_PANEL)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines["left"].set_color(BORDER)
    ax1.spines["bottom"].set_color(BORDER)
    ax1.tick_params(length=0, labelsize=9.5, colors=SUBTEXT)
    ax1.grid(axis="y", color=GRID_COL, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.spines[["top", "left"]].set_visible(False)
    ax2.spines["right"].set_color(BORDER)
    ax2.spines["bottom"].set_color(BORDER)
    ax2.tick_params(length=0, labelsize=9.5, colors=C_SLO)

    xl = x - w / 2 - gap / 2
    xr = x + w / 2 + gap / 2

    # throughput bars (left axis)
    ax1.bar(xl, THROUGHPUT, width=w, color=C_TPUT, alpha=0.85, zorder=3)

    # SLO bars (right axis) — plot on ax1 coords via transform trick; use ax2 for scale
    # draw on ax1 with the SLO values scaled to ax1's range (0-700)
    slo_scaled = [s / 100 * 700 for s in SLO_RATE]
    ax1.bar(xr, slo_scaled, width=w, color=C_SLO, alpha=0.85, zorder=3)

    # throughput value labels
    for xi, v in zip(xl, THROUGHPUT):
        ax1.text(
            xi,
            v + 8,
            str(int(v)),
            ha="center",
            va="bottom",
            fontsize=8,
            color=C_TPUT,
            fontweight="bold",
        )

    # SLO value labels (using ax1 scaled positions, label shows real %)
    for xi, vs, vr in zip(xr, slo_scaled, SLO_RATE):
        ax1.text(
            xi,
            vs + 8,
            f"{vr:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color=C_SLO,
            fontweight="bold",
        )

    # divergence shading over awq + stack
    ax1.axvspan(
        DIVERGE_START - 0.48, DIVERGE_END + 0.48, color="#ff9eb5", alpha=0.15, zorder=0
    )
    ax1.text(
        (DIVERGE_START + DIVERGE_END) / 2,
        560,
        "throughput rises,\nSLO compliance does not",
        ha="center",
        va="top",
        fontsize=8.5,
        color=C_SLO,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=BG_PANEL,
            edgecolor=BORDER,
            linewidth=0.8,
        ),
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(CONFIGS, fontsize=10, color=TEXT)
    ax1.set_ylabel("throughput  (tok/s)", fontsize=10, labelpad=8, color=SUBTEXT)
    ax1.set_ylim(0, 700)

    ax2.set_ylabel("mean SLO compliance  (%)", fontsize=10, labelpad=8, color=C_SLO)
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    ax1.set_title(
        "throughput vs per-profile slo compliance, across configurations",
        fontsize=12,
        pad=14,
        color=TEXT,
    )

    from matplotlib.patches import Patch

    legend_items = [
        Patch(facecolor=C_TPUT, alpha=0.85, label="throughput (tok/s)"),
        Patch(facecolor=C_SLO, alpha=0.85, label="mean slo compliance (%)"),
    ]
    ax1.legend(
        handles=legend_items,
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
        edgecolor=BORDER,
        labelcolor=TEXT,
    )

    plt.tight_layout()
    out = out_dir / "divergence.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
