"""
uv run prefix-surprise [out_dir]

Grouped bar chart: TTFT p50 baseline vs prefix-cache-on, per profile.
Highlights the SISO anomaly — biggest gain despite no shared prefix.
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DM Sans"
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── rose palette ──────────────────────────────────────────────────────────────
BG = "#fff5f7"
BG_PANEL = "#fdf0f4"
GRID_COL = "#f0cad8"
TEXT = "#1a0f14"
SUBTEXT = "#9e8490"
BORDER = "#f0cad8"

C_BASELINE = "#9e8490"  # --muted  — baseline bars
C_CACHE = "#d94f7a"  # mid rose — cache-on bars
C_BL = "#9e8490"  # --muted  — all baseline bars
C_CA = "#d94f7a"  # mid rose — all cache-on bars

# narrative order: expected-benefit ascending left to right, SISO punchline at right
PROFILES = ["liso", "lilo", "silo", "siso"]
LABELS = ["LISO", "LILO", "SILO", "SISO"]

BASELINE = {"siso": 56.58, "silo": 35.50, "liso": 22.00, "lilo": 38.13}
CACHE_ON = {"siso": 4.67, "silo": 4.56, "liso": 4.40, "lilo": 3.92}
SPEEDUP = {"siso": 12.1, "silo": 7.8, "liso": 5.0, "lilo": 9.7}


def main():
    out_dir = (
        pathlib.Path(sys.argv[1])
        if len(sys.argv) > 1
        else sorted(pathlib.Path("results/prefix_on_default").iterdir())[-1]
    )

    x = np.arange(len(PROFILES))
    w = 0.30
    gap = 0.06

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_PANEL)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0, labelsize=10, colors=SUBTEXT)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for i, p in enumerate(PROFILES):
        bl = BASELINE[p]
        ca = CACHE_ON[p]
        sx = SPEEDUP[p]

        xl = x[i] - w / 2 - gap / 2
        xr = x[i] + w / 2 + gap / 2

        ax.bar(xl, bl, width=w, color=C_BL, alpha=0.88, zorder=3)
        ax.bar(
            xr,
            ca,
            width=w,
            color=C_CA,
            alpha=0.95,
            zorder=3,
            edgecolor=BORDER,
            linewidth=0.6,
        )

        # value labels inside bars
        for xpos, val in [(xl, bl), (xr, ca)]:
            ax.text(
                xpos,
                val * 0.5,
                f"{val:.0f}s" if val >= 5 else f"{val:.1f}s",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

        # speedup label centered above the pair
        ax.text(
            x[i],
            bl + 2.5,
            f"{sx}×",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=TEXT,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=11, color=TEXT)
    ax.set_ylabel("ttft p50  (seconds)", fontsize=10, labelpad=8, color=SUBTEXT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}s"))

    from matplotlib.patches import Patch

    legend_items = [
        Patch(facecolor=C_BL, alpha=0.88, label="baseline  (apc off)"),
        Patch(facecolor=C_CA, alpha=0.95, label="prefix cache on"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
        edgecolor=BORDER,
        labelcolor=TEXT,
    )

    ax.set_title(
        "TTFT p50 by profile, with speedup from prefix caching",
        fontsize=11,
        pad=12,
        color=TEXT,
    )

    plt.tight_layout()
    out = out_dir / "prefix_surprise.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
