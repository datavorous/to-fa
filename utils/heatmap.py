"""
uv run heatmap [results/baseline/RUN_DIR]

Plots a smoothed KDE heatmap of prompt vs completion tokens, one blob per
workload profile (siso / silo / liso / lilo). If no path is given, uses the
latest run under results/baseline/.
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import gaussian_kde


def main():
    results_root = pathlib.Path("results/baseline")
    if len(sys.argv) > 1:
        run_dir = pathlib.Path(sys.argv[1])
    else:
        run_dir = sorted(results_root.iterdir())[-1]

    recs = [
        json.loads(l)
        for l in (run_dir / "requests.jsonl").read_text().splitlines()
        if l and json.loads(l)["status"] == "ok"
    ]
    print(f"loaded {len(recs)} records from {run_dir.name}")

    prompt_tokens = np.array([r["prompt_tokens"] for r in recs], dtype=float)
    completion_tokens = np.array([r["completion_tokens"] for r in recs], dtype=float)

    # KDE in log-space keeps clusters sharp and avoids bleeding across decades
    lx = np.log1p(prompt_tokens)
    ly = np.log1p(completion_tokens)

    GRID = 400
    xi = np.linspace(lx.min() - 0.2, lx.max() + 0.2, GRID)
    yi = np.linspace(ly.min() - 0.2, ly.max() + 0.2, GRID)
    Xi, Yi = np.meshgrid(xi, yi)

    kde = gaussian_kde(np.vstack([lx, ly]), bw_method=0.12)
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(GRID, GRID)

    pastel_cmap = mcolors.LinearSegmentedColormap.from_list(
        "pastel_teal",
        ["#f7fafa", "#d4eceb", "#96ceca", "#55aca8", "#277a78", "#134f4e"],
        N=512,
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7fafa")

    ax.contourf(Xi, Yi, Zi, levels=60, cmap=pastel_cmap)
    ax.contour(Xi, Yi, Zi, levels=12, colors="white", linewidths=0.3, alpha=0.5)

    # label each cluster at its median position
    for p in ("siso", "silo", "liso", "lilo"):
        pr = [r for r in recs if r["profile"] == p]
        if not pr:
            continue
        mx = np.log1p(np.median([r["prompt_tokens"] for r in pr]))
        my = np.log1p(np.median([r["completion_tokens"] for r in pr]))
        ax.text(
            mx,
            my,
            p.upper(),
            color="white",
            alpha=0.92,
            fontsize=13,
            fontweight="bold",
            fontfamily="monospace",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#134f4e",
                alpha=0.4,
                edgecolor="none",
            ),
        )

    # axes: real token values on log-space grid
    def ticks(vals):
        return [(np.log1p(v), str(v)) for v in vals]

    xt = ticks([50, 100, 150, 250, 500, 1000, 2000, 4000])
    yt = ticks([10, 50, 100, 250, 500, 1000, 1500, 2000, 3000])
    ax.set_xticks([t for t, _ in xt])
    ax.set_xticklabels([l for _, l in xt], fontsize=9)
    ax.set_yticks([t for t, _ in yt])
    ax.set_yticklabels([l for _, l in yt], fontsize=9)

    ax.set_xlabel("Input tokens (prompt)", fontsize=11, labelpad=8)
    ax.set_ylabel("Output tokens (completion)", fontsize=11, labelpad=8)
    ax.set_title(
        f"Request token distribution  ::  {run_dir.name}  (n={len(recs)})",
        fontsize=12,
        pad=14,
    )

    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)

    sm = plt.cm.ScalarMappable(
        cmap=pastel_cmap,
        norm=mcolors.Normalize(vmin=Zi.min(), vmax=Zi.max()),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("density", fontsize=9)
    cbar.ax.tick_params(labelsize=8, length=0)
    cbar.set_ticks([])

    plt.tight_layout()
    out = run_dir / "heatmap_tokens.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
