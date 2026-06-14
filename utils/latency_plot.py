"""
uv run latency-plot [run_dir]

Two-panel latency breakdown:
  left  — TTFT: p50 bar + extension to p95 + dot at p99, per profile
  right — E2E:  same layout
Each row is one profile. The bar shows where 50% of users land;
the thin extension shows the p95 tail; the dot is p99.
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DM Sans"
import matplotlib.pyplot as plt
import numpy as np

PROFILES = ["siso", "silo", "liso", "lilo"]
COLORS = {"siso": "#1a0f14", "silo": "#d94f7a", "liso": "#ff9eb5", "lilo": "#9e8490"}
BG = "#fff5f7"
PANEL_BG = "#fdf0f4"


def _pct(vals, p):
    return float(np.percentile(vals, p))


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
    print(f"loaded {len(recs)} ok records from {run_dir.name}")

    # collect stats per profile
    stats = {}
    for p in PROFILES:
        vals_ttft = [r["ttft_s"] for r in recs if r["profile"] == p]
        vals_e2e = [r["total_s"] for r in recs if r["profile"] == p]
        if not vals_ttft:
            continue
        stats[p] = {
            "n": len(vals_ttft),
            "ttft": (_pct(vals_ttft, 50), _pct(vals_ttft, 95), _pct(vals_ttft, 99)),
            "e2e": (_pct(vals_e2e, 50), _pct(vals_e2e, 95), _pct(vals_e2e, 99)),
        }

    profiles_present = [p for p in PROFILES if p in stats]
    y = np.arange(len(profiles_present))
    height = 0.28

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"latency breakdown  ::  {run_dir.name}  (n={len(recs)} ok requests)",
        fontsize=12,
        y=1.02,
    )

    for ax, key, title in [
        (ax1, "ttft", "time to first token (s)"),
        (ax2, "e2e", "end-to-end latency (s)"),
    ]:
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, fontsize=10, pad=8)

        # draw bars first, collect label positions after
        label_items = []
        for i, p in enumerate(profiles_present):
            p50, p95, p99 = stats[p][key]
            color = COLORS[p]

            # solid bar: 0 → p50
            ax.barh(i, p50, height=height, color=color, alpha=0.9, zorder=3)

            # thin extension: p50 → p95
            ax.barh(
                i,
                p95 - p50,
                left=p50,
                height=height * 0.35,
                color=color,
                alpha=0.45,
                zorder=3,
            )

            # p99 dot
            ax.scatter(
                p99, i, s=60, color=color, zorder=5, edgecolors="white", linewidths=0.8
            )

            # p50 label inside bar
            ax.text(
                p50 * 0.5,
                i,
                f"{p50:.0f}s",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
                fontweight="bold",
                zorder=6,
            )

            label_items.append((i, p99, p95, color))

        # p95 labels placed after all drawing so xlim is final
        x_max = ax.get_xlim()[1]
        offset = x_max * 0.015
        for i, p99, p95, color in label_items:
            ax.text(
                p99 + offset,
                i,
                f"p95={p95:.0f}s  p99={p99:.0f}s",
                ha="left",
                va="center",
                fontsize=7,
                color="#555555",
                zorder=6,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{p.lower()}  (n={stats[p]['n']})" for p in profiles_present], fontsize=9
        )
        ax.set_xlabel("seconds", fontsize=9, labelpad=6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0, labelsize=8)
        ax.grid(axis="x", color="#f0cad8", linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)

    # legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_items = [
        Patch(facecolor="#888888", alpha=0.9, label="p50 (median)"),
        Patch(facecolor="#888888", alpha=0.4, label="p50  p95 tail"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#888888",
            markersize=7,
            label="p99",
        ),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=3,
        fontsize=8,
        framealpha=0.85,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    out = run_dir / "latency_cdf.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
