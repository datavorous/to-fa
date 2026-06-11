"""
uv run workload-timeline [run_dir]

Two-panel chart:
  top    — stacked bar: count of each profile per completion window
  bottom — stacked area: % share of each profile, smoothed over windows
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROFILES = ["siso", "silo", "liso", "lilo"]
COLORS = {"siso": "#7ec8c8", "silo": "#f4a261", "liso": "#a8d5a2", "lilo": "#c9a0dc"}
N_BINS = 40


def main():
    results_root = pathlib.Path("results/baseline")
    if len(sys.argv) > 1:
        run_dir = pathlib.Path(sys.argv[1])
    else:
        run_dir = sorted(results_root.iterdir())[-1]

    recs = [
        json.loads(l)
        for l in (run_dir / "requests.jsonl").read_text().splitlines()
        if l
    ]
    print(f"loaded {len(recs)} records from {run_dir.name}")

    # bin by completion order (only time proxy available — no absolute timestamps)
    bin_size = max(1, len(recs) // N_BINS)
    bins = [recs[i : i + bin_size] for i in range(0, len(recs), bin_size)]

    counts = {
        p: np.array([sum(1 for r in b if r["profile"] == p) for b in bins], dtype=float)
        for p in PROFILES
    }
    x = np.arange(len(bins))
    labels = [f"{i*100//len(bins)}%" for i in range(len(bins))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Workload profile distribution  ::  {run_dir.name}  (n={len(recs)})",
        fontsize=13,
        y=0.98,
    )

    # --- top: stacked bar ---
    ax1.set_facecolor("#f9f9f9")
    bottom = np.zeros(len(bins))
    for p in PROFILES:
        ax1.bar(
            x,
            counts[p],
            bottom=bottom,
            color=COLORS[p],
            label=p.upper(),
            width=0.85,
            linewidth=0,
        )
        bottom += counts[p]
    ax1.set_xticks(x[::4])
    ax1.set_xticklabels(labels[::4], fontsize=8)
    ax1.set_ylabel("requests completed in window", fontsize=10)
    ax1.set_title(
        "Count per window (completion order proxy for time)", fontsize=10, pad=6
    )
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(length=0)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.85, edgecolor="#cccccc")

    # --- bottom: smoothed % share ---
    ax2.set_facecolor("#f9f9f9")
    totals = sum(counts[p] for p in PROFILES)
    totals[totals == 0] = 1

    def smooth(arr, w=4):
        return np.convolve(arr, np.ones(w) / w, mode="same")

    bottom = np.zeros(len(bins))
    for p in PROFILES:
        share = smooth(counts[p] / totals * 100)
        ax2.fill_between(
            x, bottom, bottom + share, color=COLORS[p], alpha=0.85, label=p.upper()
        )
        bottom += share

    ax2.set_xticks(x[::4])
    ax2.set_xticklabels(labels[::4], fontsize=8)
    ax2.set_ylabel("% of completions in window", fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_title("Profile share over time (smoothed %)", fontsize=10, pad=6)
    ax2.set_xlabel(
        "run progress (% of total requests, by completion order)",
        fontsize=10,
        labelpad=6,
    )
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(length=0)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.85, edgecolor="#cccccc")

    total_ok = sum(1 for r in recs if r["status"] == "ok")
    total_err = len(recs) - total_ok
    fig.text(
        0.01,
        0.005,
        f"ok={total_ok}  aborted={total_err} ({100*total_err/len(recs):.0f}%)",
        fontsize=8,
        color="#888888",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = run_dir / "workload_timeline.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
