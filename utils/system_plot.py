"""
uv run system-plot [run_dir]

Three-panel system health timeline from system.jsonl:
  top    — KV cache usage % over wall time
  middle — requests running vs waiting
  bottom — per-interval prefix cache hit rate
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

BG = "#f7fafa"
PANEL_BG = "#f0f7f7"
TEAL = "#4a9e9a"
TEAL_MID = "#7fbfb8"
TEAL_LT = "#b8ddd4"
WARN = "#f4a261"
DANGER = "#c47ec4"


def main():
    results_root = pathlib.Path("results/baseline")
    if len(sys.argv) > 1:
        run_dir = pathlib.Path(sys.argv[1])
    else:
        run_dir = sorted(results_root.iterdir())[-1]

    snaps = [
        json.loads(l) for l in (run_dir / "system.jsonl").read_text().splitlines() if l
    ]
    print(f"loaded {len(snaps)} snapshots from {run_dir.name}")

    t0 = snaps[0]["t"]
    ts = np.array([(s["t"] - t0) for s in snaps])
    kv = np.array([s["kv_cache_usage"] * 100 for s in snaps])
    run = np.array([s["requests_running"] for s in snaps])
    wait = np.array([s["requests_waiting"] for s in snaps])

    # per-interval prefix hit rate
    q = np.array([s["prefix_cache_queries"] for s in snaps])
    h = np.array([s["prefix_cache_hits"] for s in snaps])
    dq = np.diff(q)
    dh = np.diff(h)
    with np.errstate(invalid="ignore", divide="ignore"):
        phr = np.where(dq > 0, dh / dq * 100, np.nan)
    ts_mid = (ts[:-1] + ts[1:]) / 2

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"System health timeline  ::  {run_dir.name}",
        fontsize=13,
        y=1.01,
    )

    # --- KV cache ---
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    ax.fill_between(ts, kv, alpha=0.35, color=TEAL)
    ax.plot(ts, kv, color=TEAL, linewidth=1.5)
    ax.axhline(
        100,
        color=DANGER,
        linewidth=1,
        linestyle="--",
        alpha=0.7,
        label="100% (eviction)",
    )
    ax.axhline(
        90, color=WARN, linewidth=0.8, linestyle=":", alpha=0.6, label="90% threshold"
    )
    ax.set_ylabel("KV cache %", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, framealpha=0.85, edgecolor="#cccccc", loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    ax.grid(axis="y", color="white", linewidth=0.8)

    # --- running / waiting ---
    ax2 = axes[1]
    ax2.set_facecolor(PANEL_BG)
    ax2.fill_between(ts, wait, alpha=0.25, color=WARN, label="waiting")
    ax2.plot(ts, wait, color=WARN, linewidth=1.5)
    ax2.fill_between(ts, run, alpha=0.4, color=TEAL, label="running")
    ax2.plot(ts, run, color=TEAL, linewidth=1.5)
    ax2.set_ylabel("request count", fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.85, edgecolor="#cccccc", loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(length=0, labelsize=8)
    ax2.grid(axis="y", color="white", linewidth=0.8)

    # --- prefix hit rate ---
    ax3 = axes[2]
    ax3.set_facecolor(PANEL_BG)
    ax3.fill_between(ts_mid, np.nan_to_num(phr), alpha=0.3, color=TEAL_MID)
    ax3.plot(ts_mid, phr, color=TEAL_MID, linewidth=1.2)
    ax3.set_ylabel("prefix hit rate %", fontsize=10)
    ax3.set_ylim(0, 105)
    ax3.set_xlabel("wall time (s)", fontsize=10, labelpad=6)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.tick_params(length=0, labelsize=8)
    ax3.grid(axis="y", color="white", linewidth=0.8)

    plt.tight_layout()
    out = run_dir / "system_timeline.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
