"""
uv run heatmap [results/baseline/RUN_DIR]

  heatmap_tokens.png  — 2D histogram: input vs output tokens, completed requests
  heatmap_fate.png    — stacked bar: completed vs failed by input token bucket, per profile
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DM Sans"
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np

BG = "#fff5f7"  # --bg-from
BG_PANEL = "#fdf0f4"  # --code-bg
GRID_COL = "#f0cad8"  # --border
TEXT = "#1a0f14"  # --text
SUBTEXT = "#9e8490"  # --muted

# --bg-from → --select-bg → deep rose
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rose",
    ["#fff5f7", "#fce4ed", "#ff9eb5", "#d94f7a", "#7a1035"],
    N=256,
)

OK_COLOR = "#ff9eb5"  # --select-bg
FAIL_COLOR = "#7a1035"  # deep rose

PROFILE_COLORS = {
    "siso": "#1a0f14",  # --text (near-black)
    "silo": "#d94f7a",  # mid rose
    "liso": "#ff9eb5",  # --select-bg
    "lilo": "#9e8490",  # --muted
}

PROFILE_META = {
    "siso": {"label": "siso", "desc": "short in / short out"},
    "silo": {"label": "silo", "desc": "short in / long out"},
    "liso": {"label": "liso", "desc": "long in / short out"},
    "lilo": {"label": "lilo", "desc": "long in / long out"},
}


def _run_dir():
    if len(sys.argv) > 1:
        return pathlib.Path(sys.argv[1])
    return sorted(pathlib.Path("results/baseline").iterdir())[-1]


def _load(run_dir):
    return [
        json.loads(l)
        for l in (run_dir / "requests.jsonl").read_text().splitlines()
        if l
    ]


# ── plot 1: 2D histogram ──────────────────────────────────────────────────────
def _plot_kde(recs, run_dir):
    ok = [r for r in recs if r["status"] == "ok"]
    if not ok:
        print("no ok records — skipping heatmap_tokens.png")
        return

    x = np.array([r["prompt_tokens"] for r in ok], dtype=float)
    y = np.array([r["completion_tokens"] for r in ok], dtype=float)

    x_bins = np.logspace(np.log10(max(x.min(), 1)), np.log10(x.max() + 1), 40)
    y_bins = np.logspace(np.log10(max(y.min(), 1)), np.log10(y.max() + 1), 40)
    H, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_PANEL)

    im = ax.pcolormesh(xedges, yedges, H.T, cmap=CMAP, shading="auto")
    ax.set_xscale("log")
    ax.set_yscale("log")

    for axis, vals in [
        (ax.xaxis, [30, 50, 100, 200, 500, 1000, 2000, 5000]),
        (ax.yaxis, [5, 20, 50, 100, 300, 500, 1000, 2000, 3000]),
    ]:
        axis.set_major_formatter(mticker.NullFormatter())
        axis.set_minor_formatter(mticker.NullFormatter())
        ticks = (
            [v for v in vals if x.min() <= v <= x.max() * 1.1]
            if axis is ax.xaxis
            else [v for v in vals if y.min() <= v <= y.max() * 1.1]
        )
        axis.set_major_locator(mticker.FixedLocator(ticks))
        axis.set_major_formatter(mticker.FixedFormatter([str(v) for v in ticks]))

    ax.tick_params(axis="both", length=0, labelsize=9, colors=SUBTEXT)
    ax.grid(True, which="major", color=GRID_COL, linewidth=0.6, zorder=0)

    for p, meta in PROFILE_META.items():
        pr = [r for r in ok if r["profile"] == p]
        if not pr:
            continue
        mx = np.median([r["prompt_tokens"] for r in pr])
        my = np.median([r["completion_tokens"] for r in pr])
        ax.text(
            mx,
            my,
            meta["label"],
            color="white",
            fontsize=10,
            fontweight="bold",
            fontfamily="monospace",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=PROFILE_COLORS[p],
                alpha=0.85,
                edgecolor="none",
            ),
        )

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.025)
    cbar.set_label("requests per cell", fontsize=9, color=SUBTEXT)  # already lowercase
    cbar.ax.tick_params(labelsize=8, length=0, colors=SUBTEXT)
    cbar.outline.set_visible(False)

    ax.set_xlabel("input tokens", fontsize=11, labelpad=8, color=TEXT)
    ax.set_ylabel("output tokens", fontsize=11, labelpad=8, color=TEXT)

    n_total = len(recs)
    n_ok = len(ok)
    ax.set_title(
        f"token distribution  ·  {run_dir.name}  ({n_ok}/{n_total} completed)",
        fontsize=11,
        pad=10,
        color=TEXT,
    )
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    plt.tight_layout()
    out = run_dir / "heatmap_tokens.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"saved → {out}")


# ── plot 2: fate by input token bucket ───────────────────────────────────────
def _plot_fate(recs, run_dir):
    try:
        import tiktoken
        from factory.config import load
        from factory.workload import generate
    except ImportError as e:
        print(f"skipping heatmap_fate.png — {e}")
        return

    summary_path = run_dir / "summary.json"
    exp = "baseline"
    if summary_path.exists():
        exp = json.loads(summary_path.read_text()).get("experiment", "baseline")

    CFG = load(exp)
    requests = generate(CFG)
    enc = tiktoken.get_encoding("o200k_base")
    tok_map = {
        r.id: len(enc.encode(r.system)) + len(enc.encode(r.user)) for r in requests
    }

    profiles = ["siso", "silo", "liso", "lilo"]
    fig, axes = plt.subplots(len(profiles), 1, figsize=(12, 10), sharex=False)
    fig.patch.set_facecolor(BG)

    fig.text(
        0.5,
        0.998,
        f"request fate by input token length  ·  {run_dir.name}",
        ha="center",
        va="top",
        fontsize=12,
        color=TEXT,
    )

    for ax, p in zip(axes, profiles):
        meta = PROFILE_META[p]
        ax.set_facecolor(BG_PANEL)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(length=0, labelsize=9, colors=SUBTEXT, labelcolor=SUBTEXT)

        pr = [r for r in recs if r["profile"] == p]
        if not pr:
            continue

        tok_vals = [tok_map.get(r["id"], 0) for r in pr]
        ok_toks = [t for r, t in zip(pr, tok_vals) if r["status"] == "ok"]
        fail_toks = [t for r, t in zip(pr, tok_vals) if r["status"] != "ok"]

        lo, hi = min(tok_vals), max(tok_vals)
        bins = np.linspace(lo, hi + 1, 10)  # 9 bins — tighter spacing
        centres = (bins[:-1] + bins[1:]) / 2
        width = (bins[1] - bins[0]) * 0.38

        ok_counts, _ = np.histogram(ok_toks, bins=bins)
        fail_counts, _ = np.histogram(fail_toks, bins=bins)
        total_counts = ok_counts + fail_counts

        pc = PROFILE_COLORS[p]
        # lighten for ok, use full colour for fail
        ok_hex = OK_COLOR
        fail_hex = FAIL_COLOR

        ax.bar(
            centres,
            ok_counts,
            width=width,
            color=ok_hex,
            alpha=0.9,
            zorder=2,
            label=f"completed ({len(ok_toks)})",
        )
        ax.bar(
            centres,
            fail_counts,
            width=width,
            color=fail_hex,
            alpha=0.85,
            bottom=ok_counts,
            zorder=2,
            label=f"failed ({len(fail_toks)})",
        )

        ymax = max(total_counts.max(), 1)
        ax.set_ylim(0, ymax * 1.25)

        for cx, ok_c, fail_c, tot in zip(centres, ok_counts, fail_counts, total_counts):
            if tot == 0:
                continue
            rate = fail_c / tot * 100
            ax.text(
                cx,
                ok_c + fail_c + ymax * 0.025,
                f"{rate:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=SUBTEXT,
            )

        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, integer=True))
        ax.set_ylabel("n", fontsize=8, color=SUBTEXT, labelpad=4)
        ax.grid(axis="y", color=GRID_COL, linewidth=0.6, zorder=0)

        tick_vals = np.linspace(lo, hi, 4).astype(int)
        ax.set_xticks(tick_vals)
        ax.set_xticklabels(tick_vals, fontsize=8, color=SUBTEXT)
        ax.set_xlim(lo - width, hi + width * 2)

        fail_rate_overall = len(fail_toks) / len(pr) * 100
        ax.text(
            0.012,
            0.88,
            f"{meta['label'].lower()}  ·  {fail_rate_overall:.0f}% fail",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color="white",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=pc, alpha=0.85, edgecolor="none"
            ),
        )

        ax.legend(
            loc="upper right",
            fontsize=8,
            framealpha=0,
            edgecolor="none",
            labelcolor=SUBTEXT,
        )

    axes[-1].set_xlabel(
        "input tokens  (system + user)", fontsize=10, labelpad=8, color=TEXT
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.5)
    out = run_dir / "heatmap_fate.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"saved → {out}")


def main():
    run_dir = _run_dir()
    recs = _load(run_dir)
    print(f"loaded {len(recs)} records from {run_dir.name}")
    _plot_kde(recs, run_dir)
    _plot_fate(recs, run_dir)


if __name__ == "__main__":
    main()
