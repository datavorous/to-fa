"""
uv run summary-table [run_dir]

Prints a clean per-profile summary table from summary.json to stdout.
"""

import json
import pathlib
import sys


def _bar(val, max_val, width=20, fill="█", empty="░"):
    filled = int(round(val / max_val * width)) if max_val else 0
    return fill * filled + empty * (width - filled)


def main():
    results_root = pathlib.Path("results/baseline")
    if len(sys.argv) > 1:
        run_dir = pathlib.Path(sys.argv[1])
    else:
        run_dir = sorted(results_root.iterdir())[-1]

    s = json.loads((run_dir / "summary.json").read_text())
    bp = s["by_profile"]

    print(f"\n{'─'*72}")
    print(f"  run      : {s['run_id']}")
    print(f"  model    : {s['model']}")
    print(
        f"  requests : {s['n_ok']}/{s['n_requests']} ok  "
        f"({s['n_error']} errors, {100*s['n_error']/s['n_requests']:.0f}% abort rate)"
    )
    print(
        f"  wall     : {s['wall_s']:.1f}s   throughput: {s['throughput_tok_s']:.0f} tok/s"
    )
    print(
        f"  TTFT     : p50={s['ttft_p50_s']:.2f}s  p95={s['ttft_p95_s']:.2f}s  p99={s['ttft_p99_s']:.2f}s"
    )
    print(
        f"  E2E      : p50={s['e2e_p50_s']:.2f}s  p95={s['e2e_p95_s']:.2f}s  p99={s['e2e_p99_s']:.2f}s"
    )
    sys_s = s.get("system", {})
    if sys_s:
        print(
            f"  KV cache : mean={sys_s.get('kv_cache_usage_mean',0):.1%}  peak={sys_s.get('kv_cache_usage_peak',0):.1%}"
        )
        print(f"  prefix   : hit_rate={sys_s.get('prefix_cache_hit_rate',0):.1%}")
        print(
            f"  queue    : running_mean={sys_s.get('requests_running_mean',0):.1f}  "
            f"waiting_max={sys_s.get('requests_waiting_max',0)}"
        )
    print(f"{'─'*72}\n")

    HDR = f"  {'profile':<8} {'n':>5}  {'TTFT p50':>9}  {'TTFT p95':>9}  {'E2E p50':>8}  {'ITL p50':>8}  {'tok/s':>6}  {'budget':>7}  {'out_mean':>8}"
    print(HDR)
    print(f"  {'─'*66}")

    max_budget = max((bp[p]["budget_utilization_mean"] for p in bp), default=1)

    for p in ("siso", "silo", "liso", "lilo"):
        if p not in bp:
            continue
        d = bp[p]
        budget = d["budget_utilization_mean"]
        bar = _bar(budget, 1.0, width=8)
        print(
            f"  {p.upper():<8} {d['n']:>5}"
            f"  {d['ttft_p50_s']:>8.2f}s"
            f"  {d['ttft_p95_s']:>8.2f}s"
            f"  {d['e2e_p50_s']:>7.2f}s"
            f"  {d['itl_p50_ms']:>6.1f}ms"
            f"  {d['tokens_per_sec_mean']:>5.0f}/s"
            f"  {bar} {budget:.0%}"
            f"  {d['completion_tokens_mean']:>6.0f}tok"
        )
    print(f"\n  {'─'*66}")
    print(f"  budget bar: {'█'*8} = 100% of max_tokens used (truncated)\n")


if __name__ == "__main__":
    main()
