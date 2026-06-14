import asyncio
import sys
import time

import httpx

from .config import load
from .metrics import store, summarise
from .runner import run
from .system import aggregate
from .workload import generate


def progress(i, total, r):
    status = "ok" if r.status == "ok" else f"ERR {r.error[:50]}"
    print(
        f"[{i:>3}/{total}] {r.id:<18} ttft={r.ttft_s:.2f}s e2e={r.total_s:.2f}s {r.completion_tokens}tok {r.tokens_per_sec:.0f}tok/s {status}",
        flush=True,
    )


def print_summary(s):
    print(f"--- {s.experiment} / {s.run_id} ---")
    print(f"model {s.model}  concurrency {s.concurrency}  wall {s.wall_s:.1f}s")
    print(
        f"requests {s.n_ok}/{s.n_requests} ok  throughput {s.throughput_tok_s:.0f}tok/s"
    )
    print(
        f"tokens {s.total_prompt_tokens} prompt / {s.total_completion_tokens} completion"
    )
    print(
        f"TTFT p50={s.ttft_p50_s:.3f}s p95={s.ttft_p95_s:.3f}s p99={s.ttft_p99_s:.3f}s"
    )
    print(f"E2E  p50={s.e2e_p50_s:.3f}s p95={s.e2e_p95_s:.3f}s p99={s.e2e_p99_s:.3f}s")

    if s.error_types:
        print(f"errors {s.error_types}")

    for profile, m in s.by_profile.items():
        print(
            f"[{profile}] n={m['n']} prompt_mean={m['prompt_tokens_mean']:.0f}tok budget_util={m['budget_utilization_mean']:.0%}"
        )
        print(
            f"  TTFT p50={m['ttft_p50_s']:.3f}s p95={m['ttft_p95_s']:.3f}s p99={m['ttft_p99_s']:.3f}s"
        )
        print(
            f"  E2E  p50={m['e2e_p50_s']:.3f}s p95={m['e2e_p95_s']:.3f}s p99={m['e2e_p99_s']:.3f}s"
        )
        print(f"  ITL  p50={m['itl_p50_ms']:.1f}ms p95={m['itl_p95_ms']:.1f}ms")
        print(
            f"  output p50={m['completion_tokens_p50']:.0f} p95={m['completion_tokens_p95']:.0f} mean={m['completion_tokens_mean']:.0f}tok"
        )
        print(
            f"  tok/s mean={m['tokens_per_sec_mean']:.0f} p50={m['tokens_per_sec_p50']:.0f} p95={m['tokens_per_sec_p95']:.0f}"
        )
        for b, bm in m["by_length"].items():
            print(
                f"  [{b}] n={bm['n']} ttft={bm['ttft_p50_s']:.3f}s e2e={bm['e2e_p50_s']:.3f}s {bm['tokens_per_sec_mean']:.0f}tok/s mean={bm['completion_tokens_mean']:.0f}tok prompt={bm['prompt_tokens_mean']:.0f}tok budget={bm['budget_utilization_mean']:.0%}"
            )

    if s.system:
        sys_s = s.system
        print(
            f"[system] kv_cache mean={sys_s.get('kv_cache_usage_mean')} peak={sys_s.get('kv_cache_usage_peak')}  prefix_hit_rate={sys_s.get('prefix_cache_hit_rate')}"
        )
        print(
            f"[system] requests_running_mean={sys_s.get('requests_running_mean')}  waiting_max={sys_s.get('requests_waiting_max')}"
        )


def _drain(base_url, timeout_s=120):
    metrics_url = base_url.replace("/v1", "") + "/metrics"
    deadline = time.time() + timeout_s
    print("draining server...", end="", flush=True)
    while time.time() < deadline:
        try:
            r = httpx.get(metrics_url, timeout=5.0)
            running = waiting = 0
            for line in r.text.splitlines():
                if line.startswith("#"):
                    continue
                if "vllm:num_requests_running{" in line:
                    running = int(float(line.split()[-1]))
                elif "vllm:num_requests_waiting{" in line:
                    waiting = int(float(line.split()[-1]))
            if running == 0 and waiting == 0:
                print(" clear")
                return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    print(" timeout, proceeding anyway")


def _run_once(requests, CFG):
    t0 = time.perf_counter()
    results, snapshots = asyncio.run(run(requests, CFG, on_result=progress))
    wall_s = time.perf_counter() - t0

    system = aggregate(snapshots)
    summary = summarise(results, wall_s, CFG, system=system)
    out_dir = store(results, summary, snapshots, CFG)

    print_summary(summary)
    print(f"results -> {out_dir}")


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else None
    CFG = load(exp)

    print(f"experiment={CFG.exp} model={CFG.model} load_mode={CFG.load_mode}")
    print(
        f"siso={CFG.siso_n}req {CFG.siso_max_tokens[0]}-{CFG.siso_max_tokens[1]}tok  silo={CFG.silo_n}req {CFG.silo_max_tokens[0]}-{CFG.silo_max_tokens[1]}tok"
    )

    requests = generate(CFG)
    print(f"generated {len(requests)} requests")

    if CFG.load_mode == "sweep":
        print(
            f"sweep {CFG.sweep_param} over {CFG.sweep_values}  (base_mode={CFG.sweep_base_mode})"
        )
        CFG.load_mode = CFG.sweep_base_mode
        for i, val in enumerate(CFG.sweep_values):
            setattr(CFG, CFG.sweep_param, val)
            print(f"\n=== {CFG.sweep_param}={val} ===")
            _run_once(requests, CFG)
            if i < len(CFG.sweep_values) - 1:
                _drain(CFG.base_url)
    else:
        _run_once(requests, CFG)


if __name__ == "__main__":
    main()
