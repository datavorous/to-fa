import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .runner import Result


@dataclass
class Summary:
    run_id: str
    experiment: str
    timestamp: str
    model: str
    concurrency: int
    n_requests: int
    n_ok: int
    n_error: int
    error_types: dict
    wall_s: float
    throughput_tok_s: float
    total_prompt_tokens: int
    total_completion_tokens: int
    ttft_p50_s: float
    ttft_p95_s: float
    ttft_p99_s: float
    e2e_p50_s: float
    e2e_p95_s: float
    e2e_p99_s: float
    by_profile: dict = field(default_factory=dict)
    system: dict = field(default_factory=dict)


def _pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return round(s[min(int(len(s) * p / 100), len(s) - 1)], 4)


def _profile_stats(results, profile):
    pr = [r for r in results if r.profile == profile]
    tok = [r.completion_tokens for r in pr]

    by_length = {}
    for b in ("short", "medium", "long"):
        br = [r for r in pr if r.length == b]
        if not br:
            continue
        by_length[b] = {
            "n": len(br),
            "ttft_p50_s": _pct([r.ttft_s for r in br], 50),
            "e2e_p50_s": _pct([r.total_s for r in br], 50),
            "tokens_per_sec_mean": round(
                statistics.mean(r.tokens_per_sec for r in br), 2
            ),
            "completion_tokens_mean": round(
                statistics.mean(r.completion_tokens for r in br), 1
            ),
            "prompt_tokens_mean": round(
                statistics.mean(r.prompt_tokens for r in br), 1
            ),
            "budget_utilization_mean": round(
                statistics.mean(r.completion_tokens / r.max_tokens for r in br), 3
            ),
        }

    return {
        "n": len(pr),
        "ttft_p50_s": _pct([r.ttft_s for r in pr], 50),
        "ttft_p95_s": _pct([r.ttft_s for r in pr], 95),
        "ttft_p99_s": _pct([r.ttft_s for r in pr], 99),
        "e2e_p50_s": _pct([r.total_s for r in pr], 50),
        "e2e_p95_s": _pct([r.total_s for r in pr], 95),
        "e2e_p99_s": _pct([r.total_s for r in pr], 99),
        "itl_p50_ms": _pct([r.itl_ms for r in pr], 50),
        "itl_p95_ms": _pct([r.itl_ms for r in pr], 95),
        "tokens_per_sec_mean": round(statistics.mean(r.tokens_per_sec for r in pr), 2),
        "tokens_per_sec_p50": _pct([r.tokens_per_sec for r in pr], 50),
        "tokens_per_sec_p95": _pct([r.tokens_per_sec for r in pr], 95),
        "prompt_tokens_mean": round(statistics.mean(r.prompt_tokens for r in pr), 1),
        "completion_tokens_mean": round(statistics.mean(tok), 1),
        "completion_tokens_p50": _pct(tok, 50),
        "completion_tokens_p95": _pct(tok, 95),
        "budget_utilization_mean": round(
            statistics.mean(r.completion_tokens / r.max_tokens for r in pr), 3
        ),
        "by_length": by_length,
    }


def summarise(results, wall_s, CFG, system=None):
    ok = [r for r in results if r.status == "ok"]
    errors = [r for r in results if r.status == "error"]
    total_completion = sum(r.completion_tokens for r in ok)
    now = datetime.now(timezone.utc)

    return Summary(
        run_id=now.strftime("%Y%m%dT%H%M%SZ"),
        experiment=CFG.exp,
        timestamp=now.isoformat(),
        model=CFG.model,
        concurrency=CFG.concurrency,
        n_requests=len(results),
        n_ok=len(ok),
        n_error=len(errors),
        error_types=dict(Counter(e.error.split()[0] for e in errors if e.error)),
        wall_s=round(wall_s, 2),
        throughput_tok_s=round(total_completion / wall_s, 2),
        total_prompt_tokens=sum(r.prompt_tokens for r in ok),
        total_completion_tokens=total_completion,
        ttft_p50_s=_pct([r.ttft_s for r in ok], 50),
        ttft_p95_s=_pct([r.ttft_s for r in ok], 95),
        ttft_p99_s=_pct([r.ttft_s for r in ok], 99),
        e2e_p50_s=_pct([r.total_s for r in ok], 50),
        e2e_p95_s=_pct([r.total_s for r in ok], 95),
        e2e_p99_s=_pct([r.total_s for r in ok], 99),
        by_profile={
            p: _profile_stats(ok, p)
            for p in ("siso", "silo", "liso", "lilo")
            if any(r.profile == p for r in ok)
        },
        system=system or {},
    )


def store(results, summary, snapshots, CFG):
    import os
    from .system import store_system

    out_dir = f"{CFG.results_dir}/{summary.experiment}/{summary.run_id}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/requests.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(asdict(summary), f, indent=2)

    store_system(snapshots, out_dir)

    return out_dir
