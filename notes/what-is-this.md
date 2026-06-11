# to-fa: What This Is

A single-GPU inference benchmarking harness. You point it at a running vLLM server, it fires a mixed workload, and it tells you how the server behaved.

## Output Metrics

**Per request**
- `ttft_s`: time to first token
- `total_s`: end-to-end latency
- `itl_ms`: inter-token latency (mean time between tokens)
- `tokens_per_sec`: decode rate
- `prompt_tokens`: input tokens (as counted by the model)
- `completion_tokens`: output tokens generated
- `budget_utilization`: completion_tokens / max_tokens

**System (scraped every 2s from vLLM /metrics)**
- `kv_cache_usage`: fraction of KV cache pool in use
- `prefix_cache_hit_rate`: fraction of prefill work skipped via cache
- `requests_running`: active sequences in the batch
- `requests_waiting`: queued, not yet scheduled

**Summary (aggregated across the run)**
- p50 / p95 / p99 for TTFT and E2E, overall and per profile
- p50 / p95 for ITL and tok/s per profile
- per-length-bucket breakdown (short / medium / long)
- wall time, total throughput (tok/s), total prompt and completion tokens

## What You Can Modify

**Client-side (no server restart)**
- `concurrency`: max simultaneous in-flight requests
- `n`: number of requests per profile
- `max_tokens`: output token budget range [min, max]
- `temperature`: sampling temperature range [min, max]
- `seed`: for reproducibility
- `load_mode`: how requests are dispatched (see below)
- `rate_rps`: arrival rate for constant / poisson modes
- `sweep_param` / `sweep_values`: parameter to vary and values to iterate over
- `sweep_base_mode`: the dispatch mode each sweep step runs under

**Server-side (requires vLLM restart via sbatch)**
- `gpu_memory_utilization`: fraction of VRAM given to vLLM
- `max_model_len`: maximum context length
- `enable_prefix_caching`: toggle prefix caching on/off
- `extra_args`: any additional vLLM flags (e.g. `--enable-chunked-prefill`, `--kv-cache-dtype fp8`, `--speculative-model`)

## Load Modes

`load_mode` controls how and when requests are dispatched. It is the single most important variable for studying server behavior under different traffic shapes.

**synchronous**: one request at a time, next starts only after previous completes. Concurrency is always 1. Establishes the single-request latency floor with zero queuing.

**concurrent**: N requests in flight simultaneously (closed-loop). As one completes, the next starts immediately. The server is always under exactly N requests of pressure. This is the default and the most useful mode for finding the throughput-latency tradeoff.

**constant**: requests are dispatched at a fixed rate of `rate_rps` per second, regardless of how long they take. If the server is slower than the arrival rate, a queue builds. Models a steady stream of users.

**poisson**: same as constant but inter-arrival times are exponentially distributed with mean `1/rate_rps`. This is the standard model for real user traffic — most arrivals are close together, some are far apart. At the same mean rate, Poisson is harder on the server than constant because of bursts.

**sweep**: runs the workload multiple times, varying `sweep_param` over `sweep_values`. Each step uses `sweep_base_mode` as the underlying dispatch mode. Produces one result directory per step.

```yaml
# sweep concurrency from 1 to 32, using closed-loop dispatch
run:
  load_mode: sweep
  sweep_param: concurrency
  sweep_values: [1, 4, 8, 16, 32]
  sweep_base_mode: concurrent

# sweep arrival rate from 1 to 16 req/s, using Poisson arrivals
run:
  load_mode: sweep
  sweep_param: rate_rps
  sweep_values: [1, 2, 4, 8, 16]
  sweep_base_mode: poisson
```

## Running Experiments

### Client-side (no restart needed)

Add a named block under `experiments:` in `config.yaml`. Only specify what changes, everything else inherits from the top-level `run:` block.

```yaml
experiments:
  baseline:
    note: "default config"

  concurrency_16:
    note: "double concurrency"
    run:
      concurrency: 16

  high_budget:
    note: "larger output budgets"
    run:
      concurrency: 16
      code:
        n: 20
        max_tokens: [2048, 4096]
        temperature: [0.1, 0.4]
```

Run it:

```bash
uv run python -m factory.run concurrency_16
uv run python -m factory.run high_budget
```

Results land in `results/<experiment_name>/<run_id>/`.

### Server-side (restart required)

1. Edit `config.yaml`
2. Sync to cluster: `./sync.sh`
3. Cancel the running job: `scancel <jobid>`
4. Resubmit: `sbatch serve.slurm`
5. Wait for `>>> READY` in the job log
6. Re-establish tunnel: `ssh -NL 8000:node01:8000 <user>@turing.iiit.ac.in`
7. Run: `uv run python -m factory.run no_prefix_cache`

Example config for a server-side experiment:

```yaml
experiments:
  no_prefix_cache:
    note: "measure cost of cold prefill"
    vllm:
      enable_prefix_caching: false
    run:
      concurrency: 16
```

### Comparing runs

```bash
# one number across all runs
grep ttft_p95 results/*/*/summary.json
```





## how to read the plots

### `summary-table` — first thing to check after every run

```
requests : 337/725 ok  (388 errors, 54% abort rate)
TTFT     : p50=141s  p95=351s
KV cache : mean=87.8%  peak=100%
queue    : waiting_max=234
```

| metric | good | bad |
|---|---|---|
| abort rate | < 5% | > 20% — server dropping connections under load |
| TTFT p50 | < 5s for siso | > 30s — queue starvation |
| TTFT p95 / p50 ratio | < 3× | > 10× — bimodal cliff, SLO is lying |
| KV peak | < 90% | 100% — eviction cascade begins |
| waiting_max | < 20 | > 100 — no admission control |
| lilo budget p50 | < 0.8 | 1.0 — model being truncated mid-output |

---

### `heatmap_tokens.png` — where requests actually land

KDE density plot of prompt tokens (x) vs completion tokens (y). Four clusters should be visible:

- **SISO** bottom-left — short in, short out
- **SILO** bottom-right — short in, long out
- **LISO** top-left — long in, short out
- **LILO** top-right — long in, long out

**good:** four distinct tight clusters. **bad:** clusters bleeding into each other (prefix cache hit, or max_tokens truncation pulling lilo/liso down).

---

### `latency_cdf.png` — where latency pain lives

One row per profile. Left panel = TTFT, right panel = E2E.

- **solid bar** = p50 — where half your users land
- **thin extension** = p50→p95 tail
- **dot** = p99 outlier
- **label** = exact p95 and p99 values

**good:** short bars, thin tails, dot close to bar end. **improved:** bars shorten, tail shrinks relative to the bar. **bad:** tail is 10–100× longer than the bar (the p95/p99 cliff we see in baseline).

watch siso specifically — it should always have the shortest bar. if siso p50 TTFT > 10s, lilo is starving it.

---

### `system_timeline.png` — what the server was doing moment to moment

Three panels over wall time:

- **top:** KV cache % — should stay below 90%. once it hits 100% eviction begins and throughput drops non-linearly.
- **middle:** requests running (teal) vs waiting (orange) — a healthy server has waiting ≈ 0. a flooded server has waiting >> running.
- **bottom:** prefix cache hit rate per interval — high hit rate means requests share prompts (good for throughput, but can be artificial). after the preamble fix this should be low for liso/lilo.

**good:** KV stays under 90%, waiting stays near 0, hit rate is stable. **improved:** KV peak drops, waiting queue shrinks. **bad:** KV flatlined at 100% the whole run with waiting at 200+ (our baseline).

---

### `workload_timeline.png` — traffic mix over the run

Two panels (stacked bar + smoothed area) showing what share of completions belonged to each profile at each point in the run.

**what to expect:** siso completes fast and dominates early completions. lilo lingers and dominates late completions. **bad sign:** if siso disappears from the middle of the run entirely, it means siso requests were being aborted while lilo jobs held all the KV slots.

---

when we need to stop: `scancel <JOBID>` and `pkill -f "8000:<NODE>"`.
