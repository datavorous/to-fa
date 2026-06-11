# Experiment bed :: context for a larger model

This document describes the full inference serving experiment harness end-to-end — what it does, how it works, what it produces, and what levers exist. It is intended as a self-contained briefing.

---

## 1. What we are running

**Model:** Qwen/Qwen2.5-14B-Instruct  
**Server:** vLLM 0.8.5, OpenAI-compatible API, single RTX 6000 Ada (48GB VRAM)  
**Cluster:** SLURM, partition u22, turing.iiit.ac.in  
**Context window:** 8192 tokens (configurable)  
**Precision:** BF16 weights, BF16 KV cache (default)

The server is intentionally run at saturation — the workload is designed to exceed comfortable capacity so that optimisation techniques have visible, measurable impact.

---

## 2. Synthetic workload generation

### Why synthetic

Real production traffic is unavailable. Synthetic generation lets us control the exact shape of the load — input length, output length, topic diversity — while ensuring every request is unique (no repeated prompts that would artificially inflate prefix cache hit rates).

### The four profiles

Every request belongs to one of four profiles, each representing a distinct real-world traffic class:

| profile | input tokens | output tokens | real-world equivalent |
|---|---|---|---|
| **siso** | 44–101 | 30–120 | chat Q&A, support bot, autocomplete |
| **silo** | 137–157 | 800–1500 | code generation from a brief spec |
| **liso** | 481–4594 | 40–250 | code review / summarisation of a real source file |
| **lilo** | 555–4665 | 900–2000 | code conversion / rewrite of a real source file |

The names encode the shape: **S**hort/**L**ong **I**nput, **S**hort/**L**ong **O**utput.

### How prompts are built

**siso:** filled templates like `"What's the best way to {fix} my {thing}?"` with random vocabulary slots. Every request draws different slot values → no two prompts are identical.

**silo:** code generation tasks like `"Write a Python {pattern} that handles {scenario} with {constraint}"` with randomised slot fills.

**liso / lilo:** real source files from the codebase (`factory/*.py`, `utils/*.py`) are loaded as a corpus. Each request picks one file at random and prepends a **randomly chosen preamble sentence** (8 options per profile) before the file contents. The preamble is different per request, which means the shared prefix between any two requests is zero tokens — vLLM's radix-tree prefix cache finds no match and must compute the full KV for every request. This is intentional: without the preamble, all requests on the same file would share a multi-thousand-token prefix and the cache would do most of the work for free, producing an artificially inflated throughput number.

liso tasks: summarise, list functions, find bugs, rate quality, identify data flows  
lilo tasks: rewrite in Rust, translate to x86-64 assembly, convert to Go, port to JavaScript, full async refactor

### Workload sizing (baseline experiment)

```
siso : 300 requests   max_tokens [128, 512]    temperature [0.5, 1.0]
silo :  75 requests   max_tokens [2048, 3072]  temperature [0.1, 0.4]
liso : 200 requests   max_tokens [64, 256]
lilo : 150 requests   max_tokens [1024, 2048]
total: 725 requests
```

Requests are generated once at run start with a fixed seed (reproducible), then shuffled before injection.

---

## 3. Load injection

### Load modes

The runner (`factory/runner.py`) supports five modes, configured via `config.yaml`:

| mode | behaviour | use case |
|---|---|---|
| `synchronous` | one request at a time, wait for response | baseline latency floor |
| `concurrent` | up to N requests in flight simultaneously (semaphore) | closed-loop burst |
| `constant` | one new request every `1/rate_rps` seconds | steady arrival rate |
| `poisson` | inter-arrival time drawn from `expovariate(rate_rps)` | realistic bursty traffic |
| `sweep` | loops over a list of values for one param (e.g. concurrency) | throughput/latency curves |

**Baseline uses poisson at 32 rps, concurrency cap 128.**

### What the runner actually does

For `poisson` mode:
1. A dispatch coroutine fires requests at exponentially-distributed intervals (`random.expovariate(32.0)` → mean gap 31ms between arrivals).
2. Each request is fired as an independent `asyncio` task with no semaphore — the server's own queue is the backpressure mechanism.
3. Completions arrive out-of-order and are collected via an `asyncio.Queue`.
4. All 725 tasks are in flight simultaneously from the server's perspective once dispatch completes (~22s into the run).

Each request is sent as a **streaming SSE chat completion** (`stream=True`). The client measures:
- **TTFT:** wall time from request send to first non-empty `delta.content` chunk
- **E2E:** wall time from send to `[DONE]`
- **ITL (inter-token latency):** `(E2E - TTFT) / (completion_tokens - 1)` — average time per generated token after the first
- **prompt_tokens / completion_tokens:** from the `usage` field in the final chunk

### System polling

Concurrently with request dispatch, a background coroutine polls `GET /metrics` (Prometheus endpoint) every 2 seconds and records:

```
kv_cache_usage        — fraction of KV cache blocks in use (0.0–1.0)
requests_running      — sequences currently being decoded
requests_waiting      — sequences queued, not yet scheduled
prefix_cache_hits     — cumulative count of KV blocks served from cache
prefix_cache_queries  — cumulative count of KV block lookups
```

These snapshots are written to `system.jsonl` alongside the per-request `requests.jsonl`.

---

## 4. What gets written to disk

Every run produces a directory `results/<experiment>/<run_id>/` containing:

### `requests.jsonl`
One JSON line per request:
```json
{
  "id": "lilo-0042",
  "profile": "lilo",
  "length": "long",
  "status": "ok",
  "max_tokens": 1536,
  "ttft_s": 166.1,
  "total_s": 275.2,
  "prompt_tokens": 2172,
  "completion_tokens": 1460,
  "tokens_per_sec": 5.31,
  "itl_ms": 81.7,
  "error": ""
}
```
`status` is `"ok"` or `"error"`. Aborted requests (connection drops under KV pressure) land as `"error"` with blank `error` field and `ttft_s=0`, `completion_tokens=0`.

### `system.jsonl`
One JSON line per 2-second snapshot:
```json
{
  "t": 1781161974.154,
  "kv_cache_usage": 0.997,
  "requests_running": 34,
  "requests_waiting": 218,
  "prefix_cache_hits": 627181.0,
  "prefix_cache_queries": 957917.0
}
```

### `summary.json`
Aggregated stats: overall throughput, TTFT/E2E percentiles, per-profile breakdowns, system snapshot aggregates.

---

## 5. Workload composition at a given moment — what is and isn't possible

### What we cannot know
`requests.jsonl` has no absolute timestamps — only durations (`ttft_s`, `total_s`). The system poller records absolute `t` but only total counts (`requests_running`, `requests_waiting`), not per-profile breakdowns. vLLM's `/metrics` endpoint does not expose per-profile in-flight counts.

### What we can infer
Given that we know each request's approximate position in the completion order (requests.jsonl is written in completion order), and given that poisson dispatch takes ~22s to send all 725 requests, we can estimate the profile mix at any moment by:
- Assuming requests arrive at their configured poisson rate
- Knowing each profile's typical E2E duration (siso ~127s, liso ~174s, silo ~212s, lilo ~275s at baseline)
- At any time T, a request is "in flight" if it arrived before T and its E2E hasn't elapsed yet

At steady state (~t=40s to ~t=450s in the baseline), the GPU was running 25–46 sequences. Given relative E2E durations:
- lilo holds slots ~275s → dominates running set (~40–50% of slots)
- silo holds slots ~212s → ~25% of slots
- liso holds slots ~174s → ~20% of slots
- siso holds slots ~127s → ~10% of slots (rotates fastest)

### `workload_timeline.png` approximation
The timeline plot uses **completion order as a time proxy**. Since siso completes fastest it appears early; lilo appears late. This is a valid approximation for understanding the shift in mix over a run, but not a precise instantaneous picture.

**To get exact per-profile in-flight counts:** add a `start_time` field to each request at dispatch time and record it in `requests.jsonl`. Then any snapshot at time T can be joined against requests whose `[start_time, start_time + total_s]` interval contains T. This is not currently implemented.

---

## 6. Metrics produced and what they signify

### Per-request metrics (from `requests.jsonl`)

| metric | what it measures | good value | bad value |
|---|---|---|---|
| `ttft_s` | time from send to first token — user's perceived "response start" | siso < 2s, liso < 5s | > 30s means queue starvation |
| `total_s` (E2E) | full response time | proportional to output length | disproportionately high = ITL inflation |
| `itl_ms` | average ms per generated token after first | 80–120ms on healthy GPU | > 250ms means GPU decode congestion |
| `tokens_per_sec` | completion tokens / total_s | silo/lilo: 6–12 tok/s | < 3 tok/s means severe contention |
| `completion_tokens / max_tokens` (budget utilisation) | how much of the token budget was used | 0.3–0.8 = natural stop | 1.0 = truncated mid-output |
| `status=error, ttft=0` | silent abort — connection dropped under KV pressure | 0 aborts | > 10% = admission control needed |

### Per-snapshot system metrics (from `system.jsonl`)

| metric | what it measures | good | bad |
|---|---|---|---|
| `kv_cache_usage` | fraction of KV memory blocks occupied | < 0.85 | = 1.0 → eviction cascade begins |
| `requests_running` | sequences currently on GPU | stable, matches concurrency target | collapsing despite full queue = eviction |
| `requests_waiting` | queue depth — requests admitted but not yet scheduled | < 20 | > 100 = no admission control |
| per-interval prefix hit rate | KV blocks reused from cache / total lookups | depends on workload | > 0.7 with diverse prompts = suspicious |

### Derived / summary metrics

| metric | formula | what it means |
|---|---|---|
| abort rate | errors with ttft=0 / total requests | fraction of users silently dropped |
| TTFT p95/p50 ratio | p95 / p50 | > 10× = bimodal cliff; SLO monitoring on p50 alone is misleading |
| throughput (tok/s) | total completion tokens / wall time | aggregate decode rate |
| KV eviction pressure | snapshots at 100% / total snapshots | fraction of run spent in eviction-recompute cycle |
| waiting / running ratio | mean(waiting) / mean(running) | > 5 = server is a waiting room, not a server |

---

## 7. Plots and how to read them

### `heatmap_tokens.png`
KDE density plot: x = prompt tokens, y = completion tokens, log-scale colour. Four blobs should be visible at siso (bottom-left), silo (bottom-right), liso (top-left), lilo (top-right). Blobs bleeding downward = lilo/liso being truncated at max_tokens.

### `latency_cdf.png`
Horizontal bar chart, one row per profile, two panels (TTFT and E2E). Solid bar = p50, thin extension = p50→p95 tail, dot = p99, label = exact p95/p99. Short bar with thin tail is good. Tail 10–100× longer than bar = cliff.

### `system_timeline.png`
Three panels over wall time: KV cache %, requests running vs waiting, per-interval prefix hit rate. KV flatlined at 100% = eviction cascade. Waiting >> running = queue overflow. Prefix hit rate spike = shared prompt prefixes (check if workload is diverse).

### `workload_timeline.png`
Stacked bar + smoothed area of profile completions over run. Siso disappearing from mid-run = siso being aborted while lilo holds KV slots.

### `summary-table` (terminal)
First thing to check. Shows abort rate, TTFT p50/p95, KV mean/peak, queue max, per-profile budget utilisation with bar indicators.

---

## 8. Knobs available for optimisation experiments

### vLLM server flags (set via `config.yaml` → `vllm.extra_args` or named fields)

| flag | what it does | expected effect |
|---|---|---|
| `--preemption-mode swap` | evict KV blocks to CPU RAM instead of recomputing | lower TTFT p95, higher KV mean |
| `--max-num-seqs N` | cap concurrent sequences at N | lower abort rate, higher waiting queue |
| `--enable-chunked-prefill` | break large prefills into chunks interleaved with decode | lower liso/lilo TTFT, slight ITL increase |
| `--max-num-batched-tokens N` | max tokens processed per scheduler step | tune chunked prefill chunk size |
| `--num-scheduler-steps N` | run N decode steps per scheduling cycle | higher decode throughput, higher scheduling latency |
| `--block-size 32` | KV block granularity (default 16) | less fragmentation for long sequences |
| `--kv-cache-dtype fp8` | halve KV memory footprint | ~2× more KV blocks, lower eviction rate |
| `--quantization awq` | 4-bit weights (~9GB vs ~28GB) | ~3× more KV capacity, slight quality loss |
| `--gpu-memory-utilization` | fraction of VRAM reserved for KV cache | tune headroom vs capacity |
| `--swap-space N` | GB of CPU RAM for KV swap | only relevant with `--preemption-mode swap` |
| `enable_prefix_caching: false` | disable radix-tree KV reuse | honest throughput measurement, lower throughput |

### Workload knobs (set via `config.yaml` → `experiments.<name>.run`)

| param | effect |
|---|---|
| `rate_rps` | arrival rate — higher = more queue depth, more aborts |
| `concurrency` | max in-flight for closed-loop modes |
| `siso/silo/liso/lilo.n` | request count per profile — changes mix |
| `siso/silo/liso/lilo.max_tokens` | output budget — affects KV slot hold time |
| `load_mode` | synchronous / concurrent / constant / poisson / sweep |
| `seed` | RNG seed — same seed = reproducible request set |

### Adding a new experiment
Add a block under `experiments:` in `config.yaml`. Only fields that differ from baseline need to be specified — they are deep-merged:

```yaml
experiments:
  my_experiment:
    note: "what I'm testing"
    vllm:
      extra_args: "--preemption-mode swap --swap-space 8"
    run:
      rate_rps: 20.0
```

Then `EXPERIMENT=my_experiment sbatch serve.slurm` and `uv run bench my_experiment`.

---

## 9. Known limitations and gaps

- **No absolute request timestamps:** `requests.jsonl` has durations only. Exact per-profile in-flight counts at time T require adding `start_time` to the Result dataclass and runner dispatch.
- **Silent aborts misclassified:** aborted requests land as `status=error` with blank `error` field. They should be `status=aborted` for cleaner metrics separation. The runner catches only generic exceptions; connection resets under vLLM abort are swallowed silently.
- **No cross-run comparison plot:** each run produces its own plots. No tool yet exists to overlay two runs for before/after optimisation comparison.
- **prefix_cache_hit_rate in summary.json is cumulative:** it's computed as `total_hits / total_queries` over the entire run, not the mean of per-interval rates. This understates peak hit rates and overstates low-hit-rate periods. Per-interval rates are computable from `system.jsonl` deltas.
- **lilo max_tokens not clamped to available context:** `prompt_tokens + max_tokens` can exceed `max_model_len`, causing silent mid-generation truncation. A per-request clamp of `max_tokens = min(desired, max_model_len - prompt_tokens - 128)` is not yet implemented.
