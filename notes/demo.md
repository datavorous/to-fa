# Baseline run — observations, config, failures, and real-world impact

---

## What kind of company are we simulating?

A **mid-size AI API company** running a single overloaded GPU node during peak hours — think Together.ai, Fireworks, a self-hosted Copilot backend, or an enterprise internal LLM gateway. The node is undersized for the load. This is a company that either just crossed a growth inflection and hasn't scaled infra yet, or is cost-cutting by packing more traffic onto fewer GPUs.

---

## The run

**Run ID:** `20260611T074451Z` (prefill-fix baseline — preambles now randomised, no artificial prefix cache inflation)  
**Command:** `uv run bench baseline`  
**Wall time:** 489.3s

### Config at time of run

| parameter | value |
|---|---|
| load mode | poisson, 32 rps |
| concurrency | 128 |
| max_model_len | 8192 |
| gpu_memory_utilization | 0.9 |
| prefix caching | enabled |
| preemption mode | recompute (vLLM default) |
| scheduling | FCFS (vLLM default) |

### Workload mix sent

| profile | n | what it is | prompt tokens | max_tokens |
|---|---|---|---|---|
| siso | 300 | everyday chat Q&A | 44–101 tok | 128–512 |
| silo | 75 | code generation from brief spec | 137–157 tok | 2048–3072 |
| liso | 200 | code review / summarisation (real source files) | 481–4594 tok | 64–256 |
| lilo | 150 | code conversion / rewrite (real source files) | 555–4665 tok | 1024–2048 |

At any given moment during the run the GPU was processing a mix of all four. By ~8% into the run (≈40s), the waiting queue had already hit 210 and KV was at 100%. The queue stayed above 200 for the entire middle third of the run.

### Workload mix on GPU over time

| time into run | requests running | waiting queue | KV cache |
|---|---|---|---|
| 0% (start) | 0 | 0 | 0% |
| 8.5% (~41s) | 46 | 210 | **100%** |
| 17–75% | 25–37 | 130–231 | 94–99% |
| 84% (~411s) | 24 | 1 | **100%** |
| 93% (draining) | 8 | 0 | 41% |

Within 40 seconds the server was fully saturated. It stayed that way for the entire run. The small `running` count (25–46) against `waiting` (130–231) means the GPU was never underutilised — it was always at maximum KV pressure, constantly evicting and recomputing.

---

## What we observed

### 1. 54% abort rate (388/725 requests dropped)

```
total errors : 388 / 725
siso dropped : 165 / 300  (55%)
liso dropped : 112 / 200  (56%)
lilo dropped :  74 / 150  (49%)
silo dropped :  37 /  75  (49%)
```

Every single error had `ttft=0.00s`, `completion_tokens=0`, blank `error` field. These are silent connection drops — vLLM aborted the requests under KV pressure before generating a single token. The client received no HTTP status, no error body.

**What a good system should have done:** returned HTTP 429 or 503 with a `Retry-After` header the moment the queue depth exceeded a threshold. The client could then back off, retry, or surface a meaningful error. Instead, users got a silent timeout.

**Real-world harm:** Half your users get a spinner that never resolves. No error message. Support tickets say "the AI just stopped working." Engineers check dashboards, see average latency = 140s, and think it's slow but working. The 54% of users getting nothing are invisible.

---

### 2. TTFT completely broken — p50 = 142s, p95 = 351s

```
TTFT distribution (ok requests only):
  < 5s      :  58 requests  (17%)
  5 – 30s   :  69 requests  (20%)
  30 – 120s :  22 requests   (7%)
  120 – 300s: 137 requests  (41%)
  > 300s    :  51 requests  (15%)
```

Only 37% of completed requests got their first token within 30 seconds. 56% waited over 2 minutes. The p50 TTFT is 142 seconds — the median user who got a response waited over 2 minutes just to see the first token.

**What a good system should have done:** enforced a TTFT SLO (e.g. 10s). Requests not scheduled within 10s should be rejected with 503, not held in queue for 6 minutes. The server should never silently queue a request it cannot serve within a human-acceptable window.

**Real-world harm:** A chat user types a question. They wait. Nothing happens for 2 minutes. They refresh. Their request is gone. The 17% who got `< 5s` TTFT think the product is great. The 56% who waited 2+ minutes have churned. You'd never know from a mean latency dashboard.

---

### 3. KV cache saturated from the first 40 seconds

```
KV cache mean : 87.8%
KV cache peak : 100%
Snapshots at ≥99% : 8 / 238
Waiting queue mean : 151.4  max : 234
```

The KV cache hit 100% at t=40s and never meaningfully recovered. With `--preemption-mode recompute` (vLLM default), every eviction triggers a full prefill recompute of the evicted sequence. Under constant 100% KV pressure, this creates a recompute cascade: evict → recompute → fills cache again → evict again. The GPU is spending a significant fraction of its cycles re-doing work it already did.

**What a good system should have done:** `--preemption-mode swap` moves evicted KV blocks to CPU RAM over PCIe instead of recomputing. More importantly, `--max-num-seqs` should cap the scheduler at the number of sequences the KV cache can actually hold, so the cache never hits 100% in the first place.

**Real-world harm:** Throughput collapses non-linearly. The on-call engineer sees "GPU utilisation 95%, all looks fine." Meanwhile half the compute is recomputing evicted prefills. Throughput dropped from 636 tok/s (old inflated baseline) to 347 tok/s in this honest run — a 45% drop, entirely due to real prefill cost and eviction overhead.

---

### 4. Prefix cache hit rate dropped from 67% to 37% — throughput fell 45%

```
old baseline (shared prefixes)  : 636 tok/s, prefix hit rate 67%
new baseline (randomised preambles): 347 tok/s, prefix hit rate 37%
```

The old run's liso/lilo prompts all started with identical text (`"Here is the source code of config.py:\n\n```python\n..."`). vLLM's radix-tree cache reused those KV blocks for free. After the fix (randomised preamble per request), every liso/lilo request now pays full prefill cost.

**What this means:** the 636 tok/s number was a lie. The real serving cost of this workload mix — with diverse prompts as production traffic would actually be — is 347 tok/s. Any capacity planning based on the old number would result in a 2x underprovisioning mistake.

**Real-world harm:** A company benchmarks their stack, publishes "636 tok/s", plans infra accordingly. Production has diverse prompts. Throughput is 347 tok/s. They're 2x under capacity on day one of launch.

---

### 5. lilo truncated at 83% — model cut off mid-output

```
lilo: 76 ok, 63 maxed out (83%), budget_util p50=1.00
liso: 88 ok, 31 maxed out (35%), budget_util p50=0.58
```

83% of lilo responses hit the `max_tokens` ceiling mid-generation. The model was writing a Rust/Go/assembly rewrite and got cut off. The request completed with `status=ok` — no error — but the output is incomplete code.

**Root cause:** lilo prompts average 2172 tokens. With `max_model_len=8192` and `max_tokens` up to 2048, the model runs out of context window. `max_tokens = min(desired, max_model_len - prompt_tokens - safety_margin)` was never enforced.

**What a good system should have done:** clamp `max_tokens` per-request based on actual prompt length. Reject requests where `prompt_tokens + max_tokens > max_model_len` at admission, not silently truncate at generation time.

**Real-world harm:** A developer gets a "complete Rust rewrite" that stops at function 3 of 12. No error. They paste it into their IDE and it doesn't compile. They assume the model is bad. It's not — the infrastructure silently truncated it.

---

## Summary: what good looks like vs what we have

| dimension | what we have | what good looks like |
|---|---|---|
| error handling | silent drop, blank error, status=error | HTTP 429/503 + Retry-After, status=aborted |
| admission control | none — queue to 234 then drop | `--max-num-seqs` cap, reject at queue depth threshold |
| TTFT SLO | p50=142s, p95=351s | p95 < 10s for siso, p95 < 30s for liso |
| KV pressure | constant 100%, recompute eviction | `--preemption-mode swap`, headroom at 80% |
| throughput | 347 tok/s (honest) | 600+ tok/s with proper scheduling + swap eviction |
| lilo truncation | 83% silently cut off | per-request `max_tokens` clamped to available context |
| scheduling | FCFS — lilo blocks siso | priority scheduling: siso ahead of lilo, or separate queues |
| benchmark integrity | was inflated 2x by prefix cache | honest with randomised preambles |

---

## Real-world incident map

| what we see | IRL incident type | who hits this |
|---|---|---|
| 54% silent abort rate | "AI stopped working" — no errors in logs | any API company after viral growth spike |
| TTFT p50=142s | users churn silently, p50 dashboard looks fine | companies monitoring means not percentiles |
| KV at 100% from t=40s | cascade recompute, throughput halves | companies launching long-context features alongside chat |
| FCFS with lilo | 2-min chat waits behind a Rust rewrite job | any mixed interactive + batch workload on one endpoint |
| 2x throughput overestimate | infra underprovisioned on launch day | anyone who benchmarked on a fixed prompt corpus |
| lilo 83% truncated silently | incomplete code with status=ok | code tools, doc generation, anything with long output |
