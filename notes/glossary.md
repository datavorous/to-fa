# Inference Serving Glossary

A reference for metrics, bottlenecks, and system concepts in LLM inference. Written for someone who understands deep learning but is new to production serving. Organized from fundamentals to advanced topics.



## Latency Metrics

### TTFT: Time to First Token
Elapsed time from when the client sends a request to when it receives the first response token. Two components:

1. **Queue wait**: time sitting in the scheduler waiting for a GPU slot
2. **Prefill**: time for the GPU to process the entire input prompt and produce the first output token

In a lightly loaded system, queue wait ≈ 0 and TTFT ≈ prefill time. Under load, queue wait dominates. TTFT is the metric conversational users feel most directly: it is the blank before the response starts streaming. For a 14B model at bf16, prefill of a 400-token prompt takes ~80–120ms unloaded.

### ITL: Inter-Token Latency
Average time between consecutive output tokens during generation.

`ITL = (E2E − TTFT) / (completion_tokens − 1)`

ITL is determined by model size, batch size, and hardware memory bandwidth: not by prompt length or content. For a 14B bf16 model: each decode step reads ~28 GB of weights from HBM. At 864 GB/s bandwidth, that is ~32ms per step at batch size 1. With batching, memory reads are amortized across multiple sequences: ITL improves as batch size grows until the compute ceiling is reached.

### E2E: End-to-End Latency
`E2E = TTFT + (completion_tokens − 1) × ITL`

For chat (short output): dominated by TTFT. For code (long output): dominated by `completion_tokens × ITL`. A 2000-token code response at 30ms ITL = 60s decode time, regardless of serving stack choices. The only ways to reduce this are: reduce ITL (quantization, speculative decoding) or reduce output length (not in your control).

### Throughput (tok/s)
Total completion tokens generated across all requests divided by wall-clock run time. A system-level metric: it measures how well the GPU is utilized across all concurrent users, not how fast any individual request is served. Throughput and per-request latency are in fundamental tension.

### Budget Utilization
`completion_tokens / max_tokens`. Whether the model hit its token ceiling or stopped early via EOS. High (>85%): model was cut off, response may be incomplete. Low (<50%): budget was over-allocated, request held a GPU slot longer than necessary, reducing throughput for others.



## The Roofline Model

The single most important mental model for LLM inference. Every GPU operation is bounded by one of two limits:

**Peak compute:** FLOPs the GPU can execute per second. RTX 6000 Ada: 182 TFLOPS bf16.

**Peak memory bandwidth:** How fast data moves between HBM and compute units. RTX 6000 Ada: 864 GB/s.

**Arithmetic intensity:** FLOPs executed per byte of data read from memory. Operations above the roofline ridge point are compute-bound; below it are memory-bandwidth bound.

For LLM **decode** at small batch size: one forward pass reads ~28 GB (14B bf16) and performs ~28 GFLOPs (one multiply-accumulate per weight per new token). Arithmetic intensity = 28G / 28G = **1 FLOP/byte**. To be compute-bound at this intensity requires 182 TB/s of bandwidth: you have 864 GB/s. **Decode is always memory-bandwidth bound.**

For LLM **prefill** at N=1000 tokens: the same weights are reused for all 1000 tokens simultaneously. FLOPs = 28G × 1000, bytes read ≈ 28G. Arithmetic intensity = 1000 FLOP/byte. Now the compute ceiling applies. **Prefill is compute-bound for long prompts.**

This split explains everything:
- Why quantization (halving bytes) helps decode dramatically but not prefill proportionally
- Why bigger batch size helps decode (amortizes memory reads) but barely helps prefill
- Why chunked prefill makes sense: interleave memory-bound decode with compute-bound prefill



## Memory Architecture

### HBM: High Bandwidth Memory
The VRAM on a GPU. Fast (864 GB/s on RTX 6000 Ada), small (48 GB). All model weights, KV cache, and activations must live here during inference. Every decode step reads the full weight tensor from HBM: this is the primary bottleneck.

### SRAM: On-Chip Cache
Extremely fast (~10 TB/s), tiny (~100 MB total on the GPU). FlashAttention exploits SRAM by keeping attention intermediates in SRAM rather than round-tripping to HBM. This matters for long sequences where attention alone would otherwise dominate HBM traffic.

### Memory Hierarchy (approximate)
```
Registers         ~10 TB/s      < 1 KB per SM
Shared Memory     ~10 TB/s      ~100 KB per SM
L2 Cache          ~3 TB/s       ~80 MB
HBM (VRAM)        ~864 GB/s     48 GB
System RAM         ~50 GB/s      hundreds of GB
NVMe SSD           ~7 GB/s       TBs
```

### The 14B Memory Budget on RTX 6000 Ada
At bf16: `14B params × 2 bytes = 28 GB` weights.  
With `gpu_memory_utilization=0.9`: vLLM claims 43.2 GB → 28 GB weights → **~15 GB for KV cache**.

KV cache per token for Qwen2.5-14B (40 layers, 8 KV heads GQA, 128 head-dim, bf16):  
`40 × 2 × 8 × 128 × 2 bytes = 163,840 bytes ≈ 160 KB/token`

15 GB / 160 KB = **~93,750 concurrent token slots**

- Concurrency=16, 4096-tok outputs: `16 × 4096 = 65,536` → fits, ~70% of pool
- Concurrency=32, 4096-tok outputs: `32 × 4096 = 131,072` → **exceeds pool → preemptions**

This is the regime where memory optimization has real consequences.



## KV Cache

### What It Is
Stores attention keys and values for every processed token, every layer, every active sequence. Without it, generating token N would require reprocessing tokens 1..N−1 from scratch: O(N²) total cost. With KV cache, each step attends over cached history: O(N) per step.

### KV Cache Size Scaling
`size = layers × 2 × kv_heads × head_dim × bytes × context_len × batch_size`

Moving from 7B (32 layers, 32 KV heads) to 14B (40 layers, 8 KV heads via GQA): per-token cost goes from 512 KB to 160 KB: actually smaller, because GQA reduces KV heads drastically. But the weight memory doubles. Total VRAM pressure is much higher with 14B.

### PagedAttention
vLLM's KV cache allocator. Divides the KV pool into fixed-size pages (blocks of 16 tokens by default). Sequences are assigned pages non-contiguously as they grow. Before PagedAttention, each sequence reserved memory for its full `max_context_length` upfront: a 100-token response would hold a 8192-token reservation, wasting 98.8%. PagedAttention allocates only what is used.

### Prefix Caching
Server-side cache of KV blocks for shared prompt prefixes. When two requests share a prefix (e.g., the same system prompt), the second request reuses the cached KV blocks: skipping their prefill entirely. Hit rate of 95% means 95% of prefill work is skipped across the request population.

**Hit rate decays under memory pressure.** When the KV pool fills, prefix cache blocks are the first to be evicted (active sequence blocks cannot be evicted without preemption). On a 14B model at high concurrency, the prefix cache may be evicted frequently, negating its benefit. This is a failure mode that does not exist on 7B with abundant headroom.

### KV Cache Quantization
Store KV blocks in fp8 or int8 instead of bf16. Halves KV memory, allowing 2× more concurrent token slots before hitting the ceiling. Does not reduce weight loading time: only reduces attention memory reads. Significant for long-context requests where KV traffic competes with weight traffic on the HBM bus. vLLM flag: `--kv-cache-dtype fp8`.

### Preemption
When the KV pool fills, vLLM must free space. It preempts the lowest-priority active sequence: evicts its KV blocks and either recomputes from scratch when a slot reopens (recompute mode) or swaps KV state to CPU RAM (swap mode).

**Recompute preemption:** Wastes all compute already spent on that sequence. Causes a latency spike equal to the full prefill time of the preempted sequence.

**Swap preemption:** KV state written to CPU RAM over PCIe (~32 GB/s). Moving 1 GB of KV takes ~31ms. Swap-in adds the same cost. Useful if preemptions are rare; catastrophic if frequent.

With a 14B model at concurrency=32 and long outputs, preemptions are expected. `vllm:num_preemptions_total` in the Prometheus metrics tracks this. Zero preemptions = memory pressure not reached; nonzero = you have found the ceiling.



## Prefill vs Decode

Two fundamentally different computational regimes within one request.

| | Prefill | Decode |
|---|---|---|
| Tokens per GPU step | All N input tokens | 1 per active sequence |
| Bound by | Compute (for N > ~100) | Memory bandwidth |
| Scales with batch size | Weakly | Strongly |
| Parallelism available | High (matrix over N tokens) | Low (vector per token) |
| Duration, 14B, 400-tok prompt | ~80–120 ms |: |
| Duration, 14B, 2000-tok output |: | ~60 s at 33ms ITL |

**The competition:** Prefill and decode share the same GPU. A long prefill monopolizes the GPU, stalling all ongoing decode operations: every active sequence's ITL spikes. This is head-of-line blocking at the GPU level.

### Prefill Compute Cost
O(N²) in sequence length due to self-attention, linear in model size. For 14B: roughly 0.2ms per input token. A 4000-token document context: ~800ms TTFT even with no queue wait. RAG systems (retrieve → stuff → generate) pay this cost on every request.

### Disaggregated Prefill
Run prefill and decode on separate GPU instances. Prefill instances are compute-optimized (large matrices, high FLOPS). Decode instances are bandwidth-optimized (wide KV cache access, many small batches). Used in large-scale production systems (Together AI, Lepton). Not feasible on a single GPU, but represents the architectural ceiling of this design space.



## Quantization

### Why It Matters
A 14B bf16 model reads 28 GB per decode step at 864 GB/s: **32ms per step**. An AWQ int4 14B reads 7 GB: **8ms per step**. ITL drops 4×. This follows directly from the roofline model: halve the bytes, halve the time in the memory-bound regime.

### Weight-Only Quantization
Weights stored in low precision, dequantized to fp16/bf16 before the matrix multiply. Arithmetic stays in fp16. No accuracy loss from quantized activations.

**GPTQ:** Uses second-order Hessian information to minimize quantization error. Adjusts remaining weights to compensate for quantized ones. Standard int4 quantization. Most widely deployed.

**AWQ (Activation-aware Weight Quantization):** Identifies which weight channels are multiplied by large activations and protects them from aggressive quantization. Better quality than GPTQ at the same bit-width, particularly for instruction following. `Qwen/Qwen2.5-14B-Instruct-AWQ` exists on HuggingFace.

**With AWQ int4:** 14B model weights = ~7 GB. On 48 GB with `gpu_memory_utilization=0.9`: `43 GB − 7 GB = 36 GB for KV cache`. This nearly triples KV headroom compared to bf16 (15 GB → 36 GB), shifting the memory ceiling from concurrency=32 to concurrency=~70 before preemptions.

### FP8
8-bit floating point (E4M3 or E5M2 format). Native hardware support on Ada Lovelace. vLLM flag: `--dtype fp8`. Weights and activations both in fp8. 2× memory reduction vs bf16. Quality loss ~0.5–1% on benchmarks: nearly imperceptible for most tasks. Arithmetic intensity doubles. Easier to deploy than int4 (no calibration required).

### Weight + Activation Quantization
Both weights and activations quantized. Enables int8 matrix multiply units. More complex: activation ranges are dynamic and hard to bound statically.

**SmoothQuant:** Migrates quantization difficulty from activations to weights via a mathematically equivalent rescaling. Makes int8 activation quantization stable for most LLMs without per-tensor dynamic quantization.

### Calibration
Post-training quantization requires a calibration dataset: a few hundred representative prompts: to determine optimal quantization parameters (scales, zero-points). Quality depends entirely on calibration representativeness. A model calibrated on general web text may degrade on code or structured output tasks if those were underrepresented. AWQ is more robust to calibration data than GPTQ.

### Quantization and the KV Budget
Quantization does not directly help KV cache size: the KV cache is stored in fp16/bf16 regardless of weight precision (unless `--kv-cache-dtype fp8` is also set). But it frees VRAM that was used by weights, which can be reallocated to KV cache. With int4 weights on the 14B model, you go from 15 GB KV headroom to 36 GB: a 2.4× increase in concurrent sequence capacity.



## Attention Variants

### MHA: Multi-Head Attention
Original architecture. H query heads, H key heads, H value heads. KV cache per token = `layers × 2 × H × head_dim × 2 bytes`. For a hypothetical 14B with 40 heads MHA: `40 × 2 × 40 × 128 × 2 = 819 KB/token`. At concurrency=16, 4096 tokens each: 53 GB KV alone: does not fit on a 48 GB GPU. MHA at 14B scale is effectively unserveable on consumer hardware.

### GQA: Grouped Query Attention
Multiple query heads share a single KV head. Qwen2.5-14B uses 40 query heads, 8 KV heads (5:1 ratio). KV cost: `40 × 2 × 8 × 128 × 2 = 160 KB/token`: a 5× reduction vs MHA. GQA is what makes 14B models practical on single 48 GB GPUs. It is an architectural decision baked into the model, not a serving optimization you can add.

### MQA: Multi-Query Attention
All query heads share a single KV head. Maximum KV compression (effectively H× reduction), some quality degradation. Appears in older/smaller models (Falcon-7B). Modern models prefer GQA for better quality-efficiency balance.

### MLA: Multi-head Latent Attention (DeepSeek-V2/V3)
Projects keys and values into a compressed low-dimensional latent vector before caching. The full KV is reconstructed on demand from the latent. Achieves 5–13× KV cache reduction beyond GQA without MQA's quality loss. Not available in Qwen2.5, but represents the current state of the art for KV compression. Models using MLA can serve far more concurrent sequences at the same VRAM budget.

### FlashAttention
A kernel-level optimization, not an architectural change. Standard attention reads Q, K, V, and score matrices from HBM multiple times per step. FlashAttention tiles the computation so intermediates stay in SRAM, reducing HBM reads from O(N²) to O(N) for the attention step. No change to outputs: mathematically identical. For sequences longer than ~512 tokens, attention HBM traffic becomes significant; FlashAttention is essential at 4096+ context. vLLM uses it by default.



## Speculative Decoding

### The Core Idea
A small draft model generates k candidate tokens. The main model verifies all k in one forward pass: the same cost as generating 1 token, since verification is a parallel operation. If draft tokens 1..i are accepted and token i+1 is rejected, tokens 1..i are committed and token i+1 is resampled from the main model's distribution.

**Expected tokens per main model step:** For acceptance probability α per token and k speculation steps:  
`E[tokens] ≈ (1 − α^(k+1)) / (1 − α)`  
At α=0.7, k=4: E[tokens] ≈ 2.65. At α=0.85, k=4: E[tokens] ≈ 3.48.

**Effective ITL reduction:** If baseline ITL = 30ms and E[tokens] = 2.65, effective ITL = 30/2.65 ≈ 11ms. A 2000-token code response goes from 60s to 22s. This is the only serving optimization that breaks through the hardware memory bandwidth ceiling.

### When It Works
- Low temperature (code at 0.1–0.2): draft acceptance rates 75–85%
- Repetitive or templated outputs: function signatures, JSON, markdown headers
- Long outputs: amortizes the overhead of running the draft model

### When It Fails
- High temperature (chat at 0.8–1.0): acceptance rates 30–50%, overhead may negate gains
- When draft model is slow: if the draft model costs more than 1/k of the main model's step time, throughput drops
- Very short outputs: draft overhead is not amortized

### Draft Model Selection
Must use the same tokenizer. Should be 1/10th to 1/20th of main model parameters. For Qwen2.5-14B: `Qwen2.5-1.5B-Instruct` (~3 GB) is the natural draft. vLLM flag: `--speculative-model Qwen/Qwen2.5-1.5B-Instruct --num-speculative-tokens 4`.

### Advanced Variants
**EAGLE:** Uses a small head attached to the main model's hidden states as the draft, rather than a separate model. Higher acceptance rates than an independent draft model because it has access to main model context. Lower overhead.

**Medusa:** Multiple independent draft heads at different lookahead distances attached to the main model. Requires fine-tuning.

**Tree Attention (EAGLE-2):** Draft candidates form a tree: multiple branching continuations verified in parallel. Higher expected acceptance than linear speculation.



## Chunked Prefill

### Problem
A long-prompt request monopolizes the GPU for 80–800ms. All concurrent decode sequences stop receiving tokens: ITL spikes for everyone in the batch. In a mixed chat+code workload at concurrency=16, a 4000-token code prompt arriving causes a 600ms stall for 15 active chat sessions.

### Solution
`--enable-chunked-prefill --max-num-batched-tokens N` splits prefill into chunks of N tokens. Each GPU step processes one prefill chunk and all pending decode tokens interleaved. Total prefill cost increases slightly (more steps), but no single step is dominated by prefill.

**Optimal chunk size:** Smaller N = smoother ITL, longer TTFT for the new request. Larger N = faster TTFT, spikier ITL for others. Empirically: 512–1024 tokens works well for mixed chat+code workloads. The right value depends on prompt length distribution.

### Effect by Profile
- **Chat (short prompts):** Minimal change: 100-token prompts chunk in 1 step regardless
- **Code (long prompts, long context):** TTFT increases ~10–20%, but ITL for all concurrent requests stabilizes



## Scheduling

### FCFS: First Come First Served
Default vLLM policy. Simple, fair. Suffers from head-of-line blocking: a 4000-token code request blocks all subsequent short chat requests that could have been served in 100ms.

### Max Sequence Length Capping
`--max-model-len` sets the maximum context length. Setting this lower than the model's native maximum (e.g., 8192 instead of 32768) reduces the KV memory reserved per sequence: allowing more concurrent sequences before hitting the pool ceiling. If your workload doesn't use long contexts, this is free throughput.

### Max Batched Tokens
`--max-num-batched-tokens` caps total tokens processed per GPU step across all sequences. Controls GPU step latency. Setting it lower makes each step faster but requires more steps for the same work.

### Max Concurrent Sequences
`--max-num-seqs` caps concurrent active sequences at the scheduler level, regardless of concurrency the client sends. Useful for reserving KV headroom when prefix cache retention is important.



## Common Failure Modes by Model Size

### 7B on 48 GB: Too Comfortable
Weights: 14 GB (29% of VRAM). KV headroom: ~29 GB. Preemptions: never. KV peak usage: <5% at realistic concurrency. The interesting failure modes do not trigger. Optimization experiments show small, hard-to-interpret gains. Useful for studying TTFT, scheduler behavior, and prefix caching in isolation: not for studying memory pressure.

### 14B on 48 GB: Well-Matched
Weights: 28 GB (58% of VRAM). KV headroom: ~15 GB at 0.9 utilization. Preemptions trigger at concurrency≥32 with long outputs. KV cache fills to 70%+ at concurrency=16. Prefix cache begins evicting under load. Quantization doubles KV headroom and makes a visible latency difference. This is the correct operating regime for studying inference optimization.

### 30B on 48 GB: Quantization Required
bf16: ~60 GB: does not fit. AWQ int4: ~15 GB weights, ~28 GB KV headroom. Fits, but decode throughput is now limited by the dequantization overhead rather than pure bandwidth. Quality vs speed tradeoff becomes the main story. Concurrency is severely limited by KV pressure even with int4.

### 70B on 48 GB: Wrong Tool
bf16: 140 GB: requires 3–4 GPUs with tensor parallelism. AWQ int4: ~35 GB: fits, but ~8 GB left for KV cache. One sequence at 4096 tokens needs `4096 × 320 KB = 1.3 GB` KV (70B has 80 layers). Maximum concurrent sequences ≈ 6. Impractical for any real workload.



## Memory-Bound Optimization Strategies (Summary)

| Strategy | Targets | Mechanism | Tradeoff |
|---|---|---|---|
| Higher concurrency | Throughput | Amortizes weight reads across more sequences | ITL degrades, TTFT increases |
| AWQ int4 quantization | ITL, KV headroom | 4× fewer bytes to read per decode step | ~2–4% quality loss |
| FP8 weights + activations | ITL | 2× fewer bytes, native Ada hardware | ~0.5% quality loss |
| KV cache quantization (fp8) | Concurrent sequences | Halves KV memory per token | Slight attention precision loss |
| Prefix caching | TTFT | Skips prefill for shared prompt prefixes | Evicted under KV pressure |
| Chunked prefill | ITL stability | Prevents prefill monopoly | Slightly higher TTFT for long prompts |
| Speculative decoding | ITL | Multiple tokens per main model step | Acceptance rate-dependent; hurts at high temperature |
| Reduced max_model_len | KV headroom | Smaller per-sequence KV reservation | Can't serve long contexts |
| Disaggregated prefill | TTFT + ITL | Separate compute for prefill vs decode | Multi-GPU or multi-node required |
| MLA attention | KV headroom | Compressed KV latent space | Requires model support (DeepSeek only currently) |
