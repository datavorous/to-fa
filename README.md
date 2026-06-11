# to-fa

a study on inference serving

squeezing performance out of RTX 6000 Ada (48GB) running `Qwen2.5-14B-Instruct` under realistic mixed workloads.

## setup

```bash
sbatch serve.slurm
tail -f ~/<JOBID>.out
```

the log prints the exact tunnel command once ready.

open the tunnel (on your laptop)

```bash
ssh -NL 8000:<NODE>:8000 <user>@turing.iiit.ac.in
curl http://localhost:8000/health
```

## run benchmarks

```bash
uv run bench baseline # busy-monday baseline (32 rps poisson, 725 requests)
uv run bench # dev config (small, quick)
```

config is in `config.yaml`. results land in `results/<experiment>/<run_id>/`.

## analysis tools

run any of these after a bench run. all tools default to the latest run under `results/baseline/` if no path is given.

```bash
uv run summary-table [run_dir] # terminal: key metrics at a glance
uv run heatmap [run_dir] # png: token distribution heatmap
uv run latency-plot [run_dir] # png: TTFT and E2E latency per profile
uv run system-plot [run_dir] # png: KV cache, queue depth, prefix hit rate over time
uv run workload-timeline [run_dir] # png: profile mix (siso/silo/liso/lilo) over run
uv run token-counts [experiment] # terminal: prompt/budget token stats (pre-run)
```
