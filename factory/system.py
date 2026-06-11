import asyncio
import json
import statistics
import time

import httpx


async def poll(base_url, interval_s, snapshots, stop_event):
    async with httpx.AsyncClient() as client:
        while not stop_event.is_set():
            snap = {"t": round(time.time(), 3)}

            try:
                r = await client.get(
                    f"{base_url.replace('/v1', '')}/metrics", timeout=5.0
                )
                for line in r.text.splitlines():
                    if line.startswith("#"):
                        continue
                    if "vllm:gpu_cache_usage_perc{" in line:
                        snap["kv_cache_usage"] = float(line.split()[-1])
                    elif "vllm:num_requests_running{" in line:
                        snap["requests_running"] = int(float(line.split()[-1]))
                    elif "vllm:num_requests_waiting{" in line:
                        snap["requests_waiting"] = int(float(line.split()[-1]))
                    elif "vllm:gpu_prefix_cache_hits_total{" in line:
                        snap["prefix_cache_hits"] = float(line.split()[-1])
                    elif "vllm:gpu_prefix_cache_queries_total{" in line:
                        snap["prefix_cache_queries"] = float(line.split()[-1])
            except Exception:
                pass

            snapshots.append(snap)
            await asyncio.sleep(interval_s)


def aggregate(snapshots):
    if not snapshots:
        return {}

    def _mean(key):
        vals = [s[key] for s in snapshots if key in s]
        return round(statistics.mean(vals), 3) if vals else None

    def _max(key):
        vals = [s[key] for s in snapshots if key in s]
        return max(vals) if vals else None

    last_hits = next(
        (
            s["prefix_cache_hits"]
            for s in reversed(snapshots)
            if "prefix_cache_hits" in s
        ),
        None,
    )
    last_queries = next(
        (
            s["prefix_cache_queries"]
            for s in reversed(snapshots)
            if "prefix_cache_queries" in s
        ),
        None,
    )
    hit_rate = (
        round(last_hits / last_queries, 4) if last_hits and last_queries else None
    )

    return {
        "kv_cache_usage_mean": _mean("kv_cache_usage"),
        "kv_cache_usage_peak": _max("kv_cache_usage"),
        "prefix_cache_hit_rate": hit_rate,
        "requests_running_mean": _mean("requests_running"),
        "requests_waiting_max": _max("requests_waiting"),
        "n_snapshots": len(snapshots),
    }


def store_system(snapshots, out_dir):
    with open(f"{out_dir}/system.jsonl", "w") as f:
        for s in snapshots:
            f.write(json.dumps(s) + "\n")
