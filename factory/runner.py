import asyncio
import contextlib
import json
import random
import time
from dataclasses import dataclass

import httpx


@dataclass
class Result:
    id: str
    profile: str
    length: str
    status: str
    max_tokens: int = 0
    ttft_s: float = 0.0
    total_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_sec: float = 0.0
    itl_ms: float = 0.0
    error: str = ""


async def fire(client, sem, req, CFG):
    result = Result(
        id=req.id,
        profile=req.profile,
        length=req.length,
        status="error",
        max_tokens=req.max_tokens,
    )

    body = {
        "model": CFG.model,
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.user},
        ],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    async with sem if sem else contextlib.nullcontext():
        t0 = time.perf_counter()
        first_token_at = None
        prompt_tokens = 0
        completion_tokens = 0

        try:
            async with client.stream(
                "POST", f"{CFG.base_url}/chat/completions", json=body, timeout=180.0
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break

                    chunk = json.loads(payload)
                    choices = chunk.get("choices", [])

                    if (
                        choices
                        and first_token_at is None
                        and choices[0].get("delta", {}).get("content")
                    ):
                        first_token_at = time.perf_counter()

                    if usage := chunk.get("usage"):
                        prompt_tokens = usage["prompt_tokens"]
                        completion_tokens = usage["completion_tokens"]

            total_s = time.perf_counter() - t0
            ttft_s = (first_token_at - t0) if first_token_at else total_s
            gen_s = total_s - ttft_s

            result.status = "ok"
            result.ttft_s = round(ttft_s, 4)
            result.total_s = round(total_s, 4)
            result.prompt_tokens = prompt_tokens
            result.completion_tokens = completion_tokens
            result.tokens_per_sec = round(completion_tokens / total_s, 2)
            result.itl_ms = (
                round(gen_s / (completion_tokens - 1) * 1000, 2)
                if completion_tokens > 1
                else 0.0
            )

        except Exception as exc:
            result.error = str(exc)

    return result


async def _closed_loop(client, requests, concurrency, CFG, on_result):
    sem = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(fire(client, sem, r, CFG)) for r in requests]
    results = []
    for i, done in enumerate(asyncio.as_completed(tasks), 1):
        result = await done
        results.append(result)
        if on_result:
            on_result(i, len(tasks), result)
    return results


async def _open_loop(client, requests, mode, rate_rps, CFG, on_result):
    total = len(requests)
    queue = asyncio.Queue()

    async def _fire_and_enqueue(req):
        result = await fire(client, None, req, CFG)
        await queue.put(result)

    async def _dispatch():
        for i, req in enumerate(requests):
            if i > 0:
                interval = (
                    (1.0 / rate_rps)
                    if mode == "constant"
                    else random.expovariate(rate_rps)
                )
                await asyncio.sleep(interval)
            asyncio.create_task(_fire_and_enqueue(req))

    dispatch_task = asyncio.create_task(_dispatch())

    results = []
    for i in range(total):
        result = await queue.get()
        results.append(result)
        if on_result:
            on_result(i + 1, total, result)

    await dispatch_task
    return results


async def run(requests, CFG, on_result=None, snapshots=None):
    from .system import poll

    if snapshots is None:
        snapshots = []

    stop_event = asyncio.Event()
    poller = asyncio.create_task(poll(CFG.base_url, 2.0, snapshots, stop_event))

    mode = getattr(CFG, "load_mode", "concurrent")
    limits = httpx.Limits(max_connections=256, max_keepalive_connections=64)

    async with httpx.AsyncClient(limits=limits) as client:
        if mode == "synchronous":
            results = await _closed_loop(client, requests, 1, CFG, on_result)
        elif mode in ("concurrent", "sweep"):
            results = await _closed_loop(
                client, requests, CFG.concurrency, CFG, on_result
            )
        elif mode in ("constant", "poisson"):
            results = await _open_loop(
                client, requests, mode, CFG.rate_rps, CFG, on_result
            )
        else:
            raise ValueError(f"unknown load_mode: {mode!r}")

    stop_event.set()
    await poller

    return results, snapshots
