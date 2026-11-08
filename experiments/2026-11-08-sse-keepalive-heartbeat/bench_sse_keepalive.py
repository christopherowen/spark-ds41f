#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark and verification harness for SSE keep-alive heartbeat and streaming abort cleanup."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import anyio
import uvloop
from fastapi.responses import StreamingResponse

from vllm.entrypoints.serve.utils.api_utils import (
    load_aware_call,
    with_cancellation,
)
from vllm.entrypoints.serve.utils.sse_keep_alive import (
    SSE_KEEP_ALIVE_COMMENT,
    with_sse_keep_alive,
)


async def baseline_old_keep_alive_stream(generator, interval: float):
    next_task = None
    try:
        while True:
            if next_task is None:
                next_task = asyncio.create_task(anext(generator))
            done, _ = await asyncio.wait({next_task}, timeout=interval)
            if done:
                try:
                    chunk = next_task.result()
                except StopAsyncIteration:
                    next_task = None
                    break
                next_task = None
                yield chunk
            else:
                yield SSE_KEEP_ALIVE_COMMENT
    finally:
        if next_task is not None and not next_task.done():
            next_task.cancel()
            await asyncio.gather(next_task, return_exceptions=True)
        await generator.aclose()


def wrap_stream(gen, mode: str, interval: float = 15.0):
    if mode == "disabled_0s":
        return with_sse_keep_alive(gen, 0.0)
    if mode == "baseline_old_15s":
        return baseline_old_keep_alive_stream(gen, interval)
    if mode == "candidate_new_15s":
        return with_sse_keep_alive(gen, interval)
    raise ValueError(f"Unknown mode: {mode}")


class DummyState:
    server_load_metrics = 0


class DummyApp:
    def __init__(self):
        self.state = DummyState()


class DummyRequest:
    def __init__(self):
        self.scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
        self.app = DummyApp()
        self._receive = None

    async def is_disconnected(self) -> bool:
        return False


async def bench_micro_stream(
    mode: str, num_streams: int, chunks_per_stream: int, concurrency: int
) -> dict:
    chunk_payload = 'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"x"}}]}\n\n'

    async def make_gen():
        for _ in range(chunks_per_stream):
            yield chunk_payload
        yield "data: [DONE]\n\n"

    async def consume_one():
        t0 = time.perf_counter()
        n = 0
        async for _ in wrap_stream(make_gen(), mode, 15.0):
            n += 1
        return time.perf_counter() - t0, n

    # Warmup
    for _ in range(8):
        await consume_one()

    total_chunks = 0
    stream_latencies_ms = []
    t_start = time.perf_counter()
    batches = num_streams // concurrency
    for _ in range(batches):
        results = await asyncio.gather(*(consume_one() for _ in range(concurrency)))
        for dt, n in results:
            stream_latencies_ms.append(dt * 1000.0)
            total_chunks += n
    elapsed = time.perf_counter() - t_start
    us_per_chunk = (elapsed * 1e6) / total_chunks
    chunks_per_sec = total_chunks / elapsed
    return {
        "mode": mode,
        "concurrency": concurrency,
        "num_streams": num_streams,
        "chunks_per_stream": chunks_per_stream + 1,
        "total_chunks": total_chunks,
        "elapsed_s": round(elapsed, 6),
        "us_per_chunk": round(us_per_chunk, 4),
        "chunks_per_sec": round(chunks_per_sec, 1),
        "stream_latency_mean_ms": round(statistics.mean(stream_latencies_ms), 4),
        "stream_latency_p95_ms": round(
             sorted(stream_latencies_ms)[int(0.95 * len(stream_latencies_ms))], 4
        ),
    }


async def bench_asgi_streaming_response(
    mode: str, num_streams: int, chunks_per_stream: int, concurrency: int
) -> dict:
    chunk_payload = 'data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"x"}}]}\n\n'

    async def make_gen():
        for i in range(chunks_per_stream):
            if (i & 3) == 0:
                await asyncio.sleep(0)
            yield chunk_payload
        yield "data: [DONE]\n\n"

    @with_cancellation
    @load_aware_call
    async def handle(_, raw_request):
        return StreamingResponse(
            content=wrap_stream(make_gen(), mode, 15.0),
            media_type="text/event-stream",
        )

    async def run_one():
        req = DummyRequest()
        done_evt = asyncio.Event()
        body_messages = 0

        async def receive():
            await done_evt.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            nonlocal body_messages
            if message["type"] == "http.response.body":
                if message.get("body"):
                    body_messages += 1
                if not message.get("more_body", False):
                    done_evt.set()

        req._receive = receive
        t0 = time.perf_counter()
        resp = await handle(None, req)
        await resp(req.scope, receive, send)
        dt = time.perf_counter() - t0
        assert req.app.state.server_load_metrics == 0
        return dt, body_messages

    # Warmup
    for _ in range(4):
        await run_one()

    total_chunks = 0
    latencies_ms = []
    t_start = time.perf_counter()
    batches = num_streams // concurrency
    for _ in range(batches):
        results = await asyncio.gather(*(run_one() for _ in range(concurrency)))
        for dt, n in results:
            latencies_ms.append(dt * 1000.0)
            total_chunks += n
    elapsed = time.perf_counter() - t_start
    return {
        "mode": mode,
        "concurrency": concurrency,
        "num_streams": num_streams,
        "chunks_per_stream": chunks_per_stream + 1,
        "total_chunks": total_chunks,
        "elapsed_s": round(elapsed, 6),
        "us_per_chunk": round((elapsed * 1e6) / total_chunks, 4),
        "chunks_per_sec": round(total_chunks / elapsed, 1),
        "stream_latency_mean_ms": round(statistics.mean(latencies_ms), 4),
        "stream_latency_p95_ms": round(
            sorted(latencies_ms)[int(0.95 * len(latencies_ms))], 4
        ),
    }


async def bench_prefill_heartbeat_and_ttft_handoff() -> dict:
    interval_s = 0.020
    prefill_wait_s = 0.072
    first_token_ready_at = 0.0

    async def slow_prefill_gen():
        nonlocal first_token_ready_at
        await asyncio.sleep(prefill_wait_s)
        first_token_ready_at = time.perf_counter()
        yield 'data: {"id":"cmpl-1","choices":[{"delta":{"content":"Hello"}}]}\n\n'
        yield 'data: {"id":"cmpl-1","choices":[{"delta":{"content":" world"}}]}\n\n'
        yield "data: [DONE]\n\n"

    t0 = time.perf_counter()
    heartbeats_ts = []
    data_ts = []
    data_payloads = []
    async for chunk in with_sse_keep_alive(slow_prefill_gen(), interval_s):
        now = time.perf_counter()
        if chunk.startswith(":"):
            heartbeats_ts.append(now - t0)
        else:
            data_ts.append(now - t0)
            data_payloads.append(chunk)

    handoff_overhead_us = (data_ts[0] + t0 - first_token_ready_at) * 1e6
    intervals_ms = [
        (heartbeats_ts[0]) * 1000.0,
        *(
            (heartbeats_ts[i] - heartbeats_ts[i - 1]) * 1000.0
            for i in range(1, len(heartbeats_ts))
        ),
    ]
    return {
        "configured_interval_ms": interval_s * 1000.0,
        "prefill_wait_ms": prefill_wait_s * 1000.0,
        "heartbeats_emitted": len(heartbeats_ts),
        "heartbeat_intervals_ms": [round(x, 3) for x in intervals_ms],
        "ttft_ms": round(data_ts[0] * 1000.0, 3),
        "first_token_handoff_overhead_us": round(handoff_overhead_us, 2),
        "data_frames_received": len(data_payloads),
    }


async def bench_disconnect_abort_completion(mode: str, trials: int = 32) -> dict:
    completed_aborts = 0
    leaked_load = 0

    async def run_trial():
        nonlocal completed_aborts, leaked_load
        req = DummyRequest()
        started = asyncio.Event()
        abort_done = False

        async def gen():
            started.set()
            try:
                await asyncio.Event().wait()
                yield "data: never\n\n"
            finally:
                nonlocal abort_done
                if mode == "candidate_new":
                    with anyio.CancelScope(shield=True):
                        await asyncio.sleep(0)
                        await asyncio.sleep(0.002)
                        abort_done = True
                else:
                    await asyncio.sleep(0)
                    await asyncio.sleep(0.002)
                    abort_done = True

        @with_cancellation
        @load_aware_call
        async def handle(_, raw_request):
            content = (
                with_sse_keep_alive(gen(), 15.0)
                if mode == "candidate_new"
                else baseline_old_keep_alive_stream(gen(), 15.0)
            )
            return StreamingResponse(content=content, media_type="text/event-stream")

        async def receive():
            await started.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        req._receive = receive
        resp = await handle(None, req)
        await resp(req.scope, receive, send)
        if abort_done:
            completed_aborts += 1
        if req.app.state.server_load_metrics != 0:
            leaked_load += 1

    await asyncio.gather(*(run_trial() for _ in range(trials)))
    return {
        "mode": mode,
        "trials": trials,
        "completed_aborts": completed_aborts,
        "leaked_load_metrics": leaked_load,
    }


async def main_async():
    micro_c1 = []
    micro_c16 = []
    asgi_c1 = []
    asgi_c16 = []
    for mode in ("disabled_0s", "baseline_old_15s", "candidate_new_15s"):
        micro_c1.append(await bench_micro_stream(mode, 200, 256, 1))
        micro_c16.append(await bench_micro_stream(mode, 320, 256, 16))
        asgi_c1.append(await bench_asgi_streaming_response(mode, 120, 256, 1))
        asgi_c16.append(await bench_asgi_streaming_response(mode, 160, 256, 16))

    prefill_cadence = await bench_prefill_heartbeat_and_ttft_handoff()
    abort_old = await bench_disconnect_abort_completion("baseline_old", 32)
    abort_new = await bench_disconnect_abort_completion("candidate_new", 32)

    out = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "event_loop": type(asyncio.get_running_loop()).__name__,
        "micro_stream_c1": micro_c1,
        "micro_stream_c16": micro_c16,
        "asgi_streaming_response_c1": asgi_c1,
        "asgi_streaming_response_c16": asgi_c16,
        "prefill_heartbeat_and_ttft": prefill_cadence,
        "disconnect_abort_cleanup": {
            "baseline_old": abort_old,
            "candidate_new": abort_new,
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    uvloop.run(main_async())
