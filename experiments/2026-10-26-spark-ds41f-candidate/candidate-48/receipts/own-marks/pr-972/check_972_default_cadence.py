"""Supplementary check (ours, not the PR's): the image's default SSE keep-alive cadence.

Resolves the interval the way the routers do (get_sse_keep_alive_interval with no request and
no VLLM_SSE_KEEPALIVE_INTERVAL_S), then streams a generator that stalls 46 s before its first
data chunk through with_sse_keep_alive and records when each ': keep-alive' frame arrives.
"""
import asyncio
import json
import os
import time

import uvloop

from vllm.entrypoints.serve.utils import sse_keep_alive as ska

os.environ.pop("VLLM_SSE_KEEPALIVE_INTERVAL_S", None)
os.environ.pop("VLLM_SSE_KEEP_ALIVE_INTERVAL", None)
STALL_S = 46.0


async def main():
    resolve = getattr(ska, "get_sse_keep_alive_interval", None)
    interval = resolve() if resolve else None
    out = {
        "has_get_sse_keep_alive_interval": resolve is not None,
        "DEFAULT_SSE_KEEP_ALIVE_INTERVAL_S": getattr(ska, "DEFAULT_SSE_KEEP_ALIVE_INTERVAL_S", None),
        "resolved_default_interval_s": interval,
    }
    if not interval:
        print(json.dumps(out, indent=2))
        return

    async def gen():
        await asyncio.sleep(STALL_S)
        yield "data: first\n\n"

    t0 = time.monotonic()
    frames, first_data_s = [], None
    async for chunk in ska.with_sse_keep_alive(gen(), interval):
        t = round(time.monotonic() - t0, 3)
        if chunk == ska.SSE_KEEP_ALIVE_COMMENT:
            frames.append(t)
        else:
            first_data_s = t
    out.update({
        "stall_s": STALL_S,
        "keep_alive_frame_times_s": frames,
        "keep_alive_gaps_s": [round(b - a, 3) for a, b in zip([0.0] + frames, frames)],
        "first_data_at_s": first_data_s,
    })
    print(json.dumps(out, indent=2))


uvloop.run(main())
