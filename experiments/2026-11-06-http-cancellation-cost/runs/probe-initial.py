"""Small CPU-only comparisons of exact API source; no model or HTTP server."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time
from types import SimpleNamespace

os.environ["VLLM_PLUGINS"] = ""
import vllm.entrypoints.serve.utils.api_utils as original
from fastapi.responses import JSONResponse


class Request:
    def __init__(self, state):
        self.app = SimpleNamespace(state=state)
        self.state = SimpleNamespace()
        self.messages = asyncio.Queue()
        self.listening = asyncio.Event()

    async def receive(self):
        self.listening.set()
        return await self.messages.get()


def route(module, handler):
    return module.with_cancellation(module.load_aware_call(handler))


async def deliver(response):
    assert response.body == b'{"ok":true}'
    if response.background:
        await response.background()


async def capacity(module, trigger):
    """Four simulated engine slots: 200ms jobs, 5ms abort, 5ms new job."""
    before = asyncio.all_tasks()
    state = SimpleNamespace(enable_server_load_tracking=True, server_load_metrics=0)
    slots = asyncio.Semaphore(4)
    active = set()
    acquired = [asyncio.Event() for _ in range(4)]
    finished = [asyncio.Event() for _ in range(4)]
    requests = [Request(state) for _ in range(4)]
    cancellation_time = None
    post_cancel_steps = [0] * 4
    releases = [None] * 4

    async def long_job(index, raw_request):
        async with slots:
            active.add(index)
            acquired[index].set()
            deadline = time.perf_counter() + 0.200
            try:
                while time.perf_counter() < deadline:
                    await asyncio.sleep(0.001)
                    if cancellation_time is not None:
                        post_cancel_steps[index] += 1
            finally:
                await asyncio.sleep(0.005)
                releases[index] = time.perf_counter()
                active.remove(index)
                finished[index].set()
        return JSONResponse({"ok": True})

    async def short_job(_, raw_request):
        async with slots:
            await asyncio.sleep(0.005)
        return JSONResponse({"ok": True})

    long_route, short_route = route(module, long_job), route(module, short_job)
    parents = [asyncio.create_task(long_route(i, r)) for i, r in enumerate(requests)]
    try:
        await asyncio.gather(*(e.wait() for e in acquired))
        await asyncio.gather(*(r.listening.wait() for r in requests))
        cancellation_time = time.perf_counter()
        if trigger == "caller_cancel":
            for task in parents:
                task.cancel("request deadline")
        else:
            for request in requests:
                request.messages.put_nowait({"type": "http.disconnect"})

        async def probe():
            await deliver(await short_route(None, Request(state)))
            return (time.perf_counter() - cancellation_time) * 1000

        probe_task = asyncio.create_task(probe())
        await asyncio.sleep(0.020)
        at_20ms = {"active_long_jobs": len(active), "load_count": state.server_load_metrics}
        probe_ms = await probe_task
        await asyncio.gather(*parents, return_exceptions=True)
        await asyncio.gather(*(e.wait() for e in finished))
        await asyncio.sleep(0)
        orphans = [t for t in asyncio.all_tasks() - before if not t.done()]
        return {
            "probe_completion_ms": probe_ms,
            "post_cancel_work_steps": sum(post_cancel_steps),
            "held_slot_ms": sum((t - cancellation_time) * 1000 for t in releases),
            "at_20ms": at_20ms,
            "load_after_completion": state.server_load_metrics,
            "orphan_tasks_before_fixture_cleanup": len(orphans),
        }
    finally:
        # Original source leaks listeners. Count first, then clean every trial.
        remaining = [t for t in asyncio.all_tasks() - before if not t.done()]
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
        assert not [t for t in asyncio.all_tasks() - before if not t.done()]


async def successful(module, count=1000):
    state = SimpleNamespace(enable_server_load_tracking=True, server_load_metrics=0)

    async def handler(_, raw_request):
        return JSONResponse({"ok": True})

    wrapped = route(module, handler)
    # Complete the response background callback, as the normal ASGI path does.
    for _ in range(50):
        await deliver(await wrapped(None, Request(state)))
    await asyncio.sleep(0)
    before = asyncio.all_tasks()
    start = time.perf_counter()
    for _ in range(count):
        await deliver(await wrapped(None, Request(state)))
    elapsed = time.perf_counter() - start
    await asyncio.sleep(0)
    assert state.server_load_metrics == 0
    assert not [t for t in asyncio.all_tasks() - before if not t.done()]
    return {"requests": count, "microseconds_per_request": elapsed * 1e6 / count}


def summarize(rows, pairs):
    summary = {}
    for scenario, metric in (("caller_cancel", "probe_completion_ms"),
                             ("disconnect", "probe_completion_ms"),
                             ("successful", "microseconds_per_request")):
        arms = {arm: [r["metrics"][metric] for r in rows
                      if r["scenario"] == scenario and r["arm"] == arm]
                for arm in ("original", "corrected")}
        delta = [b - a for a, b in zip(arms["original"], arms["corrected"])]
        # Twelve paired trials: t(11), two-sided 95%; within this process only.
        assert pairs == 12
        margin = 2.200985 * statistics.stdev(delta) / math.sqrt(pairs)
        mean_delta = statistics.mean(delta)
        summary[scenario] = {
            "metric": metric,
            "arm_means": {k: statistics.mean(v) for k, v in arms.items()},
            "paired_difference_corrected_minus_original": mean_delta,
            "paired_difference_95_ci": [mean_delta - margin, mean_delta + margin],
        }
    return summary


async def run(args, corrected):
    rows = []
    with (args.output / "trials.jsonl").open("x") as log:
        for scenario in ("caller_cancel", "disconnect", "successful"):
            for pair in range(args.pairs):
                order = ("original", "corrected") if (pair + args.boot) % 2 else ("corrected", "original")
                for arm in order:
                    module = original if arm == "original" else corrected
                    metrics = await (successful(module) if scenario == "successful"
                                     else capacity(module, scenario))
                    row = {"boot": args.boot, "pair": pair, "scenario": scenario,
                           "arm": arm, "metrics": metrics}
                    rows.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if arm == "corrected" and scenario != "successful":
                        assert metrics["post_cancel_work_steps"] == 0, row
                        assert metrics["orphan_tasks_before_fixture_cleanup"] == 0, row
                        assert metrics["load_after_completion"] == 0, row
        return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replacement", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--boot", required=True, type=int)
    parser.add_argument("--pairs", default=12, type=int)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    spec = importlib.util.spec_from_file_location("http_cancellation_corrected", args.replacement)
    corrected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(corrected)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt = {
        "boot": args.boot, "pairs": args.pairs,
        "original_sha256": sha(original.__file__),
        "corrected_sha256": sha(args.replacement),
        "harness_sha256": sha(__file__),
        "packages": {p: importlib.metadata.version(p) for p in ("anyio", "starlette", "fastapi", "vllm")},
        "parameters": {"slots": 4, "long_job_ms": 200, "abort_cleanup_ms": 5,
                       "short_job_ms": 5, "step_ms": 1, "snapshot_ms": 20,
                       "successful_requests_per_trial": 1000},
    }
    assert receipt["original_sha256"] == "555b3a999237e1cd8d6313eb9aaa3f2f4ad8d040bf80563c01557da7a451d1ed"
    assert receipt["corrected_sha256"] == "1aeef18eb870532fde596a83077deb0764a7b3fc04680592e63750b1afde37ca"
    status = 1
    try:
        rows = asyncio.run(run(args, corrected))
        summary = summarize(rows, args.pairs)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        status = 0
    finally:
        receipt["exit_code"] = status
        receipt["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
