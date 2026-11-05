"""Exercise the cancellation capacity probe over real loopback HTTP."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import socket
import time
from types import SimpleNamespace

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import probe


async def start_server(module):
    app = FastAPI()
    app.state.enable_server_load_tracking = True
    app.state.server_load_metrics = 0

    async def work(_, raw_request):
        fixture = app.state.fixture
        job = raw_request.state.job
        async with fixture.slots:
            if not job["long"]:
                await asyncio.sleep(0.005)
                return JSONResponse({"id": job["id"], "ok": True})
            fixture.active.add(job["id"])
            deadline = time.perf_counter() + 0.200
            try:
                while time.perf_counter() < deadline:
                    await asyncio.sleep(0.001)
                    if raw_request.state.abandoned:
                        fixture.wasted_steps += 1
            finally:
                await asyncio.sleep(0.005)
                fixture.active.remove(job["id"])
                fixture.finished[job["id"]].set()
        return JSONResponse({"id": job["id"], "ok": True})

    wrapped = probe.route(module, work)

    @app.post("/work")
    async def endpoint(raw_request: Request):
        # Consume the body before the native disconnect listener owns receive().
        raw_request.state.job = await raw_request.json()
        raw_request.state.abandoned = False
        if not raw_request.state.job["long"]:
            return await wrapped(None, raw_request)
        try:
            return await asyncio.wait_for(wrapped(None, raw_request), timeout=0.035)
        except TimeoutError:
            raw_request.state.abandoned = True
            return JSONResponse({"error": "request deadline"}, status_code=504)

    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        if task.done():
            await task
            raise RuntimeError("server stopped before startup")
        await asyncio.sleep(0.001)
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{sock.getsockname()[1]}", timeout=2)
    return app, server, task, client


async def trial(app, client, pair):
    names = [f"distinct-job-{pair}-{i}" for i in range(4)]
    fixture = SimpleNamespace(slots=asyncio.Semaphore(4), active=set(), wasted_steps=0,
                              finished={n: asyncio.Event() for n in names})
    app.state.fixture = fixture
    start = time.perf_counter()
    responses = await asyncio.gather(*(client.post("/work", json={"id": n, "long": True}) for n in names))
    assert all(r.status_code == 504 and r.json() == {"error": "request deadline"} for r in responses)
    active_after_deadlines = len(fixture.active)
    retry_start = time.perf_counter()
    response = await client.post("/work", json={"id": "useful-short", "long": False})
    end = time.perf_counter()
    assert response.status_code == 200 and response.json() == {"id": "useful-short", "ok": True}
    await asyncio.gather(*(event.wait() for event in fixture.finished.values()))
    return {"probe_completion_ms": (end-start)*1000,
            "retry_latency_ms": (end-retry_start)*1000,
            "active_after_deadlines": active_after_deadlines,
            "post_timeout_work_steps": fixture.wasted_steps,
            "response_statuses": [r.status_code for r in responses] + [response.status_code]}


async def run(args, corrected):
    servers = {arm: await start_server(module) for arm, module in
               (("original", probe.original), ("corrected", corrected))}
    rows = []
    try:
        with (args.output / "trials.jsonl").open("x") as log:
            for pair in range(12):
                order = ("original", "corrected") if (pair + args.boot) % 2 else ("corrected", "original")
                for arm in order:
                    app, _, _, client = servers[arm]
                    metrics = await trial(app, client, pair)
                    row = {"boot": args.boot, "pair": pair, "arm": arm, "metrics": metrics}
                    rows.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    if arm == "corrected":
                        assert metrics["active_after_deadlines"] == 0
                        assert metrics["post_timeout_work_steps"] == 0
        return rows
    finally:
        for _, server, _, client in servers.values():
            await client.aclose()
            server.should_exit = True
        await asyncio.gather(*(task for _, _, task, _ in servers.values()))


def main():
    import statistics
    import math
    parser = argparse.ArgumentParser()
    parser.add_argument("--replacement", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--boot", required=True, type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("corrected_http_cost", args.replacement)
    corrected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(corrected)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt = {"boot": args.boot, "harness_sha256": sha(__file__),
               "helper_sha256": sha(probe.__file__),
               "original_sha256": sha(probe.original.__file__),
               "corrected_sha256": sha(args.replacement)}
    assert receipt["corrected_sha256"] == args.expected_sha256
    assert receipt["original_sha256"] == "555b3a999237e1cd8d6313eb9aaa3f2f4ad8d040bf80563c01557da7a451d1ed"
    status = 1
    try:
        rows = asyncio.run(run(args, corrected))
        summary = {}
        for metric in ("probe_completion_ms", "retry_latency_ms"):
            arms = {arm: [r["metrics"][metric] for r in rows if r["arm"] == arm]
                    for arm in ("original", "corrected")}
            delta = [b-a for a,b in zip(arms["original"], arms["corrected"])]
            mean = statistics.mean(delta)
            margin = 2.200985 * statistics.stdev(delta) / math.sqrt(12)
            summary[metric] = {"arm_means": {a:statistics.mean(v) for a,v in arms.items()},
                               "paired_difference_corrected_minus_original": mean,
                               "paired_difference_95_ci": [mean-margin,mean+margin]}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        status = 0
    finally:
        receipt["exit_code"] = status
        receipt["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
