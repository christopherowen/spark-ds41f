"""CPU transport check for the hardware client and identical fault middleware."""

import asyncio
import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from http_cancellation_deadline import DeadlineMiddleware

ROOT = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


async def check(arm):
    api = module("fixture_" + arm, ROOT / f"{arm}_api.py")
    probe = module("hardware_probe", ROOT / "http-cancellation-hardware-probe.py")
    probe.DEADLINE_MS = 500  # The transport-only fixture has 800 ms simulated jobs.
    app = FastAPI()
    app.state.enable_server_load_tracking = True
    app.state.server_load_metrics = 0
    state = {"active": 0, "generated": 0, "finished": 0}

    async def long_job(_, raw_request):
        state["active"] += 1
        try:
            await asyncio.sleep(0.8)
            state["generated"] += 64
            state["finished"] += 1
            return JSONResponse({"ok": True})
        finally:
            await asyncio.sleep(0.005)
            state["active"] -= 1

    wrapped = api.with_cancellation(api.load_aware_call(long_job))

    @app.post("/v1/completions")
    async def completions(raw_request: Request):
        await raw_request.json()
        return await wrapped(None, raw_request)

    @app.post("/v1/chat/completions")
    async def chat(raw_request: Request):
        await raw_request.json()

        async def stream():
            await asyncio.sleep(0.005)
            state["generated"] += 32
            state["finished"] += 1
            yield (
                "data: "
                + json.dumps({"choices": [{"delta": {"content": "fixture response"}}]})
                + "\n\n"
            )
            yield (
                "data: "
                + json.dumps({"choices": [], "usage": {"completion_tokens": 32}})
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "deepseek-v4.1-flash"}]}

    @app.get("/metrics")
    async def metrics():
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(
            "\n".join(
                f"{name} {value}"
                for name, value in (
                    ("vllm:num_requests_running", state["active"]),
                    ("vllm:num_requests_waiting", 0),
                    ("vllm:generation_tokens_total", state["generated"]),
                    ("vllm:prompt_tokens_total", 0),
                    ("vllm:request_success_total", state["finished"]),
                    ("vllm:kv_cache_usage_perc", 0),
                )
            )
        )

    app.add_middleware(DeadlineMiddleware)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.001)
    try:
        output = Path("/results") / arm
        await probe.run(
            SimpleNamespace(
                base=f"http://127.0.0.1:{sock.getsockname()[1]}",
                arm=arm,
                boot="cpu-harness-check",
                contexts=[32],
                trials=1,
                output=output,
                model="deepseek-v4.1-flash",
            )
        )
        fault = json.loads((output / "server_cancel-32-0.json").read_text())
        abandoned = fault["native_generation_tokens"] - fault["useful_output_tokens"]
        assert abandoned == (0 if arm == "candidate" else 256), (arm, abandoned)
        assert (
            len(json.loads((output / "normal-c1.json").read_text())[0]["responses"])
            == 16
        )
        print(arm, "hardware client and middleware transport checks pass", flush=True)
    finally:
        server.should_exit = True
        await task


async def main():
    for arm in ("parent", "candidate"):
        await check(arm)


if __name__ == "__main__":
    asyncio.run(main())
