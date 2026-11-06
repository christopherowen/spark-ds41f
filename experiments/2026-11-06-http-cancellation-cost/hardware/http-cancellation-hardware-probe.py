"""Bounded real-engine cancellation probes; run only inside an owned window."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import httpx

PROMPTS = [
    "Review a Python payment webhook handler: explain how to make retries idempotent and test a duplicate delivery.",
    "Plan a minimal SQL migration that adds a nullable customer locale without locking a large orders table.",
    "Explain how to debug a service whose HTTP health check passes but whose background jobs stop processing.",
    "Design tests for a timezone conversion function around the autumn daylight saving transition.",
    "Review a cache invalidation design for product prices and explain how to test concurrent updates.",
    "Describe a safe rollout for replacing an API field while old mobile clients still send it.",
    "Investigate an intermittent queue consumer timeout and list the evidence needed to distinguish CPU from I/O waits.",
    "Write a concise plan for testing that an authorization check rejects access to another customer's invoice.",
    "Explain how to reproduce and fix a lost update when two workers change the same account balance.",
    "Review a file upload service and propose tests for partial uploads and interrupted connections.",
    "Plan a rollback for a database migration that accidentally changed a timestamp's timezone.",
    "Explain how to measure and reduce an N plus one query pattern in an order history endpoint.",
    "Design a small test for a background scheduler that must not run the same task twice.",
    "Propose an API pagination contract that remains consistent when new records arrive.",
    "Explain how to debug a memory leak in a long lived Python worker without restarting production.",
    "Describe tests for a streaming parser when UTF-8 characters are split across network packets.",
]
DEADLINE_MS = 2000

METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:request_success_total",
    "vllm:kv_cache_usage_perc",
)


async def metrics(client):
    response = await client.get("/metrics")
    response.raise_for_status()
    values = {name: 0.0 for name in METRICS}
    found = set()
    for line in response.text.splitlines():
        if line.startswith("#") or not line:
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in values:
            values[name] += float(line.rsplit(" ", 1)[1])
            found.add(name)
    for required in METRICS[:3]:
        if required not in found:
            raise RuntimeError(f"required native metric missing: {required}")
    return {"wall_time": time.time(), **values}


async def idle(client, timeout=180):
    start = time.monotonic()
    observations = []
    while True:
        point = await metrics(client)
        observations.append(point)
        if point[METRICS[0]] == 0 and point[METRICS[1]] == 0:
            return observations
        if time.monotonic() - start > timeout:
            raise RuntimeError("engine did not become idle within the bounded wait")
        await asyncio.sleep(0.1)


async def short_request(client, model, index, case):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}],
        "max_tokens": 32,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
        "cache_salt": f"http-cancellation-{case}-short-{index}",
    }
    start = time.monotonic()
    first = None
    pieces = []
    tokens = None
    async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            if chunk.get("usage"):
                tokens = chunk["usage"]["completion_tokens"]
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    first = first or time.monotonic()
                    pieces.append(content)
    end = time.monotonic()
    if first is None or tokens is None:
        raise RuntimeError("short stream did not report text and native token usage")
    return {
        "index": index,
        "prompt_sha256": hashlib.sha256(
            PROMPTS[index % len(PROMPTS)].encode()
        ).hexdigest(),
        "ttft_ms": (first - start) * 1000,
        "completion_ms": (end - start) * 1000,
        "output_tokens": tokens,
        "text_sha256": hashlib.sha256("".join(pieces).encode()).hexdigest(),
    }


async def wave(client, model, mode, length, trial, output):
    case = f"{mode}-{length}-{trial}"
    await idle(client)
    before = await metrics(client)
    stop_sampling = asyncio.Event()
    samples = []

    async def sample():
        while not stop_sampling.is_set():
            samples.append(await metrics(client))
            await asyncio.sleep(0.05)

    async def long_request(index):
        # Deterministic synthetic long context; useful work uses distinct agent prompts.
        ids = [1000 + index] + [
            200 + ((j * 17 + index * 31) % 2000) for j in range(length - 1)
        ]
        payload = {
            "model": model,
            "prompt": ids,
            "max_tokens": 64,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
            "cache_salt": f"http-cancellation-{case}-long-{index}",
        }
        headers = (
            {"x-experiment-deadline-ms": str(DEADLINE_MS)}
            if mode == "server_cancel"
            else {}
        )
        response = await client.post("/v1/completions", json=payload, headers=headers)
        if mode == "server_cancel":
            if response.status_code != 504 or response.json() != {
                "error": "experiment request deadline"
            }:
                raise RuntimeError(
                    f"fault did not trigger: {response.status_code}: {response.text[:200]}"
                )
        else:
            response.raise_for_status()
        return response.status_code

    sampler = asyncio.create_task(sample())
    parents = [asyncio.create_task(long_request(i)) for i in range(4)]
    start = time.monotonic()
    try:
        if mode == "disconnect":
            await asyncio.sleep(DEADLINE_MS / 1000)
            for task in parents:
                task.cancel()
        outcomes = await asyncio.gather(*parents, return_exceptions=True)
        if mode == "server_cancel":
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
        elif not all(
            isinstance(outcome, asyncio.CancelledError) for outcome in outcomes
        ):
            raise RuntimeError("one long request ended before client cancellation")
        abandoned = await metrics(client)
        useful = await asyncio.gather(
            *(short_request(client, model, i, case) for i in range(8))
        )
        useful_finished = time.monotonic()
        quiet = await idle(client)
        final = quiet[-1]
        record = {
            "case": case,
            "mode": mode,
            "context_tokens_each": length,
            "long_requests": 4,
            "deadline_ms": DEADLINE_MS,
            "long_max_output_tokens": 64,
            "useful": useful,
            "before": before,
            "after_abandonment": abandoned,
            "after_idle": final,
            "wave_to_useful_completion_ms": (useful_finished - start) * 1000,
            "wave_to_idle_ms": (time.monotonic() - start) * 1000,
            "native_generation_tokens": final[METRICS[2]] - before[METRICS[2]],
            "useful_output_tokens": sum(r["output_tokens"] for r in useful),
            "native_active_observed": any(p[METRICS[0]] > 0 for p in samples),
            "samples": samples,
            "idle_samples": quiet,
        }
        (output / f"{case}.json").write_text(json.dumps(record, indent=2) + "\n")
        if not record["native_active_observed"]:
            raise RuntimeError(
                "no native running request was observed; cannot claim engine work was cancelled"
            )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in record.items()
                    if k not in ("samples", "idle_samples", "useful")
                }
            ),
            flush=True,
        )
    except BaseException as error:
        (output / f"{case}-failure.json").write_text(
            json.dumps(
                {
                    "case": case,
                    "error": repr(error),
                    "before": before,
                    "samples": samples,
                },
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        stop_sampling.set()
        await asyncio.gather(sampler, return_exceptions=True)
        for task in parents:
            if not task.done():
                task.cancel()
        await asyncio.gather(*parents, return_exceptions=True)


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "identity.json").write_text(
        json.dumps(
            {
                "arm": args.arm,
                "boot": args.boot,
                "base_url": args.base,
                "harness_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "contexts": args.contexts,
                "trials": args.trials,
                "input_policy": "distinct synthetic long token IDs, distinct useful agent prompts, unique fixed cache salts",
            },
            indent=2,
        )
        + "\n"
    )
    async with httpx.AsyncClient(
        base_url=args.base, timeout=180, limits=httpx.Limits(max_connections=32)
    ) as client:
        response = await client.get("/v1/models")
        response.raise_for_status()
        ids = [item["id"] for item in response.json()["data"]]
        if args.model not in ids:
            raise RuntimeError(f"required stock model missing: {args.model}")
        model = args.model
        await idle(client)
        await short_request(client, model, 0, "warmup")
        for length in args.contexts:
            for mode in ("disconnect", "server_cancel"):
                for trial in range(args.trials):
                    await wave(client, model, mode, length, trial, args.output)
        for concurrency in (1, 8, 16):
            rows = []
            for trial in range(args.trials):
                await idle(client)
                start = time.monotonic()
                samples = []
                for start_index in range(0, len(PROMPTS), concurrency):
                    samples.extend(
                        await asyncio.gather(
                            *(
                                short_request(
                                    client, model, i, f"normal-{concurrency}-{trial}"
                                )
                                for i in range(
                                    start_index,
                                    min(start_index + concurrency, len(PROMPTS)),
                                )
                            )
                        )
                    )
                rows.append(
                    {
                        "trial": trial,
                        "concurrency": concurrency,
                        "wall_ms": (time.monotonic() - start) * 1000,
                        "responses": samples,
                    }
                )
            (args.output / f"normal-c{concurrency}.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--boot", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 131072])
    parser.add_argument("--trials", type=int, default=6)
    asyncio.run(run(parser.parse_args()))
