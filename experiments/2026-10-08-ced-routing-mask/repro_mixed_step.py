#!/usr/bin/env python3
"""Reproduce #15's mixed step: a CED prefill admitted while other requests decode.

Streams decode with DSpark (ignore_eos keeps them decoding) while requests
arrive whose prompts are a prefix-cached document plus an uncached tail longer
than CED's 128-token window. r6 and r6a can stop at the TileKernels router's
NaN/Inf assertion at such a step; with 0045 every request completes.

Each mixed request reports how many decode streams were still running when its
prefill finished (its first token). A round counts only if every mixed request
overlapped at least one decoding stream, so a pass means mixed steps happened.

Standard library only, so it runs from any node:

    python3 repro_mixed_step.py --url http://10.0.1.71:8000 --rounds 3

Exit status 0 when every request in every round completes, every mixed request
overlapped decoding streams and the server is still healthy afterwards; 1
otherwise.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
import uuid

PREFIX_PARAGRAPHS = 120  # about 3,300 tokens of cached prefix
TOPICS = (
    "write a long, detailed maintenance log for a GPU cluster",
    "write a Python module that parses and validates a network configuration file",
    "describe, step by step, how a city water system is inspected over one year",
    "write a C function library for ring buffers with thorough comments",
)


def paragraph(index: int) -> str:
    return (
        f"Section {index}. The cluster logs a reading every minute; reading {index} "
        f"records fan speed, board temperature and power for node {index % 4 + 1}, "
        f"and the operator compares it with the previous {index % 7 + 3} readings "
        "before deciding whether the node needs attention."
    )


def post(url: str, payload: dict, timeout: float = 600.0):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=timeout)


def model_name(base: str) -> str:
    with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as response:
        return json.load(response)["data"][0]["id"]


def stream(base: str, model: str, prompt: str, max_tokens: int, started: threading.Event | None,
           results: dict, key: str) -> None:
    """Record status, chunk count, and the first and last chunk times."""
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0, "stream": True, "ignore_eos": True}
    finish = None
    chunks = 0
    first = last = None
    try:
        with post(f"{base}/v1/completions", payload) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                choice = json.loads(line[6:])["choices"][0]
                last = time.monotonic()
                first = first or last
                chunks += 1
                if started is not None and chunks == 8:
                    started.set()
                finish = choice.get("finish_reason") or finish
        status = "ok" if finish == "length" else f"finish={finish}"
    except (urllib.error.URLError, OSError, ValueError) as error:
        status = f"error: {error}"
    finally:
        if started is not None:
            started.set()
    results[key] = (status, chunks, first, last)


def healthy(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=30) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def run_round(base: str, model: str, number: int, streams: int, mixed: int) -> bool:
    salt = uuid.uuid4().hex[:8]
    prefix = f"Round {number} ({salt}).\n" + "\n".join(paragraph(i) for i in range(PREFIX_PARAGRAPHS))
    # Put the prefix in the cache first, so each mixed request schedules only its tail.
    with post(f"{base}/v1/completions", {"model": model, "prompt": prefix, "max_tokens": 1,
                                         "temperature": 0}) as response:
        response.read()
    results: dict[str, tuple] = {}
    threads = []
    started = []
    for i in range(streams):
        event = threading.Event()
        prompt = f"Round {number} ({salt}), stream {i}: {TOPICS[i % len(TOPICS)]}."
        thread = threading.Thread(target=stream, args=(base, model, prompt, 600, event, results, f"decode-{i}"))
        thread.start()
        threads.append(thread)
        started.append(event)
    for event in started:
        event.wait(timeout=120)
    for j in range(mixed):
        time.sleep(0.75)  # spread the prefills across the decoding window
        tail = "\n".join(paragraph(1000 * number + 100 * j + i) for i in range(12 + 3 * j))
        thread = threading.Thread(
            target=stream,
            args=(base, model, prefix + "\n" + tail + "\nSummarize the readings.", 32, None, results, f"mixed-{j}"),
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(timeout=900)
    alive = healthy(base)
    complete = len(results) == streams + mixed and all(r[0] == "ok" for r in results.values())
    overlaps = []
    for j in range(mixed):
        first = results.get(f"mixed-{j}", (None, 0, None, None))[2]
        overlaps.append(sum(
            1 for i in range(streams)
            if first is not None and (results.get(f"decode-{i}", (None, 0, None, None))[3] or 0) > first
        ))
    overlapped = all(count > 0 for count in overlaps)
    ok = alive and complete and overlapped
    failures = ", ".join(f"{key} {r[0]}" for key, r in sorted(results.items()) if r[0] != "ok") or "none"
    print(f"round {number}: {'PASS' if ok else 'FAIL'}; server {'healthy' if alive else 'NOT healthy'}; "
          f"streams decoding at each mixed prefill: {overlaps}; failures: {failures}", flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="API base, e.g. http://10.0.1.71:8000")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--streams", type=int, default=8, help="decoding streams per round")
    parser.add_argument("--mixed", type=int, default=4, help="prefix-cached requests per round")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    model = model_name(base)
    passed = all(run_round(base, model, number, args.streams, args.mixed) for number in range(1, args.rounds + 1))
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
