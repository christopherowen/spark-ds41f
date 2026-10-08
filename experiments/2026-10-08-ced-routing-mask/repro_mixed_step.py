#!/usr/bin/env python3
"""Reproduce #15's mixed step: a CED prefill admitted while other requests decode.

Five streams decode with DSpark while a sixth request arrives whose prompt is a
prefix-cached document plus an uncached tail longer than CED's 128-token window.
r6 and r6a stop at the TileKernels router's NaN/Inf assertion at the first such
step; with 0045 every request completes.

Standard library only, so it runs from any node:

    python3 repro_mixed_step.py --url http://10.0.1.71:8000 --rounds 3

Exit status 0 when every request in every round completes and the server is
still healthy afterwards, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request

PREFIX_PARAGRAPHS = 120  # about 3,300 tokens of cached prefix
TAIL_PARAGRAPHS = 12     # about 330 uncached tokens, well past the 128-token window


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
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0, "stream": True}
    finish = None
    tokens = 0
    try:
        with post(f"{base}/v1/completions", payload) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                choice = json.loads(line[6:])["choices"][0]
                tokens += 1
                if started is not None and tokens == 8:
                    started.set()
                finish = choice.get("finish_reason") or finish
        results[key] = ("ok" if finish in ("length", "stop") else f"finish={finish}", tokens)
    except (urllib.error.URLError, OSError, ValueError) as error:
        results[key] = (f"error: {error}", tokens)
    finally:
        if started is not None:
            started.set()


def healthy(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=30) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def run_round(base: str, model: str, number: int) -> bool:
    prefix = "\n".join(paragraph(i) for i in range(PREFIX_PARAGRAPHS))
    tail = "\n".join(paragraph(1000 * number + i) for i in range(TAIL_PARAGRAPHS))
    # Put the prefix in the cache first, so the sixth request schedules only its tail.
    with post(f"{base}/v1/completions", {"model": model, "prompt": prefix, "max_tokens": 1,
                                         "temperature": 0}) as response:
        response.read()
    results: dict[str, tuple[str, int]] = {}
    threads = []
    started = []
    for i in range(5):
        event = threading.Event()
        prompt = f"Round {number}, stream {i}: write a long, detailed maintenance log for a GPU cluster."
        thread = threading.Thread(target=stream, args=(base, model, prompt, 400, event, results, f"decode-{i}"))
        thread.start()
        threads.append(thread)
        started.append(event)
    for event in started:
        event.wait(timeout=120)
    time.sleep(0.5)  # every stream is decoding with DSpark now
    mixed = threading.Thread(target=stream, args=(base, model, prefix + "\n" + tail + "\nSummarize the readings.",
                                                  64, None, results, "mixed"))
    mixed.start()
    threads.append(mixed)
    for thread in threads:
        thread.join(timeout=900)
    alive = healthy(base)
    ok = alive and len(results) == 6 and all(status == "ok" for status, _ in results.values())
    detail = ", ".join(f"{key} {status} ({count} chunks)" for key, (status, count) in sorted(results.items()))
    print(f"round {number}: {'PASS' if ok else 'FAIL'}; server {'healthy' if alive else 'NOT healthy'}; {detail}",
          flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="API base, e.g. http://10.0.1.71:8000")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    model = model_name(base)
    passed = all(run_round(base, model, number) for number in range(1, args.rounds + 1))
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
