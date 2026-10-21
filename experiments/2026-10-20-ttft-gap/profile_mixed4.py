#!/usr/bin/env python3
"""Profile the step that carries a short prompt beside three decoding streams (a
profiler-enabled arm must be serving; its schedule records the steps after
/start_profile).

usage: profile_mixed4.py BASE_URL [--repo PATH] [--settle S]

Three of servebench's default prompts decode (thinking off, greedy, 512 tokens each); after
--settle seconds the profiler starts and a fourth prompt of about 20 tokens arrives, so the
recorded steps hold plain three-stream steps and the four-stream step that carries the
prompt. Prints one JSON line with a "summary" key: the short request's TTFT and prompt
tokens. Stdlib only.
"""
import argparse
import json
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path


def post(base, path, body=None):
    request = urllib.request.Request(base + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def stream(base, body):
    body = dict(body, stream=True, stream_options={"include_usage": True, "continuous_usage_stats": True},
                cache_salt=uuid.uuid4().hex)
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start, first, prompt_tokens = time.monotonic(), None, None
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            usage = json.loads(line[5:]).get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            if first is None and usage.get("completion_tokens", 0) > 0:
                first = time.monotonic()
    return {"ttft": (first or time.monotonic()) - start, "prompt_tokens": prompt_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--settle", type=float, default=4.0)
    a = ap.parse_args()
    base = a.base.rstrip("/")
    sys.path.insert(0, str(Path(a.repo).resolve()))
    from servebench.prompts import default_prompts

    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        model = json.load(response)["data"][0]["id"]
    prompts = default_prompts()

    def body(i, tokens):
        return {"model": model, "messages": [{"role": "user", "content": prompts[i]["text"]}],
                "max_tokens": tokens, "min_tokens": tokens, "ignore_eos": True, "temperature": 0,
                "chat_template_kwargs": {"thinking": False}}

    stream(base, body(0, 8))  # warm-up
    decoders = [threading.Thread(target=stream, args=(base, body(i, 512))) for i in (1, 2, 3)]
    for t in decoders:
        t.start()
    time.sleep(a.settle)
    post(base, "/start_profile")
    short = stream(base, body(4, 16))
    time.sleep(2.0)
    post(base, "/stop_profile")
    for t in decoders:
        t.join()
    print(json.dumps({"summary": {"short_ttft_ms": round(1000 * short["ttft"], 1),
                                  "prompt_tokens": short["prompt_tokens"]}}), flush=True)


if __name__ == "__main__":
    main()
