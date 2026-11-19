#!/usr/bin/env python3
"""The screen's single stream under the torch profiler: a few of servebench's prose prompts,
one after the other, timed and then traced.

usage: profile_decode.py BASE_URL [--tokens 256] [--requests 3]

A profiler-enabled arm must be serving. The requests are the single-stream screen's
(servebench's default prompts, temperature 0, a fixed length, thinking off):

1. warm: the first prompt, not measured;
2. timed: the next --requests prompts, not profiled: the engine's steps and tokens over
   the span, from the server's counters;
3. traced: the same prompts again between /start_profile and /stop_profile. Every rank
   writes one trace.

One JSON line per phase.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from spark_ds41f import sonar  # noqa: E402

prom = sonar.module("prom")
default_prompts = sonar.module("prompts").default_prompts


def option(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


BASE = sys.argv[1].rstrip("/")
TOKENS = int(option("--tokens", "256"))
REQUESTS = int(option("--requests", "3"))


def post(path, body=None, timeout=900):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read()
        return json.loads(text) if text.strip().startswith(b"{") else None


def emit(**record):
    print(json.dumps(record), flush=True)


with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]
PROMPTS = default_prompts()


def chat(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt["text"]}], "max_tokens": TOKENS,
            "min_tokens": TOKENS, "ignore_eos": True, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}}
    return post("/v1/chat/completions", body)["usage"]


def run(phase, prompts):
    before, start = prom.scrape(BASE, prom.VLLM) or {}, time.time()
    usage = [chat(prompt) for prompt in prompts]
    seconds, after = time.time() - start, prom.scrape(BASE, prom.VLLM) or {}
    delta = {key: after.get(key, 0) - before.get(key, 0) for key in ("steps", "step_tokens", "generation_tokens", "drafts",
                                                                     "draft_tokens", "accepted_tokens")}
    emit(phase=phase, prompts=[prompt["id"] for prompt in prompts], seconds=round(seconds, 3),
         completion_tokens=sum(u["completion_tokens"] for u in usage), **delta,
         step_ms=round(1e3 * seconds / delta["steps"], 3) if delta["steps"] else None,
         tokens_per_step=round(delta["generation_tokens"] / delta["steps"], 4) if delta["steps"] else None)


run("warm", PROMPTS[:1])
window = PROMPTS[1 : 1 + REQUESTS]
run("timed", window)
post("/start_profile")
run("traced", window)
post("/stop_profile")
emit(phase="done")
