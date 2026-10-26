#!/usr/bin/env python3
"""One stream's decode steps on every rank, and the engine grid's dummy steps at its rows
(a profiler-enabled arm with VLLM_SERVER_DEV_MODE=1 must be serving).

usage: profile_decode.py BASE_URL [--tokens N] [--replays R]

Three torch-profiler windows, one trace a rank each, in this order:
1. c1-greedy: one distinct prompt alone, temperature 0;
2. c1-sampled: another alone at temperature 1 (seed 7), what most clients send;
3. grid-c1: the engine grid's dummy decode step for one request at 3 and 6 rows (draft
   depths 2 and 5) at 4K context, R replays each: the step the grid times, on the same
   boot and ranks, so a serving step can be diffed against it kernel by kernel.
Each request is warmed first by one of its own kind on another prompt. Thinking is off and
every request has a fresh cache salt. One JSON line a window: its name, wall-clock start
and end, completion tokens (requests) or the grid's step times (grid).
"""
import json
import sys
import time
import uuid
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from distinct_streams import PROMPTS, model_id  # noqa: E402
from servebench import engine  # noqa: E402

BASE = sys.argv[1].rstrip("/")
TOKENS = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 128
REPLAYS = int(sys.argv[sys.argv.index("--replays") + 1]) if "--replays" in sys.argv else 20
MODEL = model_id(BASE)


def post(path, body=None, timeout=900):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read()
        return json.loads(text) if text.startswith(b"{") else None


def chat(prompt, **fields):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": TOKENS,
            "chat_template_kwargs": {"thinking": False}, "cache_salt": uuid.uuid4().hex, **fields}
    return post("/v1/chat/completions", body)


def profiled(name, work):
    post("/start_profile")
    began = time.time()
    try:
        result = work()
    finally:
        ended = time.time()
        post("/stop_profile")
        time.sleep(30)  # the ranks write their traces before the next window starts
    return {"window": name, "start": round(began, 3), "end": round(ended, 3), **result}


chat(PROMPTS[1], temperature=0)
print(json.dumps(profiled("c1-greedy", lambda: {
    "completion_tokens": chat(PROMPTS[0], temperature=0)["usage"]["completion_tokens"]})), flush=True)
chat(PROMPTS[3], temperature=1.0, seed=7)
print(json.dumps(profiled("c1-sampled", lambda: {
    "completion_tokens": chat(PROMPTS[2], temperature=1.0, seed=7)["usage"]["completion_tokens"]})), flush=True)

shapes = [{"kind": "decode", "num_reqs": 1, "num_tokens": rows, "context_len": 4096} for rows in (3, 6)]
engine.run(BASE, shapes, replays=2, warmup=1)


def grid():
    result = engine.run(BASE, shapes, replays=REPLAYS, warmup=2)
    return {"grid": engine.summarize(result)}


print(json.dumps(profiled("grid-c1", grid)), flush=True)
