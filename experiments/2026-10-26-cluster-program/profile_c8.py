#!/usr/bin/env python3
"""Eight streams' decode steps on every rank, and the engine grid's dummy steps at their
rows (a profiler-enabled arm with VLLM_SERVER_DEV_MODE=1 must be serving).

usage: profile_c8.py BASE_URL [--streams N] [--tokens T] [--replays R]

Two torch-profiler windows, one trace a rank each, in this order:
1. c8-greedy: N distinct prompts at once (each its own text, so the streams route to
   their own experts), temperature 0;
2. grid-c8: the engine grid's dummy decode step for N requests at 2 and 3 rows a request
   (draft depths 1 and 2, what adaptive verification runs under load) at 4K context, R
   replays each.
The streams are warmed first by N others at once. Thinking is off and every request has
a fresh cache salt. One JSON line a window: its name, wall-clock start and end, completion
tokens (requests) or the grid's step times (grid).
"""
import json
import sys
import threading
import time
import uuid
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from distinct_streams import PROMPTS, model_id  # noqa: E402
from servebench import engine  # noqa: E402

BASE = sys.argv[1].rstrip("/")
STREAMS = int(sys.argv[sys.argv.index("--streams") + 1]) if "--streams" in sys.argv else 8
TOKENS = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 128
REPLAYS = int(sys.argv[sys.argv.index("--replays") + 1]) if "--replays" in sys.argv else 20
MODEL = model_id(BASE)
if 2 * STREAMS > len(PROMPTS):
    raise SystemExit(f"--streams {STREAMS}: {len(PROMPTS)} prompts hold two sets of at most {len(PROMPTS) // 2}")


def post(path, body=None, timeout=900):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read()
        return json.loads(text) if text.startswith(b"{") else None


def chat(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": TOKENS,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}, "cache_salt": uuid.uuid4().hex}
    return post("/v1/chat/completions", body)


def together(prompts):
    results = [None] * len(prompts)

    def run(index):
        results[index] = chat(prompts[index])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return {"completion_tokens": sum(r["usage"]["completion_tokens"] for r in results if r),
            "requests": sum(1 for r in results if r)}


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


together(PROMPTS[STREAMS:2 * STREAMS])
print(json.dumps(profiled("c8-greedy", lambda: together(PROMPTS[:STREAMS]))), flush=True)

shapes = [{"kind": "decode", "num_reqs": STREAMS, "num_tokens": STREAMS * rows, "context_len": 4096}
          for rows in (2, 3)]
engine.run(BASE, shapes, replays=2, warmup=1)
print(json.dumps(profiled("grid-c8", lambda: {"grid": engine.summarize(
    engine.run(BASE, shapes, replays=REPLAYS, warmup=2))})), flush=True)
