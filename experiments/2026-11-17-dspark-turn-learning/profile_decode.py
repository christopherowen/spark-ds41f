#!/usr/bin/env python3
"""The engine grid's decode steps at a few shapes, timed and then profiled, one profiler window a shape.

usage: profile_decode.py BASE_URL [--shapes 16x96,16x48,1x3] [--context 4096] [--replays 20]

A profiler-enabled arm with VLLM_SERVER_DEV_MODE=1 must be serving. For each shape (requests x rows at
the context length), in order:

1. grid: the engine grid's fixed decode step, not profiled (the boot's own grid times for the shape);
2. grid-profiled: the same under the torch profiler, its own window.

One JSON line a phase. Every profiler window writes one trace a rank, in the shapes' order.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from spark_ds41f import sonar  # noqa: E402

engine = sonar.module("engine")


def option(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


BASE = sys.argv[1].rstrip("/")
SHAPES = [tuple(int(v) for v in s.split("x")) for s in option("--shapes", "16x96,16x48,1x3").split(",")]
CONTEXT = int(option("--context", "4096"))
REPLAYS = int(option("--replays", "20"))


def post(path, body=None, timeout=900):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def emit(**record):
    print(json.dumps(record), flush=True)


for window, (requests, rows) in enumerate(SHAPES):
    shape = [{"kind": "decode", "num_reqs": requests, "num_tokens": rows, "context_len": CONTEXT}]
    emit(phase="grid", shape=f"{requests}x{rows}", context_len=CONTEXT,
         grid=engine.summarize(engine.run(BASE, shape, replays=REPLAYS, warmup=2)))
    post("/start_profile")
    began = time.time()
    try:
        result = engine.summarize(engine.run(BASE, shape, replays=REPLAYS, warmup=2))
    finally:
        ended = time.time()
        post("/stop_profile")
    emit(phase="grid-profiled", shape=f"{requests}x{rows}", context_len=CONTEXT, window=window,
         replays=REPLAYS, warmup=2, start=round(began, 3), end=round(ended, 3), grid=result)
    time.sleep(20)  # the window's traces are written after /stop_profile
