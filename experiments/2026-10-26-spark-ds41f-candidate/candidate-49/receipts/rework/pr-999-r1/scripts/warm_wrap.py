#!/usr/bin/env python3
"""Run a bench script with bench_2boot.bench_us preceded by a GPU clock warm-up.

On this GB10 the SM clock idles at ~200 MHz; a bench section that starts after a
CPU-only phase (kernel construction / TileLang compile) times its first, launch-bound
cells while the clock ramps, and the arm timed first (1107) absorbs most of it
(1107 reindex M=1: 130-140 us instead of 24 us). This wrapper keeps every bench
unchanged except that each bench_us call is preceded by >= 100 ms of bf16 matmul,
for every arm alike.

usage: warm_wrap.py <script.py> [args...]
"""
import runpy
import sys
import time

import torch

sys.path.insert(0, "/tmp")
import bench_2boot as b  # noqa: E402

_x = None


def gpu_warm(seconds=0.1):
    global _x
    if _x is None:
        _x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(4):
            _x @ _x
        torch.cuda.synchronize()


_orig = b.bench_us


def bench_us(fn, warmup=8, rep=25):
    gpu_warm()
    return _orig(fn, warmup=warmup, rep=rep)


b.bench_us = bench_us
script = sys.argv[1]
sys.argv = sys.argv[1:]
if script.endswith("bench_2boot.py"):
    b.main()
else:
    runpy.run_path(script, run_name="__main__")
