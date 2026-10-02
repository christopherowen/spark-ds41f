#!/usr/bin/env python3
"""4 KiB O_DIRECT random-read latency on the model's NVMe, at queue depth 1 and in bursts.

usage: latency.py FILE [--seconds S] [--burst N] [--bursts K]
QD1: one read at a time for S seconds; reports per-read p50/p90/p99 (us).
Burst: N reads issued together from N threads (like a step's Engram rows), K times;
reports burst completion time p50/p90/p99 (us). Read-only; no page cache (O_DIRECT).
"""
import json
import mmap
import os
import random
import sys
import threading
import time

path = sys.argv[1]
arg = lambda n, d: type(d)(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d  # noqa: E731
SECONDS, BURST, BURSTS = arg("--seconds", 5.0), arg("--burst", 32), arg("--bursts", 400)
BLOCK = 4096
fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
blocks = os.fstat(fd).st_size // BLOCK - 1
rng = random.Random(0)


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 1)


buf = mmap.mmap(-1, BLOCK)
lat = []
end = time.perf_counter() + SECONDS
while time.perf_counter() < end:
    off = rng.randrange(blocks) * BLOCK
    t = time.perf_counter_ns()
    os.preadv(fd, [buf], off)
    lat.append((time.perf_counter_ns() - t) / 1000)

bufs = [mmap.mmap(-1, BLOCK) for _ in range(BURST)]
go = [threading.Barrier(BURST + 1) for _ in range(2)]
offsets = [0] * BURST


def worker(i):
    while True:
        go[0].wait()
        if offsets[i] < 0:
            return
        os.preadv(fd, [bufs[i]], offsets[i])
        go[1].wait()


threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(BURST)]
for th in threads:
    th.start()
burst = []
for _ in range(BURSTS):
    for i in range(BURST):
        offsets[i] = rng.randrange(blocks) * BLOCK
    t = time.perf_counter_ns()
    go[0].wait()
    go[1].wait()
    burst.append((time.perf_counter_ns() - t) / 1000)
for i in range(BURST):
    offsets[i] = -1
go[0].wait()
print(json.dumps({"host": os.uname().nodename, "file": os.path.basename(path), "qd1_reads": len(lat),
                  "qd1_us": {"p50": pct(lat, .5), "p90": pct(lat, .9), "p99": pct(lat, .99)},
                  "burst": BURST, "bursts": BURSTS,
                  "burst_us": {"p50": pct(burst, .5), "p90": pct(burst, .9), "p99": pct(burst, .99)}}))
