#!/usr/bin/env python3
"""Placement, round 2: is the fast buffer a place in memory or a place in the order?

Round 1 (placement.py) found one buffer per scenario streaming at about 265 GB/s and the
rest at about 222 GB/s. It was always the buffer whose CUDA graph was built last, even
the last slice of a single allocation, and the same address was fast in one scenario and
slow in the next. Here:
- A: four buffers, their graphs built in order and timed in a shuffled order; then the
  graphs rebuilt in reverse order and timed again. A fast buffer that moves with the build
  order is an ordering effect; one that stays is placement.
- B: the same four buffers timed without graphs (20 eager launches between events), in a
  shuffled order.
- C: one 1 GiB allocation in twenty 48 MiB slices, each timed without graphs, five
  shuffled passes: a map of the allocation.
One JSON line per measurement.
"""
import json
import random
import statistics

import torch

DEVICE = torch.device("cuda")
N = 96 * 129280
CALLS = 20


def graph(x):
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    for _ in range(2):
        x.mul_(0.999)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(CALLS):
            x.mul_(0.999)
    return g


def timed(run):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS


def eager(x):
    def run():
        for _ in range(CALLS):
            x.mul_(0.999)
    return run


def emit(**row):
    print(json.dumps(row), flush=True)


def shuffled_times(runs, passes, rng):
    times = [[] for _ in runs]
    for _ in range(passes):
        order = list(range(len(runs)))
        rng.shuffle(order)
        for i in order:
            times[i].append(timed(runs[i]))
    return [statistics.median(t) for t in times]


def main():
    rng = random.Random(7)
    torch.manual_seed(0)
    buffers = [torch.randn(N, device=DEVICE) for _ in range(4)]
    for label, order in (("A: graphs built 0..3", [0, 1, 2, 3]),
                         ("A: graphs built 3..0", [3, 2, 1, 0])):
        graphs = {i: graph(buffers[i]) for i in order}
        medians = shuffled_times([graphs[i].replay for i in range(4)], 12, rng)
        for i, us in enumerate(medians):
            emit(scenario=label, buffer=i, built=order.index(i), us=round(us, 2),
                 gb_s=round(2 * N * 4 / us / 1e3, 1), address=hex(buffers[i].data_ptr()))
        del graphs
    medians = shuffled_times([eager(x) for x in buffers], 12, rng)
    for i, us in enumerate(medians):
        emit(scenario="B: eager", buffer=i, us=round(us, 2), gb_s=round(2 * N * 4 / us / 1e3, 1),
             address=hex(buffers[i].data_ptr()))
    del buffers
    torch.cuda.empty_cache()
    slice_elems = (48 << 20) // 4
    big = torch.randn(20 * slice_elems, device=DEVICE)
    slices = [big[i * slice_elems:(i + 1) * slice_elems] for i in range(20)]
    medians = shuffled_times([eager(x) for x in slices], 5, rng)
    for i, us in enumerate(medians):
        emit(scenario="C: slices of 1 GiB, eager", slice=i, us=round(us, 2),
             gb_s=round(2 * slice_elems * 4 / us / 1e3, 1), address=hex(slices[i].data_ptr()))


if __name__ == "__main__":
    main()
