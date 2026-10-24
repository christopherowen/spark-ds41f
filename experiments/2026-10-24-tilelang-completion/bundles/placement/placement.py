#!/usr/bin/env python3
"""Why one of two same-size buffers streams 18% slower (round 4b: Triton's temperature
kernel against itself took 458.8 us on a 96 x 129280 FP32 buffer and 387.8 us on its
clone, allocated right after it, in every one of eight fresh pairs).

Each scenario allocates buffers one way and times an in-place streaming pass over each
(``x.mul_(0.7)``: read and write every element) in a CUDA graph, buffers in alternating
order. Each buffer's address is printed (its offset into 2 MiB and 64 KiB pages and
from the previous buffer) with its time, so the slow buffers can be matched to how and
where they were allocated. One JSON line per buffer and scenario.
"""
import json
import statistics

import torch

DEVICE = torch.device("cuda")
N = 96 * 129280  # round 4b's 96-token logits, FP32 (49.6 MB)
CALLS = 20
SAMPLES = 6


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


def us(g):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS


def time_buffers(scenario, buffers):
    graphs = [graph(x) for x in buffers]
    times = [[] for _ in buffers]
    for sample in range(SAMPLES):
        order = range(len(buffers)) if sample % 2 == 0 else reversed(range(len(buffers)))
        for i in order:
            times[i].append(us(graphs[i]))
    previous = None
    for i, x in enumerate(buffers):
        ptr = x.data_ptr()
        print(json.dumps({
            "scenario": scenario,
            "buffer": i,
            "us": round(statistics.median(times[i]), 2),
            "gb_s": round(2 * x.numel() * x.element_size() / statistics.median(times[i]) / 1e3, 1),
            "address": hex(ptr),
            "mod_2mib": ptr % (2 << 20),
            "mod_64kib": ptr % (64 << 10),
            "from_previous_mib": None if previous is None else round((ptr - previous) / 2**20, 3),
        }), flush=True)
        previous = ptr
    del graphs


def main():
    torch.manual_seed(0)
    # Round 4b's pair: randn, then a clone.
    first = torch.randn(N, device=DEVICE)
    time_buffers("randn then clone", [first, first.clone()])
    torch.cuda.empty_cache()
    # The same with the allocation order reversed: an empty buffer first, filled after.
    second = torch.empty(N, device=DEVICE)
    first = torch.randn(N, device=DEVICE)
    second.copy_(first)
    time_buffers("empty, then randn, copied in", [first, second])
    del first, second
    torch.cuda.empty_cache()
    # Eight in a row.
    buffers = [torch.randn(N, device=DEVICE) for _ in range(8)]
    time_buffers("eight in a row", buffers)
    del buffers
    torch.cuda.empty_cache()
    # Sizes rounded up to whole 2 MiB pages.
    whole = -(-N * 4 // (2 << 20)) * (2 << 20) // 4
    buffers = [torch.randn(whole, device=DEVICE) for _ in range(4)]
    time_buffers("whole 2 MiB pages", buffers)
    del buffers
    torch.cuda.empty_cache()
    # One allocation sliced into four buffers.
    big = torch.randn(4 * N, device=DEVICE)
    time_buffers("slices of one allocation", [big[i * N:(i + 1) * N] for i in range(4)])
    del big
    torch.cuda.empty_cache()
    # Reused blocks: free a pair and allocate again from the cache.
    a, b = torch.randn(N, device=DEVICE), torch.randn(N, device=DEVICE)
    del a
    c = torch.randn(N, device=DEVICE)
    time_buffers("a block reused from the cache", [c, b])
    print(json.dumps({"memory": torch.cuda.memory_stats().get("num_alloc_retries"),
                      "reserved_mib": torch.cuda.memory_reserved() >> 20}), flush=True)


if __name__ == "__main__":
    main()
