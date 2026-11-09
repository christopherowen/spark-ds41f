#!/usr/bin/env python3
"""CUDA-graph companion of PR #991's bench_hw_e2m1x2_quant.py.

Runs the PR bench unchanged (same cases, same inputs, same torch.equal checks,
same eager AB/BA timing), and after each case's eager timing also times the two
arms with their launches captured in CUDA graphs (``repeats`` launches per graph,
80 interleaved AB/BA graph replays). Decode serves these kernels from CUDA
graphs, where the host launch path is not on the critical path; the eager bench
at small M measures mostly that host path.

usage: bench_graph_companion.py --out /out/x.json [bench args]
"""

import importlib.util
import sys

import torch

spec = importlib.util.spec_from_file_location("bench", "/tmp/bench_hw_e2m1x2_quant.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

_eager = bench._paired_bench


def _graph_of(fn, reps):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    return g


def _paired_both(fn_sw, fn_hw, *, warmup=40, rounds=80, repeats_per_round=25):
    res = _eager(fn_sw, fn_hw, warmup=warmup, rounds=rounds, repeats_per_round=repeats_per_round)
    g_sw = _graph_of(fn_sw, repeats_per_round)
    g_hw = _graph_of(fn_hw, repeats_per_round)
    gr = _eager(g_sw.replay, g_hw.replay, warmup=10, rounds=rounds, repeats_per_round=1)
    # per-launch microseconds
    for arm in ("sw", "hw"):
        for k in list(gr[arm]):
            gr[arm][k] = round(gr[arm][k] / repeats_per_round, 4)
    gr["delta_median_us"] = round(gr["hw"]["median_us"] - gr["sw"]["median_us"], 4)
    gr["launches_per_graph"] = repeats_per_round
    res["graph"] = gr
    del g_sw, g_hw
    return res


bench._paired_bench = _paired_both

if __name__ == "__main__":
    sys.argv = ["bench_hw_e2m1x2_quant.py", *sys.argv[1:]]
    bench.main()
