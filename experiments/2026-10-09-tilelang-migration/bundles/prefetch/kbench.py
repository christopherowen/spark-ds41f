"""Shared timing and error helpers of the port benches (copied into each bundle).

Time per call under CUDA graphs: warm repeats the call inside one replay with the
weights in L2; cold evicts L2 with a 128 MiB read before a one-call replay. Both sides
of a comparison pay the same graph launch.
"""
import statistics
import sys

import torch

sys.path.insert(0, "/opt/spark3/candidate/vllm")

DEVICE = torch.device("cuda")
# The serving decode capture sizes up to 16 streams x 6 tokens, and prefill chunks.
DECODE = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 72, 80, 96)
CAPACITY = 8192
PREFILL = (512, 2048, CAPACITY)
SLOWER = 1.03  # a port may be at most 3% slower per call, warm or cold
_EVICT = torch.ones(128 << 20, dtype=torch.uint8, device=DEVICE).view(torch.int32)


def graph(fn, calls: int) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    return g


def _replay(g: torch.cuda.CUDAGraph, cold: bool) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    if cold:
        _EVICT.max()
    start.record()
    g.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000


def per_call(fn, rows: int) -> tuple[float, float]:
    """Median microseconds per call of ``fn`` (one call at ``rows``), warm and cold."""
    calls = 20 if rows <= DECODE[-1] else 3
    warm, cold = graph(fn, calls), graph(fn, 1)
    _replay(warm, False)
    _replay(cold, False)
    w = statistics.median(_replay(warm, False) / calls for _ in range(15))
    c = statistics.median(_replay(cold, True) for _ in range(25))
    return w, c


def errors(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Max and RMS error of ``out`` against an FP64 reference."""
    diff = out.double() - ref
    return diff.abs().max().item(), diff.pow(2).mean().sqrt().item()


def no_worse(port: tuple[float, float], base: tuple[float, float]) -> bool:
    """The port's (max, RMS) error is within 10% / 5% of the replaced kernel's."""
    return port[0] <= base[0] * 1.10 + 1e-12 and port[1] <= base[1] * 1.05 + 1e-12


def timing_line(rows: int, base: tuple[float, float], port: tuple[float, float]) -> str:
    return (f"  rows {rows:5d}  b12x {base[0]:8.2f} / {base[1]:8.2f}"
            f"  tilelang {port[0]:8.2f} / {port[1]:8.2f}"
            f"  ratio {port[0] / base[0]:.3f} / {port[1] / base[1]:.3f}")


def slower(base: tuple[float, float], port: tuple[float, float]) -> bool:
    return port[0] > base[0] * SLOWER or port[1] > base[1] * SLOWER
