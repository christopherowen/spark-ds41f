"""Shared timing and error helpers of the port benches (copied into each bundle).

Time per call under CUDA graphs: warm repeats the call inside one replay with the
weights in L2; cold evicts L2 with a 128 MiB read before a one-call replay. Both sides
of a comparison pay the same graph launch.

Repeatability: a port must give the same bits on every run (temperature-0 serving
replays the same graphs), so each bench replays its kernels' graphs, warm and cold,
against an eager call.

Buckets: a port may run its own schedule per row bucket (decode, mid, prefill: tiles,
split over CTAs, fusion), so a decode win need not cost prefill. A row's arithmetic may
not change with the bucket (shard count, reduction order, rounding points): every size
must give its rows' bits of the full batch (``batch_variance``), and the floor is judged
per bucket (``bucket_lines``), so a win in one bucket cannot hide a loss in another.
"""
import statistics
import sys
from collections.abc import Callable

import torch

sys.path.insert(0, "/opt/spark3/candidate/vllm")

# With its index: B12X compares a plan's device with its session's (cuda:0).
DEVICE = torch.device("cuda", torch.cuda.current_device())
# The serving decode capture sizes up to 16 streams x 6 tokens, and prefill chunks.
DECODE = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 72, 80, 96)
CAPACITY = 8192
PREFILL = (512, 2048, CAPACITY)
REPEAT = (1, 6, 16, 48, 96, 128, 512, CAPACITY)  # sizes whose graphs are replayed for repeatability
REPLAYS = 200
SLOWER = 1.03  # a port may be at most 3% slower per call, warm or cold
BUCKETS = (("decode", 1, DECODE[-1]), ("mid", DECODE[-1] + 1, PREFILL[1] - 1), ("prefill", PREFILL[1], CAPACITY))
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


def batch_variance(call: Callable[[int], torch.Tensor], sizes=DECODE + PREFILL[:-1],
                   full_rows: int = CAPACITY) -> list[int]:
    """Row counts whose rows differ in any bit from the same rows of the full batch.

    ``call(rows)`` runs the kernel on the first ``rows`` rows of fixed inputs (copying
    them fresh first if the kernel writes its inputs) and returns its output for those
    rows. Each size takes its own bucket's schedule, so this checks every bucket against
    the prefill one."""
    full = call(full_rows).clone()
    return [rows for rows in sizes if not torch.equal(call(rows), full[:rows])]


def bucket_lines(times: dict[int, tuple[tuple[float, float], tuple[float, float]]]) -> tuple[list[str], list[str]]:
    """Per bucket, from ``times[rows] = (base, port)`` (warm, cold microseconds): the
    port's best and worst ratio to the base, warm and cold, and the sizes past the
    ``SLOWER`` budget. Returns the report lines and the failing buckets."""
    lines, failing = [], []
    for name, low, high in BUCKETS:
        rows = sorted(r for r in times if low <= r <= high)
        if not rows:
            continue
        warm = [times[r][1][0] / times[r][0][0] for r in rows]
        cold = [times[r][1][1] / times[r][0][1] for r in rows]
        over = [r for r in rows if slower(*times[r])]
        lines.append(f"  {name:7s} rows {rows[0]}-{rows[-1]}: ratio warm {min(warm):.3f}-{max(warm):.3f}, "
                     f"cold {min(cold):.3f}-{max(cold):.3f}, " + (f"slower at {over}" if over else "at the floor"))
        if over:
            failing.append(name)
    return lines, failing


def repeatable(fn, outputs: Callable[[], list[torch.Tensor]], replays: int = REPLAYS) -> int:
    """Replays of a one-call CUDA graph of ``fn`` whose outputs differ in any bit from an
    eager call's (0: repeatable). Replays cycle through warm, after an L2 eviction, and
    beside a 128 MiB read on another stream (contention like the L2 prefetch's), so
    tiles and pipeline stages land at different times. ``outputs`` is read after the
    eager call and again after capture (a call that allocates its result writes it to
    the graph's pool)."""
    fn()
    torch.cuda.synchronize()
    refs = [o.clone() for o in outputs()]
    g = graph(fn, 1)
    outs = outputs()
    side = torch.cuda.Stream()
    bad = torch.zeros((), dtype=torch.int64, device=DEVICE)
    for i in range(replays):
        if i % 3 == 1:
            _EVICT.max()
        elif i % 3 == 2:
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                _EVICT.max()
        g.replay()
        bad += torch.stack([(o.view(torch.uint8) != r.view(torch.uint8)).any() for o, r in zip(outs, refs)]).any()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    return int(bad)
