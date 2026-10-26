#!/usr/bin/env python3
"""The layer chain at each row count the engine grid judges, with the shared expert on
its side stream (as the model runs it) and inline: time per layer and bits.

usage (in a build's tree, as pdl_chain_ab.py runs it):
    pdl_rows.py [--rows 2,3,6,...] [--modes aux,inline] [--layers 4] [--replays 50]

The chain is tests/kernels/deepseek_v41_pdl_chains.LayerChain, timed as
benchmarks/kernels/benchmark_deepseek_v41_pdl_chain.py times it: ``--layers`` copies in
one CUDA graph, the median per layer over ``--replays`` replays, warm and after a
128 MiB L2 eviction. Inline, the chain's aux stream is the capturing stream itself: the
shared expert's kernels run between the FFN entry and the routed dispatch, and nothing
runs beside the routed experts. One JSON line per row count and mode, shaped like the
benchmark's, so pdl_chain_ab.py compares them across builds.
"""

import argparse
import json
import statistics

import torch

from tests.kernels.deepseek_v41_pdl_chains import LayerChain
from vllm.config import VllmConfig, set_current_vllm_config

DEVICE = "cuda"
# The engine grid's decode rows: requests x (1 + depth) (servebench/engine.py).
ROWS = (2, 3, 6, 8, 12, 16, 24, 32, 48, 72, 96)


def graph(fn, layers: int) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(layers):
            fn()
    return g


def replay_us(g: torch.cuda.CUDAGraph, evict: torch.Tensor | None) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    if evict is not None:
        evict.max()
    start.record()
    g.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000


def runner(chain: LayerChain, side: torch.cuda.Stream, mode: str):
    """The chain's run with the shared expert on ``side`` or on the current stream."""

    def run() -> None:
        chain.aux = side if mode == "aux" else torch.cuda.current_stream()
        chain.run()

    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    ap.add_argument("--modes", default="aux,inline")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    evict = torch.ones(128 << 20, dtype=torch.uint8, device=DEVICE).view(torch.int32)
    with set_current_vllm_config(VllmConfig()):
        for rows in map(int, args.rows.split(",")):
            chain = LayerChain(rows, seed=args.seed)
            side = chain.aux
            for mode in args.modes.split(","):
                run = runner(chain, side, mode)
                run()
                torch.cuda.synchronize()
                eager = chain.digests()
                g = graph(run, args.layers)
                for _ in range(3):
                    replay_us(g, None)
                warm = [replay_us(g, None) / args.layers for _ in range(args.replays)]
                cold = [replay_us(g, evict) / args.layers for _ in range(args.replays)]
                replayed = chain.digests()
                print(json.dumps({
                    "shape": f"{rows}-{mode}",
                    "rows": rows,
                    "mode": mode,
                    "layers": args.layers,
                    "warm_us": round(statistics.median(warm), 2),
                    "cold_us": round(statistics.median(cold), 2),
                    "warm_p10_p90": [round(x, 2) for x in statistics.quantiles(warm, n=10)[::8]],
                    "cold_p10_p90": [round(x, 2) for x in statistics.quantiles(cold, n=10)[::8]],
                    "finite": bool(torch.isfinite(chain.out).all()),
                    "replay_equals_eager": replayed == eager,
                    "digests": replayed,
                }), flush=True)  # fmt: skip
                del g
            del chain
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
