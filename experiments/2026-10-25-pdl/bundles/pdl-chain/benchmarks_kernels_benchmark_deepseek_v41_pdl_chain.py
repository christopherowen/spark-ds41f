# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One DeepSeek-V4.1 decode layer's kernel chain in a CUDA graph: time and bits.

The chain is ``tests/kernels/deepseek_v41_pdl_chains.LayerChain``: the layer without
its attention core, router and all-reduces, 23 kernels, at a decode step's rows (one
stream, eight mixed streams, eight streams of six rows). Each shape's graph holds
``--layers`` copies of the chain back to back, as a forward replays layer after layer.

Per shape it prints one JSON line: the median microseconds per layer over
``--replays`` replays, warm (the previous replay's weights partly in L2) and cold
(L2 evicted by a 128 MiB read before each replay), and a digest of every buffer the
chain writes. Run it on two builds, the one without PDL and the one with it, and
compare: the digests must be equal (PDL changes no arithmetic) and the difference in
time is what PDL recovers on this chain. ``pdl_chain_ab.py`` in the experiment's
kernel bundle runs both builds in alternation on one GPU.
"""

import argparse
import json
import statistics

import torch

from tests.kernels.deepseek_v41_pdl_chains import ROWS, LayerChain
from vllm.config import VllmConfig, set_current_vllm_config

DEVICE = "cuda"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(ROWS))
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    evict = torch.ones(128 << 20, dtype=torch.uint8, device=DEVICE).view(torch.int32)
    with set_current_vllm_config(VllmConfig()):
        for name in args.shapes.split(","):
            chain = LayerChain(ROWS[name], seed=args.seed)
            chain.run()
            torch.cuda.synchronize()
            eager = chain.digests()
            g = graph(chain.run, args.layers)
            for _ in range(3):
                replay_us(g, None)
            warm = [replay_us(g, None) / args.layers for _ in range(args.replays)]
            cold = [replay_us(g, evict) / args.layers for _ in range(args.replays)]
            replayed = chain.digests()
            print(
                json.dumps(
                    {
                        "shape": name,
                        "rows": ROWS[name],
                        "layers": args.layers,
                        "warm_us": round(statistics.median(warm), 2),
                        "cold_us": round(statistics.median(cold), 2),
                        "warm_p10_p90": [
                            round(x, 2) for x in statistics.quantiles(warm, n=10)[::8]
                        ],
                        "cold_p10_p90": [
                            round(x, 2) for x in statistics.quantiles(cold, n=10)[::8]
                        ],
                        "finite": bool(torch.isfinite(chain.out).all()),
                        "replay_equals_eager": replayed == eager,
                        "digests": replayed,
                    }
                ),
                flush=True,
            )
            del g, chain
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
