# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What programmatic dependent launch recovers per kernel boundary on this GPU, in
CUDA graphs, with synthetic kernels.

Every kernel reads a word its predecessor wrote (all CTAs) and writes one its
successor reads, so each depends on the one before it. Launch modes:

- ``plain``: no PDL, each kernel launches when its predecessor completes;
- ``wait``: ``griddepcontrol.wait`` first, no trigger (launch as the predecessor's
  CTAs exit);
- ``chain``: wait, then ``launch_dependents`` (``pdl.chain()``, the serving kernels');
- ``late``: wait first, trigger after the store (TileKernels' placement).

1. Boundaries: chains of ``--depth`` kernels of one shape (1, 6, 48 and 192 CTAs;
   no work, about 2 us and about 10 us of it), median microseconds per kernel in
   each mode.
2. Join: A on the main stream, B forked to a side stream, C on the main stream after
   both (``moe_combine``'s shape), C plain and chained.
3. The traced layer (``--layer``, from ``critical_path.py --layer-sequence``): one
   decode layer's critical path at one stream, each kernel a spin of its traced CTAs,
   threads and duration, replayed plain and with every kernel that serving chains
   chained (the all-reduces and warp-specialized GEMMs stay plain). The difference is
   the launch time PDL removes from that sequence, without the prologue overlap of
   real kernels.

One JSON line per measurement.
"""

import argparse
import json
import statistics

import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import pdl

DEVICE = "cuda"
SLOTS = 1 << 16
MODES = ("plain", "wait", "chain", "late")
_PASS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


@tilelang.jit(pass_configs=_PASS)
def spin(ctas: int, threads: int, iters: int, mode: str):
    """Read ``token[k]``, ``iters`` FMA steps per thread, write ``token[k + 1]``."""

    @T.prim_func
    def pdl_spin(token: T.Tensor((SLOTS,), T.float32), k: T.int32):
        with T.Kernel(ctas, threads=threads) as bx:
            if mode == "chain":
                pdl.chain()
            elif mode != "plain":
                T.pdl_sync()
            tx = T.get_thread_binding()
            acc = T.alloc_var(T.float32, init=token[k])
            for _ in T.serial(iters):
                acc = acc * 0.999999 + 1e-6
            if bx == 0 and tx == 0:
                token[k + 1] = acc
            if mode == "late":
                T.pdl_trigger()

    return pdl_spin


def graph(fn) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def timed(g: torch.cuda.CUDAGraph, replays: int) -> float:
    """Median microseconds per replay."""
    times = []
    for _ in range(replays + 3):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) * 1000)
    return statistics.median(times[3:])


def iters_for(ctas: int, threads: int, us: float, token: torch.Tensor) -> int:
    """FMA steps that make one ``plain`` kernel of this shape take about ``us``."""
    if us <= 0:
        return 1

    def alone(iters: int) -> float:
        kernel = spin(ctas, threads, iters, "plain")
        one = timed(graph(lambda: kernel(token, 0)), 10)
        ten = timed(graph(lambda: [kernel(token, i) for i in range(10)]), 10)
        return (ten - one) / 9

    low, high = alone(1), alone(4096)
    return max(1, round(1 + (us - low) / max(high - low, 1e-3) * 4095))


def chain_of(kernels, token):
    def run():
        for i, kernel in enumerate(kernels):
            kernel(token, i)

    return run


def boundaries(args, token):
    for ctas in (1, 6, 48, 192):
        for work in (0.0, 2.0, 10.0):
            iters = iters_for(ctas, 128, work, token)
            row = {"test": "boundary", "ctas": ctas, "work_us": work, "iters": iters}
            for mode in MODES:
                kernel = spin(ctas, 128, iters, mode)
                g = graph(chain_of([kernel] * args.depth, token))
                row[mode] = round(timed(g, args.replays) / args.depth, 3)
            row["saved_us"] = round(row["plain"] - row["chain"], 3)
            print(json.dumps(row), flush=True)


def join(args, token):
    """A -> (B on a side stream) -> C, C after both."""
    iters = iters_for(48, 128, 10.0, token)
    a = spin(48, 128, iters, "chain")
    b = spin(6, 128, iters, "plain")
    side = torch.cuda.Stream()
    row = {"test": "join", "ctas": 48, "work_us": 10.0}
    for mode in ("plain", "chain"):
        c = spin(48, 128, iters, mode)

        def run(c=c):
            for layer in range(args.depth):
                k = 4 * layer
                a(token, k)
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    b(token, k + 1)
                torch.cuda.current_stream().wait_stream(side)
                c(token, k + 2)

        row[mode] = round(timed(graph(run), args.replays) / args.depth, 3)
    row["saved_us"] = round(row["plain"] - row["chain"], 3)
    print(json.dumps(row), flush=True)


def traced_layer(args, token):
    with open(args.layer) as handle:
        sequence = json.load(handle)
    iters = {}
    for entry in sequence["kernels"]:
        key = (entry["ctas"], entry["threads"], entry["us"])
        if key not in iters:
            iters[key] = iters_for(*key, token)
    row = {"test": "traced layer", "source": sequence["source"],
           "kernels": len(sequence["kernels"]),
           "traced_us": sequence["path_us"]}  # fmt: skip
    for mode in ("plain", "chain"):
        kernels = [
            spin(
                e["ctas"],
                e["threads"],
                iters[(e["ctas"], e["threads"], e["us"])],
                "chain" if mode == "chain" and e["pdl"] else "plain",
            )
            for e in sequence["kernels"]
        ]
        row[mode] = round(timed(graph(chain_of(kernels, token)), args.replays), 2)
    row["saved_us"] = round(row["plain"] - row["chain"], 2)
    row["saved_percent_of_layer"] = round(100 * row["saved_us"] / row["plain"], 2)
    print(json.dumps(row), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=64)
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--layer", help="one layer's critical path (critical_path.py)")
    args = ap.parse_args()
    token = torch.zeros(SLOTS, dtype=torch.float32, device=DEVICE)
    boundaries(args, token)
    join(args, token)
    if args.layer:
        traced_layer(args, token)


if __name__ == "__main__":
    main()
