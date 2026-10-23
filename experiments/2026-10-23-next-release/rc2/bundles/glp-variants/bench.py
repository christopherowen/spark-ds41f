#!/usr/bin/env python3
"""GLP projection variants against the compiled torch projection, per call.

vLLM 0261's kernel (one block of 128 threads per stream row, eight values a thread a
load, the row and the direction both held in registers) was 1.2-1.4% slower than the
compiled torch projection at 8K and 16K-token prefill chunks (next-rc2-tp4, 2026-10-23),
both near the memory's bandwidth. These variants change what a thread holds and how many
threads share a row: (threads, vec) sets the threads per row and the values a load, and
``reload`` re-reads the direction (20 KiB, L1-resident) in the write pass instead of
keeping it in registers. Every variant is checked against FP64 first (within one BF16
rounding) and must be bit-identical across batch sizes; then each shape is timed on eight
fresh inputs with graph pairs alternating which goes first, as benchmark_deepseek_v41_glp.py
does. One JSON line per variant and shape.
"""
import json
import math
import statistics

import tilelang
import tilelang.language as T
import torch

STREAMS, HIDDEN = 4, 5120
SHAPES = {"decode c1": 6, "decode c16": 96, "prefill 8K": 8192, "prefill 16K": 16384}
VARIANTS = [(128, 8, False), (128, 8, True), (256, 4, False), (256, 4, True), (160, 8, True),
            (320, 8, False), (320, 8, True), (640, 8, False), (640, 8, True), (320, 16, True)]


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def variant_kernel(hc_mult: int, hidden: int, threads: int, vec: int, reload: bool):
    assert hidden % (threads * vec) == 0 and threads % 32 == 0
    chunks = hidden // (threads * vec)
    warps = threads // 32
    tokens = T.dynamic("tokens")

    if reload:

        @T.prim_func
        def glp_project(
            streams: T.Tensor((tokens, hc_mult, hidden), T.bfloat16),
            direction: T.Tensor((hidden,), T.float32),
            alpha: T.Tensor((1,), T.float32),
            out: T.Tensor((tokens, hc_mult, hidden), T.bfloat16),
        ):
            with T.Kernel(hc_mult, tokens, threads=threads) as (h, t):
                tx = T.get_thread_binding()
                xs = T.alloc_local((chunks * vec,), T.float32)
                ds = T.alloc_local((vec,), T.float32)
                acc = T.alloc_local((1,), T.float32)
                partial = T.alloc_shared((warps,), T.float32)
                scale = T.alloc_shared((1,), T.float32)
                acc[0] = T.float32(0)
                for c in T.serial(chunks):
                    for i in T.vectorized(vec):
                        xs[c * vec + i] = streams[t, h, (c * threads + tx) * vec + i]
                    for i in T.vectorized(vec):
                        ds[i] = direction[(c * threads + tx) * vec + i]
                    for i in T.serial(vec):
                        acc[0] += xs[c * vec + i] * ds[i]
                acc[0] = T.warp_reduce_sum(acc[0])
                if tx % 32 == 0:
                    partial[tx // 32] = acc[0]
                T.sync_threads()
                if tx == 0:
                    acc[0] = T.float32(0)
                    for warp in T.serial(warps):
                        acc[0] += partial[warp]
                    scale[0] = alpha[0] * acc[0]
                T.sync_threads()
                for c in T.serial(chunks):
                    for i in T.vectorized(vec):
                        ds[i] = direction[(c * threads + tx) * vec + i]
                    for i in T.vectorized(vec):
                        out[t, h, (c * threads + tx) * vec + i] = T.cast(
                            xs[c * vec + i] - scale[0] * ds[i], T.bfloat16
                        )

        return glp_project

    @T.prim_func
    def glp_project(
        streams: T.Tensor((tokens, hc_mult, hidden), T.bfloat16),
        direction: T.Tensor((hidden,), T.float32),
        alpha: T.Tensor((1,), T.float32),
        out: T.Tensor((tokens, hc_mult, hidden), T.bfloat16),
    ):
        with T.Kernel(hc_mult, tokens, threads=threads) as (h, t):
            tx = T.get_thread_binding()
            xs = T.alloc_local((chunks * vec,), T.float32)
            ds = T.alloc_local((chunks * vec,), T.float32)
            acc = T.alloc_local((1,), T.float32)
            partial = T.alloc_shared((warps,), T.float32)
            scale = T.alloc_shared((1,), T.float32)
            acc[0] = T.float32(0)
            for c in T.serial(chunks):
                for i in T.vectorized(vec):
                    xs[c * vec + i] = streams[t, h, (c * threads + tx) * vec + i]
                for i in T.vectorized(vec):
                    ds[c * vec + i] = direction[(c * threads + tx) * vec + i]
                for i in T.serial(vec):
                    acc[0] += xs[c * vec + i] * ds[c * vec + i]
            acc[0] = T.warp_reduce_sum(acc[0])
            if tx % 32 == 0:
                partial[tx // 32] = acc[0]
            T.sync_threads()
            if tx == 0:
                acc[0] = T.float32(0)
                for warp in T.serial(warps):
                    acc[0] += partial[warp]
                scale[0] = alpha[0] * acc[0]
            T.sync_threads()
            for c in T.serial(chunks):
                for i in T.vectorized(vec):
                    out[t, h, (c * threads + tx) * vec + i] = T.cast(
                        xs[c * vec + i] - scale[0] * ds[c * vec + i], T.bfloat16
                    )

    return glp_project


def torch_projection(h, direction, alpha):
    x = h.float()
    return (x - (alpha * (x @ direction)).unsqueeze(-1) * direction).to(h.dtype)


def graph(function):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            function()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for _ in range(100):
            function()
    return capture


def elapsed(capture) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 10


def inputs(tokens: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(tokens, STREAMS, HIDDEN, generator=generator, device="cuda").bfloat16()
    d = torch.randn(HIDDEN, generator=generator, device="cuda")
    d /= d.norm()
    return h, d, torch.tensor([0.75], device="cuda")


def correct(kernel) -> bool:
    h, d, alpha = inputs(2048, 7)
    out = torch.empty_like(h)
    kernel(h, d, alpha, out)
    exact = h.double() - (0.75 * (h.double() @ d.double())).unsqueeze(-1) * d.double()
    _, exponent = torch.frexp(exact.abs().clamp_min(2.0**-126))
    ulp = torch.ldexp(torch.ones_like(exact), exponent - 8)
    small = torch.empty_like(h[:7])
    kernel(h[:7].contiguous(), d, alpha, small)
    return bool(((out.double() - exact).abs() <= ulp + 2.0**-24).all()) and torch.equal(small, out[:7])


def main() -> None:
    compiled = torch.compile(torch_projection, dynamic=False)
    failed = False
    for threads, vec, reload in VARIANTS:
        name = f"t{threads}-v{vec}{'-reload' if reload else ''}"
        kernel = variant_kernel(STREAMS, HIDDEN, threads, vec, reload)
        if not correct(kernel):
            print(json.dumps({"variant": name, "correct": False}), flush=True)
            failed = True
            continue
        for shape, tokens in SHAPES.items():
            ours, theirs, differences = [], [], []
            for sample in range(8):
                h, d, alpha = inputs(tokens, sample)
                out = torch.empty_like(h)
                a = graph(lambda: kernel(h, d, alpha, out))
                b = graph(lambda: compiled(h, d, alpha))
                pair = (a, b) if sample % 2 == 0 else (b, a)
                times = {id(c): elapsed(c) for c in pair}
                ours.append(times[id(a)])
                theirs.append(times[id(b)])
                differences.append(times[id(a)] - times[id(b)])
            mean = statistics.fmean(differences)
            half = 2.365 * statistics.stdev(differences) / math.sqrt(8)
            print(json.dumps({"variant": name, "shape": shape, "tokens": tokens, "correct": True,
                              "tilelang_us": round(statistics.median(ours), 2),
                              "torch_us": round(statistics.median(theirs), 2),
                              "difference_us": round(mean, 2),
                              "interval_us": [round(mean - half, 2), round(mean + half, 2)]}), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
