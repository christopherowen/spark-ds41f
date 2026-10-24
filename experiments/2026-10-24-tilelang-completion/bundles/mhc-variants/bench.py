#!/usr/bin/env python3
"""mHC stream collapse variants against TileKernels' mhc_pre_apply_mix_fwd, per call.

The TileLang collapse (vLLM tilelang-mhc-post) is bit-identical to TileKernels' but
slower at long prefill on GB10 (image -next-4, 2026-10-24): one 128-thread block per
1,024 columns of a token, a thread 8 columns, was +1.1% at 8K and 16K tokens; TileKernels'
own shape (64 threads, a thread two 8-column chunks 512 columns apart, no shared memory)
was +2.5%, although its loads and stores are TileKernels' to the offset. TileKernels also
holds 8 KiB of shared memory a block, which caps it at 11 blocks an SM, where the other
two run 12 and 18. These variants change only the memory side; the arithmetic of every
element is TileKernels': from positive zero, fma(mix[m], state[m], acc) for m = 0..3, one
rounding to BF16.

A variant is (threads, rows, vec, chunks) plus options:
- threads a block, rows (tokens) a block: threads // rows threads share a row;
- vec: BF16 values a load (8: 16 bytes, 4: 8 bytes); chunks: loads a stream a thread;
- layout: "strided" puts a thread's chunks one block-row width apart (TileKernels'),
  "adjacent" makes them contiguous;
- order: "stream" loads stream by stream, chunk by chunk (TileKernels'), "chunk" the
  other way round;
- grid: "token" puts a token's column blocks in adjacent blocks, "column" puts a column
  block's tokens adjacent;
- persist: k > 0 launches k blocks an SM (fewer if there is less work) that loop over
  the (token, column block) items;
- reserve: bytes of shared memory a block holds without using them, which caps the
  blocks an SM (TileKernels: 8 KiB).

Every variant is first checked bit for bit against TileKernels (random rows with
negative-zero columns, the order-of-operations rows, non-finite values, the mean) and
across batch sizes; then each shape is timed on eight fresh inputs (the vLLM
benchmark's collapse inputs), with CUDA-graph pairs of 100 calls alternating which goes
first, as benchmark_deepseek_v41_mhc_post.py does. One JSON line per variant and shape;
exit status 1 if any variant fails its checks or does not build.
"""

import json
import math
import statistics
import traceback
from dataclasses import asdict, dataclass

import tilelang
import tilelang.language as T
import torch

STREAMS, HIDDEN = 4, 5120
SHAPES = {"decode c1": 6, "decode c16": 96, "prefill 8K": 8192, "prefill 16K": 16384}
PASS_CONFIGS = {  # as vllm/models/deepseek_v4_1/tilelang/mhc.py
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}


@dataclass(frozen=True)
class Variant:
    threads: int
    rows: int
    vec: int
    chunks: int
    layout: str = "strided"
    order: str = "stream"
    grid: str = "token"
    persist: int = 0
    reserve: int = 0
    note: str = ""

    @property
    def name(self) -> str:
        name = f"t{self.threads}-r{self.rows}-v{self.vec}-c{self.chunks}"
        if self.layout != "strided":
            name += f"-{self.layout}"
        if self.order != "stream":
            name += f"-{self.order}order"
        if self.grid != "token":
            name += f"-{self.grid}grid"
        if self.persist:
            name += f"-persist{self.persist}"
        if self.reserve:
            name += f"-smem{self.reserve // 1024}k"
        return name


V = Variant
VARIANTS = [
    # The two measured shapes.
    V(128, 1, 8, 1, note="round 1: vLLM tilelang-mhc-post collapse_streams"),
    V(64, 1, 8, 2, note="TileKernels' shape without its shared memory (d781f2a04)"),
    # Blocks an SM capped by unused shared memory (TileKernels holds 8 KiB: 11 an SM).
    V(64, 1, 8, 2, reserve=8192, note="TileKernels' shape and residency"),
    V(128, 1, 8, 1, reserve=8192),
    V(128, 1, 8, 1, reserve=16384),
    V(128, 1, 8, 1, reserve=24576),
    V(64, 1, 8, 2, reserve=16384),
    # Persistent grids: k blocks an SM loop over the items.
    V(128, 1, 8, 1, persist=4),
    V(128, 1, 8, 1, persist=8),
    V(128, 1, 8, 1, persist=10),
    V(64, 1, 8, 2, persist=6),
    V(64, 1, 8, 2, persist=11),
    # Threads a block, one row.
    V(32, 1, 8, 1),
    V(64, 1, 8, 1),
    V(256, 1, 4, 1),
    # Rows a block.
    V(128, 2, 8, 1),
    V(128, 4, 8, 1),
    V(256, 2, 8, 1),
    V(128, 2, 8, 2),
    V(256, 4, 8, 2),
    # Columns a thread.
    V(32, 1, 8, 4),
    V(128, 1, 8, 5, note="a whole token a block"),
    V(64, 1, 8, 2, layout="adjacent"),
    V(128, 2, 8, 2, layout="adjacent"),
    # Load order.
    V(64, 1, 8, 2, order="chunk"),
    V(128, 2, 8, 2, order="chunk"),
    # Vector width.
    V(128, 1, 4, 1),
    V(128, 1, 4, 2),
    V(64, 1, 4, 4),
    # Grid order.
    V(128, 1, 8, 1, grid="column"),
    V(64, 1, 8, 2, grid="column"),
]


@tilelang.jit(pass_configs=PASS_CONFIGS)
def collapse_variant(
    hidden: int,
    threads: int,
    rows: int,
    vec: int,
    chunks: int,
    layout: str,
    order: str,
    grid: str,
    persist: int,
    reserve: int,
    sms: int,
):
    assert threads % rows == 0 and (threads // rows) % 32 == 0
    tpr = threads // rows  # threads a row
    cols = tpr * vec * chunks  # columns a block covers in each of its rows
    assert hidden % cols == 0, (hidden, cols)
    blocks = hidden // cols
    # Column of chunk c of lane l: base + l * lane_stride + c * chunk_stride.
    if layout == "strided":
        lane_stride, chunk_stride = vec, tpr * vec
    else:
        lane_stride, chunk_stride = chunks * vec, vec
    M = T.dynamic("M")
    groups = (M + rows - 1) // rows  # row groups (blocks a column block)
    items = groups * blocks  # (row group, column block) items
    launched = T.min(items, persist * sms) if persist else items

    @T.macro
    def item(w, row, lane, state, mix, out, sv, ov, mv, acc):
        if grid == "token":
            g = w // blocks
            b = w % blocks
        else:
            g = w % groups
            b = w // groups
        token = g * rows + row
        if rows > 1:
            src = T.min(token, M - 1)
        else:
            src = token
        col = b * cols + lane * lane_stride
        for m in T.vectorized(STREAMS):
            mv[m] = mix[src, m]
        if order == "stream":
            for m in T.unroll(STREAMS):
                for c in T.unroll(chunks):
                    for v in T.vectorized(vec):
                        sv[(m * chunks + c) * vec + v] = state[src, m, col + c * chunk_stride + v]
        else:
            for c in T.unroll(chunks):
                for m in T.unroll(STREAMS):
                    for v in T.vectorized(vec):
                        sv[(m * chunks + c) * vec + v] = state[src, m, col + c * chunk_stride + v]
        # TileKernels' arithmetic: from +0, one fma a stream in order, one rounding.
        for c in T.unroll(chunks):
            for v in T.unroll(vec):
                acc[0] = T.float32(0)
                for m in T.unroll(STREAMS):
                    acc[0] = T.fma(mv[m], T.Cast(T.float32, sv[(m * chunks + c) * vec + v]), acc[0])
                ov[c * vec + v] = T.Cast(T.bfloat16, acc[0])
        if rows > 1:
            if token < M:
                for c in T.unroll(chunks):
                    for v in T.vectorized(vec):
                        out[token, col + c * chunk_stride + v] = ov[c * vec + v]
        else:
            for c in T.unroll(chunks):
                for v in T.vectorized(vec):
                    out[token, col + c * chunk_stride + v] = ov[c * vec + v]

    @T.prim_func
    def mhc_collapse(
        state: T.Tensor((M, STREAMS, hidden), T.bfloat16),
        mix: T.Tensor((M, STREAMS), T.float32),
        out: T.Tensor((M, hidden), T.bfloat16),
    ):
        with T.Kernel(launched, threads=threads) as pid:
            tx = T.get_thread_binding()
            sv = T.alloc_local((STREAMS * chunks * vec,), T.bfloat16)
            ov = T.alloc_local((chunks * vec,), T.bfloat16)
            mv = T.alloc_local((STREAMS,), T.float32)
            acc = T.alloc_local((1,), T.float32)
            if reserve:
                # Held, not used: one write keeps the allocation (and the residency cap).
                pad = T.alloc_shared((reserve // 4,), T.float32)
                if tx == 0:
                    pad[0] = T.float32(0)
            if persist:
                for it in T.serial(T.ceildiv(items - pid, launched)):
                    item(
                        pid + it * launched,
                        tx // tpr,
                        tx % tpr,
                        state,
                        mix,
                        out,
                        sv,
                        ov,
                        mv,
                        acc,
                    )
            else:
                item(pid, tx // tpr, tx % tpr, state, mix, out, sv, ov, mv, acc)

    return mhc_collapse


def build(variant: Variant, sms: int):
    return collapse_variant(
        HIDDEN,
        variant.threads,
        variant.rows,
        variant.vec,
        variant.chunks,
        variant.layout,
        variant.order,
        variant.grid,
        variant.persist,
        variant.reserve,
        sms,
    )


def tile_kernels_collapse(state, mix, out=None):
    import tile_kernels

    if out is None:
        out = torch.empty(state.shape[0], HIDDEN, device="cuda", dtype=torch.bfloat16)
    tile_kernels.mhc.mhc_pre_apply_mix_fwd(state, mix, out)
    return out


def run(kernel, state, mix):
    out = torch.empty(state.shape[0], HIDDEN, device="cuda", dtype=torch.bfloat16)
    kernel(state, mix, out)
    return out


def same_bits(a, b) -> bool:
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def check_inputs(tokens: int, seed: int):
    """The vLLM tests' collapse inputs: BF16 streams whose first eight columns are
    negative zeros, a sigmoid pre-mix plus hc_eps."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.randn(tokens, STREAMS, HIDDEN, generator=g, device="cuda").bfloat16()
    state[:, :, :8] = -0.0
    mix = torch.sigmoid(torch.randn(tokens, STREAMS, generator=g, device="cuda")) + 1e-6
    return state, mix


def order_inputs():
    """test_collapse_order_of_operations' rows: overflow before cancellation, a fused
    product, four negative-zero products."""
    big, eps = 2.0**127, 2.0**-23
    state = torch.zeros(3, STREAMS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    mix = torch.zeros(3, STREAMS, device="cuda")
    state[0, 0], state[0, 1], state[0, 2] = big, big, -big
    mix[0, :3] = 1.0
    state[1, 0], state[1, 1] = -(1 + 2.0**-7), 1 + 2.0**-7
    mix[1, 0], mix[1, 1] = 1.0, 1 + eps
    state[2] = -0.0
    mix[2] = 0.25
    return state, mix


def correctness(kernel) -> str | None:
    """None if the variant gives TileKernels' bits everywhere, else what differed."""
    state, mix = check_inputs(2048, 1)
    full = run(kernel, state, mix)
    if not same_bits(full, tile_kernels_collapse(state, mix)):
        return "2048 rows: bits differ from TileKernels'"
    for rows in (1, 2, 3, 5, 6, 7, 31, 96, 97, 300, 1025):
        part = run(kernel, state[:rows].contiguous(), mix[:rows].contiguous())
        if not same_bits(part, full[:rows]):
            return f"first {rows} rows differ from the 2048-row batch"
    lone = run(kernel, state[1000:1001].contiguous(), mix[1000:1001].contiguous())
    if not same_bits(lone, full[1000:1001]):
        return "row 1000 alone differs from the 2048-row batch"
    quarter = torch.full((2048, STREAMS), 0.25, device="cuda")
    if not same_bits(run(kernel, state, quarter), tile_kernels_collapse(state, quarter)):
        return "the mean: bits differ from TileKernels'"
    state, mix = order_inputs()
    if not same_bits(run(kernel, state, mix), tile_kernels_collapse(state, mix)):
        return "order-of-operations rows: bits differ from TileKernels'"
    state, mix = check_inputs(33, 61)
    nan, inf = float("nan"), float("inf")
    state[3, 0, 11] = nan
    state[8, 2, 2048] = inf
    state[14, 1, 77], state[14, 3, 77] = inf, -inf
    mix[19, 1] = nan
    state[23, 2, 5000], mix[23, 2] = -inf, 0.0
    mix[29, 0] = inf
    if not same_bits(run(kernel, state, mix), tile_kernels_collapse(state, mix)):
        return "non-finite rows: bits differ from TileKernels'"
    return None


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
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 10  # ms / 100 calls to microseconds


def bench_inputs(tokens: int, seed: int):
    """benchmark_deepseek_v41_mhc_post.py's collapse_case inputs."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.randn(tokens, STREAMS, HIDDEN, generator=generator, device="cuda")
    state = state.bfloat16()
    mix = torch.randn(tokens, STREAMS, generator=generator, device="cuda").sigmoid()
    return state, mix


def measure(kernel, tokens: int, samples: int = 8) -> dict:
    ours_us, theirs_us, differences = [], [], []
    for sample in range(samples):
        state, mix = bench_inputs(tokens, sample)
        ours_out = torch.empty(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
        theirs_out = torch.empty_like(ours_out)

        def ours(state=state, mix=mix, out=ours_out):
            kernel(state, mix, out)

        def theirs(state=state, mix=mix, out=theirs_out):
            tile_kernels_collapse(state, mix, out)

        ours()
        theirs()
        torch.cuda.synchronize()
        if not same_bits(ours_out, theirs_out):
            raise RuntimeError(f"{tokens} tokens, sample {sample}: bits differ")
        a, b = graph(ours), graph(theirs)
        pair = (a, b) if sample % 2 == 0 else (b, a)
        times = {id(capture): elapsed(capture) for capture in pair}
        ours_us.append(times[id(a)])
        theirs_us.append(times[id(b)])
        differences.append(times[id(a)] - times[id(b)])
        del a, b, pair
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(samples)  # t(7), 95%
    theirs = statistics.median(theirs_us)
    return {
        "tokens": tokens,
        "tilelang_us": round(statistics.median(ours_us), 3),
        "tile_kernels_us": round(theirs, 3),
        "difference_us": round(mean, 3),
        "difference_pct": round(100 * mean / theirs, 3),
        "interval_us": [round(mean - half, 3), round(mean + half, 3)],
    }


def main() -> None:
    import tile_kernels

    properties = torch.cuda.get_device_properties(0)
    sms = properties.multi_processor_count
    print(
        json.dumps(
            {
                "device": properties.name,
                "sms": sms,
                "tilelang": tilelang.__version__,
                "tile_kernels": getattr(tile_kernels, "__version__", "unknown"),
                "variants": [v.name for v in VARIANTS],
            }
        ),
        flush=True,
    )
    failed = False
    for variant in VARIANTS:
        config = {k: v for k, v in asdict(variant).items() if k != "note"}
        head = {"variant": variant.name, "config": config, "note": variant.note}
        try:
            kernel = build(variant, sms)
            problem = correctness(kernel)
        except Exception as error:  # noqa: BLE001 (a variant that does not build: report, go on)
            traceback.print_exc()
            print(json.dumps({**head, "correct": False, "error": repr(error)}), flush=True)
            failed = True
            continue
        if problem is not None:
            print(json.dumps({**head, "correct": False, "error": problem}), flush=True)
            failed = True
            continue
        for shape, tokens in SHAPES.items():
            try:
                result = measure(kernel, tokens)
            except Exception as error:  # noqa: BLE001 (report the shape, go on)
                traceback.print_exc()
                print(json.dumps({**head, "shape": shape, "error": repr(error)}), flush=True)
                failed = True
                break
            print(json.dumps({**head, "shape": shape, "correct": True, **result}), flush=True)
        torch.cuda.empty_cache()
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
