"""Rows 65-128 on the production TileLang projections (TP4 16-stream steps reach 96
rows). Today they take the prefill tiles (128 x 128 for FP8: a handful of CTAs) and,
for the BF16 router, the unsplit GEMM. Decode tiles run any row count padded to whole
tiles, and every tile accumulates K in the same order, so the bits are the same.

For each production FP8 shape: whole calls (activation cast + GEMM) at 72, 80, 96 and
128 rows with the prefill tile, the 64-row decode tile and the 32-row decode tile;
warm (same weight), cold (distinct weights) and self (each call follows the L2
prefetch of its own weight). For the narrow BF16 projections (the router and the
indexer's head weights), at 72 to 1024 rows: the shard GEMM against split-K partials
plus reduce, warm and cold. Checks the bits of every variant against the prefill
GEMM. Exits 1 only if bits differ.
"""
import sys

import tile_kernels
import torch

sys.path.insert(0, "/b")
from kbench import DEVICE, per_call  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang import gemm as g  # noqa: E402
from vllm.models.glm5next.nvidia import l2_prefetch as L  # noqa: E402

GEN = torch.Generator(device=DEVICE).manual_seed(17)
PF = L.L2Prefetcher.get(DEVICE)
SHAPES = {"q_b": (8192, 1280), "indexer_q_b": (4096, 1280), "q_a_kv": (1792, 5120),
          "shared_gate_up": (1152, 5120), "shared_down": (5120, 576), "draft_main": (6400, 6144)}
ROWS = (72, 80, 96, 128)
failures = []


def timed(calls, rounds=10):
    for c in calls:
        c()
    PF.join()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
        PF.join()
    graph.replay()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000 / (rounds * len(calls))


def fp8(shape):
    return (torch.randn(shape, device=DEVICE, generator=GEN) * 0.5).to(torch.float8_e4m3fn)


for name, (N, K) in SHAPES.items():
    copies = max(4, -(-160 * 2**20 // (N * K)))
    ws_ = [fp8((N, K)) for _ in range(copies)]
    wsf_ = [g.pack_scale_words(torch.randint(124, 131, (N // 32, K // 32), device=DEVICE, generator=GEN,
                                             dtype=torch.uint8), rows=N) for _ in range(copies)]
    plans = [L.make_plan([L.tensor_segment("w", w), L.tensor_segment("sf", s, min_bytes=0)], 20 * 2**20, DEVICE)[0]
             for w, s in zip(ws_, wsf_)]
    variants = {
        "prefill": (g.mxfp8_gemm(N, K, **g.fp8_prefill_config(N, K)), None),
        "decode64": (g.mxfp8_gemm_decode(N, K, **g.fp8_decode_config(N, K, 64), padded_rows=True), 64),
        "decode32": (g.mxfp8_gemm_decode(N, K, **g.fp8_decode_config(N, K, 32), padded_rows=True), 32),
    }
    x = (torch.randn(max(ROWS), K, device=DEVICE, generator=GEN) * 0.5).bfloat16()
    xq = torch.empty(max(ROWS), K, dtype=torch.float8_e4m3fn, device=DEVICE)
    sf = torch.empty(max(ROWS), 4 * g.scale_words(K), dtype=torch.uint8, device=DEVICE)
    out = torch.empty(max(ROWS), N, dtype=torch.bfloat16, device=DEVICE)
    print(f"{name} {N}x{K}: us warm/cold/self per call", flush=True)
    for rows in ROWS:
        line = f"  rows {rows:3d}"
        ref = None
        for label, (kernel, tile) in variants.items():
            padded = rows if tile is None else -(-rows // tile) * tile

            def run_one(i, kernel=kernel, padded=padded, rows=rows):
                tile_kernels.quant.per_token_cast(x[:rows], "e4m3", 32, round_sf=True, use_packed_ue8m0=True,
                                                  out=(xq[:rows], sf[:rows]))
                kernel(xq[:padded], ws_[i], sf[:padded].view(torch.uint32), wsf_[i], out[:rows])
            run_one(0)
            torch.cuda.synchronize()
            bits = out[:rows].clone()
            if ref is None:
                ref = bits
            elif not torch.equal(bits.view(torch.int16), ref.view(torch.int16)):
                failures.append(f"{name} rows {rows} {label}: bits differ from prefill")
            warm = timed([lambda: run_one(0)] * 16)
            cold = timed([(lambda i=i: run_one(i % copies)) for i in range(2 * copies)])

            def selfcall(i, run_one=run_one, rows=rows):
                def run():
                    PF.issue(plans[i % copies], rows)
                    run_one(i % copies)
                return run
            own = timed([selfcall(i) for i in range(2 * copies)])
            line += f"  {label} {warm:6.1f}/{cold:6.1f}/{own:6.1f}"
        print(line, flush=True)

# Narrow BF16 projections: the router (384 x 5120, FP32 out) and the indexer's head
# weights (32 x 5120, BF16 out), which split K up to 64 rows today.
for label, (N, K, out_dtype, torch_dtype) in {
        "router": (384, 5120, "float32", torch.float32),
        "index weights": (32, 5120, "bfloat16", torch.bfloat16)}.items():
    shards = g.bf16_shards(N, K)
    bf16_rows = (72, 96, 128, 256, 512, 1024)
    w = (torch.randn(N, K, device=DEVICE, generator=GEN) * 0.02).bfloat16()
    x = torch.randn(max(bf16_rows), K, device=DEVICE, generator=GEN).bfloat16()
    full = g.bf16_gemm(N, K, out_dtype=out_dtype, shards=shards)
    partials = g.bf16_gemm_partials(N, K, shards, block_M=64)
    reduce = g.splitk_reduce(N, shards, out_dtype=out_dtype)
    out = torch.empty(max(bf16_rows), N, dtype=torch_dtype, device=DEVICE)
    p = torch.empty(shards * max(bf16_rows) * N, dtype=torch.float32, device=DEVICE)
    print(f"{label} {N}x{K} ({shards} shards): us warm/cold, shard GEMM vs split-K partials + reduce", flush=True)
    for rows in bf16_rows:
        full(x[:rows], w, out[:rows])
        torch.cuda.synchronize()
        ref = out[:rows].clone()
        view = p[:shards * rows * N].view(shards, rows, N)

        def split(rows=rows, view=view):
            partials(x[:rows], w, view)
            reduce(view, out[:rows])
        split()
        torch.cuda.synchronize()
        if not torch.equal(out[:rows], ref):
            failures.append(f"{label} rows {rows}: split differs")
        a = per_call(lambda: full(x[:rows], w, out[:rows]), rows)
        b = per_call(split, rows)
        print(f"  rows {rows:5d}  full {a[0]:7.1f}/{a[1]:7.1f}  split {b[0]:7.1f}/{b[1]:7.1f}  "
              f"ratio {b[0] / a[0]:.3f}/{b[1] / a[1]:.3f}", flush=True)

for failure in failures:
    print("FAIL", failure)
sys.exit(1 if failures else 0)
