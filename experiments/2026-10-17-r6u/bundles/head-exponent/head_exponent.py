"""The packed vocabulary head with its exponent base compiled in (r6s) versus passed as a
runtime argument: bit-identical logits, and the time of each decode row count.

Builds the runtime-argument variant from vocab.py's own constants and tile table (one
kernel signature change: an int32 ``base_minus_one`` parameter instead of a constant),
compiles both for the TP3 and TP4 vocabulary shards, checks every row count gives the same
bits on random packed weights (with out-of-window exceptions), and times them with CUDA
events (median of RUNS launches after warm-up, weights L2-cold between launches).
Exit status 0 when every row count is bit-identical; the timing table is the result.
"""
import statistics
import sys

import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import vocab as V

RUNS = 200
ROWS = (1, 3, 6, 12, 16, 24, 32, 48, 64, 96)
SHAPES = {"tp3": (43136, 4096), "tp4": (32320, 4096)}


@tilelang.jit
def packed_vocab_projection_rt(N, K, block_M=16, block_N=64, num_stages=3, threads=128,
                               step_groups=1, out_dtype=T.float32):
    assert K % (V.GROUP * step_groups) == 0
    steps = K // (V.GROUP * step_groups)
    STEP = V.GROUP * step_groups
    row_bytes = K + K // 2
    slices = V.HALF // V.VEC
    items = block_N * step_groups * slices
    M = T.dynamic("M")
    E = T.dynamic("E")
    GROUP, HALF, VEC = V.GROUP, V.HALF, V.VEC

    @T.prim_func
    def vocab_packed_projection(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, row_bytes), T.uint8),
        exception_rows: T.Tensor((E,), T.int32),
        exception_entries: T.Tensor((E,), T.int32),
        row_offsets: T.Tensor((N + 1,), T.int32),
        C: T.Tensor((M, N), out_dtype),
        base_minus_one: T.int32,
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (by, bx):
            A_shared = T.alloc_shared((block_M, STEP), T.bfloat16)
            low_shared = T.alloc_shared((block_N, STEP), T.uint8)
            code_shared = T.alloc_shared((block_N, STEP // 2), T.uint8)
            B_shared = T.alloc_shared((block_N, STEP), T.bfloat16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            low_a = T.alloc_local((VEC,), T.uint8)
            low_b = T.alloc_local((VEC,), T.uint8)
            codes = T.alloc_local((VEC,), T.uint8)
            out_a = T.alloc_local((VEC,), T.bfloat16)
            out_b = T.alloc_local((VEC,), T.bfloat16)
            tx = T.get_thread_binding()
            T.clear(C_local)
            last = num_stages - 1
            for s in T.Pipelined(steps, order=[0, 1, 2, 3, 4], stage=[0, 0, 0, last, last]):
                T.copy(A[by * block_M, s * STEP], A_shared)
                T.copy(W[bx * block_N, s * STEP], low_shared)
                T.copy(W[bx * block_N, K + s * (STEP // 2)], code_shared)
                for step in T.serial(T.ceildiv(items, threads)):
                    item = step * threads + tx
                    if item < items:
                        j = item // (step_groups * slices)
                        g = item // slices % step_groups
                        c = item % slices * VEC
                        for v in T.vectorized(VEC):
                            low_a[v] = low_shared[j, g * GROUP + c + v]
                        for v in T.vectorized(VEC):
                            low_b[v] = low_shared[j, g * GROUP + HALF + c + v]
                        for v in T.vectorized(VEC):
                            codes[v] = code_shared[j, g * HALF + c + v]
                        for v in T.unroll(VEC):
                            out_a[v] = V._decode(low_a[v], codes[v] & 0xF, base_minus_one)
                            out_b[v] = V._decode(low_b[v], codes[v] >> 4, base_minus_one)
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + c + v] = out_a[v]
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + HALF + c + v] = out_b[v]
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            first = bx * block_N
            start = row_offsets[first]
            end = row_offsets[T.min(first + block_N, N)]
            for index in T.serial(start, end):
                row = exception_rows[index] - first
                entry = exception_entries[index]
                value = T.Cast(T.float32, T.reinterpret(T.Cast(T.uint16, entry & 0xFFFF), T.bfloat16))
                column = entry >> 16
                for i, j in T.Parallel(block_M, block_N):
                    token = T.min(by * block_M + i, M - 1)
                    C_local[i, j] = T.if_then_else(
                        j == row, C_local[i, j] + value * T.Cast(T.float32, A[token, column]), C_local[i, j])
            T.copy(C_local, C[by * block_M, bx * block_N])

    return vocab_packed_projection


def packed_weights(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    low = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda", generator=g)
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=g)
    w = torch.cat([low, codes], dim=1).contiguous()
    rows = torch.sort(torch.randint(0, n, (4096,), device="cuda", generator=g)).values.to(torch.int32)
    entries = (torch.randint(0, k, (4096,), device="cuda", generator=g).to(torch.int32) << 16) | 0x3F80
    offsets = torch.searchsorted(rows, torch.arange(n + 1, device="cuda", dtype=torch.int32)).to(torch.int32)
    return w, rows, entries, offsets


def timed(fn):
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for _ in range(10):
        fn()
    times = []
    for _ in range(RUNS):
        flush.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return statistics.median(times)


def main():
    base = 120  # any exponent window start; both variants decode with it
    same = True
    print(f"{'shape':5s} {'rows':>5s} {'constant us':>12s} {'argument us':>12s} {'change':>8s} bits")
    for name, (n, k) in SHAPES.items():
        w, rows, entries, offsets = packed_weights(n, k, 0)
        for m in ROWS:
            tile = V.row_tile(m)
            config = V.TILE_CONFIGS[tile]
            constant = V.packed_vocab_projection(n, k, base, block_M=tile, out_dtype=T.float32, **config)
            argument = packed_vocab_projection_rt(n, k, block_M=tile, out_dtype=T.float32, **config)
            a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            c1 = torch.empty((m, n), dtype=torch.float32, device="cuda")
            c2 = torch.empty_like(c1)
            constant(a, w, rows, entries, offsets, c1)
            argument(a, w, rows, entries, offsets, c2, base - 1)
            torch.cuda.synchronize()
            identical = torch.equal(c1.view(torch.int32), c2.view(torch.int32))
            same &= identical
            t1 = timed(lambda: constant(a, w, rows, entries, offsets, c1))
            t2 = timed(lambda: argument(a, w, rows, entries, offsets, c2, base - 1))
            print(f"{name:5s} {m:5d} {t1:12.2f} {t2:12.2f} {100 * (t2 / t1 - 1):+7.2f}% {'same' if identical else 'DIFFER'}", flush=True)
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
