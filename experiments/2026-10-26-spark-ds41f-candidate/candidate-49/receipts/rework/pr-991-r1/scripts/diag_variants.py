#!/usr/bin/env python3
"""Rework diagnostics for PR #991: paged-key and GEMM-epilogue E2M1 variants.

Runs inside spark-ds41f:candidate-49 with the PR bench mounted at
/tmp/bench_hw_e2m1x2_quant.py (its sw arms are 1107's kernels). Every variant is
checked torch.equal against the sw arm before timing. Timing: N variants
interleaved per round in rotating order (eager launches, as the bench), and
optionally the same launches captured in one CUDA graph per variant.

usage: diag_variants.py --what paged|gemm|sass --out /out/x.json [--graph]
"""

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import pdl
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    SF_BLOCK,
    SF_WORD_K,
    TILE_WEIGHT_K,
    TILE_WEIGHT_N,
    _E2M1_HEADER,
    block_scale_words,
    mxfp8_gemm_rope_mxfp4_decode,
    scale_words,
)
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    INDEX_DIM,
    _mxfp4_store_group,
    _mxfp4_word,
    mxfp4_quantize_v41,
    page_bytes,
)

spec = importlib.util.spec_from_file_location("bench", "/tmp/bench_hw_e2m1x2_quant.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

EXTRA = r"""
__device__ __forceinline__ void dsv41_st_u32(uint8_t* ptr, uint32_t w) {
  *reinterpret_cast<uint32_t*>(ptr) = w;
}
__device__ __forceinline__ void dsv41_st_u64(uint8_t* ptr, uint32_t w0, uint32_t w1) {
  *reinterpret_cast<uint2*>(ptr) = make_uint2(w0, w1);
}
"""


def _e2m1x2(v0, v1):
    return T.call_extern("uint8", "dsv41_e2m1x2", v0, v1)


@tilelang.jit
def paged_variant(mode: str, page_size: int = 64, threads: int = 128):
    rows, pool_pages, pool_stride = T.dynamic("rows, pool_pages, pool_stride")
    groups = INDEX_DIM // SF_BLOCK
    half = SF_BLOCK // 2

    @T.prim_func
    def indexer_mxfp4_quantize(
        X: T.Tensor([rows, INDEX_DIM], T.bfloat16),
        Slots: T.Tensor([rows], T.int64),
        Pool: T.StridedTensor(
            [pool_pages, page_bytes(page_size)], [pool_stride, 1], T.uint8
        ),
    ):
        with T.Kernel(T.ceildiv(rows * groups, threads), threads=threads) as bx:
            T.import_source(_E2M1_HEADER + EXTRA)
            tx = T.get_thread_binding()
            g = bx * threads + tx
            row, k = g // groups, g % groups
            vals = T.alloc_local([SF_BLOCK], T.bfloat16)
            slot_l = T.alloc_local([1], T.int64)
            amax = T.alloc_var(T.float32)
            if row < rows:
                slot_l[0] = Slots[row]
            pdl.chain()
            if row < rows:
                for i in T.serial(SF_BLOCK):
                    vals[i] = X[row, k * SF_BLOCK + i]
                amax = 6.0 * 2.0**-126
                for i in T.serial(SF_BLOCK):
                    amax = T.max(amax, T.abs(T.cast(vals[i], T.float32)))
                bits = T.reinterpret(T.fmul(amax, T.float32(1.0 / 6.0)), T.uint32)
                sf = T.cast(T.shift_right(bits, 23) & 0xFF, T.int32) + T.cast(
                    (bits & 0x7FFFFF) != 0, T.int32
                )
                scale = T.reinterpret(
                    T.cast(T.shift_left(254 - sf, 23), T.uint32), T.float32
                )
                slot = slot_l[0]
                page, token = slot // page_size, slot % page_size
                if (slot >= 0) & (page < pool_pages):
                    base = token * (INDEX_DIM // 2) + k * half
                    if mode == "sw":
                        for i in T.serial(half):
                            Pool[page, base + i] = bench._sw_mxfp4_byte(vals, 2 * i, scale)
                    elif mode == "hw128":
                        _mxfp4_store_group(T.address_of(Pool[page, base]), vals, scale)
                    elif mode == "hw8":
                        for i in T.serial(half):
                            Pool[page, base + i] = _e2m1x2(
                                T.fmul(T.cast(vals[2 * i], T.float32), scale),
                                T.fmul(T.cast(vals[2 * i + 1], T.float32), scale),
                            )
                    elif mode == "hw32":
                        for w in T.unroll(4):
                            T.call_extern(
                                "void",
                                "dsv41_st_u32",
                                T.address_of(Pool[page, base + 4 * w]),
                                _mxfp4_word(vals, 8 * w, scale),
                            )
                    elif mode == "hw64":
                        for w in T.unroll(2):
                            T.call_extern(
                                "void",
                                "dsv41_st_u64",
                                T.address_of(Pool[page, base + 8 * w]),
                                _mxfp4_word(vals, 16 * w, scale),
                                _mxfp4_word(vals, 16 * w + 8, scale),
                            )
                    elif mode == "null":
                        Pool[page, base] = T.uint8(0)
                    Pool[page, page_size * (INDEX_DIM // 2) + token * groups + k] = T.cast(
                        sf, T.uint8
                    )

    return indexer_mxfp4_quantize


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def gemm_variant(
    N: int,
    K: int,
    conv: str,
    rolled: bool,
    head_dim: int = 128,
    rope_dim: int = 64,
    block_M: int = 16,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 256,
):
    pairs_per_row = block_N // 2
    k_blocks = K // block_K
    M = T.dynamic("M")
    Mp = T.dynamic("Mp")
    max_pos = T.dynamic("max_pos")
    in_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    sf_words = scale_words(K)
    block_words = block_scale_words(block_K)
    k_tiles = K // TILE_WEIGHT_K
    b_shape = (N * k_tiles, TILE_WEIGHT_K)
    half_n = N // 2
    scale_n = N // SF_BLOCK
    rope_start = head_dim - rope_dim
    rope_half = rope_dim // 2
    half = SF_BLOCK // 2
    rows_per_iter = threads // pairs_per_row
    iters = block_M // rows_per_iter

    @T.prim_func
    def gemm_mxfp8_rope_mxfp4(
        A: T.Tensor((Mp, K), in_dtype),
        B: T.Tensor(b_shape, in_dtype),
        SFA: T.Tensor((Mp, sf_words), T.uint32),
        SFB: T.Tensor((N // 32, sf_words), T.uint32),
        Positions: T.Tensor((M,), T.int64),
        CosSin: T.Tensor((max_pos, rope_dim), T.float32),
        Q: T.Tensor((M, half_n), T.uint8),
        S: T.Tensor((M, scale_n), T.uint8),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            T.import_source(_E2M1_HEADER)
            pdl.chain()
            tx = T.get_thread_binding()
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, block_words), T.uint32)
            SFB_shared = T.alloc_shared((block_N, block_words), T.uint32)
            C_shared = T.alloc_shared((block_M, block_N), T.bfloat16)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                for i, kk in T.Parallel(block_M, block_K):
                    A_shared[i, kk] = A[T.min(by * block_M + i, M - 1), ko * block_K + kk]
                row0 = ((bx * block_N // TILE_WEIGHT_N) * k_tiles + ko) * TILE_WEIGHT_N + (
                    bx * block_N % TILE_WEIGHT_N
                )
                T.copy(B[row0, 0], B_shared)
                word0 = ko * block_K // SF_WORD_K
                for i, w in T.Parallel(block_M, block_words):
                    SFA_shared[i, w] = SFA[
                        T.min(by * block_M + i, M - 1), T.min(word0 + w, sf_words - 1)
                    ]
                for j, w in T.Parallel(block_N, block_words):
                    row = T.min(bx * block_N + j, N - 1)
                    SFB_shared[j, w] = SFB[row // 32, T.min(word0 + w, sf_words - 1)]
                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    k_start=ko * block_K % SF_WORD_K,
                    sf_a_granularity_k=SF_BLOCK,
                    sf_b_granularity_k=SF_BLOCK,
                )

            T.sync_threads()
            T.copy(C_local, C_shared)
            col_pair = tx % pairs_per_row
            row_in_iter = tx // pairs_per_row
            g = col_pair // half
            p = tx % half
            col0 = bx * block_N + g * SF_BLOCK
            in_head = col0 % head_dim
            c0 = 2 * col_pair

            v0 = T.alloc_var(T.float32)
            v1 = T.alloc_var(T.float32)
            pos = T.alloc_var(T.int64)
            cosine = T.alloc_var(T.float32)
            sine = T.alloc_var(T.float32)
            amax = T.alloc_var(T.float32)
            bits = T.alloc_var(T.uint32)
            sf = T.alloc_var(T.int32)
            scale = T.alloc_var(T.float32)
            sv0 = T.alloc_var(T.float32)
            sv1 = T.alloc_var(T.float32)
            c_lo = T.alloc_var(T.int32)
            c_hi = T.alloc_var(T.int32)

            for it in (
                T.unroll(iters, explicit=False, annotations={"pragma_unroll_factor": 1})
                if rolled
                else T.serial(iters)
            ):
                row_local = it * rows_per_iter + row_in_iter
                m = by * block_M + row_local
                if m < M:
                    v0 = T.cast(C_shared[row_local, c0], T.float32)
                    v1 = T.cast(C_shared[row_local, c0 + 1], T.float32)
                    if in_head >= rope_start:
                        pos = Positions[m]
                        pair_base = (in_head - rope_start) // 2
                        cosine = CosSin[pos, pair_base + p]
                        sine = CosSin[pos, rope_half + pair_base + p]
                        r0 = T.cast(T.ieee_fmaf(v0, cosine, -(v1 * sine)), T.bfloat16)
                        r1 = T.cast(T.ieee_fmaf(v0, sine, v1 * cosine), T.bfloat16)
                        v0 = T.cast(r0, T.float32)
                        v1 = T.cast(r1, T.float32)
                    amax = T.max(T.float32(6.0 * 2.0**-126), T.max(T.abs(v0), T.abs(v1)))
                    amax = T.max(amax, T.shfl_xor(amax, 8))
                    amax = T.max(amax, T.shfl_xor(amax, 4))
                    amax = T.max(amax, T.shfl_xor(amax, 2))
                    amax = T.max(amax, T.shfl_xor(amax, 1))
                    bits = T.reinterpret(T.fmul(amax, T.float32(1.0 / 6.0)), T.uint32)
                    sf = T.cast(T.shift_right(bits, 23) & 0xFF, T.int32) + T.cast(
                        (bits & 0x7FFFFF) != 0, T.int32
                    )
                    scale = T.reinterpret(
                        T.cast(T.shift_left(254 - sf, 23), T.uint32), T.float32
                    )
                    sv0 = T.fmul(v0, scale)
                    sv1 = T.fmul(v1, scale)
                    if conv == "sw":
                        c_lo = bench._sw_e2m1_code(sv0)
                        c_hi = bench._sw_e2m1_code(sv1)
                        Q[m, col0 // 2 + p] = T.cast(c_lo | (c_hi << 4), T.uint8)
                    else:
                        Q[m, col0 // 2 + p] = _e2m1x2(sv0, sv1)
                    if p == 0:
                        S[m, col0 // SF_BLOCK] = T.cast(sf, T.uint8)

    return gemm_mxfp8_rope_mxfp4


def _stats(s):
    s = sorted(s)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    mean = sum(s) / n
    return {
        "median_us": round(med, 4),
        "q25_us": round(s[int(0.25 * (n - 1))], 4),
        "q75_us": round(s[int(0.75 * (n - 1))], 4),
        "mean_us": round(mean, 4),
        "std_us": round(math.sqrt(sum((x - mean) ** 2 for x in s) / max(n - 1, 1)), 4),
        "min_us": round(s[0], 4),
    }


def interleaved(fns, rounds, reps, warmup=40):
    names = list(fns)
    for _ in range(warmup):
        for n in names:
            fns[n]()
    torch.cuda.synchronize()
    samples = {n: [] for n in names}
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for r in range(rounds):
        order = names[r % len(names):] + names[: r % len(names)]
        if (r // len(names)) % 2:
            order = order[::-1]
        for n in order:
            e0.record()
            for _ in range(reps):
                fns[n]()
            e1.record()
            torch.cuda.synchronize()
            samples[n].append(e0.elapsed_time(e1) * 1000.0 / reps)
    return {n: _stats(samples[n]) for n in names}


def graphs(fns, rounds, reps):
    """Each variant's ``reps`` launches captured in one CUDA graph; per-launch us."""
    gs = {}
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for n, f in fns.items():
            for _ in range(3):
                f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    for n, f in fns.items():
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                f()
        gs[n] = g
    return interleaved({n: g.replay for n, g in gs.items()}, rounds, 1, warmup=10)


def dump(args, kerns):
    if not args.dump:
        return
    args.dump.mkdir(parents=True, exist_ok=True)
    for name, k in kerns.items():
        (args.dump / f"{name}.cu").write_text(k.get_kernel_source())
        try:
            (args.dump / f"{name}.sass").write_text(k._get_sass())
        except Exception as e:  # noqa: BLE001
            (args.dump / f"{name}.sass").write_text(f"sass failed: {e}\n")


def run_paged(args, out):
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(20261108)
    modes = args.modes.split(",")
    kern = {}
    for m in modes:
        if m == "sw":
            kern[m] = bench._sw_mxfp4_quantize_v41(paged=True, page_size=64)
        elif m == "pr":
            kern[m] = mxfp4_quantize_v41(paged=True, page_size=64)
        else:
            kern[m] = paged_variant(m, 64)
    out["sass"] = {m: bench._sass_stats(k) for m, k in kern.items()}
    dump(args, {f"paged_{m}": k for m, k in kern.items()})
    pb = page_bytes(64)
    pools = {m: torch.zeros((256, pb), dtype=torch.uint8, device=dev) for m in modes}
    bench._warmup_gpu_clock(dev)
    for M in [int(x) for x in args.ms.split(",")]:
        x = torch.randn((M, INDEX_DIM), generator=gen, device=dev, dtype=torch.bfloat16)
        slots = torch.arange(M, device=dev, dtype=torch.int64)
        for p in pools.values():
            p.zero_()
        for m in modes:
            kern[m](x, slots, pools[m])
        torch.cuda.synchronize()
        exact = {m: bool(torch.equal(pools["sw"], pools[m])) for m in modes if m != "null"}
        reps = args.repeats if M <= 512 else max(5, args.repeats // 5)
        fns = {m: (lambda m=m: kern[m](x, slots, pools[m])) for m in modes}
        row = {"M": M, "exact_vs_sw": exact, "eager": interleaved(fns, args.rounds, reps)}
        if args.graph:
            row["graph"] = graphs(fns, args.rounds, reps)
        out["rows"].append(row)
        print(M, {m: row["eager"][m]["median_us"] for m in modes},
              {m: row["graph"][m]["median_us"] for m in modes} if args.graph else "", exact, flush=True)


def run_gemm(args, out):
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(20261108)
    n, k, hd, rd = 4096, 1280, 128, 64
    sf_words = scale_words(k)
    b = torch.randn((n, k), generator=gen, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    sfb = torch.randint(120, 126, (n // 32, sf_words * 4), generator=gen, device=dev,
                        dtype=torch.int32).to(torch.uint8).view(torch.uint32)
    cos_sin = torch.randn((16384, rd), generator=gen, device=dev, dtype=torch.float32)
    b_tc = b.view(n * (k // TILE_WEIGHT_K), TILE_WEIGHT_K)
    modes = args.modes.split(",")

    def make(mode, bm):
        if mode == "sw":
            return bench._sw_mxfp8_gemm_rope_mxfp4_decode(n, k, hd, rd, block_M=bm, padded_rows=True,
                                                          tile_contiguous=True)
        if mode == "pr":
            return mxfp8_gemm_rope_mxfp4_decode(n, k, hd, rd, block_M=bm, padded_rows=True,
                                                tile_contiguous=True)
        conv, rolled = {"hw_serial": ("hw", False), "hw_rolled": ("hw", True),
                        "sw_rolled": ("sw", True), "sw_serial": ("sw", False)}[mode]
        return gemm_variant(n, k, conv, rolled, hd, rd, block_M=bm)

    kern = {}
    for bm in (16, 32, 64):
        for m in modes:
            kern[(m, bm)] = make(m, bm)
            out["sass"][f"{m}_bm{bm}"] = bench._sass_stats(kern[(m, bm)])
    dump(args, {f"gemm_{m}_bm{bm}": kk for (m, bm), kk in kern.items()})
    bench._warmup_gpu_clock(dev)
    for M in [int(x) for x in args.ms.split(",")]:
        bm = 16 if M <= 16 else (32 if M <= 32 else 64)
        a = torch.randn((bm, k), generator=gen, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        sfa = torch.randint(120, 126, (bm, sf_words * 4), generator=gen, device=dev,
                            dtype=torch.int32).to(torch.uint8).view(torch.uint32)
        pos = torch.randint(0, 16384, (M,), generator=gen, device=dev, dtype=torch.int64)
        qs = {m: torch.empty((M, n // 2), dtype=torch.uint8, device=dev) for m in modes}
        ss = {m: torch.empty((M, n // 32), dtype=torch.uint8, device=dev) for m in modes}
        for m in modes:
            kern[(m, bm)](a, b_tc, sfa, sfb, pos, cos_sin, qs[m], ss[m])
        torch.cuda.synchronize()
        exact = {m: bool(torch.equal(qs["sw"], qs[m]) and torch.equal(ss["sw"], ss[m])) for m in modes}
        fns = {m: (lambda m=m: kern[(m, bm)](a, b_tc, sfa, sfb, pos, cos_sin, qs[m], ss[m])) for m in modes}
        row = {"M": M, "block_M": bm, "exact_vs_sw": exact, "eager": interleaved(fns, args.rounds, args.repeats)}
        if args.graph:
            row["graph"] = graphs(fns, args.rounds, args.repeats)
        out["rows"].append(row)
        print(M, {m: row["eager"][m]["median_us"] for m in modes},
              {m: row["graph"][m]["median_us"] for m in modes} if args.graph else "", exact, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modes", default="sw,pr")
    ap.add_argument("--ms", default="1,2,4,6,8,16,32,512,2048,8192")
    ap.add_argument("--rounds", type=int, default=80)
    ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--dump", type=Path, default=None)
    args = ap.parse_args()
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(args) | {
        "out": str(args.out), "dump": str(args.dump)}, "sass": {}, "rows": []}
    if args.what == "paged":
        run_paged(args, out)
    elif args.what == "gemm":
        run_gemm(args, out)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
