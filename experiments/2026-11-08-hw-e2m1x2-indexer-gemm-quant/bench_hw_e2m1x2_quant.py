#!/usr/bin/env python3
"""Benchmark & bit-exactness verification for hardware cvt.rn.satfinite.e2m1x2.f32
in DeepSeek-V4.1 TileLang indexer and fused GEMM+RoPE+MXFP4 kernels.
"""

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import pdl
from vllm.models.deepseek_v4_1.tilelang.front import ROPE_DIM, _THREADS, _VEC, _row_split
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    SF_BLOCK,
    SF_WORD_K,
    TILE_WEIGHT_K,
    TILE_WEIGHT_N,
    _E2M1_HEADER,
    _e2m1x2_byte,
    block_scale_words,
    mxfp8_gemm_rope_mxfp4_decode,
    scale_words,
)
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    INDEX_DIM,
    INDEX_HEADS,
    mxfp4_norm_rope_quantize_v41,
    mxfp4_quantize_v41,
    page_bytes,
)

LAYERS = 61


def _sw_e2m1_code(x):
    """Baseline 7-comparison software E2M1 quantization."""
    m = T.abs(x)
    code = (
        T.cast(m > 0.25, T.int32)
        + T.cast(m >= 0.75, T.int32)
        + T.cast(m > 1.25, T.int32)
        + T.cast(m >= 1.75, T.int32)
        + T.cast(m > 2.5, T.int32)
        + T.cast(m >= 3.5, T.int32)
        + T.cast(m > 5.0, T.int32)
    )
    sign = T.cast(T.shift_right(T.reinterpret(x, T.uint32), 31), T.int32)
    return code | (sign << 3)


def _sw_mxfp4_byte(vals, first, scale):
    lo = _sw_e2m1_code(T.fmul(T.cast(vals[first], T.float32), scale))
    hi = _sw_e2m1_code(T.fmul(T.cast(vals[first + 1], T.float32), scale))
    return T.cast(lo | (hi << 4), T.uint8)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _sw_mxfp8_gemm_rope_mxfp4_decode(
    N: int,
    K: int,
    head_dim: int = 128,
    rope_dim: int = 64,
    block_M: int = 16,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 256,
    padded_rows: bool = True,
    swizzle_panel: int = 0,
    tile_contiguous: bool = False,
    ratio: int = 1,
):
    pairs_per_row = block_N // 2
    k_blocks = K // block_K
    M = T.dynamic("M")
    Mp = T.dynamic("Mp") if padded_rows else M
    max_pos = T.dynamic("max_pos")
    in_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    sf_words = scale_words(K)
    block_words = block_scale_words(block_K)
    k_tiles = K // TILE_WEIGHT_K
    sub_tiles_k = block_K // TILE_WEIGHT_K
    b_shape = (N * k_tiles, TILE_WEIGHT_K) if tile_contiguous else (N, K)
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
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            if swizzle_panel:
                T.use_swizzle(panel_size=swizzle_panel)
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
                if padded_rows:
                    for i, k in T.Parallel(block_M, block_K):
                        A_shared[i, k] = A[
                            T.min(by * block_M + i, M - 1),
                            ko * block_K + k,
                        ]
                else:
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                if not tile_contiguous:
                    T.copy(B[bx * block_N, ko * block_K], B_shared)
                elif sub_tiles_k == 1:
                    row0 = (
                        ((bx * block_N // TILE_WEIGHT_N) * k_tiles + ko) * TILE_WEIGHT_N
                        + (bx * block_N % TILE_WEIGHT_N)
                    )
                    T.copy(B[row0, 0], B_shared)
                else:
                    for sub_k in T.serial(sub_tiles_k):
                        row0 = (
                            (
                                (bx * block_N // TILE_WEIGHT_N) * k_tiles
                                + ko * sub_tiles_k
                                + sub_k
                            )
                            * TILE_WEIGHT_N
                            + (bx * block_N % TILE_WEIGHT_N)
                        )
                        T.copy(
                            B[row0 : row0 + block_N, 0:TILE_WEIGHT_K],
                            B_shared[
                                0:block_N,
                                sub_k * TILE_WEIGHT_K : (sub_k + 1) * TILE_WEIGHT_K,
                            ],
                        )
                word0 = ko * block_K // SF_WORD_K
                for i, w in T.Parallel(block_M, block_words):
                    SFA_shared[i, w] = SFA[
                        T.min(by * block_M + i, M - 1),
                        T.min(word0 + w, sf_words - 1),
                    ]
                for j, w in T.Parallel(block_N, block_words):
                    row = T.min(bx * block_N + j, N - 1)
                    SFB_shared[j, w] = SFB[
                        row // 32,
                        T.min(word0 + w, sf_words - 1),
                    ]
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

            for it in T.serial(iters):
                row_local = it * rows_per_iter + row_in_iter
                m = by * block_M + row_local
                if m < M:
                    v0 = T.cast(C_shared[row_local, c0], T.float32)
                    v1 = T.cast(C_shared[row_local, c0 + 1], T.float32)
                    if in_head >= rope_start:
                        pos = (
                            Positions[m] // ratio * ratio
                            if ratio > 1
                            else Positions[m]
                        )
                        pair_base = (in_head - rope_start) // 2
                        cosine = CosSin[pos, pair_base + p]
                        sine = CosSin[pos, rope_half + pair_base + p]
                        r0 = T.cast(T.ieee_fmaf(v0, cosine, -(v1 * sine)), T.bfloat16)
                        r1 = T.cast(T.ieee_fmaf(v0, sine, v1 * cosine), T.bfloat16)
                        v0 = T.cast(r0, T.float32)
                        v1 = T.cast(r1, T.float32)
                    amax = T.max(
                        T.float32(6.0 * 2.0**-126),
                        T.max(T.abs(v0), T.abs(v1)),
                    )
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
                    c_lo = _sw_e2m1_code(sv0)
                    c_hi = _sw_e2m1_code(sv1)
                    Q[m, col0 // 2 + p] = T.cast(c_lo | (c_hi << 4), T.uint8)
                    if p == 0:
                        S[m, col0 // SF_BLOCK] = T.cast(sf, T.uint8)

    return gemm_mxfp8_rope_mxfp4


@tilelang.jit
def _sw_mxfp4_quantize_v41(
    paged: bool = False,
    page_size: int = 64,
    threads: int = 128,
    rope_heads: int = 0,
    rope_dim: int = 0,
    ratio: int = 1,
):
    rows, pool_pages, pool_stride = T.dynamic("rows, pool_pages, pool_stride")
    tokens, positions_max = T.dynamic("tokens, positions_max")
    groups = INDEX_DIM // SF_BLOCK
    half = SF_BLOCK // 2
    start_group = groups - (rope_dim // SF_BLOCK) if rope_heads > 0 else groups
    rope_half = rope_dim // 2

    @T.macro
    def body(X, Positions, CosSin, Q, S, Slots, Pool):
        with T.Kernel(T.ceildiv(rows * groups, threads), threads=threads) as bx:
            tx = T.get_thread_binding()
            g = bx * threads + tx
            row, k = g // groups, g % groups
            vals = T.alloc_local([SF_BLOCK], T.bfloat16)
            pos_l = T.alloc_local([1], T.int64)
            slot_l = T.alloc_local([1], T.int64)
            amax = T.alloc_var(T.float32)
            if row < rows:
                if rope_heads > 0:
                    if k >= start_group:
                        token = row // rope_heads
                        pos_l[0] = T.cast(
                            Positions[token] // ratio * ratio
                            if ratio > 1
                            else Positions[token],
                            T.int64,
                        )
                if paged:
                    slot_l[0] = Slots[row]
            pdl.chain()
            if row < rows:
                for i in T.serial(SF_BLOCK):
                    vals[i] = X[row, k * SF_BLOCK + i]
                if rope_heads > 0:
                    if k >= start_group:
                        pos = pos_l[0]
                        pair_base = (k - start_group) * half
                        for i in T.unroll(half):
                            first = T.cast(vals[2 * i], T.float32)
                            second = T.cast(vals[2 * i + 1], T.float32)
                            cosine = CosSin[pos, pair_base + i]
                            sine = CosSin[pos, rope_half + pair_base + i]
                            vals[2 * i] = T.cast(
                                T.ieee_fmaf(first, cosine, -(second * sine)),
                                T.bfloat16,
                            )
                            vals[2 * i + 1] = T.cast(
                                T.ieee_fmaf(first, sine, second * cosine),
                                T.bfloat16,
                            )
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
                if paged:
                    slot = slot_l[0]
                    page, token = slot // page_size, slot % page_size
                    if (slot >= 0) & (page < pool_pages):
                        for i in T.serial(half):
                            Pool[page, token * (INDEX_DIM // 2) + k * half + i] = (
                                _sw_mxfp4_byte(vals, 2 * i, scale)
                            )
                        Pool[
                            page, page_size * (INDEX_DIM // 2) + token * groups + k
                        ] = T.cast(sf, T.uint8)
                else:
                    for i in T.serial(half):
                        Q[row, k * half + i] = _sw_mxfp4_byte(vals, 2 * i, scale)
                    S[row, k] = T.cast(sf, T.uint8)

    if paged:
        @T.prim_func
        def indexer_mxfp4_quantize(
            X: T.Tensor([rows, INDEX_DIM], T.bfloat16),
            Slots: T.Tensor([rows], T.int64),
            Pool: T.StridedTensor(
                [pool_pages, page_bytes(page_size)], [pool_stride, 1], T.uint8
            ),
        ):
            body(X, None, None, None, None, Slots, Pool)

        return indexer_mxfp4_quantize

    if rope_heads > 0:
        @T.prim_func
        def indexer_mxfp4_quantize(
            X: T.Tensor([rows, INDEX_DIM], T.bfloat16),
            Positions: T.Tensor([tokens], T.int64),
            CosSin: T.Tensor([positions_max, rope_dim], T.float32),
            Q: T.Tensor([rows, INDEX_DIM // 2], T.uint8),
            S: T.Tensor([rows, groups], T.uint8),
        ):
            body(X, Positions, CosSin, Q, S, None, None)

        return indexer_mxfp4_quantize

    @T.prim_func
    def indexer_mxfp4_quantize(
        X: T.Tensor([rows, INDEX_DIM], T.bfloat16),
        Q: T.Tensor([rows, INDEX_DIM // 2], T.uint8),
        S: T.Tensor([rows, groups], T.uint8),
    ):
        body(X, None, None, Q, S, None, None)

    return indexer_mxfp4_quantize


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _sw_mxfp4_norm_rope_quantize_v41(
    eps: float,
    page_size: int = 64,
    ratio: int = 1,
    slot_dtype: str = "int64",
    shards: int = 1,
):
    k = _row_split(INDEX_DIM)
    kb, kt = k["block_m"], k["row_threads"]
    half = ROPE_DIM // 2
    pairs = _VEC // 2
    out_scale = 1.0
    rows, positions_max = T.dynamic("rows, positions_max")
    pool_pages, pool_stride = T.dynamic("pool_pages, pool_stride")

    @T.macro
    def body(X, Weight, Positions, CosSin, Slots, Pool):
        with T.Kernel(T.ceildiv(rows, kb), threads=_THREADS) as bx:
            tid = T.get_thread_binding()
            k_sum_partial = T.alloc_fragment((kb, kt), T.float32)
            k_warp_reducer = T.alloc_reducer((kb, 1), T.float32, op="sum")
            k_warp_fragment = T.alloc_fragment((kb, 1), T.float32)
            k_sum_shared = T.alloc_shared((kb, 1), T.float32)
            k_row_partial = T.alloc_fragment((kb, kt, 1), T.float32)
            k_sum_reducer = T.alloc_reducer((kb, kt), T.float32, op="sum")
            k_sum_fragment = T.alloc_fragment((kb, kt), T.float32)
            T.annotate_layout(
                {
                    k_sum_partial: k["row_layout"],
                    k_warp_fragment: k["warp_layout"],
                    k_row_partial: k["row_reduce_layout"],
                    k_sum_fragment: k["row_layout"],
                }
            )
            k_x = T.alloc_local((k["num_thread_vec"], _VEC), T.bfloat16)
            k_y = T.alloc_local((k["num_thread_vec"], _VEC), T.bfloat16)
            acc_local = T.alloc_local((_VEC,), T.float32)
            y_fp32_local = T.alloc_local((_VEC,), T.float32)
            sum_local = T.alloc_local((2,), T.float32)
            weight_local = T.alloc_local((k["num_thread_vec"], _VEC), T.bfloat16)
            cs = T.alloc_local((_VEC,), T.float32)
            slot_local = T.alloc_local((1,), T.int64)

            row_id = tid // kt
            lane = tid % kt
            token_id = bx * kb + row_id
            for vi in T.unroll(k["num_thread_vec"]):
                token_offset = (vi * kt + lane) * _VEC
                for j in T.vectorized(_VEC):
                    weight_local[vi, j] = Weight[token_offset + j]
            if token_id < rows:
                slot_local[0] = T.cast(Slots[token_id], T.int64)
                pos = (
                    Positions[token_id] // ratio * ratio
                    if ratio > 1
                    else Positions[token_id]
                )
                for i in T.vectorized(pairs):
                    cs[i] = CosSin[pos, lane * pairs + i]
                for i in T.vectorized(pairs):
                    cs[pairs + i] = CosSin[pos, half + lane * pairs + i]
            else:
                slot_local[0] = T.int64(-1)
            pdl.chain()
            T.clear(sum_local)
            for vi in T.unroll(k["num_thread_vec"]):
                token_offset = (vi * kt + lane) * _VEC
                if token_id < rows:
                    if shards > 1:
                        for j in T.vectorized(_VEC):
                            acc_local[j] = X[0, token_id, token_offset + j]
                        for s in T.serial(1, shards):
                            for j in T.vectorized(_VEC):
                                acc_local[j] += X[s, token_id, token_offset + j]
                        for j in T.unroll(_VEC):
                            k_x[vi, j] = T.cast(acc_local[j], T.bfloat16)
                    else:
                        for j in T.vectorized(_VEC):
                            k_x[vi, j] = X[token_id, token_offset + j]
                else:
                    T.clear(k_x[vi, :])
                for j in T.vectorized(_VEC):
                    sum_local[j % 2] += T.float32(k_x[vi, j]) * T.float32(k_x[vi, j])
            k_sum_partial[row_id, lane] = sum_local[0] + sum_local[1]
            T.reducer_init(k_warp_reducer)
            for i, w, j in T.Parallel(
                kb, 1, kt, loop_layout=k["warp_reduce_layout"]
            ):
                T.reducer_update(k_warp_reducer[i, w], k_sum_partial[i, w * kt + j])
            T.finalize_reducer(k_warp_reducer, k_warp_fragment)
            if lane == 0:
                k_sum_shared[row_id, 0] = k_warp_fragment[row_id, 0]
            T.sync_threads()
            for i, j, w in T.Parallel(kb, kt, 1, loop_layout=k["row_reduce_layout"]):
                k_row_partial[i, j, w] = k_sum_shared[i, w]
            T.reducer_init(k_sum_reducer)
            for i, j, w in T.Parallel(kb, kt, 1, loop_layout=k["row_reduce_layout"]):
                T.reducer_update(k_sum_reducer[i, j], k_row_partial[i, j, w])
            T.finalize_reducer(k_sum_reducer, k_sum_fragment)

            rstd_value = T.alloc_var(
                T.float32,
                init=T.rsqrt(k_sum_fragment[row_id, lane] / INDEX_DIM + eps),
            )
            for vi in T.unroll(k["num_thread_vec"]):
                for j in T.vectorized(_VEC):
                    y_fp32_local[j] = (
                        T.float32(k_x[vi, j])
                        * rstd_value
                        * (T.float32(weight_local[vi, j]) * out_scale)
                    )
                for j in T.vectorized(_VEC):
                    k_y[vi, j] = y_fp32_local[j]

            if token_id < rows:
                for j in T.unroll(pairs):
                    first = T.cast(k_y[1, 2 * j], T.float32)
                    second = T.cast(k_y[1, 2 * j + 1], T.float32)
                    cosine = cs[j]
                    sine = cs[pairs + j]
                    k_y[1, 2 * j] = T.cast(
                        T.ieee_fmaf(first, cosine, -(second * sine)), T.bfloat16
                    )
                    k_y[1, 2 * j + 1] = T.cast(
                        T.ieee_fmaf(first, sine, second * cosine), T.bfloat16
                    )

            for vi in T.unroll(k["num_thread_vec"]):
                group = vi * 2 + (lane // 4)
                group_lane = lane % 4
                amax = T.alloc_var(T.float32, init=6.0 * 2.0**-126)
                for j in T.unroll(_VEC):
                    amax = T.max(amax, T.abs(T.cast(k_y[vi, j], T.float32)))
                amax = T.max(amax, T.shfl_xor(amax, 1))
                amax = T.max(amax, T.shfl_xor(amax, 2))
                if token_id < rows:
                    slot = slot_local[0]
                    page = slot // page_size
                    tok = slot % page_size
                    if (slot >= 0) & (page < pool_pages):
                        bits = T.reinterpret(
                            T.fmul(amax, T.float32(1.0 / 6.0)), T.uint32
                        )
                        sf = T.cast(T.shift_right(bits, 23) & 0xFF, T.int32) + T.cast(
                            (bits & 0x7FFFFF) != 0, T.int32
                        )
                        scale = T.reinterpret(
                            T.cast(T.shift_left(254 - sf, 23), T.uint32), T.float32
                        )
                        for i in T.unroll(pairs):
                            lo = _sw_e2m1_code(
                                T.fmul(T.cast(k_y[vi, 2 * i], T.float32), scale)
                            )
                            hi = _sw_e2m1_code(
                                T.fmul(T.cast(k_y[vi, 2 * i + 1], T.float32), scale)
                            )
                            Pool[
                                page,
                                tok * (INDEX_DIM // 2)
                                + group * 16
                                + group_lane * pairs
                                + i,
                            ] = T.cast(lo | (hi << 4), T.uint8)
                        if group_lane == 0:
                            Pool[
                                page,
                                page_size * (INDEX_DIM // 2) + tok * 4 + group,
                            ] = T.cast(sf, T.uint8)

    if shards > 1:
        @T.prim_func
        def indexer_mxfp4_quantize(
            X: T.Tensor([shards, rows, INDEX_DIM], T.float32),
            Weight: T.Tensor([INDEX_DIM], T.bfloat16),
            Positions: T.Tensor([rows], T.int64),
            CosSin: T.Tensor([positions_max, ROPE_DIM], T.float32),
            Slots: T.Tensor([rows], slot_dtype),
            Pool: T.StridedTensor(
                [pool_pages, page_bytes(page_size)], [pool_stride, 1], T.uint8
            ),
        ):
            body(X, Weight, Positions, CosSin, Slots, Pool)

        return indexer_mxfp4_quantize

    @T.prim_func
    def indexer_mxfp4_quantize(
        X: T.Tensor([rows, INDEX_DIM], T.bfloat16),
        Weight: T.Tensor([INDEX_DIM], T.bfloat16),
        Positions: T.Tensor([rows], T.int64),
        CosSin: T.Tensor([positions_max, ROPE_DIM], T.float32),
        Slots: T.Tensor([rows], slot_dtype),
        Pool: T.StridedTensor(
            [pool_pages, page_bytes(page_size)], [pool_stride, 1], T.uint8
        ),
    ):
        body(X, Weight, Positions, CosSin, Slots, Pool)

    return indexer_mxfp4_quantize


def _sass_stats(kernel) -> dict[str, int | bool | list[str]]:
    k_insns = -1
    cubin_insns_count = -1
    max_reg = -1
    stgs: list[str] = []
    f2fp_count = 0
    try:
        sass = kernel._get_sass()
        cubin_insns_count = len(re.findall(r"/\*[0-9a-f]+\*/", sass))
        m = re.search(r"\.text\.\w+_kernel:.*?(?=//---|\Z)", sass, re.S)
        body = m.group(0) if m else sass
        k_insns = len(re.findall(r"/\*[0-9a-f]+\*/", body))
        regs = [int(x) for x in re.findall(r"\bR(\d+)\b", body)]
        max_reg = max(regs) if regs else 0
        stgs = [
            re.sub(r"/\*.*?\*/\s*", "", line).strip()
            for line in body.splitlines()
            if "STG." in line
        ]
        f2fp_count = len(re.findall(r"F2FP\.SATFINITE\.E2M1\.F32\.PACK_AB", body))
    except Exception:
        pass
    return {
        "static_sass_instructions": k_insns,
        "cubin_static_sass_instructions": cubin_insns_count,
        "max_register": max_reg,
        "global_stores": len(stgs),
        "global_store_ops": stgs,
        "f2fp_e2m1_pack_ab_count": f2fp_count,
        "has_f2fp_e2m1_pack_ab": f2fp_count > 0,
    }


def _warmup_gpu_clock(device: torch.device) -> None:
    w = torch.ones((2048, 2048), dtype=torch.bfloat16, device=device)
    for _ in range(30):
        w = torch.mm(w, w)
    torch.cuda.synchronize()


def _stats(samples_us: list[float]) -> dict[str, float]:
    s = sorted(samples_us)
    n = len(s)
    med = s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    q25 = s[int(0.25 * (n - 1))]
    q75 = s[int(0.75 * (n - 1))]
    mean = sum(s) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in s) / max(n - 1, 1))
    return {
        "median_us": round(med, 4),
        "iqr_us": round(q75 - q25, 4),
        "mean_us": round(mean, 4),
        "std_us": round(std, 4),
        "min_us": round(s[0], 4),
        "max_us": round(s[-1], 4),
    }


def _paired_bench(
    fn_sw,
    fn_hw,
    *,
    warmup: int = 40,
    rounds: int = 80,
    repeats_per_round: int = 25,
) -> dict:
    for _ in range(warmup):
        fn_sw()
        fn_hw()
    torch.cuda.synchronize()

    sw_samples: list[float] = []
    hw_samples: list[float] = []
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    for r in range(rounds):
        order = (("sw", fn_sw), ("hw", fn_hw)) if r % 2 == 0 else (("hw", fn_hw), ("sw", fn_sw))
        for tag, fn in order:
            ev_start.record()
            for _ in range(repeats_per_round):
                fn()
            ev_end.record()
            torch.cuda.synchronize()
            us = (ev_start.elapsed_time(ev_end) * 1000.0) / repeats_per_round
            if tag == "sw":
                sw_samples.append(us)
            else:
                hw_samples.append(us)

    sw_st = _stats(sw_samples)
    hw_st = _stats(hw_samples)
    delta_us = hw_st["median_us"] - sw_st["median_us"]
    delta_pct = (delta_us / sw_st["median_us"]) * 100.0 if sw_st["median_us"] > 0 else 0.0
    return {
        "sw": sw_st,
        "hw": hw_st,
        "delta_median_us": round(delta_us, 4),
        "delta_pct": round(delta_pct, 2),
        "speedup": round(sw_st["median_us"] / hw_st["median_us"], 4) if hw_st["median_us"] > 0 else 0.0,
        "step_61_layers_delta_us": round(delta_us * LAYERS, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    gen = torch.Generator(device=device).manual_seed(20261108)

    results: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": torch.cuda.get_device_name(device),
        "sass": {},
        "benchmarks": {
            "mxfp8_gemm_rope_mxfp4_decode": [],
            "mxfp4_quantize_v41_query_rope": [],
            "mxfp4_quantize_v41_paged_keys": [],
            "mxfp4_norm_rope_quantize_v41_shards1": [],
            "mxfp4_norm_rope_quantize_v41_shards6": [],
        },
    }

    # 1. mxfp8_gemm_rope_mxfp4_decode (indexer.wq_b: N=4096, K=1280, head_dim=128, rope_dim=64)
    n, k, head_dim, rope_dim = 4096, 1280, 128, 64
    sf_words = scale_words(k)
    b = torch.randn((n, k), generator=gen, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    sfb = (
        torch.randint(
            120, 126, (n // 32, sf_words * 4), generator=gen, device=device, dtype=torch.int32
        )
        .to(torch.uint8)
        .view(torch.uint32)
    )
    cos_sin = torch.randn((16384, rope_dim), generator=gen, device=device, dtype=torch.float32)

    gemm_sw_kernels = {}
    gemm_hw_kernels = {}
    for bm in (16, 32, 64):
        gemm_sw_kernels[bm] = _sw_mxfp8_gemm_rope_mxfp4_decode(
            n, k, head_dim, rope_dim, block_M=bm, padded_rows=True, tile_contiguous=True
        )
        gemm_hw_kernels[bm] = mxfp8_gemm_rope_mxfp4_decode(
            n, k, head_dim, rope_dim, block_M=bm, padded_rows=True, tile_contiguous=True
        )
        results["sass"][f"gemm_mxfp8_rope_mxfp4_bm{bm}"] = {
            "sw": _sass_stats(gemm_sw_kernels[bm]),
            "hw": _sass_stats(gemm_hw_kernels[bm]),
        }
    _warmup_gpu_clock(device)

    for m in (1, 2, 4, 6, 8, 16, 32, 64):
        bm = 16 if m <= 16 else (32 if m <= 32 else 64)
        a = torch.randn((bm, k), generator=gen, device=device, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        sfa = (
            torch.randint(
                120, 126, (bm, sf_words * 4), generator=gen, device=device, dtype=torch.int32
            )
            .to(torch.uint8)
            .view(torch.uint32)
        )
        pos = torch.randint(0, 16384, (m,), generator=gen, device=device, dtype=torch.int64)
        q_sw = torch.empty((m, n // 2), dtype=torch.uint8, device=device)
        s_sw = torch.empty((m, n // 32), dtype=torch.uint8, device=device)
        q_hw = torch.empty((m, n // 2), dtype=torch.uint8, device=device)
        s_hw = torch.empty((m, n // 32), dtype=torch.uint8, device=device)

        k_sw = gemm_sw_kernels[bm]
        k_hw = gemm_hw_kernels[bm]
        b_tc = b.view(n * (k // TILE_WEIGHT_K), TILE_WEIGHT_K)
        k_sw(a, b_tc, sfa, sfb, pos, cos_sin, q_sw, s_sw)
        k_hw(a, b_tc, sfa, sfb, pos, cos_sin, q_hw, s_hw)
        torch.cuda.synchronize()
        exact_q = bool(torch.equal(q_sw, q_hw))
        exact_s = bool(torch.equal(s_sw, s_hw))
        assert exact_q and exact_s, f"GEMM+RoPE+MXFP4 mismatch at M={m}"

        perf = _paired_bench(
            lambda: k_sw(a, b_tc, sfa, sfb, pos, cos_sin, q_sw, s_sw),
            lambda: k_hw(a, b_tc, sfa, sfb, pos, cos_sin, q_hw, s_hw),
            rounds=args.rounds,
            repeats_per_round=args.repeats,
        )
        results["benchmarks"]["mxfp8_gemm_rope_mxfp4_decode"].append(
            {"M": m, "block_M": bm, "bit_exact": exact_q and exact_s, **perf}
        )

    # 2. mxfp4_quantize_v41 (query with RoPE: rows = M * 32, rope_heads=32, rope_dim=64)
    q_rope_sw = _sw_mxfp4_quantize_v41(rope_heads=INDEX_HEADS, rope_dim=64, ratio=1)
    q_rope_hw = mxfp4_quantize_v41(rope_heads=INDEX_HEADS, rope_dim=64, ratio=1)
    results["sass"]["mxfp4_quantize_v41_query_rope"] = {
        "sw": _sass_stats(q_rope_sw),
        "hw": _sass_stats(q_rope_hw),
    }
    _warmup_gpu_clock(device)

    for m in (1, 2, 4, 6, 8, 16, 32, 512, 2048, 8192):
        rows = m * INDEX_HEADS
        x = torch.randn((rows, INDEX_DIM), generator=gen, device=device, dtype=torch.bfloat16)
        # Inject exact tie midpoints on a subset of elements to verify tie-to-even parity
        if rows >= 4:
            x[0, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device=device, dtype=torch.bfloat16)
        pos = torch.randint(0, 16384, (m,), generator=gen, device=device, dtype=torch.int64)
        q_sw = torch.empty((rows, INDEX_DIM // 2), dtype=torch.uint8, device=device)
        s_sw = torch.empty((rows, INDEX_DIM // 32), dtype=torch.uint8, device=device)
        q_hw = torch.empty((rows, INDEX_DIM // 2), dtype=torch.uint8, device=device)
        s_hw = torch.empty((rows, INDEX_DIM // 32), dtype=torch.uint8, device=device)

        q_rope_sw(x, pos, cos_sin, q_sw, s_sw)
        q_rope_hw(x, pos, cos_sin, q_hw, s_hw)
        torch.cuda.synchronize()
        exact = bool(torch.equal(q_sw, q_hw) and torch.equal(s_sw, s_hw))
        assert exact, f"mxfp4_quantize_v41 query_rope mismatch at M={m}"

        reps = args.repeats if m <= 512 else max(5, args.repeats // 5)
        perf = _paired_bench(
            lambda: q_rope_sw(x, pos, cos_sin, q_sw, s_sw),
            lambda: q_rope_hw(x, pos, cos_sin, q_hw, s_hw),
            rounds=args.rounds,
            repeats_per_round=reps,
        )
        results["benchmarks"]["mxfp4_quantize_v41_query_rope"].append(
            {"M": m, "rows": rows, "bit_exact": exact, **perf}
        )

    # 3. mxfp4_quantize_v41 (paged key write: rows = M, paged=True, page_size=64)
    page_size = 64
    q_paged_sw = _sw_mxfp4_quantize_v41(paged=True, page_size=page_size)
    q_paged_hw = mxfp4_quantize_v41(paged=True, page_size=page_size)
    results["sass"]["mxfp4_quantize_v41_paged_keys"] = {
        "sw": _sass_stats(q_paged_sw),
        "hw": _sass_stats(q_paged_hw),
    }
    _warmup_gpu_clock(device)

    max_pages = 256
    pbytes = page_bytes(page_size)
    pool_sw = torch.zeros((max_pages, pbytes), dtype=torch.uint8, device=device)
    pool_hw = torch.zeros((max_pages, pbytes), dtype=torch.uint8, device=device)

    for m in (1, 2, 4, 6, 8, 16, 32, 512, 2048, 8192):
        x = torch.randn((m, INDEX_DIM), generator=gen, device=device, dtype=torch.bfloat16)
        slots = torch.arange(m, device=device, dtype=torch.int64)
        pool_sw.zero_()
        pool_hw.zero_()
        q_paged_sw(x, slots, pool_sw)
        q_paged_hw(x, slots, pool_hw)
        torch.cuda.synchronize()
        exact = bool(torch.equal(pool_sw, pool_hw))
        assert exact, f"mxfp4_quantize_v41 paged mismatch at M={m}"

        reps = args.repeats if m <= 512 else max(5, args.repeats // 5)
        perf = _paired_bench(
            lambda: q_paged_sw(x, slots, pool_sw),
            lambda: q_paged_hw(x, slots, pool_hw),
            rounds=args.rounds,
            repeats_per_round=reps,
        )
        results["benchmarks"]["mxfp4_quantize_v41_paged_keys"].append(
            {"M": m, "bit_exact": exact, **perf}
        )

    # 4. mxfp4_norm_rope_quantize_v41 (shards=1 and shards=6)
    eps = 1e-6
    weight = torch.randn((INDEX_DIM,), generator=gen, device=device, dtype=torch.bfloat16)
    for shards, bench_key in (
        (1, "mxfp4_norm_rope_quantize_v41_shards1"),
        (6, "mxfp4_norm_rope_quantize_v41_shards6"),
    ):
        nrq_sw = _sw_mxfp4_norm_rope_quantize_v41(
            eps, page_size=page_size, ratio=1, slot_dtype="int64", shards=shards
        )
        nrq_hw = mxfp4_norm_rope_quantize_v41(
            eps, page_size=page_size, ratio=1, slot_dtype="int64", shards=shards
        )
        results["sass"][bench_key] = {
            "sw": _sass_stats(nrq_sw),
            "hw": _sass_stats(nrq_hw),
        }
        _warmup_gpu_clock(device)

        m_list = (1, 2, 4, 6, 8, 16, 32, 512, 2048, 8192) if shards == 1 else (1, 2, 4, 6, 8, 16, 32)
        for m in m_list:
            if shards == 1:
                x = torch.randn((m, INDEX_DIM), generator=gen, device=device, dtype=torch.bfloat16)
            else:
                x = torch.randn((shards, m, INDEX_DIM), generator=gen, device=device, dtype=torch.float32)
            pos = torch.randint(0, 16384, (m,), generator=gen, device=device, dtype=torch.int64)
            slots = torch.arange(m, device=device, dtype=torch.int64)
            pool_sw.zero_()
            pool_hw.zero_()
            nrq_sw(x, weight, pos, cos_sin, slots, pool_sw)
            nrq_hw(x, weight, pos, cos_sin, slots, pool_hw)
            torch.cuda.synchronize()
            exact = bool(torch.equal(pool_sw, pool_hw))
            assert exact, f"{bench_key} mismatch at M={m}"

            reps = args.repeats if m <= 512 else max(5, args.repeats // 5)
            perf = _paired_bench(
                lambda: nrq_sw(x, weight, pos, cos_sin, slots, pool_sw),
                lambda: nrq_hw(x, weight, pos, cos_sin, slots, pool_hw),
                rounds=args.rounds,
                repeats_per_round=reps,
            )
            results["benchmarks"][bench_key].append(
                {"M": m, "shards": shards, "bit_exact": exact, **perf}
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
