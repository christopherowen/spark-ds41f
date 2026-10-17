# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang vocabulary heads: the exact 12-bit packed BF16 head and BF16 heads.

The packed row layout is B12X's: ``K`` sign-and-mantissa bytes, then ``K / 2``
code bytes. Byte ``j`` of a 128-column group's 64 code bytes holds the 4-bit
exponent code of column ``j`` in its low nibble and of column ``j + 64`` in its
high nibble; code 0 is exponent 0 and code ``c`` is ``base + c - 1``. The rare
values outside that window sit in the packed bytes as +0, and their exact BF16
bits in a per-row exception list (``row_offsets`` indexes it by row; each
entry is the value's bits in the low half and its column in the high half).

Each output is its row's 128-column groups in order, then its exceptions in
list order. Neither depends on the token count or the tile, so a token's
logits are the same bits in any batch.

Plain BF16 heads (DS4.1's 256-wide DSpark Markov projection) run TileLang's
BF16 GEMM with the same token tiles. Every head declares one kernel per token
tile; its owner declares it when the worker prepares kernels, after loading.
"""

import tilelang
import tilelang.language as T
import torch
from torch import nn

from . import prepare
from .gemm import bf16_gemm

GROUP = 128  # columns per code group
HALF = GROUP // 2  # one code byte holds a column of each half
VEC = 16  # columns of each half one thread decodes at a time
# Token tiles: one decode row in a 16-row tile reads the weights once, like 64.
ROW_TILES = (16, 32, 64)
SHARED_MEMORY = 101_376  # dynamic shared memory per block on SM120 / SM121


def shared_memory_bytes(
    block_M: int, block_N: int, num_stages: int, step_groups: int = 1
) -> int:
    """One buffer per pipeline stage of the activations and the packed bytes
    and codes, and one of the decoded weights, for ``step_groups`` groups per
    step."""
    staged = block_M * GROUP * 2 + block_N * (GROUP + HALF)
    decoded = block_N * GROUP * 2
    return step_groups * (num_stages * staged + decoded)


# Vocabulary tile, threads, groups per step and pipeline depth for each token
# tile, measured on GB10 for DS4.1's TP4 and TP3 vocabulary shards (K = 5,120).
TILE_CONFIGS = {
    16: dict(block_N=32, threads=128, step_groups=2, num_stages=4),
    32: dict(block_N=16, threads=128, step_groups=2, num_stages=4),
    64: dict(block_N=32, threads=256, step_groups=1, num_stages=4),
}


# BF16 heads: vocabulary tile and K block for each token tile (K = 256).
BF16_HEAD_CONFIGS = {
    16: dict(block_N=64, block_K=256),
    32: dict(block_N=64, block_K=256),
    64: dict(block_N=64, block_K=128),
}


def tile_config(block_M: int, k: int) -> dict:
    """The token tile's configuration; one group per step when two do not divide K."""
    config = dict(TILE_CONFIGS[block_M])
    if k % (GROUP * config["step_groups"]):
        config["step_groups"] = 1
    return config


def row_tile(rows: int) -> int:
    """The smallest token tile that holds ``rows`` (64 beyond that)."""
    for block_M in ROW_TILES:
        if rows <= block_M:
            return block_M
    return ROW_TILES[-1]


def _decode(low, code, base_minus_one):
    """The BF16 value of a sign-and-mantissa byte and its exponent code."""
    low = T.Cast(T.int32, low)
    code = T.Cast(T.int32, code)
    exponent = T.if_then_else(code == 0, 0, code + base_minus_one)
    bits = ((low & 0x80) << 8) | (exponent << 7) | (low & 0x7F)
    return T.reinterpret(T.Cast(T.uint16, bits), T.bfloat16)


@tilelang.jit
def packed_vocab_projection(
    N: int,
    K: int,
    exponent_base: int,
    block_M: int = 16,
    block_N: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    step_groups: int = 1,
    out_dtype: T.dtype = T.bfloat16,
):
    """``C[M, N] = A @ W^T`` for BF16 ``A[M, K]`` and the packed ``W``, accumulated
    in FP32 and written as ``out_dtype`` (FP32 keeps the reference head's logits).

    Each pipeline step loads ``step_groups`` consecutive groups: longer
    contiguous runs of each row's bytes and codes.
    """
    assert K % (GROUP * step_groups) == 0
    assert (
        shared_memory_bytes(block_M, block_N, num_stages, step_groups) <= SHARED_MEMORY
    )
    steps = K // (GROUP * step_groups)
    STEP = GROUP * step_groups
    row_bytes = K + K // 2
    base_minus_one = exponent_base - 1
    slices = HALF // VEC
    items = block_N * step_groups * slices
    M = T.dynamic("M")
    E = T.dynamic("E")

    @T.prim_func
    def vocab_packed_projection(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, row_bytes), T.uint8),
        exception_rows: T.Tensor((E,), T.int32),
        exception_entries: T.Tensor((E,), T.int32),
        row_offsets: T.Tensor((N + 1,), T.int32),
        C: T.Tensor((M, N), out_dtype),
    ):
        # Token tiles vary fastest: both tiles of a 96-row step read a vocabulary
        # block's packed weights back to back, the second from L2, instead of the
        # whole shard twice from DRAM. Each block computes the same as before.
        with T.Kernel(
            T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads
        ) as (by, bx):
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
            # The copies run num_stages - 1 steps ahead; decoding and the GEMM
            # share the last stage, so the decoded weights need one buffer.
            last = num_stages - 1
            for s in T.Pipelined(
                steps, order=[0, 1, 2, 3, 4], stage=[0, 0, 0, last, last]
            ):
                T.copy(A[by * block_M, s * STEP], A_shared)
                T.copy(W[bx * block_N, s * STEP], low_shared)
                T.copy(W[bx * block_N, K + s * (STEP // 2)], code_shared)
                # Each item is 16 columns of a group's low half and the same
                # 16 of its high half, which share 16 code bytes.
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
                            out_a[v] = _decode(low_a[v], codes[v] & 0xF, base_minus_one)
                            out_b[v] = _decode(low_b[v], codes[v] >> 4, base_minus_one)
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + c + v] = out_a[v]
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + HALF + c + v] = out_b[v]
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            # Out-of-window values, added after the dot product in list order.
            # Only the exception's own row changes; the others are selected
            # unchanged, so no other output gains a +0.0 (which would turn a
            # -0.0 into +0.0).
            first = bx * block_N
            start = row_offsets[first]
            end = row_offsets[T.min(first + block_N, N)]
            for index in T.serial(start, end):
                row = exception_rows[index] - first
                entry = exception_entries[index]
                value = T.Cast(
                    T.float32,
                    T.reinterpret(T.Cast(T.uint16, entry & 0xFFFF), T.bfloat16),
                )
                column = entry >> 16
                for i, j in T.Parallel(block_M, block_N):
                    token = T.min(by * block_M + i, M - 1)
                    C_local[i, j] = T.if_then_else(
                        j == row,
                        C_local[i, j] + value * T.Cast(T.float32, A[token, column]),
                        C_local[i, j],
                    )
            T.copy(C_local, C[by * block_M, bx * block_N])

    return vocab_packed_projection


_OUT = {torch.bfloat16: T.bfloat16, torch.float32: T.float32}


def _packed_head_kernels(layer: nn.Module, out_dtype: torch.dtype) -> dict:
    """A packed head's projection for every token tile, as calls (its exponent base
    comes from the packed weights)."""
    n, k = int(layer.weight.shape[0]), layer.packed_bf16_in_features
    return {
        block_M: prepare.call(
            packed_vocab_projection,
            n,
            k,
            layer.packed_bf16_exponent_base,
            block_M=block_M,
            out_dtype=_OUT[out_dtype],
            **tile_config(block_M, k),
        )
        for block_M in ROW_TILES
    }


def _bf16_head_kernels(layer: nn.Module, out_dtype: torch.dtype) -> dict:
    """A BF16 head's projection for every token tile, as calls."""
    n, k = map(int, layer.weight.shape)
    kernels = {}
    for block_M in ROW_TILES:
        config = dict(BF16_HEAD_CONFIGS[block_M])
        if k % config["block_K"]:
            config["block_K"] = 64
        kernels[block_M] = prepare.call(
            bf16_gemm,
            n,
            k,
            block_M=block_M,
            num_stages=2,
            out_dtype=_OUT[out_dtype],
            **config,
        )
    return kernels


def _declare_head(layer: nn.Module, kernels: dict, out_dtype: torch.dtype) -> None:
    layer.tilelang_vocab_dtype = out_dtype

    def bind() -> None:
        layer.tilelang_vocab = {
            block_M: function(*args, **kwargs)
            for block_M, (function, args, kwargs) in kernels.items()
        }

    prepare.declare(lambda: list(kernels.values()), bind=bind)


def prepare_packed_head(
    layer: nn.Module, out_dtype: torch.dtype = torch.bfloat16
) -> None:
    """Declare a packed head's projection for every token tile, writing ``out_dtype``
    logits (BF16, or FP32 as DeepSeek's reference head computes them). The head must
    be packed: the kernels depend on its exponent base."""
    _declare_head(layer, _packed_head_kernels(layer, out_dtype), out_dtype)


def prepare_bf16_head(
    layer: nn.Module, out_dtype: torch.dtype = torch.bfloat16
) -> None:
    """Declare a BF16 head's projection for every token tile, writing ``out_dtype``."""
    _declare_head(layer, _bf16_head_kernels(layer, out_dtype), out_dtype)


def project_head(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Logits of ``x[rows, K]`` through a prepared head, in its prepared dtype."""
    x = x.contiguous()
    rows = x.shape[0]
    out = torch.empty(
        (rows, layer.weight.shape[0]),
        dtype=getattr(layer, "tilelang_vocab_dtype", torch.bfloat16),
        device=x.device,
    )
    kernel = layer.tilelang_vocab[row_tile(rows)]
    if hasattr(layer, "packed_bf16_exponent_base"):
        kernel(
            x,
            layer.weight,
            layer.weight_exception_rows,
            layer.weight_exception_entries,
            layer.weight_row_offsets,
            out,
        )
    else:
        kernel(x, layer.weight, out)
    return out


__all__ = [
    "BF16_HEAD_CONFIGS",
    "ROW_TILES",
    "TILE_CONFIGS",
    "packed_vocab_projection",
    "prepare_bf16_head",
    "prepare_packed_head",
    "project_head",
    "row_tile",
    "shared_memory_bytes",
    "tile_config",
]
