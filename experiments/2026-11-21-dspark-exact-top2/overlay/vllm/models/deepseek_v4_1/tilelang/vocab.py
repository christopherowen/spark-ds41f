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

from dataclasses import dataclass
import functools
from math import prod

import tilelang
import tilelang.language as T
import torch
from torch import nn

from vllm.utils.math_utils import round_up
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

from . import pdl, prepare
from .gemm import bf16_gemm, bf16_gemm_decode

GROUP = 128  # columns per code group
HALF = GROUP // 2  # one code byte holds a column of each half
WINDOW = 15  # exponents encoded per tensor (codes 1..15; code 0 is exponent 0)
VEC = 16  # columns of each half one thread decodes at a time
_PACK_BLOCK_ROWS = 1_024
# From row 3 * _PACK_BLOCK_ROWS onward, row r's 1.5*K-byte output end is at or
# before row (r - _PACK_BLOCK_ROWS)'s 2*K-byte input start, so packing directly
# into the in-place output view cannot overlap unread input rows.
_PACK_SAFE_DIRECT_ROW = 3 * _PACK_BLOCK_ROWS
# Token tiles: one decode row in a 16-row tile reads the weights once, like 64;
# 65-128 verification rows read the packed weight shard once in a 128-row tile.
ROW_TILES = (16, 32, 64, 128)
NVFP4_GROUP = 16  # columns per FP8 E4M3 scale byte in NVFP4
NVFP4_PACKED = NVFP4_GROUP // 2  # packed E2M1 bytes per NVFP4 group
SHARED_MEMORY = 101_376  # dynamic shared memory per block on SM120 / SM121
# dspark.greedy._BLOCK: greedy_markov_shard_pair keeps one partial per part of
# this many shard columns, so the runner-up of its top-2 pairs is the best
# maximum of the row's other parts. Vocabulary tiles divide it.
GREEDY_PART = 2048


def nvfp4_shared_memory_bytes(
    block_M: int,
    block_N: int,
    num_stages: int,
    step_groups: int = 1,
    act_stages: int | None = None,
) -> int:
    """Shared memory for ``nvfp4_vocab_projection`` across ``step_groups`` groups
    per step."""
    a_stages = num_stages if act_stages is None else act_stages
    act = a_stages * (block_M * GROUP * 2)
    packed = num_stages * (block_N * (GROUP // 2 + GROUP // NVFP4_GROUP))
    decoded = block_N * GROUP * 2
    return step_groups * (act + packed + decoded)


@dataclass(frozen=True)
class PackedBf16Weight:
    """A packed ``[out_features, in_features]`` BF16 weight."""

    values: torch.Tensor
    exponent_base: int
    exception_rows: torch.Tensor
    exception_entries: torch.Tensor
    row_offsets: torch.Tensor
    exceptions: int

    @property
    def out_features(self) -> int:
        return int(self.values.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.values.shape[1]) * 2 // 3


_PACK_HEADER = r"""
template <int GROUP>
__device__ __forceinline__ void vllm_packed_bf16_histogram(
    const unsigned short* __restrict__ weight,
    int* __restrict__ histogram,
    int row, int K, int tx,
    int* __restrict__ s_hist) {
  s_hist[tx] = 0;
  s_hist[tx + GROUP] = 0;
  __syncthreads();
  const unsigned short* row_ptr = weight + (long long)row * K;
  for (int col = tx; col < K; col += GROUP) {
    const int exp = ((int)row_ptr[col] >> 7) & 0xFF;
    atomicAdd(&s_hist[exp], 1);
  }
  __syncthreads();
  const int c0 = s_hist[tx];
  const int c1 = s_hist[tx + GROUP];
  if (c0 != 0) atomicAdd(&histogram[tx], c0);
  if (c1 != 0) atomicAdd(&histogram[tx + GROUP], c1);
}

template <int GROUP, int WINDOW>
__device__ __forceinline__ void vllm_packed_bf16_exception_count(
    const unsigned short* __restrict__ weight,
    int* __restrict__ counts,
    int row, int K, int base, int tx,
    int* __restrict__ s_count) {
  if (tx == 0) s_count[0] = 0;
  __syncthreads();
  const unsigned short* row_ptr = weight + (long long)row * K;
  int local_count = 0;
  for (int col = tx; col < K; col += GROUP) {
    const int exp = ((int)row_ptr[col] >> 7) & 0xFF;
    if (exp != 0 && (exp < base || exp >= base + WINDOW)) {
      local_count += 1;
    }
  }
  if (local_count != 0) atomicAdd(&s_count[0], local_count);
  __syncthreads();
  if (tx == 0) counts[row] = s_count[0];
}

template <int GROUP, int WINDOW>
__device__ __forceinline__ void vllm_packed_bf16_pack_row(
    const unsigned short* __restrict__ weight,
    unsigned char* __restrict__ values,
    const int* __restrict__ offsets,
    int* __restrict__ rows,
    int* __restrict__ entries,
    int* __restrict__ mismatches,
    int prog_row, int row_start, int K, int base, int tx,
    int* __restrict__ s_warp_counts) {
  constexpr int HALF = GROUP / 2;
  const int row_index = row_start + prog_row;
  const unsigned short* source = weight + (long long)row_index * K;
  unsigned char* out = values + (long long)prog_row * (K + K / 2);
  int written = offsets[row_index];
  int bad = 0;
  const int warp_id = tx >> 5;
  const int lane = tx & 31;
  const unsigned int lane_mask = (1u << lane) - 1u;
  for (int group = 0; group < K / GROUP; ++group) {
    const int column = group * GROUP;
    const int col_a = column + tx;
    const int bits_a = (int)source[col_a] & 0xFFFF;
    const int exp_a = (bits_a >> 7) & 0xFF;
    const bool out_a = (exp_a != 0) && (exp_a < base || exp_a >= base + WINDOW);
    const int code_a = (exp_a != 0 && !out_a) ? (exp_a - base + 1) : 0;
    const int low_a = out_a ? 0 : (((bits_a >> 8) & 0x80) | (bits_a & 0x7F));
    const int dec_a = ((low_a & 0x80) << 8) |
                      (((code_a == 0) ? 0 : (code_a + base - 1)) << 7) |
                      (low_a & 0x7F);
    if (!out_a && dec_a != bits_a) bad += 1;
    const unsigned int mask_a = __ballot_sync(0xffffffffu, out_a);
    if (lane == 0) s_warp_counts[warp_id] = __popc(mask_a);
    __syncthreads();
    const int pos_a = written + (warp_id ? s_warp_counts[0] : 0) +
                      __popc(mask_a & lane_mask);
    if (out_a) {
      rows[pos_a] = row_index;
      entries[pos_a] = (col_a << 16) | bits_a;
    }
    written += s_warp_counts[0] + s_warp_counts[1];
    __syncthreads();

    const int col_b = column + HALF + tx;
    const int bits_b = (int)source[col_b] & 0xFFFF;
    const int exp_b = (bits_b >> 7) & 0xFF;
    const bool out_b = (exp_b != 0) && (exp_b < base || exp_b >= base + WINDOW);
    const int code_b = (exp_b != 0 && !out_b) ? (exp_b - base + 1) : 0;
    const int low_b = out_b ? 0 : (((bits_b >> 8) & 0x80) | (bits_b & 0x7F));
    const int dec_b = ((low_b & 0x80) << 8) |
                      (((code_b == 0) ? 0 : (code_b + base - 1)) << 7) |
                      (low_b & 0x7F);
    if (!out_b && dec_b != bits_b) bad += 1;
    const unsigned int mask_b = __ballot_sync(0xffffffffu, out_b);
    if (lane == 0) s_warp_counts[warp_id] = __popc(mask_b);
    __syncthreads();
    const int pos_b = written + (warp_id ? s_warp_counts[0] : 0) +
                      __popc(mask_b & lane_mask);
    if (out_b) {
      rows[pos_b] = row_index;
      entries[pos_b] = (col_b << 16) | bits_b;
    }
    written += s_warp_counts[0] + s_warp_counts[1];
    __syncthreads();

    out[col_a] = (unsigned char)low_a;
    out[col_b] = (unsigned char)low_b;
    out[K + group * HALF + tx] = (unsigned char)(code_a | (code_b << 4));
  }
  if (tx == 0 && written != offsets[row_index + 1]) bad += 1;
  if (bad != 0) atomicAdd(mismatches, bad);
}
"""


@functools.lru_cache(maxsize=None)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def packed_bf16_histogram(K: int):
    """Per-tensor 256-bin exponent histogram over ``weight[M, K]``."""
    assert K > 0 and K % GROUP == 0
    M = T.dynamic("M")

    @T.prim_func
    def vocab_packed_histogram(
        weight: T.Tensor((M, K), T.uint16),
        histogram: T.Tensor((256,), T.int32),
    ):
        with T.Kernel(M, threads=GROUP) as (row,):
            T.import_source(_PACK_HEADER)
            s_hist = T.alloc_shared((256,), T.int32)
            tx = T.get_thread_binding()
            T.call_extern(
                "handle",
                f"vllm_packed_bf16_histogram<{GROUP}>",
                T.address_of(weight[0, 0]),
                T.address_of(histogram[0]),
                row,
                K,
                tx,
                T.address_of(s_hist[0]),
            )

    return vocab_packed_histogram


@functools.lru_cache(maxsize=None)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def packed_bf16_exception_count(K: int):
    """Per-row count of nonzero exponents outside ``base..base + WINDOW - 1``."""
    assert K > 0 and K % GROUP == 0
    M = T.dynamic("M")

    @T.prim_func
    def vocab_packed_exception_count(
        weight: T.Tensor((M, K), T.uint16),
        counts: T.Tensor((M,), T.int32),
        base: T.int32,
    ):
        with T.Kernel(M, threads=GROUP) as (row,):
            T.import_source(_PACK_HEADER)
            s_count = T.alloc_shared((1,), T.int32)
            tx = T.get_thread_binding()
            T.call_extern(
                "handle",
                f"vllm_packed_bf16_exception_count<{GROUP}, {WINDOW}>",
                T.address_of(weight[0, 0]),
                T.address_of(counts[0]),
                row,
                K,
                base,
                tx,
                T.address_of(s_count[0]),
            )

    return vocab_packed_exception_count


@functools.lru_cache(maxsize=None)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def packed_bf16_pack_rows(K: int):
    """Pack rows ``row_start .. row_start + M - 1`` of ``weight[N, K]`` into ``values[M, K + K // 2]``."""
    assert K > 0 and K % GROUP == 0
    row_bytes = K + K // 2
    N = T.dynamic("N")
    M = T.dynamic("M")
    O = T.dynamic("O")
    E = T.dynamic("E")

    @T.prim_func
    def vocab_packed_pack_rows(
        weight: T.Tensor((N, K), T.uint16),
        values: T.Tensor((M, row_bytes), T.uint8),
        offsets: T.Tensor((O,), T.int32),
        rows: T.Tensor((E,), T.int32),
        entries: T.Tensor((E,), T.int32),
        mismatches: T.Tensor((1,), T.int32),
        base: T.int32,
        row_start: T.int32,
    ):
        with T.Kernel(M, threads=HALF) as (prog_row,):
            T.import_source(_PACK_HEADER)
            s_warp_counts = T.alloc_shared((2,), T.int32)
            tx = T.get_thread_binding()
            T.call_extern(
                "handle",
                f"vllm_packed_bf16_pack_row<{GROUP}, {WINDOW}>",
                T.address_of(weight[0, 0]),
                T.address_of(values[0, 0]),
                T.address_of(offsets[0]),
                T.address_of(rows[0]),
                T.address_of(entries[0]),
                T.address_of(mismatches[0]),
                prog_row,
                row_start,
                K,
                base,
                tx,
                T.address_of(s_warp_counts[0]),
            )

    return vocab_packed_pack_rows


def _check_pack_weight(weight: torch.Tensor) -> None:
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise TypeError("packing needs a 2-D torch.bfloat16 weight")
    if not weight.is_cuda or not weight.is_contiguous():
        raise ValueError("packing needs a contiguous CUDA weight")
    if weight.shape[0] == 0 or weight.shape[1] == 0 or weight.shape[1] % GROUP:
        raise ValueError(
            f"packing needs rows and K divisible by {GROUP}, got {tuple(weight.shape)}"
        )


def pack_bf16_weight(
    weight: torch.Tensor,
    *,
    in_place: bool = False,
) -> PackedBf16Weight:
    """Pack ``weight`` into exact 12-bit storage using TileLang; raise if any value
    would not round-trip.

    ``in_place=True`` writes the packed rows into the first three quarters of
    ``weight``'s own storage, staging only the first 3,072 rows where input and
    output byte ranges overlap and writing the remaining rows directly.
    """
    _check_pack_weight(weight)
    out_features, in_features = map(int, weight.shape)
    width = in_features + in_features // 2
    device = weight.device
    bits = weight.view(torch.uint16)
    histogram = torch.zeros(256, dtype=torch.int32, device=device)
    packed_bf16_histogram(in_features)(bits, histogram)
    hist = histogram.to(torch.int64).cpu()
    hist[0] = 0
    windows = hist.unfold(0, WINDOW, 1).sum(1)
    windows[0] = -1
    base = int(windows.argmax())
    row_offsets = torch.zeros(out_features + 1, dtype=torch.int32, device=device)
    packed_bf16_exception_count(in_features)(bits, row_offsets[1:], base)
    row_offsets[1:] = torch.cumsum(row_offsets[1:], 0, dtype=torch.int32)
    exceptions = int(row_offsets[-1])
    rows = torch.zeros(max(exceptions, 1), dtype=torch.int32, device=device)
    entries = torch.zeros_like(rows)
    mismatches = torch.zeros(1, dtype=torch.int32, device=device)
    pack_kernel = packed_bf16_pack_rows(in_features)
    if in_place:
        values = (
            weight.view(torch.uint8)
            .view(-1)[: out_features * width]
            .view(out_features, width)
        )
        staged_limit = min(out_features, _PACK_SAFE_DIRECT_ROW)
        staging = torch.empty(
            (min(_PACK_BLOCK_ROWS, staged_limit), width),
            dtype=torch.uint8,
            device=device,
        )
        for start in range(0, staged_limit, _PACK_BLOCK_ROWS):
            count = min(_PACK_BLOCK_ROWS, staged_limit - start)
            pack_kernel(
                bits,
                staging[:count],
                row_offsets,
                rows,
                entries,
                mismatches,
                base,
                start,
            )
            values[start : start + count].copy_(staging[:count])
        del staging
        start = staged_limit
        while start < out_features:
            # Output bytes [1.5*K*start, 1.5*K*end) are strictly below unread input
            # bytes [2*K*start, 2*K*out_features) iff 3*end <= 4*start.
            end = min(out_features, (4 * start) // 3)
            pack_kernel(
                bits,
                values[start:end],
                row_offsets,
                rows,
                entries,
                mismatches,
                base,
                start,
            )
            start = end
    else:
        values = torch.empty((out_features, width), dtype=torch.uint8, device=device)
        pack_kernel(
            bits,
            values,
            row_offsets,
            rows,
            entries,
            mismatches,
            base,
            0,
        )
    if int(mismatches) != 0:
        raise ValueError(
            f"packed BF16 weight does not round-trip ({int(mismatches)} mismatches)"
        )
    return PackedBf16Weight(values, base, rows, entries, row_offsets, exceptions)


def unpack_bf16_weight(packed: PackedBf16Weight) -> torch.Tensor:
    """The exact BF16 weight that ``packed`` stores (for tests and inspection)."""
    count, width = packed.values.shape
    k = width * 2 // 3
    low = packed.values[:, :k].to(torch.int32)
    codes = packed.values[:, k:].to(torch.int32).view(count, k // GROUP, 1, GROUP // 2)
    code = torch.cat((codes & 0xF, codes >> 4), dim=2).reshape(count, k)
    exponent = torch.where(code == 0, 0, code + packed.exponent_base - 1)
    bits = ((low & 0x80) << 8) | (exponent << 7) | (low & 0x7F)
    if packed.exceptions:
        entries = packed.exception_entries[: packed.exceptions]
        bits[
            packed.exception_rows[: packed.exceptions].long(),
            (entries >> 16).long(),
        ] = entries & 0xFFFF
    return bits.to(torch.int16).view(torch.bfloat16)


pack = pack_bf16_weight
unpack = unpack_bf16_weight


def shared_memory_bytes(
    block_M: int,
    block_N: int,
    num_stages: int,
    step_groups: int = 1,
    act_stages: int | None = None,
) -> int:
    """One buffer per pipeline stage of the activations and the packed bytes
    and codes, and one of the decoded weights, for ``step_groups`` groups per
    step. ``act_stages`` overrides the activation buffer count when smaller
    than ``num_stages``."""
    a_stages = num_stages if act_stages is None else act_stages
    act = a_stages * (block_M * GROUP * 2)
    packed = num_stages * (block_N * (GROUP + HALF))
    decoded = block_N * GROUP * 2
    return step_groups * (act + packed + decoded)


# Vocabulary tile, threads, groups per step and pipeline depth for each token
# tile, measured on GB10 for DS4.1's TP4 and TP3 vocabulary shards (K = 5,120).
TILE_CONFIGS = {
    16: dict(block_N=32, threads=128, step_groups=2, num_stages=4),
    32: dict(block_N=16, threads=128, step_groups=2, num_stages=4),
    64: dict(block_N=32, threads=256, step_groups=1, num_stages=4),
    128: dict(block_N=32, threads=256, step_groups=1, num_stages=2),
}

SCREEN_TILE_CONFIGS = {
    16: dict(block_N=32, threads=128, step_groups=2, num_stages=4),
    32: dict(block_N=32, threads=128, step_groups=2, num_stages=4, act_stages=2),
    64: dict(block_N=32, threads=256, step_groups=1, num_stages=4),
    128: dict(block_N=32, threads=256, step_groups=1, num_stages=2),
}


# BF16 heads: vocabulary tile and K block for each token tile (K = 256).
BF16_HEAD_CONFIGS = {
    16: dict(block_N=64, block_K=256),
    32: dict(block_N=64, block_K=256),
    64: dict(block_N=64, block_K=128),
    128: dict(block_N=64, block_K=128),
}


def tile_config(block_M: int, k: int) -> dict:
    """The token tile's configuration; one group per step when two do not divide K."""
    if k <= 2 * GROUP:
        return dict(block_N=32, threads=128, step_groups=1, num_stages=1)
    config = dict(TILE_CONFIGS[block_M])
    if k % (GROUP * config["step_groups"]):
        config["step_groups"] = 1
    if k // (GROUP * config["step_groups"]) < config["num_stages"]:
        config["num_stages"] = max(1, k // (GROUP * config["step_groups"]))
        if config.get("act_stages") is not None:
            config["act_stages"] = min(config["act_stages"], config["num_stages"])
    return config


def screen_tile_config(block_M: int, k: int) -> dict:
    """The token tile's configuration for 8-bit vocabulary screening."""
    if k <= 2 * GROUP:
        return dict(block_N=32, threads=128, step_groups=1, num_stages=1)
    config = dict(SCREEN_TILE_CONFIGS[block_M])
    if k % (GROUP * config["step_groups"]):
        config["step_groups"] = 1
    if k // (GROUP * config["step_groups"]) < config["num_stages"]:
        config["num_stages"] = max(1, k // (GROUP * config["step_groups"]))
        if config.get("act_stages") is not None:
            config["act_stages"] = min(config["act_stages"], config["num_stages"])
    return config


def row_tile(rows: int) -> int:
    """The smallest token tile that holds ``rows`` (128 beyond that)."""
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


def _decode_e2m1(nibble):
    """The FP32 value of a 4-bit E2M1 nibble (1 sign bit + 3 magnitude bits)."""
    n = T.Cast(T.int32, nibble)
    mag = n & 0x07
    sign = (n >> 3) & 1
    fp32_bits = 0x3F000000 + (mag << 22)
    val = T.reinterpret(T.Cast(T.uint32, fp32_bits), T.float32)
    val = T.if_then_else(mag == 0, T.float32(0.0), val)
    val = T.if_then_else(mag == 1, T.float32(0.5), val)
    return T.if_then_else(sign == 1, -val, val)


_GREEDY_HEADER = r"""
#include <math_constants.h>

template <int BASE_MINUS_ONE>
__device__ __forceinline__ void vllm_vocab_decode_vec16(
    const void* __restrict__ low_a_ptr,
    const void* __restrict__ low_b_ptr,
    const void* __restrict__ codes_ptr,
    void* __restrict__ out_a_ptr,
    void* __restrict__ out_b_ptr) {
  const uint4 low_a4 = *reinterpret_cast<const uint4*>(low_a_ptr);
  const uint4 low_b4 = *reinterpret_cast<const uint4*>(low_b_ptr);
  const uint4 codes4 = *reinterpret_cast<const uint4*>(codes_ptr);
  const unsigned int* la = reinterpret_cast<const unsigned int*>(&low_a4);
  const unsigned int* lb = reinterpret_cast<const unsigned int*>(&low_b4);
  const unsigned int* cd = reinterpret_cast<const unsigned int*>(&codes4);
  unsigned int* oa = reinterpret_cast<unsigned int*>(out_a_ptr);
  unsigned int* ob = reinterpret_cast<unsigned int*>(out_b_ptr);
  constexpr unsigned int BASE_SCALAR = (unsigned int)(BASE_MINUS_ONE & 0xFFFF);

  #pragma unroll
  for (int w = 0; w < 4; ++w) {
    const unsigned int c_word = cd[w];
    const unsigned int la_word = la[w];
    const unsigned int lb_word = lb[w];

    const unsigned int c01 = __byte_perm(c_word, 0u, 0x4140u);
    const unsigned int c23 = __byte_perm(c_word, 0u, 0x4342u);

    const unsigned int ca01 = c01 & 0x000F000Fu;
    const unsigned int cb01 = (c01 >> 4) & 0x000F000Fu;
    const unsigned int ca23 = c23 & 0x000F000Fu;
    const unsigned int cb23 = (c23 >> 4) & 0x000F000Fu;

    const unsigned int nza01 = ((ca01 + 0x000F000Fu) >> 4) & 0x00010001u;
    const unsigned int nzb01 = ((cb01 + 0x000F000Fu) >> 4) & 0x00010001u;
    const unsigned int nza23 = ((ca23 + 0x000F000Fu) >> 4) & 0x00010001u;
    const unsigned int nzb23 = ((cb23 + 0x000F000Fu) >> 4) & 0x00010001u;

    const unsigned int expa01 = (ca01 + nza01 * BASE_SCALAR) << 7;
    const unsigned int expb01 = (cb01 + nzb01 * BASE_SCALAR) << 7;
    const unsigned int expa23 = (ca23 + nza23 * BASE_SCALAR) << 7;
    const unsigned int expb23 = (cb23 + nzb23 * BASE_SCALAR) << 7;

    const unsigned int la01 = __byte_perm(la_word, 0u, 0x1100u);
    const unsigned int la23 = __byte_perm(la_word, 0u, 0x3322u);
    const unsigned int lb01 = __byte_perm(lb_word, 0u, 0x1100u);
    const unsigned int lb23 = __byte_perm(lb_word, 0u, 0x3322u);

    oa[2 * w + 0] = (la01 & 0x807F807Fu) | expa01;
    oa[2 * w + 1] = (la23 & 0x807F807Fu) | expa23;
    ob[2 * w + 0] = (lb01 & 0x807F807Fu) | expb01;
    ob[2 * w + 1] = (lb23 & 0x807F807Fu) | expb23;
  }
}

template <int BM, int BN, int THREADS, int N_ROWS, int K_COLS>
__device__ __forceinline__ void vllm_vocab_apply_row_exceptions(
    float* __restrict__ c_frag,
    const void* __restrict__ a_ptr,
    const int* __restrict__ exception_rows,
    const int* __restrict__ exception_entries,
    const int* __restrict__ row_offsets,
    int by, int bx, int M, int num_exceptions, int tx) {
  (void)exception_rows;
  if (num_exceptions <= 0) return;
  const uint16_t* __restrict__ A = reinterpret_cast<const uint16_t*>(a_ptr);
  constexpr int WARPS = THREADS >> 5;
  constexpr int WARPS_M = (BM == 16) ? 1 : 2;
  constexpr int WARPS_N = WARPS / WARPS_M;
  constexpr int MMA_M = BM / (WARPS_M * 16);
  constexpr int MMA_N = BN / (WARPS_N * 8);
  const int warp_id = tx >> 5;
  const int lane = tx & 31;
  const int warp_m = (BM == 16) ? 0 : (warp_id & 1);
  const int warp_n = (BM == 16) ? warp_id : (warp_id >> 1);
  const int lane_row = lane >> 2;
  const int lane_col = (lane & 3) * 2;
  const int base_col = bx * BN + warp_n * (MMA_N * 8) + lane_col;

  #pragma unroll
  for (int n_tile = 0; n_tile < MMA_N; ++n_tile) {
    const int col0 = base_col + n_tile * 8;
    const int col1 = col0 + 1;
    const int s0 = (col0 < N_ROWS) ? __ldg(&row_offsets[col0]) : 0;
    const int e0 = (col0 < N_ROWS) ? __ldg(&row_offsets[col0 + 1]) : 0;
    const int e1 = (col1 < N_ROWS) ? __ldg(&row_offsets[col1 + 1]) : e0;
    for (int idx = s0; idx < e0; ++idx) {
      const unsigned int entry = (unsigned int)__ldg(&exception_entries[idx]);
      const int k_col = (int)(entry >> 16);
      const unsigned int v_bits = (entry & 0xFFFFu) << 16;
      const float val = __uint_as_float(v_bits);
      #pragma unroll
      for (int m_tile = 0; m_tile < MMA_M; ++m_tile) {
        #pragma unroll
        for (int r_half = 0; r_half < 2; ++r_half) {
          const int r = warp_m * (MMA_M * 16) + m_tile * 16 + r_half * 8 + lane_row;
          const int tok = (by * BM + r < M) ? (by * BM + r) : (M - 1);
          const unsigned int a_bits = ((unsigned int)A[(long long)tok * K_COLS + k_col]) << 16;
          const float a_val = __uint_as_float(a_bits);
          const int frag_idx = ((m_tile * MMA_N + n_tile) * 2 + r_half) * 2;
          c_frag[frag_idx] += val * a_val;
        }
      }
    }
    for (int idx = e0; idx < e1; ++idx) {
      const unsigned int entry = (unsigned int)__ldg(&exception_entries[idx]);
      const int k_col = (int)(entry >> 16);
      const unsigned int v_bits = (entry & 0xFFFFu) << 16;
      const float val = __uint_as_float(v_bits);
      #pragma unroll
      for (int m_tile = 0; m_tile < MMA_M; ++m_tile) {
        #pragma unroll
        for (int r_half = 0; r_half < 2; ++r_half) {
          const int r = warp_m * (MMA_M * 16) + m_tile * 16 + r_half * 8 + lane_row;
          const int tok = (by * BM + r < M) ? (by * BM + r) : (M - 1);
          const unsigned int a_bits = ((unsigned int)A[(long long)tok * K_COLS + k_col]) << 16;
          const float a_val = __uint_as_float(a_bits);
          const int frag_idx = ((m_tile * MMA_N + n_tile) * 2 + r_half) * 2 + 1;
          c_frag[frag_idx] += val * a_val;
        }
      }
    }
  }
}
#define vllm_vocab_apply_row_exceptions_clocal(BM, BN, THREADS, N_ROWS, K_COLS, A_ptr, exc_rows, exc_entries, row_offs, by, bx, M, E, tx) \
  vllm_vocab_apply_row_exceptions<BM, BN, THREADS, N_ROWS, K_COLS>(C_local, A_ptr, exc_rows, exc_entries, row_offs, by, bx, M, E, tx)

__device__ __forceinline__ int vllm_vocab_first_argmax_keep(
    float av, int ai, float bv, int bi) {
  const bool a_nan = av != av;
  const bool b_nan = bv != bv;
  return (a_nan & (!b_nan | (ai < bi))) |
         (!a_nan & !b_nan & ((av > bv) | ((av == bv) & (ai < bi))));
}

template <int BM, int BN, int THREADS>
__device__ __forceinline__ void vllm_vocab_tile_argmax(
    const float* __restrict__ c_frag,
    float* __restrict__ smem_v,
    int* __restrict__ smem_i,
    float* __restrict__ partial_vals,
    int* __restrict__ partial_idxs,
    int by, int bx, int M, int vocab, int num_tiles, int tx) {
  constexpr int WARPS = THREADS >> 5;
  constexpr int WARPS_M = (BM == 16) ? 1 : 2;
  constexpr int WARPS_N = WARPS / WARPS_M;
  constexpr int MMA_M = BM / (WARPS_M * 16);
  constexpr int MMA_N = BN / (WARPS_N * 8);
  const int warp_id = tx >> 5;
  const int lane = tx & 31;
  const int warp_m = (BM == 16) ? 0 : (warp_id & 1);
  const int warp_n = (BM == 16) ? warp_id : (warp_id >> 1);
  const int lane_row = lane >> 2;
  const int lane_col = (lane & 3) * 2;
  const int base_col = bx * BN + warp_n * (MMA_N * 8) + lane_col;

  #pragma unroll
  for (int m_tile = 0; m_tile < MMA_M; ++m_tile) {
    #pragma unroll
    for (int r_half = 0; r_half < 2; ++r_half) {
      const int r = warp_m * (MMA_M * 16) + m_tile * 16 + r_half * 8 + lane_row;
      float best_v = -CUDART_INF_F;
      int best_i = 2147483647;
      #pragma unroll
      for (int n_tile = 0; n_tile < MMA_N; ++n_tile) {
        const int col0 = base_col + n_tile * 8;
        const int idx0 = ((m_tile * MMA_N + n_tile) * 2 + r_half) * 2;
        if (col0 < vocab) {
          const float v0 = c_frag[idx0];
          if (vllm_vocab_first_argmax_keep(v0, col0, best_v, best_i)) {
            best_v = v0;
            best_i = col0;
          }
        }
        const int col1 = col0 + 1;
        if (col1 < vocab) {
          const float v1 = c_frag[idx0 + 1];
          if (vllm_vocab_first_argmax_keep(v1, col1, best_v, best_i)) {
            best_v = v1;
            best_i = col1;
          }
        }
      }
      #pragma unroll
      for (int offset = 2; offset > 0; offset >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
        const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
        if (vllm_vocab_first_argmax_keep(ov, oi, best_v, best_i)) {
          best_v = ov;
          best_i = oi;
        }
      }
      if (WARPS_N == 1) {
        const int row = by * BM + r;
        if ((lane & 3) == 0 && row < M) {
          const int slot = row * num_tiles + bx;
          partial_vals[slot] = best_v;
          partial_idxs[slot] = best_i;
        }
      } else {
        if ((lane & 3) == 0) {
          smem_v[r * WARPS_N + warp_n] = best_v;
          smem_i[r * WARPS_N + warp_n] = best_i;
        }
      }
    }
  }
  if (WARPS_N > 1) {
    __syncthreads();
    for (int r = tx; r < BM; r += THREADS) {
      const int row = by * BM + r;
      if (row < M) {
        float best_v = smem_v[r * WARPS_N];
        int best_i = smem_i[r * WARPS_N];
        #pragma unroll
        for (int wn = 1; wn < WARPS_N; ++wn) {
          const float v = smem_v[r * WARPS_N + wn];
          const int i = smem_i[r * WARPS_N + wn];
          if (vllm_vocab_first_argmax_keep(v, i, best_v, best_i)) {
            best_v = v;
            best_i = i;
          }
        }
        const int slot = row * num_tiles + bx;
        partial_vals[slot] = best_v;
        partial_idxs[slot] = best_i;
      }
    }
  }
}

__device__ __forceinline__ void vllm_vocab_partials_top2_reduce(
    const float* __restrict__ partial_vals,
    const int* __restrict__ partial_idxs,
    float* __restrict__ smem_v,
    int* __restrict__ smem_i,
    float* __restrict__ smem_v2,
    int* __restrict__ smem_i2,
    int* __restrict__ smem_win_bx,
    int row,
    int num_tiles,
    int rescore_BN,
    int rescore_tiles,
    int tx) {
  const int base = row * num_tiles;
  float b1_v = -CUDART_INF_F;
  int b1_i = 2147483647;
  float b2_v = -CUDART_INF_F;
  int b2_i = 2147483647;
  for (int t = tx; t < num_tiles; t += 128) {
    const float v = partial_vals[base + t];
    const int idx = partial_idxs[base + t];
    if (vllm_vocab_first_argmax_keep(v, idx, b1_v, b1_i)) {
      b2_v = b1_v;
      b2_i = b1_i;
      b1_v = v;
      b1_i = idx;
    } else if (vllm_vocab_first_argmax_keep(v, idx, b2_v, b2_i)) {
      b2_v = v;
      b2_i = idx;
    }
  }
  const int lane = tx & 31;
  const int warp = tx >> 5;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float o1_v = __shfl_xor_sync(0xffffffffu, b1_v, off);
    const int o1_i = __shfl_xor_sync(0xffffffffu, b1_i, off);
    const float o2_v = __shfl_xor_sync(0xffffffffu, b2_v, off);
    const int o2_i = __shfl_xor_sync(0xffffffffu, b2_i, off);
    if (vllm_vocab_first_argmax_keep(o1_v, o1_i, b1_v, b1_i)) {
      const float old_v = b1_v;
      const int old_i = b1_i;
      b1_v = o1_v;
      b1_i = o1_i;
      if (vllm_vocab_first_argmax_keep(old_v, old_i, o2_v, o2_i)) {
        b2_v = old_v;
        b2_i = old_i;
      } else {
        b2_v = o2_v;
        b2_i = o2_i;
      }
    } else if (vllm_vocab_first_argmax_keep(o1_v, o1_i, b2_v, b2_i)) {
      b2_v = o1_v;
      b2_i = o1_i;
    }
  }
  if (lane == 0) {
    smem_v[warp] = b1_v;
    smem_i[warp] = b1_i;
    smem_v2[warp] = b2_v;
    smem_i2[warp] = b2_i;
  }
  __syncthreads();
  if (tx == 0) {
    float g1_v = smem_v[0];
    int g1_i = smem_i[0];
    float g2_v = smem_v2[0];
    int g2_i = smem_i2[0];
    #pragma unroll
    for (int w = 1; w < 4; ++w) {
      const float w1_v = smem_v[w];
      const int w1_i = smem_i[w];
      const float w2_v = smem_v2[w];
      const int w2_i = smem_i2[w];
      if (vllm_vocab_first_argmax_keep(w1_v, w1_i, g1_v, g1_i)) {
        const float old_v = g1_v;
        const int old_i = g1_i;
        g1_v = w1_v;
        g1_i = w1_i;
        if (vllm_vocab_first_argmax_keep(old_v, old_i, w2_v, w2_i)) {
          g2_v = old_v;
          g2_i = old_i;
        } else {
          g2_v = w2_v;
          g2_i = w2_i;
        }
      } else if (vllm_vocab_first_argmax_keep(w1_v, w1_i, g2_v, g2_i)) {
        g2_v = w1_v;
        g2_i = w1_i;
      }
    }
    smem_v[0] = g1_v;
    smem_i[0] = g1_i;
    smem_v2[0] = g2_v;
    smem_i2[0] = g2_i;
    int win_bx = (g1_i >= 0 && g1_i < 2147483647) ? (g1_i / rescore_BN) : 0;
    if (win_bx >= rescore_tiles) win_bx = rescore_tiles - 1;
    smem_win_bx[0] = win_bx;
  }
  __syncthreads();
}

template <int BN>
__device__ __forceinline__ void vllm_vocab_tile_second_from_clocal(
    const float* __restrict__ c_frag,
    float* __restrict__ smem_v,
    int* __restrict__ smem_i,
    float* __restrict__ smem_v2,
    int* __restrict__ smem_i2,
    float* __restrict__ pairs,
    int row,
    int win_bx,
    int vocab,
    int vocab_start,
    long long pair_stride,
    int tx) {
  constexpr int WARPS_N = 4;
  constexpr int MMA_N = BN / (WARPS_N * 8);
  const int warp_id = tx >> 5;
  const int lane = tx & 31;
  const int lane_row = lane >> 2;
  const int lane_col = (lane & 3) * 2;
  const int best_i = smem_i[0];
  float sec_v = -CUDART_INF_F;
  int sec_i = 2147483647;
  if (lane_row == 0) {
    const int base_col = win_bx * BN + warp_id * (MMA_N * 8) + lane_col;
    #pragma unroll
    for (int n_tile = 0; n_tile < MMA_N; ++n_tile) {
      const int col0 = base_col + n_tile * 8;
      const int idx0 = n_tile * 4;
      if (col0 < vocab && col0 != best_i) {
        const float v0 = c_frag[idx0];
        if (vllm_vocab_first_argmax_keep(v0, col0, sec_v, sec_i)) {
          sec_v = v0;
          sec_i = col0;
        }
      }
      const int col1 = col0 + 1;
      if (col1 < vocab && col1 != best_i) {
        const float v1 = c_frag[idx0 + 1];
        if (vllm_vocab_first_argmax_keep(v1, col1, sec_v, sec_i)) {
          sec_v = v1;
          sec_i = col1;
        }
      }
    }
  }
  #pragma unroll
  for (int off = 2; off > 0; off >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, sec_v, off);
    const int oi = __shfl_xor_sync(0xffffffffu, sec_i, off);
    if (lane_row == 0 && vllm_vocab_first_argmax_keep(ov, oi, sec_v, sec_i)) {
      sec_v = ov;
      sec_i = oi;
    }
  }
  __syncthreads();
  if (lane == 0) {
    smem_v[4 + warp_id] = sec_v;
    smem_i[4 + warp_id] = sec_i;
  }
  __syncthreads();
  if (tx == 0) {
    const float g1_v = smem_v[0];
    const int g1_i = smem_i[0];
    float g2_v = smem_v2[0];
    int g2_i = smem_i2[0];
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
      const float wv = smem_v[4 + w];
      const int wi = smem_i[4 + w];
      if (wi < 2147483647 && vllm_vocab_first_argmax_keep(wv, wi, g2_v, g2_i)) {
        g2_v = wv;
        g2_i = wi;
      }
    }
    const long long out_off = (long long)row * pair_stride;
    pairs[out_off] = g1_v;
    pairs[out_off + 1] = (g1_i < 2147483647) ? (float)(vocab_start + g1_i) : (float)(1 << 30);
    pairs[out_off + 2] = (g2_i < 2147483647) ? g2_v : -CUDART_INF_F;
    pairs[out_off + 3] = (g2_i < 2147483647) ? (float)(vocab_start + g2_i) : (float)(1 << 30);
  }
}

#define vllm_vocab_tile_second_clocal(BN, smem_v, smem_i, smem_v2, smem_i2, pairs, row, win_bx, vocab, vocab_start, pair_stride, tx) \
  vllm_vocab_tile_second_from_clocal<BN>(C_local, smem_v, smem_i, smem_v2, smem_i2, pairs, row, win_bx, vocab, vocab_start, pair_stride, tx)

#define vllm_vocab_tile_argmax_clocal(BM, BN, THREADS, smem_v, smem_i, pvals, pidxs, by, bx, M, vocab, num_tiles, tx) \
  vllm_vocab_tile_argmax<BM, BN, THREADS>(C_local, smem_v, smem_i, pvals, pidxs, by, bx, M, vocab, num_tiles, tx)

template <int THREADS, int PAIR_WIDTH>
__device__ __forceinline__ void vllm_vocab_finish_argmax(
    const float* __restrict__ partial_vals,
    const int* __restrict__ partial_idxs,
    float* __restrict__ pairs,
    int row, int num_tiles, int vocab_start, int tx,
    float* __restrict__ smem_v, int* __restrict__ smem_i) {
  float best_v = -CUDART_INF_F;
  int best_i = 2147483647;
  const int base = row * num_tiles;
  for (int t = tx; t < num_tiles; t += THREADS) {
    const float v = partial_vals[base + t];
    const int idx = partial_idxs[base + t];
    if (vllm_vocab_first_argmax_keep(v, idx, best_v, best_i)) {
      best_v = v;
      best_i = idx;
    }
  }
  const int lane = tx & 31;
  const int warp_id = tx >> 5;
  constexpr int WARPS = THREADS >> 5;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
    const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
    if (vllm_vocab_first_argmax_keep(ov, oi, best_v, best_i)) {
      best_v = ov;
      best_i = oi;
    }
  }
  if (lane == 0) {
    smem_v[warp_id] = best_v;
    smem_i[warp_id] = best_i;
  }
  __syncthreads();
  if (warp_id == 0) {
    best_v = (lane < WARPS) ? smem_v[lane] : -CUDART_INF_F;
    best_i = (lane < WARPS) ? smem_i[lane] : 2147483647;
    #pragma unroll
    for (int offset = (WARPS >> 1); offset > 0; offset >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
      const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
      if (vllm_vocab_first_argmax_keep(ov, oi, best_v, best_i)) {
        best_v = ov;
        best_i = oi;
      }
    }
    if (lane == 0) {
      pairs[row * PAIR_WIDTH + 0] = best_v;
      pairs[row * PAIR_WIDTH + 1] = (float)(best_i + vocab_start);
    }
  }
}

// The block's first maximum of ``count`` (value, shard index) partials outside
// the PART-column part ``skip_part`` (-1: none), in every thread.
template <int THREADS, int PART>
__device__ __forceinline__ void vllm_vocab_block_first_argmax(
    const float* __restrict__ vals,
    const int* __restrict__ idxs,
    int count, int skip_part, int tx,
    float* __restrict__ smem_v, int* __restrict__ smem_i,
    float& out_v, int& out_i) {
  float best_v = -CUDART_INF_F;
  int best_i = 2147483647;
  for (int t = tx; t < count; t += THREADS) {
    const float v = vals[t];
    const int idx = idxs[t];
    if (idx / PART != skip_part &&
        vllm_vocab_first_argmax_keep(v, idx, best_v, best_i)) {
      best_v = v;
      best_i = idx;
    }
  }
  const int lane = tx & 31;
  const int warp_id = tx >> 5;
  constexpr int WARPS = THREADS >> 5;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
    const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
    if (vllm_vocab_first_argmax_keep(ov, oi, best_v, best_i)) {
      best_v = ov;
      best_i = oi;
    }
  }
  if (lane == 0) {
    smem_v[warp_id] = best_v;
    smem_i[warp_id] = best_i;
  }
  __syncthreads();
  best_v = smem_v[0];
  best_i = smem_i[0];
  #pragma unroll
  for (int w = 1; w < WARPS; ++w) {
    if (vllm_vocab_first_argmax_keep(smem_v[w], smem_i[w], best_v, best_i)) {
      best_v = smem_v[w];
      best_i = smem_i[w];
    }
  }
  __syncthreads();
  out_v = best_v;
  out_i = best_i;
}

// vllm_vocab_finish_argmax and the row's runner-up as
// dspark.greedy.greedy_markov_shard_pair(top2=True) reports it: the first
// maximum of the tiles outside the pair's PART-column part, into columns 2 and
// 3 (index 2**30 when the shard has no other part).
template <int THREADS, int PAIR_WIDTH, int PART>
__device__ __forceinline__ void vllm_vocab_finish_argmax_top2(
    const float* __restrict__ partial_vals,
    const int* __restrict__ partial_idxs,
    float* __restrict__ pairs,
    int row, int num_tiles, int vocab_start, int tx,
    float* __restrict__ smem_v, int* __restrict__ smem_i) {
  const int base = row * num_tiles;
  float best_v, second_v;
  int best_i, second_i;
  vllm_vocab_block_first_argmax<THREADS, PART>(
      partial_vals + base, partial_idxs + base, num_tiles, -1, tx, smem_v, smem_i,
      best_v, best_i);
  vllm_vocab_block_first_argmax<THREADS, PART>(
      partial_vals + base, partial_idxs + base, num_tiles, best_i / PART, tx,
      smem_v, smem_i, second_v, second_i);
  if (tx == 0) {
    pairs[row * PAIR_WIDTH + 0] = best_v;
    pairs[row * PAIR_WIDTH + 1] = (float)(best_i + vocab_start);
    pairs[row * PAIR_WIDTH + 2] = second_v;
    pairs[row * PAIR_WIDTH + 3] = second_i < 2147483647
                                      ? (float)(second_i + vocab_start)
                                      : (float)(1 << 30);
  }
}

#include <cuda_bf16.h>

template <int THREADS, int PAIR_WIDTH, bool TOP2_PAIRS = false>
__device__ __forceinline__ void vllm_vocab_finish_active_argmax(
    const float* __restrict__ partial_vals,
    const int* __restrict__ partial_idxs,
    const float* __restrict__ refined_vals,
    const int* __restrict__ active_cols,
    const int* __restrict__ active_count,
    int block_N, int N, int top_k,
    float* __restrict__ pairs,
    float* __restrict__ row_topk,
    int row, int num_tiles, int vocab_start, int tx,
    float* __restrict__ smem_v, int* __restrict__ smem_i,
    int* __restrict__ smem_chosen) {
  const int lane = tx & 31;
  const int warp_id = tx >> 5;
  constexpr int WARPS = THREADS >> 5;
  if (top_k <= 1) {
    int cnt = active_count[0];
    if (cnt <= 0) cnt = 1;
    const int active_tiles = (cnt + block_N - 1) / block_N;
    float best_v = -CUDART_INF_F;
    int best_i = 2147483647;
    const int base = row * num_tiles;
    for (int t = tx; t < active_tiles; t += THREADS) {
      const float v = partial_vals[base + t];
      const int idx = partial_idxs[base + t];
      if (vllm_vocab_first_argmax_keep(v, idx, best_v, best_i)) {
        best_v = v;
        best_i = idx;
      }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
      const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
      if (vllm_vocab_first_argmax_keep(ov, oi, best_v, best_i)) {
        best_v = ov;
        best_i = oi;
      }
    }
    if (lane == 0) {
      smem_v[warp_id] = best_v;
      smem_i[warp_id] = best_i;
    }
    __syncthreads();
    if (warp_id == 0) {
      best_v = (lane < WARPS) ? smem_v[lane] : -CUDART_INF_F;
      best_i = (lane < WARPS) ? smem_i[lane] : 2147483647;
      #pragma unroll
      for (int offset = (WARPS >> 1); offset > 0; offset >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
        const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
        if (vllm_vocab_first_argmax_keep(ov, oi, best_v, best_i)) {
          best_v = ov;
          best_i = oi;
        }
      }
      if (lane == 0) {
        pairs[row * PAIR_WIDTH + 0] = best_v;
        pairs[row * PAIR_WIDTH + 1] = (float)((long long)best_i + (long long)vocab_start);
      }
    }
    return;
  }

  const int cnt = active_count[0];
  const int k_sel = (top_k > 16) ? 16 : top_k;
  const size_t val_base = (size_t)row * N;
  const size_t topk_base = (size_t)row * (2 * top_k);
  for (int r = 0; r < k_sel; ++r) {
    float best_v = -CUDART_INF_F;
    int best_i = 2147483647;
    for (int s = tx; s < cnt; s += THREADS) {
      const int col = active_cols[s];
      bool already = false;
      for (int q = 0; q < r; ++q) {
        if (col == smem_chosen[q]) {
          already = true;
          break;
        }
      }
      if (!already) {
        const float v = refined_vals[val_base + s];
        if (vllm_vocab_first_argmax_keep(v, col, best_v, best_i)) {
          best_v = v;
          best_i = col;
        }
      }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
      const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
      if (vllm_vocab_first_argmax_keep(ov, oi, best_v, best_i)) {
        best_v = ov;
        best_i = oi;
      }
    }
    if (lane == 0) {
      smem_v[warp_id] = best_v;
      smem_i[warp_id] = best_i;
    }
    __syncthreads();
    if (warp_id == 0) {
      best_v = (lane < WARPS) ? smem_v[lane] : -CUDART_INF_F;
      best_i = (lane < WARPS) ? smem_i[lane] : 2147483647;
      #pragma unroll
      for (int offset = (WARPS >> 1); offset > 0; offset >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
        const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
        if (vllm_vocab_first_argmax_keep(ov, oi, best_v, best_i)) {
          best_v = ov;
          best_i = oi;
        }
      }
      if (lane == 0) {
        smem_chosen[r] = best_i;
        if (r == 0) {
          pairs[row * PAIR_WIDTH + 0] = best_v;
          pairs[row * PAIR_WIDTH + 1] = (float)((long long)best_i + (long long)vocab_start);
        } else if (TOP2_PAIRS && r == 1) {
          pairs[row * PAIR_WIDTH + 2] = (best_v != best_v) ? -CUDART_INF_F : best_v;
          pairs[row * PAIR_WIDTH + 3] = (best_i < 2147483647)
                                            ? (float)((long long)best_i + (long long)vocab_start)
                                            : (float)(1 << 30);
        }
        if constexpr (!TOP2_PAIRS) {
          row_topk[topk_base + r] = (best_v != best_v) ? -CUDART_INF_F : best_v;
          row_topk[topk_base + top_k + r] = (float)((long long)best_i + (long long)vocab_start);
        }
      }
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void vllm_vocab_decode_q8_vec16(
    const void* __restrict__ low_a_ptr,
    const void* __restrict__ low_b_ptr,
    void* __restrict__ out_a_ptr,
    void* __restrict__ out_b_ptr) {
  const uint4 low_a4 = *reinterpret_cast<const uint4*>(low_a_ptr);
  const uint4 low_b4 = *reinterpret_cast<const uint4*>(low_b_ptr);
  const unsigned int* la = reinterpret_cast<const unsigned int*>(&low_a4);
  const unsigned int* lb = reinterpret_cast<const unsigned int*>(&low_b4);
  __nv_bfloat162* oa = reinterpret_cast<__nv_bfloat162*>(out_a_ptr);
  __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(out_b_ptr);

  #pragma unroll
  for (int w = 0; w < 4; ++w) {
    const unsigned int la_word = la[w];
    const unsigned int lb_word = lb[w];

    const unsigned int la01 = __byte_perm(la_word, 0u, 0x4140u);
    const unsigned int la23 = __byte_perm(la_word, 0u, 0x4342u);
    const unsigned int lb01 = __byte_perm(lb_word, 0u, 0x4140u);
    const unsigned int lb23 = __byte_perm(lb_word, 0u, 0x4342u);

    const unsigned int sa01 = ((la01 & 0x00800080u) << 8) | 0x3F803F80u;
    const unsigned int sa23 = ((la23 & 0x00800080u) << 8) | 0x3F803F80u;
    const unsigned int sb01 = ((lb01 & 0x00800080u) << 8) | 0x3F803F80u;
    const unsigned int sb23 = ((lb23 & 0x00800080u) << 8) | 0x3F803F80u;

    const unsigned int va01 = sa01 | (la01 & 0x007F007Fu);
    const unsigned int va23 = sa23 | (la23 & 0x007F007Fu);
    const unsigned int vb01 = sb01 | (lb01 & 0x007F007Fu);
    const unsigned int vb23 = sb23 | (lb23 & 0x007F007Fu);

    oa[2 * w + 0] = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&va01),
                            *reinterpret_cast<const __nv_bfloat162*>(&sa01));
    oa[2 * w + 1] = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&va23),
                            *reinterpret_cast<const __nv_bfloat162*>(&sa23));
    ob[2 * w + 0] = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&vb01),
                            *reinterpret_cast<const __nv_bfloat162*>(&sb01));
    ob[2 * w + 1] = __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&vb23),
                            *reinterpret_cast<const __nv_bfloat162*>(&sb23));
  }
}

template <int THREADS, int K>
__device__ __forceinline__ void vllm_vocab_compute_x_norm(
    const void* __restrict__ A_ptr,
    float* __restrict__ x_norm,
    int* __restrict__ active_count,
    int* __restrict__ default_cut_cols,
    int row, int tx, float* __restrict__ smem) {
  if (row == 0 && tx == 0) {
    active_count[0] = 0;
  }
  if (tx == 0) {
    default_cut_cols[row] = -1;
  }
  float sum_sq = 0.0f;
  const __nv_bfloat162* A2 = reinterpret_cast<const __nv_bfloat162*>(
      reinterpret_cast<const __nv_bfloat16*>(A_ptr) + (size_t)row * K);
  for (int k2 = tx; k2 < K / 2; k2 += THREADS) {
    const float2 f2 = __bfloat1622float2(A2[k2]);
    sum_sq = fmaf(f2.x, f2.x, sum_sq);
    sum_sq = fmaf(f2.y, f2.y, sum_sq);
  }
  const int lane = tx & 31;
  const int warp_id = tx >> 5;
  constexpr int WARPS = THREADS >> 5;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, offset);
  }
  if (lane == 0) smem[warp_id] = sum_sq;
  __syncthreads();
  if (warp_id == 0) {
    sum_sq = (lane < WARPS) ? smem[lane] : 0.0f;
    #pragma unroll
    for (int offset = (WARPS >> 1); offset > 0; offset >>= 1) {
      sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, offset);
    }
    if (lane == 0) {
      x_norm[row] = sqrtf(sum_sq) * 1.0001f + 1e-12f;
    }
  }
}

template <int BM, int BN, int THREADS, int OUT_BF16>
__device__ __forceinline__ void vllm_vocab_screen_tile_epilogue(
    const float* __restrict__ c_frag,
    const float* __restrict__ row_scale,
    const float* __restrict__ row_bound,
    const float* __restrict__ x_norm,
    const int* __restrict__ cut_cols,
    float* __restrict__ u_buf,
    float* __restrict__ smem_v,
    int* __restrict__ smem_i,
    float* __restrict__ smem_u,
    float* __restrict__ partial_vals,
    int* __restrict__ partial_idxs,
    float* __restrict__ tile_max_u,
    int by, int bx, int M, int N, int vocab, int num_tiles, int tx) {
  constexpr int WARPS = THREADS >> 5;
  constexpr int WARPS_M = (BM == 16) ? 1 : 2;
  constexpr int WARPS_N = WARPS / WARPS_M;
  constexpr int MMA_M = BM / (WARPS_M * 16);
  constexpr int MMA_N = BN / (WARPS_N * 8);
  const int warp_id = tx >> 5;
  const int lane = tx & 31;
  const int warp_m = (BM == 16) ? 0 : (warp_id & 1);
  const int warp_n = (BM == 16) ? warp_id : (warp_id >> 1);
  const int lane_row = lane >> 2;
  const int lane_col = (lane & 3) * 2;
  const int base_col = bx * BN + warp_n * (MMA_N * 8) + lane_col;

  #pragma unroll
  for (int m_tile = 0; m_tile < MMA_M; ++m_tile) {
    #pragma unroll
    for (int r_half = 0; r_half < 2; ++r_half) {
      const int r = warp_m * (MMA_M * 16) + m_tile * 16 + r_half * 8 + lane_row;
      const int row = by * BM + r;
      const float xn = (row < M) ? x_norm[row] : 0.0f;
      const int cut_col = (row < M) ? cut_cols[row] : -1;
      float best_v = -CUDART_INF_F;
      int best_i = 2147483647;
      float max_u = -CUDART_INF_F;
      #pragma unroll
      for (int n_tile = 0; n_tile < MMA_N; ++n_tile) {
        const int col0 = base_col + n_tile * 8;
        const int idx0 = ((m_tile * MMA_N + n_tile) * 2 + r_half) * 2;
        if (row < M && col0 < vocab) {
          float u0 = -CUDART_INF_F;
          if (col0 != cut_col) {
            const float v0 = c_frag[idx0] * row_scale[col0];
            const float rb0 = row_bound[col0];
            u0 = v0 + xn * rb0;
            if (OUT_BF16 && rb0 < CUDART_INF_F) {
              u0 = __bfloat162float(__float2bfloat16_rn(u0));
            }
            if (rb0 < CUDART_INF_F && vllm_vocab_first_argmax_keep(v0, col0, best_v, best_i)) {
              best_v = v0;
              best_i = col0;
            }
          }
          u_buf[(size_t)row * N + col0] = u0;
          if (!(u0 < max_u)) max_u = u0;
        }
        const int col1 = col0 + 1;
        if (row < M && col1 < vocab) {
          float u1 = -CUDART_INF_F;
          if (col1 != cut_col) {
            const float v1 = c_frag[idx0 + 1] * row_scale[col1];
            const float rb1 = row_bound[col1];
            u1 = v1 + xn * rb1;
            if (OUT_BF16 && rb1 < CUDART_INF_F) {
              u1 = __bfloat162float(__float2bfloat16_rn(u1));
            }
            if (rb1 < CUDART_INF_F && vllm_vocab_first_argmax_keep(v1, col1, best_v, best_i)) {
              best_v = v1;
              best_i = col1;
            }
          }
          u_buf[(size_t)row * N + col1] = u1;
          if (!(u1 < max_u)) max_u = u1;
        }
      }
      #pragma unroll
      for (int offset = 2; offset > 0; offset >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
        const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
        const float ou = __shfl_xor_sync(0xffffffffu, max_u, offset);
        if (vllm_vocab_first_argmax_keep(ov, oi, best_v, best_i)) {
          best_v = ov;
          best_i = oi;
        }
        if (!(ou < max_u)) max_u = ou;
      }
      if (WARPS_N == 1) {
        if ((lane & 3) == 0 && row < M) {
          const int slot = row * num_tiles + bx;
          partial_vals[slot] = best_v;
          partial_idxs[slot] = best_i;
          tile_max_u[slot] = max_u;
        }
      } else {
        if ((lane & 3) == 0) {
          smem_v[r * WARPS_N + warp_n] = best_v;
          smem_i[r * WARPS_N + warp_n] = best_i;
          smem_u[r * WARPS_N + warp_n] = max_u;
        }
      }
    }
  }
  if (WARPS_N > 1) {
    __syncthreads();
    for (int r = tx; r < BM; r += THREADS) {
      const int row = by * BM + r;
      if (row < M) {
        float best_v = smem_v[r * WARPS_N];
        int best_i = smem_i[r * WARPS_N];
        float max_u = smem_u[r * WARPS_N];
        #pragma unroll
        for (int wn = 1; wn < WARPS_N; ++wn) {
          const float v = smem_v[r * WARPS_N + wn];
          const int i = smem_i[r * WARPS_N + wn];
          const float u = smem_u[r * WARPS_N + wn];
          if (vllm_vocab_first_argmax_keep(v, i, best_v, best_i)) {
            best_v = v;
            best_i = i;
          }
          if (!(u < max_u)) max_u = u;
        }
        const int slot = row * num_tiles + bx;
        partial_vals[slot] = best_v;
        partial_idxs[slot] = best_i;
        tile_max_u[slot] = max_u;
      }
    }
  }
}
#define vllm_vocab_screen_epilogue_clocal(BM, BN, THREADS, OUT_BF16, rscale, rbound, xnorm, cutcols, ubuf, smem_v, smem_i, smem_u, pvals, pidxs, tmaxu, by, bx, M, N, vocab, num_tiles, tx) \
  vllm_vocab_screen_tile_epilogue<BM, BN, THREADS, OUT_BF16>(C_local, rscale, rbound, xnorm, cutcols, ubuf, smem_v, smem_i, smem_u, pvals, pidxs, tmaxu, by, bx, M, N, vocab, num_tiles, tx)

template <int BM, int BN, int THREADS, int OUT_BF16>
__device__ __forceinline__ void vllm_vocab_markov_screen_tile_epilogue(
    const float* __restrict__ c_frag,
    const float* __restrict__ u_base,
    const float* __restrict__ row_bound,
    const float* __restrict__ x_norm,
    float* __restrict__ bias_buf,
    float* __restrict__ u_buf,
    float* __restrict__ smem_v,
    int* __restrict__ smem_i,
    float* __restrict__ smem_u,
    float* __restrict__ partial_vals,
    int* __restrict__ partial_idxs,
    float* __restrict__ tile_max_u,
    int* __restrict__ active_count,
    int* __restrict__ cut_cols,
    int by, int bx, int M, int N, int vocab, int num_tiles, int tx) {
  if (bx == 0 && by == 0 && tx == 0) {
    active_count[0] = 0;
  }
  if (bx == 0 && (by * BM + tx) < M) {
    cut_cols[by * BM + tx] = -1;
  }
  constexpr int WARPS = THREADS >> 5;
  constexpr int WARPS_M = (BM == 16) ? 1 : 2;
  constexpr int WARPS_N = WARPS / WARPS_M;
  constexpr int MMA_M = BM / (WARPS_M * 16);
  constexpr int MMA_N = BN / (WARPS_N * 8);
  const int warp_id = tx >> 5;
  const int lane = tx & 31;
  const int warp_m = (BM == 16) ? 0 : (warp_id & 1);
  const int warp_n = (BM == 16) ? warp_id : (warp_id >> 1);
  const int lane_row = lane >> 2;
  const int lane_col = (lane & 3) * 2;
  const int base_col = bx * BN + warp_n * (MMA_N * 8) + lane_col;

  #pragma unroll
  for (int m_tile = 0; m_tile < MMA_M; ++m_tile) {
    #pragma unroll
    for (int r_half = 0; r_half < 2; ++r_half) {
      const int r = warp_m * (MMA_M * 16) + m_tile * 16 + r_half * 8 + lane_row;
      const int row = by * BM + r;
      const float xn = (row < M) ? x_norm[row] : 0.0f;
      float best_v = -CUDART_INF_F;
      int best_i = 2147483647;
      float max_u = -CUDART_INF_F;
      #pragma unroll
      for (int n_tile = 0; n_tile < MMA_N; ++n_tile) {
        const int col0 = base_col + n_tile * 8;
        const int idx0 = ((m_tile * MMA_N + n_tile) * 2 + r_half) * 2;
        if (row < M && col0 < vocab) {
          const float bias0 = c_frag[idx0];
          const float ub0 = u_base[(size_t)row * N + col0];
          const float rb0 = row_bound[col0];
          bias_buf[(size_t)row * N + col0] = bias0;
          float u0;
          if (ub0 != ub0 || bias0 != bias0 || !(rb0 < CUDART_INF_F) || !(ub0 < CUDART_INF_F)) {
            u0 = CUDART_INF_F;
          } else {
            u0 = __fadd_ru(ub0, bias0);
            if (OUT_BF16) {
              u0 += (fabsf(ub0) + fabsf(u0)) * (1.0f / 256.0f) + 1e-5f;
            }
            const float v0 = (ub0 - xn * rb0) + bias0;
            if (vllm_vocab_first_argmax_keep(v0, col0, best_v, best_i)) {
              best_v = v0;
              best_i = col0;
            }
          }
          u_buf[(size_t)row * N + col0] = u0;
          if (!(u0 < max_u)) max_u = u0;
        }
        const int col1 = col0 + 1;
        if (row < M && col1 < vocab) {
          const float bias1 = c_frag[idx0 + 1];
          const float ub1 = u_base[(size_t)row * N + col1];
          const float rb1 = row_bound[col1];
          bias_buf[(size_t)row * N + col1] = bias1;
          float u1;
          if (ub1 != ub1 || bias1 != bias1 || !(rb1 < CUDART_INF_F) || !(ub1 < CUDART_INF_F)) {
            u1 = CUDART_INF_F;
          } else {
            u1 = __fadd_ru(ub1, bias1);
            if (OUT_BF16) {
              u1 += (fabsf(ub1) + fabsf(u1)) * (1.0f / 256.0f) + 1e-5f;
            }
            const float v1 = (ub1 - xn * rb1) + bias1;
            if (vllm_vocab_first_argmax_keep(v1, col1, best_v, best_i)) {
              best_v = v1;
              best_i = col1;
            }
          }
          u_buf[(size_t)row * N + col1] = u1;
          if (!(u1 < max_u)) max_u = u1;
        }
      }
      #pragma unroll
      for (int offset = 2; offset > 0; offset >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
        const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
        const float ou = __shfl_xor_sync(0xffffffffu, max_u, offset);
        if (vllm_vocab_first_argmax_keep(ov, oi, best_v, best_i)) {
          best_v = ov;
          best_i = oi;
        }
        if (!(ou < max_u)) max_u = ou;
      }
      if (WARPS_N == 1) {
        if ((lane & 3) == 0 && row < M) {
          const int slot = row * num_tiles + bx;
          partial_vals[slot] = best_v;
          partial_idxs[slot] = best_i;
          tile_max_u[slot] = max_u;
        }
      } else {
        if ((lane & 3) == 0) {
          smem_v[r * WARPS_N + warp_n] = best_v;
          smem_i[r * WARPS_N + warp_n] = best_i;
          smem_u[r * WARPS_N + warp_n] = max_u;
        }
      }
    }
  }
  if (WARPS_N > 1) {
    __syncthreads();
    for (int r = tx; r < BM; r += THREADS) {
      const int row = by * BM + r;
      if (row < M) {
        float best_v = smem_v[r * WARPS_N];
        int best_i = smem_i[r * WARPS_N];
        float max_u = smem_u[r * WARPS_N];
        #pragma unroll
        for (int wn = 1; wn < WARPS_N; ++wn) {
          const float v = smem_v[r * WARPS_N + wn];
          const int i = smem_i[r * WARPS_N + wn];
          const float u = smem_u[r * WARPS_N + wn];
          if (vllm_vocab_first_argmax_keep(v, i, best_v, best_i)) {
            best_v = v;
            best_i = i;
          }
          if (!(u < max_u)) max_u = u;
        }
        const int slot = row * num_tiles + bx;
        partial_vals[slot] = best_v;
        partial_idxs[slot] = best_i;
        tile_max_u[slot] = max_u;
      }
    }
  }
}
#define vllm_vocab_markov_screen_epilogue_clocal(BM, BN, THREADS, OUT_BF16, ubase, rbound, xnorm, bbuf, ubuf, smem_v, smem_i, smem_u, pvals, pidxs, tmaxu, actcnt, ccols, by, bx, M, N, vocab, num_tiles, tx) \
  vllm_vocab_markov_screen_tile_epilogue<BM, BN, THREADS, OUT_BF16>(C_local, ubase, rbound, xnorm, bbuf, ubuf, smem_v, smem_i, smem_u, pvals, pidxs, tmaxu, actcnt, ccols, by, bx, M, N, vocab, num_tiles, tx)

template <int THREADS, int K, int BASE_MINUS_ONE, int OUT_BF16, int BN, bool HAS_BIAS = false>
__device__ __forceinline__ void vllm_vocab_row_thresholds(
    const void* __restrict__ A_ptr,
    const unsigned char* __restrict__ W,
    const int* __restrict__ exception_entries,
    const int* __restrict__ row_offsets,
    const float* __restrict__ x_norm,
    const float* __restrict__ row_mma_tol,
    const float* __restrict__ partial_vals,
    const int* __restrict__ partial_idxs,
    float* __restrict__ tile_max_u,
    float* __restrict__ u_buf,
    float* __restrict__ row_thresh,
    float* __restrict__ smem_v,
    int* __restrict__ smem_i,
    int* __restrict__ smem_chosen,
    int m, int N, int vocab, int num_tiles, int E, int top_k, int tx,
    const float* __restrict__ bias_buf = nullptr) {
  constexpr int WARPS = THREADS >> 5;
  const int warp_id = tx >> 5;
  const int lane = tx & 31;
  if (top_k > 16) {
    if (tx == 0) row_thresh[m] = -CUDART_INF_F;
    return;
  }
  const int n_sel = (top_k < 1) ? 1 : top_k;
  float min_t_val = -CUDART_INF_F;
  const int base = m * num_tiles;
  constexpr int ROW_BYTES = K + K / 2;
  constexpr int GROUPS = K / 128;
  constexpr int TOTAL_SLICES = GROUPS * 4;
  const __nv_bfloat16* a_row = reinterpret_cast<const __nv_bfloat16*>(A_ptr) + (size_t)m * K;

  for (int k_idx = 0; k_idx < n_sel; ++k_idx) {
    float best_v = -CUDART_INF_F;
    int best_i = 2147483647;
    for (int t = tx; t < num_tiles; t += THREADS) {
      const int idx = partial_idxs[base + t];
      bool already = false;
      for (int q = 0; q < k_idx; ++q) {
        if (idx == smem_chosen[q]) {
          already = true;
          break;
        }
      }
      if (!already) {
        const float v = partial_vals[base + t];
        if (vllm_vocab_first_argmax_keep(v, idx, best_v, best_i)) {
          best_v = v;
          best_i = idx;
        }
      }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
      const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
      if (vllm_vocab_first_argmax_keep(ov, oi, best_v, best_i)) {
        best_v = ov;
        best_i = oi;
      }
    }
    if (lane == 0) {
      smem_v[warp_id] = best_v;
      smem_i[warp_id] = best_i;
    }
    __syncthreads();
    if (warp_id == 0) {
      best_v = (lane < WARPS) ? smem_v[lane] : -CUDART_INF_F;
      best_i = (lane < WARPS) ? smem_i[lane] : 2147483647;
      #pragma unroll
      for (int offset = (WARPS >> 1); offset > 0; offset >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
        const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
        if (vllm_vocab_first_argmax_keep(ov, oi, best_v, best_i)) {
          best_v = ov;
          best_i = oi;
        }
      }
      if (lane == 0) {
        smem_chosen[k_idx] = best_i;
      }
    }
    __syncthreads();
    const int v_hat = smem_chosen[k_idx];
    if (v_hat < 0 || v_hat >= vocab) {
      if (tx == 0) min_t_val = -CUDART_INF_F;
      break;
    }

    const unsigned char* w_row = W + (size_t)v_hat * ROW_BYTES;
    float dot = 0.0f;
    for (int s_idx = tx; s_idx < TOTAL_SLICES; s_idx += THREADS) {
      const int g = s_idx >> 2;
      const int c = (s_idx & 3) * 16;
      alignas(16) __nv_bfloat16 out_a[16];
      alignas(16) __nv_bfloat16 out_b[16];
      vllm_vocab_decode_vec16<BASE_MINUS_ONE>(
          w_row + g * 128 + c,
          w_row + g * 128 + 64 + c,
          w_row + K + g * 64 + c,
          out_a,
          out_b);
      const __nv_bfloat162* a_va = reinterpret_cast<const __nv_bfloat162*>(a_row + g * 128 + c);
      const __nv_bfloat162* a_vb = reinterpret_cast<const __nv_bfloat162*>(a_row + g * 128 + 64 + c);
      const __nv_bfloat162* w_va = reinterpret_cast<const __nv_bfloat162*>(out_a);
      const __nv_bfloat162* w_vb = reinterpret_cast<const __nv_bfloat162*>(out_b);
      #pragma unroll
      for (int v = 0; v < 8; ++v) {
        const float2 af_a = __bfloat1622float2(a_va[v]);
        const float2 wf_a = __bfloat1622float2(w_va[v]);
        dot = fmaf(af_a.x, wf_a.x, dot);
        dot = fmaf(af_a.y, wf_a.y, dot);
        const float2 af_b = __bfloat1622float2(a_vb[v]);
        const float2 wf_b = __bfloat1622float2(w_vb[v]);
        dot = fmaf(af_b.x, wf_b.x, dot);
        dot = fmaf(af_b.y, wf_b.y, dot);
      }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      dot += __shfl_xor_sync(0xffffffffu, dot, offset);
    }
    if (lane == 0) {
      smem_v[warp_id] = dot;
    }
    __syncthreads();
    if (warp_id == 0) {
      dot = (lane < WARPS) ? smem_v[lane] : 0.0f;
      #pragma unroll
      for (int offset = (WARPS >> 1); offset > 0; offset >>= 1) {
        dot += __shfl_xor_sync(0xffffffffu, dot, offset);
      }
      if (lane == 0) {
        if (E > 0) {
          const int start = row_offsets[v_hat];
          const int end = row_offsets[v_hat + 1];
          for (int index = start; index < end; ++index) {
            const int entry = exception_entries[index];
            const unsigned short bits = (unsigned short)(entry & 0xFFFF);
            const float val = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
            const int exc_col = entry >> 16;
            dot += val * __bfloat162float(a_row[exc_col]);
          }
        }
        float tol = x_norm[m] * row_mma_tol[v_hat];
        float t_val = dot - tol;
        if constexpr (HAS_BIAS) {
          const float bias_val = bias_buf[(size_t)m * N + v_hat];
          if (OUT_BF16) {
            t_val = __bfloat162float(__float2bfloat16_rn(t_val)) - fabsf(t_val) * (1.0f / 128.0f) - 1e-5f;
          }
          t_val += bias_val;
          if (OUT_BF16) {
            t_val = t_val - fabsf(t_val) * (1.0f / 128.0f) - 1e-5f;
          } else {
            t_val = t_val - fabsf(t_val) * 2e-7f - 1e-6f;
          }
        } else if (OUT_BF16) {
          t_val = t_val - fabsf(t_val) * (1.0f / 128.0f) - 1e-5f;
        }
        if (t_val != t_val) {
          t_val = -CUDART_INF_F;
        }
        min_t_val = (k_idx == 0) ? t_val : fminf(min_t_val, t_val);
        u_buf[(size_t)m * N + v_hat] = CUDART_INF_F;
        tile_max_u[m * num_tiles + (v_hat / BN)] = CUDART_INF_F;
      }
    }
    __syncthreads();
  }
  if (tx == 0) {
    row_thresh[m] = min_t_val;
  }
}

template <int GRID_BLOCKS, int THREADS, int BN>
__device__ __forceinline__ void vllm_vocab_compact_tiles(
    const float* __restrict__ row_thresh,
    const float* __restrict__ tile_max_u,
    const float* __restrict__ u_buf,
    int* __restrict__ active_cols,
    int* __restrict__ active_count,
    float* __restrict__ T_smem,
    int M, int N, int vocab, int num_tiles, int block_idx, int tx) {
  for (int m = tx; m < M; m += THREADS) {
    T_smem[m] = row_thresh[m];
  }
  __syncthreads();

  constexpr int WARPS_PER_BLOCK = THREADS >> 5;
  constexpr int TOTAL_WARPS = GRID_BLOCKS * WARPS_PER_BLOCK;
  const int global_warp = block_idx * WARPS_PER_BLOCK + (tx >> 5);
  const int lane = tx & 31;

  for (int bx = global_warp; bx < num_tiles; bx += TOTAL_WARPS) {
    bool tile_alive = false;
    for (int m = lane; m < M; m += 32) {
      const float tu = tile_max_u[m * num_tiles + bx];
      const float tm = T_smem[m];
      if (!(tu < tm)) {
        tile_alive = true;
      }
    }
    if (__any_sync(0xffffffffu, tile_alive)) {
      const int col = bx * BN + lane;
      bool col_alive = false;
      if (lane < BN && col < vocab) {
        for (int m = 0; m < M; ++m) {
          const float u = u_buf[(size_t)m * N + col];
          const float tm = T_smem[m];
          if (!(u < tm)) {
            col_alive = true;
            break;
          }
        }
      }
      const unsigned int mask = __ballot_sync(0xffffffffu, col_alive);
      if (mask != 0u) {
        int base_idx = 0;
        if (lane == 0) {
          base_idx = atomicAdd(&active_count[0], __popc(mask));
        }
        base_idx = __shfl_sync(0xffffffffu, base_idx, 0);
        if (col_alive) {
          const int offset = __popc(mask & ((1u << lane) - 1u));
          active_cols[base_idx + offset] = col;
        }
      }
    }
  }
}

template <int BM, int BN, int THREADS, int K, int OUT_BF16, bool HAS_BIAS = false>
__device__ __forceinline__ void vllm_vocab_refine_tile_argmax(
    const float* __restrict__ c_frag,
    const void* __restrict__ A_ptr,
    const int* __restrict__ exception_entries,
    const int* __restrict__ row_offsets,
    const int* __restrict__ cols_shared,
    const int* __restrict__ cut_cols,
    float* __restrict__ smem_v,
    int* __restrict__ smem_i,
    float* __restrict__ refined_vals,
    float* __restrict__ partial_vals,
    int* __restrict__ partial_idxs,
    int by, int bx, int M, int N, int num_active, int num_tiles, int E, int tx,
    const float* __restrict__ bias_buf = nullptr) {
  constexpr int WARPS = THREADS >> 5;
  constexpr int WARPS_M = (BM == 16) ? 1 : 2;
  constexpr int WARPS_N = WARPS / WARPS_M;
  constexpr int MMA_M = BM / (WARPS_M * 16);
  constexpr int MMA_N = BN / (WARPS_N * 8);
  const int warp_id = tx >> 5;
  const int lane = tx & 31;
  const int warp_m = (BM == 16) ? 0 : (warp_id & 1);
  const int warp_n = (BM == 16) ? warp_id : (warp_id >> 1);
  const int lane_row = lane >> 2;
  const int lane_col = (lane & 3) * 2;
  const int local_base_col = warp_n * (MMA_N * 8) + lane_col;
  const __nv_bfloat16* A_bf16 = reinterpret_cast<const __nv_bfloat16*>(A_ptr);

  #pragma unroll
  for (int m_tile = 0; m_tile < MMA_M; ++m_tile) {
    #pragma unroll
    for (int r_half = 0; r_half < 2; ++r_half) {
      const int r = warp_m * (MMA_M * 16) + m_tile * 16 + r_half * 8 + lane_row;
      const int row = by * BM + r;
      const int cut_col = (row < M) ? cut_cols[row] : -1;
      const int token = (row < M) ? row : (M - 1);
      const __nv_bfloat16* a_row = A_bf16 + (size_t)token * K;
      float best_v = -CUDART_INF_F;
      int best_i = 2147483647;
      #pragma unroll
      for (int n_tile = 0; n_tile < MMA_N; ++n_tile) {
        const int j0 = local_base_col + n_tile * 8;
        const int slot0 = bx * BN + j0;
        const int idx0 = ((m_tile * MMA_N + n_tile) * 2 + r_half) * 2;
        if (row < M && slot0 < num_active) {
          const int col0 = cols_shared[j0];
          float v0;
          if (col0 == cut_col) {
            v0 = -CUDART_INF_F;
          } else {
            v0 = c_frag[idx0];
            if (E > 0) {
              const int start = row_offsets[col0];
              const int end = row_offsets[col0 + 1];
              for (int index = start; index < end; ++index) {
                const int entry = exception_entries[index];
                const unsigned short bits = (unsigned short)(entry & 0xFFFF);
                const float val = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
                const int exc_col = entry >> 16;
                v0 += val * __bfloat162float(a_row[exc_col]);
              }
            }
            if (OUT_BF16) {
              v0 = __bfloat162float(__float2bfloat16_rn(v0));
            }
            if constexpr (HAS_BIAS) {
              v0 += bias_buf[(size_t)row * N + col0];
              if (OUT_BF16) {
                v0 = __bfloat162float(__float2bfloat16_rn(v0));
              }
            }
          }
          refined_vals[(size_t)row * N + slot0] = v0;
          if (vllm_vocab_first_argmax_keep(v0, col0, best_v, best_i)) {
            best_v = v0;
            best_i = col0;
          }
        }
        const int j1 = j0 + 1;
        const int slot1 = slot0 + 1;
        if (row < M && slot1 < num_active) {
          const int col1 = cols_shared[j1];
          float v1;
          if (col1 == cut_col) {
            v1 = -CUDART_INF_F;
          } else {
            v1 = c_frag[idx0 + 1];
            if (E > 0) {
              const int start = row_offsets[col1];
              const int end = row_offsets[col1 + 1];
              for (int index = start; index < end; ++index) {
                const int entry = exception_entries[index];
                const unsigned short bits = (unsigned short)(entry & 0xFFFF);
                const float val = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
                const int exc_col = entry >> 16;
                v1 += val * __bfloat162float(a_row[exc_col]);
              }
            }
            if (OUT_BF16) {
              v1 = __bfloat162float(__float2bfloat16_rn(v1));
            }
            if constexpr (HAS_BIAS) {
              v1 += bias_buf[(size_t)row * N + col1];
              if (OUT_BF16) {
                v1 = __bfloat162float(__float2bfloat16_rn(v1));
              }
            }
          }
          refined_vals[(size_t)row * N + slot1] = v1;
          if (vllm_vocab_first_argmax_keep(v1, col1, best_v, best_i)) {
            best_v = v1;
            best_i = col1;
          }
        }
      }
      #pragma unroll
      for (int offset = 2; offset > 0; offset >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
        const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
        if (vllm_vocab_first_argmax_keep(ov, oi, best_v, best_i)) {
          best_v = ov;
          best_i = oi;
        }
      }
      if (WARPS_N == 1) {
        if ((lane & 3) == 0 && row < M) {
          const int slot = row * num_tiles + bx;
          partial_vals[slot] = best_v;
          partial_idxs[slot] = best_i;
        }
      } else {
        if ((lane & 3) == 0) {
          smem_v[r * WARPS_N + warp_n] = best_v;
          smem_i[r * WARPS_N + warp_n] = best_i;
        }
      }
    }
  }
  if (WARPS_N > 1) {
    __syncthreads();
    for (int r = tx; r < BM; r += THREADS) {
      const int row = by * BM + r;
      if (row < M) {
        float best_v = smem_v[r * WARPS_N];
        int best_i = smem_i[r * WARPS_N];
        #pragma unroll
        for (int wn = 1; wn < WARPS_N; ++wn) {
          const float v = smem_v[r * WARPS_N + wn];
          const int i = smem_i[r * WARPS_N + wn];
          if (vllm_vocab_first_argmax_keep(v, i, best_v, best_i)) {
            best_v = v;
            best_i = i;
          }
        }
        const int slot = row * num_tiles + bx;
        partial_vals[slot] = best_v;
        partial_idxs[slot] = best_i;
      }
    }
  }
}
#define vllm_vocab_refine_tile_argmax_clocal(BM, BN, THREADS, K, OUT_BF16, A_ptr, exc_entries, row_offs, cols_sh, cutcols, smem_v, smem_i, rvals, pvals, pidxs, by, bx, M, N, num_active, num_tiles, E, tx) \
  vllm_vocab_refine_tile_argmax<BM, BN, THREADS, K, OUT_BF16>(C_local, A_ptr, exc_entries, row_offs, cols_sh, cutcols, smem_v, smem_i, rvals, pvals, pidxs, by, bx, M, N, num_active, num_tiles, E, tx)
#define vllm_vocab_markov_refine_tile_argmax_clocal(BM, BN, THREADS, K, OUT_BF16, A_ptr, exc_entries, row_offs, cols_sh, cutcols, smem_v, smem_i, rvals, pvals, pidxs, by, bx, M, N, num_active, num_tiles, E, tx, bbuf) \
  vllm_vocab_refine_tile_argmax<BM, BN, THREADS, K, OUT_BF16, true>(C_local, A_ptr, exc_entries, row_offs, cols_sh, cutcols, smem_v, smem_i, rvals, pvals, pidxs, by, bx, M, N, num_active, num_tiles, E, tx, bbuf)
"""


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
    act_stages: int | None = None,
    out_dtype: T.dtype = T.bfloat16,
):
    """``C[M, N] = A @ W^T`` for BF16 ``A[M, K]`` and the packed ``W``, accumulated
    in FP32 and written as ``out_dtype`` (FP32 keeps the reference head's logits).

    Each pipeline step loads ``step_groups`` consecutive groups: longer
    contiguous runs of each row's bytes and codes.
    """
    assert K % (GROUP * step_groups) == 0
    assert (
        shared_memory_bytes(
            block_M, block_N, num_stages, step_groups, act_stages=act_stages
        )
        <= SHARED_MEMORY
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
            T.import_source(_GREEDY_HEADER)
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
            act_stage = 0 if act_stages is None else last - act_stages + 1
            for s in (
                T.serial(steps)
                if num_stages <= 1
                else T.Pipelined(
                    steps, order=[0, 1, 2, 3, 4], stage=[act_stage, 0, 0, last, last]
                )
            ):
                T.copy(A[by * block_M, s * STEP], A_shared)
                T.copy(W[bx * block_N, s * STEP], low_shared)
                T.copy(W[bx * block_N, K + s * (STEP // 2)], code_shared)
                # Each item is 16 columns of a group's low half and the same
                # 16 of its high half, which share 16 code bytes.
                for step in T.serial(T.ceildiv(items, threads)):
                    item = step * threads + tx
                    if items % threads == 0 or item < items:
                        j = item // (step_groups * slices)
                        g = item // slices % step_groups
                        c = item % slices * VEC
                        for v in T.vectorized(VEC):
                            low_a[v] = low_shared[j, g * GROUP + c + v]
                        for v in T.vectorized(VEC):
                            low_b[v] = low_shared[j, g * GROUP + HALF + c + v]
                        for v in T.vectorized(VEC):
                            codes[v] = code_shared[j, g * HALF + c + v]
                        T.call_extern(
                            "handle",
                            f"vllm_vocab_decode_vec16<{base_minus_one}>",
                            T.address_of(low_a[0]),
                            T.address_of(low_b[0]),
                            T.address_of(codes[0]),
                            T.address_of(out_a[0]),
                            T.address_of(out_b[0]),
                        )
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + c + v] = out_a[v]
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + HALF + c + v] = out_b[v]
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            # Out-of-window values, added after the dot product in list order.
            # Only the exception's own row changes; the others are selected
            # unchanged, so no other output gains a +0.0 (which would turn a
            # -0.0 into +0.0).
            T.call_extern(
                "handle",
                "vllm_vocab_apply_row_exceptions_clocal",
                block_M,
                block_N,
                threads,
                N,
                K,
                T.address_of(A[0, 0]),
                T.address_of(exception_rows[0]),
                T.address_of(exception_entries[0]),
                T.address_of(row_offsets[0]),
                by,
                bx,
                M,
                E,
                tx,
            )
            T.copy(C_local, C[by * block_M, bx * block_N])

    return vocab_packed_projection


@tilelang.jit
def packed_vocab_greedy_partials(
    N: int,
    K: int,
    exponent_base: int,
    block_M: int = 16,
    block_N: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    step_groups: int = 1,
    act_stages: int | None = None,
    out_dtype: T.dtype = T.bfloat16,
):
    """Per-tile ``(max, shard index)`` of ``A @ W^T`` for BF16 ``A[M, K]`` and the
    packed ``W``, reduced in each CTA's epilogue without writing the ``[M, N]``
    logits tensor to DRAM."""
    assert K % (GROUP * step_groups) == 0
    assert (
        shared_memory_bytes(
            block_M, block_N, num_stages, step_groups, act_stages=act_stages
        )
        <= SHARED_MEMORY
    )
    steps = K // (GROUP * step_groups)
    STEP = GROUP * step_groups
    row_bytes = K + K // 2
    base_minus_one = exponent_base - 1
    slices = HALF // VEC
    items = block_N * step_groups * slices
    num_tiles = (N + block_N - 1) // block_N
    warps_n = (threads // 32) // (1 if block_M == 16 else 2)
    M = T.dynamic("M")
    E = T.dynamic("E")

    @T.prim_func
    def vocab_packed_greedy_partials(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, row_bytes), T.uint8),
        exception_rows: T.Tensor((E,), T.int32),
        exception_entries: T.Tensor((E,), T.int32),
        row_offsets: T.Tensor((N + 1,), T.int32),
        partial_vals: T.Tensor((M, num_tiles), T.float32),
        partial_idxs: T.Tensor((M, num_tiles), T.int32),
        vocab: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads
        ) as (by, bx):
            T.import_source(_GREEDY_HEADER)
            A_shared = T.alloc_shared((block_M, STEP), T.bfloat16)
            low_shared = T.alloc_shared((block_N, STEP), T.uint8)
            code_shared = T.alloc_shared((block_N, STEP // 2), T.uint8)
            B_shared = T.alloc_shared((block_N, STEP), T.bfloat16)
            smem_v = T.alloc_shared((block_M * warps_n,), T.float32)
            smem_i = T.alloc_shared((block_M * warps_n,), T.int32)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_round = T.alloc_fragment((block_M, block_N), out_dtype)
            low_a = T.alloc_local((VEC,), T.uint8)
            low_b = T.alloc_local((VEC,), T.uint8)
            codes = T.alloc_local((VEC,), T.uint8)
            out_a = T.alloc_local((VEC,), T.bfloat16)
            out_b = T.alloc_local((VEC,), T.bfloat16)
            tx = T.get_thread_binding()
            T.clear(C_local)
            last = num_stages - 1
            act_stage = 0 if act_stages is None else last - act_stages + 1
            for s in (
                T.serial(steps)
                if num_stages <= 1
                else T.Pipelined(
                    steps, order=[0, 1, 2, 3, 4], stage=[act_stage, 0, 0, last, last]
                )
            ):
                T.copy(A[by * block_M, s * STEP], A_shared)
                T.copy(W[bx * block_N, s * STEP], low_shared)
                T.copy(W[bx * block_N, K + s * (STEP // 2)], code_shared)
                for step in T.serial(T.ceildiv(items, threads)):
                    item = step * threads + tx
                    if items % threads == 0 or item < items:
                        j = item // (step_groups * slices)
                        g = item // slices % step_groups
                        c = item % slices * VEC
                        for v in T.vectorized(VEC):
                            low_a[v] = low_shared[j, g * GROUP + c + v]
                        for v in T.vectorized(VEC):
                            low_b[v] = low_shared[j, g * GROUP + HALF + c + v]
                        for v in T.vectorized(VEC):
                            codes[v] = code_shared[j, g * HALF + c + v]
                        T.call_extern(
                            "handle",
                            f"vllm_vocab_decode_vec16<{base_minus_one}>",
                            T.address_of(low_a[0]),
                            T.address_of(low_b[0]),
                            T.address_of(codes[0]),
                            T.address_of(out_a[0]),
                            T.address_of(out_b[0]),
                        )
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + c + v] = out_a[v]
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + HALF + c + v] = out_b[v]
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.call_extern(
                "handle",
                "vllm_vocab_apply_row_exceptions_clocal",
                block_M,
                block_N,
                threads,
                N,
                K,
                T.address_of(A[0, 0]),
                T.address_of(exception_rows[0]),
                T.address_of(exception_entries[0]),
                T.address_of(row_offsets[0]),
                by,
                bx,
                M,
                E,
                tx,
            )
            if out_dtype != T.float32:
                T.copy(C_local, C_round)
                T.copy(C_round, C_local)
            T.call_extern(
                "handle",
                "vllm_vocab_tile_argmax_clocal",
                        block_M,
                        block_N,
                        threads,
                T.address_of(smem_v[0]),
                T.address_of(smem_i[0]),
                T.address_of(partial_vals[0, 0]),
                T.address_of(partial_idxs[0, 0]),
                by,
                bx,
                M,
                vocab,
                num_tiles,
                tx,
            )

    return vocab_packed_greedy_partials


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def packed_vocab_markov_greedy_partials(
    N: int,
    K: int,
    exponent_base: int,
    block_M: int = 16,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 128,
    step_groups: int = 1,
    act_stages: int | None = None,
    out_dtype: T.dtype = T.bfloat16,
):
    """Per-tile ``(max, shard index)`` of ``Base + A @ W^T`` for BF16 Markov
    embeddings ``A[M, K]``, 12-bit packed weights ``W``, and strided base logits
    ``Base[M, N]``, reduced in each CTA's epilogue without materializing the
    ``[M, N]`` Markov bias tensor."""
    assert K % (GROUP * step_groups) == 0
    assert (
        shared_memory_bytes(
            block_M, block_N, num_stages, step_groups, act_stages=act_stages
        )
        <= SHARED_MEMORY
    )
    steps = K // (GROUP * step_groups)
    STEP = GROUP * step_groups
    row_bytes = K + K // 2
    base_minus_one = exponent_base - 1
    slices = HALF // VEC
    items = block_N * step_groups * slices
    num_tiles = (N + block_N - 1) // block_N
    warps_n = (threads // 32) // (1 if block_M == 16 else 2)
    M = T.dynamic("M")
    E = T.dynamic("E")
    base_stride = T.dynamic("base_stride", "int64")

    @T.prim_func
    def vocab_packed_markov_greedy_partials(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, row_bytes), T.uint8),
        exception_rows: T.Tensor((E,), T.int32),
        exception_entries: T.Tensor((E,), T.int32),
        row_offsets: T.Tensor((N + 1,), T.int32),
        Base: T.StridedTensor((M, N), (base_stride, 1), out_dtype),
        partial_vals: T.Tensor((M, num_tiles), T.float32),
        partial_idxs: T.Tensor((M, num_tiles), T.int32),
        vocab: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            pdl.chain()
            T.import_source(_GREEDY_HEADER)
            A_shared = T.alloc_shared((block_M, STEP), T.bfloat16)
            low_shared = T.alloc_shared((block_N, STEP), T.uint8)
            code_shared = T.alloc_shared((block_N, STEP // 2), T.uint8)
            B_shared = T.alloc_shared((block_N, STEP), T.bfloat16)
            smem_v = T.alloc_shared((block_M * warps_n,), T.float32)
            smem_i = T.alloc_shared((block_M * warps_n,), T.int32)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_round = T.alloc_fragment((block_M, block_N), out_dtype)
            low_a = T.alloc_local((VEC,), T.uint8)
            low_b = T.alloc_local((VEC,), T.uint8)
            codes = T.alloc_local((VEC,), T.uint8)
            out_a = T.alloc_local((VEC,), T.bfloat16)
            out_b = T.alloc_local((VEC,), T.bfloat16)
            tx = T.get_thread_binding()
            T.clear(C_local)
            last = num_stages - 1
            act_stage = 0 if act_stages is None else last - act_stages + 1
            for s in (
                T.serial(steps)
                if num_stages <= 1
                else T.Pipelined(
                    steps, order=[0, 1, 2, 3, 4], stage=[act_stage, 0, 0, last, last]
                )
            ):
                T.copy(A[by * block_M, s * STEP], A_shared)
                T.copy(W[bx * block_N, s * STEP], low_shared)
                T.copy(W[bx * block_N, K + s * (STEP // 2)], code_shared)
                for step in T.serial(T.ceildiv(items, threads)):
                    item = step * threads + tx
                    if items % threads == 0 or item < items:
                        j = item // (step_groups * slices)
                        g = item // slices % step_groups
                        c = item % slices * VEC
                        for v in T.vectorized(VEC):
                            low_a[v] = low_shared[j, g * GROUP + c + v]
                        for v in T.vectorized(VEC):
                            low_b[v] = low_shared[j, g * GROUP + HALF + c + v]
                        for v in T.vectorized(VEC):
                            codes[v] = code_shared[j, g * HALF + c + v]
                        T.call_extern(
                            "handle",
                            f"vllm_vocab_decode_vec16<{base_minus_one}>",
                            T.address_of(low_a[0]),
                            T.address_of(low_b[0]),
                            T.address_of(codes[0]),
                            T.address_of(out_a[0]),
                            T.address_of(out_b[0]),
                        )
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + c + v] = out_a[v]
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + HALF + c + v] = out_b[v]
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.call_extern(
                "handle",
                "vllm_vocab_apply_row_exceptions_clocal",
                block_M,
                block_N,
                threads,
                N,
                K,
                T.address_of(A[0, 0]),
                T.address_of(exception_rows[0]),
                T.address_of(exception_entries[0]),
                T.address_of(row_offsets[0]),
                by,
                bx,
                M,
                E,
                tx,
            )
            if out_dtype != T.float32:
                T.copy(C_local, C_round)
                T.copy(C_round, C_local)
            for i, j in T.Parallel(block_M, block_N):
                row = T.min(by * block_M + i, M - 1)
                col = T.min(bx * block_N + j, N - 1)
                C_local[i, j] = T.Cast(T.float32, Base[row, col]) + C_local[i, j]
            if out_dtype != T.float32:
                T.copy(C_local, C_round)
                T.copy(C_round, C_local)
            T.call_extern(
                "handle",
                "vllm_vocab_tile_argmax_clocal",
                        block_M,
                        block_N,
                        threads,
                T.address_of(smem_v[0]),
                T.address_of(smem_i[0]),
                T.address_of(partial_vals[0, 0]),
                T.address_of(partial_idxs[0, 0]),
                by,
                bx,
                M,
                vocab,
                num_tiles,
                tx,
            )

    return vocab_packed_markov_greedy_partials


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def bf16_vocab_markov_greedy_partials(
    N: int,
    K: int,
    block_M: int = 16,
    block_N: int = 64,
    block_K: int = 256,
    num_stages: int = 2,
    threads: int = 128,
    out_dtype: T.dtype = T.bfloat16,
):
    """Per-tile ``(max, shard index)`` of ``Base + A @ B^T`` for BF16 Markov
    embeddings ``A[M, K]``, BF16 weights ``B[N, K]``, and strided base logits
    ``Base[M, N]``, reduced in each CTA's epilogue without materializing the
    ``[M, N]`` Markov bias tensor."""
    k_blocks = tilelang.cdiv(K, block_K)
    num_tiles = (N + block_N - 1) // block_N
    warps_n = (threads // 32) // (1 if block_M == 16 else 2)
    M = T.dynamic("M")
    base_stride = T.dynamic("base_stride", "int64")

    @T.prim_func
    def vocab_bf16_markov_greedy_partials(
        A: T.Tensor((M, K), T.bfloat16),
        B: T.Tensor((N, K), T.bfloat16),
        Base: T.StridedTensor((M, N), (base_stride, 1), out_dtype),
        partial_vals: T.Tensor((M, num_tiles), T.float32),
        partial_idxs: T.Tensor((M, num_tiles), T.int32),
        vocab: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            pdl.chain()
            T.import_source(_GREEDY_HEADER)
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            B_shared = T.alloc_shared((block_N, block_K), T.bfloat16)
            smem_v = T.alloc_shared((block_M * warps_n,), T.float32)
            smem_i = T.alloc_shared((block_M * warps_n,), T.int32)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_round = T.alloc_fragment((block_M, block_N), out_dtype)
            tx = T.get_thread_binding()
            T.clear(C_local)
            for ko in (
                T.serial(k_blocks)
                if num_stages <= 1
                else T.Pipelined(k_blocks, num_stages=num_stages)
            ):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            if out_dtype != T.float32:
                T.copy(C_local, C_round)
                T.copy(C_round, C_local)
            for i, j in T.Parallel(block_M, block_N):
                row = T.min(by * block_M + i, M - 1)
                col = T.min(bx * block_N + j, N - 1)
                C_local[i, j] = T.Cast(T.float32, Base[row, col]) + C_local[i, j]
            if out_dtype != T.float32:
                T.copy(C_local, C_round)
                T.copy(C_round, C_local)
            T.call_extern(
                "handle",
                "vllm_vocab_tile_argmax_clocal",
                        block_M,
                        block_N,
                        threads,
                T.address_of(smem_v[0]),
                T.address_of(smem_i[0]),
                T.address_of(partial_vals[0, 0]),
                T.address_of(partial_idxs[0, 0]),
                by,
                bx,
                M,
                vocab,
                num_tiles,
                tx,
            )

    return vocab_bf16_markov_greedy_partials


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def packed_vocab_markov_screen_partials(
    N: int,
    K: int,
    exponent_base: int,
    block_M: int = 16,
    block_N: int = 32,
    num_stages: int = 1,
    threads: int = 128,
    step_groups: int = 1,
    act_stages: int | None = None,
    out_dtype: T.dtype = T.bfloat16,
):
    """Markov bias projection combined with pre-screened ``U_base[M, N]`` upper
    bounds to emit exact ``bias_buf``, certified Markov upper bounds ``u_buf``,
    tile max upper bounds, and candidate argmax partials."""
    assert K % (GROUP * step_groups) == 0
    assert (
        shared_memory_bytes(
            block_M, block_N, num_stages, step_groups, act_stages=act_stages
        )
        <= SHARED_MEMORY
    )
    steps = K // (GROUP * step_groups)
    STEP = GROUP * step_groups
    row_bytes = K + K // 2
    base_minus_one = exponent_base - 1
    out_bf16 = 1 if out_dtype != T.float32 else 0
    slices = HALF // VEC
    items = block_N * step_groups * slices
    num_tiles = (N + block_N - 1) // block_N
    warps_n = (threads // 32) // (1 if block_M == 16 else 2)
    M = T.dynamic("M")
    E = T.dynamic("E")

    @T.prim_func
    def vocab_packed_markov_screen_partials(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, row_bytes), T.uint8),
        exception_rows: T.Tensor((E,), T.int32),
        exception_entries: T.Tensor((E,), T.int32),
        row_offsets: T.Tensor((N + 1,), T.int32),
        U_base: T.Tensor((M, N), T.float32),
        row_bound: T.Tensor((N,), T.float32),
        x_norm: T.Tensor((M,), T.float32),
        bias_buf: T.Tensor((M, N), T.float32),
        u_buf: T.Tensor((M, N), T.float32),
        partial_vals: T.Tensor((M, num_tiles), T.float32),
        partial_idxs: T.Tensor((M, num_tiles), T.int32),
        tile_max_u: T.Tensor((M, num_tiles), T.float32),
        active_count: T.Tensor((1,), T.int32),
        cut_cols: T.Tensor((M,), T.int32),
        vocab: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            pdl.chain()
            T.import_source(_GREEDY_HEADER)
            A_shared = T.alloc_shared((block_M, STEP), T.bfloat16)
            low_shared = T.alloc_shared((block_N, STEP), T.uint8)
            code_shared = T.alloc_shared((block_N, STEP // 2), T.uint8)
            B_shared = T.alloc_shared((block_N, STEP), T.bfloat16)
            smem_v = T.alloc_shared((block_M * warps_n,), T.float32)
            smem_i = T.alloc_shared((block_M * warps_n,), T.int32)
            smem_u = T.alloc_shared((block_M * warps_n,), T.float32)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_round = T.alloc_fragment((block_M, block_N), out_dtype)
            low_a = T.alloc_local((VEC,), T.uint8)
            low_b = T.alloc_local((VEC,), T.uint8)
            codes = T.alloc_local((VEC,), T.uint8)
            out_a = T.alloc_local((VEC,), T.bfloat16)
            out_b = T.alloc_local((VEC,), T.bfloat16)
            tx = T.get_thread_binding()
            T.clear(C_local)
            last = num_stages - 1
            act_stage = 0 if act_stages is None else last - act_stages + 1
            for s in (
                T.serial(steps)
                if num_stages <= 1
                else T.Pipelined(
                    steps, order=[0, 1, 2, 3, 4], stage=[act_stage, 0, 0, last, last]
                )
            ):
                T.copy(A[by * block_M, s * STEP], A_shared)
                T.copy(W[bx * block_N, s * STEP], low_shared)
                T.copy(W[bx * block_N, K + s * (STEP // 2)], code_shared)
                for step in T.serial(T.ceildiv(items, threads)):
                    item = step * threads + tx
                    if items % threads == 0 or item < items:
                        j = item // (step_groups * slices)
                        g = item // slices % step_groups
                        c = item % slices * VEC
                        for v in T.vectorized(VEC):
                            low_a[v] = low_shared[j, g * GROUP + c + v]
                        for v in T.vectorized(VEC):
                            low_b[v] = low_shared[j, g * GROUP + HALF + c + v]
                        for v in T.vectorized(VEC):
                            codes[v] = code_shared[j, g * HALF + c + v]
                        T.call_extern(
                            "handle",
                            f"vllm_vocab_decode_vec16<{base_minus_one}>",
                            T.address_of(low_a[0]),
                            T.address_of(low_b[0]),
                            T.address_of(codes[0]),
                            T.address_of(out_a[0]),
                            T.address_of(out_b[0]),
                        )
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + c + v] = out_a[v]
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + HALF + c + v] = out_b[v]
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.call_extern(
                "handle",
                "vllm_vocab_apply_row_exceptions_clocal",
                block_M,
                block_N,
                threads,
                N,
                K,
                T.address_of(A[0, 0]),
                T.address_of(exception_rows[0]),
                T.address_of(exception_entries[0]),
                T.address_of(row_offsets[0]),
                by,
                bx,
                M,
                E,
                tx,
            )
            if out_dtype != T.float32:
                T.copy(C_local, C_round)
                T.copy(C_round, C_local)
            T.call_extern(
                "handle",
                "vllm_vocab_markov_screen_epilogue_clocal",
                block_M,
                block_N,
                threads,
                out_bf16,
                T.address_of(U_base[0, 0]),
                T.address_of(row_bound[0]),
                T.address_of(x_norm[0]),
                T.address_of(bias_buf[0, 0]),
                T.address_of(u_buf[0, 0]),
                T.address_of(smem_v[0]),
                T.address_of(smem_i[0]),
                T.address_of(smem_u[0]),
                T.address_of(partial_vals[0, 0]),
                T.address_of(partial_idxs[0, 0]),
                T.address_of(tile_max_u[0, 0]),
                T.address_of(active_count[0]),
                T.address_of(cut_cols[0]),
                by,
                bx,
                M,
                N,
                vocab,
                num_tiles,
                tx,
            )

    return vocab_packed_markov_screen_partials


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def bf16_vocab_markov_screen_partials(
    N: int,
    K: int,
    block_M: int = 16,
    block_N: int = 32,
    block_K: int = 256,
    num_stages: int = 1,
    threads: int = 128,
    out_dtype: T.dtype = T.bfloat16,
):
    """BF16 Markov screen counterpart of ``packed_vocab_markov_screen_partials``."""
    k_blocks = tilelang.cdiv(K, block_K)
    num_tiles = (N + block_N - 1) // block_N
    warps_n = (threads // 32) // (1 if block_M == 16 else 2)
    out_bf16 = 1 if out_dtype != T.float32 else 0
    M = T.dynamic("M")

    @T.prim_func
    def vocab_bf16_markov_screen_partials(
        A: T.Tensor((M, K), T.bfloat16),
        B: T.Tensor((N, K), T.bfloat16),
        U_base: T.Tensor((M, N), T.float32),
        row_bound: T.Tensor((N,), T.float32),
        x_norm: T.Tensor((M,), T.float32),
        bias_buf: T.Tensor((M, N), T.float32),
        u_buf: T.Tensor((M, N), T.float32),
        partial_vals: T.Tensor((M, num_tiles), T.float32),
        partial_idxs: T.Tensor((M, num_tiles), T.int32),
        tile_max_u: T.Tensor((M, num_tiles), T.float32),
        active_count: T.Tensor((1,), T.int32),
        cut_cols: T.Tensor((M,), T.int32),
        vocab: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            pdl.chain()
            T.import_source(_GREEDY_HEADER)
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            B_shared = T.alloc_shared((block_N, block_K), T.bfloat16)
            smem_v = T.alloc_shared((block_M * warps_n,), T.float32)
            smem_i = T.alloc_shared((block_M * warps_n,), T.int32)
            smem_u = T.alloc_shared((block_M * warps_n,), T.float32)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_round = T.alloc_fragment((block_M, block_N), out_dtype)
            tx = T.get_thread_binding()
            T.clear(C_local)
            for ko in (
                T.serial(k_blocks)
                if num_stages <= 1
                else T.Pipelined(k_blocks, num_stages=num_stages)
            ):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            if out_dtype != T.float32:
                T.copy(C_local, C_round)
                T.copy(C_round, C_local)
            T.call_extern(
                "handle",
                "vllm_vocab_markov_screen_epilogue_clocal",
                block_M,
                block_N,
                threads,
                out_bf16,
                T.address_of(U_base[0, 0]),
                T.address_of(row_bound[0]),
                T.address_of(x_norm[0]),
                T.address_of(bias_buf[0, 0]),
                T.address_of(u_buf[0, 0]),
                T.address_of(smem_v[0]),
                T.address_of(smem_i[0]),
                T.address_of(smem_u[0]),
                T.address_of(partial_vals[0, 0]),
                T.address_of(partial_idxs[0, 0]),
                T.address_of(tile_max_u[0, 0]),
                T.address_of(active_count[0]),
                T.address_of(cut_cols[0]),
                by,
                bx,
                M,
                N,
                vocab,
                num_tiles,
                tx,
            )

    return vocab_bf16_markov_screen_partials


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def finish_vocab_greedy_pairs(
    num_tiles: int,
    pair_width: int = 4,
    threads: int = 256,
    top2: bool = False,
):
    """Reduce per-tile ``(max, shard index)`` partials into each row's ``(max, global
    index)`` pair in the first two columns of ``pairs[M, pair_width]``; with
    ``top2``, also the runner-up ``greedy_markov_shard_pair(top2=True)`` reports
    (the best maximum of the row's other ``GREEDY_PART``-column parts) in columns
    2 and 3."""
    assert not top2 or pair_width >= 4
    warps = threads // 32
    finish = (
        f"vllm_vocab_finish_argmax_top2<{threads}, {pair_width}, {GREEDY_PART}>"
        if top2
        else f"vllm_vocab_finish_argmax<{threads}, {pair_width}>"
    )
    M = T.dynamic("M")

    @T.prim_func
    def vocab_greedy_finish(
        partial_vals: T.Tensor((M, num_tiles), T.float32),
        partial_idxs: T.Tensor((M, num_tiles), T.int32),
        pairs: T.Tensor((M, pair_width), T.float32),
        vocab_start: T.int32,
    ):
        with T.Kernel(M, threads=threads) as (row,):
            pdl.chain()
            T.import_source(_GREEDY_HEADER)
            smem_v = T.alloc_shared((warps,), T.float32)
            smem_i = T.alloc_shared((warps,), T.int32)
            tx = T.get_thread_binding()
            T.call_extern(
                "handle",
                finish,
                T.address_of(partial_vals[0, 0]),
                T.address_of(partial_idxs[0, 0]),
                T.address_of(pairs[0, 0]),
                row,
                num_tiles,
                vocab_start,
                tx,
                T.address_of(smem_v[0]),
                T.address_of(smem_i[0]),
            )

    return vocab_greedy_finish


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def finish_packed_markov_top2_exact(
    N: int,
    K: int,
    exponent_base: int,
    num_tiles: int,
    block_N: int,
    step_groups: int = 1,
    out_dtype: T.dtype = T.bfloat16,
):
    """Exact full-vocabulary top-2 finish for packed BF16 Markov heads.

    Reduces ``[M, num_tiles]`` partials to find the global winner and best
    other-tile runner-up, then rescores only the single winning ``rescore_BN``
    tile using the exact same ``T.gemm`` and exception accumulation as
    ``packed_vocab_markov_greedy_partials``, guaranteeing bit-exact
    full-vocabulary top-2 in both ``bfloat16`` and ``float32``.
    """
    rescore_BN = max(block_N, 32)
    rescore_tiles = (N + rescore_BN - 1) // rescore_BN
    assert K % (GROUP * step_groups) == 0
    steps = K // (GROUP * step_groups)
    STEP = GROUP * step_groups
    row_bytes = K + K // 2
    base_minus_one = exponent_base - 1
    slices = HALF // VEC
    items = rescore_BN * step_groups * slices
    threads = 128
    M = T.dynamic("M")
    E = T.dynamic("E")
    base_stride = T.dynamic("base_stride", "int64")
    pair_width = T.dynamic("pair_width")
    pair_stride = T.dynamic("pair_stride", "int64")

    @T.prim_func
    def finish_packed_markov_top2_exact_kernel(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, row_bytes), T.uint8),
        exception_rows: T.Tensor((E,), T.int32),
        exception_entries: T.Tensor((E,), T.int32),
        row_offsets: T.Tensor((N + 1,), T.int32),
        Base: T.StridedTensor((M, N), (base_stride, 1), out_dtype),
        partial_vals: T.Tensor((M, num_tiles), T.float32),
        partial_idxs: T.Tensor((M, num_tiles), T.int32),
        pairs: T.StridedTensor((M, pair_width), (pair_stride, 1), T.float32),
        vocab: T.int32,
        vocab_start: T.int32,
    ):
        with T.Kernel(M, threads=threads) as row:
            pdl.chain()
            T.import_source(_GREEDY_HEADER)
            A_shared = T.alloc_shared((16, STEP), T.bfloat16)
            low_shared = T.alloc_shared((rescore_BN, STEP), T.uint8)
            code_shared = T.alloc_shared((rescore_BN, STEP // 2), T.uint8)
            B_shared = T.alloc_shared((rescore_BN, STEP), T.bfloat16)
            smem_v = T.alloc_shared((8,), T.float32)
            smem_i = T.alloc_shared((8,), T.int32)
            smem_v2 = T.alloc_shared((4,), T.float32)
            smem_i2 = T.alloc_shared((4,), T.int32)
            smem_win_bx = T.alloc_shared((1,), T.int32)
            C_local = T.alloc_fragment((16, rescore_BN), T.float32)
            C_round = T.alloc_fragment((16, rescore_BN), out_dtype)
            low_a = T.alloc_local((VEC,), T.uint8)
            low_b = T.alloc_local((VEC,), T.uint8)
            codes = T.alloc_local((VEC,), T.uint8)
            out_a = T.alloc_local((VEC,), T.bfloat16)
            out_b = T.alloc_local((VEC,), T.bfloat16)
            tx = T.get_thread_binding()

            T.call_extern(
                "handle",
                "vllm_vocab_partials_top2_reduce",
                T.address_of(partial_vals[0, 0]),
                T.address_of(partial_idxs[0, 0]),
                T.address_of(smem_v[0]),
                T.address_of(smem_i[0]),
                T.address_of(smem_v2[0]),
                T.address_of(smem_i2[0]),
                T.address_of(smem_win_bx[0]),
                row,
                num_tiles,
                rescore_BN,
                rescore_tiles,
                tx,
            )
            win_bx = smem_win_bx[0]
            T.clear(C_local)
            for s in T.serial(steps):
                for i, k in T.Parallel(16, STEP):
                    A_shared[i, k] = A[row, s * STEP + k]
                for j, k in T.Parallel(rescore_BN, STEP):
                    col = T.min(win_bx * rescore_BN + j, N - 1)
                    low_shared[j, k] = W[col, s * STEP + k]
                for j, k in T.Parallel(rescore_BN, STEP // 2):
                    col = T.min(win_bx * rescore_BN + j, N - 1)
                    code_shared[j, k] = W[col, K + s * (STEP // 2) + k]
                for step in T.serial(T.ceildiv(items, threads)):
                    item = step * threads + tx
                    if items % threads == 0 or item < items:
                        j = item // (step_groups * slices)
                        rem = item % (step_groups * slices)
                        g = rem // slices
                        c = (rem % slices) * VEC
                        for v in T.vectorized(VEC):
                            low_a[v] = low_shared[j, g * GROUP + c + v]
                        for v in T.vectorized(VEC):
                            low_b[v] = low_shared[j, g * GROUP + HALF + c + v]
                        for v in T.vectorized(VEC):
                            codes[v] = code_shared[j, g * HALF + c + v]
                        T.call_extern(
                            "handle",
                            f"vllm_vocab_decode_vec16<{base_minus_one}>",
                            T.address_of(low_a[0]),
                            T.address_of(low_b[0]),
                            T.address_of(codes[0]),
                            T.address_of(out_a[0]),
                            T.address_of(out_b[0]),
                        )
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + c + v] = out_a[v]
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + HALF + c + v] = out_b[v]
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.call_extern(
                "handle",
                "vllm_vocab_apply_row_exceptions_clocal",
                16,
                rescore_BN,
                threads,
                N,
                K,
                T.address_of(A[row, 0]),
                T.address_of(exception_rows[0]),
                T.address_of(exception_entries[0]),
                T.address_of(row_offsets[0]),
                0,
                win_bx,
                1,
                E,
                tx,
            )
            if out_dtype != T.float32:
                T.copy(C_local, C_round)
                T.copy(C_round, C_local)
            for i, j in T.Parallel(16, rescore_BN):
                col = T.min(win_bx * rescore_BN + j, N - 1)
                C_local[i, j] = T.Cast(T.float32, Base[row, col]) + C_local[i, j]
            if out_dtype != T.float32:
                T.copy(C_local, C_round)
                T.copy(C_round, C_local)
            T.call_extern(
                "handle",
                "vllm_vocab_tile_second_clocal",
                rescore_BN,
                T.address_of(smem_v[0]),
                T.address_of(smem_i[0]),
                T.address_of(smem_v2[0]),
                T.address_of(smem_i2[0]),
                T.address_of(pairs[0, 0]),
                row,
                win_bx,
                vocab,
                vocab_start,
                pair_stride,
                tx,
            )

    return finish_packed_markov_top2_exact_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def finish_bf16_markov_top2_exact(
    N: int,
    K: int,
    num_tiles: int,
    block_N: int,
    block_K: int = 256,
    out_dtype: T.dtype = T.bfloat16,
):
    """Exact full-vocabulary top-2 finish for BF16 Markov heads.

    Uses the exact same ``T.gemm`` over the single winning ``rescore_BN`` tile
    as ``bf16_vocab_markov_greedy_partials``, guaranteeing bit-exact
    full-vocabulary top-2 in both ``bfloat16`` and ``float32``.
    """
    rescore_BN = max(block_N, 32)
    rescore_tiles = (N + rescore_BN - 1) // rescore_BN
    k_blocks = tilelang.cdiv(K, block_K)
    threads = 128
    M = T.dynamic("M")
    base_stride = T.dynamic("base_stride", "int64")
    pair_width = T.dynamic("pair_width")
    pair_stride = T.dynamic("pair_stride", "int64")

    @T.prim_func
    def finish_bf16_markov_top2_exact_kernel(
        A: T.Tensor((M, K), T.bfloat16),
        B: T.Tensor((N, K), T.bfloat16),
        Base: T.StridedTensor((M, N), (base_stride, 1), out_dtype),
        partial_vals: T.Tensor((M, num_tiles), T.float32),
        partial_idxs: T.Tensor((M, num_tiles), T.int32),
        pairs: T.StridedTensor((M, pair_width), (pair_stride, 1), T.float32),
        vocab: T.int32,
        vocab_start: T.int32,
    ):
        with T.Kernel(M, threads=threads) as row:
            pdl.chain()
            T.import_source(_GREEDY_HEADER)
            A_shared = T.alloc_shared((16, block_K), T.bfloat16)
            B_shared = T.alloc_shared((rescore_BN, block_K), T.bfloat16)
            smem_v = T.alloc_shared((8,), T.float32)
            smem_i = T.alloc_shared((8,), T.int32)
            smem_v2 = T.alloc_shared((4,), T.float32)
            smem_i2 = T.alloc_shared((4,), T.int32)
            smem_win_bx = T.alloc_shared((1,), T.int32)
            C_local = T.alloc_fragment((16, rescore_BN), T.float32)
            C_round = T.alloc_fragment((16, rescore_BN), out_dtype)
            tx = T.get_thread_binding()

            T.call_extern(
                "handle",
                "vllm_vocab_partials_top2_reduce",
                T.address_of(partial_vals[0, 0]),
                T.address_of(partial_idxs[0, 0]),
                T.address_of(smem_v[0]),
                T.address_of(smem_i[0]),
                T.address_of(smem_v2[0]),
                T.address_of(smem_i2[0]),
                T.address_of(smem_win_bx[0]),
                row,
                num_tiles,
                rescore_BN,
                rescore_tiles,
                tx,
            )
            win_bx = smem_win_bx[0]
            T.clear(C_local)
            for ko in T.serial(k_blocks):
                for i, k in T.Parallel(16, block_K):
                    kk = ko * block_K + k
                    A_shared[i, k] = T.if_then_else(
                        kk < K, A[row, T.min(kk, K - 1)], T.Cast(T.bfloat16, 0)
                    )
                for j, k in T.Parallel(rescore_BN, block_K):
                    col = T.min(win_bx * rescore_BN + j, N - 1)
                    kk = ko * block_K + k
                    B_shared[j, k] = T.if_then_else(
                        kk < K, B[col, T.min(kk, K - 1)], T.Cast(T.bfloat16, 0)
                    )
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            if out_dtype != T.float32:
                T.copy(C_local, C_round)
                T.copy(C_round, C_local)
            for i, j in T.Parallel(16, rescore_BN):
                col = T.min(win_bx * rescore_BN + j, N - 1)
                C_local[i, j] = T.Cast(T.float32, Base[row, col]) + C_local[i, j]
            if out_dtype != T.float32:
                T.copy(C_local, C_round)
                T.copy(C_round, C_local)
            T.call_extern(
                "handle",
                "vllm_vocab_tile_second_clocal",
                rescore_BN,
                T.address_of(smem_v[0]),
                T.address_of(smem_i[0]),
                T.address_of(smem_v2[0]),
                T.address_of(smem_i2[0]),
                T.address_of(pairs[0, 0]),
                row,
                win_bx,
                vocab,
                vocab_start,
                pair_stride,
                tx,
            )

    return finish_bf16_markov_top2_exact_kernel


@tilelang.jit
def nvfp4_vocab_projection(
    N: int,
    K: int,
    block_M: int = 16,
    block_N: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    step_groups: int = 1,
    act_stages: int | None = None,
    out_dtype: T.dtype = T.bfloat16,
):
    """``C[M, N] = A @ W^T`` for BF16 ``A[M, K]`` and online NVFP4 ``W``
    (packed E2M1 nibbles ``W[N, K // 2]``, group-16 E4M3 scales
    ``W_scale[N, K // 16]``, and FP32 ``W_global_scale[1]``), accumulated in FP32
    and written as ``out_dtype``."""
    assert K % (GROUP * step_groups) == 0
    assert (
        nvfp4_shared_memory_bytes(
            block_M, block_N, num_stages, step_groups, act_stages=act_stages
        )
        <= SHARED_MEMORY
    )
    STEP = GROUP * step_groups
    steps = K // STEP
    packed_cols = K // 2
    scale_cols = K // NVFP4_GROUP
    step_packed = STEP // 2
    step_scales = STEP // NVFP4_GROUP
    items = block_N * step_scales
    M = T.dynamic("M")

    @T.prim_func
    def vocab_nvfp4_projection(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, packed_cols), T.uint8),
        W_scale: T.Tensor((N, scale_cols), T.float8_e4m3fn),
        W_global_scale: T.Tensor((1,), T.float32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads
        ) as (by, bx):
            A_shared = T.alloc_shared((block_M, STEP), T.bfloat16)
            packed_shared = T.alloc_shared((block_N, step_packed), T.uint8)
            scale_shared = T.alloc_shared((block_N, step_scales), T.float8_e4m3fn)
            B_shared = T.alloc_shared((block_N, STEP), T.bfloat16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            packed_local = T.alloc_local((NVFP4_PACKED,), T.uint8)
            out_local = T.alloc_local((NVFP4_GROUP,), T.bfloat16)
            tx = T.get_thread_binding()
            T.clear(C_local)
            last = num_stages - 1
            act_stage = 0 if act_stages is None else last - act_stages + 1
            for s in (
                T.serial(steps)
                if num_stages <= 1
                else T.Pipelined(
                    steps, order=[0, 1, 2, 3, 4], stage=[act_stage, 0, 0, last, last]
                )
            ):
                T.copy(A[by * block_M, s * STEP], A_shared)
                T.copy(W[bx * block_N, s * step_packed], packed_shared)
                T.copy(W_scale[bx * block_N, s * step_scales], scale_shared)
                for step in T.serial(T.ceildiv(items, threads)):
                    item = step * threads + tx
                    if item < items:
                        j = item // step_scales
                        g = item % step_scales
                        scale = T.Cast(T.float32, scale_shared[j, g])
                        for v in T.vectorized(NVFP4_PACKED):
                            packed_local[v] = packed_shared[j, g * NVFP4_PACKED + v]
                        for v in T.unroll(NVFP4_PACKED):
                            out_local[2 * v] = T.Cast(
                                T.bfloat16, _decode_e2m1(packed_local[v] & 0xF) * scale
                            )
                            out_local[2 * v + 1] = T.Cast(
                                T.bfloat16, _decode_e2m1(packed_local[v] >> 4) * scale
                            )
                        for v in T.vectorized(NVFP4_GROUP):
                            B_shared[j, g * NVFP4_GROUP + v] = out_local[v]
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            global_scale = W_global_scale[0]
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = C_local[i, j] * global_scale
            T.copy(C_local, C[by * block_M, bx * block_N])

    return vocab_nvfp4_projection


@tilelang.jit
def nvfp4_vocab_greedy_partials(
    N: int,
    K: int,
    block_M: int = 16,
    block_N: int = 64,
    num_stages: int = 3,
    threads: int = 128,
    step_groups: int = 1,
    act_stages: int | None = None,
    out_dtype: T.dtype = T.bfloat16,
):
    """Per-tile ``(max, shard index)`` of ``A @ W^T`` for BF16 ``A[M, K]`` and
    online NVFP4 ``W``, reduced in each CTA's epilogue without writing ``[M, N]``
    logits to DRAM."""
    assert K % (GROUP * step_groups) == 0
    assert (
        nvfp4_shared_memory_bytes(
            block_M, block_N, num_stages, step_groups, act_stages=act_stages
        )
        <= SHARED_MEMORY
    )
    STEP = GROUP * step_groups
    steps = K // STEP
    packed_cols = K // 2
    scale_cols = K // NVFP4_GROUP
    step_packed = STEP // 2
    step_scales = STEP // NVFP4_GROUP
    items = block_N * step_scales
    num_tiles = (N + block_N - 1) // block_N
    warps_n = (threads // 32) // (1 if block_M == 16 else 2)
    M = T.dynamic("M")

    @T.prim_func
    def vocab_nvfp4_greedy_partials(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, packed_cols), T.uint8),
        W_scale: T.Tensor((N, scale_cols), T.float8_e4m3fn),
        W_global_scale: T.Tensor((1,), T.float32),
        partial_vals: T.Tensor((M, num_tiles), T.float32),
        partial_idxs: T.Tensor((M, num_tiles), T.int32),
        vocab: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads
        ) as (by, bx):
            T.import_source(_GREEDY_HEADER)
            A_shared = T.alloc_shared((block_M, STEP), T.bfloat16)
            packed_shared = T.alloc_shared((block_N, step_packed), T.uint8)
            scale_shared = T.alloc_shared((block_N, step_scales), T.float8_e4m3fn)
            B_shared = T.alloc_shared((block_N, STEP), T.bfloat16)
            smem_v = T.alloc_shared((block_M * warps_n,), T.float32)
            smem_i = T.alloc_shared((block_M * warps_n,), T.int32)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            packed_local = T.alloc_local((NVFP4_PACKED,), T.uint8)
            out_local = T.alloc_local((NVFP4_GROUP,), T.bfloat16)
            tx = T.get_thread_binding()
            T.clear(C_local)
            last = num_stages - 1
            act_stage = 0 if act_stages is None else last - act_stages + 1
            for s in (
                T.serial(steps)
                if num_stages <= 1
                else T.Pipelined(
                    steps, order=[0, 1, 2, 3, 4], stage=[act_stage, 0, 0, last, last]
                )
            ):
                T.copy(A[by * block_M, s * STEP], A_shared)
                T.copy(W[bx * block_N, s * step_packed], packed_shared)
                T.copy(W_scale[bx * block_N, s * step_scales], scale_shared)
                for step in T.serial(T.ceildiv(items, threads)):
                    item = step * threads + tx
                    if item < items:
                        j = item // step_scales
                        g = item % step_scales
                        scale = T.Cast(T.float32, scale_shared[j, g])
                        for v in T.vectorized(NVFP4_PACKED):
                            packed_local[v] = packed_shared[j, g * NVFP4_PACKED + v]
                        for v in T.unroll(NVFP4_PACKED):
                            out_local[2 * v] = T.Cast(
                                T.bfloat16, _decode_e2m1(packed_local[v] & 0xF) * scale
                            )
                            out_local[2 * v + 1] = T.Cast(
                                T.bfloat16, _decode_e2m1(packed_local[v] >> 4) * scale
                            )
                        for v in T.vectorized(NVFP4_GROUP):
                            B_shared[j, g * NVFP4_GROUP + v] = out_local[v]
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            global_scale = W_global_scale[0]
            for i, j in T.Parallel(block_M, block_N):
                val = C_local[i, j] * global_scale
                if out_dtype == T.bfloat16:
                    val = T.Cast(T.float32, T.Cast(T.bfloat16, val))
                C_local[i, j] = val
            T.call_extern(
                "handle",
                "vllm_vocab_tile_argmax_clocal",
                        block_M,
                        block_N,
                        threads,
                T.address_of(smem_v[0]),
                T.address_of(smem_i[0]),
                T.address_of(partial_vals[0, 0]),
                T.address_of(partial_idxs[0, 0]),
                by,
                bx,
                M,
                vocab,
                num_tiles,
                tx,
            )

    return vocab_nvfp4_greedy_partials




@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def x_norm_kernel(K: int, threads: int = 256):
    warps = threads // 32
    M = T.dynamic("M")

    @T.prim_func
    def vocab_x_norm(
        A: T.Tensor((M, K), T.bfloat16),
        x_norm: T.Tensor((M,), T.float32),
        active_count: T.Tensor((1,), T.int32),
        default_cut_cols: T.Tensor((M,), T.int32),
    ):
        with T.Kernel(M, threads=threads) as (row,):
            pdl.chain()
            T.import_source(_GREEDY_HEADER)
            smem = T.alloc_shared((warps,), T.float32)
            tx = T.get_thread_binding()
            T.call_extern(
                "handle",
                f"vllm_vocab_compute_x_norm<{threads}, {K}>",
                T.address_of(A[0, 0]),
                T.address_of(x_norm[0]),
                T.address_of(active_count[0]),
                T.address_of(default_cut_cols[0]),
                row,
                tx,
                T.address_of(smem[0]),
            )

    return vocab_x_norm


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def screen_vocab_greedy_partials(
    N: int,
    K: int,
    block_M: int = 16,
    block_N: int = 32,
    num_stages: int = 4,
    threads: int = 128,
    step_groups: int = 2,
    act_stages: int | None = None,
    out_dtype: T.dtype = T.bfloat16,
):
    steps = K // (GROUP * step_groups)
    STEP = GROUP * step_groups
    slices = HALF // VEC
    items = block_N * step_groups * slices
    num_tiles = (N + block_N - 1) // block_N
    warps_n = (threads // 32) // (1 if block_M == 16 else 2)
    M = T.dynamic("M")

    @T.prim_func
    def vocab_screen_greedy_partials(
        A: T.Tensor((M, K), T.bfloat16),
        W_q: T.Tensor((N, K), T.uint8),
        row_scale: T.Tensor((N,), T.float32),
        row_bound: T.Tensor((N,), T.float32),
        x_norm: T.Tensor((M,), T.float32),
        cut_cols: T.Tensor((M,), T.int32),
        partial_vals: T.Tensor((M, num_tiles), T.float32),
        partial_idxs: T.Tensor((M, num_tiles), T.int32),
        tile_max_u: T.Tensor((M, num_tiles), T.float32),
        u_buf: T.Tensor((M, N), T.float32),
        vocab: T.int32,
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            pdl.chain()
            T.import_source(_GREEDY_HEADER)
            A_shared = T.alloc_shared((block_M, STEP), T.bfloat16)
            q_shared = T.alloc_shared((block_N, STEP), T.uint8)
            B_shared = T.alloc_shared((block_N, STEP), T.bfloat16)
            smem_v = T.alloc_shared((block_M * warps_n,), T.float32)
            smem_i = T.alloc_shared((block_M * warps_n,), T.int32)
            smem_u = T.alloc_shared((block_M * warps_n,), T.float32)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            low_a = T.alloc_local((VEC,), T.uint8)
            low_b = T.alloc_local((VEC,), T.uint8)
            out_a = T.alloc_local((VEC,), T.bfloat16)
            out_b = T.alloc_local((VEC,), T.bfloat16)
            tx = T.get_thread_binding()
            T.clear(C_local)
            last = num_stages - 1
            act_stage = 0 if act_stages is None else last - act_stages + 1
            for s in (
                T.serial(steps)
                if num_stages <= 1
                else T.Pipelined(
                    steps, order=[0, 1, 2, 3], stage=[act_stage, 0, last, last]
                )
            ):
                T.copy(A[by * block_M, s * STEP], A_shared)
                T.copy(W_q[bx * block_N, s * STEP], q_shared)
                for step in T.serial(T.ceildiv(items, threads)):
                    item = step * threads + tx
                    if items % threads == 0 or item < items:
                        j = item // (step_groups * slices)
                        g = item // slices % step_groups
                        c = item % slices * VEC
                        for v in T.vectorized(VEC):
                            low_a[v] = q_shared[j, g * GROUP + c + v]
                        for v in T.vectorized(VEC):
                            low_b[v] = q_shared[j, g * GROUP + HALF + c + v]
                        T.call_extern(
                            "handle",
                            "vllm_vocab_decode_q8_vec16",
                            T.address_of(low_a[0]),
                            T.address_of(low_b[0]),
                            T.address_of(out_a[0]),
                            T.address_of(out_b[0]),
                        )
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + c + v] = out_a[v]
                        for v in T.vectorized(VEC):
                            B_shared[j, g * GROUP + HALF + c + v] = out_b[v]
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.call_extern(
                "handle",
                "vllm_vocab_screen_epilogue_clocal",
                block_M,
                block_N,
                threads,
                1 if out_dtype != T.float32 else 0,
                T.address_of(row_scale[0]),
                T.address_of(row_bound[0]),
                T.address_of(x_norm[0]),
                T.address_of(cut_cols[0]),
                T.address_of(u_buf[0, 0]),
                T.address_of(smem_v[0]),
                T.address_of(smem_i[0]),
                T.address_of(smem_u[0]),
                T.address_of(partial_vals[0, 0]),
                T.address_of(partial_idxs[0, 0]),
                T.address_of(tile_max_u[0, 0]),
                by,
                bx,
                M,
                N,
                vocab,
                num_tiles,
                tx,
            )

    return vocab_screen_greedy_partials


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def row_thresholds_kernel(
    N: int,
    K: int,
    exponent_base: int,
    block_N: int = 32,
    threads: int = 256,
    out_dtype: T.dtype = T.bfloat16,
    *,
    has_bias: bool = False,
):
    row_bytes = K + K // 2
    num_tiles = (N + block_N - 1) // block_N
    base_minus_one = exponent_base - 1
    out_bf16 = 1 if out_dtype != T.float32 else 0
    warps = threads // 32
    M = T.dynamic("M")
    E = T.dynamic("E")

    @T.macro
    def _body(A, W, exception_entries, row_offsets, x_norm, row_mma_tol, partial_vals, partial_idxs, tile_max_u, u_buf, row_thresh, vocab, top_k, bias_buf=None):
        with T.Kernel(M, threads=threads) as (m,):
            pdl.chain()
            T.import_source(_GREEDY_HEADER)
            smem_v = T.alloc_shared((warps,), T.float32)
            smem_i = T.alloc_shared((warps,), T.int32)
            smem_chosen = T.alloc_shared((16,), T.int32)
            tx = T.get_thread_binding()
            if has_bias:
                T.call_extern(
                    "handle",
                    f"vllm_vocab_row_thresholds<{threads}, {K}, {base_minus_one}, {out_bf16}, {block_N}, true>",
                    T.address_of(A[0, 0]),
                    T.address_of(W[0, 0]),
                    T.address_of(exception_entries[0]),
                    T.address_of(row_offsets[0]),
                    T.address_of(x_norm[0]),
                    T.address_of(row_mma_tol[0]),
                    T.address_of(partial_vals[0, 0]),
                    T.address_of(partial_idxs[0, 0]),
                    T.address_of(tile_max_u[0, 0]),
                    T.address_of(u_buf[0, 0]),
                    T.address_of(row_thresh[0]),
                    T.address_of(smem_v[0]),
                    T.address_of(smem_i[0]),
                    T.address_of(smem_chosen[0]),
                    m,
                    N,
                    vocab,
                    num_tiles,
                    E,
                    top_k,
                    tx,
                    T.address_of(bias_buf[0, 0]),
                )
            else:
                T.call_extern(
                    "handle",
                    f"vllm_vocab_row_thresholds<{threads}, {K}, {base_minus_one}, {out_bf16}, {block_N}>",
                    T.address_of(A[0, 0]),
                    T.address_of(W[0, 0]),
                    T.address_of(exception_entries[0]),
                    T.address_of(row_offsets[0]),
                    T.address_of(x_norm[0]),
                    T.address_of(row_mma_tol[0]),
                    T.address_of(partial_vals[0, 0]),
                    T.address_of(partial_idxs[0, 0]),
                    T.address_of(tile_max_u[0, 0]),
                    T.address_of(u_buf[0, 0]),
                    T.address_of(row_thresh[0]),
                    T.address_of(smem_v[0]),
                    T.address_of(smem_i[0]),
                    T.address_of(smem_chosen[0]),
                    m,
                    N,
                    vocab,
                    num_tiles,
                    E,
                    top_k,
                    tx,
                )

    if has_bias:
        @T.prim_func
        def vocab_markov_row_thresholds(
            A: T.Tensor((M, K), T.bfloat16),
            W: T.Tensor((N, row_bytes), T.uint8),
            exception_entries: T.Tensor((E,), T.int32),
            row_offsets: T.Tensor((N + 1,), T.int32),
            x_norm: T.Tensor((M,), T.float32),
            row_mma_tol: T.Tensor((N,), T.float32),
            partial_vals: T.Tensor((M, num_tiles), T.float32),
            partial_idxs: T.Tensor((M, num_tiles), T.int32),
            tile_max_u: T.Tensor((M, num_tiles), T.float32),
            u_buf: T.Tensor((M, N), T.float32),
            bias_buf: T.Tensor((M, N), T.float32),
            row_thresh: T.Tensor((M,), T.float32),
            vocab: T.int32,
            top_k: T.int32,
        ):
            _body(A, W, exception_entries, row_offsets, x_norm, row_mma_tol, partial_vals, partial_idxs, tile_max_u, u_buf, row_thresh, vocab, top_k, bias_buf)

        return vocab_markov_row_thresholds

    @T.prim_func
    def vocab_row_thresholds(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, row_bytes), T.uint8),
        exception_entries: T.Tensor((E,), T.int32),
        row_offsets: T.Tensor((N + 1,), T.int32),
        x_norm: T.Tensor((M,), T.float32),
        row_mma_tol: T.Tensor((N,), T.float32),
        partial_vals: T.Tensor((M, num_tiles), T.float32),
        partial_idxs: T.Tensor((M, num_tiles), T.int32),
        tile_max_u: T.Tensor((M, num_tiles), T.float32),
        u_buf: T.Tensor((M, N), T.float32),
        row_thresh: T.Tensor((M,), T.float32),
        vocab: T.int32,
        top_k: T.int32,
    ):
        _body(A, W, exception_entries, row_offsets, x_norm, row_mma_tol, partial_vals, partial_idxs, tile_max_u, u_buf, row_thresh, vocab, top_k)

    return vocab_row_thresholds


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def compact_tiles_kernel(
    N: int,
    block_N: int = 32,
    grid_blocks: int = 64,
    threads: int = 256,
    max_rows: int = 256,
):
    num_tiles = (N + block_N - 1) // block_N
    M = T.dynamic("M")

    @T.prim_func
    def vocab_compact_tiles(
        row_thresh: T.Tensor((M,), T.float32),
        tile_max_u: T.Tensor((M, num_tiles), T.float32),
        u_buf: T.Tensor((M, N), T.float32),
        active_cols: T.Tensor((N,), T.int32),
        active_count: T.Tensor((1,), T.int32),
        vocab: T.int32,
    ):
        with T.Kernel(grid_blocks, threads=threads) as (block_idx,):
            pdl.chain()
            T.import_source(_GREEDY_HEADER)
            T_smem = T.alloc_shared((max_rows,), T.float32)
            tx = T.get_thread_binding()
            T.call_extern(
                "handle",
                f"vllm_vocab_compact_tiles<{grid_blocks}, {threads}, {block_N}>",
                T.address_of(row_thresh[0]),
                T.address_of(tile_max_u[0, 0]),
                T.address_of(u_buf[0, 0]),
                T.address_of(active_cols[0]),
                T.address_of(active_count[0]),
                T.address_of(T_smem[0]),
                M,
                N,
                vocab,
                num_tiles,
                block_idx,
                tx,
            )

    return vocab_compact_tiles


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def refine_vocab_greedy_partials(
    N: int,
    K: int,
    exponent_base: int,
    block_M: int = 16,
    block_N: int = 32,
    threads: int = 128,
    step_groups: int = 2,
    out_dtype: T.dtype = T.bfloat16,
    *,
    has_bias: bool = False,
    grid_x: int = 0,
):
    steps = K // (GROUP * step_groups)
    STEP = GROUP * step_groups
    row_bytes = K + K // 2
    base_minus_one = exponent_base - 1
    out_bf16 = 1 if out_dtype != T.float32 else 0
    slices = HALF // VEC
    items = block_N * step_groups * slices
    num_tiles = (N + block_N - 1) // block_N
    launch_x = min(grid_x, num_tiles) if grid_x > 0 else num_tiles
    warps_n = (threads // 32) // (1 if block_M == 16 else 2)
    M = T.dynamic("M")
    E = T.dynamic("E")

    @T.macro
    def _body(A, W, exception_entries, row_offsets, active_cols, active_count, cut_cols, refined_vals, partial_vals, partial_idxs, bias_buf=None):
        with T.Kernel(
            launch_x, T.ceildiv(M, block_M), threads=threads
        ) as (bx_grid, by):
            pdl.chain()
            T.import_source(_GREEDY_HEADER)
            A_shared = T.alloc_shared((block_M, STEP), T.bfloat16)
            B_shared = T.alloc_shared((block_N, STEP), T.bfloat16)
            cols_shared = T.alloc_shared((block_N,), T.int32)
            smem_v = T.alloc_shared((block_M * warps_n,), T.float32)
            smem_i = T.alloc_shared((block_M * warps_n,), T.int32)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            low_a = T.alloc_local((VEC,), T.uint8)
            low_b = T.alloc_local((VEC,), T.uint8)
            codes = T.alloc_local((VEC,), T.uint8)
            out_a = T.alloc_local((VEC,), T.bfloat16)
            out_b = T.alloc_local((VEC,), T.bfloat16)
            tx = T.get_thread_binding()
            num_active = T.alloc_var(T.int32, init=T.max(active_count[0], 1))
            active_tiles = T.ceildiv(num_active, block_N)
            rem_tiles = T.max(active_tiles - bx_grid, 0)
            iters = T.ceildiv(rem_tiles, launch_x) if grid_x > 0 else 1
            for tile_it in T.serial(iters):
                bx = bx_grid + tile_it * launch_x if grid_x > 0 else bx_grid
                if grid_x > 0 or bx * block_N < num_active:
                    for j_init in T.serial(T.ceildiv(block_N, threads)):
                        j_idx = j_init * threads + tx
                        if j_idx < block_N:
                            slot = T.min(bx * block_N + j_idx, num_active - 1)
                            cols_shared[j_idx] = T.if_then_else(
                                active_count[0] > 0, active_cols[slot], 0
                            )
                    T.clear(C_local)
                    if steps % 2 == 0 and items == 2 * threads:
                        low_a0_e = T.alloc_local((VEC,), T.uint8)
                        low_b0_e = T.alloc_local((VEC,), T.uint8)
                        codes0_e = T.alloc_local((VEC,), T.uint8)
                        low_a1_e = T.alloc_local((VEC,), T.uint8)
                        low_b1_e = T.alloc_local((VEC,), T.uint8)
                        codes1_e = T.alloc_local((VEC,), T.uint8)
                        low_a0_o = T.alloc_local((VEC,), T.uint8)
                        low_b0_o = T.alloc_local((VEC,), T.uint8)
                        codes0_o = T.alloc_local((VEC,), T.uint8)
                        low_a1_o = T.alloc_local((VEC,), T.uint8)
                        low_b1_o = T.alloc_local((VEC,), T.uint8)
                        codes1_o = T.alloc_local((VEC,), T.uint8)
                        T.sync_threads()
                        it0 = tx
                        j0 = it0 // (step_groups * slices)
                        g0 = it0 // slices % step_groups
                        c0 = it0 % slices * VEC
                        r0 = cols_shared[j0]
                        it1 = threads + tx
                        j1 = it1 // (step_groups * slices)
                        g1 = it1 // slices % step_groups
                        c1 = it1 % slices * VEC
                        r1 = cols_shared[j1]
                        for v in T.vectorized(VEC):
                            low_a0_e[v] = W[r0, g0 * GROUP + c0 + v]
                        for v in T.vectorized(VEC):
                            low_b0_e[v] = W[r0, g0 * GROUP + HALF + c0 + v]
                        for v in T.vectorized(VEC):
                            codes0_e[v] = W[r0, K + g0 * HALF + c0 + v]
                        for v in T.vectorized(VEC):
                            low_a1_e[v] = W[r1, g1 * GROUP + c1 + v]
                        for v in T.vectorized(VEC):
                            low_b1_e[v] = W[r1, g1 * GROUP + HALF + c1 + v]
                        for v in T.vectorized(VEC):
                            codes1_e[v] = W[r1, K + g1 * HALF + c1 + v]
                        for s2 in T.serial(steps // 2):
                            s_e = s2 * 2
                            s_o = s_e + 1
                            T.copy(A[by * block_M, s_e * STEP], A_shared)
                            ns_o = s_o * STEP
                            nsk_o = K + s_o * (STEP // 2)
                            for v in T.vectorized(VEC):
                                low_a0_o[v] = W[r0, ns_o + g0 * GROUP + c0 + v]
                            for v in T.vectorized(VEC):
                                low_b0_o[v] = W[r0, ns_o + g0 * GROUP + HALF + c0 + v]
                            for v in T.vectorized(VEC):
                                codes0_o[v] = W[r0, nsk_o + g0 * HALF + c0 + v]
                            for v in T.vectorized(VEC):
                                low_a1_o[v] = W[r1, ns_o + g1 * GROUP + c1 + v]
                            for v in T.vectorized(VEC):
                                low_b1_o[v] = W[r1, ns_o + g1 * GROUP + HALF + c1 + v]
                            for v in T.vectorized(VEC):
                                codes1_o[v] = W[r1, nsk_o + g1 * HALF + c1 + v]
                            T.call_extern(
                                "handle",
                                f"vllm_vocab_decode_vec16<{base_minus_one}>",
                                T.address_of(low_a0_e[0]),
                                T.address_of(low_b0_e[0]),
                                T.address_of(codes0_e[0]),
                                T.address_of(out_a[0]),
                                T.address_of(out_b[0]),
                            )
                            for v in T.vectorized(VEC):
                                B_shared[j0, g0 * GROUP + c0 + v] = out_a[v]
                            for v in T.vectorized(VEC):
                                B_shared[j0, g0 * GROUP + HALF + c0 + v] = out_b[v]
                            T.call_extern(
                                "handle",
                                f"vllm_vocab_decode_vec16<{base_minus_one}>",
                                T.address_of(low_a1_e[0]),
                                T.address_of(low_b1_e[0]),
                                T.address_of(codes1_e[0]),
                                T.address_of(out_a[0]),
                                T.address_of(out_b[0]),
                            )
                            for v in T.vectorized(VEC):
                                B_shared[j1, g1 * GROUP + c1 + v] = out_a[v]
                            for v in T.vectorized(VEC):
                                B_shared[j1, g1 * GROUP + HALF + c1 + v] = out_b[v]
                            T.sync_threads()
                            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                            T.sync_threads()
                            T.copy(A[by * block_M, s_o * STEP], A_shared)
                            if s_o + 1 < steps:
                                ns_e = (s_o + 1) * STEP
                                nsk_e = K + (s_o + 1) * (STEP // 2)
                                for v in T.vectorized(VEC):
                                    low_a0_e[v] = W[r0, ns_e + g0 * GROUP + c0 + v]
                                for v in T.vectorized(VEC):
                                    low_b0_e[v] = W[r0, ns_e + g0 * GROUP + HALF + c0 + v]
                                for v in T.vectorized(VEC):
                                    codes0_e[v] = W[r0, nsk_e + g0 * HALF + c0 + v]
                                for v in T.vectorized(VEC):
                                    low_a1_e[v] = W[r1, ns_e + g1 * GROUP + c1 + v]
                                for v in T.vectorized(VEC):
                                    low_b1_e[v] = W[r1, ns_e + g1 * GROUP + HALF + c1 + v]
                                for v in T.vectorized(VEC):
                                    codes1_e[v] = W[r1, nsk_e + g1 * HALF + c1 + v]
                            T.call_extern(
                                "handle",
                                f"vllm_vocab_decode_vec16<{base_minus_one}>",
                                T.address_of(low_a0_o[0]),
                                T.address_of(low_b0_o[0]),
                                T.address_of(codes0_o[0]),
                                T.address_of(out_a[0]),
                                T.address_of(out_b[0]),
                            )
                            for v in T.vectorized(VEC):
                                B_shared[j0, g0 * GROUP + c0 + v] = out_a[v]
                            for v in T.vectorized(VEC):
                                B_shared[j0, g0 * GROUP + HALF + c0 + v] = out_b[v]
                            T.call_extern(
                                "handle",
                                f"vllm_vocab_decode_vec16<{base_minus_one}>",
                                T.address_of(low_a1_o[0]),
                                T.address_of(low_b1_o[0]),
                                T.address_of(codes1_o[0]),
                                T.address_of(out_a[0]),
                                T.address_of(out_b[0]),
                            )
                            for v in T.vectorized(VEC):
                                B_shared[j1, g1 * GROUP + c1 + v] = out_a[v]
                            for v in T.vectorized(VEC):
                                B_shared[j1, g1 * GROUP + HALF + c1 + v] = out_b[v]
                            T.sync_threads()
                            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                            T.sync_threads()
                    else:
                        for s in T.serial(steps):
                            T.copy(A[by * block_M, s * STEP], A_shared)
                            for step in T.serial(T.ceildiv(items, threads)):
                                item = step * threads + tx
                                if items % threads == 0 or item < items:
                                    j = item // (step_groups * slices)
                                    g = item // slices % step_groups
                                    c = item % slices * VEC
                                    w_row = cols_shared[j]
                                    for v in T.vectorized(VEC):
                                        low_a[v] = W[w_row, s * STEP + g * GROUP + c + v]
                                    for v in T.vectorized(VEC):
                                        low_b[v] = W[w_row, s * STEP + g * GROUP + HALF + c + v]
                                    for v in T.vectorized(VEC):
                                        codes[v] = W[
                                            w_row, K + s * (STEP // 2) + g * HALF + c + v
                                        ]
                                    T.call_extern(
                                        "handle",
                                        f"vllm_vocab_decode_vec16<{base_minus_one}>",
                                        T.address_of(low_a[0]),
                                        T.address_of(low_b[0]),
                                        T.address_of(codes[0]),
                                        T.address_of(out_a[0]),
                                        T.address_of(out_b[0]),
                                    )
                                    for v in T.vectorized(VEC):
                                        B_shared[j, g * GROUP + c + v] = out_a[v]
                                    for v in T.vectorized(VEC):
                                        B_shared[j, g * GROUP + HALF + c + v] = out_b[v]
                            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                    if has_bias:
                        T.call_extern(
                            "handle",
                            "vllm_vocab_markov_refine_tile_argmax_clocal",
                            block_M,
                            block_N,
                            threads,
                            K,
                            out_bf16,
                            T.address_of(A[0, 0]),
                            T.address_of(exception_entries[0]),
                            T.address_of(row_offsets[0]),
                            T.address_of(cols_shared[0]),
                            T.address_of(cut_cols[0]),
                            T.address_of(smem_v[0]),
                            T.address_of(smem_i[0]),
                            T.address_of(refined_vals[0, 0]),
                            T.address_of(partial_vals[0, 0]),
                            T.address_of(partial_idxs[0, 0]),
                            by,
                            bx,
                            M,
                            N,
                            num_active,
                            num_tiles,
                            E,
                            tx,
                            T.address_of(bias_buf[0, 0]),
                        )
                    else:
                        T.call_extern(
                            "handle",
                            "vllm_vocab_refine_tile_argmax_clocal",
                            block_M,
                            block_N,
                            threads,
                            K,
                            out_bf16,
                            T.address_of(A[0, 0]),
                            T.address_of(exception_entries[0]),
                            T.address_of(row_offsets[0]),
                            T.address_of(cols_shared[0]),
                            T.address_of(cut_cols[0]),
                            T.address_of(smem_v[0]),
                            T.address_of(smem_i[0]),
                            T.address_of(refined_vals[0, 0]),
                            T.address_of(partial_vals[0, 0]),
                            T.address_of(partial_idxs[0, 0]),
                            by,
                            bx,
                            M,
                            N,
                            num_active,
                            num_tiles,
                            E,
                            tx,
                        )
                    if grid_x > 0:
                        T.sync_threads()

    if has_bias:
        @T.prim_func
        def vocab_packed_markov_refine_partials(
            A: T.Tensor((M, K), T.bfloat16),
            W: T.Tensor((N, row_bytes), T.uint8),
            exception_entries: T.Tensor((E,), T.int32),
            row_offsets: T.Tensor((N + 1,), T.int32),
            active_cols: T.Tensor((N,), T.int32),
            active_count: T.Tensor((1,), T.int32),
            cut_cols: T.Tensor((M,), T.int32),
            bias_buf: T.Tensor((M, N), T.float32),
            refined_vals: T.Tensor((M, N), T.float32),
            partial_vals: T.Tensor((M, num_tiles), T.float32),
            partial_idxs: T.Tensor((M, num_tiles), T.int32),
        ):
            _body(A, W, exception_entries, row_offsets, active_cols, active_count, cut_cols, refined_vals, partial_vals, partial_idxs, bias_buf)

        return vocab_packed_markov_refine_partials

    @T.prim_func
    def vocab_packed_refine_partials(
        A: T.Tensor((M, K), T.bfloat16),
        W: T.Tensor((N, row_bytes), T.uint8),
        exception_entries: T.Tensor((E,), T.int32),
        row_offsets: T.Tensor((N + 1,), T.int32),
        active_cols: T.Tensor((N,), T.int32),
        active_count: T.Tensor((1,), T.int32),
        cut_cols: T.Tensor((M,), T.int32),
        refined_vals: T.Tensor((M, N), T.float32),
        partial_vals: T.Tensor((M, num_tiles), T.float32),
        partial_idxs: T.Tensor((M, num_tiles), T.int32),
    ):
        _body(A, W, exception_entries, row_offsets, active_cols, active_count, cut_cols, refined_vals, partial_vals, partial_idxs)

    return vocab_packed_refine_partials


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def finish_active_vocab_greedy_pairs(
    N: int,
    num_tiles: int,
    block_N: int = 32,
    pair_width: int = 4,
    threads: int = 256,
    *,
    top2_pairs: bool = False,
):
    assert not top2_pairs or pair_width >= 4
    warps = threads // 32
    top2_flag = "true" if top2_pairs else "false"
    M = T.dynamic("M")
    topk_elems = T.dynamic("topk_elems")

    @T.prim_func
    def vocab_active_greedy_finish(
        partial_vals: T.Tensor((M, num_tiles), T.float32),
        partial_idxs: T.Tensor((M, num_tiles), T.int32),
        refined_vals: T.Tensor((M, N), T.float32),
        active_cols: T.Tensor((N,), T.int32),
        active_count: T.Tensor((1,), T.int32),
        pairs: T.Tensor((M, pair_width), T.float32),
        row_topk: T.Tensor((topk_elems,), T.float32),
        vocab_start: T.int32,
        top_k: T.int32,
    ):
        with T.Kernel(M, threads=threads) as (row,):
            pdl.chain()
            T.import_source(_GREEDY_HEADER)
            smem_v = T.alloc_shared((warps,), T.float32)
            smem_i = T.alloc_shared((warps,), T.int32)
            smem_chosen = T.alloc_shared((16,), T.int32)
            tx = T.get_thread_binding()
            T.call_extern(
                "handle",
                f"vllm_vocab_finish_active_argmax<{threads}, {pair_width}, {top2_flag}>",
                T.address_of(partial_vals[0, 0]),
                T.address_of(partial_idxs[0, 0]),
                T.address_of(refined_vals[0, 0]),
                T.address_of(active_cols[0]),
                T.address_of(active_count[0]),
                block_N,
                N,
                top_k,
                T.address_of(pairs[0, 0]),
                T.address_of(row_topk[0]),
                row,
                num_tiles,
                vocab_start,
                tx,
                T.address_of(smem_v[0]),
                T.address_of(smem_i[0]),
                T.address_of(smem_chosen[0]),
            )

    return vocab_active_greedy_finish


def build_screen_weights(
    packed: PackedBf16Weight,
    *,
    wq_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the 8-bit sign-magnitude screening table and certified Cauchy-Schwarz
    residual bounds in 1,024-row blocks."""
    count, width = packed.values.shape
    k = width * 2 // 3
    device = packed.values.device
    W_q = (
        wq_out
        if wq_out is not None
        else torch.empty((count, k), dtype=torch.uint8, device=device)
    )
    row_scale = torch.empty((count,), dtype=torch.float32, device=device)
    row_bound = torch.empty((count,), dtype=torch.float32, device=device)
    row_mma_tol = torch.empty((count,), dtype=torch.float32, device=device)
    offsets_cpu = packed.row_offsets.cpu().tolist()
    entries = packed.exception_entries[: packed.exceptions]
    exc_rows_all = packed.exception_rows[: packed.exceptions]
    for start in range(0, count, _PACK_BLOCK_ROWS):
        end = min(start + _PACK_BLOCK_ROWS, count)
        m = end - start
        vals = packed.values[start:end]
        low = vals[:, :k].to(torch.int32)
        codes = vals[:, k:].to(torch.int32).view(m, k // GROUP, 1, GROUP // 2)
        code = torch.cat((codes & 0xF, codes >> 4), dim=2).reshape(m, k)
        exponent = torch.where(code == 0, 0, code + packed.exponent_base - 1)
        bits = ((low & 0x80) << 8) | (exponent << 7) | (low & 0x7F)
        w_in_f = bits.to(torch.int16).view(torch.bfloat16).float()
        in_amax = w_in_f.abs().amax(dim=1).clamp(min=1e-12)
        exc_lo = offsets_cpu[start]
        exc_hi = offsets_cpu[end]
        if exc_hi > exc_lo:
            sub_entries = entries[exc_lo:exc_hi]
            sub_rows = (exc_rows_all[exc_lo:exc_hi] - start).long()
            sub_cols = (sub_entries >> 16).long()
            bits[sub_rows, sub_cols] = sub_entries & 0xFFFF
        w_full_f = bits.to(torch.int16).view(torch.bfloat16).float()
        full_amax = w_full_f.abs().amax(dim=1).clamp(min=1e-12)
        finite_row = torch.isfinite(full_amax)
        extreme_mask = (full_amax > 8.0 * in_amax) | ~finite_row
        use_amax = torch.where(extreme_mask, in_amax, full_amax)
        base_scale = use_amax / (127.0 / 128.0)
        # Preserve sign and clamp finite extreme outliers rather than zeroing them.
        w_finite = torch.where(torch.isfinite(w_full_f), w_full_f, w_in_f)
        sign = (w_finite < 0).to(torch.uint8)
        abs_target = w_finite.abs()

        def _eval_scale(
            cand_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            scaled = (abs_target / cand_scale.unsqueeze(1)) * 128.0
            u7 = torch.clamp(torch.round(scaled), 0, 127).to(torch.uint8)
            q_signed = torch.where(sign == 1, -u7.float(), u7.float())
            w_hat = (q_signed / 128.0).bfloat16().float() * cand_scale.unsqueeze(1)
            cand_r_v = torch.linalg.vector_norm(w_full_f - w_hat, dim=1)
            return cand_scale, u7, cand_r_v

        r_scale, best_u7, r_v = _eval_scale(base_scale)
        for alpha in (
            0.96,
            0.93,
            0.89,
            0.85,
            0.80,
            0.74,
            0.68,
            0.62,
            0.55,
            0.48,
            0.40,
            0.32,
            0.25,
        ):
            cand_scale, u7, cand_r_v = _eval_scale(base_scale * alpha)
            better = cand_r_v < r_v
            if better.any():
                r_scale = torch.where(better, cand_scale, r_scale)
                best_u7 = torch.where(better.unsqueeze(1), u7, best_u7)
                r_v = torch.where(better, cand_r_v, r_v)
        extreme_finite = extreme_mask & finite_row
        if extreme_finite.any():
            full_base = torch.where(extreme_finite, full_amax, in_amax) / (127.0 / 128.0)
            for f_alpha in (1.0, 0.75, 0.50, 0.30):
                cand_scale, u7, cand_r_v = _eval_scale(full_base * f_alpha)
                better = extreme_finite & (cand_r_v < r_v)
                if better.any():
                    r_scale = torch.where(better, cand_scale, r_scale)
                    best_u7 = torch.where(better.unsqueeze(1), u7, best_u7)
                    r_v = torch.where(better, cand_r_v, r_v)
        W_q[start:end] = (sign << 7) | best_u7
        w_norm = torch.linalg.vector_norm(w_full_f, dim=1)
        r_bound = r_v * 1.0005 + w_norm * 5e-4 + 1e-7
        r_bound = torch.where(
            ~torch.isfinite(r_bound),
            torch.full_like(r_bound, float("inf")),
            r_bound,
        )
        r_tol = w_norm * 5e-4 + 1e-6
        row_scale[start:end] = r_scale
        row_bound[start:end] = r_bound
        row_mma_tol[start:end] = r_tol
    return W_q, row_scale, row_bound, row_mma_tol


def _ensure_screen_weights(
    layer: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    screen = getattr(layer, "_tilelang_vocab_screen", None)
    if screen is not None and screen[0].device == layer.weight.device:
        return screen
    exceptions = getattr(
        layer,
        "packed_bf16_exceptions",
        int(layer.weight_row_offsets[-1].item()),
    )
    packed = PackedBf16Weight(
        layer.weight,
        layer.packed_bf16_exponent_base,
        layer.weight_exception_rows,
        layer.weight_exception_entries,
        layer.weight_row_offsets,
        exceptions,
    )
    carveout_wq = getattr(layer, "_carveout_screen_wq", None)
    if carveout_wq is not None and carveout_wq.device != layer.weight.device:
        carveout_wq = None
    screen = build_screen_weights(packed, wq_out=carveout_wq)
    layer._tilelang_vocab_screen = screen
    return screen


_OUT = {torch.bfloat16: T.bfloat16, torch.float32: T.float32}


def _head_row_tiles(layer: nn.Module) -> tuple[int, ...]:
    """The token tiles a head can launch: up to ``tilelang_max_rows`` when set."""
    max_rows = getattr(layer, "tilelang_max_rows", None)
    return tuple(
        block_M
        for i, block_M in enumerate(ROW_TILES)
        if i == 0 or max_rows is None or max_rows > ROW_TILES[i - 1]
    )


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
        for block_M in _head_row_tiles(layer)
    }


def _refine_tile_config(n: int, k: int, block_M: int, fallback: dict) -> tuple[int, int, int]:
    """Use 1-warp (block_N=8, threads=32, step_groups=2) for 16-row decode tiles
    with register ping-pong when K fits evenly, multiplying active-SM parallelism."""
    if (
        block_M == 16
        and n >= 1024
        and k % (GROUP * 2) == 0
        and (k // (GROUP * 2)) % 2 == 0
    ):
        return 8, 32, 2
    return 32, fallback["threads"], fallback["step_groups"]


def _packed_head_greedy_kernels(layer: nn.Module, out_dtype: torch.dtype) -> dict:
    """A packed head's certified two-pass greedy-pair screening, compaction,
    refinement and reduction for every token tile, as calls."""
    n, k = int(layer.weight.shape[0]), layer.packed_bf16_in_features
    bn = 32
    num_tiles = (n + bn - 1) // bn
    xnorm = prepare.call(x_norm_kernel, k)
    thresh = prepare.call(
        row_thresholds_kernel,
        n,
        k,
        layer.packed_bf16_exponent_base,
        block_N=bn,
        out_dtype=_OUT[out_dtype],
    )
    compact = prepare.call(compact_tiles_kernel, n, block_N=bn)
    screens = {}
    refines = {}
    finishes = {}
    refine_num_tiles_map = {}
    for block_M in _head_row_tiles(layer):
        config = screen_tile_config(block_M, k)
        r_bn, r_th, r_sg = _refine_tile_config(n, k, block_M, config)
        r_grid_x = 40 if (r_bn == 8 and n >= 1024) else 0
        r_nt = (n + r_bn - 1) // r_bn
        screens[block_M] = prepare.call(
            screen_vocab_greedy_partials,
            n,
            k,
            block_M=block_M,
            out_dtype=_OUT[out_dtype],
            **config,
        )
        refines[block_M] = prepare.call(
            refine_vocab_greedy_partials,
            n,
            k,
            layer.packed_bf16_exponent_base,
            block_M=block_M,
            block_N=r_bn,
            threads=r_th,
            step_groups=r_sg,
            out_dtype=_OUT[out_dtype],
            grid_x=r_grid_x,
        )
        if r_nt not in finishes:
            finishes[r_nt] = prepare.call(
                finish_active_vocab_greedy_pairs, n, r_nt, block_N=r_bn
            )
        refine_num_tiles_map[block_M] = r_nt
    return {
        "xnorm": xnorm,
        "thresh": thresh,
        "compact": compact,
        "finishes": finishes,
        "screens": screens,
        "refines": refines,
        "num_tiles": num_tiles,
        "refine_num_tiles_map": refine_num_tiles_map,
    }


def _nvfp4_head_kernels(layer: nn.Module, out_dtype: torch.dtype) -> dict:
    """An online NVFP4 head's projection for every token tile, as calls."""
    n = int(layer.weight.shape[0])
    k = getattr(layer, "nvfp4_in_features", int(layer.weight.shape[1]) * 2)
    return {
        block_M: prepare.call(
            nvfp4_vocab_projection,
            n,
            k,
            block_M=block_M,
            out_dtype=_OUT[out_dtype],
            **tile_config(block_M, k),
        )
        for block_M in _head_row_tiles(layer)
    }


def _nvfp4_head_greedy_kernels(layer: nn.Module, out_dtype: torch.dtype) -> dict:
    """An online NVFP4 head's fused greedy-pair projection and reduction for every
    token tile, as calls."""
    n = int(layer.weight.shape[0])
    k = getattr(layer, "nvfp4_in_features", int(layer.weight.shape[1]) * 2)
    partials = {}
    finishes = {}
    for block_M in _head_row_tiles(layer):
        config = tile_config(block_M, k)
        num_tiles = (n + config["block_N"] - 1) // config["block_N"]
        partials[block_M] = prepare.call(
            nvfp4_vocab_greedy_partials,
            n,
            k,
            block_M=block_M,
            out_dtype=_OUT[out_dtype],
            **config,
        )
        finishes.setdefault(
            num_tiles,
            prepare.call(finish_vocab_greedy_pairs, num_tiles),
        )
    return {"partials": partials, "finishes": finishes}


def _bf16_head_kernels(layer: nn.Module, out_dtype: torch.dtype) -> dict:
    """A BF16 head's projection for every token tile, as calls."""
    n, k = map(int, layer.weight.shape)
    kernels = {}
    for block_M in _head_row_tiles(layer):
        config = dict(BF16_HEAD_CONFIGS[block_M])
        if k % config["block_K"]:
            config["block_K"] = 64
        single_k_block = k <= config["block_K"]
        kernels[block_M] = prepare.call(
            bf16_gemm_decode if single_k_block else bf16_gemm,
            n,
            k,
            block_M=block_M,
            num_stages=1 if single_k_block else 2,
            out_dtype=_OUT[out_dtype],
            **config,
        )
    return kernels


def _markov_greedy_finishes(finishes: dict, num_tiles: int) -> None:
    """The pair finish of ``num_tiles`` partials (the call the other greedy heads
    declare) and its top-2 variant, keyed ``(num_tiles, top2)``."""
    finishes.setdefault(
        (num_tiles, False), prepare.call(finish_vocab_greedy_pairs, num_tiles)
    )
    finishes.setdefault(
        (num_tiles, True),
        prepare.call(finish_vocab_greedy_pairs, num_tiles, top2=True),
    )


def _packed_head_markov_greedy_kernels(
    layer: nn.Module, out_dtype: torch.dtype
) -> dict:
    """A packed Markov head's fused Markov-bias projection, base-logit addition and
    per-tile greedy reduction for every token tile, as calls."""
    n, k = int(layer.weight.shape[0]), layer.packed_bf16_in_features
    partials = {}
    finishes = {}
    tiles = {}
    for block_M in _head_row_tiles(layer):
        config = tile_config(block_M, k)
        assert GREEDY_PART % config["block_N"] == 0
        tiles[block_M] = (n + config["block_N"] - 1) // config["block_N"]
        partials[block_M] = prepare.call(
            packed_vocab_markov_greedy_partials,
            n,
            k,
            layer.packed_bf16_exponent_base,
            block_M=block_M,
            out_dtype=_OUT[out_dtype],
            **config,
        )
        num_tiles = tiles[block_M]
        finishes.setdefault(
            (num_tiles, False), prepare.call(finish_vocab_greedy_pairs, num_tiles)
        )
        finishes.setdefault(
            (num_tiles, True),
            prepare.call(
                finish_packed_markov_top2_exact,
                n,
                k,
                layer.packed_bf16_exponent_base,
                num_tiles,
                config["block_N"],
                step_groups=config["step_groups"],
                out_dtype=_OUT[out_dtype],
            ),
        )
    scr_cfg = tile_config(16, k)
    scr_cfg["block_N"] = 32
    scr_cfg["threads"] = 128
    scr_ntiles = (n + 31) // 32
    screen_call = (
        *prepare.call(
            packed_vocab_markov_screen_partials,
            n,
            k,
            layer.packed_bf16_exponent_base,
            block_M=16,
            out_dtype=_OUT[out_dtype],
            **scr_cfg,
        ),
        scr_ntiles,
    )
    return {"partials": partials, "finishes": finishes, "tiles": tiles, "screen": screen_call}


def _bf16_head_greedy_kernels(layer: nn.Module, out_dtype: torch.dtype) -> dict:
    """A BF16 head's fused Markov-bias projection, base-logit addition and
    per-tile greedy reduction for every token tile, as calls."""
    n, k = map(int, layer.weight.shape)
    partials = {}
    finishes = {}
    tiles = {}
    for block_M in _head_row_tiles(layer):
        config = dict(BF16_HEAD_CONFIGS[block_M])
        if k % config["block_K"]:
            config["block_K"] = 64
        stages = 1 if k <= config["block_K"] else 2
        assert GREEDY_PART % config["block_N"] == 0
        tiles[block_M] = (n + config["block_N"] - 1) // config["block_N"]
        partials[block_M] = prepare.call(
            bf16_vocab_markov_greedy_partials,
            n,
            k,
            block_M=block_M,
            num_stages=stages,
            out_dtype=_OUT[out_dtype],
            **config,
        )
        num_tiles = tiles[block_M]
        finishes.setdefault(
            (num_tiles, False), prepare.call(finish_vocab_greedy_pairs, num_tiles)
        )
        finishes.setdefault(
            (num_tiles, True),
            prepare.call(
                finish_bf16_markov_top2_exact,
                n,
                k,
                num_tiles,
                config["block_N"],
                block_K=config["block_K"],
                out_dtype=_OUT[out_dtype],
            ),
        )
    scr_cfg = dict(BF16_HEAD_CONFIGS[16])
    scr_cfg["block_N"] = 32
    if k % scr_cfg["block_K"]:
        scr_cfg["block_K"] = 64
    scr_stages = 1 if k <= scr_cfg["block_K"] else 2
    scr_ntiles = (n + 31) // 32
    screen_call = (
        *prepare.call(
            bf16_vocab_markov_screen_partials,
            n,
            k,
            block_M=16,
            num_stages=scr_stages,
            out_dtype=_OUT[out_dtype],
            **scr_cfg,
        ),
        scr_ntiles,
    )
    return {"partials": partials, "finishes": finishes, "tiles": tiles, "screen": screen_call}


def _declare_markov_greedy(layer: nn.Module, kernels: dict) -> None:
    """Declare a Markov head's fused greedy calls, bound as
    ``layer.tilelang_vocab_markov_greedy[block_M]`` = (partials, pair finish,
    top-2 finish, tiles)."""
    partials = kernels["partials"]
    finishes = kernels["finishes"]
    tiles = kernels["tiles"]

    def bind() -> None:
        compiled = {
            key: function(*args, **kwargs)
            for key, (function, args, kwargs) in finishes.items()
        }
        layer.tilelang_vocab_markov_greedy = {
            block_M: (
                function(*args, **kwargs),
                compiled[tiles[block_M], False],
                compiled[tiles[block_M], True],
                tiles[block_M],
            )
            for block_M, (function, args, kwargs) in partials.items()
        }

    screen_call = kernels.get("screen")

    def bind_all() -> None:
        bind()
        if screen_call is not None:
            scr_fn, scr_args, scr_kw, num_screen_tiles = screen_call
            layer.tilelang_vocab_markov_screen = {
                16: (scr_fn(*scr_args, **scr_kw), num_screen_tiles)
            }

    calls = list(partials.values()) + list(finishes.values())
    if screen_call is not None:
        calls.append(screen_call[:3])
    prepare.declare(lambda: calls, bind=bind_all)


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
    """Declare a packed head's projection and certified two-pass greedy pairs for
    every token tile, writing ``out_dtype`` logits (BF16, or FP32 as DeepSeek's
    reference head computes them). The head must be packed: the kernels depend on
    its exponent base."""
    _ensure_screen_weights(layer)
    _declare_head(layer, _packed_head_kernels(layer, out_dtype), out_dtype)
    greedy = _packed_head_greedy_kernels(layer, out_dtype)
    xnorm_call = greedy["xnorm"]
    thresh_call = greedy["thresh"]
    compact_call = greedy["compact"]
    finishes = greedy["finishes"]
    screens = greedy["screens"]
    refines = greedy["refines"]
    num_tiles = greedy["num_tiles"]
    refine_num_tiles_map = greedy["refine_num_tiles_map"]

    def bind_greedy() -> None:
        xnorm_fn, xnorm_args, xnorm_kw = xnorm_call
        thresh_fn, thresh_args, thresh_kw = thresh_call
        compact_fn, compact_args, compact_kw = compact_call
        compiled_xnorm = xnorm_fn(*xnorm_args, **xnorm_kw)
        compiled_thresh = thresh_fn(*thresh_args, **thresh_kw)
        compiled_compact = compact_fn(*compact_args, **compact_kw)
        compiled_finishes = {
            r_nt: f_call[0](*f_call[1], **f_call[2])
            for r_nt, f_call in finishes.items()
        }
        layer.tilelang_vocab_greedy = {
            block_M: (
                compiled_xnorm,
                scr_fn(*scr_args, **scr_kw),
                compiled_thresh,
                compiled_compact,
                refines[block_M][0](*refines[block_M][1], **refines[block_M][2]),
                compiled_finishes[refine_num_tiles_map[block_M]],
                num_tiles,
            )
            for block_M, (scr_fn, scr_args, scr_kw) in screens.items()
        }
        layer._tilelang_vocab_refine_num_tiles = dict(refine_num_tiles_map)

    prepare.declare(
        lambda: [xnorm_call, thresh_call, compact_call]
        + list(finishes.values())
        + list(screens.values())
        + list(refines.values()),
        bind=bind_greedy,
    )
    if getattr(layer, "tilelang_markov_head", False):
        _declare_markov_greedy(
            layer, _packed_head_markov_greedy_kernels(layer, out_dtype)
        )
    else:
        n, k = int(layer.weight.shape[0]), layer.packed_bf16_in_features
        cfg16 = screen_tile_config(16, k)
        r_bn16, r_th16, r_sg16 = _refine_tile_config(n, k, 16, cfg16)
        r_grid_x16 = 40 if (r_bn16 == 8 and n >= 1024) else 0
        r_nt16 = (n + r_bn16 - 1) // r_bn16
        m_thresh_call = prepare.call(
            row_thresholds_kernel,
            n,
            k,
            layer.packed_bf16_exponent_base,
            block_N=32,
            out_dtype=_OUT[out_dtype],
            has_bias=True,
        )
        m_refine_call = prepare.call(
            refine_vocab_greedy_partials,
            n,
            k,
            layer.packed_bf16_exponent_base,
            block_M=16,
            block_N=r_bn16,
            threads=r_th16,
            step_groups=r_sg16,
            out_dtype=_OUT[out_dtype],
            has_bias=True,
            grid_x=r_grid_x16,
        )
        m_finish_top2_call = prepare.call(
            finish_active_vocab_greedy_pairs,
            n,
            r_nt16,
            block_N=r_bn16,
            top2_pairs=True,
        )

        def bind_markov_refine() -> None:
            layer.tilelang_vocab_markov_refine = (
                m_thresh_call[0](*m_thresh_call[1], **m_thresh_call[2]),
                m_refine_call[0](*m_refine_call[1], **m_refine_call[2]),
                m_finish_top2_call[0](*m_finish_top2_call[1], **m_finish_top2_call[2]),
            )
            if layer.weight.is_cuda:
                _vocab_screen_base_buffers(
                    layer, 16, n, greedy["num_tiles"], layer.weight.device
                )
                _vocab_markov_bias_buf(layer, 16, n, layer.weight.device)

        prepare.declare(
            lambda: [m_thresh_call, m_refine_call, m_finish_top2_call],
            bind=bind_markov_refine,
        )


def prepare_nvfp4_head(
    layer: nn.Module, out_dtype: torch.dtype = torch.bfloat16
) -> None:
    """Declare an online NVFP4 head's projection and fused greedy pairs for every
    token tile, writing ``out_dtype`` logits."""
    if not hasattr(layer, "nvfp4_in_features"):
        layer.nvfp4_in_features = int(layer.weight.shape[1]) * 2
    _declare_head(layer, _nvfp4_head_kernels(layer, out_dtype), out_dtype)
    greedy = _nvfp4_head_greedy_kernels(layer, out_dtype)
    partials = greedy["partials"]
    finishes = greedy["finishes"]

    def bind_greedy() -> None:
        n, k = int(layer.weight.shape[0]), layer.nvfp4_in_features
        compiled_finishes = {
            num_tiles: function(*args, **kwargs)
            for num_tiles, (function, args, kwargs) in finishes.items()
        }
        layer.tilelang_vocab_greedy = {
            block_M: (
                function(*args, **kwargs),
                compiled_finishes[
                    (n + tile_config(block_M, k)["block_N"] - 1)
                    // tile_config(block_M, k)["block_N"]
                ],
                (n + tile_config(block_M, k)["block_N"] - 1)
                // tile_config(block_M, k)["block_N"],
            )
            for block_M, (function, args, kwargs) in partials.items()
        }

    prepare.declare(
        lambda: list(partials.values()) + list(finishes.values()),
        bind=bind_greedy,
    )


def prepare_bf16_head(
    layer: nn.Module, out_dtype: torch.dtype = torch.bfloat16
) -> None:
    """Declare a BF16 head's projection for every token tile, writing
    ``out_dtype``, and a Markov head's fused Markov greedy pairs."""
    _declare_head(layer, _bf16_head_kernels(layer, out_dtype), out_dtype)
    if getattr(layer, "tilelang_markov_head", False):
        _declare_markov_greedy(layer, _bf16_head_greedy_kernels(layer, out_dtype))


def project_head(
    layer: nn.Module, x: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Logits of ``x[rows, K]`` through a prepared head, in its prepared dtype."""
    if not x.is_contiguous():
        x = x.contiguous()
    rows = x.shape[0]
    if out is None:
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
    elif hasattr(layer, "nvfp4_in_features"):
        kernel(
            x,
            layer.weight,
            layer.weight_scale.view(torch.float8_e4m3fn),
            layer.weight_global_scale.reshape(1),
            out,
        )
    else:
        kernel(x, layer.weight, out)
    return out


def _vocab_greedy_buffers(
    layer: nn.Module,
    rows: int,
    num_tiles: int,
    pair_width: int,
    device: torch.device,
    *,
    scratch_pairs: bool = False,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    val_shape = (rows, num_tiles)
    pair_shape = (rows, pair_width)
    specs: list[tuple[tuple[int, ...], torch.dtype]] = [
        (val_shape, torch.float32),
        (val_shape, torch.int32),
    ]
    if out is None and scratch_pairs:
        specs.append((pair_shape, torch.float32))
    if is_workspace_manager_initialized():
        manager = current_workspace_manager()
        required_bytes = sum(round_up(prod(s) * d.itemsize, 256) for s, d in specs)
        if (
            manager._device.type == device.type
            and manager._device.index in (None, device.index)
            and (not manager.is_locked() or manager.available_bytes() >= required_bytes)
        ):
            buffers = manager.get_simultaneous(*specs)
            if out is not None:
                return buffers[0], buffers[1], out
            if scratch_pairs:
                return buffers[0], buffers[1], buffers[2]
            pairs = torch.empty(pair_shape, dtype=torch.float32, device=device)
            return buffers[0], buffers[1], pairs
    part_elems = rows * num_tiles
    pair_elems = rows * pair_width
    scratch = getattr(layer, "_tilelang_vocab_greedy_scratch", None)
    if (
        scratch is None
        or scratch[0].device != device
        or scratch[0].numel() < part_elems
        or scratch[2].numel() < pair_elems
    ):
        cap_parts = max(part_elems, scratch[0].numel() if scratch is not None else 0)
        cap_pairs = max(pair_elems, scratch[2].numel() if scratch is not None else 0)
        scratch = (
            torch.empty(cap_parts, dtype=torch.float32, device=device),
            torch.empty(cap_parts, dtype=torch.int32, device=device),
            torch.empty(cap_pairs, dtype=torch.float32, device=device),
        )
        layer._tilelang_vocab_greedy_scratch = scratch
    partial_vals = scratch[0][:part_elems].view(rows, num_tiles)
    partial_idxs = scratch[1][:part_elems].view(rows, num_tiles)
    if out is not None:
        pairs = out
    elif scratch_pairs:
        pairs = scratch[2][:pair_elems].view(rows, pair_width)
    else:
        pairs = torch.empty(pair_shape, dtype=torch.float32, device=device)
    return partial_vals, partial_idxs, pairs


def _vocab_screen_buffers(
    layer: nn.Module,
    rows: int,
    n: int,
    num_tiles: int,
    pair_width: int,
    device: torch.device,
    *,
    scratch_pairs: bool = False,
    refine_num_tiles: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if refine_num_tiles is None:
        refine_num_tiles = num_tiles
    val_shape = (rows, num_tiles)
    ref_shape = (rows, refine_num_tiles)
    pair_shape = (rows, pair_width)
    u_shape = (rows, n)
    r_shape = (rows,)
    col_shape = (n,)
    cnt_shape = (1,)
    specs: list[tuple[tuple[int, ...], torch.dtype]] = [
        (val_shape, torch.float32),
        (val_shape, torch.int32),
        (val_shape, torch.float32),
        (u_shape, torch.float32),
        (r_shape, torch.float32),
        (r_shape, torch.float32),
        (col_shape, torch.int32),
        (cnt_shape, torch.int32),
        (r_shape, torch.int32),
    ]
    if refine_num_tiles != num_tiles:
        specs.extend([(ref_shape, torch.float32), (ref_shape, torch.int32)])
    if scratch_pairs:
        specs.append((pair_shape, torch.float32))
    if is_workspace_manager_initialized():
        manager = current_workspace_manager()
        required_bytes = sum(round_up(prod(s) * d.itemsize, 256) for s, d in specs)
        if (
            manager._device.type == device.type
            and manager._device.index in (None, device.index)
            and (not manager.is_locked() or manager.available_bytes() >= required_bytes)
        ):
            buffers = manager.get_simultaneous(*specs)
            pairs = (
                buffers[-1]
                if scratch_pairs
                else torch.empty(pair_shape, dtype=torch.float32, device=device)
            )
            ref_pv = buffers[9] if refine_num_tiles != num_tiles else buffers[0]
            ref_pi = buffers[10] if refine_num_tiles != num_tiles else buffers[1]
            return (
                buffers[0],
                buffers[1],
                pairs,
                buffers[2],
                buffers[3],
                buffers[4],
                buffers[5],
                buffers[6],
                buffers[7],
                buffers[8],
                ref_pv,
                ref_pi,
            )
    part_elems = rows * num_tiles
    pair_elems = rows * pair_width
    u_elems = rows * n
    scratch = getattr(layer, "_tilelang_vocab_greedy_scratch", None)
    if (
        scratch is None
        or len(scratch) < 10
        or scratch[0].device != device
        or scratch[0].numel() < part_elems
        or scratch[2].numel() < pair_elems
        or scratch[4].numel() < u_elems
        or scratch[5].numel() < rows
        or scratch[7].numel() < n
    ):
        cap_parts = max(part_elems, scratch[0].numel() if scratch is not None else 0)
        cap_pairs = max(pair_elems, scratch[2].numel() if scratch is not None else 0)
        cap_u = max(
            u_elems,
            scratch[4].numel() if scratch is not None and len(scratch) >= 9 else 0,
        )
        cap_r = max(
            rows,
            scratch[5].numel() if scratch is not None and len(scratch) >= 9 else 0,
        )
        cap_n = max(
            n,
            scratch[7].numel() if scratch is not None and len(scratch) >= 9 else 0,
        )
        scratch = (
            torch.empty(cap_parts, dtype=torch.float32, device=device),
            torch.empty(cap_parts, dtype=torch.int32, device=device),
            torch.empty(cap_pairs, dtype=torch.float32, device=device),
            torch.empty(cap_parts, dtype=torch.float32, device=device),
            torch.empty(cap_u, dtype=torch.float32, device=device),
            torch.empty(cap_r, dtype=torch.float32, device=device),
            torch.empty(cap_r, dtype=torch.float32, device=device),
            torch.empty(cap_n, dtype=torch.int32, device=device),
            torch.empty(1, dtype=torch.int32, device=device),
            torch.empty(cap_r, dtype=torch.int32, device=device),
        )
        layer._tilelang_vocab_greedy_scratch = scratch
    partial_vals = scratch[0][:part_elems].view(rows, num_tiles)
    partial_idxs = scratch[1][:part_elems].view(rows, num_tiles)
    pairs = (
        scratch[2][:pair_elems].view(rows, pair_width)
        if scratch_pairs
        else torch.empty(pair_shape, dtype=torch.float32, device=device)
    )
    tile_max_u = scratch[3][:part_elems].view(rows, num_tiles)
    u_buf = scratch[4][:u_elems].view(rows, n)
    x_norm = scratch[5][:rows]
    row_thresh = scratch[6][:rows]
    active_cols = scratch[7][:n]
    active_count = scratch[8]
    default_cut_cols = scratch[9][:rows]
    if refine_num_tiles == num_tiles:
        ref_partial_vals, ref_partial_idxs = partial_vals, partial_idxs
    else:
        ref_elems = rows * refine_num_tiles
        r_scratch = getattr(layer, "_tilelang_vocab_refine_scratch", None)
        if (
            r_scratch is None
            or r_scratch[0].device != device
            or r_scratch[0].numel() < ref_elems
        ):
            cap_ref = max(ref_elems, r_scratch[0].numel() if r_scratch is not None else 0)
            r_scratch = (
                torch.empty(cap_ref, dtype=torch.float32, device=device),
                torch.empty(cap_ref, dtype=torch.int32, device=device),
            )
            layer._tilelang_vocab_refine_scratch = r_scratch
        ref_partial_vals = r_scratch[0][:ref_elems].view(rows, refine_num_tiles)
        ref_partial_idxs = r_scratch[1][:ref_elems].view(rows, refine_num_tiles)
    return (
        partial_vals,
        partial_idxs,
        pairs,
        tile_max_u,
        u_buf,
        x_norm,
        row_thresh,
        active_cols,
        active_count,
        default_cut_cols,
        ref_partial_vals,
        ref_partial_idxs,
    )


def project_head_greedy_pairs(
    layer: nn.Module,
    x: torch.Tensor,
    vocab: int | None = None,
    vocab_start: int = 0,
    pair_width: int = 4,
    *,
    scratch_pairs: bool = False,
    cut_cols: torch.Tensor | None = None,
    top_k: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Each row's ``(max, global index)`` pair of ``x[rows, K]`` through a prepared
    packed head over its ``vocab`` real columns starting at ``vocab_start``, written
    into the first two columns of ``[rows, pair_width]`` FP32 without materializing
    the shard's logits. When ``top_k > 1``, also returns ``[rows, 2 * top_k]`` FP32
    per-row top candidates (``values`` then ``global_indices``)."""
    if not x.is_contiguous():
        x = x.contiguous()
    rows = x.shape[0]
    n = int(layer.weight.shape[0])
    if vocab is None:
        vocab = n
    if hasattr(layer, "nvfp4_in_features"):
        if cut_cols is not None or top_k > 1:
            raise ValueError("NVFP4 head greedy pairs do not support cut_cols or top_k > 1")
        partials_kernel, finish_kernel, num_tiles = layer.tilelang_vocab_greedy[
            row_tile(rows)
        ]
        partial_vals, partial_idxs, pairs = _vocab_greedy_buffers(
            layer,
            rows,
            num_tiles,
            pair_width,
            x.device,
            scratch_pairs=scratch_pairs,
        )
        partials_kernel(
            x,
            layer.weight,
            layer.weight_scale.view(torch.float8_e4m3fn),
            layer.weight_global_scale.reshape(1),
            partial_vals,
            partial_idxs,
            vocab,
        )
        finish_kernel(partial_vals, partial_idxs, pairs, vocab_start)
        return pairs
    if top_k < 1 or top_k > 16:
        raise ValueError(f"top_k must be in [1, 16], got {top_k}")
    rt = row_tile(rows)
    (
        xnorm_kernel,
        screen_kernel,
        thresh_kernel,
        compact_kernel,
        refine_kernel,
        finish_kernel,
        num_tiles,
    ) = layer.tilelang_vocab_greedy[rt]
    W_q, row_scale, row_bound, row_mma_tol = _ensure_screen_weights(layer)
    refine_num_tiles = getattr(layer, "_tilelang_vocab_refine_num_tiles", {}).get(
        rt, num_tiles
    )
    (
        partial_vals,
        partial_idxs,
        pairs,
        tile_max_u,
        u_buf,
        x_norm,
        row_thresh,
        active_cols,
        active_count,
        default_cut_cols,
        ref_partial_vals,
        ref_partial_idxs,
    ) = _vocab_screen_buffers(
        layer,
        rows,
        n,
        num_tiles,
        pair_width,
        x.device,
        scratch_pairs=scratch_pairs,
        refine_num_tiles=refine_num_tiles,
    )
    if cut_cols is None:
        cut_tensor = default_cut_cols
    else:
        cut_tensor = cut_cols if cut_cols.is_contiguous() else cut_cols.contiguous()
    row_topk = (
        torch.empty((rows, 2 * top_k), dtype=torch.float32, device=x.device)
        if top_k > 1
        else None
    )
    xnorm_kernel(x, x_norm, active_count, default_cut_cols)
    screen_kernel(
        x,
        W_q,
        row_scale,
        row_bound,
        x_norm,
        cut_tensor,
        partial_vals,
        partial_idxs,
        tile_max_u,
        u_buf,
        vocab,
    )
    thresh_kernel(
        x,
        layer.weight,
        layer.weight_exception_entries,
        layer.weight_row_offsets,
        x_norm,
        row_mma_tol,
        partial_vals,
        partial_idxs,
        tile_max_u,
        u_buf,
        row_thresh,
        vocab,
        top_k,
    )
    compact_kernel(
        row_thresh,
        tile_max_u,
        u_buf,
        active_cols,
        active_count,
        vocab,
    )
    refine_kernel(
        x,
        layer.weight,
        layer.weight_exception_entries,
        layer.weight_row_offsets,
        active_cols,
        active_count,
        cut_tensor,
        u_buf,
        ref_partial_vals,
        ref_partial_idxs,
    )
    finish_kernel(
        ref_partial_vals,
        ref_partial_idxs,
        u_buf,
        active_cols,
        active_count,
        pairs,
        row_topk.view(-1) if row_topk is not None else pairs.view(-1),
        vocab_start,
        top_k,
    )
    if row_topk is not None:
        return pairs, row_topk
    return pairs


def project_head_markov_greedy_pairs(
    layer: nn.Module,
    x: torch.Tensor,
    base_logits: torch.Tensor,
    vocab: int | None = None,
    vocab_start: int = 0,
    pair_width: int = 4,
    out: torch.Tensor | None = None,
    *,
    top2: bool = False,
) -> torch.Tensor:
    """Each row's ``(max, global index)`` pair of ``base_logits + x @ W^T``
    through a prepared Markov head over its ``vocab`` real columns starting
    at ``vocab_start``, written into the first two columns of ``[rows, pair_width]``
    FP32 without materializing the Markov bias tensor: the pairs
    ``greedy_markov_shard_pair`` writes from the materialized bias, to the bit
    (with ``top2``, its runner-up in columns 2 and 3 too)."""
    if not x.is_contiguous():
        x = x.contiguous()
    rows = x.shape[0]
    n = int(layer.weight.shape[0])
    if vocab is None:
        vocab = n
    partials_kernel, finish_kernel, finish_top2_kernel, num_tiles = (
        layer.tilelang_vocab_markov_greedy[row_tile(rows)]
    )
    partial_vals, partial_idxs, pairs = _vocab_greedy_buffers(
        layer,
        rows,
        num_tiles,
        pair_width,
        x.device,
        out=out,
    )
    if hasattr(layer, "packed_bf16_exponent_base"):
        partials_kernel(
            x,
            layer.weight,
            layer.weight_exception_rows,
            layer.weight_exception_entries,
            layer.weight_row_offsets,
            base_logits,
            partial_vals,
            partial_idxs,
            vocab,
        )
    else:
        partials_kernel(
            x,
            layer.weight,
            base_logits,
            partial_vals,
            partial_idxs,
            vocab,
        )
    if top2:
        if hasattr(layer, "packed_bf16_exponent_base"):
            finish_top2_kernel(
                x,
                layer.weight,
                layer.weight_exception_rows,
                layer.weight_exception_entries,
                layer.weight_row_offsets,
                base_logits,
                partial_vals,
                partial_idxs,
                pairs,
                vocab,
                vocab_start,
            )
        else:
            finish_top2_kernel(
                x,
                layer.weight,
                base_logits,
                partial_vals,
                partial_idxs,
                pairs,
                vocab,
                vocab_start,
            )
    else:
        finish_kernel(partial_vals, partial_idxs, pairs, vocab_start)
    return pairs



def _vocab_screen_base_buffers(
    layer: nn.Module,
    rows: int,
    n: int,
    num_tiles: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Persistent screen-base scratch kept separate from per-step refinement
    buffers so upper bounds stay intact across sequential Markov steps."""
    cap = max(rows, 16)
    scratch = getattr(layer, "_tilelang_screen_base_scratch", None)
    if (
        scratch is None
        or scratch[0].device != device
        or scratch[0].shape[0] < cap
        or scratch[0].shape[1] != n
        or scratch[2].shape[1] != num_tiles
    ):
        scratch = (
            torch.empty((cap, n), dtype=torch.float32, device=device),
            torch.empty((cap,), dtype=torch.float32, device=device),
            torch.empty((cap, num_tiles), dtype=torch.float32, device=device),
            torch.empty((cap, num_tiles), dtype=torch.int32, device=device),
            torch.empty((cap, num_tiles), dtype=torch.float32, device=device),
            torch.empty((1,), dtype=torch.int32, device=device),
            torch.empty((cap,), dtype=torch.int32, device=device),
        )
        layer._tilelang_screen_base_scratch = scratch
    return (
        scratch[0][:rows],
        scratch[1][:rows],
        scratch[2][:rows],
        scratch[3][:rows],
        scratch[4][:rows],
        scratch[5],
        scratch[6][:rows],
    )


def screen_head_base(
    layer: nn.Module,
    x: torch.Tensor,
    vocab: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run only the 8-bit base-head screen on ``x[rows, K]``, returning
    certified upper bounds ``u_base[rows, n]`` and activation norms ``x_norm[rows]``."""
    if not x.is_contiguous():
        x = x.contiguous()
    rows = x.shape[0]
    n = int(layer.weight.shape[0])
    if vocab is None:
        vocab = n
    xnorm_kernel, screen_kernel, _, _, _, _, num_tiles = layer.tilelang_vocab_greedy[
        row_tile(rows)
    ]
    W_q, row_scale, row_bound, _row_mma_tol = _ensure_screen_weights(layer)
    u_buf, x_norm, partial_vals, partial_idxs, tile_max_u, active_count, cut_cols = (
        _vocab_screen_base_buffers(layer, rows, n, num_tiles, x.device)
    )
    xnorm_kernel(x, x_norm, active_count, cut_cols)
    screen_kernel(
        x,
        W_q,
        row_scale,
        row_bound,
        x_norm,
        cut_cols,
        partial_vals,
        partial_idxs,
        tile_max_u,
        u_buf,
        vocab,
    )
    return u_buf, x_norm


def _vocab_markov_bias_buf(
    layer: nn.Module,
    rows: int,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    cap = max(rows, 16)
    buf = getattr(layer, "_tilelang_vocab_markov_bias_scratch", None)
    if (
        buf is None
        or buf.device != device
        or buf.shape[0] < cap
        or buf.shape[1] != n
    ):
        buf = torch.empty((cap, n), dtype=torch.float32, device=device)
        layer._tilelang_vocab_markov_bias_scratch = buf
    return buf[:rows]


def project_head_screened_markov_greedy_pairs(
    head_layer: nn.Module,
    markov_layer: nn.Module,
    x_step: torch.Tensor,
    u_base_step: torch.Tensor,
    x_norm_step: torch.Tensor,
    markov_embed: torch.Tensor,
    vocab: int | None = None,
    vocab_start: int = 0,
    pair_width: int = 4,
    out: torch.Tensor | None = None,
    *,
    top2: bool = False,
) -> torch.Tensor:
    """Exact ``(max, global index)`` (and runner-up in columns 2 and 3 with
    ``top2``) of ``x_step @ W_head^T + markov_embed @ W_markov^T`` using the
    certified 8-bit base upper bounds ``u_base_step`` and exact 12-bit refinement."""
    if not x_step.is_contiguous():
        x_step = x_step.contiguous()
    if not u_base_step.is_contiguous():
        u_base_step = u_base_step.contiguous()
    if not x_norm_step.is_contiguous():
        x_norm_step = x_norm_step.contiguous()
    if not markov_embed.is_contiguous():
        markov_embed = markov_embed.contiguous()
    rows = x_step.shape[0]
    if rows > 16:
        raise ValueError(f"Screened Markov greedy supports at most 16 rows, got {rows}")
    n = int(head_layer.weight.shape[0])
    if vocab is None:
        vocab = n
    _, _, _, compact_kernel, _, finish_top1_kernel, num_tiles = (
        head_layer.tilelang_vocab_greedy[16]
    )
    thresh_bias_kernel, refine_bias_kernel, finish_top2_kernel = (
        head_layer.tilelang_vocab_markov_refine
    )
    markov_screen_kernel, markov_tiles = markov_layer.tilelang_vocab_markov_screen[16]
    assert markov_tiles == num_tiles, (markov_tiles, num_tiles)
    _W_q, _row_scale, row_bound, row_mma_tol = _ensure_screen_weights(head_layer)
    refine_num_tiles = getattr(
        head_layer, "_tilelang_vocab_refine_num_tiles", {}
    ).get(16, num_tiles)
    (
        partial_vals,
        partial_idxs,
        pairs,
        tile_max_u,
        u_buf,
        _scratch_x_norm,
        row_thresh,
        active_cols,
        active_count,
        cut_cols,
        ref_partial_vals,
        ref_partial_idxs,
    ) = _vocab_screen_buffers(
        head_layer,
        rows,
        n,
        num_tiles,
        pair_width,
        x_step.device,
        scratch_pairs=out is not None,
        refine_num_tiles=refine_num_tiles,
    )
    if out is not None:
        pairs = out
    bias_buf = _vocab_markov_bias_buf(head_layer, rows, n, x_step.device)
    if hasattr(markov_layer, "packed_bf16_exponent_base"):
        markov_screen_kernel(
            markov_embed,
            markov_layer.weight,
            markov_layer.weight_exception_rows,
            markov_layer.weight_exception_entries,
            markov_layer.weight_row_offsets,
            u_base_step,
            row_bound,
            x_norm_step,
            bias_buf,
            u_buf,
            partial_vals,
            partial_idxs,
            tile_max_u,
            active_count,
            cut_cols,
            vocab,
        )
    else:
        markov_screen_kernel(
            markov_embed,
            markov_layer.weight,
            u_base_step,
            row_bound,
            x_norm_step,
            bias_buf,
            u_buf,
            partial_vals,
            partial_idxs,
            tile_max_u,
            active_count,
            cut_cols,
            vocab,
        )
    top_k = 2 if top2 else 1
    thresh_bias_kernel(
        x_step,
        head_layer.weight,
        head_layer.weight_exception_entries,
        head_layer.weight_row_offsets,
        x_norm_step,
        row_mma_tol,
        partial_vals,
        partial_idxs,
        tile_max_u,
        u_buf,
        bias_buf,
        row_thresh,
        vocab,
        top_k,
    )
    compact_kernel(
        row_thresh,
        tile_max_u,
        u_buf,
        active_cols,
        active_count,
        vocab,
    )
    refine_bias_kernel(
        x_step,
        head_layer.weight,
        head_layer.weight_exception_entries,
        head_layer.weight_row_offsets,
        active_cols,
        active_count,
        cut_cols,
        bias_buf,
        u_buf,
        ref_partial_vals,
        ref_partial_idxs,
    )
    finish_kernel = finish_top2_kernel if top2 else finish_top1_kernel
    row_topk_arg = pairs.view(-1)
    finish_kernel(
        ref_partial_vals,
        ref_partial_idxs,
        u_buf,
        active_cols,
        active_count,
        pairs,
        row_topk_arg,
        vocab_start,
        top_k,
    )
    return pairs


__all__ = [
    "BF16_HEAD_CONFIGS",
    "GREEDY_PART",
    "GROUP",
    "NVFP4_GROUP",
    "PackedBf16Weight",
    "ROW_TILES",
    "TILE_CONFIGS",
    "WINDOW",
    "bf16_vocab_markov_greedy_partials",
    "finish_vocab_greedy_pairs",
    "nvfp4_shared_memory_bytes",
    "nvfp4_vocab_greedy_partials",
    "nvfp4_vocab_projection",
    "pack",
    "pack_bf16_weight",
    "packed_bf16_exception_count",
    "packed_bf16_histogram",
    "packed_bf16_pack_rows",
    "packed_vocab_greedy_partials",
    "packed_vocab_markov_greedy_partials",
    "packed_vocab_projection",
    "prepare_bf16_head",
    "prepare_nvfp4_head",
    "prepare_packed_head",
    "project_head",
    "project_head_greedy_pairs",
    "project_head_markov_greedy_pairs",
    "project_head_screened_markov_greedy_pairs",
    "screen_head_base",
    "row_tile",
    "shared_memory_bytes",
    "tile_config",
    "unpack",
    "unpack_bf16_weight",
]
