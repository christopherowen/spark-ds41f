# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang RoPE for DeepSeek-V4.1, in place on the last ``rope_dim`` columns.

TileKernels' ``apply_rotary`` arithmetic (GPT-J interleaved pairs, FP32 with
``ieee_fmaf``, cos and sin halves of one FP32 table row), with what V4.1 adds:
compressed rows floor their position to a multiple of the compression ratio, and
the rotation can run inverse. Only the rotated columns are read and written; the
leading columns stay where they are.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import tilelang
import tilelang.language as T
import torch

_PAIRS = 4  # pairs per thread: 8 BF16, one 16-byte access


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def rope_kernel(heads: int, dim: int, rope_dim: int, ratio: int, inverse: bool):
    half = rope_dim // 2
    assert rope_dim <= dim and half % _PAIRS == 0
    lanes = half // _PAIRS  # threads per head
    heads_per_block = min(heads, 256 // lanes)
    threads = heads_per_block * lanes
    start = dim - rope_dim
    tokens = T.dynamic("tokens")
    positions_max = T.dynamic("positions_max")

    @T.prim_func
    def rope(
        x: T.Tensor((tokens, heads, dim), T.bfloat16),
        positions: T.Tensor((tokens,), T.int64),
        cos_sin: T.Tensor((positions_max, rope_dim), T.float32),
    ):
        with T.Kernel(tokens, T.ceildiv(heads, heads_per_block), threads=threads) as (
            t,
            hb,
        ):
            tx = T.get_thread_binding()
            head = hb * heads_per_block + tx // lanes
            lane = tx % lanes
            xs = T.alloc_local((2 * _PAIRS,), T.bfloat16)
            cs = T.alloc_local((2 * _PAIRS,), T.float32)
            position = positions[t] // ratio * ratio if ratio > 1 else positions[t]
            if head < heads:
                for i in T.vectorized(_PAIRS):
                    cs[i] = cos_sin[position, lane * _PAIRS + i]
                for i in T.vectorized(_PAIRS):
                    cs[_PAIRS + i] = cos_sin[position, half + lane * _PAIRS + i]
                for j in T.vectorized(2 * _PAIRS):
                    xs[j] = x[t, head, start + lane * 2 * _PAIRS + j]
                for j in T.unroll(_PAIRS):
                    first = T.cast(xs[2 * j], T.float32)
                    second = T.cast(xs[2 * j + 1], T.float32)
                    cosine = cs[j]
                    sine = cs[_PAIRS + j]
                    if inverse:
                        xs[2 * j] = T.cast(
                            T.ieee_fmaf(first, cosine, second * sine), T.bfloat16
                        )
                        xs[2 * j + 1] = T.cast(
                            T.ieee_fmaf(first, -sine, second * cosine), T.bfloat16
                        )
                    else:
                        xs[2 * j] = T.cast(
                            T.ieee_fmaf(first, cosine, -(second * sine)), T.bfloat16
                        )
                        xs[2 * j + 1] = T.cast(
                            T.ieee_fmaf(first, sine, second * cosine), T.bfloat16
                        )
                for j in T.vectorized(2 * _PAIRS):
                    x[t, head, start + lane * 2 * _PAIRS + j] = xs[j]

    return rope


@torch.library.custom_op("vllm::dsv41_tilelang_rope", mutates_args=("x",))
def rope_(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    ratio: int,
    inverse: bool,
) -> None:
    """Rotate the last ``cos_sin_cache.shape[1]`` columns of contiguous BF16 ``x``
    ([T, D] or [T, H, D]) in place at ``positions`` (int64, non-negative)."""
    rows = x.shape[0]
    if rows == 0:
        return
    heads = x.shape[1] if x.ndim == 3 else 1
    if not x.is_contiguous() or x.dtype != torch.bfloat16:
        raise ValueError("TileLang RoPE rotates contiguous BF16 rows in place")
    kernel = rope_kernel(heads, x.shape[-1], cos_sin_cache.shape[1], ratio, inverse)
    kernel(x.view(rows, heads, x.shape[-1]), positions, cos_sin_cache)


@rope_.register_fake
def _rope_fake(x, positions, cos_sin_cache, ratio, inverse):
    return None


__all__ = ["rope_", "rope_kernel"]
