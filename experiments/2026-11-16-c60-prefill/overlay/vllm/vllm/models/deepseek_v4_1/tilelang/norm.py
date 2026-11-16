# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native TileLang RMSNorm and fused per-token cast for DeepSeek-V4.1."""

import functools
import math
from typing import Optional, Union

import tilelang
import torch
from tilelang.cuda import language as T
from torch import nn

from vllm.models.deepseek_v4_1.tilelang import pdl, prepare
from vllm.models.deepseek_v4_1.tilelang.gemm import SF_BLOCK
from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows, empty_mxfp8
from vllm.models.deepseek_v4_1.tilelang.quant_common import (
    IMPORT_BF16X2_FP8_CVT,
    CastOutputConfig,
    QuantTensor,
    align,
    alloc_scaling_factors,
    cast_epilogue,
    ceil_div,
    get_cast_output_config,
    get_packed_ue8m0_pack_factor,
    get_packed_ue8m0_torch_dtype,
    get_sf_and_inv,
    get_sf_index,
    get_sf_inv_dtype,
    get_sf_shape,
)


@tilelang.jit
def get_norm_forward_and_per_token_cast_kernel_cuda(
    hidden: int,
    with_weight: bool,
    with_residual: bool,
    eps: float,
    out_scale: float,
    x_dtype: T.dtype,
    out_config: CastOutputConfig,
    with_out_bf16: bool = False,
    mega_moe_block_m: int = 0,
    use_pdl: bool = False,
):
    assert 0 < hidden <= 8192
    assert out_config.with_sf or out_config.dtype == x_dtype
    assert not out_config.with_sf or out_config.sf_block == (1, 32)
    assert not with_out_bf16 or out_config.with_sf

    # A row is split into 16-byte vectors, thread `row_offset` of the row holds
    # vectors `vi * row_threads + row_offset`.
    num_threads = 128
    num_elems_per_thread = 32
    vec_size = math.gcd(16 // x_dtype.bytes, hidden)
    num_vecs = hidden // vec_size
    row_threads = min(
        num_threads,
        max(
            min(8, tilelang.next_power_of_2(num_vecs)),
            tilelang.next_power_of_2(ceil_div(hidden, num_elems_per_thread)),
        ),
    )

    num_per_channels = out_config.sf_block[1]
    group_lanes = num_per_channels // vec_size
    num_sf_cols = ceil_div(hidden, num_per_channels)
    packed_sf_pad = (
        out_config.use_packed_ue8m0
        and out_config.use_tma_aligned_col_major_sf
        and num_sf_cols % 4 != 0
    )
    sf_inv_dtype = get_sf_inv_dtype(x_dtype, out_config)
    if out_config.with_sf:
        assert hidden % num_per_channels == 0
    num_thread_vec = ceil_div(num_vecs, row_threads)
    block_m = num_threads // row_threads
    warp_threads = min(32, row_threads)
    num_row_warps = row_threads // warp_threads
    row_layout = T.Fragment(
        (block_m, row_threads), forward_thread_fn=lambda i, j: i * row_threads + j
    )
    warp_layout = T.Fragment(
        (block_m, num_row_warps),
        forward_thread_fn=lambda i, w, rep: i * row_threads + w * warp_threads + rep,
        replicate=warp_threads,
    )
    warp_reduce_layout = T.Fragment(
        (block_m, num_row_warps, warp_threads),
        forward_thread_fn=lambda i, w, j: i * row_threads + w * warp_threads + j,
    )
    row_reduce_layout = T.Fragment(
        (block_m, row_threads, num_row_warps),
        forward_thread_fn=lambda i, j, w: i * row_threads + j,
    )

    mega_sf_tile_rows = 128
    mega_sf_lane_rows = 32
    mega_sf_cols = mega_sf_tile_rows // mega_sf_lane_rows
    if mega_moe_block_m > 0:
        assert (
            out_config.with_sf
            and out_config.round_sf
            and hidden % mega_sf_tile_rows == 0
        )
    mega_moe_aligned_block_m = align(mega_moe_block_m, mega_sf_tile_rows)

    batch = T.dynamic("batch")
    num_tokens = T.dynamic("num_tokens")
    x_stride = T.dynamic("x_stride")
    x_batch_stride = T.dynamic("x_batch_stride")
    out_stride = T.dynamic("out_stride")
    out_batch_stride = T.dynamic("out_batch_stride")
    out_bf16_stride = T.dynamic("out_bf16_stride")
    out_bf16_batch_stride = T.dynamic("out_bf16_batch_stride")
    residual_stride = T.dynamic("residual_stride")
    residual_batch_stride = T.dynamic("residual_batch_stride")
    residual_out_stride = T.dynamic("residual_out_stride")
    residual_out_batch_stride = T.dynamic("residual_out_batch_stride")
    out_sf_stride = T.dynamic("out_sf_stride")
    out_sf_batch_stride = T.dynamic("out_sf_batch_stride")
    mega_moe_sf_rows = T.dynamic("mega_moe_sf_rows")
    sf_shape = (
        get_sf_shape((num_tokens, hidden), out_config) if out_config.with_sf else (1, 1)
    )

    @T.macro
    def get_mega_moe_sf_row(token_id: int):
        i = token_id % mega_moe_block_m
        return (
            token_id // mega_moe_block_m * mega_moe_aligned_block_m
            + i // mega_sf_tile_rows * mega_sf_tile_rows
            + i % mega_sf_lane_rows * mega_sf_cols
            + i % mega_sf_tile_rows // mega_sf_lane_rows
        )

    @T.prim_func
    def norm_forward_and_per_token_cast_kernel(
        x: T.StridedTensor[
            (batch, num_tokens, hidden), (x_batch_stride, x_stride, 1), x_dtype
        ],
        out: T.StridedTensor[
            (batch, num_tokens, hidden),
            (out_batch_stride, out_stride, 1),
            out_config.dtype,
        ],
        out_sf: T.StridedTensor[
            (batch, *sf_shape),
            (out_sf_batch_stride, out_sf_stride, 1),
            out_config.sf_dtype,
        ],
        out_bf16: T.StridedTensor[
            (batch, num_tokens, hidden),
            (out_bf16_batch_stride, out_bf16_stride, 1),
            T.bfloat16,
        ],
        mega_moe_sf: T.StridedTensor[
            (mega_moe_sf_rows, hidden // 128, 4),
            (4, mega_moe_sf_rows * 4, 1),
            T.uint8,
        ],
        weight: T.Tensor[(hidden,), x_dtype],
        rstd: T.Tensor[(batch, num_tokens), T.float32],
        residual: T.StridedTensor[
            (batch, num_tokens, hidden),
            (residual_batch_stride, residual_stride, 1),
            x_dtype,
        ],
        residual_out: T.StridedTensor[
            (batch, num_tokens, hidden),
            (residual_out_batch_stride, residual_out_stride, 1),
            x_dtype,
        ],
    ):
        with T.Kernel(
            T.ceildiv(num_tokens, block_m), batch, threads=num_threads
        ) as (pid_token, batch_id):
            if sf_inv_dtype == T.bfloat16:
                T.import_source(IMPORT_BF16X2_FP8_CVT)
            T.assume(x_stride % 16 == 0)
            T.assume(x_batch_stride % 16 == 0)
            T.assume(out_stride % 16 == 0)
            T.assume(out_batch_stride % 16 == 0)
            if with_out_bf16:
                T.assume(out_bf16_stride % 16 == 0)
                T.assume(out_bf16_batch_stride % 16 == 0)
            if with_residual:
                T.assume(residual_stride % 16 == 0)
                T.assume(residual_batch_stride % 16 == 0)
                T.assume(residual_out_stride % 16 == 0)
                T.assume(residual_out_batch_stride % 16 == 0)
            T.annotate_min_blocks_per_sm(5)
            if use_pdl:
                T.pdl_sync()
            tid = T.get_thread_binding()
            row_id = tid // row_threads
            row_offset = tid % row_threads
            token_id = pid_token * block_m + row_id

            sum_partial = T.alloc_fragment((block_m, row_threads), T.float32)
            warp_reducer = T.alloc_reducer(
                (block_m, num_row_warps), T.float32, op="sum"
            )
            warp_fragment = T.alloc_fragment((block_m, num_row_warps), T.float32)
            sum_shared = T.alloc_shared((block_m, num_row_warps), T.float32)
            row_partial = T.alloc_fragment(
                (block_m, row_threads, num_row_warps), T.float32
            )
            sum_reducer = T.alloc_reducer((block_m, row_threads), T.float32, op="sum")
            sum_fragment = T.alloc_fragment((block_m, row_threads), T.float32)
            T.annotate_layout(
                {
                    sum_partial: row_layout,
                    warp_fragment: warp_layout,
                    row_partial: row_reduce_layout,
                    sum_fragment: row_layout,
                }
            )
            x_local = T.alloc_local((num_thread_vec, vec_size), x_dtype)
            y_fp32_local = T.alloc_local((vec_size,), T.float32)
            y_local = T.alloc_local((num_thread_vec, vec_size), x_dtype)
            sum_local = T.alloc_local((2,), T.float32)
            if with_residual:
                residual_local = T.alloc_local((vec_size,), x_dtype)
            if with_weight:
                weight_local = T.alloc_local((vec_size,), x_dtype)
            if out_config.with_sf:
                sf_local = T.alloc_local((num_thread_vec,), out_config.sf_dtype)
                amax_local = T.alloc_local((num_thread_vec,), T.float32)

            T.clear(sum_local)
            for vi in T.unroll(num_thread_vec):
                token_offset = (vi * row_threads + row_offset) * vec_size
                if token_id < num_tokens and token_offset < hidden:
                    for j in T.vectorized(vec_size):
                        x_local[vi, j] = x[batch_id, token_id, token_offset + j]
                    if with_residual:
                        for j in T.vectorized(vec_size):
                            residual_local[j] = residual[
                                batch_id, token_id, token_offset + j
                            ]
                        for j in T.vectorized(vec_size):
                            x_local[vi, j] += residual_local[j]
                        for j in T.vectorized(vec_size):
                            residual_out[batch_id, token_id, token_offset + j] = (
                                x_local[vi, j]
                            )
                else:
                    T.clear(x_local[vi, :])
                for j in T.vectorized(vec_size):
                    sum_local[j % 2] += T.float32(x_local[vi, j]) * T.float32(
                        x_local[vi, j]
                    )

            sum_partial[row_id, row_offset] = sum_local[0] + sum_local[1]
            T.reducer_init(warp_reducer)
            for i, w, j in T.Parallel(
                block_m, num_row_warps, warp_threads, loop_layout=warp_reduce_layout
            ):
                T.reducer_update(
                    warp_reducer[i, w], sum_partial[i, w * warp_threads + j]
                )
            T.finalize_reducer(warp_reducer, warp_fragment)
            if row_offset % warp_threads == 0:
                sum_shared[row_id, row_offset // warp_threads] = warp_fragment[
                    row_id, row_offset // warp_threads
                ]
            T.sync_threads()
            for i, j, w in T.Parallel(
                block_m, row_threads, num_row_warps, loop_layout=row_reduce_layout
            ):
                row_partial[i, j, w] = sum_shared[i, w]
            T.reducer_init(sum_reducer)
            for i, j, w in T.Parallel(
                block_m, row_threads, num_row_warps, loop_layout=row_reduce_layout
            ):
                T.reducer_update(sum_reducer[i, j], row_partial[i, j, w])
            T.finalize_reducer(sum_reducer, sum_fragment)
            rstd_value = T.alloc_var(
                T.float32,
                init=T.rsqrt(sum_fragment[row_id, row_offset] / hidden + eps),
            )
            if row_offset == 0 and token_id < num_tokens:
                rstd[batch_id, token_id] = rstd_value

            for vi in T.unroll(num_thread_vec):
                token_offset = (vi * row_threads + row_offset) * vec_size
                is_valid = token_id < num_tokens and token_offset < hidden
                if with_weight:
                    if token_offset < hidden:
                        for j in T.vectorized(vec_size):
                            weight_local[j] = weight[token_offset + j]
                    for j in T.vectorized(vec_size):
                        y_fp32_local[j] = (
                            T.float32(x_local[vi, j])
                            * rstd_value
                            * (T.float32(weight_local[j]) * out_scale)
                        )
                else:
                    for j in T.vectorized(vec_size):
                        y_fp32_local[j] = (
                            T.float32(x_local[vi, j]) * rstd_value * out_scale
                        )
                for j in T.vectorized(vec_size):
                    y_local[vi, j] = y_fp32_local[j]
                if not out_config.with_sf:
                    if is_valid:
                        for j in T.vectorized(vec_size):
                            out[batch_id, token_id, token_offset + j] = y_local[vi, j]
                else:
                    if with_out_bf16 and is_valid:
                        for j in T.vectorized(vec_size):
                            out_bf16[batch_id, token_id, token_offset + j] = y_local[
                                vi, j
                            ]

                    amax = T.alloc_var(T.float32, init=0.0)
                    for j in T.unroll(vec_size):
                        amax = T.max(amax, T.abs(y_fp32_local[j]))
                    amax = T.float32(T.cast(amax, x_dtype))
                    for i in T.unroll(int(math.log2(group_lanes))):
                        amax = T.max(amax, T.shfl_xor(amax, 1 << i))
                    amax_local[vi] = amax

            if out_config.with_sf:
                for vi in T.unroll(num_thread_vec):
                    token_offset = (vi * row_threads + row_offset) * vec_size
                    sf, sf_inv = get_sf_and_inv(amax_local[vi], out_config, sf_inv_dtype)
                    sf_local[vi] = sf
                    if token_id < num_tokens and token_offset < hidden:
                        for j in T.vectorized(vec_size):
                            out[batch_id, token_id, token_offset + j] = (
                                y_local[vi, j] * sf_inv
                            )

            if out_config.with_sf:
                if token_id < num_tokens and row_offset % group_lanes == 0:
                    for vi in T.unroll(num_thread_vec):
                        token_offset = (vi * row_threads + row_offset) * vec_size
                        if token_offset < hidden:
                            sf_col = token_offset // num_per_channels
                            sf_row, sf_column = get_sf_index(
                                token_id, sf_col, out_config
                            )
                            out_sf[batch_id, sf_row, sf_column] = sf_local[vi]
                            if packed_sf_pad and sf_col == num_sf_cols - 1:
                                for pad_col in T.unroll(
                                    num_sf_cols, align(num_sf_cols, 4)
                                ):
                                    _, pad_column = get_sf_index(
                                        token_id, pad_col, out_config
                                    )
                                    out_sf[batch_id, sf_row, pad_column] = 0
                            if mega_moe_block_m > 0:
                                exponent = (
                                    sf_local[vi]
                                    if out_config.use_packed_ue8m0
                                    else T.cast(
                                        T.reinterpret(sf_local[vi], T.uint32) >> 23,
                                        T.uint8,
                                    )
                                )
                                mega_moe_sf[
                                    get_mega_moe_sf_row(token_id),
                                    sf_col // 4,
                                    sf_col % 4,
                                ] = exponent

            if use_pdl:
                T.pdl_trigger()

    return norm_forward_and_per_token_cast_kernel


def norm_forward_and_per_token_cast_impl(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    fmt: str,
    num_per_channels: Optional[int],
    residual: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    with_out_bf16: bool = False,
    out: Optional[torch.Tensor] = None,
    out_sf: Optional[torch.Tensor] = None,
    out_bf16: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    mega_moe_sf: Optional[torch.Tensor] = None,
    mega_moe_block_m: int = 0,
    use_pdl: bool = False,
) -> tuple[
    Union[torch.Tensor, QuantTensor],
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    assert (
        x.ndim in (2, 3)
        and x.dtype in (torch.bfloat16, torch.float32)
        and x.stride(-1) == 1
    )
    num_tokens, hidden = x.shape[-2:]
    batch = x.shape[0] if x.ndim == 3 else 1
    assert hidden > 0
    assert hidden % 16 == 0 and hidden <= 8192

    def batched(tensor):
        return tensor.unsqueeze(0) if tensor is not None and x.ndim == 2 else tensor

    if weight is not None:
        assert (
            weight.shape == (hidden,)
            and weight.dtype == x.dtype
            and weight.device == x.device
            and weight.is_contiguous()
        )

    assert (fmt, num_per_channels) in [("e4m3", 32), ("bf16", None), ("fp32", None)]
    out_config = get_cast_output_config(
        fmt,
        (1, num_per_channels or 1),
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
    )
    assert (
        out_config.with_sf or out_config.torch_dtype == x.dtype
    ), "the unquantized output must match the input dtype"
    assert not out_config.with_sf or hidden % num_per_channels == 0
    assert (
        not with_out_bf16 or out_config.with_sf
    ), "the extra output is only needed with casting"
    assert (mega_moe_sf is None) == (mega_moe_block_m == 0)
    if mega_moe_sf is not None:
        assert x.ndim == 2 and mega_moe_block_m > 0
        assert out_config.with_sf and round_sf and hidden % 128 == 0
        assert mega_moe_sf.device == x.device
        assert (
            mega_moe_sf.dtype == torch.int32
            and mega_moe_sf.shape == (num_tokens, hidden // 128)
            and mega_moe_sf.stride(0) == 1
        )
        num_mega_moe_sf_rows = mega_moe_sf.stride(1)
        assert (
            ceil_div(num_tokens, mega_moe_block_m) * align(mega_moe_block_m, 128)
            <= num_mega_moe_sf_rows
        )

    if out is None:
        out = torch.empty(x.shape, dtype=out_config.torch_dtype, device=x.device)
    if out_config.with_sf:
        if out_sf is None:
            alloc_shape = (num_tokens, hidden)
            if x.ndim == 3:
                if use_tma_aligned_col_major_sf:
                    sf_pack = (
                        get_packed_ue8m0_pack_factor() if use_packed_ue8m0 else 1
                    )
                    alloc_shape = (
                        num_tokens,
                        batch * align(hidden, num_per_channels * sf_pack),
                    )
                else:
                    alloc_shape = (batch * num_tokens, hidden)
            out_sf = alloc_scaling_factors(alloc_shape, out_config, x.device)
            if x.ndim == 3:
                sf_rows, sf_cols = get_sf_shape((num_tokens, hidden), out_config)
                out_sf = out_sf.as_strided(
                    (batch, sf_rows, sf_cols),
                    (sf_rows * out_sf.stride(0), out_sf.stride(0), 1),
                )
        else:
            assert not use_tma_aligned_col_major_sf
            if use_packed_ue8m0:
                assert out_sf.dtype == get_packed_ue8m0_torch_dtype()
                out_sf = out_sf.view(torch.uint8)
            expected_sf_shape = get_sf_shape((num_tokens, hidden), out_config)
            assert (
                out_sf.shape == (*x.shape[:-2], *expected_sf_shape)
                and out_sf.stride(-1) == 1
            )
            assert (
                out_sf.dtype == out_config.sf_torch_dtype
                and out_sf.device == x.device
            )
    else:
        assert out_sf is None
    if with_out_bf16:
        if out_bf16 is None:
            out_bf16 = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    else:
        assert out_bf16 is None
    residual_out = (
        torch.empty_like(x)
        if residual is not None and residual_out is None
        else residual_out
    )
    if out_bf16 is not None:
        assert (
            out_bf16.shape == x.shape
            and out_bf16.dtype == torch.bfloat16
            and out_bf16.device == x.device
            and out_bf16.stride(-1) == 1
        )
    for tensor in (residual, residual_out):
        if tensor is not None:
            assert (
                tensor.shape == x.shape
                and tensor.dtype == x.dtype
                and tensor.device == x.device
                and tensor.stride(-1) == 1
            )
    assert (
        out.shape == x.shape
        and out.dtype == out_config.torch_dtype
        and out.device == x.device
        and out.stride(-1) == 1
    )
    rstd = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)

    kernel_kwargs = dict(
        hidden=hidden,
        with_weight=weight is not None,
        with_residual=residual is not None,
        eps=eps,
        out_scale=out_scale,
        x_dtype=T.dtype(x.dtype),
        out_config=out_config,
        with_out_bf16=with_out_bf16,
        mega_moe_block_m=mega_moe_block_m,
    )
    if batch * num_tokens > 0:
        for tensor in (x, out, out_bf16, residual, residual_out):
            if tensor is not None:
                assert all(stride % 16 == 0 for stride in tensor.stride()[:-1])
        kernel = get_norm_forward_and_per_token_cast_kernel_cuda(
            **kernel_kwargs, use_pdl=use_pdl
        )
        if mega_moe_sf is not None:
            mega_moe_sf = (
                mega_moe_sf.mT.view(torch.uint8).as_strided(
                    (num_mega_moe_sf_rows, hidden // 128, 4),
                    (4, num_mega_moe_sf_rows * 4, 1),
                )
            )
        args = (
            batched(x),
            batched(out),
            batched(out_sf),
            batched(out_bf16),
            mega_moe_sf,
            weight,
            batched(rstd),
            batched(residual),
            batched(residual_out),
        )
        kernel(*args)

    norm_input = x if residual is None else residual_out
    if out_config.with_sf:
        if x.ndim == 3 and use_tma_aligned_col_major_sf:
            sf_rows, sf_cols = out_sf.shape[-2:]
            out_sf = out_sf.as_strided(
                (batch * sf_rows, sf_cols), (out_sf.stride(1), 1)
            )
            out_sf = cast_epilogue(out_sf, out_config)
            out_sf = out_sf.as_strided(
                (batch, out_sf.shape[0], sf_rows),
                (sf_rows * out_sf.stride(1), *out_sf.stride()),
            )
        else:
            out_sf = cast_epilogue(out_sf, out_config)
        return (out, out_sf), out_bf16, rstd, norm_input
    return out, None, rstd, norm_input


def norm_forward_and_per_token_cast(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    fmt: str,
    num_per_channels: int,
    residual: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    with_out_bf16: bool = False,
    out: Optional[QuantTensor] = None,
    out_bf16: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    mega_moe_sf: Optional[torch.Tensor] = None,
    mega_moe_block_m: int = 0,
    *,
    use_pdl: bool = False,
) -> tuple[QuantTensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """RMSNorm with optional residual addition, cast to FP8 with per-token scales."""
    out_x, out_sf = (None, None) if out is None else out
    return norm_forward_and_per_token_cast_impl(
        x,
        weight,
        eps,
        fmt,
        num_per_channels,
        residual,
        out_scale,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
        with_out_bf16=with_out_bf16,
        out=out_x,
        out_sf=out_sf,
        out_bf16=out_bf16,
        residual_out=residual_out,
        mega_moe_sf=mega_moe_sf,
        mega_moe_block_m=mega_moe_block_m,
        use_pdl=use_pdl,
    )


def norm_forward(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    residual: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    *,
    use_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm with optional input-dtype residual addition."""
    assert x.dtype in (torch.bfloat16, torch.float32)
    out, _, rstd, norm_input = norm_forward_and_per_token_cast_impl(
        x,
        weight,
        eps,
        "bf16" if x.dtype == torch.bfloat16 else "fp32",
        None,
        residual,
        out_scale,
        out=out,
        residual_out=residual_out,
        use_pdl=use_pdl,
    )
    return out, rstd, norm_input


def _rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float
) -> None:
    pdl.norm_forward(x, weight, eps, out=out)


@functools.cache
def _warm(hidden: int, eps: float, dtype: torch.dtype) -> None:
    """Native TileLang RMSNorm of one zero row of ``hidden`` (``dtype`` input and
    weight), so its kernel (keyed by the width, eps and dtype, not the rows) is
    compiled before the cache freezes."""
    x = torch.zeros((1, hidden), dtype=dtype, device="cuda")
    weight = torch.ones(hidden, dtype=dtype, device="cuda")
    _rmsnorm(x, weight, torch.empty_like(x), eps)


@functools.cache
def _warm_with_cast(hidden: int, eps: float, dtype: torch.dtype) -> None:
    """TileKernels' fused RMSNorm + MXFP8 per-token cast of one zero row."""
    x = torch.zeros((1, hidden), dtype=dtype, device="cuda")
    weight = torch.ones(hidden, dtype=dtype, device="cuda")
    out_bf16 = torch.empty_like(x)
    cast = empty_mxfp8(out_bf16)
    pdl.norm_forward_and_per_token_cast(
        x,
        weight,
        eps,
        "e4m3",
        SF_BLOCK,
        round_sf=True,
        use_packed_ue8m0=True,
        with_out_bf16=True,
        out=(cast.values[:1], cast.scales[:1].view(torch.int32)),
        out_bf16=out_bf16,
    )


class TileLangRMSNorm(nn.Module):
    """``y = x * rsqrt(mean(x^2) + eps) * weight`` in FP32, in the input dtype."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16), requires_grad=False
        )
        self.variance_epsilon = eps
        self.with_cast = False

    def declare_kernels(self) -> None:
        """Native TileLang norm of this width; its input has the weight's dtype."""
        hidden, eps, dtype = (
            self.weight.shape[0],
            self.variance_epsilon,
            self.weight.dtype,
        )
        prepare.declare(warm=lambda: _warm(hidden, eps, dtype))
        if self.with_cast:
            prepare.declare(warm=lambda: _warm_with_cast(hidden, eps, dtype))

    def forward_with_cast(
        self, x: torch.Tensor, out: torch.Tensor | None = None
    ) -> MXFP8Rows:
        """The normalized rows (written into ``out`` when given) beside their
        MXFP8 cast, in one TileKernels ``norm_forward_and_per_token_cast`` launch."""
        if x.ndim != 2 or (out is not None and out.ndim != 2):
            raise ValueError("forward_with_cast requires 2D input and output")
        if out is None:
            out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        cast = empty_mxfp8(out)
        m = x.shape[0]
        if m:
            pdl.norm_forward_and_per_token_cast(
                x,
                self.weight,
                self.variance_epsilon,
                "e4m3",
                SF_BLOCK,
                round_sf=True,
                use_packed_ue8m0=True,
                with_out_bf16=True,
                out=(cast.values[:m], cast.scales[:m].view(torch.int32)),
                out_bf16=out,
            )
        return cast

    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """The normalized rows, into ``out`` (contiguous, ``x``'s rows) when given."""
        if x.ndim == 2 and (out is None or out.ndim == 2):
            if out is None:
                out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            pdl.norm_forward(x, self.weight, self.variance_epsilon, out=out)
            return out
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        if out is None:
            out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        pdl.norm_forward(x, self.weight, self.variance_epsilon, out=out.view(x.shape))
        return out.view(shape)
