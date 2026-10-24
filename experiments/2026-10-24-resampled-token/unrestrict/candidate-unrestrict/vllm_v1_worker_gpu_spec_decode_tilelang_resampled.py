# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Choose and commit the resampled or bonus token after verification."""

import tilelang
import tilelang.language as T
import torch

_PASS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}
_THREADS = 32


@tilelang.jit(pass_configs=_PASS)
def insert_kernel(
    blocks: int,
    value_dtype: str,
    index_dtype: str,
    threads: int,
    sampled_stride: int,
    id_stride: int,
    value_stride: int,
):
    requests = T.dynamic("requests")
    width = T.dynamic("width")
    mapping_len = T.dynamic("mapping_len")
    state_len = T.dynamic("state_len")
    padded = 1 << (blocks - 1).bit_length()

    @T.prim_func
    def insert(
        sampled: T.StridedTensor((requests, width), (sampled_stride, 1), T.int64),
        counts: T.Tensor((requests,), T.int32),
        local_ids: T.StridedTensor((requests, blocks), (id_stride, 1), T.int64),
        local_max: T.StridedTensor((requests, blocks), (value_stride, 1), value_dtype),
        cu: T.Tensor((requests + 1,), T.int32),
        mapping: T.Tensor((mapping_len,), index_dtype),
        temperature: T.Tensor((state_len,), T.float32),
    ):
        T.annotate_restrict_buffers(sampled, counts, local_ids, local_max, cu, mapping, temperature)
        with T.Kernel(requests, threads=threads) as row:
            thread = T.get_thread_binding()
            if blocks <= 4:
                if thread == 0:
                    count = counts[row]
                    counts[row] = count + 1
                    position = cu[row] + count
                    state = mapping[position]
                    if temperature[state] != 0 or position == cu[row + 1] - 1:
                        best = T.alloc_var(value_dtype, init=-T.infinity(value_dtype))
                        winner = T.alloc_var(T.int32, init=0)
                        for i in T.unroll(blocks):
                            value = local_max[row, i]
                            if value > best:
                                best = value
                                winner = i
                        sampled[row, count] = local_ids[row, winner]
            else:
                if threads == 32:
                    count_local = T.alloc_var(T.int32, init=0)
                    if thread == 0:
                        count_local = counts[row]
                        counts[row] = count_local + 1
                    count = T.shfl_sync(count_local, 0)
                else:
                    count = counts[row]
                position = cu[row] + count
                state = mapping[position]
                if temperature[state] != 0 or position == cu[row + 1] - 1:
                    values = T.alloc_fragment((padded,), value_dtype)
                    maximum = T.alloc_fragment((1,), value_dtype)
                    indices = T.alloc_fragment((padded,), T.int32)
                    winner = T.alloc_fragment((1,), T.int32)
                    for i in T.Parallel(padded):
                        values[i] = -T.infinity(value_dtype)
                        if i < blocks:
                            value = local_max[row, i]
                            values[i] = T.if_then_else(
                                T.isnan(value), -T.infinity(value_dtype), value
                            )
                    T.reduce_max(values, maximum, dim=0)
                    for i in T.Parallel(padded):
                        indices[i] = T.if_then_else(
                            i < blocks and values[i] == maximum[0], i, padded
                        )
                    T.reduce_min(indices, winner, dim=0)
                    if thread == 0:
                        sampled[row, count] = local_ids[row, winner[0]]
                if threads != 32:
                    T.sync_threads()
                    if thread == 0:
                        counts[row] = count + 1

    return insert


def insert_call(
    blocks: int,
    value_dtype: torch.dtype,
    index_dtype: torch.dtype,
    strides: tuple[int, int, int],
):
    from vllm.models.deepseek_v4_1.tilelang.prepare import call

    return call(
        insert_kernel,
        blocks,
        "float64" if value_dtype == torch.float64 else "float32",
        "int64" if index_dtype == torch.int64 else "int32",
        1 if blocks <= 4 else _THREADS,
        *strides,
    )


def register_insert(blocks: int, use_fp64: bool, sampled_width: int) -> None:
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    if JitWarmupRegistry.active() is None:
        return
    from vllm.models.deepseek_v4_1.tilelang.prepare import declare

    dtype = torch.float64 if use_fp64 else torch.float32
    declare(
        lambda: [
            insert_call(blocks, dtype, d, (sampled_width, blocks, blocks))
            for d in (torch.int32, torch.int64)
        ]
    )


def insert_resampled(
    sampled: torch.Tensor,
    counts: torch.Tensor,
    local_ids: torch.Tensor,
    local_max: torch.Tensor,
    cu_num_logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    if counts.numel() == 0:
        return
    _, args, kwargs = insert_call(
        local_max.shape[1],
        local_max.dtype,
        expanded_idx_mapping.dtype,
        (sampled.stride(0), local_ids.stride(0), local_max.stride(0)),
    )
    insert_kernel(*args, **kwargs)(
        sampled,
        counts,
        local_ids,
        local_max,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
    )
