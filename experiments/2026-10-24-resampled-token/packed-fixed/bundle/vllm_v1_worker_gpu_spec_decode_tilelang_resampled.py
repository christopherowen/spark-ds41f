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
def insert_kernel(blocks: int, value_dtype: str, index_dtype: str, threads: int):
    requests = T.dynamic("requests")
    width = T.dynamic("width")
    sampled_stride = T.dynamic("sampled_stride")
    id_stride = T.dynamic("id_stride")
    value_stride = T.dynamic("value_stride")
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
        with T.Kernel(requests, threads=threads) as row:
            if threads == 32:
                count_local = T.alloc_var(T.int32, init=0)
                if T.get_thread_binding() == 0:
                    count_local = counts[row]
                count = T.shfl_sync(count_local, 0)
            else:
                count = counts[row]
            position = cu[row] + count
            state = mapping[position]
            if temperature[state] != 0 or position == cu[row + 1] - 1:
                if value_dtype == "float32":
                    keys = T.alloc_fragment((padded,), T.uint64)
                    top = T.alloc_fragment((1,), T.uint64)
                    for i in T.Parallel(padded):
                        keys[i] = T.uint64(0)
                        if i < blocks:
                            value = local_max[row, i]
                            clean = T.if_then_else(
                                T.isnan(value), -T.infinity(T.float32), value
                            )
                            bits = T.reinterpret(clean, T.int32)
                            ordered = T.if_then_else(
                                clean == 0, 0, bits ^ ((bits >> 31) & 0x7FFFFFFF)
                            )
                            unsigned = T.reinterpret(ordered, T.uint32) ^ T.uint32(
                                0x80000000
                            )
                            keys[i] = (T.cast(unsigned, T.uint64) << 32) | T.cast(
                                padded - i, T.uint64
                            )
                    T.reduce_max(keys, top, dim=0)
                    if T.get_thread_binding() == 0:
                        winner_idx = padded - T.cast(
                            top[0] & T.uint64(0xFFFFFFFF), T.int32
                        )
                        sampled[row, count] = local_ids[row, winner_idx]
                else:
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
                    if T.get_thread_binding() == 0:
                        sampled[row, count] = local_ids[row, winner[0]]
            if threads != 32:
                T.sync_threads()
            if T.get_thread_binding() == 0:
                counts[row] = count + 1

    return insert


def insert_call(blocks: int, value_dtype: torch.dtype, index_dtype: torch.dtype):
    from vllm.models.deepseek_v4_1.tilelang.prepare import call

    return call(
        insert_kernel,
        blocks,
        "float64" if value_dtype == torch.float64 else "float32",
        "int64" if index_dtype == torch.int64 else "int32",
        _THREADS,
    )


def register_insert(blocks: int, use_fp64: bool) -> None:
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    if JitWarmupRegistry.active() is None:
        return
    from vllm.models.deepseek_v4_1.tilelang.prepare import declare

    dtype = torch.float64 if use_fp64 else torch.float32
    declare(lambda: [insert_call(blocks, dtype, d) for d in (torch.int32, torch.int64)])


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
        local_max.shape[1], local_max.dtype, expanded_idx_mapping.dtype
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
