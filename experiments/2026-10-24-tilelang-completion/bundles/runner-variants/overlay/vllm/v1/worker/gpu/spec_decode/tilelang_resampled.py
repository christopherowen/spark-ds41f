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


@T.macro
def _take_better(best, best_index, best_id, value, index, value_id, valid):
    """Take ``(value, index, value_id)`` if it is valid and larger, or equal with
    a lower index."""
    take = T.alloc_var(
        T.int32,
        init=T.cast(
            valid
            & ((value > best[0]) | ((value == best[0]) & (index < best_index[0]))),
            T.int32,
        ),
    )
    best_id[0] = T.if_then_else(take != 0, value_id, best_id[0])
    best_index[0] = T.if_then_else(take != 0, index, best_index[0])
    best[0] = T.if_then_else(take != 0, value, best[0])


@T.macro
def _exchange(best, best_index, best_id, distance):
    """Combine with the lane at XOR ``distance``."""
    _take_better(
        best, best_index, best_id,
        T.shfl_xor(best[0], distance), T.shfl_xor(best_index[0], distance),
        T.shfl_xor(best_id[0], distance), T.bool(True),
    )  # fmt: skip


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
    per = (blocks + 31) // 32
    if blocks > 4 and threads != 32:
        raise ValueError("more than four blocks take a warp a row")

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
            thread = T.get_thread_binding()
            if blocks <= 4:
                if thread == 0:
                    count = counts[row]
                    counts[row] = count + 1
                    position = cu[row] + count
                    state = mapping[position]
                    if (
                        T.cast(temperature[state] != 0, T.int32)
                        | T.cast(position == cu[row + 1] - 1, T.int32)
                    ) != 0:
                        best = T.alloc_var(value_dtype, init=-T.infinity(value_dtype))
                        winner = T.alloc_var(T.int32, init=0)
                        for i in T.unroll(blocks):
                            value = local_max[row, i]
                            if value > best:
                                best = value
                                winner = i
                        sampled[row, count] = local_ids[row, winner]
            else:
                # A warp a row. Every block's maximum and id is loaded first (they
                # depend only on the row), then the count, the request's state and
                # its temperature; the reduction carries each candidate's id, so no
                # load waits on the winner. As before: NaN counts as -inf, and the
                # lowest block index wins a tie.
                values = T.alloc_local((per,), value_dtype)
                ids = T.alloc_local((per,), T.int64)
                for k in T.unroll(per):
                    i = T.min(k * 32 + thread, blocks - 1)
                    values[k] = local_max[row, i]
                    ids[k] = local_ids[row, i]
                first = T.alloc_var(T.int32, init=cu[row])
                last = T.alloc_var(T.int32, init=cu[row + 1] - 1)
                count_local = T.alloc_var(T.int32, init=0)
                if thread == 0:
                    count_local = counts[row]
                    counts[row] = count_local + 1
                count = T.shfl_sync(count_local, 0)
                position = first + count
                state = mapping[position]
                if (
                    T.cast(temperature[state] != 0, T.int32)
                    | T.cast(position == last, T.int32)
                ) != 0:
                    best = T.alloc_local((1,), value_dtype)
                    best_index = T.alloc_local((1,), T.int32)
                    best_id = T.alloc_local((1,), T.int64)
                    best[0] = -T.infinity(value_dtype)
                    best_index[0] = padded
                    best_id[0] = 0
                    for k in T.unroll(per):
                        _take_better(
                            best, best_index, best_id,
                            T.if_then_else(
                                T.isnan(values[k]), -T.infinity(value_dtype), values[k]
                            ),
                            k * 32 + thread, ids[k], k * 32 + thread < blocks,
                        )  # fmt: skip
                    _exchange(best, best_index, best_id, 16)
                    _exchange(best, best_index, best_id, 8)
                    _exchange(best, best_index, best_id, 4)
                    _exchange(best, best_index, best_id, 2)
                    _exchange(best, best_index, best_id, 1)
                    if thread == 0:
                        sampled[row, count] = best_id[0]

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
