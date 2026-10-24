# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Commit counts after sampling or speculative verification, one lane per request."""

import tilelang
import tilelang.language as T
import torch

_PASS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
_THREADS = 128


@tilelang.jit(pass_configs=_PASS)
def counts_kernel(dead_rows: bool, index_dtype: str, threads: int):
    requests = T.dynamic("requests")
    sampled_len = T.dynamic("sampled_len")
    seq_len = T.dynamic("seq_len")
    cu_len = T.dynamic("cu_len")
    state_len = T.dynamic("state_len")
    cut_len = T.dynamic("cut_len")

    @T.prim_func
    def token_counts(
        sampled: T.Tensor((sampled_len,), T.int32),
        rejected: T.Tensor((sampled_len,), T.int32),
        seq_lens: T.Tensor((seq_len,), T.int32),
        cu_num_logits: T.Tensor((cu_len,), T.int32),
        idx_mapping: T.Tensor((requests,), index_dtype),
        prefill_len: T.Tensor((state_len,), T.int32),
        dead_cut: T.Tensor((cut_len,), T.int32),
    ):
        with T.Kernel(T.ceildiv(requests, threads), threads=threads) as block:
            row = block * threads + T.get_thread_binding()
            if row < requests:
                req = idx_mapping[row]
                chunked = seq_lens[row] < prefill_len[req]
                count = sampled[row]
                if dead_rows:
                    count = T.min(count, dead_cut[row] + 1)
                committed = T.if_then_else(chunked, 0, count)
                sampled[row] = committed
                rejected[row] = T.if_then_else(
                    chunked, 0, cu_num_logits[row + 1] - cu_num_logits[row] - committed
                )

    return token_counts


def counts_call(dead_rows: bool, index_dtype: torch.dtype):
    from vllm.models.deepseek_v4_1.tilelang.prepare import call

    return call(
        counts_kernel,
        dead_rows,
        "int64" if index_dtype == torch.int64 else "int32",
        _THREADS,
    )


def register_counts(dead_rows: bool) -> None:
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    if JitWarmupRegistry.active() is None:
        return
    from vllm.models.deepseek_v4_1.tilelang.prepare import declare

    declare(
        lambda: [counts_call(dead_rows, dtype) for dtype in (torch.int32, torch.int64)]
    )


def count_tokens(
    sampled: torch.Tensor,
    rejected: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
    dead_cut: torch.Tensor | None,
) -> None:
    if idx_mapping.numel() == 0:
        return
    _, args, kwargs = counts_call(dead_cut is not None, idx_mapping.dtype)
    counts_kernel(*args, **kwargs)(
        sampled,
        rejected,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
        dead_cut if dead_cut is not None else sampled,
    )
