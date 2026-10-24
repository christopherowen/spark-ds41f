# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selected-token ranks, with the V2 runner's Triton rank semantics."""

from collections.abc import Sequence

import tilelang
import tilelang.language as T
import torch

_PASS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}
_DTYPES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int32: "int32",
    torch.int64: "int64",
}
_BLOCK = 8192


@tilelang.jit(pass_configs=_PASS)
def ranks_kernel(
    vocab: int, stride: int, dtype: str, id_dtype: str, threads: int = 512
):
    rows = T.symbolic("rows")

    @T.prim_func
    def token_ranks(
        logits: T.StridedTensor((rows, vocab), (stride, 1), dtype),
        token_ids: T.Tensor((rows,), id_dtype),
        output: T.Tensor((rows,), "int64"),
    ):
        with T.Kernel(rows, threads=threads) as row:
            counts = T.alloc_fragment((_BLOCK,), "int32")
            total = T.alloc_fragment((1,), "int32")
            selected = T.alloc_local((1,), "float32")
            selected[0] = T.cast(
                logits[T.cast(row, "int64"), token_ids[row]], "float32"
            )
            T.clear(counts)
            for block in T.serial(T.ceildiv(vocab, _BLOCK)):
                for col in T.Parallel(_BLOCK):
                    index = block * _BLOCK + col
                    value = T.if_then_else(
                        index < vocab,
                        T.cast(logits[T.cast(row, "int64"), index], "float32"),
                        -T.infinity("float32"),
                    )
                    # Triton counts its -inf padding when the selected logit
                    # is -inf; keep that behaviour as well as ties and NaNs.
                    counts[col] += T.cast(value >= selected[0], "int32")
            T.reduce_sum(counts, total, dim=0)
            output[row] = T.cast(total[0], "int64")

    return token_ranks


def rank_call(dtype: torch.dtype, stride: int, vocab: int, id_dtype: torch.dtype):
    from vllm.models.deepseek_v4_1.tilelang.prepare import call

    return call(ranks_kernel, vocab, stride, _DTYPES[dtype], _DTYPES[id_dtype])


def register_ranks(logits: Sequence[tuple[torch.dtype, int]], vocab_size: int) -> None:
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    if JitWarmupRegistry.active() is None:
        return
    from vllm.models.deepseek_v4_1.tilelang.prepare import declare

    declare(
        lambda: [
            rank_call(dtype, stride, vocab_size, id_dtype)
            for dtype, stride in logits
            for id_dtype in (torch.int32, torch.int64)
        ]
    )


def token_ranks(
    logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
) -> None:
    if logits.shape[0] == 0:
        return
    _, args, kwargs = rank_call(
        logits.dtype, logits.stride(0), logits.shape[1], token_ids.dtype
    )
    ranks_kernel(*args, **kwargs)(logits, token_ids, output)
