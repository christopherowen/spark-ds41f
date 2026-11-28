# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The grammar bitmask in TileLang, the contract of
``structured_outputs._apply_grammar_bitmask_kernel`` bit for bit (masked tokens -inf,
the others untouched)."""

import torch

from vllm.tilelang_utils import T, tilelang

_PASS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}
_DTYPES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}
_BLOCK = 8192  # Triton's BLOCK_SIZE: tokens a block
_THREADS = _BLOCK // 32  # a bitmask word a thread to load


@tilelang.jit(pass_configs=_PASS)
def grammar_bitmask_kernel(logits_dtype: str, mask_stride: int, invalid: bool):
    """A block a bitmask row and 8192 tokens, as Triton's grid. The row's key is
    ``request * mask_stride + position`` (with ``invalid``, a negative key marks a
    rejected draft input: ``-key - 1``, and the first block sets that input id to -1);
    the logits row is the request's first plus the position, when the request has
    that many. The block's 256 bitmask words go through shared memory, so each thread
    masks tokens 256 apart and the stores coalesce."""
    rows, vocab, logits_stride = T.dynamic("rows, vocab, logits_stride")
    num_masks, cu_len, num_ids, num_logits = T.dynamic(
        "num_masks, cu_len, num_ids, num_logits"
    )
    words, bitmask_rows, bitmask_stride = T.dynamic(
        "words, bitmask_rows, bitmask_stride"
    )

    @T.prim_func
    def grammar_bitmask(
        logits: T.StridedTensor((rows, vocab), (logits_stride, 1), logits_dtype),
        keys: T.Tensor((num_masks,), T.int32),
        cu_num_logits: T.Tensor((cu_len,), T.int32),
        input_ids: T.Tensor((num_ids,), T.int32),
        input_logits_indices: T.Tensor((num_logits,), T.int64),
        bitmask: T.StridedTensor((bitmask_rows, words), (bitmask_stride, 1), T.int32),
    ):
        with T.Kernel(num_masks, T.ceildiv(vocab, _BLOCK), threads=_THREADS) as (
            m,
            block,
        ):
            tx = T.get_thread_binding()
            packed = T.alloc_shared((_THREADS,), T.int32)
            key = T.alloc_var(T.int32, init=keys[m])
            rejected = T.alloc_var(T.int32, init=0)
            if invalid:
                rejected = T.cast(key < 0, T.int32)
                key = T.if_then_else(key < 0, -key - 1, key)
            req = key // mask_stride
            position = key % mask_stride
            first = T.alloc_var(T.int32, init=cu_num_logits[req])
            count = T.alloc_var(T.int32, init=cu_num_logits[req + 1] - first)
            row = first + position
            word = block * _THREADS + tx
            packed[tx] = T.if_then_else(
                word < words, bitmask[m, T.min(word, words - 1)], 0
            )
            if invalid and (
                (block == 0) & (tx == 0) & (position < count) & (rejected != 0)
            ):
                input_ids[input_logits_indices[row]] = -1
            T.sync_threads()
            if position < count:
                for i in T.unroll(32):
                    token = block * _BLOCK + i * _THREADS + tx
                    bit = (packed[i * (_THREADS // 32) + tx // 32] >> (tx % 32)) & 1
                    if (bit == 0) & (token < vocab):
                        logits[T.cast(row, T.int64), token] = -T.infinity(logits_dtype)

    return grammar_bitmask


def grammar_bitmask_key(
    logits_dtype: torch.dtype, mask_stride: int, invalid: bool
) -> tuple:
    return (_DTYPES[logits_dtype], mask_stride, invalid)


def apply_grammar_bitmask(
    logits: torch.Tensor,
    keys: torch.Tensor,
    cu_num_logits: torch.Tensor,
    input_ids: torch.Tensor | None,
    input_logits_indices: torch.Tensor | None,
    bitmask: torch.Tensor,
    mask_stride: int,
) -> None:
    """``StructuredOutputsWorker.apply_grammar_bitmask``'s launch. Without
    ``input_ids`` (no invalid draft inputs to mark), ``keys`` and an empty int64
    tensor stand in for the unread input tensors (no conversion is launched)."""
    if keys.shape[0] == 0:
        return
    invalid = input_ids is not None
    grammar_bitmask_kernel(*grammar_bitmask_key(logits.dtype, mask_stride, invalid))(
        logits,
        keys,
        cu_num_logits,
        input_ids if invalid else keys,
        input_logits_indices
        if invalid
        else torch.empty(1, dtype=torch.int64, device=keys.device),
        bitmask,
    )
