# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The vocabulary-parallel embedding lookup in TileLang.

Each rank of the tensor-parallel group holds a shard of the token table; its partial
embeddings are the shard's row for every id it owns and zeros for every other id, so
the group's all-reduce (or prefill SP's reduce-scatter) adds exactly one row to zeros.
vLLM's CUDA kernel (``csrc/libtorch_stable/vocab_parallel_embedding_kernels.cu``)
does this in one pass; this kernel does the same thing, for the same arguments:

- an id in ``[org_start, org_end)`` reads shard row ``id - org_start``;
- an id in ``[added_start, added_end)`` (LoRA's added vocabulary, empty for
  DeepSeek-V4.1) reads row ``id + added_offset``;
- any other id, another rank's or none at all (negative, padding), writes zeros.

A row is copied as 32-bit words, sixteen bytes a thread, whatever the table's dtype,
so the output carries the table's bits exactly (NaN payloads, signed zeros); rows
must be sixteen-byte aligned and a multiple of sixteen bytes wide, as DS4.1's
10240-byte BF16 rows are. A token's row is spread over ``ceil(width / (16 *
threads))`` blocks (five for DS4.1 at 128 threads), one block per piece of the row,
so a decode step's few tokens still put every load of their rows in flight at once.
The table and the ids are not marked restrict: their loads take the coherent path,
which on GB10 is faster for a load whose address depends on the load before it
(measured for CED's row copies, ``ced.py``).
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
from functools import cache

import tilelang
import tilelang.language as T
import torch
from torch import nn

from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.models.deepseek_v4_1.tilelang import prepare

# 32-bit words in one sixteen-byte piece of a row.
_WORDS = 4
_ID_DTYPES = {torch.int32: "int32", torch.int64: "int64"}


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # Rows are in the table by construction (the shard bounds come from the
        # layer); the CUDA kernel does not guard them either.
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def vocab_embedding_kernel(
    words: int, id_dtype: str, threads: int = 128, coherent: bool = True
):
    """``out[t] = table[row(ids[t])]`` or zeros, for rows of ``words`` 32-bit words.

    ``coherent`` leaves the table and the ids unmarked (plain loads); False marks
    them restrict, as the CUDA kernel does (read-only loads). A benchmark variant.
    """
    assert words % _WORDS == 0 and threads % 32 == 0
    pieces = words // _WORDS
    blocks = -(-pieces // threads)
    tokens = T.dynamic("tokens")
    span = T.dynamic("span")

    @T.prim_func
    def vocab_embedding(
        ids: T.Tensor((tokens,), id_dtype),
        table: T.Tensor((span,), T.int32),
        out: T.Tensor((tokens, words), T.int32),
        org_start: T.int64,
        org_end: T.int64,
        added_start: T.int64,
        added_end: T.int64,
        added_offset: T.int64,
    ):
        with T.Kernel(tokens, blocks, threads=threads) as (t, bx):
            if coherent:
                T.annotate_restrict_buffers(ids, table)
            piece = bx * threads + T.get_thread_binding()
            data = T.alloc_local((_WORDS,), T.int32)
            token = T.alloc_var(T.int64, init=T.cast(ids[t], T.int64))
            # The shard row, or -1 for an id this rank does not own.
            row = T.alloc_var(
                T.int64,
                init=T.if_then_else(
                    (token >= org_start) & (token < org_end),
                    token - org_start,
                    T.if_then_else(
                        (token >= added_start) & (token < added_end),
                        token + added_offset,
                        T.int64(-1),
                    ),
                ),
            )
            if piece < pieces:
                if row >= 0:
                    for i in T.vectorized(_WORDS):
                        data[i] = table[row * words + piece * _WORDS + i]
                else:
                    for i in T.vectorized(_WORDS):
                        data[i] = T.int32(0)
                for i in T.vectorized(_WORDS):
                    out[t, piece * _WORDS + i] = data[i]

    return vocab_embedding


@cache
def _kernel(words: int, id_dtype: str):
    return vocab_embedding_kernel(words, id_dtype)


def row_words(weight: torch.Tensor) -> int:
    """32-bit words in one row of ``weight``; rows must be whole sixteen-byte pieces."""
    row_bytes = weight.shape[1] * weight.element_size()
    if row_bytes % (4 * _WORDS):
        raise ValueError(
            f"TileLang vocabulary embedding needs rows of whole 16-byte pieces, got "
            f"{row_bytes} bytes"
        )
    return row_bytes // 4


def kernel_calls(words: int) -> list[prepare.TileLangCall]:
    """The calls ``vocab_embedding`` makes for rows of ``words`` words: int32 ids
    (the model runner's) and int64 ids."""
    return [prepare.call(vocab_embedding_kernel, words, d) for d in _ID_DTYPES.values()]


def declare(words: int) -> None:
    """Declare the lookups of rows of ``words`` words, and keep them."""
    prepare.declare(
        lambda: kernel_calls(words),
        bind=lambda: [_kernel(words, d) for d in _ID_DTYPES.values()],
    )


@torch.library.custom_op("vllm::dsv41_tilelang_vocab_embedding", mutates_args=("out",))
def vocab_embedding(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> None:
    """This rank's partial embeddings of ``input_ids`` ([T] int32 or int64) from its
    shard ``weight`` into ``out`` ([T, width], the weight's dtype): the arguments of
    ``ops.vocab_parallel_embedding``, the same result bit for bit."""
    tokens = input_ids.shape[0]
    if (
        input_ids.dim() != 1
        or input_ids.dtype not in _ID_DTYPES
        or weight.dim() != 2
        or out.shape != (tokens, weight.shape[1])
        or out.dtype != weight.dtype
        or not (input_ids.is_contiguous() and weight.is_contiguous())
        or not out.is_contiguous()
    ):
        raise ValueError(
            "TileLang vocabulary embedding needs 1-D int32/int64 ids, a contiguous "
            f"2-D table and a contiguous [ids, width] output of its dtype: ids "
            f"{tuple(input_ids.shape)} {input_ids.dtype}, table "
            f"{tuple(weight.shape)} {weight.dtype}, output {tuple(out.shape)} "
            f"{out.dtype}"
        )
    words = row_words(weight)
    if weight.data_ptr() % 16 or out.data_ptr() % 16:
        raise ValueError("TileLang vocabulary embedding needs 16-byte aligned rows")
    if tokens == 0:
        return
    # Maps an added-vocabulary id onto its shard row, as get_masked_input_and_mask().
    added_offset = (
        org_vocab_end_index
        - org_vocab_start_index
        + num_org_vocab_padding
        - added_vocab_start_index
    )
    _kernel(words, _ID_DTYPES[input_ids.dtype])(
        input_ids,
        weight.view(torch.int32).view(-1),
        out.view(torch.int32),
        org_vocab_start_index,
        org_vocab_end_index,
        added_vocab_start_index,
        added_vocab_end_index,
        added_offset,
    )


@vocab_embedding.register_fake
def _vocab_embedding_fake(
    input_ids,
    weight,
    out,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    return None


class TileLangEmbeddingMethod(UnquantizedEmbeddingMethod):
    """The unquantized embedding's sharded weights and ties, with the tensor-parallel
    lookup in TileLang (``VocabParallelEmbedding``'s fused path)."""

    def declare_layer_kernels(self, layer: nn.Module) -> None:
        declare(row_words(layer.weight))

    def parallel_embedding(
        self,
        layer: nn.Module,
        input_ids: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty(
                (input_ids.shape[0], layer.weight.shape[1]),
                dtype=layer.weight.dtype,
                device=layer.weight.device,
            )
        shard = layer.shard_indices
        vocab_embedding(
            input_ids,
            layer.weight,
            out,
            shard.org_vocab_start_index,
            shard.org_vocab_end_index,
            shard.num_org_vocab_padding,
            shard.added_vocab_start_index,
            shard.added_vocab_end_index,
        )
        return out


__all__ = [
    "TileLangEmbeddingMethod",
    "declare",
    "kernel_calls",
    "row_words",
    "vocab_embedding",
    "vocab_embedding_kernel",
]
