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

A row is copied as 32-bit words, sixteen bytes a thread a piece, whatever the
table's dtype, so the output carries the table's bits exactly (NaN payloads, signed
zeros); rows must be sixteen-byte aligned and a multiple of sixteen bytes wide, as
DS4.1's 10240-byte BF16 rows are. Below ``_PREFILL_TOKENS`` tokens a token's row is
spread over ``ceil(width / (16 * threads))`` blocks (five for DS4.1 at 128 threads),
a piece a thread, so a decode step's few tokens still put every load of their rows
in flight at once; from it on a row is one block, every piece of a thread loaded
before any is stored, so a long prefill chunk is a block a token instead of five
blocks of one load each.
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
from vllm.model_executor.utils import replace_parameter
from vllm.models.deepseek_v4_1.tilelang import prepare

# 32-bit words in one sixteen-byte piece of a row.
_WORDS = 4
# Tokens from which a row is one block instead of a block a piece.
_PREFILL_TOKENS = 512
_THREADS = 128
_GROUP = 128
_ID_DTYPES = {torch.int32: "int32", torch.int64: "int64"}

_PACKED_EMBED_HEADER = r"""
#include <stdint.h>

template <int BASE_MINUS_ONE>
__device__ __forceinline__ void vllm_embed_decode_vec16(
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

template <int K, int BASE_MINUS_ONE, int THREADS>
__device__ __forceinline__ void vllm_packed_vocab_embedding_row(
    const uint8_t* values,
    const int* exception_entries,
    const int* row_offsets,
    void* __restrict__ out_ptr,
    int64_t row,
    int t,
    int tx) {
  constexpr int GROUP = 128;
  constexpr int HALF = 64;
  constexpr int TOTAL_SLICES = K / 32;
  constexpr int64_t ROW_BYTES = K + K / 2;
  constexpr int ITERS = (TOTAL_SLICES + THREADS - 1) / THREADS;
  uint16_t* __restrict__ out_row = reinterpret_cast<uint16_t*>(out_ptr) + (int64_t)t * K;
  if (row >= 0) {
    const int exc_start = row_offsets[row];
    const int exc_end = row_offsets[row + 1];
    const uint8_t* w_row = values + row * ROW_BYTES;
    uint4 out_a[ITERS][2];
    uint4 out_b[ITERS][2];
    #pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int s = it * THREADS + tx;
      if (s < TOTAL_SLICES) {
        const int g = s >> 2;
        const int c = (s & 3) << 4;
        vllm_embed_decode_vec16<BASE_MINUS_ONE>(
            w_row + g * GROUP + c,
            w_row + g * GROUP + HALF + c,
            w_row + K + g * HALF + c,
            out_a[it],
            out_b[it]);
      }
    }
    #pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int s = it * THREADS + tx;
      if (s < TOTAL_SLICES) {
        const int g = s >> 2;
        const int c = (s & 3) << 4;
        uint4* dst_a = reinterpret_cast<uint4*>(out_row + g * GROUP + c);
        uint4* dst_b = reinterpret_cast<uint4*>(out_row + g * GROUP + HALF + c);
        dst_a[0] = out_a[it][0];
        dst_a[1] = out_a[it][1];
        dst_b[0] = out_b[it][0];
        dst_b[1] = out_b[it][1];
      }
    }
    if (exc_start < exc_end) {
      __syncthreads();
      for (int idx = exc_start + tx; idx < exc_end; idx += THREADS) {
        const unsigned int entry = (unsigned int)exception_entries[idx];
        out_row[entry >> 16] = (uint16_t)(entry & 0xFFFFu);
      }
    }
  } else {
    const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
    #pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const int s = it * THREADS + tx;
      if (s < TOTAL_SLICES) {
        const int g = s >> 2;
        const int c = (s & 3) << 4;
        uint4* dst_a = reinterpret_cast<uint4*>(out_row + g * GROUP + c);
        uint4* dst_b = reinterpret_cast<uint4*>(out_row + g * GROUP + HALF + c);
        dst_a[0] = zero4;
        dst_a[1] = zero4;
        dst_b[0] = zero4;
        dst_b[1] = zero4;
      }
    }
  }
}
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # Rows are in the table by construction (the shard bounds come from the
        # layer); the CUDA kernel does not guard them either.
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def vocab_embedding_kernel(
    words: int,
    id_dtype: str,
    threads: int = _THREADS,
    coherent: bool = True,
    per_thread: int = 1,
):
    """``out[t] = table[row(ids[t])]`` or zeros, for rows of ``words`` 32-bit words,
    ``per_thread`` pieces a thread.

    ``coherent`` leaves the table and the ids unmarked (plain loads); False marks
    them restrict, as the CUDA kernel does (read-only loads). A benchmark variant.
    """
    assert words % _WORDS == 0 and threads % 32 == 0
    pieces = words // _WORDS
    blocks = -(-pieces // (threads * per_thread))
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
            tx = T.get_thread_binding()
            data = T.alloc_local((per_thread * _WORDS,), T.int32)
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
            for k in T.unroll(per_thread):
                piece = (bx * per_thread + k) * threads + tx
                if piece < pieces:
                    if row >= 0:
                        for i in T.vectorized(_WORDS):
                            data[k * _WORDS + i] = table[
                                row * words + piece * _WORDS + i
                            ]
                    else:
                        for i in T.vectorized(_WORDS):
                            data[k * _WORDS + i] = T.int32(0)
            for k in T.unroll(per_thread):
                piece = (bx * per_thread + k) * threads + tx
                if piece < pieces:
                    for i in T.vectorized(_WORDS):
                        out[t, piece * _WORDS + i] = data[k * _WORDS + i]

    return vocab_embedding


def _packed_threads(k: int) -> int:
    del k
    return _THREADS


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def packed_vocab_embedding_kernel(
    k: int,
    exponent_base: int,
    id_dtype: str,
    threads: int = _THREADS,
):
    """``out[t] = unpack(values[row(ids[t])])`` or zeros, for 12-bit packed BF16
    rows of ``k`` elements (`k` divisible by 128) with SWAR ``__byte_perm`` decode
    and exact out-of-window exception patching."""
    assert k > 0 and k % _GROUP == 0 and threads % 32 == 0
    row_bytes = k + k // 2
    base_minus_one = exponent_base - 1
    tokens = T.dynamic("tokens")
    rows = T.dynamic("rows")
    exceptions = T.dynamic("exceptions")
    offsets = T.dynamic("offsets")

    @T.prim_func
    def vocab_packed_embedding(
        ids: T.Tensor((tokens,), id_dtype),
        values: T.Tensor((rows, row_bytes), T.uint8),
        exception_entries: T.Tensor((exceptions,), T.int32),
        row_offsets: T.Tensor((offsets,), T.int32),
        out: T.Tensor((tokens, k), T.bfloat16),
        org_start: T.int64,
        org_end: T.int64,
        added_start: T.int64,
        added_end: T.int64,
        added_offset: T.int64,
    ):
        with T.Kernel(tokens, threads=threads) as (t,):
            T.import_source(_PACKED_EMBED_HEADER)
            T.annotate_restrict_buffers(ids, values, exception_entries, row_offsets)
            tx = T.get_thread_binding()
            token = T.alloc_var(T.int64, init=T.cast(ids[t], T.int64))
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
            T.call_extern(
                "handle",
                f"vllm_packed_vocab_embedding_row<{k}, {base_minus_one}, {threads}>",
                T.address_of(values[0, 0]),
                T.address_of(exception_entries[0]),
                T.address_of(row_offsets[0]),
                T.address_of(out[0, 0]),
                row,
                t,
                tx,
            )

    return vocab_packed_embedding


@cache
def _packed_kernel(k: int, exponent_base: int, id_dtype: str):
    return packed_vocab_embedding_kernel(
        k, exponent_base, id_dtype, threads=_packed_threads(k)
    )


def packed_kernel_calls(k: int, exponent_base: int) -> list[prepare.TileLangCall]:
    threads = _packed_threads(k)
    return [
        prepare.call(
            packed_vocab_embedding_kernel, k, exponent_base, d, threads=threads
        )
        for d in _ID_DTYPES.values()
    ]


def declare_packed(k: int, exponent_base: int) -> None:
    prepare.declare(
        lambda: packed_kernel_calls(k, exponent_base),
        bind=lambda: [
            _packed_kernel(k, exponent_base, d) for d in _ID_DTYPES.values()
        ],
    )


def vocab_embedding_packed(
    input_ids: torch.Tensor,
    values: torch.Tensor,
    exception_entries: torch.Tensor,
    row_offsets: torch.Tensor,
    in_features: int,
    exponent_base: int,
    out: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> None:
    """This rank's partial BF16 embeddings of ``input_ids`` ([T] int32 or int64)
    decoded directly from its 12-bit packed shard ``values`` into ``out`` ([T, K]
    BF16): bit-identical to ``vocab_embedding`` on the unpacked table."""
    tokens = input_ids.shape[0]
    row_bytes = in_features + in_features // 2
    if (
        input_ids.dim() != 1
        or input_ids.dtype not in _ID_DTYPES
        or values.dim() != 2
        or values.dtype != torch.uint8
        or values.shape[1] != row_bytes
        or out.shape != (tokens, in_features)
        or out.dtype != torch.bfloat16
        or not (input_ids.is_contiguous() and values.is_contiguous())
        or not out.is_contiguous()
    ):
        raise ValueError(
            "TileLang packed vocabulary embedding needs 1-D int32/int64 ids, a "
            "contiguous [rows, K + K // 2] uint8 packed table and a contiguous "
            f"[ids, K] bfloat16 output: ids {tuple(input_ids.shape)} "
            f"{input_ids.dtype}, table {tuple(values.shape)} {values.dtype}, "
            f"output {tuple(out.shape)} {out.dtype}"
        )
    if values.data_ptr() % 16 or out.data_ptr() % 16:
        raise ValueError(
            "TileLang packed vocabulary embedding needs 16-byte aligned rows"
        )
    if tokens == 0:
        return
    added_offset = (
        org_vocab_end_index
        - org_vocab_start_index
        + num_org_vocab_padding
        - added_vocab_start_index
    )
    _packed_kernel(in_features, exponent_base, _ID_DTYPES[input_ids.dtype])(
        input_ids,
        values,
        exception_entries,
        row_offsets,
        out,
        org_vocab_start_index,
        org_vocab_end_index,
        added_vocab_start_index,
        added_vocab_end_index,
        added_offset,
    )


def _per_thread(words: int, tokens: int) -> int:
    """Pieces a thread for a lookup of ``tokens`` rows of ``words`` words."""
    if tokens < _PREFILL_TOKENS:
        return 1
    return -(-(words // _WORDS) // _THREADS)


def _layouts(words: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys((1, -(-(words // _WORDS) // _THREADS))))


@cache
def _kernel(words: int, id_dtype: str, per_thread: int):
    return vocab_embedding_kernel(words, id_dtype, per_thread=per_thread)


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
    (the model runner's) and int64 ids, in both layouts."""
    return [
        prepare.call(vocab_embedding_kernel, words, d, per_thread=p)
        for d in _ID_DTYPES.values()
        for p in _layouts(words)
    ]


def declare(words: int) -> None:
    """Declare the lookups of rows of ``words`` words, and keep them."""
    prepare.declare(
        lambda: kernel_calls(words),
        bind=lambda: [
            _kernel(words, d, p) for d in _ID_DTYPES.values() for p in _layouts(words)
        ],
    )


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
    _kernel(words, _ID_DTYPES[input_ids.dtype], _per_thread(words, tokens))(
        input_ids,
        weight.view(torch.int32).view(-1),
        out.view(torch.int32),
        org_vocab_start_index,
        org_vocab_end_index,
        added_vocab_start_index,
        added_vocab_end_index,
        added_offset,
    )


class TileLangEmbeddingMethod(UnquantizedEmbeddingMethod):
    """The unquantized embedding's sharded weights and ties, with the tensor-parallel
    lookup in TileLang (``VocabParallelEmbedding``'s fused path).

    After loading, BF16 tables whose width is divisible by 128 are packed in-place
    into B12X/TileLang's exact 12-bit representation (75% of BF16 bytes) and decoded
    on the fly during lookup.
    """

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        weight = getattr(layer, "weight", None)
        if (
            weight is None
            or not weight.is_cuda
            or weight.dtype != torch.bfloat16
            or weight.ndim != 2
            or weight.shape[0] == 0
            or weight.shape[1] == 0
            or weight.shape[1] % _GROUP != 0
        ):
            return
        from .vocab import pack_bf16_weight

        data = weight.data
        in_features = int(data.shape[1])
        from vllm.v1.worker import display_carveout

        packed = pack_bf16_weight(data, in_place=True)
        del data
        replace_parameter(layer, "weight", display_carveout.compact(packed.values))
        replace_parameter(layer, "weight_exception_rows", packed.exception_rows)
        replace_parameter(layer, "weight_exception_entries", packed.exception_entries)
        replace_parameter(layer, "weight_row_offsets", packed.row_offsets)
        layer.packed_bf16_in_features = in_features
        layer.packed_bf16_exponent_base = packed.exponent_base
        layer.packed_bf16_exceptions = packed.exceptions
        layer._already_called_process_weights_after_loading = True

    def declare_layer_kernels(self, layer: nn.Module) -> None:
        if hasattr(layer, "packed_bf16_exponent_base"):
            declare_packed(
                layer.packed_bf16_in_features, layer.packed_bf16_exponent_base
            )
            return
        declare(row_words(layer.weight))

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if not hasattr(layer, "packed_bf16_exponent_base"):
            return super().embedding(layer, input_)
        flat = input_ if input_.ndim == 1 else input_.reshape(-1)
        if not flat.is_contiguous():
            flat = flat.contiguous()
        in_features = layer.packed_bf16_in_features
        out = torch.empty(
            (flat.shape[0], in_features),
            dtype=torch.bfloat16,
            device=layer.weight.device,
        )
        shard = layer.shard_indices
        vocab_embedding_packed(
            flat,
            layer.weight,
            layer.weight_exception_entries,
            layer.weight_row_offsets,
            in_features,
            layer.packed_bf16_exponent_base,
            out,
            shard.org_vocab_start_index,
            shard.org_vocab_end_index,
            shard.num_org_vocab_padding,
            shard.added_vocab_start_index,
            shard.added_vocab_end_index,
        )
        return out if input_.ndim == 1 else out.view(*input_.shape, in_features)

    def parallel_embedding(
        self,
        layer: nn.Module,
        input_ids: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shard = layer.shard_indices
        if hasattr(layer, "packed_bf16_exponent_base"):
            in_features = layer.packed_bf16_in_features
            if out is None:
                out = torch.empty(
                    (input_ids.shape[0], in_features),
                    dtype=torch.bfloat16,
                    device=layer.weight.device,
                )
            vocab_embedding_packed(
                input_ids,
                layer.weight,
                layer.weight_exception_entries,
                layer.weight_row_offsets,
                in_features,
                layer.packed_bf16_exponent_base,
                out,
                shard.org_vocab_start_index,
                shard.org_vocab_end_index,
                shard.num_org_vocab_padding,
                shard.added_vocab_start_index,
                shard.added_vocab_end_index,
            )
            return out
        if out is None:
            out = torch.empty(
                (input_ids.shape[0], layer.weight.shape[1]),
                dtype=layer.weight.dtype,
                device=layer.weight.device,
            )
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
    "declare_packed",
    "kernel_calls",
    "packed_kernel_calls",
    "packed_vocab_embedding_kernel",
    "row_words",
    "vocab_embedding",
    "vocab_embedding_kernel",
    "vocab_embedding_packed",
]
