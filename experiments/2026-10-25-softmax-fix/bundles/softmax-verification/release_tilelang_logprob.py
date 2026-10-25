# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The selected tokens' ranks and log probabilities over the real vocabulary.

- ``token_ranks``: each row's selected-token rank, ties included
  (``logprob._ranks_kernel`` without its phantom padding columns)
- ``topk_log_softmax``: ``logprob._topk_log_softmax_kernel``, bit for bit, NaN and
  infinite logits included (the Triton kernel stays importable as the tests'
  reference)

``topk_log_softmax``'s arithmetic is Triton 3.7.1's for sm_121a, read from the PTX
and SASS of the image's cached specializations (FP32: cache entry
KYOO2UC5JP6EBVTSY75OOJE5ZXFRIGKV6WFOJYRUR4SAK4V3UBDQ, bf16:
I3S7WK7AC43HFSJJKZZRSDS4GPL6K6SRZ4CZBOPEA2EPC5RXPGNQ, both with 16-byte logits and
a row stride of 16 elements):

- ``max_val`` is ``max.f32`` (a NaN gives way) of every logit and -inf, so any
  order gives it: here each thread folds its columns of every block, the row read
  last block first, and the block reduces once (``_block_reduce``). Triton reduces
  each 1024-column block in turn, three barriers a block. (The PTX ISA orders -0
  below +0; were the sign of a zero maximum to depend on the order, it could not
  reach an output: two zero maxima add 2 to ``se``, so ``lse`` is not 0.)
- ``se`` is the blocks' sums of ``tl.exp(logit - max_val)`` added to 0 in block
  order. ``tl.exp`` is ``ex2.approx.f32`` (subnormals kept) of the product with
  log2(e) (``vllm_exp_from``); a column past the row adds 0. A block's sum is
  Triton's: a thread adds its eight exponentials in register order (``_element``:
  Triton 3.7.1 lays the block out by the load's specialization, ``_triton_run``),
  the warp adds at XOR distances 16 to 1, and the warps' sums ``w`` combine as
  threads 0-3 do, ``(w0 + w2) + (w1 + w3)``. Here a chunk of eight blocks is
  loaded at once and its sums cross the warps through one barrier.
- ``lse`` is libdevice's ``__nv_logf`` as Triton bundles it
  (``vllm_control_log``), and a token's log probability ``(logit - max_val) -
  lse``, each rounded.

Every add, product and difference is the header's, rounded to nearest as Triton
writes it (nvcc would contract them otherwise).
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
from collections.abc import Sequence

import tilelang
import tilelang.language as T
import torch

from vllm.v1.worker.gpu.sample import tilelang_sampling
from vllm.v1.worker.gpu.sample.tilelang_sampling import (
    _THREADS,
    _call,
    _element,
    _kernel,
    _runs,
    _triton_run,
    declare,
)
from vllm.v1.worker.gpu.spec_decode.tilelang_rejection import (
    _HEADER,
    _block_reduce,
)

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
def ranks_kernel(vocab: int, stride: int, dtype: str, id_dtype: str, threads: int):
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
                    counts[col] += T.cast(
                        (index < vocab) & (value >= selected[0]), "int32"
                    )
            T.reduce_sum(counts, total, dim=0)
            output[row] = T.cast(total[0], "int64")

    return token_ranks


def rank_call(dtype: torch.dtype, stride: int, vocab: int, id_dtype: torch.dtype):
    from vllm.models.deepseek_v4_1.tilelang.prepare import call

    return call(
        ranks_kernel,
        vocab,
        stride,
        _DTYPES[dtype],
        _DTYPES[id_dtype],
        rank_threads(dtype),
    )


def rank_threads(dtype: torch.dtype) -> int:
    return 256 if dtype == torch.float32 else 512


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


# --- Log probabilities of the selected tokens -----------------------------------

# logprob._topk_log_softmax_kernel's BLOCK_SIZE and TOPK_BLOCK_SIZE, at Triton's
# default four warps (_THREADS).
SOFTMAX_BLOCK = 1024
_SOFTMAX_PER_THREAD = SOFTMAX_BLOCK // _THREADS
# Vocabulary blocks a thread loads before it reduces them: a chunk.
_SOFTMAX_UNROLL = 8
_SOFTMAX_CHUNK = SOFTMAX_BLOCK * _SOFTMAX_UNROLL


def _softmax_column(block, j: int, tx, run: int):
    """The column of thread ``tx``'s ``j``-th element of vocabulary block ``block``
    in Triton's layout (``_element``)."""
    return block * SOFTMAX_BLOCK + _element(j, tx, run)


def _masked(value, column, vocab, guarded: bool, other):
    """``value``, or ``other`` where a guarded block's ``column`` is past the row."""
    if guarded:
        return T.if_then_else(column < vocab, value, other)
    return value


@tilelang.jit(pass_configs=tilelang_sampling._PASS)
def topk_log_softmax_kernel(logits_dtype: str, run: int, masked: bool = False):
    """``_topk_log_softmax_kernel`` (BLOCK_SIZE and TOPK_BLOCK_SIZE 1024, four
    warps): one block of 128 threads a row, whose ``topk`` selected tokens' log
    probabilities it writes; with ``masked``, -inf where ``valid`` is false (the
    custom-ids path's ``masked_fill``, in the same store; unmasked, ``valid`` is
    not read). A thread holds its eight columns of each 1024-column
    block in runs of ``run`` (``_triton_run``); chunks of eight whole blocks load
    their runs as vectors (the launch checked the alignment), and the row's last
    chunk loads each column clamped to the row, the columns past it masked."""
    rows, vocab, topk = T.dynamic("rows, vocab, topk")
    logits_stride = T.dynamic("logits_stride", "int64")
    valid_rows, valid_cols = T.dynamic("valid_rows, valid_cols")
    per = _SOFTMAX_PER_THREAD
    unroll = _SOFTMAX_UNROLL

    @T.macro
    def load_blocks(logits, raw, row, block0, tx, guarded):
        """The thread's columns of blocks ``block0`` to ``block0 + unroll - 1``,
        block ``u``'s at ``raw[u * per:]`` in register order."""
        for u in tuple(range(unroll)):
            if guarded:
                for j in tuple(range(per)):
                    raw[u * per + j] = logits[
                        row, T.min(_softmax_column(block0 + u, j, tx, run), vocab - 1)
                    ]
            elif run == 1:
                for j in tuple(range(per)):
                    raw[u * per + j] = logits[
                        row, _softmax_column(block0 + u, j, tx, run)
                    ]
            else:
                for r in tuple(range(per // run)):
                    col0 = _softmax_column(block0 + u, r * run, tx, run)
                    for k in T.vectorized(run):
                        raw[u * per + r * run + k] = logits[row, col0 + k]

    @T.macro
    def max_blocks(raw, peak, best, block0, tx, guarded):
        """Fold the loaded blocks into the thread's maximum ``best[0]``; a column
        past the row is -inf."""
        for u in tuple(range(unroll)):
            for j in tuple(range(per)):
                value = _masked(
                    T.cast(raw[u * per + j], T.float32),
                    _softmax_column(block0 + u, j, tx, run),
                    vocab,
                    guarded,
                    -T.infinity(T.float32),
                )
                if j == 0:
                    peak[u] = value
                else:
                    peak[u] = T.call_extern("float32", "vllm_max", peak[u], value)
        for u in tuple(range(unroll)):
            best[0] = T.call_extern("float32", "vllm_max", best[0], peak[u])

    @T.macro
    def sum_blocks(raw, total, sums, se, maximum, block0, buf, blocks, tx, guarded):
        """Add the loaded blocks' sums of exponentials to ``se[0]`` in block order,
        each block's summed as Triton sums it (``_block_reduce`` with ``vllm_add``,
        the blocks' warp sums crossing in one barrier through ``sums[buf]``); a
        column past the row adds 0, a block past it nothing."""
        for u in tuple(range(unroll)):
            for j in tuple(range(per)):
                term = _masked(
                    T.call_extern(
                        "float32",
                        "vllm_exp_from",
                        T.cast(raw[u * per + j], T.float32),
                        maximum,
                    ),
                    _softmax_column(block0 + u, j, tx, run),
                    vocab,
                    guarded,
                    T.float32(0),
                )
                if j == 0:
                    total[u] = term
                else:
                    total[u] = T.call_extern("float32", "vllm_add", total[u], term)
        for distance in (16, 8, 4, 2, 1):
            for u in tuple(range(unroll)):
                total[u] = T.call_extern(
                    "float32", "vllm_add", total[u], T.shfl_xor(total[u], distance)
                )
        if tx % 32 == 0:
            for u in tuple(range(unroll)):
                sums[buf, u, tx // 32] = total[u]
        T.sync_threads()
        # Threads 0-3 add the warps' sums at XOR distances 2 and 1; every thread
        # takes thread 0's.
        for u in tuple(range(unroll)):
            block_sum = T.call_extern(
                "float32",
                "vllm_add",
                T.call_extern("float32", "vllm_add", sums[buf, u, 0], sums[buf, u, 2]),
                T.call_extern("float32", "vllm_add", sums[buf, u, 1], sums[buf, u, 3]),
            )
            if guarded:
                se[0] = T.if_then_else(
                    block0 + u < blocks,
                    T.call_extern("float32", "vllm_add", se[0], block_sum),
                    se[0],
                )
            else:
                se[0] = T.call_extern("float32", "vllm_add", se[0], block_sum)

    @T.prim_func
    def topk_log_softmax(
        output: T.Tensor((rows, topk), T.float32),
        logits: T.StridedTensor((rows, vocab), (logits_stride, 1), logits_dtype),
        token_ids: T.Tensor((rows, topk), T.int64),
        valid: T.Tensor((valid_rows, valid_cols), T.bool),
    ):
        with T.Kernel(rows, threads=_THREADS) as row:
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            raw = T.alloc_local((unroll * per,), logits_dtype)
            peak = T.alloc_local((unroll,), T.float32)
            total = T.alloc_local((unroll,), T.float32)
            best = T.alloc_local((1,), T.float32)
            se = T.alloc_local((1,), T.float32)
            tokens = T.alloc_local((per,), T.int64)
            picked = T.alloc_local((per,), T.float32)
            partial = T.alloc_shared((_THREADS // 32,), T.float32)
            row_max = T.alloc_shared((1,), T.float32)
            # Two chunks' warp sums: a chunk writes one while the last is read.
            sums = T.alloc_shared((2, unroll, _THREADS // 32), T.float32)
            if run > 1:
                # The launch checked it: a 16-byte base and a stride of 16 elements.
                T.assume(logits_stride % 16 == 0)

            blocks = T.ceildiv(vocab, SOFTMAX_BLOCK)
            # Whole chunks; the rest of the row (up to eight blocks, the last of
            # them partial) is the guarded chunk.
            full = vocab // _SOFTMAX_CHUNK

            # max_val, the row read last block first: the sum then starts on the
            # blocks most recently read.
            best[0] = -T.infinity(T.float32)
            if full * unroll < blocks:
                load_blocks(logits, raw, row, full * unroll, tx, True)
                max_blocks(raw, peak, best, full * unroll, tx, True)
            for step in T.serial(full):
                load_blocks(logits, raw, row, (full - 1 - step) * unroll, tx, False)
                max_blocks(raw, peak, best, (full - 1 - step) * unroll, tx, False)
            _block_reduce(best, partial, tx, "vllm_max")
            if tx == 0:
                row_max[0] = best[0]
            T.sync_threads()
            maximum = T.alloc_var(T.float32, init=row_max[0])

            # se, in block order.
            se[0] = T.float32(0)
            for chunk in T.serial(full):
                load_blocks(logits, raw, row, chunk * unroll, tx, False)
                sum_blocks(raw, total, sums, se, maximum, chunk * unroll, chunk % 2,
                           blocks, tx, False)  # fmt: skip
            if full * unroll < blocks:
                load_blocks(logits, raw, row, full * unroll, tx, True)
                sum_blocks(raw, total, sums, se, maximum, full * unroll, full % 2,
                           blocks, tx, True)  # fmt: skip
            lse = T.alloc_var(
                T.float32, init=T.call_extern("float32", "vllm_control_log", se[0])
            )

            # The selected tokens, 1024 at a time: the ids (clamped to the last),
            # then their logits, then the stores of the requested ones. Row offsets
            # are 64-bit, as Triton's req_idx * topk.
            row64 = T.cast(row, T.int64)
            for c in T.serial(T.ceildiv(topk, SOFTMAX_BLOCK)):
                for u in tuple(range(per)):
                    tokens[u] = token_ids[
                        row64, T.min(c * SOFTMAX_BLOCK + u * _THREADS + tx, topk - 1)
                    ]
                for u in tuple(range(per)):
                    picked[u] = T.cast(logits[row, tokens[u]], T.float32)
                for u in tuple(range(per)):
                    if c * SOFTMAX_BLOCK + u * _THREADS + tx < topk:
                        col = c * SOFTMAX_BLOCK + u * _THREADS + tx
                        value = T.call_extern(
                            "float32",
                            "vllm_sub",
                            T.call_extern("float32", "vllm_sub", picked[u], maximum),
                            lse,
                        )
                        if masked:
                            value = T.if_then_else(
                                valid[row64, col], value, -T.infinity(T.float32)
                            )
                        output[row64, col] = value

    return topk_log_softmax


def topk_log_softmax_key(
    logits_dtype: torch.dtype, run: int = 1, masked: bool = False
) -> tuple:
    return (_DTYPES[logits_dtype], run, masked)


def log_softmax_calls(logits_dtypes: Sequence[torch.dtype]) -> list:
    """``topk_log_softmax``'s calls for logits of each of ``logits_dtypes``, in both
    layouts a launch may take (``_runs``), unmasked and masked (custom ids)."""
    return [
        _call(topk_log_softmax_kernel, *topk_log_softmax_key(dtype, run, masked))
        for dtype in dict.fromkeys(logits_dtypes)
        for run in _runs(dtype)
        for masked in (False, True)
    ]


def register_log_softmax(logits: Sequence[tuple[torch.dtype, int]]) -> None:
    """Declare ``log_softmax_calls`` for logits given as ``(dtype, row stride)``
    pairs with the runner's JIT registry (a no-op where none collects). The stride
    is the launch's argument, not part of the key."""
    dtypes = tuple(dtype for dtype, _ in logits)
    declare(lambda: log_softmax_calls(dtypes))


def topk_log_softmax(
    logits: torch.Tensor,  # [rows, vocab_size]
    token_ids: torch.Tensor,  # [rows, topk], int64, contiguous
    output: torch.Tensor,  # [rows, topk], FP32, contiguous
    valid: torch.Tensor | None = None,  # [rows, topk], bool, contiguous
) -> None:
    """``_topk_log_softmax_kernel``'s launch: each row's log-softmax at its
    ``token_ids``, into ``output``."""
    if logits.shape[0] == 0 or token_ids.shape[1] == 0:
        return
    kernel = _kernel(
        topk_log_softmax_kernel,
        *topk_log_softmax_key(logits.dtype, _triton_run(logits), valid is not None),
    )
    # Unmasked, the kernel does not read ``valid``: the ids' first row stands in.
    kernel(
        output,
        logits,
        token_ids,
        token_ids[:1].view(torch.bool) if valid is None else valid,
    )
