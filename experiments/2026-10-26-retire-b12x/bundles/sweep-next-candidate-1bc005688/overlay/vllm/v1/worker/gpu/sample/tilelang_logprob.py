# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The selected tokens' ranks and log probabilities over the real vocabulary.

- ``token_ranks``: each row's selected-token rank, ties included
  (``logprob._ranks_kernel`` without its phantom padding columns)
- ``topk_log_softmax``: the log-softmax at the selected tokens by Barron,
  Gnanapragasam and Tschannen's fix ("A Fix for Softmax and Cross-Entropy", Eq. 5),
  in place of ``logprob._topk_log_softmax_kernel``'s arithmetic (the Triton kernel
  stays importable as the tests' baseline)

The Triton kernel sums every column's ``exp(logit - max_val)``, the maximum's
``exp(0) = 1`` among them, and returns ``(logit - max_val) - log(sum)``. Once the
other columns add less than half an FP32 ULP of 1 (2^-24: a margin past 16.6 nats)
the sum rounds to 1 and the top token's log probability to exactly 0; short of
that, every block added after the maximum's is rounded against 1, losing what it
holds below 2^-24, and the log probability lands on a multiple of 2^-23. The sum
here leaves the maximum out (S), so no 1 absorbs it, and the log-sum-exp past the
maximum is log1p(S):

- ``max_val`` is ``max.f32`` (a NaN gives way) of every logit and -inf, so any
  order gives it: a block of 1024 threads takes the row, last step first, each
  thread folding its columns into one maximum a load, then the block
  (``_row_reduce``). The sum then reads the row from its first step, the last
  read: at a batch's rows, the steps still in L2.
- ``S`` is the sum of ``exp(logit - max_val)`` over the columns that are not the
  maximum, and of the maximum's ties all but one, ``exp(max_val - max_val)`` each
  (``vllm_others``: 1, or NaN for an infinite maximum, as the paper's z - z_max).
  ``exp`` is ``ex2.approx.f32`` (subnormals kept) of the product with log2(e)
  (``vllm_other_exp``); a column past the row adds 0. A thread adds the columns of
  each of its two loads a step, in order, to that load's sum, step after step, and
  folds the two in load order; the warp adds at XOR distances 16 to 1, and every
  thread adds the 32 warps' sums in warp order. The ties are counted exactly in
  FP32 the same way.
- ``lse`` is CUDA's ``log1pf(S)`` (libdevice, within an ULP), the paper's
  softplus(-m), and a token's log probability ``shifted - lse``, ``shifted`` its
  ``logit - max_val``, exactly 0 at the maximum (``vllm_shifted``).

A row's outputs depend on its own logits only, in a fixed order: the same bits in
any batch and on every run. Every add, product and difference is the header's,
rounded to nearest (nvcc would contract them otherwise).
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
from collections.abc import Sequence

import tilelang
import tilelang.language as T
import torch

from vllm.v1.worker.gpu.sample import tilelang_sampling
from vllm.v1.worker.gpu.sample.tilelang_sampling import (
    _call,
    _kernel,
    _runs,
    _triton_run,
    declare,
)
from vllm.v1.worker.gpu.spec_decode.tilelang_rejection import _HEADER

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

# A row's block of threads, and the vector loads a thread issues a step: 2048 runs
# in flight a row, so one row (a decode step's single request) is not a few warps'
# round trips to memory one after another. Two loads keep it within 64 registers,
# which 1024 threads allow (dgx4, image -next-5: 1024 x 2 was level with the
# release's kernel at 96 rows and 35% faster at 1 to 16; 512 x 4 and 256 x 4 were
# slower at 96 FP32 rows, and 1024 x 4 spills).
_SOFTMAX_THREADS = 1024
_SOFTMAX_WARPS = _SOFTMAX_THREADS // 32
_SOFTMAX_LOADS = 2


def _masked(value, column, vocab, guarded: bool, other):
    """``value``, or ``other`` where a guarded step's ``column`` is past the row."""
    if guarded:
        return T.if_then_else(column < vocab, value, other)
    return value


@T.macro
def _row_reduce(value, partial, tx, combine):
    """Every thread's ``value[0]`` becomes the block's: lanes at XOR distances 16 to
    1, then each warp's in warp order, every thread combining them alike.
    ``combine`` is the header's ``vllm_max`` or ``vllm_add``; ``partial`` holds one
    value a warp and serves this reduction only."""
    for distance in (16, 8, 4, 2, 1):
        value[0] = T.call_extern(
            "float32", combine, value[0], T.shfl_xor(value[0], distance)
        )
    if tx % 32 == 0:
        partial[tx // 32] = value[0]
    T.sync_threads()
    value[0] = partial[0]
    for w in tuple(range(1, _SOFTMAX_WARPS)):
        value[0] = T.call_extern("float32", combine, value[0], partial[w])


@tilelang.jit(pass_configs=tilelang_sampling._PASS)
def topk_log_softmax_kernel(logits_dtype: str, run: int, masked: bool = False):
    """The selected tokens' log-softmax (the module's Eq. 5): one block of 1024
    threads a row, whose ``topk`` selected tokens' log probabilities it writes;
    with ``masked``, -inf where ``valid`` is false (the custom-ids path's
    ``masked_fill``, in the same store; unmasked, ``valid`` is not read). The row is
    read twice in steps of ``1024 * 2 * run`` columns: a thread loads two runs of
    ``run`` columns (one vector each; the launch checked the alignment), runs
    ``1024 * run`` apart, and the row's last step loads each column clamped to the
    row, the columns past it masked."""
    rows, vocab, topk = T.dynamic("rows, vocab, topk")
    logits_stride = T.dynamic("logits_stride", "int64")
    valid_rows, valid_cols = T.dynamic("valid_rows, valid_cols")
    threads = _SOFTMAX_THREADS
    loads = _SOFTMAX_LOADS
    per = loads * run
    span = threads * per

    def column(step, j, tx):
        """The column of thread ``tx``'s ``j``-th element of step ``step``."""
        return step * span + (j // run) * (threads * run) + tx * run + j % run

    @T.macro
    def load(logits, raw, row, step, tx, guarded):
        """The thread's elements of step ``step``, load ``u``'s at ``raw[u * run:]``."""
        for u in tuple(range(loads)):
            col0 = column(step, u * run, tx)
            if guarded:
                for k in tuple(range(run)):
                    raw[u * run + k] = logits[row, T.min(col0 + k, vocab - 1)]
            elif run == 1:
                raw[u] = logits[row, col0]
            else:
                for k in T.vectorized(run):
                    raw[u * run + k] = logits[row, col0 + k]

    @T.macro
    def fold_max(raw, peak, step, tx, guarded):
        """Fold the step's elements into the thread's maxima, one a load; a column
        past the row is -inf."""
        for j in tuple(range(per)):
            value = _masked(
                T.cast(raw[j], T.float32),
                column(step, j, tx),
                vocab,
                guarded,
                -T.infinity(T.float32),
            )
            peak[j // run] = T.call_extern("float32", "vllm_max", peak[j // run], value)

    @T.macro
    def fold_sum(raw, total, tally, maximum, step, tx, guarded):
        """Add the step's other columns' exponentials to the thread's sums, one a
        load, each load's columns in order, and count the maximum's columns; a
        column past the row adds 0."""
        for j in tuple(range(per)):
            value = T.cast(raw[j], T.float32)
            col = column(step, j, tx)
            term = _masked(
                T.call_extern("float32", "vllm_other_exp", value, maximum),
                col,
                vocab,
                guarded,
                T.float32(0),
            )
            tie = _masked(
                T.call_extern("float32", "vllm_tie", value, maximum),
                col,
                vocab,
                guarded,
                T.float32(0),
            )
            total[j // run] = T.call_extern(
                "float32", "vllm_add", total[j // run], term
            )
            tally[j // run] = T.call_extern("float32", "vllm_add", tally[j // run], tie)

    @T.prim_func
    def topk_log_softmax(
        output: T.Tensor((rows, topk), T.float32),
        logits: T.StridedTensor((rows, vocab), (logits_stride, 1), logits_dtype),
        token_ids: T.Tensor((rows, topk), T.int64),
        valid: T.Tensor((valid_rows, valid_cols), T.bool),
    ):
        with T.Kernel(rows, threads=threads) as row:
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            raw = T.alloc_local((per,), logits_dtype)
            peak = T.alloc_local((loads,), T.float32)
            total = T.alloc_local((loads,), T.float32)
            tally = T.alloc_local((loads,), T.float32)
            best = T.alloc_local((1,), T.float32)
            se = T.alloc_local((1,), T.float32)
            ties = T.alloc_local((1,), T.float32)
            max_partial = T.alloc_shared((_SOFTMAX_WARPS,), T.float32)
            sum_partial = T.alloc_shared((_SOFTMAX_WARPS,), T.float32)
            tie_partial = T.alloc_shared((_SOFTMAX_WARPS,), T.float32)
            if run > 1:
                # The launch checked it: a 16-byte base and a stride of 16 elements.
                T.assume(logits_stride % 16 == 0)

            # Whole steps; the rest of the row is the guarded step.
            full = vocab // span
            guarded = full * span < vocab

            # max_val, the row read last step first (NaN gives way, so any order
            # gives it): the sum then starts on the steps most recently read, still
            # in L2 at a batch's rows.
            for u in tuple(range(loads)):
                peak[u] = -T.infinity(T.float32)
            if guarded:
                load(logits, raw, row, full, tx, True)
                fold_max(raw, peak, full, tx, True)
            for step in T.serial(full):
                load(logits, raw, row, full - 1 - step, tx, False)
                fold_max(raw, peak, full - 1 - step, tx, False)
            best[0] = peak[0]
            for u in tuple(range(1, loads)):
                best[0] = T.call_extern("float32", "vllm_max", best[0], peak[u])
            _row_reduce(best, max_partial, tx, "vllm_max")
            maximum = T.alloc_var(T.float32, init=best[0])

            # S: the other columns' exponentials, a thread's sums in step order,
            # folded in load order, then the block's; and the maximum's ties.
            for u in tuple(range(loads)):
                total[u] = T.float32(0)
                tally[u] = T.float32(0)
            for step in T.serial(full):
                load(logits, raw, row, step, tx, False)
                fold_sum(raw, total, tally, maximum, step, tx, False)
            if guarded:
                load(logits, raw, row, full, tx, True)
                fold_sum(raw, total, tally, maximum, full, tx, True)
            se[0] = total[0]
            ties[0] = tally[0]
            for u in tuple(range(1, loads)):
                se[0] = T.call_extern("float32", "vllm_add", se[0], total[u])
                ties[0] = T.call_extern("float32", "vllm_add", ties[0], tally[u])
            _row_reduce(se, sum_partial, tx, "vllm_add")
            _row_reduce(ties, tie_partial, tx, "vllm_add")
            # The log-sum-exp past the maximum, log1p(S).
            lse = T.alloc_var(
                T.float32,
                init=T.call_extern(
                    "float32",
                    "vllm_softplus",
                    T.call_extern("float32", "vllm_others", se[0], ties[0], maximum),
                ),
            )

            # The selected tokens, 1024 at a time. Row offsets are 64-bit.
            row64 = T.cast(row, T.int64)
            for c in T.serial(T.ceildiv(topk, threads)):
                col = c * threads + tx
                if col < topk:
                    picked = T.cast(logits[row, token_ids[row64, col]], T.float32)
                    value = T.call_extern(
                        "float32",
                        "vllm_sub",
                        T.call_extern("float32", "vllm_shifted", picked, maximum),
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
    """Each row's log-softmax (Eq. 5) at its ``token_ids``, into ``output``: the
    launch ``_topk_log_softmax_kernel`` took."""
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
