# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's acceptance-confidence head in one TileLang launch.

For each drafted position the head scores the drafter's head hidden ``h`` (BF16,
``hidden`` wide) and the Markov embedding ``e`` of the token before it (BF16,
``rank`` wide) with one FP32 weight row ``w`` (``hidden + rank``) and returns the
acceptance probability ``sigmoid(w . [h | e])``. vLLM ran it as torch ops: a
concatenation, a widening FP32 copy, ``F.linear`` (a cuBLAS dot product and its
reduction for a few rows, a GEMV for more, so a row's order of summation depended
on how many requests the step drafted for) and ``torch.sigmoid``. This kernel reads
both halves where they are, one block per row:

- every load first: each thread's sixteen-byte pieces of ``h`` and ``e`` and their
  weights (thread ``x`` takes the pieces ``x``, ``x + threads``, ...);
- eight FP32 accumulators a thread, one per lane of a piece, each adding its
  products in piece order, so a value's chain is the thread's pieces (six at
  DS4.1's 5120 + 256 with 128 threads); then the eight lanes in a tree (three
  levels), the warp's lanes by shuffles (five) and the warps in order (three);
- ``1 / (1 + exp(-z))`` in FP32 with IEEE division and the accurate ``expf``,
  the expression ATen's sigmoid evaluates.

The products are of exactly widened BF16 values with FP32 weights, accumulated in
FP32 (fused multiply-adds), so the sum carries at most 17 roundings along any path
(``|z - z_exact| <= 17 * 2^-24 * sum(|h w| + |e w|)`` to first order; 23 without
FMA contraction), and the probability adds the sigmoid's own few roundings. The
order is fixed per row: a row's result does not depend on the batch, and a dot
product that is exact in FP32 gives the torch head's bits.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
from functools import cache

import tilelang
import tilelang.language as T
import torch

from vllm.model_executor.models.qwen3_dspark import DSparkConfidenceHead
from vllm.models.deepseek_v4_1.tilelang import prepare

# BF16 values in one sixteen-byte piece; one accumulator per value of a piece.
_VEC = 8


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def dspark_confidence_kernel(hidden: int, rank: int, threads: int = 128):
    """``p[m] = sigmoid(sum(h[m] * w[:hidden]) + sum(e[m] * w[hidden:]))``."""
    assert hidden % _VEC == 0 and rank % _VEC == 0 and threads % 32 == 0
    warps = threads // 32
    hidden_pieces = hidden // _VEC
    rank_pieces = rank // _VEC
    hidden_chunks = -(-hidden_pieces // threads)
    rank_chunks = -(-rank_pieces // threads)
    chunks = hidden_chunks + rank_chunks
    rows = T.dynamic("rows")

    @T.prim_func
    def dspark_confidence(
        h: T.Tensor((rows, hidden), T.bfloat16),
        e: T.Tensor((rows, rank), T.bfloat16),
        weight: T.Tensor((hidden + rank,), T.float32),
        out: T.Tensor((rows,), T.float32),
    ):
        with T.Kernel(rows, threads=threads) as m:
            tx = T.get_thread_binding()
            xs = T.alloc_local((chunks * _VEC,), T.bfloat16)
            ws = T.alloc_local((chunks * _VEC,), T.float32)
            acc = T.alloc_local((_VEC,), T.float32)
            total = T.alloc_local((1,), T.float32)
            partial = T.alloc_shared((warps,), T.float32)
            # Every load before any arithmetic; a thread past the last piece of a
            # half holds zeros there, which add nothing.
            for c in T.unroll(hidden_chunks):
                piece = c * threads + tx
                if piece < hidden_pieces:
                    for i in T.vectorized(_VEC):
                        xs[c * _VEC + i] = h[m, piece * _VEC + i]
                    for i in T.vectorized(4):
                        ws[c * _VEC + i] = weight[piece * _VEC + i]
                    for i in T.vectorized(4):
                        ws[c * _VEC + 4 + i] = weight[piece * _VEC + 4 + i]
                else:
                    for i in T.vectorized(_VEC):
                        xs[c * _VEC + i] = T.bfloat16(0)
                    for i in T.vectorized(_VEC):
                        ws[c * _VEC + i] = T.float32(0)
            for c in T.unroll(rank_chunks):
                piece = c * threads + tx
                k = hidden_chunks + c
                if piece < rank_pieces:
                    for i in T.vectorized(_VEC):
                        xs[k * _VEC + i] = e[m, piece * _VEC + i]
                    for i in T.vectorized(4):
                        ws[k * _VEC + i] = weight[hidden + piece * _VEC + i]
                    for i in T.vectorized(4):
                        ws[k * _VEC + 4 + i] = weight[hidden + piece * _VEC + 4 + i]
                else:
                    for i in T.vectorized(_VEC):
                        xs[k * _VEC + i] = T.bfloat16(0)
                    for i in T.vectorized(_VEC):
                        ws[k * _VEC + i] = T.float32(0)
            for i in T.unroll(_VEC):
                acc[i] = T.float32(0)
            for c in T.unroll(chunks):
                for i in T.unroll(_VEC):
                    acc[i] += T.cast(xs[c * _VEC + i], T.float32) * ws[c * _VEC + i]
            for i in T.unroll(4):
                acc[i] = acc[i] + acc[i + 4]
            for i in T.unroll(2):
                acc[i] = acc[i] + acc[i + 2]
            acc[0] = acc[0] + acc[1]
            acc[0] = T.warp_reduce_sum(acc[0])
            if tx % 32 == 0:
                partial[tx // 32] = acc[0]
            T.sync_threads()
            if tx == 0:
                total[0] = T.float32(0)
                for warp in T.serial(warps):
                    total[0] += partial[warp]
                out[m] = T.float32(1) / (T.float32(1) + T.exp(-total[0]))

    return dspark_confidence


@cache
def _kernel(hidden: int, rank: int):
    return dspark_confidence_kernel(hidden, rank)


def declare(hidden: int, rank: int) -> None:
    """Declare the head over ``hidden``-wide hidden states and ``rank``-wide Markov
    embeddings, and keep it."""
    prepare.declare(
        lambda: [prepare.call(dspark_confidence_kernel, hidden, rank)],
        bind=lambda: _kernel(hidden, rank),
    )


def dspark_confidence(
    hidden: torch.Tensor,
    markov_embed: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``out[m] = sigmoid(weight . [hidden[m] | markov_embed[m]])`` for BF16 rows
    ``hidden`` ([M, H]) and ``markov_embed`` ([M, R]) and the FP32 head row
    ``weight`` ([H + R]), into ``out`` ([M], FP32)."""
    rows, width = hidden.shape
    rank = markov_embed.shape[-1]
    if (
        hidden.dtype != torch.bfloat16
        or markov_embed.dtype != torch.bfloat16
        or weight.dtype != torch.float32
        or out.dtype != torch.float32
        or markov_embed.shape != (rows, rank)
        or weight.shape != (width + rank,)
        or out.shape != (rows,)
        or not (hidden.is_contiguous() and markov_embed.is_contiguous())
        or not (weight.is_contiguous() and out.is_contiguous())
    ):
        raise ValueError(
            "The TileLang DSpark confidence head needs contiguous BF16 [M, H] hidden "
            "and [M, R] Markov rows, an FP32 [H + R] weight and an FP32 [M] output: "
            f"hidden {tuple(hidden.shape)} {hidden.dtype}, Markov "
            f"{tuple(markov_embed.shape)} {markov_embed.dtype}, weight "
            f"{tuple(weight.shape)} {weight.dtype}, output {tuple(out.shape)} "
            f"{out.dtype}"
        )
    if rows == 0:
        return
    _kernel(width, rank)(hidden, markov_embed, weight, out)


class TileLangConfidenceHead(DSparkConfidenceHead):
    """``DSparkConfidenceHead`` (its FP32 projection and weight loading) whose
    probability is the one-launch kernel."""

    tp_invariant: bool = True

    def __init__(self, hidden_size: int, markov_rank: int, prefix: str = ""):
        super().__init__(hidden_size + markov_rank, prefix=prefix)
        self.hidden_size = hidden_size
        self.markov_rank = markov_rank

    def declare_kernels(self) -> None:
        declare(self.hidden_size, self.markov_rank)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "The TileLang DSpark confidence head returns probabilities: probability()"
        )

    def probability(
        self,
        hidden: torch.Tensor,
        markov_embed: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Each row's acceptance probability, into ``out`` (FP32 [rows],
        contiguous) when given."""
        if out is None:
            out = torch.empty(
                hidden.shape[0], dtype=torch.float32, device=hidden.device
            )
        dspark_confidence(hidden, markov_embed, self.proj.weight.view(-1), out)
        return out


__all__ = [
    "TileLangConfidenceHead",
    "declare",
    "dspark_confidence",
    "dspark_confidence_kernel",
]
