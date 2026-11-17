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


@T.macro
def _dot_to_total(
    m,
    h,
    e,
    weight,
    total,
    partial,
    hidden: int,
    rank: int,
    threads: int,
):
    """Fixed-order BF16xFP32 dot product of ``[h[m] | e[m]] . weight`` into ``total[0]``."""
    warps = threads // 32
    hidden_pieces = hidden // _VEC
    rank_pieces = rank // _VEC
    hidden_chunks = -(-hidden_pieces // threads)
    rank_chunks = -(-rank_pieces // threads)
    chunks = hidden_chunks + rank_chunks

    tx = T.get_thread_binding()
    xs = T.alloc_local((chunks * _VEC,), T.bfloat16)
    ws = T.alloc_local((chunks * _VEC,), T.float32)
    acc = T.alloc_local((_VEC,), T.float32)

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


@T.macro
def _calibrate_from_logit(total_val, slope, intercepts, step, cal_out):
    """Calibrate raw confidence logit into ``cal_out[0]=feat, cal_out[1]=prob``."""
    raw_prob = T.float32(1) / (T.float32(1) + T.exp(-total_val))
    clamped = T.min(
        T.max(raw_prob, T.float32(1e-6)),
        T.float32(1.0 - 1e-6),
    )
    feat = T.min(
        T.max(
            T.log(clamped / (T.float32(1.0) - clamped)),
            T.float32(-40.0),
        ),
        T.float32(40.0),
    )
    cal = slope[0] * feat + intercepts[step]
    prob = T.float32(1) / (T.float32(1) + T.exp(-cal))
    cal_out[0] = feat
    cal_out[1] = prob


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def dspark_confidence_kernel(hidden: int, rank: int, threads: int = 128):
    """``p[m] = sigmoid(sum(h[m] * w[:hidden]) + sum(e[m] * w[hidden:]))``."""
    assert hidden % _VEC == 0 and rank % _VEC == 0 and threads % 32 == 0
    warps = threads // 32
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
            total = T.alloc_local((1,), T.float32)
            partial = T.alloc_shared((warps,), T.float32)
            _dot_to_total(m, h, e, weight, total, partial, hidden, rank, threads)
            if tx == 0:
                out[m] = T.float32(1) / (T.float32(1) + T.exp(-total[0]))

    return dspark_confidence


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def dspark_confidence_calibrate_kernel(
    hidden: int, rank: int, steps: int = 5, threads: int = 128
):
    """``out[m] = sigmoid(slope * clamp(logit(clamp(sigmoid(w . [h[m] | e[m]]), eps, 1 - eps)), -40, 40) + intercepts[m % steps])``,
    also recording ``features[slot, m % steps]`` and ``predictions[slot, m % steps]``
    when ``slot = idx_mapping[m] >= 0``."""
    assert (
        hidden % _VEC == 0 and rank % _VEC == 0 and steps > 0 and threads % 32 == 0
    )
    warps = threads // 32
    rows = T.dynamic("rows")
    slots = T.dynamic("slots")

    @T.prim_func
    def dspark_confidence_calibrate(
        h: T.Tensor((rows, hidden), T.bfloat16),
        e: T.Tensor((rows, rank), T.bfloat16),
        weight: T.Tensor((hidden + rank,), T.float32),
        out: T.Tensor((rows,), T.float32),
        slope: T.Tensor((1,), T.float32),
        intercepts: T.Tensor((steps,), T.float32),
        idx_mapping: T.Tensor((rows,), T.int32),
        features: T.Tensor((slots, steps), T.float32),
        predictions: T.Tensor((slots, steps), T.float32),
    ):
        with T.Kernel(rows, threads=threads) as m:
            tx = T.get_thread_binding()
            total = T.alloc_local((1,), T.float32)
            partial = T.alloc_shared((warps,), T.float32)
            _dot_to_total(m, h, e, weight, total, partial, hidden, rank, threads)
            cal_out = T.alloc_local((2,), T.float32)
            if tx == 0:
                step = m % steps
                _calibrate_from_logit(
                    total[0], slope, intercepts, step, cal_out
                )
                feat = cal_out[0]
                prob = cal_out[1]
                out[m] = prob
                slot = idx_mapping[m]
                if slot >= 0:
                    features[slot, step] = feat
                    predictions[slot, step] = prob

    return dspark_confidence_calibrate


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def dspark_confidence_step_and_gate_kernel(
    hidden: int,
    rank: int,
    steps: int = 5,
    single_row: bool = True,
    threads: int = 128,
):
    """Score and calibrate one draft step ``step`` across ``rows`` requests,
    compute the exact Hillis-Steele cumulative log-survival matching
    ``combine_sampled_and_draft_tokens_kernel``, zero tail ``out_probs[r, >step]``
    when a request falls below ``log_tau``, and update ``handle_i64`` directly
    when ``single_row`` is True."""
    assert (
        hidden % _VEC == 0 and rank % _VEC == 0 and steps > 0 and threads % 32 == 0
    )
    warps = threads // 32
    # Match DeadRows scan power-of-two width: next_pow2(steps + 1) = 1 << steps.bit_length()
    block = 1 << steps.bit_length()
    scan_strides = tuple(1 << i for i in range(max(block - 1, 0).bit_length()))
    rows = T.dynamic("rows")
    h_stride = T.dynamic("h_stride")
    e_stride = T.dynamic("e_stride")
    prob_rows = T.dynamic("prob_rows")
    out_stride = T.dynamic("out_stride")
    idx_stride = T.dynamic("idx_stride")
    slots = T.dynamic("slots")
    log_rows = T.dynamic("log_rows")
    log_stride = T.dynamic("log_stride")

    @T.prim_func
    def dspark_confidence_step_and_gate(
        h: T.StridedTensor((rows, hidden), (h_stride, 1), T.bfloat16),
        e: T.StridedTensor((rows, rank), (e_stride, 1), T.bfloat16),
        weight: T.Tensor((hidden + rank,), T.float32),
        out_probs: T.StridedTensor((prob_rows, steps), (out_stride, 1), T.float32),
        slope: T.Tensor((1,), T.float32),
        intercepts: T.Tensor((steps,), T.float32),
        idx_mapping: T.StridedTensor((rows,), (idx_stride,), T.int32),
        features: T.Tensor((slots, steps), T.float32),
        predictions: T.Tensor((slots, steps), T.float32),
        log_probs: T.StridedTensor((log_rows, steps), (log_stride, 1), T.float32),
        row_alive: T.Tensor((log_rows,), T.int32),
        log_tau: T.Tensor((1,), T.float32),
        step: T.int32,
        handle_i64: T.int64,
    ):
        with T.Kernel(1 if single_row else rows, threads=threads) as m:
            T.assume(h_stride >= hidden)
            T.assume(h_stride % _VEC == 0)
            T.assume(e_stride >= rank)
            T.assume(e_stride % _VEC == 0)
            T.assume(out_stride >= steps)
            T.assume(log_stride >= steps)
            T.assume(idx_stride >= 1)
            tx = T.get_thread_binding()
            total = T.alloc_local((1,), T.float32)
            partial = T.alloc_shared((warps,), T.float32)
            _dot_to_total(m, h, e, weight, total, partial, hidden, rank, threads)
            cal_out = T.alloc_local((2,), T.float32)
            if tx == 0:
                _calibrate_from_logit(
                    total[0], slope, intercepts, step, cal_out
                )
                feat = cal_out[0]
                prob = cal_out[1]
                out_probs[m, step] = prob
                slot = idx_mapping[m]
                if slot >= 0:
                    features[slot, step] = feat
                    predictions[slot, step] = prob
                lp = T.log(prob)
                log_probs[m, step] = lp

                surv = T.alloc_local((block,), T.float32)
                for j in tuple(range(block)):
                    if j < step:
                        surv[j] = log_probs[m, j]
                    elif j == step:
                        surv[j] = lp
                    else:
                        surv[j] = T.float32(0.0)
                for stride in scan_strides:
                    for j in tuple(range(block - 1, stride - 1, -1)):
                        surv[j] = surv[j - stride] + surv[j]

                step_surv = T.alloc_var(T.float32, init=T.float32(0.0))
                for j in T.unroll(steps):
                    if j == step:
                        step_surv = surv[j]
                alive = T.if_then_else(
                    ((slot >= 0) | (log_tau[0] < T.float32(-1e30)))
                    & (step_surv >= log_tau[0]),
                    1,
                    0,
                )
                row_alive[m] = alive
                if alive == 0:
                    for j in T.unroll(steps):
                        if j > step:
                            out_probs[m, j] = T.float32(0.0)
                if single_row:
                    if handle_i64 != T.int64(0):
                        T.call_extern(
                            "void",
                            "cudaGraphSetConditional",
                            T.Cast("uint64", handle_i64),
                            T.Cast("uint32", alive),
                        )

    return dspark_confidence_step_and_gate


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def dspark_confidence_gate_reduce_kernel(threads: int = 32):
    """Reduce per-request ``row_alive`` flags in a single CTA and set the CUDA
    graph conditional handle once without concurrent writers."""
    assert threads == 32
    max_rows = T.dynamic("max_rows")

    @T.prim_func
    def dspark_confidence_gate_reduce(
        row_alive: T.Tensor((max_rows,), T.int32),
        num_rows: T.int32,
        handle_i64: T.int64,
    ):
        with T.Kernel(1, threads=threads) as _:
            tx = T.get_thread_binding()
            local_alive = T.alloc_var(T.int32, init=0)
            for i in T.serial(T.ceildiv(num_rows, threads)):
                idx = i * threads + tx
                if idx < num_rows:
                    if row_alive[idx] != 0:
                        local_alive = 1
            any_alive = T.warp_reduce_sum(local_alive)
            if tx == 0 and handle_i64 != T.int64(0):
                T.call_extern(
                    "void",
                    "cudaGraphSetConditional",
                    T.Cast("uint64", handle_i64),
                    T.Cast("uint32", T.if_then_else(any_alive > 0, 1, 0)),
                )

    return dspark_confidence_gate_reduce


@cache
def _kernel(hidden: int, rank: int):
    return dspark_confidence_kernel(hidden, rank)


@cache
def _calibrate_kernel(hidden: int, rank: int, steps: int):
    return dspark_confidence_calibrate_kernel(hidden, rank, steps)


@cache
def _step_and_gate_kernel(hidden: int, rank: int, steps: int, single_row: bool):
    return dspark_confidence_step_and_gate_kernel(
        hidden, rank, steps, single_row=single_row
    )


@cache
def _gate_reduce_kernel():
    return dspark_confidence_gate_reduce_kernel()


def declare(hidden: int, rank: int, steps: int = 5) -> None:
    """Declare the head over ``hidden``-wide hidden states and ``rank``-wide Markov
    embeddings, and keep it."""
    prepare.declare(
        lambda: [
            prepare.call(dspark_confidence_kernel, hidden, rank),
            prepare.call(dspark_confidence_calibrate_kernel, hidden, rank, steps),
        ],
        bind=lambda: (
            _kernel(hidden, rank),
            _calibrate_kernel(hidden, rank, steps),
        ),
    )


def declare_step_and_gate(hidden: int, rank: int, steps: int = 5) -> None:
    """Declare the stepwise early-halting confidence kernels."""
    prepare.declare(
        lambda: [
            prepare.call(
                dspark_confidence_step_and_gate_kernel,
                hidden,
                rank,
                steps,
                single_row=True,
            ),
            prepare.call(
                dspark_confidence_step_and_gate_kernel,
                hidden,
                rank,
                steps,
                single_row=False,
            ),
            prepare.call(dspark_confidence_gate_reduce_kernel),
        ],
        bind=lambda: (
            _step_and_gate_kernel(hidden, rank, steps, True),
            _step_and_gate_kernel(hidden, rank, steps, False),
            _gate_reduce_kernel(),
        ),
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


def dspark_confidence_calibrate(
    hidden: torch.Tensor,
    markov_embed: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    slope: torch.Tensor,
    intercepts: torch.Tensor,
    idx_mapping: torch.Tensor,
    features: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    """Fused confidence head projection and online logistic calibration."""
    rows, width = hidden.shape
    rank = markov_embed.shape[-1]
    steps = intercepts.shape[0]
    if (
        hidden.dtype != torch.bfloat16
        or markov_embed.dtype != torch.bfloat16
        or weight.dtype != torch.float32
        or out.dtype != torch.float32
        or slope.dtype != torch.float32
        or intercepts.dtype != torch.float32
        or idx_mapping.dtype != torch.int32
        or features.dtype != torch.float32
        or predictions.dtype != torch.float32
        or markov_embed.shape != (rows, rank)
        or weight.shape != (width + rank,)
        or out.shape != (rows,)
        or slope.shape != (1,)
        or idx_mapping.shape != (rows,)
        or features.ndim != 2
        or features.shape[1] != steps
        or predictions.shape != features.shape
        or rows % steps != 0
        or not (hidden.is_contiguous() and markov_embed.is_contiguous())
        or not (weight.is_contiguous() and out.is_contiguous())
        or not (slope.is_contiguous() and intercepts.is_contiguous())
        or not (idx_mapping.is_contiguous() and features.is_contiguous())
        or not predictions.is_contiguous()
    ):
        raise ValueError(
            "Invalid inputs for fused TileLang DSpark confidence calibration"
        )
    if rows == 0:
        return
    _calibrate_kernel(width, rank, steps)(
        hidden,
        markov_embed,
        weight,
        out,
        slope,
        intercepts,
        idx_mapping,
        features,
        predictions,
    )


def dspark_confidence_step_and_gate(
    hidden: torch.Tensor,
    markov_embed: torch.Tensor,
    weight: torch.Tensor,
    out_probs: torch.Tensor,
    slope: torch.Tensor,
    intercepts: torch.Tensor,
    idx_mapping: torch.Tensor,
    features: torch.Tensor,
    predictions: torch.Tensor,
    log_probs: torch.Tensor,
    row_alive: torch.Tensor,
    log_tau: torch.Tensor,
    step: int,
    handle_i64: int = 0,
) -> None:
    """Evaluate calibrated confidence for a single draft step across ``rows``
    requests and update the cumulative-survival conditional handle."""
    rows, width = hidden.shape
    rank = markov_embed.shape[-1]
    steps = intercepts.shape[0]
    if (
        hidden.dtype != torch.bfloat16
        or markov_embed.dtype != torch.bfloat16
        or weight.dtype != torch.float32
        or out_probs.dtype != torch.float32
        or slope.dtype != torch.float32
        or intercepts.dtype != torch.float32
        or idx_mapping.dtype != torch.int32
        or features.dtype != torch.float32
        or predictions.dtype != torch.float32
        or log_probs.dtype != torch.float32
        or row_alive.dtype != torch.int32
        or log_tau.dtype != torch.float32
        or hidden.stride(1) != 1
        or markov_embed.stride(1) != 1
        or hidden.stride(0) % _VEC != 0
        or markov_embed.stride(0) % _VEC != 0
        or markov_embed.shape != (rows, rank)
        or weight.shape != (width + rank,)
        or out_probs.ndim != 2
        or out_probs.shape[0] < rows
        or out_probs.shape[1] != steps
        or out_probs.stride(1) != 1
        or slope.shape != (1,)
        or idx_mapping.shape != (rows,)
        or features.ndim != 2
        or features.shape[1] != steps
        or predictions.shape != features.shape
        or log_probs.ndim != 2
        or log_probs.shape[0] < rows
        or log_probs.shape[1] != steps
        or log_probs.stride(1) != 1
        or row_alive.shape[0] < rows
        or log_tau.shape != (1,)
        or not (0 <= step < steps)
        or not (weight.is_contiguous() and slope.is_contiguous())
        or not (intercepts.is_contiguous() and features.is_contiguous())
        or not (predictions.is_contiguous() and row_alive.is_contiguous())
        or not log_tau.is_contiguous()
    ):
        raise ValueError(
            "Invalid inputs for stepwise TileLang DSpark confidence gating"
        )
    if rows == 0:
        return
    single_row = rows == 1
    _step_and_gate_kernel(width, rank, steps, single_row)(
        hidden,
        markov_embed,
        weight,
        out_probs,
        slope,
        intercepts,
        idx_mapping,
        features,
        predictions,
        log_probs,
        row_alive,
        log_tau,
        int(step),
        int(handle_i64 if single_row else 0),
    )
    if not single_row and handle_i64 != 0:
        _gate_reduce_kernel()(row_alive, rows, int(handle_i64))


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

    def probability_and_calibrate(
        self,
        hidden: torch.Tensor,
        markov_embed: torch.Tensor,
        out: torch.Tensor | None,
        slope: torch.Tensor,
        intercepts: torch.Tensor,
        idx_mapping: torch.Tensor,
        features: torch.Tensor,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """Each row's calibrated acceptance probability in one TileLang launch."""
        if out is None:
            out = torch.empty(
                hidden.shape[0], dtype=torch.float32, device=hidden.device
            )
        dspark_confidence_calibrate(
            hidden,
            markov_embed,
            self.proj.weight.view(-1),
            out,
            slope,
            intercepts,
            idx_mapping,
            features,
            predictions,
        )
        return out

    def probability_step_and_gate(
        self,
        hidden: torch.Tensor,
        markov_embed: torch.Tensor,
        out_probs: torch.Tensor,
        slope: torch.Tensor,
        intercepts: torch.Tensor,
        idx_mapping: torch.Tensor,
        features: torch.Tensor,
        predictions: torch.Tensor,
        log_probs: torch.Tensor,
        row_alive: torch.Tensor,
        log_tau: torch.Tensor,
        step: int,
        handle_i64: int = 0,
    ) -> None:
        """Evaluate step ``step`` calibrated confidence and update survival gate."""
        dspark_confidence_step_and_gate(
            hidden,
            markov_embed,
            self.proj.weight.view(-1),
            out_probs,
            slope,
            intercepts,
            idx_mapping,
            features,
            predictions,
            log_probs,
            row_alive,
            log_tau,
            step,
            handle_i64=handle_i64,
        )


__all__ = [
    "TileLangConfidenceHead",
    "declare",
    "dspark_confidence",
    "dspark_confidence_calibrate",
    "dspark_confidence_calibrate_kernel",
    "dspark_confidence_gate_reduce_kernel",
    "dspark_confidence_kernel",
    "dspark_confidence_step_and_gate",
    "dspark_confidence_step_and_gate_kernel",
]
