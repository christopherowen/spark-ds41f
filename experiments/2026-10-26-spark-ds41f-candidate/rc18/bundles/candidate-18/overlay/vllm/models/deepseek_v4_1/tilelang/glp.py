# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLP steering of the rows a step marks, in one launch a layer.

A layer leaves its write ``x`` and the mixes that fold it into the streams for the
next layer's fused post/pre: ``h[n] = bf16(post[n] x + sum_m comb[m, n] residual[m])``.
For each marked row (``rows[t] != 0``) one block folds the write as ``post_streams``
does (the ranks' unreduced parts of ``x`` first added in rank order, as
``rank_order_sum`` does), projects every stream, ``h <- h - alpha (h . d) d``, and
writes the projected streams back as the row's ``residual``, with ``comb`` the
identity and ``post`` zero. The next layer's fold of that row then gives the
projected streams back exactly (``1 * r + 0 * ...`` in FP32 is ``r``; only a
negative zero becomes positive), whichever path reads it: the fused post/pre, an
Engram layer's post, the drafter's auxiliary states or the final collapse.

A row that is not marked is not read or written: its block returns after loading
its mark, so unsteered rows keep their bits whatever else the step carries.

The kernel is on the decode chain's PDL (``pdl.chain()``): it waits for the layer
before it reads its mark, so a step with no marked row keeps the chain's overlap.

``direction`` ([hidden], FP32) is the layer's unit row and ``alpha`` ([1], FP32) a
buffer, so a file's contents never change the traced graph. A block's 320 threads
each own sixteen columns of every stream (two loads of eight); a stream's ``h . d``
is reduced in a fixed order (each thread's values in sequence, then the warp's
lanes, then the warps), the order of the projection this replaces
(experiments/2026-10-23-next-release, glp-variants), so a row's result never
depends on the batch.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
from functools import cache

import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import pdl, prepare


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def glp_steer_kernel(
    hc_mult: int, hidden: int, world: int, threads: int = 320, vec: int = 8
):
    assert hidden % (threads * vec) == 0 and threads % 32 == 0
    assert 1 <= world <= 4
    chunks = hidden // (threads * vec)
    warps = threads // 32
    span = chunks * vec
    tokens = T.dynamic("tokens")

    @T.prim_func
    def glp_steer(
        parts: T.Tensor((world, tokens, hidden), T.bfloat16),
        residual: T.Tensor((tokens, hc_mult, hidden), T.bfloat16),
        post: T.Tensor((tokens, hc_mult), T.float32),
        comb: T.Tensor((tokens, hc_mult, hc_mult), T.float32),
        rows: T.Tensor((tokens,), T.int32),
        direction: T.Tensor((hidden,), T.float32),
        alpha: T.Tensor((1,), T.float32),
    ):
        with T.Kernel(tokens, threads=threads) as t:
            pdl.chain()
            tx = T.get_thread_binding()
            # The whole block leaves an unmarked row, before any barrier.
            if rows[t] == 0:
                T.thread_return()
            hs = T.alloc_local((hc_mult * span,), T.float32)
            ds = T.alloc_local((span,), T.float32)
            xs = T.alloc_local((vec,), T.float32)
            rv = T.alloc_local((hc_mult * vec,), T.bfloat16)
            pv = T.alloc_local((world * vec,), T.bfloat16)
            pm = T.alloc_local((hc_mult,), T.float32)
            cm = T.alloc_local((hc_mult * hc_mult,), T.float32)
            acc = T.alloc_local((hc_mult,), T.float32)
            fold = T.alloc_var(T.float32)
            total = T.alloc_var(T.float32)
            partial = T.alloc_shared((hc_mult, warps), T.float32)
            scale = T.alloc_shared((hc_mult,), T.float32)
            for n in T.unroll(hc_mult):
                pm[n] = post[t, n]
                acc[n] = T.float32(0)
            for m in T.unroll(hc_mult):
                for n in T.unroll(hc_mult):
                    cm[m * hc_mult + n] = comb[t, m, n]
            for c in T.serial(chunks):
                for r in T.unroll(world):
                    for v in T.vectorized(vec):
                        pv[r * vec + v] = parts[r, t, (c * threads + tx) * vec + v]
                for m in T.unroll(hc_mult):
                    for v in T.vectorized(vec):
                        rv[m * vec + v] = residual[t, m, (c * threads + tx) * vec + v]
                for v in T.vectorized(vec):
                    ds[c * vec + v] = direction[(c * threads + tx) * vec + v]
                # rank_order_sum: the first part as is, then the others in order.
                for v in T.unroll(vec):
                    xs[v] = T.Cast(T.float32, pv[v])
                    for r in T.serial(world - 1):
                        xs[v] = xs[v] + T.Cast(T.float32, pv[(r + 1) * vec + v])
                    xs[v] = T.Cast(T.float32, T.Cast(T.bfloat16, xs[v]))
                # post_streams: the first stream's rounded product, then FMAs in order.
                for n in T.unroll(hc_mult):
                    for v in T.unroll(vec):
                        fold = cm[n] * T.Cast(T.float32, rv[v])
                        fold = T.fma(pm[n], xs[v], fold)
                        for m in T.unroll(1, hc_mult):
                            fold = T.fma(
                                cm[m * hc_mult + n],
                                T.Cast(T.float32, rv[m * vec + v]),
                                fold,
                            )
                        hs[n * span + c * vec + v] = T.Cast(
                            T.float32, T.Cast(T.bfloat16, fold)
                        )
            for n in T.unroll(hc_mult):
                for i in T.serial(span):
                    acc[n] += hs[n * span + i] * ds[i]
                acc[n] = T.warp_reduce_sum(acc[n])
                if tx % 32 == 0:
                    partial[n, tx // 32] = acc[n]
            T.sync_threads()
            if tx < hc_mult:
                total = T.float32(0)
                for w in T.serial(warps):
                    total += partial[tx, w]
                scale[tx] = alpha[0] * total
            T.sync_threads()
            for c in T.serial(chunks):
                for n in T.unroll(hc_mult):
                    for v in T.vectorized(vec):
                        residual[t, n, (c * threads + tx) * vec + v] = T.Cast(
                            T.bfloat16,
                            hs[n * span + c * vec + v] - scale[n] * ds[c * vec + v],
                        )
            if tx < hc_mult * hc_mult:
                comb[t, tx // hc_mult, tx % hc_mult] = T.if_then_else(
                    tx // hc_mult == tx % hc_mult, T.float32(1), T.float32(0)
                )
            if tx < hc_mult:
                post[t, tx] = T.float32(0)

    return glp_steer


@cache
def _kernel(hc_mult: int, hidden: int, world: int):
    return glp_steer_kernel(hc_mult, hidden, world)


def glp_steer(
    parts: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    rows: torch.Tensor,
    direction: torch.Tensor,
    alpha: torch.Tensor,
) -> None:
    """Steer the rows ``rows`` (int32, [T]) marks: fold ``parts`` ([world, T, H],
    BF16, the layer's write or the ranks' unreduced parts of it) into ``residual``
    ([T, hc_mult, H], BF16) with ``post`` ([T, hc_mult]) and ``comb`` ([T, hc_mult,
    hc_mult]), project ``direction`` off every stream scaled by ``alpha`` ([1]),
    and leave the result as the row's ``residual`` with ``comb`` the identity and
    ``post`` zero. Unmarked rows are untouched."""
    world, tokens, hidden = parts.shape
    if tokens == 0:
        return
    _kernel(residual.shape[1], hidden, world)(
        parts, residual, post, comb, rows, direction, alpha
    )


def declare(hc_mult: int, hidden: int, worlds: tuple[int, ...]) -> None:
    """Declare the steering of ``hc_mult`` streams of ``hidden`` for a write of
    each of ``worlds`` parts, and keep them."""
    prepare.declare(
        lambda: [prepare.call(glp_steer_kernel, hc_mult, hidden, w) for w in worlds],
        bind=lambda: [_kernel(hc_mult, hidden, w) for w in worlds],
    )


__all__ = ["declare", "glp_steer", "glp_steer_kernel"]
