# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang Engram gate for DeepSeek-V4.1.

DeepSeek's signed-sqrt gate (TileKernels ``engram_gate_fwd``, inference form) over
the multi-stream residual. For each token and stream, with the stream's residual
``x``, key ``k``, the token's shared value ``v`` and the stream's fused norm weight
``w`` (the product of the query and key RMSNorm weights)::

    dot  = sum(x * w * k) * rsqrt(mean(x^2) + eps) * rsqrt(mean(k^2) + eps) / sqrt(D)
    gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))
    out  = x + gate * v

in FP32, rounded to BF16. Image tokens keep ``x``. One block per (token, stream)
spreads a row's loads over the whole block, and every load (the value too) is
issued before the reduction, so a decode row costs one DRAM round trip.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
import tilelang
import tilelang.language as T
import torch

# Gate magnitudes below this clamp to it before the square root.
CLAMP = 1e-6


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def engram_gate_kernel(
    hc_mult: int,
    hidden: int,
    eps: float,
    has_mask: bool,
    threads: int = 256,
    vec: int = 4,
):
    assert hidden % (threads * vec) == 0 and threads % 32 == 0
    chunks = hidden // (threads * vec)
    warps = threads // 32
    scalar = hidden**-0.5
    tokens = T.dynamic("tokens")

    @T.prim_func
    def engram_gate(
        x: T.Tensor((tokens, hc_mult, hidden), T.bfloat16),
        kv: T.Tensor((tokens, hc_mult + 1, hidden), T.bfloat16),
        weight: T.Tensor((hc_mult, hidden), T.float32),
        image_mask: T.Tensor((tokens,), T.bool),
        out: T.Tensor((tokens, hc_mult, hidden), T.bfloat16),
    ):
        with T.Kernel(hc_mult, tokens, threads=threads) as (h, t):
            tx = T.get_thread_binding()
            xs = T.alloc_local((chunks * vec,), T.float32)
            vs = T.alloc_local((chunks * vec,), T.float32)
            ks = T.alloc_local((vec,), T.float32)
            ws = T.alloc_local((vec,), T.float32)
            acc = T.alloc_local((3,), T.float32)
            partial = T.alloc_shared((3, warps), T.float32)
            gate = T.alloc_shared((1,), T.float32)
            for c in T.serial(chunks):
                for i in T.vectorized(vec):
                    xs[c * vec + i] = x[t, h, (c * threads + tx) * vec + i]
            if has_mask and image_mask[t]:
                for c in T.serial(chunks):
                    for i in T.vectorized(vec):
                        out[t, h, (c * threads + tx) * vec + i] = T.cast(
                            xs[c * vec + i], T.bfloat16
                        )
            else:
                for j in T.serial(3):
                    acc[j] = T.float32(0)
                for c in T.serial(chunks):
                    # The value loads with the key: no second round trip after
                    # the gate.
                    for i in T.vectorized(vec):
                        vs[c * vec + i] = kv[t, hc_mult, (c * threads + tx) * vec + i]
                    for i in T.vectorized(vec):
                        ks[i] = kv[t, h, (c * threads + tx) * vec + i]
                    for i in T.vectorized(vec):
                        ws[i] = weight[h, (c * threads + tx) * vec + i]
                    for i in T.serial(vec):
                        acc[0] += xs[c * vec + i] * xs[c * vec + i]
                        acc[1] += ks[i] * ks[i]
                        acc[2] += xs[c * vec + i] * ws[i] * ks[i]
                for j in T.serial(3):
                    acc[j] = T.warp_reduce_sum(acc[j])
                if tx % 32 == 0:
                    for j in T.serial(3):
                        partial[j, tx // 32] = acc[j]
                T.sync_threads()
                if tx == 0:
                    for j in T.serial(3):
                        acc[j] = T.float32(0)
                        for warp in T.serial(warps):
                            acc[j] += partial[j, warp]
                    dot = (
                        acc[2]
                        * T.rsqrt(acc[0] / hidden + eps)
                        * T.rsqrt(acc[1] / hidden + eps)
                        * scalar
                    )
                    gate[0] = T.sigmoid(
                        T.copysign(T.sqrt(T.max(T.abs(dot), T.float32(CLAMP))), dot)
                    )
                T.sync_threads()
                for c in T.serial(chunks):
                    for i in T.vectorized(vec):
                        out[t, h, (c * threads + tx) * vec + i] = T.cast(
                            xs[c * vec + i] + gate[0] * vs[c * vec + i], T.bfloat16
                        )

    return engram_gate


@torch.library.custom_op("vllm::dsv41_tilelang_engram_gate", mutates_args=("out",))
def engram_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    weight: torch.Tensor,
    image_mask: torch.Tensor | None,
    out: torch.Tensor,
    eps: float,
    hc_mult: int,
) -> None:
    """Gate ``x`` ([T, hc_mult * D]) with ``kv`` ([T, (hc_mult + 1) * D]: the keys,
    then the shared value) and the fused norm ``weight`` ([hc_mult * D], FP32) into
    ``out``. ``image_mask`` marks tokens that pass through unchanged."""
    tokens = x.shape[0]
    if tokens == 0:
        return
    hidden = weight.numel() // hc_mult
    kernel = engram_gate_kernel(hc_mult, hidden, eps, image_mask is not None)
    kernel(
        x.view(tokens, hc_mult, hidden),
        kv.view(tokens, hc_mult + 1, hidden),
        weight.view(hc_mult, hidden),
        image_mask,
        out.view(tokens, hc_mult, hidden),
    )


@engram_gate.register_fake
def _engram_gate_fake(x, kv, weight, image_mask, out, eps, hc_mult):
    return None


__all__ = ["CLAMP", "engram_gate", "engram_gate_kernel"]
