# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shared expert's clamped SwiGLU at DeepSeek's reference precision.

DeepSeek's reference (``Expert.forward``) widens the gate and up projections to FP32,
clamps the gate from above and the up projection on both sides at ``swiglu_limit``,
multiplies ``silu(gate)`` by ``up`` in FP32 and rounds the product to BF16 once.
vLLM's ``SiluAndMulWithClamp`` rounds ``silu(gate)`` to BF16 before the multiply.
This kernel follows the reference: one launch, eight values a thread.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shape must resolve.
from functools import cache

import tilelang
import tilelang.language as T
import torch
from torch import nn

from vllm.models.deepseek_v4_1.tilelang import prepare

_VEC = 8


@tilelang.jit
def swiglu_clamp_kernel(n: int, limit: float, threads: int = 128):
    """``Y[m, j] = bf16(silu(min(G, limit)) * clamp(U, -limit, limit))`` in FP32, for
    ``X = [G | U]`` of shape ``[M, 2n]`` (BF16). The clamps compare and select, as
    ``torch.clamp`` does, so a NaN passes through; ``min``/``max`` (``fminf``,
    ``fmaxf``) would return the limit and hide it."""
    assert n % _VEC == 0
    M = T.dynamic("M")
    per_block = threads * _VEC

    @T.prim_func
    def activation_swiglu_clamp(
        X: T.Tensor((M, 2 * n), T.bfloat16),
        Y: T.Tensor((M, n), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(n, per_block), M, threads=threads) as (bx, by):
            col = (bx * threads + T.get_thread_binding()) * _VEC
            gate = T.alloc_local((_VEC,), T.bfloat16)
            up = T.alloc_local((_VEC,), T.bfloat16)
            out = T.alloc_local((_VEC,), T.bfloat16)
            if col < n:
                for v in T.vectorized(_VEC):
                    gate[v] = X[by, col + v]
                for v in T.vectorized(_VEC):
                    up[v] = X[by, n + col + v]
                for v in T.unroll(_VEC):
                    g = T.alloc_var(T.float32, init=T.Cast(T.float32, gate[v]))
                    u = T.alloc_var(T.float32, init=T.Cast(T.float32, up[v]))
                    g = T.if_then_else(g > limit, T.float32(limit), g)
                    u = T.if_then_else(u > limit, T.float32(limit), u)
                    u = T.if_then_else(u < -limit, T.float32(-limit), u)
                    out[v] = T.Cast(T.bfloat16, g / (1.0 + T.exp(-g)) * u)
                for v in T.vectorized(_VEC):
                    Y[by, col + v] = out[v]

    return activation_swiglu_clamp


@cache
def _kernel(n: int, limit: float):
    return swiglu_clamp_kernel(n, limit)


class SiluAndMulClampFP32(nn.Module):
    """Drop-in for ``SiluAndMulWithClamp`` with the reference's FP32 product.

    ``width`` is the gate (and up) width its owner runs it on, ``[M, 2 * width]``
    inputs; the module declares that kernel. Without it (kernel tests) nothing is
    declared.
    """

    def __init__(self, swiglu_limit: float, width: int | None = None):
        super().__init__()
        self.limit = float(swiglu_limit)
        self.width = width

    def declare_kernels(self) -> None:
        """Declare the kernel of the owner's width, and keep it in ``_kernel``."""
        if self.width is None:
            return
        width, limit = self.width, self.limit
        prepare.declare(
            lambda: [prepare.call(swiglu_clamp_kernel, width, limit)],
            bind=lambda: _kernel(width, limit),
        )

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        n = gate_up.shape[-1] // 2
        x = gate_up.reshape(-1, 2 * n).contiguous()
        out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
        if x.shape[0]:
            _kernel(n, self.limit)(x, out)
        return out.view(*gate_up.shape[:-1], n)


__all__ = ["SiluAndMulClampFP32", "swiglu_clamp_kernel"]
