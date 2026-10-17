# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang RMSNorm for DeepSeek-V4.1, through TileKernels' ``norm_forward``."""

import functools

import torch
from torch import nn

from vllm.models.deepseek_v4_1.tilelang import prepare


@torch.library.custom_op("vllm::dsv41_tilelang_rmsnorm", mutates_args=("out",))
def _rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, eps: float
) -> None:
    import tile_kernels

    tile_kernels.quant.norm_forward(x, weight, eps, out=out)


@_rmsnorm.register_fake
def _rmsnorm_fake(x, weight, out, eps):
    return None


@functools.cache
def _warm(hidden: int, eps: float, dtype: torch.dtype) -> None:
    """TileKernels' RMSNorm of one zero row of ``hidden`` (``dtype`` input and weight),
    so its kernel (keyed by the width, eps and dtype, not the rows) is compiled
    before the cache freezes."""
    x = torch.zeros((1, hidden), dtype=dtype, device="cuda")
    weight = torch.ones(hidden, dtype=dtype, device="cuda")
    _rmsnorm(x, weight, torch.empty_like(x), eps)


class TileLangRMSNorm(nn.Module):
    """``y = x * rsqrt(mean(x^2) + eps) * weight`` in FP32, in the input dtype."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16), requires_grad=False
        )
        self.variance_epsilon = eps

    def declare_kernels(self) -> None:
        """TileKernels' norm of this width; its input has the weight's dtype."""
        hidden, eps, dtype = (
            self.weight.shape[0],
            self.variance_epsilon,
            self.weight.dtype,
        )
        prepare.declare(warm=lambda: _warm(hidden, eps, dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        out = torch.empty_like(x)
        _rmsnorm(x, self.weight, out, self.variance_epsilon)
        return out.view(shape)
