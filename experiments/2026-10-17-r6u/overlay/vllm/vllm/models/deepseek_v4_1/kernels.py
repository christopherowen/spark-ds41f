# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek V4.1 kernel family: B12X or TileLang.

``VLLM_DS41_KERNEL_BACKEND`` selects one family for the whole model, drafter
included; ``DeepseekV41ForCausalLMConfig`` checks that explicit attention,
linear and MoE backends agree with it. Model code asks this module for the
implementation of each component instead of naming a family.
"""

import torch
from torch import nn

from vllm import envs


def uses_tilelang() -> bool:
    return envs.VLLM_DS41_KERNEL_BACKEND == "tilelang"


def linear_method():
    """Method for an unquantized (BF16) projection."""
    if uses_tilelang():
        from .tilelang.linear import TileLangLinearMethod

        return TileLangLinearMethod()
    from .b12x_layers import B12xLinearMethod

    return B12xLinearMethod()


def block_fp8_linear_method(quant_config):
    """Method for a checkpoint block-32 FP8 projection."""
    if uses_tilelang():
        from .tilelang.linear import TileLangFP8LinearMethod

        return TileLangFP8LinearMethod(quant_config)
    from .b12x_layers import B12xFP8LinearMethod

    return B12xFP8LinearMethod(quant_config)


def embedding_method():
    if uses_tilelang():
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            UnquantizedEmbeddingMethod,
        )

        return UnquantizedEmbeddingMethod()
    from .b12x_layers import B12xEmbeddingMethod

    return B12xEmbeddingMethod()


def moe_output_dtype() -> torch.dtype:
    """The routed experts' output. FP32 keeps the routes' combined sum unrounded,
    so the rank's partial (shared expert added in FP32) rounds once before the
    cross-rank sum, as DeepSeek's row-parallel linears round theirs; BF16 rounds
    at the combine and again at the shared add."""
    return torch.float32 if MOE_FP32_COMBINE and uses_tilelang() else torch.bfloat16


# The MoE combine stays FP32 until the rank's partial rounds, under the TileLang family.
MOE_FP32_COMBINE = True


def shared_expert_activation(act_fn: nn.Module, width: int) -> nn.Module:
    """The shared expert's clamped SwiGLU on ``width``-wide gate and up halves;
    TileLang's keeps the reference's FP32 product (one BF16 rounding) where vLLM's
    rounds ``silu(gate)`` first."""
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    if (
        SHARED_SWIGLU_FP32
        and uses_tilelang()
        and isinstance(act_fn, SiluAndMulWithClamp)
    ):
        from .tilelang.activation import SiluAndMulClampFP32

        return SiluAndMulClampFP32(act_fn.swiglu_limit, width)
    return act_fn


# The shared expert's SwiGLU at the reference's precision under the TileLang family.
SHARED_SWIGLU_FP32 = True


def moe_router_class():
    """Router class for the MoE layers; None keeps vLLM's for the config."""
    if uses_tilelang():
        from .tilelang.router import TileLangGateRouter

        return TileLangGateRouter
    return None


def mhc(config) -> nn.Module:
    """A decoder layer's lagged mHC: pre, post_pre, post and collapse."""
    if uses_tilelang():
        from .tilelang.mhc import TileKernelsMHC

        return TileKernelsMHC(config)
    from .b12x_layers import B12xMHC

    return B12xMHC(config)


def rms_norm(hidden_size: int, eps: float = 1e-6) -> nn.Module:
    if uses_tilelang():
        from .tilelang.norm import TileLangRMSNorm

        return TileLangRMSNorm(hidden_size, eps)
    from .b12x_layers import B12xRMSNorm

    return B12xRMSNorm(hidden_size, eps)


def is_native_linear_method(method) -> bool:
    from .b12x_layers import B12xFP8LinearMethod, B12xLinearMethod

    native = (B12xFP8LinearMethod, B12xLinearMethod)
    if uses_tilelang():
        from .tilelang.linear import TileLangFP8LinearMethod, TileLangLinearMethod

        native = (TileLangFP8LinearMethod, TileLangLinearMethod)
    return isinstance(method, native)
