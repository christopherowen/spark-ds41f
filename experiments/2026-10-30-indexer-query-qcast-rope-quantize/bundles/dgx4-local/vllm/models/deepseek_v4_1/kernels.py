# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek V4.1 kernel family: TileLang.

One family runs the whole model, drafter included; ``DeepseekV41ForCausalLMConfig``
checks that explicit attention, linear and MoE backends name it. Model code asks
this module for the implementation of each component instead of naming it.
"""

import contextlib

import torch
from torch import nn


def linear_method():
    """Method for an unquantized (BF16) projection."""
    from .tilelang.linear import TileLangLinearMethod

    return TileLangLinearMethod()


def block_fp8_linear_method(quant_config):
    """Method for a checkpoint block-32 FP8 projection."""
    from .tilelang.linear import TileLangFP8LinearMethod

    return TileLangFP8LinearMethod(quant_config)


def embedding_method():
    from .tilelang.embedding import TileLangEmbeddingMethod

    return TileLangEmbeddingMethod()


def moe_output_dtype() -> torch.dtype:
    """The routed experts' output. FP32 keeps the routes' combined sum unrounded,
    so the rank's partial (shared expert added in FP32) rounds once before the
    cross-rank sum, as DeepSeek's row-parallel linears round theirs."""
    return torch.float32


def shared_expert_activation(act_fn: nn.Module, width: int) -> nn.Module:
    """The shared expert's clamped SwiGLU on ``width``-wide gate and up halves,
    at the reference's FP32 product (one BF16 rounding) where vLLM's rounds
    ``silu(gate)`` first."""
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    if isinstance(act_fn, SiluAndMulWithClamp):
        from .tilelang.activation import SiluAndMulClampFP32

        return SiluAndMulClampFP32(act_fn.swiglu_limit, width)
    return act_fn


def moe_router_class():
    """Router class for the MoE layers."""
    from .tilelang.router import TileLangGateRouter

    return TileLangGateRouter


def mhc(config) -> nn.Module:
    """A decoder layer's lagged mHC: pre, post_pre, post and collapse."""
    from .tilelang.mhc import TileKernelsMHC

    return TileKernelsMHC(config)


def cast_mxfp8(rows: torch.Tensor):
    """The MXFP8 cast of rows regrouped after their producer (``linear.MXFP8Rows``),
    for projections that read their input's cast."""
    from .tilelang.linear import cast_mxfp8

    return cast_mxfp8(rows)


def slice_mxfp8(rows, first: int, last: int):
    """Rows ``[first, last)`` of ``rows`` (``linear.MXFP8Rows``) for a split consumer."""
    if rows is None:
        return None
    from .tilelang.linear import slice_mxfp8

    return slice_mxfp8(rows, first, last)


def mxfp8_inputs(rows) -> contextlib.AbstractContextManager:
    """Hand a sublayer's projections their input's MXFP8 cast (``linear.MXFP8Rows``,
    from the mHC entry or ``cast_mxfp8``) for the calls inside; nothing for None
    (the model tests' mHC stand-ins return no cast)."""
    if rows is None:
        return contextlib.nullcontext()
    from .tilelang.linear import provide_mxfp8

    return provide_mxfp8(rows)


def rms_norm(hidden_size: int, eps: float = 1e-6) -> nn.Module:
    from .tilelang.norm import TileLangRMSNorm

    return TileLangRMSNorm(hidden_size, eps)


def is_native_linear_method(method) -> bool:
    from .tilelang.linear import TileLangFP8LinearMethod, TileLangLinearMethod

    return isinstance(method, (TileLangFP8LinearMethod, TileLangLinearMethod))


def dspark_confidence_head(hidden_size: int, markov_rank: int, prefix: str):
    """DSpark's acceptance-confidence head over the drafter's head hidden and the
    Markov embedding of the token before each draft."""
    from .tilelang.confidence import TileLangConfidenceHead

    return TileLangConfidenceHead(hidden_size, markov_rank, prefix=prefix)
