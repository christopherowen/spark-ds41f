# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, get_args

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    resolve_quant_method,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

QuantizationMethods = Literal[
    "awq",
    "auto_awq",
    "fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "modelopt_mxfp8",
    "modelopt_mixed",
    "auto_gptq",
    "gptq",
    "gptq_marlin",
    "awq_marlin",
    "humming",
    "compressed-tensors",
    "experts_int8",
    "quark",
    "moe_wna16",
    "exl3",
    "torchao",
    "inc",
    "mxfp4",
    "gpt_oss_mxfp4",
    "deepseek_v4_fp8",
    "deepseek_v41_fp8",
    "mxfp4_csf",
    "online",
    # Below are online quant shorthand names (see vllm.config.quantization).
    # Listed here as strings to avoid a circular import; kept in sync with
    # _ONLINE_SHORTHANDS by the assertion in get_quantization_config().
    "fp8_per_tensor",
    "fp8_per_block",
    "fp8_per_channel",
    "int8_per_channel_weight_only",
    "nvfp4_per_token",
    "mxfp8",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))

DEPRECATED_QUANTIZATION_METHODS = [
    "fbgemm_fp8",
    "fp_quant",
]

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from vllm.model_executor.layers.quantization import (
        ...     register_quantization_config,
        ... )
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import (
        ...     QuantizationConfig,
        ... )
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            logger.debug(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)
            # Automatically assume the custom quantization config is supported
            if sq := current_platform.supported_quantization:
                sq.append(quantization)

        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError(
                "The quantization config must be a subclass of `QuantizationConfig`."
            )
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper


def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    if quantization in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG:
        return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]

    # Lazy per-method imports avoid importing heavy native/Triton backends for
    # unrelated quantization methods during ModelConfig verification.
    if quantization in ("awq", "awq_marlin", "auto_awq"):
        from .auto_awq import AutoAWQConfig

        return AutoAWQConfig
    if quantization in ("auto_gptq", "gptq", "gptq_marlin"):
        from .auto_gptq import AutoGPTQConfig

        return AutoGPTQConfig
    if quantization == "fp8":
        from .fp8 import Fp8Config

        return Fp8Config
    if quantization == "fbgemm_fp8":
        from .fbgemm_fp8 import FBGEMMFp8Config

        return FBGEMMFp8Config
    if quantization == "fp_quant":
        from .fp_quant import FPQuantConfig

        return FPQuantConfig
    if quantization == "modelopt":
        from .modelopt import ModelOptFp8Config

        return ModelOptFp8Config
    if quantization == "modelopt_fp4":
        from .modelopt import ModelOptNvFp4Config

        return ModelOptNvFp4Config
    if quantization in ("modelopt_mxfp8", "mxfp8"):
        from .modelopt import ModelOptMxFp8Config

        return ModelOptMxFp8Config
    if quantization == "modelopt_mixed":
        from .modelopt import ModelOptMixedPrecisionConfig

        return ModelOptMixedPrecisionConfig
    if quantization == "compressed-tensors":
        from .compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )

        return CompressedTensorsConfig
    if quantization == "experts_int8":
        from .experts_int8 import ExpertsInt8Config

        return ExpertsInt8Config
    if quantization == "quark":
        from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

        return QuarkConfig
    if quantization == "moe_wna16":
        from .moe_wna16 import MoeWNA16Config

        return MoeWNA16Config
    if quantization == "exl3":
        from .exl3 import Exl3Config

        return Exl3Config
    if quantization == "torchao":
        from .torchao import TorchAOConfig

        return TorchAOConfig
    if quantization == "inc":
        from .inc import INCConfig

        return INCConfig
    if quantization == "mxfp4":
        from .mxfp4 import Mxfp4Config

        return Mxfp4Config
    if quantization == "gpt_oss_mxfp4":
        from .mxfp4 import GptOssMxfp4Config

        return GptOssMxfp4Config
    if quantization == "deepseek_v4_fp8":
        if current_platform.is_cuda() or current_platform.is_rocm():
            from vllm.models.deepseek_v41.quant_config import (
                DeepseekV4FP8Config as UpstreamDeepseekV41FP8Config,
            )

            return UpstreamDeepseekV41FP8Config
        from vllm.models.deepseek_v4 import DeepseekV4FP8Config

        return DeepseekV4FP8Config
    if quantization == "deepseek_v41_fp8":
        from vllm.models.deepseek_v4_1.quant_config import DeepseekV41FP8Config

        return DeepseekV41FP8Config
    if quantization == "mxfp4_csf":
        from .mxfp4_csf import Mxfp4CsfConfig

        return Mxfp4CsfConfig
    if quantization == "humming":
        from .humming import HummingConfig

        return HummingConfig

    from vllm.config.quantization import _ONLINE_SHORTHANDS

    from .online.base import OnlineQuantizationConfig

    if quantization == "online" or quantization in _ONLINE_SHORTHANDS:
        return OnlineQuantizationConfig

    raise ValueError(f"Invalid quantization method: {quantization}")


__all__ = [
    "QuantizationConfig",
    "resolve_quant_method",
    "QuantizationMethods",
    "get_quantization_config",
    "register_quantization_config",
    "QUANTIZATION_METHODS",
]
