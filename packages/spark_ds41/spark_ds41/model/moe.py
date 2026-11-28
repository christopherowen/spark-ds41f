# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from collections.abc import Callable, Iterable
from inspect import signature
from itertools import islice

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.kernel import MEGA_MOE_BACKENDS
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_fused_post_pre_tilelang,
    mhc_post_tilelang,
    mhc_pre_broadcast_tilelang,
    mhc_pre_tilelang,
)
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
    spec_decode_needs_target_embed,
)
from vllm.model_executor.utils import set_weight_attrs
# weight_transfer not needed in DeepseekV4MoE
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.ubatching import dbo_current_ubatch_id


MEGA_MOE_BACKENDS = frozenset()
def is_fi_moe_ep_backend(x): return False
def validate_fi_moe_ep_config(x): return None
IMAGE_SENTINEL_BASE_ID = 128800
logger = init_logger(__name__)



class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        #
        # Block-FP8 shards in whole 128-blocks; cdiv rounds the per-rank block
        # count up so the linear's even TP split stays block-aligned, with the
        # trailing ranks zero-filled by load_weights.
        block_size = getattr(quant_config, "weight_block_size", None)
        if block_size is not None and not is_sequence_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            n_local = cdiv(intermediate_size // block_size[0], tp_size)
            intermediate_size = n_local * block_size[0] * tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        forward_mxfp8 = getattr(self.act_fn, "forward_mxfp8", None)
        if forward_mxfp8 is not None and getattr(
            self.down_proj, "tilelang_mxfp8_input", False
        ):
            from spark_ds41.tilelang.linear import provide_mxfp8

            act_q = forward_mxfp8(gate_up)
            with provide_mxfp8(act_q):
                out, _ = self.down_proj(act_q.rows)
            return out.view(*gate_up.shape[:-1], out.shape[-1])
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

def make_deepseek_v4_expert_params_mapping(
    num_experts: int,
) -> list[tuple[str, str, int, str]]:
    return [
        (
            "experts.w13_" if shard_id in ("w1", "w3") else "experts.w2_",
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in [
            ("w1", "w1"),
            ("w2", "w2"),
            ("w3", "w3"),
        ]
    ]

class DeepseekV4MegaMoEExperts(nn.Module):
    _symm_buffer_cache: dict[tuple[int, int, int, int, int, int, int, int], object] = {}

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        num_shared_experts: int = 0,
        prefix: str = "",
        num_logical_experts: int | None = None,
    ):
        super().__init__()
        self.prefix = prefix
        self.capture_fn: Callable[[torch.Tensor], None] | None = None
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_shared_experts = num_shared_experts
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        self.num_logical_experts = (
            num_logical_experts if num_logical_experts is not None else num_experts
        )

        self.eplb_state = EplbLayerState()

        weight_attrs = {"weight_loader": self.weight_loader}
        self.w13_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, weight_attrs)

        self.w13_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale, weight_attrs)
        self.w13_weight_scale.quant_method = "block"

        self.w2_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, weight_attrs)

        self.w2_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale, weight_attrs)
        self.w2_weight_scale.quant_method = "block"

        self._transformed_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._transformed_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._transformed_shared_l1_weights: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None
        self._transformed_shared_l2_weights: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None

        # Register in the static forward context so the custom-op wrapper
        # can look up this module by name from within a torch.compile graph.
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _map_global_expert_id(self, expert_id: int) -> list[int]:
        """Return local (per-rank) slot offsets where logical expert
        `expert_id` should land on this rank.
        """
        physical_ids: list[int] = []
        for p in range(self.experts_start_idx, self.experts_end_idx):
            if p % self.num_logical_experts == expert_id:
                physical_ids.append(p - self.experts_start_idx)
        return physical_ids

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local_expert_ids = self._map_global_expert_id(expert_id)
        if not local_expert_ids:
            return False if return_success else None

        loaded_any = False
        for local_expert_id in local_expert_ids:
            expert_data = param.data[local_expert_id]
            if shard_id in ("w1", "w3"):
                if "w13_" not in weight_name:
                    continue
                shard_offset = 0 if shard_id == "w1" else self.intermediate_size
                expert_data = expert_data.narrow(
                    0, shard_offset, self.intermediate_size
                )
            elif shard_id == "w2":
                if "w2_" not in weight_name:
                    continue
            else:
                raise ValueError(f"Unsupported expert shard id: {shard_id}")

            if expert_data.shape != loaded_weight.shape:
                raise ValueError(
                    f"DeepSeek V4 MegaMoE expert weight shape mismatch for "
                    f"{weight_name}: parameter shard {tuple(expert_data.shape)} "
                    f"vs checkpoint {tuple(loaded_weight.shape)}"
                )
            copy_weight(expert_data, loaded_weight)
            loaded_any = True

        if return_success:
            return loaded_any
        return None

    @staticmethod
    def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
        return (sf.to(torch.int32) << 23).view(torch.float32)

    def _check_runtime_supported(self) -> None:
        device = self.w13_weight.device
        if torch.cuda.get_device_capability(device)[0] != 10:
            raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")
        if self.hidden_size % 128 != 0 or self.intermediate_size % 128 != 0:
            raise ValueError(
                "DeepGEMM MegaMoE requires hidden and intermediate sizes "
                "to be multiples of 128."
            )

    @staticmethod
    def _deep_gemm_supports_shared_experts(deep_gemm) -> bool:
        """Check the Python API before touching a symmetric-memory group.

        This also gives users of an older precompiled vLLM wheel a safe serial
        fallback instead of failing halfway through multi-rank buffer setup.
        """
        try:
            buffer_params = signature(deep_gemm.get_symm_buffer_for_mega_moe).parameters
            kernel_params = signature(deep_gemm.fp8_fp4_mega_moe).parameters
        except (TypeError, ValueError):
            return False
        return (
            hasattr(deep_gemm, "get_block_m_for_mega_moe")
            and hasattr(deep_gemm, "transform_weights_for_mega_moe")
            and "num_shared_experts" in buffer_params
            and "shared_l1_weights" in kernel_params
            and "shared_l2_weights" in kernel_params
        )

    def _finalize_shared_expert_weights(
        self, deep_gemm, shared_experts: DeepseekV4MLP
    ) -> None:
        gate_up = shared_experts.gate_up_proj
        down = shared_experts.down_proj
        gate_up_weight = gate_up.weight.data
        gate_up_scale = (
            gate_up.weight_scale
            if hasattr(gate_up, "weight_scale")
            else gate_up.weight_scale_inv
        ).data
        down_weight = down.weight.data
        down_scale = (
            down.weight_scale
            if hasattr(down, "weight_scale")
            else down.weight_scale_inv
        ).data

        # MegaMoE's shared FP8 MMA consumes a 1x32 scale for every weight row,
        # while the checkpoint uses coarser block-FP8 scales (usually
        # 128x128). Build a dedicated, numerically equivalent scale view before
        # the generic linear post-load hook replaces the raw checkpoint scales
        # with its 128x128 DeepGEMM layout.
        checkpoint_scale_dtypes = (torch.float8_e8m0fnu, torch.uint8)
        if (
            gate_up_scale.dtype in checkpoint_scale_dtypes
            and down_scale.dtype in checkpoint_scale_dtypes
        ):
            gate_up_scale = self._prepare_shared_expert_scale(
                deep_gemm,
                gate_up,
                gate_up_scale,
                gate_up_weight.shape[0],
                gate_up_weight.shape[1],
            )
            down_scale = self._prepare_shared_expert_scale(
                deep_gemm,
                down,
                down_scale,
                down_weight.shape[0],
                down_weight.shape[1],
            )

        if gate_up_scale is None or down_scale is None:
            self.num_shared_experts = 0
            return

        shared_intermediate_size = self.intermediate_size * self.num_shared_experts
        expected_gate_up_shape = (
            2 * shared_intermediate_size,
            self.hidden_size,
        )
        expected_down_shape = (self.hidden_size, shared_intermediate_size)
        if (
            gate_up_weight.dtype != torch.float8_e4m3fn
            or down_weight.dtype != torch.float8_e4m3fn
            or gate_up_scale.dtype != torch.int32
            or down_scale.dtype != torch.int32
            or tuple(gate_up_weight.shape) != expected_gate_up_shape
            or tuple(down_weight.shape) != expected_down_shape
        ):
            logger.warning(
                "Disabling native MegaMoE shared-expert fusion for %s: expected "
                "replicated block-FP8 weights with gate_up=%s, down=%s, and "
                "DeepGEMM int32 scales; got gate_up=%s/%s/%s and down=%s/%s/%s.",
                self.prefix,
                expected_gate_up_shape,
                expected_down_shape,
                tuple(gate_up_weight.shape),
                gate_up_weight.dtype,
                gate_up_scale.dtype,
                tuple(down_weight.shape),
                down_weight.dtype,
                down_scale.dtype,
            )
            self.num_shared_experts = 0
            return

        transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
            (gate_up_weight, gate_up_scale),
            (down_weight, down_scale),
        )
        # L1 interleaving allocates a full copy. Re-home the loader Parameter on
        # that storage so the original 2*intermediate*hidden FP8 tensor can be
        # released instead of adding roughly 0.7 GiB per rank on DSV4-Flash.
        # The generic linear post-load hook may still repack the serial scales,
        # but this shared MLP is never called after native fusion is enabled.
        gate_up.weight.data = transformed_l1[0]
        self._transformed_shared_l1_weights = (
            gate_up.weight.data,
            transformed_l1[1],
        )
        self._transformed_shared_l2_weights = transformed_l2

    def _prepare_shared_expert_scale(
        self,
        deep_gemm,
        linear: nn.Module,
        scale: torch.Tensor,
        mn: int,
        k: int,
    ) -> torch.Tensor | None:
        block_size = getattr(linear, "weight_block_size", None)
        if block_size is None or len(block_size) != 2:
            logger.warning(
                "Disabling native MegaMoE shared-expert fusion for %s: "
                "shared FP8 weight block size is unavailable.",
                self.prefix,
            )
            return None

        block_m, block_k = block_size
        expected_shape = (
            (mn + block_m - 1) // block_m,
            (k + block_k - 1) // block_k,
        )
        if block_k % 32 != 0 or tuple(scale.shape) != expected_shape:
            logger.warning(
                "Disabling native MegaMoE shared-expert fusion for %s: "
                "cannot convert shared scale shape %s with block size %s "
                "to MegaMoE's 1x32 layout for weight (%d, %d).",
                self.prefix,
                tuple(scale.shape),
                tuple(block_size),
                mn,
                k,
            )
            return None

        scale_fp32 = self._ue8m0_uint8_to_float(scale.view(torch.uint8))
        scale_1x32 = (
            scale_fp32.repeat_interleave(block_m, dim=0)
            .repeat_interleave(block_k // 32, dim=1)[:mn, : k // 32]
            .contiguous()
        )
        # The grouped API is used with a singleton dimension to request the
        # MN-major, TMA-aligned packed-UE8M0 strides, then squeezed back to the
        # 2D layout required for a shared expert.
        return deep_gemm.transform_sf_into_required_layout(
            scale_1x32.unsqueeze(0),
            mn,
            k,
            (1, 32),
            1,
        ).squeeze(0)

    def finalize_weights(self, shared_experts: DeepseekV4MLP | None = None) -> None:
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        if self._transformed_l1_weights is None:
            self._check_runtime_supported()
            # MegaMoE's 1x32 activation scales need 16-byte TMA rows, so
            # pad each gate/up half and the down projection to a multiple of 512.
            padded_size = (self.intermediate_size + 511) // 512 * 512
            padding = padded_size - self.intermediate_size
            if padding:
                for param in (self.w13_weight, self.w13_weight_scale):
                    gate_up = param.data.unflatten(1, (2, self.intermediate_size))
                    param.data = torch.nn.functional.pad(
                        gate_up, (0, 0, 0, padding)
                    ).flatten(1, 2)
                self.w2_weight.data = torch.nn.functional.pad(
                    self.w2_weight.data, (0, padding // 2)
                )
                self.w2_weight_scale.data = torch.nn.functional.pad(
                    self.w2_weight_scale.data, (0, padding // 32)
                )
                self.intermediate_size = padded_size
            w13_scale = deep_gemm.transform_sf_into_required_layout(
                self._ue8m0_uint8_to_float(self.w13_weight_scale.data).contiguous(),
                2 * self.intermediate_size,
                self.hidden_size,
                (1, 32),
                self.num_local_experts,
            )
            w2_scale = deep_gemm.transform_sf_into_required_layout(
                self._ue8m0_uint8_to_float(self.w2_weight_scale.data).contiguous(),
                self.hidden_size,
                self.intermediate_size,
                (1, 32),
                self.num_local_experts,
            )
            self._transformed_l1_weights, self._transformed_l2_weights = (
                deep_gemm.transform_weights_for_mega_moe(
                    (self.w13_weight.data.view(torch.int8).contiguous(), w13_scale),
                    (self.w2_weight.data.view(torch.int8).contiguous(), w2_scale),
                )
            )
            # Drop the original loader-side parameters: the MegaMoE kernels only
            # consume the transformed views above. transform_weights_for_mega_moe
            # allocates a fresh tensor for the L1 weight (see
            # _interleave_l1_weights) and fresh SF tensors for L1/L2; the L2
            # weight is the only tensor that aliases the original storage, and
            # _transformed_l2_weights still holds it, so the storage stays live
            # after we drop the Parameter.
            self.w13_weight = None
            self.w13_weight_scale = None
            self.w2_weight = None
            self.w2_weight_scale = None

        if shared_experts is None or self.num_shared_experts == 0:
            return
        if self._transformed_shared_l1_weights is not None:
            return
        if not self._deep_gemm_supports_shared_experts(deep_gemm):
            logger.warning_once(
                "Disabling native MegaMoE shared-expert fusion because the "
                "installed DeepGEMM Python API is older than the vLLM "
                "source. Rebuild the vendored _deep_gemm_C extension to enable it.",
            )
            self.num_shared_experts = 0
            return
        self._finalize_shared_expert_weights(deep_gemm, shared_experts)

    @property
    def has_fused_shared_experts(self) -> bool:
        return self._transformed_shared_l1_weights is not None

    def get_symm_buffer(self):
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        group = get_ep_group().device_group
        device = torch.accelerator.current_device_index()
        key = (
            id(group),
            device,
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_shared_experts if self.has_fused_shared_experts else 0,
        )
        symm_buffer = self._symm_buffer_cache.get(key)
        if symm_buffer is None:
            symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                num_shared_experts=(
                    self.num_shared_experts if self.has_fused_shared_experts else 0
                ),
            )
            self._symm_buffer_cache[key] = symm_buffer
        return symm_buffer

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.eplb_state.set_layer_state(
            moe_layer_idx,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )

    def get_expert_weights(self) -> list[torch.Tensor]:
        self.finalize_weights()
        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None

        def _to_eplb_view(name: str, t: torch.Tensor) -> torch.Tensor:
            """Return a (num_local_experts, -1) view with contiguous memory layout."""
            assert t.shape[0] == self.num_local_experts
            if t.is_contiguous():
                return t.view(self.num_local_experts, -1)
            elif t.dim() == 3 and t.stride(1) == 1 and t.stride(2) == t.shape[1]:
                # scales have shape (E, M, N) with memory layout (E, N, M)
                back = torch.transpose(t, 1, 2)
                assert back.is_contiguous()
                return back.view(self.num_local_experts, -1)

            raise AssertionError(
                f"DSv4 EPLB {name}: non-contiguous expert tensor with "
                f"unexpected layout shape={tuple(t.shape)} "
                f"stride={tuple(t.stride())} dtype={t.dtype}"
            )

        return [
            _to_eplb_view("l1_packed", self._transformed_l1_weights[0]),
            _to_eplb_view("l1_scale", self._transformed_l1_weights[1]),
            _to_eplb_view("l2_weight", self._transformed_l2_weights[0]),
            _to_eplb_view("l2_scale", self._transformed_l2_weights[1]),
        ]

    def update_expert_map(self) -> None:
        pass

    @property
    def layer_id(self) -> int:
        return extract_layer_index(self.prefix)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
        fast_math: bool = True,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"DeepSeek V4 MegaMoE got {hidden_states.shape[0]} tokens, "
                f"but the symmetric buffer was sized for {self.max_num_tokens}."
            )
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)

        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()

        symm_buffer = self.get_symm_buffer()
        num_tokens = hidden_states.shape[0]
        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                is_padding = is_padding[:num_tokens]

        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        # EPLB: map logical expert IDs to physical replicas and record load.
        eplb_state = self.eplb_state
        if eplb_state.logical_to_physical_map is not None:
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            if is_padding is not None:
                topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=eplb_state.expert_load_view,
                logical_to_physical_map=eplb_state.logical_to_physical_map,
                logical_replica_count=eplb_state.logical_replica_count,
                record_enabled=eplb_state.should_record_tensor,
                num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[
                    dbo_current_ubatch_id()
                ]
                if eplb_state.num_unpadded_tokens_tensors is not None
                else None,
            )

        shared_x_sf = None
        shared_block_m = None
        if self.has_fused_shared_experts:
            shared_x_sf = symm_buffer.shared_l1_acts_sf
            shared_block_m = deep_gemm.get_block_m_for_mega_moe(
                get_ep_group().world_size,
                self.num_experts,
                symm_buffer.num_max_tokens_per_rank,
                num_tokens,
                self.top_k,
                "fp8xfp4",
            )

        prepare_megamoe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
            is_padding=is_padding,
            shared_x_sf=shared_x_sf,
            shared_block_m=shared_block_m,
        )

        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None
        if self.has_fused_shared_experts:
            deep_gemm.fp8_fp4_mega_moe(
                y,
                self._transformed_l1_weights,
                self._transformed_l2_weights,
                symm_buffer,
                shared_l1_weights=self._transformed_shared_l1_weights,
                shared_l2_weights=self._transformed_shared_l2_weights,
                activation_clamp=activation_clamp,
                fast_math=fast_math,
            )
        else:
            deep_gemm.fp8_fp4_mega_moe(
                y,
                self._transformed_l1_weights,
                self._transformed_l2_weights,
                symm_buffer,
                activation_clamp=activation_clamp,
                fast_math=fast_math,
            )
        return y

class DeepseekV4MoE(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_sequence_parallel: bool = False,
        *,
        num_hash_layers: int | None = None,
        n_routed_experts: int | None = None,
        n_activated_experts: int | None = None,
        gate: nn.Module | None = None,
        image_sentinel_lo: int | None = None,
        image_sentinel_count: int = 5,
        router_cls: type | None = None,
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        # A FusedTopKBiasRouter subclass that routes these experts instead of
        # vLLM's choice for the config (the DS4.1 kernel family picks it).
        self.router_cls = router_cls
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix
        self.use_sequence_parallel = use_sequence_parallel
        moe_backend = vllm_config.kernel_config.moe_backend
        validate_fi_moe_ep_config(vllm_config)
        self.use_mega_moe = moe_backend in MEGA_MOE_BACKENDS
        if self.use_mega_moe and router_cls is not None:
            raise ValueError("MegaMoE routes in its own kernel; it takes no router")
        self.use_fi_mega_moe = is_fi_moe_ep_backend(moe_backend)
        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently requires expert parallel. "
                "Enable it with --enable-expert-parallel, or pick a different "
                "moe backend."
            )

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size

        self.n_routed_experts = (
            config.n_routed_experts if n_routed_experts is None else n_routed_experts
        )
        self.n_activated_experts = (
            config.num_experts_per_tok
            if n_activated_experts is None
            else n_activated_experts
        )
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
        if self.use_mega_moe and self.scoring_func != "sqrtsoftplus":
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently supports sqrtsoftplus routing only."
            )
        if self.use_mega_moe and getattr(config, "expert_dtype", "fp4") != "fp4":
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE only supports fp4 experts; got expert_dtype="
                f"{config.expert_dtype!r}. Drop --kernel-config moe_backend="
                f"{moe_backend} for this checkpoint."
            )

        self.gate = (
            gate
            if gate is not None
            else GateLinear(
                input_size=config.hidden_size,
                output_size=self.n_routed_experts,
                bias=False,
                out_dtype=torch.float32,
                params_dtype=(
                    torch.float32
                    if getattr(config, "router_dtype", None) == "float32"
                    else None
                ),
                prefix=f"{prefix}.gate",
            )
        )

        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        self.gate.bias_vl = None
        # Image tokens borrow five consecutive reserved in-vocab ids starting
        # at IMAGE_SENTINEL_BASE_ID; 0 disables vision routing (text model).
        self.image_sentinel_lo = (
            image_sentinel_lo
            if image_sentinel_lo is not None
            else IMAGE_SENTINEL_BASE_ID
            if getattr(config, "vision_n_layers", 0) > 0
            else 0
        )
        self.image_sentinel_count = image_sentinel_count
        if num_hash_layers is None:
            num_hash_layers = getattr(config, "num_hash_layers", 0)
        is_hash_moe = extract_layer_index(prefix) < num_hash_layers
        self.hash_indices_dtype = torch.int64 if self.use_mega_moe else torch.int32
        if is_hash_moe:
            # hash MoE doesn't use e_score_correction_bias
            # Use randint instead of empty to avoid garbage values causing
            # invalid memory access in dummy mode (--load-format="dummy")
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    self.n_routed_experts,
                    (config.vocab_size, self.n_activated_experts),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        if getattr(config, "topk_method", None) == "noaux_tc" and (
            not is_hash_moe or self.image_sentinel_lo > 0
        ):
            # Vision checkpoints ship a gate bias on hash layers too (it is
            # unused for routing there; image tokens use bias_vl instead).
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if getattr(config, "vision_n_layers", 0) > 0:
            # Vision checkpoints route image sentinel tokens with bias_vl
            # instead of e_score_correction_bias / the hash table. Created on
            # every MoE layer, hash layers included.
            self.gate.bias_vl = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=self.use_mega_moe,
                is_sequence_parallel=use_sequence_parallel,
                prefix=f"{prefix}.shared_experts",
            )

        if self.use_mega_moe:
            self._init_mega_moe_experts(vllm_config, config, prefix)
        else:
            self._init_fused_moe_experts(vllm_config, config, quant_config, prefix)

    def _init_mega_moe_experts(
        self,
        vllm_config: VllmConfig,
        config,
        prefix: str,
    ) -> None:
        self.ep_group = get_ep_group()
        self.ep_size = self.ep_group.world_size
        self.ep_rank = self.ep_group.rank_in_group

        eplb_config = vllm_config.parallel_config.eplb_config
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_shared_experts = config.n_shared_experts or 0
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        assert self.n_physical_experts % self.ep_size == 0, (
            f"n_physical_experts={self.n_physical_experts} must be divisible by "
            f"ep_size={self.ep_size}. Adjust num_redundant_experts."
        )
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.n_local_experts = self.n_local_physical_experts
        self.experts_start_idx = self.physical_expert_start
        self.experts_end_idx = self.physical_expert_end

        # Native DeepGEMM fusion requires each EP rank to own the complete
        # shared MLP. Sequence parallel replicates those weights while sharding
        # tokens. TP=1 is also naturally replicated. With PP+TP the shared MLP
        # remains tensor-sharded, so retain the serial path. The FlashInfer
        # megakernel has no shared-expert fusion, so it keeps the serial path.
        fuse_shared_experts = bool(
            self.shared_experts is not None
            and not envs.VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION
            and (self.use_sequence_parallel or self.tp_size == 1)
            and not self.use_fi_mega_moe
        )

        activation_clamp = (
            float(self.swiglu_limit) if self.swiglu_limit is not None else None
        )
        if self.use_fi_mega_moe:
            # Deferred: fi_moe subclasses DeepseekV4MegaMoEExperts, so a
            # module-level import here would be circular.
            from vllm.models.deepseek_v4.nvidia.fi_moe import (
                DeepseekV4MegaMoEExpertsFI,
            )

            experts_cls: type[DeepseekV4MegaMoEExperts] = DeepseekV4MegaMoEExpertsFI
        else:
            experts_cls = DeepseekV4MegaMoEExperts
        expert_kwargs: dict[str, typing.Any] = dict(
            num_experts=self.n_physical_experts,
            num_local_experts=self.n_local_physical_experts,
            experts_start_idx=self.physical_expert_start,
            num_logical_experts=self.n_logical_experts,
            top_k=self.n_activated_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_shared_experts=(self.n_shared_experts if fuse_shared_experts else 0),
            prefix=f"{prefix}.experts",
        )
        if self.use_fi_mega_moe:
            expert_kwargs["activation_clamp"] = activation_clamp
        self.experts = experts_cls(vllm_config, **expert_kwargs)

    def _init_fused_moe_experts(
        self,
        vllm_config: VllmConfig,
        config,
        quant_config,
        prefix: str,
    ) -> None:
        parallel_config = vllm_config.parallel_config
        self.tp_rank = get_tensor_model_parallel_rank()

        eplb_config = parallel_config.eplb_config
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_shared_experts = config.n_shared_experts or 0
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts

        router = None
        runner_gate = self.gate
        if self.router_cls is not None:
            if parallel_config.enable_eplb:
                raise ValueError(f"{self.router_cls.__name__} does not support EPLB")
            extra = {}
            if getattr(self.router_cls, "routes_hidden_states", False):
                # The router runs the gate itself; the runner passes it the
                # hidden states in place of logits.
                extra["gate"], runner_gate = self.gate, None
            router = self.router_cls(
                top_k=self.n_activated_experts,
                global_num_experts=self.n_routed_experts,
                e_score_correction_bias=self.gate.e_score_correction_bias,
                renormalize=config.norm_topk_prob,
                routed_scaling_factor=self.routed_scaling_factor,
                scoring_func=self.scoring_func,
                hash_indices_table=self.gate.tid2eid,
                bias_vl=getattr(self.gate, "bias_vl", None),
                image_sentinel_lo=self.image_sentinel_lo,
                    **extra,
            )
        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            gate=runner_gate,
            router=router,
            num_experts=self.n_routed_experts,
            top_k=self.n_activated_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            bias_vl=getattr(self.gate, "bias_vl", None),
            image_sentinel_lo=self.image_sentinel_lo,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=eplb_config.num_redundant_experts,
            is_sequence_parallel=self.use_sequence_parallel,
        )
        self.n_local_physical_experts = self.experts.routed_experts.local_num_experts
        self.n_local_experts = self.n_local_physical_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        *,
        reduce_scatter_rows: int | None = None,
    ) -> torch.Tensor:
        """Route ``hidden_states`` through the experts.

        ``reduce_scatter_rows`` (sequence parallelism) replaces the final TP
        all-reduce with a dim-0 reduce-scatter of the output padded to that
        many rows; the result holds this rank's rows only.
        """
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
        # getattr for test doubles that fake the gate module.
        bias_vl = getattr(self.gate, "bias_vl", None)
        if bias_vl is not None and input_ids is None:
            raise ValueError("DeepSeek V4 vision MoE routing requires input_ids.")

        if not self.use_mega_moe:
            return self._forward_fused_moe(
                hidden_states, input_ids, reduce_scatter_rows
            )
        if reduce_scatter_rows is not None:
            raise NotImplementedError("DeepSeek V4 MegaMoE output reduce-scatter")

        org_shape = hidden_states.shape
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias.data
            if self.gate.e_score_correction_bias is not None
            else None,
            topk=self.n_activated_experts,
            renormalize=self.renormalize,
            indices_type=self.hash_indices_dtype,
            input_tokens=input_ids,
            hash_indices_table=self.gate.tid2eid,
            routed_scaling_factor=self.routed_scaling_factor,
            bias_vl=bias_vl.data if bias_vl is not None else None,
            image_sentinel_lo=self.image_sentinel_lo if bias_vl is not None else 0,
        )
        activation_clamp = (
            float(self.swiglu_limit) if self.swiglu_limit is not None else None
        )
        final_hidden_states = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
            activation_clamp=activation_clamp,
        )

        if (
            self.shared_experts is not None
            and not self.experts.has_fused_shared_experts
        ):
            shared_output = self.shared_experts(hidden_states)
            final_hidden_states += shared_output

        return final_hidden_states.view(org_shape)

    def _forward_fused_moe(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        reduce_scatter_rows: int | None = None,
    ) -> torch.Tensor:
        org_shape = hidden_states.shape
        if reduce_scatter_rows is not None:
            return self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
                input_ids=input_ids,
                reduce_scatter_rows=reduce_scatter_rows,
            ).to(hidden_states.dtype)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=hidden_states,
            input_ids=input_ids,
        )

        return final_hidden_states.to(hidden_states.dtype).view(org_shape)

    def finalize_mega_moe_weights(self) -> None:
        if self.use_mega_moe:
            self.experts.finalize_weights(self.shared_experts)
