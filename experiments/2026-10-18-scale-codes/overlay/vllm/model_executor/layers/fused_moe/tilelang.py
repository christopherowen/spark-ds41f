# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang MXFP4 x MXFP8 routed experts for SM12x (``--moe-backend tilelang``).

One layer runs, all from the modular kernel's workspace:

1. dispatch: routes grouped by expert in the contiguous grouped-GEMM layout;
2. MXFP8 cast of the tokens (TileKernels ``per_token_cast``);
3. grouped gate/up GEMM over the dispatched rows, gathering tokens with
   ``cp.async`` from tile-major expert weights;
4. SwiGLU with DeepSeek's clamp, the route weight and the MXFP8 cast of the
   down projection's input, fused (TileKernels);
5. grouped down GEMM;
6. combine: each token's expert outputs summed in slot order in FP32.

Route weights are applied before the down projection, as DeepSeek does. No
step uses atomics on results, so a token's output does not depend on the
rest of the batch.
"""

from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp8Dynamic,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang

# Output columns come in 128-wide tiles; K blocks are multiples of 64.
_BLOCK_N = 128
_K_ALIGN = 64


def _kernels():
    from vllm.models.deepseek_v4_1.tilelang import moe

    return moe


def _output_dtype() -> torch.dtype:
    from vllm.models.deepseek_v4_1.kernels import moe_output_dtype

    return moe_output_dtype()


class TileLangExperts(mk.FusedMoEExpertsModular):
    """MXFP4 experts with MXFP8 activations, in TileLang and TileKernels."""

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        self._plan = None
        # The 64-wide K tails of each GEMM's tile-major weights (None without one).
        self._tails: tuple[torch.Tensor | None, ...] = (None, None)

    # -- support --------------------------------------------------------

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
            and has_tilelang()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, kMxfp8Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_ep
            and moe_parallel_config.ep_size == 1
            and not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    @staticmethod
    def _supports_batch_invariance() -> bool:
        return True

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: mk.FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if moe_config.has_bias:
            return False, "kernel does not support expert biases"
        if moe_config.in_dtype != torch.bfloat16:
            return False, f"kernel does not support {moe_config.in_dtype} activations"
        intermediate = moe_config.intermediate_size_per_partition
        # Hidden is the down projection's N; the gate/up N is 2 * intermediate.
        if moe_config.hidden_dim % _BLOCK_N or intermediate % _K_ALIGN:
            reason = (
                f"hidden size must be a multiple of {_BLOCK_N} and per-rank "
                f"intermediate size a multiple of {_K_ALIGN}"
            )
            return False, reason
        return mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    # Set by the MoE runner around one call whose shared expert runs before the
    # routed experts, or beside them on the aux stream: the combine adds
    # ``combine_addend`` (the shared expert's output, or a callable that waits for
    # it and returns it) to the FP32 sum and rounds the rank's partial once, in
    # BF16 out.
    fold_shared: bool = False
    combine_addend: torch.Tensor | Callable[[], torch.Tensor] | None = None
    # The plan resolves a callable addend just before its combine launch, so an
    # aux-stream shared expert overlaps the routed experts' GEMMs.
    folds_deferred_shared_output: bool = True

    @property
    def folds_shared_output(self) -> bool:
        """Whether the combine can add the shared expert's output itself: when it
        would otherwise write an FP32 partial for the runner to add and round."""
        return _output_dtype() == torch.float32

    @property
    def output_dtype(self) -> torch.dtype:
        """FP32 under the MoE FP32 combine switch: the runner adds the shared
        expert in FP32 and rounds the rank's partial once before the reduction.
        With the shared output folded into the combine, the layer's dtype."""
        return self.moe_config.in_dtype if self.fold_shared else _output_dtype()

    # -- weights ----------------------------------------------------------

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.apply_router_weight_on_input:
            raise ValueError("TileLang MoE applies route weights inside the experts")
        moe = _kernels()
        w13, w2 = layer.w13_weight, layer.w2_weight  # [E, 2I, H/2], [E, H, I/2]
        limit = self.quant_config.gemm1_clamp_limit
        # The plan's geometry; its kernels are declared when the worker prepares
        # kernels (``declare_kernels``).
        self._plan = moe.MoeExpertsV41Plan(
            w13.shape[0],
            self.moe_config.experts_per_token,
            w2.shape[1],
            w2.shape[2] * 2,
            swiglu_limit=self.moe_config.swiglu_limit if limit is None else limit,
            out_dtype=self.output_dtype,
        )
        # Tile-major copies replace the checkpoint layout, so each GEMM tile's
        # weights and scale codes are contiguous. A K that ends 64 past a multiple
        # of 128 (TP4's 576-wide rows) keeps its last 64 columns as compact tails.
        (t13, tail13, codes13, base13), (t2, tail2, codes2, base2) = (
            self._plan.prepare_weights(
                w13, layer.w13_weight_scale, w2, layer.w2_weight_scale
            )
        )
        for name, tensor in (
            ("w13_weight", t13),
            ("w13_weight_scale", codes13),
            ("w2_weight", t2),
            ("w2_weight_scale", codes2),
        ):
            replace_parameter(layer, name, tensor)
        self._tails = (tail13, tail2)
        self._bases = (base13, base2)
        self.quant_config._w1.scale = layer.w13_weight_scale
        self.quant_config._w2.scale = layer.w2_weight_scale

    def declare_kernels(self) -> None:
        """Declare the plan's kernels (the layer's MoE method calls this after
        loading, when the worker prepares kernels)."""
        self._plan.declare_kernels()

    # -- execution ----------------------------------------------------------

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        # The weights are tile-major; report the logical geometry.
        plan = self._plan
        tokens = int(a1.shape[0] if a1.ndim == 2 else a1.shape[1])
        return (
            plan.num_experts,
            tokens,
            2 * plan.intermediate,
            plan.hidden,
            int(topk_ids.shape[1]),
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del N, topk, global_num_experts, local_num_experts, expert_tokens_meta
        nbytes = _kernels().workspace_bytes(self._plan.workspace_specs(M))
        itemsize = self.moe_config.in_dtype.itemsize
        return (0,), ((nbytes + itemsize - 1) // itemsize,), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        del global_num_experts, a1q_scale, a2_scale, workspace13, expert_tokens_meta
        if expert_map is not None:
            raise ValueError("TileLang TP MoE does not support expert maps")
        if apply_router_weight_on_input:
            raise ValueError("TileLang MoE applies route weights inside the experts")
        if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
            raise TypeError("TileLang MoE topk_weights must be contiguous float32")
        if not topk_ids.is_contiguous():
            raise TypeError("TileLang MoE topk_ids must be contiguous")
        tail13, tail2 = self._tails
        base13, base2 = self._bases
        weights = (
            (w1, tail13, self.w1_scale, base13),
            (w2, tail2, self.w2_scale, base2),
        )
        workspace = workspace2.view(-1).view(torch.uint8)
        shared = None
        if self.fold_shared:
            assert self.combine_addend is not None, "the shared output must be given"
            shared = self.combine_addend
        self._plan.run(
            hidden_states, topk_ids, topk_weights, weights, output, workspace, shared
        )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for TileLangExperts")
