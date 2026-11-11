# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark draft model for DeepSeek-V4.1 (semi-autoregressive speculative decoding).

See: qwen3_dspark.py for base architecture. This one is specialized to the
DSV4.1 DSpark, which reuses the target model's architecture similarly to MTP.
Its weights ship in the target checkpoint under the ``mtp.{0,1,2}.*`` prefix.

To implement non-causal attention, we leverage the sparse attention implementation to
include the future query tokens in the top-k indices for each query token.
"""

import copy
from collections.abc import Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor,
    _is_packed_head,
    _is_plain_bf16_head,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.weight_transfer import copy_weight
from vllm.models.deepseek_v4_1.common.engram import _PaddedColumnParallelLinear
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.cudagraph_utils import (
    CudaGraphManager,
    dense_varlen_decode_shapes,
)

from .. import kernels, l2_prefetch
from ..tilelang import columns, dspark_context, prepare
from .model import (
    DeepseekV4DecoderLayer,
    _linear_scale_param_name,
    build_l2_prefetch_plans,
)

logger = init_logger(__name__)


def _vocab_parallel_greedy() -> bool:
    """Use the fused vocabulary-shard path at every TP size.

    One rank owns the full vocabulary and its pair gather is a view. It still
    needs fused Markov selection/embedding and the final confidence layout.
    """
    return True


# MoE expert scale suffix differs by expert dtype (mirrors deepseek_v4 loaders):
# fp4 experts register ``.weight_scale``; block-fp8 experts ``.weight_scale_inv``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def _context_kv_projection(attn: nn.Module):
    """The KV rows of ``attn``'s fused Q-A/KV weight as one projection."""
    from ..tilelang.linear import TileLangBlock32Rows

    projection = TileLangBlock32Rows(attn.fused_wqa_wkv, attn.q_lora_rank)
    projection.tilelang_mxfp8_input = True
    return projection


def _context_capture_sizes(vllm_config: VllmConfig, limit: int) -> list[int]:
    """The context graphs' row counts: the target's decode graph sizes up to ``limit``.

    A context graph runs at the target step's live row count. The target captures
    its FULL decode graphs at the capture ladder and at every exact low-concurrency
    and full-depth speculative total (``dense_varlen_decode_shapes`` at one
    request's verification width), so these sizes pad the context no further than
    the target pads itself: a full-depth step keeps its 6, 24 or 48 rows through
    main_proj, its gather and the draft layers' KV front instead of 8, 32 or 64.
    The count is the target's own decode graph count below ``limit``, which is
    always captured: the largest decode step.
    """
    if limit <= 0:
        return []
    decode_query_len = 1 + vllm_config.num_verification_tokens
    sizes = {limit}
    sizes.update(
        num_tokens
        for _, num_tokens in dense_varlen_decode_shapes(
            vllm_config.scheduler_config.max_num_seqs,
            decode_query_len,
            limit,
            vllm_config.uniform_decode_query_len,
        )
    )
    sizes.update(
        size
        for size in vllm_config.compilation_config.cudagraph_capture_sizes or ()
        if size <= limit
    )
    return sorted(sizes)


class DSparkContextCudaGraphs:
    """Serial, draft-workspace-lane owner for auxiliary projection and KV prep.

    Only bounded decode capacities are captured. Live source tensors are copied,
    never retained by a graph; rejected rows keep their caller-provided PAD slots.
    Larger prefills continue through the eager hooks with their actual row count.
    """

    def __init__(
        self,
        model: nn.Module,
        vllm_config: VllmConfig,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mappings: torch.Tensor,
        layer_group_idx: list[int] | None,
        max_decode_tokens: int,
    ) -> None:
        self.model = model
        self.hidden_states = hidden_states
        self.positions = positions
        self.slot_mappings = slot_mappings
        self.layer_group_idx = layer_group_idx
        self.width = model.config.hidden_size
        self.num_aux = len(model.config.dspark_target_layer_ids)
        limit = min(
            max_decode_tokens,
            hidden_states.shape[0],
            vllm_config.compilation_config.max_cudagraph_capture_size,
        )
        capacities = _context_capture_sizes(vllm_config, limit)
        self.aux = torch.zeros(
            (limit, self.width * self.num_aux),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        config = copy.copy(vllm_config)
        config.compilation_config = copy.copy(vllm_config.compilation_config)
        config.compilation_config.cudagraph_capture_sizes = capacities
        config.compilation_config.max_cudagraph_capture_size = limit
        self.manager = CudaGraphManager(
            config, hidden_states.device, CUDAGraphMode.FULL, decode_query_len=1
        )

    def _forward(self, capacity: int) -> None:
        # The norm writes the combined rows into the context buffer itself: no
        # copy between it and the context KV projections.
        self.model.combine_hidden_states(
            self.aux[:capacity], out=self.hidden_states[:capacity]
        )
        slots = (
            self.slot_mappings[0, :capacity]
            if self.layer_group_idx is None
            else [self.slot_mappings[i, :capacity] for i in self.layer_group_idx]
        )
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:capacity], self.positions[:capacity], slots
        )

    def capture(self) -> None:
        self.positions.zero_()
        self.slot_mappings.fill_(PAD_SLOT_ID)
        # Reserve all native scratch and resolve every planned capacity BEFORE
        # the first graph. Never unlock a serving workspace to make capture fit.
        for capacity in reversed(
            self.manager.compilation_config.cudagraph_capture_sizes
        ):
            self._forward(capacity)
        self.model._context_main_cast = None
        if torch.cuda.is_available():
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()

        def create_forward_fn(desc, warmup):
            return lambda mode: self._forward(desc.num_tokens)

        self.manager.capture(
            create_forward_fn, progress_bar_desc="Capturing DSpark context CUDA graphs"
        )

    def can_run(self, num_tokens: int) -> bool:
        desc = self.manager.dispatch(1, num_tokens, None, 0)
        return desc in self.manager.graphs

    def run(
        self,
        aux_hidden_states: list[torch.Tensor],
        num_tokens: int,
        *,
        context_kv_is_restored: bool = False,
    ) -> None:
        if context_kv_is_restored:
            return
        desc = self.manager.dispatch(1, num_tokens, None, 0)
        if desc not in self.manager.graphs:
            raise ValueError("DSpark context rows exceed captured decode capacities")
        if len(aux_hidden_states) != self.num_aux or any(
            x.ndim != 2 or x.shape[0] < num_tokens or x.shape[1] != self.width
            for x in aux_hidden_states
        ):
            raise ValueError("DSpark context auxiliary states have invalid dimensions")
        # The sources side by side in aux. Context input preparation initializes
        # rejected rows, but not rows beyond the target batch: the same launch
        # scrubs that tail on EVERY replay, including transitions from a larger
        # batch to a smaller one.
        dspark_context.context_inputs(
            self.aux,
            aux_hidden_states,
            self.positions,
            self.slot_mappings,
            num_tokens,
            desc.num_tokens,
        )
        self.manager.run_fullgraph(desc)

    def close(self) -> None:
        """Destroy context graphs before their prepared executions are released."""
        self.manager.reset_graphs()
        self.manager.graphs.clear()
        self.manager.graph_capture_resources.clear()


class DSparkDeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.num_hidden_layers = config.num_hidden_layers
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        # The context graphs' input launch (DSparkContextCudaGraphs.run).
        context_key = dspark_context.context_inputs_key(
            self.hidden_size, vllm_config.model_config.dtype
        )
        # combine_aux_hidden_states_into: the eager context step's join.
        join_widths = (self.hidden_size,) * len(self.target_layer_ids)
        join_key = columns.split_columns_key(
            join_widths,
            vllm_config.model_config.dtype,
            join=True,
        )
        calls = [
            prepare.call(dspark_context.context_inputs_kernel, *context_key),
            prepare.call(columns.split_columns_kernel, *join_key),
        ]
        if (
            vllm_config.model_config.dtype == torch.bfloat16
            and self.hidden_size % 128 == 0
        ):
            join_cast_key = columns.join_columns_and_cast_mxfp8_key(join_widths)
            calls.append(
                prepare.call(
                    columns.join_columns_and_cast_mxfp8_kernel, *join_cast_key
                )
            )
        prepare.declare(lambda: calls)
        self._context_kv_projections: list = []

        self.num_dspark_layers = (
            getattr(config, "n_mtp_layers", None)
            or getattr(config, "num_nextn_predict_layers", None)
            or 3
        )

        # Bound from the loaded target, as in K3DSparkModel. Do not create an
        # unused vocabulary allocation while loading the draft checkpoint.
        self.embed_tokens: nn.Module | None = None

        # Column-shard the projection of the target's auxiliary hidden states when
        # TP > 1 and gather the output: same arithmetic.
        main_proj_tp = get_tensor_model_parallel_world_size() > 1
        main_proj_cls = (
            _PaddedColumnParallelLinear if main_proj_tp else ReplicatedLinear
        )
        self.main_proj = main_proj_cls(
            config.hidden_size * len(self.target_layer_ids),
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "main_proj"),
            **({"gather_output": True} if main_proj_tp else {}),
        )
        self.main_norm = kernels.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        if hasattr(self.main_norm, "with_cast"):
            self.main_norm.with_cast = True

        current_vllm_config = get_current_vllm_config()
        self._l2pf_ready = False
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),
                )
                for i in range(self.num_dspark_layers)
            ]
        )
        # The drafter's layers attend to their window only (compress ratio 0): no
        # indexed attention reads a top-k buffer, so none is allocated for them.
        if any(layer.attn.compress_ratio for layer in self.layers):
            raise ValueError(
                "DSpark draft layers must be window-only (compress ratio 0)"
            )
        # A draft forward routes one query block per request, as DSparkSpeculator
        # lays it out: the speculative tokens, plus the anchor unless the drafter
        # predicts from it.
        query_rows = vllm_config.speculative_config.num_speculative_tokens + (
            0 if getattr(config, "sample_from_anchor", True) else 1
        )
        for layer in self.layers:
            router = getattr(layer.ffn.experts, "router", None)
            if hasattr(router, "max_rows"):
                router.max_rows = vllm_config.scheduler_config.max_num_seqs * query_rows

        # Heads: final norm, and the Markov + confidence heads.
        # Loaded from the "final" MTP layer weights (mtp.*) in the target
        # checkpoint. v4.1 has no learned hc_head: the hc copies are
        # collapsed with a pre-mix derived from the last layer's hc_ffn
        # projection at the end of forward().
        self.norm = kernels.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        from vllm.model_executor.layers.quantization.online.packed_bf16 import (
            packed_bf16_supported,
        )

        markov_quantization = (
            "packed_bf16"
            if current_vllm_config.kernel_config.linear_backend == "tilelang"
            and packed_bf16_supported(
                config.dspark_markov_rank, torch.get_default_dtype()
            )
            else None
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            # Pack the native BF16 transition head losslessly into 12 bits
            # per weight (75% of BF16 bytes) on the supported TileLang path.
            lm_head_quantization=markov_quantization,
            vocab_parallel=_vocab_parallel_greedy(),
        )
        # Each sequential Markov step projects one row per request.
        self.markov_head.markov_w2.tilelang_max_rows = (
            vllm_config.scheduler_config.max_num_seqs
        )
        self.markov_head.markov_w2.tilelang_markov_head = True
        self.confidence_head: DSparkConfidenceHead | None = None
        if getattr(config, "enable_confidence_head", True):
            self.confidence_head = kernels.dspark_confidence_head(
                config.hidden_size,
                config.dspark_markov_rank,
                prefix=maybe_prefix(prefix, "confidence_head"),
            )

    def declare_kernels(self) -> None:
        """Declare the context KV projections' TileLang kernels (built after loading).

        The draft layers' modules declare their own kernels, WO included, and the
        Markov head's are declared by the logits processor it is registered with.
        """
        from ..tilelang.linear import _warm_casts

        for projection in self._context_kv_projections:
            projection.declare_kernels()
        if self._context_kv_projections:
            if hasattr(self.main_norm, "with_cast"):
                self.main_norm.with_cast = True
            prepare.declare(warm=lambda: _warm_casts(self.hidden_size))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    def combine_hidden_states(
        self, aux_hidden_states: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """main_x = main_norm(main_proj(concat of target aux hidden states)).

        ``aux_hidden_states`` is [T, hidden_size * len(target_layer_ids)]. With
        ``out`` (contiguous, [T, hidden_size]), the norm writes main_x there; when
        the context KV projections read a producer's MXFP8 cast, ``main_norm``
        also emits ``main_x``'s MXFP8 cast in the same launch.
        """
        projected = self.main_proj(aux_hidden_states)
        if _casts_context_main_x(self) and hasattr(
            self.main_norm, "forward_with_cast"
        ):
            cast = self.main_norm.forward_with_cast(projected, out=out)
            self._context_main_cast = cast
            return cast.rows
        self._context_main_cast = None
        return self.main_norm(projected, out=out)

    def combine_aux_hidden_states_into(
        self, aux_hidden_states: list[torch.Tensor], out: torch.Tensor
    ) -> None:
        """``combine_hidden_states`` of the target aux hidden states' first
        ``out.shape[0]`` rows, written into ``out``: the states joined side by side
        in one TileLang launch and the norm writing ``out`` (torch's cat, and the
        speculator's copy into its context buffer, gone)."""
        rows = out.shape[0]
        joined = out.new_empty(rows, len(aux_hidden_states) * self.hidden_size)
        parts = [aux[:rows] for aux in aux_hidden_states]
        if (
            joined.dtype == torch.bfloat16
            and self.hidden_size % 128 == 0
            and _casts_main_proj_input(self)
        ):
            joined_cast = columns.join_columns_and_cast_mxfp8(parts, joined)
            with kernels.mxfp8_inputs(joined_cast):
                self.combine_hidden_states(joined, out=out)
            return
        columns.join_columns(parts, joined)
        self.combine_hidden_states(joined, out=out)

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Insert the sliding-window context KV for every draft layer.

        Mirrors the reference DSparkAttention: each layer derives its context KV
        from the SAME projected target hidden ``main_x``, via that layer's own
        ``wkv`` + ``kv_norm`` + RoPE + quant, then writes it at the
        layer's context slots.

        ``context_slot_mappings`` is either common to all layers or a per-layer
        list (each entry is the mapping for that layer's kv-cache group).
        ``None`` (or a ``None`` entry) runs the projection to reserve workspace
        but writes nothing (profiling).
        """
        cached_cast = getattr(self, "_context_main_cast", None)
        self._context_main_cast = None
        if (
            cached_cast is not None
            and cached_cast.rows.data_ptr() == main_x.data_ptr()
            and cached_cast.rows.shape == main_x.shape
        ):
            main_cast = cached_cast
        elif _casts_context_main_x(self):
            main_cast = kernels.cast_mxfp8(main_x)
        else:
            main_cast = None
        with kernels.mxfp8_inputs(main_cast):
            for i, layer in enumerate(self.layers):
                slot_mapping = (
                    context_slot_mappings
                    if context_slot_mappings is None
                    or isinstance(context_slot_mappings, torch.Tensor)
                    else context_slot_mappings[i]
                )
                attn = layer.attn
                kv_raw = self._context_kv_projections[i](main_x)
                if slot_mapping is None:
                    continue
                if hasattr(attn, "insert_raw_context_kv"):
                    attn.insert_raw_context_kv(
                        kv_raw, context_positions, slot_mapping
                    )
                else:
                    _insert_context_kv(
                        attn, attn.kv_norm(kv_raw), context_positions, slot_mapping
                    )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # The first layer's post-load broadcast weights consume [T, H] directly.
        hidden_states = inputs_embeds

        residual = post_mix = res_mix = pre_mix = None
        if l2_prefetch.ENABLED and not self._l2pf_ready:
            self._l2pf_ready = build_l2_prefetch_plans(
                list(self.layers), self.config.num_hidden_layers
            )
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
            )
        l2_prefetch.join()
        assert pre_mix is not None
        if hasattr(layer.mhc, "post_collapse"):
            return layer.mhc.post_collapse(
                hidden_states, residual, post_mix, res_mix, pre_mix
            )
        hidden_states = layer.mhc.post(hidden_states, residual, post_mix, res_mix)
        # Collapse the hc copies with the pre-mix from the last layer's FFN
        # mixes — the mix the reference forward_head applies (hc_pre with the
        # last block's ffn pre-mix). Return the PRE-norm head hidden;
        # compute_logits applies self.norm.
        hidden_states = layer.mhc.collapse(hidden_states, pre_mix)
        return hidden_states


def _casts_main_proj_input(model: object) -> bool:
    main_proj = getattr(model, "main_proj", None)
    if getattr(main_proj, "tilelang_mxfp8_input", False):
        return True
    quant_method = getattr(main_proj, "quant_method", None)
    return type(quant_method).__name__ == "TileLangFP8LinearMethod"


def _casts_context_main_x(model: object) -> bool:
    return any(
        getattr(proj, "tilelang_mxfp8_input", False)
        for proj in getattr(model, "_context_kv_projections", ())
    )


def _insert_context_kv(
    attn: nn.Module,
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    attn.insert_context_kv(kv, positions, slot_mapping)


class DSparkDeepseekV4ForCausalLM(nn.Module):
    # Draft weights ship in the target checkpoint (mtp.*) without embed/head, so
    # load_dspark_model always aliases the target's.
    has_own_embed_tokens = False
    has_own_lm_head = False
    # Full-vocab draft: draft ids are target ids, no remapping needed.
    draft_id_to_target_id = None
    checkpoint_weight_name_prefixes: tuple[str, ...] = ("mtp.",)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.linear_scale_name = _linear_scale_param_name(
            vllm_config, getattr(self.config, "expert_dtype", "fp4")
        )
        self.model = DSparkDeepseekV4Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Bound by load_dspark_model after loading. An unused online-quantized
        # head otherwise remains on meta and cannot be natively precompiled.
        self.lm_head: nn.Module | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    # --- Hooks used by the speculator -------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(
        self, aux_hidden_states: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.model.combine_hidden_states(aux_hidden_states, out)

    def combine_aux_hidden_states_into(
        self, aux_hidden_states: list[torch.Tensor], out: torch.Tensor
    ) -> None:
        self.model.combine_aux_hidden_states_into(aux_hidden_states, out)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        # DSV4 MLA path: each draft layer's sliding-window cache is a separate
        # layer, named by its prefix.
        return [layer.attn.swa_cache_layer.prefix for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mappings
        )

    def capture_context_preparation(
        self,
        vllm_config: VllmConfig,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mappings: torch.Tensor,
        layer_group_idx: list[int] | None,
        max_decode_tokens: int,
    ) -> DSparkContextCudaGraphs:
        context = DSparkContextCudaGraphs(
            self,
            vllm_config,
            hidden_states,
            positions,
            slot_mappings,
            layer_group_idx,
            max_decode_tokens,
        )
        context.capture()
        return context

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Returns the pre-norm collapsed head hidden ([T, hidden_size]).
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Base logits U_k = lm_head(norm(head_hidden))."""
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, self.model.norm(hidden_states))

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Full-vocab draft: base logits, no d2t scatter.
        return self.compute_logits(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids  # full-vocab: draft ids are target ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_embed_table(self) -> torch.Tensor | None:
        """``markov_embed``'s table when W1 is a replicated ``nn.Embedding`` (the
        speculator's kernels then look rows up themselves), else None."""
        w1 = self.model.markov_head.markov_w1
        return w1.weight if type(w1) is nn.Embedding else None

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    # --- Vocabulary-parallel greedy drafting ---
    # Each rank keeps its vocabulary shard of the draft logits and Markov bias
    # and reduces only (max, index) pairs across ranks: exact greedy drafts
    # without the draft-logits all-gather or a replicated Markov projection.

    @property
    def vocab_parallel_greedy(self) -> bool:
        return not self.model.markov_head.markov_w2.disable_tp

    @property
    def draft_vocab_shard(self) -> tuple[int, int]:
        """(first global vocabulary index, real entries) of this rank's shard.

        The draft head is bound only after loading, so the check that it and
        the Markov projection share one partition runs here, on first use.
        """
        assert self.lm_head is not None
        head = self.lm_head.shard_indices
        markov = self.model.markov_head.markov_w2.shard_indices
        shard = head.org_vocab_start_index, head.num_org_elements
        if shard != (markov.org_vocab_start_index, markov.num_org_elements):
            raise RuntimeError(
                "Vocabulary-parallel greedy drafting needs the draft head and Markov "
                f"projection sharded alike: {head} != {markov}"
            )
        return shard

    def compute_draft_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.logits_processor(
            self.lm_head, self.model.norm(hidden_states), skip_gather=True
        )

    def markov_bias_local(self, markov_embed: torch.Tensor) -> torch.Tensor:
        w2 = self.model.markov_head.markov_w2
        if (
            self.logits_processor.use_tilelang_vocab_projection
            and hasattr(w2, "tilelang_vocab")
            and self.logits_processor.soft_cap is None
            and self.logits_processor.scale == 1.0
        ):
            from ..tilelang.vocab import project_head

            n = int(w2.weight.shape[0])
            rows = markov_embed.shape[0]
            dtype = getattr(w2, "tilelang_vocab_dtype", torch.bfloat16)
            scratch = getattr(w2, "_tilelang_vocab_bias_scratch", None)
            if (
                scratch is None
                or scratch.device != markov_embed.device
                or scratch.dtype != dtype
                or scratch.shape[0] < rows
                or scratch.shape[1] != n
            ):
                cap = max(rows, getattr(w2, "tilelang_max_rows", rows) or rows)
                scratch = torch.empty(
                    (cap, n), dtype=dtype, device=markov_embed.device
                )
                w2._tilelang_vocab_bias_scratch = scratch
            return project_head(w2, markov_embed, out=scratch[:rows])
        return self.logits_processor(
            w2, markov_embed, skip_gather=True
        )

    @property
    def draft_logits_local_layout(self) -> tuple[torch.dtype, int, int]:
        """(dtype, draft logits row width, Markov bias row width) of the fresh
        tensors ``compute_draft_logits_local`` and ``markov_bias_local`` return:
        both heads write the head dtype, one padded vocabulary shard per row."""
        assert self.lm_head is not None
        return (
            self.logits_processor.head_dtype,
            self.lm_head.num_embeddings_per_partition,
            self.model.markov_head.markov_w2.num_embeddings_per_partition,
        )

    @property
    def fuses_markov_greedy(self) -> bool:
        """Whether ``markov_greedy_pairs_local`` can run: a TileLang packed or
        BF16 Markov head declares the fused projection (an NVFP4 one does not),
        and nothing scales or caps the bias."""
        w2 = self.model.markov_head.markov_w2
        processor = self.logits_processor
        return (
            processor.use_tilelang_vocab_projection
            and processor.scale == 1.0
            and processor.soft_cap is None
            and (_is_packed_head(w2) or _is_plain_bf16_head(w2))
        )

    def markov_greedy_pairs_local(
        self,
        markov_embed: torch.Tensor,
        base_logits: torch.Tensor,
        pairs: torch.Tensor,
        *,
        top2: bool = False,
    ) -> None:
        """Each row's ``(max, global index)`` of this rank's shard of
        ``base_logits`` ([requests, padded shard], any row stride) plus the Markov
        bias of ``markov_embed``, into the first two columns of ``pairs`` (with
        ``top2``, the runner-up ``greedy_markov_shard_pair`` reports into the
        other two), without the bias tensor: the Markov projection adds the base
        logits and reduces each vocabulary tile in its epilogue."""
        from ..tilelang.vocab import project_head_markov_greedy_pairs

        vocab_start, shard_vocab = self.draft_vocab_shard
        project_head_markov_greedy_pairs(
            self.model.markov_head.markov_w2,
            markov_embed,
            base_logits,
            vocab=shard_vocab,
            vocab_start=vocab_start,
            out=pairs,
            top2=top2,
        )

    def apply_markov_bias_gathered(
        self,
        markov_embed: torch.Tensor,
        logits: torch.Tensor,
        values: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """Add the Markov bias to the top-k base-logit candidates only.

        Reads k rows of the replicated ``markov_w2`` per request instead of
        the whole vocabulary projection at every draft position. ``logits``
        is the caller's ``-inf``-filled buffer; the corrected candidates are
        scattered into it. The target still verifies every draft, so this
        can change acceptance but not the output.
        """
        w2 = self.model.markov_head.markov_w2
        if hasattr(w2, "packed_bf16_exponent_base"):
            from ..tilelang.embedding import vocab_embedding_packed

            flat_index = index.reshape(-1).contiguous()
            in_features = w2.packed_bf16_in_features
            weight = torch.empty(
                (flat_index.shape[0], in_features),
                dtype=torch.bfloat16,
                device=w2.weight.device,
            )
            shard = w2.shard_indices
            vocab_embedding_packed(
                flat_index,
                w2.weight,
                w2.weight_exception_entries,
                w2.weight_row_offsets,
                in_features,
                w2.packed_bf16_exponent_base,
                weight,
                shard.org_vocab_start_index,
                shard.org_vocab_end_index,
                shard.num_org_vocab_padding,
                shard.added_vocab_start_index,
                shard.added_vocab_end_index,
            )
            weight = weight.view(*index.shape, in_features)
        else:
            weight = w2.weight[index]
        bias = torch.bmm(weight, markov_embed.to(weight.dtype).unsqueeze(-1))
        bias = bias.squeeze(-1).to(values.dtype)
        if self.logits_processor.scale != 1.0:
            bias = bias * self.logits_processor.scale
        return logits.scatter_(1, index, (values + bias).to(logits.dtype))

    def compute_confidence(
        self,
        head_hidden: torch.Tensor,
        markov_embed: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-position acceptance probability for each drafted token, into
        ``out`` when given."""
        assert self.model.confidence_head is not None
        return self.model.confidence_head.probability(head_hidden, markov_embed, out)

    def compute_and_calibrate_confidence(
        self,
        head_hidden: torch.Tensor,
        markov_embed: torch.Tensor,
        out: torch.Tensor,
        estimator: object,
        idx_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position acceptance probability calibrated online by ``estimator``."""
        assert self.model.confidence_head is not None
        head = self.model.confidence_head
        if (
            estimator is not None
            and hasattr(head, "probability_and_calibrate")
            and head_hidden.is_cuda
            and idx_mapping.dtype == torch.int32
            and idx_mapping.is_contiguous()
            and idx_mapping.numel() == head_hidden.shape[0]
        ):
            return head.probability_and_calibrate(
                head_hidden,
                markov_embed,
                out,
                estimator.slope,
                estimator.intercepts,
                idx_mapping,
                estimator.features,
                estimator.predictions,
            )
        probs = self.compute_confidence(head_hidden, markov_embed, out=out)
        if estimator is not None:
            num_steps = estimator.num_speculative_steps
            estimator.calibrate_confidence_probs(
                probs.view(-1, num_steps), idx_mapping
            )
        return probs

    # --- Weight loading ----------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the ``mtp.{0,1,2}.*`` draft weights from the target checkpoint.

        Non-mtp weights (embed/head/main layers) belong to the target model and
        are skipped here. ``embed_tokens``/``lm_head`` are aliased from the target.
        """
        # Draft MoE layers use the dspark_* expert counts, not the
        # backbone's (see DeepseekV4MoE and the reference
        # ModelArgs.get_moe_config).
        n_draft_experts = (
            getattr(self.config, "dspark_n_routed_experts", 0)
            or self.config.n_routed_experts
        )
        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=n_draft_experts,
        )
        expert_mapping_by_key: dict[str, list[tuple[str, str, int, str]]] = {}
        for mapping in expert_mapping:
            expert_mapping_by_key.setdefault(mapping[1], []).append(mapping)
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        # (param_name, ckpt_shard_name, shard_id) for non-expert stacked params.
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        loaded_confidence_head = False

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)

        for name, loaded_weight in weights:
            mapped = self._remap_dspark_name(name)
            if mapped is None:
                continue
            name = mapped
            if "confidence_head." in name:
                loaded_confidence_head = True

            # ``.scale`` -> per-method scale suffix.
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else f".{self.linear_scale_name}"
                )
                name = name.removesuffix(".scale") + suffix
            if ".shared_experts.w2" in name:
                name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")

            # E8M0 expert scales: keep raw exponent bytes.
            if ".experts." in name:
                if (
                    "weight_scale" in name
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                _, _, rest = name.partition(".experts.")
                d1 = rest.find(".")
                d2 = rest.find(".", d1 + 1) if d1 != -1 else -1
                candidates = (
                    expert_mapping_by_key.get(f"experts.{rest[: d2 + 1]}")
                    if d2 != -1
                    else None
                )
                for param_name, weight_name, expert_id, shard_id in (
                    candidates if candidates is not None else expert_mapping
                ):
                    if weight_name not in name:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    param = params_dict[name_mapped]
                    success = param.weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(name_mapped)
                        break
                continue

            # Stacked rules only apply to decoder-layer weights. Head-stack params
            # (main_proj/norm/markov_head/confidence_head) load directly —
            # otherwise e.g. "markov_w1" would collide with the "w1" shard rule.
            is_layer_param = name.startswith("model.layers.")
            for param_name, weight_name, stacked_shard_id in stacked_params_mapping:
                if not is_layer_param or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, stacked_shard_id)
                loaded_params.add(name)
                break
            else:
                if "attn_sink" in name:
                    narrow = loaded_weight[head_start:head_end]
                    copy_weight(params_dict[name][: narrow.shape[0]], narrow)
                    loaded_params.add(name)
                    continue
                if name.endswith(".ffn.gate.bias"):
                    name = name.replace(
                        ".ffn.gate.bias", ".ffn.gate.e_score_correction_bias"
                    )
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if self.has_own_lm_head and "lm_head.weight" not in loaded_params:
            raise RuntimeError("The DSpark NVFP4 draft head was not loaded")
        if self.model.confidence_head is not None and not loaded_confidence_head:
            self.model.confidence_head = None
        logger.info_once("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def process_weights_after_loading(self) -> None:
        w2 = self.model.markov_head.markov_w2
        self.logits_processor.prepare_b12x_vocab_projection(w2)
        if hasattr(w2, "tilelang_vocab") and hasattr(w2, "weight"):
            cap = int(getattr(w2, "tilelang_max_rows", 16) or 16)
            n = int(w2.weight.shape[0])
            dtype = getattr(w2, "tilelang_vocab_dtype", torch.bfloat16)
            w2._tilelang_vocab_bias_scratch = torch.empty(
                (cap, n), dtype=dtype, device=w2.weight.device
            )
        for layer in self.model.layers:
            layer.attn.setup_wo_projection()
        self.model._context_kv_projections = [
            _context_kv_projection(layer.attn) for layer in self.model.layers
        ]
        first_layer = self.model.layers[0]
        broadcast = (
            first_layer.hc_attn_fn.detach()
            .view(-1, first_layer.hc_mult, first_layer.hidden_size)
            .sum(dim=1)
        )
        if first_layer.hc_attn_fn_broadcast is None:
            first_layer.hc_attn_fn_broadcast = broadcast
        else:
            first_layer.hc_attn_fn_broadcast.copy_(broadcast)

    def _remap_dspark_name(self, name: str) -> str | None:
        """Map a checkpoint ``mtp.{i}.*`` name to this model's parameter path.

        Returns None for non-mtp weights (owned by the target model).
        """
        if name == "head.weight":
            return "lm_head.weight" if self.has_own_lm_head else None
        m = re.match(r"mtp\.(\d+)\.(.*)", name)
        if m is None:
            return None
        stage = int(m.group(1))
        rest = m.group(2)
        if rest.startswith("confidence_head.") and self.model.confidence_head is None:
            return None
        # The checkpoint calls the Markov head's factors ``embed``/``head``;
        # DSparkMarkovHead registers them as ``markov_w1``/``markov_w2``.
        if rest.startswith("markov_head.embed."):
            return "model.markov_head.markov_w1." + rest.removeprefix(
                "markov_head.embed."
            )
        if rest.startswith("markov_head.head."):
            return "model.markov_head.markov_w2." + rest.removeprefix(
                "markov_head.head."
            )
        # Head-stack params live at model level (mtp.last), context combiner at
        # model level (mtp.0); everything else is a per-layer decoder block.
        head_prefixes = (
            "norm.",
            "markov_head.",
            "confidence_head.",
        )
        if rest.startswith(("main_proj.", "main_norm.")) or rest.startswith(
            head_prefixes
        ):
            return f"model.{rest}"
        return f"model.layers.{stage}.{rest}"


# ---------------------------------------------------------------------------
# Drafter taps (boot-differences overlay): off unless SPARK_DS41F_ROW_TRACE_DIR and
# SPARK_DS41F_ROW_TRACE_DRAFTER=1. Kinds: 10 the target's auxiliary states copied in
# (context graph), 11 the combined context states the KV projections read, 12 the
# query forward's head hidden state, 13 the confidence head's hidden input and its
# output, 14 its Markov embeddings, 15 each draft layer's raw context KV, 16 its
# slots, 17 the confidence rows' request slots, 18 the estimator's slope and
# intercepts; collectives and attention outputs inside these forwards record with
# their own kinds (1-3, 7).
from vllm.distributed.device_communicators.cuda_communicator import (  # noqa: E402
    DRAFT_CTX_TRACE as _DRAFT_CTX_TRACE,
    DRAFT_QUERY_TRACE as _DRAFT_QUERY_TRACE,
)

if _DRAFT_CTX_TRACE is not None and _DRAFT_QUERY_TRACE is not None:
    _untraced_context_forward = DSparkContextCudaGraphs._forward
    _untraced_store_context = DSparkDeepseekV4Model.precompute_and_store_context_kv
    _untraced_draft_forward = DSparkDeepseekV4Model.forward
    _untraced_confidence = DSparkDeepseekV4ForCausalLM.compute_and_calibrate_confidence

    def _traced_context_forward(self, capacity):
        _DRAFT_CTX_TRACE.begin(None, self.positions[:capacity])
        _DRAFT_CTX_TRACE.record(10, self.aux[:capacity], self.aux[:capacity])
        _untraced_context_forward(self, capacity)
        _DRAFT_CTX_TRACE.end()

    def _traced_store_context(self, main_x, context_positions, context_slot_mappings=None):
        began = not _DRAFT_CTX_TRACE.active
        if began:  # the eager context path (the graph path began in _forward)
            _DRAFT_CTX_TRACE.begin(None, context_positions)
        _DRAFT_CTX_TRACE.record(11, main_x, main_x)
        for layer in self.layers:
            _wrap_context_insert(layer.attn)
        _untraced_store_context(self, main_x, context_positions, context_slot_mappings)
        if began:
            _DRAFT_CTX_TRACE.end()

    def _traced_draft_forward(self, input_ids, positions, inputs_embeds=None):
        _DRAFT_QUERY_TRACE.begin(input_ids, positions)
        out = _untraced_draft_forward(self, input_ids, positions, inputs_embeds)
        _DRAFT_QUERY_TRACE.record(12, out, out)
        return out

    def _wrap_context_insert(attn):
        # Each draft layer's context KV (before its norm, RoPE and quantization) and
        # the slots it is written to.
        if getattr(attn, "_bd_traced_insert", False) or not hasattr(attn, "insert_raw_context_kv"):
            return
        untraced = attn.insert_raw_context_kv

        def traced(kv_raw, *args, **kwargs):
            _DRAFT_CTX_TRACE.record(15, kv_raw, kv_raw)
            slots = kwargs.get("slot_mapping", args[-1] if args else None)
            if isinstance(slots, torch.Tensor):
                _DRAFT_CTX_TRACE.record(16, slots, slots)
            return untraced(kv_raw, *args, **kwargs)

        attn.insert_raw_context_kv = traced
        attn._bd_traced_insert = True

    def _traced_confidence(self, head_hidden, markov_embed, out, estimator, idx_mapping):
        probs = _untraced_confidence(self, head_hidden, markov_embed, out, estimator, idx_mapping)
        _DRAFT_QUERY_TRACE.record(13, head_hidden, probs)
        _DRAFT_QUERY_TRACE.record(14, markov_embed, markov_embed)
        if isinstance(idx_mapping, torch.Tensor):
            _DRAFT_QUERY_TRACE.record(17, idx_mapping, idx_mapping)
        slope = getattr(estimator, "slope", None)
        intercepts = getattr(estimator, "intercepts", None)
        if isinstance(slope, torch.Tensor) and isinstance(intercepts, torch.Tensor):
            _DRAFT_QUERY_TRACE.record(18, slope.reshape(1, -1), intercepts.reshape(1, -1))
        _DRAFT_QUERY_TRACE.end()
        return probs

    DSparkContextCudaGraphs._forward = _traced_context_forward
    DSparkDeepseekV4Model.precompute_and_store_context_kv = _traced_store_context
    DSparkDeepseekV4Model.forward = _traced_draft_forward
    DSparkDeepseekV4ForCausalLM.compute_and_calibrate_confidence = _traced_confidence
