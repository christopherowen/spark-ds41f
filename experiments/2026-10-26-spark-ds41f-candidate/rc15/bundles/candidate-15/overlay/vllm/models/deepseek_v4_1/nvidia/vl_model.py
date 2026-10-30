# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 DeepSeek V4.1 with native vision and streaming checkpoint loading."""

from collections.abc import Iterable
import os

import torch
from torch import nn

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.interfaces import _require_is_multimodal
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.models.deepseek_v41.common.mm_preprocess import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)
from vllm.models.deepseek_v41.nvidia.vl_model import (
    DeepseekV4VLImagePixelInputs,
)
from vllm.models.deepseek_v41.nvidia.vl_model import (
    DeepseekV41ForCausalLM as UpstreamDeepseekV41ForCausalLM,
)
from vllm.models.deepseek_v41.nvidia.vl_model import (
    _make_deepseek_v4_vl_weights_mapper as _make_deepseek_v4_vl_weights_mapper,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.torch_utils import async_tensor_h2d

from ..tilelang import ced as ced_kernels
from ..tilelang import prepare
from .model import (
    DeepseekV41LLMForCausalLM,
    _linear_scale_param_name,
)


def _vision_tower():
    """The TileLang ViT, aligner and data-parallel image runner, over the
    checkpoint's parameter tree."""
    from .tilelang_vision import (
        DeepseekV4Aligner,
        DeepseekV4ViT,
        run_dp_sharded_vision_tower,
    )

    return DeepseekV4ViT, DeepseekV4Aligner, run_dp_sharded_vision_tower


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor,
    info=DeepseekV4VLProcessingInfo,
    dummy_inputs=DeepseekV4VLDummyInputsBuilder,
)
class DeepseekV41ForCausalLM(UpstreamDeepseekV41ForCausalLM):
    """Multimodal entry point for DeepSeek-V4.1 checkpoints with a vision tower.

    ``SupportsEagle3`` (aux hidden-state plumbing for MTP/DSpark drafters)
    delegates through ``language_model`` via the protocol defaults.
    """

    supports_encoder_tp_data = True

    # The MoE router applies bias_vl to raw image-span token IDs.
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        nn.Module.__init__(self)
        model_config = vllm_config.model_config
        config = model_config.hf_config
        self.config = config
        # The embeddings' dtype: the image spans' rows and delimiters take it.
        self.embed_dtype = model_config.dtype
        self.multimodal_config = model_config.multimodal_config
        assert self.multimodal_config is not None

        # The tower is always built; _mark_tower_model stubs it out
        # (StageMissingLayer, weights skipped) when the image limit is 0.
        vit, aligner, self._run_dp_sharded_vision_tower = _vision_tower()
        with self._mark_tower_model(vllm_config, {"image"}):
            self.use_data_parallel = is_vit_use_data_parallel() or (
                os.environ.get("SPARK_DS41F_VISION_TP", "0") != "1"
            )
            self.vision = vit(config, use_data_parallel=self.use_data_parallel)
            self.aligner = aligner(config, use_data_parallel=self.use_data_parallel)
            self.image_start = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )
            self.image_end = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )
            self.image_newline = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )

        with self._mark_language_model(vllm_config):
            self.language_model = DeepseekV41LLMForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        # The outer mapper (see load_weights) fully resolves HF names into
        # this wrapper's namespace before AutoWeightsLoader strips the
        # "language_model." prefix and delegates to the child's load_weights,
        # so the child's own mapper must be a no-op. Its suffix rules are not
        # idempotent (e.g. "lm_head.weight".endswith("head.weight") would
        # re-fire "head.weight" -> "lm_head.weight").
        self.language_model.hf_to_vllm_mapper = WeightsMapper()
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )

        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.hf_to_vllm_mapper = _make_deepseek_v4_vl_weights_mapper(
            expert_dtype, _linear_scale_param_name(vllm_config, expert_dtype)
        )

    def _process_image_input(
        self,
        image_input: DeepseekV4VLImagePixelInputs,
    ) -> tuple[torch.Tensor, ...]:
        patches = image_input.patches.to(self.aligner.w1.weight.dtype)
        vit_grid = image_input.vit_grid.tolist()

        image_embeds_list: list[torch.Tensor]
        if self.use_data_parallel and get_tensor_model_parallel_world_size() > 1:
            # Data-parallel ViT: shard images across TP ranks and all-gather
            # the per-image embeddings (weights are replicated on every rank).
            image_embeds_list = self._run_dp_sharded_vision_tower(
                self.vision, self.aligner, patches, vit_grid
            )
        else:
            image_embeds_list = []
            vit_offset = 0
            for n_vit_h, n_vit_w in vit_grid:
                n_vit = n_vit_h * n_vit_w
                image_embeds_list.append(
                    self._encode_image(
                        patches[vit_offset : vit_offset + n_vit], n_vit_h, n_vit_w
                    )
                )
                vit_offset += n_vit

        embeds: list[torch.Tensor] = []
        span_offset = 0
        for image_embeds, (n_llm_h, n_llm_w) in zip(
            image_embeds_list, image_input.llm_grid.tolist(), strict=True
        ):
            span_len = n_llm_h * (n_llm_w + 1) + 2
            embeds.append(
                self._build_image_span(
                    image_embeds,
                    image_input.types[span_offset : span_offset + span_len],
                )
            )
            span_offset += span_len
        return tuple(embeds)

    # The span's delimiter rows, in the order _build_image_span indexes them.
    _DELIMITERS = (IMAGE_START, IMAGE_END, IMAGE_NEW_LINE)

    def _build_image_span(
        self, image_embeds: torch.Tensor, types: torch.Tensor
    ) -> torch.Tensor:
        """Full image span: aligner rows at IMAGE slots, the learned delimiter
        vectors at IMAGE_START/IMAGE_NEW_LINE/IMAGE_END, by CED's TileLang row copy:
        the delimiters gathered from their table (an IMAGE slot's index is invalid
        and reads zero), then the aligner rows scattered over the IMAGE slots. The
        processor keeps ``types`` on the host, so the slots are found without a
        device sync (torch's masked assignments synced once a role)."""
        if types.is_cuda:
            raise ValueError("image span roles are host data")
        kinds = types.tolist()
        rows = {kind: row for row, kind in enumerate(self._DELIMITERS)}
        delimiter_rows = [rows.get(kind, -1) for kind in kinds]
        image_slots = [slot for slot, kind in enumerate(kinds) if kind == IMAGE]
        if len(image_slots) != image_embeds.shape[0]:
            raise ValueError(
                f"{image_embeds.shape[0]} aligner rows for {len(image_slots)} image "
                "slots"
            )
        device = image_embeds.device
        span = image_embeds.new_empty(len(kinds), image_embeds.shape[-1])
        ced_kernels.copy_rows(
            self._image_delimiters,
            async_tensor_h2d(delimiter_rows, device=device, dtype=torch.int64),
            span,
            scatter=False,
        )
        ced_kernels.copy_rows(
            image_embeds,
            async_tensor_h2d(image_slots, device=device, dtype=torch.int64),
            span,
            scatter=True,
        )
        return span

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The token embeddings with each image span's rows at its slots, scattered
        by CED's TileLang row copy, a span at a time. The mask is host data (the
        runner's pinned mask), so the slots are found without a device sync
        (torch's index_put of the flattened spans before)."""
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.get_language_model().embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        mask = _require_is_multimodal(is_multimodal)
        if mask.is_cuda:
            raise ValueError("the image slot mask is host data")
        slots = mask.nonzero().flatten().tolist()
        spans = list(multimodal_embeddings)
        if sum(span.shape[0] for span in spans) != len(slots):
            raise ValueError(
                f"{sum(span.shape[0] for span in spans)} image embeddings for "
                f"{len(slots)} placeholders"
            )
        offset = 0
        for span in spans:
            if span.dim() != 2 or span.dtype != inputs_embeds.dtype:
                raise ValueError("image spans are rows of the embeddings' dtype")
            rows = span.shape[0]
            ced_kernels.copy_rows(
                span,
                async_tensor_h2d(
                    slots[offset : offset + rows],
                    device=inputs_embeds.device,
                    dtype=torch.int64,
                ),
                inputs_embeds,
                scatter=True,
            )
            offset += rows
        return inputs_embeds

    def declare_kernels(self) -> None:
        """Declare the image path's row copies: the span's delimiter gather and image
        scatter, and the spans' scatter into the embeddings."""
        hidden = self.config.hidden_size
        key = (str(self.embed_dtype).removeprefix("torch."), (hidden,))
        prepare.declare(
            lambda: [
                prepare.call(
                    ced_kernels.copy_rows_kernel,
                    *ced_kernels.packed_copy_rows_key(*key, scatter),
                )
                for scatter in (False, True)
            ]
        )

    def get_sink_residual_columns(self) -> torch.Tensor | None:
        """The attention-sink detector's residual column per stream."""
        return self.language_model.get_sink_residual_columns()

    def get_graph_output_buffers(
        self,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
        return self.language_model.get_graph_output_buffers()

    @staticmethod
    def get_model_state_cls():
        from .model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    requires_accepted_token_lookback = True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        ced_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
            ced_indices=ced_indices,
        )

    def checkpoint_file_weight_filter(self, name: str) -> bool:
        return self.language_model.checkpoint_file_weight_filter(name)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Prefix groups may recur across checkpoint files. Consume each tensor
        # immediately; sorting payloads retains the entire checkpoint on device.
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )

    def process_weights_after_loading(self) -> None:
        # Native vision and aligner declarations are collected from their loaded
        # owners by the worker preparation registry.
        self.language_model.process_weights_after_loading()
        # The span's delimiter rows in the aligner's dtype, stacked once (the
        # learned vectors are FP32 parameters; a span casts them as torch did).
        # A stubbed tower (no images allowed) has none.
        if self.image_start.is_meta:
            return
        self._image_delimiters = torch.stack(
            [
                getattr(self, name).to(self.embed_dtype)
                for name in ("image_start", "image_end", "image_newline")
            ]
        )
