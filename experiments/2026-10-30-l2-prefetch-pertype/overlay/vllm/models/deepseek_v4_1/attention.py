# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 Full/Reindex/Reuse topology: caches, metadata and ownership. The TileLang
layer (``tilelang/attention.py``) runs the indexer and sparse MLA kernels."""

import os
from typing import cast

import regex as re
import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import (
    eager_break_during_capture,
    is_breakable_cudagraph_enabled,
)
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearMethodBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4_1 import kernels, l2_prefetch, sp_prefill
from vllm.models.deepseek_v4_1.ced import ced_decoder_start
from vllm.models.deepseek_v4_1.compressor import DeepseekCompressor
from vllm.models.deepseek_v4_1.quant_config import Block32FP8LinearMethod
from vllm.models.deepseek_v4_1.sparse_mla import (
    SLOT_DTYPE,
    DeepseekV41B12xBackend,
    DeepseekV41B12xMetadata,
    tilelang_builder_kernels,
)
from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang.gemm import DECODE_ROWS
from vllm.models.deepseek_v4_1.tilelang.indexer import CANDIDATE_BLOCKS
from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    retain_cuda_graph_capture_resource,
)

logger = init_logger(__name__)

# Validation only: under a sequence-parallel indexer split, every rank also runs
# the full-row indexer twice and compares its own rows with the split result.
_INDEX_SPLIT_CHECK = os.environ.get("SPARK_DS41F_INDEXER_SPLIT_CHECK") == "1"


def _source(prefix, layer):
    return re.sub(r"(layers\.)\d+", rf"\g<1>{layer}", prefix)


def _native_linear(layer):
    method = layer.quant_method
    if kernels.is_native_linear_method(method):
        return
    if type(method) is UnquantizedLinearMethod:
        layer.quant_method = kernels.linear_method()
        return
    raise ValueError("V4.1 attention only supports native linear methods")


class _Cache(nn.Module, AttentionLayerBase):
    def __init__(self, config, prefix, *, kind, ratio=1, window=0, draft=False):
        super().__init__()
        self.prefix, self.kind, self.ratio, self.window = prefix, kind, ratio, window
        self.draft = draft
        self.block_size = (
            config.cache_config.swa_block_size
            or CacheConfig.DEFAULT_DS41_SWA_BLOCK_SIZE
            if kind == "swa"
            else config.cache_config.block_size
        )
        self.kv_cache = torch.tensor([])
        context = config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate V4.1 cache {prefix}")
        context[prefix] = self

    def bind_kv_cache(self, cache):
        # Keep allocator page stride: padding belongs to the allocator, not the ABI.
        self.kv_cache = cache.view(cache.shape[0], -1)

    def get_kv_cache_spec(self, config):
        # No V4.1 kernel uses a slot its block's owner has not written, as in
        # DeepSeek's reference: windows start at the request's replay start, the
        # sparse MLA never loads a masked entry, the indexer reads no key past the
        # visible positions, and the compressor reads only a slot it wrote.
        # Reused blocks need no zeros.
        common = dict(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=512 if self.kind == "swa" else 68,
            state_content_bytes=528 if self.kind == "swa" else 68,
            dtype=torch.uint8,
            tokens_per_state=self.ratio,
            cache_dtype_str="b12x_dsv41",
            alignment=None,
            uses_only_written_slots=True,
        )
        if self.kind == "swa":
            boundary = ced_decoder_start(config.model_config.hf_config)
            layer_id = int(re.search(r"layers\.(\d+)", self.prefix).group(1))
            bounded_replay = self.draft or (
                boundary is not None and layer_id >= boundary
            )
            return SlidingWindowMLASpec(
                **common,
                sliding_window=self.window,
                extra_retained_tokens=int(self.draft),
                prefix_cache_enabled=not bounded_replay,
                prefill_replay_window=128 if bounded_replay else 0,
            )
        return MLAAttentionSpec(**common)

    def get_attn_backend(self):
        return DeepseekV41B12xBackend

    def forward(self):
        raise RuntimeError("V4.1 caches are consumed by their owning attention layer")


class _WOProjectionWeightMethod(LinearMethodBase):
    """Keep checkpoint FP8 tensors unmodified for the fused WO projection."""

    def __init__(self, original):
        if not isinstance(original, Block32FP8LinearMethod):
            raise ValueError("V4.1 output projection requires block32 FP8 weights")
        self.original = original

    def create_weights(self, *args, **kwargs):
        return self.original.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer):
        if layer.weight.dtype != torch.float8_e4m3fn:
            raise ValueError("V4.1 output projection requires FP8 checkpoint weights")
        if not hasattr(layer, "weight_scale_inv"):
            raise ValueError("V4.1 output projection requires checkpoint scales")
        if layer.weight_scale_inv.dtype != torch.float8_e8m0fnu:
            raise ValueError("V4.1 output projection requires UE8M0 checkpoint scales")

    def apply(self, layer, x, bias=None):
        raise RuntimeError("V4.1 WO weights are consumed only by the fused projection")


class DeepseekV4Indexer(nn.Module):
    def __init__(self, config, prefix, *, owns_k, k_cache, ratio):
        super().__init__()
        hf = config.model_config.hf_config
        self.prefix, self.owns_k, self.k_cache = prefix, owns_k, k_cache
        # Replicating the small indexer avoids reducing a rows-by-context score
        # matrix across TP ranks. Every rank selects from all index heads.
        self.heads = hf.index_n_heads
        self.wq_b = ReplicatedLinear(
            hf.q_lora_rank,
            hf.index_n_heads * 128,
            bias=False,
            return_bias=False,
            quant_config=config.quant_config,
            prefix=f"{prefix}.wq_b",
        )
        _native_linear(self.wq_b)
        self.weights_proj = ReplicatedLinear(
            hf.hidden_size,
            hf.index_n_heads,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        from vllm.models.deepseek_v4_1.tilelang.linear import (
            TileLangScaledLinearMethod,
        )

        # DeepSeek scales the head weights by (heads * head_dim) ** -0.5, a
        # power of two folded exactly into the projection.
        self.weights_proj.quant_method = TileLangScaledLinearMethod(
            (hf.index_n_heads * 128) ** -0.5
        )
        if owns_k:
            self.wk = ReplicatedLinear(
                512,
                128,
                bias=False,
                return_bias=False,
                quant_config=None,
                prefix=f"{prefix}.wk",
            )
            self.wk.quant_method = kernels.linear_method()
            self.k_norm = kernels.rms_norm(128, hf.rms_norm_eps)
        self.ratio = ratio


@torch.library.custom_op("vllm::dsv41_b12x_attention", mutates_args=())
def _attention(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    prefix: str,
    global_kv_ready: torch.Tensor | None,
) -> torch.Tensor:
    context = get_forward_context()
    layer = context.no_compile_layers[prefix]
    # WO projection and TP reduction own this result; it cannot alias the
    # borrowed attention scratch or the operation's input tensors.
    return layer._forward(positions, hidden)


@_attention.register_fake
def _attention_fake(hidden, positions, prefix, global_kv_ready):
    return torch.empty_like(hidden, memory_format=torch.contiguous_format)


@torch.library.custom_op("vllm::dsv41_b12x_prepare_global_kv", mutates_args=())
def _prepare_global_kv(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    prefix: str,
) -> torch.Tensor:
    context = get_forward_context()
    context.no_compile_layers[prefix]._prepare_global_kv(positions, hidden)
    # A small explicit data dependency orders the following opaque attention
    # call without asking functionalization to clone the aliased KV pool.
    return torch.empty(1, dtype=torch.uint8, device=hidden.device)


@_prepare_global_kv.register_fake
def _prepare_global_kv_fake(hidden, positions, prefix):
    return torch.empty(1, dtype=torch.uint8, device=hidden.device)


class DeepseekV4Attention(nn.Module, AttentionLayerBase):
    """The V4.1 attention layer: caches, metadata, projections and ownership. Its
    kernels are the TileLang layer's (``DeepseekV41TileLangAttention``), which
    defines ``_check_attention_rows``, ``_attention_scratch_specs``,
    ``_quantize_index_query``, ``_write_index_keys``, ``_index_chunk`` and
    ``_attend``."""

    backend_cls = DeepseekV41B12xBackend
    # Index scores need bounded row scratch; attention handles the whole batch.
    INDEX_CHUNK = 256
    DECODE_CHUNK = 64

    @classmethod
    def get_padded_num_q_heads(cls, num_heads):
        return num_heads

    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer=None,
        aux_stream_list=None,
    ):
        super().__init__()
        self.config = vllm_config
        hf = vllm_config.model_config.hf_config
        self.prefix = prefix
        layer_match = re.search(r"layers\.(\d+)", prefix)
        if layer_match is None:
            raise ValueError("V4.1 attention prefix must contain a layer index")
        self.layer_id = int(layer_match.group(1))
        self.hidden_size, self.head_dim = hf.hidden_size, hf.head_dim
        self.rope_head_dim = hf.qk_rope_head_dim
        tp = get_tensor_model_parallel_world_size()
        self.n_local_heads = hf.num_attention_heads // tp
        self.n_local_groups = hf.o_groups // tp
        self.n_groups, self.o_lora_rank = hf.o_groups, hf.o_lora_rank
        self.q_lora_rank, self.window_size = hf.q_lora_rank, hf.sliding_window
        spec = vllm_config.speculative_config
        self.is_draft = (
            self.layer_id >= hf.num_hidden_layers
            and spec is not None
            and spec.use_dspark()
        )
        boundary = ced_decoder_start(hf)
        self.is_ced_decoder = (
            not self.is_draft and boundary is not None and self.layer_id >= boundary
        )
        # With a CED decoder the drafter's context rows are its kept rows, so a
        # draft window starts at CED's replay start (bind_draft_replay_start).
        self._draft_replays = self.is_draft and boundary is not None
        self._draft_replay_start: torch.Tensor | None = None
        self.swa_width = self.window_size
        if self.is_draft:
            from vllm.v1.attention.backends.mla.compressor_utils import (
                get_dspark_swa_index_width,
            )

            self.swa_width = get_dspark_swa_index_width(
                self.window_size, spec.num_speculative_tokens
            )
        self.capacity = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.compress_ratio = (
            hf.compress_ratios[self.layer_id]
            if self.layer_id < len(hf.compress_ratios)
            else 0
        )
        self._main_page = vllm_config.cache_config.block_size // max(
            self.compress_ratio, 1
        )
        self._main_width = (
            self.max_model_len + vllm_config.cache_config.block_size - 1
        ) // vllm_config.cache_config.block_size
        self._index_page, self._index_width = self._main_page, self._main_width
        self.is_kv_source = self.layer_id in hf.kv_source_layer_ids
        self.is_index_source = self.layer_id in hf.index_source_layer_ids
        self.kv_source_layer_id = (
            max(s for s in hf.kv_source_layer_ids if s <= self.layer_id)
            if self.compress_ratio
            else None
        )
        self.index_source_layer_id = (
            max(s for s in hf.index_source_layer_ids if s <= self.layer_id)
            if self.compress_ratio
            else None
        )
        self.candidate_source_layer = hf.candidate_source_layer_id
        if (
            self.head_dim != 512
            or self.rope_head_dim != 64
            or self.compress_ratio not in (0, 1, 2)
        ):
            raise ValueError("unsupported V4.1 native attention geometry")
        if (
            vllm_config.parallel_config.decode_context_parallel_size != 1
            or vllm_config.parallel_config.prefill_context_parallel_size != 1
        ):
            raise ValueError(
                "V4.1 TP shards heads, not context; context parallel is unsupported"
            )
        self._context = vllm_config.compilation_config.static_forward_context
        if prefix in self._context:
            raise ValueError(f"Duplicate attention layer {prefix}")
        self._context[prefix] = self
        self.kv_cache = torch.tensor([])
        self.attn_sink = nn.Parameter(
            torch.full((self.n_local_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            hf.hidden_size,
            [hf.q_lora_rank, 512],
            bias=False,
            quant_config=vllm_config.quant_config,
            disable_tp=True,
            prefix=f"{prefix}.fused_wqa_wkv",
        )
        self.q_norm = kernels.rms_norm(hf.q_lora_rank, hf.rms_norm_eps)
        self.kv_norm = kernels.rms_norm(512, hf.rms_norm_eps)
        self.wq_b = ColumnParallelLinear(
            hf.q_lora_rank,
            hf.num_attention_heads * 512,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wo_a = ColumnParallelLinear(
            hf.num_attention_heads * 512 // hf.o_groups,
            hf.o_groups * hf.o_lora_rank,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm, self.wo_a.bmm_batch_size = True, self.n_local_groups
        self.wo_a.quant_method = _WOProjectionWeightMethod(self.wo_a.quant_method)
        self.wo_b = RowParallelLinear(
            hf.o_groups * hf.o_lora_rank,
            hf.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.wo_b",
            reduce_results=False,
        )
        self.wo_b.quant_method = _WOProjectionWeightMethod(self.wo_b.quant_method)
        if getattr(hf, "original_num_attention_heads", hf.num_attention_heads) != (
            hf.num_attention_heads
        ):
            for linear in (self.wq_b, self.wo_a, self.wo_b):
                for param in linear.parameters():
                    set_weight_attrs(param, {"allow_tp_padding": True})
        self._wo_projection_weights = None
        self._l2pf_wo = None
        self._l2pf_ffn = None
        self._l2pf_comp = None
        for linear in (self.fused_wqa_wkv, self.wq_b):
            _native_linear(linear)
        # Q-B reads its input's MXFP8 cast from the input's producer: the front's one
        # kernel at decode sizes (``_front``), TileKernels' cast otherwise.
        self.wq_b.tilelang_mxfp8_input = True
        self.rotary_emb = build_deepseek_v4_rope(
            hf,
            head_dim=512,
            rope_head_dim=64,
            max_position_embeddings=hf.max_position_embeddings,
            compress_ratio=self.compress_ratio,
        )
        self.swa_cache_layer = _Cache(
            vllm_config,
            f"{prefix}.swa_cache",
            kind="swa",
            window=self.window_size,
            draft=self.is_draft,
        )
        self.compressor = (
            DeepseekCompressor(
                vllm_config,
                self.compress_ratio,
                hf.hidden_size,
                512,
                prefix=f"{prefix}.compressor",
            )
            if self.is_kv_source
            else None
        )
        self.indexer = None
        if self.is_index_source:
            if self.is_kv_source:
                k_cache = _Cache(
                    vllm_config,
                    f"{prefix}.indexer.k_cache",
                    kind="index",
                    ratio=self.compress_ratio,
                )
            else:
                k_cache = self._context[
                    f"{_source(prefix, self.kv_source_layer_id)}.indexer.k_cache"
                ]
            self.indexer = DeepseekV4Indexer(
                vllm_config,
                f"{prefix}.indexer",
                owns_k=self.is_kv_source,
                k_cache=k_cache,
                ratio=self.compress_ratio,
            )
        self.topk_indices_buffer = topk_indices_buffer
        self._ready = False

    def get_attn_backend(self):
        return self.backend_cls

    def get_kv_cache_spec(self, config):
        if not self.is_kv_source:
            return None
        return MLAAttentionSpec(
            block_size=config.cache_config.block_size,
            num_kv_heads=1,
            head_size=512,
            state_content_bytes=288,
            dtype=torch.uint8,
            tokens_per_state=self.compress_ratio,
            cache_dtype_str="b12x_dsv41",
            alignment=None,
            uses_only_written_slots=True,
        )

    def bind_kv_cache(self, cache):
        self.kv_cache = cache.view(cache.shape[0], -1)

    def _owner(self):
        return self._context[_source(self.prefix, self.kv_source_layer_id)]

    def _attention_geometry(self) -> None:
        """The metadata arena of each attention mode and the staging this layer
        owns, from configuration alone."""
        # Page geometry stays as computed at construction, from the block size
        # the caches were allocated with. The engine later lowers
        # cache_config.block_size to its smallest prefix-cacheable group (the
        # SWA block); re-reading it here would plan half-size main pages
        # whenever the worker shares the engine's config (the uniproc executor).
        spec = self.config.speculative_config
        # Parallel drafting can admit two draft spans during verifier profiling.
        query_width = (
            1 + (2 if spec.parallel_drafting else 1) * spec.num_speculative_tokens
            if spec is not None
            else 1
        )
        decode_rows = self.config.scheduler_config.max_num_seqs * query_width
        # Graph buffers include padding beyond the live decode-token bound.
        decode_rows = max(
            decode_rows,
            self.config.compilation_config.max_cudagraph_capture_size or 0,
        )
        self._attention_workspace_specs = {}
        self._decode_capacity = min(self.capacity, decode_rows)
        for mode, capacity in (
            ("decode", self._decode_capacity),
            ("extend", self.capacity),
        ):
            metadata_specs: tuple[tuple[tuple[int, ...], torch.dtype], ...] = (
                ((capacity, self.swa_width), torch.int32),
                ((capacity,), torch.int32),
                ((capacity,), torch.int32),
            )
            if self.compress_ratio:
                metadata_specs += (((capacity, self._main_width), torch.int32),)
            self._attention_workspace_specs[mode] = metadata_specs
        if not hasattr(self, "_owns_topk_indices"):
            self._owns_topk_indices = (
                self.topk_indices_buffer is None and self.is_index_source
            )
        specs: list[tuple[str, tuple[int, ...], torch.dtype]] = []
        if self._owns_topk_indices:
            specs.append(("topk_indices_buffer", (self.capacity, 512), torch.int32))
        if self.indexer is not None:
            specs.extend(
                (
                    (
                        "_index_pages",
                        (self.INDEX_CHUNK, self._index_width),
                        torch.int32,
                    ),
                    ("_active", (1,), torch.int32),
                )
            )
            if self.layer_id == self.candidate_source_layer:
                specs.extend(
                    (
                        # The block ids the candidate source publishes, a row.
                        ("_candidates", (self.capacity, CANDIDATE_BLOCKS), torch.int32),
                        ("_candidate_lens", (self.capacity,), torch.int32),
                    )
                )
        self._short_index_shape = None
        if (
            self.indexer is not None
            and not self.is_ced_decoder
            and self.capacity > self.INDEX_CHUNK
            and self.layer_id < self.candidate_source_layer
        ):
            short_rows = min(1024, self.capacity)
            short_width = min(self._index_width, cdiv(16384, self._index_page))
            self._short_index_shape = (short_rows, short_width)
            specs.append(("_short_index_pages", self._short_index_shape, torch.int32))
        self._staging_specs = tuple(specs)

    def declare_kernels(self) -> None:
        """Declare every TileLang kernel this layer can launch, its WO projection's
        included (the worker calls this after loading, before any distributed
        work)."""
        from vllm.models.deepseek_v4_1.tilelang.linear import _warm_casts

        self._attention_geometry()
        calls = self._tilelang_calls()
        # Q-B's input cast where the front's one kernel does not write it.
        prepare.declare(lambda: calls, warm=lambda: _warm_casts(self.q_lora_rank))
        declare_wo = getattr(self._wo_projection_weights, "declare_kernels", None)
        if declare_wo is not None:
            declare_wo()

    def _owned_caches(self) -> list:
        """The cache layers this layer allocates: one per KV-cache group it joins."""
        caches = [self.swa_cache_layer]
        if self.is_kv_source:
            caches.append(self)
        if self.indexer is not None and self.indexer.owns_k:
            caches.append(self.indexer.k_cache)
        if self.compressor is not None and self.compressor.state_cache is not None:
            caches.append(self.compressor.state_cache)
        return caches

    def _tilelang_calls(self) -> list:
        """The TileLang kernels of the shared layer paths, as ``prepare.call``s:
        window metadata and index page tables, the metadata builders of the cache
        groups this layer joins, the RoPE of every role and the cache writers, each
        in its call site's call form."""
        from vllm.models.deepseek_v4_1.tilelang import metadata as tl_metadata
        from vllm.models.deepseek_v4_1.tilelang.cache_writer import writer_call
        from vllm.models.deepseek_v4_1.tilelang.front import front_call
        from vllm.models.deepseek_v4_1.tilelang.rope import rope_call

        calls = []
        # _write_decode_metadata: the window metadata, with the main page table in
        # the same launch on compressed layers. A CED decoder's rows start no
        # earlier than their request's replay start when the step has one; a
        # drafter's beside a CED decoder always do.
        replays = (
            (True,)
            if self._draft_replays
            else (False, True)
            if self.is_ced_decoder
            else (False,)
        )
        for replay in replays:
            args = (
                self.swa_cache_layer.block_size,
                self.window_size,
                self.swa_width,
                self.is_draft,
                replay,
            )
            if self.compress_ratio:
                calls.append(
                    prepare.call(
                        tl_metadata.decode_metadata_kernel,
                        *args,
                        self._main_width,
                        self._main_page,
                    )
                )
            else:
                calls.append(prepare.call(tl_metadata.chunk_kernel, *args))
        # forward_mqa: the page tables of the index chunks, full or short.
        if self.indexer is not None:
            widths = {self._index_width}
            if self._short_index_shape is not None:
                widths.add(self._short_index_shape[1])
            calls.extend(
                prepare.call(tl_metadata.pages_kernel, width, self._index_page)
                for width in widths
            )
        # Each group's metadata builder: its spec as the KV-cache spec code makes
        # it, for the scheduler's requests.
        requests = self.config.scheduler_config.max_num_seqs
        for cache in self._owned_caches():
            calls.extend(
                tilelang_builder_kernels(cache.get_kv_cache_spec(self.config), requests)
            )
        # _rope (heads, columns, ratio) by role (WO's input is rotated back in the
        # attention kernels' output store).
        table_width = self.rotary_emb.cos_sin_cache.shape[1]
        roles = [
            (self.n_local_heads, self.head_dim, 1, False),  # q
            (1, 512, 1, False),  # kv
        ]
        if self.indexer is not None:
            roles.append((self.indexer.heads, 128, 1, False))  # index query
        if self.compressor is not None:
            roles.append((1, 128, self.compress_ratio, False))  # index key
            roles.append((1, 512, self.compress_ratio, False))  # latent
        calls.extend(
            rope_call(heads, dim, table_width, ratio, inverse)
            for heads, dim, ratio, inverse in roles
        )
        # _write_v41_cache: SWA records of every layer, indexed records of the
        # compressor's latents, at the builders' (and compressor's) slot dtype.
        calls.append(writer_call("swa", self.swa_cache_layer.block_size, SLOT_DTYPE))
        # _front: the decode front's one kernel.
        calls.append(
            front_call(
                self.q_norm.variance_epsilon,
                self.swa_cache_layer.block_size,
                SLOT_DTYPE,
            )
        )
        if self.compressor is not None:
            calls.append(writer_call("indexed", self._main_page, SLOT_DTYPE))
        else:
            # _gather_front: prefill SP's BF16 front, the index head weights beside
            # the normalized halves.
            from vllm.models.deepseek_v4_1.tilelang import columns

            widths = (self.q_lora_rank, 512)
            if self.indexer is not None:
                widths += (self.indexer.heads,)
                key = columns.copy_columns_key(self.indexer.heads, torch.bfloat16)
                calls.append(prepare.call(columns.copy_columns_kernel, *key))
            key = columns.split_columns_key(widths, torch.bfloat16)
            calls.append(prepare.call(columns.split_columns_kernel, *key))
        return calls

    def _prepare(self, device):
        """Materialize admitted integration-owned metadata, never launch choices."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V4.1 metadata must be prepared before graph capture")
        if not hasattr(self, "_staging_specs"):
            self._attention_geometry()
        for name, shape, dtype in self._staging_specs:
            tensor = getattr(self, name, None)
            if tensor is None or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                setattr(self, name, torch.empty(shape, dtype=dtype, device=device))
        if self.indexer is not None:
            self._active.fill_(self._index_width * self._index_page)
        self._ready = True

    def _rope(self, x, positions, role: str):
        """``x`` rotated at ``positions``, in place."""
        from vllm.models.deepseek_v4_1.tilelang.rope import rope_

        table = self.rotary_emb.cos_sin_cache
        compressed = role in ("index_key", "latent")
        rope_(x, positions, table, self.compress_ratio if compressed else 1, False)
        return x

    def _write_v41_cache(self, x, cache, slots, *, page_size, cache_kind):
        """Write ``x``'s V4.1 records (``cache_kind`` swa or indexed) at ``slots``."""
        from .tilelang.cache_writer import write_cache

        write_cache(x, cache, slots, page_size=page_size, cache_kind=cache_kind)

    def insert_context_kv(self, kv, positions, slot_mapping):
        if not self._ready:
            raise RuntimeError("V4.1 attention metadata is not prepared")
        rotated = self._rope(kv, positions, "kv")
        self._write_v41_cache(
            rotated,
            self.swa_cache_layer.kv_cache,
            slot_mapping,
            page_size=self.swa_cache_layer.block_size,
            cache_kind="swa",
        )

    def _query_metadata(self, metadata):
        if self.is_ced_decoder and metadata.decoder is not None:
            return metadata.decoder
        return metadata

    @eager_break_during_capture
    def _cache_context_kv(self, kv, positions):
        # Resolve live request metadata inside the eager segment. A captured
        # slot-mapping argument could refer to a previous request's cache pages.
        metadata = get_forward_context().attn_metadata
        swa = self._query_metadata(metadata[self.swa_cache_layer.prefix])
        self.insert_context_kv(kv, positions, swa.slot_mapping[: kv.shape[0]])

    def prepare_global_kv(self, positions, hidden_states):
        """Build full-row global KV before the CED boundary gathers decoder rows."""
        if self.compressor is None or hidden_states.shape[0] == 0:
            return
        return _prepare_global_kv(hidden_states, positions, self.prefix)

    @eager_break_during_capture
    def _prepare_global_kv(self, positions, hidden_states):
        if not self._ready:
            raise RuntimeError("V4.1 attention metadata is not prepared")
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or self.compressor is None:
            return
        indexer = self.indexer
        if indexer is None:
            raise RuntimeError("V4.1 global KV source requires an indexer")
        rows = hidden_states.shape[0]
        # These are deliberately original metadata, never the decoder views.
        main = metadata[self._owner().prefix]
        state = (
            metadata[self.compressor.state_cache.prefix]
            if self.compressor.state_cache is not None
            else None
        )
        latent, slots = self.compressor(hidden_states, main, state)
        self._issue_wo_prefetch(rows, after="compressor")
        # Index K consumes the ordinary-normalized PRE-RoPE latent.
        key = indexer.k_norm(indexer.wk(latent))
        key = self._rope(key, positions, "index_key")
        index_meta = cast(DeepseekV41B12xMetadata, metadata[indexer.k_cache.prefix])
        self._write_index_keys(key, index_meta.slot_mapping[:rows])
        latent = self._rope(latent, positions, "latent")
        self._write_v41_cache(
            latent,
            self.kv_cache,
            slots,
            page_size=self._main_page,
            cache_kind="indexed",
        )

    def forward(
        self, positions, hidden_states, llama_4_scaling=None, *, global_kv_ready=None
    ):
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)
        return _attention(hidden_states, positions, self.prefix, global_kv_ready)

    def _gather_front(self, sp_rows, hidden_states, qr, kv):
        """Sequence-parallel prefill's row-wise front of every row, as contiguous
        parts (normalized Q, normalized KV, the indexer's head weights, all BF16):
        one all-gather of this rank's rows side by side. The norms write their
        halves into the gather's buffer in place and the head weights are copied
        beside them."""
        from vllm.models.deepseek_v4_1.tilelang import columns

        q = self.q_lora_rank
        widths = (q, 512) + (() if self.indexer is None else (self.indexer.heads,))
        front = qr.new_empty(qr.shape[0], sum(widths))
        self.q_norm(qr, out=front[:, :q])
        self.kv_norm(kv, out=front[:, q : q + 512])
        if self.indexer is not None:
            weights = self.indexer.weights_proj(hidden_states)
            columns.copy_columns(weights, front[:, q + 512 :])
        return columns.split_columns(sp_rows.gather(front), widths)

    def _front(self, qr_kv, positions, sp_rows):
        """The decode front after the fused projection in one launch
        (``tilelang.front``): the normalized q latent and its MXFP8 cast, and the
        normalized kv latent rotated, with its SWA record written. None where the
        separate kernels run: sequence-parallel or prefill rows past a decode tile,
        memory profiling (no cache pages), and breakable graphs, which write the
        record from live metadata in an eager segment (``_cache_context_kv``)."""
        from vllm.models.deepseek_v4_1.tilelang.front import attention_front
        from vllm.models.deepseek_v4_1.tilelang.linear import empty_mxfp8

        rows = qr_kv.shape[0]
        if (
            sp_rows is not None
            or rows > DECODE_ROWS
            or is_breakable_cudagraph_enabled()
        ):
            return None
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None
        swa = self._query_metadata(metadata[self.swa_cache_layer.prefix])
        qr = torch.empty(
            (rows, self.q_lora_rank), dtype=qr_kv.dtype, device=qr_kv.device
        )
        kv = torch.empty((rows, 512), dtype=qr_kv.dtype, device=qr_kv.device)
        q_cast = empty_mxfp8(qr)
        attention_front(
            qr_kv,
            self.q_norm.weight,
            self.kv_norm.weight,
            self.q_norm.variance_epsilon,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.swa_cache_layer.kv_cache,
            swa.slot_mapping[:rows],
            self.swa_cache_layer.block_size,
            qr,
            q_cast.values,
            q_cast.scales,
            kv,
        )
        return qr, kv, q_cast

    def _forward(self, positions, hidden_states):
        if not self._ready:
            raise RuntimeError("V4.1 attention metadata is not prepared")
        sp_rows = sp_prefill.current()
        index_weights = None
        front = None
        if sp_rows is not None and self.compressor is None:
            # Sequence-parallel prefill without a compressor: the row-wise front
            # (projection, norms, index weights) runs on this rank's rows, and
            # its product, narrower than the hidden state, is gathered.
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            qr, kv = qr_kv.split((self.q_lora_rank, 512), dim=-1)
            parts = self._gather_front(sp_rows, hidden_states, qr, kv)
            qr, kv = parts[0], parts[1]
            if self.indexer is not None:
                index_weights = parts[2]
            hidden_states = None
            rows = sp_rows.num_tokens
        else:
            gathered = None
            if sp_rows is not None:
                # Sequence-parallel prefill: attention runs on every row, and the
                # projection reads the gathered rows' MXFP8 cast.
                hidden_states = sp_rows.gather(hidden_states)
                gathered = kernels.cast_mxfp8(hidden_states)
            rows = hidden_states.shape[0]
            with kernels.mxfp8_inputs(gathered):
                qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            front = self._front(qr_kv, positions, sp_rows)
            if front is None:
                qr, kv = qr_kv.split((self.q_lora_rank, 512), dim=-1)
                qr, kv = self.q_norm(qr), self.kv_norm(kv)
        if front is None:
            q_cast = kernels.cast_mxfp8(qr)
        else:
            qr, kv, q_cast = front
        with kernels.mxfp8_inputs(q_cast):
            q = self.wq_b(qr).view(rows, self.n_local_heads, 512)
        self._issue_wo_prefetch(rows, after="q-b")
        q = self._rope(q, positions, "q")
        output = torch.empty_like(q)
        metadata = cast(
            dict[str, DeepseekV41B12xMetadata] | None,
            get_forward_context().attn_metadata,
        )
        if not isinstance(metadata, dict):
            # Memory profiling has no cache pages; serving does not use this branch.
            output.zero_()
        else:
            original_swa = metadata[self.swa_cache_layer.prefix]
            swa = self._query_metadata(original_swa)
            if front is None:  # the front's kernel wrote the record
                self._cache_context_kv(kv, positions)
            if swa is original_swa and hidden_states is not None:
                self._prepare_global_kv(positions, hidden_states)
            index_query = None
            if self.indexer is not None:
                # Sequence-parallel prefill: the indexer is replicated, so each
                # rank scores and selects only its own rows, and forward_mqa
                # gathers the selected positions. Every row is independent.
                split = sp_rows if sp_rows is not None and not swa.is_decode else None
                first = split.start if split is not None else 0
                last = first + split.real_rows if split is not None else rows
                query = self._index_query(
                    qr, positions, hidden_states, index_weights, first, last, swa
                )
                reference = None
                if split is not None and _INDEX_SPLIT_CHECK:
                    reference = self._index_query(
                        qr, positions, hidden_states, index_weights, 0, rows, swa
                    )
                index_query = (*query, split, reference)
            self.forward_mqa(q, kv, positions, output, index_query=index_query)
        return self._o_proj(output, sp_rows=sp_rows)

    def _issue_wo_prefetch(self, rows, *, after):
        """Issue the WO window's L2 prefetch at its point in this layer's forward
        (``l2_prefetch``): after Q-B, or right after the compressor in a layer that
        owns one. There the compressor's projection between Q-B and WO-A would evict
        WO's lines; the COMP window, the compressor's own weight, takes WO's place
        after Q-B, and WO's fill queues behind it."""
        if after == ("q-b" if self.compressor is None else "compressor"):
            l2_prefetch.issue(self._l2pf_wo, rows)
        if after == "q-b":
            l2_prefetch.issue(self._l2pf_comp, rows)

    def _index_query(
        self, qr, positions, hidden_states, index_weights, first, last, swa
    ):
        """MXFP4 index queries and scaled head weights for rows [first, last)."""
        rows = last - first
        h = self.indexer.heads
        iq = self.indexer.wq_b(qr[first:last]).view(rows, h, 128)
        iq = self._rope(iq, positions[first:last], "index_query")
        iq_data = torch.empty((rows, h, 64), dtype=torch.uint8, device=qr.device)
        iq_scale = torch.empty((rows, h, 4), dtype=torch.uint8, device=qr.device)
        self._quantize_index_query(
            "decode" if swa.is_decode else "prefill", iq, iq_data, iq_scale
        )
        # The BF16 head-weight projection runs on every row: its GEMV rounds
        # differently at another row count, which would move top-k selections.
        # The FP8 wq_b quantizes per row and does not depend on the row count.
        weights = (
            index_weights[first:last]
            if index_weights is not None
            else self.indexer.weights_proj(hidden_states)[first:last]
        )
        return iq_data, iq_scale, weights  # the projection applied the scale

    @staticmethod
    def _selection_spread(actual, expected):
        """Rows whose selected positions differ, and positions absent from `expected`.

        Unused slots (-1) are ignored; only real positions count.
        """
        actual = actual.sort(dim=-1).values
        expected = expected.sort(dim=-1).values
        where = torch.searchsorted(expected, actual).clamp_max(expected.shape[-1] - 1)
        missing = ((expected.gather(-1, where) != actual) & (actual >= 0)).sum(dim=-1)
        return (
            int((missing > 0).sum()),
            int(missing.sum()),
            int(missing.max()) if missing.numel() else 0,
        )

    def _check_index_split(
        self, split, selected, expected, repeat, expected_candidates
    ):
        """Compare this rank's split selection with the full-row indexer.

        The radix top-k keeps an arbitrary subset of the positions tied at its
        threshold, so two full-row runs can differ; `repeat` is a second run
        and measures that spread. The split passes when it differs from the
        first run no more than that, writes every row, and leaves the
        candidate lists unchanged.
        """
        real = split.real_rows
        rows = slice(split.start, split.start + real)
        actual = selected[:real]
        unwritten = int((actual == -2).any(dim=-1).sum())
        split_rows, split_positions, split_max = self._selection_spread(
            actual, expected
        )
        repeat_rows, repeat_positions, repeat_max = self._selection_spread(
            repeat, expected
        )
        bad_candidates = 0
        if expected_candidates is not None:
            candidates, lengths = expected_candidates
            bad_candidates = int(
                (
                    (self._candidates[rows] != candidates).any(dim=-1)
                    | (self._candidate_lens[rows] != lengths)
                ).sum()
            )
        log = logger.warning if unwritten or bad_candidates else logger.info
        log(
            "DeepSeek V4.1 indexer split check: layer %d rank %d rows %d-%d of %d: "
            "split vs full %d rows, %d positions (max %d); full vs full %d rows, "
            "%d positions (max %d); %d unwritten rows; %d candidate rows differ",
            self.layer_id,
            split.rank,
            rows.start,
            rows.stop,
            split.num_tokens,
            split_rows,
            split_positions,
            split_max,
            repeat_rows,
            repeat_positions,
            repeat_max,
            unwritten,
            bad_candidates,
        )

    @eager_break_during_capture
    def forward_mqa(self, q, kv, positions, output, *, index_query=None):
        metadata = get_forward_context().attn_metadata
        swa = self._query_metadata(metadata[self.swa_cache_layer.prefix])
        main = (
            self._query_metadata(metadata[self._owner().prefix])
            if self.compress_ratio
            else None
        )
        mode = "decode" if swa.is_decode else "extend"
        rows = q.shape[0]
        self._check_attention_rows(mode, rows)
        # A mixed step's decode requests, its leading rows (the batch is reordered
        # decode-first), keep the decode plan and the decode index mode: a row's
        # arithmetic must not depend on whether a prefill shares its step.
        decode_rows = 0
        if mode == "extend":
            decode_rows = min(getattr(swa, "num_decode_tokens", 0), rows)
            if decode_rows:
                self._check_attention_rows("decode", decode_rows)
        owner = (
            self._context[_source(self.prefix, self.index_source_layer_id)]
            if self.compress_ratio
            else self
        )

        # Only the score matrix needs row chunking. All selected positions
        # survive in the source-owned buffer until their reuse interval ends.
        if self.indexer is not None:
            index_mode = "decode" if mode == "decode" else "prefill"
            *query, split, reference = index_query
            im = self._query_metadata(metadata[self.indexer.k_cache.prefix])
            score_width = None
            if (
                mode == "extend"
                and self.layer_id <= self.candidate_source_layer
                and not torch.cuda.is_current_stream_capturing()
            ):
                score_width = min(
                    self._index_width * self._index_page,
                    max(1, cdiv(im.max_seq_len, self.compress_ratio)),
                )
            chunk_rows = self.DECODE_CHUNK if mode == "decode" else self.INDEX_CHUNK
            index_pages = self._index_pages
            if (
                mode == "extend"
                and score_width is not None
                and score_width <= 16384
                and self._short_index_shape is not None
            ):
                index_mode = "prefill_short"
                chunk_rows = self._short_index_shape[0]
                index_pages = self._short_index_pages
            rest = (index_mode, chunk_rows, index_pages, score_width)
            decode = ("decode", self.DECODE_CHUNK, self._index_pages, None)

            def select(query, first, selected):
                # Decode requests' rows (global rows below decode_rows) take the
                # decode index mode, as in a step of decode requests alone.
                count = query[0].shape[0]
                cut = min(max(decode_rows - first, 0), count)
                if cut:
                    select_rows([t[:cut] for t in query], first, selected[:cut], decode)
                if cut < count:
                    select_rows(
                        [t[cut:] for t in query], first + cut, selected[cut:], rest
                    )

            def select_rows(query, first, selected, params):
                # Query rows start at global row `first`; metadata and
                # candidates are indexed by global row, `selected` by query row.
                index_mode, chunk_rows, index_pages, score_width = params
                index_width = index_pages.shape[1]
                iq_data, iq_scale, iw = query
                index_rows = iq_data.shape[0]
                for offset in range(0, index_rows, chunk_rows):
                    end = min(offset + chunk_rows, index_rows)
                    count = end - offset
                    row, row_end = first + offset, first + end
                    from vllm.models.deepseek_v4_1.tilelang import (
                        metadata as tl_metadata,
                    )

                    tl_metadata.pages(
                        im.req_id_per_token,
                        im.cache_lengths,
                        im.block_table,
                        index_pages,
                        row,
                        count,
                        min(im.block_table.shape[1], index_width),
                        page=self._index_page,
                    )
                    candidate_args = {}
                    if self.layer_id == self.candidate_source_layer:
                        candidate_args = dict(
                            candidate_output=self._candidates[row:row_end],
                            candidate_output_lengths=self._candidate_lens[row:row_end],
                        )
                    elif self.layer_id > self.candidate_source_layer:
                        source = self._context[
                            _source(self.prefix, self.candidate_source_layer)
                        ]
                        candidate_args = dict(
                            candidate_indices=source._candidates[row:row_end],
                            candidate_lengths=source._candidate_lens[row:row_end],
                        )
                    self._index_chunk(
                        index_mode,
                        count,
                        q_mxfp4=iq_data[offset:end],
                        q_scales=iq_scale[offset:end],
                        query_weights=iw[offset:end],
                        page_table=index_pages[:count],
                        cache_lengths=im.cache_lengths[row:row_end],
                        score_width=score_width,
                        output_indices=selected[offset:end],
                        **candidate_args,
                    )

            if split is None:
                select(query, 0, self.topk_indices_buffer)
            else:
                real = split.real_rows
                local = slice(split.start, split.start + real)
                selected = torch.empty(
                    (split.local_rows, 512), dtype=torch.int32, device=q.device
                )
                owns_candidates = self.layer_id == self.candidate_source_layer
                expected = repeat = expected_candidates = None
                if reference is not None:
                    # Sentinels make unwritten rows visible to the comparison.
                    self.topk_indices_buffer[:rows].fill_(-2)
                    if owns_candidates:
                        self._candidates[:rows].fill_(-2)
                        self._candidate_lens[:rows].fill_(-2)
                    select(reference, 0, self.topk_indices_buffer)
                    expected = self.topk_indices_buffer[local].clone()
                    if owns_candidates:
                        expected_candidates = (
                            self._candidates[local].clone(),
                            self._candidate_lens[local].clone(),
                        )
                    # A second full-row run measures the top-k's own spread.
                    repeat = torch.full(
                        (rows, 512), -2, dtype=torch.int32, device=q.device
                    )
                    select(reference, 0, repeat)
                    repeat = repeat[local]
                    selected.fill_(-2)
                    if owns_candidates:
                        self._candidates[local].fill_(-2)
                        self._candidate_lens[local].fill_(-2)
                select(query, split.start, selected)
                if reference is not None:
                    self._check_index_split(
                        split, selected, expected, repeat, expected_candidates
                    )
                self.topk_indices_buffer[:rows].copy_(split.gather(selected))

        if decode_rows:
            self._attend_split(
                q, output, positions, swa, main, owner, rows, decode_rows
            )
            return
        # Reuse the shared arena only after indexing completes. Keep full-batch
        # metadata beside, not overlapping, the native attention scratch.
        metadata_specs = self._attention_workspace_specs[mode]
        arena = current_workspace_manager().get_simultaneous(
            *metadata_specs, *self._attention_scratch_specs(mode, rows)
        )
        buffers = arena
        metadata_count = len(metadata_specs)
        # Decode layers sharing cache groups map identical pages, so the first
        # layer of each group keeps its metadata for the rest of the forward.
        # Every replay rewrites it before use, so it is an ordinary graph-pool
        # activation: retaining it would pin one copy per captured graph.
        if mode == "decode":
            key = (
                id(swa),
                id(main),
                self.is_draft,
                self.is_ced_decoder,
                rows,
                metadata_specs,
            )
            context = get_forward_context()
            shared = context.__dict__.setdefault("_ds41_decode_metadata", {})
            entry = shared.get(key)
            if entry is None or entry[0] is not swa or entry[1] is not main:
                metadata_buffers = tuple(
                    torch.empty(shape, dtype=dtype, device=q.device)
                    for shape, dtype in metadata_specs
                )
                self._write_decode_metadata(metadata_buffers, swa, main, rows)
                entry = shared[key] = (swa, main, metadata_buffers)
            buffers = [*entry[2], *arena[metadata_count:]]
        else:
            self._write_decode_metadata(buffers, swa, main, rows)
        swa_indices, swa_lengths, top_lengths = buffers[:3]
        kwargs = {}
        if main is not None:
            kwargs = dict(
                indexed_indices=owner.topk_indices_buffer[:rows],
                indexed_lengths=top_lengths[:rows],
                indexed_page_table=buffers[3][:rows],
            )
        # Keep the borrowed arena, not aliases of the transient query storage.
        retain_cuda_graph_capture_resource(arena)
        self._attend(
            mode,
            q,
            output,
            scratch=buffers[metadata_count:],
            positions=positions,
            swa_indices=swa_indices[:rows],
            swa_lengths=swa_lengths[:rows],
            **kwargs,
        )

    def _attend_split(self, q, output, positions, swa, main, owner, rows, decode_rows):
        """A mixed step's attention: its decode requests' rows on the decode plan,
        the rest on the extend plan. Mixed steps run eagerly, so the per-row
        metadata is an ordinary allocation and each plan borrows the workspace in
        turn (stream order keeps the second from overwriting the first's scratch
        while it runs)."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V4.1 mixed-step attention is never captured")
        metadata = tuple(
            torch.empty((rows, *shape[1:]), dtype=dtype, device=q.device)
            for shape, dtype in self._attention_workspace_specs["extend"]
        )
        self._write_decode_metadata(metadata, swa, main, rows)
        swa_indices, swa_lengths, top_lengths = metadata[:3]
        for mode, lo, hi in (("decode", 0, decode_rows), ("extend", decode_rows, rows)):
            if hi == lo:
                continue
            scratch = current_workspace_manager().get_simultaneous(
                *self._attention_scratch_specs(mode, hi - lo)
            )
            kwargs = {}
            if main is not None:
                kwargs = dict(
                    indexed_indices=owner.topk_indices_buffer[lo:hi],
                    indexed_lengths=top_lengths[lo:hi],
                    indexed_page_table=metadata[3][lo:hi],
                )
            self._attend(
                mode,
                q[lo:hi],
                output[lo:hi],
                scratch=list(scratch),
                positions=positions[lo:hi],
                swa_indices=swa_indices[lo:hi],
                swa_lengths=swa_lengths[lo:hi],
                **kwargs,
            )

    def bind_draft_replay_start(self, replay_start: torch.Tensor) -> None:
        """A drafter layer's window bound: CED's per-request replay start (the
        model state's persistent buffer, rewritten every step before the drafter
        runs). Before it, the request's draft window was never written: the
        decoder kept no rows there, or they precede a prefix hit."""
        assert self._draft_replays
        self._draft_replay_start = replay_start

    def _replay_start(self, swa):
        if self.is_ced_decoder:
            return swa.swa_replay_start
        if self._draft_replays:
            return self._draft_replay_start
        return None

    def _write_decode_metadata(self, buffers, swa, main, rows):
        from vllm.models.deepseek_v4_1.tilelang import metadata as tl_metadata

        swa_indices, swa_lengths, top_lengths = buffers[:3]
        visible = main.cache_lengths if main is not None else swa.cache_lengths
        window = dict(
            page=self.swa_cache_layer.block_size,
            window=self.window_size,
            draft=self.is_draft,
            replay_start=self._replay_start(swa),
        )
        common = (
            swa.positions,
            swa.req_id_per_token,
            swa.block_table,
            swa_indices,
            swa_lengths,
            top_lengths,
            visible,
            swa.query_start_loc,
            swa.request_positions,
        )
        if main is None:
            tl_metadata.chunk(*common, 0, rows, **window)
        else:
            # The window metadata and the main page table in one launch.
            tl_metadata.decode_metadata(
                *common,
                main.req_id_per_token,
                main.block_table,
                buffers[3],
                0,
                rows,
                main.block_table.shape[1],
                main_page=self._main_page,
                **window,
            )

    def setup_wo_projection(self):
        groups = self.n_local_groups
        heads_per_group = self.n_local_heads // groups
        group_width = heads_per_group * self.head_dim
        rank = self.o_lora_rank
        hidden = self.hidden_size
        expected = (
            ("WO-A weight", self.wo_a.weight, (groups * rank, group_width)),
            (
                "WO-A scale",
                self.wo_a.weight_scale_inv,
                (groups * (rank // 32), group_width // 32),
            ),
            ("WO-B weight", self.wo_b.weight, (hidden, groups * rank)),
            (
                "WO-B scale",
                self.wo_b.weight_scale_inv,
                (hidden // 32, groups * rank // 32),
            ),
        )
        for name, tensor, shape in expected:
            if tuple(tensor.shape) != shape:
                raise RuntimeError(
                    f"V4.1 {name} shape mismatch: expected {shape}, "
                    f"got {tuple(tensor.shape)}"
                )
        from vllm.models.deepseek_v4_1.tilelang.wo import TileLangWOProjection

        # The checkpoint tensors in place, no packing.
        self._wo_projection_weights = TileLangWOProjection(
            self.wo_a, self.wo_b, groups=groups
        )

    def _o_proj(self, o, sp_rows=None):
        """The TileLang WO projection of WO-A's input ``o``, which the attention
        kernels store already rotated back and through ``act_quant``.

        A sequence-parallel step projects through ``project_reduce_scatter``: in
        slices, and as unreduced rank-order parts when the step keeps them, so a
        row's sum over ranks is the non-SP step's all-reduce."""
        rows, hidden = o.shape[0], self.hidden_size
        if sp_rows is not None:
            if rows != sp_rows.num_tokens:
                raise ValueError(f"V4.1 SP WO expects {sp_rows.num_tokens} rows")

            def project(start, stop, out):
                self._wo_projection_weights(o[start:stop], out)

            return sp_rows.project_reduce_scatter(
                project,
                (hidden,),
                torch.bfloat16,
                o.device,
                slices=sp_prefill.wo_slices(rows),
                then=lambda: l2_prefetch.issue(self._l2pf_ffn, rows),
            )
        local = torch.empty((rows, hidden), dtype=torch.bfloat16, device=o.device)
        self._wo_projection_weights(o, local)
        l2_prefetch.issue(self._l2pf_ffn, rows)
        if get_tensor_model_parallel_world_size() > 1:
            local = get_tp_group().all_reduce(local)
        return local
