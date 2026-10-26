# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-exact V4.1 cache metadata. No foreign attention scheduler is involved."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import CacheConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheLayout


@dataclass
class DeepseekV41Metadata:
    num_actual_tokens: int
    num_reqs: int
    max_query_len: int
    block_size: int
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    request_positions: torch.Tensor
    live_counts: torch.Tensor
    positions: torch.Tensor
    req_id_per_token: torch.Tensor
    slot_mapping: torch.Tensor
    cache_lengths: torch.Tensor
    is_decode: bool
    max_seq_len: int
    decoder: "DeepseekV41Metadata | None" = None
    swa_replay_start: torch.Tensor | None = None
    # Rows the CED decoder's MoE routes (False for padding), in its compact layout.
    routing_mask: torch.Tensor | None = None
    # The leading rows of decode requests (the batch is reordered decode-first):
    # every row of a decode step; in a mixed step, the rows attention keeps on the
    # decode plan so their arithmetic does not depend on the step.
    num_decode_tokens: int = 0


def _group_geometry(kv_cache_spec) -> tuple[int, int, bool]:
    """A cache group's ``(tokens per state, states per page, request ring)``."""
    return (
        int(kv_cache_spec.tokens_per_state),
        int(kv_cache_spec.num_states),
        isinstance(kv_cache_spec, CircularBufferSpec),
    )


def tilelang_builder_kernels(kv_cache_spec, max_requests: int) -> list:
    """The TileLang metadata kernels a builder of ``kv_cache_spec``'s group launches
    for ``max_requests`` requests, as ``prepare.call``s. The attention layers that
    own the group's caches declare them; the builder finds them prepared."""
    from vllm.models.deepseek_v4_1.tilelang import metadata as tl_metadata
    from vllm.models.deepseek_v4_1.tilelang import prepare

    ratio, page, circular = _group_geometry(kv_cache_spec)
    return [
        prepare.call(tl_metadata.builder_kernel, max_requests, page, ratio, circular)
    ]


# The slot-mapping rows a builder publishes: the cache writers' slot dtype.
SLOT_DTYPE = torch.int64


class DeepseekV41MetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.requests = vllm_config.scheduler_config.max_num_seqs
        self.ratio, self.page, self.circular = _group_geometry(kv_cache_spec)

        def alloc(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=device)

        self.starts = alloc((self.requests + 1,), torch.int32)
        self.request_positions = alloc((self.requests,), torch.int64)
        self.counts = alloc((2,), torch.int32)
        self.positions = alloc((self.tokens,), torch.int64)
        self.reqs = alloc((self.tokens,), torch.int32)
        self.slots = alloc((self.tokens,), SLOT_DTYPE)
        self.lengths = alloc((self.tokens,), torch.int32)
        # The attention layers owning this group's caches declare these kernels;
        # here, and at every rebuild, they are cache hits.
        for function, args, kwargs in tilelang_builder_kernels(
            kv_cache_spec, self.requests
        ):
            function(*args, **kwargs)

    def build(
        self,
        common_prefix_len,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build=False,
    ):
        cm = common_attn_metadata
        table = cm.block_table_tensor
        self._build_tilelang(cm, table)
        return self._metadata(cm, table)

    def _build_tilelang(self, cm, table) -> None:
        """The step's request starts and positions, then each token's request,
        position, slot and visible length, in one launch."""
        from vllm.models.deepseek_v4_1.tilelang import metadata as tl_metadata

        tl_metadata.builder(
            cm.query_start_loc,
            cm.seq_lens,
            table,
            cm.slot_mapping,
            self.starts,
            self.request_positions,
            self.counts,
            self.positions,
            self.reqs,
            self.slots,
            self.lengths,
            cm.num_reqs,
            cm.num_actual_tokens,
            page=self.page,
            ratio=self.ratio,
            circular=self.circular,
        )

    def _metadata(self, cm, table):
        is_decode = (
            self.reorder_batch_threshold is not None
            and cm.max_query_len <= self.reorder_batch_threshold
        )
        if is_decode:
            decode_tokens = cm.num_actual_tokens
        elif self.reorder_batch_threshold is not None:
            decode_tokens = split_decodes_and_prefills(
                cm, decode_threshold=self.reorder_batch_threshold
            )[2]
        else:
            decode_tokens = 0
        return DeepseekV41Metadata(
            cm.num_actual_tokens,
            cm.num_reqs,
            cm.max_query_len,
            self.page,
            table,
            self.starts,
            self.request_positions,
            self.counts,
            self.positions,
            self.reqs,
            self.slots,
            self.lengths,
            is_decode,
            cm.max_seq_len,
            num_decode_tokens=decode_tokens,
        )


class DeepseekV41Backend(AttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "fp8", "fp8_ds_mla"]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # SWA, main KV, index keys, and partial states have different byte pages.
        return (KVCacheLayout.BLHNC,)

    @staticmethod
    def get_name():
        return "DSV41"

    @staticmethod
    def get_builder_cls():
        return DeepseekV41MetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(32)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size):
        return CacheConfig.DEFAULT_DS41_BLOCK_SIZE

    @classmethod
    def get_supported_head_sizes(cls):
        return [68, 288, 512, 528, 1024]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (num_blocks, block_size, head_size)

    @classmethod
    def is_mla(cls):
        return True

    @classmethod
    def is_sparse(cls):
        return True

    @classmethod
    def supports_sink(cls):
        return True

    @classmethod
    def supports_sliding_window(cls):
        return True

    @classmethod
    def supports_compute_capability(cls, capability):
        return capability.major == 12
