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

# Diagnostic (experiments/2026-11-03-copy-drafter-shared-budget, never part of a
# patch): a device-side census of the attention metadata every build, no host
# synchronization. A row the metadata treats as invalid (request index or position
# below 0, which zeroes its lengths: window_row's ``valid``) among the step's live
# rows, or a query_start_loc end that differs from the step's token count, is a
# hit; the first hit's call number, counts and first bad row are kept until read.
_META_CALLS: torch.Tensor | None = None
_META_FIRST: torch.Tensor | None = None
_META_HOST_CALLS = 0
_META_FIELDS = ("hit", "call", "qsl_end", "num_tokens", "first_bad_row", "num_reqs",
                "builder", "bad_rows")


def _meta_census(builder, cm) -> None:
    global _META_CALLS, _META_FIRST, _META_HOST_CALLS
    device = builder.reqs.device
    if _META_CALLS is None:
        _META_CALLS = torch.zeros(1, dtype=torch.int64, device=device)
        _META_FIRST = torch.zeros(len(_META_FIELDS), dtype=torch.int64, device=device)
    _META_HOST_CALLS += 1
    _META_CALLS += 1
    nt, nr = int(cm.num_actual_tokens), int(cm.num_reqs)
    bad = (builder.reqs[:nt] < 0) | (builder.positions[:nt] < 0)
    any_bad = bad.any()
    qsl_end = cm.query_start_loc[nr].to(torch.int64)
    # A batch with every row invalid is a dummy or warm-up batch (startup, graph
    # capture); a stale-cap hit leaves a few rows invalid, not all of them.
    bad_count = bad.sum().to(torch.int64)
    hit = (any_bad & (bad_count < nt)) | (qsl_end != nt)
    const = lambda v: torch.full((), int(v), dtype=torch.int64, device=device)  # noqa: E731
    first_bad = torch.where(any_bad, bad.to(torch.int32).argmax().to(torch.int64), const(-1))
    record = torch.stack([const(1), _META_CALLS[0], qsl_end, const(nt), first_bad, const(nr),
                          const(getattr(builder, "_census_id", -1)), bad_count])
    unset = _META_FIRST[0] == 0
    _META_FIRST.copy_(torch.where(unset & hit, record, _META_FIRST))


def metadata_census() -> dict | None:
    """The first metadata hit since the last read, or None. Synchronizes the GPU."""
    if _META_FIRST is None:
        return None
    values = _META_FIRST.tolist()
    if not values[0]:
        return None
    _META_FIRST.zero_()
    return dict(zip(_META_FIELDS, values))


@dataclass
class DeepseekV41B12xMetadata:
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
    decoder: "DeepseekV41B12xMetadata | None" = None
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


class DeepseekV41B12xMetadataBuilder(AttentionMetadataBuilder):
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
        cls = type(self)
        cls._census_next = getattr(cls, "_census_next", 0) + 1
        self._census_id = cls._census_next
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
        _meta_census(self, cm)

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
        return DeepseekV41B12xMetadata(
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


class DeepseekV41B12xBackend(AttentionBackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "fp8", "fp8_ds_mla"]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # SWA, main KV, index keys, and partial states have different byte pages.
        return (KVCacheLayout.BLHNC,)

    @staticmethod
    def get_name():
        return "B12X"

    @staticmethod
    def get_builder_cls():
        return DeepseekV41B12xMetadataBuilder

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
