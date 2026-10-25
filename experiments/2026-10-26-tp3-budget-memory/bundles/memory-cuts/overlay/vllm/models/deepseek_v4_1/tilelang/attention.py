# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang kernels behind the DeepSeek-V4.1 attention layer.

``--attention-backend TILELANG`` selects this layer. It keeps the shared
layer's caches, metadata and Full/Reindex/Reuse ownership, and runs the
indexer (MXFP4 query quantization, index-key writes, scores, top-k and the
candidate list) and the sparse MLA with this package's kernels.

The kernels hold no buffers. Their scratch comes from vLLM's workspace,
reserved for the worst case before memory profiling, beside the attention
metadata arena exactly as B12X's scratch is.
"""

import functools
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xBackend
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    retain_cuda_graph_capture_resource,
)

from . import prepare
from .indexer import (
    CANDIDATE_BLOCK,
    CANDIDATE_BLOCKS,
    IndexerV41Plan,
    page_bytes,
    quantize_call,
    quantize_mxfp4,
    write_index_keys,
)
from .sparse_mla import SparseMlaV41Plan


class DeepseekV41TileLangBackend(DeepseekV41B12xBackend):
    """The V4.1 cache layout and metadata builder, under the TileLang name."""

    @staticmethod
    def get_name() -> str:
        return "TILELANG"


def _mla_options(heads, swa_width, idx_width, idx_page, swa_page, max_rows, device):
    """``SparseMlaV41Plan``'s arguments for one layer shape."""
    return (heads, swa_width, idx_width, idx_page), dict(
        swa_page_size=swa_page, max_rows=max_rows, device=device
    )


# Plans are kernels only, so layers of one shape share them.
@functools.cache
def _mla_plan(*shape):
    args, kwargs = _mla_options(*shape)
    return SparseMlaV41Plan(*args, **kwargs)


def _mla_calls(*shape) -> list:
    """The kernel calls ``_mla_plan(*shape)`` makes."""
    args, kwargs = _mla_options(*shape)
    return SparseMlaV41Plan.calls(*args, **kwargs)


@functools.cache
def _index_plan(width, page_size, candidates, candidate_blocks):
    return IndexerV41Plan(
        width, page_size, candidates=candidates, candidate_blocks=candidate_blocks
    )


def _index_calls(width, page_size, candidates, candidate_blocks) -> list:
    """The kernel calls ``_index_plan`` of these arguments makes."""
    return IndexerV41Plan.calls(
        width, page_size, candidates=candidates, candidate_blocks=candidate_blocks
    )


class DeepseekV41TileLangAttention(DeepseekV4Attention):
    backend_cls = DeepseekV41TileLangBackend
    # The TileLang indexer publishes candidates as block ids: 8 KiB a budget token
    # where 16,384 positions took 64.
    CANDIDATE_WIDTH = CANDIDATE_BLOCKS

    def _declare_mla(self, mode, capacity, metadata_specs, device):
        return None

    def get_b12x_preparation_units(self, layer, workload):
        # B12X's state unit prepares its own sparse MLA and indexer plans,
        # which this layer never launches.
        return ()

    def declare_kernels(self) -> None:
        """Also declare the sparse MLA, indexer and MXFP4 quantizer kernels; binding
        allocates the layer's staging, builds the plans and reserves their scratch,
        before memory profiling."""
        super().declare_kernels()
        calls = self._plan_calls()
        prepare.declare(lambda: calls, bind=self._bind_plans)

    def _plan_args(self) -> tuple[tuple, dict, dict]:
        """``_mla_plan``'s arguments, and per index mode ``_index_plan``'s
        arguments and the chunk rows its scratch covers."""
        self._attention_geometry()
        mla = (
            self.n_local_heads,
            self.swa_width,
            512 if self.compress_ratio else 0,
            self._main_page,
            self.swa_cache_layer.block_size,
            self.capacity,
            self.rotary_emb.cos_sin_cache.device,
        )
        index, chunks = {}, {}
        if self.indexer is not None:
            reindex = self.layer_id > self.candidate_source_layer
            width = (
                CANDIDATE_BLOCKS * CANDIDATE_BLOCK
                if reindex
                else self._index_width * self._index_page
            )
            full = (
                width,
                self._index_page,
                reindex,
                self.layer_id == self.candidate_source_layer,
            )
            index = {"decode": full, "prefill": full}
            chunks = {"decode": self.DECODE_CHUNK, "prefill": self.INDEX_CHUNK}
            if self._short_index_shape is not None:
                rows, pages = self._short_index_shape
                index["prefill_short"] = (
                    pages * self._index_page,
                    self._index_page,
                    False,
                    False,
                )
                chunks["prefill_short"] = rows
        return mla, index, chunks

    def _plan_calls(self) -> list:
        """The kernels this layer's plans make (``_build_tilelang_plans``) and its
        MXFP4 quantizers launch, as ``prepare.call``s."""
        mla, index, _ = self._plan_args()
        calls = _mla_calls(*mla)
        # Decode and prefill share one plan.
        for args in dict.fromkeys(index.values()):
            calls.extend(_index_calls(*args))
        if self.indexer is not None:
            calls.append(quantize_call())  # _quantize_index_query
        if self.compressor is not None:
            calls.append(quantize_call(self._index_page))  # _write_index_keys
        return calls

    def _bind_plans(self) -> None:
        # The staging is mutable serving state: ordinary tensors, not inference
        # tensors of the worker's profiling scope.
        with torch.inference_mode(False):
            self._prepare(self.rotary_emb.cos_sin_cache.device)
        self._tilelang_plans = self._build_tilelang_plans()

    def _tilelang(self) -> SimpleNamespace:
        """This layer's plans (built when its kernels are bound; the cache pages
        are checked on first use)."""
        plans = self.__dict__.get("_tilelang_plans")
        if plans is None:
            raise RuntimeError(
                "V4.1 TileLang attention plans are built when the layer's kernels "
                "are bound: declare_kernels() was not run for this layer"
            )
        if not self.__dict__.get("_cache_pages_checked"):
            self._check_cache_pages()
            self._cache_pages_checked = True
        return plans

    def _check_cache_pages(self) -> None:
        """Each cache page must be exactly the records the kernels address in it.

        Rows may be strided (BLHNC interleaves layers); a width that differs
        from the configured page means the page holds a different number of
        tokens than the plans assume.
        """
        caches = [
            (
                "SWA",
                self.swa_cache_layer.kv_cache,
                self.swa_cache_layer.block_size * 528,
            )
        ]
        if self.compress_ratio:
            caches.append(("indexed", self._owner().kv_cache, self._main_page * 288))
        if self.indexer is not None:
            caches.append(
                ("index K", self.indexer.k_cache.kv_cache, page_bytes(self._index_page))
            )
        for name, cache, page in caches:
            if cache.dim() != 2 or cache.dtype != torch.uint8 or cache.shape[1] != page:
                raise ValueError(
                    f"{self.prefix}: {name} cache pages {tuple(cache.shape)} "
                    f"{cache.dtype} are not {page}-byte pages"
                )

    def _build_tilelang_plans(self) -> SimpleNamespace:
        mla_args, index_args, chunks = self._plan_args()
        mla = _mla_plan(*mla_args)
        index = {mode: _index_plan(*args) for mode, args in index_args.items()}
        # The largest scratch each phase borrows: indexing, then the attention
        # metadata arena with the decode route's segment workspace. They are
        # reserved in the workspace lane this layer runs in, before profiling
        # (``reserve_profile_scratch``), so the drafter's lane does not hold the
        # target's indexing scratch.
        split_rows = min(mla.decode_max_rows, self.capacity)
        self._scratch_reservations = (
            *(plan.scratch_specs(chunks[mode]) for mode, plan in index.items()),
            *(
                (*metadata_specs, *mla.scratch_specs(split_rows))
                for metadata_specs in self._attention_workspace_specs.values()
            ),
        )
        return SimpleNamespace(mla=mla, index=index)

    def reserve_profile_scratch(self) -> None:
        """Reserve this layer's largest borrows (``_build_tilelang_plans``) in the
        current workspace lane: the model runner calls this in the lane of each
        model that holds the layer, before memory profiling, and then evens every
        lane's microbatch slots out (``reserve_by_lane``)."""
        workspace = current_workspace_manager()
        for specs in self.__dict__.get("_scratch_reservations", ()):
            workspace.get_simultaneous(*specs)

    def _check_attention_rows(self, mode: str, rows: int) -> None:
        capacity = self._tilelang().mla.max_rows
        if rows > capacity:
            raise ValueError(f"V4.1 {mode} rows {rows} exceed capacity {capacity}")

    def _attention_scratch_specs(self, mode: str, rows: int):
        return self._tilelang().mla.scratch_specs(rows)

    def _quantize_index_query(self, mode: str, iq, iq_data, iq_scale) -> None:
        quantize_mxfp4(iq, iq_data, iq_scale)

    def _write_index_keys(self, key, slot_mapping) -> None:
        write_index_keys(
            key, self.indexer.k_cache.kv_cache, slot_mapping, self._index_page
        )

    def _index_chunk(
        self,
        mode: str,
        rows: int,
        *,
        output_indices,
        q_mxfp4,
        q_scales,
        query_weights,
        page_table,
        cache_lengths,
        score_width=None,
        candidate_indices=None,
        candidate_lengths=None,
        candidate_output=None,
        candidate_output_lengths=None,
    ):
        # The kernels bound their work by each row's live length on device,
        # so the host-side ``score_width`` hint has nothing left to trim.
        del score_width
        plan = self._tilelang().index[mode]
        scratch = current_workspace_manager().get_simultaneous(
            *plan.scratch_specs(rows)
        )
        retain_cuda_graph_capture_resource(scratch)
        plan.run(
            q_mxfp4,
            q_scales,
            query_weights,
            self.indexer.k_cache.kv_cache,
            page_table,
            cache_lengths,
            self._active,
            candidate_indices,
            candidate_lengths,
            out=output_indices,
            candidate_out=candidate_output,
            candidate_out_lengths=candidate_output_lengths,
            scratch=scratch,
        )

    def _attend(self, mode: str, q, output, *, scratch, positions, **indices):
        indexed = {}
        if "indexed_indices" in indices:
            indexed = dict(
                idx_cache=self._owner().kv_cache,
                idx_indices=indices["indexed_indices"],
                idx_lengths=indices["indexed_lengths"],
                idx_page_table=indices["indexed_page_table"],
            )
        self._tilelang().mla.run(
            q,
            self.swa_cache_layer.kv_cache,
            indices["swa_indices"],
            indices["swa_lengths"],
            self.attn_sink,
            positions,
            self.rotary_emb.cos_sin_cache,
            output,
            scratch=scratch or None,
            **indexed,
        )


__all__ = ["DeepseekV41TileLangAttention", "DeepseekV41TileLangBackend"]
