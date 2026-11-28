# SPDX-License-Identifier: Apache-2.0
"""Spark GB10 GPUModelRunner mixin: step-kernel declarations, engine_grid,
sharded/greedy TileLang verification, and sink decoding helpers."""

from __future__ import annotations

import gc
import inspect
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from spark_ds41.runner.step_kernels import (
    dead_rows_enabled,
    declare_block_table_kernels,
    declare_input_batch_kernels,
    declare_staged_write_kernels,
    declare_zero_kv_blocks_kernel,
    iter_layer_specs,
    register_drafting_warmups,
    register_input_batch_warmups,
    select_kernel_block_sizes,
)
from spark_ds41.sampling import (
    greedy_verification,
    tilelang_inputs,
    tilelang_structured,
    tilelang_token_counts,
)
from spark_ds41.sampling.gpu_sink import SinkAlternates, SinkSampler
from spark_ds41.sampling.greedy_verification import (
    greedy_rejection_sample,
    register_greedy_verification_warmups,
    register_sampled_verification_warmups,
    sample_tokens_from_shards,
    sampled_rejection_sample_from_shards,
    vocab_parallel_pairs,
    vocab_parallel_pairs_from_head,
)
from spark_ds41.sampling.sink import SinkConfig
from spark_ds41.sampling.tilelang_inputs import logits_rows
from spark_ds41.tilelang import prepare as tilelang_prepare
from vllm import envs
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheGroup,
    KVCacheSpec,
    get_block_table_width,
)
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.worker.gpu.cudagraph_utils import (
    _init_minimal_kv_cache_for_profiling,
    _teardown_profiling_state,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.attn_utils import get_shared_kv_cache_layers
from vllm.v1.worker.utils import add_kv_sharing_layers_to_kv_cache_groups
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from spark_ds41.spec_decode.dflash import DFlashSpeculator, DraftBlockTableLayout
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace
from vllm.v1.worker.workspace import current_workspace_manager, use_workspace_lane

logger = init_logger(__name__)


class SparkModelRunnerMixin:
    _REAL_CONTEXT_ALIGN = 256
    _REAL_CONTEXT_CHUNK = 4096
    _b12x_a4_prefill_enabled: bool = False
    _filling_real_context: bool = False
    _greedy_verification: bool | None = None
    _sampled_verification: bool | None = None
    _real_context_depths: list[int] | None = None
    _real_context_max_len: int = 0
    _real_context_ring_offsets: dict[int, int] | None = None
    _real_context_ring_snapshots: dict[int, Any] | None = None
    _real_dummy_block_counts: Any = None
    _real_dummy_token_ids: Any = None
    _real_ring_prepared: bool = False
    _sink_decoding: Any = None
    copy_drafter: Any = None
    routed_experts_capturer: Any = None
    sink: Any = None

    def init_routed_experts_capturer(self) -> None:
        """Initialize target-model capture on every participating worker."""
        self.routed_experts_capturer = RoutedExpertsCapturer(
            max_num_batched_tokens=self.max_num_tokens,
            vllm_config=self.vllm_config,
            kv_cache_config=self.kv_cache_config,
        )
        bind_routed_experts_capturer(self.model, self.routed_experts_capturer)


    # get_kv_cache_spec is provided by canonical GPUModelRunner

    def _block_table_max_model_len(self) -> int:
        """The longest sequence a block-table row must index."""
        if not self.is_encoder_decoder:
            return self.max_model_len
        # Cross-attention block tables need to index encoder tokens, which
        # can exceed the decoder's max_model_len.
        return max(
            self.max_model_len,
            self.scheduler_config.max_num_encoder_input_tokens,
            getattr(self.model_config.hf_config, "max_source_positions", 0),
        )

    def _block_table_width(self, group_or_spec, max_len: int) -> int:
        if hasattr(group_or_spec, "block_table_width"):
            return group_or_spec.block_table_width(self.vllm_config, max_len)
        return get_block_table_width(
            cdiv(max_len, group_or_spec.block_size),
            group_or_spec.block_size,
        )

    def _group_cp_size(self, layer_spec: KVCacheSpec) -> int:
        return 1 if getattr(layer_spec, "dcp_replicated", False) else self.dcp_size

    @torch.inference_mode()
    def engine_grid(self, spec: dict) -> dict:
        """Time dummy steps at fixed shapes: the engine cost grid servebench reads.

        ``spec`` holds ``shapes`` (each ``num_tokens``, optional ``num_reqs``,
        ``context_len`` and ``prefill`` for a single request's chunk), ``replays``,
        ``warmup`` and ``settle``. Every shape runs ``warmup`` times untimed, then
        ``replays`` timed rounds alternate the shape order, as the cost profile does,
        so drift spreads over every shape. Each timed run follows ``settle`` untimed
        runs of the same shape (default 1): switching shapes rebuilds the dummy
        metadata (an 11 ms step after a jump between 1M and no context), which is the
        harness's cost, not the engine's. Returns each shape's whole-step
        milliseconds (events around the dummy run) and, when every run reached the
        drafter, the target forward and drafter split, step by step.

        The steps are the cost profile's dummy runs: identical work on every build,
        with no text, acceptance or scheduling in them. Run only on an idle server,
        and reset the prefix cache afterwards, since dummy steps write KV slots.

        A dummy row is a padding row with no cache slot: the MoE routers skip it
        and it sees no compressed context, whatever ``context_len`` is. A shape
        with ``real_rows`` runs its rows as real ones instead (``_real_dummy_rows``):
        they route, and they read ``context_len`` of context that real-row prefill
        chunks wrote before the first timed step (``_fill_real_context``, about
        the time of a real prefill of the longest such context; a spec with
        ``fill_real_context`` false skips it, for a call that follows one which
        filled the same context on a server idle since). The answer repeats
        ``real_rows`` for each shape that ran so.
        """
        shapes = spec["shapes"]
        replays = int(spec.get("replays", 30))
        warmup = int(spec.get("warmup", 2))
        settle = int(spec.get("settle", 1))
        fill_real_context = bool(spec.get("fill_real_context", True))
        real_depths = sorted(
            {
                int(shape.get("context_len", 0))
                for shape in shapes
                if shape.get("real_rows")
            }
        )
        if not fill_real_context and real_depths:
            saved_depths = getattr(self, "_real_context_depths", set())
            missing = [d for d in real_depths if d not in saved_depths]
            if missing:
                raise ValueError(
                    "fill_real_context=False requires prior real-context "
                    f"initialization for depths {missing}"
                )
        if (
            getattr(self, "kv_block_zeroer", None) is not None
            and getattr(self, "kv_cache_config", None) is not None
            and self.kv_cache_config.num_blocks > 0
        ):
            # Zero null block and circular SWA dummy-block slices once so
            # prior target traffic cannot leave NaN in dummy drafter blocks.
            # Preserve saved per-depth ring snapshots when fill_real_context=False.
            bt = self.block_tables
            num_groups = len(bt.input_block_tables)
            usable = self.kv_cache_config.num_blocks - 1
            preserved: set[int] = set()
            if not fill_real_context:
                for group_snaps in getattr(
                    self, "_real_context_ring_snapshots", {}
                ).values():
                    preserved.update(group_snaps.values())
            ring_blocks: set[int] = {0}
            if num_groups > 0 and usable >= 1:
                share = max(usable // num_groups, 1)
                for group, is_ring in enumerate(bt._slot_mapping_circular):
                    if is_ring:
                        for off in range(share):
                            bid = (group * share + off) % usable + 1
                            if bid not in preserved:
                                ring_blocks.add(bid)
            self.kv_block_zeroer._fill_block_ids(sorted(ring_blocks), 0)
        if fill_real_context:
            if real_depths:
                self._fill_real_context(max(real_depths), depths=real_depths)
            else:
                self._fill_real_context(0)

        def prepare_ring(shape: dict) -> bool:
            if not shape.get("real_rows"):
                return False
            prep = getattr(self, "_prepare_real_row_ring_blocks", None)
            if prep is None:
                return False
            n_reqs = (
                1
                if shape.get("prefill")
                else int(shape.get("num_reqs") or 1)
            )
            ctx = int(shape.get("context_len", 0))
            prep([ctx] * n_reqs)
            return True

        def run(shape: dict, *, ring_prepared: bool = False) -> None:
            kwargs: dict[str, Any] = {
                "num_tokens": int(shape["num_tokens"]),
                "context_len": int(shape.get("context_len", 0)),
            }
            if shape.get("prefill"):
                kwargs["single_request_prefill"] = True
            elif shape.get("num_reqs") is not None:
                kwargs["profile_num_reqs"] = int(shape["num_reqs"])
            if shape.get("real_rows"):
                kwargs["real_rows"] = True
            if ring_prepared:
                self._real_ring_prepared = True
            try:
                self._dummy_run(**kwargs)
            finally:
                if ring_prepared:
                    self._real_ring_prepared = False

        for shape in shapes:
            for _ in range(warmup):
                run(shape)
        torch.cuda.synchronize()
        gc.collect()
        if (
            get_tensor_model_parallel_world_size() > 1
            and torch.distributed.is_initialized()
        ):
            tp_group = get_tp_group()
            cpu_group = getattr(tp_group, "cpu_group", None)
            if cpu_group is not None:
                torch.distributed.barrier(group=cpu_group)
        order: list[int] = []
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with self.step_timing.collect() as timings:
                for replay in range(replays):
                    indices = range(len(shapes))
                    for index in indices if replay % 2 == 0 else reversed(indices):
                        for _ in range(settle):
                            run(shapes[index])
                        ring_ready = prepare_ring(shapes[index])
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        run(shapes[index], ring_prepared=ring_ready)
                        end.record()
                        events.append((start, end))
                        order.append(index)
            torch.cuda.synchronize()
        finally:
            if gc_was_enabled:
                gc.enable()
        results = [
            {
                "shape": shape,
                "real_rows": bool(shape.get("real_rows")),
                "step_ms": [],
                "forward_ms": [],
                "drafter_ms": [],
                "modes": [],
            }
            for shape in shapes
        ]
        for index, (start, end) in zip(order, events):
            # The whole dummy run on the stream: target, drafter and every gap
            # between their kernels.
            results[index]["step_ms"].append(start.elapsed_time(end))
        # The target/drafter split exists only when every run reached the drafter;
        # each timed run's sample follows its settle runs' samples.
        if len(timings) == len(order) * (settle + 1):
            for index, sample in zip(order, timings[settle :: settle + 1]):
                results[index]["forward_ms"].append(sample.forward_ms)
                results[index]["drafter_ms"].append(sample.drafter_ms)
                results[index]["modes"].append(
                    [sample.full_cudagraph, sample.num_target_tokens, sample.num_reqs]
                )
        return {
            "replays": replays,
            "warmup": warmup,
            "settle": settle,
            "shapes": results,
        }

    def _real_context_pool_layout(self) -> tuple[int, int]:
        bt = self.block_tables
        num_groups = len(bt.input_block_tables)
        usable = self.kv_cache_config.num_blocks - 1
        if usable < 1 or num_groups < 1:
            raise ValueError(
                "real_rows requires at least one KV cache group and non-null block"
            )
        share = max(usable // num_groups, 1)
        return usable, share

    @staticmethod
    def _real_context_pool_block(
        group: int, off: int, usable: int, share: int
    ) -> int:
        if off < 0 or off >= share:
            raise ValueError(
                f"real_rows KV block offset {off} exceeds group {group} share {share}"
            )
        return (group * share + off) % usable + 1

    def _snapshot_real_context_ring(self, depth: int) -> None:
        bt = getattr(self, "block_tables", None)
        if bt is None or getattr(self, "kv_cache_config", None) is None:
            return
        usable, share = self._real_context_pool_layout()
        snapshots: dict[int, dict[int, int]] = getattr(
            self, "_real_context_ring_snapshots", {}
        )
        offsets: dict[int, int] = getattr(self, "_real_context_ring_offsets", {})
        if depth in offsets:
            snap_off = offsets[depth]
        else:
            snap_off = 1 + len(offsets)
            offsets[depth] = snap_off
        group_map: dict[int, int] = {}
        copies: list[tuple[int, int]] = []
        for group, is_ring in enumerate(bt._slot_mapping_circular):
            if not is_ring:
                continue
            src = self._real_context_pool_block(group, 0, usable, share)
            dst = self._real_context_pool_block(group, snap_off, usable, share)
            group_map[group] = dst
            copies.append((src, dst))
        if copies and getattr(self, "kv_caches", None):
            copy_kv_cache_blocks_inplace(
                self.kv_caches, self.kv_cache_config.num_blocks, copies
            )
        snapshots[depth] = group_map
        self._real_context_ring_snapshots = snapshots
        self._real_context_ring_offsets = offsets

    def _prepare_real_row_ring_blocks(
        self, depths: list[int] | tuple[int, ...]
    ) -> None:
        bt = getattr(self, "block_tables", None)
        if bt is None or getattr(self, "kv_cache_config", None) is None:
            return
        snapshots: dict[int, dict[int, int]] = getattr(
            self, "_real_context_ring_snapshots", {}
        )
        if not snapshots:
            return
        usable, share = self._real_context_pool_layout()
        num_snapshots = len(snapshots)
        copies: list[tuple[int, int]] = []
        for group, is_ring in enumerate(bt._slot_mapping_circular):
            if not is_ring:
                continue
            for req, depth in enumerate(depths):
                snap = snapshots.get(int(depth), {}).get(group)
                if snap is None:
                    continue
                dst = self._real_context_pool_block(
                    group, 1 + num_snapshots + req, usable, share
                )
                copies.append((snap, dst))
        if copies and getattr(self, "kv_caches", None):
            copy_kv_cache_blocks_inplace(
                self.kv_caches, self.kv_cache_config.num_blocks, copies
            )

    def _fill_real_context(
        self,
        context_len: int,
        *,
        depths: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """Write real-row prefix context up to ``context_len`` and snapshot ring
        KV state at every requested depth in ``depths``.

        Non-ring (paged) groups store immutable history in group offsets
        ``0 .. history_kv_blocks - 1``. Circular ring groups fill into offset
        ``0`` and snapshot each requested depth into its own read-only block
        (offsets ``1 .. len(depths)``) so every subsequent decode/grid step
        copies from the depth's immutable snapshot into per-request writable
        ring blocks (offsets ``1 + len(depths) + req``)."""
        target_depths = sorted(
            {
                int(d)
                for d in (
                    depths
                    if depths is not None
                    else ([context_len] if context_len > 0 else [])
                )
            }
        )
        if context_len < 0 or context_len % self._REAL_CONTEXT_ALIGN:
            raise ValueError(
                f"real_rows needs a context of whole {self._REAL_CONTEXT_ALIGN}-token "
                f"blocks, not {context_len}"
            )
        for d in target_depths:
            if d < 0 or d % self._REAL_CONTEXT_ALIGN:
                raise ValueError(
                    f"real_rows needs a context of whole {self._REAL_CONTEXT_ALIGN}-token "
                    f"blocks, not {d}"
                )
        max_len = max([context_len, *target_depths], default=0)
        self._real_context_max_len = max_len
        self._real_context_depths = set(target_depths)
        self._real_context_ring_snapshots = {}
        self._real_context_ring_offsets = {}

        if 0 in target_depths:
            self._snapshot_real_context_ring(0)

        offset = 0
        self._filling_real_context = True
        try:
            while offset < max_len:
                next_targets = [d for d in target_depths if d > offset]
                stop = next_targets[0] if next_targets else max_len
                step = min(self._REAL_CONTEXT_CHUNK, stop - offset)
                self._dummy_run(
                    num_tokens=step,
                    context_len=offset,
                    single_request_prefill=True,
                    real_rows=True,
                )
                offset += step
                if offset in target_depths:
                    self._snapshot_real_context_ring(offset)
        finally:
            self._filling_real_context = False

    def _real_dummy_rows(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        context_len: int | list[int] | tuple[int, ...],
    ) -> torch.Tensor:
        """Make a dummy batch's rows real, for the engine grid's ``real_rows``
        shapes, and return their slot mappings.

        Every request reads its depth's immutable paged history (offsets
        ``0 .. history_kv_blocks - 1``) and writes new tokens into strictly
        disjoint per-request tail blocks (offsets ``>= history_kv_blocks``) and
        its own isolated writable ring block (copied from the depth's snapshot),
        so concurrent requests and mixed-depth steps never clobber each other or
        corrupt saved history across ``fill_real_context=False`` repeats."""
        if self.pcp_manager is not None:
            raise ValueError("real_rows is not supported with context parallelism")
        bt = self.block_tables
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        query_lens = [int(x) for x in input_batch.num_scheduled_tokens[:num_reqs]]
        query_len = input_batch.max_query_len or max(query_lens)
        if isinstance(context_len, (list, tuple)):
            if len(context_len) != num_reqs:
                raise ValueError(
                    f"context_len list length {len(context_len)} != num_reqs {num_reqs}"
                )
            req_depths = [
                max(min(int(d), self.max_model_len - q), 0)
                for d, q in zip(context_len, query_lens)
            ]
        else:
            clamped = max(min(int(context_len), self.max_model_len - query_len), 0)
            req_depths = [clamped] * num_reqs

        for d in req_depths:
            if d % self._REAL_CONTEXT_ALIGN:
                raise ValueError(
                    f"real_rows needs a context of whole {self._REAL_CONTEXT_ALIGN}-token "
                    f"blocks, not {d}"
                )

        filling = bool(getattr(self, "_filling_real_context", False))
        if not any(req_depths) or len(set(req_depths)) > 1:
            local_pos = np.arange(num_tokens, dtype=np.int64) - np.repeat(
                input_batch.query_start_loc_np[:num_reqs],
                input_batch.num_scheduled_tokens[:num_reqs],
            )
            depth_per_tok = np.repeat(
                np.asarray(req_depths, dtype=np.int64),
                input_batch.num_scheduled_tokens[:num_reqs],
            )
            input_batch.positions[:num_tokens].copy_(
                torch.from_numpy(depth_per_tok + local_pos)
            )

        usable, share = self._real_context_pool_layout()
        if filling:
            assert num_reqs == 1
            d0 = req_depths[0]
            q0 = query_lens[0]
            for group, (table, block_size, bpk, ring) in enumerate(
                zip(
                    block_tables,
                    bt.kernel_block_sizes,
                    bt.blocks_per_kv_block,
                    bt._slot_mapping_circular,
                )
            ):
                device = table.device
                if ring:
                    pool_id = self._real_context_pool_block(group, 0, usable, share)
                    k_ids = torch.arange(bpk, dtype=table.dtype, device=device) + (
                        pool_id * bpk
                    )
                    cols = min(bpk, table.shape[1])
                    table[0, :cols] = k_ids[:cols]
                else:
                    total_blocks = min(cdiv(d0 + q0, block_size), table.shape[1])
                    kv_blocks = cdiv(total_blocks, bpk)
                    if kv_blocks > share:
                        raise ValueError(
                            f"real_rows fill needs {kv_blocks} blocks > group {group} share {share}"
                        )
                    kv_ids = torch.tensor(
                        [
                            self._real_context_pool_block(group, off, usable, share)
                            for off in range(kv_blocks)
                        ],
                        dtype=torch.int64,
                        device=device,
                    )
                    kernel_ids = (
                        kv_ids.unsqueeze(1) * bpk
                        + torch.arange(bpk, dtype=torch.int64, device=device)
                    ).reshape(-1)
                    table[0, :total_blocks] = kernel_ids[:total_blocks].to(table.dtype)
        else:
            if not getattr(self, "_real_ring_prepared", False):
                self._prepare_real_row_ring_blocks(req_depths)
            num_snapshots = len(getattr(self, "_real_context_ring_snapshots", {}))
            max_hist_len = max(
                getattr(self, "_real_context_max_len", 0),
                max(req_depths, default=0),
            )
            for group, (table, block_size, bpk, ring) in enumerate(
                zip(
                    block_tables,
                    bt.kernel_block_sizes,
                    bt.blocks_per_kv_block,
                    bt._slot_mapping_circular,
                )
            ):
                device = table.device
                if ring:
                    if 1 + num_snapshots + num_reqs > share:
                        raise ValueError(
                            f"real_rows needs {1 + num_snapshots + num_reqs} ring blocks "
                            f"in group {group}, only {share} available"
                        )
                    cols = min(bpk, table.shape[1])
                    for req in range(num_reqs):
                        pool_id = self._real_context_pool_block(
                            group, 1 + num_snapshots + req, usable, share
                        )
                        k_ids = torch.arange(bpk, dtype=table.dtype, device=device) + (
                            pool_id * bpk
                        )
                        table[req, :cols] = k_ids[:cols]
                else:
                    hist_blocks = min(max_hist_len // block_size, table.shape[1])
                    hist_kv_blocks = cdiv(hist_blocks, bpk)
                    next_off = hist_kv_blocks
                    for req, (d, q) in enumerate(zip(req_depths, query_lens)):
                        if d % block_size:
                            raise ValueError(
                                f"real_rows context {d} splits a {block_size}-token block"
                            )
                        shared = min(d // block_size, table.shape[1])
                        total_blocks = min(cdiv(d + q, block_size), table.shape[1])
                        tail_blocks = max(total_blocks - shared, 0)
                        if shared > 0:
                            shared_kv = cdiv(shared, bpk)
                            hist_ids = torch.tensor(
                                [
                                    self._real_context_pool_block(
                                        group, off, usable, share
                                    )
                                    for off in range(shared_kv)
                                ],
                                dtype=torch.int64,
                                device=device,
                            )
                            hist_k = (
                                hist_ids.unsqueeze(1) * bpk
                                + torch.arange(bpk, dtype=torch.int64, device=device)
                            ).reshape(-1)
                            table[req, :shared] = hist_k[:shared].to(table.dtype)
                        if tail_blocks > 0:
                            tail_kv = cdiv(tail_blocks, bpk)
                            if next_off + tail_kv > share:
                                raise ValueError(
                                    f"real_rows needs {next_off + tail_kv} paged blocks "
                                    f"in group {group}, only {share} available"
                                )
                            tail_ids = torch.tensor(
                                [
                                    self._real_context_pool_block(
                                        group, next_off + k, usable, share
                                    )
                                    for k in range(tail_kv)
                                ],
                                dtype=torch.int64,
                                device=device,
                            )
                            tail_k = (
                                tail_ids.unsqueeze(1) * bpk
                                + torch.arange(bpk, dtype=torch.int64, device=device)
                            ).reshape(-1)
                            table[req, shared : shared + tail_blocks] = tail_k[
                                :tail_blocks
                            ].to(table.dtype)
                            next_off += tail_kv

        ids = getattr(self, "_real_dummy_token_ids", None)
        if ids is None:
            span = 1 << 20
            generator = torch.Generator().manual_seed(0)
            ids = torch.randint(
                1000,
                min(self.vocab_size, 100000),
                (span + self.input_buffers.input_ids.shape[0],),
                generator=generator,
                dtype=torch.int32,
            ).to(self.device)
            self._real_dummy_token_ids = ids
            self._real_dummy_block_counts = torch.tensor(
                [[table.shape[1]] * bt.max_num_reqs for table in bt.input_block_tables],
                dtype=torch.int32,
                device=self.device,
            )
        max_span = ids.shape[0] - self.input_buffers.input_ids.shape[0]
        if len(set(req_depths)) == 1 and num_reqs == 1:
            start = req_depths[0] % max_span
            input_batch.input_ids[:num_tokens].copy_(ids[start : start + num_tokens])
        else:
            q_starts = input_batch.query_start_loc_np
            for req, (d, q) in enumerate(zip(req_depths, query_lens)):
                s_dst = int(q_starts[req])
                s_src = d % max_span
                input_batch.input_ids[s_dst : s_dst + q].copy_(
                    ids[s_src : s_src + q]
                )
        input_batch.is_padding.fill_(False)

        tilelang_inputs.compute_slot_mappings(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            bt._input_block_table_ptrs_i64,
            bt.block_table_strides,
            self._real_dummy_block_counts,
            bt.block_sizes_tensor,
            bt.kernel_block_sizes_tensor,
            bt.slot_mapping_enabled,
            bt.slot_mapping_circular,
            bt.group_cp_sizes,
            bt.slot_mappings,
            bt.cp_rank,
            bt.cp_size,
            bt.cp_interleave,
        )
        return bt.slot_mappings[:, : input_batch.num_tokens_after_padding]

    def _reserve_profile_scratch(self) -> None:
        model_lanes: dict[int, set[int]] = {}
        lane_modules: list[tuple[torch.nn.Module, int]] = []
        for model, lane in (
            (self.get_model(), 0),
            (self.get_draft_model(), self._draft_workspace_lane),
        ):
            if model is not None:
                for module in model.modules():
                    model_lanes.setdefault(id(module), set()).add(lane)
                    if hasattr(module, "reserve_profile_scratch"):
                        lane_modules.append((module, lane))
        seen: set[tuple[int, int]] = set()
        candidates: list[tuple[object, int]] = [
            (module, lane)
            for module in self.compilation_config.static_forward_context.values()
            for lane in sorted(model_lanes.get(id(module), {0}))
        ]
        candidates.extend(lane_modules)
        for module, lane in candidates:
            key = (id(module), lane)
            if key in seen:
                continue
            seen.add(key)
            reserve = getattr(module, "reserve_profile_scratch", None)
            if reserve is not None:
                with use_workspace_lane(lane):
                    if inspect.signature(reserve).parameters:
                        reserve(self.max_num_tokens)
                    else:
                        reserve()
        current_workspace_manager().reserve_by_lane()

    @torch.inference_mode()
    def _profile_deepseek_v4_attention(
        self, prepare_profile_state: Callable[[], None] | None = None
    ) -> None:
        """Include the maximum DeepSeek V4 prefill peak in KV admission.

        The generic profile omits attention and distributes its token budget
        across requests. DeepSeek V4 can execute the complete scheduler token
        budget as one prefill, where query projection and auxiliary-stream
        indexer work overlap. Run that reachable shape while multimodal encoder
        outputs from ``profile_run`` remain resident. A minimal temporary cache
        makes attention executable without reserving the production KV pool.
        """
        if self.model_config.architecture not in {
            "DeepseekV4ForCausalLM",
            "DeepseekV4ForConditionalGeneration",
            "DeepseekV41ForCausalLM",
        }:
            return

        from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

        # One block per KV cache group besides the null block: the groups
        # overlay the same memory, so a block shared by every group would let
        # one group read another's writes in its own format.
        groups = get_kv_cache_groups(self.vllm_config, self.get_kv_cache_spec())
        try:
            _init_minimal_kv_cache_for_profiling(self, num_blocks=1 + len(groups))
            if prepare_profile_state is not None:
                prepare_profile_state()
            self._dummy_run(
                self.max_num_tokens,
                skip_eplb=True,
                is_profile=True,
                single_request_prefill=True,
                profile_all_kv_cache_groups=True,
            )
            torch.accelerator.synchronize()
        finally:
            _teardown_profiling_state(self)

    @torch.inference_mode()
    def profile_glm_dcp_attention(
        self, prepare_profile_state: Callable[[], None] | None = None
    ) -> None:
        """Measure the GLM DCP query-gather peak before KV cache sizing.

        The general activation profile deliberately skips attention and splits
        its token budget across many requests. A GLM sparse-MLA prefill can put
        the complete scheduler budget in one request, causing the DCP query
        all-gather to require substantially more temporary memory. Bind a
        one-block split cache for that single request, prepare its bound kernel
        plans, then release the temporary cache before production allocation.
        """
        if (
            self.model_config.architecture != "Glm5NextForConditionalGeneration"
            or self.dcp_size <= 1
        ):
            return

        try:
            _init_minimal_kv_cache_for_profiling(self, num_blocks=1)
            if prepare_profile_state is not None:
                prepare_profile_state()
            self._dummy_run(
                self.max_num_tokens,
                context_len=self.dcp_size * self.cp_interleave,
                skip_eplb=True,
                is_profile=True,
                single_request_prefill=True,
                profile_all_kv_cache_groups=True,
            )
            torch.accelerator.synchronize()
        finally:
            _teardown_profiling_state(self)

    def post_kv_cache_wake_up(self) -> None:
        self.block_tables.init_block_table_layout_tensors()

    def _build_moe_prefill_metadata(
        self, input_batch: InputBatch
    ) -> MoEPrefillMetadata | None:
        if not self._b12x_a4_prefill_enabled:
            return None
        from vllm.utils.b12x import build_moe_prefill_metadata

        return build_moe_prefill_metadata(
            input_batch.num_tokens_after_padding,
            input_batch.query_start_loc_np[: input_batch.num_reqs + 1],
            input_batch.num_computed_tokens_np,
            self.req_states.prompt_len.np[input_batch.idx_mapping_np],
        )

    def _language_model(self) -> nn.Module:
        """The model that owns the LM head (a multimodal wrapper delegates to it)."""
        model = self.model
        while getattr(model, "lm_head", None) is None and hasattr(
            model, "language_model"
        ):
            model = model.language_model
        return model

    def _greedy_verification_available(self) -> bool:
        if self._greedy_verification is None:
            language_model = self._language_model()
            processor = getattr(language_model, "logits_processor", None)
            head = getattr(language_model, "lm_head", None)
            self._greedy_verification = bool(
                os.environ.get("VLLM_GREEDY_VERIFICATION", "1") == "1"
                and type(self.rejection_sampler) is RejectionSampler
                and self.rejection_sampler.watermark_key is None
                and self.rejection_sampler.synthetic_conditional_rates is None
                and self.sampler is not None
                and not isinstance(self.sampler, GPUWatermarkSampler)
                and not self.sampler.compute_nans
                and not self.sampler.return_sampling_mask
                and self.sampler.trace_replay_state is None
                and self.batch_sharder is None
                and hasattr(self.model, "compute_logits_local")
                and hasattr(getattr(head, "shard_indices", None), "num_org_elements")
                and processor is not None
                and processor.scale == 1.0
                and processor.soft_cap is None
            )
            logger.info(
                "Greedy verification from vocabulary shards: %s",
                "on" if self._greedy_verification else "off",
            )
        return self._greedy_verification

    def _sampled_verification_available(self) -> bool:
        if self._sampled_verification is None:
            self._sampled_verification = bool(
                self._greedy_verification_available()
                and os.environ.get("VLLM_SAMPLED_VERIFICATION", "1") == "1"
                and self.rejection_sampler is not None
                and self.rejection_sampler.use_block_verification
                and getattr(self.speculator, "draft_logits", None) is None
                and self.sampler is not None
                and not getattr(self.sampler, "use_fp64_gumbel", False)
            )
            logger.info(
                "Sampled verification from vocabulary shards: %s",
                "on" if self._sampled_verification else "off",
            )
        return self._sampled_verification

    def declare_step_kernels(self) -> None:
        """Declare the kernels that input preparation, sampling, greedy
        verification and the drafter launch: for the runner's buffers, the loaded
        LM head's dtype and vocabulary shards, and the block tables of the KV cache
        groups its spec gives (the KV cache does not exist yet). Input preparation
        and the commit of sampled tokens (TileLang) run on every rank; a stage
        before the last commits the last stage's tokens without query starts or
        output-token counts. Sampling and verification run on the last.

        The worker calls this while preparing kernels, after the engine core set
        the KV cache layout the groups depend on.
        """
        from vllm.v1.core.kv_cache_planning import get_kv_cache_groups

        kv_cache_groups = get_kv_cache_groups(
            self.vllm_config, self.get_kv_cache_layer_configs()
        )
        add_kv_sharing_layers_to_kv_cache_groups(
            get_shared_kv_cache_layers(self.vllm_config), kv_cache_groups
        )
        kernel_block_sizes = select_kernel_block_sizes(
            kv_cache_groups, self.vllm_config
        )
        self._register_kv_cache_warmups(kv_cache_groups, kernel_block_sizes)
        # The manager is built with the KV cache, after preparation; the
        # speculator's flag is what decides it exists.
        spec_cfg = getattr(self, "speculative_config", None)
        adaptive = bool(
            getattr(self.speculator, "enable_adaptive_verification", False)
            if getattr(self, "is_last_pp_rank", True)
            else (
                spec_cfg is not None
                and spec_cfg.enable_adaptive_verification
            )
        )
        dead_rows = adaptive and dead_rows_enabled() and self.batch_sharder is None
        samples = self.is_last_pp_rank and not self.is_pooling_model
        if not samples:
            # A stage before the last commits what the last stage sampled.
            commits = () if self.is_pooling_model else ((False, False, False),)
        elif getattr(self.vllm_config, "use_request_boundary_checkpoints", False):
            # With boundary captures, and logits-only steps without query starts.
            commits = ((True, True, True), (True, False, False), (True, True, False))
        else:
            commits = ((True, True, False),)
        num_draft_steps = getattr(self, "num_draft_steps", self.num_speculative_steps)
        declare_input_batch_kernels(
            num_lookahead=self.req_states.next_prefill_tokens.shape[0],
            num_speculative_steps=self.req_states.draft_tokens.shape[-1],
            num_new_sampled_tokens=self.model_state.num_new_sampled_tokens_per_step,
            dead_rows=dead_rows,
            draft_widths=(
                (num_draft_steps, self.req_states.draft_tokens.shape[-1])
                if 0 < num_draft_steps < self.req_states.draft_tokens.shape[-1]
                else ()
            ),
            # The sampler's one column, and verification's num_speculative_steps + 1.
            sampled_widths=tuple(
                dict.fromkeys(
                    (1,)
                    + (
                        (num_draft_steps + 1,)
                        if num_draft_steps
                        and num_draft_steps != self.num_speculative_steps
                        else ()
                    )
                    + (
                        (self.num_speculative_steps + 1,)
                        if self.num_speculative_steps
                        else ()
                    )
                )
            ),
            commits=commits,
        )
        if self.speculator is not None and self.num_speculative_steps:
            # The commit of a step's drafts (sample_tokens), and of a boundary
            # checkpoint's replayed drafts.
            draft_dtype = self.req_states.draft_tokens.dtype
            tilelang_inputs.declare(
                lambda: tilelang_inputs.scatter_rows_calls(
                    torch.int64,
                    draft_dtype,
                    gather=not getattr(getattr(self, "draft_tokens_handler", None), "track_consumed_drafts", False),
                )
                + (
                    tilelang_inputs.scatter_rows_calls(
                        torch.int64, draft_dtype, gather=False
                    )
                    if getattr(self.vllm_config, "use_request_boundary_checkpoints", False)
                    and not getattr(getattr(self, "draft_tokens_handler", None), "track_consumed_drafts", False)
                    else []
                )
            )
            if getattr(getattr(self, "draft_tokens_handler", None), "track_consumed_drafts", False):
                # The drafts a step's inputs consumed, handed back to the
                # scheduler for grammar validation (set_consumed_draft_tokens).
                tilelang_inputs.declare(
                    lambda: tilelang_inputs.gather_rows_calls(torch.int64, draft_dtype)
                )
        if envs.VLLM_MOE_SKIP_PADDING:
            # The target's padding rows (prepare_inputs).
            tilelang_inputs.declare(tilelang_inputs.padding_rows_calls)
        if adaptive:
            step_widths = tuple(
                dict.fromkeys(
                    (num_draft_steps, self.num_speculative_steps)
                    if num_draft_steps > 0
                    else (self.num_speculative_steps,)
                )
            )
            tilelang_inputs.declare(
                lambda: [
                    call
                    for steps in step_widths
                    for call in (
                        tilelang_inputs.adaptive_offsets_calls(
                            steps + self.model_state.num_new_sampled_tokens_per_step
                        )
                        + tilelang_inputs.assign_draft_budget_calls(steps)
                    )
                ]
                + tilelang_inputs.scatter_rows_calls(
                    torch.int64, torch.float32, gather=False
                )
                # The new requests' confidences (apply_staged_writes).
                + tilelang_inputs.fill_rows_calls(torch.int64, torch.float32)
            )
        self.req_states.register_warmups()
        if getattr(self, "copy_drafter", None) is not None:
            self.copy_drafter.register_warmups()
        if self.is_pooling_model or not self.is_last_pp_rank:
            tilelang_inputs.declare(
                tilelang_inputs.post_update_num_computed_tokens_calls
            )
        if not samples:
            return
        assert self.sampler is not None
        register_input_batch_warmups(dead_rows=dead_rows)
        # The hidden rows the LM head projects (gather_hidden_rows).
        hidden_size = self.model_config.get_hidden_size()
        tilelang_inputs.declare(
            lambda: tilelang_inputs.gather_hidden_rows_calls(self.dtype, hidden_size)
        )
        # The logits rows' inputs (InputBatch.logits_rows): every batch's logits
        # indices are int64, a dummy's too.
        tilelang_inputs.declare(
            lambda: tilelang_inputs.logits_rows_calls(torch.int32)
            + tilelang_inputs.logits_rows_calls(torch.int64)
        )

        language_model = self._language_model()
        head = getattr(language_model, "lm_head", None)
        processor = getattr(language_model, "logits_processor", None)
        if isinstance(head, VocabParallelEmbedding) and isinstance(
            processor, LogitsProcessor
        ):
            if hasattr(processor, "declare_kernels"):
                processor.declare_kernels()
            vocab_size = processor.org_vocab_size
            # compute_logits gathers the padded vocabulary shards and drops the
            # padding columns. Logits processing copies them to FP32 densely: rows
            # vocab_size apart, or as gathered when there is a single row.
            head_dtype = processor.head_dtype or head.weight.dtype
            logits = ((head_dtype, head.num_embeddings_padded),)
            processed_strides = tuple(
                dict.fromkeys((vocab_size, head.num_embeddings_padded))
            )
            if hasattr(self.sampler, "register_warmups"):
                self.sampler.register_warmups(
                    logits=logits,
                    processed_logits_strides=processed_strides,
                    vocab_size=vocab_size,
                )
            if hasattr(self.rejection_sampler, "register_warmups"):
                for _rej_steps in dict.fromkeys(
                    (num_draft_steps, self.num_speculative_steps)
                    if num_draft_steps > 0
                    else (self.num_speculative_steps,)
                ):
                    self.rejection_sampler.register_warmups(
                        logits=logits,
                        vocab_size=vocab_size,
                        num_speculative_steps=_rej_steps,
                        dead_rows=dead_rows,
                    )
            if self.prompt_logprobs_worker is not None:
                self.prompt_logprobs_worker.register_warmups(
                    logits=logits,
                    vocab_size=vocab_size,
                    all_token_ids_stride=self.req_states.all_token_ids.gpu.stride(0),
                )
            if self.structured_outputs_worker is not None:
                self.structured_outputs_worker.register_warmups(
                    logits=logits,
                    vocab_size=vocab_size,
                    invalid_spec_tokens=self.speculative_config is not None,
                )
            if getattr(self, "sink", None) is not None:
                self.sink.register_warmups(
                    logits=logits,
                    processed_logits_strides=processed_strides,
                    sampled_widths=tuple(
                        dict.fromkeys(
                            (1,)
                            + (
                                (num_draft_steps + 1,)
                                if num_draft_steps > 0
                                and num_draft_steps != self.num_speculative_steps
                                else ()
                            )
                            + (
                                (self.num_speculative_steps + 1,)
                                if self.num_speculative_steps > 0
                                else ()
                            )
                        )
                    ),
                    bin_counts_stride=self.sampler.penalties_state.output_bin_counts.stride(
                        0
                    ),
                    vocab_size=vocab_size,
                    logits_mode=self.sampler.logprobs_mode
                    in ("raw_logits", "processed_logits"),
                )
        if self.rejection_sampler is not None and self._greedy_verification_available():
            assert head is not None and processor is not None
            shard = head.shard_indices
            register_greedy_verification_warmups(
                logits_dtype=(processor.head_dtype or head.weight.dtype),
                logits_stride=head.num_embeddings_per_partition,
                vocab=shard.num_org_elements,
                vocab_start=shard.org_vocab_start_index,
                tp_size=get_tensor_model_parallel_world_size(),
                num_speculative_steps=self.rejection_sampler.num_speculative_steps,
                step_widths=tuple(
                    dict.fromkeys(
                        (
                            num_draft_steps,
                            self.rejection_sampler.num_speculative_steps,
                        )
                        if num_draft_steps > 0
                        else (self.rejection_sampler.num_speculative_steps,)
                    )
                ),
            )
        if (
            self.rejection_sampler is not None
            and getattr(self, "_sampled_verification_available", lambda: False)()
        ):
            assert head is not None and processor is not None
            register_sampled_verification_warmups(
                logits_dtype=(processor.head_dtype or head.weight.dtype),
                shard_vocab_size=head.num_embeddings_per_partition,
                tp_size=get_tensor_model_parallel_world_size(),
                num_speculative_steps=self.rejection_sampler.num_speculative_steps,
                step_widths=tuple(
                    dict.fromkeys(
                        (
                            num_draft_steps,
                            self.rejection_sampler.num_speculative_steps,
                        )
                        if num_draft_steps > 0
                        else (self.rejection_sampler.num_speculative_steps,)
                    )
                ),
                dead_rows=dead_rows,
            )
        if isinstance(self.speculator, DFlashSpeculator):
            self.speculator.register_warmups(
                self._draft_block_table_layouts(kv_cache_groups, kernel_block_sizes)
            )
        draft_logits = getattr(self.speculator, "draft_logits", None)
        if draft_logits is not None:
            # Probabilistic drafting samples the draft logits with the logits
            # cache: the head's dtype, or the cache's when it scatters into it.
            head_dtype = getattr(processor, "head_dtype", None)
            register_drafting_warmups(
                logits_dtypes=tuple(
                    dtype
                    for dtype in (head_dtype, draft_logits.dtype, torch.float32)
                    if dtype is not None
                ),
                cache_dtype=draft_logits.dtype,
                vocab_size=draft_logits.shape[-1],
                use_fp64=self.model_config.use_fp64_gumbel,
            )

    def _register_kv_cache_warmups(
        self, kv_cache_groups: list[KVCacheGroup], kernel_block_sizes: list[int]
    ) -> None:
        """Declare the staged writes, the block-table kernels and KV block zeroing
        for the KV cache groups ``initialize_kv_cache`` will receive (their layout
        does not depend on the number of blocks)."""
        # Every staged tensor's write (request and sampling states, block
        # tables): the kernels are keyed by element type alone.
        declare_staged_write_kernels()
        if not kv_cache_groups:
            return
        declare_block_table_kernels(
            cp_size=self.dcp_size,
            cp_interleave=self.parallel_config.cp_kv_cache_interleave_size,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=0, kv_cache_tensors=[], kv_cache_groups=kv_cache_groups
        )
        if not getattr(kv_cache_config, "needs_kv_cache_zeroing", False):
            return
        # KVBlockZeroer clears each attention layer's kernel-block page.
        page_sizes = [
            layer_spec.page_size_bytes
            // (layer_spec.block_size // kernel_block_sizes[group_id])
            for group_id, group in enumerate(kv_cache_groups)
            if group_id < len(kernel_block_sizes)
            for layer_spec in group.kv_cache_specs.values()
            if isinstance(layer_spec, AttentionSpec)
        ]
        if page_sizes:
            declare_zero_kv_blocks_kernel()

    def _draft_block_table_layouts(
        self, kv_cache_groups: list[KVCacheGroup], kernel_block_sizes: list[int]
    ) -> list[DraftBlockTableLayout]:
        """The block table of each KV cache group the drafter attends through, as
        ``initialize_kv_cache`` lays it out for memory profiling's minimal KV
        cache and for serving's."""
        assert isinstance(self.speculator, DraftModelSpeculator)
        max_len = self._block_table_max_model_len()
        layouts = []
        for group_id, group in enumerate(kv_cache_groups):
            if self.speculator.draft_attn_layer_names.isdisjoint(group.layer_names):
                continue
            layer_spec = next(iter(group.kv_cache_specs.values()))
            spec = layer_spec
            cp_size = self._group_cp_size(layer_spec)
            kernel_block_size = kernel_block_sizes[group_id]
            layouts.append(
                DraftBlockTableLayout(
                    group_id=group_id,
                    width=group.block_table_width(self.vllm_config, max_len)
                    * (spec.block_size // kernel_block_size),
                    kernel_block_size=kernel_block_size,
                    cp_rank=self.dcp_rank if cp_size > 1 else 0,
                    cp_size=cp_size,
                    cp_interleave=(
                        self.parallel_config.cp_kv_cache_interleave_size
                        if cp_size > 1
                        else 1
                    ),
                )
            )
        return layouts

    def _batch_verifies_greedily(
        self,
        idx_mapping_np: np.ndarray,
        num_reqs: int,
        num_draft_tokens: int,
        has_structured_output: bool,
    ) -> bool:
        if (
            has_structured_output
            or num_reqs == 0
            or num_draft_tokens == 0
            or not self._greedy_verification_available()
        ):
            return False
        assert self.sampler is not None
        states = self.sampler.sampling_states
        # No logprobs: neither top logprobs nor logprob_token_ids.
        return bool(
            np.all(states.temperature.np[idx_mapping_np] == 0.0)
            and not np.any(self.sampler.needs_logits_processing[idx_mapping_np])
            and self.sampler.get_logprobs_dims(idx_mapping_np) is None
        )

    def _verifies_greedily(
        self, input_batch: InputBatch, grammar_output: GrammarOutput | None
    ) -> bool:
        """An all-greedy verification step that nothing else reads the logits of."""
        return self._batch_verifies_greedily(
            input_batch.idx_mapping_np,
            input_batch.num_reqs,
            input_batch.num_draft_tokens,
            grammar_output is not None,
        )

    def _verify_greedily(
        self,
        sample_hidden_states: torch.Tensor,
        input_batch: InputBatch,
        sink_flags: torch.Tensor | None = None,
    ) -> SamplerOutput:
        """The rejection sampler's output for an all-greedy step, without the
        logits all-gather: the same committed tokens, bit for bit."""
        assert self.rejection_sampler is not None and self.sampler is not None
        head = self._language_model().lm_head
        shard = head.shard_indices
        tp_size = get_tensor_model_parallel_world_size()
        drafts = logits_rows(input_batch)[0]
        rejection = self.sink is not None and self.sink.rejection
        local_logits: torch.Tensor | None = None
        row_topk: torch.Tensor | None = None
        if hasattr(head, "tilelang_vocab_greedy") and (
            not rejection
            or (
                hasattr(head, "packed_bf16_exponent_base")
                and self.sink.num_candidates <= 16
            )
        ):
            if rejection:
                assert self.sink is not None and sink_flags is not None
                logger.info_once(
                    "Using certified W_q screened greedy verification under sink rejection (top_k=%d)",
                    self.sink.num_candidates,
                )
                cut_cols = self.sink.cut_cols(
                    sink_flags,
                    drafts,
                    input_batch.cu_num_logits,
                    input_batch.num_reqs,
                    shard.org_vocab_start_index,
                    shard.num_org_elements,
                )
                pairs, row_topk = vocab_parallel_pairs_from_head(
                    head,
                    sample_hidden_states,
                    shard.num_org_elements,
                    shard.org_vocab_start_index,
                    tp_size,
                    cut_cols=cut_cols,
                    top_k=self.sink.num_candidates,
                )
            else:
                pairs = vocab_parallel_pairs_from_head(
                    head,
                    sample_hidden_states,
                    shard.num_org_elements,
                    shard.org_vocab_start_index,
                    tp_size,
                )
        else:
            local_logits = self.model.compute_logits_local(sample_hidden_states)
            if rejection:
                assert self.sink is not None and sink_flags is not None
                # The rank that holds a sunk draft's column masks it.
                self.sink.cut(
                    local_logits,
                    sink_flags,
                    drafts,
                    input_batch.cu_num_logits,
                    input_batch.num_reqs,
                    shard.org_vocab_start_index,
                    shard.num_org_elements,
                )
            pairs = vocab_parallel_pairs(
                local_logits,
                shard.num_org_elements,
                shard.org_vocab_start_index,
                tp_size,
            )
        greedy_steps = min(
            getattr(
                self,
                "_step_draft_steps",
                self.rejection_sampler.num_speculative_steps,
            ),
            self.rejection_sampler.num_speculative_steps,
        )
        sampled, num_sampled = greedy_rejection_sample(
            pairs,
            drafts,
            input_batch.cu_num_logits,
            greedy_steps,
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
            dead_row_cut=getattr(self, "_dead_row_cut", None),
        )
        output = SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        if rejection:
            assert self.sink is not None
            if row_topk is not None:
                output.sink_alternates = SinkAlternates(
                    self.sink.alternates_of_greedy_top_candidates(
                        row_topk,
                        input_batch.cu_num_logits,
                        num_sampled,
                        sampled,
                        tp_size,
                    )
                )
            else:
                assert local_logits is not None
                output.sink_alternates = SinkAlternates(
                    self.sink.alternates_of_greedy_shards(
                        local_logits,
                        input_batch.cu_num_logits,
                        num_sampled,
                        sampled,
                        shard.org_vocab_start_index,
                        shard.num_org_elements,
                        tp_size,
                    )
                )
        return output

    def _batch_verifies_from_shards(
        self,
        idx_mapping_np: np.ndarray,
        num_reqs: int,
        num_draft_tokens: int,
        has_structured_output: bool,
        has_sink_rejection: bool = False,
    ) -> bool:
        if (
            has_structured_output
            or has_sink_rejection
            or num_reqs == 0
            or num_draft_tokens == 0
            or not self._sampled_verification_available()
        ):
            return False
        assert self.sampler is not None
        return bool(
            not self.sampler.any_non_temperature_logits_processing(idx_mapping_np)
            and self.sampler.get_logprobs_dims(idx_mapping_np) is None
        )

    def _verifies_from_shards(
        self,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
        sink_flags: torch.Tensor | None = None,
    ) -> bool:
        """A speculative verification step with temperature >= 0 that needs no
        non-temperature logits processing, logprobs, or sink rejection."""
        has_sink_rejection = (
            sink_flags is not None
            and self.sink is not None
            and self.sink.rejection
        )
        return self._batch_verifies_from_shards(
            input_batch.idx_mapping_np,
            input_batch.num_reqs,
            input_batch.num_draft_tokens,
            grammar_output is not None,
            has_sink_rejection,
        )

    def _verify_from_shards(
        self,
        sample_hidden_states: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        """Shard-local block verification and resampling without all-gathering
        the full vocabulary logits across tensor-parallel ranks."""
        assert self.rejection_sampler is not None and self.sampler is not None
        head = self._language_model().lm_head
        shard = head.shard_indices
        tp_size = get_tensor_model_parallel_world_size()
        draft_sampled, pos = logits_rows(input_batch)
        local_logits = self.model.compute_logits_local(sample_hidden_states)
        kernel_steps = None
        drafts_per_req = input_batch.num_draft_tokens_per_req
        step_draft_steps = min(
            getattr(
                self,
                "_step_draft_steps",
                self.rejection_sampler.num_draft_steps,
            ),
            self.rejection_sampler.num_speculative_steps,
        )
        if step_draft_steps < self.rejection_sampler.num_speculative_steps and (
            drafts_per_req is None
            or min(int(drafts_per_req.max(initial=0)), input_batch.num_draft_tokens)
            <= step_draft_steps
        ):
            kernel_steps = step_draft_steps
        apply_temperature = bool(
            np.any(self.sampler.needs_logits_processing[input_batch.idx_mapping_np])
        )
        dead_row_cut = getattr(self, "_dead_row_cut", None)
        sampled, num_sampled = sampled_rejection_sample_from_shards(
            local_logits=local_logits,
            draft_sampled=draft_sampled,
            cu_num_logits=input_batch.cu_num_logits,
            pos=pos,
            idx_mapping=input_batch.idx_mapping,
            expanded_idx_mapping=input_batch.expanded_idx_mapping,
            expanded_local_pos=input_batch.expanded_local_pos,
            temperature=self.sampler.sampling_states.temperature.gpu,
            seed=self.sampler.sampling_states.seeds.gpu,
            num_speculative_steps=self.rejection_sampler.num_speculative_steps,
            vocab=shard.num_org_elements,
            vocab_start=shard.org_vocab_start_index,
            vocab_size=self.vocab_size,
            tp_size=tp_size,
            dead_row_cut=dead_row_cut,
            kernel_steps=kernel_steps,
            apply_temperature=apply_temperature,
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
            dead_row_cut=dead_row_cut,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def _samples_from_shards(
        self,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
        sink_flags: torch.Tensor | None = None,
    ) -> bool:
        """A 1-token-per-request step (e.g. prefill bonus token or non-draft
        step) with at least one sampled stream and no non-temperature logits
        processing or logprobs."""
        has_sink_rejection = (
            sink_flags is not None
            and self.sink is not None
            and self.sink.rejection
        )
        if (
            grammar_output is not None
            or has_sink_rejection
            or input_batch.num_reqs == 0
            or input_batch.num_draft_tokens != 0
            or not self._sampled_verification_available()
        ):
            return False
        assert self.sampler is not None
        idx_mapping_np = input_batch.idx_mapping_np
        states = self.sampler.sampling_states
        return bool(
            np.any(states.temperature.np[idx_mapping_np] != 0.0)
            and not self.sampler.any_non_temperature_logits_processing(idx_mapping_np)
            and self.sampler.get_logprobs_dims(idx_mapping_np) is None
        )

    def _sample_from_shards(
        self,
        sample_hidden_states: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        """Shard-local Gumbel/greedy token sampling for 1-token-per-request
        steps without all-gathering the full vocabulary logits."""
        assert self.sampler is not None
        head = self._language_model().lm_head
        shard = head.shard_indices
        tp_size = get_tensor_model_parallel_world_size()
        _, pos = logits_rows(input_batch)
        local_logits = self.model.compute_logits_local(sample_hidden_states)
        apply_temperature = bool(
            np.any(self.sampler.needs_logits_processing[input_batch.idx_mapping_np])
        )
        sampled = sample_tokens_from_shards(
            local_logits=local_logits,
            cu_num_logits=input_batch.cu_num_logits,
            pos=pos,
            expanded_idx_mapping=input_batch.expanded_idx_mapping,
            temperature=self.sampler.sampling_states.temperature.gpu,
            seed=self.sampler.sampling_states.seeds.gpu,
            vocab=shard.num_org_elements,
            vocab_start=shard.org_vocab_start_index,
            tp_size=tp_size,
            apply_temperature=apply_temperature,
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            torch.ones(
                input_batch.num_reqs, dtype=torch.int32, device=sampled.device
            ),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def _samples_split_from_shards(
        self,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
        sink_flags: torch.Tensor | None = None,
    ) -> bool:
        """Mixed batch where some requests are temperature-only/greedy (eligible
        for shard-local sampling/verification) and others require non-temperature
        logits controls (top_p, top_k, min_p, penalties, logit_bias, bad_words,
        thinking_budget). Partitioning per request keeps the fast subset off the
        full-vocab all-gather, FP32 logit copy, and top-k/top-p kernels, while
        giving the control subset full SM parallelism sized to its active rows."""
        has_sink_rejection = (
            sink_flags is not None
            and self.sink is not None
            and self.sink.rejection
        )
        if (
            grammar_output is not None
            or has_sink_rejection
            or input_batch.num_reqs < 2
            or not getattr(self, "_cu_num_logits_np_exact", True)
            or not self._sampled_verification_available()
        ):
            return False
        assert self.sampler is not None
        idx_mapping_np = input_batch.idx_mapping_np
        if self.sampler.get_logprobs_dims(idx_mapping_np) is not None:
            return False
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            processor = getattr(self._language_model(), "logits_processor", None)
            if processor is None or not getattr(processor, "use_all_gather", False):
                return False
        ctrl_mask = self.sampler.needs_non_temperature_logits_processing[idx_mapping_np]
        return bool(ctrl_mask.any() and not ctrl_mask.all())

    def _sample_split_from_shards(
        self,
        sample_hidden_states: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        """Execute a mixed sampling/verification batch by routing temperature-only
        and greedy requests through the shard-local path and gathering/processing
        full-vocabulary logits only for requests with non-temperature controls."""
        from dataclasses import replace

        assert self.sampler is not None
        device = sample_hidden_states.device
        lm = self._language_model()
        head = lm.lm_head
        shard = head.shard_indices
        processor = lm.logits_processor
        tp_size = get_tensor_model_parallel_world_size()

        idx_mapping_np = input_batch.idx_mapping_np
        ctrl_mask = self.sampler.needs_non_temperature_logits_processing[idx_mapping_np]
        counts = np.diff(input_batch.cu_num_logits_np)

        fast_req_np = np.flatnonzero(~ctrl_mask)
        ctrl_req_np = np.flatnonzero(ctrl_mask)
        row_is_ctrl = np.repeat(ctrl_mask, counts)
        fast_row_np = np.flatnonzero(~row_is_ctrl)
        ctrl_row_np = np.flatnonzero(row_is_ctrl)

        fast_cu_np = np.zeros(fast_req_np.shape[0] + 1, dtype=np.int32)
        np.cumsum(counts[fast_req_np], out=fast_cu_np[1:])
        ctrl_cu_np = np.zeros(ctrl_req_np.shape[0] + 1, dtype=np.int32)
        np.cumsum(counts[ctrl_req_np], out=ctrl_cu_np[1:])

        fast_req_idx = async_tensor_h2d(fast_req_np.astype(np.int64), device=device)
        ctrl_req_idx = async_tensor_h2d(ctrl_req_np.astype(np.int64), device=device)
        fast_row_idx = async_tensor_h2d(fast_row_np.astype(np.int64), device=device)
        ctrl_row_idx = async_tensor_h2d(ctrl_row_np.astype(np.int64), device=device)
        fast_cu = async_tensor_h2d(fast_cu_np, device=device)
        ctrl_cu = async_tensor_h2d(ctrl_cu_np, device=device)

        draft_sampled, pos = logits_rows(input_batch)
        local_logits = self.model.compute_logits_local(sample_hidden_states)

        # 1. Fast subset: shard-local greedy / temperature-only verification or sampling.
        fast_local = local_logits.index_select(0, fast_row_idx)
        fast_idx_mapping = input_batch.idx_mapping.index_select(0, fast_req_idx)
        fast_idx_mapping_np = idx_mapping_np[fast_req_np]
        fast_exp_idx = input_batch.expanded_idx_mapping.index_select(0, fast_row_idx)
        fast_exp_lpos = input_batch.expanded_local_pos.index_select(0, fast_row_idx)
        fast_seq_lens = input_batch.seq_lens.index_select(0, fast_req_idx)
        fast_pos = pos.index_select(0, fast_row_idx)
        fast_apply_temp = bool(
            np.any(self.sampler.needs_logits_processing[fast_idx_mapping_np])
        )
        dead_row_cut = getattr(self, "_dead_row_cut", None)
        prefill_len = self.sampler.req_states.prefill_len.gpu

        use_rejection = (
            input_batch.num_draft_tokens > 0 and self.rejection_sampler is not None
        )
        if use_rejection:
            assert self.rejection_sampler is not None
            fast_draft_sampled = draft_sampled.index_select(0, fast_row_idx)
            fast_dead_cut = (
                None
                if dead_row_cut is None
                else dead_row_cut.index_select(0, fast_req_idx)
            )
            kernel_steps = None
            drafts_per_req = input_batch.num_draft_tokens_per_req
            fast_drafts_per_req = (
                None if drafts_per_req is None else drafts_per_req[fast_req_np]
            )
            step_draft_steps = min(
                getattr(
                    self,
                    "_step_draft_steps",
                    self.rejection_sampler.num_draft_steps,
                ),
                self.rejection_sampler.num_speculative_steps,
            )
            if step_draft_steps < self.rejection_sampler.num_speculative_steps and (
                fast_drafts_per_req is None
                or min(
                    int(fast_drafts_per_req.max(initial=0)),
                    input_batch.num_draft_tokens,
                )
                <= step_draft_steps
            ):
                kernel_steps = step_draft_steps
            fast_sampled, fast_num_sampled = sampled_rejection_sample_from_shards(
                local_logits=fast_local,
                draft_sampled=fast_draft_sampled,
                cu_num_logits=fast_cu,
                pos=fast_pos,
                idx_mapping=fast_idx_mapping,
                expanded_idx_mapping=fast_exp_idx,
                expanded_local_pos=fast_exp_lpos,
                temperature=self.sampler.sampling_states.temperature.gpu,
                seed=self.sampler.sampling_states.seeds.gpu,
                num_speculative_steps=self.rejection_sampler.num_speculative_steps,
                vocab=shard.num_org_elements,
                vocab_start=shard.org_vocab_start_index,
                vocab_size=self.vocab_size,
                tp_size=tp_size,
                dead_row_cut=fast_dead_cut,
                kernel_steps=kernel_steps,
                apply_temperature=fast_apply_temp,
            )
            fast_num_sampled, fast_num_rejected = get_num_sampled_and_rejected(
                fast_num_sampled,
                fast_seq_lens,
                fast_cu,
                fast_idx_mapping,
                prefill_len,
                dead_row_cut=fast_dead_cut,
            )
        else:
            fast_sampled = sample_tokens_from_shards(
                local_logits=fast_local,
                cu_num_logits=fast_cu,
                pos=fast_pos,
                expanded_idx_mapping=fast_exp_idx,
                temperature=self.sampler.sampling_states.temperature.gpu,
                seed=self.sampler.sampling_states.seeds.gpu,
                vocab=shard.num_org_elements,
                vocab_start=shard.org_vocab_start_index,
                tp_size=tp_size,
                apply_temperature=fast_apply_temp,
            )
            fast_num_sampled, fast_num_rejected = get_num_sampled_and_rejected(
                torch.ones(fast_req_np.shape[0], dtype=torch.int32, device=device),
                fast_seq_lens,
                fast_cu,
                fast_idx_mapping,
                prefill_len,
            )

        # 2. Control subset: gather full vocabulary only for non-temperature rows.
        ctrl_local = local_logits.index_select(0, ctrl_row_idx)
        ctrl_logits = processor._gather_logits(ctrl_local) if tp_size > 1 else ctrl_local
        ctrl_logits = ctrl_logits[:, : self.vocab_size]
        ctrl_idx_mapping = input_batch.idx_mapping.index_select(0, ctrl_req_idx)
        ctrl_idx_mapping_np = idx_mapping_np[ctrl_req_np]
        ctrl_exp_idx = input_batch.expanded_idx_mapping.index_select(0, ctrl_row_idx)
        ctrl_exp_lpos = input_batch.expanded_local_pos.index_select(0, ctrl_row_idx)
        ctrl_seq_lens = input_batch.seq_lens.index_select(0, ctrl_req_idx)
        ctrl_drafts_per_req = (
            None
            if input_batch.num_draft_tokens_per_req is None
            else input_batch.num_draft_tokens_per_req[ctrl_req_np]
        )
        ctrl_batch = replace(
            input_batch,
            req_ids=[input_batch.req_ids[i] for i in ctrl_req_np],
            num_reqs=int(ctrl_req_np.shape[0]),
            num_reqs_after_padding=int(ctrl_req_np.shape[0]),
            idx_mapping=ctrl_idx_mapping,
            idx_mapping_np=ctrl_idx_mapping_np,
            expanded_idx_mapping=ctrl_exp_idx,
            expanded_local_pos=ctrl_exp_lpos,
            num_scheduled_tokens=input_batch.num_scheduled_tokens[ctrl_req_np],
            num_draft_tokens=(
                int(ctrl_cu_np[-1] - ctrl_req_np.shape[0])
                if not use_rejection
                else input_batch.num_draft_tokens
            ),
            num_draft_tokens_per_req=ctrl_drafts_per_req,
            seq_lens=ctrl_seq_lens,
            logits_indices=input_batch.logits_indices.index_select(0, ctrl_row_idx),
            cu_num_logits=ctrl_cu,
            cu_num_logits_np=ctrl_cu_np,
        )
        ctrl_batch._logits_rows = (
            draft_sampled.index_select(0, ctrl_row_idx),
            pos.index_select(0, ctrl_row_idx),
        )
        if use_rejection:
            assert self.rejection_sampler is not None
            ctrl_dead_cut = (
                None
                if dead_row_cut is None
                else dead_row_cut.index_select(0, ctrl_req_idx)
            )
            ctrl_out = self.rejection_sampler(
                ctrl_logits,
                ctrl_batch,
                None,
                dead_row_cut=ctrl_dead_cut,
                sink_flags=None,
            )
        else:
            ctrl_out = self.sampler(ctrl_logits, ctrl_batch)

        # 3. Merge fast and control outputs back into batch order.
        ctrl_sampled = ctrl_out.sampled_token_ids
        if fast_sampled.ndim == 1 and ctrl_sampled.ndim == 2:
            fast_sampled = fast_sampled.view(-1, 1)
        elif ctrl_sampled.ndim == 1 and fast_sampled.ndim == 2:
            ctrl_sampled = ctrl_sampled.view(-1, 1)
        merged_sampled = torch.empty(
            (input_batch.num_reqs, *ctrl_sampled.shape[1:]),
            dtype=ctrl_sampled.dtype,
            device=device,
        )
        merged_sampled.index_copy_(0, fast_req_idx, fast_sampled.to(ctrl_sampled.dtype))
        merged_sampled.index_copy_(0, ctrl_req_idx, ctrl_sampled)

        assert ctrl_out.num_sampled is not None and ctrl_out.num_rejected is not None
        merged_num_sampled = torch.empty(
            (input_batch.num_reqs,), dtype=ctrl_out.num_sampled.dtype, device=device
        )
        merged_num_sampled.index_copy_(
            0, fast_req_idx, fast_num_sampled.to(ctrl_out.num_sampled.dtype)
        )
        merged_num_sampled.index_copy_(0, ctrl_req_idx, ctrl_out.num_sampled)

        merged_num_rejected = torch.empty(
            (input_batch.num_reqs,), dtype=ctrl_out.num_rejected.dtype, device=device
        )
        merged_num_rejected.index_copy_(
            0, fast_req_idx, fast_num_rejected.to(ctrl_out.num_rejected.dtype)
        )
        merged_num_rejected.index_copy_(0, ctrl_req_idx, ctrl_out.num_rejected)

        return SamplerOutput(
            sampled_token_ids=merged_sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=merged_num_sampled,
            num_rejected=merged_num_rejected,
        )

    def _init_sink(self, config: SinkConfig, custom_sampler: bool) -> None:
        """Builds the sink detector's sampling state, after the samplers, and
        refuses the configurations it does not handle."""
        parallel = self.parallel_config
        unsupported = [
            ("batch-sharded sampling", self.batch_sharder is not None),
            ("prefill context parallelism", parallel.prefill_context_parallel_size > 1),
            (
                "dual-batch overlap",
                parallel.use_ubatching and parallel.data_parallel_size > 1,
            ),
        ]
        if config.rejection:
            spec = self.speculative_config
            unsupported += [
                ("a model-specific sampler", custom_sampler),
                (
                    "drafters other than DSpark and DFlash",
                    spec is not None and spec.method not in ("dspark", "dflash"),
                ),
                (
                    "probabilistic drafts",
                    getattr(self.speculator, "draft_logits", None) is not None,
                ),
                ("watermarking", self.vllm_config.watermark_config is not None),
                ("sampling masks", self.model_config.return_sampling_mask),
                ("trace replay", self.model_config.enable_trace_replay),
                ("routed experts", self.model_config.enable_return_routed_experts),
                ("KV transfer", self.vllm_config.kv_transfer_config is not None),
                (
                    "request boundary checkpoints",
                    getattr(self.vllm_config, "use_request_boundary_checkpoints", False),
                ),
            ]
        names = [name for name, present in unsupported if present]
        if names:
            raise ValueError(
                f"The sink detector{' and rejection' if config.rejection else ''} "
                f"(spark_ds41.sampling.sink) do not support {', '.join(names)}."
            )
        get_columns = getattr(self.model, "get_sink_residual_columns", None)
        if get_columns is None or get_columns() is None:
            raise ValueError(
                "The sink detector needs the model to keep its residual column "
                "(get_sink_residual_columns); DeepSeek V4.1 on SM120 does."
            )
        assert self.sampler is not None
        self.sink = SinkSampler(
            config,
            self.max_num_reqs,
            self.model_config.get_hidden_size(),
            self.vocab_size,
            self.num_speculative_steps,
            self.device,
            self.model_config.use_fp64_gumbel,
        )
        if config.rejection:
            # A draw's alternates are the next of its Gumbel scores, which
            # FlashInfer's sampler does not expose.
            self.sampler.use_flashinfer = False
            self.sampler.sink = self.sink
            if self.rejection_sampler is not None:
                self.rejection_sampler.sink = self.sink
        logger.info(
            "Sink detector: residual column %d, threshold %g, rejection %s "
            "(%d alternates per provisional token)",
            config.column,
            config.threshold,
            "on" if config.rejection else "off",
            config.num_alternates,
        )

    def _drafter_spec_tokens(
        self, scheduler_output: SchedulerOutput
    ) -> dict[str, list[int]]:
        """The scheduled drafter tokens per request. Draft zero (sink
        rejection) is not one: it is the request's last sampled token, the
        input of its first row, which the request state already holds."""
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        draft_zero = scheduler_output.sink_draft_zero_req_ids
        if not draft_zero:
            return spec_tokens
        return {
            req_id: tokens[1:] if req_id in draft_zero else tokens
            for req_id, tokens in spec_tokens.items()
            if req_id not in draft_zero or len(tokens) > 1
        }

    def _set_sink_decoding(
        self, scheduler_output: SchedulerOutput, batch_req_state: "BatchReqState"
    ) -> None:
        """Marks the requests that are decoding (not finishing a prefill): the
        rows the detector counts as draft zero. With rejection on these are
        exactly the requests the scheduler gave a draft zero."""
        decoding = ~batch_req_state.is_prefilling_np
        assert self.sink is not None
        if self.sink.rejection:
            draft_zero = scheduler_output.sink_draft_zero_req_ids or set()
            for req_id, is_decoding in zip(batch_req_state.req_ids, decoding):
                assert (req_id in draft_zero) == bool(is_decoding), (
                    f"{req_id}: scheduler draft zero {req_id in draft_zero}, "
                    f"decoding {bool(is_decoding)}"
                )
        async_tensor_h2d(
            decoding.astype(np.int8), out=self._sink_decoding[: decoding.shape[0]]
        )

    def _sink_flags(self, input_batch: InputBatch) -> torch.Tensor:
        assert self.sink is not None
        columns = self.model.get_sink_residual_columns()
        flags = self.sink.flags(
            columns,
            input_batch.logits_indices,
            input_batch.cu_num_logits,
            input_batch.num_reqs,
            getattr(self, "_dead_row_cut", None),
        )
        if envs.VLLM_SINK_DETECTOR_CHECK_RANKS:
            self._check_sink_flags_across_ranks(flags)
        return flags

    @staticmethod
    def _check_sink_flags_across_ranks(flags: torch.Tensor) -> None:
        """Lab check: every tensor-parallel rank flagged the same rows."""
        if get_tensor_model_parallel_world_size() == 1:
            return
        gathered = tensor_model_parallel_all_gather(flags[None], dim=0)
        if not bool((gathered == flags[None]).all()):
            raise RuntimeError(
                "Sink detector flags differ across tensor-parallel ranks: "
                f"{gathered.tolist()}"
            )

    def _apply_sink(
        self,
        sampler_output: SamplerOutput,
        input_batch: InputBatch,
        sink_flags: torch.Tensor,
    ) -> None:
        """The draft-zero policy and the step's sink events, on the device."""
        assert self.sink is not None and self.sampler is not None
        assert sampler_output.num_sampled is not None
        assert sampler_output.num_rejected is not None
        num_reqs = input_batch.num_reqs
        sampler_output.sink = self.sink.apply(
            sampler_output.sampled_token_ids,
            sampler_output.num_sampled,
            sampler_output.num_rejected,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self._sink_decoding[:num_reqs],
            sink_flags,
            logits_rows(input_batch)[0],
            sampler_output.sink_alternates,
            self.req_states.total_len.gpu,
            self.sampler.penalties_state.output_bin_counts,
        )


    @property
    def _use_spark_tilelang(self) -> bool:
        return (
            getattr(self.model_config, "architecture", None)
            == "DeepseekV41ForCausalLM"
        )

    def _spark_try_sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor] | None:
        if not self._use_spark_tilelang or self.batch_sharder is not None:
            return None
        if input_batch.num_reqs == 0:
            return None
        if getattr(input_batch, "_logits_rows_contiguous", False):
            sample_hidden_states = hidden_states[: input_batch.logits_indices.shape[0]]
        else:
            sample_hidden_states = tilelang_inputs.gather_hidden_rows(
                hidden_states, input_batch.logits_indices
            )
        sink_flags = self._sink_flags(input_batch) if self.sink is not None else None
        if self._verifies_greedily(input_batch, grammar_output):
            output = self._verify_greedily(
                sample_hidden_states, input_batch, sink_flags
            )
            if sink_flags is not None:
                self._apply_sink(output, input_batch, sink_flags)
            assert output.num_sampled is not None and output.num_rejected is not None
            return output, output.num_sampled, output.num_rejected
        if hasattr(input_batch, "ensure_expanded_idx_mapping"):
            input_batch.ensure_expanded_idx_mapping(self.input_buffers)
        if self._verifies_from_shards(input_batch, grammar_output, sink_flags):
            output = self._verify_from_shards(sample_hidden_states, input_batch)
            if sink_flags is not None:
                self._apply_sink(output, input_batch, sink_flags)
            assert output.num_sampled is not None and output.num_rejected is not None
            return output, output.num_sampled, output.num_rejected
        if self._samples_from_shards(input_batch, grammar_output, sink_flags):
            output = self._sample_from_shards(sample_hidden_states, input_batch)
            if sink_flags is not None:
                self._apply_sink(output, input_batch, sink_flags)
            assert output.num_sampled is not None and output.num_rejected is not None
            return output, output.num_sampled, output.num_rejected
        if self._samples_split_from_shards(input_batch, grammar_output, sink_flags):
            output = self._sample_split_from_shards(sample_hidden_states, input_batch)
            if sink_flags is not None:
                self._apply_sink(output, input_batch, sink_flags)
            assert output.num_sampled is not None and output.num_rejected is not None
            return output, output.num_sampled, output.num_rejected
        return None
