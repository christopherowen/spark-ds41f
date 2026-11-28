# SPDX-License-Identifier: Apache-2.0
"""Spark GB10 worker mixin: order-13 contiguous CUDA pool, RoCE guard, startup
rendezvous, TileLang kernel preparation, and engine_grid RPC."""

from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

import contextlib
import re
import sys

from spark_ds41.worker import display_carveout
from spark_ds41.worker.cpu_affinity import apply_worker_thread_affinity
from vllm.config import set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.platforms import current_platform
from vllm.utils.gc_utils import freeze_gc_heap, maybe_attach_gc_debug_callback
from vllm.utils.gpu_sync_debug import enable_gpu_sync_check
from vllm.utils.torch_utils import set_random_seed, set_torch_threads_for_runtime
from vllm.v1.worker.startup_plan import maybe_save_startup_plan
from spark_ds41.worker import startup_rendezvous
from vllm.v1.worker.worker_base import CompilationTimes
from spark_ds41.worker.mem_utils import reclaim_uma_memory, trim_host_heap
from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphMode
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
)
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.utils.mem_constants import GiB_bytes, MiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

logger = init_logger(__name__)

@contextmanager
def _collective_cause(failure: Callable[[], str | None]):
    """Re-raise an error as the one-shot collectives' failure report, if any.

    A collective that waited past its timeout stopped the GPU, so CUDA raises
    a launch failure at the next sync, which does not say which peer stalled.
    """
    try:
        yield
    except Exception as error:
        report = failure()
        if report is not None:
            raise RuntimeError(report) from error
        raise


class _B12xRoceCheckedAsyncOutput(AsyncModelRunnerOutput):
    """An asynchronous output whose completion is followed by the RoCEnante check."""

    def __init__(
        self,
        inner: AsyncModelRunnerOutput,
        check: Callable[[], None],
        failure: Callable[[], str | None],
    ) -> None:
        self._inner = inner
        self._check = check
        self._failure = failure

    def get_output(self) -> ModelRunnerOutput:
        """Wait for the wrapped output, then run the fail-stop check.

        Returns:
            The completed ModelRunnerOutput.

        Raises:
            RuntimeError: When a RoCEnante wait timed out or its proxy died.
        """
        with _collective_cause(self._failure):
            output = self._inner.get_output()
        self._check()
        return output


def log_startup_memory_breakdown(
    worker: Any,
    cuda_graph_memory_bytes: int = 0,
) -> dict[str, int]:
    """Record and log a per-component GPU/UMA memory breakdown after warmup."""
    device = getattr(worker, "device", None)
    if device is None:
        return {}

    snapshot = MemorySnapshot(device=device)
    model_runner = getattr(worker, "model_runner", None)
    cache_config = getattr(worker, "cache_config", None)

    weights_bytes = int(getattr(model_runner, "model_memory_usage", 0) or 0)
    kv_cache_bytes = int(
        getattr(worker, "available_kv_cache_memory_bytes", 0)
        or getattr(cache_config, "kv_cache_memory_bytes", 0)
        or 0
    )

    from vllm.v1.worker import workspace as workspace_mod

    mgr = workspace_mod._manager
    workspace_bytes = 0
    if mgr is not None:
        workspace_bytes = sum(
            workspace_mod.WorkspaceManager._workspace_size_bytes(ws)
            for ws in getattr(mgr, "_current_workspaces", ())
        ) + int(getattr(mgr, "_moe_output_bytes", 0) or 0)

    torch_allocated_bytes = int(getattr(snapshot, "torch_allocated", 0) or 0)
    torch_reserved_bytes = int(getattr(snapshot, "torch_memory", 0) or 0)
    torch_allocator_slack_bytes = max(
        torch_reserved_bytes - torch_allocated_bytes, 0
    )
    other_torch_allocated_bytes = max(
        torch_allocated_bytes - weights_bytes - kv_cache_bytes - workspace_bytes,
        0,
    )
    peak_activation_bytes = int(
        getattr(worker, "peak_activation_memory", 0) or 0
    )
    cudagraph_bytes = int(cuda_graph_memory_bytes or 0)
    non_torch_cuda_bytes = int(getattr(snapshot, "non_torch_memory", 0) or 0)
    free_memory_bytes = int(getattr(snapshot, "free_memory", 0) or 0)

    breakdown = {
        "weights_bytes": weights_bytes,
        "kv_cache_bytes": kv_cache_bytes,
        "workspace_bytes": workspace_bytes,
        "other_torch_allocated_bytes": other_torch_allocated_bytes,
        "torch_allocated_bytes": torch_allocated_bytes,
        "torch_reserved_bytes": torch_reserved_bytes,
        "torch_allocator_slack_bytes": torch_allocator_slack_bytes,
        "peak_activation_bytes": peak_activation_bytes,
        "cudagraph_bytes": cudagraph_bytes,
        "non_torch_cuda_bytes": non_torch_cuda_bytes,
        "free_memory_bytes": free_memory_bytes,
    }
    worker.startup_memory_breakdown = breakdown
    logger.info(
        "[memory_breakdown] rank=%s weights=%s GiB kv_cache=%s GiB "
        "workspace=%s GiB other_torch_allocated=%s GiB "
        "torch_allocated=%s GiB torch_reserved=%s GiB "
        "torch_allocator_slack=%s GiB peak_activation=%s GiB "
        "cudagraph=%s GiB non_torch_cuda=%s GiB free_memory=%s GiB",
        getattr(worker, "rank", 0),
        format_gib(weights_bytes),
        format_gib(kv_cache_bytes),
        format_gib(workspace_bytes),
        format_gib(other_torch_allocated_bytes),
        format_gib(torch_allocated_bytes),
        format_gib(torch_reserved_bytes),
        format_gib(torch_allocator_slack_bytes),
        format_gib(peak_activation_bytes),
        format_gib(cudagraph_bytes),
        format_gib(non_torch_cuda_bytes),
        format_gib(free_memory_bytes),
    )
    return breakdown


class SparkWorkerMixin:
    _b12x_roce_health_check: Any = None
    _b12x_session: Any = None
    _contiguous_pool_anchor: Any = None
    _contiguous_pool_bytes: int = 0
    _contiguous_pool_front_clean: bool = False
    _contiguous_pool_finalized: bool = False
    _contiguous_pool_max_bytes: int = 0
    _contiguous_pool_override: bool = False
    _contiguous_pool_range: tuple[int, int] = (0, 0)

    @staticmethod
    def _parse_unmovable_sub_order13_bytes(pagetypeinfo_text: str) -> int:
        """Sum free bytes in Zone Normal MIGRATE_UNMOVABLE orders < 13 (64 KiB pages)."""
        total = 0
        for line in pagetypeinfo_text.splitlines():
            if "Normal" in line and "Unmovable" in line and "type" in line:
                parts = line.split()
                try:
                    idx = parts.index("Unmovable")
                    counts = [int(x) for x in parts[idx + 1 :]]
                except (ValueError, IndexError):
                    continue
                if len(counts) >= 14:
                    total += sum(c * (65536 << order) for order, c in enumerate(counts[:-1]))
        return total

    @staticmethod
    def _read_proc_uma_memory_state() -> tuple[int, int, int]:
        """Return ``(order13_bytes, avail_bytes, unmov_sub_bytes)`` on 64 KiB Linux."""
        o13_bytes = 0
        avail_bytes = 0
        unmov_sub_bytes = 0
        try:
            from spark_ds41.direct_io.direct_loader import (
                parse_buddyinfo_order13,
            )

            _, o13_bytes = parse_buddyinfo_order13(
                Path("/proc/buddyinfo").read_text(encoding="utf-8"),
                page_bytes=65536,
            )
        except OSError:
            o13_bytes = 0
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    avail_bytes = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            avail_bytes = 0
        try:
            unmov_sub_bytes = SparkWorkerMixin._parse_unmovable_sub_order13_bytes(
                Path("/proc/pagetypeinfo").read_text(encoding="utf-8")
            )
        except OSError:
            unmov_sub_bytes = 0
        return o13_bytes, avail_bytes, unmov_sub_bytes

    def _allocate_soaked_contiguous_pool(
        self, pool_bytes: int
    ) -> tuple[int, int, int, bool]:
        """Soak MIGRATE_UNMOVABLE sub-order-13 blocks and allocate one anchored
        contiguous CUDA pool of ``pool_bytes``.

        Returns ``(soak_bytes, unmov_sub_bytes, avail_bytes, front_clean)``.
        """
        block_bytes = 512 * 1024 * 1024
        _, avail_bytes, unmov_sub_bytes = self._read_proc_uma_memory_state()

        # One-shot soak of MIGRATE_UNMOVABLE sub-order-13 blocks so nvidia.ko
        # draws the entire weight pool from pure 512 MiB order-13 blocks starting
        # at offset 0. Keep >= 8 GiB MemAvailable above startup memguard.
        soak_headroom_bytes = 8 * 1024**3
        soak_align = 2 * 1024 * 1024
        max_soak = (
            max(0, avail_bytes - pool_bytes - soak_headroom_bytes)
            if avail_bytes > 0
            else 0
        )
        soak_bytes = (
            (min(unmov_sub_bytes, max_soak) // soak_align) * soak_align
            if unmov_sub_bytes >= block_bytes and max_soak >= block_bytes
            else 0
        )

        # Return stale cached-free segments left over from runner/speculator
        # init before allocating the pool so the 2 MiB anchor lands inside the
        # newly allocated segment rather than being swallowed by earlier holes.
        torch.accelerator.empty_cache()
        soak: torch.Tensor | None = None
        try:
            if soak_bytes > 0:
                try:
                    soak = torch.empty(
                        (soak_bytes,), dtype=torch.uint8, device=self.device
                    )
                    soak[::65536].zero_()
                except torch.OutOfMemoryError:
                    soak = None
                    soak_bytes = 0
                    torch.accelerator.empty_cache()

            pool = torch.empty((pool_bytes,), dtype=torch.uint8, device=self.device)
            pool_ptr = int(pool.data_ptr()) if hasattr(pool, "data_ptr") else 0
            pool_end = pool_ptr + pool_bytes if pool_ptr > 0 else 0
            del pool
            # A >= 2 MiB allocation lands in PyTorch's large_pool. Drain any
            # remaining cached-free holes in partially-occupied earlier segments
            # so the 2 MiB anchor is guaranteed to sit inside [pool_ptr, pool_end)
            # and pin the segment across empty_cache() and GC threshold sweeps.
            holders: list[torch.Tensor] = []
            anchor: torch.Tensor | None = None
            try:
                for _ in range(1024):
                    cand = torch.empty(
                        (2 * 1024 * 1024,), dtype=torch.uint8, device=self.device
                    )
                    cand_ptr = int(cand.data_ptr()) if hasattr(cand, "data_ptr") else 0
                    if pool_ptr == 0 or (pool_ptr <= cand_ptr < pool_end):
                        anchor = cand
                        break
                    holders.append(cand)
            finally:
                holders.clear()
            if anchor is None:
                raise RuntimeError(
                    "Failed to place contiguous CUDA pool anchor inside "
                    f"[{pool_ptr:#x}, {pool_end:#x})"
                )
            self._contiguous_pool_anchor = anchor
            self._contiguous_pool_bytes = pool_bytes
            self._contiguous_pool_range = (pool_ptr, pool_end)
            if soak is not None:
                del soak
                soak = None
                torch.accelerator.empty_cache()
            front_clean = (unmov_sub_bytes - soak_bytes) < (1024**3)
            self._contiguous_pool_front_clean = front_clean
            return soak_bytes, unmov_sub_bytes, avail_bytes, front_clean
        except (torch.OutOfMemoryError, RuntimeError):
            if soak is not None:
                del soak
            self._contiguous_pool_anchor = None
            self._contiguous_pool_bytes = 0
            self._contiguous_pool_range = (0, 0)
            self._contiguous_pool_front_clean = False
            torch.accelerator.empty_cache()
            raise

    def _release_contiguous_cuda_pool(self) -> None:
        """Atomically release the pre-allocated contiguous CUDA pool (evacuating
        and restoring any live tensors inside its address range) so fallback paths
        such as direct_loader's buddy sponge never double-allocate on top of it."""
        if (
            getattr(self, "_contiguous_pool_bytes", 0) <= 0
            and getattr(self, "_contiguous_pool_anchor", None) is None
        ):
            return
        pool_ptr, pool_end = getattr(self, "_contiguous_pool_range", (0, 0))
        _, evac_groups, evac_snapshots = self._evacuate_tensors_in_pool_range(
            pool_ptr, pool_end
        )
        try:
            self._contiguous_pool_anchor = None
            self._contiguous_pool_bytes = 0
            self._contiguous_pool_range = (0, 0)
            self._contiguous_pool_front_clean = False
            self._contiguous_pool_finalized = True
            from spark_ds41.direct_io.direct_loader import (
                set_preallocated_cuda_pool_bytes,
            )

            set_preallocated_cuda_pool_bytes(0)
            gc.collect()
            torch.accelerator.empty_cache()
        finally:
            self._restore_evacuated_tensors(evac_groups, evac_snapshots)

    def _evacuate_tensors_in_pool_range(
        self, pool_ptr: int, pool_end: int
    ) -> tuple[
        int,
        dict[
            int,
            tuple[
                int,
                list[tuple[torch.Tensor, int, tuple[int, ...], tuple[int, ...]]],
            ],
        ],
        dict[int, torch.Tensor],
    ]:
        """Snapshot and detach any live CUDA tensors whose storage lies inside
        ``[pool_ptr, pool_end)`` (excluding the pool anchor) so ``empty_cache()``
        can cleanly release the oversized scratch segment."""
        if pool_ptr <= 0 or pool_end <= pool_ptr:
            return 0, {}, {}
        anchor = self._contiguous_pool_anchor
        empty_storage = torch.empty(
            (0,), dtype=torch.uint8, device=self.device
        ).untyped_storage()
        groups: dict[
            int,
            tuple[
                int,
                list[tuple[torch.Tensor, int, tuple[int, ...], tuple[int, ...]]],
            ],
        ] = {}
        snapshots: dict[int, torch.Tensor] = {}
        gc.collect()
        with torch.no_grad():
            for obj in gc.get_objects():
                try:
                    if not isinstance(obj, torch.Tensor):
                        continue
                    if obj is anchor or obj.is_meta or not obj.is_cuda:
                        continue
                    if obj.numel() == 0:
                        continue
                    storage = obj.untyped_storage()
                    nbytes = storage.nbytes()
                    if nbytes == 0:
                        continue
                    ptr = int(storage.data_ptr())
                    if not (pool_ptr <= ptr < pool_end):
                        continue
                    if ptr not in groups:
                        groups[ptr] = (nbytes, [])
                    groups[ptr][1].append(
                        (
                            obj,
                            int(obj.storage_offset()),
                            tuple(obj.shape),
                            tuple(obj.stride()),
                        )
                    )
                except Exception:
                    continue
            for ptr, (nbytes, views) in groups.items():
                orig_storage = views[0][0].untyped_storage()
                src = torch.empty((0,), dtype=torch.uint8, device=self.device).set_(
                    orig_storage
                )
                cpu_copy = torch.empty((nbytes,), dtype=torch.uint8, device="cpu")
                cpu_copy.copy_(src)
                snapshots[ptr] = cpu_copy
                del src, orig_storage
                for tensor, _, _, _ in views:
                    tensor.set_(empty_storage, 0, (0,), (1,))
        evac_bytes = sum(nbytes for nbytes, _ in groups.values())
        return evac_bytes, groups, snapshots

    def _restore_evacuated_tensors(
        self,
        groups: dict[
            int,
            tuple[
                int,
                list[tuple[torch.Tensor, int, tuple[int, ...], tuple[int, ...]]],
            ],
        ],
        snapshots: dict[int, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            for ptr, (nbytes, views) in groups.items():
                cpu_copy = snapshots.pop(ptr, None)
                if cpu_copy is None:
                    continue
                restored = torch.empty(
                    (nbytes,), dtype=torch.uint8, device=self.device
                )
                restored.copy_(cpu_copy)
                restored_storage = restored.untyped_storage()
                for tensor, orig_offset, shape, stride in views:
                    tensor.set_(restored_storage, orig_offset, shape, stride)
                del restored, restored_storage, cpu_copy
            snapshots.clear()

    def _compute_desired_pool_bytes(
        self, total_model_bytes: int, max_moe_bytes: int
    ) -> int:
        block_bytes = 512 * 1024 * 1024
        spec_cfg = getattr(self, "speculative_config", None)
        repack_and_ws = max(int(max_moe_bytes), int(1.5 * 1024**3))
        par_cfg = getattr(self, "parallel_config", None)
        pp_size = int(getattr(par_cfg, "pipeline_parallel_size", 1) or 1)
        has_local_drafter = spec_cfg is not None
        if has_local_drafter and pp_size > 1:
            try:
                from vllm.distributed.parallel_state import get_pp_group

                if torch.distributed.is_initialized():
                    has_local_drafter = bool(get_pp_group().is_last_rank)
            except Exception:
                pass
        if has_local_drafter:
            default_tp = int(getattr(par_cfg, "tensor_parallel_size", 1) or 1)
            draft_tp = int(
                getattr(spec_cfg, "draft_tensor_parallel_size", None) or default_tp
            )
            draft_tp = max(1, draft_tp)
            draft_and_other = max(
                int(1.0 * 1024**3), int((8.0 * 1024**3) // draft_tp)
            )
        else:
            draft_and_other = int(1.0 * 1024**3)
        raw_desired = int(total_model_bytes) + repack_and_ws + draft_and_other
        desired_bytes = ((raw_desired + block_bytes - 1) // block_bytes) * block_bytes
        return max(block_bytes, min(desired_bytes, self._contiguous_pool_max_bytes))

    def _resize_contiguous_cuda_pool_for_model(
        self, model_bytes: int, max_moe_bytes: int = 0
    ) -> tuple[int, bool]:
        """Shrink the initial scratch contiguous pool to the model's actual rank
        weight + draft + workspace footprint so shapes with large KV pools or
        offloaded experts (TP1, TP2, TP4..TP8, or smaller models) never pin
        unused host RAM inside Segment #1."""
        if (
            self._contiguous_pool_finalized
            or self._contiguous_pool_override
            or self._contiguous_pool_bytes <= 0
            or model_bytes <= 0
        ):
            return self._contiguous_pool_bytes, self._contiguous_pool_front_clean

        self._contiguous_pool_finalized = True
        block_bytes = 512 * 1024 * 1024
        old_bytes = self._contiguous_pool_bytes

        def _cached_free_bytes() -> int:
            anchor = getattr(self, "_contiguous_pool_anchor", None)
            try:
                reserved = int(torch.cuda.memory_reserved(self.device))
                allocated = int(torch.cuda.memory_allocated(self.device))
                if reserved > 0 or (isinstance(anchor, torch.Tensor) and anchor.is_cuda):
                    return max(0, reserved - allocated)
            except Exception:
                pass
            return old_bytes

        # Fast path: keep the already-soaked segment only when it is within
        # <= 1 GiB of initial pool AND actually has >= model_bytes cached free.
        est_desired = self._compute_desired_pool_bytes(model_bytes, max_moe_bytes)
        if (
            est_desired >= old_bytes - 2 * block_bytes
            and _cached_free_bytes() >= model_bytes
        ):
            logger.info(
                "Keeping %.1f GiB contiguous CUDA pool (exact fit for %.1f GiB "
                "weights, desired %.1f GiB, front_clean=%s)",
                old_bytes / (1024**3),
                model_bytes / (1024**3),
                est_desired / (1024**3),
                self._contiguous_pool_front_clean,
            )
            return old_bytes, self._contiguous_pool_front_clean

        t0 = time.perf_counter()
        pool_ptr, pool_end = getattr(self, "_contiguous_pool_range", (0, 0))
        evac_bytes, evac_groups, evac_snapshots = self._evacuate_tensors_in_pool_range(
            pool_ptr, pool_end
        )
        try:
            # Include any evacuated model/vocab/buffer tensors in the final size!
            desired_bytes = self._compute_desired_pool_bytes(
                model_bytes + evac_bytes, max_moe_bytes
            )
            if (
                desired_bytes >= old_bytes - 2 * block_bytes
                and _cached_free_bytes() >= model_bytes
            ):
                logger.info(
                    "Keeping %.1f GiB contiguous CUDA pool (exact fit for %.1f + %.1f GiB "
                    "weights, desired %.1f GiB, front_clean=%s)",
                    old_bytes / (1024**3),
                    model_bytes / (1024**3),
                    evac_bytes / (1024**3),
                    desired_bytes / (1024**3),
                    self._contiguous_pool_front_clean,
                )
                return old_bytes, self._contiguous_pool_front_clean

            try:
                pre_release_reserved = int(torch.cuda.memory_reserved(self.device))
            except Exception:
                pre_release_reserved = old_bytes
            self._contiguous_pool_anchor = None
            gc.collect()
            torch.accelerator.empty_cache()
            try:
                post_release_reserved = int(torch.cuda.memory_reserved(self.device))
            except Exception:
                post_release_reserved = 0
            freed_bytes = pre_release_reserved - post_release_reserved
            if post_release_reserved >= desired_bytes and freed_bytes < (old_bytes // 2):
                self._contiguous_pool_anchor = torch.empty(
                    (2 * 1024 * 1024,), dtype=torch.uint8, device=self.device
                )
                logger.info(
                    "Skipped shrinking %.1f GiB contiguous CUDA pool to %.1f GiB "
                    "(freed %.1f GiB, %.1f GiB remained reserved after empty_cache)",
                    old_bytes / (1024**3),
                    desired_bytes / (1024**3),
                    freed_bytes / (1024**3),
                    post_release_reserved / (1024**3),
                )
                return old_bytes, self._contiguous_pool_front_clean

            from spark_ds41.direct_io.direct_loader import (
                set_preallocated_cuda_pool_bytes,
            )

            try:
                soak_bytes, unmov_sub_bytes, avail_bytes, front_clean = (
                    self._allocate_soaked_contiguous_pool(desired_bytes)
                )
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                set_preallocated_cuda_pool_bytes(0)
                logger.warning(
                    "Resizing contiguous CUDA pool to %.1f GiB failed (%s); "
                    "falling back to standard allocator",
                    desired_bytes / (1024**3),
                    exc,
                )
                return 0, False

            set_preallocated_cuda_pool_bytes(
                desired_bytes,
                front_clean=front_clean,
                resize_cb=None,
                release_cb=self._release_contiguous_cuda_pool,
            )
            logger.info(
                "Resized contiguous CUDA pool %.1f -> %.1f GiB for %.1f GiB weights "
                "(+%.2f GiB evacuated) in %.2f s (freed %.1f GiB, soaked %.2f/%.2f GiB "
                "Unmovable sub-O13, MemAvailable was %.1f GiB, front_clean=%s)",
                old_bytes / (1024**3),
                desired_bytes / (1024**3),
                model_bytes / (1024**3),
                evac_bytes / (1024**3),
                time.perf_counter() - t0,
                (old_bytes - desired_bytes) / (1024**3),
                soak_bytes / (1024**3),
                unmov_sub_bytes / (1024**3),
                avail_bytes / (1024**3),
                front_clean,
            )
            return desired_bytes, front_clean
        finally:
            if evac_groups:
                self._restore_evacuated_tensors(evac_groups, evac_snapshots)

    def _preallocate_contiguous_cuda_pool(self) -> int:
        """Pre-allocate one large contiguous caching-allocator segment on 64 KiB
        integrated UMA CUDA hosts when native cudaMalloc caching is active.

        On 64 KiB ARM64 UMA hosts (GB10) with iommu.passthrough=0, hundreds of
        small cuMemCreate/cudaMalloc segments interleave immortal SMMU L3 page
        tables into 512 MiB order-13 pageblocks and cause permanent boot-to-boot
        slowdown. Pre-allocating one contiguous scratch segment keeps model
        construction inside a single cudaMalloc; bind_weight_arena then shrinks
        it to the rank's exact weight + draft + workspace footprint so every
        TP shape (TP1..TP8) and model size avoids pinning unused host RAM.
        """
        self._contiguous_pool_anchor: torch.Tensor | None = None
        self._contiguous_pool_bytes = 0
        self._contiguous_pool_range = (0, 0)
        self._contiguous_pool_max_bytes = 0
        self._contiguous_pool_front_clean = False
        self._contiguous_pool_finalized = False
        self._contiguous_pool_override = False
        from spark_ds41.direct_io.direct_loader import (
            set_preallocated_cuda_pool_bytes,
        )

        set_preallocated_cuda_pool_bytes(0)

        raw_override = os.environ.get("VLLM_CUDA_PREALLOC_POOL_MIB", "").strip()
        if raw_override.lower() in ("0", "false", "off", "no"):
            return 0
        if not current_platform.is_cuda():
            return 0
        conf = torch._C._accelerator_getAllocatorSettings()
        if re.search(
            r"(?:^|,)\s*(?:expandable_segments\s*:\s*True|backend\s*:\s*cudaMallocAsync)\s*(?:,|$)",
            conf,
        ):
            return 0

        override_mib: int | None = None
        if raw_override:
            try:
                override_mib = int(raw_override)
            except ValueError:
                logger.warning(
                    "Ignoring invalid VLLM_CUDA_PREALLOC_POOL_MIB=%r", raw_override
                )
                return 0
            if override_mib <= 0:
                return 0

        if override_mib is None:
            try:
                page_bytes = os.sysconf("SC_PAGE_SIZE")
            except (AttributeError, OSError, ValueError):
                page_bytes = 0
            if page_bytes < 65536:
                return 0
            try:
                props = torch.cuda.get_device_properties(self.device)
                is_integrated = bool(getattr(props, "is_integrated", False))
            except Exception:
                is_integrated = False
            if not is_integrated:
                return 0

        block_bytes = 512 * 1024 * 1024
        o13_bytes, avail_bytes, _ = self._read_proc_uma_memory_state()

        if override_mib is not None:
            pool_bytes = (override_mib * 1024 * 1024 // block_bytes) * block_bytes
            if pool_bytes <= 0:
                pool_bytes = override_mib * 1024 * 1024
            self._contiguous_pool_override = True
            self._contiguous_pool_finalized = True
        else:
            # Auto-activate only when enough 512 MiB blocks exist to benefit;
            # on an already-fragmented host, let direct_loader's buddy sponge run.
            if o13_bytes < 32 * 1024**3:
                return 0
            # Exclude explicit kv_cache_memory_bytes from the upper-bound weight
            # pool so weights + draft + workspace fit in Segment #1 and KV cache
            # takes one clean contiguous Segment #2 with zero slack spill.
            kv_bytes = int(getattr(self.cache_config, "kv_cache_memory_bytes", 0) or 0)
            req_bytes = int(self.requested_memory)
            if kv_bytes > 0 and req_bytes > kv_bytes + 16 * 1024**3:
                req_bytes -= kv_bytes
            headroom_bytes = 12 * 1024**3
            cap_bytes = (
                max(0, avail_bytes - headroom_bytes)
                if avail_bytes > 0
                else req_bytes
            )
            target_bytes = min(req_bytes, cap_bytes)
            pool_bytes = (target_bytes // block_bytes) * block_bytes
            if pool_bytes < 16 * 1024**3:
                return 0

        self._contiguous_pool_max_bytes = pool_bytes
        t0 = time.perf_counter()
        try:
            soak_bytes, unmov_sub_bytes, avail_bytes, front_clean = (
                self._allocate_soaked_contiguous_pool(pool_bytes)
            )
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            logger.warning(
                "Pre-allocation of %.1f GiB contiguous CUDA pool failed (%s); "
                "falling back to standard allocator",
                pool_bytes / (1024**3),
                exc,
            )
            return 0

        set_preallocated_cuda_pool_bytes(
            pool_bytes,
            front_clean=front_clean,
            resize_cb=(
                None
                if self._contiguous_pool_override
                else self._resize_contiguous_cuda_pool_for_model
            ),
            release_cb=self._release_contiguous_cuda_pool,
        )
        logger.info(
            "Pre-allocated %.1f GiB single contiguous CUDA pool in %.2f s "
            "(soaked %.2f/%.2f GiB Unmovable sub-O13, order-13 free was %.1f GiB, "
            "MemAvailable was %.1f GiB, front_clean=%s)",
            pool_bytes / (1024**3),
            time.perf_counter() - t0,
            soak_bytes / (1024**3),
            unmov_sub_bytes / (1024**3),
            o13_bytes / (1024**3),
            avail_bytes / (1024**3),
            front_clean,
        )
        return pool_bytes

    def _compile_or_warm_up_model_after_preparation(self) -> CompilationTimes:
        # Drop to the serving thread count before any warmup. Dynamo guards
        # compiled functions on torch.get_num_threads(), so changing it after
        # warmup and graph capture made the first request recompile every
        # torch.compile'd helper. Weight loading, the only startup phase that
        # benefits from intra-op parallelism, is already done.
        set_torch_threads_for_runtime()
        warmup_sizes: list[int] = []

        if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            # warm up sizes that are not in cudagraph capture sizes,
            # but users still want to compile for better performance,
            # e.g. for the max-num-batched token size in chunked prefill.
            compile_sizes = self.vllm_config.compilation_config.compile_sizes
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []  # type: ignore[assignment]
            cg_capture_sizes: list[int] = []

            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
        from vllm.v1.worker.gpu.warmup import warmup_kernels
        kernel_warmup(self)
        # The JIT warm-up compiles locally, for as long as each rank's cache needs.
        self._startup_rendezvous("warm-up")

        if getattr(self, "use_v2_model_runner", True):
            # A workspace resize after capture frees what the graphs point at.
            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
            gc.collect()
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()
            trim_host_heap()

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            # Under capture the b12x session refuses kernel resolution and an
            # unprepared plan is an error; captured shapes are always planned.
            session = self._b12x_session
            guard = (
                session.capture()
                if session is not None and session.state != "CLOSED"
                else contextlib.nullcontext()
            )
            with guard, self._get_cudagraph_capture_context():
                cuda_graph_memory_bytes = self.model_runner.capture_model()

        # Compare actual vs estimated CUDA graph memory (if we did profiling)
        if (
            hasattr(self, "cudagraph_memory_estimate")
            and self.cudagraph_memory_estimate > 0
        ):
            GiB = lambda b: round(b / GiB_bytes, 2)
            diff = abs(cuda_graph_memory_bytes - self.cudagraph_memory_estimate)
            logger.info(
                "CUDA graph pool memory: %s GiB (actual), %s GiB (estimated), "
                "difference: %s GiB (%.1f%%).",
                GiB(cuda_graph_memory_bytes),
                GiB(self.cudagraph_memory_estimate),
                GiB(diff),
                100 * diff / max(cuda_graph_memory_bytes, 1),
            )

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            redundancy_buffer_memory = 150 * (1 << 20)

            non_kv_cache_memory = (
                self.total_consumed
                + self.peak_activation_memory
                + cuda_graph_memory_bytes
            )
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_memory)} GiB). "
                f"Actual usage is {format_gib(self.total_consumed)} "
                f"GiB for consumed memory (weights + non-torch), "
                f"{format_gib(self.peak_activation_memory)} GiB "
                f"for peak activation, and {format_gib(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.info(msg)

            maybe_save_startup_plan(self, kv_cache_memory_bytes_to_requested_limit)

        if getattr(self, "use_v2_model_runner", True):
            if (
                not self.model_config.enforce_eager
                and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            ):
                # Exercise captured execution paths before serving requests.
                from vllm.v1.worker.gpu.warmup import warmup_kernels
                warmup_kernels(
                    self.model_runner,
                    self.execute_model,
                    self.sample_tokens,
                    post_capture=True,
                )
                gc.collect()
                torch.accelerator.synchronize()
                torch.accelerator.empty_cache()
                trim_host_heap()
        elif get_pp_group().is_last_rank:
            # V1: Warm up sampler and preallocate memory buffer for logits and other
            # sampling related tensors of max possible shape to avoid memory
            # fragmentation issue.
            # NOTE: This is called after `capture_model` on purpose to prevent
            # memory buffers from being cleared by `torch.accelerator.empty_cache`.
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        # Eagerly trigger inductor's once-per-process lazy inits during
        # warmup (rather than on a later compile cache-miss at runtime).
        c_config = self.compilation_config
        if c_config.mode != CompilationMode.NONE and c_config.backend == "inductor":
            from vllm.compilation.compiler_interface import (
                trigger_inductor_lazy_init,
            )

            trigger_inductor_lazy_init(self.device)

        if self._startup_heartbeat is not None:
            self._startup_heartbeat.stop()
            self._startup_heartbeat = None

        # All warmup is done — start monitoring for unexpected JIT
        # compilations that would cause latency spikes during inference.
        from vllm.utils.jit_monitor import activate as activate_jit_monitor

        activate_jit_monitor(
            mode=self.observability_config.jit_monitor_mode,
            verbose=self.observability_config.jit_monitor_verbose,
        )
        if "sparknet.oneshot" in sys.modules:
            # sparknet prepared its kernels before capture; one compiling now
            # would stall a step, so it raises instead.
            sys.modules["sparknet.oneshot"].freeze_kernel_resolution(
                "vLLM warm-up is complete"
            )

        apply_worker_thread_affinity(getattr(self, "_cpu_affinity_plan", None))

        # Freeze the worker heap and trim host arenas before logging the
        # steady-state startup memory breakdown.
        freeze_gc_heap()
        maybe_attach_gc_debug_callback()

        log_startup_memory_breakdown(self, cuda_graph_memory_bytes)

        # Warmup / first-compile is done — activate the `VLLM_GPU_SYNC_CHECK`
        # gate so subsequent `execute_model` / `sample_tokens` calls enforce it.
        enable_gpu_sync_check()

        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def _pp_p2p_runtimes(self) -> list[Any]:
        runtimes: list[Any] = []
        pp_p2p = getattr(self, "_pp_p2p", None)
        if pp_p2p is not None and not getattr(pp_p2p, "disabled", False):
            runtimes.append(pp_p2p)
        runner = getattr(self, "model_runner", None)
        pp_handler = getattr(runner, "pp_handler", None)
        pp_bcast = getattr(pp_handler, "_pp_bcast_p2p", None)
        if pp_bcast is not None and not getattr(pp_bcast, "disabled", False):
            runtimes.append(pp_bcast)
        return runtimes

    def _roce_failure(self) -> str | None:
        """One-shot TP collective and PP P2P failure report, or None."""
        parts: list[str] = []
        communicator = get_tp_group().device_communicator
        failure = getattr(communicator, "roce_failure", None)
        if failure is not None:
            report = failure()
            if report:
                parts.append(report)
        for rt in self._pp_p2p_runtimes():
            report = rt.failure()
            if report:
                parts.append(report)
        return "; ".join(parts) if parts else None

    def _start_startup_heartbeat(self) -> None:
        if get_world_group().world_size == 1:
            return
        from spark_ds41.worker.startup_rendezvous import StartupHeartbeat
        store = torch.distributed.distributed_c10d._get_default_store()
        self._startup_heartbeat = StartupHeartbeat(
            store, get_world_group().rank, self._startup_progress
        ).start()

    def _startup_progress(self) -> str:
        from spark_ds41.tilelang.prepare import PROGRESS
        done, total = PROGRESS.snapshot()
        return f"prepared {done}/{total} kernels" if total else ""

    def _set_startup_phase(self, phase: str) -> None:
        if self._startup_heartbeat is not None:
            self._startup_heartbeat.phase = phase

    def _prepare_kernels(self) -> None:
        """Compile every declared JIT kernel before any distributed work.

        DeepSeek V4.1 declares each TileLang call its loaded modules can make;
        the runner's JIT registry holds the Triton kernels its samplers and
        model state registered, and the runner adds its step's kernels here, once
        the KV cache layout is set. They compile here on this rank alone,
        TileLang's in one thread pool, and TileLang's kernel cache is then frozen,
        so profiling, warm-up, capture and serving compile nothing: a kernel that
        was not declared fails by name on every rank alike.
        """
        registry = getattr(self.model_runner, "jit_warmup_registry", None)
        if registry is None:
            return
        self._set_startup_phase("preparing kernels")
        started = time.monotonic()
        # Declarations read the scheduler sizing and model configuration.
        with set_current_vllm_config(self.vllm_config):
            from spark_ds41.tilelang import prepare

            speculator = getattr(self.model_runner, "speculator", None)
            with registry.activate():
                prepare.declare_modules(
                    self.model_runner.model, getattr(speculator, "model", None)
                )
            tilelang_calls = prepare.prepare_declared(registry)
            with torch.inference_mode():
                prepare.warm_declared(registry)
            # A step's Triton kernels (input preparation, sampling, verification,
            # drafting), specialized on the runner's buffers, the loaded heads and
            # the block tables of the KV cache groups (the layout is set by now).
            declare_step_kernels = getattr(
                self.model_runner, "declare_step_kernels", None
            )
            if declare_step_kernels is not None:
                with registry.activate():
                    declare_step_kernels()
            registry.warmup()
            if tilelang_calls:
                prepare.release_host_memory()
        if tilelang_calls:
            prepare.freeze("every declared kernel was prepared before memory profiling")
            with set_current_vllm_config(self.vllm_config):
                prepare.bind_declared(registry)
            # Nothing compiles from here on: the JIT monitor reports (or, in error
            # mode, refuses) any other compile, Triton's included, from profiling on.
            from vllm.utils.jit_monitor import activate as activate_jit_monitor

            if self.observability_config.jit_monitor_mode != "off":
                activate_jit_monitor(
                    mode=self.observability_config.jit_monitor_mode,
                    verbose=self.observability_config.jit_monitor_verbose,
                )
        logger.info(
            "Prepared %d TileLang calls and the registry's other kernels in %.1f s.",
            tilelang_calls,
            time.monotonic() - started,
        )

    def _startup_rendezvous(self, phase: str) -> None:
        """Wait on the host until every rank reaches ``phase``.

        Ranks finish loading and preparing kernels at their own pace. The wait
        lasts while every missing rank is alive, however long it takes, and fails
        naming a rank whose heartbeat stopped (see ``startup_rendezvous``).
        """
        group = get_world_group()
        if group.world_size == 1:
            return
        from spark_ds41.worker.startup_rendezvous import rendezvous

        self._set_startup_phase(f"waiting before {phase}")
        waited = rendezvous(
            torch.distributed.distributed_c10d._get_default_store(),
            group.rank,
            group.world_size,
            phase,
            envs.VLLM_STARTUP_HEARTBEAT_TIMEOUT_S,
            logger.info,
        )
        self._set_startup_phase(phase)
        if waited >= 1.0:
            logger.info("Waited %.1f s for every rank before %s.", waited, phase)

    def _roce_health_check(self) -> Callable[[], None] | None:
        """The RoCEnante health check of the TP communicator and PP P2P runtimes.

        Returns:
            The check callable, or None when neither is in use.
        """
        checks: list[Callable[[], None]] = []
        communicator = get_tp_group().device_communicator
        comm = getattr(communicator, "roce_ar_comm", None) or getattr(
            communicator, "b12x_ar_comm", None
        )
        tp_check = getattr(comm, "check_health", None)
        if tp_check is not None:
            checks.append(tp_check)
        for rt in self._pp_p2p_runtimes():
            pp_check = getattr(rt, "check_health", None)
            if pp_check is not None:
                checks.append(pp_check)
        if not checks:
            return None
        if len(checks) == 1:
            return checks[0]

        def _check_all() -> None:
            for fn in checks:
                fn()

        return _check_all

    def _roce_guarded(self, output):
        """Fail-stop RoCEnante check once the step's output is on the host.

        A RoCEnante wait that timed out records itself and freezes the runtime.
        A synchronous output already holds the sampled tokens on the host, so
        every collective of the step has completed and the check runs now; an
        asynchronous output is wrapped so the check runs right after its
        ``get_output()`` completes the copy.  Either way a failed collective's
        output never leaves the worker.  Every rank reaches the same state on
        its own (a stalled rank starves its peers' waits), so the raise is
        coordinated without a supervisor.  Two pinned-memory reads; no added
        synchronization.

        Args:
            output: The model runner's output for this step, possibly None.

        Returns:
            The same output, or a wrapper for an asynchronous output.

        Raises:
            RuntimeError: When a RoCEnante wait timed out or its proxy died.
        """
        check = self._b12x_roce_health_check()
        if check is None:
            return output
        if isinstance(output, AsyncModelRunnerOutput):
            return _B12xRoceCheckedAsyncOutput(output, check, self._roce_failure)
        check()
        return output

    def engine_grid(self, spec: str) -> dict:
        """The engine cost grid (``GPUModelRunner.engine_grid``) for the dev-mode
        collective RPC, whose arguments arrive as strings: ``spec`` is JSON."""
        import json

        grid = getattr(self.model_runner, "engine_grid", None)
        if grid is None:
            raise NotImplementedError("engine_grid needs the V2 GPU model runner")
        result = grid(json.loads(spec))
        result["rank"] = self.rank
        return result
