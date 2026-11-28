# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

import errno
import time
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.logger import init_logger
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
from vllm.v1.simple_kv_offload.disk_backend import DiskBackend, disk_slot_bytes
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        kv_offload_backend: str = "cpu",
        disk_path: str | None = None,
        disk_capacity_bytes: int = 0,
        disk_buffer_slots: int = 2,
        use_page_cache: bool = False,
        disk_io_timeout_s: float = 15.0,
        disk_io_slot_budget_s: float = 0.02,
        disk_checksum: str = "crc32",
        disk_preallocate: bool = True,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes
        self.disk_path = disk_path
        self.disk_capacity_bytes = disk_capacity_bytes
        self.disk_buffer_slots = disk_buffer_slots
        self.use_page_cache = use_page_cache
        self.disk_mode = kv_offload_backend == "disk"

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        self._backend: DmaCopyBackend | DiskBackend | None = None

        # Ordered (event_idx, Event). Events pre-allocated on main thread.
        self._load_events: list[tuple[int, torch.Event]] = []
        self._store_events: list[tuple[int, torch.Event]] = []
        # High-water marks: highest event_idx completed per stream.
        # When the event list is empty, the hwm covers all prior events.
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None
        self._store_submitted = False

        # Compute-done event recorded before each store; reused across steps
        # (get_finished runs once per step, copy queue is FIFO).
        self._store_compute_done: torch.Event | None = None

        # Pending event index sets, populated in bind_connector_metadata
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()
        # Completed store events to report via build_connector_worker_meta
        self._completed_store_events: dict[int, int] = {}

        # Disk fault handling (disk mode). A transfer under I/O that has made
        # no progress for disk_io_timeout_s (no slot completed since its IO
        # thread took it from the queue or since its last slot) is a fault:
        # it is abandoned and reported failed, and the tier on this rank
        # degrades; time in the queue does not count. So is every pending
        # transfer of an IO thread that is not running, and every transfer
        # still queued on a rank whose tier has a fault. A transfer that runs
        # longer than disk_io_timeout_s plus disk_io_slot_budget_s for each
        # of its slots, counted from its dequeue, while it still makes
        # progress, is an overrun: it is abandoned and reported failed with
        # no fault, and the queue behind it runs on. The IO thread's own
        # Python work (GIL handoffs) counts against that bound, so an overrun
        # is not a disk fault. No bound counts from submission. A failed load
        # is still reported finished, so all ranks release the request
        # together, and it is also reported in failed_recving, which the
        # scheduler turns into a cache miss (kv_load_failure_policy
        # "recompute"). A failed store is reported complete and failed, so
        # the scheduler drops its blocks instead of publishing them.
        self.disk_io_timeout_s = disk_io_timeout_s
        self.disk_io_slot_budget_s = disk_io_slot_budget_s
        self.disk_checksum = disk_checksum
        self.disk_preallocate = disk_preallocate
        self._forced_failed_loads: set[int] = set()
        # Load events abandoned at their total bound since the last worker
        # metadata, so the scheduler keeps the tier up for them.
        self._overrun_load_events: set[int] = set()
        self._failed_store_events: set[int] = set()
        self._fault_reported = False
        self._rank = 0

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register GPU KV caches and allocate pinned CPU tensors.
        The worker will infer the underlying raw storage from the kv_caches.

        Args:
            kv_caches: Per-layer GPU KV caches. Values are either a single
                tensor (attention layers) or a list of tensors (Mamba layers
                in hybrid models). All values are included for offloading
                by resolving to their underlying raw storage.
        """
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        self.device = next(iter(kv_caches.values())).device
        self._rank = self._global_rank()

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks
        assert self.kv_cache_config.kv_cache_tensors

        # The DMA backend copies blocks as base + block_id * stride(0), so every
        # region is a [num_blocks, block_bytes] view. Block bytes come from each
        # tensor's placement -- layers in one allocation can differ in page size.
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        seen: set[tuple[torch.device, int]] = set()
        for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
            name = next((n for n in kv_cache_tensor.layers if n in kv_caches), None)
            if name is None:
                continue
            num_layers = len(kv_cache_tensor.layers)
            block_bytes = kv_cache_tensor.block_stride
            blocks_span = num_blocks * block_bytes
            layer_outermost = kv_cache_tensor.layer_stride >= blocks_span
            if layer_outermost:
                # One region per layer, or per KV head group under LHBNC.
                start = kv_cache_tensor.offset
                span = num_layers * kv_cache_tensor.layer_stride
                layer_names = kv_cache_tensor.layers
            else:
                start = 0
                span = blocks_span
                layer_names = kv_cache_tensor.layers[:1]

            cache = kv_caches[name]
            raw = torch.empty(0, dtype=torch.int8, device=cache.device).set_(
                cache.untyped_storage()
            )
            assert raw.numel() >= start + span, (
                f"KV cache {name!r} storage has {raw.numel()} bytes, smaller than "
                f"the {span} bytes it places at offset {start}"
            )
            regions = raw[start : start + span].view(-1, num_blocks, block_bytes)
            groups_per_layer, remainder = divmod(len(regions), len(layer_names))
            assert remainder == 0, (
                f"KV cache {name!r} places {len(layer_names)} layers in "
                f"{len(regions)} block-strided regions"
            )
            for idx, region in enumerate(regions):
                key = (region.device, region.data_ptr())
                if key in seen:
                    continue
                seen.add(key)
                layer_name = layer_names[idx // groups_per_layer]
                key_name = (
                    layer_name
                    if groups_per_layer == 1
                    else f"{layer_name}.{idx % groups_per_layer}"
                )
                unique_gpu_caches[key_name] = region

        # Compute per-tensor bytes_per_block. Tensors may have different
        # page_size_bytes (e.g., UniformTypeKVCacheSpecs with varying head_size).
        per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in unique_gpu_caches.values()
        ]
        total_bytes_per_block = sum(per_tensor_bpb)

        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)

        # Use lowest priority so KV cache I/O yields to compute streams.
        low_pri, _ = torch.cuda.Stream.priority_range()
        self.load_stream = torch.cuda.Stream(priority=low_pri)
        self.store_stream = torch.cuda.Stream(priority=low_pri)

        self.gpu_kv_caches = unique_gpu_caches

        if self.disk_mode:
            self._init_disk_mode(unique_gpu_caches, total_bytes_per_block, self.device)
        else:
            self._init_cpu_mode(unique_gpu_caches, total_bytes_per_block, self.device)

    @staticmethod
    def _global_rank() -> int:
        try:
            from vllm.distributed.parallel_state import get_world_group

            return get_world_group().rank
        except Exception:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank()
            return 0

    def _init_disk_mode(
        self,
        unique_gpu_caches: dict[str, torch.Tensor],
        total_bytes_per_block: int,
        device: torch.device,
    ) -> None:
        # Must match SimpleCPUOffloadScheduler's slot count: both divide the
        # capacity by the padded slot size (disk_slot_bytes).
        slot_bytes = disk_slot_bytes(total_bytes_per_block, self.use_page_cache)
        num_disk_slots = max(1, self.disk_capacity_bytes // slot_bytes)
        self.num_cpu_blocks = num_disk_slots

        logger.info(
            "SimpleCPUOffloadWorker [DISK]: %d tensors, %d disk slots (%.2f GB), "
            "block %d bytes in %d-byte slots",
            len(unique_gpu_caches),
            num_disk_slots,
            (num_disk_slots * slot_bytes) / (1024**3),
            total_bytes_per_block,
            slot_bytes,
        )
        # One line a rank, so a boot log shows the bounds of every rank.
        logger.info(
            "SimpleCPUOffloadWorker [DISK]: disk_io_timeout_s=%g (no progress: "
            "a fault), disk_io_slot_budget_s=%g (past the total bound: a miss), "
            "rank %d",
            self.disk_io_timeout_s,
            self.disk_io_slot_budget_s,
            self._rank,
        )

        assert self.disk_path is not None
        rank_path = f"{self.disk_path}.rank_{device.index or 0}"
        self._backend = DiskBackend()
        self._backend.init(
            unique_gpu_caches,
            device,
            self.load_stream,
            self.store_stream,
            rank_path,
            num_disk_slots,
            total_bytes_per_block,
            self.disk_buffer_slots,
            self.use_page_cache,
            checksum=self.disk_checksum,
            preallocate=self.disk_preallocate,
            rank=self._rank,
        )

    def _init_cpu_mode(
        self,
        unique_gpu_caches: dict[str, torch.Tensor],
        total_bytes_per_block: int,
        device: torch.device,
    ) -> None:
        logger.info(
            "SimpleCPUOffloadWorker [CPU]: %d tensors, %d CPU blocks (%.2f GB)",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

        pin_memory = PIN_MEMORY
        if not pin_memory:
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        self.cpu_kv_caches = {}
        for name, gpu_tensor in unique_gpu_caches.items():
            cpu_shape = (self.num_cpu_blocks,) + gpu_tensor.shape[1:]
            # Allocate non-pinned first, then pin via cudaHostRegister to
            # bypass PyTorch's CUDACachingHostAllocator which rounds up to
            # the next power of 2 (e.g. 100 GB -> 128 GB).
            tensor = torch.zeros(cpu_shape, dtype=gpu_tensor.dtype, device="cpu")
            if pin_memory:
                pin_tensor(tensor)
            self.cpu_kv_caches[name] = tensor

        self._backend = DmaCopyBackend()
        self._backend.init(
            unique_gpu_caches,
            self.cpu_kv_caches,
            device,
            self.load_stream,
            self.store_stream,
        )

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        self._store_submitted = False
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        # No-forward steps skip the normal wait_for_save hook.
        self.wait_for_save()
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        # Invoked after the forward launch on steps without sync loads
        # (SchedulerOutput.has_sync_kv_loads), so the CPU-side block copy
        # op overhead (~5ms) stays hidden behind GPU compute. Stores are
        # issued in wait_for_save().
        metadata = self._connector_metadata
        if metadata is not None and metadata.load_cpu_blocks:
            backend = self._backend
            assert backend is not None
            keys = (
                {"keys": metadata.load_keys or None}
                if isinstance(backend, DiskBackend)
                else {}
            )
            backend.launch_copy(
                metadata.load_cpu_blocks,
                metadata.load_gpu_blocks,
                is_store=False,
                event_idx=metadata.load_event,
                events_list=self._load_events,
                **keys,
            )

    def wait_for_save(self) -> None:
        """Submit async stores.

        Stores (GPU->CPU) read the live KV cache, which the compute stream may
        still be writing under v1 overlapped execution, so they are ordered
        after a compute-done event recorded on the current stream. See
        #45704 for the bug and #39306 for the srcAccessOrder rationale.
        """
        metadata = self._connector_metadata
        if (
            metadata is not None
            and metadata.store_gpu_blocks
            and not self._store_submitted
        ):
            backend = self._backend
            assert backend is not None
            if self._store_compute_done is None:
                self._store_compute_done = torch.Event()
            self._store_compute_done.record(torch.cuda.current_stream())
            keys = (
                {"keys": metadata.store_keys or None}
                if isinstance(backend, DiskBackend)
                else {}
            )
            backend.launch_copy(
                metadata.store_gpu_blocks,
                metadata.store_cpu_blocks,
                is_store=True,
                event_idx=metadata.store_event,
                events_list=self._store_events,
                wait_event=self._store_compute_done,
                **keys,
            )
            self._store_submitted = True

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report completed transfer events to the scheduler.

        Loads are submitted in start_load_kv() and stores in wait_for_save().

        Returns:
            tuple of (finished_sending, finished_recving).
            - finished_sending: always None (stores use worker metadata).
            - finished_recving: req_ids whose loads have completed, failed
              loads included (see get_transfer_results).
        """
        results = self.get_transfer_results(finished_req_ids)
        return None, results.finished_recving or None

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        """Completed loads, with the failed ones also in failed_recving."""
        metadata = self._connector_metadata
        finished_recving: set[str] = set()
        failed_recving: set[str] = set()
        self._check_overdue_transfers()

        def reqs_of(j: int) -> list[str]:
            return (
                (metadata.load_event_to_reqs.get(j) or [])
                if metadata is not None
                else []
            )

        if self._pending_load_event_indices:
            load_wm = self._poll_stream_events(is_store=False)
            for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
                self._pending_load_event_indices.discard(j)
                req_ids = reqs_of(j)
                finished_recving.update(req_ids)
                backend = self._backend
                if isinstance(backend, DiskBackend) and backend.take_failed(False, j):
                    failed_recving.update(req_ids)
        for j in self._forced_failed_loads:
            req_ids = reqs_of(j)
            finished_recving.update(req_ids)
            failed_recving.update(req_ids)
        self._forced_failed_loads.clear()

        if self._pending_store_event_indices:
            store_wm = self._poll_stream_events(is_store=True)
            for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
                self._pending_store_event_indices.discard(j)
                self._completed_store_events[j] = 1
                backend = self._backend
                if isinstance(backend, DiskBackend) and backend.take_failed(True, j):
                    self._failed_store_events.add(j)

        return KVConnectorTransferResults(
            finished_recving=finished_recving, failed_recving=failed_recving
        )

    def _check_overdue_transfers(self) -> None:
        """Abandon disk transfers that stopped making progress or that ran
        past their total bound.

        No progress: a transfer under I/O with no completed slot for
        disk_io_timeout_s, counted from the moment its IO thread took it from
        the queue or from its last completed slot, is a fault. It is reported
        failed at the first get_finished call after that, the tier on this
        rank degrades, and every transfer still queued fails at once (its
        thread may be blocked in a hung read). The check also covers a
        transfer already abandoned at its total bound: its thread runs
        nothing else until it returns. Time in the queue does not count, so a
        burst of loads queued behind each other does not fail.

        Total bound: a transfer that keeps completing slots but runs longer
        than disk_io_timeout_s plus disk_io_slot_budget_s for each of its
        slots, from the moment it left the queue, is an overrun. It is
        reported failed (a load becomes a cache miss, a store is dropped)
        with no fault: its thread is alive and moves on to the next transfer
        at its next slot, and the tier stays up. The IO thread's Python work
        counts against the bound, so a busy GIL alone can cause an overrun.

        Every pending transfer fails at once, as a fault, when its IO thread
        is not running. Abandoned copies stop before their next DMA
        (DiskBackend.abandon); copies already queued on the load stream are
        drained before the request's blocks are released.
        """
        backend = self._backend
        if not isinstance(backend, DiskBackend):
            return
        now = time.monotonic()
        for is_store, pending in (
            (False, self._pending_load_event_indices),
            (True, self._pending_store_event_indices),
        ):
            if not pending:
                continue
            op = "store" if is_store else "load"
            overdue: list[int] = []
            if not backend.thread_alive(is_store):
                overdue = list(pending)
                backend.record_fault(op, -1, errno.EPIPE, f"the {op} thread stopped")
            else:
                running = backend.running(is_store)
                if running is not None:
                    elapsed = now - running.started_at
                    bound = (
                        self.disk_io_timeout_s
                        + running.slots * self.disk_io_slot_budget_s
                    )
                    if now - running.progressed_at > self.disk_io_timeout_s:
                        if running.event_idx in pending:
                            overdue.append(running.event_idx)
                        backend.record_fault(
                            op,
                            backend.current_slot(is_store),
                            errno.ETIMEDOUT,
                            f"{op} event {running.event_idx} made no progress "
                            f"for {self.disk_io_timeout_s:g} s (I/O timeout)",
                        )
                    elif running.event_idx in pending and elapsed > bound:
                        overdue.append(running.event_idx)
                        if not is_store:
                            self._overrun_load_events.add(running.event_idx)
                        logger.warning(
                            "SimpleCPUOffloadWorker [DISK]: rank %d %s event %d "
                            "ran past its %g s bound (%g s + %d slots x %g s), "
                            "%d of %d slots in %.1f s; reported failed, the tier "
                            "stays up",
                            self._rank,
                            op,
                            running.event_idx,
                            bound,
                            self.disk_io_timeout_s,
                            running.slots,
                            self.disk_io_slot_budget_s,
                            running.done,
                            running.slots,
                            elapsed,
                        )
                if backend.fault is not None:
                    overdue.extend(
                        j
                        for j in pending
                        if j not in overdue and not backend.started(is_store, j)
                    )
            if not overdue:
                continue
            for j in overdue:
                backend.abandon(is_store, j)
            if not is_store and self.load_stream is not None:
                # No DMA of an abandoned load starts after abandon(); finish
                # the ones already queued before the blocks are released.
                self.load_stream.synchronize()
            for j in overdue:
                pending.discard(j)
                if is_store:
                    self._completed_store_events[j] = 1
                    self._failed_store_events.add(j)
                else:
                    self._forced_failed_loads.add(j)

    def build_connector_worker_meta(self) -> SimpleCPUOffloadWorkerMetadata | None:
        """Return completed store events since the last call, the failed ones
        among them, the load events abandoned at their total bound, and this
        rank's first disk fault once."""
        faults: list[tuple[int, str, int, int, str]] = []
        backend = self._backend
        if (
            isinstance(backend, DiskBackend)
            and backend.fault is not None
            and not self._fault_reported
        ):
            fault = backend.fault
            faults.append(
                (self._rank, fault.op, fault.slot, fault.errno, fault.message)
            )
            self._fault_reported = True
        if (
            not self._completed_store_events
            and not faults
            and not self._overrun_load_events
        ):
            return None
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
            failed_store_events=self._failed_store_events,
            faults=faults,
            overrun_load_events=self._overrun_load_events,
        )
        self._completed_store_events = {}
        self._failed_store_events = set()
        self._overrun_load_events = set()
        return meta

    def handle_preemptions(
        self, kv_connector_metadata: SimpleCPUOffloadMetadata
    ) -> None:
        """Sync all in-flight transfers before preempted blocks are reused."""
        if not kv_connector_metadata.need_flush:
            return
        self._flush_and_sync_all()

    def _flush_and_sync_all(self) -> None:
        """Synchronize all in-flight transfer events."""
        for event_idx, event in self._load_events:
            event.synchronize()
            self._load_hwm = event_idx
        self._load_events.clear()

        for event_idx, event in self._store_events:
            event.synchronize()
            self._store_hwm = event_idx
        self._store_events.clear()

    def _poll_stream_events(self, is_store: bool) -> int:
        """Non-blocking poll for completed events and return the high-water mark."""
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            event_idx, event = events[0]
            if not event.query():
                break
            hwm = event_idx
            events.pop(0)
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm
