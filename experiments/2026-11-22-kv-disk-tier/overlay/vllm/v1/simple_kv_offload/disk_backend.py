# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk I/O backend for GPU<->NVMe block transfers via pinned staging buffers.

Uses separate IO threads for store and load so that loads (latency-critical)
never block behind stores (background work). Each thread owns its own pinned
staging buffers to avoid contention.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import queue
import struct
import threading
import time
import zlib
from typing import NamedTuple

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
    pin_tensor,
)

logger = init_logger(__name__)

O_DIRECT = getattr(os, "O_DIRECT", 0)
_ALIGNMENT = 4096
# Debug only, off by default: "<read>:<seconds>[:<rank>]" makes the <read>-th
# slot read of this process (from 1) sleep <seconds> before its preadv, on
# every rank or on <rank> only. Read once at init. It stands in for a hung
# read in tests and in one window row.
DEBUG_READ_DELAY_ENV = "VLLM_KVTIER_DEBUG_READ_DELAY"


def parse_debug_read_delay(value: str | None, rank: int) -> tuple[int, float] | None:
    """(read number, seconds) of the debug read delay for this rank, or None."""
    if not value:
        return None
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"{DEBUG_READ_DELAY_ENV}={value!r}: expected <read>:<seconds>[:<rank>]"
        )
    read, seconds = int(parts[0]), float(parts[1])
    if read < 1 or seconds <= 0:
        raise ValueError(f"{DEBUG_READ_DELAY_ENV}={value!r}: read >= 1, seconds > 0")
    if len(parts) == 3 and int(parts[2]) != rank:
        return None
    return read, seconds

# Slot header, written into the slot's padding after the block bytes:
# magic, version, checksum kind, slot id, generation, block key, checksum.
# The slot id catches a misdirected read, the generation (the slot's store
# count on this rank) a lost or stale write, the block key (slot_key of the
# block hash that the scheduler stored or expects, sent with every store and
# load) a slot rewritten for another block between the scheduler's lookup and
# the load, and the checksum corrupted or torn bytes.
_HEADER = struct.Struct("<4sHHQQQQ")
_HEADER_MAGIC = b"KVT1"
_HEADER_VERSION = 2
CHECKSUMS = {"none": 0, "crc32": 1, "sum64": 2}


def slot_key(block_hash: bytes | None) -> int:
    """The 8-byte identity of a block in its disk slot header.

    A BLAKE2b digest of the whole BlockHashWithGroupId, not a prefix of it:
    a group whose block spans two hash blocks has a hash that starts with the
    hash of a smaller group's block at the same position, and the group id
    sits in the last 4 bytes. 0 means no block (never a real digest in
    practice: one chance in 2**64).
    """
    if block_hash is None:
        return 0
    digest = hashlib.blake2b(block_hash, digest_size=8).digest()
    return int.from_bytes(digest, "little")


class DiskTierError(OSError):
    """An I/O fault on one tier slot: the errno, the message and the slot."""

    def __init__(self, err: int, message: str, slot: int = -1) -> None:
        super().__init__(err, message)
        self.slot = slot


class _Abandoned(Exception):
    """The worker gave up on this transfer (I/O timeout); skip its copies."""


class RunningTransfer(NamedTuple):
    """The transfer under I/O on one IO thread, for the worker's I/O timeout."""

    event_idx: int
    # time.monotonic() when the IO thread took the transfer from its queue
    started_at: float
    # time.monotonic() of the last completed slot, or started_at before one
    progressed_at: float
    # slots the transfer reads or writes
    slots: int
    # slots completed so far
    done: int = 0


class DiskFault(NamedTuple):
    """The first fault of a backend: once set, the tier on this rank is degraded."""

    op: str  # "load" or "store"
    slot: int  # disk slot of the failed I/O, -1 when unknown
    errno: int
    message: str


def disk_slot_bytes(block_bytes: int, use_page_cache: bool = False) -> int:
    """Bytes one offload block occupies in the disk file and in a staging slot.

    O_DIRECT needs every file offset, iovec length and buffer address aligned
    to the device's logical block (4096 covers 512e and 4Kn drives). A block
    whose bytes are not a multiple of 4096 is padded up to the next multiple,
    so every slot starts aligned and moves in one aligned I/O. Page-cache I/O
    has no such rule and keeps the exact block size.
    """
    if use_page_cache:
        return block_bytes
    return -(-block_bytes // _ALIGNMENT) * _ALIGNMENT


def _alloc_aligned(num_slots: int, bpb: int) -> torch.Tensor:
    """Allocate a staging buffer whose base address is O_DIRECT aligned.

    The CPU allocator only guarantees 64-byte alignment, so over-allocate by
    one alignment unit and return an aligned view. The view keeps the backing
    storage alive.
    """
    nbytes = num_slots * bpb
    raw = torch.zeros(nbytes + _ALIGNMENT, dtype=torch.int8, device="cpu")
    offset = -raw.data_ptr() % _ALIGNMENT
    return raw[offset : offset + nbytes].view(num_slots, bpb)


class DiskBackend:
    """Async disk offload backend with pipelined GPU DMA and interleaved IO.

    Architecture:
    - Separate coordinator threads for store and load (never block each other)
    - Interleaved pipeline: DMA slot N while preadv/pwritev slot N-1
    - O_DIRECT by default; page cache is opt-in via use_page_cache

    Same launch_copy interface as DmaCopyBackend so the worker can swap
    backends without changing calling code.
    """

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._store_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._load_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._store_thread: threading.Thread | None = None
        self._load_thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._fd: int = -1
        self._disk_path: str = ""
        self._total_block_bytes: int = 0
        # Bytes of one block on disk and in one staging slot: the block's
        # bytes, padded to 4096 under O_DIRECT (see disk_slot_bytes).
        self._slot_bytes: int = 0
        # One [num_buffer_slots, slot_bytes] buffer per direction. A slot
        # holds the tensors' bytes back to back, then the padding.
        self._store_buffer: torch.Tensor | None = None
        self._load_buffer: torch.Tensor | None = None
        self._store_slot_views: list[list[memoryview]] = []
        self._load_slot_views: list[list[memoryview]] = []
        # Per staging slot, the DMA parameters that address that slot as
        # block 0, so the slot can start on any aligned address.
        self._store_slot_params: list[BatchMemcpyParams] = []
        self._load_slot_params: list[BatchMemcpyParams] = []
        self._per_tensor_bpb: list[int] = []
        self._tensor_names: list[str] = []
        # Fault state. The first fault degrades the tier on this rank: every
        # transfer queued after it fails at once without I/O. Failed event ids
        # are recorded before the event is published, so the worker sees the
        # failure no later than the completion. Keyed by is_store.
        self._fault: DiskFault | None = None
        self._failed_events: dict[bool, set[int]] = {True: set(), False: set()}
        # Events the worker abandoned after its I/O timeout. The IO threads
        # check this under the lock before every DMA, so no copy of an
        # abandoned event starts after abandon() returns. The worker reports
        # an abandoned event at the abandon, so the IO thread drops both
        # marks of the event once the thread is done with the event.
        self._abandoned: dict[bool, set[int]] = {True: set(), False: set()}
        self._abandon_lock = threading.Lock()
        # Slot under I/O in each thread, for the fault report of a timeout.
        self._current_slot: dict[bool, int] = {True: -1, False: -1}
        # Progress of each IO thread, for the worker's I/O timeout. The clock
        # of a transfer starts when its thread takes it from the queue, so
        # time in the queue never counts. _running holds the transfer under
        # I/O: its event, the time of its start, the time of its last
        # completed slot and its slot count; _started is the highest event
        # index taken from the queue (events leave each queue in index order).
        self._running: dict[bool, RunningTransfer | None] = {
            True: None,
            False: None,
        }
        self._started: dict[bool, int] = {True: -1, False: -1}
        self._debug_read_delay: tuple[int, float] | None = None
        self._reads = 0
        # Slot integrity (see _HEADER): kind 0 is off, the header with it.
        self._checksum_kind = 0
        self._generation: np.ndarray | None = None
        self._store_rows: list[np.ndarray] = []
        self._load_rows: list[np.ndarray] = []

    # --- fault state (thread-safe) ---

    @property
    def fault(self) -> DiskFault | None:
        return self._fault

    def record_fault(self, op: str, slot: int, err: int, message: str) -> None:
        with self._abandon_lock:
            if self._fault is None:
                self._fault = DiskFault(op, slot, err, message)

    def take_failed(self, is_store: bool, event_idx: int) -> bool:
        """Whether a completed event failed; forgets it."""
        failed = self._failed_events[is_store]
        if event_idx in failed:
            failed.discard(event_idx)
            return True
        return False

    def abandon(self, is_store: bool, event_idx: int) -> None:
        """Stop all further copies of an event (the worker's I/O timeout)."""
        with self._abandon_lock:
            self._abandoned[is_store].add(event_idx)

    def _forget_if_abandoned(self, is_store: bool, event_idx: int) -> None:
        """Drop the abandon mark and the failure mark of an event that the
        worker abandoned: the worker reported that event as failed at the
        abandon and never asks for the event again, and event indices never
        repeat. Called by the IO thread when the thread is done with the
        event."""
        with self._abandon_lock:
            if event_idx in self._abandoned[is_store]:
                self._abandoned[is_store].discard(event_idx)
                self._failed_events[is_store].discard(event_idx)

    def current_slot(self, is_store: bool) -> int:
        return self._current_slot[is_store]

    def thread_alive(self, is_store: bool) -> bool:
        thread = self._store_thread if is_store else self._load_thread
        return thread is not None and thread.is_alive()

    def started(self, is_store: bool, event_idx: int) -> bool:
        """Whether the IO thread has taken this event from its queue."""
        return event_idx <= self._started[is_store]

    def running(self, is_store: bool) -> RunningTransfer | None:
        """The transfer under I/O on this thread, or None."""
        return self._running[is_store]

    def _begin(self, is_store: bool, event_idx: int, slots: int) -> None:
        self._started[is_store] = max(self._started[is_store], event_idx)
        now = time.monotonic()
        self._running[is_store] = RunningTransfer(event_idx, now, now, slots)

    def _progress(self, is_store: bool) -> None:
        running = self._running[is_store]
        if running is not None:
            self._running[is_store] = running._replace(
                progressed_at=time.monotonic(), done=running.done + 1
            )

    def _copy_unless_abandoned(
        self,
        is_store: bool,
        event_idx: int,
        src_blocks: list[int],
        dst_blocks: list[int],
        params: BatchMemcpyParams,
        stream: torch.cuda.Stream,
    ) -> torch.Event:
        with self._abandon_lock:
            if event_idx in self._abandoned[is_store]:
                raise _Abandoned()
            copy_blocks(src_blocks, dst_blocks, params)
            ev = torch.Event()
            ev.record(stream)
        return ev

    def _run_transfer(
        self,
        is_store: bool,
        src_blocks: list[int],
        dst_blocks: list[int],
        stream: torch.cuda.Stream,
        event_idx: int,
        keys: list[int] | None = None,
    ) -> None:
        """One queued transfer. Never raises: a fault marks the event failed."""
        if self._fault is not None:
            # Degraded: no I/O. The event still completes, as a failure.
            self._failed_events[is_store].add(event_idx)
            self._running[is_store] = None
            self._forget_if_abandoned(is_store, event_idx)
            return
        try:
            if is_store:
                self._do_store(src_blocks, dst_blocks, stream, event_idx, keys)
            else:
                self._do_load(src_blocks, dst_blocks, stream, event_idx, keys)
        except _Abandoned:
            self._failed_events[is_store].add(event_idx)
        except Exception as e:  # OSError, a CUDA RuntimeError, a bad header
            self._failed_events[is_store].add(event_idx)
            self.record_fault(
                "store" if is_store else "load",
                getattr(e, "slot", self._current_slot[is_store]),
                getattr(e, "errno", None) or errno.EIO,
                f"{type(e).__name__}: {e}",
            )
        finally:
            self._current_slot[is_store] = -1
            self._running[is_store] = None
            self._forget_if_abandoned(is_store, event_idx)

    def _slot_tensor_views(
        self, buffer: torch.Tensor, slot: int
    ) -> dict[str, torch.Tensor]:
        """[1, bpb] byte views of each tensor's part of one staging slot."""
        row = buffer[slot]
        views: dict[str, torch.Tensor] = {}
        offset = 0
        for name, bpb in zip(self._tensor_names, self._per_tensor_bpb):
            views[name] = row[offset : offset + bpb].view(1, bpb)
            offset += bpb
        return views

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        disk_path: str,
        num_disk_slots: int,
        total_block_bytes: int,
        num_buffer_slots: int = 2,
        use_page_cache: bool = False,
        checksum: str = "none",
        preallocate: bool = False,
        rank: int = 0,
    ) -> None:
        self._load_stream = load_stream
        self._debug_read_delay = parse_debug_read_delay(
            os.environ.get(DEBUG_READ_DELAY_ENV), rank
        )
        if self._debug_read_delay is not None:
            logger.warning(
                "DiskBackend: debug read delay on (%s): read %d sleeps %.1f s",
                DEBUG_READ_DELAY_ENV,
                *self._debug_read_delay,
            )
        self._store_stream = store_stream
        self._total_block_bytes = total_block_bytes
        self._num_buffer_slots = num_buffer_slots
        self._tensor_names = list(gpu_caches.keys())
        self._per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in gpu_caches.values()
        ]
        assert sum(self._per_tensor_bpb) == total_block_bytes, (
            f"tensor block bytes {self._per_tensor_bpb} do not sum to "
            f"total_block_bytes={total_block_bytes}"
        )
        # O_DIRECT pads each slot to 4096; page-cache I/O keeps the exact size.
        self._slot_bytes = disk_slot_bytes(total_block_bytes, use_page_cache)

        # Separate buffer pools for store and load threads. Each slot is one
        # contiguous row of slot_bytes, so slot starts stay 4096-aligned and a
        # slot moves to or from disk in a single aligned iovec.
        self._store_buffer = _alloc_aligned(num_buffer_slots, self._slot_bytes)
        pin_tensor(self._store_buffer)
        self._load_buffer = _alloc_aligned(num_buffer_slots, self._slot_bytes)
        pin_tensor(self._load_buffer)

        # Pre-built iovec views per slot (avoid per-transfer .numpy() calls)
        self._store_slot_views = [
            [memoryview(self._store_buffer[slot].numpy())]
            for slot in range(num_buffer_slots)
        ]
        self._load_slot_views = [
            [memoryview(self._load_buffer[slot].numpy())]
            for slot in range(num_buffer_slots)
        ]

        self._store_slot_params = [
            build_params(
                gpu_caches,
                self._slot_tensor_views(self._store_buffer, slot),
                store_stream,
                src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
            )
            for slot in range(num_buffer_slots)
        ]
        self._load_slot_params = [
            build_params(
                self._slot_tensor_views(self._load_buffer, slot),
                gpu_caches,
                load_stream,
                src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
            )
            for slot in range(num_buffer_slots)
        ]
        self._store_params = self._store_slot_params[0]
        self._load_params = self._load_slot_params[0]

        if checksum not in CHECKSUMS:
            raise ValueError(f"disk_checksum {checksum!r}: expected {list(CHECKSUMS)}")
        kind = CHECKSUMS[checksum]
        if kind and self._slot_bytes - total_block_bytes < _HEADER.size:
            logger.warning(
                "DiskBackend: no room for the %d-byte slot header in %d-byte "
                "slots of %d-byte blocks; slot integrity checks are off",
                _HEADER.size,
                self._slot_bytes,
                total_block_bytes,
            )
            kind = 0
        if kind == CHECKSUMS["sum64"] and total_block_bytes % 8:
            kind = CHECKSUMS["crc32"]
        self._checksum_kind = kind
        if kind:
            self._generation = np.zeros(num_disk_slots, dtype=np.uint64)
            self._store_rows = [
                self._store_buffer[slot].numpy().view(np.uint8)
                for slot in range(num_buffer_slots)
            ]
            self._load_rows = [
                self._load_buffer[slot].numpy().view(np.uint8)
                for slot in range(num_buffer_slots)
            ]

        os.makedirs(os.path.dirname(disk_path) or ".", exist_ok=True)
        # Slot contents never outlive the process, so unlink then O_EXCL rather
        # than reopening: a pre-existing file would otherwise keep its own
        # (possibly world-readable) mode, and blocks may encode user prompts.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(disk_path)
        # O_DIRECT by default: page cache would consume the very host DRAM this
        # backend exists to conserve, and doubles the copy on the store path.
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if not use_page_cache:
            flags |= O_DIRECT
        self._fd = os.open(disk_path, flags, 0o600)
        self._disk_path = disk_path
        size = num_disk_slots * self._slot_bytes
        if preallocate:
            # Reserve every slot now: a sparse file on a filling disk fails
            # hours later, as a write error in the middle of serving. On ext4
            # and xfs this allocates unwritten extents (reads return zeros)
            # without writing the data.
            try:
                os.posix_fallocate(self._fd, 0, size)
            except OSError as e:
                os.close(self._fd)
                self._fd = -1
                with contextlib.suppress(OSError):
                    os.unlink(disk_path)
                raise OSError(
                    e.errno,
                    f"DiskBackend: cannot reserve {size / (1024**3):.2f} GiB for "
                    f"{disk_path}: {e.strerror}; free space on that filesystem or "
                    "lower disk_capacity_bytes",
                ) from e
        else:
            os.ftruncate(self._fd, size)

        logger.info(
            "DiskBackend: path=%s, slots=%d, total=%.2f GB, buf=%dx%d bytes, "
            "block=%d bytes (page_cache=%s)",
            disk_path,
            num_disk_slots,
            (num_disk_slots * self._slot_bytes) / (1024**3),
            num_buffer_slots,
            self._slot_bytes,
            total_block_bytes,
            use_page_cache,
        )

        self._store_thread = threading.Thread(
            target=self._store_loop,
            args=(device, store_stream),
            daemon=True,
        )
        self._load_thread = threading.Thread(
            target=self._load_loop,
            args=(device, load_stream),
            daemon=True,
        )
        self._store_thread.start()
        self._load_thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
        keys: list[int] | None = None,
    ) -> None:
        """Queue a transfer. ``keys`` holds the block key (slot_key) of each
        disk slot of the transfer, in order: written into the header by a
        store, compared with the header by a load. None means key 0."""
        q = self._store_queue if is_store else self._load_queue
        slots = dst_blocks if is_store else src_blocks
        if keys is None:
            keys = [0] * len(slots)
        assert len(keys) == len(slots), (len(keys), len(slots))
        q.put((src_blocks, dst_blocks, event_idx, events_list, wait_event, keys))

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._store_queue.put(None)
        self._load_queue.put(None)
        if self._store_thread is not None:
            self._store_thread.join(timeout=10.0)
        if self._load_thread is not None:
            self._load_thread.join(timeout=10.0)
        if self._fd < 0:
            return
        # Slot contents can encode user prompts, so drop the name now rather
        # than leaving them readable until the next run overwrites the file.
        # Unlinking only removes the directory entry: any thread still holding
        # the fd keeps writing to the (now anonymous) inode, which the kernel
        # frees once the last fd goes away.
        with contextlib.suppress(OSError):
            os.unlink(self._disk_path)
        # Closing under a still-running IO thread would let the fd number be
        # reused by an unrelated open(), turning its next pwritev into a write
        # into that file. Leaking one fd for the remaining process lifetime is
        # the cheaper failure.
        if any(
            t is not None and t.is_alive()
            for t in (self._store_thread, self._load_thread)
        ):
            logger.warning(
                "IO thread still running after shutdown timeout; leaking fd %d",
                self._fd,
            )
            return
        os.close(self._fd)
        self._fd = -1

    def _store_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._store_queue.get()
            if item is None:
                return
            (src_blocks, dst_blocks, event_idx, events_list, wait_event, keys) = item
            self._begin(True, event_idx, len(dst_blocks))
            if wait_event is not None:
                stream.wait_event(wait_event)
            self._run_transfer(True, src_blocks, dst_blocks, stream, event_idx, keys)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))

    def _checksum(self, data: np.ndarray) -> int:
        if self._checksum_kind == CHECKSUMS["crc32"]:
            return zlib.crc32(memoryview(data))
        return int(data.view(np.uint64).sum(dtype=np.uint64))

    def _seal_slot(self, buf_slot: int, slot: int, key: int) -> int:
        """Write the header of a staged block; returns its generation."""
        assert self._generation is not None
        row = self._store_rows[buf_slot]
        n = self._total_block_bytes
        generation = int(self._generation[slot]) + 1
        _HEADER.pack_into(
            row,
            n,
            _HEADER_MAGIC,
            _HEADER_VERSION,
            self._checksum_kind,
            slot,
            generation,
            key,
            self._checksum(row[:n]),
        )
        return generation

    def _verify_slot(self, buf_slot: int, slot: int, expected_key: int) -> None:
        """Check a read slot's header and checksum before any DMA."""
        assert self._generation is not None
        row = self._load_rows[buf_slot]
        n = self._total_block_bytes
        magic, version, kind, got_slot, generation, key, checksum = (
            _HEADER.unpack_from(row, n)
        )
        expected_generation = int(self._generation[slot])
        problem = None
        if magic != _HEADER_MAGIC or version != _HEADER_VERSION:
            problem = "no slot header"
        elif kind != self._checksum_kind:
            problem = f"checksum kind {kind}, expected {self._checksum_kind}"
        elif got_slot != slot:
            problem = f"header of slot {got_slot}"
        elif generation != expected_generation:
            problem = f"generation {generation}, expected {expected_generation}"
        elif key != expected_key:
            problem = f"block key {key:016x}, expected {expected_key:016x}"
        elif checksum != self._checksum(row[:n]):
            problem = "checksum mismatch"
        if problem is not None:
            raise DiskTierError(
                errno.EBADMSG,
                f"slot {slot} failed its integrity check: {problem}",
                slot,
            )

    def _writev_slot(
        self, buf_slot: int, file_offset: int, slot: int = -1, key: int = 0
    ) -> None:
        self._current_slot[True] = slot
        generation = self._seal_slot(buf_slot, slot, key) if self._checksum_kind else 0
        try:
            written = os.pwritev(
                self._fd, self._store_slot_views[buf_slot], file_offset
            )
        except OSError as e:
            raise DiskTierError(
                e.errno or errno.EIO, f"write of slot {slot}: {e.strerror}", slot
            ) from e
        if written < self._slot_bytes:
            raise DiskTierError(
                errno.EIO,
                f"short write of slot {slot}: {written} of {self._slot_bytes} bytes",
                slot,
            )
        if self._generation is not None:
            self._generation[slot] = generation
        self._progress(True)

    def _readv_slot(
        self, buf_slot: int, file_offset: int, slot: int = -1, key: int = 0
    ) -> None:
        self._current_slot[False] = slot
        self._reads += 1
        if self._debug_read_delay and self._reads == self._debug_read_delay[0]:
            time.sleep(self._debug_read_delay[1])
        try:
            bytes_read = os.preadv(
                self._fd, self._load_slot_views[buf_slot], file_offset
            )
        except OSError as e:
            raise DiskTierError(
                e.errno or errno.EIO, f"read of slot {slot}: {e.strerror}", slot
            ) from e
        if bytes_read < self._slot_bytes:
            raise DiskTierError(
                errno.EIO,
                f"short read of slot {slot}: {bytes_read} of {self._slot_bytes} bytes",
                slot,
            )
        if self._checksum_kind:
            self._verify_slot(buf_slot, slot, key)
        self._progress(False)

    def _load_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._load_queue.get()
            if item is None:
                return
            (src_blocks, dst_blocks, event_idx, events_list, wait_event, keys) = item
            self._begin(False, event_idx, len(src_blocks))
            if wait_event is not None:
                stream.wait_event(wait_event)
            self._run_transfer(False, src_blocks, dst_blocks, stream, event_idx, keys)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))

    def _do_store(
        self,
        gpu_blocks: list[int],
        disk_slots: list[int],
        stream: torch.cuda.Stream,
        event_idx: int = -1,
        keys: list[int] | None = None,
    ) -> None:
        """GPU -> buffer (DMA) -> disk (pwritev), interleaved double-buffer."""
        assert self._store_slot_params
        n = self._num_buffer_slots
        if keys is None:
            keys = [0] * len(disk_slots)
        # (DMA event, disk slot, block key) of the block staged in each slot.
        pending: list[tuple[torch.Event, int, int] | None] = [None] * n

        for i, (gpu_blk, disk_slot) in enumerate(zip(gpu_blocks, disk_slots)):
            buf_slot = i % n
            prev = pending[buf_slot]
            if prev is not None:
                prev[0].synchronize()
                self._writev_slot(
                    buf_slot, prev[1] * self._slot_bytes, prev[1], prev[2]
                )

            ev = self._copy_unless_abandoned(
                True,
                event_idx,
                [gpu_blk],
                [0],
                self._store_slot_params[buf_slot],
                stream,
            )
            pending[buf_slot] = (ev, disk_slot, keys[i])

        for slot, last in enumerate(pending):
            if last is not None:
                last[0].synchronize()
                self._writev_slot(slot, last[1] * self._slot_bytes, last[1], last[2])

    def _do_load(
        self,
        disk_slots: list[int],
        gpu_blocks: list[int],
        stream: torch.cuda.Stream,
        event_idx: int = -1,
        keys: list[int] | None = None,
    ) -> None:
        """Disk (preadv) -> buffer -> GPU (DMA), interleaved double-buffer."""
        assert self._load_slot_params
        n = self._num_buffer_slots
        if keys is None:
            keys = [0] * len(disk_slots)
        prev_dma_events: list[torch.Event | None] = [None] * n

        for i, (disk_slot, gpu_blk) in enumerate(zip(disk_slots, gpu_blocks)):
            buf_slot = i % n
            prev = prev_dma_events[buf_slot]
            if prev is not None:
                prev.synchronize()

            self._readv_slot(
                buf_slot, disk_slot * self._slot_bytes, disk_slot, keys[i]
            )

            prev_dma_events[buf_slot] = self._copy_unless_abandoned(
                False,
                event_idx,
                [0],
                [gpu_blk],
                self._load_slot_params[buf_slot],
                stream,
            )

        for ev in prev_dma_events:
            if ev is not None:
                ev.synchronize()
