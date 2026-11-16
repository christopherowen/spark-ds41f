# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One Engram table's rows on one rank, read from its checkpoint file as steps ask.

A step's lookups split by when their row IDs exist. The IDs the host hashed when the
runner prepared the step (prompt tokens, ``host_hash.py``) come with the step as its
early IDs, and the native reader (``_reader.c``), which never takes the GIL or calls
CUDA, reads those rows as soon as the GPU has gathered the previous step, so they are
read while the previous step computes. For the rest (tokens the GPU samples) the
step's row IDs are copied to the host on the model's stream and the stream writes the
step's sequence number into the control block; the reader then reads those rows and
publishes the step. One worker serves every table in the order the steps are staged
(layer order), so the first Engram layer's rows are never held back by the second's.
In the forward, one thread waits for the table's step, bounded by a timeout, and stops
the GPU with a record of the step if the rows do not come or the reader failed; then
the gather dequantizes this rank's rows and tells the reader the buffers are free. No
Python thread, CUDA event or stream wait is involved, so the rows arrive however busy
the Python side is.
"""

import math
import sys
from contextlib import suppress

import numpy as np
import torch

from vllm.models.deepseek_v4_1.engram_rows._native import load
from vllm.v1.utils import mlock_memory, munlock_memory

ROW_BYTES = 256  # E4M3 values of one row
SCALE_BYTES = 8  # E8M0 scales of one row, one per 32 values
PACKED_ROW_BYTES = ROW_BYTES + SCALE_BYTES  # 264-byte row-interleaved storage
PACKED_ROWS_PER_PAGE = 15  # 15 * 264 = 3,960 B + 136 B padding per 4096 B NVMe page
PACKED_PAGE_BYTES = 4096
DEFAULT_CACHE_ROWS = 65536  # 4-way set-associative row cache (16,384 sets x 4 ways)
PHASES = {
    0: "idle",
    1: "waiting for the step's row IDs",
    2: "reading",
    3: "failed",
    4: "waiting for the GPU to gather the previous step's rows",
}


def packed_table_bytes(rows: int) -> int:
    """Exact byte size of ``rows`` 264-byte rows packed at 15 rows per 4096-byte page."""
    rows = int(rows)
    if rows <= 0:
        return 0
    return ((rows + PACKED_ROWS_PER_PAGE - 1) // PACKED_ROWS_PER_PAGE) * PACKED_PAGE_BYTES


def _check(result, operation: str):
    """Unpack a CUDA runtime or driver result (both report success as 0)."""
    error, *values = result
    if getattr(error, "value", error) != 0:
        raise RuntimeError(f"{operation} failed: {error}")
    return values[0] if values else None


def _indexed(device) -> torch.device:
    """``device`` with its index: a storage and its tensor must name the same one."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _tensor_at(pointer: int, shape, dtype, device) -> torch.Tensor:
    nbytes = math.prod(shape) * dtype.itemsize
    storage = torch._C._construct_storage_from_data_pointer(int(pointer), device, nbytes)
    strides = [math.prod(shape[i + 1:]) for i in range(len(shape))]
    return torch.empty(0, dtype=dtype, device=device).set_(storage, 0, shape, strides)


class MappedHostAllocation:
    """Page-locked host memory the GPU reads and writes in place, with a CPU and a
    CUDA tensor over it."""

    def __init__(self, shape, dtype, device: torch.device, *, write_combined: bool):
        from cuda.bindings import runtime as cudart

        flags = cudart.cudaHostAllocMapped
        if write_combined:  # only the CPU writes it, only the GPU reads it
            flags |= cudart.cudaHostAllocWriteCombined
        self.device = device = _indexed(device)
        self.nbytes = math.prod(shape) * dtype.itemsize
        self._mlocked = False
        with torch.cuda.device(device):
            self.pointer = int(_check(cudart.cudaHostAlloc(self.nbytes, flags), "cudaHostAlloc"))
            try:
                device_pointer = _check(cudart.cudaHostGetDevicePointer(self.pointer, 0),
                                        "cudaHostGetDevicePointer")
                self.host = _tensor_at(self.pointer, shape, dtype, torch.device("cpu"))
                self.device_view = _tensor_at(int(device_pointer), shape, dtype, device)
                self._mlocked = mlock_memory(self.pointer, self.nbytes)
            except Exception:
                cudart.cudaFreeHost(self.pointer)
                self.pointer = 0
                raise

    def close(self) -> None:
        from cuda.bindings import runtime as cudart

        if self.pointer:
            if self._mlocked:
                munlock_memory(self.pointer, self.nbytes)
                self._mlocked = False
            _check(cudart.cudaFreeHost(self.pointer), "cudaFreeHost")
            self.pointer = 0

    def __del__(self):
        if self.pointer and sys is not None and not sys.is_finalizing():
            with suppress(Exception):
                self.close()


ALL_COLUMNS = (1 << 24) - 1


def column_compaction_map(
    offsets,
    table_rows: int,
    columns: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per-column compaction tables for ``columns`` (a subset of ``0..23``).

    Returns ``(col_start, col_end, col_delta, owned_starts, owned_rows)`` as contiguous
    ``int64`` arrays, where global row ``id`` of owned column ``c`` maps to physical
    compacted row ``id - col_delta[c]`` in ``[0, owned_rows.sum())``.
    """
    starts = np.asarray(offsets, dtype=np.int64)
    if starts.shape != (24,):
        raise ValueError(f"Engram column offsets must have 24 entries, got {starts.shape}")
    ends = np.empty(24, dtype=np.int64)
    ends[:23] = starts[1:]
    ends[23] = int(table_rows)

    owned = sorted(int(c) for c in columns)
    col_start = np.zeros(24, dtype=np.int64)
    col_end = np.zeros(24, dtype=np.int64)
    col_delta = np.zeros(24, dtype=np.int64)
    owned_starts = np.empty(len(owned), dtype=np.int64)
    owned_rows = np.empty(len(owned), dtype=np.int64)

    phys_offset = 0
    for idx, col in enumerate(owned):
        if not 0 <= col < 24:
            raise ValueError(f"Engram column index out of range: {col}")
        s = int(starts[col])
        e = int(ends[col])
        r = e - s
        if r < 0:
            raise ValueError(f"Invalid column range [{s}, {e}) for column {col}")
        col_start[col] = s
        col_end[col] = e
        col_delta[col] = s - phys_offset
        owned_starts[idx] = s
        owned_rows[idx] = r
        phys_offset += r

    return col_start, col_end, col_delta, owned_starts, owned_rows


def pack_engram_columns(
    weight_path: str,
    weight_offset: int,
    scale_path: str,
    scale_offset: int,
    out_path: str,
    col_starts: np.ndarray,
    col_rows: np.ndarray,
) -> None:
    """Stream ``(col_starts, col_rows)`` from ``weight_path`` and ``scale_path`` into a
    single-plane 4 KiB-page-aligned (15 x 264-byte rows per 4096-byte page) file at
    ``out_path`` without polluting the OS page cache."""
    starts = np.ascontiguousarray(col_starts, dtype=np.int64).reshape(-1)
    rows = np.ascontiguousarray(col_rows, dtype=np.int64).reshape(-1)
    load().pack_columns(
        str(weight_path),
        int(weight_offset),
        str(scale_path),
        int(scale_offset),
        str(out_path),
        starts,
        rows,
    )


class EngramDiskRows:
    """Rows ``[shard_start, shard_end)`` of the hash columns in the 24-bit mask ``columns``
    of one Engram table (lookup i of a step is column i % 24), ``max_lookups`` per step.

    ``ids_timeout_s`` bounds the reader's waits for the GPU (a step's row IDs, or its
    gather of the previous step) and ``rows_timeout_s`` the GPU's wait for its rows;
    ``delay_s`` (tests only) delays every read; ``check_early`` (debug) fails a step
    whose early ID differs from the GPU's.
    """

    def __init__(self, *, device, table_rows: int, shard_start: int, shard_end: int,
                 max_lookups: int, columns: int = ALL_COLUMNS, queue_depth: int = 512,
                 ids_timeout_s: float = 20.0,
                 rows_timeout_s: float = 10.0, delay_s: float = 0.0,
                 check_early: bool = False, timeline: bool = False,
                 cache_rows: int = DEFAULT_CACHE_ROWS):
        self.native = load()
        self.device = _indexed(device)
        self.shard_start, self.shard_end = int(shard_start), int(shard_end)
        self.columns = int(columns)
        if not 0 < self.columns <= ALL_COLUMNS:
            raise ValueError(f"Engram reader columns must be a nonzero 24-bit mask: {columns:#x}")
        self.max_lookups = int(max_lookups)
        self.check_early = bool(check_early)
        self.timeline_enabled = bool(timeline)
        self.cache_rows = int(cache_rows)
        self._allocations = []
        self.ids = torch.empty(self.max_lookups, dtype=torch.int64, pin_memory=True)
        self._ids_mlocked = mlock_memory(self.ids.data_ptr(), self.ids.nbytes)
        self.weights = self._mapped((self.max_lookups, ROW_BYTES), torch.uint8, True)
        self.scales = self._mapped((self.max_lookups, SCALE_BYTES), torch.uint8, True)
        self.control = self._mapped((self.native.CTRL_WORDS,), torch.int32, False)
        self.control.host.zero_()
        self._control = self.control.host.numpy().view(np.uint32)
        self._control[self.native.CTRL_TIMEOUT_US] = int(rows_timeout_s * 1e6)
        self.rows_e4m3 = self.weights.device_view.view(torch.float8_e4m3fn)
        self._reader = self.native.reader(
            table_rows, self.shard_start, self.shard_end, self.columns, self.max_lookups, queue_depth,
            self.ids.data_ptr(), self.weights.pointer, self.scales.pointer, self.control.pointer,
            float(ids_timeout_s), float(delay_s), self.check_early,
            self.timeline_enabled, self.cache_rows,
        )
        self._ids_seq_device = self.control.device_view.data_ptr() + 4 * self.native.CTRL_IDS_SEQ
        self._sequence = 0

    def _mapped(self, shape, dtype, write_combined: bool) -> MappedHostAllocation:
        allocation = MappedHostAllocation(shape, dtype, self.device, write_combined=write_combined)
        self._allocations.append(allocation)
        return allocation

    def add_source(self, path: str, offset: int, *, scale: bool) -> None:
        """Register the table's E4M3 rows (``scale=False``) or E8M0 scale rows, which
        start at ``offset`` in the immutable file ``path``."""
        self.native.add(self._reader, str(path), int(offset), bool(scale))

    def add_packed_source(
        self,
        path: str,
        offset: int = 0,
        *,
        col_start: np.ndarray | None = None,
        col_end: np.ndarray | None = None,
        col_delta: np.ndarray | None = None,
    ) -> None:
        """Register a single-plane 4 KiB-page-aligned (15 x 264B rows per 4096B page)
        source at ``offset`` in ``path``, optionally compacted to this rank's hash
        columns via ``(col_start, col_end, col_delta)``."""
        if (col_start is None) != (col_end is None) or (col_start is None) != (
            col_delta is None
        ):
            raise ValueError(
                "col_start, col_end and col_delta must all be provided together or omitted"
            )
        if col_start is not None:
            col_start = np.ascontiguousarray(col_start, dtype=np.int64).reshape(-1)
            col_end = np.ascontiguousarray(col_end, dtype=np.int64).reshape(-1)
            col_delta = np.ascontiguousarray(col_delta, dtype=np.int64).reshape(-1)
        self.native.add_packed(
            self._reader, str(path), int(offset), col_start, col_end, col_delta
        )

    def stage(
        self,
        stream: torch.cuda.Stream,
        ids: torch.Tensor,
        count: int,
        early: np.ndarray | None = None,
    ) -> None:
        """Queue a step: ``ids`` holds its ``count`` row IDs (contiguous int64 on the
        device, written by the GPU's hash); ``early`` (``count`` int64 on the host, or
        None) the IDs the host hashed, ``host_hash.UNKNOWN`` where only the GPU knows
        one. Returns at once; the forward's ``gather`` waits for the rows."""
        self.raise_if_failed()
        from cuda.bindings import driver

        has_unknown = True
        if early is not None:
            if not (
                isinstance(early, np.ndarray)
                and early.dtype == np.int64
                and early.flags.c_contiguous
            ):
                early = np.ascontiguousarray(early, dtype=np.int64)
            early = early.reshape(-1)
            if early.shape[0] != count:
                raise ValueError(
                    f"Engram early IDs: {early.shape[0]} for a step of {count} lookups"
                )
            has_unknown = bool(self.native.has_unknown(early))
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF or 1
        if has_unknown or self.check_early:
            with torch.cuda.stream(stream):
                self.ids[:count].copy_(ids.view(-1)[:count], non_blocking=True)
        _check(driver.cuStreamWriteValue32(stream.cuda_stream, self._ids_seq_device, self._sequence, 0),
               "cuStreamWriteValue32")
        self.native.submit(
            self._reader, self._sequence, count, 60.0, early, int(has_unknown)
        )

    def prefetch(self, ids: np.ndarray) -> None:
        """Warm the reader's user-space row cache in the background for ``ids`` (int64
        on the host) without advancing or modifying the GPU step buffers."""
        self.raise_if_failed()
        arr = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        self.native.prefetch(self._reader, arr, 60.0)

    def gather(self, out: torch.Tensor, ids: torch.Tensor, num_tokens: torch.Tensor) -> None:
        """Wait for the staged step's rows and dequantize the first ``num_tokens``
        tokens' rows into ``out`` (``[tokens, columns * 256]`` BF16, this rank's hash
        columns side by side), zero elsewhere and for rows outside this rank's shard.
        Capturable in a CUDA graph."""
        from vllm.models.deepseek_v4_1.tilelang.engram_rows import engram_rows_gather

        engram_rows_gather(ids, num_tokens, self.rows_e4m3, self.scales.device_view,
                           self.control.device_view, out, self.shard_start, self.shard_end,
                           self.columns)

    def failure(self) -> str | None:
        """Why the rows stopped, or None: the reader's failure or the GPU's timeout."""
        n = self.native
        status = n.status(self._reader)
        failed = int(self._control[n.CTRL_FAILED_SEQ])
        if status["failure"] is None and not failed:
            return None
        parts = []
        if failed:
            parts.append(
                f"the GPU waited {int(self._control[n.CTRL_WAITED_US])} us for the rows of "
                f"Engram step {failed} and stopped"
            )
        phase = PHASES.get(int(self._control[n.CTRL_PHASE]), "unknown")
        parts.append(f"the reader is {phase} on step {int(self._control[n.CTRL_PHASE_SEQ])}, "
                     f"last published {int(self._control[n.CTRL_ROWS_SEQ])}")
        if status["failure"] is not None:
            parts.append(status["failure"])
        return "; ".join(parts)

    def raise_if_failed(self) -> None:
        """Two control-block reads per step; the message is built only on a failure."""
        n = self.native
        if self._control[n.CTRL_PHASE] == 3 or self._control[n.CTRL_FAILED_SEQ]:
            raise RuntimeError(f"Engram rows: {self.failure()}")

    def stats(self) -> dict:
        return self.native.status(self._reader)

    TIMELINE_FIELDS = ("sequence", "count", "early", "gpu_sequence", "gpu_wait_ns",
                       "submitted", "started", "consumed", "ids", "read", "published", "plan_ns")

    def timeline(self) -> list[dict]:
        """Debug: the steps served since the last call, oldest first (``timeline`` on).
        Times are CLOCK_MONOTONIC ns: the host queued the step (submitted), the worker
        began it (started), saw the previous step gathered (consumed, early rows only),
        saw the GPU's IDs (ids), read the rows (read) and published them (published).
        gpu_wait_ns is the GPU's wait for gpu_sequence, the step before this one; plan_ns
        the time the step's reads spent planning (sorting and merging its lookups)."""
        rows = self.native.timeline(self._reader)
        return [dict(zip(self.TIMELINE_FIELDS, row)) for row in rows]

    def close(self) -> None:
        """Stop the reader, then free the buffers; the GPU must be done with them."""
        self._reader = None
        torch.cuda.synchronize(self.device)
        if getattr(self, "_ids_mlocked", False):
            munlock_memory(self.ids.data_ptr(), self.ids.nbytes)
            self._ids_mlocked = False
        for allocation in self._allocations:
            allocation.close()
        self._allocations = []
