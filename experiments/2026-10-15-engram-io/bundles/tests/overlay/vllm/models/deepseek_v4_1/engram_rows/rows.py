# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One Engram table's rows on one rank, read from its checkpoint file as steps ask.

Per step, on the model's stream: the step's row IDs are copied to the host and the
stream writes the step's sequence number into the control block; the native reader
(``_reader.c``), which never takes the GIL or calls CUDA, then reads the rows into
mapped host buffers and publishes the step. The forward's gather kernel waits for
that, bounded by a timeout that stops the GPU with a record of the step, and
dequantizes the rows. No Python thread, CUDA event or stream wait is involved, so
the rows arrive however busy the Python side is.
"""

import math
import sys
from contextlib import suppress

import numpy as np
import torch

from vllm.models.deepseek_v4_1.engram_rows._native import load

ROW_BYTES = 256  # E4M3 values of one row
SCALE_BYTES = 8  # E8M0 scales of one row, one per 32 values
PHASES = {0: "idle", 1: "waiting for the step's row IDs", 2: "reading", 3: "failed"}


def _check(result, operation: str):
    """Unpack a CUDA runtime or driver result (both report success as 0)."""
    error, *values = result
    if getattr(error, "value", error) != 0:
        raise RuntimeError(f"{operation} failed: {error}")
    return values[0] if values else None


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
        self.device = device
        self.nbytes = math.prod(shape) * dtype.itemsize
        with torch.cuda.device(device):
            self.pointer = int(_check(cudart.cudaHostAlloc(self.nbytes, flags), "cudaHostAlloc"))
            try:
                device_pointer = _check(cudart.cudaHostGetDevicePointer(self.pointer, 0),
                                        "cudaHostGetDevicePointer")
                self.host = _tensor_at(self.pointer, shape, dtype, torch.device("cpu"))
                self.device_view = _tensor_at(int(device_pointer), shape, dtype, device)
            except Exception:
                cudart.cudaFreeHost(self.pointer)
                self.pointer = 0
                raise

    def close(self) -> None:
        from cuda.bindings import runtime as cudart

        if self.pointer:
            _check(cudart.cudaFreeHost(self.pointer), "cudaFreeHost")
            self.pointer = 0

    def __del__(self):
        if self.pointer and sys is not None and not sys.is_finalizing():
            with suppress(Exception):
                self.close()


class EngramDiskRows:
    """Rows ``[shard_start, shard_end)`` of one Engram table, ``max_lookups`` per step.

    ``ids_timeout_s`` bounds the reader's wait for a step's row IDs and
    ``rows_timeout_s`` the GPU's wait for its rows; ``delay_s`` (tests only) delays
    every read.
    """

    def __init__(self, *, device, table_rows: int, shard_start: int, shard_end: int,
                 max_lookups: int, queue_depth: int = 128, ids_timeout_s: float = 20.0,
                 rows_timeout_s: float = 10.0, delay_s: float = 0.0):
        self.native = load()
        self.device = torch.device(device)
        self.shard_start, self.shard_end = int(shard_start), int(shard_end)
        self.max_lookups = int(max_lookups)
        self._allocations = []
        self.ids = torch.empty(self.max_lookups, dtype=torch.int64, pin_memory=True)
        self.weights = self._mapped((self.max_lookups, ROW_BYTES), torch.uint8, True)
        self.scales = self._mapped((self.max_lookups, SCALE_BYTES), torch.uint8, True)
        self.control = self._mapped((self.native.CTRL_WORDS,), torch.int32, False)
        self.control.host.zero_()
        self._control = self.control.host.numpy().view(np.uint32)
        self._control[self.native.CTRL_TIMEOUT_US] = int(rows_timeout_s * 1e6)
        self.rows_e4m3 = self.weights.device_view.view(torch.float8_e4m3fn)
        self._reader = self.native.reader(
            table_rows, self.shard_start, self.shard_end, self.max_lookups, queue_depth,
            self.ids.data_ptr(), self.weights.pointer, self.scales.pointer, self.control.pointer,
            float(ids_timeout_s), float(delay_s),
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

    def stage(self, stream: torch.cuda.Stream, ids: torch.Tensor, count: int) -> None:
        """Queue a step: ``ids`` holds its ``count`` row IDs (contiguous int64 on the
        device). Returns at once; the forward's ``gather`` waits for the rows."""
        self.raise_if_failed()
        from cuda.bindings import driver

        self._sequence = (self._sequence + 1) & 0xFFFFFFFF or 1
        with torch.cuda.stream(stream):
            self.ids[:count].copy_(ids.view(-1)[:count], non_blocking=True)
        _check(driver.cuStreamWriteValue32(stream.cuda_stream, self._ids_seq_device, self._sequence, 0),
               "cuStreamWriteValue32")
        self.native.submit(self._reader, self._sequence, count, 60.0)

    def gather(self, out: torch.Tensor, ids: torch.Tensor, num_tokens: torch.Tensor) -> None:
        """Wait for the staged step's rows and dequantize the first ``num_tokens``
        tokens' rows into ``out`` (``[tokens, 24 * 256]`` BF16), zero elsewhere and for
        rows outside this rank's shard. Capturable in a CUDA graph."""
        from vllm.models.deepseek_v4_1.tilelang.engram_rows import engram_rows_gather

        engram_rows_gather(ids, num_tokens, self.rows_e4m3, self.scales.device_view,
                           self.control.device_view, out, self.shard_start, self.shard_end)

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

    def close(self) -> None:
        """Stop the reader, then free the buffers; the GPU must be done with them."""
        self._reader = None
        torch.cuda.synchronize(self.device)
        for allocation in self._allocations:
            allocation.close()
        self._allocations = []
