# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from functools import partial

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import (
    async_tensor_h2d,
    get_accelerator_view_from_cpu_tensor,
)

logger = init_logger(__name__)

# Default round-robin depth for the UVA buffer pools. Must be >= the number of
# concurrent in-flight steps (engine batch_queue_size).
_DEFAULT_MAX_CONCURRENCY = 2


def set_default_max_concurrency(n: int) -> None:
    global _DEFAULT_MAX_CONCURRENCY
    _DEFAULT_MAX_CONCURRENCY = max(2, n)


class UvaBuffer:
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        if not is_uva_available():
            raise RuntimeError("UVA is not available")
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self.np = self.cpu.numpy()
        self._uva = get_accelerator_view_from_cpu_tensor(self.cpu)

    def uva(self, n: int | None = None) -> torch.Tensor:
        return self._uva[:n] if n is not None else self._uva


class NonUvaBuffer:
    """Explicit-copy fallback for platforms without pinned memory."""

    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        from vllm.platforms import current_platform

        logger.warning_once(
            "Pinned memory is not available on this platform; falling back "
            "to device memory for UVA buffers."
        )
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()
        self._uva = torch.zeros(size, dtype=dtype, device=current_platform.device_type)

    def uva(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return self._uva.copy_(self.cpu)
        return self._uva[:n].copy_(self.cpu[:n])


class UvaBufferPool:
    """Preallocate each slot at size, growing its first dimension as needed.

    Callers must retire a slot's GPU readers before reuse, including growth.
    """

    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int | None = None,
    ):
        if max_concurrency is None:
            max_concurrency = _DEFAULT_MAX_CONCURRENCY
        self.size = size
        self.dtype = dtype
        self.max_concurrency = max_concurrency

        # UVA buffers for concurrency
        self._buffer_cls = UvaBuffer if is_uva_available() else NonUvaBuffer
        self._uva_bufs = [self._buffer_cls(size, dtype) for _ in range(max_concurrency)]
        # Current buffer index
        self._curr = 0

    def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        # Round robin to the next buffer.
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        n = len(x)
        if n > buf.cpu.shape[0]:
            capacity = 1 << (n - 1).bit_length()
            buf = self._buffer_cls((capacity, *buf.cpu.shape[1:]), self.dtype)
            self._uva_bufs[self._curr] = buf
        # CPU-to-CPU copy
        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
        dst[:n] = x
        return buf.uva(n)

    def copy_to_gpu(
        self,
        x: torch.Tensor | np.ndarray,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uva = self.copy_to_uva(x)
        # CPU-to-GPU copy
        return uva.clone() if out is None else out.copy_(uva, non_blocking=True)


class UvaBackedTensor:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int | None = None,
    ):
        self.dtype = dtype

        # Source of truth
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
        self.np = self.cpu.numpy()

        # Buffers for concurrency
        self.pool = UvaBufferPool(size, dtype, max_concurrency)
        self.gpu = self.pool.copy_to_uva(self.np)

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        # CPU-to-CPU copy
        self.gpu = self.pool.copy_to_uva(self.np[:n] if n is not None else self.np)
        return self.gpu


class StagedWriteTensor:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int | None = None,
        uva_instead_of_gpu: bool = False,
    ):
        if max_concurrency is None:
            max_concurrency = _DEFAULT_MAX_CONCURRENCY
        supported_dtypes = [torch.int32, torch.int64, torch.float32]
        if dtype not in supported_dtypes:
            raise ValueError(
                f"Unsupported dtype {dtype}: should be one of {supported_dtypes}"
            )
        self.num_rows = size if isinstance(size, int) else size[0]
        self.dtype = dtype
        self.device = device
        self.max_concurrency = max_concurrency

        # Without UVA, large-but-cold tensors live on the GPU directly.
        uva_instead_of_gpu = uva_instead_of_gpu and is_uva_available()

        if not uva_instead_of_gpu:
            # Create a GPU tensor (default)
            self.gpu = torch.zeros(size, dtype=dtype, device=device)
        else:
            # For a large but not-frequently-accessed tensor, we can use UVA instead of
            # GPU to save GPU memory
            self._uva_buf = UvaBuffer(size, dtype)
            self.gpu = self._uva_buf.uva()
        # The write kernel addresses rows of the flat tensor by its row stride.
        self._flat_gpu = self.gpu.view(-1)

        self._np_dtype = {
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.float32: np.float32,
        }[dtype]
        self._staged_write_indices: list[int] = []
        self._staged_write_starts: list[int] = []
        self._staged_write_contents: list[int | float] = []
        self._staged_write_arrays: list[np.ndarray] = []
        self._staged_write_cu_lens: list[int] = []

        new_buffer = partial(UvaBufferPool, max_concurrency=max_concurrency)

        self.write_indices = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_starts = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_cu_lens = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_contents = new_buffer(1, dtype=dtype) if uva_instead_of_gpu else None

    def _flush_staged_arrays(self) -> None:
        if self._staged_write_arrays:
            for arr in self._staged_write_arrays:
                self._staged_write_contents.extend(arr.tolist())
            self._staged_write_arrays.clear()

    def stage_write(
        self, index: int, start: int, x: Iterable[int] | Iterable[float]
    ) -> None:
        assert index >= 0
        assert start >= 0
        if isinstance(x, np.ndarray) or hasattr(x, "__array__"):
            arr = np.asarray(x, dtype=self._np_dtype).reshape(-1)
            if arr.size == 0:
                return
            if not self._staged_write_contents:
                self._staged_write_indices.append(index)
                self._staged_write_starts.append(start)
                self._staged_write_arrays.append(arr)
                prev_cu = (
                    self._staged_write_cu_lens[-1]
                    if self._staged_write_cu_lens
                    else 0
                )
                self._staged_write_cu_lens.append(prev_cu + int(arr.size))
                return
            x = arr.tolist()
        if not x:
            return
        self._flush_staged_arrays()
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(start)
        self._staged_write_contents.extend(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def stage_write_elem(self, index: int, x: int) -> None:
        assert index >= 0
        self._flush_staged_arrays()
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(0)
        self._staged_write_contents.append(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def apply_write(self) -> None:
        n = len(self._staged_write_indices)
        if n == 0:
            return

        indices_uva = self.write_indices.copy_to_uva(self._staged_write_indices)
        starts_uva = self.write_starts.copy_to_uva(self._staged_write_starts)
        cu_lens_uva = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)

        if self._staged_write_arrays and not self._staged_write_contents:
            contents_src: np.ndarray | list[int | float] = (
                self._staged_write_arrays[0]
                if len(self._staged_write_arrays) == 1
                else np.concatenate(self._staged_write_arrays)
            )
        else:
            self._flush_staged_arrays()
            contents_src = self._staged_write_contents

        if self.write_contents is None:
            write_contents = async_tensor_h2d(
                contents_src, device=self.device, dtype=self.dtype
            )
        else:
            write_contents = self.write_contents.copy_to_uva(contents_src)

        from vllm.v1.worker.gpu import tilelang_inputs

        # Write diffs to the GPU buffer
        tilelang_inputs.apply_staged_writes(
            self._flat_gpu,
            self.gpu.stride(0),
            indices_uva,
            starts_uva,
            write_contents,
            cu_lens_uva,
            self._staged_write_cu_lens,
        )
        # Clear the staged writes
        self.clear_staged_writes()

    def register_warmup(self) -> None:
        """Declare ``apply_write`` for this tensor's element type."""
        from vllm.v1.worker.gpu import tilelang_inputs

        dtype = self.dtype
        tilelang_inputs.declare(
            lambda: tilelang_inputs.staged_write_calls(dtypes=(dtype,), fused=False)
        )

    def clear_staged_writes(self) -> None:
        self._staged_write_indices.clear()
        self._staged_write_starts.clear()
        self._staged_write_contents.clear()
        self._staged_write_arrays.clear()
        self._staged_write_cu_lens.clear()


class FusedStagedWriter:
    """Applies the staged writes of several `StagedWriteTensor`s at once."""

    def __init__(
        self, device: torch.device, max_writes: int, max_concurrency: int | None = None
    ):
        new_pool = partial(
            UvaBufferPool, dtype=torch.int32, max_concurrency=max_concurrency
        )
        self.group_ids = new_pool(max_writes)
        self.indices = new_pool(max_writes)
        self.starts = new_pool(max_writes)
        self.cu_lens = new_pool(max_writes)
        self.device = device

    @staticmethod
    def register_warmup() -> None:
        """Declare ``apply`` (int32 contents through per-group pointers)."""
        from vllm.v1.worker.gpu import tilelang_inputs

        tilelang_inputs.declare(
            lambda: tilelang_inputs.staged_write_calls(dtypes=(), fused=True)
        )

    def apply(
        self,
        tensors: Sequence[StagedWriteTensor],
        output_ptrs: torch.Tensor,
        output_strides: torch.Tensor,
    ) -> None:
        """Apply and clear the staged writes of `tensors` with one kernel.

        ``output_ptrs``: the tensors' data pointers as int64."""
        group_ids: list[int] = []
        indices: list[int] = []
        starts: list[int] = []
        contents: list[int | float] = []
        cu_lens: list[int] = []

        for group_id, t in enumerate(tensors):
            n = len(t._staged_write_indices)
            if n == 0:
                continue

            t._flush_staged_arrays()
            group_ids.extend([group_id] * n)
            indices.extend(t._staged_write_indices)
            starts.extend(t._staged_write_starts)
            content_base = len(contents)
            contents.extend(t._staged_write_contents)
            cu_lens.extend(content_base + cu_len for cu_len in t._staged_write_cu_lens)

        if not group_ids:
            return

        group_ids_uva = self.group_ids.copy_to_uva(group_ids)
        indices_uva = self.indices.copy_to_uva(indices)
        starts_uva = self.starts.copy_to_uva(starts)
        cu_lens_uva = self.cu_lens.copy_to_uva(cu_lens)
        contents_gpu = async_tensor_h2d(contents, device=self.device, dtype=torch.int32)

        from vllm.v1.worker.gpu import tilelang_inputs

        tilelang_inputs.apply_fused_staged_writes(
            output_ptrs,
            output_strides,
            indices_uva,
            starts_uva,
            contents_gpu,
            cu_lens_uva,
            group_ids_uva,
            cu_lens,
        )
        for t in tensors:
            t.clear_staged_writes()


def declare_staged_write_kernels() -> None:
    """Declare the write of every element type a ``StagedWriteTensor`` may have and
    ``FusedStagedWriter``'s: keyed by element type alone, they cover the request
    and sampling states' tensors, including those that declare nothing."""
    from vllm.v1.worker.gpu import tilelang_inputs

    tilelang_inputs.declare(tilelang_inputs.staged_write_calls)


_TRITON_REFS = frozenset(("_apply_write_kernel", "_load_ptr"))


def __getattr__(name: str):
    if name in _TRITON_REFS:
        from vllm.v1.worker.gpu import _triton_refs

        return getattr(_triton_refs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
