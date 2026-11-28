# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scoped transport for checkpoint copies into model parameter views."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

WeightWriter = Callable[[torch.Tensor, torch.Tensor], bool]
_writer: ContextVar[WeightWriter | None] = ContextVar("weight_writer", default=None)


@dataclass(frozen=True)
class FileTensorSource:
    """Immutable checkpoint tensor range, without materializing its payload."""

    path: str
    offset: int
    shape: tuple[int, ...]
    dtype: torch.dtype


def get_file_tensor_source(tensor: torch.Tensor) -> FileTensorSource | None:
    """Return the backing range carried by an opted-in checkpoint tensor."""
    return getattr(tensor, "_vllm_file_tensor_source", None)


@contextmanager
def weight_transfer(
    writer: WeightWriter,
) -> Iterator[None]:
    """Install a synchronous writer during exclusive model weight loading.

    Return True after accepting a copy, retaining its owners until completion.
    Call flush_weight_transfers before consuming queued parameter values.
    Return False to use Torch's usual copy semantics.
    """
    token = _writer.set(writer)
    try:
        yield
    finally:
        _writer.reset(token)


def copy_weight(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    writer = _writer.get()
    if destination.is_meta or writer is None or not writer(destination, source):
        if (
            destination.is_cuda
            and source.device.type == "cpu"
            and not source.is_pinned()
        ):
            # A checkpoint tensor is often a view of a memory-mapped file (safetensors).
            # On GB10, a pageable copy above about 128 KiB reads the source through the
            # host page tables, and with driver R580 on a 64 KiB-page kernel it never
            # completes when the file pages are not yet mapped and the copy does not
            # start on a page boundary (the host spins in the copy, the GPU is idle).
            # A copy into anonymous memory reads the file on the CPU first. (Not
            # pin_memory: Torch's host allocator keeps every pinned block, rounded up
            # to a power of two, and on GB10 that memory is the GPU's.)
            source = source.clone()
        destination.copy_(source)
    return destination


def is_weight_transfer_active() -> bool:
    """Whether checkpoint copies use a scoped writer with deferred completion."""
    return _writer.get() is not None


def flush_weight_transfers() -> None:
    """Complete queued checkpoint reads before numerical weight preparation."""
    writer = _writer.get()
    flush = getattr(writer, "flush", None)
    if flush is not None:
        flush()


def finish_weight_transfers() -> None:
    """Complete a model routing epoch on every participating loading rank.

    Writers may exchange rank-local descriptors at this explicit boundary.
    Eager numerical consumers continue to use flush_weight_transfers.
    """
    writer = _writer.get()
    finish = getattr(writer, "finish", None)
    if finish is not None:
        finish()
    else:
        flush_weight_transfers()


def materialize_weight(source: torch.Tensor) -> torch.Tensor:
    """Own checkpoint values needed by a numerical loading transform."""
    writer = _writer.get()
    materialize = getattr(writer, "materialize", None)
    return materialize(source) if materialize is not None else source.clone()


_SAFETENSORS_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": getattr(torch, "float8_e8m0fnu", torch.uint8),
    "I16": torch.int16,
    "U16": torch.uint16,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "U64": torch.uint64,
}


def safetensors_file_sources(path: str) -> dict[str, FileTensorSource]:
    import json
    import os
    from safetensors import safe_open

    path = os.path.abspath(path)
    with open(path, "rb", buffering=0) as f:
        size = os.fstat(f.fileno()).st_size
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Missing safetensors header in {path}")
        length = int.from_bytes(length_bytes, "little")
        if length > 100 << 20 or length > size - 8:
            raise ValueError(f"Invalid safetensors header length in {path}")
        header = json.loads(f.read(length))
    with safe_open(path, framework="pt"):
        pass
    return {
        name: FileTensorSource(
            path=path,
            offset=8 + length + entry["data_offsets"][0],
            shape=tuple(entry["shape"]),
            dtype=_SAFETENSORS_TO_TORCH_DTYPE[entry["dtype"]],
        )
        for name, entry in header.items()
        if name != "__metadata__"
    }


def file_source_tensor(source: FileTensorSource) -> torch.Tensor:
    tensor = torch.empty(source.shape, dtype=source.dtype, device="meta")
    tensor._vllm_file_tensor_source = source
    return tensor
