# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the b12x project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Safetensors metadata routing with O_DIRECT reads into ordinary CUDA tensors.

The weights iterator yields meta tensors that carry each checkpoint tensor's file
range. Model weight loaders slice, narrow and view them as they would real tensors and
hand each copy to the session through vLLM's ``weight_transfer`` hook, which records
it as descriptors (``view_records``): a file range, a destination pointer, and the
rows and strides of a strided copy. ``flush`` runs the recorded descriptors through the
native helper's bounce ring (``_bounce.c``): O_DIRECT reads into registered host slots,
each copied into its destination by the copy engine. No payload is memory-mapped or
read through the page cache, and no kernel runs.

Plain copies, strided slices (tensor-parallel shards) and BF16 to FP32 expansion (done
by the CPU in the ring slot) are recorded. A view the descriptors cannot express fails
closed. Another dtype conversion of a contiguous tensor reads it through an 8 MiB GPU
scratch and converts it with Torch (``scratch_chunks``). Scalars and declared metadata
are read with O_DIRECT into CPU tensors, and a copy from a CPU tensor whose layout
matches its destination is queued as a host copy. The session declines every other
copy from a real tensor, and the copy goes to Torch.

Ported from local-inference-lab/b12x b12x/loader/_checkpoint.py at 236ddff04a5f
(Apache-2.0). Changed: only the bounce ring is kept (no GDS transport, shared read
group or progress callbacks), and the descriptor planning is in functions of its own
so it can be checked without a GPU.
"""

from __future__ import annotations

import json
import math
import os
from array import array
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from vllm.model_executor.model_loader.direct_io._native import load

# A descriptor is eight unsigned 64-bit words, as _batch.c reads them: (fd, offset,
# bytes, destination, transform, rows, source stride, destination stride). Transform
# READ copies `rows` rows of `bytes` from the file, EXPAND_BF16 also expands each BF16
# value to FP32 (destination rows twice as wide), and HOST_COPY copies `bytes` from the
# host address `offset` (fd 0, one row).
READ, EXPAND_BF16, HOST_COPY = 0, 1, 2
SCRATCH_BYTES = 8 << 20


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safetensors key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class TensorRange:
    name: str
    fd: int
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: object


def view_offset(entry: TensorRange, source, destination) -> int:
    """The byte offset of ``source``, a view of the checkpoint tensor ``entry``, in the
    tensor's range; refuses a copy the descriptors cannot express."""
    if (
        source._version != 0
        or source.element_size() != entry.dtype.itemsize
        or source.is_neg()
        or source.is_conj()
        or destination.is_neg()
        or destination.is_conj()
        or source.shape != destination.shape
        or any(stride < 0 for stride in (*source.stride(), *destination.stride()))
    ):
        raise NotImplementedError(
            f"direct load needs an explicit transform for {entry.name}: "
            f"{source.shape}/{source.stride()}/{source.dtype} -> "
            f"{destination.shape}/{destination.stride()}/{destination.dtype}"
        )
    offset = source.storage_offset() * source.element_size()
    source_extent = (
        1
        + sum(
            (size - 1) * stride
            for size, stride in zip(source.shape, source.stride(), strict=False)
        )
    ) * source.element_size()
    if source.numel() and offset + source_extent > entry.nbytes:
        raise ValueError(f"source view exceeds checkpoint range: {entry.name}")
    return offset


def view_records(
    entry: TensorRange, offset: int, source, destination, expand: bool
) -> list[tuple[int, ...]]:
    """Descriptors that copy ``source``, a view at byte ``offset`` of ``entry``, into
    ``destination`` of the same shape (and dtype, or FP32 for a BF16 source when
    ``expand``). The innermost axes that both lay out contiguously make one row, the
    next axis the rows of one strided copy, and each index of the axes outside it one
    descriptor."""
    elements = 1
    axis = source.ndim - 1
    while axis >= 0:
        size = source.shape[axis]
        if size != 1 and (
            source.stride(axis) != elements or destination.stride(axis) != elements
        ):
            break
        elements *= size
        axis -= 1
    rows = source.shape[axis] if axis >= 0 else 1
    source_stride = source.stride(axis) * source.element_size() if axis >= 0 else 0
    destination_stride = (
        destination.stride(axis) * destination.element_size() if axis >= 0 else 0
    )
    outer = source.shape[:axis] if axis >= 0 else ()
    records = []
    for index in product(*(range(size) for size in outer)):
        source_delta = (
            sum(i * s for i, s in zip(index, source.stride(), strict=False))
            * source.element_size()
        )
        destination_delta = (
            sum(i * s for i, s in zip(index, destination.stride(), strict=False))
            * destination.element_size()
        )
        records.append(
            (
                entry.fd,
                entry.offset + offset + source_delta,
                elements * source.element_size(),
                destination.data_ptr() + destination_delta,
                EXPAND_BF16 if expand else READ,
                rows,
                source_stride,
                destination_stride,
            )
        )
    return records


def scratch_chunks(entry: TensorRange, offset: int, source, capacity: int):
    """``(start, count, file offset, bytes)`` of each read that passes the contiguous
    view ``source`` (at byte ``offset`` of ``entry``) through a scratch of ``capacity``
    elements of its dtype, ``count`` elements from element ``start``."""
    for start in range(0, source.numel(), capacity):
        count = min(capacity, source.numel() - start)
        yield (
            start,
            count,
            entry.offset + offset + start * source.element_size(),
            count * source.element_size(),
        )


class DirectWeightSession:
    """Retain source metadata and destination owners through synchronous reads.

    Meta tensors carry only shape, dtype and view offsets through model routing.
    They never substitute for numerical input: unregistered transformations fail
    closed. Scalars and explicitly declared metadata are read for value access.
    No checkpoint payload is mapped or read through the page cache.
    """

    def __init__(self, device=0, io_threads=8):
        import torch

        if not 1 <= io_threads <= 16:
            raise ValueError("io_threads must be between 1 and 16")
        torch.cuda.init()
        self.native = load()
        self.reader = self.native.direct_reader(device)
        self.device = device
        self.io_threads = io_threads
        self.executor = None
        self.records = array("Q")
        self.destinations = []
        self.files = []
        self.sources = {}
        self.torch_copy_bytes = 0
        self.payload_bytes = 0
        self.metadata_bytes = 0
        self.metadata_copy_bytes = 0
        self.loaded_tensors = 0
        self.transform_bytes = 0
        self.bf16_expansion_bytes = 0
        self.transform_scratch = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            if exc[0] is None:
                self.flush()
        finally:
            self._close()

    def _close(self):
        self.records = array("Q")
        self.destinations.clear()
        self.executor = None
        self.transform_scratch = None
        self.sources.clear()
        for fd in self.files:
            os.close(fd)
        self.files.clear()
        self.reader = None

    def flush(self):
        """Complete accepted destinations before a numerical consumer runs."""
        if not self.records:
            return
        try:
            self._execute(self.records)
        finally:
            self.records = array("Q")
            self.destinations.clear()

    def _execute(self, records):
        import torch

        with torch.accelerator.device_index(self.device):
            if self.executor is None:
                self.executor = self.native.batch_executor(self.device, self.io_threads)
            self.native.batch_execute(
                self.executor,
                records,
                torch.cuda.current_stream(self.device).cuda_stream,
            )

    def materialize(self, source):
        """Read owned transform inputs without routing arithmetic through metadata."""
        import torch

        if source.device.type != "meta":
            return source.clone()
        destination = torch.empty(
            source.shape, dtype=source.dtype, device=torch.device("cuda", self.device)
        )
        self(destination, source)
        self.flush()
        return destination

    def _metadata_values(self, entries):
        import torch

        values = {}
        ordered = sorted(entries, key=lambda entry: entry.offset)
        start = 0
        while start < len(ordered):
            first = ordered[start]
            end = first.offset + first.nbytes
            stop = start + 1
            while stop < len(ordered):
                following = ordered[stop]
                if following.offset - end > 4096 or (
                    following.offset + following.nbytes - first.offset > 8 << 20
                ):
                    break
                end = following.offset + following.nbytes
                stop += 1
            self.metadata_bytes += end - first.offset
            if self.metadata_bytes > 64 << 20:
                raise ValueError(
                    "declared metadata spans exceed the 64 MiB session budget"
                )
            raw = bytearray(
                self.native.direct_bytes(
                    self.reader, first.fd, first.offset, end - first.offset
                )
            )
            for entry in ordered[start:stop]:
                values[entry.name] = (
                    torch.frombuffer(
                        raw,
                        dtype=entry.dtype,
                        count=math.prod(entry.shape),
                        offset=entry.offset - first.offset,
                    ).reshape(entry.shape)
                    if entry.nbytes
                    else torch.empty(entry.shape, dtype=entry.dtype, device="cpu")
                )
            start = stop
        return values

    def manifest(self, path):
        import torch

        fd = os.open(path, os.O_RDONLY | os.O_DIRECT | os.O_CLOEXEC)
        self.files.append(fd)
        size = os.fstat(fd).st_size
        length = int.from_bytes(
            self.native.direct_bytes(self.reader, fd, 0, 8), "little"
        )
        if length > 100 << 20 or length > size - 8:
            raise ValueError(f"invalid safetensors header length in {path}")
        header = json.loads(
            self.native.direct_bytes(self.reader, fd, 8, length),
            object_pairs_hook=_unique_object,
        )
        if not isinstance(header, dict):
            raise ValueError("safetensors header must be an object")
        dtypes = {
            "BOOL": torch.bool,
            "U8": torch.uint8,
            "I8": torch.int8,
            "I16": torch.int16,
            "I32": torch.int32,
            "I64": torch.int64,
            "U16": torch.uint16,
            "U32": torch.uint32,
            "U64": torch.uint64,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "F64": torch.float64,
            "F8_E4M3": torch.float8_e4m3fn,
            "F8_E5M2": torch.float8_e5m2,
            "F8_E8M0": torch.float8_e8m0fnu,
        }
        result = []
        for name, entry in header.items():
            if name == "__metadata__":
                if not isinstance(entry, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in entry.items()
                ):
                    raise ValueError("invalid safetensors metadata")
                continue
            if not isinstance(entry, dict):
                raise ValueError(f"invalid tensor entry: {name}")
            shape = entry.get("shape")
            offsets = entry.get("data_offsets")
            dtype = dtypes.get(entry.get("dtype"))
            if (
                dtype is None
                or not isinstance(shape, list)
                or any(type(d) is not int or d < 0 for d in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(d) is not int or d < 0 for d in offsets)
            ):
                raise ValueError(f"invalid tensor metadata: {name}")
            start, end = offsets
            nbytes = math.prod(shape) * dtype.itemsize
            if end - start != nbytes or end > size - length - 8:
                raise ValueError(f"invalid tensor file range: {name}")
            result.append(
                TensorRange(name, fd, 8 + length + start, nbytes, tuple(shape), dtype)
            )
        end = 8 + length
        for tensor in sorted(result, key=lambda t: (t.offset, t.nbytes)):
            if tensor.offset != end:
                raise ValueError(
                    f"overlap or hole in safetensors payload: {tensor.name}"
                )
            end += tensor.nbytes
        if end != size:
            raise ValueError("safetensors payload does not cover file")
        return result

    def weights(
        self,
        files,
        *,
        prefixes=None,
        prefix="",
        index_path=None,
        needs_values=None,
        skip=None,
    ):
        import torch

        weight_map = None
        indexed_paths = {}
        if index_path is not None:
            index_file_path = Path(index_path)
            if index_file_path.is_file():
                resolved_index = index_file_path.resolve()
                cached = getattr(self, "_index_cache", None)
                if cached is not None and cached[0] == resolved_index:
                    _, weight_map, indexed_paths = cached
                else:
                    weight_map = json.loads(index_file_path.read_text())["weight_map"]
                    index_parent = index_file_path.parent
                    for indexed_file in set(weight_map.values()):
                        indexed_paths[indexed_file] = (
                            index_parent / indexed_file
                        ).resolve()
                    self._index_cache = (resolved_index, weight_map, indexed_paths)
        for path in files:
            manifest = self.manifest(path)
            resolved_path = Path(path).resolve()
            selected = []
            for entry in sorted(manifest, key=lambda t: t.name):
                if prefixes and not entry.name.startswith(prefixes):
                    continue
                if weight_map is not None:
                    indexed_file = weight_map.get(entry.name)
                    if indexed_file is None:
                        continue
                    if indexed_paths[indexed_file] != resolved_path:
                        continue
                if skip is not None and skip(entry.name):
                    continue
                selected.append(entry)
            values = self._metadata_values(
                [
                    entry
                    for entry in selected
                    if needs_values is not None and needs_values(entry)
                ]
            )
            for entry in selected:
                if entry.name in values:
                    tensor = values[entry.name]
                else:
                    tensor = torch.empty(entry.shape, dtype=entry.dtype, device="meta")
                    storage = tensor.untyped_storage()
                    self.sources[storage._cdata] = (storage, entry)
                yield prefix + entry.name, tensor

    def __call__(self, destination, source):
        import torch

        if source.device.type != "meta":
            if (
                source.device.type == "cpu"
                and source.dtype == destination.dtype
                and (
                    source.shape == destination.shape
                    or source.numel() == destination.numel() == 1
                )
                and source.is_contiguous()
                and not source.is_neg()
                and not source.is_conj()
                and not destination.is_neg()
                and not destination.is_conj()
                and destination.device.index == self.device
                and destination.is_cuda
                and destination.is_contiguous()
            ):
                if source.nbytes:
                    self.records.extend(
                        (
                            0,
                            source.data_ptr(),
                            source.nbytes,
                            destination.data_ptr(),
                            HOST_COPY,
                            1,
                            0,
                            0,
                        )
                    )
                    self.destinations.extend((destination, source))
                    self.metadata_copy_bytes += source.nbytes
                    torch.autograd.graph.increment_version(destination)
                return True
            self.flush()
            self.torch_copy_bytes += destination.nbytes
            return False
        item = self.sources.get(source.untyped_storage()._cdata)
        if item is None:
            raise NotImplementedError(
                "checkpoint routing performed an unsupported data transformation"
            )
        _, entry = item
        offset = view_offset(entry, source, destination)
        if source.nbytes == 0:
            return True
        if destination.device.type != "cuda" or destination.device.index != self.device:
            raise ValueError("destination must use the session CUDA device")
        storage = destination.untyped_storage()
        extent = destination.element_size() * (
            1
            + sum(
                (size - 1) * stride
                for size, stride in zip(
                    destination.shape, destination.stride(), strict=True
                )
            )
        )
        if destination.data_ptr() - storage.data_ptr() + extent > storage.nbytes():
            raise ValueError(f"destination exceeds tensor storage: {entry.name}")
        expand = source.dtype == torch.bfloat16 and destination.dtype == torch.float32
        byte_cast = {source.dtype, destination.dtype} <= {torch.int8, torch.uint8}
        if destination.dtype == source.dtype or expand or byte_cast:
            for record in view_records(entry, offset, source, destination, expand):
                self.records.extend(record)
            self.destinations.append(destination)
            if expand:
                self.bf16_expansion_bytes += source.nbytes
        else:
            if not source.is_contiguous() or not destination.is_contiguous():
                raise NotImplementedError(f"strided dtype conversion: {entry.name}")
            with torch.accelerator.device_index(destination.device.index):
                self.flush()
                if self.transform_scratch is None:
                    self.transform_scratch = torch.empty(
                        SCRATCH_BYTES, dtype=torch.uint8, device=destination.device
                    )
                scratch = self.transform_scratch.view(source.dtype)
                flat = destination.view(-1)
                for start, count, position, nbytes in scratch_chunks(
                    entry, offset, source, scratch.numel()
                ):
                    self._execute(
                        array(
                            "Q",
                            (
                                entry.fd,
                                position,
                                nbytes,
                                scratch.data_ptr(),
                                READ,
                                1,
                                0,
                                0,
                            ),
                        )
                    )
                    flat[start : start + count].copy_(scratch[:count])
                self.transform_bytes += source.nbytes
        torch.autograd.graph.increment_version(destination)
        self.payload_bytes += source.nbytes
        self.loaded_tensors += 1
        return True

    def stats(self, *, flush=True):
        if flush:
            self.flush()
        io = self.native.direct_stats(self.reader)
        if self.executor is not None:
            for name, value in self.native.batch_stats(self.executor).items():
                io[name] = io.get(name, 0) + value
        return {
            **io,
            "payload_bytes": self.payload_bytes,
            "metadata_bytes": self.metadata_bytes,
            "loaded_tensors": self.loaded_tensors,
            "transform_bytes": self.transform_bytes,
            "bf16_expansion_bytes": self.bf16_expansion_bytes,
            "transform_scratch_bytes": (
                0 if self.transform_scratch is None else self.transform_scratch.nbytes
            ),
            "metadata_host_copy_bytes": self.metadata_copy_bytes,
            "torch_copy_bytes": self.torch_copy_bytes,
        }
