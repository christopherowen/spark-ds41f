# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import ctypes
import gc
import struct
import sys
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import psutil
import torch
import torch.types

from vllm.logger import init_logger
from vllm.platforms import current_platform

from .mem_constants import GiB_bytes, KiB_bytes, MiB_bytes

logger = init_logger(__name__)


def format_kib(b: int) -> str:
    return f"{round(b / KiB_bytes, 2)}"


def format_mib(b: int) -> str:
    return f"{round(b / MiB_bytes, 2)}"


def format_gib(b: int) -> str:
    return f"{round(b / GiB_bytes, 2)}"


@cache
def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    from vllm import _custom_ops as ops

    max_shared_mem = ops.get_max_shared_memory_per_block_device_attribute(gpu)
    # value 0 will cause MAX_SEQ_LEN become negative and test_attention.py
    # will fail
    assert max_shared_mem > 0, "max_shared_mem cannot be zero"
    return int(max_shared_mem)


_UMA_PRESSURE_THRESHOLD = 0.8
_UMA_MIN_RELEASE_BYTES = 512 * MiB_bytes


@cache
def _get_malloc_trim() -> Callable[[int], int] | None:
    if sys.platform != "linux":
        return None
    with contextlib.suppress(Exception):
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        return malloc_trim
    return None


@cache
def _get_c10_mi_collect() -> Callable[[bool], None] | None:
    """Resolve mimalloc's ``mi_collect(bool force)`` inside ``libc10.so`` when
    PyTorch's CPU tensor allocator (``c10::alloc_cpu`` / ``c10::free_cpu``) is
    built with statically linked mimalloc.
    """
    if sys.platform != "linux":
        return None
    with contextlib.suppress(Exception):
        libc10_path = Path(torch.__file__).resolve().parent / "lib" / "libc10.so"
        if not libc10_path.is_file():
            return None
        lib = ctypes.CDLL(str(libc10_path))
        mi_collect = getattr(lib, "mi_collect", None)
        if mi_collect is not None:
            mi_collect.argtypes = [ctypes.c_bool]
            mi_collect.restype = None
            return mi_collect

        free_cpu_ptr = ctypes.cast(lib._ZN3c108free_cpuEPv, ctypes.c_void_p).value
        if not free_cpu_ptr:
            return None

        with libc10_path.open("rb") as f:
            ident = f.read(64)
            if (
                len(ident) < 64
                or ident[:4] != b"\x7fELF"
                or ident[4] != 2
                or ident[5] != 1
            ):
                return None
            e_shoff = struct.unpack_from("<Q", ident, 40)[0]
            e_shentsize, e_shnum = struct.unpack_from("<HH", ident, 58)
            if e_shoff == 0 or e_shnum == 0 or e_shentsize < 64:
                return None
            f.seek(e_shoff)
            shdrs = f.read(e_shentsize * e_shnum)
            symtab_hdr = None
            for idx in range(e_shnum):
                sh = struct.unpack_from("<IIQQQQIIQQ", shdrs, idx * e_shentsize)
                if sh[1] == 2:  # SHT_SYMTAB
                    symtab_hdr = sh
                    break
            if symtab_hdr is None:
                return None
            _, _, _, _, sh_offset, sh_size, sh_link, _, _, sh_entsize = symtab_hdr
            if sh_entsize != 24 or sh_link >= e_shnum:
                return None
            strtab_sh = struct.unpack_from(
                "<IIQQQQIIQQ", shdrs, sh_link * e_shentsize
            )
            str_offset, str_size = strtab_sh[4], strtab_sh[5]
            f.seek(str_offset)
            strtab = f.read(str_size)
            f.seek(sh_offset)
            symtab = f.read(sh_size)

        target_free = b"_ZN3c108free_cpuEPv\x00"
        target_mi = b"mi_collect\x00"
        free_val = 0
        mi_val = 0
        for offset in range(0, len(symtab) - 23, 24):
            st_name, st_info, _, _, st_value, _ = struct.unpack_from(
                "<IBBHQQ", symtab, offset
            )
            if st_value == 0 or (st_info & 0xF) != 2:  # STT_FUNC
                continue
            if strtab[st_name : st_name + 20] == target_free:
                free_val = st_value
            elif strtab[st_name : st_name + 11] == target_mi:
                mi_val = st_value
            if free_val and mi_val:
                break

        if free_val and mi_val:
            fn_type = ctypes.CFUNCTYPE(None, ctypes.c_bool)
            return fn_type(free_cpu_ptr - free_val + mi_val)
    return None


def trim_host_heap() -> bool:
    """Return unused host heap pages from PyTorch's CPU tensor allocator and
    glibc's malloc arenas back to the OS.

    On Unified Memory Architecture (UMA) platforms such as GB10, freed CPU
    tensor segments in ``libc10`` (mimalloc) and freed heap chunks in ``glibc``
    (ptmalloc) remain committed against the shared physical memory pool until
    explicitly purged.
    """
    trimmed = False
    mi_collect = _get_c10_mi_collect()
    if mi_collect is not None:
        with contextlib.suppress(Exception):
            mi_collect(True)
            trimmed = True
    malloc_trim = _get_malloc_trim()
    if malloc_trim is not None:
        with contextlib.suppress(Exception):
            malloc_trim(0)
            trimmed = True
    return trimmed


def reclaim_uma_memory(
    device: torch.types.Device | None = None,
    *,
    synchronize: bool = False,
) -> None:
    """Collect unreachable Python objects, release cached device allocator
    blocks, and return unused host heap pages to the OS.
    """
    gc.collect()
    if torch.cuda.is_available():
        if synchronize:
            if device is None:
                torch.accelerator.synchronize()
            else:
                torch.accelerator.synchronize(device)
        torch.accelerator.empty_cache()
    trim_host_heap()


def release_device_memory_under_pressure(device: torch.device) -> bool:
    """On integrated (UMA) GPUs, release caching-allocator memory back to the
    OS when system memory pressure is high. The OS may start thrashing before
    an allocation failure would trigger PyTorch's own cache release.

    Returns:
        True if memory was released.
    """
    if device.type != "cuda" or not current_platform.is_integrated_gpu(device.index):
        return False

    releasable = torch.accelerator.memory_reserved(
        device
    ) - torch.accelerator.memory_allocated(device)
    if releasable < _UMA_MIN_RELEASE_BYTES:
        return False

    # cudaMemGetInfo underreports free memory on UMA, see MemorySnapshot.measure
    mem = psutil.virtual_memory()
    if mem.available > (1 - _UMA_PRESSURE_THRESHOLD) * mem.total:
        return False

    gc.collect()
    torch.accelerator.synchronize(device)
    torch.accelerator.empty_cache()
    trim_host_heap()
    logger.debug(
        "Released %sGiB of cached device memory under memory pressure",
        format_gib(releasable),
    )
    return True


class DeviceMemoryProfiler:
    def __init__(self, device: torch.types.Device | None = None):
        self.device = device

    def current_memory_usage(self) -> float:
        # Return the memory usage in bytes.
        gc.collect()
        trim_host_heap()
        return current_platform.get_current_memory_usage(self.device)

    def __enter__(self):
        self.initial_memory = self.current_memory_usage()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self.current_memory_usage()
        self.consumed_memory = self.final_memory - self.initial_memory

        # Force garbage collection and return host arena slack to the OS.
        gc.collect()
        trim_host_heap()


@dataclass
class MemorySnapshot:
    """Memory snapshot."""

    torch_peak: int = 0
    torch_allocated: int = 0
    free_memory: int = 0
    total_memory: int = 0
    cuda_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0
    timestamp: float = 0.0

    device: torch.types.Device = None
    auto_measure: bool = True

    def __post_init__(self) -> None:
        if self.device is None:
            device_fn = current_platform.current_device
            assert device_fn is not None
            self.device_ = torch.device(device_fn())
        else:
            self.device_ = torch.device(self.device)

        if self.auto_measure:
            self.measure()

    def measure(self) -> None:
        device = self.device_

        # we measure the torch peak memory usage via allocated_bytes,
        # rather than `torch.accelerator.memory_reserved()` .
        # After `torch.accelerator.reset_peak_memory_stats()`,
        # `torch.accelerator.memory_reserved()` will keep growing, and only shrink
        # when we call `torch.accelerator.empty_cache()` or OOM happens.
        stats = torch.accelerator.memory_stats(device)
        self.torch_peak = stats.get("allocated_bytes.all.peak", 0)
        self.torch_allocated = stats.get("allocated_bytes.all.current", 0)

        self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)
        if current_platform.is_integrated_gpu(device.index):
            # On UMA (Unified Memory Architecture) platforms where CPU and
            # GPU share physical memory (e.g. GH200, DGX Spark, Jetson Orin),
            # cudaMemGetInfo underreports free memory because it does not
            # account for reclaimable OS memory (page cache, buffers).
            # Use psutil to get the true available memory.
            # https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/#estimating-total-allocatable-device-memory-on-an-integrated-gpu-device
            self.free_memory = psutil.virtual_memory().available

        self.cuda_memory = self.total_memory - self.free_memory

        # torch.accelerator.memory_reserved() is how many bytes
        # PyTorch gets from cuda (by calling cudaMalloc, etc.)
        # this is used to measure the non-torch memory usage
        self.torch_memory = torch.accelerator.memory_reserved(device)

        self.non_torch_memory = self.cuda_memory - self.torch_memory
        self.timestamp = time.time()

    def __sub__(self, other: "MemorySnapshot") -> "MemorySnapshot":
        if self.device_ != other.device_:
            raise ValueError(
                "The two snapshots should be from the same device! "
                f"Found: {self.device_} vs. {other.device_}"
            )

        return MemorySnapshot(
            torch_peak=self.torch_peak - other.torch_peak,
            torch_allocated=self.torch_allocated - other.torch_allocated,
            free_memory=self.free_memory - other.free_memory,
            total_memory=self.total_memory - other.total_memory,
            cuda_memory=self.cuda_memory - other.cuda_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
            device=self.device_,
            auto_measure=False,
        )

    def __repr__(self) -> str:
        return (
            f"torch_peak={format_gib(self.torch_peak)}GiB, "
            f"torch_allocated={format_gib(self.torch_allocated)}GiB, "
            f"free_memory={format_gib(self.free_memory)}GiB, "
            f"total_memory={format_gib(self.total_memory)}GiB, "
            f"{current_platform.device_name}_memory={format_gib(self.cuda_memory)}GiB, "
            f"torch_memory={format_gib(self.torch_memory)}GiB, "
            f"non_torch_memory={format_gib(self.non_torch_memory)}GiB, "
            f"timestamp={self.timestamp}, "
            f"auto_measure={self.auto_measure}"
        )


@dataclass
class MemoryProfilingResult:
    """Memory profiling result. All numbers are in bytes."""

    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    total_consumed: int = 0
    transient_peak_headroom: int = 0
    weights_memory: int = 0
    before_create: MemorySnapshot = field(default_factory=MemorySnapshot)
    profile_time: float = 0.0

    def __post_init__(self) -> None:
        device = self.before_create.device_

        self.before_profile = MemorySnapshot(device=device, auto_measure=False)
        self.after_profile = MemorySnapshot(device=device, auto_measure=False)

    def __repr__(self) -> str:
        return (
            f"Memory profiling takes {self.profile_time:.2f} seconds. "
            f"Total non KV cache memory: "
            f"{format_gib(self.non_kv_cache_memory)}GiB; "
            f"torch peak memory increase: "
            f"{format_gib(self.torch_peak_increase)}GiB; "
            f"total consumed (from mem_get_info): "
            f"{format_gib(self.total_consumed)}GiB; "
            f"weights memory: {format_gib(self.weights_memory)}GiB."
        )


@contextlib.contextmanager
def memory_profiling(
    baseline_snapshot: MemorySnapshot,
    weights_memory: int = 0,
) -> Generator[MemoryProfilingResult, None, None]:
    """
    Memory profiling context manager.

    baseline_snapshot: the memory snapshot before the current vLLM instance.
    weights_memory: memory used by PyTorch when loading the model weights.
        Note that, before loading the model weights, we also initialize the device
        and distributed environment, which may consume some memory. This part is not
        included in the weights_memory because PyTorch does not control it.

    The memory in one GPU can be classified into 3 categories:
    1. memory used by anything other than the current vLLM instance.
    2. memory used by torch in the current vLLM instance.
    3. memory used in the current vLLM instance, but not by torch.

    A quantitive example:

    Before creating the current vLLM instance:
        category 1: 1 GiB
        category 2: 0 GiB
        category 3: 0 GiB

    After creating the current vLLM instance and loading the model,
    (i.e. before profiling):
        category 1: 1 GiB
        category 2: 2 GiB (model weights take 2 GiB)
        category 3: 0.5 GiB (memory used by NCCL)

    During profiling (peak):
        category 1: 1 GiB
        category 2: 4 GiB (peak activation tensors take 2 GiB)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    After profiling:
        category 1: 1 GiB
        category 2: 3 GiB (after garbage-collecting activation tensors)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    In this case, non-kv cache takes 5 GiB in total, including:
    a. 2 GiB used by the model weights (category 2)
    b. 2 GiB reserved for the peak activation tensors (category 2)
    c. 1 GiB used by non-torch components (category 3)

    The memory used for loading weights (a.) is directly given from the
    argument `weights_memory`.

    The increase of `torch.accelerator.memory_stats()["allocated_bytes.all.peak"]`
    during profiling gives (b.).

    The increase of `non_torch_memory` from creating the current vLLM instance
    until after profiling to get (c.).
    """
    gc.collect()
    torch.accelerator.empty_cache()
    trim_host_heap()
    torch.accelerator.reset_peak_memory_stats(baseline_snapshot.device_)

    result = MemoryProfilingResult(
        before_create=baseline_snapshot,
        # the part of memory used for holding the model weights
        weights_memory=weights_memory,
    )

    result.before_profile.measure()

    yield result

    gc.collect()
    torch.accelerator.empty_cache()
    trim_host_heap()

    result.after_profile.measure()

    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp

    # Measure total consumption via mem_get_info() instead of
    # memory_reserved(), which goes negative when pluggable allocators
    # (e.g. cumem) bypass PyTorch's tracking.
    result.total_consumed = (
        result.before_create.free_memory - result.after_profile.free_memory
    )

    # total_consumed already covers persistent torch allocations; add only the
    # transient peak headroom to avoid double-counting.
    result.transient_peak_headroom = (
        result.after_profile.torch_peak - result.after_profile.torch_allocated
    )
    result.non_kv_cache_memory = result.total_consumed + result.transient_peak_headroom
