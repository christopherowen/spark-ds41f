# SPDX-License-Identifier: Apache-2.0
"""DGX Spark UMA memory reclaim and mlock/munlock helpers."""

from __future__ import annotations

import contextlib
import ctypes
import gc
import os
import resource
import struct
import sys
import threading
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.mem_constants import MiB_bytes

logger = init_logger(__name__)

_UMA_PRESSURE_THRESHOLD = 0.8


_UMA_MIN_RELEASE_BYTES = 512 * MiB_bytes


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


_mlock_state_lock = threading.Lock()


_mlock_fn: Any = None


_munlock_fn: Any = None


_mlock_warned: bool = False


_MIN_MEMLOCK_LIMIT = 64 << 20  # reserve tight (<64 MiB) RLIMIT_MEMLOCK for io_uring


def _ensure_mlock_fns() -> tuple[Any, Any]:
    global _mlock_fn, _munlock_fn
    if _mlock_fn is not None:
        return _mlock_fn, _munlock_fn
    with _mlock_state_lock:
        if _mlock_fn is None:
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(
                    resource.RLIMIT_MEMLOCK,
                    (resource.RLIM_INFINITY, resource.RLIM_INFINITY),
                )
            with contextlib.suppress(ValueError, OSError):
                soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
                if soft != resource.RLIM_INFINITY and (
                    hard == resource.RLIM_INFINITY or soft < hard
                ):
                    resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
            libc = ctypes.CDLL(None, use_errno=True)
            mlock = libc.mlock
            mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            mlock.restype = ctypes.c_int
            munlock = libc.munlock
            munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            munlock.restype = ctypes.c_int
            _munlock_fn = munlock
            _mlock_fn = mlock
    return _mlock_fn, _munlock_fn


def mlock_memory(addr: int, nbytes: int) -> bool:
    """Lock a host virtual address range into RAM via ``mlock(2)``."""
    global _mlock_warned
    if addr <= 0 or nbytes <= 0:
        return False
    mlock, _ = _ensure_mlock_fns()
    with contextlib.suppress(ValueError, OSError):
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if soft != resource.RLIM_INFINITY and soft < _MIN_MEMLOCK_LIMIT:
            return False
    if mlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes)) == 0:
        return True
    if not _mlock_warned:
        err = ctypes.get_errno()
        _mlock_warned = True
        logger.debug(
            "mlock(%#x, %d) failed: %s (errno=%d)",
            addr,
            nbytes,
            os.strerror(err),
            err,
        )
    return False


def munlock_memory(addr: int, nbytes: int) -> None:
    """Unlock a host virtual address range previously locked with ``mlock_memory``."""
    if addr <= 0 or nbytes <= 0:
        return
    _, munlock = _ensure_mlock_fns()
    munlock(ctypes.c_void_p(addr), ctypes.c_size_t(nbytes))


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
