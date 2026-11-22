# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the b12x project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--load-format direct``: safetensors weights read with O_DIRECT through a registered
host ring and copied straight into the model's CUDA parameters.

The loader is for unified-memory GPUs such as GB10, where the GPU reads pageable
memory through the host page tables. There a load through the page cache costs twice:

- every byte read stays in the page cache, which counts as available memory but takes
  the free memory the CUDA allocator needs for the KV cache and graphs allocated after
  loading;
- a copy from a memory-mapped checkpoint page the process has not touched can hang
  (driver R580, 64 KiB pages); ``weight_transfer.copy_weight`` copies such sources into
  anonymous memory first.

So nothing here maps the checkpoint or reads it through the page cache:

- The weights iterator yields meta tensors that carry each tensor's checkpoint range
  (``direct_io.DirectWeightSession``). Model weight loaders route them into parameter
  views as usual, and ``weight_transfer`` hands each copy to the session, which
  records it and later reads it with O_DIRECT through a 16 MiB registered ring into
  its destination. Scalars and the model's declared metadata are read into CPU
  tensors, and copies the session declines go to Torch.
- Tensors the model reads from the checkpoint files while serving
  (``checkpoint_file_weight_filter``, such as DeepSeek-V4.1's Engram tables) never pass
  through the reader: the iterator yields vLLM's file-source descriptors for them
  (``weight_utils.file_source_tensor``), in name order with the file's other tensors.

The device must report pageable memory access through the host page tables; the
loader refuses others, since it has no GPU-direct-storage path. The native helper
(``direct_io/_native.py``) is built on first use. ``model_loader_extra_config`` takes
``io_threads``, the reads in flight (1-16, default 8). The loader logs one summary line
once the weights are loaded.

Ported from local-inference-lab/b12x b12x/integration/vllm/loader.py at 236ddff04a5f
(Apache-2.0) and vLLM's former copy of it (``b12x_loader.py``, the ``b12x`` load
format). Changed: no GDS transport, shared read group, IPC-exportable allocations or
progress panel.
"""

from contextlib import contextmanager
import dataclasses
import gc
import heapq
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.logger import init_logger
from vllm.utils.mem_utils import reclaim_uma_memory, trim_host_heap
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.direct_io import (
    DirectWeightSession,
    capabilities,
)
from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight
from vllm.model_executor.weight_transfer import weight_transfer
from vllm.v1.worker import display_carveout

logger = init_logger(__name__)

_ARENA_ALIGN_BYTES = 256
_PRESERVE_STORAGE_MAX_BYTES = 1 << 20
_TOUCH_STRIDE_BYTES = 1 << 21
_SPONGE_CHUNK_BYTES = 1 << 27
_SPONGE_MIN_CHUNK_BYTES = 1 << 21
_SPONGE_HEADROOM_BYTES = 12 << 30
_SPONGE_POST_LOAD_HEADROOM_BYTES = 16 << 30
_SPONGE_POST_LOAD_MIN_HIGH_BYTES = 256 << 20
_SPONGE_MIN_MODEL_BYTES = 256 << 20
_SPONGE_POST_LOAD_SCRATCH_BYTES = 512 << 20
_SPONGE_RESERVE_HIGH_MAX_BYTES = 512 << 20
_NVIDIA_MIN_SYS_ALLOC_BYTES = 65536
_GPU_HUGEPAGE_BYTES = 1 << 21
_HIGH_ORDER_SPONGE_TARGETS_64K = tuple(65536 << order for order in range(13, 5, -1))
_HELD_SPONGE_STACK: list[list[torch.nn.Module]] = []
_PREALLOCATED_CUDA_POOL_BYTES: int = 0
_PREALLOCATED_CUDA_POOL_FRONT_CLEAN: bool = True
_PREALLOCATED_CUDA_POOL_RESIZE_CB = None
_DEFER_ARENA_WEIGHT_ALLOC: int = 0


def is_deferring_arena_weight_alloc() -> bool:
    """True while ``DirectModelLoader.create_model`` defers large uninitialized
    parameter allocations to ``meta`` so ``bind_weight_arena`` allocates them
    once under the buddy sponge instead of allocating and freeing them twice."""
    return _DEFER_ARENA_WEIGHT_ALLOC > 0


@contextmanager
def defer_arena_weight_alloc():
    global _DEFER_ARENA_WEIGHT_ALLOC
    _DEFER_ARENA_WEIGHT_ALLOC += 1
    try:
        yield
    finally:
        _DEFER_ARENA_WEIGHT_ALLOC -= 1


def set_preallocated_cuda_pool_bytes(
    num_bytes: int, *, front_clean: bool = True, resize_cb=None
) -> None:
    """Record whether Worker.init_device pre-allocated a single contiguous CUDA
    caching-allocator segment so bind_weight_arena skips redundant buddy soaking
    and (when resize_cb is provided) sizes the permanent pool to the model."""
    global _PREALLOCATED_CUDA_POOL_BYTES
    global _PREALLOCATED_CUDA_POOL_FRONT_CLEAN
    global _PREALLOCATED_CUDA_POOL_RESIZE_CB
    _PREALLOCATED_CUDA_POOL_BYTES = max(0, int(num_bytes))
    _PREALLOCATED_CUDA_POOL_FRONT_CLEAN = bool(front_clean)
    _PREALLOCATED_CUDA_POOL_RESIZE_CB = resize_cb


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def parse_buddyinfo_order13(
    text: str, page_bytes: int | None = None
) -> tuple[int, int]:
    """Return ``(sub_order13_bytes, order13_plus_bytes)`` from ``/proc/buddyinfo``.

    On 64 KiB-page kernels (such as GB10), buddy order 13 is a 512 MiB contiguous
    block. Returns ``(0, 0)`` when ``page_bytes`` is not 64 KiB.
    """
    if page_bytes is None:
        try:
            page_bytes = os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, OSError, ValueError):
            return 0, 0
    if page_bytes != 65536:
        return 0, 0
    sub_order13 = 0
    order13_plus = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) <= 4 or parts[0] != "Node" or parts[2] != "zone":
            continue
        for order, raw in enumerate(parts[4:]):
            try:
                count = int(raw)
            except ValueError:
                continue
            block_bytes = count * (page_bytes << order)
            if order < 13:
                sub_order13 += block_bytes
            else:
                order13_plus += block_bytes
    return sub_order13, order13_plus


def parse_buddyinfo_blocks(
    text: str, page_bytes: int | None = None
) -> dict[int, int]:
    """Return ``{block_size_bytes: total_free_bytes}`` across all zones in
    ``/proc/buddyinfo`` for any base OS page size (both 4 KiB and 64 KiB)."""
    if page_bytes is None:
        try:
            page_bytes = os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, OSError, ValueError):
            return {}
    if page_bytes <= 0:
        return {}
    blocks: dict[int, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) <= 4 or parts[0] != "Node" or parts[2] != "zone":
            continue
        for order, raw in enumerate(parts[4:]):
            try:
                count = int(raw)
            except ValueError:
                continue
            if count <= 0:
                continue
            bsz = page_bytes << order
            blocks[bsz] = blocks.get(bsz, 0) + count * bsz
    return blocks


def select_buddy_sponge_target(
    text: str,
    *,
    required_bytes: int,
    page_bytes: int | None = None,
    min_alloc_bytes: int = _NVIDIA_MIN_SYS_ALLOC_BYTES,
) -> tuple[int, int, int]:
    """Choose ``(target_block_bytes, sub_target_bytes, at_or_above_target_bytes)``.

    ``nvidia.ko`` (`nv_alloc_system_pages`) satisfies `cuMemCreate` allocations on
    UMA by draining smaller buddy orders from ``64 KiB`` upward before splitting
    higher-order physical blocks. Low-order physical fragments self-conflict across
    GB10's physically-indexed 24 MiB L2 slices and DRAM banks.

    1. On 64 KiB kernels, walk descending high-order targets from ``512 MiB``
       (Order 13) down to ``4 MiB`` (Order 6) and choose the largest target whose
       cumulative free bytes at or above that order cover
       ``max(required_bytes, _SPONGE_CHUNK_BYTES)``.
    2. Otherwise, pick the highest block size in ``(2 MiB, 1 MiB, 512 KiB)`` whose
       cumulative free bytes at or above that order cover ``min(required_bytes, 512 MiB)``,
       counting only ``>= 64 KiB`` blocks in ``sub_target_bytes`` (the orders ``nvidia.ko``
       consumes ahead of the target).
    """
    blocks = parse_buddyinfo_blocks(text, page_bytes=page_bytes)
    if not blocks:
        return 0, 0, 0

    eff_page_bytes = page_bytes
    if eff_page_bytes is None:
        try:
            eff_page_bytes = os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, OSError, ValueError):
            eff_page_bytes = 0

    if eff_page_bytes == 65536:
        full_cover = max(required_bytes, _SPONGE_CHUNK_BYTES)
        for target_bsz in _HIGH_ORDER_SPONGE_TARGETS_64K:
            high_bytes = sum(tot for bsz, tot in blocks.items() if bsz >= target_bsz)
            sub_bytes = sum(
                tot for bsz, tot in blocks.items() if min_alloc_bytes <= bsz < target_bsz
            )
            if high_bytes >= full_cover and sub_bytes > 0:
                return target_bsz, sub_bytes, high_bytes

    min_high_needed = min(max(required_bytes // 4, _GPU_HUGEPAGE_BYTES), 512 << 20)
    for target_bsz in (_GPU_HUGEPAGE_BYTES, 1 << 20, 512 << 10):
        high_bytes = sum(tot for bsz, tot in blocks.items() if bsz >= target_bsz)
        sub_bytes = sum(
            tot for bsz, tot in blocks.items() if min_alloc_bytes <= bsz < target_bsz
        )
        if high_bytes >= min_high_needed and sub_bytes > 0:
            return target_bsz, sub_bytes, high_bytes
    return 0, 0, 0


def _available_uma_or_gpu_bytes(
    device: torch.device,
    *,
    meminfo_path: Path | None = Path("/proc/meminfo"),
) -> int:
    """Return usable free bytes before KV allocation.

    On integrated UMA GPUs (GB10), ``cudaMemGetInfo`` reports ``MemFree`` and omits
    reclaimable clean VFS page cache, underreporting usable memory by tens of GiB
    compared to ``/proc/meminfo`` ``MemAvailable`` (the metric vLLM's ``MemorySnapshot``
    and ``memguard`` enforce).
    """
    gpu_free = 0
    try:
        gpu_free, _ = torch.accelerator.get_memory_info(device)
    except Exception:  # noqa: BLE001
        gpu_free = 0
    if meminfo_path is not None:
        try:
            if capabilities(device.index or 0).get("integrated", 0):
                for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("MemAvailable:"):
                        avail_kib = int(line.split()[1])
                        return avail_kib * 1024
        except Exception:  # noqa: BLE001
            pass
    return gpu_free


def _try_thp_buddy_defrag(
    deficit_bytes: int,
    *,
    avail_bytes: int,
    headroom_bytes: int = _SPONGE_HEADROOM_BYTES,
) -> None:
    # Best-effort unprivileged buddy compaction via MADV_HUGEPAGE page faults
    # when transparent_hugepage/defrag is set to madvise on 4 KiB kernels.
    import ctypes
    import mmap

    if deficit_bytes < _GPU_HUGEPAGE_BYTES:
        return
    try:
        page_sz = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return
    if page_sz != 4096:
        return
    max_safe = avail_bytes - headroom_bytes - (2 << 30)
    span = min(deficit_bytes, max_safe, 32 << 30)
    span = (span // _GPU_HUGEPAGE_BYTES) * _GPU_HUGEPAGE_BYTES
    if span < _GPU_HUGEPAGE_BYTES:
        return
    mm = None
    try:
        libc = ctypes.CDLL("libc.so.6")
        mm = mmap.mmap(
            -1,
            span + _GPU_HUGEPAGE_BYTES,
            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        aligned = _align_up(base, _GPU_HUGEPAGE_BYTES)
        if (
            libc.madvise(
                ctypes.c_void_p(aligned),
                ctypes.c_size_t(span),
                ctypes.c_int(14),  # MADV_HUGEPAGE on Linux
            )
            == 0
        ):
            mv = memoryview(mm)[aligned - base : aligned - base + span]
            mv[::_GPU_HUGEPAGE_BYTES] = b"\x01" * (span // _GPU_HUGEPAGE_BYTES)
            del mv
    except Exception:  # noqa: BLE001
        pass
    finally:
        if mm is not None:
            try:
                mm.close()
            except Exception:  # noqa: BLE001
                pass


def _allocate_order13_sponge(
    device: torch.device,
    *,
    required_bytes: int,
    buddyinfo_path: Path = Path("/proc/buddyinfo"),
    meminfo_path: Path | None = Path("/proc/meminfo"),
    page_bytes: int | None = None,
    headroom_bytes: int = _SPONGE_HEADROOM_BYTES,
    chunk_bytes: int = _SPONGE_CHUNK_BYTES,
) -> list[torch.Tensor]:
    """Temporarily soak fragmented sub-hugepage buddy blocks (``64 KiB .. 1 MiB``,
    or ``< 512 MiB`` when Order-13 blocks exist) so subsequent model weight and
    post-load allocations draw from ``>= 1-2 MiB`` (or ``512 MiB``) contiguous
    physical blocks without exhausting ``>= 64 KiB`` buddy pools into order-4
    page-cache slowpath reclaim."""
    try:
        text = buddyinfo_path.read_text(encoding="utf-8")
    except OSError:
        return []
    target_bsz, sub_target_bytes, high_target_bytes = select_buddy_sponge_target(
        text, required_bytes=required_bytes, page_bytes=page_bytes
    )
    if target_bsz == 0:
        return []
    eff_chunk = min(chunk_bytes, max(_SPONGE_MIN_CHUNK_BYTES, sub_target_bytes))
    eff_chunk = max(
        _SPONGE_MIN_CHUNK_BYTES,
        (eff_chunk // _SPONGE_MIN_CHUNK_BYTES) * _SPONGE_MIN_CHUNK_BYTES,
    )
    if sub_target_bytes < eff_chunk or high_target_bytes < _SPONGE_MIN_CHUNK_BYTES:
        return []
    avail_bytes = _available_uma_or_gpu_bytes(device, meminfo_path=meminfo_path)
    if avail_bytes <= 0:
        return []
    if (
        buddyinfo_path == Path("/proc/buddyinfo")
        and target_bsz == _GPU_HUGEPAGE_BYTES
        and high_target_bytes < required_bytes
    ):
        _try_thp_buddy_defrag(
            required_bytes - high_target_bytes,
            avail_bytes=avail_bytes,
            headroom_bytes=headroom_bytes,
        )
        try:
            text = buddyinfo_path.read_text(encoding="utf-8")
            target_bsz, sub_target_bytes, high_target_bytes = (
                select_buddy_sponge_target(
                    text, required_bytes=required_bytes, page_bytes=page_bytes
                )
            )
            if target_bsz == 0 or sub_target_bytes < eff_chunk:
                return []
        except OSError:
            pass
    # Cap soak so (sub + high) keeps >= required_bytes + reserve_high in >=64K buddy
    # lists, preventing __alloc_pages_slowpath(order=4) 64 KiB fallback when
    # high_target_bytes < required_bytes while still draining all 64K..512K blocks.
    soak_goal = sub_target_bytes + (
        eff_chunk if target_bsz <= _GPU_HUGEPAGE_BYTES else 0
    )
    eff_page_bytes = page_bytes
    if eff_page_bytes is None:
        try:
            eff_page_bytes = os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, OSError, ValueError):
            eff_page_bytes = 0
    # 4 KiB kernels spend ~2% (~20 MiB/GiB) of additional non-reclaimable RAM on
    # CPU/GPU PTEs and UVM metadata across sponge + weight mappings.
    usable_after_headroom = avail_bytes - headroom_bytes
    if eff_page_bytes == 4096 and headroom_bytes > 0 and usable_after_headroom > 0:
        usable_after_headroom = (usable_after_headroom * 48) // 49
    reserve_high = min(_SPONGE_RESERVE_HIGH_MAX_BYTES, high_target_bytes // 4)
    buddy_cap = (sub_target_bytes + high_target_bytes) - required_bytes - reserve_high
    mem_cap = usable_after_headroom - required_bytes
    budget = min(soak_goal, buddy_cap, mem_cap)
    num_chunks = int(budget // eff_chunk)
    if num_chunks <= 0:
        return []
    live_buddy = buddyinfo_path == Path("/proc/buddyinfo")
    t0 = time.monotonic()
    sponge: list[torch.Tensor] = []
    try:
        for idx in range(num_chunks):
            chunk = torch.empty((eff_chunk,), dtype=torch.uint8, device=device)
            if eff_chunk >= _TOUCH_STRIDE_BYTES:
                chunk[::_TOUCH_STRIDE_BYTES].zero_()
            elif eff_chunk > 0:
                chunk[:1].zero_()
            sponge.append(chunk)
            if live_buddy and target_bsz < (65536 << 13):
                # Linux serves nvidia.ko GFP_KERNEL (__GFP_COMP) allocations from
                # MIGRATE_UNMOVABLE sub-target lists first, then >= target_bsz
                # Unmovable blocks BEFORE stealing Movable sub-target lists.
                # Stop the instant >= target_bsz free bytes drop by one chunk so we
                # never waste >= 2 MiB blocks inside the sponge!
                if idx == 0 or (idx + 1) % 2 == 0:
                    torch.accelerator.synchronize(device)
                    try:
                        cur_blocks = parse_buddyinfo_blocks(
                            buddyinfo_path.read_text(encoding="utf-8"),
                            page_bytes=page_bytes,
                        )
                        cur_high = sum(
                            tot for bsz, tot in cur_blocks.items() if bsz >= target_bsz
                        )
                        if high_target_bytes - cur_high >= eff_chunk:
                            # Pop the single chunk that crossed into >= target_bsz so
                            # even that block returns to the high-order pool.
                            sponge.pop()
                            torch.accelerator.empty_cache()
                            high_target_bytes = cur_high + eff_chunk
                            break
                    except OSError:
                        pass
        if sponge:
            torch.accelerator.synchronize(device)
            logger.info(
                "Buddy hugepage sponge soaked %.1f MiB (%d chunks) in %.2f s so %.1f MiB "
                "of weights draw >= %.1f MiB contiguous physical blocks (%.1f MiB available)",
                (len(sponge) * eff_chunk) / (1 << 20),
                len(sponge),
                time.monotonic() - t0,
                required_bytes / (1 << 20),
                target_bsz / (1 << 20),
                high_target_bytes / (1 << 20),
            )
    except torch.OutOfMemoryError:
        pass
    return sponge


def _trim_sponge_for_post_load(
    sponge: list[torch.Tensor],
    device: torch.device,
    *,
    target_bsz: int,
    buddyinfo_path: Path = Path("/proc/buddyinfo"),
    meminfo_path: Path | None = Path("/proc/meminfo"),
    page_bytes: int | None = None,
    post_load_headroom_bytes: int = _SPONGE_POST_LOAD_HEADROOM_BYTES,
    min_high_bytes: int = _SPONGE_POST_LOAD_MIN_HIGH_BYTES,
) -> list[torch.Tensor]:
    """Release or trim the sponge after arena + MoE allocation when either no
    ``>= target_bsz`` blocks remain to protect or ``MemAvailable`` would fall
    below ``post_load_headroom_bytes`` during ``load_weights`` / post-load."""
    if not sponge:
        return sponge
    chunk_bytes = sponge[0].numel() * sponge[0].element_size()
    initial_mib = (len(sponge) * chunk_bytes) / (1 << 20)
    if target_bsz > 0:
        try:
            cur_blocks = parse_buddyinfo_blocks(
                buddyinfo_path.read_text(encoding="utf-8"),
                page_bytes=page_bytes,
            )
            cur_high = sum(
                tot for bsz, tot in cur_blocks.items() if bsz >= target_bsz
            )
            if cur_high < min_high_bytes:
                logger.info(
                    "Releasing %.1f MiB buddy sponge before weight load "
                    "(%.1f MiB of >= %.1f MiB blocks remain; weights already "
                    "claimed the high-order pool)",
                    initial_mib,
                    cur_high / (1 << 20),
                    target_bsz / (1 << 20),
                )
                sponge.clear()
                torch.accelerator.empty_cache()
                return sponge
        except OSError:
            pass
    avail_after = _available_uma_or_gpu_bytes(device, meminfo_path=meminfo_path)
    if 0 < avail_after < post_load_headroom_bytes and chunk_bytes > 0:
        deficit = post_load_headroom_bytes - avail_after
        drop_chunks = min(len(sponge), (deficit + chunk_bytes - 1) // chunk_bytes)
        if drop_chunks >= len(sponge):
            logger.info(
                "Releasing %.1f MiB buddy sponge before weight load to preserve "
                "%.1f MiB post-load MemAvailable headroom (%.1f MiB available)",
                initial_mib,
                post_load_headroom_bytes / (1 << 20),
                avail_after / (1 << 20),
            )
            sponge.clear()
        elif drop_chunks > 0:
            del sponge[-drop_chunks:]
            logger.info(
                "Trimmed buddy sponge by %.1f MiB (retaining %.1f MiB) to preserve "
                "%.1f MiB post-load MemAvailable headroom",
                (drop_chunks * chunk_bytes) / (1 << 20),
                (len(sponge) * chunk_bytes) / (1 << 20),
                post_load_headroom_bytes / (1 << 20),
            )
        torch.accelerator.empty_cache()
    return sponge


def release_weight_arena_sponge(
    model: torch.nn.Module,
    device: torch.device | None = None,
) -> None:
    """Release a retained ``model._direct_buddy_sponge`` after post-load processing."""
    sponge = getattr(model, "_direct_buddy_sponge", None)
    if hasattr(model, "_direct_buddy_sponge"):
        delattr(model, "_direct_buddy_sponge")
    if hasattr(model, "_direct_buddy_sponge_target_bsz"):
        delattr(model, "_direct_buddy_sponge_target_bsz")
    had_sponge = bool(sponge)
    if sponge:
        sponge.clear()
    reclaim_uma_memory(device, synchronize=had_sponge and device is not None and device.type == "cuda")


def defer_weight_arena_sponge_release(model: torch.nn.Module) -> bool:
    """Defer releasing ``model._direct_buddy_sponge`` when a ``hold_weight_arena_sponge``
    scope is active (for example, to keep the target's buddy sponge pinned until a
    speculative drafter finishes loading)."""
    if not _HELD_SPONGE_STACK or not hasattr(model, "_direct_buddy_sponge"):
        return False
    held = _HELD_SPONGE_STACK[-1]
    if model not in held:
        held.append(model)
    return True


def trim_held_weight_arena_sponges(
    device: torch.device,
    *,
    buddyinfo_path: Path = Path("/proc/buddyinfo"),
    meminfo_path: Path | None = Path("/proc/meminfo"),
    page_bytes: int | None = None,
    post_load_headroom_bytes: int = _SPONGE_POST_LOAD_HEADROOM_BYTES,
) -> None:
    """Trim or release any sponges held in the active ``hold_weight_arena_sponge``
    scope so ``MemAvailable`` stays at or above ``post_load_headroom_bytes`` before
    a subsequent model (such as a speculative drafter) allocates weights."""
    if not _HELD_SPONGE_STACK:
        return
    for model in list(_HELD_SPONGE_STACK[-1]):
        sponge = getattr(model, "_direct_buddy_sponge", None)
        if not sponge:
            continue
        target_bsz = int(getattr(model, "_direct_buddy_sponge_target_bsz", 0))
        remaining = _trim_sponge_for_post_load(
            sponge,
            device,
            target_bsz=target_bsz,
            buddyinfo_path=buddyinfo_path,
            meminfo_path=meminfo_path,
            page_bytes=page_bytes,
            post_load_headroom_bytes=post_load_headroom_bytes,
        )
        if not remaining:
            if hasattr(model, "_direct_buddy_sponge"):
                delattr(model, "_direct_buddy_sponge")
            if hasattr(model, "_direct_buddy_sponge_target_bsz"):
                delattr(model, "_direct_buddy_sponge_target_bsz")


@contextmanager
def hold_weight_arena_sponge(device: torch.device | None = None):
    """Hold retained buddy sponges across sequential model loads (target + drafter)
    and release them together on exit."""
    held: list[torch.nn.Module] = []
    _HELD_SPONGE_STACK.append(held)
    try:
        yield held
    finally:
        _HELD_SPONGE_STACK.pop()
        had_sponge = False
        for model in held:
            sponge = getattr(model, "_direct_buddy_sponge", None)
            if hasattr(model, "_direct_buddy_sponge"):
                delattr(model, "_direct_buddy_sponge")
            if hasattr(model, "_direct_buddy_sponge_target_bsz"):
                delattr(model, "_direct_buddy_sponge_target_bsz")
            if sponge:
                had_sponge = True
                sponge.clear()
        if had_sponge:
            reclaim_uma_memory(
                device,
                synchronize=device is not None and device.type == "cuda",
            )


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = acc
        acc *= shape[i]
    return tuple(strides)


def _refresh_module_weight_views(model: torch.nn.Module) -> None:
    """Rebind cached tensor slice views (e.g. ``shard_weight``) after ``param.set_``."""
    for module in model.modules():
        refresh = getattr(module, "_refresh_views", None)
        if callable(refresh):
            refresh()


def bind_weight_arena(
    model: torch.nn.Module,
    device: torch.device,
    *,
    order13_sponge: bool = True,
    sponge_min_model_bytes: int = _SPONGE_MIN_MODEL_BYTES,
    buddyinfo_path: Path = Path("/proc/buddyinfo"),
    page_bytes: int | None = None,
    retain_sponge: bool = False,
) -> torch.Tensor | None:
    """Pack persistent CUDA model parameters into a 256-byte-aligned contiguous
    ``uint8`` arena (``model._direct_weight_arena``) before ``load_weights``.

    Parameters on ``FusedMoEMethodBase`` modules (which ``process_weights_after_loading``
    replaces layer by layer via ``replace_parameter``) keep individual storages so their
    pre-tile weights can be freed during post-processing, but are re-allocated largest
    first under the same Order-13 buddy sponge. Display carve-out tensors and file-backed
    ``load_view`` parameters are left untouched.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    moe_storage_ptrs: set[int] = set()
    deferred_moe_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(getattr(module, "quant_method", None), FusedMoEMethodBase):
            for param in module.parameters(recurse=True):
                data = param.data
                if (
                    getattr(param, "_direct_deferred_alloc", False)
                    and data.device.type == "cuda"
                    and data.numel() > 0
                ):
                    deferred_moe_param_ids.add(id(param))
                elif (
                    data.device.type == "cuda"
                    and data.numel() > 0
                    and data.untyped_storage().nbytes() > 0
                ):
                    moe_storage_ptrs.add(data.untyped_storage().data_ptr())

    arena_groups: dict[
        int,
        tuple[
            int,
            list[
                tuple[
                    torch.nn.Parameter, int, int, tuple[int, ...], tuple[int, ...]
                ]
            ],
        ],
    ] = {}
    moe_groups: dict[
        int,
        tuple[
            int,
            list[
                tuple[
                    torch.nn.Parameter, int, int, tuple[int, ...], tuple[int, ...]
                ]
            ],
        ],
    ] = {}
    for param in model.parameters():
        if hasattr(param, "load_view"):
            continue
        data = param.data
        if (
            getattr(param, "_direct_deferred_alloc", False)
            and data.device.type == "cuda"
            and data.numel() > 0
        ):
            shape = tuple(data.shape)
            nbytes = data.numel() * data.element_size()
            ptr = -id(param)
            target = (
                moe_groups if id(param) in deferred_moe_param_ids else arena_groups
            )
            target[ptr] = (
                nbytes,
                [
                    (
                        param,
                        data.element_size(),
                        0,
                        shape,
                        _contiguous_strides(shape),
                    )
                ],
            )
            continue
        if data.device.type != "cuda" or data.numel() == 0:
            continue
        if any(display_carveout._holds(c, data) for c in display_carveout._carveouts):
            continue
        storage = data.untyped_storage()
        nbytes = storage.nbytes()
        if nbytes == 0:
            continue
        ptr = storage.data_ptr()
        target = moe_groups if ptr in moe_storage_ptrs else arena_groups
        if ptr not in target:
            target[ptr] = (nbytes, [])
        target[ptr][1].append(
            (
                param,
                data.element_size(),
                data.storage_offset(),
                tuple(data.shape),
                tuple(data.stride()),
            )
        )

    # Drop the final loop iteration's local references to the last parameter's
    # CUDA storage (e.g. lm_head.weight) before detaching & freeing scratch pools.
    param = data = storage = None  # type: ignore[assignment]
    if not arena_groups and not moe_groups:
        return None

    arena_bytes = sum(
        _align_up(nbytes, _ARENA_ALIGN_BYTES) for nbytes, _ in arena_groups.values()
    )
    moe_bytes = sum(nbytes for nbytes, _ in moe_groups.values())
    total_bytes = arena_bytes + moe_bytes

    cpu_snapshots: dict[int, torch.Tensor] = {}
    empty_storage = torch.empty((0,), dtype=torch.uint8, device=device).untyped_storage()
    with torch.no_grad():
        for groups in (arena_groups, moe_groups):
            for ptr, (nbytes, views) in groups.items():
                if ptr < 0:
                    for param, _, _, _, _ in views:
                        param.set_(empty_storage, 0, (0,), (1,))
                    continue
                if nbytes <= _PRESERVE_STORAGE_MAX_BYTES:
                    orig_storage = views[0][0].data.untyped_storage()
                    src = torch.empty((0,), dtype=torch.uint8, device=device).set_(
                        orig_storage
                    )
                    cpu_copy = torch.empty((nbytes,), dtype=torch.uint8, device="cpu")
                    cpu_copy.copy_(src)
                    cpu_snapshots[ptr] = cpu_copy
                    del src, orig_storage
                for param, _, _, _, _ in views:
                    param.set_(empty_storage, 0, (0,), (1,))
        _refresh_module_weight_views(model)
    max_moe_bytes = max((nbytes for nbytes, _ in moe_groups.values()), default=0)

    # One-shot model-exact sizing of the pre-allocated contiguous CUDA pool.
    global _PREALLOCATED_CUDA_POOL_RESIZE_CB
    resize_cb = _PREALLOCATED_CUDA_POOL_RESIZE_CB
    if (
        resize_cb is not None
        and _PREALLOCATED_CUDA_POOL_BYTES > 0
        and device.type == "cuda"
        and torch.cuda.is_available()
    ):
        _PREALLOCATED_CUDA_POOL_RESIZE_CB = None
        resize_cb(total_bytes, max_moe_bytes)

    cached_free_bytes = 0
    if (
        _PREALLOCATED_CUDA_POOL_BYTES > 0
        and device.type == "cuda"
        and torch.cuda.is_available()
    ):
        cached_free_bytes = max(
            0,
            int(torch.cuda.memory_reserved(device))
            - int(torch.cuda.memory_allocated(device)),
        )
    use_preallocated_pool = (
        _PREALLOCATED_CUDA_POOL_BYTES > 0
        and total_bytes >= sponge_min_model_bytes
        and cached_free_bytes >= total_bytes
    )
    if not use_preallocated_pool:
        torch.accelerator.empty_cache()
    post_load_allowance = max_moe_bytes + min(
        _SPONGE_POST_LOAD_SCRATCH_BYTES,
        max(total_bytes // 4, _GPU_HUGEPAGE_BYTES),
    )
    live_buddy = buddyinfo_path == Path("/proc/buddyinfo")
    avail_pre_sponge = (
        _available_uma_or_gpu_bytes(device) if live_buddy and retain_sponge else 0
    )
    can_retain_post_load = retain_sponge and (
        not live_buddy
        or avail_pre_sponge - total_bytes
        >= _SPONGE_POST_LOAD_HEADROOM_BYTES + post_load_allowance
    )
    sponge_required_bytes = (
        total_bytes + post_load_allowance if can_retain_post_load else total_bytes
    )
    sponge_target_bsz = 0
    if use_preallocated_pool:
        logger.info(
            "Pre-allocated CUDA pool has %.1f MiB cached free for %.1f MiB of "
            "weights; skipping buddy sponge (front_clean=%s)",
            cached_free_bytes / (1 << 20),
            total_bytes / (1 << 20),
            _PREALLOCATED_CUDA_POOL_FRONT_CLEAN,
        )
        sponge: list[torch.Tensor] = []
    else:
        if order13_sponge and total_bytes >= sponge_min_model_bytes:
            try:
                sponge_target_bsz, _, _ = select_buddy_sponge_target(
                    buddyinfo_path.read_text(encoding="utf-8"),
                    required_bytes=sponge_required_bytes,
                    page_bytes=page_bytes,
                )
            except OSError:
                sponge_target_bsz = 0
        sponge = (
            _allocate_order13_sponge(
                device,
                required_bytes=sponge_required_bytes,
                buddyinfo_path=buddyinfo_path,
                page_bytes=page_bytes,
            )
            if order13_sponge and total_bytes >= sponge_min_model_bytes
            else []
        )
    arena: torch.Tensor | None = None
    kept_sponge = False
    try:
        with torch.no_grad():
            def _alloc_arena() -> torch.Tensor | None:
                if arena_bytes <= 0:
                    return None
                buf = torch.empty((arena_bytes,), dtype=torch.uint8, device=device)
                if arena_bytes >= _TOUCH_STRIDE_BYTES:
                    buf[::_TOUCH_STRIDE_BYTES].zero_()
                arena_storage = buf.untyped_storage()
                offset = 0
                for ptr, (nbytes, views) in arena_groups.items():
                    if ptr in cpu_snapshots:
                        buf[offset : offset + nbytes].copy_(cpu_snapshots.pop(ptr))
                    for param, elem_size, orig_offset, shape, stride in views:
                        param.set_(
                            arena_storage,
                            (offset // elem_size) + orig_offset,
                            shape,
                            stride,
                        )
                        if hasattr(param, "_direct_deferred_alloc"):
                            delattr(param, "_direct_deferred_alloc")
                    offset += _align_up(nbytes, _ARENA_ALIGN_BYTES)
                model._direct_weight_arena = buf
                return buf

            def _alloc_moe() -> None:
                for ptr, (nbytes, views) in sorted(
                    moe_groups.items(), key=lambda item: item[1][0], reverse=True
                ):
                    moe_buf = torch.empty((nbytes,), dtype=torch.uint8, device=device)
                    if nbytes >= _TOUCH_STRIDE_BYTES:
                        moe_buf[::_TOUCH_STRIDE_BYTES].zero_()
                    if ptr in cpu_snapshots:
                        moe_buf.copy_(cpu_snapshots.pop(ptr))
                    moe_storage = moe_buf.untyped_storage()
                    for param, _, orig_offset, shape, stride in views:
                        param.set_(moe_storage, orig_offset, shape, stride)
                        if hasattr(param, "_direct_deferred_alloc"):
                            delattr(param, "_direct_deferred_alloc")
                    del moe_buf, moe_storage

            if use_preallocated_pool and _PREALLOCATED_CUDA_POOL_FRONT_CLEAN:
                # Front of the pre-allocated segment is pure 512 MiB order-13
                # blocks: place the dense arena at the aligned front first.
                arena = _alloc_arena()
                _alloc_moe()
            else:
                # Either no pre-allocated pool or unsoaked Unmovable sub-O13
                # remains at offset 0: place temporary pre-tile MoE storages
                # first so the permanent dense arena lands in the 512 MiB band.
                _alloc_moe()
                arena = _alloc_arena()

            _refresh_module_weight_views(model)

        if sponge or total_bytes >= _TOUCH_STRIDE_BYTES:
            torch.accelerator.synchronize(device)
        if retain_sponge and sponge and live_buddy:
            sponge = _trim_sponge_for_post_load(
                sponge,
                device,
                target_bsz=sponge_target_bsz,
                buddyinfo_path=buddyinfo_path,
                page_bytes=page_bytes,
            )
        if retain_sponge and sponge:
            model._direct_buddy_sponge = sponge
            model._direct_buddy_sponge_target_bsz = sponge_target_bsz
            kept_sponge = True
    finally:
        cpu_snapshots.clear()
        if not kept_sponge:
            if sponge:
                sponge.clear()
            reclaim_uma_memory(device, synchronize=False)
        else:
            gc.collect()
            trim_host_heap()

    return arena


class DirectModelLoader(DefaultModelLoader):
    """Route checkpoint views through an asynchronous O_DIRECT read ring."""

    def __init__(self, load_config):
        options = dict(load_config.model_loader_extra_config)
        self.io_threads = options.pop("io_threads", 8)
        self.weight_arena = bool(options.pop("weight_arena", True))
        self.order13_sponge = bool(options.pop("order13_sponge", True))
        if options.get("enable_multithread_load"):
            raise ValueError("the direct loader routes checkpoint shards in one thread")
        if load_config.safetensors_load_strategy not in (None, "lazy"):
            raise ValueError(
                "the direct loader's O_DIRECT reads use no safetensors read strategy"
            )
        super().__init__(
            dataclasses.replace(
                load_config,
                load_format="safetensors",
                model_loader_extra_config=options,
            )
        )
        self._session = None

    def create_model(self, vllm_config, model_config, prefix=""):
        device = torch.device(
            self.load_config.device or vllm_config.device_config.device
        )
        if self.weight_arena and device.type == "cuda":
            with defer_arena_weight_alloc():
                model = super().create_model(vllm_config, model_config, prefix)
            bind_weight_arena(
                model,
                device,
                order13_sponge=self.order13_sponge,
                retain_sponge=True,
            )
            return model
        model = super().create_model(vllm_config, model_config, prefix)
        if self.weight_arena:
            bind_weight_arena(
                model,
                device,
                order13_sponge=self.order13_sponge,
                retain_sponge=True,
            )
        return model

    def load_model(self, vllm_config, model_config, prefix=""):
        device = torch.device(
            self.load_config.device or vllm_config.device_config.device
        )
        if device.type != "cuda":
            raise ValueError("the direct loader requires a CUDA device")
        index = (
            torch.accelerator.current_device_index()
            if device.index is None
            else device.index
        )
        if not capabilities(index)["host_page_tables"]:
            raise RuntimeError(
                f"--load-format direct needs a GPU that reads pageable memory through "
                f"the host page tables, such as GB10, and CUDA device {index} does "
                f"not; it has no GPU-direct-storage path"
            )
        with (
            DirectWeightSession(index, io_threads=self.io_threads) as session,
            weight_transfer(session),
        ):
            self._session = session
            try:
                model = super().load_model(vllm_config, model_config, prefix)
                stats = session.stats()
            finally:
                self._session = None
        reclaim_uma_memory(device, synchronize=True)
        seconds = (
            self.counter_after_loading_weights - self.counter_before_loading_weights
        )
        payload = stats["payload_bytes"] / 1e9
        logger.info(
            "Direct loader: weights loaded, %.2f GB of checkpoint payload in %.2f s "
            "(%.2f GB/s); %.2f GB read from disk, %.2f GB BF16 expanded in the ring, "
            "%.2f GB converted in the GPU scratch, %.2f GB copied by Torch",
            payload,
            seconds,
            payload / seconds if seconds > 0 else 0.0,
            stats["physical_bytes"] / 1e9,
            stats["bf16_expansion_bytes"] / 1e9,
            stats["transform_bytes"] / 1e9,
            stats["torch_copy_bytes"] / 1e9,
        )
        logger.debug("Direct loader I/O counters: %s", stats)
        return model

    @staticmethod
    def _needs_values(entry):
        return math.prod(entry.shape) == 1 or entry.name.rsplit(".", 1)[-1] in {
            "layer_multipliers",
            "ngram_heads_offsets",
            "ngram_heads_vocab_sizes",
        }

    def _file_backed_weights_iterator(self, files, source, index_path):
        session = self._session
        if session is None:
            raise RuntimeError("the direct loader reads weights only in load_model")
        names_by_path = None
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            indexed_paths = {}
            names_by_path = {}
            for name, indexed_file in weight_map.items():
                resolved_indexed = indexed_paths.get(indexed_file)
                if resolved_indexed is None:
                    resolved_indexed = (index_path.parent / indexed_file).resolve()
                    indexed_paths[indexed_file] = resolved_indexed
                names_by_path.setdefault(resolved_indexed, []).append(name)
        prefixes = source.weight_name_prefixes
        local_expert_ids = self.local_expert_ids
        file_filter = source.file_weight_filter
        for path in files:
            resolved_path = Path(path).resolve()
            if names_by_path is not None:
                indexed_names = names_by_path.get(resolved_path, ())
                candidate_names = [
                    name
                    for name in indexed_names
                    if (not prefixes or name.startswith(prefixes))
                    and not should_skip_weight(name, local_expert_ids)
                ]
                if not candidate_names:
                    continue
                if not any(file_filter(name) for name in candidate_names):
                    allowed_names = set(candidate_names)
                    yield from session.weights(
                        [path],
                        prefix=source.prefix,
                        needs_values=self._needs_values,
                        skip=lambda name, allowed=allowed_names: name not in allowed,
                    )
                    continue
                allowed_indexed = set(candidate_names)
            else:
                allowed_indexed = None
            sources = weight_utils.safetensors_file_sources(path)
            if allowed_indexed is not None:
                selected = [name for name in sorted(sources) if name in allowed_indexed]
            else:
                selected = [
                    name
                    for name in sorted(sources)
                    if (not prefixes or name.startswith(prefixes))
                    and not should_skip_weight(name, local_expert_ids)
                ]
            file_names = {name for name in selected if file_filter(name)}
            ordinary_names = {name for name in selected if name not in file_names}
            file_backed = (
                (source.prefix + name, weight_utils.file_source_tensor(sources[name]))
                for name in selected
                if name in file_names
            )
            # File-backed entries never enter the checkpoint direct reader, even
            # for mixed files or before metadata-value reads and routing.
            ordinary = (
                session.weights(
                    [path],
                    prefix=source.prefix,
                    needs_values=self._needs_values,
                    skip=lambda name, ordinary_names=ordinary_names: (
                        name not in ordinary_names
                    ),
                )
                if ordinary_names
                else ()
            )
            yield from heapq.merge(file_backed, ordinary, key=lambda item: item[0])

    def _get_weights_iterator(self, source):
        session = self._session
        if session is None:
            raise RuntimeError("the direct loader reads weights only in load_model")
        folder, files, _ = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            False,
            source.allow_patterns_overrides,
            source.weight_name_prefixes,
        )
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        index_path = Path(folder) / SAFE_WEIGHTS_INDEX_NAME
        if source.file_weight_filter is not None:
            yield from self._file_backed_weights_iterator(files, source, index_path)
            return
        yield from session.weights(
            files,
            prefixes=source.weight_name_prefixes,
            prefix=source.prefix,
            index_path=index_path,
            needs_values=self._needs_values,
            skip=lambda name: should_skip_weight(name, self.local_expert_ids),
        )
