# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero-copy ``pin_memory`` + ``io_uring`` (``READ_FIXED``) routed MoE expert slot cache.

On integrated-memory SM121 (GB10), ``torch.empty(..., pin_memory=True)`` allocates
page-locked LPDDR5x memory via ``cudaHostAlloc`` that is simultaneously:

1. Registrable with Linux ``io_uring_register_buffers`` for zero-copy ``O_DIRECT``
   ``IORING_OP_READ_FIXED`` NVMe DMA; and
2. Addressable directly by SM121 hardware TMA (``CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B``)
   and ``cp.async`` inside ``grouped_fp8_fp4_gemm`` when wrapped zero-copy via
   ``__cuda_array_interface__``.

Each layer's ``num_slots < num_experts`` physical slots (e.g. 288 of 384 routed
experts at TP2: 240 static hot slots + 48 dynamic slots) hold uniform 4-bit scale
codes so every expert has an identical 4 KiB-aligned byte footprint. On a cold
expert miss, the C ``io_uring`` reader submits all ``w13`` role SQEs before ``w2``
role SQEs (role-major ordering), allowing ``gate_up_proj`` and ``moe_swiglu_cast``
to execute on the GPU while ``w2`` streams over PCIe Gen5 NVMe. Cross-layer router
lookahead applies layer L+1's router gate to layer L's hidden state to prefetch
cold experts into layer L+1's dynamic slots before layer L's MoE GEMMs begin.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import errno
import json
import logging
import math
import mmap
import os
from pathlib import Path
import queue
import random
import shutil
import struct
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import _native

logger = logging.getLogger(__name__)

PAGE_BYTES = 4096
PACK_HEADER_BYTES = 4096
PACK_MAGIC = b"DS41EXPK"
PACK_VERSION = 1
MIN_FREE_DISK_BYTES = 8 * (1 << 30)
DEFAULT_PIN_ALIGN = 65536

_TYPESTR = {
    torch.uint8: "|u1",
    torch.int8: "|i1",
    torch.uint16: "<u2",
    torch.int16: "<i2",
    torch.int32: "<i4",
    torch.int64: "<i8",
    torch.float16: "<f2",
    torch.bfloat16: "<u2",
    torch.float32: "<f4",
}


def _align_up(value: int, align: int = PAGE_BYTES) -> int:
    return ((value + align - 1) // align) * align


class _ZeroCopyCudaArray:
    """``__cuda_array_interface__`` wrapper keeping the ``pin_memory`` owner alive."""

    def __init__(self, host_tensor: torch.Tensor, owner: object = None) -> None:
        if not host_tensor.is_contiguous():
            raise ValueError("as_cuda_zero_copy requires a contiguous host tensor")
        if host_tensor.dtype not in _TYPESTR:
            raise TypeError(f"unsupported dtype for zero-copy wrapper: {host_tensor.dtype}")
        self._host_tensor = host_tensor
        self._owner = owner
        itemsize = host_tensor.element_size()
        self.__cuda_array_interface__ = {
            "shape": tuple(host_tensor.shape),
            "strides": tuple(s * itemsize for s in host_tensor.stride()),
            "typestr": _TYPESTR[host_tensor.dtype],
            "data": (host_tensor.data_ptr(), False),
            "version": 3,
        }


def as_cuda_zero_copy(
    host_tensor: torch.Tensor,
    device: torch.device | str | None = None,
    *,
    owner: object = None,
) -> torch.Tensor:
    """Wrap a contiguous ``pin_memory`` host tensor as a zero-copy CUDA tensor."""
    target_device = torch.device("cuda" if device is None else device)
    raw_u8 = host_tensor.view(torch.uint8)
    wrapper = _ZeroCopyCudaArray(raw_u8, owner=(host_tensor, owner))
    cuda_u8 = torch.as_tensor(wrapper, device=target_device)
    return cuda_u8.view(host_tensor.dtype).view(host_tensor.shape)


def alloc_pinned_aligned(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.uint8,
    *,
    align: int = DEFAULT_PIN_ALIGN,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate an ``align``-byte-aligned ``pin_memory`` tensor and its zero-copy
    CUDA view ``(host_tensor, cuda_tensor)`` sharing the same physical address."""
    numel = 1
    for dim in shape:
        numel *= dim
    nbytes = numel * dtype.itemsize
    raw = torch.empty(nbytes + align, dtype=torch.uint8, pin_memory=True)
    ptr = raw.data_ptr()
    offset = (align - (ptr % align)) % align
    host_u8 = raw[offset : offset + nbytes]
    host_tensor = host_u8.view(dtype).view(shape)
    cuda_tensor = as_cuda_zero_copy(host_tensor, device=device, owner=raw)
    return host_tensor, cuda_tensor


@dataclass(frozen=True)
class ExpertPackMeta:
    """Byte geometry of one layer's 4 KiB-aligned role-major expert pack file."""

    num_experts: int
    hidden: int
    intermediate: int
    block_N: int
    n_blocks_13: int
    k_full_13: int
    k_blocks_13: int
    tail_13: int
    n_blocks_2: int
    k_full_2: int
    k_blocks_2: int
    tail_2: int

    w13_tiles_slot_bytes: int
    w13_codes_slot_bytes: int
    w13_tails_slot_bytes: int
    w13_base_slot_bytes: int
    w13_meta_aligned_bytes: int

    w2_tiles_slot_bytes: int
    w2_codes_slot_bytes: int
    w2_tails_slot_bytes: int
    w2_base_slot_bytes: int
    w2_meta_aligned_bytes: int

    @property
    def w13_role_bytes(self) -> int:
        return self.w13_tiles_slot_bytes + self.w13_meta_aligned_bytes

    @property
    def w2_role_bytes(self) -> int:
        return self.w2_tiles_slot_bytes + self.w2_meta_aligned_bytes

    @property
    def expert_stride(self) -> int:
        return self.w13_role_bytes + self.w2_role_bytes

    @classmethod
    def from_plan(cls, plan) -> ExpertPackMeta:
        num_experts = plan.num_experts
        hidden = plan.hidden
        inter = plan.intermediate
        block_N = plan.block_N
        block_K = 128

        n_blocks_13 = (2 * inter) // block_N
        k_full_13, tail_13 = divmod(hidden, block_K)
        k_blocks_13 = k_full_13 + (1 if tail_13 else 0)

        n_blocks_2 = hidden // block_N
        k_full_2, tail_2 = divmod(inter, block_K)
        k_blocks_2 = k_full_2 + (1 if tail_2 else 0)

        w13_tiles_slot_bytes = n_blocks_13 * k_full_13 * block_N * (block_K // 2)
        w13_codes_slot_bytes = n_blocks_13 * k_blocks_13 * block_N * 2
        w13_tails_slot_bytes = (
            n_blocks_13 * block_N * (tail_13 // 2) if tail_13 else 0
        )
        w13_base_slot_bytes = 2 * inter
        w13_meta_aligned_bytes = _align_up(
            w13_codes_slot_bytes + w13_tails_slot_bytes + w13_base_slot_bytes,
            PAGE_BYTES,
        )

        w2_tiles_slot_bytes = n_blocks_2 * k_full_2 * block_N * (block_K // 2)
        w2_codes_slot_bytes = n_blocks_2 * k_blocks_2 * block_N * 2
        w2_tails_slot_bytes = (
            n_blocks_2 * block_N * (tail_2 // 2) if tail_2 else 0
        )
        w2_base_slot_bytes = hidden
        w2_meta_aligned_bytes = _align_up(
            w2_codes_slot_bytes + w2_tails_slot_bytes + w2_base_slot_bytes,
            PAGE_BYTES,
        )

        return cls(
            num_experts=num_experts,
            hidden=hidden,
            intermediate=inter,
            block_N=block_N,
            n_blocks_13=n_blocks_13,
            k_full_13=k_full_13,
            k_blocks_13=k_blocks_13,
            tail_13=tail_13,
            n_blocks_2=n_blocks_2,
            k_full_2=k_full_2,
            k_blocks_2=k_blocks_2,
            tail_2=tail_2,
            w13_tiles_slot_bytes=w13_tiles_slot_bytes,
            w13_codes_slot_bytes=w13_codes_slot_bytes,
            w13_tails_slot_bytes=w13_tails_slot_bytes,
            w13_base_slot_bytes=w13_base_slot_bytes,
            w13_meta_aligned_bytes=w13_meta_aligned_bytes,
            w2_tiles_slot_bytes=w2_tiles_slot_bytes,
            w2_codes_slot_bytes=w2_codes_slot_bytes,
            w2_tails_slot_bytes=w2_tails_slot_bytes,
            w2_base_slot_bytes=w2_base_slot_bytes,
            w2_meta_aligned_bytes=w2_meta_aligned_bytes,
        )


def pack_expert_chunk_bytes(
    meta: ExpertPackMeta,
    gate_up_prepared: tuple[torch.Tensor | None, ...],
    down_prepared: tuple[torch.Tensor | None, ...],
    *,
    num_chunk_experts: int | None = None,
) -> bytes:
    """Serialize ``K`` prepared experts into a contiguous ``K * meta.expert_stride``
    4 KiB-aligned role-major byte slab without a per-expert Python file loop."""
    t13, tail13, codes13, _, _, base13 = gate_up_prepared
    t2, tail2, codes2, _, _, base2 = down_prepared
    K = int(meta.num_experts if num_chunk_experts is None else num_chunk_experts)
    if K <= 0:
        return b""

    t13_np = t13.detach().cpu().view(torch.uint8).reshape(K, meta.w13_tiles_slot_bytes).numpy()
    c13_np = codes13.detach().cpu().view(torch.uint8).reshape(K, meta.w13_codes_slot_bytes).numpy()
    b13_np = base13.detach().cpu().view(torch.uint8).reshape(K, meta.w13_base_slot_bytes).numpy()
    tail13_np = (
        tail13.detach().cpu().view(torch.uint8).reshape(K, meta.w13_tails_slot_bytes).numpy()
        if tail13 is not None and meta.w13_tails_slot_bytes > 0
        else None
    )

    t2_np = t2.detach().cpu().view(torch.uint8).reshape(K, meta.w2_tiles_slot_bytes).numpy()
    c2_np = codes2.detach().cpu().view(torch.uint8).reshape(K, meta.w2_codes_slot_bytes).numpy()
    b2_np = base2.detach().cpu().view(torch.uint8).reshape(K, meta.w2_base_slot_bytes).numpy()
    tail2_np = (
        tail2.detach().cpu().view(torch.uint8).reshape(K, meta.w2_tails_slot_bytes).numpy()
        if tail2 is not None and meta.w2_tails_slot_bytes > 0
        else None
    )

    out = np.zeros((K, meta.expert_stride), dtype=np.uint8)
    off = 0
    out[:, off : off + meta.w13_tiles_slot_bytes] = t13_np
    off += meta.w13_tiles_slot_bytes
    m13 = off
    out[:, m13 : m13 + meta.w13_codes_slot_bytes] = c13_np
    m13 += meta.w13_codes_slot_bytes
    if tail13_np is not None:
        out[:, m13 : m13 + meta.w13_tails_slot_bytes] = tail13_np
        m13 += meta.w13_tails_slot_bytes
    out[:, m13 : m13 + meta.w13_base_slot_bytes] = b13_np
    off += meta.w13_meta_aligned_bytes

    out[:, off : off + meta.w2_tiles_slot_bytes] = t2_np
    off += meta.w2_tiles_slot_bytes
    m2 = off
    out[:, m2 : m2 + meta.w2_codes_slot_bytes] = c2_np
    m2 += meta.w2_codes_slot_bytes
    if tail2_np is not None:
        out[:, m2 : m2 + meta.w2_tails_slot_bytes] = tail2_np
        m2 += meta.w2_tails_slot_bytes
    out[:, m2 : m2 + meta.w2_base_slot_bytes] = b2_np
    return out.tobytes()


def build_pack_header(meta: ExpertPackMeta, checkpoint_key: str = "") -> bytes:
    """Build a 4 KiB-aligned header page describing ``meta`` and ``checkpoint_key``."""
    payload = {
        "magic": PACK_MAGIC.decode("ascii"),
        "version": PACK_VERSION,
        "format": "role_major_w13_w2_4k",
        "checkpoint_key": str(checkpoint_key),
        "num_experts": int(meta.num_experts),
        "hidden": int(meta.hidden),
        "intermediate": int(meta.intermediate),
        "block_N": int(meta.block_N),
        "expert_stride": int(meta.expert_stride),
        "w13_role_bytes": int(meta.w13_role_bytes),
        "w2_role_bytes": int(meta.w2_role_bytes),
        "w13_tiles_slot_bytes": int(meta.w13_tiles_slot_bytes),
        "w13_codes_slot_bytes": int(meta.w13_codes_slot_bytes),
        "w13_tails_slot_bytes": int(meta.w13_tails_slot_bytes),
        "w13_base_slot_bytes": int(meta.w13_base_slot_bytes),
        "w13_meta_aligned_bytes": int(meta.w13_meta_aligned_bytes),
        "w2_tiles_slot_bytes": int(meta.w2_tiles_slot_bytes),
        "w2_codes_slot_bytes": int(meta.w2_codes_slot_bytes),
        "w2_tails_slot_bytes": int(meta.w2_tails_slot_bytes),
        "w2_base_slot_bytes": int(meta.w2_base_slot_bytes),
        "w2_meta_aligned_bytes": int(meta.w2_meta_aligned_bytes),
    }
    raw_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if 16 + len(raw_json) > PACK_HEADER_BYTES:
        raise ValueError(f"pack header JSON ({len(raw_json)} B) exceeds {PACK_HEADER_BYTES} B")
    buf = bytearray(PACK_HEADER_BYTES)
    buf[0:8] = PACK_MAGIC
    struct.pack_into("<II", buf, 8, PACK_VERSION, len(raw_json))
    buf[16 : 16 + len(raw_json)] = raw_json
    return bytes(buf)


def read_and_validate_pack_header(
    path: Path | str,
    meta: ExpertPackMeta,
    checkpoint_key: str = "",
) -> tuple[bool, str, dict]:
    """Validate the 4 KiB header and file size of a cached layer expert file."""
    target = Path(path)
    try:
        st = target.stat()
    except OSError as exc:
        return False, f"stat failed: {exc}", {}
    expected_size = PACK_HEADER_BYTES + meta.num_experts * meta.expert_stride
    if st.st_size != expected_size:
        return (
            False,
            f"file size {st.st_size} != expected {expected_size}",
            {},
        )
    try:
        with open(target, "rb") as f:
            hdr = f.read(PACK_HEADER_BYTES)
    except OSError as exc:
        return False, f"header read failed: {exc}", {}
    if len(hdr) != PACK_HEADER_BYTES or hdr[:8] != PACK_MAGIC:
        return False, "invalid header magic", {}
    version, json_len = struct.unpack_from("<II", hdr, 8)
    if version != PACK_VERSION:
        return False, f"unsupported pack version {version} != {PACK_VERSION}", {}
    if json_len == 0 or 16 + json_len > PACK_HEADER_BYTES:
        return False, f"invalid header json_len {json_len}", {}
    try:
        info = json.loads(hdr[16 : 16 + json_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"corrupt header JSON: {exc}", {}
    if checkpoint_key and info.get("checkpoint_key", "") != str(checkpoint_key):
        return (
            False,
            f"checkpoint_key mismatch ({info.get('checkpoint_key')!r} != {checkpoint_key!r})",
            info,
        )
    checks = (
        ("num_experts", meta.num_experts),
        ("hidden", meta.hidden),
        ("intermediate", meta.intermediate),
        ("block_N", meta.block_N),
        ("expert_stride", meta.expert_stride),
        ("w13_role_bytes", meta.w13_role_bytes),
        ("w2_role_bytes", meta.w2_role_bytes),
    )
    for key, expected_val in checks:
        if int(info.get(key, -1)) != int(expected_val):
            return (
                False,
                f"header {key}={info.get(key)} != expected {expected_val}",
                info,
            )
    return True, "ok", info


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def cleanup_stale_tmp_files(directory: Path | str) -> int:
    """Remove any stale ``*.tmp.<pid>`` staging files left behind by dead processes
    and return the number of files removed."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return 0
    removed = 0
    for candidate in sorted(dir_path.iterdir()):
        name = candidate.name
        if ".tmp." not in name:
            continue
        suffix = name.rsplit(".tmp.", 1)[-1]
        try:
            pid = int(suffix)
        except ValueError:
            continue
        if pid != os.getpid() and not _is_pid_alive(pid):
            try:
                candidate.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def write_layer_expert_file(
    path: Path | str,
    meta: ExpertPackMeta,
    records: bytes | memoryview | Sequence[bytes | memoryview],
    *,
    checkpoint_key: str = "",
    min_free_bytes: int = MIN_FREE_DISK_BYTES,
    free_bytes_override: int | None = None,
    use_o_direct: bool = True,
) -> Path | None:
    """Write a 4 KiB-header + 4 KiB-aligned expert pack file into a PID-scoped
    staging path (``<name>.tmp.<pid>``) using ``O_DIRECT`` (with buffered +
    ``fsync`` fallback) after checking the ``min_free_bytes`` (8 GiB) disk
    space safety floor, then atomically publish via ``os.replace``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    expected_body = meta.num_experts * meta.expert_stride
    total_write_bytes = PACK_HEADER_BYTES + expected_body
    free_bytes = (
        int(free_bytes_override)
        if free_bytes_override is not None
        else int(shutil.disk_usage(target.parent).free)
    )
    if free_bytes < int(min_free_bytes) or (free_bytes - total_write_bytes) < int(min_free_bytes):
        logger.warning(
            "Skipping expert pack write for %s: free disk space %.2f GiB would fall below %.2f GiB safety floor",
            target,
            free_bytes / (1024**3),
            int(min_free_bytes) / (1024**3),
        )
        return None

    if isinstance(records, (bytes, bytearray, memoryview)):
        chunks: list[memoryview] = [memoryview(records)]
    else:
        chunks = [memoryview(c) for c in records]

    actual_body = sum(len(c) for c in chunks)
    if actual_body != expected_body:
        raise ValueError(
            f"records byte length {actual_body} != expected {expected_body}"
        )
    for c in chunks:
        if len(c) % PAGE_BYTES != 0:
            raise ValueError(
                f"chunk length {len(c)} is not a multiple of {PAGE_BYTES} B"
            )

    header = build_pack_header(meta, checkpoint_key=checkpoint_key)
    tmp_path = target.parent / f"{target.name}.tmp.{os.getpid()}"

    max_chunk = max([PACK_HEADER_BYTES, *[len(c) for c in chunks]])
    bounce_cap = _align_up(min(max_chunk, 16 * 1024 * 1024), PAGE_BYTES)
    mm = mmap.mmap(-1, bounce_cap)
    mm_view = memoryview(mm)

    fd = -1
    used_odirect = False
    o_direct_flag = getattr(os, "O_DIRECT", 0) if use_o_direct else 0
    try:
        if o_direct_flag:
            try:
                fd = os.open(
                    str(tmp_path),
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC | o_direct_flag | os.O_CLOEXEC,
                    0o644,
                )
                used_odirect = True
            except OSError:
                fd = -1
                used_odirect = False
        if fd < 0:
            fd = os.open(
                str(tmp_path),
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC,
                0o644,
            )

        def _write_all_aligned(payload: memoryview) -> None:
            nonlocal fd, used_odirect
            pos = 0
            total = len(payload)
            while pos < total:
                step = min(bounce_cap, total - pos)
                mm_view[:step] = payload[pos : pos + step]
                try:
                    n = os.write(fd, mm_view[:step])
                except OSError as exc:
                    if used_odirect and exc.errno == errno.EINVAL and pos == 0:
                        os.close(fd)
                        fd = os.open(
                            str(tmp_path),
                            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC,
                            0o644,
                        )
                        used_odirect = False
                        continue
                    raise
                if n <= 0:
                    raise OSError(f"short write to {tmp_path}")
                pos += n

        _write_all_aligned(memoryview(header))
        for chunk in chunks:
            if len(chunk) > 0:
                _write_all_aligned(chunk)
        if not used_odirect:
            os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(tmp_path, target)
        return target
    except Exception:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        del mm_view
        mm.close()


def select_spot_check_experts(
    num_experts: int,
    seed: int | None = None,
) -> list[int]:
    """Select 4 experts for byte-for-byte cached-start spot verification:
    expert ``0`` (first), expert ``1`` (first + 1), expert ``E - 1`` (last),
    and ``1`` pseudo-randomly sampled expert in ``2 .. E - 2``."""
    if num_experts <= 0:
        return []
    if num_experts <= 4:
        return list(range(num_experts))
    rng = random.Random(seed)
    sampled = rng.randint(2, num_experts - 2)
    return [0, 1, sampled, num_experts - 1]


def verify_spot_experts(
    pack_path: Path | str,
    meta: ExpertPackMeta,
    plan,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    checkpoint_key: str = "",
    expected_checkpoint_key: str | None = None,
    spot_seed: int | None = None,
    rng_seed: int | None = None,
) -> tuple[bool, str, list[int]]:
    """Validate the 4 KiB header of ``pack_path`` and compare 4 spot-checked
    experts (``0``, ``1``, ``sampled``, ``E - 1``) byte-for-byte against fresh
    post-load preparation from the raw checkpoint weights."""
    target = Path(pack_path)
    ckpt_key = (
        checkpoint_key
        if expected_checkpoint_key is None
        else expected_checkpoint_key
    )
    seed = spot_seed if rng_seed is None else rng_seed
    ok_hdr, reason, _ = read_and_validate_pack_header(
        target, meta, checkpoint_key=ckpt_key
    )
    if not ok_hdr:
        return False, reason, []

    spot_ids = select_spot_check_experts(meta.num_experts, seed=seed)
    if not spot_ids:
        return True, "ok", []

    idx = torch.tensor(spot_ids, dtype=torch.long, device=w13.device)
    w13_sub = w13.index_select(0, idx)
    s13_sub = w13_scale.index_select(0, idx)
    w2_sub = w2.index_select(0, idx)
    s2_sub = w2_scale.index_select(0, idx)

    g13_sub = plan.prepare_weight(w13_sub, s13_sub)
    g2_sub = plan.prepare_weight(w2_sub, s2_sub)
    expected_blob = pack_expert_chunk_bytes(
        meta,
        g13_sub,
        g2_sub,
        num_chunk_experts=len(spot_ids),
    )
    del w13_sub, s13_sub, w2_sub, s2_sub, g13_sub, g2_sub

    stride = meta.expert_stride
    try:
        with open(target, "rb") as f:
            for k, exp_id in enumerate(spot_ids):
                f.seek(PACK_HEADER_BYTES + exp_id * stride)
                on_disk = f.read(stride)
                expected_slice = expected_blob[k * stride : (k + 1) * stride]
                if len(on_disk) != stride or on_disk != expected_slice:
                    return (
                        False,
                        f"spot check byte mismatch at expert {exp_id}",
                        spot_ids,
                    )
    except OSError as exc:
        return False, f"spot check read failed: {exc}", spot_ids

    return True, "ok", spot_ids


def pack_layer_experts(
    path: Path | str,
    meta: ExpertPackMeta,
    gate_up_prepared: tuple[torch.Tensor | None, ...],
    down_prepared: tuple[torch.Tensor | None, ...],
) -> Path:
    """Write a 4 KiB-aligned role-major expert pack file for ``meta.num_experts``
    experts from uniform 4-bit prepared weights ``(tiles, tails, codes, None, None, base)``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = pack_expert_chunk_bytes(
        meta,
        gate_up_prepared,
        down_prepared,
        num_chunk_experts=meta.num_experts,
    )
    with open(target, "wb") as f:
        f.write(blob)
        f.flush()
    return target




def _to_1d_float64(values: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().to(torch.float64).reshape(-1).numpy().copy()
    return np.asarray(values, dtype=np.float64).reshape(-1).copy()


def _zscore(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    std = float(np.std(arr))
    if std <= 1e-12:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / std


def rank_experts_by_router_bias(
    e_score_correction_bias: torch.Tensor | np.ndarray | Sequence[float],
    gate_weight: torch.Tensor | None = None,
    hotlist_counts: torch.Tensor | np.ndarray | Sequence[float] | None = None,
) -> list[int]:
    """Rank experts ``0 .. E - 1`` from hottest to coldest using checkpoint router
    bias (``e_score_correction_bias``), optional router ``gate_weight`` row L2
    norms, and optional empirical ``hotlist_counts``."""
    bias = _to_1d_float64(e_score_correction_bias)
    num_experts = int(bias.size)
    if num_experts == 0:
        return []

    prior = _zscore(bias)
    if gate_weight is not None:
        gw = gate_weight.detach().cpu().to(torch.float64)
        if gw.ndim == 2 and gw.shape[0] == num_experts:
            row_norms = torch.linalg.vector_norm(gw, dim=-1).numpy()
            prior = prior + 0.25 * _zscore(row_norms)

    if hotlist_counts is not None:
        counts = _to_1d_float64(hotlist_counts)
        if counts.size != num_experts:
            raise ValueError(
                f"hotlist_counts length {counts.size} != num_experts {num_experts}"
            )
        log_counts = np.log1p(np.maximum(counts, 0.0))
        if float(np.max(log_counts)) > 0.0:
            prior = prior + 2.0 * _zscore(log_counts)

    return [
        idx
        for idx, _ in sorted(
            enumerate(prior.tolist()),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def plan_layer_slot_budgets(
    layer_biases: Sequence[torch.Tensor | np.ndarray | Sequence[float]],
    total_slot_budget: int,
    min_slots_per_layer: int = 64,
    max_slots_per_layer: int = 384,
    dynamic_slots_per_layer: int = 48,
    *,
    temperature: float | None = None,
) -> list[tuple[int, int]]:
    """Distribute ``total_slot_budget`` across MoE layers proportionally to each
    layer's routing Shannon entropy, returning ``[(num_slots_l, static_slots_l), ...]``."""
    num_layers = len(layer_biases)
    if num_layers == 0:
        return []
    if min_slots_per_layer < 1 or min_slots_per_layer > max_slots_per_layer:
        raise ValueError(
            f"invalid slot bounds: min={min_slots_per_layer}, max={max_slots_per_layer}"
        )

    arrays = [_to_1d_float64(b) for b in layer_biases]
    stds = [float(np.std(a)) for a in arrays if a.size > 0]
    median_std = float(np.median(stds)) if stds else 0.0
    temp = (
        float(temperature)
        if temperature is not None
        else (0.05 if 0.0 < median_std < 0.25 else 1.0)
    )
    if temp <= 0.0:
        raise ValueError(f"temperature must be positive, got {temp}")

    weights: list[float] = []
    for arr in arrays:
        if arr.size <= 1:
            weights.append(1.0)
            continue
        centered = (arr - float(np.mean(arr))) / temp
        shifted = centered - float(np.max(centered))
        exps = np.exp(shifted)
        probs = exps / float(np.sum(exps))
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-30))))
        weights.append(max(1e-6, entropy))

    min_total = num_layers * min_slots_per_layer
    max_total = num_layers * max_slots_per_layer
    target_total = max(min_total, min(max_total, int(total_slot_budget)))

    alloc = [float(min_slots_per_layer) for _ in range(num_layers)]
    remaining = float(target_total - min_total)
    active = {i for i in range(num_layers) if alloc[i] < max_slots_per_layer}

    while remaining > 1e-9 and active:
        w_sum = sum(weights[i] for i in active)
        if w_sum <= 0.0:
            share = remaining / len(active)
            for i in list(active):
                add = min(float(max_slots_per_layer) - alloc[i], share)
                alloc[i] += add
                remaining -= add
                if alloc[i] >= max_slots_per_layer - 1e-9:
                    active.remove(i)
            continue

        capped_any = False
        for i in list(active):
            ideal_add = remaining * (weights[i] / w_sum)
            if alloc[i] + ideal_add >= max_slots_per_layer - 1e-9:
                used = float(max_slots_per_layer) - alloc[i]
                alloc[i] = float(max_slots_per_layer)
                remaining -= used
                active.remove(i)
                capped_any = True
        if not capped_any:
            for i in active:
                alloc[i] += remaining * (weights[i] / w_sum)
            remaining = 0.0
            break

    floors = [
        max(min_slots_per_layer, min(max_slots_per_layer, int(math.floor(x + 1e-9))))
        for x in alloc
    ]
    deficit = target_total - sum(floors)
    if deficit > 0:
        order = sorted(
            range(num_layers),
            key=lambda i: (-(alloc[i] - floors[i]), -weights[i], i),
        )
        for i in order:
            if deficit <= 0:
                break
            if floors[i] < max_slots_per_layer:
                floors[i] += 1
                deficit -= 1
    elif deficit < 0:
        order = sorted(
            range(num_layers),
            key=lambda i: ((alloc[i] - floors[i]), weights[i], -i),
        )
        for i in order:
            if deficit >= 0:
                break
            if floors[i] > min_slots_per_layer:
                floors[i] -= 1
                deficit += 1

    result: list[tuple[int, int]] = []
    for slots in floors:
        dyn = min(max(0, int(dynamic_slots_per_layer)), slots)
        static = max(0, slots - dyn)
        result.append((int(slots), int(static)))
    return result


def _read_proc_memavailable_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024
    except OSError:
        pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and phys_pages > 0:
            return page_size * phys_pages
    except (ValueError, OSError, AttributeError):
        pass
    return 120 * (1 << 30)


def compute_uma_slot_budget(
    num_layers: int = 40,
    expert_bytes: int = 9_134_080,
    non_routed_bytes: int = 0,
    kv_reserve_bytes: int = 0,
    system_reserve_bytes: int = 8 * (1 << 30),
    available_bytes: int | None = None,
    *,
    min_slots: int = 64,
    max_slots: int = 384,
) -> int:
    """Compute a safe per-layer resident MoE slot budget from Linux ``MemAvailable``
    on integrated-memory (UMA) hosts after subtracting system, KV, and non-routed
    weight reserves."""
    if num_layers <= 0 or expert_bytes <= 0:
        raise ValueError(
            f"num_layers ({num_layers}) and expert_bytes ({expert_bytes}) must be positive"
        )
    avail = (
        int(available_bytes)
        if available_bytes is not None
        else _read_proc_memavailable_bytes()
    )
    usable = max(
        0,
        avail - int(system_reserve_bytes) - int(kv_reserve_bytes) - int(non_routed_bytes),
    )
    per_layer_bytes = usable // int(num_layers)
    raw_slots = per_layer_bytes // int(expert_bytes)
    return int(max(min_slots, min(max_slots, raw_slots)))


class CountDecayLFU:
    """Decayed-frequency (``count_decay``) slot replacement with cold-prefill protection.

    Slots ``0 .. static_slots - 1`` are permanently pinned. Dynamic slots
    ``static_slots .. num_slots - 1`` track exponentially decayed access scores
    (``half_life_steps``) and last-used step timestamps. During cold prefill
    sweeps (``is_prefill=True``), decode-hot dynamic slots (``score >= hot_threshold``)
    are protected from eviction whenever a cold dynamic slot is available, and
    one-off prefill touches do not decay or over-inflate decode scores.
    """

    def __init__(
        self,
        num_experts: int,
        num_slots: int,
        static_slots: int,
        *,
        initial_experts: Iterable[int] | None = None,
        half_life_steps: float = 64.0,
        hot_threshold: float = 1.5,
    ) -> None:
        if not (0 <= static_slots <= num_slots <= num_experts):
            raise ValueError(
                f"invalid slot counts: static={static_slots}, slots={num_slots}, experts={num_experts}"
            )
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.static_slots = static_slots
        self.dynamic_slots = num_slots - static_slots
        self.half_life_steps = float(half_life_steps)
        self.decay_factor = (
            0.5 ** (1.0 / self.half_life_steps) if self.half_life_steps > 0 else 1.0
        )
        self.hot_threshold = float(hot_threshold)

        order = (
            list(initial_experts)
            if initial_experts is not None
            else list(range(num_slots))
        )
        if len(order) != num_slots or len(set(order)) != num_slots:
            raise ValueError("initial_experts must contain num_slots unique expert IDs")

        self.expert_to_slot = np.full((num_experts,), -1, dtype=np.int32)
        self.slot_to_expert = np.full((num_slots,), -1, dtype=np.int32)
        self.scores = np.zeros((num_experts,), dtype=np.float32)
        self.last_used = np.zeros((num_slots,), dtype=np.int64)
        self.probationary = np.zeros((num_slots,), dtype=bool)
        self.step = 0

        for slot, exp in enumerate(order):
            self.expert_to_slot[exp] = slot
            self.slot_to_expert[slot] = exp
            self.scores[exp] = 10.0 if slot < static_slots else 1.0

    def record_step(
        self,
        active_experts: Iterable[int],
        *,
        is_prefill: bool = False,
    ) -> None:
        self.step += 1
        if not is_prefill:
            self.scores *= self.decay_factor
        bump = 0.05 if is_prefill else 1.0
        for exp in active_experts:
            if 0 <= exp < self.num_experts:
                self.scores[exp] += bump
                slot = int(self.expert_to_slot[exp])
                if slot >= 0:
                    self.probationary[slot] = False
                    self.last_used[slot] = self.step

    def promote_probationary(self, expert_id: int) -> bool:
        """Promote an unread lookahead slot to a normal resident slot when demanded."""
        if 0 <= expert_id < self.num_experts:
            slot = int(self.expert_to_slot[expert_id])
            if slot >= self.static_slots and bool(self.probationary[slot]):
                self.probationary[slot] = False
                self.last_used[slot] = self.step
                return True
        return False

    def clear_probation(self) -> int:
        """Clear temporary lookahead protection on any unpromoted probationary slots,
        leaving their score at 0.0 and last_used at 0 so they are evicted first."""
        if self.dynamic_slots <= 0:
            return 0
        dyn_prob = self.probationary[self.static_slots : self.num_slots]
        count = int(np.count_nonzero(dyn_prob))
        if count > 0:
            dyn_prob[:] = False
        return count

    def _candidate_dynamic_slots(
        self,
        protected_experts: set[int],
        *,
        is_prefill: bool = False,
    ) -> list[tuple[int, float, int, int]]:
        candidates: list[tuple[int, float, int, int]] = []
        for slot in range(self.static_slots, self.num_slots):
            if bool(self.probationary[slot]):
                continue
            occ = int(self.slot_to_expert[slot])
            if occ in protected_experts:
                continue
            score = float(self.scores[occ]) if occ >= 0 else -1.0
            hot_flag = 1 if (is_prefill and score >= self.hot_threshold) else 0
            candidates.append((hot_flag, score, int(self.last_used[slot]), slot))
        candidates.sort()
        return candidates

    def _bind_slot(
        self,
        victim_slot: int,
        new_expert: int,
        *,
        is_prefill: bool,
        probationary: bool,
    ) -> None:
        old_expert = int(self.slot_to_expert[victim_slot])
        if old_expert >= 0:
            self.expert_to_slot[old_expert] = -1
        self.expert_to_slot[new_expert] = victim_slot
        self.slot_to_expert[victim_slot] = new_expert
        if probationary:
            self.probationary[victim_slot] = True
            self.scores[new_expert] = 0.0
            self.last_used[victim_slot] = 0
        else:
            self.probationary[victim_slot] = False
            self.last_used[victim_slot] = self.step
            if self.scores[new_expert] <= 0.0:
                self.scores[new_expert] = 0.25 if is_prefill else 1.0

    def allocate_dynamic_slot(
        self,
        new_expert: int,
        protected_experts: set[int],
        *,
        is_prefill: bool = False,
        probationary: bool = False,
    ) -> int:
        """Select a victim dynamic slot ``s in [static_slots, num_slots - 1]`` for
        ``new_expert`` without evicting any expert in ``protected_experts``."""
        existing = int(self.expert_to_slot[new_expert])
        if existing >= 0:
            return existing
        if self.dynamic_slots <= 0:
            raise RuntimeError(
                f"expert {new_expert} missed but dynamic_slots == 0"
            )

        candidates = self._candidate_dynamic_slots(
            protected_experts, is_prefill=is_prefill
        )
        if not candidates:
            raise RuntimeError(
                f"no unpinned dynamic slot available for expert {new_expert} "
                f"(protected={len(protected_experts)}, dynamic_slots={self.dynamic_slots})"
            )

        victim_slot = candidates[0][3]
        self._bind_slot(
            victim_slot,
            new_expert,
            is_prefill=is_prefill,
            probationary=probationary,
        )
        return victim_slot

    def allocate_dynamic_slots_lba(
        self,
        miss_experts: Iterable[int],
        protected_experts: set[int],
        *,
        is_prefill: bool = False,
        probationary: bool = False,
    ) -> tuple[list[int], list[int]]:
        """Allocate dynamic slots for multiple missed experts by sorting both the
        missed expert IDs (ascending LBA file offset) and the chosen victim dynamic
        slots (ascending buffer address) before binding ``e_k -> s_k``."""
        sorted_experts = sorted(
            {
                int(e)
                for e in miss_experts
                if 0 <= int(e) < self.num_experts
                and int(self.expert_to_slot[int(e)]) < 0
            }
        )
        if not sorted_experts:
            return [], []
        if self.dynamic_slots <= 0:
            raise RuntimeError(
                f"{len(sorted_experts)} experts missed but dynamic_slots == 0"
            )

        candidates = self._candidate_dynamic_slots(
            protected_experts | set(sorted_experts),
            is_prefill=is_prefill,
        )
        needed = len(sorted_experts)
        if len(candidates) < needed:
            raise RuntimeError(
                f"need {needed} dynamic slots but only {len(candidates)} available "
                f"(protected={len(protected_experts)}, dynamic_slots={self.dynamic_slots})"
            )

        sorted_slots = sorted(c[3] for c in candidates[:needed])
        for exp, slot in zip(sorted_experts, sorted_slots):
            self._bind_slot(
                slot,
                exp,
                is_prefill=is_prefill,
                probationary=probationary,
            )
        return sorted_experts, sorted_slots


class RouterLookaheadPredictor:
    """Cross-layer router lookahead using layer L+1's gate weights on layer L's state."""

    def __init__(
        self,
        gate_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
        *,
        top_k: int = 6,
        max_rows: int = 16,
    ) -> None:
        self.gate_weight = gate_weight
        self.bias = e_score_correction_bias
        self.top_k = top_k
        self.max_rows = max_rows

    @torch.no_grad()
    def predict_experts(self, hidden_states: torch.Tensor) -> list[int]:
        """Predict unique routed expert IDs for the next layer on decode/verify steps."""
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        if x.shape[0] == 0 or x.shape[0] > self.max_rows:
            return []
        logits = F.linear(x.float(), self.gate_weight.float())
        scores = torch.sqrt(
            torch.where(logits > 20.0, logits, F.softplus(logits))
        )
        if self.bias is not None:
            scores = scores + self.bias.float().unsqueeze(0)
        _, topk_ids = torch.topk(scores, k=self.top_k, dim=-1, largest=True, sorted=False)
        return [int(e) for e in torch.unique(topk_ids).cpu().tolist() if e >= 0]


class ExpertSlotCache:
    """Zero-copy ``pin_memory`` + ``io_uring`` (``READ_FIXED``) slot cache for one MoE layer."""

    def __init__(
        self,
        meta: ExpertPackMeta,
        pack_path: Path | str,
        *,
        num_slots: int,
        static_slots: int,
        initial_experts: Iterable[int] | None = None,
        router_bias: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        gate_weight: torch.Tensor | None = None,
        hotlist_counts: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        half_life_steps: float = 64.0,
        prefill_protect_rows: int = 16,
        device: torch.device | str | None = None,
        populate_on_init: bool = True,
        static_only_on_init: bool = False,
        disk_ready_event: threading.Event | None = None,
    ) -> None:
        self.meta = meta
        self.pack_path = Path(pack_path)
        self.num_experts = meta.num_experts
        self.num_slots = num_slots
        self.static_slots = static_slots
        self.dynamic_slots = num_slots - static_slots
        self.prefill_protect_rows = prefill_protect_rows
        self.device = torch.device("cuda" if device is None else device)
        self._disk_ready_event = disk_ready_event

        if initial_experts is None and (
            router_bias is not None or hotlist_counts is not None
        ):
            bias_input = (
                router_bias
                if router_bias is not None
                else np.zeros((meta.num_experts,), dtype=np.float32)
            )
            ranked = rank_experts_by_router_bias(
                bias_input,
                gate_weight=gate_weight,
                hotlist_counts=hotlist_counts,
            )
            initial_experts = ranked[:num_slots]

        self.lfu = CountDecayLFU(
            meta.num_experts,
            num_slots,
            static_slots,
            initial_experts=initial_experts,
            half_life_steps=half_life_steps,
        )

        S = num_slots
        self.w13_tiles_host, self.w13_tiles = alloc_pinned_aligned(
            (S * meta.n_blocks_13 * meta.k_full_13, meta.block_N, 64),
            torch.int8,
            device=self.device,
        )
        self.w13_codes_host, self.w13_codes = alloc_pinned_aligned(
            (S * meta.n_blocks_13 * meta.k_blocks_13, meta.block_N),
            torch.uint16,
            device=self.device,
        )
        if meta.w13_tails_slot_bytes > 0:
            self.w13_tails_host, self.w13_tails = alloc_pinned_aligned(
                (S * meta.n_blocks_13, meta.block_N, meta.tail_13 // 2),
                torch.uint8,
                device=self.device,
            )
        else:
            self.w13_tails_host, self.w13_tails = None, None
        self.w13_base_host, self.w13_base = alloc_pinned_aligned(
            (S, 2 * meta.intermediate),
            torch.uint8,
            device=self.device,
        )

        self.w2_tiles_host, self.w2_tiles = alloc_pinned_aligned(
            (S * meta.n_blocks_2 * meta.k_full_2, meta.block_N, 64),
            torch.int8,
            device=self.device,
        )
        self.w2_codes_host, self.w2_codes = alloc_pinned_aligned(
            (S * meta.n_blocks_2 * meta.k_blocks_2, meta.block_N),
            torch.uint16,
            device=self.device,
        )
        if meta.w2_tails_slot_bytes > 0:
            self.w2_tails_host, self.w2_tails = alloc_pinned_aligned(
                (S * meta.n_blocks_2, meta.block_N, meta.tail_2 // 2),
                torch.uint8,
                device=self.device,
            )
        else:
            self.w2_tails_host, self.w2_tails = None, None
        self.w2_base_host, self.w2_base = alloc_pinned_aligned(
            (S, meta.hidden),
            torch.uint8,
            device=self.device,
        )

        self.expert_to_slot_host, self.expert_to_slot_gpu = alloc_pinned_aligned(
            (meta.num_experts,),
            torch.int32,
            device=self.device,
        )

        meta_stride = meta.w13_meta_aligned_bytes + meta.w2_meta_aligned_bytes
        meta_staging_bytes = max(1, self.dynamic_slots * meta_stride)
        self._meta_staging_host, _ = alloc_pinned_aligned(
            (meta_staging_bytes,),
            torch.uint8,
            device=self.device,
        )

        self._mod = _native.load()
        self._reader = None
        self._in_flight_lookahead: set[int] = set()
        self._w2_pending: bool = False
        self.hits: int = 0
        self.misses: int = 0
        self.lookahead_submitted: int = 0
        self.lookahead_hits: int = 0
        self.lookahead_evictable: int = 0
        self.prefill_windows: int = 0
        self.slots_filled_from_disk: int = 0
        self.slots_populated_in_ram: int = 0

        if populate_on_init:
            self.fill_from_disk_cache(
                self.pack_path,
                static_only=static_only_on_init,
            )
        else:
            self._sync_slot_map()

    def populate_slots_from_chunk_bytes(
        self,
        chunk_bytes: bytes | bytearray | memoryview,
        chunk_start_expert: int,
        chunk_count: int,
    ) -> int:
        """Populate any initial resident slots whose expert ID lies in
        ``[chunk_start_expert, chunk_start_expert + chunk_count)`` directly from
        a packed chunk buffer in RAM (zero disk I/O during Boot 1 build)."""
        m = self.meta
        mv = memoryview(chunk_bytes)
        o_w13_codes = m.w13_tiles_slot_bytes
        o_w13_meta = o_w13_codes + m.w13_codes_slot_bytes
        o_w2_base_off = m.w13_role_bytes
        o_w2_codes = o_w2_base_off + m.w2_tiles_slot_bytes
        o_w2_meta = o_w2_codes + m.w2_codes_slot_bytes

        w13_tiles_u8 = self.w13_tiles_host.view(torch.uint8).reshape(-1).numpy()
        w13_codes_u8 = self.w13_codes_host.view(torch.uint8).reshape(-1).numpy()
        w13_tails_u8 = (
            self.w13_tails_host.view(torch.uint8).reshape(-1).numpy()
            if self.w13_tails_host is not None
            else None
        )
        w13_base_u8 = self.w13_base_host.view(torch.uint8).reshape(-1).numpy()

        w2_tiles_u8 = self.w2_tiles_host.view(torch.uint8).reshape(-1).numpy()
        w2_codes_u8 = self.w2_codes_host.view(torch.uint8).reshape(-1).numpy()
        w2_tails_u8 = (
            self.w2_tails_host.view(torch.uint8).reshape(-1).numpy()
            if self.w2_tails_host is not None
            else None
        )
        w2_base_u8 = self.w2_base_host.view(torch.uint8).reshape(-1).numpy()

        populated = 0
        for e_local in range(chunk_count):
            exp = chunk_start_expert + e_local
            if not (0 <= exp < self.num_experts):
                continue
            slot = int(self.lfu.expert_to_slot[exp])
            if slot < 0:
                continue
            rec = mv[e_local * m.expert_stride : (e_local + 1) * m.expert_stride]

            s_t13 = slot * m.w13_tiles_slot_bytes
            w13_tiles_u8[s_t13 : s_t13 + m.w13_tiles_slot_bytes] = np.frombuffer(
                rec[0 : m.w13_tiles_slot_bytes], dtype=np.uint8
            )
            s_c13 = slot * m.w13_codes_slot_bytes
            w13_codes_u8[s_c13 : s_c13 + m.w13_codes_slot_bytes] = np.frombuffer(
                rec[o_w13_codes : o_w13_codes + m.w13_codes_slot_bytes], dtype=np.uint8
            )
            if w13_tails_u8 is not None and m.w13_tails_slot_bytes > 0:
                s_tl13 = slot * m.w13_tails_slot_bytes
                w13_tails_u8[s_tl13 : s_tl13 + m.w13_tails_slot_bytes] = np.frombuffer(
                    rec[o_w13_meta : o_w13_meta + m.w13_tails_slot_bytes],
                    dtype=np.uint8,
                )
            s_b13 = slot * m.w13_base_slot_bytes
            w13_base_u8[s_b13 : s_b13 + m.w13_base_slot_bytes] = np.frombuffer(
                rec[
                    o_w13_meta + m.w13_tails_slot_bytes : o_w13_meta
                    + m.w13_tails_slot_bytes
                    + m.w13_base_slot_bytes
                ],
                dtype=np.uint8,
            )

            s_t2 = slot * m.w2_tiles_slot_bytes
            w2_tiles_u8[s_t2 : s_t2 + m.w2_tiles_slot_bytes] = np.frombuffer(
                rec[o_w2_base_off : o_w2_base_off + m.w2_tiles_slot_bytes],
                dtype=np.uint8,
            )
            s_c2 = slot * m.w2_codes_slot_bytes
            w2_codes_u8[s_c2 : s_c2 + m.w2_codes_slot_bytes] = np.frombuffer(
                rec[o_w2_codes : o_w2_codes + m.w2_codes_slot_bytes], dtype=np.uint8
            )
            if w2_tails_u8 is not None and m.w2_tails_slot_bytes > 0:
                s_tl2 = slot * m.w2_tails_slot_bytes
                w2_tails_u8[s_tl2 : s_tl2 + m.w2_tails_slot_bytes] = np.frombuffer(
                    rec[o_w2_meta : o_w2_meta + m.w2_tails_slot_bytes], dtype=np.uint8
                )
            s_b2 = slot * m.w2_base_slot_bytes
            w2_base_u8[s_b2 : s_b2 + m.w2_base_slot_bytes] = np.frombuffer(
                rec[
                    o_w2_meta + m.w2_tails_slot_bytes : o_w2_meta
                    + m.w2_tails_slot_bytes
                    + m.w2_base_slot_bytes
                ],
                dtype=np.uint8,
            )
            populated += 1

        self.slots_populated_in_ram += populated
        self._sync_slot_map()
        return populated

    def fill_from_disk_cache(
        self,
        pack_path: Path | str | None = None,
        *,
        static_only: bool = False,
    ) -> int:
        """Read initial resident slots directly from ``pack_path`` via ``O_DIRECT`` +
        ``io_uring`` (``READ_FIXED``) in LBA-sorted order. When ``static_only=True``,
        only the ``static_slots`` router-bias-ranked static slots are read upfront and
        dynamic slots start empty for demand/lookahead streaming."""
        if pack_path is not None:
            self.pack_path = Path(pack_path)
        if self._disk_ready_event is not None:
            self._disk_ready_event.wait()

        if self._reader is not None:
            self._mod.close_reader(self._reader)
            self._reader = None

        if static_only and self.dynamic_slots > 0:
            for slot in range(self.static_slots, self.num_slots):
                old_exp = int(self.lfu.slot_to_expert[slot])
                if old_exp >= 0:
                    self.lfu.expert_to_slot[old_exp] = -1
                    self.lfu.scores[old_exp] = 0.0
                self.lfu.slot_to_expert[slot] = -1
                self.lfu.last_used[slot] = 0
                self.lfu.probationary[slot] = False
            fill_count = self.static_slots
        else:
            fill_count = self.num_slots

        if fill_count > 0:
            meta_stride = (
                self.meta.w13_meta_aligned_bytes + self.meta.w2_meta_aligned_bytes
            )
            init_meta_host, _ = alloc_pinned_aligned(
                (max(1, fill_count * meta_stride),),
                torch.uint8,
                device=self.device,
            )
            init_reader = self._open_native_reader(
                static_slots=0,
                dynamic_slots=fill_count,
                meta_staging_ptr=init_meta_host.data_ptr(),
            )
            try:
                pairs = sorted(
                    (
                        (int(self.lfu.slot_to_expert[slot]), int(slot))
                        for slot in range(fill_count)
                        if int(self.lfu.slot_to_expert[slot]) >= 0
                    ),
                    key=lambda item: item[0],
                )
                init_experts = np.asarray([p[0] for p in pairs], dtype=np.int32)
                init_slots = np.asarray([p[1] for p in pairs], dtype=np.int32)
                for start in range(0, len(pairs), 128):
                    end = min(len(pairs), start + 128)
                    self._mod.submit_reads(
                        init_reader,
                        init_experts[start:end].tobytes(),
                        init_slots[start:end].tobytes(),
                    )
                    self._mod.wait_role(init_reader, self._mod.ROLE_ALL)
                self.slots_filled_from_disk += len(pairs)
            finally:
                self._mod.close_reader(init_reader)
                del init_meta_host

        self._reader = self._open_native_reader(
            static_slots=self.static_slots,
            dynamic_slots=self.dynamic_slots,
        )
        self._sync_slot_map()
        return fill_count

    def _ensure_reader_open(self) -> None:
        if self._reader is not None:
            return
        if self._disk_ready_event is not None:
            self._disk_ready_event.wait()
        if not self.pack_path.exists():
            raise RuntimeError(
                f"expert pack file {self.pack_path} is not available for cold-miss streaming"
            )
        self._reader = self._open_native_reader(
            static_slots=self.static_slots,
            dynamic_slots=self.dynamic_slots,
        )

    def _open_native_reader(
        self,
        *,
        static_slots: int,
        dynamic_slots: int,
        meta_staging_ptr: int | None = None,
    ):
        m = self.meta
        if meta_staging_ptr is None:
            meta_staging_ptr = self._meta_staging_host.data_ptr()
        return self._mod.open_reader(
            str(self.pack_path),
            m.num_experts,
            static_slots,
            dynamic_slots,
            self.w13_tiles_host.data_ptr() + static_slots * m.w13_tiles_slot_bytes,
            m.w13_tiles_slot_bytes,
            self.w13_codes_host.data_ptr() + static_slots * m.w13_codes_slot_bytes,
            m.w13_codes_slot_bytes,
            (
                self.w13_tails_host.data_ptr() + static_slots * m.w13_tails_slot_bytes
                if self.w13_tails_host is not None
                else 0
            ),
            m.w13_tails_slot_bytes,
            self.w13_base_host.data_ptr() + static_slots * m.w13_base_slot_bytes,
            m.w13_base_slot_bytes,
            m.w13_meta_aligned_bytes,
            self.w2_tiles_host.data_ptr() + static_slots * m.w2_tiles_slot_bytes,
            m.w2_tiles_slot_bytes,
            self.w2_codes_host.data_ptr() + static_slots * m.w2_codes_slot_bytes,
            m.w2_codes_slot_bytes,
            (
                self.w2_tails_host.data_ptr() + static_slots * m.w2_tails_slot_bytes
                if self.w2_tails_host is not None
                else 0
            ),
            m.w2_tails_slot_bytes,
            self.w2_base_host.data_ptr() + static_slots * m.w2_base_slot_bytes,
            m.w2_base_slot_bytes,
            m.w2_meta_aligned_bytes,
            meta_staging_ptr,
        )

    def _sync_slot_map(self) -> None:
        self.expert_to_slot_host.numpy()[:] = self.lfu.expert_to_slot

    @property
    def weights(self) -> tuple[tuple[torch.Tensor | None, ...], tuple[torch.Tensor | None, ...]]:
        """Prepared weight tuple ``(gate_up, down)`` for ``MoeExpertsV41Plan.run``."""
        return (
            (
                self.w13_tiles,
                self.w13_tails,
                self.w13_codes,
                None,
                None,
                self.w13_base,
            ),
            (
                self.w2_tiles,
                self.w2_tails,
                self.w2_codes,
                None,
                None,
                self.w2_base,
            ),
        )

    def reader_stats(self) -> dict:
        if self._reader is None and self.pack_path.exists():
            try:
                self._ensure_reader_open()
            except RuntimeError:
                pass
        stats = (
            dict(self._mod.stats(self._reader))
            if self._reader is not None
            else {"total_bytes_read": 0}
        )
        stats.update(
            {
                "bytes_read": int(stats.get("total_bytes_read", 0)),
                "hits": self.hits,
                "misses": self.misses,
                "lookahead_submitted": self.lookahead_submitted,
                "lookahead_hits": self.lookahead_hits,
                "lookahead_evictable": self.lookahead_evictable,
                "prefill_windows": self.prefill_windows,
                "slots_filled_from_disk": self.slots_filled_from_disk,
                "slots_populated_in_ram": self.slots_populated_in_ram,
            }
        )
        return stats

    def close(self) -> None:
        if getattr(self, "_reader", None) is not None:
            self._mod.close_reader(self._reader)
            self._reader = None

    def prefetch_async(
        self,
        predicted_experts: Iterable[int],
        *,
        max_prefetch: int | None = None,
    ) -> int:
        """Issue non-blocking LBA-sorted role-major ``io_uring`` reads for predicted
        cold experts into probationary dynamic slots (``score = 0.0`` until demanded)."""
        if self.dynamic_slots <= 0:
            return 0
        self._ensure_reader_open()
        if self._w2_pending:
            self._mod.wait_role(self._reader, self._mod.ROLE_ALL)
            self._w2_pending = False
            self.lookahead_evictable += self.lfu.clear_probation()
            self._in_flight_lookahead.clear()

        unique_pred: list[int] = []
        seen: set[int] = set()
        for e in predicted_experts:
            ei = int(e)
            if 0 <= ei < self.num_experts and ei not in seen:
                seen.add(ei)
                unique_pred.append(ei)

        cap = (
            self.dynamic_slots
            if max_prefetch is None
            else min(max_prefetch, self.dynamic_slots)
        )
        cold_candidates: list[int] = []
        protected = set(unique_pred) | self._in_flight_lookahead
        avail_candidates = self.lfu._candidate_dynamic_slots(
            protected, is_prefill=False
        )
        max_alloc = min(cap, len(avail_candidates))
        for exp in unique_pred:
            if int(self.lfu.expert_to_slot[exp]) >= 0:
                continue
            if len(cold_candidates) >= max_alloc:
                break
            cold_candidates.append(exp)

        if not cold_candidates:
            return 0

        miss_experts, miss_slots = self.lfu.allocate_dynamic_slots_lba(
            cold_candidates,
            protected,
            is_prefill=False,
            probationary=True,
        )
        if miss_experts:
            for exp in miss_experts:
                self._in_flight_lookahead.add(exp)
            miss_dyn_slots = [slot - self.static_slots for slot in miss_slots]
            exp_arr = np.asarray(miss_experts, dtype=np.int32)
            slot_arr = np.asarray(miss_dyn_slots, dtype=np.int32)
            self._mod.submit_reads(self._reader, exp_arr.tobytes(), slot_arr.tobytes())
            self._w2_pending = True
            self._sync_slot_map()
            self.lookahead_submitted += len(miss_experts)

        return len(miss_experts)

    def ensure_active(
        self,
        active_experts: Iterable[int],
        *,
        is_prefill: bool = False,
    ) -> Callable[[], None] | None:
        """Ensure all ``active_experts`` are mapped to slots, promote demanded
        lookahead slots while demoting unused lookahead slots to evictable
        ``score = 0.0``, submit LBA-sorted demand misses, wait for ``ROLE_W13``,
        and return a ``wait_w2`` callback that waits for ``ROLE_W2`` before ``down_proj``."""
        unique_active: list[int] = []
        seen: set[int] = set()
        for e in active_experts:
            ei = int(e)
            if 0 <= ei < self.num_experts and ei not in seen:
                seen.add(ei)
                unique_active.append(ei)

        raw_misses: list[int] = []
        for exp in unique_active:
            slot = int(self.lfu.expert_to_slot[exp])
            if exp in self._in_flight_lookahead or (
                slot >= self.static_slots and bool(self.lfu.probationary[slot])
            ):
                self.lfu.promote_probationary(exp)
                self._in_flight_lookahead.discard(exp)
                self.lookahead_hits += 1
                self.hits += 1
            elif slot >= 0:
                self.hits += 1
            else:
                self.misses += 1
                raw_misses.append(exp)

        # Any lookahead expert not demanded by the router on this step loses its
        # probationary protection and remains at score=0.0 (last_used=0), making it
        # the first candidate for eviction without displacing warm decode experts.
        unhit_probation = self.lfu.clear_probation()
        if unhit_probation > 0:
            self.lookahead_evictable += unhit_probation
        self._in_flight_lookahead.clear()

        miss_experts: list[int] = []
        miss_slots: list[int] = []
        if raw_misses:
            self._ensure_reader_open()
            if self._w2_pending:
                self._mod.wait_role(self._reader, self._mod.ROLE_ALL)
                self._w2_pending = False
            miss_experts, miss_slots = self.lfu.allocate_dynamic_slots_lba(
                raw_misses,
                set(unique_active),
                is_prefill=is_prefill,
                probationary=False,
            )

        self.lfu.record_step(unique_active, is_prefill=is_prefill)

        if miss_experts:
            miss_dyn_slots = [slot - self.static_slots for slot in miss_slots]
            exp_arr = np.asarray(miss_experts, dtype=np.int32)
            slot_arr = np.asarray(miss_dyn_slots, dtype=np.int32)
            self._mod.submit_reads(self._reader, exp_arr.tobytes(), slot_arr.tobytes())
            self._w2_pending = True
            self._sync_slot_map()

        if self._w2_pending:
            self._ensure_reader_open()
            self._mod.wait_role(self._reader, self._mod.ROLE_W13)

            def wait_w2() -> None:
                if self._w2_pending:
                    self._mod.wait_role(self._reader, self._mod.ROLE_W2)
                    self._w2_pending = False
                    self._in_flight_lookahead.clear()

            return wait_w2
        return None

    def _partition_windows(self, topk_ids_np: np.ndarray) -> list[tuple[int, int, list[int]]]:
        """Partition ``[rows, topk]`` into maximal contiguous token windows ``(start, stop)``
        where each window requires at most ``dynamic_slots`` non-static experts."""
        rows = topk_ids_np.shape[0]
        if rows == 0:
            return []
        static_set = {
            int(self.lfu.slot_to_expert[s]) for s in range(self.static_slots)
        }
        windows: list[tuple[int, int, list[int]]] = []
        start = 0
        while start < rows:
            active_all: list[int] = []
            active_set: set[int] = set()
            non_static: set[int] = set()
            stop = start
            while stop < rows:
                row_exps = [int(e) for e in topk_ids_np[stop] if e >= 0]
                new_non_static = non_static | {
                    e for e in row_exps if e not in static_set
                }
                if len(new_non_static) > self.dynamic_slots and stop > start:
                    break
                if len(new_non_static) > self.dynamic_slots and stop == start:
                    raise RuntimeError(
                        f"single row {stop} requires {len(new_non_static)} non-static "
                        f"experts, exceeding dynamic_slots={self.dynamic_slots}"
                    )
                non_static = new_non_static
                for e in row_exps:
                    if e not in active_set:
                        active_set.add(e)
                        active_all.append(e)
                stop += 1
            windows.append((start, stop, active_all))
            start = stop
        return windows

    def run(
        self,
        plan,
        x,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None,
        workspace: torch.Tensor | None = None,
        shared=None,
        *,
        next_lookahead: tuple[RouterLookaheadPredictor, ExpertSlotCache] | None = None,
        lookahead_hidden: torch.Tensor | None = None,
    ):
        """Run ``plan`` using this layer's zero-copy slot buffers, role-major
        ``w13``/``w2`` overlap, and optional cross-layer router lookahead."""
        from vllm.models.deepseek_v4_1.tilelang.linear import slice_mxfp8

        if torch.cuda.is_current_stream_capturing():
            return plan.run(
                x,
                topk_ids,
                topk_weights,
                self.weights,
                out,
                workspace=workspace,
                shared=shared,
                expert_to_slot=self.expert_to_slot_gpu,
                wait_w2=None,
            )

        rows = topk_ids.shape[0]
        is_prefill = rows > self.prefill_protect_rows
        topk_ids_np = topk_ids.detach().cpu().numpy()
        windows = self._partition_windows(topk_ids_np)

        if next_lookahead is not None and lookahead_hidden is not None and not is_prefill:
            predictor, next_cache = next_lookahead
            pred_experts = predictor.predict_experts(lookahead_hidden)
            if pred_experts:
                next_cache.prefetch_async(pred_experts)

        if len(windows) <= 1:
            active_experts = windows[0][2] if windows else []
            wait_w2 = self.ensure_active(active_experts, is_prefill=is_prefill)
            # Ensure GPU sees any updated slot indices or dynamic slot weights.
            return plan.run(
                x,
                topk_ids,
                topk_weights,
                self.weights,
                out,
                workspace=workspace,
                shared=shared,
                expert_to_slot=self.expert_to_slot_gpu,
                wait_w2=wait_w2,
            )

        # Multi-window prefill when a large batch touches > dynamic_slots cold experts.
        if callable(shared):
            shared = shared()
        if out is not None:
            for start, stop, active_experts in windows:
                self.prefill_windows += 1
                wait_w2 = self.ensure_active(active_experts, is_prefill=True)
                x_slice = slice_mxfp8(x, start, stop)
                shared_slice = None if shared is None else shared[start:stop]
                plan.run(
                    x_slice,
                    topk_ids[start:stop],
                    topk_weights[start:stop],
                    self.weights,
                    out[start:stop],
                    workspace=None,
                    shared=shared_slice,
                    expert_to_slot=self.expert_to_slot_gpu,
                    wait_w2=wait_w2,
                )
                if wait_w2 is not None:
                    wait_w2()
                torch.cuda.current_stream(self.device).synchronize()
            return out

        temp_out = torch.empty(
            (rows, plan.hidden),
            dtype=torch.bfloat16 if shared is not None else plan.out_dtype,
            device=self.device,
        )
        for start, stop, active_experts in windows:
            self.prefill_windows += 1
            wait_w2 = self.ensure_active(active_experts, is_prefill=True)
            x_slice = slice_mxfp8(x, start, stop)
            shared_slice = None if shared is None else shared[start:stop]
            plan.run(
                x_slice,
                topk_ids[start:stop],
                topk_weights[start:stop],
                self.weights,
                temp_out[start:stop],
                workspace=None,
                shared=shared_slice,
                expert_to_slot=self.expert_to_slot_gpu,
                wait_w2=wait_w2,
            )
            if wait_w2 is not None:
                wait_w2()
            torch.cuda.current_stream(self.device).synchronize()

        def combine(start: int, stop: int, rows_out: torch.Tensor) -> None:
            rows_out.copy_(temp_out[start:stop])

        return combine


class ExpertDiskCache:
    """Startup disk-cache coordinator with a single background ``O_DIRECT`` layer-file
    writer thread on Boot 1 and 4-expert byte-for-byte spot verification on Boot 2+."""

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        min_free_bytes: int = MIN_FREE_DISK_BYTES,
        chunk_experts: int = 64,
        use_o_direct: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_free_bytes = int(min_free_bytes)
        self.chunk_experts = max(1, int(chunk_experts))
        self.use_o_direct = bool(use_o_direct)

        self._queue: queue.Queue[
            tuple[
                Path,
                ExpertPackMeta,
                list[bytes | bytearray | memoryview],
                str,
                int,
                bool,
                threading.Event,
            ]
            | None
        ] = queue.Queue(maxsize=2)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False

        self.background_writes_queued: int = 0
        self.background_writes_completed: int = 0
        self.background_writes_skipped_low_disk: int = 0
        self.background_write_errors: int = 0
        self.cached_starts: int = 0
        self.spot_checks_passed: int = 0
        self.spot_checks_failed: int = 0
        self.rebuilds: int = 0
        self.stale_tmp_cleaned: int = cleanup_stale_tmp_files(self.cache_dir)
        self.prepared_experts_total: int = 0

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._closed = False
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="ds41-expert-disk-writer",
                daemon=True,
            )
            self._thread.start()

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            (
                pack_path,
                meta,
                chunks,
                checkpoint_key,
                min_free_bytes,
                use_o_direct,
                done_event,
            ) = item
            try:
                wrote = write_layer_expert_file(
                    pack_path,
                    meta,
                    chunks,
                    checkpoint_key=checkpoint_key,
                    min_free_bytes=min_free_bytes,
                    use_o_direct=use_o_direct,
                )
                with self._lock:
                    if wrote:
                        self.background_writes_completed += 1
                    else:
                        self.background_writes_skipped_low_disk += 1
            except Exception as exc:
                with self._lock:
                    self.background_write_errors += 1
                logger.warning(
                    "Background expert pack write failed for %s: %s",
                    pack_path,
                    exc,
                )
            finally:
                done_event.set()
                self._queue.task_done()

    def enqueue_write(
        self,
        pack_path: Path | str,
        meta: ExpertPackMeta,
        chunks: Sequence[bytes | bytearray | memoryview],
        *,
        checkpoint_key: str = "",
        min_free_bytes: int | None = None,
        use_o_direct: bool | None = None,
    ) -> threading.Event:
        """Enqueue a layer's prepared expert chunks onto the single background
        ``O_DIRECT`` writer thread and return an Event signaled when published."""
        self._ensure_worker()
        done_event = threading.Event()
        with self._lock:
            self.background_writes_queued += 1
        self._queue.put(
            (
                Path(pack_path),
                meta,
                list(chunks),
                checkpoint_key,
                self.min_free_bytes if min_free_bytes is None else int(min_free_bytes),
                self.use_o_direct if use_o_direct is None else bool(use_o_direct),
                done_event,
            )
        )
        return done_event

    def _build_layer_chunked(
        self,
        pack_path: Path,
        meta: ExpertPackMeta,
        plan,
        w13_weight: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_scale: torch.Tensor,
        *,
        num_slots: int,
        static_slots: int,
        initial_experts: Iterable[int] | None,
        router_bias: torch.Tensor | np.ndarray | Sequence[float] | None,
        gate_weight: torch.Tensor | None,
        hotlist_counts: torch.Tensor | np.ndarray | Sequence[float] | None,
        half_life_steps: float,
        prefill_protect_rows: int,
        device: torch.device | str | None,
        checkpoint_key: str,
        background_write: bool,
    ) -> ExpertSlotCache:
        done_event = threading.Event()
        cache = ExpertSlotCache(
            meta,
            pack_path,
            num_slots=num_slots,
            static_slots=static_slots,
            initial_experts=initial_experts,
            router_bias=router_bias,
            gate_weight=gate_weight,
            hotlist_counts=hotlist_counts,
            half_life_steps=half_life_steps,
            prefill_protect_rows=prefill_protect_rows,
            device=device,
            populate_on_init=False,
            disk_ready_event=done_event,
        )

        chunks: list[bytes] = []
        E = meta.num_experts
        step = self.chunk_experts
        for start in range(0, E, step):
            end = min(E, start + step)
            count = end - start
            gu_chunk = plan.prepare_weight(
                w13_weight[start:end], w13_scale[start:end]
            )
            dn_chunk = plan.prepare_weight(
                w2_weight[start:end], w2_scale[start:end]
            )
            with self._lock:
                self.prepared_experts_total += count
            chunk_buf = pack_expert_chunk_bytes(
                meta,
                gu_chunk,
                dn_chunk,
                num_chunk_experts=count,
            )
            cache.populate_slots_from_chunk_bytes(chunk_buf, start, count)
            chunks.append(bytes(chunk_buf))
            del gu_chunk, dn_chunk

        if background_write:
            bg_event = self.enqueue_write(
                pack_path,
                meta,
                chunks,
                checkpoint_key=checkpoint_key,
            )
            cache._disk_ready_event = bg_event
        else:
            wrote = write_layer_expert_file(
                pack_path,
                meta,
                chunks,
                checkpoint_key=checkpoint_key,
                min_free_bytes=self.min_free_bytes,
                use_o_direct=self.use_o_direct,
            )
            with self._lock:
                if wrote:
                    self.background_writes_completed += 1
                else:
                    self.background_writes_skipped_low_disk += 1
            done_event.set()
            if wrote and pack_path.exists():
                cache._ensure_reader_open()
        return cache

    def build_or_load_layer(
        self,
        layer_name: str,
        meta: ExpertPackMeta,
        plan,
        w13_weight: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_scale: torch.Tensor,
        *,
        num_slots: int,
        static_slots: int,
        initial_experts: Iterable[int] | None = None,
        router_bias: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        gate_weight: torch.Tensor | None = None,
        hotlist_counts: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        half_life_steps: float = 64.0,
        prefill_protect_rows: int = 16,
        device: torch.device | str | None = None,
        checkpoint_key: str = "",
        spot_check_seed: int | None = None,
        static_only_on_cached_start: bool = False,
        background_write: bool = True,
    ) -> ExpertSlotCache:
        """Build or load one MoE layer's ``ExpertSlotCache``:
        - Boot 1 (file missing): prepare in small chunks without a full-layer
          staging tensor, populate resident slots directly in RAM, and enqueue
          an atomic background ``O_DIRECT`` write.
        - Boot 2+ (file present): run the 4-expert byte-for-byte spot check
          (``[0, 1, sampled, E - 1]``); on pass, skip preparing the remaining
          ``E - 4`` experts and fill resident slots directly from NVMe; on any
          mismatch, log a warning and transparently rebuild the layer file."""
        pack_path = (
            Path(layer_name)
            if Path(layer_name).is_absolute()
            else (self.cache_dir / f"{layer_name}.bin")
        )
        with self._lock:
            self.stale_tmp_cleaned += cleanup_stale_tmp_files(pack_path.parent)

        if pack_path.exists():
            ok, reason, spot_ids = verify_spot_experts(
                pack_path,
                meta,
                plan,
                w13_weight,
                w13_scale,
                w2_weight,
                w2_scale,
                expected_checkpoint_key=checkpoint_key,
                rng_seed=spot_check_seed,
            )
            if ok:
                with self._lock:
                    self.cached_starts += 1
                    self.spot_checks_passed += 1
                    self.prepared_experts_total += len(spot_ids)
                return ExpertSlotCache(
                    meta,
                    pack_path,
                    num_slots=num_slots,
                    static_slots=static_slots,
                    initial_experts=initial_experts,
                    router_bias=router_bias,
                    gate_weight=gate_weight,
                    hotlist_counts=hotlist_counts,
                    half_life_steps=half_life_steps,
                    prefill_protect_rows=prefill_protect_rows,
                    device=device,
                    populate_on_init=True,
                    static_only_on_init=static_only_on_cached_start,
                )

            with self._lock:
                self.spot_checks_failed += 1
                self.rebuilds += 1
                self.prepared_experts_total += len(spot_ids)
            logger.warning(
                "Cached expert file %s failed spot verification (%s); rebuilding all %d experts.",
                pack_path,
                reason,
                meta.num_experts,
            )
            return self._build_layer_chunked(
                pack_path,
                meta,
                plan,
                w13_weight,
                w13_scale,
                w2_weight,
                w2_scale,
                num_slots=num_slots,
                static_slots=static_slots,
                initial_experts=initial_experts,
                router_bias=router_bias,
                gate_weight=gate_weight,
                hotlist_counts=hotlist_counts,
                half_life_steps=half_life_steps,
                prefill_protect_rows=prefill_protect_rows,
                device=device,
                checkpoint_key=checkpoint_key,
                background_write=False,
            )

        return self._build_layer_chunked(
            pack_path,
            meta,
            plan,
            w13_weight,
            w13_scale,
            w2_weight,
            w2_scale,
            num_slots=num_slots,
            static_slots=static_slots,
            initial_experts=initial_experts,
            router_bias=router_bias,
            gate_weight=gate_weight,
            hotlist_counts=hotlist_counts,
            half_life_steps=half_life_steps,
            prefill_protect_rows=prefill_protect_rows,
            device=device,
            checkpoint_key=checkpoint_key,
            background_write=background_write,
        )

    def loading_done(self, timeout: float | None = None) -> dict[str, int]:
        """Wait for the background writer queue to drain and clean up any stale
        ``.tmp.<dead_pid>`` files in ``cache_dir``."""
        if self._thread is not None and self._thread.is_alive():
            if timeout is None:
                self._queue.join()
            else:
                deadline = time.monotonic() + float(timeout)
                while self._queue.unfinished_tasks > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "Timed out waiting for ExpertDiskCache writer queue to drain"
                        )
                    time.sleep(min(0.01, remaining))
        with self._lock:
            self.stale_tmp_cleaned += cleanup_stale_tmp_files(self.cache_dir)
        return self.stats()

    def close(self) -> None:
        self.loading_done()
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and not self._closed:
                self._closed = True
                self._queue.put(None)
                thread = self._thread
            else:
                thread = None
        if thread is not None:
            thread.join(timeout=5.0)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "background_writes_queued": self.background_writes_queued,
                "background_writes_completed": self.background_writes_completed,
                "background_writes_skipped_low_disk": self.background_writes_skipped_low_disk,
                "background_write_errors": self.background_write_errors,
                "cached_starts": self.cached_starts,
                "spot_checks_passed": self.spot_checks_passed,
                "spot_checks_failed": self.spot_checks_failed,
                "rebuilds": self.rebuilds,
                "stale_tmp_cleaned": self.stale_tmp_cleaned,
                "prepared_experts_total": self.prepared_experts_total,
            }
