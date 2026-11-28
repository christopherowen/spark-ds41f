# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Embedding, output-head, and decode-only weights in the GB10 display carve-out.

DGX Spark firmware reserves about 2 GiB at the top of physical memory for
display scanout surfaces (the DISPLAY_FRM carve-out). Linux and ordinary CUDA
allocations never use it, so on a headless node it sits idle; a text console
takes only its framebuffer (8 MiB at 1920x1080). Calling NVIDIA's Resource
Manager directly on /dev/nvidiactl (NV_ESC_RM_ALLOC with
NV01_MEMORY_SYSTEM, NVOS32_ATTR2_ISO_YES, and 2 MiB huge pages),
exporting the RM handle via NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD,
and importing it through CUDA Virtual Memory Management
(cuMemImportFromShareableHandle + cuMemAddressReserve + cuMemMap +
cuMemSetAccess) maps scanout memory into CUDA without touching
/dev/dri/card0, avoiding DRM master contention and 16 MiB dumb-buffer
rounding while staying readable by kernels and direct_io.

Measured on GB10 (spark-ds41f:candidate-60), both RM scanout modes stream
at full speed (~251-257 GB/s sequential read) and deliver ~231-234 GB/s on
random 576-byte reads (versus ~10 GB/s on DRM dumb buffers and ~225 GB/s in
cudaMalloc). NVOS32_ATTR2_GPU_CACHEABLE_YES matches cudaMalloc
(1.00x) on large-M GEMMs that re-read weights (appropriate for embed_tokens,
packed lm_head.weight, markov_w1, and multi-token tables), while
NVOS32_ATTR2_GPU_CACHEABLE_NO bypasses L2 cache line allocation for
single-pass decode-only weights such as the 8-bit vocab screening matrix W_q
and DSpark routed-expert slabs (saving ~39 us on lm_head_greedy(M=6) vs.
caching W_q).

With SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1 mandatory vocabulary tables are
created in the carve-out when the model is built (allocate), so the
checkpoint streams straight into it and they never occupy ordinary memory: on
GB10, memory freed after loading stays charged to the process, so copying them
in after loading and freeing the originals would cost host memory for good.
Once loading is done, before any kernel plan or graph captures the weights,
the worker confirms that every mandatory vocabulary-sized weight of the served
model is in the carve-out and that no carve-out buffer was left behind by a
layer the model dropped. It refuses to start otherwise, or when the carve-out
cannot hold them.

Nothing writes the weights after loading, and nothing else can reach them:
the RM client, device, memory handles, and export file descriptor all close
immediately after CUDA VMM imports the handle, leaving no open RM control fd
and keeping the imported allocation private to the worker process (reclaimed by
the kernel on close() or process exit). The scanout heap never hands out a
range already in use, and _MAX_TOTAL_BYTES keeps at least 64 MiB free for a
node's 8 MiB /dev/fb0 text console. A driver fault or GPU reset could still
leave a stale mapping, so SPARK_DS41F_DISPLAY_CARVEOUT_CHECK_SECONDS above
0 turns on periodic side-stream checksum verification; production leaves it
unset for zero runtime overhead.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import math
import os
import signal
import threading
import time

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

logger = init_logger(__name__)

_REQUIRED = os.environ.get("SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS", "0") == "1"
_CHECK_SECONDS = float(
    os.environ.get("SPARK_DS41F_DISPLAY_CARVEOUT_CHECK_SECONDS", "0")
)
# 16 MiB of 32-bit words per partial checksum (a 32 MiB int64 transient).
_CHECK_CHUNK_WORDS = 1 << 22
_RM_NODE = "/dev/nvidiactl"

# 2 MiB huge pages inside the scanout reserve match CUDA VMM mapping granularity.
_GRANULE = 2 << 20
# Vocabulary-sized tables (including DSpark's 60-64 MiB markov_w1) and
# decode-only DSpark routed-expert slabs; small embeddings stay where they are.
_MIN_WEIGHT_BYTES = 60 << 20
# Leave 64 MiB of the 2,032 MiB scanout reserve free for the console framebuffer.
_MAX_TOTAL_BYTES = 1968 << 20

# NVIDIA Resource Manager escape codes (nv_escape.h), class IDs (cl0000.h /
# cl0080.h / cl003e.h), and control commands (ctrl0000gpu.h / ctrl0000unix.h).
_NV_ESC_RM_FREE = 0x29
_NV_ESC_RM_CONTROL = 0x2A
_NV_ESC_RM_ALLOC = 0x2B

_NV01_ROOT = 0x0
_NV01_DEVICE_0 = 0x80
_NV01_MEMORY_SYSTEM = 0x3E

_NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS = 0x201
_NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2 = 0x205
_NV0000_CTRL_CMD_GPU_ATTACH_IDS = 0x215
_NV0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID = 0x275
_NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD = 0x3D05

_NV0000_CTRL_GPU_ATTACH_ALL_PROBED_IDS = 0x0000FFFF
_NV0000_CTRL_GPU_INVALID_ID = 0xFFFFFFFF
_NV0000_CTRL_CMD_GPU_GET_UUID_FLAGS_BINARY_SHA1 = 0x2
_NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM = 1
_NV0000_GPU_MAX_GID_LENGTH = 256
_NV_MAX_DEVICES = 32

# Non-zero RM owner tag ('spk1') and memory attributes (nvos.h) for contiguous
# 2 MiB ISO scanout pages with selectable GPU L2 cacheability.
_RM_OWNER = 0x73706B31
_NVOS32_TYPE_IMAGE = 0
_NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE = 0x00000100
_NVOS32_ATTR = (
    (3 << 23)  # NVOS32_ATTR_PAGE_SIZE_HUGE
    | (1 << 25)  # NVOS32_ATTR_LOCATION_PCI
    | (2 << 27)  # NVOS32_ATTR_PHYSICALITY_CONTIGUOUS
    | (1 << 29)  # NVOS32_ATTR_COHERENCY_CACHED
)
_NVOS32_ATTR2_BASE = (
    (1 << 18)  # NVOS32_ATTR2_ISO_YES
    | (1 << 20)  # NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB
)
_NVOS32_ATTR2_GPU_CACHEABLE_YES = 1 << 2
_NVOS32_ATTR2_GPU_CACHEABLE_NO = 2 << 2

_CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3

_HANDLE_DEVICE = 0x10001
_HANDLE_MEMORY = 0x20001


def _iowr(nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord("F") << 8) | nr


class _NvOs21Params(ctypes.Structure):
    _fields_ = [
        ("hRoot", ctypes.c_uint32),
        ("hObjectParent", ctypes.c_uint32),
        ("hObjectNew", ctypes.c_uint32),
        ("hClass", ctypes.c_uint32),
        ("pAllocParms", ctypes.c_uint64),
        ("paramsSize", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
    ]


class _NvOs00Params(ctypes.Structure):
    _fields_ = [
        ("hRoot", ctypes.c_uint32),
        ("hObjectParent", ctypes.c_uint32),
        ("hObjectOld", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
    ]


class _NvOs54Params(ctypes.Structure):
    _fields_ = [
        ("hClient", ctypes.c_uint32),
        ("hObject", ctypes.c_uint32),
        ("cmd", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("params", ctypes.c_uint64),
        ("paramsSize", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
    ]


class _Nv0000AllocParams(ctypes.Structure):
    _fields_ = [
        ("hClient", ctypes.c_uint32),
        ("processID", ctypes.c_uint32),
        ("processName", ctypes.c_char * 100),
        ("pOsPidInfo", ctypes.c_uint64),
    ]


class _Nv0080AllocParams(ctypes.Structure):
    _fields_ = [
        ("deviceId", ctypes.c_uint32),
        ("hClientShare", ctypes.c_uint32),
        ("hTargetClient", ctypes.c_uint32),
        ("hTargetDevice", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("vaSpaceSize", ctypes.c_uint64),
        ("vaStartInternal", ctypes.c_uint64),
        ("vaLimitInternal", ctypes.c_uint64),
        ("vaMode", ctypes.c_uint32),
    ]


class _NvMemoryAllocationParams(ctypes.Structure):
    _fields_ = [
        ("owner", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pitch", ctypes.c_int32),
        ("attr", ctypes.c_uint32),
        ("attr2", ctypes.c_uint32),
        ("format", ctypes.c_uint32),
        ("comprCovg", ctypes.c_uint32),
        ("zcullCovg", ctypes.c_uint32),
        ("rangeLo", ctypes.c_uint64),
        ("rangeHi", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("alignment", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("limit", ctypes.c_uint64),
        ("address", ctypes.c_uint64),
        ("ctagOffset", ctypes.c_uint32),
        ("hVASpace", ctypes.c_uint32),
        ("internalflags", ctypes.c_uint32),
        ("tag", ctypes.c_uint32),
        ("numaNode", ctypes.c_int32),
    ]


class _Nv0000AttachIdsParams(ctypes.Structure):
    _fields_ = [
        ("gpuIds", ctypes.c_uint32 * _NV_MAX_DEVICES),
        ("failedId", ctypes.c_uint32),
    ]


class _Nv0000AttachedIdsParams(ctypes.Structure):
    _fields_ = [("gpuIds", ctypes.c_uint32 * _NV_MAX_DEVICES)]


class _Nv0000GpuIdInfoV2Params(ctypes.Structure):
    _fields_ = [
        ("gpuId", ctypes.c_uint32),
        ("gpuFlags", ctypes.c_uint32),
        ("deviceInstance", ctypes.c_uint32),
        ("subDeviceInstance", ctypes.c_uint32),
        ("sliStatus", ctypes.c_uint32),
        ("boardId", ctypes.c_uint32),
        ("gpuInstance", ctypes.c_uint32),
        ("numaId", ctypes.c_int32),
    ]


class _Nv0000UuidFromGpuIdParams(ctypes.Structure):
    _fields_ = [
        ("gpuId", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("gpuUuid", ctypes.c_uint8 * _NV0000_GPU_MAX_GID_LENGTH),
        ("uuidStrLen", ctypes.c_uint32),
    ]


class _Nv0000CtrlOsUnixExportObjectToFdParams(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("hDevice", ctypes.c_uint32),
        ("hParent", ctypes.c_uint32),
        ("hObject", ctypes.c_uint32),
        ("fd", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


_libcuda = None


def _cuda():
    global _libcuda
    if _libcuda is None:
        _libcuda = ctypes.CDLL("libcuda.so.1")
    return _libcuda


def _check(cuda, status: int, what: str) -> None:
    if status:
        text = ctypes.c_char_p()
        cuda.cuGetErrorString(status, ctypes.byref(text))
        detail = text.value.decode() if text.value else status
        raise OSError(errno.ENOMEM, f"{what} failed: {detail}")


def _rm_alloc(
    ctl_fd: int, root: int, parent: int, new_obj: int, cls: int, params
) -> int:
    req = _NvOs21Params(
        hRoot=root,
        hObjectParent=parent,
        hObjectNew=new_obj,
        hClass=cls,
        pAllocParms=ctypes.addressof(params),
        paramsSize=ctypes.sizeof(params),
    )
    fcntl.ioctl(ctl_fd, _iowr(_NV_ESC_RM_ALLOC, ctypes.sizeof(req)), req)
    if req.status != 0:
        err = errno.ENOMEM if req.status == 0x51 else errno.EIO
        raise OSError(
            err, f"NV_ESC_RM_ALLOC(class={cls:#x}) failed: status={req.status:#x}"
        )
    return req.hObjectNew


def _rm_control(ctl_fd: int, client: int, obj: int, cmd: int, params) -> None:
    req = _NvOs54Params(
        hClient=client,
        hObject=obj,
        cmd=cmd,
        flags=0,
        params=ctypes.addressof(params),
        paramsSize=ctypes.sizeof(params),
    )
    fcntl.ioctl(ctl_fd, _iowr(_NV_ESC_RM_CONTROL, ctypes.sizeof(req)), req)
    if req.status != 0:
        raise OSError(
            errno.EIO,
            f"NV_ESC_RM_CONTROL(cmd={cmd:#x}) failed: status={req.status:#x}",
        )


def _rm_free(ctl_fd: int, root: int, parent: int, obj: int) -> None:
    req = _NvOs00Params(hRoot=root, hObjectParent=parent, hObjectOld=obj)
    with contextlib.suppress(OSError):
        fcntl.ioctl(ctl_fd, _iowr(_NV_ESC_RM_FREE, ctypes.sizeof(req)), req)


def _device_uuid(cuda, device_index: int) -> bytes:
    cu_dev = ctypes.c_int()
    _check(cuda, cuda.cuDeviceGet(ctypes.byref(cu_dev), device_index), "cuDeviceGet")
    cu_uuid = (ctypes.c_uint8 * 16)()
    get_uuid = getattr(cuda, "cuDeviceGetUuid_v2", cuda.cuDeviceGetUuid)
    _check(cuda, get_uuid(cu_uuid, cu_dev), "cuDeviceGetUuid")
    return bytes(cu_uuid)


def _resolve_device_instance(
    ctl_fd: int, hclient: int, cuda, device_index: int
) -> int:
    target_uuid = _device_uuid(cuda, device_index)
    attached = _Nv0000AttachedIdsParams()
    _rm_control(ctl_fd, hclient, hclient, _NV0000_CTRL_CMD_GPU_GET_ATTACHED_IDS, attached)
    for gpu_id in attached.gpuIds:
        if gpu_id == 0 or gpu_id == _NV0000_CTRL_GPU_INVALID_ID:
            continue
        uuid_req = _Nv0000UuidFromGpuIdParams(
            gpuId=gpu_id, flags=_NV0000_CTRL_CMD_GPU_GET_UUID_FLAGS_BINARY_SHA1
        )
        try:
            _rm_control(
                ctl_fd,
                hclient,
                hclient,
                _NV0000_CTRL_CMD_GPU_GET_UUID_FROM_GPU_ID,
                uuid_req,
            )
        except OSError:
            continue
        if bytes(uuid_req.gpuUuid[:16]) == target_uuid:
            info = _Nv0000GpuIdInfoV2Params(gpuId=gpu_id)
            _rm_control(
                ctl_fd, hclient, hclient, _NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2, info
            )
            return int(info.deviceInstance)
    raise OSError(
        errno.ENODEV,
        f"No RM GPU matches CUDA device {device_index} UUID {target_uuid.hex()}",
    )


class _Carveout:
    """One RM scanout allocation mapped via CUDA VMM; alive for the process.

    /dev/nvidiactl and all RM handles (client, device, memory, and export
    fd) close right after cuMemImportFromShareableHandle retains its own
    kernel reference to the exported object.
    """

    def __init__(
        self, nbytes: int, *, device_index: int = 0, cacheable: bool = True
    ):
        self.nbytes = nbytes
        self.device_index = device_index
        self.cacheable = cacheable
        self._closed = True
        self.handle = ctypes.c_uint64()
        self.pointer = ctypes.c_ulonglong()
        self._mapped = False

        cuda = _cuda()
        torch.cuda.init()

        ctl_fd = os.open(_RM_NODE, os.O_RDWR | os.O_CLOEXEC)
        hclient = 0
        hdev = 0
        hmem = 0
        export_fd = -1
        try:
            root_params = _Nv0000AllocParams()
            # Kernel upgrades NV01_ROOT allocation hObjectNew cleanly when 0.
            req = _NvOs21Params(
                hRoot=0,
                hObjectParent=0,
                hObjectNew=0,
                hClass=_NV01_ROOT,
                pAllocParms=ctypes.addressof(root_params),
                paramsSize=ctypes.sizeof(root_params),
            )
            fcntl.ioctl(ctl_fd, _iowr(_NV_ESC_RM_ALLOC, ctypes.sizeof(req)), req)
            if req.status != 0:
                raise OSError(
                    errno.EIO,
                    f"NV_ESC_RM_ALLOC(NV01_ROOT) failed: status={req.status:#x}",
                )
            hclient = req.hObjectNew

            attach = _Nv0000AttachIdsParams()
            attach.gpuIds[0] = _NV0000_CTRL_GPU_ATTACH_ALL_PROBED_IDS
            attach.gpuIds[1] = _NV0000_CTRL_GPU_INVALID_ID
            _rm_control(
                ctl_fd, hclient, hclient, _NV0000_CTRL_CMD_GPU_ATTACH_IDS, attach
            )

            device_instance = _resolve_device_instance(
                ctl_fd, hclient, cuda, device_index
            )
            dev_params = _Nv0080AllocParams(
                deviceId=device_instance, hClientShare=hclient
            )
            hdev = _rm_alloc(
                ctl_fd, hclient, hclient, _HANDLE_DEVICE, _NV01_DEVICE_0, dev_params
            )

            cache_attr = (
                _NVOS32_ATTR2_GPU_CACHEABLE_YES
                if cacheable
                else _NVOS32_ATTR2_GPU_CACHEABLE_NO
            )
            mem_params = _NvMemoryAllocationParams(
                owner=_RM_OWNER,
                type=_NVOS32_TYPE_IMAGE,
                flags=_NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE,
                attr=_NVOS32_ATTR,
                attr2=_NVOS32_ATTR2_BASE | cache_attr,
                size=nbytes,
                alignment=_GRANULE,
            )
            hmem = _rm_alloc(
                ctl_fd, hclient, hdev, _HANDLE_MEMORY, _NV01_MEMORY_SYSTEM, mem_params
            )

            export_fd = os.open(_RM_NODE, os.O_RDWR | os.O_CLOEXEC)
            export_params = _Nv0000CtrlOsUnixExportObjectToFdParams(
                type=_NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM,
                hDevice=hdev,
                hParent=hdev,
                hObject=hmem,
                fd=export_fd,
                flags=0,
            )
            _rm_control(
                ctl_fd,
                hclient,
                hclient,
                _NV0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD,
                export_params,
            )
            _check(
                cuda,
                cuda.cuMemImportFromShareableHandle(
                    ctypes.byref(self.handle),
                    ctypes.c_void_p(export_fd),
                    _CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                ),
                "cuMemImportFromShareableHandle",
            )
        finally:
            if export_fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(export_fd)
            if hmem:
                _rm_free(ctl_fd, hclient, hdev, hmem)
            if hdev:
                _rm_free(ctl_fd, hclient, hclient, hdev)
            if hclient:
                _rm_free(ctl_fd, hclient, 0, hclient)
            os.close(ctl_fd)

        self._closed = False
        try:
            _check(
                cuda,
                cuda.cuMemAddressReserve(
                    ctypes.byref(self.pointer),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_size_t(_GRANULE),
                    ctypes.c_ulonglong(0),
                    ctypes.c_ulonglong(0),
                ),
                "cuMemAddressReserve",
            )
            _check(
                cuda,
                cuda.cuMemMap(
                    self.pointer,
                    ctypes.c_size_t(nbytes),
                    ctypes.c_size_t(0),
                    self.handle,
                    ctypes.c_ulonglong(0),
                ),
                "cuMemMap",
            )
            self._mapped = True
            access = _CUmemAccessDesc(
                location=_CUmemLocation(
                    type=_CU_MEM_LOCATION_TYPE_DEVICE, id=device_index
                ),
                flags=_CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
            )
            _check(
                cuda,
                cuda.cuMemSetAccess(
                    self.pointer,
                    ctypes.c_size_t(nbytes),
                    ctypes.byref(access),
                    ctypes.c_size_t(1),
                ),
                "cuMemSetAccess",
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        cuda = _cuda()
        if getattr(self, "_mapped", False) and getattr(self, "pointer", None) and self.pointer.value:
            cuda.cuMemUnmap(self.pointer, ctypes.c_size_t(self.nbytes))
            self._mapped = False
        if getattr(self, "pointer", None) and self.pointer.value:
            cuda.cuMemAddressFree(self.pointer, ctypes.c_size_t(self.nbytes))
            self.pointer = ctypes.c_ulonglong()
        if getattr(self, "handle", None) and self.handle.value:
            cuda.cuMemRelease(self.handle)
            self.handle = ctypes.c_uint64()

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.nbytes,),
            "typestr": "|i1",
            "data": (self.pointer.value, False),
            "version": 3,
            "strides": None,
        }


# One buffer per weight, alive for the whole process.
_carveouts: list[_Carveout] = []


def required() -> bool:
    return _REQUIRED


def allocate(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str,
    *,
    optional: bool = False,
    cacheable: bool = True,
) -> torch.Tensor | None:
    """A vocabulary-sized weight created in its own carve-out buffer, or None.

    Returns None unless the carve-out is required (or optional=True and the
    RM scanout reserve can hold it), the device is CUDA and the weight reaches
    the vocabulary-sized threshold; the caller then allocates ordinary memory as
    usual. Loading writes the checkpoint straight into the returned tensor.
    """
    device = torch.device(device)
    nbytes = math.prod(shape) * dtype.itemsize
    if device.type != "cuda" or nbytes < _MIN_WEIGHT_BYTES:
        return None
    if not _REQUIRED and not optional:
        return None
    rounded = -(-nbytes // _GRANULE) * _GRANULE
    if total_bytes() + rounded > _MAX_TOTAL_BYTES:
        if optional:
            return None
        raise RuntimeError(
            f"SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1 but the display carve-out limit "
            f"({_MAX_TOTAL_BYTES >> 20} MiB) cannot hold a {nbytes >> 20} MiB weight"
        )
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    try:
        carveout = _Carveout(rounded, device_index=device_index, cacheable=cacheable)
    except OSError as error:
        if optional:
            return None
        raise RuntimeError(
            f"SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1 but the display carve-out cannot "
            f"hold a {nbytes >> 20} MiB weight ({_RM_NODE}: {error})"
        ) from error
    _carveouts.append(carveout)
    backing = torch.as_tensor(carveout, device=device)
    assert backing.data_ptr() == carveout.pointer.value
    return backing[:nbytes].view(dtype).view(shape)


def total_bytes() -> int:
    """Total bytes currently reserved across all active carve-out buffers."""
    return sum(carveout.nbytes for carveout in _carveouts)


def compact(tensor: torch.Tensor) -> torch.Tensor:
    """Shrink a carve-out buffer to fit tensor after in-place packing.

    12-bit vocabulary packing shrinks an unpacked BF16 table to 75% of its
    bytes in place. Reallocating a tight scanout buffer and closing the unpacked
    one reclaims carve-out space (at 2 MiB granularity) without touching
    ordinary memory.

    The old buffer is closed, so tensor must be the only live view of it:
    a caller that keeps other data in the same buffer (the TileLang packed
    head's screening table) must not compact it.
    """
    nbytes = _nbytes(tensor)
    if (
        not _REQUIRED
        or not tensor.is_cuda
        or not tensor.is_contiguous()
        or nbytes <= 0
    ):
        return tensor
    rounded = -(-nbytes // _GRANULE) * _GRANULE
    for index, carveout in enumerate(_carveouts):
        if not _holds(carveout, tensor):
            continue
        if rounded >= carveout.nbytes:
            return tensor
        try:
            staging = _Carveout(
                rounded,
                device_index=carveout.device_index,
                cacheable=carveout.cacheable,
            )
        except OSError:
            return tensor
        staging_view = (
            torch.as_tensor(staging, device=tensor.device)[:nbytes]
            .view(tensor.dtype)
            .view(tensor.shape)
        )
        staging_view.copy_(tensor)
        torch.cuda.synchronize(tensor.device)
        device_index = carveout.device_index
        cacheable = carveout.cacheable
        del tensor
        carveout.close()
        # Reallocate at the bottom of the scanout heap now that carveout is
        # closed so compacted vocabulary tables leave no hole between them.
        try:
            shrunk = _Carveout(
                rounded, device_index=device_index, cacheable=cacheable
            )
        except OSError:
            _carveouts[index] = staging
            return staging_view
        backing = torch.as_tensor(shrunk, device=staging_view.device)
        assert backing.data_ptr() == shrunk.pointer.value
        out = backing[:nbytes].view(staging_view.dtype).view(staging_view.shape)
        out.copy_(staging_view)
        torch.cuda.synchronize(staging_view.device)
        del staging_view
        staging.close()
        _carveouts[index] = shrunk
        return out
    return tensor


def move(tensor: torch.Tensor, *, cacheable: bool = False) -> torch.Tensor:
    """Move a prepared decode-only weight tensor into the carve-out if space permits.

    Used after process_weights_after_loading prepares a tile-major
    decode-only weight (such as DSpark's routed-expert w13_weight and
    w2_weight) when the remaining carve-out capacity within
    _MAX_TOTAL_BYTES can hold it. Defaults to uncached L2 streaming
    (GPU_CACHEABLE_NO) for single-pass decode weights.
    """
    nbytes = _nbytes(tensor)
    if (
        not _REQUIRED
        or not tensor.is_cuda
        or not tensor.is_contiguous()
        or nbytes < _MIN_WEIGHT_BYTES
    ):
        return tensor
    if any(_holds(carveout, tensor) for carveout in _carveouts):
        return tensor
    rounded = -(-nbytes // _GRANULE) * _GRANULE
    if total_bytes() + rounded > _MAX_TOTAL_BYTES:
        return tensor
    device_index = (
        tensor.device.index
        if tensor.device.index is not None
        else torch.cuda.current_device()
    )
    try:
        carveout = _Carveout(rounded, device_index=device_index, cacheable=cacheable)
    except OSError:
        return tensor
    _carveouts.append(carveout)
    backing = torch.as_tensor(carveout, device=tensor.device)
    assert backing.data_ptr() == carveout.pointer.value
    out = backing[:nbytes].view(tensor.dtype).view(tensor.shape)
    out.copy_(tensor)
    torch.cuda.synchronize(tensor.device)
    return out


def _holds(carveout: _Carveout, tensor: torch.Tensor) -> bool:
    start = carveout.pointer.value
    return start <= tensor.data_ptr() < start + carveout.nbytes


def _checksum(tensor: torch.Tensor) -> torch.Tensor:
    """The sum of the tensor's 32-bit words in 64 bits: any flipped bit changes it.

    Summed in chunks: a 64-bit sum casts its input first, which over a whole
    421 MiB table would allocate 842 MiB, and the check stream's allocator
    pool would keep it.
    """
    raw = tensor.detach().reshape(-1).view(torch.uint8)
    words = raw.view(torch.int32) if raw.numel() % 4 == 0 else raw
    parts = words.split(_CHECK_CHUNK_WORDS)
    return torch.stack([part.sum(dtype=torch.int64) for part in parts]).sum()


def _kill_worker(reason: str) -> None:
    """Stop the engine from a background thread: its executor sees the worker die."""
    logger.error("Stopping the worker: %s", reason)
    os.kill(os.getpid(), signal.SIGKILL)


class _IntegrityCheck:
    """Periodic checksums of the weights held in the carve-out (debug mode)."""

    def __init__(
        self,
        weights: list[tuple[str, torch.Tensor]],
        expected: list[int],
        device: torch.device,
        interval: float,
        on_failure=_kill_worker,
    ):
        self.weights = weights
        self.expected = expected
        self.device = device
        self.interval = interval
        self.on_failure = on_failure
        self.failure: str | None = None
        self.passed = 0
        self._started = False
        self._stream = torch.cuda.Stream(device) if device.type == "cuda" else None

    def sums(self) -> list[int]:
        stream = (
            torch.cuda.stream(self._stream)
            if self._stream is not None
            else contextlib.nullcontext()
        )
        with stream:
            return torch.stack(
                [_checksum(weight) for _, weight in self.weights]
            ).tolist()

    def check(self) -> None:
        changed = [
            f"{name} (checksum {want:#x}, now {got:#x})"
            for (name, _), want, got in zip(self.weights, self.expected, self.sums())
            if want != got
        ]
        if changed:
            self.failure = "display carve-out weights changed: " + "; ".join(changed)
            logger.error("%s", self.failure)
            return
        self.passed += 1
        if self.passed == 1 or self.passed % 12 == 0:
            logger.info(
                "Display carve-out integrity check %d passed (every %g s)",
                self.passed,
                self.interval,
            )

    def start(self) -> None:
        if self._started or self.interval <= 0:
            return
        self._started = True
        threading.Thread(
            target=self._run, name="display-carveout-check", daemon=True
        ).start()

    def _run(self) -> None:
        if self._stream is not None:
            torch.cuda.set_device(self.device)
        while self.failure is None:
            time.sleep(self.interval)
            try:
                self.check()
            except Exception as error:
                # A check that cannot run must not pass silently either.
                self.failure = f"display carve-out integrity check failed: {error!r}"
                logger.exception("Display carve-out integrity check failed")
        self.on_failure(self.failure)


_integrity: _IntegrityCheck | None = None


def start_checks() -> None:
    """Start the debug-mode periodic check, if on; called once after graph capture.

    Starting after the last capture keeps the check's side-stream work away
    from graph captures.
    """
    if _integrity is not None:
        _integrity.start()


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _is_mandatory_vocab_module(module: torch.nn.Module) -> bool:
    return isinstance(module, VocabParallelEmbedding) or (
        isinstance(module, torch.nn.Embedding)
        and getattr(module, "_display_carveout_required", False)
    )


def streamed_weights(
    model: torch.nn.Module, *extra_models: torch.nn.Module | None
) -> list[tuple[str, torch.nn.Parameter]]:
    """Streamed carve-out weights across model and any extra_models, deduplicated.

    Includes every mandatory vocabulary-sized table (VocabParallelEmbedding
    and DSpark's markov_w1) plus any decode-only parameters moved into
    _carveouts after preparation. A drafter that aliases the target's
    modules shares their parameters, so each tensor appears once.
    """
    roots = [model, *(m for m in extra_models if m is not None)]
    found: dict[int, tuple[str, torch.nn.Parameter]] = {}
    for root in roots:
        for name, module in root.named_modules():
            if _is_mandatory_vocab_module(module):
                weight = getattr(module, "weight", None)
                if (
                    isinstance(weight, torch.nn.Parameter)
                    and weight.is_cuda
                    and weight.is_contiguous()
                    and _nbytes(weight) >= _MIN_WEIGHT_BYTES
                ):
                    label = f"{name}.weight" if name else "weight"
                    found.setdefault(weight.data_ptr(), (label, weight))
            screen_wq = getattr(module, "_carveout_screen_wq", None)
            if (
                isinstance(screen_wq, torch.Tensor)
                and screen_wq.is_cuda
                and screen_wq.is_contiguous()
                and any(_holds(carveout, screen_wq) for carveout in _carveouts)
            ):
                label = (
                    f"{name}._carveout_screen_wq"
                    if name
                    else "_carveout_screen_wq"
                )
                found.setdefault(screen_wq.data_ptr(), (label, screen_wq))
        for name, param in root.named_parameters():
            if (
                param.is_cuda
                and param.is_contiguous()
                and any(_holds(carveout, param.data) for carveout in _carveouts)
            ):
                found.setdefault(param.data_ptr(), (name, param))
    return list(found.values())


def confirm_streamed_weights(
    model: torch.nn.Module,
    device: torch.device,
    *extra_models: torch.nn.Module | None,
) -> None:
    """Confirm the streamed weights loaded into the display carve-out.

    Every vocabulary-sized weight of the served model (and any drafter) must
    sit in a buffer from allocate, and every carve-out buffer must hold a
    live parameter of the served models.
    """
    assert ctypes.sizeof(_NvOs21Params) == 32
    assert ctypes.sizeof(_NvOs00Params) == 16
    assert ctypes.sizeof(_NvOs54Params) == 32
    assert ctypes.sizeof(_Nv0000AllocParams) == 120
    assert ctypes.sizeof(_Nv0080AllocParams) == 56
    assert ctypes.sizeof(_NvMemoryAllocationParams) == 128
    assert ctypes.sizeof(_Nv0000CtrlOsUnixExportObjectToFdParams) == 24
    assert ctypes.sizeof(_CUmemLocation) == 8
    assert ctypes.sizeof(_CUmemAccessDesc) == 12
    weights = streamed_weights(model, *extra_models)
    if not weights:
        raise RuntimeError(
            "SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1 but the model has no "
            "vocabulary-sized embedding or output-head weights"
        )
    outside = [
        name
        for name, weight in weights
        if not any(_holds(carveout, weight.data) for carveout in _carveouts)
    ]
    if outside:
        raise RuntimeError(
            "SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1 but these vocabulary-sized weights "
            f"were not created in the display carve-out: {', '.join(outside)}"
        )
    orphaned = [
        carveout
        for carveout in _carveouts
        if not any(_holds(carveout, weight.data) for _, weight in weights)
    ]
    if orphaned:
        # A vocabulary-sized layer built and then dropped would hold carve-out
        # space for nothing; placeholders the model replaces after loading
        # belong on the meta device.
        raise RuntimeError(
            f"{len(orphaned)} display carve-out buffers "
            f"({sum(c.nbytes for c in orphaned) >> 20} MiB) hold no weight of the "
            "served model; build placeholder layers on the meta device"
        )
    checking = _CHECK_SECONDS > 0
    if checking:
        global _integrity
        held = [(name, weight.data) for name, weight in weights]
        torch.cuda.synchronize(device)
        expected = [int(_checksum(weight)) for _, weight in held]
        _integrity = _IntegrityCheck(held, expected, device, _CHECK_SECONDS)
    logger.info(
        "Display carve-out holds %.1f MiB of streamed weights in %d buffers, "
        "loaded in place (%s); debug integrity check %s",
        sum(_nbytes(weight) for _, weight in weights) / (1 << 20),
        len(_carveouts),
        ", ".join(name for name, _ in weights),
        f"every {_CHECK_SECONDS:g} s" if checking else "off",
    )
