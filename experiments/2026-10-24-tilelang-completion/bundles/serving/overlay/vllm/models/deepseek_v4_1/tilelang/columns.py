# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Column blocks of row-major tensors, copied in 16-byte vectors.

Prefill sequence parallelism gathers an attention layer's row-wise front (the
normalized Q and KV halves of the fused projection and the indexer's head weights)
in one all-gather: the norms write their halves into the gather's buffer in place,
``copy_columns`` puts the head weights beside them, and ``split_columns`` cuts the
gathered rows into one contiguous tensor a part. Torch's cat and three contiguous
copies, as two TileLang launches. ``join_columns`` is the split reversed: the
DSpark drafter's eager context step joins the target's auxiliary hidden states side
by side (torch's cat). A thread copies 16 bytes; the block widths are whole vectors,
so no vector straddles two parts.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
from functools import cache

import torch

from vllm.tilelang_utils import T, tilelang

_THREADS = 128
_VECTOR_BYTES = 16
_DTYPES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}
_PASS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}

_HEADER = r"""
__device__ __forceinline__ void vllm_columns_b128(void* d, const void* s) {
  *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>(s);
}

__device__ __forceinline__ void vllm_columns_zero_b128(void* d) {
  *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
}
"""


def _vector(dtype: str) -> int:
    return _VECTOR_BYTES // {"bfloat16": 2, "float16": 2, "float32": 4}[dtype]


@tilelang.jit(pass_configs=_PASS)
def copy_columns_kernel(width: int, dtype: str):
    """``dst[r, :width] = src[r, :width]`` for the first ``rows`` rows, either side
    a view of wider rows."""
    vector = _vector(dtype)
    if width % vector:
        raise ValueError(f"a width of {width} is not whole 16-byte vectors")
    span = _THREADS * vector
    src_rows, dst_rows, src_stride, dst_stride = T.dynamic(
        "src_rows, dst_rows, src_stride, dst_stride"
    )

    @T.prim_func
    def copy_columns(
        src: T.StridedTensor((src_rows, width), (src_stride, 1), dtype),
        dst: T.StridedTensor((dst_rows, width), (dst_stride, 1), dtype),
        rows: T.int32,
    ):
        with T.Kernel(T.ceildiv(width, span), rows, threads=_THREADS) as (bx, row):
            T.import_source(_HEADER)
            col = bx * span + T.get_thread_binding() * vector
            if col < width:
                T.call_extern(
                    "handle",
                    "vllm_columns_b128",
                    T.address_of(dst[row, col]),
                    T.address_of(src[row, col]),
                )

    return copy_columns


@tilelang.jit(pass_configs=_PASS)
def zero_packs_kernel():
    """Zero bits into the first ``packs`` 16-byte packs of ``out``, any dtype (viewed
    as int32 words, four a pack)."""
    words = T.dynamic("words")

    @T.prim_func
    def zero_packs(out: T.Tensor((words,), T.int32), packs: T.int32):
        with T.Kernel(T.ceildiv(packs, _THREADS), threads=_THREADS) as bx:
            T.import_source(_HEADER)
            p = bx * _THREADS + T.get_thread_binding()
            if p < packs:
                T.call_extern(
                    "handle", "vllm_columns_zero_b128", T.address_of(out[p * 4])
                )

    return zero_packs


def zero_(out: torch.Tensor) -> torch.Tensor:
    """``out.zero_()`` for a contiguous tensor of whole 16-byte packs."""
    nbytes = out.numel() * out.element_size()
    if not out.is_contiguous() or nbytes % _VECTOR_BYTES or out.data_ptr() % 16:
        raise ValueError("zero_ fills contiguous 16-byte packs")
    if nbytes:
        zero_packs_kernel()(out.view(-1).view(torch.int32), nbytes // _VECTOR_BYTES)
    return out


def _move(part, whole, join: bool):
    """A 16-byte copy between a part's and the whole row's addresses, in the
    direction ``join`` gives."""
    destination, source = (whole, part) if join else (part, whole)
    return T.call_extern("handle", "vllm_columns_b128", destination, source)


@tilelang.jit(pass_configs=_PASS)
def split_columns_kernel(widths: tuple[int, int, int], dtype: str, join: bool = False):
    """Row ``r`` of ``src`` ([rows, sum(widths)], contiguous) cut into ``out0``,
    ``out1`` and ``out2`` (contiguous, ``widths[i]`` wide; a zero width has no
    part), or with ``join`` the parts' rows side by side into ``src``."""
    vector = _vector(dtype)
    if any(width % vector for width in widths) or not widths[0] or not widths[1]:
        raise ValueError(f"widths {widths} are not whole 16-byte vectors")
    w0, w1, w2 = widths
    total = w0 + w1 + w2
    span = _THREADS * vector
    src_rows, rows0, rows1, rows2 = T.dynamic("src_rows, rows0, rows1, rows2")

    @T.prim_func
    def split_columns(
        src: T.Tensor((src_rows, total), dtype),
        out0: T.Tensor((rows0, w0), dtype),
        out1: T.Tensor((rows1, w1), dtype),
        out2: T.Tensor((rows2, max(w2, vector)), dtype),
        rows: T.int32,
    ):
        with T.Kernel(T.ceildiv(total, span), rows, threads=_THREADS) as (bx, row):
            T.import_source(_HEADER)
            col = bx * span + T.get_thread_binding() * vector
            whole = T.address_of(src[row, col])
            if col < w0:
                _move(T.address_of(out0[row, col]), whole, join)
            elif col < w0 + w1:
                _move(T.address_of(out1[row, col - w0]), whole, join)
            elif col < total:
                _move(T.address_of(out2[row, col - w0 - w1]), whole, join)

    return split_columns


@cache
def _bound(split: bool, *key):
    """The kernel of ``key``, resolved once (after the kernel cache freezes, an
    undeclared key raises by name). TileLang's JIT functions are not hashable."""
    return (split_columns_kernel if split else copy_columns_kernel)(*key)


def _check(*tensors: torch.Tensor) -> str:
    dtype = tensors[0].dtype
    vector = _VECTOR_BYTES // tensors[0].element_size()
    for tensor in tensors:
        if (
            tensor.dim() != 2
            or tensor.dtype != dtype
            or tensor.stride(-1) != 1
            or tensor.stride(0) % vector
            or tensor.data_ptr() % _VECTOR_BYTES
        ):
            raise ValueError("column blocks must be 16-byte aligned rows of one dtype")
    return _DTYPES[dtype]


def copy_columns_key(width: int, dtype: torch.dtype) -> tuple:
    return (width, _DTYPES[dtype])


def split_columns_key(
    widths: tuple[int, ...], dtype: torch.dtype, join: bool = False
) -> tuple:
    return (tuple(widths) + (0,) * (3 - len(widths)), _DTYPES[dtype], join)


def copy_columns(src: torch.Tensor, dst: torch.Tensor) -> None:
    """``dst.copy_(src)`` for 2-D views of 16-byte aligned rows."""
    if src.shape != dst.shape:
        raise ValueError(f"cannot copy {tuple(src.shape)} into {tuple(dst.shape)}")
    dtype = _check(src, dst)
    if src.shape[0]:
        _bound(False, src.shape[1], dtype)(src, dst, src.shape[0])


def split_columns(src: torch.Tensor, widths: tuple[int, ...]) -> list[torch.Tensor]:
    """``src`` ([rows, sum(widths)], contiguous) as two or three contiguous column
    blocks: ``[part.contiguous() for part in src.split(widths, -1)]``."""
    if len(widths) not in (2, 3) or src.shape[-1] != sum(widths):
        raise ValueError(f"cannot split {tuple(src.shape)} into {widths}")
    if not src.is_contiguous():
        raise ValueError("split_columns reads contiguous rows")
    rows = src.shape[0]
    parts = [src.new_empty(rows, width) for width in widths]
    dtype = _check(src, *parts)
    if rows:
        # A two-way split has no third part: the kernel never writes it, and any
        # contiguous rows of a vector serve as the argument.
        third = parts[2] if len(widths) == 3 else parts[0].view(-1, _vector(dtype))
        key = split_columns_key(widths, src.dtype)
        _bound(True, *key)(src, parts[0], parts[1], third, rows)
    return parts


def join_columns(parts: list[torch.Tensor], out: torch.Tensor) -> torch.Tensor:
    """``torch.cat(parts, -1, out=out)`` for two or three contiguous parts of the
    same rows into contiguous ``out``."""
    widths = tuple(part.shape[-1] for part in parts)
    if len(parts) not in (2, 3) or out.shape != (parts[0].shape[0], sum(widths)):
        raise ValueError(f"cannot join {widths} into {tuple(out.shape)}")
    if not all(t.is_contiguous() for t in (out, *parts)) or any(
        part.shape[0] != out.shape[0] for part in parts
    ):
        raise ValueError("join_columns joins contiguous parts of the same rows")
    dtype = _check(out, *parts)
    rows = out.shape[0]
    if rows:
        third = parts[2] if len(parts) == 3 else parts[0].view(-1, _vector(dtype))
        key = split_columns_key(widths, out.dtype, join=True)
        _bound(True, *key)(out, parts[0], parts[1], third, rows)
    return out


__all__ = [
    "zero_",
    "zero_packs_kernel",
    "copy_columns",
    "copy_columns_kernel",
    "copy_columns_key",
    "join_columns",
    "split_columns",
    "split_columns_kernel",
    "split_columns_key",
]
