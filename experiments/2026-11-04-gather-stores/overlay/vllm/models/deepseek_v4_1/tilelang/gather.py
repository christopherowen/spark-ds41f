# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write rank-major gathered column shards directly into persistent outputs."""

from functools import cache

import torch

from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.tilelang_utils import T, tilelang

_DTYPES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.int32: "int32",
}
_PASS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}
_HEADER = r"""
__device__ __forceinline__ void vllm_gather_b128(const void* d, const void* s) {
  *reinterpret_cast<uint4*>(const_cast<void*>(d)) = *reinterpret_cast<const uint4*>(s);
}
"""


@tilelang.jit(pass_configs=_PASS)
def gather_columns_kernel(
    world: int, local_width: int, width: int, parts: int, dtype: str, packed: bool
):
    """Gather ordering, optional unpadding and one/two output stores in one pass."""
    assert world >= 1 and parts in (1, 2) and 0 < width <= world * local_width
    vector = 16 // {"bfloat16": 2, "float16": 2, "float32": 4, "int32": 4}[dtype]
    run = vector if packed else 1
    if packed:
        assert local_width % vector == 0
    source_rows, rows, stride0, stride1 = T.dynamic(
        "source_rows, rows, stride0, stride1"
    )

    @T.macro
    def store(src, dst, part, row, col):
        rank = col // local_width
        source_row = (rank * parts + part) * rows + row
        if packed:
            if col + vector <= width:
                T.call_extern(
                    "handle",
                    "vllm_gather_b128",
                    T.address_of(dst[row, col]),
                    T.address_of(src[source_row, col % local_width]),
                )
            else:
                for k in T.unroll(vector):
                    if col + k < width:
                        dst[row, col + k] = src[source_row, col % local_width + k]
        else:
            dst[row, col] = src[source_row, col % local_width]

    @T.prim_func
    def gather_columns_into(
        src: T.Tensor((source_rows, local_width), dtype),
        out0: T.StridedTensor((rows, width), (stride0, 1), dtype),
        out1: T.StridedTensor((rows, width), (stride1, 1), dtype),
    ):
        with T.Kernel(rows, T.ceildiv(width, 128 * run), threads=128) as (row, block):
            T.import_source(_HEADER)
            col = (block * 128 + T.get_thread_binding()) * run
            if col < width:
                store(src, out0, 0, row, col)
                if parts == 2:
                    store(src, out1, 1, row, col)

    return gather_columns_into


@cache
def _bound(*key):
    return gather_columns_kernel(*key)


def gather_call(
    world: int,
    local_width: int,
    width: int,
    parts: int,
    dtype: torch.dtype,
    packed: bool,
):
    return prepare.call(
        gather_columns_kernel, world, local_width, width, parts, _DTYPES[dtype], packed
    )


def _range(tensor):
    begin = tensor.data_ptr()
    extent = sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())) + 1
    return begin, begin + extent * tensor.element_size()


def gather_columns_into(
    gathered: torch.Tensor, outputs: tuple[torch.Tensor, ...], world: int
) -> None:
    """A dim-0 gather of [parts * rows, local_width] into full column rows.

    Every output has the same shape and dtype. Optional padding at the end of
    the global columns is discarded. No transpose, intermediate full-width
    allocation or subsequent copy is needed. Unsupported layouts fail explicitly.
    """
    if len(outputs) not in (1, 2) or gathered.ndim != 2 or world < 1:
        raise ValueError(
            "gather_columns_into needs one/two outputs and a positive world"
        )
    first = outputs[0]
    if first.ndim != 2 or not gathered.is_contiguous():
        raise ValueError("gather input must be contiguous and outputs must be matrices")
    if not first.is_cuda or first.dtype not in _DTYPES:
        raise ValueError("gather outputs require a supported CUDA dtype")
    rows, width = first.shape
    local_width = gathered.shape[1]
    if (
        gathered.shape[0] != world * len(outputs) * rows
        or width > world * local_width
        or width <= 0
    ):
        raise ValueError("gather source shape does not match output shards")
    for tensor in (gathered, *outputs):
        if (
            tensor.dtype != first.dtype
            or tensor.device != first.device
            or tensor.ndim != 2
            or tensor.stride(1) != 1
        ):
            raise ValueError(
                "gather tensors need unit-stride columns, one dtype and device"
            )
    if any(t.shape != first.shape for t in outputs):
        raise ValueError("gather outputs must have equal shapes")
    if not rows:
        return
    ranges = [_range(t) for t in (gathered, *outputs)]
    for i, (begin, end) in enumerate(ranges):
        if any(begin < hi and lo < end for lo, hi in ranges[i + 1 :]):
            raise ValueError("gather source and outputs must not overlap")
    vector = 16 // first.element_size()
    packed = local_width % vector == 0 and all(
        t.data_ptr() % 16 == 0 and t.stride(0) % vector == 0
        for t in (gathered, *outputs)
    )
    _bound(world, local_width, width, len(outputs), _DTYPES[first.dtype], packed)(
        gathered, first, outputs[1] if len(outputs) == 2 else first
    )
