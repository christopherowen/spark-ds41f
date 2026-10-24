# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's context-graph inputs, ``DSparkContextCudaGraphs.run``'s torch prologue.

Before a context graph replays, its persistent inputs take the step's rows: the
target's three auxiliary hidden states side by side in ``aux`` (row ``r``'s
columns ``[i * width, (i + 1) * width)`` from source ``i``), and every row from
the step's ``num_tokens`` up to the graph's ``capacity`` scrubbed: ``aux`` zero,
``positions`` zero and each KV cache group's slot ``PAD_SLOT_ID``. Three strided
copies and three fills in torch; one launch here: a block a row and 1024 columns
of each source, a thread 16 bytes of each, its three loads issued before its
three stores.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
from functools import cache

import torch

from vllm.tilelang_utils import T, tilelang
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

NUM_SOURCES = 3  # dspark_target_layer_ids of DeepSeek V4.1 Flash
_THREADS = 128
_VECTOR_BYTES = 16
_DTYPES = {torch.bfloat16: "bfloat16", torch.float16: "float16"}

_HEADER = r"""
__device__ __forceinline__ void vllm_dspark_context_b128(void* d0, const void* s0,
                                                         void* d1, const void* s1,
                                                         void* d2, const void* s2,
                                                         bool live) {
  uint4 a = make_uint4(0u, 0u, 0u, 0u), b = a, c = a;
  if (live) {
    a = *reinterpret_cast<const uint4*>(s0);
    b = *reinterpret_cast<const uint4*>(s1);
    c = *reinterpret_cast<const uint4*>(s2);
  }
  *reinterpret_cast<uint4*>(d0) = a;
  *reinterpret_cast<uint4*>(d1) = b;
  *reinterpret_cast<uint4*>(d2) = c;
}
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def context_inputs_kernel(width: int, dtype: str):
    vector = _VECTOR_BYTES // 2
    if width % vector:
        raise ValueError(f"a width of {width} is not whole 16-byte vectors")
    span = _THREADS * vector
    aux_rows, positions_len, groups, slots_len, slots_stride = T.dynamic(
        "aux_rows, positions_len, groups, slots_len, slots_stride"
    )
    rows0, rows1, rows2, stride0, stride1, stride2 = T.dynamic(
        "rows0, rows1, rows2, stride0, stride1, stride2"
    )

    @T.prim_func
    def context_inputs(
        aux: T.Tensor((aux_rows, NUM_SOURCES * width), dtype),
        source0: T.StridedTensor((rows0, width), (stride0, 1), dtype),
        source1: T.StridedTensor((rows1, width), (stride1, 1), dtype),
        source2: T.StridedTensor((rows2, width), (stride2, 1), dtype),
        positions: T.Tensor((positions_len,), T.int64),
        slot_mappings: T.StridedTensor((groups, slots_len), (slots_stride, 1), T.int64),
        num_tokens: T.int32,
        capacity: T.int32,
    ):
        with T.Kernel(T.ceildiv(width, span), capacity, threads=_THREADS) as (
            bx,
            row,
        ):
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            col = bx * span + tx * vector
            if col < width:
                T.call_extern(
                    "handle",
                    "vllm_dspark_context_b128",
                    T.address_of(aux[row, col]),
                    T.address_of(source0[row, col]),
                    T.address_of(aux[row, width + col]),
                    T.address_of(source1[row, col]),
                    T.address_of(aux[row, 2 * width + col]),
                    T.address_of(source2[row, col]),
                    row < num_tokens,
                )
            if (bx == 0) & (tx == 0) & (row >= num_tokens):
                positions[row] = T.int64(0)
                for g in T.serial(groups):
                    slot_mappings[g, row] = T.int64(PAD_SLOT_ID)

    return context_inputs


def context_inputs_key(width: int, dtype: torch.dtype) -> tuple:
    return (width, _DTYPES[dtype])


@cache
def _bound(*key):
    """The kernel of ``key``, resolved once (after the kernel cache freezes, an
    undeclared key raises by name)."""
    return context_inputs_kernel(*key)


def context_inputs(
    aux: torch.Tensor,
    sources: list[torch.Tensor],
    positions: torch.Tensor,
    slot_mappings: torch.Tensor,
    num_tokens: int,
    capacity: int,
) -> None:
    """``aux[:num_tokens]`` from ``sources`` side by side, and rows ``num_tokens``
    to ``capacity`` of ``aux``, ``positions`` and every group's ``slot_mappings``
    scrubbed."""
    width = aux.shape[1] // NUM_SOURCES
    if len(sources) != NUM_SOURCES or aux.shape[1] != NUM_SOURCES * width:
        raise ValueError(f"DSpark context inputs take {NUM_SOURCES} sources")
    vector = _VECTOR_BYTES // aux.element_size()
    for source in (aux, *sources):
        if (
            source.dtype != aux.dtype
            or source.stride(-1) != 1
            or source.stride(0) % vector
            or source.data_ptr() % _VECTOR_BYTES
        ):
            raise ValueError(
                "DSpark context rows must be 16-byte aligned rows of the aux dtype"
            )
    _bound(*context_inputs_key(width, aux.dtype))(
        aux, *sources, positions, slot_mappings, num_tokens, capacity
    )
