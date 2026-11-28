# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared MXFP4 decode helper and candidate block remapping for LSA."""
import tilelang
import tilelang.language as T

from . import pdl
from .indexer import CANDIDATE_BLOCKS

BLOCK = 8


def decode(byte, scale, dimension):
    code = (T.cast(byte, T.int32) >> ((dimension % 2) * 4)) & 15
    magnitude = code & 7
    value = T.if_then_else(
        magnitude < 2,
        T.cast(magnitude, T.float32) * 0.5,
        (1.0 + T.cast(magnitude & 1, T.float32) * 0.5)
        * T.cast(1 << T.max(magnitude // 2 - 1, 0), T.float32),
    )
    value = value * T.reinterpret(T.cast(scale, T.int32) << 23, T.float32)
    return T.if_then_else((code & 8) != 0, -value, value)


@tilelang.jit
def remap_blocks(capacity: int):
    rows = T.dynamic("rows")

    @T.prim_func
    def remap(
        Selected: T.Tensor([rows, CANDIDATE_BLOCKS], T.int32),
        Candidates: T.Tensor([rows, capacity // BLOCK], T.int32),
        Original: T.Tensor([rows, CANDIDATE_BLOCKS], T.int32),
    ):
        with T.Kernel(
            tilelang.cdiv(CANDIDATE_BLOCKS, 128), rows, threads=128
        ) as (tile, row):
            pdl.chain()
            for i in T.Parallel(128):
                slot = tile * 128 + i
                entry = Selected[row, slot]
                Original[row, slot] = T.if_then_else(
                    entry >= 0, Candidates[row, T.max(entry, 0)], -1
                )

    return remap
