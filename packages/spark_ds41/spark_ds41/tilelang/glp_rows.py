# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLP steering's row marks for a step, from its requests' marks.

``rows[t]`` is ``marks[idx_mapping[r]]`` for the request ``r`` whose query holds
row ``t`` (``query_start_loc[r] <= t < query_start_loc[r + 1]``), and 0 for a row
past the last query (graph padding). The query bounds are read on the GPU, as the
step's attention reads them: adaptive verification places a verifying request's
rows there, not on the host. A thread a row scans the step's requests; a step
holds at most ``max_num_seqs`` of them.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
from functools import cache

import torch

from vllm.tilelang_utils import T, tilelang

_THREADS = 128
_DTYPES = {torch.int32: "int32", torch.int64: "int64"}


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def mark_rows_kernel(idx_dtype: str):
    tokens = T.dynamic("tokens")
    num_reqs = T.dynamic("num_reqs")
    starts_len = T.dynamic("starts_len")
    max_reqs = T.dynamic("max_reqs")

    @T.prim_func
    def mark_rows(
        rows: T.Tensor((tokens,), T.int32),
        idx_mapping: T.Tensor((num_reqs,), idx_dtype),
        query_start_loc: T.Tensor((starts_len,), T.int32),
        marks: T.Tensor((max_reqs,), T.int32),
    ):
        with T.Kernel(T.ceildiv(tokens, _THREADS), threads=_THREADS) as b:
            t = b * _THREADS + T.get_thread_binding()
            mark = T.alloc_var(T.int32, init=0)
            if t < tokens:
                for r in T.serial(num_reqs):
                    if (query_start_loc[r] <= t) & (t < query_start_loc[r + 1]):
                        mark = marks[idx_mapping[r]]
                rows[t] = mark

    return mark_rows


@cache
def _bound(idx_dtype: str):
    """The kernel for ``idx_dtype``, resolved once (after the kernel cache freezes,
    an undeclared key raises by name)."""
    return mark_rows_kernel(idx_dtype)


def mark_rows(
    rows: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    marks: torch.Tensor,
) -> None:
    """Fill ``rows`` (int32) with the marks of the requests whose queries hold
    them; ``query_start_loc`` has one more entry than ``idx_mapping``."""
    if rows.numel() == 0:
        return
    if query_start_loc.shape[0] != idx_mapping.shape[0] + 1:
        raise ValueError(
            f"{idx_mapping.shape[0]} requests need {idx_mapping.shape[0] + 1} "
            f"query bounds, not {query_start_loc.shape[0]}"
        )
    _bound(_DTYPES[idx_mapping.dtype])(rows, idx_mapping, query_start_loc, marks)


__all__ = ["mark_rows", "mark_rows_kernel"]
