# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Engram lookback window, ``model_state._gather_lookback_kernel``'s contract.

Each row of ``lookback`` ([rows, depth], int32) holds the ids of the ``depth``
tokens before its request's computed tokens, oldest first, from the runner's token
history: ``all_token_ids[req, num_computed - depth + j]``, -1 before the start of
the history and in every row past the batch. A thread an element.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
from functools import cache

import torch

from vllm.tilelang_utils import T, tilelang

_THREADS = 128
_DTYPES = {torch.int32: "int32", torch.int64: "int64"}


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def gather_lookback_kernel(depth: int, idx_dtype: str):
    rows = T.dynamic("rows")
    num_reqs = T.dynamic("num_reqs")
    counts_len = T.dynamic("counts_len")
    history_rows = T.dynamic("history_rows")
    history_cols = T.dynamic("history_cols")
    history_stride = T.dynamic("history_stride")

    @T.prim_func
    def gather_lookback(
        lookback: T.Tensor((rows, depth), T.int32),
        idx_mapping: T.Tensor((num_reqs,), idx_dtype),
        num_computed_tokens: T.Tensor((counts_len,), T.int32),
        all_token_ids: T.StridedTensor(
            (history_rows, history_cols), (history_stride, 1), T.int32
        ),
    ):
        with T.Kernel(T.ceildiv(rows * depth, _THREADS), threads=_THREADS) as bx:
            element = bx * _THREADS + T.get_thread_binding()
            if element < rows * depth:
                b = element // depth
                j = element % depth
                live = b < num_reqs
                req = T.if_then_else(live, T.cast(idx_mapping[b], T.int64), T.int64(0))
                computed = T.if_then_else(live, num_computed_tokens[req], 0)
                position = computed - depth + j
                lookback[b, j] = T.if_then_else(
                    live and position >= 0, all_token_ids[req, position], -1
                )

    return gather_lookback


def gather_lookback_key(depth: int, idx_dtype: str) -> tuple:
    return (depth, idx_dtype)


@cache
def _bound(*key):
    """The kernel of ``key``, resolved once (after the kernel cache freezes, an
    undeclared key raises by name)."""
    return gather_lookback_kernel(*key)


def gather_lookback(
    lookback: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> None:
    """Fill every row of ``lookback``: the batch's requests', then -1."""
    if lookback.numel() == 0:
        return
    key = gather_lookback_key(lookback.shape[1], _DTYPES[idx_mapping.dtype])
    _bound(*key)(lookback, idx_mapping, num_computed_tokens, all_token_ids)
