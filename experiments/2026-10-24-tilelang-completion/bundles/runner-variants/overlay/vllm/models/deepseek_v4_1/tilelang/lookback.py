# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Engram lookback window, ``model_state._gather_lookback_kernel``'s contract.

Each row of ``lookback`` ([rows, depth], int32) holds the ids of the ``depth``
tokens before its request's computed tokens, oldest first, from the runner's token
history: ``all_token_ids[req, num_computed - depth + j]``, -1 before the start of
the history and in every row past the batch. In Triton's form: a block a row
and a thread an element of the window, the request index, its computed count
and the token predicated loads (a masked lane keeps 0, or -1 for the token, and
never waits on memory) and the store predicated, so no load waits on a branch.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
from functools import cache

import torch

from vllm.tilelang_utils import T, tilelang

_THREADS = 32  # a warp a row: the window is a few tokens
_DTYPES = {torch.int32: "int32", torch.int64: "int64"}
_LOAD_BITS = {"int32": 32, "int64": 64}

_HEADER = r"""
__device__ __forceinline__ long long vllm_lookback_load_if_b64(const void* p,
                                                               long long other,
                                                               bool ok) {
  long long v = other;
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n"
               " @q ld.global.b64 %0, [%1];\n}"
               : "+l"(v) : "l"(p), "r"((int)ok));
  return v;
}
__device__ __forceinline__ int vllm_lookback_load_if_b32(const void* p, int other,
                                                        bool ok) {
  int v = other;
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n"
               " @q ld.global.b32 %0, [%1];\n}"
               : "+r"(v) : "l"(p), "r"((int)ok));
  return v;
}
__device__ __forceinline__ void vllm_lookback_store_if_b32(void* p, int v, bool ok) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %2, 0;\n"
               " @q st.global.b32 [%0], %1;\n}"
               :: "l"(p), "r"(v), "r"((int)ok) : "memory");
}
"""


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)
def gather_lookback_kernel(depth: int, idx_dtype: str):
    if depth > _THREADS:
        raise ValueError(f"a lookback window of {depth} exceeds a block's {_THREADS}")
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
        with T.Kernel(rows, threads=_THREADS) as b:
            T.import_source(_HEADER)
            j = T.get_thread_binding()
            inside = j < depth
            live = inside & (b < num_reqs)
            req = T.alloc_var(
                T.int64,
                init=T.cast(
                    T.call_extern(
                        idx_dtype,
                        f"vllm_lookback_load_if_b{_LOAD_BITS[idx_dtype]}",
                        T.address_of(idx_mapping[b]),
                        T.cast(0, idx_dtype),
                        live,
                    ),
                    T.int64,
                ),
            )
            computed = T.alloc_var(
                T.int32,
                init=T.call_extern(
                    "int32",
                    "vllm_lookback_load_if_b32",
                    T.address_of(num_computed_tokens[req]),
                    0,
                    live,
                ),
            )
            position = T.alloc_var(T.int32, init=computed - depth + j)
            token = T.alloc_var(
                T.int32,
                init=T.call_extern(
                    "int32",
                    "vllm_lookback_load_if_b32",
                    T.address_of(all_token_ids[req, position]),
                    -1,
                    live & (position >= 0),
                ),
            )
            T.call_extern(
                "handle",
                "vllm_lookback_store_if_b32",
                T.address_of(lookback[b, j]),
                token,
                inside,
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
