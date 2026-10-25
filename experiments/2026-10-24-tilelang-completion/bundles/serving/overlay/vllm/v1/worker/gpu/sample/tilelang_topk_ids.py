# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The top-k token ids of ``compute_topk_scores``, as ``torch.topk`` returns them.

- ``logprob_token_ids``: ``torch.cat((sampled[:, None], torch.topk(logits,
  k).indices), 1)``, int64 ``[rows, 1 + k]``
- ``topk_token_ids``: ``torch.topk(logits, k).indices.to(torch.int32)``

Both are one launch of ``topk_ids_kernel`` and return PyTorch 2.13.0's CUDA indices,
in its order, ties, signed zeros and NaN included, for ``k`` up to ``TOPK_IDS_MAX``.

PyTorch's path (``aten/src/ATen/native/cuda`` at v2.13.0) for a ``[rows, vocab]``
float32, bfloat16 or float16 input, ``largest=True, sorted=True``:

1. ``topk_out_cuda`` (TensorTopK.cpp) runs ``launch_gather_topk_kernel`` on the
   input made contiguous. Values are ranked by ``TopKTypeConfig<T>::convert``
   (SortingRadixSelect.cuh): an unsigned key that orders as the value, ``-0``
   below ``+0``, every NaN equal and above ``+inf``. A radix select finds the
   k-th largest key; ``gatherTopK`` writes every element whose key is greater, in
   column order, then the lowest-column elements whose key equals it, in column
   order. That holds on both slice paths (``sbtopk``: a block a row, its prefix
   scans in thread order; ``mbtopk``, which ``should_use_multiblock`` picks for
   every vocabulary and row count here: blocks in column order, each block's
   ``BlockScan``s in item then thread order). The selected set is the ``k`` largest
   by (key, then lower column).
2. ``sortKeyValueInplace(values, indices, dim, largest)`` (Sort.cu), unstable, for
   ``k <= 32``: ``SmallBitonicSort``, ``bitonicSortKVInPlace`` (SortUtils.cuh) on
   32 slots, two a thread for 16 threads, the slots past ``k`` invalid. Its
   comparator is ``GTOp<T, true>`` (SortingCommon.cuh): a NaN above any other value
   and NaNs equal; otherwise ``>`` on the values, so ``-0`` and ``+0`` are equal. A
   compare-exchange swaps equal values in some stages (``bitonicSwap``'s ``swap ==
   dir``), so ties leave in an order only the network decides: ``k`` equal values
   at columns ``0..7`` come out ``7, 6, 4, 5, 1, 0, 2, 3``. ``k == 1`` is not sorted.

Above 32, PyTorch sorts with another algorithm (``WarpMergeSort`` up to 128, a block
radix sort up to 4096, a segmented sort beyond), each with its own tie order, and a
full-vocabulary request (``prompt_logprobs=-1``) is a sort of the whole row:
``compute_topk_scores`` keeps ``torch.topk`` there.

The kernel, a block of 512 threads a row:

- Each warp keeps a descending list of 32 entries, one a lane, whose first ``k``
  are the ``k`` largest it has seen. An entry is the element's key (the key of its
  FP32 value: widening a bfloat16 or float16 is exact, so the key orders as the
  16-bit one) above the complement of its column: unique, and ordered as PyTorch
  selects. Each thread loads 32 elements of a tile, and the warp inserts those
  above its k-th entry, one at a time (``vllm_tki_insert``); a value below the
  k-th entry's value is not looked at again.
- The warps' first ``k`` entries meet in shared memory; each entry's rank is the
  count of larger ones (a binary search in each list), and the ``k`` ranked below
  ``k`` are the selection, in descending order.
- One warp lays them out as ``gatherTopK`` wrote them and runs the bitonic network
  across its lanes (``vllm_tki_order``): the comparisons are made on the keys,
  which give ``GTOp``'s answers bit for bit (no flush-to-zero).
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
from collections.abc import Sequence

import torch

from vllm.tilelang_utils import T, tilelang
from vllm.v1.worker.gpu.sample.tilelang_sampling import (
    _DTYPES,
    _PASS,
    _call,
    _kernel,
    _runs,
    _triton_run,
    declare,
)

# The largest k the kernel serves: PyTorch's bitonic sort of 32 slots.
TOPK_IDS_MAX = 32
_THREADS = 512
_WARPS = _THREADS // 32
# Elements a thread loads per tile, and a tile's columns.
_PER = 32
_TILE = _THREADS * _PER

_HEADER = r"""
// TopKTypeConfig<float>::convert: an unsigned key that orders as the value, -0 below
// +0, every NaN above +inf.
__device__ __forceinline__ unsigned int vllm_tki_key(float v) {
  const unsigned int x = __float_as_uint(v);
  if ((x & 0x7fffffffu) > 0x7f800000u) return 0xffffffffu;
  return x ^ ((x & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

// An element's entry: its key, then the complement of its column, so that entries
// are unique and the larger one is the one torch.topk selects first.
__device__ __forceinline__ unsigned long long vllm_tki_entry(float v, int col) {
  return ((unsigned long long)vllm_tki_key(v) << 32) |
         (unsigned long long)(~(unsigned int)col);
}

// Whether a value can rank above an entry of value `floor`: it is not below it (a
// NaN value, or a NaN floor, passes).
__device__ __forceinline__ int vllm_tki_pass(float v, float floor) {
  return !(v < floor);
}

// The value of an entry's key; -inf for an empty entry (0).
__device__ __forceinline__ float vllm_tki_floor(unsigned long long entry) {
  const unsigned int key = (unsigned int)(entry >> 32);
  if (key == 0u) return __uint_as_float(0xff800000u);
  return __uint_as_float(key ^ ((key & 0x80000000u) ? 0x80000000u : 0xffffffffu));
}

__device__ __forceinline__ unsigned int vllm_tki_ballot(int flag) {
  return __ballot_sync(0xffffffffu, flag);
}

__device__ __forceinline__ unsigned long long vllm_tki_lane(unsigned long long v,
                                                            int lane) {
  return __shfl_sync(0xffffffffu, v, lane);
}

// The warp's list (lane i holds its i-th largest entry, 0 where empty) with the
// entries of the lanes in `mask` inserted, the lowest lane first: the entries below
// one move down a lane, and the last falls off.
__device__ __forceinline__ unsigned long long vllm_tki_insert(unsigned long long list,
                                                              unsigned long long entry,
                                                              unsigned int mask) {
  const int lane = threadIdx.x & 31;
  while (mask != 0u) {
    const int src = __ffs(mask) - 1;
    mask &= mask - 1u;
    const unsigned long long x = __shfl_sync(0xffffffffu, entry, src);
    const unsigned long long up = __shfl_up_sync(0xffffffffu, list, 1);
    list = (lane > 0 && x > up) ? up : (x > list ? x : list);
  }
  return list;
}

__device__ __forceinline__ bool vllm_tki_zero(unsigned int key) {
  return key == 0x80000000u || key == 0x7fffffffu;
}

// GTOp<scalar_t, true> on the values of keys a and b: a NaN is above any other
// value and equal to a NaN; otherwise a > b, -0 equal to +0.
__device__ __forceinline__ bool vllm_tki_gt(unsigned int a, unsigned int b) {
  return b != 0xffffffffu && a > b && !(vllm_tki_zero(a) && vllm_tki_zero(b));
}

// bitonicSwap between this lane's slot and the slot `stride` lanes away: the lower
// slot is A, the upper B; an invalid slot (column -1) sorts last.
__device__ __forceinline__ void vllm_tki_exchange(unsigned int& key, int& col,
                                                  int stride, bool dir) {
  const int lane = threadIdx.x & 31;
  const unsigned int other_key = __shfl_xor_sync(0xffffffffu, key, stride);
  const int other_col = __shfl_xor_sync(0xffffffffu, col, stride);
  const bool lower = (lane & stride) == 0;
  const unsigned int key_a = lower ? key : other_key;
  const unsigned int key_b = lower ? other_key : key;
  const bool valid_a = (lower ? col : other_col) >= 0;
  const bool valid_b = (lower ? other_col : col) >= 0;
  const bool swap = (vllm_tki_gt(key_a, key_b) && valid_a) || !valid_b;
  if (swap == dir) {
    key = other_key;
    col = other_col;
  }
}

// torch.topk's order of its k <= 32 selected entries, lane i holding the entry
// ranked i (i < k): gatherTopK's layout (the entries above the k-th key in column
// order, then those equal to it, in column order already), then
// bitonicSortKVInPlace over 32 slots, a slot a lane. Lane p returns the column at
// position p.
__device__ __forceinline__ int vllm_tki_order(unsigned long long entry, int k) {
  const int lane = threadIdx.x & 31;
  const unsigned int key = (unsigned int)(entry >> 32);
  const int col = (int)~(unsigned int)entry;
  const unsigned int kth = __shfl_sync(0xffffffffu, key, k - 1);
  const int above = (lane < k && key != kth) ? 1 : 0;
  int before = 0;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const int above_j = __shfl_sync(0xffffffffu, above, j);
    const int col_j = __shfl_sync(0xffffffffu, col, j);
    before += (above_j != 0 && col_j < col) ? 1 : 0;
  }
  const int at = above ? before : lane;
  unsigned int slot_key = 0u;
  int slot_col = -1;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const int at_j = __shfl_sync(0xffffffffu, at, j);
    const unsigned int key_j = __shfl_sync(0xffffffffu, key, j);
    const int col_j = __shfl_sync(0xffffffffu, col, j);
    if (j < k && at_j == lane) {
      slot_key = key_j;
      slot_col = col_j;
    }
  }
  // bitonicSort<32>: the flag of thread t, (t & (size / 2)) != 0, is bit `size` of
  // its slots' positions.
#pragma unroll
  for (int size = 2; size < 32; size <<= 1) {
    const bool dir = (lane & size) != 0;
#pragma unroll
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      vllm_tki_exchange(slot_key, slot_col, stride, dir);
    }
  }
#pragma unroll
  for (int stride = 16; stride > 0; stride >>= 1) {
    vllm_tki_exchange(slot_key, slot_col, stride, false);
  }
  return slot_col;
}
"""


@tilelang.jit(pass_configs=_PASS)
def topk_ids_kernel(logits_dtype: str, ids_dtype: str, run: int):
    """A block a row: its ``k = width - first`` top columns in ``torch.topk``'s order,
    at ``out[row, first:]``. int64 ids take the sampled token at column 0 (``first``
    1); int32 ids have no sampled column. A thread loads its elements of a tile in
    runs of ``run`` (``_triton_run``: one 16-byte vector, the launch checked the
    alignment, or one element)."""
    rows, vocab, width = T.dynamic("rows, vocab, width")
    logits_stride = T.dynamic("logits_stride", "int64")
    with_sampled = ids_dtype == "int64"
    first = 1 if with_sampled else 0
    vectors = _PER // run

    def column(base, e: int, tx):
        """The column of thread ``tx``'s ``e``-th element of the tile at ``base``:
        runs of ``run`` columns, ``_THREADS * run`` apart."""
        return base + (e // run) * (_THREADS * run) + tx * run + e % run

    @T.macro
    def load(logits, raw, row, base, tx, guarded):
        """The thread's elements of the tile at ``base``; a guarded tile (the row's
        last, partial one) loads only the columns below ``vocab``."""
        for r in tuple(range(vectors)):
            col0 = base + r * (_THREADS * run) + tx * run
            if run == 1:
                if guarded:
                    if col0 < vocab:
                        raw[r] = logits[row, col0]
                else:
                    raw[r] = logits[row, col0]
            elif guarded:
                if col0 + run <= vocab:
                    for v in T.vectorized(run):
                        raw[r * run + v] = logits[row, col0 + v]
                else:
                    for v in T.unroll(run):
                        if col0 + v < vocab:
                            raw[r * run + v] = logits[row, col0 + v]
            else:
                for v in T.vectorized(run):
                    raw[r * run + v] = logits[row, col0 + v]

    @T.macro
    def scan(raw, base, tx, k, best, thr, floor, bits, entry, mask, guarded):
        """Insert the tile's elements that rank above the warp's k-th entry
        ``thr[0]`` (of value ``floor[0]``) into the warp's list ``best[0]``. Every
        element is tested against the floor first, which consumes the loads before
        any branch; an element position some lane passes then takes its exact
        entries, and the lanes whose entry is above ``thr[0]`` are inserted."""
        bits[0] = T.uint32(0)
        for e in tuple(range(_PER)):
            passes = T.call_extern(
                "int32", "vllm_tki_pass", T.cast(raw[e], T.float32), floor[0]
            )
            if guarded:
                passes = T.if_then_else(column(base, e, tx) < vocab, passes, 0)
            bits[0] = bits[0] | T.if_then_else(
                passes != 0, T.uint32(1 << e), T.uint32(0)
            )
        for e in tuple(range(_PER)):
            flag = (bits[0] & T.uint32(1 << e)) != T.uint32(0)
            if T.call_extern("uint32", "vllm_tki_ballot", T.cast(flag, T.int32)) != 0:
                entry[0] = T.call_extern(
                    "uint64",
                    "vllm_tki_entry",
                    T.cast(raw[e], T.float32),
                    column(base, e, tx),
                )
                mask[0] = T.call_extern(
                    "uint32",
                    "vllm_tki_ballot",
                    T.cast(flag & (entry[0] > thr[0]), T.int32),
                )
                if mask[0] != 0:
                    best[0] = T.call_extern(
                        "uint64", "vllm_tki_insert", best[0], entry[0], mask[0]
                    )
                    thr[0] = T.call_extern("uint64", "vllm_tki_lane", best[0], k - 1)
                    floor[0] = T.call_extern("float32", "vllm_tki_floor", thr[0])

    @T.macro
    def select(logits, out, row, tx, lists, chosen):
        """The row's top ``k`` into ``out[row, first:first + k]``."""
        lane = tx % 32
        warp = tx // 32
        k = width - first
        raw = T.alloc_local((_PER,), logits_dtype)
        best = T.alloc_local((1,), T.uint64)
        thr = T.alloc_local((1,), T.uint64)
        floor = T.alloc_local((1,), T.float32)
        bits = T.alloc_local((1,), T.uint32)
        entry = T.alloc_local((1,), T.uint64)
        mask = T.alloc_local((1,), T.uint32)
        rank = T.alloc_local((1,), T.int32)
        found = T.alloc_local((1,), T.int32)
        picked = T.alloc_local((1,), T.int32)
        if run > 1:
            # The launch checked it: a 16-byte base and a stride of 16 elements.
            T.assume(logits_stride % 16 == 0)
        best[0] = T.uint64(0)
        thr[0] = T.uint64(0)
        floor[0] = -T.infinity(T.float32)

        full = vocab // _TILE
        for t in T.serial(full):
            load(logits, raw, row, t * _TILE, tx, False)
            scan(raw, t * _TILE, tx, k, best, thr, floor, bits, entry, mask, False)
        if full * _TILE < vocab:
            load(logits, raw, row, full * _TILE, tx, True)
            scan(raw, full * _TILE, tx, k, best, thr, floor, bits, entry, mask, True)

        # The warps' lists meet: an entry's rank is the count of larger entries
        # among every list's first k (its own list's: its lane).
        lists[warp, lane] = best[0]
        T.sync_threads()
        rank[0] = 0
        for w in tuple(range(_WARPS)):
            found[0] = 0
            for step in (16, 8, 4, 2, 1):
                found[0] = found[0] + T.if_then_else(
                    (found[0] + (step - 1) < k)
                    & (lists[w, found[0] + (step - 1)] > best[0]),
                    step,
                    0,
                )
            # All 32 larger: the steps reach 31.
            found[0] = found[0] + T.if_then_else(
                (found[0] < k) & (lists[w, found[0]] > best[0]), 1, 0
            )
            rank[0] = rank[0] + found[0]
        if (lane < k) & (best[0] != T.uint64(0)) & (rank[0] < k):
            chosen[rank[0]] = best[0]
        T.sync_threads()

        if warp == 0:
            picked[0] = T.call_extern(
                "int32",
                "vllm_tki_order",
                T.if_then_else(lane < k, chosen[lane], T.uint64(0)),
                k,
            )
            if lane < k:
                out[row, first + lane] = T.cast(picked[0], ids_dtype)

    if with_sampled:

        @T.prim_func
        def topk_ids(
            logits: T.StridedTensor((rows, vocab), (logits_stride, 1), logits_dtype),
            sampled: T.Tensor((rows,), T.int64),
            out: T.Tensor((rows, width), T.int64),
        ):
            with T.Kernel(rows, threads=_THREADS) as row:
                T.import_source(_HEADER)
                tx = T.get_thread_binding()
                lists = T.alloc_shared((_WARPS, 32), T.uint64)
                chosen = T.alloc_shared((32,), T.uint64)
                select(logits, out, row, tx, lists, chosen)
                if tx == 0:
                    out[row, 0] = sampled[row]

        return topk_ids

    @T.prim_func
    def topk_ids_only(
        logits: T.StridedTensor((rows, vocab), (logits_stride, 1), logits_dtype),
        out: T.Tensor((rows, width), T.int32),
    ):
        with T.Kernel(rows, threads=_THREADS) as row:
            T.import_source(_HEADER)
            tx = T.get_thread_binding()
            lists = T.alloc_shared((_WARPS, 32), T.uint64)
            chosen = T.alloc_shared((32,), T.uint64)
            select(logits, out, row, tx, lists, chosen)

    return topk_ids_only


def topk_ids_key(logits_dtype: torch.dtype, ids_dtype: torch.dtype, run: int) -> tuple:
    return (_DTYPES[logits_dtype], _DTYPES[ids_dtype], run)


# The ids each path writes: the fast path's int64 rows with the sampled token, the
# custom-ids path's int32 top-k.
_IDS_DTYPES = (torch.int64, torch.int32)


def topk_ids_calls(logits_dtypes: Sequence[torch.dtype]) -> list:
    """The calls for logits of each of ``logits_dtypes``: both ids layouts, in both
    load layouts a launch may take (``_runs``)."""
    return [
        _call(topk_ids_kernel, *topk_ids_key(dtype, ids_dtype, run))
        for dtype in dict.fromkeys(logits_dtypes)
        for ids_dtype in _IDS_DTYPES
        for run in _runs(dtype)
    ]


def register_topk_ids(logits: Sequence[tuple[torch.dtype, int]]) -> None:
    """Declare ``topk_ids_calls`` for logits given as ``(dtype, row stride)`` pairs
    with the runner's JIT registry (a no-op where none collects). The stride and
    ``k`` are launch arguments, not part of the key."""
    dtypes = tuple(dtype for dtype, _ in logits)
    declare(lambda: topk_ids_calls(dtypes))


def _check(logits: torch.Tensor, num_logprobs: int) -> None:
    if not 0 < num_logprobs <= TOPK_IDS_MAX:
        raise ValueError(
            f"top-k ids take 1 to {TOPK_IDS_MAX} tokens, not {num_logprobs}"
        )
    if num_logprobs > logits.shape[1]:
        # torch.topk's own check.
        raise RuntimeError("selected index k out of range")


def _launch(
    logits: torch.Tensor, out: torch.Tensor, sampled: torch.Tensor | None
) -> None:
    if logits.stride(1) != 1:
        raise ValueError("top-k ids read logits rows with a unit column stride")
    kernel = _kernel(
        topk_ids_kernel, *topk_ids_key(logits.dtype, out.dtype, _triton_run(logits))
    )
    if sampled is None:
        kernel(logits, out)
    else:
        kernel(logits, sampled, out)


def logprob_token_ids(
    logits: torch.Tensor,  # [rows, vocab]
    num_logprobs: int,
    sampled: torch.Tensor,  # [rows]
) -> torch.Tensor:
    """``torch.cat((sampled[:, None], torch.topk(logits, num_logprobs).indices),
    1)``: ``[rows, 1 + num_logprobs]`` int64, ``num_logprobs`` at most
    ``TOPK_IDS_MAX``."""
    _check(logits, num_logprobs)
    rows = logits.shape[0]
    out = torch.empty((rows, 1 + num_logprobs), dtype=torch.int64, device=logits.device)
    if rows:
        _launch(logits, out, sampled.to(torch.int64).contiguous())
    return out


def topk_token_ids(logits: torch.Tensor, num_logprobs: int) -> torch.Tensor:
    """``torch.topk(logits, num_logprobs).indices.to(torch.int32)``: ``[rows,
    num_logprobs]`` int32, ``num_logprobs`` at most ``TOPK_IDS_MAX``."""
    _check(logits, num_logprobs)
    rows = logits.shape[0]
    out = torch.empty((rows, num_logprobs), dtype=torch.int32, device=logits.device)
    if rows:
        _launch(logits, out, None)
    return out
