# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang kernels of copy drafting (``copy_drafts.CopyDrafter``).

A request's context is its own token history, ``all_token_ids[slot, :total_len]``:
its prompt and output, the last token being the step's last sampled one (draft
zero with sink rejection). A copy proposes the tokens that followed the latest
earlier position whose preceding tokens equal the context's tail, at the longest
of the levels ``m, 2m, 4m, ...`` (``CopyDraftsConfig.levels``) that has one.
Nothing outside the request's own row up to its length is read, and nothing is
kept between steps but the request's copy state, reset with the slot.

- ``copy_match``: a grid of (requests, ``CHUNKS``) blocks. A block loads the
  context's last ``MAX_LEVEL`` tokens (the key) and, for each window end ``e``
  of its share of ``[m, total_len)``, the number of tokens before ``e`` that
  equal the key's (its run, at most ``MAX_LEVEL`` and ``e``); per level, the
  largest ``e`` whose run reaches it. A thread's ends ascend, so its last hit is
  its largest; the block takes the maximum over its threads (warp shuffles, then
  the warps), an order-free reduction, and writes ``sources[r, chunk, level]``
  (-1 where none). No atomics: the result is the same on every launch.
- ``copy_select``: a warp a request. First the copy state of the request's last
  proposal is updated from this step's verification of it: a copy whose live
  verified drafts (its admitted drafts, at most the dead-row cut) were all
  accepted doubles its cap (from ``initial`` up to the lookahead), a rejection
  or a proposal without a copy returns it to ``initial``, and the accepted and
  judged drafts update the request's acceptance estimate (decayed counts over a
  prior). Then the longest
  level with a source over the chunks, its latest source, and the proposal:
  ``min(cap, total_len - source)`` tokens from the source, each at the estimate
  as its confidence; DSpark's drafts and confidences where no level has one.
  It also writes the planned confidences and the allocation the scheduler lays
  out the step after next from: as if this proposal is accepted in full, the
  next one offers ``min(2 cap, lookahead, total_len - source)`` drafts; DSpark's
  confidences and block otherwise.

Every launch has a fixed grid for its request count, no host synchronization and
no dynamic allocation, so it can be captured in a CUDA graph.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
import torch

from vllm.config.speculative import CopyDraftsConfig
from vllm.tilelang_utils import T, tilelang
from vllm.v1.worker.gpu.tilelang_inputs import (
    _DTYPES,
    _PASS,
    RUNNER_IDX_DTYPE,
    _call,
    _kernel,
)

MAX_LEVEL = CopyDraftsConfig.MAX_LEVEL
# Blocks a request's history spreads over (copy_match), whatever its length.
CHUNKS = 32
_MATCH_THREADS = 256
_MATCH_WARPS = _MATCH_THREADS // 32
_SELECT_THREADS = 32
# The acceptance estimate: (hits + PRIOR_HITS) / (trials + PRIOR_TRIALS), the
# counts decayed by DECAY at each verified copy, so a copy that starts is
# trusted at 0.95 and the estimate follows the request's last ten or so copies.
PRIOR_HITS = 19.0
PRIOR_TRIALS = 20.0
DECAY = 0.9
# The cap doubles at most this many times (initial << 16 exceeds any lookahead).
MAX_DOUBLINGS = 16
# Copy state columns: int32 (doublings, pending drafts), float32 (hits, trials).
STATE_COLUMNS = 2

assert MAX_LEVEL == _MATCH_THREADS, "a thread loads one key token"


def num_levels(min_match: int) -> int:
    return len(CopyDraftsConfig(min_match=min_match).levels())


@tilelang.jit(pass_configs=_PASS)
def copy_match_kernel(min_match: int, idx_dtype: str):
    """See the module docstring: each block's latest window end per level."""
    levels = CopyDraftsConfig(min_match=min_match).levels()
    count = len(levels)
    num_reqs = T.dynamic("num_reqs")
    history_rows, history_cols, history_stride = T.dynamic(
        "history_rows, history_cols, history_stride"
    )
    total_rows = T.dynamic("total_rows")
    span = CHUNKS * _MATCH_THREADS  # window ends a pass of the grid covers

    @T.prim_func
    def copy_match(
        idx_mapping: T.Tensor((num_reqs,), idx_dtype),
        history: T.StridedTensor(
            (history_rows, history_cols), (history_stride, 1), T.int32
        ),
        total_len: T.Tensor((total_rows,), T.int32),
        sources: T.Tensor((num_reqs, CHUNKS, count), T.int32),
    ):
        with T.Kernel(num_reqs, CHUNKS, threads=_MATCH_THREADS) as (r, c):
            tx = T.get_thread_binding()
            key = T.alloc_shared((MAX_LEVEL,), T.int32)
            partial = T.alloc_shared((_MATCH_WARPS, count), T.int32)
            best = T.alloc_local((count,), T.int32)
            slot = T.alloc_var(T.int64, init=T.cast(idx_mapping[r], T.int64))
            length = T.alloc_var(T.int32, init=0)
            if slot >= 0:
                length = total_len[slot]
            # The key: the context's last tokens, latest first.
            key[tx] = -1
            if tx < length:
                key[tx] = history[slot, length - 1 - tx]
            for i in tuple(range(count)):
                best[i] = -1
            T.sync_threads()
            ends = T.max(length - min_match, 0)  # window ends min_match .. length - 1
            for k in T.serial(T.ceildiv(ends, span)):
                e = min_match + k * span + c * _MATCH_THREADS + tx
                if e < length:
                    run = T.alloc_var(T.int32, init=0)
                    # The shortest level first: most ends fail on its first token.
                    # A token is loaded only while the run reaches it.
                    for j in T.unroll(min_match):
                        if run == j:  # noqa: SIM102
                            if history[slot, e - 1 - j] == key[j]:
                                run = j + 1
                    # A longer level only past a full match of the one before.
                    for i in tuple(range(1, count)):
                        if run == levels[i - 1]:
                            for j in T.serial(levels[i - 1], levels[i]):
                                if run == j and j < e:  # noqa: SIM102
                                    if history[slot, e - 1 - j] == key[j]:
                                        run = j + 1
                    for i in tuple(range(count)):
                        if run >= levels[i]:
                            best[i] = e
            for i in tuple(range(count)):
                for distance in (16, 8, 4, 2, 1):
                    best[i] = T.max(best[i], T.shfl_xor(best[i], distance))
                if tx % 32 == 0:
                    partial[tx // 32, i] = best[i]
            T.sync_threads()
            if tx < count:
                latest = T.alloc_var(T.int32, init=-1)
                for w in T.unroll(_MATCH_WARPS):
                    latest = T.max(latest, partial[w, tx])
                sources[r, c, tx] = latest

    return copy_match


@tilelang.jit(pass_configs=_PASS)
def copy_select_kernel(
    min_match: int,
    lookahead: int,
    block: int,
    initial: int,
    idx_dtype: str,
):
    """See the module docstring: the copy state's update and the proposal."""
    count = num_levels(min_match)
    num_reqs = T.dynamic("num_reqs")
    history_rows, history_cols, history_stride = T.dynamic(
        "history_rows, history_cols, history_stride"
    )
    total_rows = T.dynamic("total_rows")
    draft_rows, draft_stride = T.dynamic("draft_rows, draft_stride")
    conf_rows, conf_stride = T.dynamic("conf_rows, conf_stride")
    state_rows = T.dynamic("state_rows")
    passes = -(-lookahead // _SELECT_THREADS)

    @T.prim_func
    def copy_select(
        idx_mapping: T.Tensor((num_reqs,), idx_dtype),
        history: T.StridedTensor(
            (history_rows, history_cols), (history_stride, 1), T.int32
        ),
        total_len: T.Tensor((total_rows,), T.int32),
        sources: T.Tensor((num_reqs, CHUNKS, count), T.int32),
        dspark_drafts: T.StridedTensor((draft_rows, block), (draft_stride, 1), T.int64),
        dspark_confidences: T.StridedTensor(
            (conf_rows, block), (conf_stride, 1), T.float32
        ),
        num_sampled: T.Tensor((num_reqs,), T.int32),
        verified: T.Tensor((num_reqs,), T.int32),
        cut: T.Tensor((num_reqs,), T.int32),
        int_state: T.Tensor((state_rows, STATE_COLUMNS), T.int32),
        float_state: T.Tensor((state_rows, STATE_COLUMNS), T.float32),
        drafts: T.Tensor((num_reqs, lookahead), T.int64),
        confidences: T.Tensor((num_reqs, lookahead), T.float32),
        planned: T.Tensor((num_reqs, lookahead), T.float32),
        alloc: T.Tensor((num_reqs,), T.int32),
    ):
        with T.Kernel(num_reqs, threads=_SELECT_THREADS) as r:
            tx = T.get_thread_binding()
            # (source, offered, planned) and the estimate, for the whole warp.
            chosen = T.alloc_shared((3,), T.int32)
            estimate = T.alloc_shared((1,), T.float32)
            slot = T.alloc_var(T.int64, init=T.cast(idx_mapping[r], T.int64))
            if tx == 0:
                doublings = T.alloc_var(T.int32, init=int_state[slot, 0])
                pending = int_state[slot, 1]
                hits = T.alloc_var(T.float32, init=float_state[slot, 0])
                trials = T.alloc_var(T.float32, init=float_state[slot, 1])
                if pending > 0 and num_sampled[r] > 0:
                    live = T.min(verified[r], cut[r])
                    if live > 0:
                        accepted = num_sampled[r] - 1
                        doublings = T.if_then_else(
                            accepted >= live, T.min(doublings + 1, MAX_DOUBLINGS), 0
                        )
                        hits = hits * DECAY + T.cast(accepted, T.float32)
                        trials = trials * DECAY + T.cast(
                            T.min(accepted + 1, live), T.float32
                        )
                cap = T.min(T.int32(initial) << doublings, lookahead)
                # The longest level with a source wins; its latest source.
                source = T.alloc_var(T.int32, init=-1)
                for i in tuple(range(count)):
                    latest = T.alloc_var(T.int32, init=-1)
                    for ch in T.serial(CHUNKS):
                        latest = T.max(latest, sources[r, ch, i])
                    if latest >= 0:
                        source = latest
                distance = total_len[slot] - source
                offered = T.if_then_else(source >= 0, T.min(cap, distance), 0)
                after = T.if_then_else(
                    source >= 0, T.min(T.min(2 * cap, lookahead), distance), 0
                )
                # A step without a copy (DSpark's) starts the next copy afresh.
                int_state[slot, 0] = T.if_then_else(source >= 0, doublings, 0)
                int_state[slot, 1] = offered
                float_state[slot, 0] = hits
                float_state[slot, 1] = trials
                chosen[0] = source
                chosen[1] = offered
                chosen[2] = after
                estimate[0] = (hits + PRIOR_HITS) / (trials + PRIOR_TRIALS)
                alloc[r] = T.max(after, block)
            T.sync_threads()
            start = chosen[0]
            for k in T.unroll(passes):
                col = k * _SELECT_THREADS + tx
                if col < lookahead:
                    if start >= 0:
                        drafts[r, col] = T.if_then_else(
                            col < chosen[1],
                            T.cast(
                                history[slot, start + T.min(col, chosen[1] - 1)],
                                T.int64,
                            ),
                            T.int64(0),
                        )
                        confidences[r, col] = T.if_then_else(
                            col < chosen[1], estimate[0], T.float32(0)
                        )
                        planned[r, col] = T.if_then_else(
                            col < chosen[2], estimate[0], T.float32(0)
                        )
                    else:
                        value = T.if_then_else(
                            col < block,
                            dspark_confidences[r, T.min(col, block - 1)],
                            T.float32(0),
                        )
                        drafts[r, col] = T.if_then_else(
                            col < block,
                            dspark_drafts[r, T.min(col, block - 1)],
                            T.int64(0),
                        )
                        confidences[r, col] = value
                        planned[r, col] = value

    return copy_select


def copy_match_key(min_match: int, idx_dtype: torch.dtype) -> tuple:
    return (min_match, _DTYPES[idx_dtype])


def copy_select_key(
    min_match: int, lookahead: int, block: int, initial: int, idx_dtype: torch.dtype
) -> tuple:
    return (min_match, lookahead, block, initial, _DTYPES[idx_dtype])


def copy_match(
    idx_mapping: torch.Tensor,
    history: torch.Tensor,
    total_len: torch.Tensor,
    sources: torch.Tensor,
    min_match: int,
) -> None:
    """``sources[:num_reqs]`` ([num_reqs, CHUNKS, levels] int32): each chunk's
    latest window end per level of the batch's requests."""
    if idx_mapping.shape[0]:
        _kernel(copy_match_kernel, *copy_match_key(min_match, idx_mapping.dtype))(
            idx_mapping, history, total_len, sources
        )


def copy_select(
    idx_mapping: torch.Tensor,
    history: torch.Tensor,
    total_len: torch.Tensor,
    sources: torch.Tensor,
    dspark_drafts: torch.Tensor,
    dspark_confidences: torch.Tensor,
    num_sampled: torch.Tensor,
    verified: torch.Tensor,
    cut: torch.Tensor,
    int_state: torch.Tensor,
    float_state: torch.Tensor,
    drafts: torch.Tensor,
    confidences: torch.Tensor,
    planned: torch.Tensor,
    alloc: torch.Tensor,
    min_match: int,
    initial: int,
) -> None:
    """The batch's proposals into ``drafts``, ``confidences``, ``planned`` (each
    ``[num_reqs, lookahead]``) and ``alloc`` (``[num_reqs]``), and its requests'
    copy state updated in place."""
    num_reqs = idx_mapping.shape[0]
    if not num_reqs:
        return
    lookahead = drafts.shape[1]
    block = dspark_drafts.shape[1]
    key = copy_select_key(min_match, lookahead, block, initial, idx_mapping.dtype)
    _kernel(copy_select_kernel, *key)(
        idx_mapping,
        history,
        total_len,
        sources,
        dspark_drafts,
        dspark_confidences,
        num_sampled,
        verified,
        cut,
        int_state,
        float_state,
        drafts,
        confidences,
        planned,
        alloc,
    )


def copy_calls(min_match: int, lookahead: int, block: int, initial: int) -> list:
    """The runner's launches: its batch index, the request state's history."""
    idx = torch.int64
    assert _DTYPES[idx] == RUNNER_IDX_DTYPE
    return [
        _call(copy_match_kernel, *copy_match_key(min_match, idx)),
        _call(
            copy_select_kernel,
            *copy_select_key(min_match, lookahead, block, initial, idx),
        ),
    ]
