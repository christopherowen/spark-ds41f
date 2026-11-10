# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TileLang kernels of copy drafting (``copy_drafts.CopyDrafter``).

A request matches its own token history up to its current length. The longest
matching level wins. Within that level, prefer a source with the full lookahead
remaining, breaking ties by latest position; if none has that much continuation,
use the source with the most remaining tokens. Short copies remain available.

``copy_match`` writes priorities rather than raw positions into ``sources``.
A full source at end ``e`` has priority ``e + lookahead``; a shorter one has
priority ``length - e`` (1 .. lookahead - 1); no match is -1. Thus a plain max
reduction implements the rule without atomics, int64 packing or extra buffers.
Valid priorities never exceed the request length. ``copy_alloc`` decodes this
priority using the same lookahead and the same step's length.

The scan visits full continuations in 32-element aligned tiles latest first,
then short ones in block 0. Because tiles visit decreasing priorities within each
warp, a warp reduces its matched levels across all 32 lanes whenever any lane
matches a new level and raises its required match length ``need_len`` to the
next unmatched level, probing a four-point anchor filter (``key[0]``, ``key[1]``,
``key[need_len // 2]``, and ``key[need_len - 1]``) before walking a candidate
suffix in 8-token parallel steps. Short contexts (``<= 32K``) are scanned in full;
longer contexts use a bounded two-tier scan over the most recent ``16K`` tokens,
a sparse ``256``-stride phase-aligned anchor scan across the first quarter for
256-token matches, and a dense scan of the initial ``16K`` prefix.

- ``copy_alloc``: a warp a request, before the drafter's proposal is needed.
  First the copy state of the request's last proposal is updated from this
  step's verification of it: a copy whose live verified drafts (its admitted
  drafts, at most the dead-row cut) were all accepted doubles its cap (from
  ``initial`` up to the lookahead), a rejection or a proposal without a copy
  returns it to ``initial``, and the accepted and judged drafts update the
  request's acceptance estimate (decayed counts over a prior). Then the longest
  level with a source over the chunks, its highest-priority source decoded to
  its position, and the choice (``chosen``: the source, the
  ``min(cap, total_len - source)`` drafts it offers, and the
  ``min(2 cap, lookahead, total_len - source)`` the next proposal offers if
  this one is accepted in full; ``estimates``: the estimate). It also writes
  the allocation the scheduler lays out the step after next from (those next
  drafts, at least DSpark's block).
- ``copy_apply``: a warp a request, once DSpark has drafted: the proposal of
  ``copy_alloc``'s choice when its expected yield (``offered * estimate``) is at
  least DSpark's cumulative survival yield
  (``sum_j prod_{m<=j} dspark_confidences[r, m]``), each
  copied token at the estimate as its confidence and as its planned confidence up
  to the next proposal's drafts; otherwise DSpark's drafts and confidences, with
  the slot's pending copy state cleared so the next step does not judge DSpark's
  tokens as a copy proposal.

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
# copy_alloc's choice of a request: (source, offered drafts, next drafts).
CHOSEN_COLUMNS = 3

assert MAX_LEVEL == _MATCH_THREADS, "a thread loads one key token"
assert CHUNKS == _SELECT_THREADS, "a lane reads one chunk's source"


def num_levels(min_match: int) -> int:
    return len(CopyDraftsConfig(min_match=min_match).levels())


# Contexts up to _DENSE_SCAN_LIMIT are scanned in full; longer contexts scan
# the newest _RECENT_WINDOW tokens densely, probe the first quarter at 256-token
# stride for 256-token matches, and scan the initial _PREFIX_WINDOW tokens densely.
_DENSE_SCAN_LIMIT = 32768
_RECENT_WINDOW = 16384
_PREFIX_WINDOW = 16384


@tilelang.jit(pass_configs=_PASS)
def copy_match_kernel(min_match: int, lookahead: int, idx_dtype: str):
    """See the module docstring: each block's best continuation priority per level."""
    assert 1 <= lookahead <= _MATCH_THREADS
    levels = CopyDraftsConfig(min_match=min_match).levels()
    count = len(levels)
    second_offset = 2 if min_match >= 2 else 1
    num_reqs = T.dynamic("num_reqs")
    history_rows, history_cols, history_stride = T.dynamic(
        "history_rows, history_cols, history_stride"
    )
    total_rows = T.dynamic("total_rows")
    span = CHUNKS * _MATCH_THREADS  # window ends a pass of the grid covers
    sparse_span = span * 8  # window ends a sparse 256-match pass covers

    @T.macro
    def count_run(slot, history, key, e, run, stepping, ok):
        run[0] = 0
        stepping[0] = 1
        while run[0] + 8 <= MAX_LEVEL and run[0] + 8 <= e and stepping[0] != 0:
            ok[0] = (
                T.if_then_else(history[slot, e - 1 - run[0]] == key[run[0]], 1, 0)
                & T.if_then_else(
                    history[slot, e - 2 - run[0]] == key[run[0] + 1], 1, 0
                )
                & T.if_then_else(
                    history[slot, e - 3 - run[0]] == key[run[0] + 2], 1, 0
                )
                & T.if_then_else(
                    history[slot, e - 4 - run[0]] == key[run[0] + 3], 1, 0
                )
                & T.if_then_else(
                    history[slot, e - 5 - run[0]] == key[run[0] + 4], 1, 0
                )
                & T.if_then_else(
                    history[slot, e - 6 - run[0]] == key[run[0] + 5], 1, 0
                )
                & T.if_then_else(
                    history[slot, e - 7 - run[0]] == key[run[0] + 6], 1, 0
                )
                & T.if_then_else(
                    history[slot, e - 8 - run[0]] == key[run[0] + 7], 1, 0
                )
            )
            if ok[0] != 0:
                run[0] = run[0] + 8
            else:
                stepping[0] = 0
        while (
            run[0] < MAX_LEVEL
            and run[0] < e
            and history[slot, e - 1 - run[0]] == key[run[0]]
        ):
            run[0] = run[0] + 1

    @T.macro
    def evaluate_end(
        slot,
        history,
        key,
        k0,
        k1,
        e,
        priority,
        need_len,
        half_idx,
        k_half,
        k_need,
        matched,
        run,
        stepping,
        ok,
        best,
        hit,
    ):
        matched[0] = 0
        if (
            e >= need_len[0]
            and history[slot, e - 1] == k0
            and history[slot, e - second_offset] == k1
            and history[slot, e - 1 - half_idx[0]] == k_half[0]
            and history[slot, e - need_len[0]] == k_need[0]
        ):
            matched[0] = 1
        if matched[0] != 0:
            count_run(slot, history, key, e, run, stepping, ok)
            if run[0] >= need_len[0]:
                hit[0] = 1
                for i in tuple(range(count)):
                    if run[0] >= levels[i]:
                        best[i] = T.max(best[i], priority)

    @T.macro
    def sync_warp_state(key, best, need_len, half_idx, k_half, k_need, searching):
        for i in tuple(range(count)):
            for distance in (16, 8, 4, 2, 1):
                best[i] = T.max(best[i], T.shfl_xor(best[i], distance))
        for i in tuple(range(count - 1)):
            if best[i] >= 0:
                need_len[0] = levels[i + 1]
        half_idx[0] = T.min(T.max(need_len[0] // 2, 1), MAX_LEVEL - 1)
        k_half[0] = key[half_idx[0]]
        k_need[0] = key[need_len[0] - 1]
        if best[count - 1] >= 0:
            searching[0] = 0

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
            hit = T.alloc_local((1,), T.int32)
            matched = T.alloc_local((1,), T.int32)
            need_len = T.alloc_local((1,), T.int32)
            half_idx = T.alloc_local((1,), T.int32)
            k_half = T.alloc_local((1,), T.int32)
            k_need = T.alloc_local((1,), T.int32)
            run = T.alloc_local((1,), T.int32)
            stepping = T.alloc_local((1,), T.int32)
            ok = T.alloc_local((1,), T.int32)
            searching = T.alloc_local((1,), T.int32)
            e_cand = T.alloc_local((1,), T.int32)
            slot = T.alloc_var(T.int64, init=T.cast(idx_mapping[r], T.int64))
            length = T.alloc_var(T.int32, init=0)
            if slot >= 0:
                length = total_len[slot]
            full_last = T.alloc_var(T.int32, init=length - lookahead)
            top_aligned = T.alloc_var(
                T.int32,
                init=T.if_then_else(
                    full_last >= min_match, ((full_last + 31) // 32) * 32, 0
                ),
            )
            if length <= min_match or (
                c > 0 and top_aligned - c * _MATCH_THREADS < min_match
            ):
                # The block's share is empty: no end to compare
                # (the branch is the whole block's, so no barrier diverges).
                if tx < count:
                    sources[r, c, tx] = -1
            else:
                # The key: the context's last tokens, latest first.
                key[tx] = -1
                if tx < length:
                    key[tx] = history[slot, length - 1 - tx]
                for i in tuple(range(count)):
                    best[i] = -1
                T.sync_threads()
                k0 = T.alloc_var(T.int32, init=key[0])
                k1 = T.alloc_var(T.int32, init=key[second_offset - 1])
                k_last = T.alloc_var(T.int32, init=key[MAX_LEVEL - 1])
                need_len[0] = min_match
                half_idx[0] = T.min(T.max(min_match // 2, 1), MAX_LEVEL - 1)
                k_half[0] = key[half_idx[0]]
                k_need[0] = key[min_match - 1]
                # Stage 1: full continuations (e in [min_match, full_last]) in
                # 32-element aligned warp tiles from highest priority down.
                warp_tile = T.alloc_var(
                    T.int32,
                    init=top_aligned - (c + 1) * _MATCH_THREADS + (tx // 32) * 32,
                )
                searching[0] = 1
                if top_aligned <= _DENSE_SCAN_LIMIT:
                    while warp_tile + 32 >= min_match and searching[0] != 0:
                        hit[0] = 0
                        e = warp_tile + tx % 32 + 1
                        if min_match <= e and e <= full_last:
                            evaluate_end(
                                slot,
                                history,
                                key,
                                k0,
                                k1,
                                e,
                                e + lookahead,
                                need_len,
                                half_idx,
                                k_half,
                                k_need,
                                matched,
                                run,
                                stepping,
                                ok,
                                best,
                                hit,
                            )
                        if (
                            T.call_extern(
                                "int32",
                                "__any_sync",
                                T.uint32(0xFFFFFFFF),
                                hit[0] != 0,
                            )
                            != 0
                        ):
                            sync_warp_state(
                                key,
                                best,
                                need_len,
                                half_idx,
                                k_half,
                                k_need,
                                searching,
                            )
                        warp_tile -= span
                else:
                    recent_floor = T.alloc_var(
                        T.int32, init=top_aligned - _RECENT_WINDOW
                    )
                    while warp_tile >= recent_floor and searching[0] != 0:
                        hit[0] = 0
                        e = warp_tile + tx % 32 + 1
                        if min_match <= e and e <= full_last:
                            evaluate_end(
                                slot,
                                history,
                                key,
                                k0,
                                k1,
                                e,
                                e + lookahead,
                                need_len,
                                half_idx,
                                k_half,
                                k_need,
                                matched,
                                run,
                                stepping,
                                ok,
                                best,
                                hit,
                            )
                        if (
                            T.call_extern(
                                "int32",
                                "__any_sync",
                                T.uint32(0xFFFFFFFF),
                                hit[0] != 0,
                            )
                            != 0
                        ):
                            sync_warp_state(
                                key,
                                best,
                                need_len,
                                half_idx,
                                k_half,
                                k_need,
                                searching,
                            )
                        warp_tile -= span

                    quarter_top = T.alloc_var(
                        T.int32,
                        init=T.min(
                            recent_floor,
                            ((length // 4 + 1024 + 255) // 256) * 256,
                        ),
                    )
                    sparse_base = T.alloc_var(
                        T.int32,
                        init=(quarter_top // 256) * 256
                        - (c + 1) * _MATCH_WARPS * 256
                        + (tx // 32) * 256,
                    )
                    tok = T.alloc_var(T.int32, init=-1)
                    while (
                        sparse_base + 256 >= _PREFIX_WINDOW and searching[0] != 0
                    ):
                        hit[0] = 0
                        e_cand[0] = 0
                        if sparse_base >= 0 and sparse_base + (tx % 32) < full_last:
                            tok = history[slot, sparse_base + (tx % 32)]
                            for m in tuple(range(8)):
                                if tok == key[32 * m]:
                                    e_try = sparse_base + 32 * m + (tx % 32) + 1
                                    if (
                                        MAX_LEVEL <= e_try
                                        and e_try <= recent_floor
                                        and history[slot, e_try - 1] == k0
                                        and history[slot, e_try - second_offset] == k1
                                        and history[slot, e_try - MAX_LEVEL]
                                        == k_last
                                    ):
                                        e_cand[0] = e_try
                        if e_cand[0] > 0:
                            count_run(
                                slot, history, key, e_cand[0], run, stepping, ok
                            )
                            if run[0] >= MAX_LEVEL:
                                hit[0] = 1
                                for i in tuple(range(count)):
                                    best[i] = T.max(
                                        best[i], e_cand[0] + lookahead
                                    )
                        if (
                            T.call_extern(
                                "int32",
                                "__any_sync",
                                T.uint32(0xFFFFFFFF),
                                hit[0] != 0,
                            )
                            != 0
                        ):
                            sync_warp_state(
                                key,
                                best,
                                need_len,
                                half_idx,
                                k_half,
                                k_need,
                                searching,
                            )
                        sparse_base -= sparse_span

                    warp_tile = (
                        _PREFIX_WINDOW
                        - (c + 1) * _MATCH_THREADS
                        + (tx // 32) * 32
                    )
                    while warp_tile + 32 >= min_match and searching[0] != 0:
                        hit[0] = 0
                        e = warp_tile + tx % 32 + 1
                        if min_match <= e and e <= full_last:
                            evaluate_end(
                                slot,
                                history,
                                key,
                                k0,
                                k1,
                                e,
                                e + lookahead,
                                need_len,
                                half_idx,
                                k_half,
                                k_need,
                                matched,
                                run,
                                stepping,
                                ok,
                                best,
                                hit,
                            )
                        if (
                            T.call_extern(
                                "int32",
                                "__any_sync",
                                T.uint32(0xFFFFFFFF),
                                hit[0] != 0,
                            )
                            != 0
                        ):
                            sync_warp_state(
                                key,
                                best,
                                need_len,
                                half_idx,
                                k_half,
                                k_need,
                                searching,
                            )
                        warp_tile -= span

                # Stage 2: short continuations (at most lookahead - 1 ends) in block 0.
                if c == 0 and searching[0] != 0:
                    e_short = T.max(min_match, full_last + 1) + tx
                    if e_short < length:
                        evaluate_end(
                            slot,
                            history,
                            key,
                            k0,
                            k1,
                            e_short,
                            length - e_short,
                            need_len,
                            half_idx,
                            k_half,
                            k_need,
                            matched,
                            run,
                            stepping,
                            ok,
                            best,
                            hit,
                        )
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
def copy_alloc_kernel(
    min_match: int,
    lookahead: int,
    block: int,
    initial: int,
    idx_dtype: str,
):
    """See the module docstring: the copy state's update and the choice."""
    count = num_levels(min_match)
    num_reqs = T.dynamic("num_reqs")
    total_rows = T.dynamic("total_rows")
    state_rows = T.dynamic("state_rows")

    @T.prim_func
    def copy_alloc(
        idx_mapping: T.Tensor((num_reqs,), idx_dtype),
        total_len: T.Tensor((total_rows,), T.int32),
        sources: T.Tensor((num_reqs, CHUNKS, count), T.int32),
        num_sampled: T.Tensor((num_reqs,), T.int32),
        verified: T.Tensor((num_reqs,), T.int32),
        cut: T.Tensor((num_reqs,), T.int32),
        int_state: T.Tensor((state_rows, STATE_COLUMNS), T.int32),
        float_state: T.Tensor((state_rows, STATE_COLUMNS), T.float32),
        chosen: T.Tensor((num_reqs, CHOSEN_COLUMNS), T.int32),
        estimates: T.Tensor((num_reqs,), T.float32),
        alloc: T.Tensor((num_reqs,), T.int32),
    ):
        with T.Kernel(num_reqs, threads=_SELECT_THREADS) as r:
            tx = T.get_thread_binding()
            slot = T.alloc_var(T.int64, init=T.cast(idx_mapping[r], T.int64))
            # The longest level wins; priorities order its continuations.
            source = T.alloc_var(T.int32, init=-1)
            for i in tuple(range(count)):
                latest = T.alloc_var(T.int32, init=sources[r, tx, i])
                for step in (16, 8, 4, 2, 1):
                    latest = T.max(latest, T.shfl_xor(latest, step))
                if latest >= 0:
                    source = latest
            if tx == 0:
                if slot < 0:
                    chosen[r, 0] = -1
                    chosen[r, 1] = 0
                    chosen[r, 2] = 0
                    estimates[r] = T.float32(PRIOR_HITS / PRIOR_TRIALS)
                    alloc[r] = block
                else:
                    length = total_len[slot]
                    if source >= 0:
                        source = T.if_then_else(
                            source >= lookahead, source - lookahead, length - source
                        )
                    doublings = T.alloc_var(T.int32, init=int_state[slot, 0])
                    pending = int_state[slot, 1]
                    hits = T.alloc_var(T.float32, init=float_state[slot, 0])
                    trials = T.alloc_var(T.float32, init=float_state[slot, 1])
                    if pending > 0 and num_sampled[r] > 0:
                        live = T.min(pending, T.min(verified[r], cut[r]))
                        if live > 0:
                            accepted = T.min(num_sampled[r] - 1, live)
                            doublings = T.if_then_else(
                                accepted >= live, T.min(doublings + 1, MAX_DOUBLINGS), 0
                            )
                            hits = hits * DECAY + T.cast(accepted, T.float32)
                            trials = trials * DECAY + T.cast(
                                T.min(accepted + 1, live), T.float32
                            )
                    cap = T.min(T.int32(initial) << doublings, lookahead)
                    distance = length - source
                    offered = T.if_then_else(source >= 0, T.min(cap, distance), 0)
                    after = T.if_then_else(
                        source >= 0, T.min(T.min(2 * cap, lookahead), distance), 0
                    )
                    # A step without a copy (DSpark's) starts the next copy afresh.
                    int_state[slot, 0] = T.if_then_else(source >= 0, doublings, 0)
                    int_state[slot, 1] = offered
                    float_state[slot, 0] = hits
                    float_state[slot, 1] = trials
                    chosen[r, 0] = source
                    chosen[r, 1] = offered
                    chosen[r, 2] = after
                    estimates[r] = (hits + PRIOR_HITS) / (trials + PRIOR_TRIALS)
                    alloc[r] = T.max(after, block)

    return copy_alloc


@tilelang.jit(pass_configs=_PASS)
def copy_apply_kernel(lookahead: int, block: int, idx_dtype: str):
    """See the module docstring: the proposal of ``copy_alloc``'s choice."""
    num_reqs = T.dynamic("num_reqs")
    history_rows, history_cols, history_stride = T.dynamic(
        "history_rows, history_cols, history_stride"
    )
    state_rows = T.dynamic("state_rows")
    draft_rows, draft_stride = T.dynamic("draft_rows, draft_stride")
    conf_rows, conf_stride = T.dynamic("conf_rows, conf_stride")
    passes = -(-lookahead // _SELECT_THREADS)

    @T.prim_func
    def copy_apply(
        idx_mapping: T.Tensor((num_reqs,), idx_dtype),
        history: T.StridedTensor(
            (history_rows, history_cols), (history_stride, 1), T.int32
        ),
        chosen: T.Tensor((num_reqs, CHOSEN_COLUMNS), T.int32),
        estimates: T.Tensor((num_reqs,), T.float32),
        int_state: T.Tensor((state_rows, STATE_COLUMNS), T.int32),
        dspark_drafts: T.StridedTensor((draft_rows, block), (draft_stride, 1), T.int64),
        dspark_confidences: T.StridedTensor(
            (conf_rows, block), (conf_stride, 1), T.float32
        ),
        drafts: T.Tensor((num_reqs, lookahead), T.int64),
        confidences: T.Tensor((num_reqs, lookahead), T.float32),
        planned: T.Tensor((num_reqs, lookahead), T.float32),
    ):
        with T.Kernel(num_reqs, threads=_SELECT_THREADS) as r:
            tx = T.get_thread_binding()
            slot = T.alloc_var(T.int64, init=T.cast(idx_mapping[r], T.int64))
            start = T.alloc_var(T.int32, init=chosen[r, 0])
            offered = T.alloc_var(T.int32, init=chosen[r, 1])
            after = T.alloc_var(T.int32, init=chosen[r, 2])
            estimate = T.alloc_var(T.float32, init=estimates[r])
            dspark_yield = T.alloc_var(T.float32, init=T.float32(0))
            dspark_survival = T.alloc_var(T.float32, init=T.float32(1))
            for j in T.unroll(block):
                dspark_survival *= dspark_confidences[r, j]
                dspark_yield += dspark_survival
            use_copy = T.alloc_var(T.int32, init=0)
            if (
                slot >= 0
                and start >= 0
                and T.cast(offered, T.float32) * estimate >= dspark_yield
            ):
                use_copy = 1
            if slot >= 0 and start >= 0 and use_copy == 0 and tx == 0:
                int_state[slot, 0] = 0
                int_state[slot, 1] = 0
            for k in T.unroll(passes):
                col = k * _SELECT_THREADS + tx
                if col < lookahead:
                    if use_copy != 0:
                        drafts[r, col] = T.if_then_else(
                            col < offered,
                            T.cast(
                                history[slot, start + T.min(col, offered - 1)],
                                T.int64,
                            ),
                            T.int64(-1),
                        )
                        confidences[r, col] = T.if_then_else(
                            col < offered, estimate, T.float32(0)
                        )
                        planned[r, col] = T.if_then_else(
                            col < after, estimate, T.float32(0)
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
                            T.int64(-1),
                        )
                        confidences[r, col] = value
                        planned[r, col] = value

    return copy_apply


def copy_match_key(
    min_match: int, lookahead: int, idx_dtype: torch.dtype
) -> tuple:
    return (min_match, lookahead, _DTYPES[idx_dtype])


def copy_alloc_key(
    min_match: int, lookahead: int, block: int, initial: int, idx_dtype: torch.dtype
) -> tuple:
    return (min_match, lookahead, block, initial, _DTYPES[idx_dtype])


def copy_apply_key(lookahead: int, block: int, idx_dtype: torch.dtype) -> tuple:
    return (lookahead, block, _DTYPES[idx_dtype])


def copy_match(
    idx_mapping: torch.Tensor,
    history: torch.Tensor,
    total_len: torch.Tensor,
    sources: torch.Tensor,
    min_match: int,
    lookahead: int = CopyDraftsConfig.max_drafts,
) -> None:
    """Write per-chunk, per-level continuation priorities (see module docstring)."""
    if idx_mapping.shape[0]:
        _kernel(
            copy_match_kernel, *copy_match_key(min_match, lookahead, idx_mapping.dtype)
        )(idx_mapping, history, total_len, sources)


def copy_alloc(
    idx_mapping: torch.Tensor,
    total_len: torch.Tensor,
    sources: torch.Tensor,
    num_sampled: torch.Tensor,
    verified: torch.Tensor,
    cut: torch.Tensor,
    int_state: torch.Tensor,
    float_state: torch.Tensor,
    chosen: torch.Tensor,
    estimates: torch.Tensor,
    alloc: torch.Tensor,
    min_match: int,
    lookahead: int,
    block: int,
    initial: int,
) -> None:
    """The batch's requests' copy state updated in place from ``sources``, their
    choice into ``chosen`` and ``estimates``, and ``alloc[:num_reqs]``."""
    num_reqs = idx_mapping.shape[0]
    if not num_reqs:
        return
    key = copy_alloc_key(min_match, lookahead, block, initial, idx_mapping.dtype)
    _kernel(copy_alloc_kernel, *key)(
        idx_mapping,
        total_len,
        sources,
        num_sampled,
        verified,
        cut,
        int_state,
        float_state,
        chosen,
        estimates,
        alloc,
    )


def copy_apply(
    idx_mapping: torch.Tensor,
    history: torch.Tensor,
    chosen: torch.Tensor,
    estimates: torch.Tensor,
    int_state: torch.Tensor,
    dspark_drafts: torch.Tensor,
    dspark_confidences: torch.Tensor,
    drafts: torch.Tensor,
    confidences: torch.Tensor,
    planned: torch.Tensor,
) -> None:
    """The batch's proposals from ``copy_alloc``'s choice into ``drafts``,
    ``confidences`` and ``planned`` (each ``[num_reqs, lookahead]``)."""
    num_reqs = idx_mapping.shape[0]
    if not num_reqs:
        return
    lookahead = drafts.shape[1]
    block = dspark_drafts.shape[1]
    key = copy_apply_key(lookahead, block, idx_mapping.dtype)
    _kernel(copy_apply_kernel, *key)(
        idx_mapping,
        history,
        chosen,
        estimates,
        int_state,
        dspark_drafts,
        dspark_confidences,
        drafts,
        confidences,
        planned,
    )


def copy_calls(min_match: int, lookahead: int, block: int, initial: int) -> list:
    """The runner's launches: its batch index, the request state's history."""
    idx = torch.int64
    assert _DTYPES[idx] == RUNNER_IDX_DTYPE
    return [
        _call(copy_match_kernel, *copy_match_key(min_match, lookahead, idx)),
        _call(
            copy_alloc_kernel,
            *copy_alloc_key(min_match, lookahead, block, initial, idx),
        ),
        _call(copy_apply_kernel, *copy_apply_key(lookahead, block, idx)),
    ]
