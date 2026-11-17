# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark drafter learning at the end of each turn, in shadow
(``VLLM_DSPARK_TURN_LEARNING=shadow``).

The learner keeps an FP32 copy of the Markov head's W1 and trains it on every
finished turn; the draft step keeps drafting with the loaded W1, so no draft, and
no output, changes. Each turn is scored with the learned copy before the copy
learns from it, so the score is what the copy would have done on traffic it had
not seen.

What the draft step keeps (``capture``): at each draft position the greedy step
already reduces this rank's vocabulary shard by 2048-column blocks (positions
after the first, and the first at one or two requests) or by smaller tiles (the
fused projection of the first position at three requests or more). The learner
points those reductions at its own per-position scratch instead of the shared
one, which costs nothing, and one kernel on a side stream, overlapping the
position's all-gather, copies each request's block maxima (value and global
token) and its anchor into the request's ring. That is everything the update
reads: no head row, no logit is recomputed.

What the end of a turn does (``finish``, called by every rank in the same step and
in sorted order, before the slot can be reused):

1. the labels: the token the target committed at each kept position;
2. at each drafted position, each candidate's teacher-forced logits: the step's
   value with the bias of the step's input (the anchor, then the previous draft)
   replaced by the committed input's, under the loaded table and under the
   learned copy as a BF16 table, through the candidate's W2 row (decoded from this
   rank's packed shard), each bias and the sum rounded to BF16 as the step rounds
   them (where the step's input was the committed one the loaded logit is the
   step's value, and a change too small to move a BF16 bias moves nothing);
3. a score limited to those maxima: each table's leading run of positions whose
   best candidate is the committed token, one rule for both; the drafts the
   maxima give are checked against the drafts the ring holds. The offline study
   shows this score does not track the learned copy's gain (it cannot see a fix
   whose committed token is not among the step's maxima), so it is a health
   signal; the gain is measured offline on exact replays, or with drafts read
   from the learned table;
4. the update: at the positions up to each block's first miss ("seen", where the
   step's input was the committed token, so its maxima are teacher forcing's),
   the cross-entropy of the committed token over the block maxima where the token
   is among them, into the W1 rows of each position's input, summed over the
   turn's positions a row, with a per-row AdaGrad step on the FP32 copy
   (``lr / sqrt(sum of squared row-gradient norms)``).

Ranks exchange one 10-value summary a position (all-gather) and the 256-wide
gradient a position (all-reduce), so every rank applies the same update to its
replica. Every step is a TileLang kernel; the update is the ``blockseen`` rule and
the score the ``tf_proxy`` of the offline study (``bundles/study``).
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.tilelang_utils import T, tilelang
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.spec_decode.dspark.turn_learning import (
    _META,
    _PASS,
    _call,
    _kernel,
    labels_kernel,
    reset_kernel,
)

logger = init_logger(__name__)

GREEDY_BLOCK = 2048
_THREADS = 256
_CAPTURE_THREADS = 512  # 16 warps: one a part of the fused first position
_NO_INDEX = 1 << 24  # above every vocabulary index, exact in FP32
_INT32_MAX = 2147483647
_LOCAL = 10  # see shadow_local_kernel
_CHUNK = 256  # entries an exchange, bounding the update's row search


class ShadowStats:
    """The counters of the shadow score, in ``shadow_combine_kernel``'s order."""

    FIELDS = (
        "held",  # kept draft steps the turn committed (see labels_kernel)
        "seen",  # positions up to each block's first miss, within its drafts
        "trained",  # seen positions whose committed token is among the maxima
        "accepted",  # the drafts' accepted run, summed over the blocks
        "tf_loaded",  # the loaded table's teacher-forced run over the step's maxima
        "tf_learned",  # the learned copy's, by the same rule
        "fixed",  # blocks whose run the learned copy lengthens
        "broken",  # blocks whose run it shortens
        "checked",  # drafts the ring holds, compared with the maxima's best
        "draft_mismatches",  # of those, drafts the maxima do not give
    )

    @classmethod
    def as_dict(cls, values: Any) -> dict[str, int]:
        return {name: int(v) for name, v in zip(cls.FIELDS, values)}


_COUNTS = len(ShadowStats.FIELDS)


def turn_learning_lr() -> float:
    value = float(os.environ.get("VLLM_DSPARK_TURN_LEARNING_LR", "0.3"))
    if not value > 0:
        raise ValueError("VLLM_DSPARK_TURN_LEARNING_LR must be positive")
    return value


# --- In-graph capture ------------------------------------------------------------


@T.macro
def _take(best, best_index, value, index):
    """``(best, best_index)`` becomes ``(value, index)`` when it is the first
    maximum of the two (a larger value, or an equal one at a lower index); an
    index below 0 is no candidate."""
    take = (index >= 0) & (
        (best_index[0] < 0)
        | (value > best[0])
        | ((value == best[0]) & (index < best_index[0]))
    )
    best[0] = T.if_then_else(take, value, best[0])
    best_index[0] = T.if_then_else(take, index, best_index[0])


@T.macro
def _warp_take(best, best_index, other, other_index):
    """Every lane's ``(best, best_index)`` becomes the warp's first maximum (the
    combine is a total order, so the pairing does not matter)."""
    for d in T.unroll(5):
        other[0] = T.shfl_xor(best[0], 16 >> d)
        other_index[0] = T.shfl_xor(best_index[0], 16 >> d)
        _take(best, best_index, other[0], other_index[0])


@tilelang.jit(pass_configs=_PASS)
def shadow_capture_kernel(n_spec: int, parts: int):
    """One block a draft row: the row's block maxima at each drafted position
    (value and global token; -inf and -1 past the drafted ones), its drafts
    before the last position, its anchor position, drafted count and anchor token
    into its request's ring at entry ``count % capacity``, then ``count + 1``.

    Position ``i``'s maxima are this rank's greedy partials of that position: the
    2048-column parts ``greedy_markov_shard_pair`` reduced (``part_values[i]``,
    [rows, parts]), one thread a part; or, for position 0 when ``fused0``, the
    tiles the fused projection reduced ([rows, num_tiles], ``tiles_per_block`` a
    part), reduced here to parts by the same first-maximum rule, one warp a part
    (a tile a lane, then a shuffle reduction). A row whose sampling slot is -1
    (padding of the captured size) writes nothing."""
    map_len, pv_len, tv_len = T.dynamic("map_len, pv_len, tv_len")
    draft_rows, draft_stride = T.dynamic("draft_rows, draft_stride")
    ids_len, anchor_len = T.dynamic("ids_len, anchor_len")
    slots, cap = T.dynamic("slots, cap")
    span = n_spec * parts
    warps = _CAPTURE_THREADS // 32
    assert span <= _CAPTURE_THREADS

    @T.prim_func
    def capture(
        part_values: T.Tensor((n_spec, pv_len), T.float32),
        part_indices: T.Tensor((n_spec, pv_len), T.int32),
        tile_values: T.Tensor((tv_len,), T.float32),
        tile_indices: T.Tensor((tv_len,), T.int32),
        sample_map: T.Tensor((map_len,), T.int32),
        sample_pos: T.Tensor((map_len,), T.int64),
        drafts: T.StridedTensor((draft_rows, n_spec), (draft_stride, 1), T.int64),
        input_ids: T.Tensor((ids_len,), T.int32),
        anchor_idx: T.Tensor((anchor_len,), T.int64),
        ring_values: T.Tensor((slots, cap, n_spec, parts), T.float32),
        ring_index: T.Tensor((slots, cap, n_spec, parts), T.int32),
        ring_meta: T.Tensor((slots, cap, _META), T.int64),
        ring_drafts: T.Tensor((slots, cap, n_spec), T.int32),
        ring_count: T.Tensor((slots,), T.int64),
        active: T.int32,
        fused0: T.int32,
        num_tiles: T.int32,
        tiles_per_block: T.int32,
        vocab_start: T.int32,
    ):
        with T.Kernel(map_len // n_spec, threads=_CAPTURE_THREADS) as r:
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            best = T.alloc_local((1,), T.float32)
            best_index = T.alloc_local((1,), T.int32)
            other = T.alloc_local((1,), T.float32)
            other_index = T.alloc_local((1,), T.int32)
            slot = T.alloc_var(T.int64, init=T.cast(sample_map[r * n_spec], T.int64))
            safe = T.alloc_var(T.int64, init=T.max(slot, T.int64(0)))
            count = T.alloc_var(T.int64, init=ring_count[safe])
            entry = T.alloc_var(T.int64, init=count % T.cast(cap, T.int64))
            if (slot >= 0) & (fused0 != 0) & (active > 0):
                # The fused first position: a warp a part, a tile a lane.
                for round_ in T.serial(T.ceildiv(parts, warps)):
                    b = round_ * warps + warp
                    if b < parts:
                        best[0] = -T.infinity(T.float32)
                        best_index[0] = -1
                        for it in T.serial(T.ceildiv(tiles_per_block, 32)):
                            t = it * 32 + lane
                            tile = b * tiles_per_block + t
                            if (t < tiles_per_block) & (tile < num_tiles):
                                x = tile_indices[r * num_tiles + tile]
                                _take(
                                    best,
                                    best_index,
                                    tile_values[r * num_tiles + tile],
                                    T.if_then_else(x == _INT32_MAX, -1, x),
                                )
                        _warp_take(best, best_index, other, other_index)
                        if lane == 0:
                            ring_values[safe, entry, 0, b] = best[0]
                            ring_index[safe, entry, 0, b] = T.if_then_else(
                                best_index[0] >= 0, best_index[0] + vocab_start, -1
                            )
            if (slot >= 0) & (tx < span):
                i = tx // parts
                b = tx % parts
                if (i != 0) | (fused0 == 0):
                    best[0] = -T.infinity(T.float32)
                    best_index[0] = -1
                    if i < active:
                        x = part_indices[i, r * parts + b]
                        _take(
                            best,
                            best_index,
                            part_values[i, r * parts + b],
                            T.if_then_else(x == _INT32_MAX, -1, x),
                        )
                    ring_values[safe, entry, i, b] = best[0]
                    ring_index[safe, entry, i, b] = T.if_then_else(
                        best_index[0] >= 0, best_index[0] + vocab_start, -1
                    )
            if slot >= 0:
                if tx < n_spec:
                    ring_drafts[safe, entry, tx] = T.if_then_else(
                        tx < active - 1, T.cast(drafts[r, tx], T.int32), -1
                    )
                if tx == 0:
                    ring_meta[safe, entry, 0] = sample_pos[r * n_spec] - 1
                    ring_meta[safe, entry, 1] = T.cast(active, T.int64)
                    ring_meta[safe, entry, 2] = T.cast(input_ids[anchor_idx[r]], T.int64)
            T.sync_threads()
            if (slot >= 0) & (tx == 0):
                ring_count[safe] = count + 1

    return capture


# --- End of turn -----------------------------------------------------------------


@tilelang.jit(pass_configs=_PASS)
def shadow_local_kernel(n_spec: int, parts: int, dim: int):
    """One block an entry. At each position the step drafted, each of this rank's
    candidates (the step's block maxima, under the step's own input: the anchor,
    then the previous draft) gets two teacher-forced logits: the step's value with
    the step input's loaded bias replaced by the committed input's (``ids``: the
    anchor, then the labels) under the loaded table, and under the learned copy as a
    BF16 table would hold it; biases through the candidate's decoded W2 row, each
    bias and the sum rounded to BF16 as the step rounds them (where the step's input
    is the committed one, the loaded logit is the step's value exactly, and a
    learned change too small to move a BF16 bias moves nothing). ``z_out`` keeps the
    learned logits; ``summary`` ([entries, n_spec, 10]) the position's share of the
    exchange: the step's first maximum and its token (wherever a held entry
    drafted, labeled or not), then at labeled positions the loaded teacher-forced
    first maximum and its token, the learned one and its token, the sum of the
    learned logits' exponentials relative to their maximum, the committed token's
    learned logit, whether it is a candidate here, and 1 (else -inf and 2**24
    tokens, zeros). Tokens are FP32, exact below 2**24. ``dec`` holds the
    candidates' W2 rows as ``(entry * n_spec + i) * parts + c``."""
    entries, finished_len, slots, cap = T.dynamic("entries, finished_len, slots, cap")
    dec_rows, table_rows = T.dynamic("dec_rows, table_rows")
    pairs = n_spec * parts

    @T.prim_func
    def local(
        entry_f: T.Tensor((entries,), T.int64),
        entry_k: T.Tensor((entries,), T.int64),
        finished: T.Tensor((finished_len,), T.int64),
        labels: T.Tensor((finished_len, cap, n_spec), T.int32),
        ids: T.Tensor((finished_len, cap, n_spec + 1), T.int32),
        ring_values: T.Tensor((slots, cap, n_spec, parts), T.float32),
        ring_index: T.Tensor((slots, cap, n_spec, parts), T.int32),
        ring_meta: T.Tensor((slots, cap, _META), T.int64),
        ring_drafts: T.Tensor((slots, cap, n_spec), T.int32),
        dec: T.Tensor((dec_rows, dim), T.bfloat16),
        master: T.Tensor((table_rows, dim), T.float32),
        table: T.Tensor((table_rows, dim), T.bfloat16),
        z_out: T.Tensor((entries, n_spec, parts), T.float32),
        summary: T.Tensor((entries, n_spec, _LOCAL), T.float32),
    ):
        with T.Kernel(entries, threads=_THREADS) as e:
            tx = T.get_thread_binding()
            step_row = T.alloc_shared((n_spec, dim), T.float32)
            loaded = T.alloc_shared((n_spec, dim), T.float32)
            learned = T.alloc_shared((n_spec, dim), T.float32)
            z0 = T.alloc_shared((n_spec, parts), T.float32)
            zl = T.alloc_shared((n_spec, parts), T.float32)
            acc_d = T.alloc_local((1,), T.float32)
            acc_0 = T.alloc_local((1,), T.float32)
            acc_l = T.alloc_local((1,), T.float32)
            bv = T.alloc_local((1,), T.float32)
            bvi = T.alloc_local((1,), T.int32)
            b0 = T.alloc_local((1,), T.float32)
            b0i = T.alloc_local((1,), T.int32)
            bz = T.alloc_local((1,), T.float32)
            bzi = T.alloc_local((1,), T.int32)
            zlab = T.alloc_local((1,), T.float32)
            has = T.alloc_local((1,), T.float32)
            total = T.alloc_local((1,), T.float32)
            f = T.alloc_var(T.int64, init=entry_f[e])
            k = T.alloc_var(T.int64, init=entry_k[e])
            slot = T.alloc_var(T.int64, init=finished[f])
            held = labels[f, k, 0] != -2
            drafted = T.alloc_var(T.int64, init=ring_meta[slot, k, 1])
            for it in T.serial(T.ceildiv(n_spec * dim, _THREADS)):
                q = it * _THREADS + tx
                if q < n_spec * dim:
                    i = q // dim
                    d = q % dim
                    here = held & (T.cast(i, T.int64) < drafted)
                    # The step's input: the anchor, then the previous draft.
                    step_in = T.if_then_else(
                        i == 0, ids[f, k, 0], ring_drafts[slot, k, T.max(i - 1, 0)]
                    )
                    p_d = T.if_then_else(
                        here & (step_in >= 0), T.cast(step_in, T.int64), T.int64(0)
                    )
                    p_l = T.if_then_else(
                        here & (labels[f, k, i] >= 0), T.cast(ids[f, k, i], T.int64), T.int64(0)
                    )
                    step_row[i, d] = T.cast(table[p_d, d], T.float32)
                    loaded[i, d] = T.cast(table[p_l, d], T.float32)
                    learned[i, d] = T.cast(T.cast(master[p_l, d], T.bfloat16), T.float32)
            T.sync_threads()
            if tx < pairs:
                i = tx // parts
                c = tx % parts
                row = (T.cast(e, T.int64) * n_spec + i) * parts + c
                acc_d[0] = T.float32(0)
                acc_0[0] = T.float32(0)
                acc_l[0] = T.float32(0)
                for d in T.serial(dim):
                    w = T.cast(dec[row, d], T.float32)
                    acc_d[0] = acc_d[0] + step_row[i, d] * w
                    acc_0[0] = acc_0[0] + loaded[i, d] * w
                    acc_l[0] = acc_l[0] + learned[i, d] * w
                base = ring_values[slot, k, i, c] - T.cast(T.cast(acc_d[0], T.bfloat16), T.float32)
                z0[i, c] = T.cast(
                    T.cast(base + T.cast(T.cast(acc_0[0], T.bfloat16), T.float32), T.bfloat16),
                    T.float32,
                )
                zl[i, c] = T.cast(
                    T.cast(base + T.cast(T.cast(acc_l[0], T.bfloat16), T.float32), T.bfloat16),
                    T.float32,
                )
                z_out[e, i, c] = zl[i, c]
            T.sync_threads()
            if tx < n_spec:
                i = tx
                label = labels[f, k, i]
                here = held & (T.cast(i, T.int64) < drafted)
                bv[0] = -T.infinity(T.float32)
                bvi[0] = -1
                b0[0] = -T.infinity(T.float32)
                b0i[0] = -1
                bz[0] = -T.infinity(T.float32)
                bzi[0] = -1
                zlab[0] = -T.infinity(T.float32)
                has[0] = T.float32(0)
                total[0] = T.float32(0)
                for c in T.serial(parts):
                    x = ring_index[slot, k, i, c]
                    _take(bv, bvi, ring_values[slot, k, i, c], x)
                    _take(b0, b0i, z0[i, c], x)
                    _take(bz, bzi, zl[i, c], x)
                    if (x >= 0) & (x == label):
                        zlab[0] = zl[i, c]
                        has[0] = T.float32(1)
                for c in T.serial(parts):
                    if ring_index[slot, k, i, c] >= 0:
                        total[0] = total[0] + T.exp(zl[i, c] - bz[0])
                # The step's draft is checked wherever it drafted, labeled or not.
                drafted_here = here & (bvi[0] >= 0)
                ok = drafted_here & (label >= 0)
                summary[e, i, 0] = T.if_then_else(drafted_here, bv[0], -T.infinity(T.float32))
                summary[e, i, 1] = T.if_then_else(
                    drafted_here, T.cast(bvi[0], T.float32), T.float32(_NO_INDEX)
                )
                summary[e, i, 2] = T.if_then_else(ok, b0[0], -T.infinity(T.float32))
                summary[e, i, 3] = T.if_then_else(
                    ok, T.cast(b0i[0], T.float32), T.float32(_NO_INDEX)
                )
                summary[e, i, 4] = T.if_then_else(ok, bz[0], -T.infinity(T.float32))
                summary[e, i, 5] = T.if_then_else(
                    ok, T.cast(bzi[0], T.float32), T.float32(_NO_INDEX)
                )
                summary[e, i, 6] = T.if_then_else(ok, total[0], T.float32(0))
                summary[e, i, 7] = T.if_then_else(ok, zlab[0], -T.infinity(T.float32))
                summary[e, i, 8] = T.if_then_else(ok, has[0], T.float32(0))
                summary[e, i, 9] = T.if_then_else(ok, T.float32(1), T.float32(0))

    return local


@T.macro
def _take_float(best, best_index, value, index):
    """``_take`` for the exchanged summaries, whose tokens are FP32."""
    take = (index < T.float32(_NO_INDEX)) & (
        (best_index[0] >= T.float32(_NO_INDEX))
        | (value > best[0])
        | ((value == best[0]) & (index < best_index[0]))
    )
    best[0] = T.if_then_else(take, value, best[0])
    best_index[0] = T.if_then_else(take, index, best_index[0])


@tilelang.jit(pass_configs=_PASS)
def shadow_combine_kernel(n_spec: int, parts: int, dim: int, ranks: int):
    """One block an entry, on every rank alike from the exchanged summaries
    (``gathered``: [entries * n_spec, 10 * ranks], each rank's 10 in rank order).
    At each position: the step's draft (the first maximum of the step's maxima),
    checked against the drafts the ring holds; the loaded and the learned table's
    teacher-forced best candidate; the log-sum-exp of the learned logits. The score
    is each table's leading run of labeled drafted positions whose best candidate
    is the committed token, one rule for both. The update reads the seen positions
    (labeled, drafted, every earlier draft the committed token: there the step's
    maxima are teacher forcing's) whose committed token is a candidate: this
    rank's share of the cross-entropy's gradient into the input's W1 row
    (``grad``, [entries * n_spec, dim]; zero elsewhere), the input row (``prev``)
    and whether the position trains."""
    entries, finished_len, slots, cap = T.dynamic("entries, finished_len, slots, cap")
    dec_rows, grows = T.dynamic("dec_rows, grows")
    pairs = n_spec * parts

    @T.prim_func
    def combine(
        entry_f: T.Tensor((entries,), T.int64),
        entry_k: T.Tensor((entries,), T.int64),
        finished: T.Tensor((finished_len,), T.int64),
        labels: T.Tensor((finished_len, cap, n_spec), T.int32),
        ids: T.Tensor((finished_len, cap, n_spec + 1), T.int32),
        ring_index: T.Tensor((slots, cap, n_spec, parts), T.int32),
        ring_meta: T.Tensor((slots, cap, _META), T.int64),
        ring_drafts: T.Tensor((slots, cap, n_spec), T.int32),
        dec: T.Tensor((dec_rows, dim), T.bfloat16),
        z_in: T.Tensor((entries, n_spec, parts), T.float32),
        gathered: T.Tensor((grows, _LOCAL * ranks), T.float32),
        grad: T.Tensor((grows, dim), T.float32),
        train_out: T.Tensor((grows,), T.int32),
        prev_out: T.Tensor((grows,), T.int32),
        counts: T.Tensor((entries, _COUNTS), T.int32),
        loss: T.Tensor((entries,), T.float32),
    ):
        with T.Kernel(entries, threads=_THREADS) as e:
            tx = T.get_thread_binding()
            lse = T.alloc_shared((n_spec,), T.float32)
            train = T.alloc_shared((n_spec,), T.int32)
            coef = T.alloc_shared((n_spec, parts), T.float32)
            acc = T.alloc_local((_COUNTS,), T.int32)
            dv = T.alloc_local((1,), T.float32)
            di = T.alloc_local((1,), T.float32)
            lv = T.alloc_local((1,), T.float32)
            li = T.alloc_local((1,), T.float32)
            qv = T.alloc_local((1,), T.float32)
            qi = T.alloc_local((1,), T.float32)
            peak = T.alloc_local((1,), T.float32)
            total = T.alloc_local((1,), T.float32)
            zlab = T.alloc_local((1,), T.float32)
            has = T.alloc_local((1,), T.float32)
            ok = T.alloc_local((1,), T.float32)
            run_draft = T.alloc_local((1,), T.int32)
            run_loaded = T.alloc_local((1,), T.int32)
            run_learned = T.alloc_local((1,), T.int32)
            loss_sum = T.alloc_local((1,), T.float32)
            g = T.alloc_local((1,), T.float32)
            f = T.alloc_var(T.int64, init=entry_f[e])
            k = T.alloc_var(T.int64, init=entry_k[e])
            slot = T.alloc_var(T.int64, init=finished[f])
            held = labels[f, k, 0] != -2
            drafted = T.alloc_var(T.int64, init=ring_meta[slot, k, 1])
            if tx == 0:
                for s in T.serial(_COUNTS):
                    acc[s] = 0
                acc[0] = T.if_then_else(held, 1, 0)
                run_draft[0] = 1
                run_loaded[0] = 1
                run_learned[0] = 1
                loss_sum[0] = T.float32(0)
                for i in T.serial(n_spec):
                    row = T.cast(e, T.int64) * n_spec + i
                    label = labels[f, k, i]
                    dv[0] = -T.infinity(T.float32)
                    di[0] = T.float32(_NO_INDEX)
                    lv[0] = -T.infinity(T.float32)
                    li[0] = T.float32(_NO_INDEX)
                    qv[0] = -T.infinity(T.float32)
                    qi[0] = T.float32(_NO_INDEX)
                    peak[0] = -T.infinity(T.float32)
                    zlab[0] = -T.infinity(T.float32)
                    has[0] = T.float32(0)
                    ok[0] = T.float32(0)
                    for r in T.serial(ranks):
                        o = r * _LOCAL
                        _take_float(dv, di, gathered[row, o], gathered[row, o + 1])
                        _take_float(lv, li, gathered[row, o + 2], gathered[row, o + 3])
                        _take_float(qv, qi, gathered[row, o + 4], gathered[row, o + 5])
                        peak[0] = T.max(peak[0], gathered[row, o + 4])
                        zlab[0] = T.max(zlab[0], gathered[row, o + 7])
                        has[0] = T.max(has[0], gathered[row, o + 8])
                        ok[0] = T.max(ok[0], gathered[row, o + 9])
                    total[0] = T.float32(0)
                    for r in T.serial(ranks):
                        o = r * _LOCAL
                        if gathered[row, o + 9] > 0:
                            total[0] = total[0] + gathered[row, o + 6] * T.exp(
                                gathered[row, o + 4] - peak[0]
                            )
                    draft = T.cast(di[0], T.int32)
                    # The drafts the ring holds (all but the last drafted one).
                    if held & (T.cast(i + 1, T.int64) < drafted):
                        acc[8] = acc[8] + 1
                        if ring_drafts[slot, k, i] != draft:
                            acc[9] = acc[9] + 1
                    scored = (ok[0] > 0) & (li[0] < T.float32(_NO_INDEX))
                    # The score: each table's teacher-forced run, by one rule.
                    if (run_loaded[0] != 0) & scored & (T.cast(li[0], T.int32) == label):
                        acc[4] = acc[4] + 1
                    else:
                        run_loaded[0] = 0
                    if (run_learned[0] != 0) & scored & (T.cast(qi[0], T.int32) == label):
                        acc[5] = acc[5] + 1
                    else:
                        run_learned[0] = 0
                    # The update's positions: the step's input was the committed one.
                    seen = scored & (run_draft[0] != 0) & (di[0] < T.float32(_NO_INDEX))
                    lse[i] = T.float32(0)
                    train[i] = 0
                    if seen:
                        acc[1] = acc[1] + 1
                        if draft == label:
                            acc[3] = acc[3] + 1
                        if has[0] > 0:
                            acc[2] = acc[2] + 1
                            train[i] = 1
                            lse[i] = peak[0] + T.log(total[0])
                            loss_sum[0] = loss_sum[0] + lse[i] - zlab[0]
                    run_draft[0] = T.if_then_else(seen & (draft == label), run_draft[0], 0)
                    train_out[row] = train[i]
                    prev_out[row] = T.if_then_else(seen, ids[f, k, i], 0)
                acc[6] = T.if_then_else(acc[5] > acc[4], 1, 0)
                acc[7] = T.if_then_else(acc[5] < acc[4], 1, 0)
                for s in T.serial(_COUNTS):
                    counts[e, s] = acc[s]
                loss[e] = loss_sum[0]
            T.sync_threads()
            if tx < pairs:
                i = tx // parts
                c = tx % parts
                x = ring_index[slot, k, i, c]
                coef[i, c] = T.if_then_else(
                    (train[i] != 0) & (x >= 0),
                    T.exp(z_in[e, i, c] - lse[i])
                    - T.if_then_else(x == labels[f, k, i], T.float32(1), T.float32(0)),
                    T.float32(0),
                )
            T.sync_threads()
            for i in T.serial(n_spec):
                for it in T.serial(T.ceildiv(dim, _THREADS)):
                    d = it * _THREADS + tx
                    if d < dim:
                        g[0] = T.float32(0)
                        if train[i] != 0:
                            for c in T.serial(parts):
                                g[0] = g[0] + coef[i, c] * T.cast(
                                    dec[(T.cast(e, T.int64) * n_spec + i) * parts + c, d],
                                    T.float32,
                                )
                        grad[T.cast(e, T.int64) * n_spec + i, d] = g[0]

    return combine


@tilelang.jit(pass_configs=_PASS)
def shadow_update_kernel(dim: int):
    """One block a position: the first training position of each W1 row sums the
    gradient of every training position with that row as input, in position
    order, adds the sum's squared norm to the row's AdaGrad accumulator and steps
    the FP32 copy's row by ``lr / (sqrt(accumulator) + 1e-6)`` of the sum. Every
    rank runs it on the same all-reduced gradient, in the same order."""
    positions, table_rows = T.dynamic("positions, table_rows")

    @T.prim_func
    def update(
        train: T.Tensor((positions,), T.int32),
        prev: T.Tensor((positions,), T.int32),
        grad: T.Tensor((positions, dim), T.float32),
        master: T.Tensor((table_rows, dim), T.float32),
        accum: T.Tensor((table_rows,), T.float32),
        lr: T.float32,
    ):
        with T.Kernel(positions, threads=_THREADS) as pos:
            tx = T.get_thread_binding()
            owner = T.alloc_shared((1,), T.int32)
            squares = T.alloc_shared((_THREADS,), T.float32)
            scale = T.alloc_shared((1,), T.float32)
            g = T.alloc_local((-(-dim // _THREADS),), T.float32)
            part = T.alloc_local((1,), T.float32)
            p = T.alloc_var(T.int32, init=prev[pos])
            if tx == 0:
                owner[0] = train[pos]
            T.sync_threads()
            if train[pos] != 0:
                for it in T.serial(T.ceildiv(pos, _THREADS)):
                    j = it * _THREADS + tx
                    if j < pos:
                        if (train[j] != 0) & (prev[j] == p):
                            owner[0] = 0
            T.sync_threads()
            part[0] = T.float32(0)
            for it in T.serial(T.ceildiv(dim, _THREADS)):
                d = it * _THREADS + tx
                g[it] = T.float32(0)
                if (owner[0] != 0) & (d < dim):
                    for j in T.serial(positions - pos):
                        q = pos + j
                        if (train[q] != 0) & (prev[q] == p):
                            g[it] = g[it] + grad[q, d]
                    part[0] = part[0] + g[it] * g[it]
            squares[tx] = part[0]
            T.sync_threads()
            if (tx == 0) & (owner[0] != 0):
                part[0] = T.float32(0)
                for t in T.serial(_THREADS):
                    part[0] = part[0] + squares[t]
                accum[p] = accum[p] + part[0]
                scale[0] = lr / (T.sqrt(accum[p]) + T.float32(1e-6))
            T.sync_threads()
            if owner[0] != 0:
                for it in T.serial(T.ceildiv(dim, _THREADS)):
                    d = it * _THREADS + tx
                    if d < dim:
                        master[p, d] = master[p, d] - scale[0] * g[it]

    return update


@tilelang.jit(pass_configs=_PASS)
def shadow_totals_kernel():
    """One block: a chunk's entry counters and losses added, in entry order, to
    the call's and the boot's totals (``loss`` last)."""
    entries = T.dynamic("entries")

    @T.prim_func
    def totals(
        counts: T.Tensor((entries, _COUNTS), T.int32),
        loss: T.Tensor((entries,), T.float32),
        call: T.Tensor((_COUNTS + 1,), T.float64),
        boot: T.Tensor((_COUNTS + 1,), T.float64),
    ):
        with T.Kernel(1, threads=32) as _:
            tx = T.get_thread_binding()
            acc = T.alloc_local((1,), T.float64)
            if tx < _COUNTS + 1:
                acc[0] = T.float64(0)
                for e in T.serial(entries):
                    acc[0] = acc[0] + T.if_then_else(
                        tx < _COUNTS,
                        T.cast(counts[e, T.min(tx, _COUNTS - 1)], T.float64),
                        T.cast(loss[e], T.float64),
                    )
                call[tx] = call[tx] + acc[0]
                boot[tx] = boot[tx] + acc[0]

    return totals


# --- The learner -----------------------------------------------------------------


def _tile_cols(rows_in_layer: int, num_tiles: int) -> int:
    """The columns of each tile the fused projection reduces: the divisor of the
    greedy part whose tiling gives ``num_tiles``."""
    cols = [c for c in (16, 32, 64, 128, 256, 512, 1024, 2048) if -(-rows_in_layer // c) == num_tiles]
    if not cols:
        raise RuntimeError(
            f"shadow turn learning: {num_tiles} fused tiles do not tile "
            f"{rows_in_layer} rows by a divisor of {GREEDY_BLOCK}"
        )
    return cols[0]


class DSparkShadowLearner:
    """Keeps each request's block maxima and trains an FP32 copy of Markov W1 on
    each finished turn, scoring the turn with the copy first."""

    mode = "shadow"

    def __init__(
        self,
        *,
        model: Any,
        table: torch.Tensor,
        markov_w2: Any,
        draft_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_idx: torch.Tensor,
        sample_idx_mapping: torch.Tensor,
        sample_pos: torch.Tensor,
        all_gather: Callable[[torch.Tensor], torch.Tensor],
        all_reduce: Callable[[torch.Tensor], torch.Tensor],
        max_num_reqs: int,
        num_speculative_steps: int,
        tp_size: int,
        tp_rank: int,
        device: torch.device,
        capacity: int,
        lr: float,
    ) -> None:
        if not hasattr(markov_w2, "packed_bf16_exponent_base"):
            raise RuntimeError(
                "shadow turn learning decodes candidates' W2 rows from a 12-bit "
                "packed Markov head; this one is not packed"
            )
        if draft_tokens.dtype != torch.int64:
            raise TypeError("turn learning reads int64 drafts")
        self.model = model
        self.table = table
        self.w2 = markov_w2
        self.draft_tokens = draft_tokens
        self.input_ids = input_ids
        self.anchor_idx = anchor_idx
        self.sample_idx_mapping = sample_idx_mapping
        self.sample_pos = sample_pos
        self.all_gather = all_gather
        self.all_reduce = all_reduce
        self.max_num_reqs = int(max_num_reqs)
        self.n_spec = int(num_speculative_steps)
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.device = device
        self.capacity = int(capacity)
        self.lr = float(lr)
        self.dim = int(table.shape[1])
        self.vocab_start, self.shard_vocab = model.draft_vocab_shard
        self.parts = -(-self.shard_vocab // GREEDY_BLOCK)
        self.chunk = min(_CHUNK, self.max_num_reqs * self.capacity)

        # The fused first position's tiles (its kernels are bound after loading, so
        # each token tile's count is read at its first call): at most one a 16
        # columns of the layer's rows.
        self.layer_rows = int(markov_w2.weight.shape[0])
        self.fused_tiles: dict[int, tuple[int, int]] = {}
        max_tiles = -(-self.layer_rows // 16)

        slots, cap, n, parts = self.max_num_reqs, self.capacity, self.n_spec, self.parts
        f32, i32, i64 = torch.float32, torch.int32, torch.int64
        # The step's partials, one scratch a position (the shared scratch is
        # reused from one position to the next).
        self.part_values = torch.empty((n, slots * parts), dtype=f32, device=device)
        self.part_indices = torch.empty((n, slots * parts), dtype=i32, device=device)
        self.tile_values = torch.empty(slots * max_tiles, dtype=f32, device=device)
        self.tile_indices = torch.empty(slots * max_tiles, dtype=i32, device=device)
        self.ring_values = torch.empty((slots, cap, n, parts), dtype=f32, device=device)
        self.ring_index = torch.full((slots, cap, n, parts), -1, dtype=i32, device=device)
        self.ring_meta = torch.full((slots, cap, _META), -1, dtype=i64, device=device)
        self.ring_drafts = torch.full((slots, cap, n), -1, dtype=i32, device=device)
        self.ring_count = torch.zeros(slots, dtype=i64, device=device)
        self.host_count = np.zeros(slots, dtype=np.int64)

        self.labels = torch.full((slots, cap, n), -1, dtype=i32, device=device)
        self.ids = torch.zeros((slots, cap, n + 1), dtype=i32, device=device)
        e = self.chunk
        self.dec = torch.empty((e * n * parts, self.dim), dtype=torch.bfloat16, device=device)
        self.z = torch.empty((e, n, parts), dtype=f32, device=device)
        self.summary = torch.empty((e * n, _LOCAL), dtype=f32, device=device)
        self.grad = torch.empty((e * n, self.dim), dtype=f32, device=device)
        self.train = torch.empty(e * n, dtype=i32, device=device)
        self.prev = torch.empty(e * n, dtype=i32, device=device)
        self.counts = torch.empty((e, _COUNTS), dtype=i32, device=device)
        self.loss = torch.empty(e, dtype=f32, device=device)
        self.call_totals = torch.zeros(_COUNTS + 1, dtype=torch.float64, device=device)
        self.boot_totals = torch.zeros(_COUNTS + 1, dtype=torch.float64, device=device)

        # The learned copy and its AdaGrad accumulators.
        self.master = table.float().clone()
        self.accum = torch.zeros(table.shape[0], dtype=f32, device=device)

        self.side = torch.cuda.Stream(device=device)
        self._forked = False
        self.turns = 0
        self.calls_since_check = 0
        self._log_queue: list[tuple[Any, Any, int, torch.Tensor, torch.Tensor]] = []
        mib = (
            sum(
                t.numel() * t.element_size()
                for t in (
                    self.part_values, self.part_indices, self.tile_values,
                    self.tile_indices, self.ring_values, self.ring_index,
                    self.ring_meta, self.ring_drafts, self.labels, self.ids,
                    self.dec, self.z, self.summary, self.grad, self.master,
                    self.accum,
                )
            )
            / (1 << 20)
        )
        logger.info(
            "DSpark turn learning (shadow): %d draft steps kept per request, %d "
            "block maxima a position, AdaGrad step %.3g, %.1f MiB a rank "
            "(%.1f MiB the FP32 copy of W1)",
            cap,
            parts,
            self.lr,
            mib,
            self.master.numel() * 4 / (1 << 20),
        )

    # -- declarations ---------------------------------------------------------

    def calls(self) -> list:
        from vllm.models.deepseek_v4_1.tilelang.embedding import packed_kernel_calls

        n, parts, dim = self.n_spec, self.parts, self.dim
        return [
            _call(shadow_capture_kernel, n, parts),
            _call(labels_kernel, n),
            _call(reset_kernel),
            _call(shadow_local_kernel, n, parts, dim),
            _call(shadow_combine_kernel, n, parts, dim, self.tp_size),
            _call(shadow_update_kernel, dim),
            _call(shadow_totals_kernel),
            *packed_kernel_calls(
                int(self.w2.packed_bf16_in_features),
                int(self.w2.packed_bf16_exponent_base),
            ),
        ]

    # -- the draft step (inside the draft graph) --------------------------------

    def partials(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Position ``position``'s scratch for ``greedy_markov_shard_pair``."""
        return self.part_values[position], self.part_indices[position]

    def fused_pairs(
        self, markov_embed: torch.Tensor, base_logits: torch.Tensor, pairs: torch.Tensor
    ) -> tuple[int, int]:
        """``model.markov_greedy_pairs_local`` (``project_head_markov_greedy_pairs``
        with the step's pairs as output), with the tiles' partials in the learner's
        scratch instead of the shared one: the same kernels and arguments.
        Returns the call's (tiles, tile columns)."""
        from vllm.models.deepseek_v4_1.tilelang.vocab import row_tile

        x = markov_embed if markov_embed.is_contiguous() else markov_embed.contiguous()
        rows = x.shape[0]
        layer = self.w2
        block_m = row_tile(rows)
        partials_kernel, finish_kernel, _, num_tiles = layer.tilelang_vocab_markov_greedy[block_m]
        if block_m not in self.fused_tiles:
            self.fused_tiles[block_m] = (int(num_tiles), _tile_cols(self.layer_rows, int(num_tiles)))
        if rows * num_tiles > self.tile_values.numel():
            raise RuntimeError("shadow turn learning: fused tiles exceed the learner's scratch")
        values = self.tile_values[: rows * num_tiles].view(rows, num_tiles)
        indices = self.tile_indices[: rows * num_tiles].view(rows, num_tiles)
        partials_kernel(
            x,
            layer.weight,
            layer.weight_exception_rows,
            layer.weight_exception_entries,
            layer.weight_row_offsets,
            base_logits,
            values,
            indices,
            self.shard_vocab,
        )
        finish_kernel(values, indices, pairs, self.vocab_start)
        return self.fused_tiles[block_m]

    def capture(self, num_reqs: int, active: int, fused: tuple[int, int] | None) -> None:
        """Fork the capture of this step's maxima onto the side stream, after the
        last position's partials; ``join`` before the graph ends."""
        if num_reqs == 0:
            return
        num_tiles, cols = fused if fused is not None else (1, GREEDY_BLOCK)
        main = torch.cuda.current_stream()
        self.side.wait_stream(main)
        with torch.cuda.stream(self.side):
            rows = num_reqs * self.n_spec
            _kernel(shadow_capture_kernel, self.n_spec, self.parts)(
                self.part_values,
                self.part_indices,
                self.tile_values,
                self.tile_indices,
                self.sample_idx_mapping[:rows],
                self.sample_pos[:rows],
                self.draft_tokens,
                self.input_ids,
                self.anchor_idx,
                self.ring_values,
                self.ring_index,
                self.ring_meta,
                self.ring_drafts,
                self.ring_count,
                int(active),
                1 if fused is not None else 0,
                int(num_tiles),
                GREEDY_BLOCK // int(cols),
                int(self.vocab_start),
            )
        self._forked = True

    def join(self) -> None:
        if self._forked:
            torch.cuda.current_stream().wait_stream(self.side)
            self._forked = False

    def add_request(self, slot: int) -> None:
        slots = async_tensor_h2d([slot], dtype=torch.int64, device=self.device)
        _kernel(reset_kernel)(slots, self.ring_count)
        self.host_count[slot] = 0

    def note_drafts(self, slots: np.ndarray) -> None:
        self.host_count[slots] += 1

    # -- end of turn ----------------------------------------------------------

    def finish(
        self,
        finished_slots: list[int],
        preempted_slots: list[int],
        all_token_ids: torch.Tensor,
        total_len: torch.Tensor,
    ) -> None:
        """Score each finished turn with the learned copy, then train the copy on
        it; clear the finished and preempted requests' rings. Every rank calls this
        in the same step with the same slots in the same order."""
        self._flush_logs()
        kept = [int(min(self.host_count[s], self.capacity)) for s in finished_slots]
        if any(kept):
            start = None
            if self.tp_rank == 0:
                start = torch.cuda.Event(enable_timing=True)
                start.record()
            finished = async_tensor_h2d(finished_slots, dtype=torch.int64, device=self.device)
            n_fin = len(finished_slots)
            _kernel(labels_kernel, self.n_spec)(
                finished,
                all_token_ids,
                total_len,
                self.ring_meta,
                self.ring_count,
                self.labels[:n_fin],
                self.ids[:n_fin],
            )
            self.call_totals.zero_()
            entries = [(f, k) for f, count in enumerate(kept) for k in range(count)]
            for s in range(0, len(entries), self.chunk):
                self._chunk(finished, finished_slots, entries[s : s + self.chunk])
            self.turns += sum(1 for c in kept if c)
            if start is not None:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._log_queue.append(
                    (
                        start,
                        end,
                        sum(1 for c in kept if c),
                        self.call_totals.to("cpu", non_blocking=True),
                        self.boot_totals.to("cpu", non_blocking=True),
                    )
                )
            self.calls_since_check += 1
            if self.calls_since_check >= 256:
                self.calls_since_check = 0
                self._check_replicas()
        cleared = list(finished_slots) + list(preempted_slots)
        if cleared:
            slots = async_tensor_h2d(cleared, dtype=torch.int64, device=self.device)
            _kernel(reset_kernel)(slots, self.ring_count)
            self.host_count[cleared] = 0

    def _chunk(
        self, finished: torch.Tensor, finished_slots: list[int], entries: list[tuple[int, int]]
    ) -> None:
        from vllm.models.deepseek_v4_1.tilelang.embedding import vocab_embedding_packed

        n, parts, dim = self.n_spec, self.parts, self.dim
        count = len(entries)
        rows = count * n
        entry_f = async_tensor_h2d([f for f, _ in entries], dtype=torch.int64, device=self.device)
        entry_k = async_tensor_h2d([k for _, k in entries], dtype=torch.int64, device=self.device)
        # The candidates' W2 rows, a run of kept entries of one turn at a time.
        w2 = self.w2
        shard = w2.shard_indices
        start = 0
        while start < count:
            f, k0 = entries[start]
            stop = start
            while stop < count and entries[stop][0] == f:
                stop += 1
            k1 = k0 + (stop - start)
            vocab_embedding_packed(
                self.ring_index[finished_slots[f], k0:k1].reshape(-1),
                w2.weight,
                w2.weight_exception_entries,
                w2.weight_row_offsets,
                int(w2.packed_bf16_in_features),
                int(w2.packed_bf16_exponent_base),
                self.dec[start * n * parts : stop * n * parts],
                shard.org_vocab_start_index,
                shard.org_vocab_end_index,
                shard.num_org_vocab_padding,
                shard.added_vocab_start_index,
                shard.added_vocab_end_index,
            )
            start = stop
        labels = self.labels[: len(finished_slots)]
        ids = self.ids[: len(finished_slots)]
        _kernel(shadow_local_kernel, n, parts, dim)(
            entry_f,
            entry_k,
            finished,
            labels,
            ids,
            self.ring_values,
            self.ring_index,
            self.ring_meta,
            self.ring_drafts,
            self.dec[: rows * parts],
            self.master,
            self.table,
            self.z[:count],
            self.summary[:rows].view(count, n, _LOCAL),
        )
        gathered = self.all_gather(self.summary[:rows]).contiguous()
        _kernel(shadow_combine_kernel, n, parts, dim, self.tp_size)(
            entry_f,
            entry_k,
            finished,
            labels,
            ids,
            self.ring_index,
            self.ring_meta,
            self.ring_drafts,
            self.dec[: rows * parts],
            self.z[:count],
            gathered,
            self.grad[:rows],
            self.train[:rows],
            self.prev[:rows],
            self.counts[:count],
            self.loss[:count],
        )
        grad = self.all_reduce(self.grad[:rows])
        _kernel(shadow_update_kernel, dim)(
            self.train[:rows], self.prev[:rows], grad, self.master, self.accum, self.lr
        )
        _kernel(shadow_totals_kernel)(
            self.counts[:count], self.loss[:count], self.call_totals, self.boot_totals
        )

    def _check_replicas(self) -> None:
        """Every rank's copy must stay the same (they apply one all-reduced update):
        a sum of each copy, compared on rank 0's log."""
        total = torch.sum(self.master, dtype=torch.float64).reshape(1)
        # Exchanged as the sum's two 32-bit words, compared bit for bit.
        sums = self.all_gather(total.view(torch.int32).reshape(1, 2)).reshape(-1, 2)
        if self.tp_rank == 0:
            words = sums.cpu()
            values = words.view(torch.float64).flatten().tolist()
            same = bool((words == words[0]).all())
            (logger.info if same else logger.warning)(
                "DSpark turn learning (shadow): the copies of W1 %s across ranks (%s)",
                "agree" if same else "differ",
                ", ".join(f"{v:.17g}" for v in values),
            )

    # -- logging (rank 0) -----------------------------------------------------

    def _flush_logs(self) -> None:
        while self._log_queue and self._log_queue[0][1].query():
            start, end, turns, call, boot = self._log_queue.pop(0)
            elapsed = start.elapsed_time(end)
            c = ShadowStats.as_dict(call[:_COUNTS].tolist())
            b = ShadowStats.as_dict(boot[:_COUNTS].tolist())

            def gain(stats: dict[str, int]) -> float:
                return 100.0 * (stats["tf_learned"] - stats["tf_loaded"]) / max(
                    stats["tf_loaded"], 1
                )

            # The maxima-limited score misses most of what the learned copy fixes
            # (the committed token is often not among the step's maxima where the
            # draft missed); offline it reads about +0.0% where the exact gain is
            # about +0.9% (README). It is logged as a health signal, not the gain.
            logger.info(
                "DSpark turn learning (shadow): %d turns %s loss %.4f, maxima-limited "
                "score %+.2f%%; since boot %s, maxima-limited score %+.2f%% "
                "(end of turn %.3f ms)",
                turns,
                c,
                float(call[_COUNTS]) / max(c["trained"], 1),
                gain(c),
                b,
                gain(b),
                elapsed,
            )
            if c["draft_mismatches"]:
                logger.warning(
                    "DSpark turn learning (shadow): %d drafts the captured maxima "
                    "do not give",
                    c["draft_mismatches"],
                )
