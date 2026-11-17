# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark drafter learning at the end of each turn (``VLLM_DSPARK_TURN_LEARNING``).

A turn is one request. While it decodes, every draft step's backbone rows are kept
in the request's ring (``capture``, a TileLang kernel inside the captured draft
graph): the ``N`` pre-norm head rows, the anchor's position and the step's drafts.
DSpark drafts its block in one parallel pass, so those rows do not depend on the
drafted tokens; only the sequential Markov head does.

When the scheduler reports the request finished (``finish``, called by every rank
in the same step and in sorted order, before the slot can be reused), every token
of the turn is committed. Each kept draft position then has its label, the token
the target committed there, and the turn is replayed with the declared draft
kernels (head projection, Markov projection, shard pairs, first-argmax):

- the replay with the drafted tokens as the chain's inputs must reproduce the
  drafts the step made, bit for bit (``replay_mismatches`` is 0);
- the replay with the labels as the chain's inputs (teacher forcing) gives the
  exact number of positions each block would have had accepted, all ``N`` of
  them, with whatever Markov W1 the chain reads.

``observe`` changes no draft and no weight. With ``VLLM_DSPARK_TURN_LEARNING_DUMP``
set, rank 0 also writes each turn's rows and labels to that directory, for an
offline study of the update. ``shadow`` (``turn_learning_shadow.py``) keeps only the
step's block maxima and trains a copy of W1 the draft step does not read.

Off (the default), nothing is allocated or launched.
"""

# No `from __future__ import annotations`: TileLang reads the prim_func
# annotations eagerly, and the symbolic shapes must resolve.
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.tilelang_utils import T, tilelang
from vllm.v1.worker.gpu import tilelang_inputs

logger = init_logger(__name__)

_MODES = ("off", "observe", "shadow")  # shadow: turn_learning_shadow.py

_PASS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    "tl.if_stmt_binding_inline_replayable_binds": False,
}
_CAPTURE_THREADS = 256
_LABEL_THREADS = 128
_STATS = 9  # see ``TurnStats``
_META = 3  # anchor position, active drafts, the anchor token the step drafted from


def turn_learning_mode() -> str:
    mode = os.environ.get("VLLM_DSPARK_TURN_LEARNING", "off").strip().lower() or "off"
    if mode not in _MODES:
        raise ValueError(
            f"VLLM_DSPARK_TURN_LEARNING={mode!r}: expected one of {', '.join(_MODES)}"
        )
    return mode


def turn_learning_capacity() -> int:
    value = int(os.environ.get("VLLM_DSPARK_TURN_LEARNING_CAPACITY", "64"))
    if value < 1:
        raise ValueError("VLLM_DSPARK_TURN_LEARNING_CAPACITY must be at least 1")
    return value


def turn_learning_dump_dir() -> str | None:
    value = os.environ.get("VLLM_DSPARK_TURN_LEARNING_DUMP", "").strip()
    return value or None


# --- Kernels ---------------------------------------------------------------------


@tilelang.jit(pass_configs=_PASS)
def capture_kernel(n_spec: int, words: int):
    """One block a draft row: the row's ``n_spec`` pre-norm head rows (as int64
    words), its drafts, its anchor position and the anchor token it drafted from
    into its request's ring at entry ``count % capacity``, then ``count + 1``. A row whose sampling slot is -1
    (padding of the captured size) writes nothing. Each request has at most one
    row a step, so no two blocks touch one ring."""
    hidden_rows, map_len = T.dynamic("hidden_rows, map_len")
    draft_rows, draft_stride = T.dynamic("draft_rows, draft_stride")
    slots, cap = T.dynamic("slots, cap")
    ids_len, anchor_len = T.dynamic("ids_len, anchor_len")
    span = n_spec * words

    @T.prim_func
    def capture(
        hidden: T.Tensor((hidden_rows, words), T.int64),
        sample_map: T.Tensor((map_len,), T.int32),
        sample_pos: T.Tensor((map_len,), T.int64),
        drafts: T.StridedTensor((draft_rows, n_spec), (draft_stride, 1), T.int64),
        input_ids: T.Tensor((ids_len,), T.int32),
        anchor_idx: T.Tensor((anchor_len,), T.int64),
        ring_hidden: T.Tensor((slots, cap, n_spec, words), T.int64),
        ring_meta: T.Tensor((slots, cap, _META), T.int64),
        ring_drafts: T.Tensor((slots, cap, n_spec), T.int32),
        ring_count: T.Tensor((slots,), T.int64),
        active: T.int32,
    ):
        with T.Kernel(hidden_rows // n_spec, threads=_CAPTURE_THREADS) as r:
            tx = T.get_thread_binding()
            slot = T.alloc_var(T.int64, init=T.cast(sample_map[r * n_spec], T.int64))
            safe = T.alloc_var(T.int64, init=T.max(slot, T.int64(0)))
            count = T.alloc_var(T.int64, init=ring_count[safe])
            entry = T.alloc_var(T.int64, init=count % T.cast(cap, T.int64))
            if slot >= 0:
                for it in T.serial(T.ceildiv(span, _CAPTURE_THREADS)):
                    e = it * _CAPTURE_THREADS + tx
                    if e < span:
                        ring_hidden[safe, entry, e // words, e % words] = hidden[
                            r * n_spec + e // words, e % words
                        ]
                if tx < n_spec:
                    ring_drafts[safe, entry, tx] = T.cast(drafts[r, tx], T.int32)
                if tx == 0:
                    ring_meta[safe, entry, 0] = sample_pos[r * n_spec] - 1
                    ring_meta[safe, entry, 1] = T.cast(active, T.int64)
                    ring_meta[safe, entry, 2] = T.cast(input_ids[anchor_idx[r]], T.int64)
            T.sync_threads()
            if (slot >= 0) & (tx == 0):
                ring_count[safe] = count + 1

    return capture


@tilelang.jit(pass_configs=_PASS)
def labels_kernel(n_spec: int):
    """A thread a kept entry of each finished request: the anchor and the
    committed tokens at the entry's ``n_spec`` predicted positions. ``labels`` is
    -1 where nothing was committed past the turn's end, and -2 at every position
    of an entry the ring does not hold (the host counted a draft step that wrote
    no row, the anchor lies outside the turn, or the step drafted from another
    token than the turn committed there);
    ``ids`` (anchor, then the labels) holds 0 there, a valid row for the Markov
    lookups that teacher forcing makes."""
    finished_len, slots, cap = T.dynamic("finished_len, slots, cap")
    token_rows, max_len, token_stride = T.dynamic("token_rows, max_len, token_stride")
    total_rows = T.dynamic("total_rows")
    grid = T.ceildiv(finished_len * cap, _LABEL_THREADS)

    @T.prim_func
    def labels(
        finished: T.Tensor((finished_len,), T.int64),
        all_tokens: T.StridedTensor(
            (token_rows, max_len), (token_stride, 1), T.int32
        ),
        total_len: T.Tensor((total_rows,), T.int32),
        ring_meta: T.Tensor((slots, cap, _META), T.int64),
        ring_count: T.Tensor((slots,), T.int64),
        labels_out: T.Tensor((finished_len, cap, n_spec), T.int32),
        ids_out: T.Tensor((finished_len, cap, n_spec + 1), T.int32),
    ):
        with T.Kernel(grid, threads=_LABEL_THREADS) as b:
            e = b * _LABEL_THREADS + T.get_thread_binding()
            f = T.min(e // cap, finished_len - 1)
            k = e % cap
            slot = T.alloc_var(T.int64, init=finished[f])
            count = T.alloc_var(T.int64, init=ring_count[slot])
            total = T.alloc_var(T.int64, init=T.cast(total_len[slot], T.int64))
            anchor_pos = T.alloc_var(T.int64, init=ring_meta[slot, k, 0])
            anchor = T.alloc_var(
                T.int32,
                init=all_tokens[
                    slot, T.min(T.max(anchor_pos, T.int64(0)), T.cast(max_len - 1, T.int64))
                ],
            )
            # Held: inside the ring, inside the turn, and drafted from the token the
            # turn committed at its anchor (a draft over any other context is not
            # this turn's, whatever path made it).
            kept = (
                (T.cast(k, T.int64) < T.min(count, T.cast(cap, T.int64)))
                & (anchor_pos >= 0)
                & (anchor_pos < total)
                & (T.cast(anchor, T.int64) == ring_meta[slot, k, 2])
            )
            if e < finished_len * cap:
                ids_out[f, k, 0] = T.if_then_else(kept, anchor, 0)
                for i in T.serial(n_spec):
                    q = anchor_pos + T.cast(i + 1, T.int64)
                    ok = kept & (q < total)
                    token = all_tokens[
                        slot, T.min(T.max(q, T.int64(0)), T.cast(max_len - 1, T.int64))
                    ]
                    labels_out[f, k, i] = T.if_then_else(
                        ok, token, T.if_then_else(kept, -1, -2)
                    )
                    ids_out[f, k, i + 1] = T.if_then_else(ok, token, 0)

    return labels


@tilelang.jit(pass_configs=_PASS)
def reset_kernel():
    """``ring_count[slots] = 0``: the finished or preempted requests' rings."""
    count_len, slots = T.dynamic("count_len, slots")

    @T.prim_func
    def reset(
        finished: T.Tensor((count_len,), T.int64),
        ring_count: T.Tensor((slots,), T.int64),
    ):
        with T.Kernel(T.ceildiv(count_len, 128), threads=128) as b:
            i = b * 128 + T.get_thread_binding()
            if i < count_len:
                ring_count[finished[i]] = T.int64(0)

    return reset


@tilelang.jit(pass_configs=_PASS)
def stats_kernel(n_spec: int):
    """One thread folds a chunk's entries into the turn's counters, in entry
    order (``TurnStats``): ``first`` overwrites them, a later chunk adds."""
    rows = T.dynamic("rows")
    pred_rows, pred_stride = T.dynamic("pred_rows, pred_stride")

    @T.prim_func
    def stats(
        labels: T.Tensor((rows, n_spec), T.int32),
        drafts: T.Tensor((rows, n_spec), T.int32),
        meta: T.Tensor((rows, _META), T.int64),
        replay: T.StridedTensor((pred_rows, n_spec), (pred_stride, 1), T.int64),
        forced: T.StridedTensor((pred_rows, n_spec), (pred_stride, 1), T.int64),
        out: T.Tensor((_STATS,), T.int64),
        first: T.int32,
    ):
        with T.Kernel(1, threads=32) as _:
            tx = T.get_thread_binding()
            acc = T.alloc_local((_STATS,), T.int64)
            if tx == 0:
                for s in T.serial(_STATS):
                    acc[s] = T.if_then_else(first != 0, T.int64(0), out[s])
                for r in T.serial(rows):
                    held = labels[r, 0] != -2
                    active = T.alloc_var(
                        T.int64, init=T.if_then_else(held, meta[r, 1], T.int64(0))
                    )
                    labeled = T.alloc_var(T.int64, init=T.int64(0))
                    run_forced = T.alloc_var(T.int64, init=T.int64(0))
                    run_draft = T.alloc_var(T.int64, init=T.int64(0))
                    live_forced = T.alloc_var(T.int32, init=1)
                    live_draft = T.alloc_var(T.int32, init=1)
                    mismatches = T.alloc_var(T.int64, init=T.int64(0))
                    for i in T.serial(n_spec):
                        y = T.cast(labels[r, i], T.int64)
                        has = y >= 0
                        drafted = T.cast(i, T.int64) < active
                        if has:
                            labeled = labeled + 1
                        if live_forced != 0:
                            if has & (forced[r, i] == y):
                                run_forced = run_forced + 1
                            else:
                                live_forced = 0
                        if live_draft != 0:
                            if has & drafted & (T.cast(drafts[r, i], T.int64) == y):
                                run_draft = run_draft + 1
                            else:
                                live_draft = 0
                        if held & drafted & (replay[r, i] != T.cast(drafts[r, i], T.int64)):
                            mismatches = mismatches + 1
                    acc[0] = acc[0] + T.if_then_else(labeled > 0, T.int64(1), T.int64(0))
                    acc[1] = acc[1] + labeled
                    acc[2] = acc[2] + run_forced
                    acc[3] = acc[3] + run_draft
                    acc[4] = acc[4] + T.min(run_forced, active)
                    acc[5] = acc[5] + mismatches
                    acc[6] = acc[6] + active
                    acc[7] = acc[7] + T.int64(1)
                    acc[8] = acc[8] + T.if_then_else(held, T.int64(0), T.int64(1))
                for s in T.serial(_STATS):
                    out[s] = acc[s]

    return stats


def _kernel(factory: Any, *key: Any) -> Any:
    return tilelang_inputs._kernel(factory, *key)


def _call(factory: Any, *key: Any) -> tuple:
    return tilelang_inputs._call(factory, *key)


# --- The learner -----------------------------------------------------------------


class TurnStats:
    """The counters of one turn, in ``stats_kernel``'s order."""

    FIELDS = (
        "entries",  # kept draft steps with at least one label
        "labeled",  # draft positions with a committed token
        "forced_accepted",  # leading positions teacher forcing predicts, all N
        "draft_accepted",  # leading drafted positions the drafts got right
        "forced_accepted_active",  # forced_accepted capped at the step's drafts
        "replay_mismatches",  # replayed drafts that differ from the step's
        "drafted",  # positions the steps drafted
        "kept",  # entries replayed
        "unheld",  # entries the ring does not hold (see labels_kernel)
    )

    @classmethod
    def as_dict(cls, values: Any) -> dict[str, int]:
        return {name: int(v) for name, v in zip(cls.FIELDS, values)}


class DSparkTurnLearner:
    """Keeps each request's draft rows and replays the turn when it finishes."""

    def __init__(
        self,
        *,
        model: Any,
        table: torch.Tensor,
        draft_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_idx: torch.Tensor,
        sample_idx_mapping: torch.Tensor,
        sample_pos: torch.Tensor,
        partial_values: torch.Tensor,
        partial_indices: torch.Tensor,
        fused_step: Callable[[int, int, int], bool],
        all_gather: Callable[[torch.Tensor], torch.Tensor],
        max_num_reqs: int,
        num_speculative_steps: int,
        hidden_size: int,
        tp_size: int,
        tp_rank: int,
        device: torch.device,
        capacity: int,
        dump_dir: str | None = None,
    ) -> None:
        if hidden_size % 4 != 0:
            raise ValueError("turn learning keeps head rows as 8-byte words")
        self.model = model
        self.table = table
        self.draft_tokens = draft_tokens
        self.input_ids = input_ids
        self.anchor_idx = anchor_idx
        self.sample_idx_mapping = sample_idx_mapping
        self.sample_pos = sample_pos
        self.partial_values = partial_values
        self.partial_indices = partial_indices
        self.fused_step = fused_step
        self.all_gather = all_gather
        self.max_num_reqs = int(max_num_reqs)
        self.n_spec = int(num_speculative_steps)
        self.hidden_size = int(hidden_size)
        self.words = self.hidden_size // 4
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.device = device
        self.capacity = int(capacity)
        self.chunk = self.max_num_reqs
        self.vocab_start, self.shard_vocab = model.draft_vocab_shard
        rank = int(table.shape[1])
        if draft_tokens.dtype != torch.int64:
            raise TypeError("turn learning reads int64 drafts")

        slots, cap, n = self.max_num_reqs, self.capacity, self.n_spec
        self.ring_hidden = torch.zeros(
            (slots, cap, n, self.hidden_size), dtype=torch.bfloat16, device=device
        )
        self._ring_words = self.ring_hidden.view(torch.int64)
        self.ring_meta = torch.full((slots, cap, _META), -1, dtype=torch.int64, device=device)
        self.ring_drafts = torch.zeros((slots, cap, n), dtype=torch.int32, device=device)
        self.ring_count = torch.zeros(slots, dtype=torch.int64, device=device)
        self.host_count = np.zeros(slots, dtype=np.int64)

        self.labels = torch.full((slots, cap, n), -1, dtype=torch.int32, device=device)
        self.ids = torch.zeros((slots, cap, n + 1), dtype=torch.int32, device=device)
        self.step_embed = torch.empty((self.chunk, rank), dtype=torch.bfloat16, device=device)
        self.conf_embed = torch.empty((self.chunk, rank), dtype=torch.bfloat16, device=device)
        self.pairs = torch.empty((self.chunk, 4), dtype=torch.float32, device=device)
        self.replayed = torch.zeros_like(draft_tokens)
        self.forced = torch.zeros_like(draft_tokens)
        # Teacher forcing reads ids[row, i] of a chunk as a flat run of
        # (n + 1)-wide rows: index i of every row, for each position.
        base = torch.arange(self.chunk, dtype=torch.int64, device=device) * (n + 1)
        self.ids_index = [base + i for i in range(n + 1)]
        self.stats = torch.zeros((slots, _STATS), dtype=torch.int64, device=device)

        self.turns = 0
        self.boot = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        self.dump_dir = dump_dir if tp_rank == 0 else None
        # Rank 0's turns awaiting their log line: (start, end, [(turn, stats)]).
        self._log_queue: list[tuple[Any, Any, list[tuple[int, torch.Tensor]]]] = []
        self._pending: list[tuple[int, torch.Tensor]] = []
        self._writer: queue.Queue | None = None
        if self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)
            self._writer = queue.Queue(maxsize=8)
            threading.Thread(target=self._write_loop, daemon=True).start()
        logger.info(
            "DSpark turn learning (observe): %d draft steps kept per request, "
            "%.1f MiB of rows a rank%s",
            cap,
            self.ring_hidden.numel() * 2 / (1 << 20),
            f", dumping turns to {self.dump_dir}" if self.dump_dir else "",
        )

    # -- declarations ---------------------------------------------------------

    def calls(self) -> list:
        from vllm.v1.worker.gpu.sample import tilelang_sampling

        n = self.n_spec
        return [
            _call(capture_kernel, n, self.words),
            _call(labels_kernel, n),
            _call(reset_kernel),
            _call(stats_kernel, n),
            *tilelang_sampling.markov_embed_calls(
                torch.int32,
                torch.int64,
                self.draft_tokens.dtype,
                self.tp_size,
                int(self.table.shape[1]),
            ),
        ]

    # -- capture (inside the draft graph) ---------------------------------------

    def capture(self, num_reqs: int, head_hidden: torch.Tensor, active: int) -> None:
        rows = num_reqs * self.n_spec
        if rows == 0:
            return
        _kernel(capture_kernel, self.n_spec, self.words)(
            head_hidden[:rows].view(torch.int64),
            self.sample_idx_mapping,
            self.sample_pos,
            self.draft_tokens,
            self.input_ids,
            self.anchor_idx,
            self._ring_words,
            self.ring_meta,
            self.ring_drafts,
            self.ring_count,
            int(active),
        )

    def add_request(self, slot: int) -> None:
        """A request takes ``slot``: its ring starts empty, whatever a warmup or
        an earlier request left there."""
        slots = async_tensor_h2d([slot], dtype=torch.int64, device=self.device)
        _kernel(reset_kernel)(slots, self.ring_count)
        self.host_count[slot] = 0

    def note_drafts(self, slots: np.ndarray) -> None:
        """The host's count of each request's draft steps (an upper bound of the
        ring's, which the replay reads only to size its launches)."""
        self.host_count[slots] += 1

    # -- end of turn ----------------------------------------------------------

    def finish(
        self,
        finished_slots: list[int],
        preempted_slots: list[int],
        all_token_ids: torch.Tensor,
        total_len: torch.Tensor,
    ) -> None:
        """Replay each finished request's turn, then clear its ring and the
        preempted requests' rings. Every rank calls this in the same step with the
        same slots in the same order."""
        self._flush_logs()
        start = end = None
        if finished_slots and self.tp_rank == 0:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        if finished_slots:
            finished = async_tensor_h2d(
                finished_slots, dtype=torch.int64, device=self.device
            )
            _kernel(labels_kernel, self.n_spec)(
                finished,
                all_token_ids,
                total_len,
                self.ring_meta,
                self.ring_count,
                self.labels[: len(finished_slots)],
                self.ids[: len(finished_slots)],
            )
            for f, slot in enumerate(finished_slots):
                kept = int(min(self.host_count[slot], self.capacity))
                if kept:
                    self._replay(f, slot, kept)
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._log_queue.append((start, end, self._pending))
            self._pending = []
        cleared = list(finished_slots) + list(preempted_slots)
        if cleared:
            slots = async_tensor_h2d(cleared, dtype=torch.int64, device=self.device)
            _kernel(reset_kernel)(slots, self.ring_count)
            self.host_count[cleared] = 0

    def _replay(self, f: int, slot: int, kept: int) -> None:
        n_spec = self.n_spec
        for start in range(0, kept, self.chunk):
            rows = min(self.chunk, kept - start)
            hidden = self.ring_hidden[slot, start : start + rows].reshape(
                rows * n_spec, self.hidden_size
            )
            base = self.model.compute_draft_logits_local(hidden)
            base = base.view(rows, n_spec, base.shape[-1])
            ids = self.ids[f, start : start + rows].reshape(-1)
            self._chain(base, rows, ids, self.replayed, teacher=False)
            self._chain(base, rows, ids, self.forced, teacher=True)
            _kernel(stats_kernel, n_spec)(
                self.labels[f, start : start + rows],
                self.ring_drafts[slot, start : start + rows],
                self.ring_meta[slot, start : start + rows],
                self.replayed,
                self.forced,
                self.stats[f],
                1 if start == 0 else 0,
            )
        self.turns += 1
        if self.tp_rank == 0:
            host = self.stats[f].to("cpu", non_blocking=True)
            self._pending.append((self.turns, host))
            if self._writer is not None:
                self._dump(f, slot, kept, host)

    def _chain(
        self,
        base: torch.Tensor,
        rows: int,
        ids: torch.Tensor,
        out: torch.Tensor,
        *,
        teacher: bool,
    ) -> None:
        """The draft step's Markov chain over ``rows`` kept steps, with its own
        declared kernels: the anchor's row of W1, then at each position the shard
        pairs of base logits plus Markov bias, their first argmax into
        ``out[:, i]``, and the next position's row of W1: the argmax's (replay) or
        the label's (teacher forcing)."""
        from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
            greedy_markov_shard_pair,
            markov_anchor_embed,
            select_first_argmax,
        )

        step = self.step_embed[:rows]
        conf = self.conf_embed[:rows]
        pairs = self.pairs[:rows]
        markov_anchor_embed(ids, self.ids_index[0][:rows], self.table, step, conf)
        for i in range(self.n_spec):
            if self.fused_step(self.tp_size, i, rows):
                self.model.markov_greedy_pairs_local(step, base[:, i], pairs, top2=False)
            else:
                bias = self.model.markov_bias_local(step)
                greedy_markov_shard_pair(
                    base[:, i, : self.shard_vocab],
                    bias[:, : self.shard_vocab],
                    self.vocab_start,
                    pairs,
                    self.partial_values,
                    self.partial_indices,
                    top2=False,
                )
            gathered = self.all_gather(pairs)
            last = i + 1 == self.n_spec
            if teacher or last:
                select_first_argmax(gathered, out[:rows, i], pair_width=4)
                if teacher and not last:
                    markov_anchor_embed(
                        ids, self.ids_index[i + 1][:rows], self.table, step, conf
                    )
            else:
                select_first_argmax(
                    gathered,
                    out[:rows, i],
                    pair_width=4,
                    embed=(self.table, step, conf),
                )

    # -- logging and dumps (rank 0) ------------------------------------------

    def _flush_logs(self) -> None:
        while self._log_queue and self._log_queue[0][1].query():
            start, end, turns = self._log_queue.pop(0)
            elapsed = start.elapsed_time(end)
            for turn, host in turns:
                stats = TurnStats.as_dict(host.tolist())
                logger.info(
                    "DSpark turn learning: turn %d %s (end of turn %.2f ms for %d turns)",
                    turn,
                    stats,
                    elapsed,
                    len(turns),
                )
                if stats["replay_mismatches"] or (
                    stats["draft_accepted"] != stats["forced_accepted_active"]
                ):
                    logger.warning(
                        "DSpark turn learning: turn %d does not replay its drafts: %s",
                        turn,
                        stats,
                    )

    def _dump(self, f: int, slot: int, kept: int, stats: torch.Tensor) -> None:
        assert self._writer is not None
        record = {
            "boot": self.boot,
            "turn": self.turns,
            "hidden": self.ring_hidden[slot, :kept].to("cpu", non_blocking=True),
            "ids": self.ids[f, :kept].to("cpu", non_blocking=True),
            "labels": self.labels[f, :kept].to("cpu", non_blocking=True),
            "drafts": self.ring_drafts[slot, :kept].to("cpu", non_blocking=True),
            "meta": self.ring_meta[slot, :kept].to("cpu", non_blocking=True),
            "stats": stats,
        }
        done = torch.cuda.Event()
        done.record()
        try:
            self._writer.put_nowait((done, record))
        except queue.Full:
            logger.warning("DSpark turn learning: dump queue full, turn %d not written", self.turns)

    def _write_loop(self) -> None:
        assert self._writer is not None and self.dump_dir is not None
        while True:
            done, record = self._writer.get()
            done.synchronize()
            path = os.path.join(
                self.dump_dir, f"{record['boot']}-turn-{record['turn']:07d}.pt"
            )
            torch.save(record, path + ".tmp")
            os.replace(path + ".tmp", path)
