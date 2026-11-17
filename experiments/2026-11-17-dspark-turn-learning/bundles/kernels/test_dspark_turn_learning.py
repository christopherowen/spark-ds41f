# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The turn learner's TileLang kernels against plain Python references: the
in-graph capture (wrap-around, padding rows), the end-of-turn labels, the ring
reset and the turn counters. The replay itself runs the draft step's declared
kernels and is checked in serving (``replay_mismatches`` must stay 0)."""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark import turn_learning as tl

DEV = "cuda"
N_SPEC = 5
HIDDEN = 5120


def _ring(slots: int, cap: int, n_spec: int = N_SPEC, hidden: int = HIDDEN):
    rows = torch.zeros((slots, cap, n_spec, hidden), dtype=torch.bfloat16, device=DEV)
    return {
        "hidden": rows,
        "words": rows.view(torch.int64),
        "meta": torch.full((slots, cap, 3), -1, dtype=torch.int64, device=DEV),
        "drafts": torch.zeros((slots, cap, n_spec), dtype=torch.int32, device=DEV),
        "count": torch.zeros(slots, dtype=torch.int64, device=DEV),
    }


def _capture(ring, hidden, sample_map, sample_pos, drafts, active, input_ids, anchor_idx):
    words = hidden.shape[1] // 4
    tl._kernel(tl.capture_kernel, N_SPEC, words)(
        hidden.view(torch.int64),
        sample_map,
        sample_pos,
        drafts,
        input_ids,
        anchor_idx,
        ring["words"],
        ring["meta"],
        ring["drafts"],
        ring["count"],
        active,
    )


def test_capture_keeps_rows_wraps_and_skips_padding():
    torch.manual_seed(0)
    slots, cap, max_reqs = 4, 3, 4
    ring = _ring(slots, cap)
    ref_hidden = torch.zeros_like(ring["hidden"])
    ref_meta = ring["meta"].clone()
    ref_drafts = ring["drafts"].clone()
    ref_count = [0] * slots
    # Row 1 is padding of the captured size (-1); slot 3 is never drafted.
    layouts = [[2, -1, 0], [0, 2, -1], [2, 0, 1], [1, -1, 2], [2, 0, -1]]
    for step, layout in enumerate(layouts):
        num_reqs = len(layout)
        hidden = torch.randn(
            (num_reqs * N_SPEC + 7, HIDDEN), dtype=torch.bfloat16, device=DEV
        )
        sample_map = torch.full((max_reqs * N_SPEC,), -1, dtype=torch.int32, device=DEV)
        sample_pos = torch.zeros(max_reqs * N_SPEC, dtype=torch.int64, device=DEV)
        drafts = torch.randint(
            0, 129280, (max_reqs, N_SPEC), dtype=torch.int64, device=DEV
        )
        active = 5 if step % 2 == 0 else 3
        # The draft step's anchor of row r is input_ids[anchor_idx[r]] (r * N_SPEC).
        anchor_idx = torch.arange(max_reqs, dtype=torch.int64, device=DEV) * N_SPEC
        input_ids = torch.randint(0, 129280, (max_reqs * N_SPEC,), dtype=torch.int32, device=DEV)
        for r, slot in enumerate(layout):
            anchor = 100 * step + 10 * r + 7
            for i in range(N_SPEC):
                sample_map[r * N_SPEC + i] = slot
                sample_pos[r * N_SPEC + i] = anchor + 1 + i
        _capture(
            ring, hidden[: num_reqs * N_SPEC], sample_map, sample_pos, drafts, active,
            input_ids, anchor_idx,
        )
        for r, slot in enumerate(layout):
            if slot < 0:
                continue
            entry = ref_count[slot] % cap
            ref_hidden[slot, entry] = hidden[r * N_SPEC : (r + 1) * N_SPEC]
            ref_meta[slot, entry, 0] = 100 * step + 10 * r + 7
            ref_meta[slot, entry, 1] = active
            ref_meta[slot, entry, 2] = int(input_ids[r * N_SPEC])
            ref_drafts[slot, entry] = drafts[r].to(torch.int32)
            ref_count[slot] += 1
    torch.cuda.synchronize()
    assert ring["count"].tolist() == ref_count
    assert torch.equal(ring["hidden"].view(torch.int16), ref_hidden.view(torch.int16))
    assert torch.equal(ring["meta"], ref_meta)
    assert torch.equal(ring["drafts"], ref_drafts)


def _labels_reference(finished, tokens, total_len, meta, count, cap):
    labels = torch.full((len(finished), cap, N_SPEC), -1, dtype=torch.int32)
    ids = torch.zeros((len(finished), cap, N_SPEC + 1), dtype=torch.int32)
    for f, slot in enumerate(finished):
        kept_entries = min(int(count[slot]), cap)
        total = int(total_len[slot])
        for k in range(cap):
            pos = int(meta[slot, k, 0])
            kept = (
                k < kept_entries
                and 0 <= pos < total
                and int(tokens[slot, pos]) == int(meta[slot, k, 2])
            )
            if not kept:
                labels[f, k] = -2
                continue
            ids[f, k, 0] = int(tokens[slot, pos])
            for i in range(N_SPEC):
                q = pos + 1 + i
                if q < total:
                    labels[f, k, i] = int(tokens[slot, q])
                    ids[f, k, i + 1] = int(tokens[slot, q])
    return labels, ids


def test_labels_read_committed_tokens_and_mask_the_rest():
    torch.manual_seed(1)
    slots, cap, max_len = 4, 6, 64
    wide = torch.randint(0, 129280, (slots, max_len + 9), dtype=torch.int32, device=DEV)
    tokens = wide[:, :max_len]  # a strided view, as the request state's rows
    total_len = torch.tensor([40, 20, 64, 10], dtype=torch.int32, device=DEV)
    meta = torch.randint(-1, max_len, (slots, cap, 3), dtype=torch.int64, device=DEV)
    meta[2, 0, 0] = 61  # the turn ends inside this block
    # Each entry drafted from the committed anchor token, except one drafted from
    # another context, which the labels must not hold.
    rows = torch.arange(slots, device=DEV)[:, None].expand(slots, cap)
    meta[:, :, 2] = tokens[rows, meta[:, :, 0].clamp(0, max_len - 1)].to(torch.int64)
    meta[1, 1, 2] += 1
    count = torch.tensor([0, 3, 8, 6], dtype=torch.int64, device=DEV)  # slot 2 wrapped
    finished_slots = [2, 1, 3, 0]
    finished = torch.tensor(finished_slots, dtype=torch.int64, device=DEV)
    labels = torch.empty((len(finished_slots), cap, N_SPEC), dtype=torch.int32, device=DEV)
    ids = torch.empty((len(finished_slots), cap, N_SPEC + 1), dtype=torch.int32, device=DEV)
    tl._kernel(tl.labels_kernel, N_SPEC)(
        finished, tokens, total_len, meta, count, labels, ids
    )
    torch.cuda.synchronize()
    ref_labels, ref_ids = _labels_reference(
        finished_slots, tokens.cpu(), total_len.cpu(), meta.cpu(), count.cpu(), cap
    )
    assert torch.equal(labels.cpu(), ref_labels)
    assert torch.equal(ids.cpu(), ref_ids)
    # Slot 2's block at 61 has labels at 62 and 63 only.
    assert labels[0, 0].tolist()[:2] == tokens[2, 62:64].tolist()
    assert labels[0, 0].tolist()[2:] == [-1, -1, -1]


def test_reset_clears_only_the_given_rings():
    count = torch.tensor([3, 9, 4, 1, 7], dtype=torch.int64, device=DEV)
    slots = torch.tensor([3, 1], dtype=torch.int64, device=DEV)
    tl._kernel(tl.reset_kernel)(slots, count)
    torch.cuda.synchronize()
    assert count.tolist() == [3, 0, 4, 0, 7]


def _stats_reference(labels, drafts, meta, replay, forced, start):
    acc = list(start)
    for r in range(labels.shape[0]):
        held = int(labels[r, 0]) != -2
        active = int(meta[r, 1]) if held else 0
        labeled = run_forced = run_draft = mismatches = 0
        live_forced = live_draft = True
        for i in range(N_SPEC):
            y = int(labels[r, i])
            has = y >= 0
            drafted = i < active
            labeled += has
            if live_forced:
                if has and int(forced[r, i]) == y:
                    run_forced += 1
                else:
                    live_forced = False
            if live_draft:
                if has and drafted and int(drafts[r, i]) == y:
                    run_draft += 1
                else:
                    live_draft = False
            if held and drafted and int(replay[r, i]) != int(drafts[r, i]):
                mismatches += 1
        acc[0] += labeled > 0
        acc[1] += labeled
        acc[2] += run_forced
        acc[3] += run_draft
        acc[4] += min(run_forced, active)
        acc[5] += mismatches
        acc[6] += active
        acc[7] += 1
        acc[8] += 0 if held else 1
    return acc


@pytest.mark.parametrize("first", [1, 0])
def test_stats_count_accepted_prefixes_and_replay_mismatches(first):
    torch.manual_seed(2 + first)
    rows, pred_rows = 7, 16
    vocab = 6  # a small vocabulary makes matches and runs common
    labels = torch.randint(0, vocab, (rows, N_SPEC), dtype=torch.int32)
    labels[1, 3:] = -1
    labels[4, :] = -1
    labels[6, :] = -2  # an entry the ring does not hold: its stale replay is not counted
    drafts = torch.randint(0, vocab, (rows, N_SPEC), dtype=torch.int32)
    meta = torch.zeros((rows, 3), dtype=torch.int64)
    meta[:, 1] = torch.tensor([5, 3, 5, 1, 5, 2, 4])
    replay = torch.zeros((pred_rows, N_SPEC), dtype=torch.int64)
    replay[:rows] = drafts.to(torch.int64)
    replay[2, 1] += 1  # one replay mismatch inside the drafted span
    replay[3, 4] += 1  # past the drafted span: not counted
    replay[6, 0] += 1  # in an entry the ring does not hold: not counted
    forced = torch.randint(0, vocab, (pred_rows, N_SPEC), dtype=torch.int64)
    forced[0] = labels[0].to(torch.int64)  # a fully accepted block
    out = torch.tensor([11, 22, 33, 44, 55, 66, 77, 88, 99], dtype=torch.int64)
    start = [0] * 9 if first else out.tolist()
    expected = _stats_reference(labels, drafts, meta, replay, forced, start)
    dev = [t.to(DEV) for t in (labels, drafts, meta, replay, forced, out)]
    tl._kernel(tl.stats_kernel, N_SPEC)(*dev, first)
    torch.cuda.synchronize()
    assert dev[-1].cpu().tolist() == expected
    assert expected[5] - start[5] == 1


def test_mode_and_capacity_are_validated(monkeypatch):
    monkeypatch.setenv("VLLM_DSPARK_TURN_LEARNING", "observe")
    assert tl.turn_learning_mode() == "observe"
    monkeypatch.setenv("VLLM_DSPARK_TURN_LEARNING", "train-everything")
    with pytest.raises(ValueError):
        tl.turn_learning_mode()
    monkeypatch.delenv("VLLM_DSPARK_TURN_LEARNING")
    assert tl.turn_learning_mode() == "off"
    monkeypatch.setenv("VLLM_DSPARK_TURN_LEARNING_CAPACITY", "0")
    with pytest.raises(ValueError):
        tl.turn_learning_capacity()
