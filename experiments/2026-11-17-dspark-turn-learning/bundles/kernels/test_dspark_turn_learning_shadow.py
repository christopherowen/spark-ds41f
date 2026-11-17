# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shadow turn learner's TileLang kernels against plain Python references: the
in-graph capture of the step's block maxima (unfused parts, the fused first
position's tiles, undrafted positions, padding rows, wrap-around), and the end of
turn across simulated ranks (each rank's candidates and W2 rows, the exchanged
summaries, both tables' teacher-forced score, the drafts' check, the gradient
summed over ranks, the per-row AdaGrad step), read from the stage the step fills
(each kept entry's labels and ring entry, against ``labels_kernel`` and the ring).

The end-of-turn data is dyadic (W1, W2 and the step's values are small multiples
of powers of two), so every dot product and BF16 rounding is exact whatever the
summation order and the rounded logits match the reference to the bit; only the
exponentials and the update are compared with a tolerance."""

import math

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark import turn_learning as tl
from vllm.v1.worker.gpu.spec_decode.dspark import turn_learning_shadow as sh

DEV = "cuda"
N_SPEC = 5
PARTS = 16
DIM = 256
INT32_MAX = 2147483647


def _first_max(pairs):
    best, best_index = -math.inf, -1
    for value, index in pairs:
        if index < 0:
            continue
        if best_index < 0 or value > best or (value == best and index < best_index):
            best, best_index = value, index
    return best, best_index


# --- capture ---------------------------------------------------------------------


def test_capture_keeps_parts_tiles_and_skips_padding():
    torch.manual_seed(0)
    slots, cap, max_reqs = 4, 3, 4
    num_tiles, cols = 505, 64
    per_block = sh.GREEDY_BLOCK // cols
    vocab_start = 32320
    ring_values = torch.zeros((slots, cap, N_SPEC, PARTS), dtype=torch.float32, device=DEV)
    ring_index = torch.full((slots, cap, N_SPEC, PARTS), -7, dtype=torch.int32, device=DEV)
    ring_meta = torch.full((slots, cap, tl._META), -1, dtype=torch.int64, device=DEV)
    ring_drafts = torch.zeros((slots, cap, N_SPEC), dtype=torch.int32, device=DEV)
    ring_count = torch.zeros(slots, dtype=torch.int64, device=DEV)
    ref_values = ring_values.clone().cpu()
    ref_index = ring_index.clone().cpu()
    ref_meta = ring_meta.clone().cpu()
    ref_drafts = ring_drafts.clone().cpu()
    ref_count = [0] * slots
    kernel = tl._kernel(sh.shadow_capture_kernel, N_SPEC, PARTS)
    # (slot layout, drafted positions, fused first position)
    steps = [
        ([2, -1, 0], 5, True),
        ([0, 2, -1], 3, False),
        ([2, 0, 1], 5, True),
        ([1, -1, 2], 1, True),
        ([2, 0, -1], 4, False),
    ]
    for step, (layout, active, fused) in enumerate(steps):
        rows = len(layout)
        # Few distinct values, so ties are common; one part a request has no column.
        part_values = (torch.randint(0, 4, (N_SPEC, max_reqs * PARTS), device=DEV) / 4).float()
        part_indices = torch.randint(0, 2048, (N_SPEC, max_reqs * PARTS), dtype=torch.int32, device=DEV)
        part_indices += (torch.arange(PARTS, device=DEV, dtype=torch.int32) * 2048).repeat(max_reqs)
        part_indices[:, 3] = INT32_MAX
        tile_values = (torch.randint(0, 3, (max_reqs * num_tiles,), device=DEV) / 2).float()
        tile_indices = (
            torch.arange(num_tiles, device=DEV, dtype=torch.int32).repeat(max_reqs) * cols
            + torch.randint(0, cols, (max_reqs * num_tiles,), dtype=torch.int32, device=DEV)
        )
        tile_indices[num_tiles + 40] = INT32_MAX
        sample_map = torch.full((rows * N_SPEC,), -1, dtype=torch.int32, device=DEV)
        sample_pos = torch.zeros(rows * N_SPEC, dtype=torch.int64, device=DEV)
        drafts = torch.randint(0, 129280, (max_reqs, N_SPEC), dtype=torch.int64, device=DEV)
        input_ids = torch.randint(0, 129280, (64,), dtype=torch.int32, device=DEV)
        anchor_idx = torch.randint(0, 64, (max_reqs,), dtype=torch.int64, device=DEV)
        for r, slot in enumerate(layout):
            for i in range(N_SPEC):
                sample_map[r * N_SPEC + i] = slot
                sample_pos[r * N_SPEC + i] = 100 * step + 10 * r + 8 + i
        kernel(
            part_values, part_indices, tile_values, tile_indices, sample_map, sample_pos,
            drafts, input_ids, anchor_idx, ring_values, ring_index, ring_meta,
            ring_drafts, ring_count, active, 1 if fused else 0, num_tiles, per_block,
            vocab_start,
        )
        pv, pi = part_values.cpu(), part_indices.cpu()
        tv, ti = tile_values.cpu(), tile_indices.cpu()
        for r, slot in enumerate(layout):
            if slot < 0:
                continue
            entry = ref_count[slot] % cap
            for i in range(N_SPEC):
                for b in range(PARTS):
                    if i >= active:
                        best, index = -math.inf, -1
                    elif i == 0 and fused:
                        pairs = []
                        for t in range(b * per_block, (b + 1) * per_block):
                            if t < num_tiles:
                                x = int(ti[r * num_tiles + t])
                                pairs.append((float(tv[r * num_tiles + t]), -1 if x == INT32_MAX else x))
                        best, index = _first_max(pairs)
                    else:
                        x = int(pi[i, r * PARTS + b])
                        best, index = _first_max([(float(pv[i, r * PARTS + b]), -1 if x == INT32_MAX else x)])
                    ref_values[slot, entry, i, b] = best
                    ref_index[slot, entry, i, b] = index + vocab_start if index >= 0 else -1
                ref_drafts[slot, entry, i] = int(drafts[r, i]) if i < active - 1 else -1
            ref_meta[slot, entry, 0] = 100 * step + 10 * r + 7
            ref_meta[slot, entry, 1] = active
            ref_meta[slot, entry, 2] = int(input_ids[int(anchor_idx[r])])
            ref_count[slot] += 1
    torch.cuda.synchronize()
    assert ring_count.tolist() == ref_count
    assert torch.equal(ring_values.cpu(), ref_values)
    assert torch.equal(ring_index.cpu(), ref_index)
    assert torch.equal(ring_meta.cpu(), ref_meta)
    assert torch.equal(ring_drafts.cpu(), ref_drafts)


# --- end of turn -----------------------------------------------------------------


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def _turn_data(ranks: int, seed: int):
    """A finished-turn layout over ``ranks`` simulated ranks: per rank, the step's
    values and candidates of each kept entry; shared, the labels, inputs, drafted
    counts and the drafts the ring holds."""
    g = torch.Generator().manual_seed(seed)
    vocab = ranks * PARTS * 8
    per_rank = vocab // ranks
    table = (torch.randint(-8, 9, (vocab, DIM), generator=g) / 16).to(torch.bfloat16)
    master = table.float() + torch.randint(-2, 3, (vocab, DIM), generator=g) / 64
    master[: vocab // 2] = table[: vocab // 2].float()  # half the rows unchanged
    w2 = (torch.randint(-4, 5, (vocab, DIM), generator=g) / 8).to(torch.bfloat16)
    slots, cap = 3, 4
    finished = [2, 0]
    kept = [3, 4]
    entries = [(f, k) for f, n in enumerate(kept) for k in range(n)]
    values = torch.full((ranks, slots, cap, N_SPEC, PARTS), -math.inf)
    index = torch.full((ranks, slots, cap, N_SPEC, PARTS), -1, dtype=torch.int32)
    meta = torch.full((slots, cap, tl._META), -1, dtype=torch.int64)
    drafts = torch.full((slots, cap, N_SPEC), -1, dtype=torch.int32)
    labels = torch.full((len(finished), cap, N_SPEC), -1, dtype=torch.int32)
    ids = torch.zeros((len(finished), cap, N_SPEC + 1), dtype=torch.int32)
    for f, k in entries:
        slot = finished[f]
        active = int(torch.randint(2, N_SPEC + 1, (1,), generator=g))
        meta[slot, k, 1] = active
        for i in range(active):
            for r in range(ranks):
                for b in range(PARTS):
                    index[r, slot, k, i, b] = r * per_rank + b * 8 + int(torch.randint(0, 8, (1,), generator=g))
                    # A few distinct dyadic values: ties across ranks and parts.
                    values[r, slot, k, i, b] = float(torch.randint(16, 24, (1,), generator=g)) / 2
        # The global first maximum of each position: the step's draft.
        for i in range(active):
            pairs = [
                (float(values[r, slot, k, i, b]), int(index[r, slot, k, i, b]))
                for r in range(ranks)
                for b in range(PARTS)
            ]
            _, draft = _first_max(pairs)
            if i < active - 1:
                drafts[slot, k, i] = draft
            u = float(torch.rand(1, generator=g))
            if u < 0.6:
                labels[f, k, i] = draft
            elif u < 0.85:
                r, b = int(torch.randint(0, ranks, (1,), generator=g)), int(torch.randint(0, PARTS, (1,), generator=g))
                labels[f, k, i] = int(index[r, slot, k, i, b])
            else:
                labels[f, k, i] = int(torch.randint(0, vocab, (1,), generator=g))
        ids[f, k, 0] = int(torch.randint(0, vocab, (1,), generator=g))
    # The turn ends inside one block; one entry the ring does not hold; one held
    # draft that the maxima do not give.
    labels[0, 1, 3:] = -1
    labels[1, 2, :] = -2
    for f, k in entries:
        for i in range(N_SPEC):
            ids[f, k, i + 1] = labels[f, k, i] if int(labels[f, k, i]) >= 0 else 0
    slot, k = finished[1], 0
    if int(meta[slot, k, 1]) >= 2:
        drafts[slot, k, 0] = drafts[slot, k, 0] + 1
    return dict(
        ranks=ranks, vocab=vocab, table=table, master=master, w2=w2, slots=slots,
        cap=cap, finished=finished, entries=entries, values=values, index=index,
        meta=meta, drafts=drafts, labels=labels, ids=ids,
    )


def _reference(d, lr):
    """The score, the summed gradient and the updated copy, by the definitions."""
    ranks, table, master, w2 = d["ranks"], d["table"], d["master"].clone(), d["w2"]
    accum = torch.zeros(master.shape[0])
    counts, losses, grads, trains, prevs = [], [], [], [], []
    for e, (f, k) in enumerate(d["entries"]):
        slot = d["finished"][f]
        held = int(d["labels"][f, k, 0]) != -2
        drafted = int(d["meta"][slot, k, 1])
        acc = [0] * len(sh.ShadowStats.FIELDS)
        acc[0] = 1 if held else 0
        run_draft, run_loaded, run_learned, loss = True, True, True, 0.0
        for i in range(N_SPEC):
            label = int(d["labels"][f, k, i])
            here = held and i < drafted
            step_in = int(d["ids"][f, k, 0]) if i == 0 else int(d["drafts"][slot, k, max(i - 1, 0)])
            p_d = step_in if here and step_in >= 0 else 0
            p_l = int(d["ids"][f, k, i]) if here and label >= 0 else 0
            z0 = torch.full((ranks, PARTS), -math.inf)
            zl = torch.full((ranks, PARTS), -math.inf)
            for r in range(ranks):
                for c in range(PARTS):
                    x = int(d["index"][r, slot, k, i, c])
                    row = w2[x].float() if x >= 0 else torch.zeros(DIM)
                    bd = _bf16((table[p_d].float() * row).sum())
                    b0 = _bf16((table[p_l].float() * row).sum())
                    bl = _bf16((_bf16(master[p_l]) * row).sum())
                    base = d["values"][r, slot, k, i, c] - bd
                    z0[r, c] = _bf16(base + b0)
                    zl[r, c] = _bf16(base + bl)
            idx = d["index"][:, slot, k, i]
            v = d["values"][:, slot, k, i]
            pairs = lambda t: [(float(t[r, c]), int(idx[r, c])) for r in range(ranks) for c in range(PARTS)]  # noqa: E731
            _, draft = _first_max(pairs(v))
            _, best0 = _first_max(pairs(z0))
            _, bestl = _first_max(pairs(zl))
            if held and i + 1 < drafted:
                acc[8] += 1
                acc[9] += int(d["drafts"][slot, k, i]) != draft
            scored = here and label >= 0 and draft >= 0
            if run_loaded and scored and best0 == label:
                acc[4] += 1
            else:
                run_loaded = False
            if run_learned and scored and bestl == label:
                acc[5] += 1
            else:
                run_learned = False
            seen = scored and run_draft
            train = False
            lse = 0.0
            if seen:
                acc[1] += 1
                acc[3] += draft == label
                if bool((idx == label).any()):
                    acc[2] += 1
                    train = True
                    lse = float(torch.logsumexp(zl[idx >= 0], dim=0))
                    loss += lse - float(zl[idx == label][0])
            run_draft = run_draft and seen and draft == label
            g = torch.zeros(DIM)
            if train:
                for r in range(ranks):
                    for c in range(PARTS):
                        x = int(idx[r, c])
                        if x >= 0:
                            coef = math.exp(float(zl[r, c]) - lse) - (1.0 if x == label else 0.0)
                            g += coef * w2[x].float()
            grads.append(g)
            trains.append(train)
            prevs.append(p_l if seen else 0)
        acc[6] = int(acc[5] > acc[4])
        acc[7] = int(acc[5] < acc[4])
        counts.append(acc)
        losses.append(loss)
    for pos, (train, p) in enumerate(zip(trains, prevs)):
        if not train or any(t and q == p for t, q in zip(trains[:pos], prevs[:pos])):
            continue
        g = sum(grads[j] for j in range(pos, len(trains)) if trains[j] and prevs[j] == p)
        accum[p] += float((g * g).sum())
        master[p] -= lr / (math.sqrt(float(accum[p])) + 1e-6) * g
    return counts, losses, torch.stack(grads), trains, prevs, master, accum


def _stage(d, first, stage_len):
    """The call's entries at stage rows ``first + e``, as ``shadow_stage_kernel``
    leaves them: shared labels, inputs, meta and drafts; each rank's maxima."""
    ranks = d["ranks"]
    labels = torch.full((stage_len, N_SPEC), -2, dtype=torch.int32)
    ids = torch.zeros((stage_len, N_SPEC + 1), dtype=torch.int32)
    meta = torch.full((stage_len, tl._META), -1, dtype=torch.int64)
    drafts = torch.full((stage_len, N_SPEC), -1, dtype=torch.int32)
    values = torch.full((ranks, stage_len, N_SPEC, PARTS), -math.inf)
    index = torch.full((ranks, stage_len, N_SPEC, PARTS), -1, dtype=torch.int32)
    for e, (f, k) in enumerate(d["entries"]):
        slot, s = d["finished"][f], first + e
        labels[s], ids[s] = d["labels"][f, k], d["ids"][f, k]
        meta[s], drafts[s] = d["meta"][slot, k], d["drafts"][slot, k]
        values[:, s], index[:, s] = d["values"][:, slot, k], d["index"][:, slot, k]
    return labels, ids, meta, drafts, values, index


@pytest.mark.parametrize("ranks", [4, 3])
def test_end_of_turn_matches_reference_across_ranks(ranks):
    lr = 0.3
    d = _turn_data(ranks, seed=10 + ranks)
    to = lambda t: t.to(DEV)  # noqa: E731
    entries = d["entries"]
    count = len(entries)
    rows = count * N_SPEC
    first = 3  # the call's rows sit inside a larger stage
    labels, ids, meta, drafts, st_values, st_index = (
        to(t) for t in _stage(d, first, count + 5)
    )
    table, w2 = to(d["table"]), to(d["w2"])
    master = to(d["master"].clone())
    accum = torch.zeros(d["vocab"], dtype=torch.float32, device=DEV)
    summaries, zs, decs = [], [], []
    local = tl._kernel(sh.shadow_local_kernel, N_SPEC, PARTS, DIM)
    for r in range(ranks):
        values = st_values[r].contiguous()
        index = st_index[r].contiguous()
        cand = index[first : first + count].reshape(-1).cpu()
        dec = torch.where(
            (cand >= 0)[:, None], d["w2"][cand.clamp_min(0)], torch.zeros((), dtype=torch.bfloat16)
        )
        dec = to(dec.contiguous())
        z = torch.empty((count, N_SPEC, PARTS), dtype=torch.float32, device=DEV)
        summary = torch.empty((count, N_SPEC, sh._LOCAL), dtype=torch.float32, device=DEV)
        local(labels, ids, values, index, meta, drafts, dec, master, table, z, summary, first)
        summaries.append(summary.view(rows, sh._LOCAL))
        zs.append(z)
        decs.append((dec, index))
    gathered = torch.stack(summaries).contiguous()  # the all-gather's rank-major layout
    combine = tl._kernel(sh.shadow_combine_kernel, N_SPEC, PARTS, DIM, ranks)
    grad_sum = torch.zeros((rows, DIM), dtype=torch.float32, device=DEV)
    for r in range(ranks):
        dec, index = decs[r]
        grad = torch.empty((rows, DIM), dtype=torch.float32, device=DEV)
        train = torch.empty(rows, dtype=torch.int32, device=DEV)
        prev = torch.empty(rows, dtype=torch.int32, device=DEV)
        counts = torch.empty((count, sh._COUNTS), dtype=torch.int32, device=DEV)
        loss = torch.empty(count, dtype=torch.float32, device=DEV)
        combine(labels, ids, index, meta, drafts, dec, zs[r], gathered, grad, train, prev,
                counts, loss, first)
        grad_sum += grad  # the all-reduce, in rank order
        if r == 0:
            first_rank = (train.clone(), prev.clone(), counts.clone(), loss.clone())
        else:  # every rank derives the same score and inputs
            assert torch.equal(train, first_rank[0]) and torch.equal(prev, first_rank[1])
            assert torch.equal(counts, first_rank[2])
    train, prev, counts, loss = first_rank
    tl._kernel(sh.shadow_update_kernel, DIM, sh._update_words(rows))(
        train, prev, grad_sum, master, accum, lr
    )
    call = torch.full((sh._COUNTS + 1,), 7.0, dtype=torch.float64, device=DEV)
    boot = torch.full((sh._COUNTS + 1,), 5.0, dtype=torch.float64, device=DEV)
    totals = tl._kernel(sh.shadow_totals_kernel)
    totals(counts, loss, call, boot, 1)  # a call's first chunk replaces the call's totals
    torch.cuda.synchronize()

    ref_counts, ref_loss, ref_grads, ref_train, ref_prev, ref_master, ref_accum = _reference(d, lr)
    assert counts.cpu().tolist() == ref_counts
    assert train.cpu().bool().tolist() == ref_train
    assert prev.cpu().tolist() == ref_prev
    torch.testing.assert_close(loss.cpu(), torch.tensor(ref_loss), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad_sum.cpu(), ref_grads, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(accum.cpu(), ref_accum, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(master.cpu(), ref_master, rtol=1e-5, atol=1e-6)
    ref_totals = torch.tensor(
        torch.tensor(ref_counts).sum(dim=0).double().tolist() + [sum(ref_loss)], dtype=torch.float64
    )
    torch.testing.assert_close(call.cpu(), ref_totals, rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(boot.cpu(), ref_totals + 5.0, rtol=1e-6, atol=1e-5)
    totals(counts, loss, call, boot, 0)  # a later chunk adds
    torch.cuda.synchronize()
    torch.testing.assert_close(call.cpu(), 2 * ref_totals, rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(boot.cpu(), 2 * ref_totals + 5.0, rtol=1e-6, atol=1e-5)
    # The data exercises what it is meant to.
    totals_by_name = sh.ShadowStats.as_dict(torch.tensor(ref_counts).sum(dim=0).tolist())
    assert totals_by_name["trained"] > 0 and totals_by_name["draft_mismatches"] == 1
    assert totals_by_name["seen"] > totals_by_name["accepted"] > 0
    assert totals_by_name["tf_loaded"] > 0 and totals_by_name["tf_loaded"] != totals_by_name["tf_learned"]


def test_stage_matches_labels_kernel_and_ring():
    """The stage holds each kept entry's labels and inputs as ``labels_kernel``
    gives them, and a copy of its ring entry, at ``first + e`` in turn order."""
    g = torch.Generator().manual_seed(7)
    slots, cap, max_len, vocab = 6, 8, 64, 500
    tokens = torch.randint(0, vocab, (slots, max_len), generator=g, dtype=torch.int32)
    total_len = torch.randint(20, max_len, (slots,), generator=g, dtype=torch.int32)
    ring_values = torch.randn((slots, cap, N_SPEC, PARTS), generator=g)
    ring_index = torch.randint(-1, vocab, (slots, cap, N_SPEC, PARTS), generator=g, dtype=torch.int32)
    ring_drafts = torch.randint(-1, vocab, (slots, cap, N_SPEC), generator=g, dtype=torch.int32)
    ring_meta = torch.zeros((slots, cap, tl._META), dtype=torch.int64)
    ring_count = torch.tensor([8, 3, 11, 5, 0, 7], dtype=torch.int64)  # 11: wrapped
    for s in range(slots):
        for k in range(cap):
            pos = int(torch.randint(0, int(total_len[s]), (1,), generator=g))
            ring_meta[s, k] = torch.tensor([pos, int(torch.randint(1, N_SPEC + 1, (1,), generator=g)), int(tokens[s, pos])])
    ring_meta[0, 2, 2] += 1  # drafted from another token: not held
    ring_meta[1, 1, 0] = int(total_len[1]) + 2  # anchored past the turn
    ring_meta[2, 4, 0] = int(total_len[2]) - 2  # the turn ends inside the block
    ring_meta[2, 4, 2] = int(tokens[2, int(total_len[2]) - 2])
    finished = [2, 0, 1, 5]
    kept = [8, 6, 4, 0]  # slot 1: one more than the ring holds
    entries = [(f, k) for f, c in enumerate(kept) for k in range(c)]
    count, first = len(entries), 5
    stage_len = count + 9
    to = lambda t: t.to(DEV)  # noqa: E731
    fin = to(torch.tensor(finished, dtype=torch.int64))
    dev = [to(t) for t in (tokens, total_len, ring_values, ring_index, ring_meta, ring_drafts, ring_count)]
    labels = torch.empty((len(finished), cap, N_SPEC), dtype=torch.int32, device=DEV)
    ids = torch.empty((len(finished), cap, N_SPEC + 1), dtype=torch.int32, device=DEV)
    tl._kernel(tl.labels_kernel, N_SPEC)(fin, dev[0], dev[1], dev[4], dev[6], labels, ids)
    st_values = torch.zeros((stage_len, N_SPEC, PARTS), dtype=torch.float32, device=DEV)
    st_index = torch.zeros((stage_len, N_SPEC, PARTS), dtype=torch.int32, device=DEV)
    st_meta = torch.zeros((stage_len, tl._META), dtype=torch.int64, device=DEV)
    st_drafts = torch.zeros((stage_len, N_SPEC), dtype=torch.int32, device=DEV)
    st_labels = torch.full((stage_len, N_SPEC), 99, dtype=torch.int32, device=DEV)
    st_ids = torch.full((stage_len, N_SPEC + 1), 99, dtype=torch.int32, device=DEV)
    entry_f = to(torch.tensor([f for f, _ in entries], dtype=torch.int64))
    entry_k = to(torch.tensor([k for _, k in entries], dtype=torch.int64))
    tl._kernel(sh.shadow_stage_kernel, N_SPEC, PARTS)(
        entry_f, entry_k, fin, dev[0], dev[1], dev[2], dev[3], dev[4], dev[5], dev[6],
        st_values, st_index, st_meta, st_drafts, st_labels, st_ids, first,
    )
    torch.cuda.synchronize()
    labels, ids = labels.cpu(), ids.cpu()
    for e, (f, k) in enumerate(entries):
        s, slot = first + e, finished[f]
        assert torch.equal(st_labels[s].cpu(), labels[f, k])
        assert torch.equal(st_ids[s].cpu(), ids[f, k])
        assert torch.equal(st_values[s].cpu(), ring_values[slot, k])
        assert torch.equal(st_index[s].cpu(), ring_index[slot, k])
        assert torch.equal(st_meta[s].cpu(), ring_meta[slot, k])
        assert torch.equal(st_drafts[s].cpu(), ring_drafts[slot, k])
    # Rows outside the call are untouched.
    assert (st_labels[:first] == 99).all() and (st_labels[first + count :] == 99).all()
    # The cases the data is meant to hold.
    staged = st_labels[first : first + count].cpu()
    assert (staged[:, 0] == -2).any() and (staged == -1).any() and (staged >= 0).any()


@pytest.mark.parametrize("positions, vocab", [(40, 12), (700, 40), (1280, 300)])
def test_update_sums_each_row_once_in_position_order(positions, vocab):
    torch.manual_seed(3)
    train = (torch.rand(positions) < 0.7).int()
    prev = torch.randint(0, vocab, (positions,), dtype=torch.int32)
    grad = (torch.randint(-8, 9, (positions, DIM)) / 16).float()
    master = (torch.randint(-8, 9, (vocab, DIM)) / 16).float()
    accum = torch.rand(vocab) * 4
    ref_master, ref_accum = master.clone(), accum.clone()
    done = set()
    for pos in range(positions):
        p = int(prev[pos])
        if not train[pos] or p in done:
            continue
        done.add(p)
        g = sum(grad[j] for j in range(positions) if train[j] and int(prev[j]) == p)
        ref_accum[p] += float((g * g).sum())
        ref_master[p] -= 0.1 / (math.sqrt(float(ref_accum[p])) + 1e-6) * g
    assert len(done) < int(train.sum())  # rows repeat
    dev = [t.to(DEV) for t in (train, prev, grad, master, accum)]
    tl._kernel(sh.shadow_update_kernel, DIM, sh._update_words(positions))(*dev, 0.1)
    torch.cuda.synchronize()
    torch.testing.assert_close(dev[4].cpu(), ref_accum, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(dev[3].cpu(), ref_master, rtol=1e-6, atol=1e-6)


def test_tile_columns_and_lr_are_validated(monkeypatch):
    assert sh._tile_cols(32320, 505) == 64
    assert sh._tile_cols(32320, 253) == 128
    with pytest.raises(RuntimeError):
        sh._tile_cols(32320, 7)
    monkeypatch.setenv("VLLM_DSPARK_TURN_LEARNING_LR", "0")
    with pytest.raises(ValueError):
        sh.turn_learning_lr()
    monkeypatch.setenv("VLLM_DSPARK_TURN_LEARNING", "shadow")
    assert tl.turn_learning_mode() == "shadow"
