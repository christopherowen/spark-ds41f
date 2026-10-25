# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sink rejection on the V2 runner's sampling layer, through the real kernels
(vllm.v1.worker.gpu.sample.sink): the detector, the drafter-row cut, the
alternates of each draw, the draft-zero policy, the request-state update, the
drafter's inputs, and the collapse sequence across steps with the real
sampler and rejection sampler (greedy and random, block verification)."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.sample.sink import (
    SinkConfig,
    SinkOutcome,
    SinkRow,
    events_to_dict,
)
from vllm.v1.worker.gpu.input_batch import post_update
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.sink import SinkAlternates, SinkSampler
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs
from vllm.v1.worker.gpu.spec_decode.greedy_verification import (
    greedy_rejection_sample,
    vocab_parallel_pairs,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.states import RequestState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

DEVICE = torch.device("cuda")
REJECTION = SinkConfig(column=2455, threshold=85000.0, rejection=True, max_retries=3)
DETECTOR = SinkConfig(column=2455, threshold=85000.0, rejection=False, max_retries=3)
SUNK_VALUE = 108544.0
CLEAN_VALUE = 190.0


def _sink(config: SinkConfig = REJECTION, *, max_num_reqs=8, vocab=2048, k=2):
    return SinkSampler(config, max_num_reqs, 5120, vocab, k, DEVICE, False)


def _cu(counts: list[int]) -> torch.Tensor:
    return torch.tensor(np.cumsum([0] + counts), dtype=torch.int32, device=DEVICE)


def test_flags_on_reference_numbers():
    """|residual[2455]| per stream at the sampled rows, from the reference
    runs and the served survey (analysis/sink_column.txt, survey/): sunk
    rows 108,032 to 125,440, number tokens up to 62,720 with normal
    predictions, reasoning words up to 10,688, the effort line's prompt sink
    7,008, normal rows 153 to 270."""
    rows = {
        "say hi sunk ' can' (layer 38)": ([108032.0] * 4, True),
        "say hi sunk ' can'": ([-108544.0, 3.0, 4.0, 5.0], True),
        "thanks sunk": ([0.0, 0.0, 119296.0, 0.0], True),
        "effort-line sunk max": ([125440.0] + [1.0] * 3, True),
        "clean ' friendly' (effort 50)": ([5500.0] * 4, False),
        "clean ' friendly' (9,984, effort 90)": ([9984.0, -9984.0, 1.0, 1.0], False),
        "clean max ' effectively' (10,688)": ([10688.0, 2.0, 3.0, 4.0], False),
        "number ' ' before '1.0' (47,360)": ([47360.0, 1.0, 1.0, 1.0], False),
        "number '202' in '2023' (51,968)": ([1.0, 51968.0, 1.0, 1.0], False),
        "survey max ' ' before '1.3%' (62,720)": ([62720.0] * 4, False),
        "prompt ' (' sink": ([7008.0] * 4, False),
        "clean ' can'": ([190.0, 64.0, 120.0, 150.0], False),
        "clean ' help'": ([270.0] * 4, False),
        "nan": ([float("nan")] * 4, False),
    }
    columns = torch.tensor([v for v, _ in rows.values()], device=DEVICE)
    n = columns.shape[0]
    sink = _sink()
    flags = sink.flags(columns, torch.arange(n, device=DEVICE), _cu([n]), 1, None)
    assert flags.tolist() == [int(f) for _, f in rows.values()]
    # Each row's largest |value| over the streams (NaN reads 0), for the log.
    assert sink.row_values.tolist() == [
        0.0 if v[0] != v[0] else max(abs(x) for x in v) for v, _ in rows.values()
    ]
    # Dead verification rows (past a request's live drafts) are never
    # flagged: their residual skipped routed MoE.
    flags = _sink().flags(
        columns[:4],
        torch.arange(4, device=DEVICE),
        _cu([2, 2]),
        2,
        torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
    )
    assert flags.tolist() == [1, 0, 1, 1]


def _torch_flags(columns, logits_indices, cu_num_logits, num_reqs, dead_cut, threshold):
    """The detector's flags as they were computed in torch (the TileLang kernel's
    reference): each row's largest |column| (NaN reads 0) and its flag."""
    rows = columns[logits_indices].abs()
    values = rows.amax(dim=1).nan_to_num(0.0)
    flags = (rows > threshold).any(dim=1)
    if dead_cut is not None:
        index = torch.arange(flags.shape[0], dtype=cu_num_logits.dtype, device=DEVICE)
        req = torch.searchsorted(cu_num_logits[1 : num_reqs + 1], index, right=True)
        flags &= index - cu_num_logits[req] <= dead_cut[req]
    return flags.to(torch.int8), values


@pytest.mark.parametrize("dead", [False, True])
def test_flags_match_the_torch_reference(dead: bool):
    """Random columns with NaN, +-inf, -0 and values at the threshold, at scattered
    tokens, five requests of up to six rows: flags and values bit for bit."""
    gen = torch.Generator(device=DEVICE).manual_seed(29)
    columns = torch.randn(300, 4, generator=gen, device=DEVICE) * 60000.0
    columns[::7, 1] = float("nan")
    columns[3::11, 2] = float("inf")
    columns[5::13, 0] = -float("inf")
    columns[::17] = -0.0
    columns[2::19, 3] = REJECTION.threshold
    columns[4::23, 3] = -REJECTION.threshold
    counts = [6, 1, 4, 6, 3]
    num_logits, num_reqs = sum(counts), len(counts)
    logits_indices = torch.randperm(300, generator=gen, device=DEVICE)[:num_logits]
    cu = _cu(counts)
    cut = torch.tensor([2, 0, 5, 1, 0], dtype=torch.int32, device=DEVICE)
    cut = cut if dead else None
    sink = _sink()
    flags = sink.flags(columns, logits_indices, cu, num_reqs, cut)
    ref_flags, ref_values = _torch_flags(
        columns, logits_indices, cu, num_reqs, cut, REJECTION.threshold
    )
    assert torch.equal(flags, ref_flags)
    assert torch.equal(sink.row_values.view(torch.int32), ref_values.view(torch.int32))


def _random_logits(rows: int, vocab: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(rows, vocab, generator=gen, device=DEVICE) * 2.0


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_random_alternates_are_the_draws_next_samples(temperature: float):
    """The alternates of a random draw are the draws of the same noise with
    the earlier tokens taken out: sampling without replacement."""
    num_reqs, vocab = 4, 3000
    sink = _sink(vocab=vocab)
    logits = _random_logits(num_reqs, vocab) / temperature
    idx = torch.tensor([3, 1, 0, 2], dtype=torch.int64, device=DEVICE)
    temp = torch.full((8,), temperature, dtype=torch.float32, device=DEVICE)
    seeds = torch.arange(8, dtype=torch.int64, device=DEVICE) * 7919
    pos = torch.tensor([5, 17, 33, 2], dtype=torch.int64, device=DEVICE)

    def draw(scores: torch.Tensor) -> torch.Tensor:
        return gumbel_sample(
            scores, idx, temp, seeds, pos, apply_temperature=False, is_drafting=False
        )

    sampled = draw(logits)
    alternates = sink.alternates_of(
        logits,
        _cu([1] * num_reqs),
        torch.ones(num_reqs, dtype=torch.int32, device=DEVICE),
        sampled.view(-1, 1),
        torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        idx,
        temp,
        seeds,
        pos,
        mask_rejected_draft=False,
    )
    taken = logits.clone()
    taken.scatter_(1, sampled[:, None], float("-inf"))
    for i in range(sink.num_alternates):
        resampled = draw(taken)
        assert torch.equal(resampled, alternates[:, i])
        taken.scatter_(1, resampled[:, None], float("-inf"))


def test_greedy_alternates_follow_the_logits():
    num_reqs, vocab = 3, 2500
    sink = _sink(vocab=vocab)
    logits = _random_logits(num_reqs, vocab, seed=3)
    temp = torch.zeros(8, dtype=torch.float32, device=DEVICE)
    sampled = logits.argmax(dim=-1)
    alternates = sink.alternates_of(
        logits,
        _cu([1] * num_reqs),
        torch.ones(num_reqs, dtype=torch.int32, device=DEVICE),
        sampled.view(-1, 1),
        torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        torch.arange(num_reqs, dtype=torch.int64, device=DEVICE),
        temp,
        torch.zeros(8, dtype=torch.int64, device=DEVICE),
        torch.zeros(num_reqs, dtype=torch.int64, device=DEVICE),
        mask_rejected_draft=False,
    )
    assert torch.equal(alternates, logits.topk(4, dim=-1).indices[:, 1:])
    # An impossible candidate (masked by the grammar, top-k...) is none.
    masked = logits.clone()
    keep = logits.topk(2, dim=-1).indices
    masked.fill_(float("-inf"))
    masked.scatter_(1, keep, logits.gather(1, keep))
    alternates = sink.alternates_of(
        masked,
        _cu([1] * num_reqs),
        torch.ones(num_reqs, dtype=torch.int32, device=DEVICE),
        keep[:, :1].contiguous(),
        torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        torch.arange(num_reqs, dtype=torch.int64, device=DEVICE),
        temp,
        torch.zeros(8, dtype=torch.int64, device=DEVICE),
        torch.zeros(num_reqs, dtype=torch.int64, device=DEVICE),
        mask_rejected_draft=False,
    )
    assert alternates[:, 0].tolist() == keep[:, 1].tolist()
    assert (alternates[:, 1:] == -1).all()


def _torch_cut(logits, flags, drafts, cu, num_reqs, vocab_start=0, vocab_cols=None):
    """The drafter-row cut as it was computed in torch (the TileLang kernel's
    reference), in place."""
    num_logits, num_cols = logits.shape
    starts = torch.zeros(num_logits + 1, dtype=torch.bool, device=DEVICE)
    starts[cu[:num_reqs].long()] = True
    cols = drafts[1:num_logits].long() - vocab_start
    cut = flags[1:num_logits].bool() & ~starts[1:num_logits] & (cols >= 0)
    cut &= cols < (num_cols if vocab_cols is None else vocab_cols)
    cols = cols.where(cut, 0)[:, None]
    rows = logits[: num_logits - 1]
    saved = rows.gather(1, cols)
    rows.scatter_(1, cols, saved.masked_fill(cut[:, None], float("-inf")))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shard", [None, (700, 600)])
def test_cut_matches_the_torch_reference(dtype, shard):
    """Random flags and drafts over five requests, the full vocabulary or a shard
    (columns 700 on, 600 of them): the cut equals torch's bit for bit, and the
    restore gives the logits back."""
    gen = torch.Generator(device=DEVICE).manual_seed(41)
    counts = [3, 1, 6, 4, 2]
    num_logits, num_reqs = sum(counts), len(counts)
    vocab = 1500
    drafts = torch.randint(0, vocab, (num_logits,), generator=gen, device=DEVICE).int()
    flags = torch.randint(0, 2, (num_logits,), generator=gen, device=DEVICE).to(
        torch.int8
    )
    cu = _cu(counts)
    start, cols = shard or (0, None)
    width = vocab if shard is None else cols
    logits = _random_logits(num_logits, width, seed=6).to(dtype)
    original = logits.clone()
    expected = logits.clone()
    _torch_cut(expected, flags, drafts, cu, num_reqs, start, cols)
    cut = _sink(vocab=vocab).cut(logits, flags, drafts, cu, num_reqs, start, cols)
    assert torch.equal(logits, expected)
    cut.restore()
    assert torch.equal(logits, original)


def test_cut_masks_sunk_drafts_and_restores_the_logits():
    """Row r verifies the draft at row r + 1: a flagged draft row masks the
    draft in the row before it, within the request only, and the restore
    gives back the logits bit for bit."""
    vocab = 1500
    sink = _sink(vocab=vocab)
    # Two requests of 3 rows: inputs [P, d1, d2], [Q, e1, e2].
    drafts = torch.tensor([5, 11, 12, 6, 21, 22], dtype=torch.int32, device=DEVICE)
    flags = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.int8, device=DEVICE)
    cu = _cu([3, 3])
    logits = _random_logits(6, vocab, seed=5)
    original = logits.clone()
    cut = sink.cut(logits, flags, drafts, cu, 2)
    masked = (logits == float("-inf")).nonzero().tolist()
    # Row 1 verifies d2 (flagged); row 3 (Q, flagged) is row 0 of the second
    # request and verifies nothing before it; row 3 verifies e1 (flagged).
    assert masked == [[1, 12], [3, 21]]
    cut.restore()
    assert torch.equal(logits, original)
    # A vocabulary shard masks only the drafts it holds.
    shard = original[:, 1000:1500].clone()
    sink.cut(shard, flags, drafts, cu, 2, vocab_start=1000, vocab_cols=500)
    assert not (shard == float("-inf")).any()
    shard = original[:, :1000].clone()
    sink.cut(shard, flags, drafts, cu, 2, vocab_start=0, vocab_cols=1000)
    assert (shard == float("-inf")).nonzero().tolist() == [[1, 12], [3, 21]]


def _torch_merge(values, indices, num_sampled, sampled, num_alternates):
    """SinkSampler's merge as it was computed in torch (the TileLang kernel's
    reference)."""
    last = (num_sampled.long() - 1).clamp(min=0)[:, None]
    excluded = (indices == sampled.gather(1, last)) | values.isnan()
    values = values.masked_fill(excluded, float("-inf"))
    values, order = values.sort(dim=1, descending=True, stable=True)
    alternates = indices.gather(1, order[:, :num_alternates])
    return alternates.masked_fill(values[:, :num_alternates] == float("-inf"), -1)


def test_merge_matches_the_torch_reference():
    """Random candidates with repeated values (the stable sort keeps their order),
    -inf, NaN and the sampled token among them."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    gen = torch.Generator(device=DEVICE).manual_seed(53)
    num_reqs, width = 7, 508
    values = torch.randint(
        -4, 4, (num_reqs, width), generator=gen, device=DEVICE
    ).float()
    values[:, ::9] = float("-inf")
    values[:, 3::31] = float("nan")
    indices = torch.randperm(4 * width, generator=gen, device=DEVICE)[:width]
    indices = indices.repeat(num_reqs, 1)
    sampled = indices[:, :3].flip(1).contiguous()
    num_sampled = torch.tensor([1, 2, 3, 1, 2, 3, 0], dtype=torch.int32, device=DEVICE)
    values[6] = float("-inf")
    values[6, 5] = 1.0
    got = tilelang_sampling.sink_merge(values, indices, sampled, num_sampled, 3)
    assert torch.equal(got, _torch_merge(values, indices, num_sampled, sampled, 3))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shard_candidates_are_the_rows_best_columns(dtype):
    """Each request's last sampled row's k best columns of a shard, packed with the
    vocabulary indices: a stable descending sort's first k (NaN first, as torch's sort
    and topk put it; equal values by column). On FP32 rows without ties that is
    torch.topk's answer too."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    gen = torch.Generator(device=DEVICE).manual_seed(61)
    cols, k, start = 32320, 4, 32320
    counts = [3, 1, 6, 2]
    num_logits = sum(counts)
    logits = (torch.randn(num_logits, cols, generator=gen, device=DEVICE) * 4).to(dtype)
    logits[1, 7] = float("nan")
    num_sampled = torch.tensor([3, 1, 4, 2], dtype=torch.int32, device=DEVICE)
    cu = _cu(counts)
    packed = tilelang_sampling.sink_shard_top(logits, cu, num_sampled, start, cols, k)
    rows = cu[:-1].long() + (num_sampled.long() - 1).clamp(min=0)
    values, order = logits[rows].float().sort(dim=-1, descending=True, stable=True)
    values = values[:, :k].masked_fill(values[:, :k].isnan(), float("-inf"))
    assert torch.equal(packed[:, :k], values)
    assert torch.equal(packed[:, k:], (order[:, :k] + start).float())
    if dtype == torch.float32:
        top, indices = logits[rows].topk(k, dim=-1)
        assert torch.equal(packed[:, k:], (indices + start).float())


def test_shard_candidate_ties_go_to_the_lower_column():
    """Equal values rank by column, as the full-logits path orders them."""
    from vllm.v1.worker.gpu.sample import tilelang_sampling

    logits = torch.zeros(1, 4096, device=DEVICE)
    logits[0, [3000, 17, 2048, 900]] = 5.0
    packed = tilelang_sampling.sink_shard_top(
        logits,
        _cu([1]),
        torch.ones(1, dtype=torch.int32, device=DEVICE),
        0,
        4096,
        3,
    )
    assert packed[0, 3:].tolist() == [17.0, 900.0, 2048.0]


def test_greedy_shards_match_the_full_logits():
    """Greedy verification from vocabulary shards (one rank here) gives the
    same tokens and alternates as the full logits."""
    vocab, k = 2048, 2
    sink = _sink(vocab=vocab, k=k)
    logits = _random_logits(6, vocab, seed=9)
    drafts = logits.argmax(dim=-1).to(torch.int32).roll(1)
    drafts[0] = 0
    drafts[3] = 0
    # Request 0 rejects its second draft; request 1 takes the bonus.
    drafts[2] = (drafts[2] + 1) % vocab
    cu = _cu([3, 3])
    pairs = vocab_parallel_pairs(logits, vocab, 0, 1)
    sampled, num_sampled = greedy_rejection_sample(pairs, drafts, cu, k)
    shard = sink.alternates_of_greedy_shards(
        logits, cu, num_sampled, sampled, 0, vocab, 1
    )
    full = sink.alternates_of(
        logits,
        cu,
        num_sampled,
        sampled,
        drafts,
        torch.arange(2, dtype=torch.int64, device=DEVICE),
        torch.zeros(8, dtype=torch.float32, device=DEVICE),
        torch.zeros(8, dtype=torch.int64, device=DEVICE),
        torch.zeros(6, dtype=torch.int64, device=DEVICE),
        mask_rejected_draft=True,
    )
    assert torch.equal(shard, full)


@dataclass
class PolicyCase:
    """One step of three requests for the policy kernel."""

    sampled: list[list[int]]
    num_sampled: list[int]
    rows: list[int]
    decoding: list[int]
    flags: list[int]
    drafts: list[int]
    new_alternates: list[list[int]]
    state: dict[int, tuple[list[int], int]] = field(default_factory=dict)


def _apply(
    sink: SinkSampler,
    case: PolicyCase,
    logprobs: bool = False,
    values: list[float] | None = None,
):
    num_reqs = len(case.rows)
    width = sink.num_speculative_steps + 1
    sampled = torch.full((num_reqs, width), -7, dtype=torch.int64, device=DEVICE)
    for i, tokens in enumerate(case.sampled):
        sampled[i, : len(tokens)] = torch.tensor(tokens)
    num_sampled = torch.tensor(case.num_sampled, dtype=torch.int32, device=DEVICE)
    num_rejected = (
        torch.tensor(case.rows, dtype=torch.int32, device=DEVICE) - num_sampled
    ).clamp(min=0)
    idx = torch.tensor([4, 0, 6][:num_reqs], dtype=torch.int64, device=DEVICE)
    for slot, (alternates, tries) in case.state.items():
        sink.alternates[slot] = torch.tensor(alternates, device=DEVICE)
        sink.alternate_logprobs[slot] = torch.tensor([-1.0, -2.0, -3.0], device=DEVICE)
        sink.alternate_ranks[slot] = torch.tensor([2, 3, 4], device=DEVICE)
        sink.tries[slot] = tries
    total_len = torch.full((8,), 40, dtype=torch.int32, device=DEVICE)
    bin_counts = torch.zeros((8, 64), dtype=torch.int32, device=DEVICE)
    bin_counts[:, :] = 1
    alternates = None
    if sink.rejection:
        new = torch.tensor(case.new_alternates, dtype=torch.int64, device=DEVICE)
        alternates = SinkAlternates(
            new,
            torch.full(new.shape, -9.0, device=DEVICE) if logprobs else None,
            torch.full(new.shape, 9, dtype=torch.int64, device=DEVICE)
            if logprobs
            else None,
        )
    out = sink.apply(
        sampled,
        num_sampled,
        num_rejected,
        _cu(case.rows),
        idx,
        torch.tensor(case.decoding, dtype=torch.int8, device=DEVICE),
        torch.tensor(case.flags, dtype=torch.int8, device=DEVICE),
        torch.tensor(case.drafts, dtype=torch.int32, device=DEVICE),
        alternates,
        total_len,
        bin_counts,
        None if values is None else torch.tensor(values, device=DEVICE),
    )
    return SimpleNamespace(
        out=out,
        sampled=sampled,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        total_len=total_len,
        bin_counts=bin_counts,
        idx=idx.tolist(),
    )


def _host_tokens(result) -> list[list[int]]:
    tokens = result.out.tokens.tolist()
    return [row[:n] for row, n in zip(tokens, result.out.num_tokens.tolist())]


def test_policy_rejects_a_sunk_draft_zero_and_leaves_its_neighbours():
    """Request 1 (slot 0) has its draft zero flagged: it is replaced by the
    next alternate of its draw and all its rows are rejected. Requests 0 and
    2 get draft zero in front of their verified tokens, and take their new
    alternates."""
    sink = _sink()
    case = PolicyCase(
        sampled=[[31, 32, 33], [41, 42], [51]],
        num_sampled=[3, 2, 1],
        rows=[3, 3, 3],
        decoding=[1, 1, 1],
        # Row 0 of each request: draft zero P, Q, R = 30, 40, 50.
        flags=[0, 0, 0, 1, 0, 0, 0, 0, 0],
        drafts=[30, 31, 32, 40, 41, 49, 50, 58, 59],
        new_alternates=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        state={0: ([60, 61, 62], 0), 4: ([70, 71, 72], 0), 6: ([80, 81, 82], 0)},
    )
    r = _apply(sink, case, logprobs=True)
    assert _host_tokens(r) == [[30, 31, 32, 33], [60], [50, 51]]
    assert r.num_sampled.tolist() == [3, 1, 1]
    assert r.num_rejected.tolist() == [0, 3, 2]
    assert r.sampled[1, 0].item() == 60
    # The request state drops draft zero; the update appends the alternate.
    assert r.total_len[0].item() == 39 and r.total_len[4].item() == 40
    assert r.bin_counts[0, 40].item() == 0 and r.bin_counts[4, 30].item() == 1
    assert sink.tries[0].item() == 1
    assert sink.alternates[0].tolist() == [60, 61, 62]
    assert sink.alternates[4].tolist() == [1, 2, 3]
    assert sink.alternates[6].tolist() == [7, 8, 9]
    assert r.out.recovered_logprobs[1].item() == -1.0
    assert r.out.recovered_ranks[1].item() == 2
    events = events_to_dict(
        r.out.events.tolist(),
        ["a", "b", "c"],
        r.out.recovered_logprobs.tolist(),
        r.out.recovered_ranks.tolist(),
    )
    assert set(events) == {"b"}
    assert events["b"].outcome == SinkOutcome.REJECTED
    assert events["b"].row0 == SinkRow.DRAFT_ZERO
    assert events["b"].recovered_rank == 2
    assert sink.counts.tolist() == [1, 0, 0]


def test_policy_retries_then_falls_through():
    sink = _sink()
    case = PolicyCase(
        sampled=[[41]],
        num_sampled=[1],
        rows=[1],
        decoding=[1],
        flags=[1],
        drafts=[40],
        new_alternates=[[4, 5, 6]],
        state={4: ([60, 61, 62], 2)},
    )
    r = _apply(sink, case)
    assert _host_tokens(r) == [[62]]
    assert sink.tries[4].item() == 3
    case.state = {}
    r = _apply(sink, case)
    # Past its alternates draft zero stands: committed, new alternates taken.
    assert _host_tokens(r) == [[40, 41]]
    assert r.out.events[0].tolist()[:4] == [SinkOutcome.FALLTHROUGH, 0, 0, -1]
    assert sink.tries[4].item() == 0
    # An impossible alternate (-1) also falls through.
    case.state = {4: ([-1, -1, -1], 0)}
    r = _apply(sink, case)
    assert r.out.events[0, 0].item() == SinkOutcome.FALLTHROUGH


def test_policy_drafter_cut_prompt_row_and_prefill_chunk():
    sink = _sink()
    case = PolicyCase(
        # Request 0: draft zero clean, d1 accepted, cut at row 2 (its draft's
        # row flagged), recovered token 33. Request 1: finishing its prefill
        # with a flagged prompt row. Request 2: a prefill chunk.
        sampled=[[31, 33], [41], [51]],
        num_sampled=[2, 1, 0],
        rows=[3, 1, 1],
        decoding=[1, 0, 0],
        flags=[0, 0, 1, 1, 0],
        drafts=[30, 31, 32, 40, 50],
        new_alternates=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    )
    r = _apply(sink, case)
    assert _host_tokens(r) == [[30, 31, 33], [41], []]
    events = r.out.events.tolist()
    assert events[0][:4] == [SinkOutcome.NONE, SinkRow.DRAFT_ZERO, 1, 2]
    assert events[1][:4] == [SinkOutcome.FLAGGED, SinkRow.PROMPT, 0, -1]
    assert events[2][0] == SinkOutcome.NONE and events[2][2] == 0
    assert sink.counts.tolist() == [0, 1, 1]
    # A finished prefill takes its draw's alternates; a chunk does not.
    assert sink.alternates[0].tolist() == [4, 5, 6]
    assert sink.alternates[6].tolist() != [7, 8, 9]


def test_detector_only_counts_and_changes_nothing():
    sink = _sink(DETECTOR)
    case = PolicyCase(
        sampled=[[31, 32], [41]],
        num_sampled=[2, 1],
        rows=[3, 1],
        decoding=[1, 1],
        flags=[1, 1, 0, 0],
        drafts=[30, 31, 32, 40],
        new_alternates=[],
    )
    r = _apply(sink, case, values=[61000.0, 70500.4, 12.0, 30.0])
    assert r.out.tokens is None
    assert r.num_sampled.tolist() == [2, 1]
    assert r.sampled[0, :2].tolist() == [31, 32]
    assert r.total_len[4].item() == 40
    # A sunk committed token and a sunk accepted draft, counted, with the
    # largest flagged value for the log line; nothing flagged in request 2.
    events = r.out.events.tolist()
    assert events[0] == [SinkOutcome.FLAGGED, SinkRow.DRAFT_ZERO, 1, -1, 70500, 1]
    assert events[1] == [SinkOutcome.NONE, SinkRow.DRAFT_ZERO, 0, -1, 0, -1]
    assert sink.counts.tolist() == [1, 1, 0]


def test_event_names_the_row_of_its_value():
    """Row 0 clean and a later drafter row flagged: the event's value is
    that row's and names it; a flagged row past the first rejected draft is
    not judged."""
    sink = _sink(DETECTOR)
    case = PolicyCase(
        sampled=[[31, 32, 33], [41]],
        num_sampled=[3, 1],
        rows=[3, 3],
        decoding=[1, 1],
        flags=[0, 0, 1, 0, 0, 1],
        drafts=[30, 31, 32, 40, 41, 42],
        new_alternates=[],
    )
    r = _apply(sink, case, values=[9.0, 40000.0, 90000.0, 7.0, 8.0, 95000.0])
    events = r.out.events.tolist()
    assert events[0] == [SinkOutcome.NONE, SinkRow.DRAFT_ZERO, 1, -1, 90000, 2]
    # Request 2 accepted no draft: its flagged row 2 lies past row 1, the
    # first rejected draft, so nothing is judged flagged.
    assert events[1] == [SinkOutcome.NONE, SinkRow.DRAFT_ZERO, 0, -1, 0, -1]


def test_warmup_rows_are_neither_logged_nor_kept(monkeypatch: pytest.MonkeyPatch):
    """The startup warmup's synthetic rows (token 0, the BOS, as draft zero)
    are flagged but never logged, and its end forgets them; afterwards a
    flagged row is logged once its count is read back."""
    from vllm.v1.worker.gpu.sample import sink as sink_module

    lines: list[str] = []
    monkeypatch.setattr(
        sink_module.logger, "info", lambda message, *args: lines.append(message % args)
    )
    sink = _sink(DETECTOR)
    case = PolicyCase(
        sampled=[[41]],
        num_sampled=[1],
        rows=[1],
        decoding=[1],
        flags=[1],
        drafts=[0],
        new_alternates=[],
    )

    def step() -> None:
        _apply(sink, case)
        sink.log_counts(0)
        torch.cuda.synchronize()
        sink.log_counts(0)

    sink.warming = True
    step()
    assert sink.counts.tolist() == [1, 0, 0]
    assert lines == []
    sink.end_warmup()
    assert sink.counts.tolist() == [0, 0, 0] and not sink.warming
    step()
    assert lines == [
        "Sink detector, rank 0: 1 flagged rows in executed steps (draft zero 1, "
        "drafter 0, prompt 0; vllm:sink_* counts committed outputs only)"
    ]


def test_post_update_replaces_draft_zero():
    """The real request-state update after the policy: a rejected draft zero
    is replaced in place (same length, same computed count, the alternate as
    the last sampled token, bin counts moved); an accepted one advances."""
    k = 2
    states = RequestState(8, 64, 32, k, 64, DEVICE)
    for slot_req, (req_id, provisional) in enumerate((("a", 30), ("b", 40))):
        states.add_request(req_id, 10, list(range(10)) + [provisional], 10, 20)
    states.apply_staged_writes()
    idx_a = states.req_id_to_index["a"]
    idx_b = states.req_id_to_index["b"]
    states.last_sampled_tokens[idx_a] = 30
    states.last_sampled_tokens[idx_b] = 40
    bin_counts = torch.zeros((8, 64), dtype=torch.int32, device=DEVICE)
    bin_counts[idx_a, 30] = 1
    bin_counts[idx_b, 40] = 1
    sink = _sink(vocab=64, k=k)
    sink.alternates[idx_b] = torch.tensor([45, 46, 47], device=DEVICE)
    sink.tries[idx_b] = 0
    idx = torch.tensor([idx_a, idx_b], dtype=torch.int64, device=DEVICE)
    sampled = torch.tensor([[31, 33, -1], [41, -1, -1]], device=DEVICE)
    num_sampled = torch.tensor([2, 1], dtype=torch.int32, device=DEVICE)
    num_rejected = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
    cu = _cu([3, 3])
    out = sink.apply(
        sampled,
        num_sampled,
        num_rejected,
        cu,
        idx,
        torch.ones(2, dtype=torch.int8, device=DEVICE),
        torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.int8, device=DEVICE),
        torch.tensor([30, 31, 32, 40, 41, 42], dtype=torch.int32, device=DEVICE),
        SinkAlternates(torch.tensor([[1, 2, 3], [4, 5, 6]], device=DEVICE)),
        states.total_len.gpu,
        bin_counts,
    )
    post_update(
        idx,
        states.num_computed_tokens.gpu,
        states.last_sampled_tokens,
        bin_counts,
        sampled,
        num_sampled,
        num_rejected,
        cu,
        states.all_token_ids.gpu,
        states.total_len.gpu,
    )
    torch.accelerator.synchronize()
    assert out.tokens is not None
    computed = states.num_computed_tokens.gpu
    assert computed[idx_a].item() == 12 and computed[idx_b].item() == 10
    assert states.total_len.gpu[idx_a].item() == 13
    assert states.total_len.gpu[idx_b].item() == 11
    assert states.last_sampled_tokens[idx_b].item() == 45
    assert states.all_token_ids.gpu[idx_b, :11].tolist() == list(range(10)) + [45]
    assert states.all_token_ids.gpu[idx_a, 10:13].tolist() == [30, 31, 33]
    assert bin_counts[idx_b, 40].item() == 0 and bin_counts[idx_b, 45].item() == 1


def test_drafter_inputs_after_draft_zero_rejected():
    """DSpark/DFlash: a request with no valid row anchors its replacement at
    draft zero's position (not the previous request's) and writes no context
    KV; its neighbours' inputs are unchanged."""

    def prepare(num_rejected_b: int, num_sampled_b: int):
        num_reqs, k, n = 3, 3, 4
        max_num_reqs, max_num_tokens = 4, 32
        bufs = SimpleNamespace(
            input_ids=torch.full(
                (max_num_tokens,), -1, dtype=torch.int32, device=DEVICE
            ),
            positions=torch.full(
                (max_num_tokens,), -1, dtype=torch.int64, device=DEVICE
            ),
            query_start_loc=torch.full(
                (max_num_reqs + 1,), -1, dtype=torch.int32, device=DEVICE
            ),
            seq_lens=torch.full((max_num_reqs,), -1, dtype=torch.int32, device=DEVICE),
        )
        positions = [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]
        batch = SimpleNamespace(
            num_reqs=num_reqs,
            num_scheduled_tokens=np.array([n] * num_reqs, dtype=np.int32),
            positions=torch.tensor(positions, dtype=torch.int64, device=DEVICE),
            query_start_loc=torch.tensor(
                [0, 4, 8, 12], dtype=torch.int32, device=DEVICE
            ),
            idx_mapping=torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE),
        )
        out = SimpleNamespace(
            query_slots=torch.full(
                (max_num_tokens,), -2, dtype=torch.int64, device=DEVICE
            ),
            context_positions=torch.full(
                (max_num_tokens,), -1, dtype=torch.int64, device=DEVICE
            ),
            context_slots=torch.full(
                (max_num_tokens,), -2, dtype=torch.int64, device=DEVICE
            ),
            sample_indices=torch.full(
                (max_num_reqs * k,), -1, dtype=torch.int64, device=DEVICE
            ),
        )
        sample_pos = torch.full_like(out.sample_indices, -1)
        sample_map = torch.full(
            out.sample_indices.shape, -1, dtype=torch.int32, device=DEVICE
        )
        prepare_dflash_inputs(
            bufs,
            out.query_slots,
            out.context_positions,
            out.context_slots,
            out.sample_indices,
            sample_pos,
            sample_map,
            torch.zeros(max_num_reqs, dtype=torch.float32, device=DEVICE),
            torch.zeros(max_num_reqs, dtype=torch.int64, device=DEVICE),
            batch,
            torch.tensor([2, num_sampled_b, 1], dtype=torch.int32, device=DEVICE),
            torch.tensor([2, num_rejected_b, 3], dtype=torch.int32, device=DEVICE),
            torch.tensor([99, 98, 97, 0], dtype=torch.int64, device=DEVICE),
            torch.zeros(4, dtype=torch.int64, device=DEVICE),
            torch.zeros(max_num_reqs, dtype=torch.float32, device=DEVICE),
            torch.zeros(max_num_reqs, dtype=torch.int64, device=DEVICE),
            torch.arange(1, 13, dtype=torch.int32, device=DEVICE).view(3, 4),
            16,
            0,
            1,
            1,
            123,
            k,
            k,
            max_num_reqs,
            max_num_tokens,
            128,
            sample_from_anchor=True,
        )
        torch.accelerator.synchronize()
        return bufs, out, sample_pos

    bufs, out, sample_pos = prepare(num_rejected_b=4, num_sampled_b=1)
    # Request 1: every row rejected, draft zero at position 20.
    assert bufs.positions[3:6].tolist() == [20, 21, 22]
    assert bufs.input_ids[3:6].tolist() == [98, 123, 123]
    assert (out.context_slots[4:8] == PAD_SLOT_ID).all()
    assert sample_pos[3:6].tolist() == [21, 22, 23]
    base, base_out, base_pos = prepare(num_rejected_b=2, num_sampled_b=2)
    for neighbour in (slice(0, 3), slice(6, 9)):
        assert torch.equal(bufs.positions[neighbour], base.positions[neighbour])
        assert torch.equal(bufs.input_ids[neighbour], base.input_ids[neighbour])
        assert torch.equal(out.query_slots[neighbour], base_out.query_slots[neighbour])
    for ctx in (slice(0, 4), slice(8, 12)):
        assert torch.equal(out.context_slots[ctx], base_out.context_slots[ctx])


# The collapse sequence through the real sampler and rejection sampler.

VOCAB = 2048
HOW, CAN, I_, HELP, YOU, TODAY, QM, DOT = 101, 7, 102, 103, 104, 105, 106, 107
SAY_HI_END = 50
SUNK = frozenset({CAN})


def _chain(start: int, length: int = 8) -> dict[int, int]:
    return {start + i: start + i + 1 for i in range(length)} | {
        start + length: start + length
    }


# The greedy chain of the scripted model (a Markov chain over tokens): the
# ``Say hi.`` reply, and three clean ones.
CHAIN = {SAY_HI_END: HOW, HOW: CAN, CAN: 1500, I_: HELP, HELP: YOU, YOU: TODAY}
CHAIN |= {TODAY: QM, QM: DOT, DOT: DOT}
CHAIN |= _chain(600) | _chain(700) | _chain(800)
# The runner-up of each step: `` I`` after `` How``.
RUNNER_UP = {HOW: I_}
REPLY = [HOW, I_, HELP, YOU, TODAY, QM, DOT]


def _table() -> torch.Tensor:
    """Logits of the next token given the input token: the chain's token by a
    wide margin, then the runner-up, then noise."""
    gen = torch.Generator(device=DEVICE).manual_seed(11)
    table = torch.randn(VOCAB, VOCAB, generator=gen, device=DEVICE) * 0.1
    for token, nxt in CHAIN.items():
        table[token, nxt] = 40.0
        table[token, RUNNER_UP.get(token, 1000 + token % 900)] = 30.0
    return table


class MiniRunner:
    """The V2 runner's sampling layer for a few decoding requests: inputs
    [draft zero, drafts], logits from the scripted model, the detector's
    columns, the real sampler / rejection sampler, the policy and the
    request-state update; and the scheduler's side (commit all the step's
    tokens but the last, hold the last)."""

    def __init__(
        self,
        requests: dict[str, tuple[int, float]],
        drafter_errors: dict[tuple[str, int], int] | None = None,
        k: int = 2,
    ):
        self.k = k
        self.req_ids = list(requests)
        self.prompt_end = {r: end for r, (end, _) in requests.items()}
        self.states = RequestState(8, 128, 64, k, VOCAB, DEVICE)
        self.sampler = Sampler(
            8, VOCAB, DEVICE, self.states, num_speculative_tokens=k + 1
        )
        spec = SimpleNamespace(
            num_speculative_tokens=k,
            enable_adaptive_verification=False,
            rejection_sample_method="block",
            synthetic_acceptance_rates=None,
        )
        self.rejection = RejectionSampler(self.sampler, spec, DEVICE)
        self.sink = SinkSampler(REJECTION, 8, 5120, VOCAB, k, DEVICE, False)
        self.sampler.use_flashinfer = False
        self.sampler.sink = self.sink
        self.rejection.sink = self.sink
        self.drafter_errors = drafter_errors or {}
        self.prompts = {}
        for i, (req_id, (end, temperature)) in enumerate(requests.items()):
            prompt = list(range(10)) + [end]
            self.prompts[req_id] = prompt
            self.states.add_request(req_id, len(prompt), prompt, len(prompt) - 1, 64)
            self.sampler.add_request(
                self.states.req_id_to_index[req_id],
                len(prompt),
                SamplingParams(temperature=temperature, seed=1234 + i),
            )
        self.states.apply_staged_writes()
        self.sampler.apply_staged_writes()
        self.committed: dict[str, list[int]] = {r: [] for r in self.req_ids}
        self.held: dict[str, int] = {}
        self.events: list[dict] = []
        self.decoding = torch.zeros(8, dtype=torch.int8, device=DEVICE)

    def _batch(self, inputs: dict[str, list[int]]):
        counts = [len(inputs[r]) for r in self.req_ids]
        idx_np = np.array([self.states.req_id_to_index[r] for r in self.req_ids])
        idx = torch.from_numpy(idx_np).to(DEVICE)
        computed = self.states.num_computed_tokens.gpu[idx]
        positions = torch.cat(
            [computed[i] + torch.arange(c, device=DEVICE) for i, c in enumerate(counts)]
        )
        cu_np = np.cumsum([0] + counts).astype(np.int32)
        cu = torch.from_numpy(cu_np).to(DEVICE)
        tokens = [t for r in self.req_ids for t in inputs[r]]
        return SimpleNamespace(
            num_reqs=len(self.req_ids),
            input_ids=torch.tensor(tokens, dtype=torch.int32, device=DEVICE),
            positions=positions.to(torch.int64),
            logits_indices=torch.arange(int(cu_np[-1]), device=DEVICE),
            cu_num_logits=cu,
            cu_num_logits_np=cu_np,
            query_start_loc=cu,
            idx_mapping=idx,
            idx_mapping_np=idx_np,
            expanded_idx_mapping=torch.from_numpy(np.repeat(idx_np, counts)).to(DEVICE),
            expanded_local_pos=torch.cat(
                [torch.arange(c, dtype=torch.int32, device=DEVICE) for c in counts]
            ),
            seq_lens=(computed + torch.tensor(counts, device=DEVICE)).to(torch.int32),
            tokens=tokens,
        )

    def _drafts(self, req_id: str, token: int) -> list[int]:
        drafts = []
        for _ in range(self.k):
            token = self.drafter_errors.get((req_id, token), CHAIN.get(token, DOT))
            drafts.append(token)
        return drafts

    def step(self, table: torch.Tensor, decoding: bool) -> None:
        last = self.states.last_sampled_tokens[:, 0].tolist()
        inputs = {}
        for req_id in self.req_ids:
            if decoding:
                token = last[self.states.req_id_to_index[req_id]]
                inputs[req_id] = [token] + self._drafts(req_id, token)
            else:
                inputs[req_id] = [self.prompt_end[req_id]]
        batch = self._batch(inputs)
        logits = table[batch.input_ids.long()].clone()
        columns = torch.tensor(
            [[SUNK_VALUE if t in SUNK else CLEAN_VALUE] * 4 for t in batch.tokens],
            device=DEVICE,
        )
        flags = self.sink.flags(
            columns, batch.logits_indices, batch.cu_num_logits, batch.num_reqs, None
        )
        if decoding:
            output = self.rejection(logits, batch, None, None, sink_flags=flags)
        else:
            output = self.sampler(logits, batch)
        self.decoding[: batch.num_reqs] = int(decoding)
        result = self.sink.apply(
            output.sampled_token_ids,
            output.num_sampled,
            output.num_rejected,
            batch.cu_num_logits,
            batch.idx_mapping,
            self.decoding[: batch.num_reqs],
            flags,
            batch.input_ids,
            output.sink_alternates,
            self.states.total_len.gpu,
            self.sampler.penalties_state.output_bin_counts,
        )
        post_update(
            batch.idx_mapping,
            self.states.num_computed_tokens.gpu,
            self.states.last_sampled_tokens,
            self.sampler.penalties_state.output_bin_counts,
            output.sampled_token_ids,
            output.num_sampled,
            output.num_rejected,
            batch.query_start_loc,
            self.states.all_token_ids.gpu,
            self.states.total_len.gpu,
        )
        assert result.tokens is not None and result.num_tokens is not None
        tokens = result.tokens.tolist()
        num = result.num_tokens.tolist()
        events = events_to_dict(result.events.tolist(), self.req_ids) or {}
        self.events.append(events)
        for i, req_id in enumerate(self.req_ids):
            out = tokens[i][: num[i]]
            if decoding:
                if len(out) == 1:
                    # Draft zero rejected, replaced by the step's one token.
                    assert events[req_id].outcome == SinkOutcome.REJECTED
                else:
                    assert out[0] == self.held[req_id]
            self.committed[req_id].extend(out[:-1])
            self.held[req_id] = out[-1]

    def run(self, table: torch.Tensor, steps: int) -> None:
        self.step(table, decoding=False)
        for _ in range(steps):
            self.step(table, decoding=True)

    def state_of(self, req_id: str):
        slot = self.states.req_id_to_index[req_id]
        total = self.states.total_len.gpu[slot].item()
        return (
            self.states.num_computed_tokens.gpu[slot].item(),
            total,
            self.states.all_token_ids.gpu[slot, :total].tolist(),
            self.states.last_sampled_tokens[slot].item(),
        )


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_collapse_sequence_draft_zero_rejected(temperature: float):
    """``How can``: the drafter misses `` can`` (it proposes 999), the step
    samples it as its last token, the next step rejects it as draft zero and
    replaces it by `` I``, its draw's next candidate. Nothing after `` can``
    was committed, and the request state does not keep it."""
    runner = MiniRunner(
        {"a": (SAY_HI_END, temperature)}, drafter_errors={("a", HOW): 999}
    )
    runner.run(_table(), steps=6)
    assert runner.committed["a"][: len(REPLY)] == REPLY
    events = [e["a"] for e in runner.events if "a" in e]
    assert [e.outcome for e in events] == [SinkOutcome.REJECTED]
    computed, _, tokens, last = runner.state_of("a")
    prompt = runner.prompts["a"]
    # Every committed token is computed; the held one is the state's last.
    assert computed == len(prompt) + len(runner.committed["a"])
    assert tokens == prompt + runner.committed["a"] + [runner.held["a"]]
    assert last == runner.held["a"]
    assert CAN not in tokens[len(prompt) :]


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_collapse_sequence_drafter_cut(temperature: float):
    """The drafter proposes `` can`` behind a clean draft zero: its own row
    is flagged, the verification is cut there and the position recovered
    without it."""
    runner = MiniRunner({"a": (SAY_HI_END, temperature)})
    runner.run(_table(), steps=5)
    assert runner.committed["a"][: len(REPLY)] == REPLY
    events = [e["a"] for e in runner.events if "a" in e]
    assert [(e.outcome, e.cut_row) for e in events] == [(SinkOutcome.NONE, 1)]


def test_rejection_leaves_the_batch_neighbours_unchanged():
    """Three requests, greedy and random; the middle one rejects its draft
    zero. The other two commit the same tokens and end in the same request
    state as when the middle one never meets a sink."""

    def run(sink: bool) -> MiniRunner:
        runner = MiniRunner(
            {
                "a": (600, 0.0),
                "b": (SAY_HI_END if sink else 800, 0.0),
                "c": (700, 1.0),
            },
            drafter_errors={("b", HOW): 999},
        )
        runner.run(_table(), steps=6)
        return runner

    with_sink = run(True)
    without = run(False)
    for req_id in ("a", "c"):
        assert with_sink.committed[req_id] == without.committed[req_id]
        assert with_sink.state_of(req_id) == without.state_of(req_id)
    assert with_sink.committed["a"][:4] == [601, 602, 603, 604]
    assert with_sink.committed["b"][: len(REPLY)] == REPLY
    assert [e["b"].outcome for e in with_sink.events if "b" in e] == [
        SinkOutcome.REJECTED
    ]
    assert not any("b" in e for e in without.events)


# --- The alternates' logprobs and ranks ---------------------------------------


def _alternate_scores_reference(sink, logits, cu, num_sampled, alternates, mode):
    """``alternate_logprobs_of`` before its port: torch's row index math, gather,
    clamps and column copies around ``compute_topk_scores`` of each alternate."""
    from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores

    num_reqs = alternates.shape[0]
    rows = cu[:num_reqs].to(torch.int64) + (num_sampled.to(torch.int64) - 1).clamp(
        min=0
    )
    row_logits = logits[rows]
    logprobs = torch.empty(
        (num_reqs, sink.num_alternates), dtype=torch.float32, device=DEVICE
    )
    ranks = torch.empty(
        (num_reqs, sink.num_alternates), dtype=torch.int64, device=DEVICE
    )
    for i in range(sink.num_alternates):
        tokens = alternates[:, i].clamp(min=0)
        scores = compute_topk_scores(row_logits, 0, tokens, None, logits_mode=mode)
        logprobs[:, i] = scores.logprobs[:, 0]
        ranks[:, i] = scores.selected_token_ranks
    return logprobs, ranks


def _alternates_case(vocab: int, stride: int, dtype: torch.dtype, seed: int):
    """Seven requests' logits rows (in a wider row stride), with NaN, infinities and
    ties in the rows the alternates are drawn from, their sampled counts (a prefill
    chunk's 0 included), and alternates with -1 (no further candidate) among them."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    counts = [6, 1, 6, 3, 2, 6, 1]
    cu = _cu(counts)
    rows = int(cu[-1])
    storage = (torch.randn(rows, stride, generator=g, device=DEVICE) * 4).to(dtype)
    logits = storage[:, :vocab]
    num_sampled = torch.tensor([3, 0, 6, 1, 2, 4, 1], dtype=torch.int32, device=DEVICE)
    # The drawing rows: cu[r] + max(num_sampled[r] - 1, 0) = 2, 6, 12, 13, 17, 21, 24.
    logits[2, 7] = float("nan")  # request 0's
    logits[12, 3] = float("inf")  # request 2's
    logits[12, 4] = -float("inf")
    logits[13, :] = logits[13, 0]  # request 3's: every column tied
    alternates = torch.randint(0, vocab, (7, 3), generator=g, device=DEVICE)
    alternates[1, 2] = -1
    alternates[4, 1:] = -1
    alternates[0, 0] = 7  # the NaN column
    return logits, cu, num_sampled, alternates


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("logits_mode", [False, True])
@pytest.mark.parametrize(
    "vocab,stride", [(129280, 129280), (129280, 129408), (1000, 1003)]
)
def test_alternate_logprobs_match_the_torch_glue(dtype, logits_mode, vocab, stride):
    """``alternate_logprobs_of`` (one copy launch, one log-softmax launch, a rank
    launch an alternate) gives the bits of the torch glue it replaces, which scored
    each alternate with its own log-softmax and rank launches and copied their
    columns: logprobs (or, in a logits mode, logits) and ranks, at DS4.1's
    vocabulary in a dense and a padded (TP3) row stride and at a width that is not
    16-element aligned."""
    sink = _sink(vocab=vocab)
    assert sink.num_alternates == 3
    logits, cu, num_sampled, alternates = _alternates_case(vocab, stride, dtype, 31)
    before = logits.clone()
    out = sink.alternate_logprobs_of(logits, cu, num_sampled, alternates, logits_mode)
    logprobs, ranks = _alternate_scores_reference(
        sink, logits, cu, num_sampled, alternates, logits_mode
    )
    assert out.tokens is alternates
    assert out.logprobs is not None and out.ranks is not None
    assert out.logprobs.shape == out.ranks.shape == (7, 3)
    assert out.logprobs.dtype == torch.float32 and out.ranks.dtype == torch.int64
    assert torch.equal(out.logprobs.view(torch.int32), logprobs.view(torch.int32))
    assert torch.equal(out.ranks, ranks)
    assert torch.equal(logits.view(torch.uint8), before.view(torch.uint8))


def test_alternate_logprobs_of_an_empty_batch():
    sink = _sink()
    logits = torch.randn(0, 2048, device=DEVICE)
    out = sink.alternate_logprobs_of(
        logits,
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
        torch.zeros(0, dtype=torch.int32, device=DEVICE),
        torch.zeros((0, 3), dtype=torch.int64, device=DEVICE),
        False,
    )
    assert out.logprobs.shape == out.ranks.shape == (0, 3)


def test_alternate_logprobs_feed_the_policy():
    """The policy keeps a step's alternate logprobs and ranks as the sink computed
    them (its ranks alternate by alternate) and recovers them for a rejected draft
    zero: the same state as from request-major copies of them."""
    results = []
    for contiguous in (False, True):
        sink = _sink()
        logits, cu, num_sampled, alternates = _alternates_case(
            2048, 2048, torch.float32, 5
        )
        out = sink.alternate_logprobs_of(logits, cu, num_sampled, alternates, False)
        if contiguous:
            out = SinkAlternates(
                out.tokens, out.logprobs.contiguous(), out.ranks.contiguous()
            )
        else:
            assert not out.ranks.is_contiguous()
        num_reqs = 7
        sampled = torch.zeros((num_reqs, 3), dtype=torch.int64, device=DEVICE)
        flags = torch.zeros(int(cu[-1]), dtype=torch.int8, device=DEVICE)
        result = sink.apply(
            sampled,
            num_sampled.clone(),
            torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
            cu,
            torch.arange(num_reqs, device=DEVICE),
            torch.ones(num_reqs, dtype=torch.int8, device=DEVICE),
            flags,
            torch.zeros(int(cu[-1]), dtype=torch.int32, device=DEVICE),
            out,
            torch.full((8,), 40, dtype=torch.int32, device=DEVICE),
            torch.ones((8, 2048), dtype=torch.int32, device=DEVICE),
            torch.zeros(int(cu[-1]), device=DEVICE),
        )
        results.append(
            (
                sink.alternate_logprobs[:num_reqs].clone(),
                sink.alternate_ranks[:num_reqs].clone(),
                result.recovered_logprobs,
                result.recovered_ranks,
            )
        )
    (lp, rk, rec_lp, rec_rk), (lp2, rk2, rec_lp2, rec_rk2) = results
    assert torch.equal(lp.view(torch.int32), lp2.view(torch.int32))
    assert torch.equal(rk, rk2)
    # Nothing was recovered: the policy wrote zeros for every request.
    assert torch.equal(rec_lp, torch.zeros_like(rec_lp))
    assert torch.equal(rec_rk, torch.zeros_like(rec_rk))
    assert torch.equal(rec_lp2, rec_lp) and torch.equal(rec_rk2, rec_rk)


def test_alternate_logprobs_kernels_are_declared():
    """The sink declares the alternates' copy (each logits dtype, the logits mode
    given), and the log-softmax and ranks of their dense rows."""
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.models.deepseek_v4_1.tilelang import prepare
    from vllm.v1.worker.gpu.sample import tilelang_logprob, tilelang_sampling

    vocab = 129280
    for logits_mode in (False, True):
        sink = _sink(vocab=vocab)
        registry = JitWarmupRegistry(None)
        with registry.activate():
            sink.register_warmups(
                logits=((torch.bfloat16, 129408),),
                processed_logits_strides=(vocab, 129408),
                sampled_widths=(1, 6),
                bin_counts_stride=vocab,
                vocab_size=vocab,
                logits_mode=logits_mode,
            )
        calls = prepare.declared_calls(registry)
        # By the factory's identity: TileLang's JIT functions are not hashable.
        keys = {(id(f), args) for f, args, _ in calls}
        for dtype in (torch.bfloat16, torch.float32):
            assert (
                id(tilelang_sampling.sink_rows_kernel),
                tilelang_sampling.sink_rows_key(dtype, 3, logits_mode),
            ) in keys
            _, rank_args, _ = tilelang_logprob.rank_call(
                dtype, vocab, vocab, torch.int64
            )
            assert (id(tilelang_logprob.ranks_kernel), rank_args) in keys
            dense = torch.empty((1, vocab), dtype=dtype, device=DEVICE)
            softmax = (
                id(tilelang_logprob.topk_log_softmax_kernel),
                tilelang_logprob.topk_log_softmax_key(
                    dtype, tilelang_sampling._triton_run(dense)
                ),
            )
            assert (softmax in keys) != logits_mode
