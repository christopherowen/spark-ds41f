# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy drafts: the TileLang kernels (``spec_decode.tilelang_copy``) against numpy
references, the proposer (``spec_decode.copy_drafts.CopyDrafter``) against the
design's walk, and temperature-0 output with and without copy drafts.

- ``copy_match``/``copy_alloc``/``copy_apply`` against references written from
  the definition: the strongest suffix match, preferring a full continuation,
  then recency; short-only matches remain available; the copy state's update,
  the allocation and the proposal.
  Contexts from 1 token to past two of the match grid's passes, vocabularies of
  3 to 129280 tokens (few to rare matches), batches in any slot order.
- Nothing but the request's own row up to its length is read: a slot's stale
  tokens past the length of its new request, and other requests' text in the
  same batch, never propose; a slot's copy state starts afresh with each request.
- The kernels replay in a CUDA graph with new inputs as they run eagerly.
- Preparation and proposals follow the current model stream, including when
  that stream is busy. Async allocation delivery survives delayed copies and
  repeated ring wrap under the runner's two-in-flight-output bound.
- The proposer, driven as the engine lays out steps (its allocation a step
  late), takes the steps experiments/2026-10-26-suffix-drafter/variants.py's walk
  takes for the go variant (exact, minimum match 16, at most 16 drafts, the cap
  growing from 8): the same proposals, caps, allocations and tokens.
- A toy target whose next token is a function of the context alone (so a row's
  logits do not depend on its batch, as DeepSeek V4.1's decode chain is built),
  verified by the runner's greedy verification kernels: the committed text is the
  target's own greedy text with DSpark-like drafts alone and with copy drafts,
  over prompts that copy a lot, a little and not at all.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("tilelang")

from vllm.config.speculative import CopyDraftsConfig  # noqa: E402
from vllm.v1.worker.gpu.spec_decode import tilelang_copy as tlc  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.copy_drafts import CopyDrafter  # noqa: E402
from vllm.v1.worker.gpu.states import RequestState  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

DEVICE = torch.device("cuda")
MIN_MATCH = 16
LOOKAHEAD = 16
BLOCK = 5  # DSpark's drafts
INITIAL = 8
VOCAB = 129280
MAX_LEVEL = CopyDraftsConfig.MAX_LEVEL
MAX_REQS = 16


# --- References -----------------------------------------------------------------


def ref_source(
    S: np.ndarray, p: int, min_match: int = MIN_MATCH, lookahead: int = LOOKAHEAD
) -> int:
    """The best continuation at an end ``e < p`` whose ``L`` preceding tokens
    equal the context's last ``L`` (``S[e - L:e] == S[p - L:p]``), at the
    longest level ``L`` that has one, else -1. Rank by continuation capped at
    lookahead, then latest end. The context is ``S[:p]``."""
    levels = CopyDraftsConfig(min_match=min_match).levels()
    if p <= min_match:
        return -1
    ends = np.arange(min_match, p)
    width = min(MAX_LEVEL, p)
    key = S[p - width : p][::-1]
    run = np.zeros(ends.shape, dtype=np.int64)
    alive = np.ones(ends.shape, dtype=bool)
    for j in range(width):
        index = ends - 1 - j
        valid = alive & (index >= 0)
        alive = valid & (S[np.maximum(index, 0)] == key[j])
        run += alive
    source = -1
    for level in levels:
        hits = ends[run >= level]
        if hits.size:
            source = int(max(hits, key=lambda e: (min(lookahead, p - int(e)), int(e))))
    return source


class RefState:
    """``copy_alloc``'s copy state of one slot, as the kernel keeps it."""

    def __init__(self):
        self.doublings, self.pending = 0, 0
        self.hits, self.trials = np.float32(0), np.float32(0)


def ref_select(
    state, source, length, num_sampled, verified, cut, dspark_yield=0.0
):
    """The state update and the proposal's (offered, after, estimate, alloc)."""
    if state.pending > 0 and num_sampled > 0:
        live = min(state.pending, verified, cut)
        if live > 0:
            accepted = min(num_sampled - 1, live)
            state.doublings = (
                min(state.doublings + 1, tlc.MAX_DOUBLINGS) if accepted >= live else 0
            )
            state.hits = np.float32(state.hits * np.float32(tlc.DECAY) + accepted)
            state.trials = np.float32(
                state.trials * np.float32(tlc.DECAY) + min(accepted + 1, live)
            )
    cap = min(INITIAL << state.doublings, LOOKAHEAD)
    if source >= 0:
        offered = min(cap, length - source)
        after = min(2 * cap, LOOKAHEAD, length - source)
    else:
        offered = after = 0
        state.doublings = 0
    alloc = max(after, BLOCK)
    estimate = float((state.hits + tlc.PRIOR_HITS) / (state.trials + tlc.PRIOR_TRIALS))
    if offered > 0 and np.float32(offered) * np.float32(estimate) < np.float32(
        dspark_yield
    ):
        offered = after = 0
        state.doublings = 0
    state.pending = offered
    return offered, after, estimate, alloc


# --- Streams ----------------------------------------------------------------------


def copy_stream(rng, vocab, doc=1000, edits=8):
    """A prompt with a document, and a reply that reproduces it with a few edits
    (the copy probe's categories, ideal replies)."""
    document = rng.integers(0, vocab, doc)
    reply = document.copy()
    for at in rng.integers(0, doc, edits):
        reply = np.concatenate([reply[:at], rng.integers(0, vocab, 3), reply[at:]])
    prompt = np.concatenate([rng.integers(0, vocab, 40), document, [7, 8, 9]])
    return np.concatenate([prompt, reply]).astype(np.int64), len(prompt)


def prose_stream(rng, vocab, length=600):
    return rng.integers(0, vocab, length).astype(np.int64), length // 3


# --- Kernels against the references --------------------------------------------


def _history(rows, width):
    return torch.full((rows, width), -9, dtype=torch.int32, device=DEVICE)


def _run_match(
    history, total_len, idx_mapping, min_match=MIN_MATCH, lookahead=LOOKAHEAD
):
    sources = torch.full(
        (idx_mapping.shape[0], tlc.CHUNKS, tlc.num_levels(min_match)),
        -7,
        dtype=torch.int32,
        device=DEVICE,
    )
    tlc.copy_match(idx_mapping, history, total_len, sources, min_match, lookahead)
    return sources


def _best(sources_row: torch.Tensor, length: int, lookahead: int = LOOKAHEAD) -> int:
    """copy_alloc's choice from one request's chunk sources."""
    source = -1
    for level in range(sources_row.shape[1]):
        latest = int(sources_row[:, level].max())
        if latest >= 0:
            source = latest
    return (
        source
        if source < 0
        else (source - lookahead if source >= lookahead else length - source)
    )


@pytest.mark.parametrize("vocab", [3, 64, VOCAB])
@pytest.mark.parametrize("min_match", [MIN_MATCH, 24, 8])
def test_copy_match_is_the_latest_longest_match(vocab, min_match):
    rng = np.random.default_rng(vocab + min_match)
    lengths = [1, 15, 16, 17, 40, 300, 8191, 8192, 8209, 16401, 20000]
    width = max(lengths) + 64
    history = _history(MAX_REQS, width)
    total_len = torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE)
    streams = {}
    slots = rng.permutation(MAX_REQS)[: len(lengths)]
    for slot, length in zip(slots, lengths):
        if vocab == VOCAB and length > 300:
            # Rare natural matches: plant copies, one of them at the tail.
            S = rng.integers(0, vocab, length)
            for _ in range(4):
                span = int(rng.integers(20, 300))
                src = int(rng.integers(0, length - 2 * span))
                dst = int(rng.integers(src + span, length - span))
                S[dst : dst + span] = S[src : src + span]
            S[-40:] = S[length // 2 : length // 2 + 40]
        else:
            S = rng.integers(0, vocab, length)
        streams[int(slot)] = S
        history[slot, :length] = torch.from_numpy(S.astype(np.int32)).to(DEVICE)
        total_len[slot] = length
    idx_mapping = torch.tensor(slots, dtype=torch.int64, device=DEVICE)
    sources = _run_match(history, total_len, idx_mapping, min_match)
    # Every chunk writes its sources, those past a short context's end too.
    assert (sources >= -1).all()
    for r, slot in enumerate(slots):
        S = streams[int(slot)]
        expected = ref_source(S, len(S), min_match)
        assert _best(sources[r], len(S)) == expected, (r, int(slot), len(S))


def test_copy_match_reads_only_its_request():
    """A slot's stale tokens past its new request's length and another request's
    text in the same batch never propose."""
    rng = np.random.default_rng(1)
    width = 4096
    history = _history(4, width)
    total_len = torch.zeros(4, dtype=torch.int32, device=DEVICE)
    prompt = rng.integers(0, VOCAB, 600)
    tail = prompt[-MIN_MATCH:]
    # Request B (slot 2): its tail occurs once, early, in its own prompt.
    b = prompt.copy()
    b[100 : 100 + MIN_MATCH] = tail
    # The previous request in slot 2 left B's tail later, past B's length.
    stale = rng.integers(0, VOCAB, width)
    stale[1500 : 1500 + MIN_MATCH] = tail
    history[2] = torch.from_numpy(stale.astype(np.int32)).to(DEVICE)
    history[2, : len(b)] = torch.from_numpy(b.astype(np.int32)).to(DEVICE)
    total_len[2] = len(b)
    # Request C (slot 0), in the same batch, holds B's tail many times.
    c = rng.integers(0, VOCAB, 3000)
    for at in (200, 900, 2500):
        c[at : at + MIN_MATCH] = tail
    history[0, : len(c)] = torch.from_numpy(c.astype(np.int32)).to(DEVICE)
    total_len[0] = len(c)
    idx_mapping = torch.tensor([0, 2], dtype=torch.int64, device=DEVICE)
    sources = _run_match(history, total_len, idx_mapping)
    assert _best(sources[1], len(b)) == 100 + MIN_MATCH == ref_source(b, len(b))
    assert _best(sources[0], len(c)) == ref_source(c, len(c))
    # Without the early copy, B has no source at all.
    b2 = prompt.copy()
    history[2, : len(b2)] = torch.from_numpy(b2.astype(np.int32)).to(DEVICE)
    sources = _run_match(history, total_len, idx_mapping)
    assert ref_source(b2, len(b2)) == -1
    assert _best(sources[1], len(b)) == -1


def _propose(
    history,
    total_len,
    idx_mapping,
    sources,
    int_state,
    float_state,
    num_sampled,
    verified,
    cut,
    dspark=None,
):
    """copy_alloc then copy_apply over one step: the proposals and the
    allocation."""
    n = idx_mapping.shape[0]
    if dspark is None:
        dspark = (
            torch.arange(n * BLOCK, device=DEVICE).view(n, BLOCK) + 1000,
            torch.full((n, BLOCK), 0.5, device=DEVICE),
        )
    chosen = torch.full((n, tlc.CHOSEN_COLUMNS), -5, dtype=torch.int32, device=DEVICE)
    estimates = torch.full((n,), -5.0, device=DEVICE)
    drafts = torch.full((n, LOOKAHEAD), -5, dtype=torch.int64, device=DEVICE)
    confidences = torch.full((n, LOOKAHEAD), -5.0, device=DEVICE)
    planned = torch.full((n, LOOKAHEAD), -5.0, device=DEVICE)
    alloc = torch.full((n,), -5, dtype=torch.int32, device=DEVICE)
    tlc.copy_alloc(
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
        MIN_MATCH,
        LOOKAHEAD,
        BLOCK,
        INITIAL,
    )
    tlc.copy_apply(
        idx_mapping,
        history,
        chosen,
        estimates,
        int_state,
        dspark[0],
        dspark[1],
        drafts,
        confidences,
        planned,
    )
    return drafts, confidences, planned, alloc, dspark


def _state(rows=MAX_REQS):
    return (
        torch.zeros((rows, tlc.STATE_COLUMNS), dtype=torch.int32, device=DEVICE),
        torch.zeros((rows, tlc.STATE_COLUMNS), dtype=torch.float32, device=DEVICE),
    )


def test_copy_alloc_and_apply_follow_the_reference_over_a_copy():
    """One request copying a document: proposals, the cap's growth and reset, the
    estimate, the planned confidences and the allocation, step by step."""
    rng = np.random.default_rng(2)
    S, start = copy_stream(rng, VOCAB, doc=600, edits=4)
    slot = 5
    history = _history(MAX_REQS, len(S) + 64)
    history[slot, : len(S)] = torch.from_numpy(S.astype(np.int32)).to(DEVICE)
    total_len = torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE)
    int_state, float_state = _state()
    ref = RefState()
    idx_mapping = torch.tensor([slot], dtype=torch.int64, device=DEVICE)
    p, num_sampled, verified = start, 1, 0
    copies = 0
    while p < len(S):
        total_len[slot] = p
        cut = verified if p % 7 else max(verified - 1, 0)  # a dead-row cut now and then
        sources = _run_match(history, total_len, idx_mapping)
        args = [
            torch.tensor([v], dtype=torch.int32, device=DEVICE)
            for v in (num_sampled, verified, cut)
        ]
        drafts, confidences, planned, alloc, dspark = _propose(
            history, total_len, idx_mapping, sources, int_state, float_state, *args
        )
        source = ref_source(S, p)
        offered, after, estimate, ref_alloc = ref_select(
            ref,
            source,
            p,
            num_sampled,
            verified,
            cut,
            dspark_yield=float(np.cumprod(dspark[1][0].cpu().numpy(), dtype=np.float32).sum()),
        )
        assert int(alloc[0]) == ref_alloc
        assert int(int_state[slot, 0]) == ref.doublings
        assert int(int_state[slot, 1]) == ref.pending
        torch.testing.assert_close(
            float_state[slot].cpu(),
            torch.tensor([ref.hits, ref.trials]),
            rtol=1e-6,
            atol=1e-5,
        )
        d = drafts[0].cpu().numpy()
        c = confidences[0].cpu().numpy()
        w = planned[0].cpu().numpy()
        if offered > 0:
            copies += 1
            np.testing.assert_array_equal(d[:offered], S[source : source + offered])
            assert (d[offered:] == -1).all()
            np.testing.assert_allclose(c[:offered], estimate, rtol=1e-6)
            assert (c[offered:] == 0).all()
            np.testing.assert_allclose(w[:after], estimate, rtol=1e-6)
            assert (w[after:] == 0).all()
            accepted = 0
            while (
                accepted < offered
                and p + accepted < len(S)
                and S[source + accepted] == S[p + accepted]
            ):
                accepted += 1
            num_sampled, verified = accepted + 1, offered
        else:
            np.testing.assert_array_equal(d[:BLOCK], dspark[0][0].cpu().numpy())
            assert (d[BLOCK:] == -1).all()
            np.testing.assert_array_equal(c[:BLOCK], dspark[1][0].cpu().numpy())
            assert (c[BLOCK:] == 0).all()
            np.testing.assert_array_equal(w, c)
            num_sampled, verified = 1 + int(rng.integers(0, 3)), BLOCK
        p += num_sampled
    # The reply copies its 600-token document a step of up to 16 drafts at a time.
    assert copies > 25


def test_copy_alloc_judges_only_its_pending_copy():
    """No proposal pending, no token sampled (a prefill chunk) or no draft
    verified: the state stays; a sunk draft zero (one token committed of a
    verified block) resets the cap."""
    rng = np.random.default_rng(3)
    S = np.tile(rng.integers(0, VOCAB, 50), 6)
    slot = 1
    history = _history(4, len(S) + 64)
    history[slot, : len(S)] = torch.from_numpy(S.astype(np.int32)).to(DEVICE)
    total_len = torch.tensor([0, len(S), 0, 0], dtype=torch.int32, device=DEVICE)
    idx_mapping = torch.tensor([slot], dtype=torch.int64, device=DEVICE)
    sources = _run_match(history, total_len, idx_mapping)

    def step(int_state, float_state, num_sampled, verified, cut):
        args = [
            torch.tensor([v], dtype=torch.int32, device=DEVICE)
            for v in (num_sampled, verified, cut)
        ]
        _propose(
            history, total_len, idx_mapping, sources, int_state, float_state, *args
        )
        return int(int_state[slot, 0]), int(int_state[slot, 1])

    int_state, float_state = _state(4)
    int_state[slot] = torch.tensor([2, 16], dtype=torch.int32)
    float_state[slot] = torch.tensor([30.0, 31.0])
    for num_sampled, verified, cut in ((0, 16, 16), (1, 0, 0), (5, 16, 0)):
        before = float_state[slot].clone()
        assert step(int_state, float_state, num_sampled, verified, cut)[0] == 2
        torch.testing.assert_close(float_state[slot], before)
    # A sunk draft zero: the block rejected whole.
    assert step(int_state, float_state, 1, 16, 16) == (0, INITIAL)
    # All accepted: doubled to the lookahead and held there.
    assert step(int_state, float_state, 9, 8, 8) == (1, 16)
    assert step(int_state, float_state, 17, 16, 16) == (2, 16)


def test_kernels_replay_in_a_cuda_graph():
    width = 9000
    history = _history(MAX_REQS, width)
    total_len = torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE)
    idx_mapping = torch.tensor([3, 9, 0], dtype=torch.int64, device=DEVICE)
    n = idx_mapping.shape[0]
    levels = tlc.num_levels(MIN_MATCH)
    sources = torch.empty((n, tlc.CHUNKS, levels), dtype=torch.int32, device=DEVICE)
    int_state, float_state = _state()
    chosen = torch.empty((n, tlc.CHOSEN_COLUMNS), dtype=torch.int32, device=DEVICE)
    estimates = torch.empty((n,), device=DEVICE)
    alloc = torch.empty((n,), dtype=torch.int32, device=DEVICE)
    proposals = (
        torch.empty((n, LOOKAHEAD), dtype=torch.int64, device=DEVICE),
        torch.empty((n, LOOKAHEAD), device=DEVICE),
        torch.empty((n, LOOKAHEAD), device=DEVICE),
    )
    outs = (*proposals, chosen, estimates, alloc)
    dspark = (
        torch.randint(0, VOCAB, (n, BLOCK), device=DEVICE),
        torch.rand((n, BLOCK), device=DEVICE),
    )
    step = [torch.ones(n, dtype=torch.int32, device=DEVICE) for _ in range(3)]

    def launch():
        tlc.copy_match(idx_mapping, history, total_len, sources, MIN_MATCH, LOOKAHEAD)
        tlc.copy_alloc(
            idx_mapping, total_len, sources, *step, int_state, float_state,
            chosen, estimates, alloc, MIN_MATCH, LOOKAHEAD, BLOCK, INITIAL,
        )  # fmt: skip
        tlc.copy_apply(idx_mapping, history, chosen, estimates, int_state, *dspark, *proposals)

    def fill(seed):
        g = np.random.default_rng(seed)
        for slot in idx_mapping.tolist():
            length = int(g.integers(100, width))
            S = np.tile(g.integers(0, VOCAB, int(g.integers(20, 400))), width)[:length]
            history[slot, :length] = torch.from_numpy(S.astype(np.int32)).to(DEVICE)
            total_len[slot] = length

    fill(10)
    launch()  # compiles outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for seed in (11, 12, 13):
        fill(seed)
        saved = (int_state.clone(), float_state.clone())
        graph.replay()
        torch.cuda.synchronize()
        replayed = [t.clone() for t in (*outs, sources, int_state, float_state)]
        int_state.copy_(saved[0])
        float_state.copy_(saved[1])
        launch()
        torch.cuda.synchronize()
        for a, b in zip(replayed, (*outs, sources, int_state, float_state)):
            assert torch.equal(a, b)


# --- The proposer -------------------------------------------------------------------


def _request_state(max_len):
    states = RequestState(
        max_num_reqs=MAX_REQS,
        max_model_len=max_len,
        max_num_batched_tokens=4096,
        num_speculative_steps=LOOKAHEAD,
        vocab_size=VOCAB,
        device=DEVICE,
    )
    return states


def _drafter(states):
    config = CopyDraftsConfig(
        min_match=MIN_MATCH, max_drafts=LOOKAHEAD, initial_drafts=INITIAL
    )
    return CopyDrafter(config, states, num_draft_steps=BLOCK, device=DEVICE)


def _set_history(states, req_id, tokens):
    """The request state of a request whose history is ``tokens``, as the runner
    holds it after committing them (``post_update`` writes the same rows)."""
    if req_id not in states.req_id_to_index:
        states.add_request(req_id, len(tokens), list(map(int, tokens)), 0, 1 << 20)
    slot = states.req_id_to_index[req_id]
    states.all_token_ids.stage_write(slot, 0, list(map(int, tokens)))
    states.total_len.stage_write_elem(slot, len(tokens))
    states.apply_staged_writes()
    return slot


def _copy_run(S, p, cap, end):
    source = ref_source(S, p, 8)
    if source < 0:
        return 0
    a = 0
    while a < cap and p + a < end and source + a < p and S[source + a] == S[p + a]:
        a += 1
    return a


PROSE = (2, 2, 2, 3)


def design_walk(S, start, end):
    """variants.py's ``walk`` for the go variant: family ``exact``, ``m = 16``,
    ``k = 16``, policy ``grow`` with the allocation lag, over one turn. Each step
    is ``(kind, allocated rows, live rows, tokens)``."""
    steps = []
    p, prose_i, alloc, grow = start, 0, 1 + BLOCK, INITIAL
    while p < end:
        source = ref_source(S, p)
        length = p - source if source >= 0 else 0
        offered = min(LOOKAHEAD, length)
        if offered > 0:
            want = min(offered, grow)
            drafts = min(want, alloc - 1)
            accepted = 0
            while (
                accepted < drafts
                and p + accepted < end
                and S[source + accepted] == S[p + accepted]
            ):
                accepted += 1
            tokens = min(accepted + 1, end - p)
            steps.append(("suffix", alloc, 1 + drafts, tokens))
            predicted = min(min(LOOKAHEAD, 2 * grow), length)
            grow = min(LOOKAHEAD, 2 * grow) if accepted == drafts else INITIAL
            next_alloc = 1 + max(BLOCK, predicted)
        else:
            run = _copy_run(S, p, BLOCK, end)
            if run >= 1:
                tokens = min(BLOCK, run + 1, end - p)
                steps.append(("dspark_copy", alloc, 1 + BLOCK, tokens))
            else:
                tokens = min(PROSE[prose_i & 3], end - p)
                prose_i += 1
                steps.append(("dspark_prose", alloc, 0, tokens))
            grow = INITIAL
            next_alloc = 1 + BLOCK
        alloc = next_alloc
        p += tokens
    return steps


def engine_walk(drafter, states, S, start, end, req_id="r"):
    """The proposer over the same turn, as the engine drives it: the proposal
    verified at a step was made after the step before; the scheduler lays out the
    step after the proposal's verification from the allocation it reported (one
    step late); verification takes every offered draft that fits the step's rows;
    DSpark's steps take the walk's tokens."""
    steps = []
    p, prose_i = start, 0
    allocs = [1 + BLOCK]  # the rows of each step, from the report two proposals back
    num_sampled = verified = 0
    dspark = (
        torch.zeros((1, BLOCK), dtype=torch.int64, device=DEVICE),
        torch.full((1, BLOCK), 0.5, device=DEVICE),
    )
    while p < end:
        slot = _set_history(states, req_id, S[:p])
        idx = torch.tensor([slot], dtype=torch.int64, device=DEVICE)
        sampled = torch.tensor([num_sampled], dtype=torch.int32, device=DEVICE)
        judged = torch.tensor([verified], dtype=torch.int32, device=DEVICE)
        drafter.prepare_alloc(idx, sampled, judged, None)
        drafts, _, _ = drafter.propose(idx, *dspark)
        # The slot's pending copy: the drafts its proposal offers (0: DSpark's).
        offered = int(drafter.int_state[slot, 1])
        allocs.append(1 + int(drafter.alloc[0]))
        alloc = allocs[-2]
        if offered > 0:
            d = min(offered, alloc - 1)
            proposal = drafts[0, :d].cpu().numpy()
            accepted = 0
            while (
                accepted < d
                and p + accepted < end
                and proposal[accepted] == S[p + accepted]
            ):
                accepted += 1
            tokens = min(accepted + 1, end - p)
            steps.append(("suffix", alloc, 1 + d, tokens))
            num_sampled, verified = accepted + 1, d
        else:
            run = _copy_run(S, p, BLOCK, end)
            if run >= 1:
                tokens = min(BLOCK, run + 1, end - p)
                steps.append(("dspark_copy", alloc, 1 + BLOCK, tokens))
            else:
                tokens = min(PROSE[prose_i & 3], end - p)
                prose_i += 1
                steps.append(("dspark_prose", alloc, 0, tokens))
            num_sampled, verified = tokens, BLOCK
        p += tokens
    return steps


@pytest.mark.parametrize("seed", range(4))
def test_proposer_takes_the_designs_steps(seed):
    rng = np.random.default_rng(100 + seed)
    streams = [copy_stream(rng, VOCAB), prose_stream(rng, 64), copy_stream(rng, 300)]
    states = _request_state(8192)
    drafter = _drafter(states)
    for i, (S, start) in enumerate(streams):
        req_id = f"r{seed}-{i}"
        _set_history(states, req_id, S[:start])
        drafter.add_request(states.req_id_to_index[req_id])
        drafter.apply_staged_writes()
        assert engine_walk(drafter, states, S, start, len(S), req_id) == design_walk(
            S, start, len(S)
        )
        states.remove_request(req_id)


def test_a_slot_starts_afresh_with_each_request():
    """The previous request's copy state (its cap, acceptance counts and pending
    proposal) does not reach the slot's next request."""
    rng = np.random.default_rng(5)
    states = _request_state(8192)
    drafter = _drafter(states)
    S, start = copy_stream(rng, VOCAB, doc=800, edits=0)
    _set_history(states, "a", S[:start])
    slot = states.req_id_to_index["a"]
    drafter.add_request(slot)
    drafter.apply_staged_writes()
    engine_walk(drafter, states, S, start, len(S), "a")
    assert int(drafter.int_state[slot, 0]) > 0
    assert float(drafter.float_state[slot, 1]) > 0
    states.remove_request("a")
    T, tstart = copy_stream(rng, VOCAB, doc=200, edits=0)
    _set_history(states, "b", T[: tstart + 30])
    assert states.req_id_to_index["b"] == slot
    drafter.add_request(slot)
    drafter.apply_staged_writes()
    assert int(drafter.int_state[slot].abs().sum()) == 0
    assert float(drafter.float_state[slot].abs().sum()) == 0
    idx = torch.tensor([slot], dtype=torch.int64, device=DEVICE)
    one = torch.ones(1, dtype=torch.int32, device=DEVICE)
    dspark = (
        torch.zeros((1, BLOCK), dtype=torch.int64, device=DEVICE),
        torch.zeros((1, BLOCK), device=DEVICE),
    )
    drafter.prepare_alloc(idx, one, one, None)
    drafts, confidences, planned = drafter.propose(idx, *dspark)
    # B's first copy: the initial cap, at the prior's estimate.
    source = ref_source(T, tstart + 30)
    assert source >= 0
    np.testing.assert_array_equal(
        drafts[0, :INITIAL].cpu().numpy(), T[source : source + INITIAL]
    )
    np.testing.assert_allclose(
        confidences[0, :INITIAL].cpu().numpy(),
        tlc.PRIOR_HITS / tlc.PRIOR_TRIALS,
        rtol=1e-6,
    )
    assert int(drafter.alloc[0]) == 2 * INITIAL


def test_prepare_alloc_writes_alloc_before_propose():
    """prepare_alloc writes the allocation before DSpark runs; propose leaves it
    as it is."""
    rng = np.random.default_rng(7)
    states = _request_state(8192)
    drafter = _drafter(states)
    S, start = copy_stream(rng, VOCAB, doc=400, edits=0)
    P, pstart = prose_stream(rng, VOCAB, length=300)
    slot_copy = _set_history(states, "copy", S[: start + 30])
    slot_prose = _set_history(states, "prose", P[:pstart])
    drafter.add_request(slot_copy)
    drafter.add_request(slot_prose)
    drafter.apply_staged_writes()

    idx = torch.tensor([slot_copy, slot_prose], dtype=torch.int64, device=DEVICE)
    one = torch.ones(2, dtype=torch.int32, device=DEVICE)
    zero = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    dspark = (
        torch.full((2, BLOCK), 42, dtype=torch.int64, device=DEVICE),
        torch.full((2, BLOCK), 0.6, device=DEVICE),
    )
    drafter.prepare_alloc(idx, one, zero, None)
    early_alloc = drafter.alloc[:2].clone()

    drafts, confidences, planned = drafter.propose(idx, *dspark)
    assert torch.equal(early_alloc, drafter.alloc[:2])
    assert int(early_alloc[0]) == 2 * INITIAL
    assert int(early_alloc[1]) == BLOCK
    assert (drafts[0, INITIAL:] == -1).all()
    assert (drafts[1, BLOCK:] == -1).all()
    assert (confidences[0, INITIAL:] == 0).all()
    assert (confidences[1, BLOCK:] == 0).all()
    assert (planned[0, : 2 * INITIAL] > 0).all()



# --- Temperature 0: the committed text does not depend on the drafts ------------


class ToyTarget:
    """A target whose next token is a function of the context alone: a table
    lookup of a hash of its last ``order`` tokens (and, for ``noisy``, of the
    position), so its rows are batch-invariant. A small order and table cycle
    soon (much copying), a position-dependent one hardly copies at all."""

    def __init__(self, seed, order, table, noisy=False):
        rng = np.random.default_rng(seed)
        self.order, self.noisy = order, noisy
        self.table = rng.integers(0, VOCAB, table)

    def next(self, context: list[int]) -> int:
        h = 1469598103934665603
        for t in context[-self.order :]:
            h = ((h ^ t) * 1099511628211) & ((1 << 64) - 1)
        if self.noisy:
            h ^= len(context) // 37
        return int(self.table[h % len(self.table)])

    def greedy(self, prompt, count):
        out = list(map(int, prompt))
        for _ in range(count):
            out.append(self.next(out))
        return out[len(prompt) :]


def _greedy_verify(target_rows, drafts_per_req, last_tokens):
    """The runner's greedy verification over a step's rows: one-hot target logits
    at each row (``target_rows``), the rows' inputs (the request's last token,
    then its drafts)."""
    from vllm.v1.worker.gpu.spec_decode.greedy_verification import (
        greedy_rejection_sample,
        vocab_parallel_pairs,
    )

    counts = [len(d) + 1 for d in drafts_per_req]
    rows = sum(counts)
    logits = torch.zeros((rows, VOCAB), dtype=torch.float32, device=DEVICE)
    targets = torch.tensor(
        [t for req in target_rows for t in req], dtype=torch.int64, device=DEVICE
    )
    logits[torch.arange(rows, device=DEVICE), targets] = 1.0
    # The rows' input ids, int32 as the runner's input buffer holds them.
    inputs = torch.tensor(
        [x for last, d in zip(last_tokens, drafts_per_req) for x in (last, *d)],
        dtype=torch.int32,
        device=DEVICE,
    )
    cu = torch.tensor(
        np.concatenate([[0], np.cumsum(counts)]), dtype=torch.int32, device=DEVICE
    )
    pairs = vocab_parallel_pairs(logits, VOCAB, 0, 1)
    sampled, num_sampled = greedy_rejection_sample(pairs, inputs, cu, LOOKAHEAD)
    return [
        sampled[r, : int(num_sampled[r])].tolist() for r in range(len(drafts_per_req))
    ]


def _decode(targets, prompts, count, copy):
    """Decode every request to ``count`` tokens, in one batch, verifying DSpark-like
    drafts alone or the copy drafter's proposals."""
    states = _request_state(8192)
    drafter = _drafter(states)
    texts = [list(map(int, p)) for p in prompts]
    slots = []
    for i, p in enumerate(prompts):
        slots.append(_set_history(states, f"q{i}", p))
        drafter.add_request(slots[-1])
    drafter.apply_staged_writes()
    idx = torch.tensor(slots, dtype=torch.int64, device=DEVICE)
    n = len(prompts)
    num_sampled = torch.ones(n, dtype=torch.int32, device=DEVICE)
    verified = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    out = [[] for _ in range(n)]
    while min(len(o) for o in out) < count:
        # DSpark-like drafts: the target's own next tokens, wrong now and then.
        dspark = []
        for i, text in enumerate(texts):
            ctx, guess = list(text), []
            for j in range(BLOCK):
                t = targets[i].next(ctx)
                if (len(ctx) + j) % 4 == 3:
                    t = (t + 1) % VOCAB
                guess.append(t)
                ctx.append(t)
            dspark.append(guess)
        dspark_drafts = torch.tensor(dspark, dtype=torch.int64, device=DEVICE)
        dspark_conf = torch.full((n, BLOCK), 0.7, device=DEVICE)
        if copy:
            for i, text in enumerate(texts):
                _set_history(states, f"q{i}", text)
            drafter.prepare_alloc(idx, num_sampled, verified, None)
            drafts, confidences, _ = drafter.propose(idx, dspark_drafts, dspark_conf)
            widths = (confidences > 0).sum(dim=1).tolist()
            proposals = [drafts[i, : widths[i]].tolist() for i in range(n)]
        else:
            proposals = dspark
        rows = []
        for i, text in enumerate(texts):
            ctx = list(text)
            row = [targets[i].next(ctx)]
            for t in proposals[i]:
                ctx.append(t)
                row.append(targets[i].next(ctx))
            rows.append(row)
        committed = _greedy_verify(rows, proposals, [t[-1] for t in texts])
        for i, tokens in enumerate(committed):
            texts[i].extend(tokens)
            out[i].extend(tokens)
        num_sampled = torch.tensor(
            [len(c) for c in committed], dtype=torch.int32, device=DEVICE
        )
        verified = torch.tensor(
            [len(p) for p in proposals], dtype=torch.int32, device=DEVICE
        )
    return [o[:count] for o in out]


def test_temperature_zero_text_does_not_depend_on_copy_drafts():
    rng = np.random.default_rng(6)
    targets = [
        ToyTarget(1, order=2, table=97),  # cycles early: mostly copies
        ToyTarget(2, order=3, table=4093),  # cycles late
        ToyTarget(3, order=4, table=65521, noisy=True),  # hardly copies
        ToyTarget(4, order=2, table=31),
    ]
    S, start = copy_stream(rng, VOCAB, doc=400, edits=2)
    prompts = [
        rng.integers(0, VOCAB, 30),
        S[:start],
        rng.integers(0, VOCAB, 300),
        np.tile(rng.integers(0, VOCAB, 25), 8),
    ]
    count = 700
    expected = [t.greedy(p, count) for t, p in zip(targets, prompts)]
    alone = _decode(targets, prompts, count, copy=False)
    with_copies = _decode(targets, prompts, count, copy=True)
    assert alone == expected
    assert with_copies == expected


def test_prepare_alloc_skips_incomplete_prefill_requests():
    """Requests still mid-prefill after a step pass -1 in match_idx_mapping so
    their uncomputed prompt tail is neither scanned nor offered as a copy."""
    rng = np.random.default_rng(11)
    states = _request_state(8192)
    drafter = _drafter(states)
    repeated = np.tile(rng.integers(0, VOCAB, 32), 4).tolist()
    s0 = _set_history(states, "decode_req", repeated[:96])
    s1 = _set_history(states, "mid_prefill_req", repeated[:96])
    drafter.add_request(s0)
    drafter.add_request(s1)
    drafter.apply_staged_writes()
    idx = torch.tensor([s0, s1], dtype=torch.int64, device=DEVICE)
    match_idx = torch.tensor([s0, -1], dtype=torch.int64, device=DEVICE)
    dspark_drafts = torch.arange(1, 1 + 2 * BLOCK, dtype=torch.int64, device=DEVICE).view(
        2, BLOCK
    )
    dspark_conf = torch.full((2, BLOCK), 0.6, device=DEVICE)
    num_sampled = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)
    verified = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    drafter.prepare_alloc(
        idx, num_sampled, verified, None, match_idx_mapping=match_idx
    )
    drafts, confs, _ = drafter.propose(idx, dspark_drafts, dspark_conf)
    # Decoding slot s0 matched and proposes INITIAL copy drafts; mid-prefill slot s1
    # was masked in copy_match and keeps DSpark's block with pending=0.
    assert int(drafter.int_state[s0, 1]) == INITIAL
    assert int(drafter.int_state[s1, 1]) == 0
    assert int(drafter.alloc[1]) == BLOCK
    assert drafts[1, :BLOCK].tolist() == dspark_drafts[1].tolist()
    assert drafts[1, BLOCK:].tolist() == [-1] * (LOOKAHEAD - BLOCK)
    torch.testing.assert_close(confs[1, :BLOCK], dspark_conf[1])


def test_stale_wide_allocation_rejects_placeholder_tail_and_doubles_cap():
    """When async scheduling allocates 16 draft rows from 2 steps back while the
    previous step only offered 8 copy drafts, columns 8..15 are -1 placeholders
    (rejected even when the target argmax is token 0), and accepting all 8 valid
    copy drafts still doubles the cap to 16."""
    rng = np.random.default_rng(19)
    states = _request_state(8192)
    drafter = _drafter(states)
    repeated = np.tile(rng.integers(1, VOCAB, 32), 6).tolist()
    slot = _set_history(states, "req", repeated[:96])
    drafter.add_request(slot)
    drafter.apply_staged_writes()

    idx = torch.tensor([slot], dtype=torch.int64, device=DEVICE)
    dspark_drafts = torch.zeros((1, BLOCK), dtype=torch.int64, device=DEVICE)
    dspark_conf = torch.full((1, BLOCK), 0.5, device=DEVICE)
    one = torch.ones(1, dtype=torch.int32, device=DEVICE)
    zero = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    # Step t proposes 8 copy drafts (columns 0..7) and pads columns 8..15 with -1.
    drafter.prepare_alloc(idx, one, zero, None)
    drafts, _, _ = drafter.propose(idx, dspark_drafts, dspark_conf)
    assert int(drafter.int_state[slot, 1]) == INITIAL
    assert (drafts[0, INITIAL:] == -1).all()

    # Step t+1 was scheduled with 16 draft slots from an earlier wide allocation.
    # Even if the target argmax at column 8 (and beyond) is token 0, the -1
    # placeholder at column 8 is rejected and commits token 0 as the bonus token
    # of the 8-draft prefix (9 tokens committed total).
    full_drafts = drafts[0].tolist()
    target_row = full_drafts[:INITIAL] + [0] * (LOOKAHEAD + 1 - INITIAL)
    committed = _greedy_verify([target_row], [full_drafts], [repeated[95]])[0]
    assert committed == full_drafts[:INITIAL] + [0]
    assert len(committed) == INITIAL + 1

    # Advance history by the 9 committed tokens (still matching the repeated span).
    _set_history(states, "req", repeated[: 96 + INITIAL + 1])
    num_sampled = torch.tensor([INITIAL + 1], dtype=torch.int32, device=DEVICE)
    verified = torch.tensor([LOOKAHEAD], dtype=torch.int32, device=DEVICE)
    drafter.prepare_alloc(idx, num_sampled, verified, None)
    _, _, _ = drafter.propose(idx, dspark_drafts, dspark_conf)
    assert int(drafter.int_state[slot, 0]) == 1
    assert int(drafter.int_state[slot, 1]) == 2 * INITIAL


def test_prepare_alloc_and_propose_follow_the_current_stream():
    """The current stream orders preparation/state/proposals without a join."""
    main = torch.cuda.current_stream(DEVICE)
    alternate = torch.cuda.Stream(device=DEVICE)
    runs = []
    for run_stream in (main, alternate):
        run_stream.wait_stream(main)
        with torch.cuda.stream(run_stream):
            states = _request_state(8192)
            drafter = _drafter(states)
            S, start = copy_stream(np.random.default_rng(29), VOCAB, doc=400, edits=2)
            P, pstart = prose_stream(np.random.default_rng(30), VOCAB, length=300)
            texts = {"copy": list(S[: start + 30]), "prose": list(P[:pstart])}
            slots = [_set_history(states, req, tokens) for req, tokens in texts.items()]
            for slot in slots:
                drafter.add_request(slot)
            drafter.apply_staged_writes()
            idx = torch.tensor(slots, dtype=torch.int64, device=DEVICE)
            num_sampled = torch.ones(2, dtype=torch.int32, device=DEVICE)
            verified = torch.zeros(2, dtype=torch.int32, device=DEVICE)
            dspark = (
                torch.full((2, BLOCK), 42, dtype=torch.int64, device=DEVICE),
                torch.full((2, BLOCK), 0.6, device=DEVICE),
            )
            steps = []
            for _ in range(20):
                if run_stream is alternate:
                    # Inputs, preparation and proposals must follow this delay.
                    torch.cuda._sleep(20_000_000)
                drafter.prepare_alloc(idx, num_sampled, verified, None)
                drafts, confidences, planned = drafter.propose(idx, *dspark)
                steps.append(
                    [
                        t.clone()
                        for t in (
                            drafts,
                            confidences,
                            planned,
                            drafter.alloc[:2],
                            drafter.int_state[idx],
                            drafter.float_state[idx],
                        )
                    ]
                )
                # Copy accepts every offered draft; prose takes two DSpark tokens.
                widths = (confidences > 0).sum(dim=1).tolist()
                counts = []
                for i, req in enumerate(texts):
                    ahead = S if req == "copy" else P
                    n = len(texts[req])
                    take = widths[i] + 1 if req == "copy" else 2
                    texts[req].extend(int(t) for t in ahead[n : n + take])
                    counts.append(len(texts[req]) - n)
                    _set_history(states, req, texts[req])
                num_sampled = torch.tensor(counts, dtype=torch.int32, device=DEVICE)
                verified = torch.tensor(widths, dtype=torch.int32, device=DEVICE)
        torch.cuda.synchronize()
        runs.append(steps)
    for on_main, on_alternate in zip(*runs):
        for a, b in zip(on_main, on_alternate):
            assert torch.equal(a, b)
    # The copying request copied: its proposals grew past DSpark's block.
    assert max(int((step[1][0] > 0).sum()) for step in runs[1]) == LOOKAHEAD


def test_async_alloc_ring_wrap_with_two_inflight_outputs_and_busy_copy_stream():
    """The PP=1 async runner allows two outstanding outputs, not an unbounded
    producer. It resolves the oldest future before another step; get_output()
    must finish alloc D2H before a three-slot ring buffer/event can be reused.
    Exercise that credit rule over four complete wraps, with real async output
    delivery and deliberately delayed copies of different allocation rows.
    """
    from vllm.v1.executor.uniproc_executor import AsyncOutputFuture
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    main = torch.cuda.current_stream(DEVICE)
    copy = torch.cuda.Stream(device=DEVICE)
    states = _request_state(256)
    drafter = _drafter(states)
    repeated = np.concatenate((np.arange(64), np.arange(32))) + 1000
    unique = np.arange(96) + 100000
    slots = [_set_history(states, req, unique) for req in ("a", "b")]
    for slot in slots:
        drafter.add_request(slot)
    drafter.apply_staged_writes()
    idx = torch.tensor(slots, dtype=torch.int64, device=DEVICE)
    one = torch.ones(2, dtype=torch.int32, device=DEVICE)
    zero = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    dspark = torch.full((2, BLOCK), 42, dtype=torch.int64, device=DEVICE)
    confidences = torch.full((2, BLOCK), 0.6, device=DEVICE)
    pending = []
    seen_buffers, seen_events = set(), set()
    completed = []

    def consume_oldest():
        future, expected = pending.pop(0)
        # This calls the actual get_output(), including alloc_event.synchronize(),
        # before reading its NumPy view and returning the host-owned list.
        output = future.result()
        assert output.copy_draft_alloc == expected
        completed.append(output.copy_draft_alloc)

    for step in range(12):
        assert len(pending) < 2
        for i, req in enumerate(("a", "b")):
            _set_history(states, req, repeated if (step + i) % 2 == 0 else unique)
        drafter.prepare_alloc(idx, one, zero, None)
        async_output = AsyncOutput(
            ModelRunnerOutput(["a", "b"], {"a": 0, "b": 1}),
            SamplerOutput(
                torch.ones((2, 1), dtype=torch.int64, device=DEVICE),
                None,
                None,
                one,
            ),
            one,
            main,
            copy,
        )
        with torch.cuda.stream(copy):
            torch.cuda._sleep(20_000_000)
        drafter.attach_async_output(async_output, 2, copy, main)
        seen_buffers.add(async_output.copy_draft_alloc.data_ptr())
        seen_events.add(id(async_output.copy_draft_alloc_event))
        drafter.propose(idx, dspark, confidences)
        expected = [LOOKAHEAD, BLOCK] if step % 2 == 0 else [BLOCK, LOOKAHEAD]
        pending.append((AsyncOutputFuture(async_output, True), expected))
        if len(pending) == 2:
            consume_oldest()
    while pending:
        consume_oldest()
    assert len(seen_buffers) == len(seen_events) == drafter._ring_size == 3
    assert len(completed) == 12
    # Reusing a pinned ring view cannot mutate an already delivered output list.
    for step, output in enumerate(completed):
        assert output == ([LOOKAHEAD, BLOCK] if step % 2 == 0 else [BLOCK, LOOKAHEAD])


# --- The shared budget: what every rank holds alike -------------------------------


def _manager(states):
    """The adaptive verification manager as the runner builds it beside a copy
    drafter. One rank."""
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        AdaptiveVerificationManager,
    )

    manager = AdaptiveVerificationManager(
        states,
        torch.zeros(MAX_REQS + 1, dtype=torch.int32, device=DEVICE),
        num_bonus_tokens=1,
        max_total_logits=4096,
        row_cost=0.035,
        planned_confidences=True,
        num_draft_steps=BLOCK,
    )
    manager.dead_rows_tau = 0.0
    return manager


def _one_rank():
    """One rank after the step that proposed a copy for "copy" (its history repeats)
    and DSpark's block for "prose", budgeting the next step, which the scheduler
    laid out at 16 rows for the copy (an earlier allocation) and the block for the
    other."""
    from types import SimpleNamespace

    rng = np.random.default_rng(3)
    histories = {
        "copy": np.tile(rng.integers(1, VOCAB, 32), 6).tolist()[:96],
        "prose": rng.integers(1, VOCAB, 96).tolist(),
    }
    states = _request_state(8192)
    drafter = _drafter(states)
    manager = _manager(states)
    slots = [_set_history(states, req_id, h) for req_id, h in histories.items()]
    for slot in slots:
        states.num_computed_tokens_np[slot] = 96
        drafter.add_request(slot)
        manager.add_request(slot)
    drafter.apply_staged_writes()
    manager.apply_staged_writes()
    idx = torch.tensor(slots, dtype=torch.int64, device=DEVICE)
    one = torch.ones(2, dtype=torch.int32, device=DEVICE)
    zero = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    dspark_drafts = torch.zeros((2, BLOCK), dtype=torch.int64, device=DEVICE)
    dspark_conf = torch.full((2, BLOCK), 0.5, device=DEVICE)
    drafter.prepare_alloc(idx, one, zero, None)
    _, confidences, planned = drafter.propose(idx, dspark_drafts, dspark_conf)
    batch = SimpleNamespace(num_reqs=2, idx_mapping=idx)
    for _ in range(2):  # the budget reads the snapshot of the record before last
        manager.record_confidences(confidences, batch, planned=planned)
    num_tokens = manager.get_num_tokens(
        {"copy": 1 + LOOKAHEAD, "prose": 1 + BLOCK},
        {"copy": [-1] * LOOKAHEAD, "prose": [-1] * BLOCK},
    )
    eligible = dict(manager._batch_budget[0])
    cu_num_logits, query_start_loc, budget, _, _ = manager.reallocate_drafts(
        list(histories), idx
    )
    torch.accelerator.synchronize()
    return SimpleNamespace(
        offered=drafter.int_state[idx, 1].tolist(),
        num_tokens=num_tokens,
        eligible=eligible,
        budget=budget,
        query_start_loc=query_start_loc[:3].tolist(),
        cu_num_logits=cu_num_logits.tolist(),
    )


def test_ranks_budget_the_schedulers_drafts_alike(monkeypatch):
    """The budget reads the scheduler's drafts and the confidence snapshot, which
    every tensor-parallel rank holds alike, and no state of a rank's own: a copy
    that offers 8 drafts in a step laid out for 16 is budgeted at the 16, and two
    ranks budget the same step, drafts per request and device layout."""
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode import adaptive_verification

    monkeypatch.setattr(
        adaptive_verification, "get_tp_group", lambda: SimpleNamespace(world_size=1)
    )
    first, second = _one_rank(), _one_rank()
    assert first.offered == second.offered == [INITIAL, 0]
    assert first.eligible == second.eligible == {"copy": LOOKAHEAD, "prose": BLOCK}
    for field in ("num_tokens", "budget", "query_start_loc", "cu_num_logits"):
        assert getattr(first, field) == getattr(second, field), field


def _decode_with_shared_budget(states, drafter, manager, target, req_id, prompt, count):
    """Decode ``count`` tokens of one request as the runner verifies them with the
    shared budget: the scheduler lays out each step from the allocation the proposal
    before it reported, the budget admits drafts from the confidence snapshot
    alone, the draft combine writes the step's input ids from the request state,
    and greedy verification commits the target's tokens. Returns the tokens and,
    per step, the proposal's drafts and the input ids its rows verified."""
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.input_batch import combine_sampled_and_draft_tokens

    slot = _set_history(states, req_id, prompt)  # its prefill: the prompt
    drafter.add_request(slot)
    manager.add_request(slot)
    drafter.apply_staged_writes()
    manager.apply_staged_writes()
    prefill = len(prompt)
    text = list(map(int, prompt))
    text.append(target.next(text))  # the token its prefill sampled
    idx = torch.tensor([slot], dtype=torch.int64, device=DEVICE)
    batch = SimpleNamespace(num_reqs=1, idx_mapping=idx)
    num_sampled = torch.ones(1, dtype=torch.int32, device=DEVICE)
    verified = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    reported = [BLOCK]  # each proposal's allocation; the first step's is the block
    out, steps = [], []

    def propose() -> int:
        _set_history(states, req_id, text)
        states.num_computed_tokens_np[slot] = len(text) - 1
        states.last_sampled_tokens[slot] = text[-1]
        # DSpark-like drafts: the target's own next tokens, wrong now and then.
        ctx, guess = list(text), []
        for j in range(BLOCK):
            t = target.next(ctx)
            guess.append((t + 1) % VOCAB if (len(ctx) + j) % 4 == 3 else t)
            ctx.append(guess[-1])
        dspark = torch.tensor([guess], dtype=torch.int64, device=DEVICE)
        conf = torch.full((1, BLOCK), 0.5, device=DEVICE)
        drafter.prepare_alloc(idx, num_sampled, verified, None)
        drafts, confidences, planned = drafter.propose(idx, dspark, conf)
        states.draft_tokens[idx] = drafts
        manager.record_confidences(confidences, batch, planned=planned)
        reported.append(int(drafter.alloc[0]))
        # The proposal's drafts: a copy's offered ones, else DSpark's block.
        return int(drafter.int_state[slot, 1]) or BLOCK

    width = propose()
    while len(out) < count:
        scheduled = reported[-2]
        num_tokens = manager.get_num_tokens(
            {req_id: 1 + scheduled}, {req_id: [-1] * scheduled}
        )
        cu_num_logits, query_start_loc, budget, _, _ = manager.reallocate_drafts(
            [req_id], idx
        )
        input_ids = torch.full((num_tokens,), -7, dtype=torch.int32, device=DEVICE)
        seq_lens = torch.tensor(
            [len(text) - 1 + num_tokens], dtype=torch.int32, device=DEVICE
        )
        combine_sampled_and_draft_tokens(
            input_ids, idx, states.last_sampled_tokens, query_start_loc[:2],
            seq_lens, states.prefill_len.gpu, states.draft_tokens, cu_num_logits,
            1 + budget, 1,
        )  # fmt: skip
        assert len(text) - 1 >= prefill
        inputs = input_ids.tolist()
        assert inputs[0] == text[-1] and len(inputs) == 1 + budget
        drafts = inputs[1:]
        rows = [target.next(text + drafts[:k]) for k in range(budget + 1)]
        committed = _greedy_verify([rows], [drafts], [text[-1]])[0]
        steps.append(SimpleNamespace(width=width, drafts=drafts, committed=committed))
        text += committed
        out += committed
        num_sampled = torch.tensor([len(committed)], dtype=torch.int32, device=DEVICE)
        verified = torch.tensor([budget], dtype=torch.int32, device=DEVICE)
        width = propose()
    return out[:count], steps


def test_verification_through_copy_growth_shrinkage_and_slot_reuse(monkeypatch):
    """With the shared budget, a request's steps verify the scheduler's rows
    whatever its latest proposal offers. Its copies grow past the initial cap, and
    after a rejection its proposal shrinks below the step the scheduler laid
    out: the step then verifies the proposal's -1 placeholders, which greedy
    verification rejects. The committed text is the target's own greedy text,
    and a request that takes the slot next starts afresh and decodes its own."""
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode import adaptive_verification

    monkeypatch.setattr(
        adaptive_verification, "get_tp_group", lambda: SimpleNamespace(world_size=1)
    )
    rng = np.random.default_rng(13)
    states = _request_state(8192)
    drafter = _drafter(states)
    manager = _manager(states)

    def prompt_for(target):
        """R, t, Y, R: with the token the prefill samples (t again, an order-2
        target's next token after R), the context ends with the R, t it began
        with, so its first copy proposes Y, which the target does not follow."""
        span = rng.integers(1, VOCAB, 24).tolist()
        t = target.next(span)
        return span + [t] + rng.integers(1, VOCAB, 10).tolist() + span

    # Each target soon cycles, so its later copies are accepted and grow.
    targets = [("a", ToyTarget(1, order=2, table=97)), ("b", ToyTarget(4, order=2, table=31))]
    slots, widths = set(), []
    for req_id, target in targets:
        prompt = prompt_for(target)
        count = 400
        out, steps = _decode_with_shared_budget(
            states, drafter, manager, target, req_id, prompt, count
        )
        slots.add(states.req_id_to_index[req_id])
        assert out == target.greedy(prompt, count + 1)[1:]
        assert -1 not in out
        # The copies grew past the initial cap (to the lookahead, or to the
        # target's cycle), and some step's proposal had shrunk below the rows it
        # verified: their inputs are the -1 placeholders.
        widths.append(max(s.width for s in steps))
        assert widths[-1] > INITIAL
        shrunk = [s for s in steps if len(s.drafts) > s.width]
        assert shrunk
        for s in shrunk:
            assert s.drafts[s.width :] == [-1] * (len(s.drafts) - s.width)
            assert len(s.committed) <= s.width + 1
        states.remove_request(req_id)
    assert LOOKAHEAD in widths
    assert len(slots) == 1  # "b" took "a"'s slot


@pytest.mark.parametrize("lookahead", [1, 5, 16, 64])
@pytest.mark.parametrize("period", [1, 3, 64])
def test_continuation_ranking_across_scan_passes(lookahead, period):
    """Independent reference at full/short boundaries and across grid passes."""
    lengths = [17, 31, 32, 33, 63, 64, 255, 256, 257, 260, 300, 8192, 16401]
    history = _history(MAX_REQS, max(lengths) + 32)
    total_len = torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE)
    streams = [(np.arange(n) % period + 100).astype(np.int32) for n in lengths]
    for slot, tokens in enumerate(streams):
        history[slot, : len(tokens)] = torch.from_numpy(tokens).to(DEVICE)
        total_len[slot] = len(tokens)
    idx = torch.arange(len(streams), dtype=torch.int64, device=DEVICE).flip(0)
    sources = _run_match(history, total_len, idx, lookahead=lookahead)
    for row, slot in enumerate(idx.tolist()):
        tokens = streams[slot]
        assert _best(sources[row], len(tokens), lookahead) == ref_source(
            tokens, len(tokens), lookahead=lookahead
        )


@pytest.mark.parametrize(
    "length,available", [(257, 1), (260, 4), (300, 16), (16401, 16)]
)
def test_constant_history_keeps_useful_continuation_and_allocation(length, available):
    """A recent one-token match must not hide a full equal-strength continuation.
    Short-only matches still work; stronger short matches beat weaker full ones.
    Allocation before DSpark and proposal after it must agree as the cap grows.
    """
    history = _history(MAX_REQS, length + 32)
    history[3, :length] = 7
    total_len = torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE)
    total_len[3] = length
    idx = torch.tensor([3], dtype=torch.int64, device=DEVICE)
    sources = _run_match(history, total_len, idx)
    assert _best(sources[0], length) == length - available
    zero = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    low_dspark = (
        torch.full((1, BLOCK), 99, dtype=torch.int64, device=DEVICE),
        torch.full((1, BLOCK), 0.1, device=DEVICE),
    )
    for doublings in (0, 1, 2):
        ints, floats = _state()
        ints[3, 0] = doublings
        drafts, conf, planned, alloc, _ = _propose(
            history,
            total_len,
            idx,
            sources,
            ints,
            floats,
            zero,
            zero,
            zero,
            dspark=low_dspark,
        )
        offered = min(INITIAL << doublings, LOOKAHEAD, available)
        after = min(2 * (INITIAL << doublings), LOOKAHEAD, available)
        assert int(ints[3, 1]) == offered
        assert drafts[0, :offered].tolist() == [7] * offered
        assert drafts[0, offered:].tolist() == [-1] * (LOOKAHEAD - offered)
        assert int((conf[0] > 0).sum()) == offered
        assert int((planned[0] > 0).sum()) == after
        assert int(alloc[0]) == max(after, BLOCK)


def test_copy_apply_gates_replacement_by_expected_yield_against_dspark():
    """A short copy continuation or a decayed copy acceptance estimate whose
    expected yield ``offered * estimate`` is below DSpark's expected yield
    ``sum(dspark_confidences[r, :BLOCK])`` keeps DSpark's proposal and clears
    ``int_state[slot]`` so the next step does not attribute DSpark's accepted
    tokens to CopyDrafter."""
    slot = 2
    idx = torch.tensor([slot], dtype=torch.int64, device=DEVICE)
    zero = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    # 1. Short continuation (available=1, offered=1, estimate=0.95 -> yield=0.95)
    # against DSpark with yield 3.0: DSpark wins and pending is cleared to 0.
    history = _history(MAX_REQS, 300)
    history[slot, :257] = 7
    total_len = torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE)
    total_len[slot] = 257
    sources = _run_match(history, total_len, idx)
    ints, floats = _state()
    ints[slot, 0] = 2
    dspark_strong = (
        torch.tensor([[101, 102, 103, 104, 105]], dtype=torch.int64, device=DEVICE),
        torch.tensor([[0.8, 0.7, 0.6, 0.5, 0.4]], dtype=torch.float32, device=DEVICE),
    )
    drafts, conf, planned, alloc, _ = _propose(
        history,
        total_len,
        idx,
        sources,
        ints,
        floats,
        zero,
        zero,
        zero,
        dspark=dspark_strong,
    )
    assert int(alloc[0]) == BLOCK
    assert int(ints[slot, 0]) == 0
    assert int(ints[slot, 1]) == 0
    assert drafts[0, :BLOCK].tolist() == [101, 102, 103, 104, 105]
    assert drafts[0, BLOCK:].tolist() == [-1] * (LOOKAHEAD - BLOCK)
    torch.testing.assert_close(conf[0, :BLOCK], dspark_strong[1][0])
    assert (conf[0, BLOCK:] == 0).all()
    torch.testing.assert_close(planned[0], conf[0])

    # Next step: verifying DSpark's 5 drafts (4 accepted -> num_sampled=5) must
    # not update CopyDrafter's hits/trials because pending was cleared to 0.
    sampled = torch.tensor([5], dtype=torch.int32, device=DEVICE)
    judged = torch.tensor([BLOCK], dtype=torch.int32, device=DEVICE)
    _propose(
        history,
        total_len,
        idx,
        sources,
        ints,
        floats,
        sampled,
        judged,
        judged,
        dspark=dspark_strong,
    )
    assert float(floats[slot].abs().sum()) == 0.0

    # 2. Low-estimate full continuation (offered=8, estimate=19/80=0.2375 ->
    # yield=1.90) yields to DSpark when cumulative survival dspark_yield=1.9312,
    # and wins at exact cumulative survival tie=1.90.
    total_len[slot] = 300
    history[slot, :300] = 7
    sources = _run_match(history, total_len, idx)
    ints, floats = _state()
    floats[slot] = torch.tensor([0.0, 60.0], device=DEVICE)
    dspark_above = (
        torch.tensor([[201, 202, 203, 204, 205]], dtype=torch.int64, device=DEVICE),
        torch.tensor([[0.8, 0.7, 0.6, 0.5, 0.4]], dtype=torch.float32, device=DEVICE),
    )
    drafts, conf, planned, alloc, _ = _propose(
        history,
        total_len,
        idx,
        sources,
        ints,
        floats,
        zero,
        zero,
        zero,
        dspark=dspark_above,
    )
    assert int(alloc[0]) == LOOKAHEAD
    assert int(ints[slot, 0]) == 0
    assert int(ints[slot, 1]) == 0
    assert drafts[0, :BLOCK].tolist() == [201, 202, 203, 204, 205]

    # At exact cumulative survival tie (0.95 + 0.95*1.0 = 1.90 == 8 * 0.2375), copy wins.
    dspark_tie = (
        dspark_above[0],
        torch.tensor([[0.95, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=DEVICE),
    )
    drafts, conf, planned, _, _ = _propose(
        history,
        total_len,
        idx,
        sources,
        ints,
        floats,
        zero,
        zero,
        zero,
        dspark=dspark_tie,
    )
    assert int(ints[slot, 1]) == INITIAL
    assert drafts[0, :INITIAL].tolist() == [7] * INITIAL
    assert drafts[0, INITIAL:].tolist() == [-1] * (LOOKAHEAD - INITIAL)

    # 3. Short 2-token copy continuation (available=2, offered=2, estimate=0.95
    # -> copy_yield=1.90) beats DSpark [0.6, 0.5, 0.4, 0.3, 0.2] whose raw sum is
    # 2.00 > 1.90 but whose true cumulative survival is 1.0632 < 1.90.
    total_len[slot] = 258
    sources = _run_match(history, total_len, idx)
    ints, floats = _state()
    dspark_raw2 = (
        dspark_above[0],
        torch.tensor([[0.6, 0.5, 0.4, 0.3, 0.2]], dtype=torch.float32, device=DEVICE),
    )
    drafts, conf, planned, alloc, _ = _propose(
        history,
        total_len,
        idx,
        sources,
        ints,
        floats,
        zero,
        zero,
        zero,
        dspark=dspark_raw2,
    )
    assert int(alloc[0]) == BLOCK
    assert int(ints[slot, 1]) == 2
    assert drafts[0, :2].tolist() == [7, 7]
    assert drafts[0, 2:].tolist() == [-1] * (LOOKAHEAD - 2)


def test_copy_match_boilerplate_anchor_pruning_is_bit_identical():
    """Contexts with thousands of repeated 16-, 32-, 64-, and 128-token
    boilerplate phrases produce bit-identical chunk priorities and selected
    sources to the exact reference."""
    rng = np.random.default_rng(20261106)
    length = 65536
    history = _history(4, length + 64)
    total_len = torch.full((4,), length, dtype=torch.int32, device=DEVICE)
    tail = rng.integers(10, VOCAB, MAX_LEVEL, dtype=np.int32)
    streams = []
    for row, boiler_len in enumerate((16, 32, 64, 128)):
        S = rng.integers(10, VOCAB, length, dtype=np.int32)
        phrase = tail[MAX_LEVEL - boiler_len :]
        # Stamp the boiler_len suffix every 48 tokens across the entire context,
        # preceded by a mismatching token so none of them reaches 2 * boiler_len.
        guard = int((tail[MAX_LEVEL - boiler_len - 1] + 1) % VOCAB)
        for pos in range(boiler_len + 1, length - MAX_LEVEL - 64, 48):
            S[pos - boiler_len - 1] = guard
            S[pos - boiler_len : pos] = phrase
        # In rows 1 and 3, also plant an earlier full 256-token match.
        if row in (1, 3):
            S[1200 : 1200 + MAX_LEVEL] = tail
        S[length - MAX_LEVEL :] = tail
        streams.append(S)
        history[row, :length] = torch.from_numpy(S).to(DEVICE)

    idx = torch.arange(4, dtype=torch.int64, device=DEVICE)
    sources = _run_match(history, total_len, idx)
    for row, S in enumerate(streams):
        assert _best(sources[row], length) == ref_source(S, length)



def test_copy_match_long_context_two_tier_scan():
    """Above 32K tokens, copy_match finds recent-window matches, initial-prefix
    matches, and sparse first-quarter 256-token matches exactly."""
    rng = np.random.default_rng(20261108)
    length = 131072
    history = _history(3, length + 64)
    total_len = torch.full((3,), length, dtype=torch.int32, device=DEVICE)
    tail = rng.integers(10, VOCAB, MAX_LEVEL, dtype=np.int32)
    streams = []

    # Row 0: 32-token match in the recent 16K window (at length - 1200) and
    # 16-token match in the 16K prefix (at 4000); longer 32-token match wins.
    S0 = rng.integers(10, VOCAB, length, dtype=np.int32)
    S0[4000 - 16 : 4000] = tail[MAX_LEVEL - 16 :]
    S0[length - 1200 - 32 : length - 1200] = tail[MAX_LEVEL - 32 :]
    S0[length - MAX_LEVEL :] = tail
    streams.append(S0)

    # Row 1: 16-token match in the initial 16K prefix (at 6000) with no recent match.
    S1 = rng.integers(10, VOCAB, length, dtype=np.int32)
    S1[6000 - 16 : 6000] = tail[MAX_LEVEL - 16 :]
    S1[length - MAX_LEVEL :] = tail
    streams.append(S1)

    # Row 2: 256-token match in the first quarter outside the 16K prefix (at 24000)
    # and a shorter 32-token match in the recent window; 256-token match wins.
    S2 = rng.integers(10, VOCAB, length, dtype=np.int32)
    S2[24000 - MAX_LEVEL : 24000] = tail
    S2[length - 800 - 32 : length - 800] = tail[MAX_LEVEL - 32 :]
    S2[length - MAX_LEVEL :] = tail
    streams.append(S2)

    for row, S in enumerate(streams):
        history[row, :length] = torch.from_numpy(S).to(DEVICE)

    idx = torch.arange(3, dtype=torch.int64, device=DEVICE)
    sources = _run_match(history, total_len, idx)
    for row, S in enumerate(streams):
        assert _best(sources[row], length) == ref_source(S, length)


def test_negative_slot_bounds_guard_in_alloc_and_apply():
    """Negative entries in idx_mapping never read or write slot -1 of history,
    total_len, int_state, or float_state, and fall back to DSpark proposals."""
    rng = np.random.default_rng(20261108)
    width = 256
    # Allocate with a leading canary row at index 0, and pass slices starting at
    # row 1 so any out-of-bounds access at slot = -1 hits row 0 of the base tensor.
    history_base = _history(5, width)
    history_base[0].fill_(777777)
    history = history_base[1:]

    total_len_base = torch.full((5,), 999999, dtype=torch.int32, device=DEVICE)
    total_len = total_len_base[1:]

    int_state_base = torch.full(
        (5, tlc.STATE_COLUMNS), 888888, dtype=torch.int32, device=DEVICE
    )
    float_state_base = torch.full(
        (5, tlc.STATE_COLUMNS), 555555.0, dtype=torch.float32, device=DEVICE
    )
    int_state = int_state_base[1:]
    float_state = float_state_base[1:]
    int_state.zero_()
    float_state.zero_()
    total_len.zero_()

    repeated = np.tile(rng.integers(10, VOCAB, 32, dtype=np.int32), 4)
    slot = 2
    history[slot, :96] = torch.from_numpy(repeated[:96]).to(DEVICE)
    total_len[slot] = 96

    idx_mapping = torch.tensor([-1, slot, -1], dtype=torch.int64, device=DEVICE)
    sources = _run_match(history, total_len, idx_mapping)
    assert (sources[0] == -1).all()
    assert (sources[2] == -1).all()

    sampled = torch.tensor([4, 1, 4], dtype=torch.int32, device=DEVICE)
    verified = torch.tensor([8, 0, 8], dtype=torch.int32, device=DEVICE)
    dspark = (
        torch.arange(3 * BLOCK, dtype=torch.int64, device=DEVICE).view(3, BLOCK) + 500,
        torch.full((3, BLOCK), 0.6, dtype=torch.float32, device=DEVICE),
    )
    drafts, confs, planned, alloc, _ = _propose(
        history,
        total_len,
        idx_mapping,
        sources,
        int_state,
        float_state,
        sampled,
        verified,
        verified,
        dspark=dspark,
    )
    # Canary row 0 (index -1 of the sliced views) must be completely untouched.
    assert (history_base[0] == 777777).all()
    assert int(total_len_base[0]) == 999999
    assert (int_state_base[0] == 888888).all()
    assert (float_state_base[0] == 555555.0).all()

    # Negative slots fall back to DSpark and report BLOCK allocation.
    for neg_row in (0, 2):
        assert int(alloc[neg_row]) == BLOCK
        assert drafts[neg_row, :BLOCK].tolist() == dspark[0][neg_row].tolist()
        assert drafts[neg_row, BLOCK:].tolist() == [-1] * (LOOKAHEAD - BLOCK)
        torch.testing.assert_close(confs[neg_row, :BLOCK], dspark[1][neg_row])
    assert int(alloc[1]) == 2 * INITIAL
    assert int(int_state[slot, 1]) == INITIAL

    # Even if copy_apply is invoked directly with a positive chosen source on a
    # negative slot, slot < 0 guards both history reads and int_state resets.
    forged_chosen = torch.tensor(
        [[16, 8, 16], [32, 8, 16], [16, 8, 16]], dtype=torch.int32, device=DEVICE
    )
    low_estimates = torch.tensor([0.1, 0.95, 0.99], dtype=torch.float32, device=DEVICE)
    tlc.copy_apply(
        idx_mapping,
        history,
        forged_chosen,
        low_estimates,
        int_state,
        dspark[0],
        dspark[1],
        drafts,
        confs,
        planned,
    )
    assert (history_base[0] == 777777).all()
    assert (int_state_base[0] == 888888).all()
    for neg_row in (0, 2):
        assert drafts[neg_row, :BLOCK].tolist() == dspark[0][neg_row].tolist()
