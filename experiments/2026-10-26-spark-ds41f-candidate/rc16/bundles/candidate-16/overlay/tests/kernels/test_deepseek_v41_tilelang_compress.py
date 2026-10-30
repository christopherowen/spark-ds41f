# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang CSA compression against DeepSeek's arithmetic and torch
references of the state ring."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from vllm.models.deepseek_v4_1.tilelang.compress import (  # noqa: E402
    DIM,
    STATE,
    compress,
    gather_partial_state,
)

DEVICE = "cuda"
REQUESTS, CAPACITY, PAGE, BLOCKS = 16, 512, 8, 40


def _step(lengths, ratio, seed, padding=3):
    """A packed step: requests of ``lengths`` tokens (0 = empty), random positions,
    one ring block each, and ``padding`` rows past the live tokens."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    nt, nr = starts[-1], len(lengths)
    rows = nt + padding
    positions = torch.randint(0, 50, (REQUESTS,), generator=g)
    positions[nr:] = -1
    starts_t = torch.full((REQUESTS + 1,), nt, dtype=torch.int32)
    starts_t[: nr + 1] = torch.tensor(starts, dtype=torch.int32)
    vdtype = torch.float32 if ratio == 2 else torch.bfloat16
    values = (torch.randn(CAPACITY, DIM, generator=g) * 3).to(vdtype)
    gates = torch.randn(CAPACITY, DIM, generator=g) * 2
    slots = torch.randint(0, 10**6, (CAPACITY,), generator=g)
    table = torch.randint(1, BLOCKS, (REQUESTS, 4), generator=g).to(torch.int32)
    # Each request owns its ring block, as the block table gives it.
    table[:, 0] = (torch.randperm(BLOCKS - 1, generator=g)[:REQUESTS] + 1).to(
        torch.int32
    )
    state_slots = torch.full((CAPACITY,), -1, dtype=torch.int64)
    for r, n in enumerate(lengths):
        for k in range(n):
            pos = int(positions[r]) + k
            state_slots[starts[r] + k] = int(table[r, 0]) * PAGE + pos % PAGE
    weight = torch.rand(DIM, generator=g) + 0.5
    counts = torch.tensor([nt, nr], dtype=torch.int32)
    # Padded rows, as the KV allocator lays out its pages.
    pool = torch.randn(BLOCKS, PAGE * STATE + 64, generator=g)[:, : PAGE * STATE]
    tensors = dict(
        starts=starts_t, positions=positions, values=values, gates=gates, slots=slots,
        table=table, state_slots=state_slots, weight=weight, counts=counts, pool=pool,
    )  # fmt: skip
    moved = {k: v.to(DEVICE) for k, v in tensors.items()}
    moved["pool"] = torch.randn(BLOCKS, PAGE * STATE + 64, device=DEVICE)[
        :, : PAGE * STATE
    ]
    moved["pool"].copy_(tensors["pool"])
    assert not moved["pool"].is_contiguous()
    return moved, rows, starts


def _ring(step):
    """Each request's carry, by the TileLang kernel."""
    pv = torch.full((REQUESTS, DIM), 9.0, device=DEVICE)
    pg = torch.full_like(pv, 9.0)
    tags = torch.full((REQUESTS,), 7, dtype=torch.int64, device=DEVICE)
    ids = torch.full_like(tags, 7)
    gather_partial_state(
        step["pool"], step["table"], step["starts"], step["positions"],
        step["counts"], pv, pg, tags, ids, page=PAGE,
    )  # fmt: skip
    return pv, pg, tags, ids


def _ref_ring(step):
    """The carry's contract: a live request (one of the step's, rows within the
    step's tokens, a position) at an odd position reads its predecessor's value
    and gate from its ring block (in the first column of its block table), tagged
    with the predecessor's position; every other request carries zeros, tag -1. A
    live request keeps its index as its state id, others -1."""
    starts, positions, table, counts = (
        step[name].cpu().long() for name in ("starts", "positions", "table", "counts")
    )
    pool = step["pool"].cpu()
    nt, nr = int(counts[0]), int(counts[1])
    r = torch.arange(REQUESTS)
    first, end = starts[:REQUESTS], starts[1 : REQUESTS + 1]
    live = (r < nr) & (first >= 0) & (first < end) & (end <= nt) & (positions >= 0)
    needed = live & (positions > 0) & (positions % 2 == 1)
    page = torch.where(needed, table[:, 0], 0)
    valid = needed & (page > 0)
    predecessor = positions - 1
    base = (torch.where(valid, predecessor, 0) % PAGE) * STATE
    column = base[:, None] + torch.arange(DIM)
    rows = pool[page]
    value = torch.where(valid[:, None], rows.gather(1, column), 0.0)
    gate = torch.where(valid[:, None], rows.gather(1, column + DIM), 0.0)
    return value, gate, torch.where(valid, predecessor, -1), torch.where(live, r, -1)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gather_matches_the_reference(seed):
    step, _, _ = _step([3, 0, 5, 1, 2, 6], 2, seed)
    for actual, expected in zip(_ring(step), _ref_ring(step)):
        assert torch.equal(actual.cpu(), expected.to(actual.dtype))


def normalized_latent(values, weight, gates=None):
    """One latent of DeepSeek's ``Compressor``: the softmax-gated pool in FP32,
    rounded to BF16 (``kv.to(dtype)``), then its ``RMSNorm`` in FP32 with the
    checkpoint's ``norm_eps`` of 1e-20."""
    pooled = (
        values
        if gates is None
        else (values.float() * gates.float().softmax(dim=0)).sum(dim=0)
    )
    rounded = pooled.to(torch.bfloat16).float()
    rstd = torch.rsqrt(rounded.square().mean(dim=-1, keepdim=True) + 1e-20)
    return (rounded * rstd * weight.float()).to(torch.bfloat16)


def _reference(step, rows, starts, ratio, carry):
    """DeepSeek's arithmetic, token by token."""
    out = torch.zeros(rows, DIM, dtype=torch.bfloat16, device=DEVICE)
    emitted = torch.zeros(rows, dtype=torch.bool, device=DEVICE)
    slots = torch.full((rows,), -1, dtype=torch.int64, device=DEVICE)
    pv, pg, tags, ids = carry
    for r in range(len(starts) - 1):
        for token in range(starts[r], starts[r + 1]):
            pos = int(step["positions"][r]) + token - starts[r]
            if ratio == 1:
                out[token] = normalized_latent(step["values"][token], step["weight"])
            elif pos % 2 == 0:
                continue
            else:
                if token == starts[r]:
                    if int(tags[ids[r]]) != pos - 1:
                        continue
                    pair_v = torch.stack((pv[ids[r]], step["values"][token]))
                    pair_g = torch.stack((pg[ids[r]], step["gates"][token]))
                else:
                    pair_v = step["values"][token - 1 : token + 1]
                    pair_g = step["gates"][token - 1 : token + 1]
                out[token] = normalized_latent(pair_v, step["weight"], pair_g)
            emitted[token] = True
            slots[token] = step["slots"][token]
    return out, emitted, slots


def _run(step, rows, ratio, carry):
    out = torch.full((CAPACITY, DIM), 5.0, dtype=torch.bfloat16, device=DEVICE)
    emitted = torch.ones(CAPACITY, dtype=torch.bool, device=DEVICE)
    eslots = torch.full((CAPACITY,), 77, dtype=torch.int64, device=DEVICE)
    pv, pg, tags, ids = carry
    state_ids = ids if ratio == 2 else torch.arange(REQUESTS, device=DEVICE)
    kwargs = {}
    if ratio == 2:
        kwargs = dict(
            pending_values=pv, pending_gates=pg, tags=tags, pool=step["pool"],
            state_slots=step["state_slots"], page=PAGE,
        )  # fmt: skip
    compress(
        rows, ratio, step["values"], step["gates"] if ratio == 2 else None,
        step["weight"], step["starts"], step["positions"], state_ids, step["slots"],
        step["counts"], out, emitted, eslots, **kwargs,
    )  # fmt: skip
    return out, emitted, eslots


def _bf16_ulps(a, b):
    ia = a.view(torch.int16).int()
    ib = b.view(torch.int16).int()
    return (ia - ib).abs()


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("lengths", [[1], [6] * 16, [3, 0, 5, 1, 2, 6], [300]])
def test_compress_matches_reference(ratio, lengths):
    step, rows, starts = _step(lengths, ratio, seed=len(lengths) * 10 + ratio)
    carry = _ring(step)
    want = _reference(step, rows, starts, ratio, carry)
    pool_before = step["pool"].clone()
    out, emitted, eslots = _run(step, rows, ratio, carry)
    assert torch.equal(emitted[:rows], want[1])
    assert torch.equal(eslots[:rows], want[2])
    ulps = _bf16_ulps(out[:rows], want[0])
    # The library exp and the reduction order each move an FP32 bit; two such flips
    # at BF16 rounding boundaries make two ulps.
    assert int(ulps.max()) <= 2 and float((ulps > 0).float().mean()) < 0.01
    # Rows past the step keep their contents.
    assert (out[rows:] == 5.0).all() and emitted[rows:].all()
    assert (eslots[rows:] == 77).all()
    if ratio == 2:
        # Each slot holds its latest position's value and gate.
        latest = pool_before.clone()
        for r in range(len(starts) - 1):
            for token in range(max(starts[r], starts[r + 1] - PAGE), starts[r + 1]):
                slot = int(step["state_slots"][token])
                base = (slot % PAGE) * STATE
                latest[slot // PAGE, base : base + DIM] = step["values"][token]
                latest[slot // PAGE, base + DIM : base + STATE] = step["gates"][token]
        assert torch.equal(step["pool"], latest)


@pytest.mark.parametrize("ratio", [1, 2])
def test_token_independent_of_batch_and_graphs(ratio):
    lengths = [4, 6, 2]
    step, rows, starts = _step(lengths, ratio, seed=40 + ratio)
    carry = _ring(step)
    together = _run(step, rows, ratio, carry)[0][:rows].clone()
    # The middle request alone: its rows packed first, the same positions.
    alone_step = {k: v.clone() for k, v in step.items()}
    a, b = starts[1], starts[2]
    for name in ("values", "gates", "slots", "state_slots"):
        alone_step[name][: b - a] = step[name][a:b]
    alone_step["starts"][:2] = torch.tensor([0, b - a], device=DEVICE)
    alone_step["starts"][2:] = b - a
    alone_step["positions"][0] = step["positions"][1]
    alone_step["positions"][1:] = -1
    alone_step["counts"].copy_(torch.tensor([b - a, 1], device=DEVICE))
    alone_step["table"][0] = step["table"][1]
    alone_carry = tuple(t.clone() for t in carry)
    for t, src in zip(alone_carry, carry):
        t[0] = src[1]
    alone_carry[3][0] = 0
    alone = _run(alone_step, b - a, ratio, alone_carry)[0][: b - a]
    assert torch.equal(alone, together[a:b])
    # Graph replays give the eager bits.
    out = torch.zeros(CAPACITY, DIM, dtype=torch.bfloat16, device=DEVICE)
    emitted = torch.zeros(CAPACITY, dtype=torch.bool, device=DEVICE)
    eslots = torch.zeros(CAPACITY, dtype=torch.int64, device=DEVICE)
    state_ids = carry[3] if ratio == 2 else torch.arange(REQUESTS, device=DEVICE)
    kwargs = {}
    if ratio == 2:
        kwargs = dict(
            pending_values=carry[0], pending_gates=carry[1], tags=carry[2],
            pool=step["pool"], state_slots=step["state_slots"], page=PAGE,
        )  # fmt: skip

    def call():
        compress(
            rows, ratio, step["values"], step["gates"] if ratio == 2 else None,
            step["weight"], step["starts"], step["positions"], state_ids,
            step["slots"], step["counts"], out, emitted, eslots, **kwargs,
        )  # fmt: skip

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(10):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[:rows], together)


def _verify_block_then_resume(draft_tokens, page, accepted):
    """One request verifies a block of ``1 + draft_tokens`` rows from an even
    position, writing every row's state to its ring; the next step resumes after
    ``accepted`` drafts. Returns the carry that step gathers (value, gate, tag) and
    the block's state at the position it carries from."""
    g = torch.Generator(device="cpu").manual_seed(draft_tokens * 100 + accepted)
    rows, first, block = 1 + draft_tokens, 40, 3

    def request(position, tokens):
        starts = torch.full((REQUESTS + 1,), tokens, dtype=torch.int32)
        starts[0] = 0
        positions = torch.full((REQUESTS,), -1, dtype=torch.int64)
        positions[0] = position
        state_slots = torch.full((CAPACITY,), -1, dtype=torch.int64)
        state_slots[:tokens] = block * page + (position + torch.arange(tokens)) % page
        return {
            "starts": starts.to(DEVICE),
            "positions": positions.to(DEVICE),
            "state_slots": state_slots.to(DEVICE),
            "counts": torch.tensor([tokens, 1], dtype=torch.int32, device=DEVICE),
        }

    table = torch.full((REQUESTS, 4), 1, dtype=torch.int32, device=DEVICE)
    table[0, 0] = block
    pool = torch.zeros(BLOCKS, page * STATE, device=DEVICE)
    values = torch.randn(CAPACITY, DIM, generator=g).to(DEVICE)
    gates = torch.randn(CAPACITY, DIM, generator=g).to(DEVICE)
    weight = (torch.rand(DIM, generator=g) + 0.5).to(DEVICE)
    slots = torch.arange(CAPACITY, dtype=torch.int64, device=DEVICE)

    def carry(step):
        pv = torch.zeros((REQUESTS, DIM), device=DEVICE)
        pg = torch.zeros_like(pv)
        tags = torch.full((REQUESTS,), 7, dtype=torch.int64, device=DEVICE)
        ids = torch.full_like(tags, 7)
        gather_partial_state(
            pool, table, step["starts"], step["positions"], step["counts"],
            pv, pg, tags, ids, page=page,
        )  # fmt: skip
        return pv, pg, tags, ids

    a = request(first, rows)
    pv, pg, tags, ids = carry(a)
    out = torch.zeros(CAPACITY, DIM, dtype=torch.bfloat16, device=DEVICE)
    emitted = torch.zeros(CAPACITY, dtype=torch.bool, device=DEVICE)
    eslots = torch.zeros(CAPACITY, dtype=torch.int64, device=DEVICE)
    compress(
        rows, 2, values, gates, weight, a["starts"], a["positions"], ids, slots,
        a["counts"], out, emitted, eslots, pending_values=pv, pending_gates=pg,
        tags=tags, pool=pool, state_slots=a["state_slots"], page=page,
    )  # fmt: skip
    resumed = first + accepted + 1
    pv, pg, tags, _ = carry(request(resumed, 1))
    row = resumed - 1 - first
    return (pv[0], pg[0], int(tags[0])), (values[row], gates[row], resumed - 1)


@pytest.mark.parametrize("draft_tokens", [5, 16, 64])
def test_a_rejected_block_leaves_every_accepted_prefix_its_carry(draft_tokens):
    """The ring a request's verification writes before acceptance is known keeps
    the state of every position a resumed step can carry from: the state ring
    (CompressorStateCache) has a slot for each of its rows and one more, so a block
    of the verification lookahead (copy drafts: 16, or 64) overwrites none of them."""
    from vllm.models.deepseek_v4_1.compressor import CompressorStateCache

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_verification_tokens=draft_tokens),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    page = CompressorStateCache(config, f"ring-{draft_tokens}").block_size
    assert page > draft_tokens + 1
    for accepted in range(0, draft_tokens + 1, 2):  # odd positions carry
        (value, gate, tag), (want_v, want_g, want_tag) = _verify_block_then_resume(
            draft_tokens, page, accepted
        )
        assert tag == want_tag
        assert torch.equal(value, want_v) and torch.equal(gate, want_g), accepted


def test_a_short_ring_loses_a_wide_blocks_carry():
    """The control: DSpark's ring of 8 under a block of 17 rows. A resumed step
    early in the block carries a later row's state, which shares its slot."""
    lost = 0
    for accepted in range(0, 9, 2):
        (value, _, _), (want_v, _, _) = _verify_block_then_resume(16, 8, accepted)
        lost += not torch.equal(value, want_v)
    assert lost > 0
