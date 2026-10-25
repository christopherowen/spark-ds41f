# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy verification from vocabulary shards gives the stock sampler's tokens."""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
    greedy_shard_pair,
    scratch_shape,
    select_first_argmax,
)
from vllm.v1.worker.gpu.spec_decode.greedy_verification import (
    PAIR_WIDTH,
    greedy_rejection_sample,
    vocab_parallel_pairs,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

DEVICE = "cuda"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _sharded_pairs(
    logits: torch.Tensor, ranks: int, padding: int, pair_width: int = 2
) -> torch.Tensor:
    """Each simulated rank's padded shard reduced to a pair (in rows of
    ``pair_width`` columns), the rows concatenated in rank order: what the
    all-gather returns."""
    rows, vocab = logits.shape
    shard = -(-vocab // ranks)
    pairs = []
    for rank in range(ranks):
        start = rank * shard
        real = max(0, min(shard, vocab - start))
        local = torch.full((rows, shard + padding), 1e30, device=DEVICE)  # junk padding
        local[:, :real] = logits[:, start : start + real]
        pair = torch.full((rows, pair_width), -3.0, device=DEVICE)
        shape = scratch_shape(rows, real)
        values = torch.empty(shape, dtype=torch.float32, device=DEVICE)
        indices = torch.empty(shape, dtype=torch.int32, device=DEVICE)
        greedy_shard_pair(local, real, start, pair, values, indices)
        pairs.append(pair)
    return torch.cat(pairs, dim=-1)


def _sharded_argmax(logits: torch.Tensor, ranks: int, padding: int) -> torch.Tensor:
    """The simulated ranks' pairs reduced to the first maximum."""
    ids = torch.empty(logits.shape[0], dtype=torch.int64, device=DEVICE)
    select_first_argmax(_sharded_pairs(logits, ranks, padding), ids)
    return ids


@pytest.mark.parametrize("ranks", [1, 3, 4])
@pytest.mark.parametrize("vocab", [129280, 50001])
def test_vocab_parallel_argmax_is_the_full_argmax(ranks, vocab):
    torch.manual_seed(0)
    rows = 96
    logits = torch.randn((rows, vocab), device=DEVICE)
    # Ties: equal maxima inside a shard, across a shard boundary, and at both ends.
    shard = -(-vocab // ranks)
    logits[0, [5, 9]] = 50.0
    if ranks > 1:
        logits[1, [shard - 1, shard]] = 50.0
    logits[2, [0, vocab - 1]] = 50.0
    logits[3, :] = 1.0  # all equal: the first index
    assert torch.equal(_sharded_argmax(logits, ranks, padding=7), logits.argmax(-1))
    # Greedy verification selects the same targets from the runner's 16-byte pair
    # rows: one logits row a request, no drafts, so each commits its row's target.
    pairs = _sharded_pairs(logits, ranks, padding=7, pair_width=PAIR_WIDTH)
    cu = torch.arange(rows + 1, dtype=torch.int32, device=DEVICE)
    drafts = torch.full((rows,), -1, dtype=torch.int32, device=DEVICE)
    sampled, num_sampled = greedy_rejection_sample(pairs, drafts, cu, 5)
    assert (num_sampled == 1).all()
    assert torch.equal(sampled[:, 0], logits.argmax(-1))


def test_vocab_parallel_pairs_on_one_rank():
    """One rank's pairs: its 16-byte rows, the pair in the first two columns."""
    torch.manual_seed(2)
    logits = torch.randn((7, 4099), device=DEVICE)
    pairs = vocab_parallel_pairs(logits, 4000, 0, 1)
    assert pairs.shape == (7, PAIR_WIDTH) and pairs.dtype == torch.float32
    assert torch.equal(pairs[:, 0], logits[:, :4000].amax(-1))
    assert torch.equal(pairs[:, 1].long(), logits[:, :4000].argmax(-1))


@pytest.mark.parametrize("ranks", [1, 3, 4])
@pytest.mark.parametrize("block_verification", [True, False])
def test_greedy_rejection_matches_rejection_sample(ranks, block_verification):
    torch.manual_seed(1)
    num_spec, vocab = 5, 32768 + 123
    drafts_per_req = [5, 5, 5, 5, 5, 3, 0, 5, 1, 5, 5, 5]
    num_reqs = len(drafts_per_req)
    cu = [0]
    for n in drafts_per_req:
        cu.append(cu[-1] + n + 1)
    num_logits = cu[-1]
    logits = torch.randn((num_logits, vocab), device=DEVICE)
    target = logits.argmax(-1)
    draft_sampled = torch.randint(0, vocab, (num_logits,), device=DEVICE)
    local_pos = []
    expanded = []
    for r, n in enumerate(drafts_per_req):
        start = cu[r]
        accept = int(torch.randint(0, n + 1, ()).item())
        for i in range(n):
            if i < accept:
                draft_sampled[start + i + 1] = target[start + i]
            elif i == accept:
                # A mismatch, or the always-rejected placeholder.
                draft_sampled[start + i + 1] = (
                    -1 if r % 3 == 0 else (target[start + i] + 1) % vocab
                )
        local_pos += list(range(n + 1))
        expanded += [r] * (n + 1)
    cu_num_logits = torch.tensor(cu, dtype=torch.int32, device=DEVICE)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = torch.tensor(expanded, dtype=torch.int32, device=DEVICE)
    expanded_local_pos = torch.tensor(local_pos, dtype=torch.int32, device=DEVICE)
    pos = torch.arange(num_logits, dtype=torch.int64, device=DEVICE) + 100
    temperature = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)
    seed = torch.zeros(num_reqs, dtype=torch.int64, device=DEVICE)
    stock, stock_n = rejection_sample(
        logits, None, draft_sampled, cu_num_logits, pos, idx_mapping,
        expanded_idx_mapping, expanded_local_pos, temperature, seed, num_spec,
        use_block_verification=block_verification,
    )  # fmt: skip
    pairs = _sharded_pairs(logits, ranks, padding=5, pair_width=PAIR_WIDTH)
    greedy, greedy_n = greedy_rejection_sample(
        pairs, draft_sampled, cu_num_logits, num_spec
    )
    assert torch.equal(greedy_n, stock_n)
    for r in range(num_reqs):
        n = int(stock_n[r])
        assert torch.equal(greedy[r, :n], stock[r, :n]), r


@pytest.mark.parametrize("tp_size", [1, 3, 4])
def test_greedy_verification_declares_its_one_launch(monkeypatch, tp_size):
    """The runner's declaration covers the fused selection and verification for
    the TP size's pair rows (the runner's int32 drafts, and int64), and no separate
    selection; the launch takes a declared key."""
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
    from vllm.models.deepseek_v4_1.tilelang import prepare
    from vllm.v1.worker.gpu.sample import tilelang_sampling as tls
    from vllm.v1.worker.gpu.spec_decode.greedy_verification import (
        register_greedy_verification_warmups,
    )

    registry = JitWarmupRegistry(None)
    with registry.activate():
        register_greedy_verification_warmups(
            logits_dtype=torch.bfloat16,
            logits_stride=32320,
            vocab=32320,
            vocab_start=0,
            tp_size=tp_size,
            num_speculative_steps=5,
        )
    calls = prepare.declared_calls(registry)
    declared = {args for f, args, _ in calls if f is tls.greedy_rejection_kernel}
    assert declared == {
        (6, "int32", tp_size, PAIR_WIDTH),
        (6, "int64", tp_size, PAIR_WIDTH),
    }
    assert not any(f is tls.select_first_argmax_kernel for f, _, _ in calls)
    prepare.prepare_declared(registry)
    launched = []
    original = tls._kernel

    def record(factory, *key):
        launched.append((factory, key))
        return original(factory, *key)

    monkeypatch.setattr(tls, "_kernel", record)
    cu = torch.tensor([0, 6, 7, 13], dtype=torch.int32, device=DEVICE)
    pairs = torch.zeros((13, PAIR_WIDTH * tp_size), device=DEVICE)
    drafts = torch.zeros(13, dtype=torch.int32, device=DEVICE)
    greedy_rejection_sample(pairs, drafts, cu, 5)
    assert [factory for factory, _ in launched] == [tls.greedy_rejection_kernel]
    assert launched[0][1] in declared
