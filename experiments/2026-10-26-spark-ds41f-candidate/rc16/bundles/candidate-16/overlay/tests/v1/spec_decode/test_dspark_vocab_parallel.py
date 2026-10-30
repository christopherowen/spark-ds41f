# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
    greedy_markov_shard_pair,
    sample_greedy_markov,
    scratch_shape,
    select_first_argmax,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _sharded(base, bias, ranks, padded):
    """Reduce per-shard pairs as TP ranks would, padding each shard like a
    vocabulary-parallel head (the padded columns are never read)."""
    rows, vocab = base.shape
    per_rank = -(-vocab // ranks)
    values = torch.empty(scratch_shape(rows, padded), dtype=torch.float32, device="cuda")
    indices = torch.empty(scratch_shape(rows, padded), dtype=torch.int32, device="cuda")
    pairs = []
    for rank in range(ranks):
        start = rank * per_rank
        stop = min(vocab, start + per_rank)
        shard_base = torch.full((rows, padded), 7.0, dtype=base.dtype, device="cuda")
        shard_bias = torch.full((rows, padded), 7.0, dtype=bias.dtype, device="cuda")
        shard_base[:, : stop - start] = base[:, start:stop]
        shard_bias[:, : stop - start] = bias[:, start:stop]
        pair = torch.empty(rows, 2, dtype=torch.float32, device="cuda")
        greedy_markov_shard_pair(
            shard_base[:, : stop - start], shard_bias[:, : stop - start], start,
            pair, values, indices,
        )
        pairs.append(pair)
    out = torch.empty(rows, 5, dtype=torch.int64, device="cuda")[:, 2]
    select_first_argmax(torch.cat(pairs, dim=-1), out)
    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ranks", [1, 2, 3])
def test_shard_pairs_reproduce_the_full_argmax(dtype, ranks):
    torch.manual_seed(0)
    rows, vocab = 8, 129280
    base = torch.randn(rows, vocab, device="cuda").to(dtype)
    bias = torch.randn(rows, vocab, device="cuda").to(dtype)
    # Ties across shard boundaries resolve to the lowest index; NaN wins.
    base[1, [100, 50000, 120000]] = 30
    bias[1, [100, 50000, 120000]] = 0
    base[2, [90000, 20000]] = 40
    bias[2, [90000, 20000]] = 0
    base[3, 70000] = float("nan")
    expected = torch.empty(rows, dtype=torch.int64, device="cuda")
    values = torch.empty(scratch_shape(rows, vocab), dtype=torch.float32, device="cuda")
    indices = torch.empty(scratch_shape(rows, vocab), dtype=torch.int32, device="cuda")
    sample_greedy_markov(base, bias, expected, values, indices)
    assert torch.equal(expected, (base + bias).argmax(dim=-1))
    got = _sharded(base, bias, ranks, padded=-(-vocab // ranks) + 64)
    assert got.tolist() == expected.tolist()
    assert expected[1].item() == 100 and expected[2].item() == 20000 and expected[3].item() == 70000
