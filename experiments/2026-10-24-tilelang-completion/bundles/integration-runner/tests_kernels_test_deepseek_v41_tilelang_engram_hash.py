# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang Engram hash against DeepSeek's integer arithmetic."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sympy import isprime

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from vllm.models.deepseek_v4_1.tilelang.engram_hash import (  # noqa: E402
    IMAGE_SENTINEL,
    engram_hash,
    geometry_tensors,
)

DEVICE = "cuda"
VOCAB, COMPRESSED = 129280, 99092
LAYERS = (1, 14)


def _geometry(layer_ids, table_size=16_000_000, ngrams=4, heads=8):
    """DeepSeek's inference/engram.py geometry: ``EngramLayout``'s primes, drawn in
    order from ``engram_vocab_size`` and never reused, their running offsets, and
    ``compute_hash_multipliers``."""
    seen, primes, offsets, multipliers = set(), [], [], []
    bound = max(1, (np.iinfo(np.int64).max // COMPRESSED) // 2)
    for layer in layer_ids:
        sizes = []
        for _ in range(ngrams - 1):
            current = table_size - 1
            for _ in range(heads):
                current += 1
                while not isprime(current) or current in seen:
                    current += 1
                seen.add(current)
                sizes.append(current)
        primes.append(sizes)
        offsets.append(np.cumsum([0, *sizes[:-1]]).tolist())
        values = np.random.default_rng(10007 * layer).integers(
            0, bound, size=(ngrams,), dtype=np.int64
        )
        multipliers.append((values * 2 + 1).tolist())
    return SimpleNamespace(primes=primes, offsets=offsets, multipliers=multipliers)


def _hash_reference(tokens, mask, starts, history, token_map, geometry, layer):
    """``NgramHashState.forward`` of DeepSeek's inference/engram.py over packed
    requests: each request's three history ids, then its compressed ids (``-1``
    where masked). Look-back stops at a dead id and pads with the pad token's id;
    each running XOR of id times multiplier lands at its prime plus its offset."""
    pad = token_map[2]  # engram_pad_id
    multipliers = geometry.multipliers[layer]
    primes, offsets = geometry.primes[layer], geometry.offsets[layer]
    compressed = [token_map[t] if keep else -1 for t, keep in zip(tokens, mask)]
    rows = []
    for request, (begin, end) in enumerate(zip(starts[:-1], starts[1:])):
        cache = list(history[request]) + compressed[begin:end]
        for position in range(3, len(cache)):
            rolling, blocked, result = 0, False, []
            for lag in range(4):
                source = cache[position - lag]
                blocked |= source == -1
                rolling ^= (pad if blocked else source) * multipliers[lag]
                if lag:
                    for head in range((lag - 1) * 8, lag * 8):
                        result.append(rolling % primes[head] + offsets[head])
            rows.append(result)
    return torch.tensor(rows, dtype=torch.int64).reshape(-1, 24)


def _case(lengths, padding, seed):
    gen = torch.Generator().manual_seed(seed)
    token_map = torch.randint(0, COMPRESSED, (VOCAB,), generator=gen)
    live = sum(lengths)
    ids = torch.randint(0, VOCAB, (live + padding,), generator=gen)
    ids[torch.rand(live + padding, generator=gen) < 0.05] = IMAGE_SENTINEL
    ids[torch.rand(live + padding, generator=gen) < 0.02] = VOCAB + 7
    history = torch.randint(0, VOCAB, (len(lengths), 3), generator=gen)
    history[torch.rand(len(lengths), 3, generator=gen) < 0.2] = -1
    history[0, 2] = IMAGE_SENTINEL
    starts = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return token_map, ids, history, starts


def _expected(token_map, ids, history, starts, geometry):
    image = ids == IMAGE_SENTINEL  # the hash finds the sentinel in the ids itself
    live = int(starts[-1])
    tokens = ids[:live].clamp(0, VOCAB - 1).tolist()
    keep = ((ids[:live] < VOCAB) & ~image[:live]).tolist()
    # vLLM's metadata compresses history: out of vocabulary and the sentinel drop.
    valid = (history >= 0) & (history < VOCAB) & (history != IMAGE_SENTINEL)
    compressed = torch.where(valid, token_map[history.clamp(0, VOCAB - 1)], -1)
    rows = [
        _hash_reference(
            tokens,
            keep,
            starts.tolist(),
            compressed.tolist(),
            token_map.tolist(),
            geometry,
            layer,
        )
        for layer in range(len(LAYERS))
    ]
    return torch.stack(rows, dim=1)


@pytest.mark.parametrize("layer_major", [False, True])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("history_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "lengths, padding",
    [((1,), 0), ((6, 1, 3, 6), 5), ((700, 1, 2, 1300), 0), ((0, 4, 0, 2), 3)],
)
def test_engram_hash(lengths, padding, ids_dtype, history_dtype, layer_major):
    """``layer_major``: ``out`` is the permutation of a [layers, rows, 24] buffer,
    as the model's disk Engram planes are."""
    geometry = _geometry(LAYERS)
    token_map, ids, history, starts = _case(lengths, padding, sum(lengths) + padding)
    expected = _expected(token_map, ids, history, starts, geometry)
    rows = ids.numel() + 2
    if layer_major:
        out = torch.full((len(LAYERS), rows, 24), 7, dtype=torch.int64)
        out = out.to(DEVICE).permute(1, 0, 2)
    else:
        out = torch.full((rows, len(LAYERS), 24), 7, dtype=torch.int64).to(DEVICE)
    num_seqs, num_tokens = (
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
    )
    engram_hash(
        ids.to(DEVICE, ids_dtype),
        starts.to(DEVICE),
        history.to(DEVICE, history_dtype),
        token_map.to(DEVICE),
        *geometry_tensors(geometry, DEVICE),
        out,
        num_seqs,
        num_tokens,
    )
    live = int(starts[-1])
    assert torch.equal(out[:live].cpu(), expected)
    assert (out[live : ids.numel()] == -1).all(), "padding rows must hash to -1"
    assert (out[ids.numel() :] == 7).all(), "rows past the ids must stay untouched"
    assert int(num_seqs) == len(lengths) and int(num_tokens) == live
