# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang Engram hash against B12X's integer oracle."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")
pytest.importorskip("b12x")

from b12x.sequence.engram.geometry import build_geometry  # noqa: E402
from b12x.sequence.engram.reference import hash_reference  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.engram_hash import (  # noqa: E402
    IMAGE_SENTINEL,
    engram_hash,
    geometry_tensors,
)

DEVICE = "cuda"
VOCAB, COMPRESSED = 129280, 99092
LAYERS = (1, 14)


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
    image = ids == IMAGE_SENTINEL
    live = int(starts[-1])
    tokens = ids[:live].clamp(0, VOCAB - 1).tolist()
    keep = ((ids[:live] < VOCAB) & ~image[:live]).tolist()
    # vLLM's metadata compresses history: out of vocabulary and the sentinel drop.
    valid = (history >= 0) & (history < VOCAB) & (history != IMAGE_SENTINEL)
    compressed = torch.where(valid, token_map[history.clamp(0, VOCAB - 1)], -1)
    slots = list(range(len(starts) - 1))
    rows = [
        hash_reference(
            tokens,
            keep,
            starts.tolist(),
            slots,
            compressed.tolist(),
            token_map.tolist(),
            geometry,
            layer,
        )
        for layer in LAYERS
    ]
    return torch.stack(rows, dim=1), image


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "lengths, padding",
    [((1,), 0), ((6, 1, 3, 6), 5), ((700, 1, 2, 1300), 0), ((0, 4, 0, 2), 3)],
)
def test_engram_hash(lengths, padding, ids_dtype):
    geometry = build_geometry(
        layer_ids=LAYERS, base_table_size=16_000_000, compressed_vocab_size=COMPRESSED
    )
    token_map, ids, history, starts = _case(lengths, padding, sum(lengths) + padding)
    expected, image = _expected(token_map, ids, history, starts, geometry)
    out = torch.full((ids.numel() + 2, len(LAYERS), 24), 7, dtype=torch.int64)
    out, num_seqs, num_tokens = (
        out.to(DEVICE),
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
        torch.zeros(1, dtype=torch.int32, device=DEVICE),
    )
    engram_hash(
        ids.to(DEVICE, ids_dtype),
        image.to(DEVICE),
        starts.to(DEVICE),
        history.to(DEVICE),
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
