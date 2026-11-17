# SPDX-License-Identifier: Apache-2.0
"""The reindex (candidate) score kernel keeps a NaN head as DeepSeek's ``relu_`` does: a row
whose index query has an infinite scale scores NaN at every unmasked candidate, masked blocks
still score ``-inf``, and every other row keeps its bits (``test_indexer_nonfinite_query_scale``
for the candidate path)."""

import torch

from tests.kernels.attention.test_deepseek_v41_tilelang import (
    CANDIDATE_BLOCK,
    CANDIDATE_BLOCKS,
    DEVICE,
    IndexerV41Plan,
    _decode_mxfp4,
    _expanded,
    _gen,
    _index_inputs,
    _ref_scores,
    assert_same_nonfinite,
)


def test_indexer_candidates_nonfinite_query_scale(page_size=64, rows=4, context=8192):
    inp = _index_inputs(rows, context, page_size)
    width = CANDIDATE_BLOCKS * CANDIDATE_BLOCK
    gen = _gen(7)
    candidates = torch.full((rows, CANDIDATE_BLOCKS), -1, dtype=torch.int32, device=DEVICE)
    candidate_lengths = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
    for r in range(rows):
        visible = int(inp["lengths"][r]) - 3
        last = (visible - 1) // CANDIDATE_BLOCK
        n = int(torch.randint(64, last + 1, (1,), generator=gen, device=DEVICE))
        chosen = torch.randperm(last, generator=gen, device=DEVICE)[: n - 1]
        ids = torch.cat([chosen.sort().values, torch.tensor([last], device=DEVICE)])
        candidates[r, :n] = ids.to(torch.int32)
        candidate_lengths[r] = (n - 1) * CANDIDATE_BLOCK + visible - last * CANDIDATE_BLOCK
    candidates[1, 2] = -1  # a masked block in the NaN row keeps -inf
    plan = IndexerV41Plan(width, page_size, candidates=True)
    names = ("q_packed", "q_scales", "weights", "pool", "page_table", "lengths", "active")
    _, scores = plan.run(*(inp[k] for k in names), candidates, candidate_lengths)
    clean = scores.clone()
    r, h = 1, 3
    inp["q_scales"][r, h, 0] = 0xFF
    inp["q_packed"][r, h, 0] = 0  # a zero code: 0 x inf is NaN in any order
    inp["q_values"] = _decode_mxfp4(inp["q_packed"], inp["q_scales"])
    _, scores = plan.run(*(inp[k] for k in names), candidates, candidate_lengths)
    extent = candidate_lengths.clamp(0, width).tolist()
    positions = _expanded(candidates, candidate_lengths)
    ref = _ref_scores(inp, page_size, extent, lambda i: positions[i, : extent[i]])
    masked = slice(2 * CANDIDATE_BLOCK, 3 * CANDIDATE_BLOCK)
    assert bool(torch.isneginf(scores[r, masked]).all())
    live = torch.ones(extent[r], dtype=torch.bool, device=DEVICE)
    live[masked] = False
    assert bool(ref[r][live].isnan().all())
    assert_same_nonfinite(scores[r, : extent[r]][live], ref[r][live], f"row {r}: ")
    for i in range(rows):
        if i != r:
            assert torch.equal(scores[i, : extent[i]], clean[i, : extent[i]]), i
