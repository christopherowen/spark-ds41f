# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-finite values in the indexer's BF16 queries and keys. The MXFP4 quantizer gives a
group holding a NaN or an infinity the NaN scale 0xFF and zero codes, so its dot is NaN
however the MMA reads 0xFF; the score's ReLU keeps that NaN, as the reference's relu_
does. A NaN key reaches exactly the positions that read its slot, a NaN query exactly its
row; everything else keeps its bits."""

import pytest
import torch

from tests.kernels.attention.test_deepseek_v41_tilelang import (
    DEVICE,
    SF_BLOCK,
    _dense_extent,
    _index_inputs,
    _mxfp4_inputs,
    _quantize_mxfp4,
)
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    INDEX_DIM,
    INDEX_HEADS,
    IndexerV41Plan,
    quantize_mxfp4,
    write_index_keys,
)

NAMES = ("q_packed", "q_scales", "weights", "pool", "page_table", "lengths", "active")


def _quantize(x):
    q = torch.empty((*x.shape[:-1], INDEX_DIM // 2), dtype=torch.uint8, device=DEVICE)
    s = torch.empty(
        (*x.shape[:-1], INDEX_DIM // SF_BLOCK), dtype=torch.uint8, device=DEVICE
    )
    quantize_mxfp4(x, q, s)
    return q, s


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_quantizer_nonfinite_group(value, n=64):
    x = _mxfp4_inputs(n, 21)
    ref_q, ref_s = _quantize_mxfp4(x)
    row, col = 9, 70
    g = col // SF_BLOCK
    x[row, col] = value
    q, s = _quantize(x)
    assert int(s[row, g]) == 0xFF
    group = q[row, g * SF_BLOCK // 2 : (g + 1) * SF_BLOCK // 2].long()
    assert bool(((group & 7) == 0).all()) and bool(((group >> 4 & 7) == 0).all()), group
    # Every other group keeps B12X's bytes.
    keep_s = torch.ones_like(s, dtype=torch.bool)
    keep_s[row, g] = False
    keep_q = torch.ones_like(q, dtype=torch.bool)
    keep_q[row, g * SF_BLOCK // 2 : (g + 1) * SF_BLOCK // 2] = False
    assert torch.equal(s[keep_s], ref_s[keep_s])
    assert torch.equal(q[keep_q], ref_q[keep_q])


def _plan(inp, context, page_size):
    width = (context + page_size - 1) // page_size * page_size
    return IndexerV41Plan(width, page_size), width


def test_indexer_nan_key_reaches_its_positions(page_size=64, rows=4, context=4096):
    inp = _index_inputs(rows, context, page_size)
    plan, width = _plan(inp, context, page_size)
    _, clean = plan.run(*(inp[k] for k in NAMES))
    clean = clean.clone()
    # Row 0's position 5 (inside every extent) gets a key with a NaN, written by the
    # cache writer's quantizer.
    page = int(inp["page_table"][0, 0])
    token = 5
    key = torch.randn((1, INDEX_DIM), device=DEVICE).bfloat16()
    key[0, 40] = float("nan")
    slot = torch.tensor([page * page_size + token], dtype=torch.int64, device=DEVICE)
    write_index_keys(key, inp["pool"], slot, page_size)
    _, scores = plan.run(*(inp[k] for k in NAMES))
    extent = _dense_extent(inp, width)
    hits = 0
    for r in range(rows):
        e = extent[r]
        pos = torch.arange(e, device=DEVICE)
        pages = inp["page_table"][r].long()[pos // page_size]
        hit = (pages == page) & (pos % page_size == token)
        hits += int(hit.sum())
        assert bool(scores[r, :e][hit].isnan().all()), f"row {r}: the NaN key's score"
        assert torch.equal(scores[r, :e][~hit], clean[r, :e][~hit]), f"row {r}"
    assert hits >= 1


def test_indexer_nan_query_poisons_its_row(page_size=64, rows=4, context=4096):
    inp = _index_inputs(rows, context, page_size)
    plan, width = _plan(inp, context, page_size)
    _, clean = plan.run(*(inp[k] for k in NAMES))
    clean = clean.clone()
    r, h = 2, 17
    query = torch.randn((INDEX_HEADS, INDEX_DIM), device=DEVICE).bfloat16()
    query[h, 3] = float("nan")
    q, s = _quantize(query)
    inp["q_packed"][r] = q
    inp["q_scales"][r] = s
    _, scores = plan.run(*(inp[k] for k in NAMES))
    extent = _dense_extent(inp, width)
    assert bool(scores[r, : extent[r]].isnan().all())
    for i in range(rows):
        if i != r:
            assert torch.equal(scores[i, : extent[i]], clean[i, : extent[i]]), i
