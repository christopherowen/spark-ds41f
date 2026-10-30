# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang attention kernels against torch references.

Caches use vLLM's page layout, including allocator padding past each page's
records, so the kernels are checked against the strides they see in serving.
The sparse MLA stores WO-A's input: its output rotated back and through
``act_quant`` (``wo_input``), checked bit for bit on its own.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

tilelang = pytest.importorskip("tilelang")
pytest.importorskip("tile_kernels")

import tilelang.language as T  # noqa: E402

from tests.kernels.deepseek_v41_contracts import (  # noqa: E402
    assert_other_rows_equal,
    assert_rows_nan,
    assert_same_nonfinite,
    mxfp8_values,
    rope_table,
    wo_input_reference,
)
from vllm.models.deepseek_v4_1.tilelang.indexer import (  # noqa: E402
    CANDIDATE_BLOCK,
    CANDIDATE_BLOCKS,
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_TOPK,
    IndexerTopK,
    IndexerV41Plan,
    norm_rope_write_index_keys,
    page_bytes,
    quantize_mxfp4,
    write_index_keys,
)
from vllm.models.deepseek_v4_1.tilelang.sparse_mla import (  # noqa: E402
    HEAD_DIM,
    IDX_GROUP,
    ROPE_DIM,
    SWA_GROUP,
    SparseMlaV41Plan,
    wo_input,
)

DEVICE = "cuda"
SF_BLOCK = 32
PAD = 256  # allocator padding past each page's records
LAYERS = 3  # BLHNC: each block holds every layer's page, so a layer's rows stride
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _page_buffer(pages, page_bytes, fill=0xAB):
    """One layer's view of a BLHNC buffer: padded pages, rows strided over layers.

    The other layers' pages and the padding hold ``fill``.
    """
    buffer = torch.full(
        (pages, LAYERS, page_bytes + PAD), fill, dtype=torch.uint8, device=DEVICE
    )
    view = buffer[:, 1]
    assert view.stride(0) == LAYERS * (page_bytes + PAD)
    return view


def _paged(records, page_size):
    """Lay ``[slots, record_bytes]`` records out as one layer's cache pages."""
    slots, record_bytes = records.shape
    pages = slots // page_size
    pool = _page_buffer(pages, page_size * record_bytes)
    pool[:, : page_size * record_bytes] = records.view(pages, -1)
    return pool


def _e2m1_codes(normalized):
    lut = torch.tensor(E2M1, device=normalized.device)
    codes = (normalized.abs().clamp(max=6.0).unsqueeze(-1) - lut).abs().argmin(-1)
    return codes.to(torch.uint8) | (torch.signbit(normalized).to(torch.uint8) << 3)


def _e2m1_values(codes):
    lut = torch.tensor(E2M1, device=codes.device)
    return torch.where((codes & 8) != 0, -lut[codes & 7], lut[codes & 7])


# --------------------------------------------------------------------------
# Indexer
# --------------------------------------------------------------------------


def _e2m1_rne(x):
    """``cvt.rn.satfinite.e2m1``: nearest code, ties to even, saturating at 6."""
    m = x.abs()
    code = sum(
        (m > t if strict else m >= t).to(torch.uint8)
        for t, strict in (
            (0.25, True),
            (0.75, False),
            (1.25, True),
            (1.75, False),
            (2.5, True),
            (3.5, False),
            (5.0, True),
        )
    )
    return code | (torch.signbit(x).to(torch.uint8) << 3)


def _quantize_mxfp4(x):
    """B12X's MXFP4 quantizer: UE8M0 ``pow2_ceil(amax * fp32(1/6))``, RNE values."""
    groups = x.float().reshape(*x.shape[:-1], INDEX_DIM // SF_BLOCK, SF_BLOCK)
    amax = groups.abs().amax(-1).clamp_min(6.0 * 2.0**-126)
    bits = (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32)).view(torch.int32)
    sf = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = torch.exp2(127.0 - sf.float())
    codes = _e2m1_rne((groups * scale.unsqueeze(-1)).reshape(x.shape))
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), sf.to(torch.uint8).contiguous()


def _decode_mxfp4(packed, scales):
    u = packed.long()
    codes = torch.stack([u & 15, u >> 4], -1).reshape(*packed.shape[:-1], INDEX_DIM)
    scale = torch.exp2(scales.float() - 127.0).repeat_interleave(SF_BLOCK, -1)
    return _e2m1_values(codes) * scale


def _index_inputs(rows, context, page_size, seed=0):
    """Exactly representable MXFP4 operands, so scores compare bit for bit."""
    gen = _gen(seed)
    logical_pages = (context + page_size - 1) // page_size
    pool_pages = logical_pages * rows + 3
    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), device=DEVICE)

    def operand(shape):
        values = lut[
            torch.randint(0, 16, (*shape, INDEX_DIM), generator=gen, device=DEVICE)
        ]
        exps = torch.randint(
            -3, 4, (*shape, INDEX_DIM // SF_BLOCK), generator=gen, device=DEVICE
        )
        return values * torch.exp2(exps.float()).repeat_interleave(SF_BLOCK, -1)

    q_packed, q_scales = _quantize_mxfp4(operand((rows, INDEX_HEADS)))
    k_packed, k_scales = _quantize_mxfp4(operand((pool_pages * page_size,)))
    pool = _page_buffer(pool_pages, page_bytes(page_size))
    pool[:, : page_size * 64].copy_(k_packed.view(pool_pages, -1))
    pool[:, page_size * 64 : page_size * 68].copy_(k_scales.view(pool_pages, -1))
    weights = (
        torch.randn((rows, INDEX_HEADS), generator=gen, device=DEVICE) / 64
    ).bfloat16()
    page_table = torch.stack(
        [
            torch.randperm(pool_pages, generator=gen, device=DEVICE)[:logical_pages]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    lengths = torch.randint(
        context // 2,
        context + 1,
        (rows,),
        generator=gen,
        device=DEVICE,
        dtype=torch.int32,
    )
    active = torch.tensor([context], dtype=torch.int32, device=DEVICE)
    return dict(
        q_packed=q_packed,
        q_scales=q_scales,
        weights=weights,
        pool=pool,
        page_table=page_table,
        lengths=lengths,
        active=active,
        q_values=_decode_mxfp4(q_packed, q_scales),
        k_values=_decode_mxfp4(k_packed, k_scales),
    )


def _mxfp4_inputs(n, seed):
    gen = _gen(seed)
    x = torch.randn((n, INDEX_DIM), generator=gen, device=DEVICE) * 3
    # Exact midpoints after scaling exercise the ties-to-even rule.
    x[:4] = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device=DEVICE
    ).repeat(16)[:INDEX_DIM]
    x[4] = -x[0]
    x[5] = 0
    x[6, :32] = -0.0
    return x.bfloat16()


def test_mxfp4_quantize(n=3000):
    x = _mxfp4_inputs(n, 11)
    q = torch.empty((n, 64), dtype=torch.uint8, device=DEVICE)
    scales = torch.empty((n, 4), dtype=torch.uint8, device=DEVICE)
    quantize_mxfp4(x, q, scales)
    ref_q, ref_scales = _quantize_mxfp4(x)
    assert torch.equal(scales, ref_scales)
    assert torch.equal(q, ref_q)


@pytest.mark.parametrize("rows", [1, 7, 37, 128])
def test_mxfp4_quantize_with_rope(rows):
    """Fused index-query RoPE + MXFP4 quantize produces the exact bytes of running
    ``rope_`` then ``quantize_mxfp4``, without mutating the BF16 input."""
    from vllm.models.deepseek_v4_1.tilelang.rope import rope_

    gen = _gen(rows + 30)
    iq = _mxfp4_inputs(rows * INDEX_HEADS, rows + 31).view(
        rows, INDEX_HEADS, INDEX_DIM
    )
    iq_orig = iq.clone()
    iq_ref = iq.clone()
    positions = torch.randint(0, 4096, (rows,), generator=gen, device=DEVICE)
    table = rope_table()
    rope_(iq_ref, positions, table, 1, False)
    ref_q = torch.empty((rows, INDEX_HEADS, 64), dtype=torch.uint8, device=DEVICE)
    ref_scales = torch.empty((rows, INDEX_HEADS, 4), dtype=torch.uint8, device=DEVICE)
    quantize_mxfp4(iq_ref, ref_q, ref_scales)

    q = torch.empty_like(ref_q)
    scales = torch.empty_like(ref_scales)
    quantize_mxfp4(iq, q, scales, positions=positions, cos_sin=table)
    assert torch.equal(iq.view(torch.int16), iq_orig.view(torch.int16))
    assert torch.equal(scales, ref_scales)
    assert torch.equal(q, ref_q)


def test_write_index_keys(page_size=64, pages=5):
    n = pages * page_size // 2
    x = _mxfp4_inputs(n, 12)
    gen = _gen(13)
    slots = torch.randperm(pages * page_size, generator=gen, device=DEVICE)[:n]
    slots[:3] = -1  # padding rows write nothing
    pool = _page_buffer(pages, page_bytes(page_size))
    expected = pool.clone()
    write_index_keys(x, pool, slots, page_size)
    ref_q, ref_scales = _quantize_mxfp4(x)
    keep = slots >= 0
    page, token = slots[keep] // page_size, slots[keep] % page_size
    data = expected[:, : page_size * 64].reshape(pages, page_size, 64)
    scale_rows = expected[:, page_size * 64 : page_size * 68].reshape(
        pages, page_size, 4
    )
    data[page, token] = ref_q[keep]
    scale_rows[page, token] = ref_scales[keep]
    expected[:, : page_size * 64] = data.reshape(pages, -1)
    expected[:, page_size * 64 : page_size * 68] = scale_rows.reshape(pages, -1)
    assert torch.equal(pool, expected)
    # Neither the other layers' pages nor the padding were written.
    assert bool((pool._base[:, [0, 2]] == 0xAB).all())


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
def test_norm_rope_write_index_keys(ratio, slot_dtype, page_size=64, pages=5):
    from vllm.models.deepseek_v4_1.tilelang import pdl
    from vllm.models.deepseek_v4_1.tilelang.rope import rope_

    n = pages * page_size // 2
    x = _mxfp4_inputs(n, 14)
    gen = _gen(15)
    weight = torch.randn((INDEX_DIM,), generator=gen, device=DEVICE).bfloat16()
    eps = 1e-6
    positions = torch.randint(
        0, 512, (n,), generator=gen, device=DEVICE, dtype=torch.int64
    )
    table = rope_table(512)
    slots = torch.randperm(pages * page_size, generator=gen, device=DEVICE)[:n].to(
        slot_dtype
    )
    slots[:3] = -1
    pool = _page_buffer(pages, page_bytes(page_size))
    expected = pool.clone()
    norm_rope_write_index_keys(
        x, weight, eps, positions, table, pool, slots, page_size, ratio
    )
    ref = torch.empty_like(x)
    pdl.norm_forward(x, weight, eps, out=ref)
    rope_(ref, positions, table, ratio, False)
    write_index_keys(ref, expected, slots, page_size)
    assert torch.equal(pool, expected)
    assert bool((pool._base[:, [0, 2]] == 0xAB).all())


def _ref_scores(inp, page_size, extent, positions_of):
    """Staged BF16 score: BF16 dot, ReLU, BF16 weighted product, ordered head sum."""
    rows = inp["q_values"].shape[0]
    scores = []
    for r in range(rows):
        positions = positions_of(r)
        safe = positions.clamp_min(0).long()
        slots = (
            inp["page_table"][r].long()[safe // page_size] * page_size
            + safe % page_size
        )
        dots = (inp["k_values"][slots].double() @ inp["q_values"][r].double().T).float()
        contrib = dots.bfloat16().float().clamp_min(0) * inp["weights"][r].float()
        contrib = contrib.bfloat16().float()
        total = torch.zeros(len(positions), device=DEVICE)
        for h in range(INDEX_HEADS):
            total = total + contrib[:, h]
        total = torch.where(positions >= 0, total, float("-inf")).bfloat16()
        scores.append(total[: extent[r]])
    return scores


def _ref_topk(scores, extent, topk, group=1, gather=None):
    """Larger scores first, then lower positions; ascending, ``-1`` padded. Grouped
    selections return block ids and their count of positions below the extent;
    ``gather`` (candidate rows) maps entries to positions."""
    rows = scores.shape[0]
    out = torch.full((rows, topk), -1, dtype=torch.int32, device=DEVICE)
    counts = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
    for r in range(rows):
        n = int(extent[r])
        values = scores[r, :n].float()
        if group > 1:
            blocks = (n + group - 1) // group
            padded = torch.full((blocks * group,), float("-inf"), device=DEVICE)
            padded[:n] = values
            values = padded.view(blocks, group).amax(1)
            if blocks:
                values[(n - 1) // group] = float("inf")
        order = torch.sort(values, descending=True, stable=True).indices
        chosen = order[values[order] > float("-inf")][:topk].sort().values
        if group > 1:
            pos = chosen[:, None] * group + torch.arange(group, device=DEVICE)
            counts[r] = int((pos < n).sum())
        elif gather is not None:
            chosen = gather[r, chosen]
        out[r, : chosen.numel()] = chosen.to(torch.int32)
    return out, counts


def _expanded(ids, counts, group=CANDIDATE_BLOCK):
    """Each block's ``group`` positions below its row's count, ``-1`` elsewhere (and
    for a masked block): the candidate positions a block-id list stands for."""
    rows, blocks = ids.shape
    j = torch.arange(blocks * group, device=DEVICE)
    block = ids[:, j // group]
    pos = block * group + j % group
    live = (j[None] < counts[:, None]) & (block >= 0)
    return torch.where(live, pos, -1).to(torch.int32)


def _release_expansion(ids, extent, group=CANDIDATE_BLOCK):
    """The release's position list of selected blocks: every block's ``group``
    positions, ``-1`` from the extent on, then ``-1`` padding (26382f5cf)."""
    rows, blocks = ids.shape
    out = torch.full((rows, blocks * group), -1, dtype=torch.int32, device=DEVICE)
    for r in range(rows):
        chosen = ids[r][ids[r] >= 0].long()
        pos = (chosen[:, None] * group + torch.arange(group, device=DEVICE)).flatten()
        out[r, : pos.numel()] = torch.where(pos < extent[r], pos, -1).to(torch.int32)
    return out


def _dense_extent(inp, width):
    return inp["lengths"].clamp(0, int(inp["active"][0])).clamp(max=width).tolist()


@pytest.mark.parametrize("context", [4096, 9000])
def test_indexer_dense(context, page_size=64, rows=4):
    inp = _index_inputs(rows, context, page_size)
    width = (context + page_size - 1) // page_size * page_size
    plan = IndexerV41Plan(width, page_size)
    out, scores = plan.run(
        *(
            inp[k]
            for k in (
                "q_packed",
                "q_scales",
                "weights",
                "pool",
                "page_table",
                "lengths",
                "active",
            )
        )
    )
    extent = _dense_extent(inp, width)
    ref = _ref_scores(
        inp, page_size, extent, lambda r: torch.arange(extent[r], device=DEVICE)
    )
    for r in range(rows):
        assert torch.equal(scores[r, : extent[r]], ref[r]), f"row {r} scores differ"
    expected, _ = _ref_topk(scores, extent, INDEX_TOPK)
    assert torch.equal(out, expected)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the indexer's score ReLU is max(dot, 0), which drops a NaN; the NaN-keeping "
    "forms tried so far either serialize the per-head loop (isnan, a 32-bit bit test) or "
    "are folded away by TileLang's simplifier (dot != dot), so the fix waits for a "
    "vectorizable NaN-keeping max (spark-ds41f experiments/2026-10-14-kernel-contracts)",
)
def test_indexer_nonfinite_query_scale(page_size=64, rows=4, context=4096):
    """An infinite index query quantizes to the UE8M0 scale ``0xFF``, which makes its
    head's dot NaN. DeepSeek's indexer takes ``relu_`` of the score and keeps the
    NaN; ``max(dot, 0)`` would drop the head silently. The other rows keep their
    bits."""
    inp = _index_inputs(rows, context, page_size)
    width = (context + page_size - 1) // page_size * page_size
    plan = IndexerV41Plan(width, page_size)
    names = (
        "q_packed",
        "q_scales",
        "weights",
        "pool",
        "page_table",
        "lengths",
        "active",
    )
    _, scores = plan.run(*(inp[k] for k in names))
    clean = scores.clone()
    r, h = 1, 3
    inp["q_scales"][r, h, 0] = 0xFF
    inp["q_packed"][r, h, 0] = 0  # a zero code: 0 x inf is NaN in any order
    inp["q_values"] = _decode_mxfp4(inp["q_packed"], inp["q_scales"])
    _, scores = plan.run(*(inp[k] for k in names))
    extent = _dense_extent(inp, width)
    ref = _ref_scores(
        inp, page_size, extent, lambda i: torch.arange(extent[i], device=DEVICE)
    )
    assert ref[r].isnan().all()
    assert_same_nonfinite(scores[r, : extent[r]], ref[r], f"row {r}: ")
    for i in range(rows):
        if i != r:
            assert torch.equal(scores[i, : extent[i]], clean[i, : extent[i]]), i


def test_indexer_candidates(page_size=64, rows=4, context=8192):
    """Reindex rows read block-id candidate lists, as the source layer publishes
    them: candidate ``j`` is position ``ids[j // 8] * 8 + j % 8``, the last block
    holding the row's last position (here only partly). A masked block's candidates
    score ``-inf``; an empty row selects nothing."""
    inp = _index_inputs(rows, context, page_size)
    width = CANDIDATE_BLOCKS * CANDIDATE_BLOCK
    gen = _gen(7)
    candidates = torch.full(
        (rows, CANDIDATE_BLOCKS), -1, dtype=torch.int32, device=DEVICE
    )
    candidate_lengths = torch.zeros(rows, dtype=torch.int32, device=DEVICE)
    for r in range(rows - 1):  # the last row stays empty
        visible = int(inp["lengths"][r]) - 3  # the last block partly visible
        last = (visible - 1) // CANDIDATE_BLOCK
        n = int(torch.randint(64, last + 1, (1,), generator=gen, device=DEVICE))
        chosen = torch.randperm(last, generator=gen, device=DEVICE)[: n - 1]
        ids = torch.cat([chosen.sort().values, torch.tensor([last], device=DEVICE)])
        candidates[r, :n] = ids.to(torch.int32)
        last_visible = visible - last * CANDIDATE_BLOCK  # 1 to 8 positions
        candidate_lengths[r] = (n - 1) * CANDIDATE_BLOCK + last_visible
    candidates[0, 1] = -1  # a masked block: its candidates score -inf
    plan = IndexerV41Plan(width, page_size, candidates=True)
    args = (
        inp[k]
        for k in (
            "q_packed",
            "q_scales",
            "weights",
            "pool",
            "page_table",
            "lengths",
            "active",
        )
    )
    out, scores = plan.run(*args, candidates, candidate_lengths)
    extent = candidate_lengths.clamp(0, width).tolist()
    positions = _expanded(candidates, candidate_lengths)
    ref = _ref_scores(inp, page_size, extent, lambda r: positions[r, : extent[r]])
    for r in range(rows):
        assert torch.equal(scores[r, : extent[r]], ref[r]), f"row {r} scores differ"
    expected, _ = _ref_topk(scores, extent, INDEX_TOPK, gather=positions)
    assert torch.equal(out, expected)
    assert bool((out[rows - 1] == -1).all())


def _compute_sanitizer() -> str | None:
    path = os.environ.get("COMPUTE_SANITIZER") or shutil.which("compute-sanitizer")
    if path is None and os.path.exists("/usr/local/cuda/bin/compute-sanitizer"):
        path = "/usr/local/cuda/bin/compute-sanitizer"
    return path


def test_indexer_reads_only_visible_rows():
    """No key or scale past a row's visible positions is read, as DeepSeek's reference
    reads ``index_k[:, : end_pos // ratio]``: compute-sanitizer's initcheck scores a
    cache written only at the visible positions, dense and by candidates, and reports
    any read of the rest (deepseek_v41_indexer_reads.py)."""
    sanitizer = _compute_sanitizer()
    if sanitizer is None:
        pytest.skip("compute-sanitizer not found (COMPUTE_SANITIZER names it)")
    script = Path(__file__).with_name("deepseek_v41_indexer_reads.py")
    run = subprocess.run(
        [sanitizer, "--tool", "initcheck", "--error-exitcode", "1",
         sys.executable, str(script)],
        env={**os.environ, "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"},
        capture_output=True,
        text=True,
        timeout=3600,
    )  # fmt: skip
    report = run.stdout + run.stderr
    assert run.returncode == 0 and "ERROR SUMMARY: 0 errors" in report, report[-6000:]


@pytest.mark.parametrize("page_size", [128, 256])
def test_indexer_scores_ignore_rows_past_the_extent(page_size, rows=4, context=9000):
    """NaN bytes in every slot past a row's extent (a UE8M0 scale of 0xFF) leave every
    stored score bit-identical: no row past the extent reaches a result."""
    inp = _index_inputs(rows, context, page_size)
    inp["lengths"][:] = torch.tensor([37, 1061, 4133, 8999], device=DEVICE)
    width = (context + page_size - 1) // page_size * page_size
    # Disjoint pages, so one row's poisoned tail is no other row's visible key.
    logical = width // page_size
    inp["page_table"] = (
        torch.randperm(inp["pool"].shape[0], generator=_gen(5), device=DEVICE)
        [: rows * logical]
        .view(rows, logical)
        .to(torch.int32)
    )
    plan = IndexerV41Plan(width, page_size)
    names = (
        "q_packed",
        "q_scales",
        "weights",
        "pool",
        "page_table",
        "lengths",
        "active",
    )
    _, clean = plan.run(*(inp[k] for k in names))
    clean = clean.clone()
    extent = _dense_extent(inp, width)
    for r in range(rows):
        for position in range(extent[r], width):
            page = int(inp["page_table"][r, position // page_size])
            offset = position % page_size
            inp["pool"][page, offset * 64 : offset * 64 + 64] = 0xFF
            words = page_size * 64 + offset * 4
            inp["pool"][page, words : words + 4] = 0xFF
    _, poisoned = plan.run(*(inp[k] for k in names))
    for r in range(rows):
        assert torch.equal(poisoned[r, : extent[r]], clean[r, : extent[r]]), r


def test_indexer_publishes_candidates(page_size=64, rows=2, context=20000):
    inp = _index_inputs(rows, context, page_size)
    width = (context + page_size - 1) // page_size * page_size
    plan = IndexerV41Plan(width, page_size, candidate_blocks=True)
    candidate_out = torch.full(
        (rows, CANDIDATE_BLOCKS), -7, dtype=torch.int32, device=DEVICE
    )
    counts = torch.full((rows,), -7, dtype=torch.int32, device=DEVICE)
    args = (
        inp[k]
        for k in (
            "q_packed",
            "q_scales",
            "weights",
            "pool",
            "page_table",
            "lengths",
            "active",
        )
    )
    out, scores = plan.run(
        *args, candidate_out=candidate_out, candidate_out_lengths=counts
    )
    extent = _dense_extent(inp, width)
    expected, _ = _ref_topk(scores, extent, INDEX_TOPK)
    assert torch.equal(out, expected)
    blocks, block_counts = _ref_topk(scores, extent, CANDIDATE_BLOCKS, CANDIDATE_BLOCK)
    assert torch.equal(candidate_out, blocks)
    assert torch.equal(counts, block_counts)


def _poison_past_live(page_table, lengths, page_size, seed=11):
    """``page_table`` with every column past each row's live pages (those holding its
    first ``lengths`` positions, all the TileLang metadata writes) set to page ids no
    pool holds, negative ones among them."""
    live = (lengths.long().clamp_min(0) + page_size - 1) // page_size
    cols = torch.arange(page_table.shape[1], device=DEVICE)
    past = cols[None, :] >= live[:, None]
    assert past.any(), "no column to poison"
    garbage = torch.randint(
        -(2**31),
        2**31 - 1,
        page_table.shape,
        generator=_gen(seed),
        device=DEVICE,
        dtype=torch.int32,
    )
    return torch.where(past, garbage, page_table)


@pytest.mark.parametrize("candidates", [False, True])
def test_indexer_reads_only_live_pages(candidates, page_size=64, rows=4, context=9000):
    """The TileLang metadata writes a row's index page table only over the pages of
    its first ``lengths`` positions (``tilelang.metadata.pages``). The indexer looks
    up no other column: with every other column poisoned, the scores and the
    selections keep their bits."""
    inp = _index_inputs(rows, context, page_size)
    inp["lengths"] = torch.tensor(
        [1, 700, 3001, context], dtype=torch.int32, device=DEVICE
    )
    keys = (
        "q_packed",
        "q_scales",
        "weights",
        "pool",
        "page_table",
        "lengths",
        "active",
    )
    extra: tuple = ()
    if candidates:
        # Block ids as the candidate source writes them: ascending, the block holding
        # the row's last position always kept (partly visible), -1 padding past them.
        width = CANDIDATE_BLOCKS * CANDIDATE_BLOCK
        gen = _gen(7)
        candidates_in = torch.full(
            (rows, CANDIDATE_BLOCKS), -1, dtype=torch.int32, device=DEVICE
        )
        counts = []
        for r, n in enumerate((1, 75, 256, 512)):
            visible = int(inp["lengths"][r])
            last = (visible - 1) // CANDIDATE_BLOCK
            chosen = torch.randperm(last, generator=gen, device=DEVICE)[: n - 1]
            ids = torch.cat([chosen.sort().values, torch.tensor([last], device=DEVICE)])
            candidates_in[r, :n] = ids.to(torch.int32)
            counts.append((n - 1) * CANDIDATE_BLOCK + visible - last * CANDIDATE_BLOCK)
        candidates_in[2, 5] = -1  # a masked block looks up column 0
        candidate_lengths = torch.tensor(counts, dtype=torch.int32, device=DEVICE)
        extra = (candidates_in, candidate_lengths)
        extent = counts
    else:
        width = (context + page_size - 1) // page_size * page_size
        extent = _dense_extent(inp, width)
    plan = IndexerV41Plan(width, page_size, candidates=candidates)
    clean_out, clean_scores = plan.run(*(inp[k] for k in keys), *extra)
    inp["page_table"] = _poison_past_live(inp["page_table"], inp["lengths"], page_size)
    out, scores = plan.run(*(inp[k] for k in keys), *extra)
    for r in range(rows):
        assert torch.equal(scores[r, : extent[r]], clean_scores[r, : extent[r]]), (
            f"row {r} looked up a page past its live pages"
        )
    assert torch.equal(out, clean_out)


def test_candidate_chain_matches_the_release_positions(page_size=64, context=20000):
    """The source layer's block ids stand for exactly the release's position list
    (blocks expanded, ``-1`` from the extent on), so the reindex layer that reads
    them scores and selects the release's positions bit for bit: rows with more
    visible blocks than are kept, a partly visible last block, few blocks, none."""
    rows = 4
    inp = _index_inputs(rows, context, page_size)
    inp["lengths"] = torch.tensor(
        [context, context - 3, 1003, 0], dtype=torch.int32, device=DEVICE
    )
    width = (context + page_size - 1) // page_size * page_size
    source = IndexerV41Plan(width, page_size, candidate_blocks=True)
    names = ("q_packed", "q_scales", "weights", "pool", "page_table", "lengths")
    ids = torch.full((rows, CANDIDATE_BLOCKS), -7, dtype=torch.int32, device=DEVICE)
    counts = torch.full((rows,), -7, dtype=torch.int32, device=DEVICE)
    _, scores = source.run(
        *(inp[k] for k in names),
        inp["active"],
        candidate_out=ids,
        candidate_out_lengths=counts,
    )
    extent = _dense_extent(inp, width)
    expected_ids, expected_counts = _ref_topk(
        scores, extent, CANDIDATE_BLOCKS, CANDIDATE_BLOCK
    )
    assert torch.equal(ids, expected_ids) and torch.equal(counts, expected_counts)
    positions = _release_expansion(ids, extent)
    # No position list of the release has a hole before its count.
    assert torch.equal(_expanded(ids, counts), positions)

    reindex = IndexerV41Plan(
        CANDIDATE_BLOCKS * CANDIDATE_BLOCK, page_size, candidates=True
    )
    out, re_scores = reindex.run(*(inp[k] for k in names), inp["active"], ids, counts)
    re_extent = counts.tolist()
    ref = _ref_scores(inp, page_size, re_extent, lambda r: positions[r, : re_extent[r]])
    for r in range(rows):
        assert torch.equal(re_scores[r, : re_extent[r]], ref[r]), f"row {r} scores"
    expected, _ = _ref_topk(re_scores, re_extent, INDEX_TOPK, gather=positions)
    assert torch.equal(out, expected)


def _tie_scores(rows, width, seed=3):
    # Few distinct values force large ties at every selection threshold.
    gen = _gen(seed)
    return (
        torch.randint(0, 6, (rows, width), generator=gen, device=DEVICE).float() * 0.25
    ).bfloat16()


@pytest.mark.parametrize(
    "width, topk, group",
    [
        (4000, INDEX_TOPK, 1),  # one pass
        (20000, INDEX_TOPK, 1),  # segments, one merge
        (300000, INDEX_TOPK, 1),  # segments, two merge levels
        (
            4000 * CANDIDATE_BLOCK + 8,
            CANDIDATE_BLOCKS,
            CANDIDATE_BLOCK,
        ),  # blocks, one merge
        (200000, CANDIDATE_BLOCKS, CANDIDATE_BLOCK),  # blocks, two merge levels
    ],
)
def test_topk_routes(width, topk, group, rows=4):
    scores = _tie_scores(rows, width)
    scores[0, 100:] = float("-inf")  # fewer selectable entries than topk
    lengths = torch.tensor(
        [width, width - 17, 1000, 0], dtype=torch.int32, device=DEVICE
    )
    active = torch.tensor([width], dtype=torch.int32, device=DEVICE)
    op = IndexerTopK(width, topk, group)
    indices = torch.full((rows, topk), -7, dtype=torch.int32, device=DEVICE)
    values = torch.empty((rows, topk), dtype=torch.bfloat16, device=DEVICE)
    counts = (
        torch.full((rows,), -7, dtype=torch.int32, device=DEVICE) if group > 1 else None
    )
    op.run(scores, lengths, active, indices, values, counts=counts)
    expected, expected_counts = _ref_topk(scores, lengths.tolist(), topk, group)
    assert torch.equal(indices, expected), (
        f"{op.segments} segments, {len(op.levels)} merges"
    )
    if counts is not None:
        assert torch.equal(counts, expected_counts)


# --------------------------------------------------------------------------
# Sparse MLA
# --------------------------------------------------------------------------


def _pack_swa(kv):
    n = kv.shape[0]
    groups = kv.float().reshape(n, HEAD_DIM // SWA_GROUP, SWA_GROUP)
    exponent = torch.ceil(
        torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0)
    ).clamp(-126, 127)
    payload = (
        (groups / torch.exp2(exponent).unsqueeze(-1))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    scales = (exponent + 127).to(torch.uint8)
    return torch.cat(
        [payload.view(torch.uint8).reshape(n, HEAD_DIM), scales], -1
    ).contiguous()


def _pack_idx(kv):
    n = kv.shape[0]
    groups = kv.float().reshape(n, HEAD_DIM // IDX_GROUP, IDX_GROUP)
    scales = (
        (groups.abs().amax(-1).clamp_min(6 * 2**-9) / 6.0)
        .clamp(max=448)
        .to(torch.float8_e4m3fn)
    )
    codes = _e2m1_codes(groups / scales.float().unsqueeze(-1)).reshape(n, HEAD_DIM)
    payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return torch.cat([payload, scales.view(torch.uint8)], -1).contiguous()


def _decode_swa(records):
    payload = records[:, :HEAD_DIM].contiguous().view(torch.float8_e4m3fn).float()
    scales = torch.exp2(records[:, HEAD_DIM:].float() - 127.0)
    return (
        payload.reshape(-1, HEAD_DIM // SWA_GROUP, SWA_GROUP) * scales.unsqueeze(-1)
    ).reshape(-1, HEAD_DIM)


def _decode_idx(records):
    packed = records[:, : HEAD_DIM // 2].long()
    codes = torch.stack([packed & 15, packed >> 4], -1).reshape(-1, HEAD_DIM)
    scales = records[:, HEAD_DIM // 2 :].contiguous().view(torch.float8_e4m3fn).float()
    values = _e2m1_values(codes).reshape(-1, HEAD_DIM // IDX_GROUP, IDX_GROUP)
    return (values * scales.unsqueeze(-1)).reshape(-1, HEAD_DIM)


def _ref_mla(q, swa, swa_idx, swa_len, sinks, idx=None, idx_slots=None, idx_len=None):
    rows, heads, _ = q.shape
    out = torch.empty((rows, heads, HEAD_DIM), dtype=torch.float32, device=DEVICE)
    for r in range(rows):
        s = swa_idx[r, : int(swa_len[r])]
        kv = [_decode_swa(swa[s[s >= 0].long()])]
        if idx is not None:
            i = idx_slots[r, : int(idx_len[r])]
            kv.append(_decode_idx(idx[i[i >= 0].long()]))
        kv = torch.cat(kv)
        scores = q[r].float() @ kv.T * HEAD_DIM**-0.5
        top = torch.maximum(scores.max(-1).values if kv.shape[0] else sinks, sinks)
        weights = torch.exp(scores - top[:, None])
        out[r] = weights @ kv / (weights.sum(-1) + torch.exp(sinks - top))[:, None]
    return out.bfloat16()


def _mla_case(idx_width, heads, rows, swa_width=128, swa_page=128, idx_page=128):
    """Random queries, sinks, packed caches and selection lists for ``rows`` rows."""
    gen = _gen(1)
    swa_pages, idx_pages, idx_context = 24, 40, 4096
    q = torch.randn((rows, heads, HEAD_DIM), generator=gen, device=DEVICE).bfloat16()
    sinks = torch.randn((heads,), generator=gen, device=DEVICE)
    swa = _pack_swa(
        torch.randn((swa_pages * swa_page, HEAD_DIM), generator=gen, device=DEVICE)
    )

    def lists(width, extent, min_len):
        lengths = torch.randint(
            min_len, width + 1, (rows,), generator=gen, device=DEVICE, dtype=torch.int32
        )
        chosen = torch.stack(
            [
                torch.randperm(extent, generator=gen, device=DEVICE)[:width]
                for _ in range(rows)
            ]
        )
        keep = torch.arange(width, device=DEVICE) < lengths[:, None]
        return torch.where(keep, chosen, -1).to(torch.int32).contiguous(), lengths

    swa_idx, swa_len = lists(swa_width, swa_pages * swa_page, 1)
    plan = SparseMlaV41Plan(
        heads, swa_width, idx_width, idx_page, swa_page_size=swa_page,
        max_rows=rows, decode_max_rows=rows, split_max_rows=rows,
    )  # fmt: skip
    case = dict(
        plan=plan, q=q, sinks=sinks, swa=swa, swa_idx=swa_idx, swa_len=swa_len,
        swa_page=swa_page, idx_page=idx_page, idx=None,
    )  # fmt: skip
    if idx_width:
        idx = _pack_idx(
            torch.randn((idx_pages * idx_page, HEAD_DIM), generator=gen, device=DEVICE)
        )
        idx_logical, idx_len = lists(idx_width, idx_context, 0)
        table = torch.stack(
            [
                torch.randperm(idx_pages, generator=gen, device=DEVICE)[
                    : idx_context // idx_page
                ]
                for _ in range(rows)
            ]
        ).to(torch.int32)
        case.update(idx=idx, idx_logical=idx_logical, idx_len=idx_len, table=table)
    case.update(
        positions=torch.randint(0, 4096, (rows,), generator=_gen(2), device=DEVICE),
        rope=rope_table(),
    )
    return case


def _idx_slots(case, logical):
    page = case["idx_page"]
    pages = case["table"].long().gather(1, logical.long().clamp_min(0) // page)
    return torch.where(logical >= 0, pages * page + logical.long() % page, -1)


def _mla_run(case, swa_pool=None, idx_pool=None, swa_idx=None, idx_logical=None):
    """Every route's output, from the case's caches unless pools are given."""
    plan, rows = case["plan"], case["q"].shape[0]
    indexed = {}
    if case["idx"] is not None:
        indexed = dict(
            idx_cache=_paged(case["idx"], case["idx_page"])
            if idx_pool is None
            else idx_pool,
            idx_indices=case["idx_logical"] if idx_logical is None else idx_logical,
            idx_lengths=case["idx_len"],
            idx_page_table=case["table"],
        )
    pool = _paged(case["swa"], case["swa_page"]) if swa_pool is None else swa_pool
    outs = {}
    for kernel in plan.kernels:
        out = torch.empty(
            (rows, case["q"].shape[1], HEAD_DIM), dtype=torch.bfloat16, device=DEVICE
        )
        plan.run(
            case["q"],
            pool,
            case["swa_idx"] if swa_idx is None else swa_idx,
            case["swa_len"],
            case["sinks"],
            case["positions"],
            case["rope"],
            out,
            kernel=kernel,
            **indexed,
        )
        outs[kernel] = out
    return outs


def _mla_ref(case, swa=None):
    ref_indexed = {}
    if case["idx"] is not None:
        ref_indexed = dict(
            idx=case["idx"],
            idx_slots=_idx_slots(case, case["idx_logical"]),
            idx_len=case["idx_len"],
        )
    swa = case["swa"] if swa is None else swa
    out = _ref_mla(
        case["q"], swa, case["swa_idx"], case["swa_len"], case["sinks"], **ref_indexed
    )
    return wo_input_reference(out, case["positions"], case["rope"])


@pytest.mark.parametrize("idx_width", [0, 512])
# Heads per rank: TP4 (16), TP3 (24 of 72 padded), TP1 (64).
@pytest.mark.parametrize("heads", [16, 24, 64])
def test_sparse_mla(idx_width, heads, rows=6):
    """Every route's WO-A input, near the reference's: within one E4M3 step (an
    attention output a rounding away from the reference's can land on the next MXFP8
    value), exactly act_quant's values (requantizing changes nothing), the same bits
    on every route."""
    case = _mla_case(idx_width, heads, rows)
    outs = _mla_run(case)
    ref = _mla_ref(case)
    torch.testing.assert_close(
        outs["extend"].float(), ref.float(), rtol=0.13, atol=2e-2
    )
    flat = outs["extend"].view(rows, -1)
    assert torch.equal(mxfp8_values(flat).bfloat16(), flat)
    for kernel, out in outs.items():
        assert torch.equal(out, outs["extend"]), f"{kernel} differs from extend"


def _selected(slots, lengths):
    """The slots any row selects within its length."""
    keep = torch.arange(slots.shape[1], device=DEVICE) < lengths[:, None].long()
    return torch.unique(slots[keep & (slots >= 0)].long())


def _poisoned_pool(records, page_size):
    """``_paged`` with the padding and the other layers' pages all ``0xFF``."""
    slots, record_bytes = records.shape
    pages = slots // page_size
    pool = _page_buffer(pages, page_size * record_bytes, fill=0xFF)
    pool[:, : page_size * record_bytes] = records.view(pages, -1)
    return pool


@pytest.mark.parametrize("idx_width", [0, 512])
def test_sparse_mla_reads_only_selected_records(idx_width, heads=16, rows=6):
    """DeepSeek's ``sparse_attn`` loads zeros for slots it does not select. Here every
    record no row selects is poisoned (``0xFF`` bytes: NaN payloads and the largest
    scale), page 0, the padding and the other layers' pages included, and each list
    points past its length at poisoned records: every route keeps the clean bits.
    A masked record that reached the arithmetic would turn its row NaN, since a zero
    weight times NaN is NaN."""
    case = _mla_case(idx_width, heads, rows)
    # Masked slots clamp to record 0, so no row may select it.
    case["swa_idx"][case["swa_idx"] == 0] = -1
    if idx_width:
        case["idx_logical"][_idx_slots(case, case["idx_logical"]) == 0] = -1
    clean = _mla_run(case)

    def poison(records, used):
        records = records.clone()
        unused = torch.ones(records.shape[0], dtype=torch.bool, device=DEVICE)
        unused[used] = False
        assert unused[0]
        records[unused] = 0xFF
        return records, unused.nonzero().flatten()

    swa, swa_unused = poison(case["swa"], _selected(case["swa_idx"], case["swa_len"]))
    swa_idx = case["swa_idx"].clone()
    past = torch.arange(swa_idx.shape[1], device=DEVICE) >= case["swa_len"][:, None]
    swa_idx[past] = int(swa_unused[-1])
    kwargs = dict(swa_pool=_poisoned_pool(swa, case["swa_page"]), swa_idx=swa_idx)
    if idx_width:
        used = _selected(_idx_slots(case, case["idx_logical"]), case["idx_len"])
        idx, idx_unused = poison(case["idx"], used)
        logical = case["idx_logical"].clone()
        everything = torch.arange(
            case["table"].shape[1] * case["idx_page"], device=DEVICE
        )
        physical = _idx_slots(case, everything.expand(rows, -1))
        for r in range(rows):
            # A logical position of this row whose record is poisoned.
            spare = everything[torch.isin(physical[r], idx_unused[1:])][-1]
            logical[r, int(case["idx_len"][r]) :] = int(spare)
        kwargs.update(
            idx_pool=_poisoned_pool(idx, case["idx_page"]), idx_logical=logical
        )
    for kernel, out in _mla_run(case, **kwargs).items():
        assert torch.equal(out, clean[kernel]), f"{kernel} read an unselected record"


@pytest.mark.parametrize("heads", [16, 24])
def test_sparse_mla_reads_only_live_pages(heads, rows=6):
    """The TileLang metadata writes a row's indexed page table only over the pages of
    its visible entries (``tilelang.metadata.decode_metadata``), which hold every
    position the row selects. The sparse MLA looks up no other column for a lane it
    keeps: with every other column poisoned and each list pointing past its length
    into them, every route keeps the clean bits."""
    case = _mla_case(512, heads, rows)
    page, width = case["idx_page"], case["idx_logical"].shape[1]
    gen = _gen(13)
    visible = [1, 130, 700, 1500, 2600, 4096][:rows]
    logical = torch.full_like(case["idx_logical"], -1)
    lengths = torch.zeros_like(case["idx_len"])
    for r, seen in enumerate(visible):
        n = min(seen, width, 1 + r * 97)
        logical[r, :n] = torch.randperm(seen, generator=gen, device=DEVICE)[:n].to(
            torch.int32
        )
        lengths[r] = n
    case.update(idx_logical=logical, idx_len=lengths)
    clean = _mla_run(case)
    seen = torch.tensor(visible, dtype=torch.int32, device=DEVICE)
    case["table"] = _poison_past_live(case["table"], seen, page)
    # Lanes past each list's length look up the first poisoned column (a row whose
    # pages are all live keeps its -1 lanes).
    live = (seen.long() + page - 1) // page
    lanes = torch.arange(width, device=DEVICE)[None, :]
    past = lanes >= lengths[:, None]
    dead = (live[:, None] * page + lanes % page).to(torch.int32)
    poisoned = torch.where(
        past & (live[:, None] < case["table"].shape[1]), dead, logical
    )
    for kernel, out in _mla_run(case, idx_logical=poisoned).items():
        assert torch.equal(out, clean[kernel]), f"{kernel} looked up a dead page"


def test_sparse_mla_selected_nan_reaches_its_rows(heads=16, rows=6):
    """A NaN in a selected key makes every head of each row that selects it NaN, as
    in the reference's softmax, and leaves the other rows' bits alone."""
    case = _mla_case(0, heads, rows)
    clean = _mla_run(case)
    slot = int(case["swa_idx"][0, 0])
    swa = case["swa"].clone()
    swa[slot, 0] = 0x7F  # E4M3 NaN
    keep = (
        torch.arange(case["swa_idx"].shape[1], device=DEVICE) < case["swa_len"][:, None]
    )
    hit = ((case["swa_idx"] == slot) & keep).any(1).nonzero().flatten().tolist()
    assert 0 in hit
    ref = _mla_ref(case, swa)
    for kernel, out in _mla_run(case, swa_pool=_paged(swa, case["swa_page"])).items():
        assert_same_nonfinite(out, ref, f"{kernel}: ")
        assert_rows_nan(out, hit, f"{kernel}: ")
        assert_other_rows_equal(out, clean[kernel], hit, f"{kernel}: ")


# Heads per rank: TP4 (16), TP3 (24 of 72 padded).
@pytest.mark.parametrize("heads", [16, 24])
@pytest.mark.parametrize("keys", [1, 2, 3, 8])
def test_sparse_mla_routes_short_rows(heads, keys, rows=1024, swa_width=128, page=128):
    """A prompt's first rows attend to a few keys. On the SWA-only layer (one segment)
    extend's fold starts from the constant state; every route must round alike (FOLD)."""
    gen = _gen(keys)
    swa_pages = 24
    q = torch.randn((rows, heads, HEAD_DIM), generator=gen, device=DEVICE).bfloat16()
    sinks = torch.randn((heads,), generator=gen, device=DEVICE)
    swa = _pack_swa(
        torch.randn((swa_pages * page, HEAD_DIM), generator=gen, device=DEVICE)
    )
    swa_idx = torch.full((rows, swa_width), -1, dtype=torch.int32, device=DEVICE)
    swa_idx[:, :keys] = torch.randint(
        0, swa_pages * page, (rows, keys), generator=gen, device=DEVICE, dtype=torch.int32
    )
    swa_len = torch.full((rows,), keys, dtype=torch.int32, device=DEVICE)
    plan = SparseMlaV41Plan(
        heads, swa_width, 0, page, swa_page_size=page,
        max_rows=rows, decode_max_rows=rows, split_max_rows=rows,
    )  # fmt: skip
    pool = _paged(swa, page)
    positions = torch.randint(0, 4096, (rows,), generator=gen, device=DEVICE)
    table = rope_table()
    outs = {}
    for kernel in plan.kernels:
        out = torch.empty((rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
        plan.run(q, pool, swa_idx, swa_len, sinks, positions, table, out, kernel=kernel)
        outs[kernel] = out
    for kernel, out in outs.items():
        differ = int((out.view(torch.int16) != outs["extend"].view(torch.int16)).sum())
        assert differ == 0, f"{kernel} differs from extend in {differ} elements"


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _wo_input_kernel(heads: int, block_heads: int, threads: int):
    """``wo_input`` alone on ``[rows, heads, 512]`` BF16 rows, ``block_heads`` heads per
    CTA, compiled as the sparse MLA kernels are (fast math)."""
    rows, positions_max = T.dynamic("rows, positions_max")
    blocks = block_heads * HEAD_DIM // 32

    @T.prim_func
    def kernel(
        Src: T.Tensor([rows, heads, HEAD_DIM], T.bfloat16),
        Positions: T.Tensor([rows], T.int64),
        CosSin: T.Tensor([positions_max, ROPE_DIM], T.float32),
        Out: T.Tensor([rows, heads, HEAD_DIM], T.bfloat16),
    ):
        with T.Kernel(heads // block_heads, rows, threads=threads) as (hb, row):
            S = T.alloc_shared([block_heads, HEAD_DIM], T.bfloat16)
            X = T.alloc_fragment([blocks, 32], T.float32)
            amax = T.alloc_fragment([blocks], T.float32)
            inv = T.alloc_fragment([blocks], T.float32)
            scale = T.alloc_fragment([blocks], T.float32)
            h0 = hb * block_heads
            T.copy(Src[row, h0 : h0 + block_heads, :], S)
            wo_input(S, block_heads, Positions[row], CosSin, X, amax, inv, scale)
            T.copy(S, Out[row, h0 : h0 + block_heads, :])

    return kernel


# The extend kernel's head block (32 heads, 256 threads) and the decode merge's (one
# head, 128 threads).
@pytest.mark.parametrize(("block_heads", "threads"), [(32, 256), (1, 128)])
def test_wo_input_is_the_reference_rotation_and_act_quant(block_heads, threads):
    """The sparse MLA's output store gives WO-A's input bit for bit as the reference
    prepares it: TileKernels' inverse rotation in BF16, then per_token_cast's values
    (byte-identical to act_quant), blocks below the amax floor and a wide range of
    magnitudes included."""
    rows, heads = 37, 64
    gen = _gen(21)
    o = torch.randn((rows, heads, HEAD_DIM), generator=gen, device=DEVICE)
    exponents = torch.randint(
        -24, 8, (rows, heads, HEAD_DIM // 32, 1), generator=gen, device=DEVICE
    )
    o *= torch.exp2(exponents).repeat_interleave(32, -1).view(rows, heads, HEAD_DIM)
    o = o.bfloat16()
    o[::7, ::5] *= 1e-7  # blocks below the amax floor
    positions = torch.randint(0, 4096, (rows,), generator=gen, device=DEVICE)
    table = rope_table()
    out = torch.empty_like(o)
    _wo_input_kernel(heads, block_heads, threads)(o, positions, table, out)
    want = wo_input_reference(o, positions, table)
    differ = int((out.view(torch.int16) != want.view(torch.int16)).sum())
    assert differ == 0, f"{differ} of {out.numel()} values differ from the reference"
