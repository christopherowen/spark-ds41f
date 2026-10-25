# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang sparse-MLA metadata against torch references of its
contracts, exactly: every row a launch writes, and the rows it must leave alone."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from vllm.models.deepseek_v4_1.tilelang import metadata  # noqa: E402

DEVICE = "cuda"
R, CAP, COLS = 16, 1024, 40
BLOCK = 128  # tokens' rows a program covers


def _step(lengths, seed, padding=5):
    g = torch.Generator().manual_seed(seed)
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    nr, nt = len(lengths), starts[-1] + padding
    qsl = torch.full((R + 1,), starts[-1], dtype=torch.int32)
    qsl[: nr + 1] = torch.tensor(starts, dtype=torch.int32)
    seq = torch.zeros(R, dtype=torch.int32)
    seq[:nr] = torch.tensor([n + int(torch.randint(0, 900, (1,), generator=g))
                             for n in lengths], dtype=torch.int32)  # fmt: skip
    seq[1 % max(nr, 1)] = 0 if nr > 1 else seq[0]  # a negative first position
    table = torch.randint(0, 500, (R, COLS + 8), generator=g, dtype=torch.int32)
    table[:, 3] = 0  # holes
    slots = torch.randint(-1, 10**6, (CAP,), generator=g)
    tensors = dict(qsl=qsl, seq=seq, table=table[:, :COLS], slots=slots)
    return {k: v.to(DEVICE) for k, v in tensors.items()}, nr, nt


STEPS = [[1], [6] * 16, [3, 0, 5, 1, 2, 6], [700], [200, 1, 300]]


# --- Torch references of the contracts, in int64 on the host -----------------


def _host(*tensors):
    return [t.cpu().long() for t in tensors]


def _ref_requests(s, nr, nt):
    """Each of the R + 1 request starts (``nt`` past the last request), each
    request's first position (-1 past the last), and (tokens, requests)."""
    qsl, seq = _host(s["qsl"], s["seq"])
    starts = torch.where(torch.arange(R + 1) <= nr, qsl, nt)
    live = torch.arange(R) < nr
    ends = torch.where(live, qsl[1:], nt)
    positions = torch.where(live, seq - (ends - starts[:R]), -1)
    return starts, positions, torch.tensor([nt, nr])


def _ref_tokens(s, nr, nt, page, ratio, circular):
    """Each token row of the first ``max(1, ceil(nt / BLOCK))`` blocks: its position
    and request (-1 when it belongs to none, its position is negative or, outside
    a ring, the worker gave it no slot), the cache slot it writes (-1 when it
    emits nothing: the page is a hole or past the table, the ratio's group is not
    complete, or, in a ring, an earlier row of a long prefill would race later
    writers) and its visible length ((position + 1) / ratio, 0 when invalid). Rows
    past those blocks are left alone (None)."""
    qsl, seq, table, input_slots = _host(s["qsl"], s["seq"], s["table"], s["slots"])
    cover = min(CAP, max(1, -(-nt // BLOCK)) * BLOCK)
    t = torch.arange(cover)
    # The first request whose rows end past t: as many as end at or before it.
    req = torch.searchsorted(qsl[1 : nr + 1].contiguous(), t, right=True)
    valid = (t < nt) & (req < nr)
    r = req.clamp(max=R - 1)
    start, end = qsl[r], qsl[r + 1]
    pos = seq[r] - (end - start) + t - start
    valid &= pos >= 0
    if not circular:
        valid &= torch.where(t < nt, input_slots[t], -1) >= 0
    logical = torch.where(valid, pos, 0) // ratio
    column = torch.zeros_like(logical) if circular else logical // page
    valid &= column < table.shape[1]
    block = torch.where(valid, table[r, column.clamp(max=table.shape[1] - 1)], 0)
    valid &= block > 0
    emit = valid & ((torch.where(valid, pos, 0) + 1) % ratio == 0)
    if circular:
        emit &= t >= end - page
    return (
        torch.where(valid, pos, -1),
        torch.where(valid, req, -1),
        torch.where(emit, block * page + logical % page, -1),
        torch.where(valid, (pos + 1) // ratio, 0),
    )


def _ref_chunk(pos, reqs, table, visible, qsl, request_positions, offset, rows,
               width, page, window, draft, replay_start):  # fmt: skip
    """Rows ``offset:offset + rows``: each row's window of physical SWA slots (-1 in
    holes and past its length), its length (clamped to ``width``; 0 when its
    request or position is invalid) and its indexed length (visible, at most 512).
    A draft row's window is its request's: prefix plus the request's rows."""
    pos, reqs, table, visible, qsl, request_positions = _host(
        pos, reqs, table, visible, qsl, request_positions
    )
    row = torch.arange(rows) + offset
    req, p = reqs[row], pos[row]
    valid = (req >= 0) & (p >= 0)
    r = torch.where(valid, req, 0)
    if draft:
        first, end = qsl[r] * valid, qsl[r + 1] * valid
        prefix = request_positions[r] * valid
        start = (prefix - window).clamp(min=0)
        length = prefix + end - first - start
    else:
        start = (p + 1 - window).clamp(min=0)
        if replay_start is not None:
            start = torch.maximum(start, replay_start.cpu().long()[r] * valid)
        length = p + 1 - start
    length = torch.where(valid, length.clamp(0, width), 0)
    col = torch.arange(width)
    logical = start[:, None] + col
    live = valid[:, None] & (logical >= 0) & (col < length[:, None])
    column = torch.where(live, logical // page, 0)
    assert int(column.max()) < table.shape[1]  # no read past a table row
    page_id = torch.where(live, table[r[:, None], column], -1)
    swa = torch.where(live & (page_id > 0), page_id * page + logical % page, -1)
    top = torch.where(valid, visible[row].clamp(0, 512), 0)
    return swa, length, top


def _ref_pages(reqs, table, offset, rows, width, read_width):
    """Rows ``offset:offset + rows``: each row's request's page table, ``width``
    columns, -1 in holes (vLLM's block zero), past ``read_width`` and for rows
    without a request."""
    reqs, table = _host(reqs, table)
    req = reqs[torch.arange(rows) + offset]
    col = torch.arange(width)
    live = (req >= 0)[:, None] & (col < read_width)
    page = torch.where(
        live, table[req.clamp(min=0)[:, None], col.clamp(max=table.shape[1] - 1)], -1
    )
    return torch.where(page > 0, page, -1)


def _rows_equal(actual, expected, rows, sentinel=7):
    """``actual``'s first rows equal the reference, and the rest are untouched."""
    actual = actual.cpu().long()
    assert torch.equal(actual[:rows], expected)
    assert bool((actual[rows:] == sentinel).all())


@pytest.mark.parametrize("lengths", STEPS)
def test_requests(lengths):
    s, nr, nt = _step(lengths, seed=len(lengths))
    out_starts = torch.full((R + 1,), 7, dtype=torch.int32, device=DEVICE)
    pos = torch.full((R,), 7, dtype=torch.int64, device=DEVICE)
    counts = torch.full((2,), 7, dtype=torch.int32, device=DEVICE)
    metadata.requests(s["qsl"], s["seq"], out_starts, pos, counts, nr, nt)
    for actual, expected in zip((out_starts, pos, counts), _ref_requests(s, nr, nt)):
        assert torch.equal(actual.cpu().long(), expected)


TOKEN_CASES = [(64, 1, False), (64, 4, False), (32, 2, False), (8, 2, True)]


@pytest.mark.parametrize("lengths", STEPS)
@pytest.mark.parametrize("page, ratio, circular", TOKEN_CASES)
def test_tokens(lengths, page, ratio, circular):
    s, nr, nt = _step(lengths, seed=page + ratio + len(lengths))
    pos = torch.full((CAP,), 7, dtype=torch.int64, device=DEVICE)
    reqs = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
    slots = torch.full((CAP,), 7, dtype=torch.int64, device=DEVICE)
    visible = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
    metadata.tokens(
        s["qsl"], s["seq"], s["table"], s["slots"], pos, reqs, slots, visible,
        nr, nt, max_requests=R, page=page, ratio=ratio, circular=circular,
    )  # fmt: skip
    expected = _ref_tokens(s, nr, nt, page, ratio, circular)
    cover = expected[0].shape[0]
    for actual, reference in zip((pos, reqs, slots, visible), expected):
        _rows_equal(actual, reference, cover)


def _token_metadata(s, nr, nt, page):
    pos = torch.empty(CAP, dtype=torch.int64, device=DEVICE)
    reqs = torch.empty(CAP, dtype=torch.int32, device=DEVICE)
    slots = torch.empty(CAP, dtype=torch.int64, device=DEVICE)
    visible = torch.empty(CAP, dtype=torch.int32, device=DEVICE)
    metadata.tokens(
        s["qsl"], s["seq"], s["table"], s["slots"], pos, reqs, slots, visible, nr, nt,
        max_requests=R, page=page, ratio=1, circular=False,
    )  # fmt: skip
    return pos, reqs, visible


@pytest.mark.parametrize("lengths", STEPS)
@pytest.mark.parametrize(
    "draft, replay", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("width", [128, 136])
def test_chunk(lengths, draft, replay, width):
    page, window = 64, 128
    s, nr, nt = _step(lengths, seed=width + 3 * draft + replay)
    pos, reqs, visible = _token_metadata(s, nr, nt, page)
    request_positions = torch.randint(-5, 900, (R,), device=DEVICE)
    replay_start = torch.randint(0, 900, (R,), device=DEVICE) if replay else None
    rows, offset = min(nt, 600), 3 if nt > 10 else 0
    rows = min(rows, nt - offset)
    swa = torch.full((CAP, width), 7, dtype=torch.int32, device=DEVICE)
    lengths_out = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
    top = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
    metadata.chunk(
        pos, reqs, s["table"], swa, lengths_out, top, visible, s["qsl"],
        request_positions, offset, rows, page=page, window=window, draft=draft,
        replay_start=replay_start,
    )  # fmt: skip
    expected = _ref_chunk(
        pos, reqs, s["table"], visible, s["qsl"], request_positions, offset, rows,
        width, page, window, draft, replay_start,
    )  # fmt: skip
    for actual, reference in zip((swa, lengths_out, top), expected):
        _rows_equal(actual, reference, rows)


@pytest.mark.parametrize("lengths", STEPS)
@pytest.mark.parametrize("width, read_width", [(128, COLS), (256, 30), (40, 40)])
def test_pages(lengths, width, read_width):
    s, nr, nt = _step(lengths, seed=width + read_width)
    _, reqs, _ = _token_metadata(s, nr, nt, 64)
    rows, offset = min(nt, 500), 2 if nt > 4 else 0
    rows = min(rows, nt - offset)
    out = torch.full((CAP, width), 7, dtype=torch.int32, device=DEVICE)
    metadata.pages(reqs, s["table"], out, offset, rows, read_width)
    expected = _ref_pages(reqs, s["table"], offset, rows, width, read_width)
    _rows_equal(out, expected, rows)


@pytest.mark.parametrize("lengths", STEPS)
@pytest.mark.parametrize(
    "draft, replay", [(False, False), (False, True), (True, False)]
)
def test_decode_metadata_equals_chunk_and_pages(lengths, draft, replay):
    page, window, width, main_width = 64, 128, 128, 256
    s, nr, nt = _step(lengths, seed=7 + 3 * draft + replay)
    pos, reqs, visible = _token_metadata(s, nr, nt, page)
    main_reqs = torch.flip(reqs, (0,)).contiguous()
    request_positions = torch.randint(-5, 900, (R,), device=DEVICE)
    replay_start = torch.randint(0, 900, (R,), device=DEVICE) if replay else None
    rows, offset = min(nt, 600), 3 if nt > 10 else 0
    rows = min(rows, nt - offset)
    swa = torch.full((CAP, width), 7, dtype=torch.int32, device=DEVICE)
    lengths_out = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
    top = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
    main_pages = torch.full((CAP, main_width), 7, dtype=torch.int32, device=DEVICE)
    metadata.decode_metadata(
        pos, reqs, s["table"], swa, lengths_out, top, visible, s["qsl"],
        request_positions, main_reqs, s["table"], main_pages, offset, rows,
        s["table"].shape[1], page=page, window=window, draft=draft,
        replay_start=replay_start,
    )  # fmt: skip
    expected = _ref_chunk(
        pos, reqs, s["table"], visible, s["qsl"], request_positions, offset, rows,
        width, page, window, draft, replay_start,
    ) + (
        _ref_pages(
            main_reqs, s["table"], offset, rows, main_width, s["table"].shape[1]
        ),
    )  # fmt: skip
    for actual, reference in zip((swa, lengths_out, top, main_pages), expected):
        _rows_equal(actual, reference, rows)


@torch.inference_mode()
def test_metadata_refresh_preserves_padded_graph_domain_and_addresses():
    from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadataBuilder

    device = torch.device("cuda", torch.accelerator.current_device_index())
    builder = DeepseekV41B12xMetadataBuilder.__new__(DeepseekV41B12xMetadataBuilder)
    builder.tokens, builder.requests = 4096, 4
    builder.page, builder.ratio, builder.circular = 128, 1, False
    builder.reorder_batch_threshold = 6
    builder.starts = torch.empty(5, dtype=torch.int32, device=device)
    builder.request_positions = torch.empty(4, dtype=torch.int64, device=device)
    builder.counts = torch.empty(2, dtype=torch.int32, device=device)
    for name, dtype in (
        ("positions", torch.int64),
        ("reqs", torch.int32),
        ("slots", torch.int64),
        ("lengths", torch.int32),
    ):
        setattr(builder, name, torch.empty(builder.tokens, dtype=dtype, device=device))
    addresses = [
        getattr(builder, name).data_ptr()
        for name in ("positions", "reqs", "slots", "lengths")
    ]
    base_page = 2**31 // builder.page + 17
    table = torch.zeros((2, 16), dtype=torch.int32, device=device)
    table[0] = torch.arange(base_page, base_page + 16, dtype=torch.int32, device=device)
    for live, padded in ((257, 320), (5, 10), (129, 160), (0, 5)):
        seq = torch.tensor([1024, 0], dtype=torch.int32, device=device)
        slots = torch.full((padded,), -1, dtype=torch.int64, device=device)
        slots[:live] = 1
        common = SimpleNamespace(
            block_table_tensor=table,
            query_start_loc=torch.tensor(
                [0, live, live], dtype=torch.int32, device=device
            ),
            seq_lens=seq,
            slot_mapping=slots,
            num_reqs=2,
            num_actual_tokens=padded,
            max_query_len=live,
            max_seq_len=2048,
        )
        builder.build(0, common)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            metadata = builder.build(0, common)
        seq[0] += 7
        graph.replay()
        torch.accelerator.synchronize()
        expected = torch.arange(1031 - live, 1031, dtype=torch.int64, device=device)
        torch.testing.assert_close(metadata.positions[:live], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            metadata.slot_mapping[:live],
            base_page * builder.page + expected,
            rtol=0,
            atol=0,
        )
        assert bool((metadata.req_id_per_token[:live] == 0).all())
        for tensor in (
            metadata.positions,
            metadata.req_id_per_token,
            metadata.slot_mapping,
        ):
            assert bool((tensor[live:padded] == -1).all())
        assert bool((metadata.cache_lengths[live:padded] == 0).all())
        assert metadata.live_counts.tolist() == [padded, 2]
        assert addresses == [
            getattr(builder, name).data_ptr()
            for name in ("positions", "reqs", "slots", "lengths")
        ]
