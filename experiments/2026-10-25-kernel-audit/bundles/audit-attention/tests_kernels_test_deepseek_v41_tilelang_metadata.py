# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang sparse-MLA metadata against the Triton kernels, exactly.

The page tables are Triton's over each row's live pages (those of its visible
entries); their other columns keep what the buffer held."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from vllm.models.deepseek_v4_1.attention import _chunk, _pages  # noqa: E402
from vllm.models.deepseek_v4_1.sparse_mla import _requests, _tokens  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import metadata  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402

DEVICE = "cuda"
R, CAP, COLS = 16, 1024, 40


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


TOKEN_CASES = [(64, 1, False), (64, 4, False), (32, 2, False), (8, 2, True)]


@pytest.mark.parametrize("lengths", STEPS)
@pytest.mark.parametrize("page, ratio, circular", TOKEN_CASES)
def test_builder(lengths, page, ratio, circular):
    """One launch writes ``_requests``' and ``_tokens``' outputs exactly."""
    s, nr, nt = _step(lengths, seed=page + ratio + len(lengths))
    outs = []
    for impl in ("triton", "tilelang"):
        out_starts = torch.full((R + 1,), 7, dtype=torch.int32, device=DEVICE)
        request_pos = torch.full((R,), 7, dtype=torch.int64, device=DEVICE)
        counts = torch.full((2,), 7, dtype=torch.int32, device=DEVICE)
        pos = torch.full((CAP,), 7, dtype=torch.int64, device=DEVICE)
        reqs = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
        slots = torch.full((CAP,), 7, dtype=torch.int64, device=DEVICE)
        visible = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
        if impl == "triton":
            _requests[(triton.cdiv(R + 1, 128),)](
                s["qsl"], s["seq"], out_starts, request_pos, counts, nr, nt, R, 128
            )
            _tokens[(max(1, triton.cdiv(nt, 128)),)](
                s["qsl"], s["seq"], s["table"], s["slots"], pos, reqs, slots, visible,
                nr, nt, s["table"].stride(0), s["table"].shape[1], CAP, page, ratio,
                128, circular,
            )  # fmt: skip
        else:
            metadata.builder(
                s["qsl"], s["seq"], s["table"], s["slots"], out_starts, request_pos,
                counts, pos, reqs, slots, visible, nr, nt, page=page, ratio=ratio,
                circular=circular,
            )  # fmt: skip
        outs.append((out_starts, request_pos, counts, pos, reqs, slots, visible))
    for a, b in zip(*outs):
        assert torch.equal(a, b)


def _token_metadata(s, nr, nt, page):
    out_starts = torch.empty(R + 1, dtype=torch.int32, device=DEVICE)
    request_pos = torch.empty(R, dtype=torch.int64, device=DEVICE)
    counts = torch.empty(2, dtype=torch.int32, device=DEVICE)
    pos = torch.empty(CAP, dtype=torch.int64, device=DEVICE)
    reqs = torch.empty(CAP, dtype=torch.int32, device=DEVICE)
    slots = torch.empty(CAP, dtype=torch.int64, device=DEVICE)
    visible = torch.empty(CAP, dtype=torch.int32, device=DEVICE)
    metadata.builder(
        s["qsl"], s["seq"], s["table"], s["slots"], out_starts, request_pos, counts,
        pos, reqs, slots, visible, nr, nt, page=page, ratio=1, circular=False,
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
    outs = []
    for impl in ("triton", "tilelang"):
        swa = torch.full((CAP, width), 7, dtype=torch.int32, device=DEVICE)
        lengths_out = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
        top = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
        if impl == "triton":
            _chunk[(rows,)](
                pos, reqs, s["table"], swa, lengths_out, top, visible, s["qsl"],
                request_positions, offset, s["table"].stride(0), page, window, width,
                draft, triton.next_power_of_2(width), swa_replay_start=replay_start,
            )  # fmt: skip
        else:
            metadata.chunk(
                pos, reqs, s["table"], swa, lengths_out, top, visible, s["qsl"],
                request_positions, offset, rows, page=page, window=window, draft=draft,
                replay_start=replay_start,
            )  # fmt: skip
        outs.append((swa, lengths_out, top))
    for a, b in zip(*outs):
        assert torch.equal(a, b)


def _live_columns(reqs, visible, offset, rows, page, read_width, width):
    """Each of ``rows`` rows' live page-table columns: the pages of its visible
    entries, none for a row without a request."""
    seen = visible[offset : offset + rows].long().clamp_min(0)
    live = (seen + page - 1) // page
    live = live.clamp_max(min(read_width, width))
    return torch.where(reqs[offset : offset + rows] >= 0, live, 0)


def _assert_live_equal(triton_out, tilelang_out, live, rows, fill=7):
    """Triton's columns where a row is live; the buffer's ``fill`` elsewhere (and in
    rows past ``rows``)."""
    cols = torch.arange(triton_out.shape[1], device=DEVICE)
    inside = torch.zeros_like(triton_out, dtype=torch.bool)
    inside[:rows] = cols[None, :] < live[:, None]
    assert torch.equal(tilelang_out[inside], triton_out[inside])
    assert torch.equal(
        tilelang_out[~inside], torch.full_like(tilelang_out[~inside], fill)
    )


@pytest.mark.parametrize("lengths", STEPS)
@pytest.mark.parametrize("width, read_width", [(128, COLS), (256, 30), (40, 40)])
@pytest.mark.parametrize("page", [64, 128])
def test_pages(lengths, width, read_width, page):
    s, nr, nt = _step(lengths, seed=width + read_width)
    _, reqs, visible = _token_metadata(s, nr, nt, 64)
    rows, offset = min(nt, 500), 2 if nt > 4 else 0
    rows = min(rows, nt - offset)
    outs = []
    for impl in ("triton", "tilelang"):
        out = torch.full((CAP, width), 7, dtype=torch.int32, device=DEVICE)
        if impl == "triton":
            _pages[(rows, triton.cdiv(width, 128))](
                reqs, s["table"], out, offset, s["table"].stride(0), read_width,
                width, 128,
            )  # fmt: skip
        else:
            metadata.pages(
                reqs, visible, s["table"], out, offset, rows, read_width, page=page
            )
        outs.append(out)
    live = _live_columns(reqs, visible, offset, rows, page, read_width, width)
    _assert_live_equal(*outs, live, rows)


@pytest.mark.parametrize("lengths", STEPS)
@pytest.mark.parametrize(
    "draft, replay", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("main_page", [64, 128])
def test_decode_metadata_equals_chunk_and_pages(lengths, draft, replay, main_page):
    page, window, width, main_width = 64, 128, 128, 256
    s, nr, nt = _step(lengths, seed=7 + 3 * draft + replay)
    pos, reqs, visible = _token_metadata(s, nr, nt, page)
    # The main rows' requests differ from the window rows' (the step's rows in
    # reverse); every row has a written request or -1, as a builder leaves them.
    main_reqs = torch.full_like(reqs, -1)
    main_reqs[:nt] = torch.flip(reqs[:nt], (0,))
    request_positions = torch.randint(-5, 900, (R,), device=DEVICE)
    replay_start = torch.randint(0, 900, (R,), device=DEVICE) if replay else None
    rows, offset = min(nt, 600), 3 if nt > 10 else 0
    rows = min(rows, nt - offset)
    outs = []
    for impl in ("triton", "tilelang"):
        swa = torch.full((CAP, width), 7, dtype=torch.int32, device=DEVICE)
        lengths_out = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
        top = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
        main_pages = torch.full((CAP, main_width), 7, dtype=torch.int32, device=DEVICE)
        if impl == "triton":
            _chunk[(rows,)](
                pos, reqs, s["table"], swa, lengths_out, top, visible, s["qsl"],
                request_positions, offset, s["table"].stride(0), page, window, width,
                draft, triton.next_power_of_2(width), swa_replay_start=replay_start,
            )  # fmt: skip
            _pages[(rows, triton.cdiv(main_width, 128))](
                main_reqs, s["table"], main_pages, offset, s["table"].stride(0),
                s["table"].shape[1], main_width, 128,
            )  # fmt: skip
        else:
            metadata.decode_metadata(
                pos, reqs, s["table"], swa, lengths_out, top, visible, s["qsl"],
                request_positions, main_reqs, s["table"], main_pages, offset, rows,
                s["table"].shape[1], page=page, window=window, draft=draft,
                main_page=main_page, replay_start=replay_start,
            )  # fmt: skip
        outs.append((swa, lengths_out, top, main_pages))
    for a, b in zip(outs[0][:3], outs[1][:3]):
        assert torch.equal(a, b)
    live = _live_columns(
        main_reqs, visible, offset, rows, main_page, s["table"].shape[1], main_width
    )
    _assert_live_equal(outs[0][3], outs[1][3], live, rows)


@pytest.mark.parametrize(
    "main_width, read_width", [(1100, 1100), (4096, 4096), (4096, 2500)]
)
@pytest.mark.parametrize("main_page", [64, 256])
def test_decode_metadata_wide_tables(main_width, read_width, main_page):
    """Rows whose live pages span several of the page part's column groups, up to the
    4,096 columns of a 1M context's table (the other tests' tables fit in one)."""
    page, window, width, rows = 64, 128, 128, 64
    g = torch.Generator().manual_seed(main_width + read_width + main_page)
    reqs = torch.randint(-1, R, (rows,), generator=g, dtype=torch.int32)
    # Visible entries from none to past the table's width: full, partial and empty
    # groups, and rows without a request.
    visible = torch.randint(
        -3, (read_width + 2) * main_page, (rows,), generator=g, dtype=torch.int32
    )
    pos = torch.randint(0, 2000, (rows,), generator=g)
    table = torch.randint(
        0, 1 << 20, (R, read_width + 8), generator=g, dtype=torch.int32
    )
    table[:, 3] = 0  # holes
    qsl = torch.arange(R + 1, dtype=torch.int32)
    request_positions = torch.randint(-5, 900, (R,), generator=g)
    reqs, visible, pos, qsl, request_positions = (
        t.to(DEVICE) for t in (reqs, visible, pos, qsl, request_positions)
    )
    table = table.to(DEVICE)[:, :read_width]
    outs = []
    for impl in ("triton", "tilelang"):
        swa = torch.full((rows, width), 7, dtype=torch.int32, device=DEVICE)
        lengths_out = torch.full((rows,), 7, dtype=torch.int32, device=DEVICE)
        top = torch.full((rows,), 7, dtype=torch.int32, device=DEVICE)
        main_pages = torch.full((rows, main_width), 7, dtype=torch.int32, device=DEVICE)
        if impl == "triton":
            _chunk[(rows,)](
                pos, reqs, table, swa, lengths_out, top, visible, qsl,
                request_positions, 0, table.stride(0), page, window, width, False,
                triton.next_power_of_2(width),
            )  # fmt: skip
            _pages[(rows, triton.cdiv(main_width, 128))](
                reqs, table, main_pages, 0, table.stride(0), read_width, main_width,
                128,
            )  # fmt: skip
        else:
            metadata.decode_metadata(
                pos, reqs, table, swa, lengths_out, top, visible, qsl,
                request_positions, reqs, table, main_pages, 0, rows, read_width,
                page=page, window=window, draft=False, main_page=main_page,
            )  # fmt: skip
        outs.append((swa, lengths_out, top, main_pages))
    for a, b in zip(outs[0][:3], outs[1][:3]):
        assert torch.equal(a, b)
    live = _live_columns(reqs, visible, 0, rows, main_page, read_width, main_width)
    assert int(live.max()) > 4 * 128, "no row spans more than one column group"
    _assert_live_equal(outs[0][3], outs[1][3], live, rows)
