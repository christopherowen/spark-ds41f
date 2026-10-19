# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang sparse-MLA metadata against the Triton kernels, exactly."""

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


@pytest.mark.parametrize("max_requests", [1, 8, 16, 32, 48, 64])
@pytest.mark.parametrize("nr", [0, 1, 3, 35, 64])
def test_ced_requests_preserves_compaction_and_replay_bounds(max_requests, nr):
    """Per-request limits preserve the old 128/full-row metadata, including padding."""
    from vllm.models.deepseek_v4_1.ced import _decoder_requests
    from vllm.models.deepseek_v4_1.tilelang import ced

    nr = min(nr, max_requests)
    for lengths in ([0] * nr, [1] * nr, [129 + i * 511 for i in range(nr)]):
        starts = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        seq = torch.full((max_requests,), 5000, dtype=torch.int32, device="cuda")
        prefix = torch.arange(max_requests, dtype=torch.int64, device="cuda") * 17
        full = torch.arange(max_requests, device="cuda") % 2 == 0
        limits = torch.where(full, 2**31 - 1, 128).to(torch.int32)
        outputs = []
        for implementation in ("triton", "tilelang"):
            out = torch.full((max_requests + 1,), 7, dtype=torch.int32, device="cuda")
            positions = torch.full((max_requests,), 7, dtype=torch.int64, device="cuda")
            replay = torch.full_like(positions, 7)
            counts = torch.full((2,), 7, dtype=torch.int32, device="cuda")
            if implementation == "triton":
                _decoder_requests[(1,)](
                    starts,
                    seq,
                    full,
                    prefix,
                    out,
                    positions,
                    replay,
                    counts,
                    nr,
                    max_requests,
                    triton.next_power_of_2(max_requests + 1),
                    128,
                )
            else:
                ced.requests(
                    starts, seq, limits, prefix, out, positions, replay, counts, nr
                )
            if implementation == "tilelang" and all(
                q <= 128 or r % 2 == 0 for r, q in enumerate(lengths)
            ):
                ced.requests(
                    starts,
                    seq,
                    limits,
                    prefix,
                    out,
                    positions,
                    replay,
                    counts,
                    nr,
                    compact=False,
                )
            outputs.append((out, positions, replay, counts))
        for reference, actual in zip(*outputs):
            assert torch.equal(reference, actual)


DEVICE = "cuda"
R, CAP, COLS = 16, 1024, 40


@pytest.mark.parametrize(
    "dtype", [torch.int64, torch.int32, torch.float32, torch.bfloat16, torch.bool]
)
@pytest.mark.parametrize("transpose", [False, True])
def test_ced_row_copy_preserves_strides_invalid_indices_and_scatter(dtype, transpose):
    from vllm.models.deepseek_v4_1.ced import gather_rows, scatter_rows
    from vllm.models.deepseek_v4_1.tilelang import ced

    source = torch.arange(7 * 3 * 5, device=DEVICE).reshape(7, 3, 5).to(dtype)
    if transpose:
        source = source.transpose(1, 2)
    indices = torch.tensor([6, 1, -1, 3, 99], device=DEVICE)
    expected = gather_rows(source, indices)
    output = torch.empty_like(expected)
    ced.copy_rows(source, indices, output, False)
    assert torch.equal(expected, output)
    restored = torch.zeros((7, *source.shape[1:]), dtype=dtype, device=DEVICE)
    ced.copy_rows(output, indices, restored, True)
    assert torch.equal(scatter_rows(output, indices, 7), restored)


@pytest.mark.parametrize("rows", [1, 48, 4048])
@pytest.mark.parametrize("tail", [(5120,), (4, 5120)])
def test_ced_row_copy_serving_widths(rows, tail):
    from vllm.models.deepseek_v4_1.ced import gather_rows, scatter_rows
    from vllm.models.deepseek_v4_1.tilelang import ced

    source = torch.randn((rows, *tail), dtype=torch.bfloat16, device=DEVICE)
    indices = torch.arange(max(0, rows - 128), rows, dtype=torch.int64, device=DEVICE)
    expected = gather_rows(source, indices)
    output = torch.empty_like(expected)
    ced.copy_rows(source, indices, output, False)
    assert torch.equal(expected, output)
    restored = torch.zeros_like(source)
    ced.copy_rows(output, indices, restored, True)
    assert torch.equal(scatter_rows(output, indices, rows), restored)


@pytest.mark.parametrize("lengths", [[], [0], [1], [129], [0, 257, 0, 1], [4096] * 8])
@pytest.mark.parametrize("padding", [0, 1, 127])
def test_ced_indices_preserves_last_rows_and_padding(lengths, padding):
    """Compacted rows map to each request's suffix; padded rows map to -1."""
    from vllm.models.deepseek_v4_1.ced import _decoder_indices
    from vllm.models.deepseek_v4_1.tilelang import ced

    nr = len(lengths)
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE
    )
    kept = [min(n, 128) for n in lengths]
    nt = sum(kept)
    compact = torch.full((17,), nt, dtype=torch.int32, device=DEVICE)
    compact[: nr + 1] = torch.tensor(
        [0, *torch.tensor(kept).cumsum(0).tolist()], dtype=torch.int32, device=DEVICE
    )
    capacity = max(1, nt + padding)
    reference = torch.full((capacity,), 7, dtype=torch.int64, device=DEVICE)
    actual = torch.full_like(reference, 7)
    _decoder_indices[(triton.cdiv(capacity, 128),)](
        starts, compact, reference, nr, nt, capacity, 128
    )
    ced.indices(starts, compact, actual, nr, nt)
    assert torch.equal(reference, actual)


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


@pytest.mark.parametrize("lengths", STEPS)
def test_requests(lengths):
    s, nr, nt = _step(lengths, seed=len(lengths))
    outs = []
    for impl in ("triton", "tilelang"):
        out_starts = torch.full((R + 1,), 7, dtype=torch.int32, device=DEVICE)
        pos = torch.full((R,), 7, dtype=torch.int64, device=DEVICE)
        counts = torch.full((2,), 7, dtype=torch.int32, device=DEVICE)
        if impl == "triton":
            _requests[(triton.cdiv(R + 1, 128),)](
                s["qsl"], s["seq"], out_starts, pos, counts, nr, nt, R, 128
            )
        else:
            metadata.requests(s["qsl"], s["seq"], out_starts, pos, counts, nr, nt)
        outs.append((out_starts, pos, counts))
    for a, b in zip(*outs):
        assert torch.equal(a, b)


TOKEN_CASES = [(64, 1, False), (64, 4, False), (32, 2, False), (8, 2, True)]


@pytest.mark.parametrize("lengths", STEPS)
@pytest.mark.parametrize("page, ratio, circular", TOKEN_CASES)
def test_tokens(lengths, page, ratio, circular):
    s, nr, nt = _step(lengths, seed=page + ratio + len(lengths))
    outs = []
    for impl in ("triton", "tilelang"):
        pos = torch.full((CAP,), 7, dtype=torch.int64, device=DEVICE)
        reqs = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
        slots = torch.full((CAP,), 7, dtype=torch.int64, device=DEVICE)
        visible = torch.full((CAP,), 7, dtype=torch.int32, device=DEVICE)
        if impl == "triton":
            _tokens[(max(1, triton.cdiv(nt, 128)),)](
                s["qsl"], s["seq"], s["table"], s["slots"], pos, reqs, slots, visible,
                nr, nt, s["table"].stride(0), s["table"].shape[1], CAP, page, ratio,
                128, circular,
            )  # fmt: skip
        else:
            metadata.tokens(
                s["qsl"], s["seq"], s["table"], s["slots"], pos, reqs, slots, visible,
                nr, nt, max_requests=R, page=page, ratio=ratio, circular=circular,
            )  # fmt: skip
        outs.append((pos, reqs, slots, visible))
    for a, b in zip(*outs):
        assert torch.equal(a, b)


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


@pytest.mark.parametrize("lengths", STEPS)
@pytest.mark.parametrize("width, read_width", [(128, COLS), (256, 30), (40, 40)])
def test_pages(lengths, width, read_width):
    s, nr, nt = _step(lengths, seed=width + read_width)
    _, reqs, _ = _token_metadata(s, nr, nt, 64)
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
            metadata.pages(reqs, s["table"], out, offset, rows, read_width)
        outs.append(out)
    assert torch.equal(outs[0], outs[1])


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
                replay_start=replay_start,
            )  # fmt: skip
        outs.append((swa, lengths_out, top, main_pages))
    for a, b in zip(*outs):
        assert torch.equal(a, b)
