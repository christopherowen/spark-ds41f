# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang CED row selection (requests, row map, row copy) against the
Triton kernels it replaced, exactly."""

import math

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from vllm.triton_utils import triton  # noqa: E402

DEVICE = "cuda"


def _requests_reference(starts, seq, limits, floors, nr, max_requests, compact):
    """The requests kernel's outputs from the CPU: each request keeps its last
    ``min(rows, limit)`` rows; ``positions`` holds the first kept row's position and
    the replay starts at ``min(floor, position)``. Padding requests keep nothing."""
    starts, seq = starts.cpu().long(), seq.cpu().long()
    limits, floors = limits.cpu().long(), floors.cpu().long()
    rows = starts[1 : nr + 1] - starts[:nr]
    kept = torch.minimum(rows, limits[:nr]) if compact else rows
    out = torch.zeros(max_requests + 1, dtype=torch.long)
    out[1 : nr + 1] = kept.cumsum(0)
    out[nr + 1 :] = out[nr]
    positions = torch.full((max_requests,), -1, dtype=torch.long)
    positions[:nr] = seq[:nr] - kept
    replay = torch.zeros(max_requests, dtype=torch.long)
    replay[:nr] = torch.minimum(floors[:nr], positions[:nr])
    counts = torch.tensor([int(out[nr]), nr])
    return out.int(), positions, replay, counts.int()


@pytest.mark.parametrize("max_requests", [1, 8, 16, 32, 48, 64])
@pytest.mark.parametrize("nr", [0, 1, 3, 35, 64])
def test_ced_requests_replay_from_the_floor(max_requests, nr):
    """Kept rows, positions and replay starts for limits of one row, part of the
    window, the window and every row, with floors before, at and after the first
    kept row; the unchanged-row kernel where every request keeps its rows."""
    from vllm.models.deepseek_v4_1.tilelang import ced

    nr = min(nr, max_requests)
    choices = (1, 37, 128, 2**31 - 1)
    for lengths in (
        [0] * nr,
        [1] * nr,
        [3 + i * 7 for i in range(nr)],
        [129 + i * 511 for i in range(nr)],
    ):
        starts = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        seq = torch.arange(max_requests, dtype=torch.int32, device="cuda") * 701
        seq += 40000
        limits = torch.tensor(
            [choices[r % 4] for r in range(max_requests)],
            dtype=torch.int32,
            device="cuda",
        )
        # Floors past every row, at the window start and before the chunk.
        floors = seq.long() - torch.tensor(
            [(0, 128, 90000)[r % 3] for r in range(max_requests)], device="cuda"
        ).clamp_max(seq.long())
        for compact in (True, False):
            if not compact and any(
                min(q, int(limits[r])) < q for r, q in enumerate(lengths)
            ):
                continue
            out = torch.full((max_requests + 1,), 7, dtype=torch.int32, device="cuda")
            positions = torch.full((max_requests,), 7, dtype=torch.int64, device="cuda")
            replay = torch.full_like(positions, 7)
            counts = torch.full((2,), 7, dtype=torch.int32, device="cuda")
            ced.requests(
                starts, seq, limits, floors, out, positions, replay, counts, nr,
                compact=compact,
            )  # fmt: skip
            expected = _requests_reference(
                starts, seq, limits, floors, nr, max_requests, compact
            )
            if not compact:
                # The unchanged-row kernel passes the step's starts through.
                passed = torch.arange(max_requests + 1).clamp_max(nr)
                expected = (starts.cpu()[passed], *expected[1:])
            for want, got in zip(expected, (out, positions, replay, counts)):
                assert torch.equal(want, got.cpu()), (compact, lengths)


@pytest.mark.parametrize("max_requests", [1, 8, 16, 32, 48, 64])
@pytest.mark.parametrize("nr", [0, 1, 3, 35, 64])
def test_ced_requests_final_chunks_keep_the_released_metadata(max_requests, nr):
    """Final prompt chunks and decode rows, whose per-chunk anchor (the frozen r6x
    kernel's) was the prompt end already: the floors ``replay_rows`` gives them
    reproduce that kernel's metadata, including padding."""
    from benchmarks.kernels.ced_reference import _decoder_requests
    from vllm.models.deepseek_v4_1.tilelang import ced

    nr = min(nr, max_requests)
    for lengths in ([0] * nr, [1] * nr, [129 + i * 511 for i in range(nr)]):
        starts = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        # Every chunk ends its sequence's prompt or decode rows: no longer than it.
        seq = torch.full((max_requests,), 40000, dtype=torch.int32, device="cuda")
        prefix = torch.arange(max_requests, dtype=torch.int64, device="cuda") * 17
        full = torch.arange(max_requests, device="cuda") % 2 == 0
        limits = torch.where(full, 2**31 - 1, 128).to(torch.int32)
        rows = torch.tensor(lengths + [0] * (max_requests - nr), device="cuda")
        # A final chunk of more than the window replays from the prompt end less
        # the window; decode rows, short prompts and prompt logprobs from the prefix.
        floors = torch.where(
            ~full & (rows > 128), torch.maximum(prefix, seq.long() - 128), prefix
        )
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
                    starts, seq, limits, floors, out, positions, replay, counts, nr
                )
            if implementation == "tilelang" and all(
                q <= 128 or r % 2 == 0 for r, q in enumerate(lengths)
            ):
                ced.requests(
                    starts,
                    seq,
                    limits,
                    floors,
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


def _triton_copy(source, indices, rows, scatter):
    """Triton's row copy, which the TileLang copy replaced (the benchmark's copy)."""
    from benchmarks.kernels.ced_reference import _copy_rows

    tail = tuple(source.shape[1:])
    width = math.prod(tail)
    output = source.new_zeros((rows, *tail))
    _copy_rows[(triton.cdiv((source.shape[0] if scatter else rows) * width, 512),)](
        source,
        indices,
        output,
        source.shape[0],
        rows,
        tail,
        source.stride(),
        width,
        scatter,
        512,
    )
    return output


def gather_rows(source, indices):
    return _triton_copy(source, indices, indices.numel(), False)


def scatter_rows(source, indices, rows):
    return _triton_copy(source, indices, rows, True)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.int64,
        torch.int32,
        torch.float32,
        torch.bfloat16,
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.float16,
        torch.float64,
    ],
)
@pytest.mark.parametrize("transpose", [False, True])
def test_ced_row_copy_preserves_strides_invalid_indices_and_scatter(dtype, transpose):
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


@pytest.mark.parametrize(
    "dtype, tail",
    [
        (torch.int32, ()),
        (torch.int64, ()),
        (torch.bool, ()),
        (torch.float32, (4,)),
        (torch.float32, (4, 4)),
        (torch.bfloat16, (5120,)),
        (torch.bfloat16, (4, 5120)),
    ],
)
@pytest.mark.parametrize("rows", [1, 129, 600])
def test_ced_row_copy_serving_layouts_skip_invalid_rows(dtype, tail, rows):
    """The serving layouts with the plan's -1 padding and out-of-range indices,
    across a block boundary, against the Triton copies."""
    from vllm.models.deepseek_v4_1.tilelang import ced

    source = torch.randint(-100, 100, (rows, *tail), device=DEVICE).to(dtype)
    kept = min(rows, 128)
    indices = torch.cat(
        [
            torch.arange(rows - kept, rows, device=DEVICE),
            torch.tensor([-1, rows, -5, 1 << 40], device=DEVICE),
            torch.full((131,), -1, device=DEVICE),
        ]
    )
    expected = gather_rows(source, indices)
    output = torch.full_like(expected, 7)
    ced.copy_rows(source, indices, output, False)
    assert torch.equal(expected, output)
    restored = torch.full_like(source, 3)
    ced.copy_rows(output, indices, restored, True)
    reference = torch.full_like(source, 3)
    reference[indices[:kept]] = output[:kept]
    assert torch.equal(reference, restored)
    assert torch.equal(
        scatter_rows(output, indices, rows)[indices[:kept]], output[:kept]
    )


@pytest.mark.parametrize(
    "dtype, width, columns",
    [
        (torch.bfloat16, 5184, slice(64, 5184)),
        (torch.float32, 20, slice(4, 20)),
        (torch.int32, 3, 1),
        (torch.bool, 5, 2),
    ],
)
def test_ced_row_copy_views_of_wider_rows(dtype, width, columns):
    """Column views of wider rows: the row stride is an argument, and the copies
    match Triton's."""
    from vllm.models.deepseek_v4_1.tilelang import ced

    wide = torch.randint(-100, 100, (300, width), device=DEVICE).to(dtype)
    source = wide[:, columns]
    if source.ndim == 2 and source.shape[1] == 16:
        source = source.unflatten(1, (4, 4))
    indices = torch.cat(
        [torch.arange(172, 300, device=DEVICE), torch.full((5,), -1, device=DEVICE)]
    )
    expected = gather_rows(source, indices)
    output = torch.full_like(expected, 7)
    ced.copy_rows(source, indices, output, False)
    assert torch.equal(expected, output)
    restored = torch.zeros((300, *source.shape[1:]), dtype=dtype, device=DEVICE)
    ced.copy_rows(output, indices, restored, True)
    assert torch.equal(scatter_rows(output, indices, 300), restored)


def test_ced_row_copy_packed_key_matches_contiguous_sources():
    from vllm.models.deepseek_v4_1.tilelang import ced

    for tail in [(), (4,), (4, 4), (4, 5120)]:
        source = torch.empty((3, *tail), device=DEVICE)
        assert ced.packed_copy_rows_key("float32", tail, True) == ced.copy_rows_key(
            "float32", tail, source.stride()[1:], source.stride(0), True
        )


def test_ced_row_copy_preconditions():
    from vllm.models.deepseek_v4_1.tilelang import ced

    indices = torch.zeros(2, dtype=torch.int64, device=DEVICE)
    rows = torch.zeros((2, 5120), dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(ValueError, match="a row to copy"):
        ced.copy_rows(rows, indices[:0], rows[:0], False)
    with pytest.raises(ValueError, match="one index"):
        ced.copy_rows(rows, indices[:1], rows, False)
    storage = torch.zeros(2 * 5120 + 1, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(ValueError, match="aligned"):
        ced.copy_rows(storage[1:].view(2, 5120), indices, rows, False)
    with pytest.raises(ValueError, match="aligned"):
        ced.copy_rows(rows, indices, storage[1:].view(2, 5120), False)
    # Element copies have no alignment requirement, and an empty source gathers
    # zeros.
    flags = torch.ones(9, dtype=torch.bool, device=DEVICE)
    ced.copy_rows(flags[1:3], indices, flags[5:7], False)
    assert flags.all()
    for dtype in (torch.bool, torch.int32, torch.bfloat16):
        output = torch.ones(2, dtype=dtype, device=DEVICE)
        ced.copy_rows(output[:0], indices, output, False)
        assert not output.any()


@pytest.mark.parametrize("lengths", [[], [0], [1], [129], [0, 257, 0, 1], [4096] * 8])
@pytest.mark.parametrize("padding", [0, 1, 127])
def test_ced_indices_preserves_last_rows_and_padding(lengths, padding):
    """Compacted rows map to each request's suffix; padded rows map to -1."""
    from benchmarks.kernels.ced_reference import _decoder_indices
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


@pytest.mark.parametrize("rows", [1, 22, 158, 4096])
def test_ced_token_metadata_matches_the_torch_chain(rows):
    """The decoder's per-token metadata in one launch: the gathered fields with
    -1 at negative indices (0 lengths) and 0 past the step's rows, as
    ``gather_rows`` then ``masked_fill`` wrote them."""
    from vllm.models.deepseek_v4_1.ced import gather_rows
    from vllm.models.deepseek_v4_1.tilelang import ced

    g = torch.Generator(device="cuda").manual_seed(rows)
    tokens = 4200
    positions = torch.randint(0, 1 << 20, (tokens,), generator=g, device="cuda")
    reqs = torch.randint(0, 16, (tokens,), generator=g, device="cuda").int()
    slots = torch.randint(-1, 1 << 24, (tokens,), generator=g, device="cuda")
    lengths = torch.randint(0, 1 << 16, (tokens,), generator=g, device="cuda").int()
    indices = torch.randint(-1, tokens + 3, (rows,), generator=g, device="cuda")
    valid = indices >= 0
    expected = [
        gather_rows(field, indices).masked_fill(~valid, -1)
        for field in (positions, reqs, slots)
    ] + [gather_rows(lengths, indices)]
    actual = ced.token_metadata(indices, positions, reqs, slots, lengths)
    for a, e in zip(actual, expected):
        assert a.dtype == e.dtype and torch.equal(a, e)


@pytest.mark.parametrize("rows", [1, 22, 158, 4096])
def test_ced_slots_and_routing_mask_match_the_torch_chain(rows):
    """The drafter's packed context slots (-1 at a negative index, 0 past the
    step's rows) and the decoder's routing mask (a valid index whose row is not
    padding; a row past the step's rows routes), each in one launch."""
    from vllm.models.deepseek_v4_1.ced import gather_rows
    from vllm.models.deepseek_v4_1.tilelang import ced
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    g = torch.Generator(device="cuda").manual_seed(rows + 7)
    tokens = 4200
    slots = torch.randint(-1, 1 << 24, (2, tokens), generator=g, device="cuda")
    is_padding = torch.rand(tokens, generator=g, device="cuda") < 0.3
    indices = torch.randint(-1, tokens + 3, (rows,), generator=g, device="cuda")
    for group in slots:
        expected = gather_rows(group, indices).masked_fill(indices < 0, PAD_SLOT_ID)
        actual = ced.gather_slots(group, indices)
        assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    expected = (indices >= 0) & ~gather_rows(is_padding, indices)
    actual = ced.routing_mask(indices, is_padding)
    assert actual.dtype == torch.bool and torch.equal(actual, expected)
