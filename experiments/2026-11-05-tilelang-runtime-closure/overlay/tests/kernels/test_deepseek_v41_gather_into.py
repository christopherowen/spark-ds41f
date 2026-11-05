# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-major gather to final buffers, against an independent concatenation."""

import pytest
import torch

pytest.importorskip("tilelang")
from vllm.models.deepseek_v4_1.tilelang.gather import gather_columns_into

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
CASES = [
    (1, 1, 7, 17, 17),
    (2, 1, 7, 16, 29),
    (3, 1, 9, 32, 91),
    (4, 2, 7, 128, 512),
    (8, 2, 17, 64, 512),
    (4, 2, 0, 128, 512),
    (3, 1, 65537, 16, 48),
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["dense", "padded", "offset"])
def test_gather_bits_and_output_bounds(case, dtype, layout):
    world, parts, rows, local, full = case
    src = torch.randn((world * parts * rows, local), dtype=dtype, device="cuda")
    reference = src.view(world, parts, rows, local)
    expected = [
        torch.cat([reference[r, p] for r in range(world)], dim=1)[:, :full]
        for p in range(parts)
    ]
    padding = (
        16 + (-full) % (16 // src.element_size())
        if layout == "padded"
        else 3
        if layout == "offset"
        else 0
    )
    offset = 1 if layout == "offset" else 0
    storage = [
        torch.full((rows, full + padding), -123, dtype=dtype, device="cuda")
        for _ in range(parts)
    ]
    outputs = tuple(t[:, offset : offset + full] for t in storage)
    gather_columns_into(src, outputs, world)
    bit_dtype = torch.int16 if dtype == torch.bfloat16 else torch.int32
    for backing, out, want in zip(storage, outputs, expected):
        assert torch.equal(out.view(bit_dtype), want.view(bit_dtype))
        assert (backing[:, :offset] == -123).all()
        assert (backing[:, offset + full :] == -123).all()


def test_floating_point_payloads_survive_the_copy():
    src = torch.randn((12, 16), device="cuda")
    src.view(torch.int32)[0, :4] = torch.tensor(
        [0, -2147483648, 2143289345, 2139095040], dtype=torch.int32, device="cuda"
    )
    out = torch.empty((3, 64), device="cuda")
    gather_columns_into(src, (out,), 4)
    expected = torch.cat(list(src.view(4, 3, 16)), dim=1)
    assert torch.equal(out.view(torch.int32), expected.view(torch.int32))


def test_graph_replay_reads_new_payloads():
    src = torch.randn((4 * 2 * 7, 128), device="cuda")
    outputs = tuple(torch.empty((7, 512), device="cuda") for _ in range(2))
    gather_columns_into(src, outputs, 4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather_columns_into(src, outputs, 4)
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        src.normal_()
        graph.replay()
        torch.cuda.synchronize()
        for part, out in enumerate(outputs):
            expected = torch.cat(
                [src.view(4, 2, 7, 128)[r, part] for r in range(4)], dim=1
            )
            assert torch.equal(out, expected)


def test_bad_layouts_fail_explicitly():
    src = torch.empty((8, 16), device="cuda")
    with pytest.raises(ValueError, match="overlap"):
        gather_columns_into(src, (src,), 1)
    with pytest.raises(ValueError, match="source shape"):
        gather_columns_into(src, (torch.empty((3, 32), device="cuda"),), 2)
    with pytest.raises(ValueError, match="contiguous"):
        gather_columns_into(src.t(), (torch.empty((8, 32), device="cuda"),), 2)


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.bfloat16, torch.float32, torch.int64, torch.complex64]
)
@pytest.mark.parametrize("width", [1, 5, 17])
def test_collective_tail_rows_use_tilelang_and_preserve_bytes(dtype, width):
    from vllm.models.deepseek_v4_1.tilelang.collectives import concat_rank_shards

    src = torch.arange(3 * 7 * width, device="cuda").to(dtype).view(21, width)
    expected = torch.cat(list(src.view(3, 7, width)), dim=1)
    got = concat_rank_shards(src, 3, (7, width))
    assert got is not None
    assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))


def test_overlapping_output_rows_are_rejected_before_launch():
    src = torch.empty((8, 16), device="cuda")
    output = torch.empty((1, 32), device="cuda").expand(4, 32)
    with pytest.raises(ValueError, match="rows must not overlap"):
        gather_columns_into(src, (output,), 2)
