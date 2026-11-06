# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-only boundary probes; no model source is replaced by this module."""

import pytest
import torch
from vllm.models.deepseek_v4_1.tilelang import dspark_context
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

ROWS = [
    1,
    3,
    5,
    7,
    11,
    15,
    16,
    17,
    18,
    24,
    30,
    31,
    32,
    33,
    34,
    36,
    42,
    47,
    48,
    51,
    58,
    59,
    63,
    64,
    65,
    68,
    84,
    85,
    95,
    96,
    97,
    102,
    106,
    107,
    127,
    128,
]


@pytest.mark.parametrize("rows", ROWS)
def test_context_inputs_replay(rows):
    """Real input kernel, fresh unwritten source tails, changed-input replay."""
    width, allocated = 5120, 132
    gen = torch.Generator(device="cuda").manual_seed(1000 + rows)
    sources = []
    for index in range(3):
        # The unused source tail is deliberately never initialized.
        storage = torch.empty(
            allocated, width + 8 * (index + 1), dtype=torch.bfloat16, device="cuda"
        )
        source = storage[:, :width]
        source[:rows].copy_(torch.randn(rows, width, generator=gen, device="cuda"))
        source[0, :3] = torch.tensor(
            [float("nan"), float("inf"), -float("inf")],
            device="cuda",
            dtype=torch.bfloat16,
        )
        sources.append(source)
    for capacity in sorted({rows, min(128, (rows + 15) // 16 * 16)}):
        aux = torch.full((allocated, 3 * width), 7, device="cuda", dtype=torch.bfloat16)
        positions = torch.full((allocated,), 99, device="cuda", dtype=torch.int64)
        slots = torch.full((2, allocated), 77, device="cuda", dtype=torch.int64)

        def run(aux=aux, positions=positions, slots=slots, capacity=capacity):
            dspark_context.context_inputs(
                aux, sources, positions, slots, rows, capacity
            )

        def check(aux=aux, positions=positions, slots=slots, capacity=capacity):
            expected = torch.full_like(aux, 7)
            for index, source in enumerate(sources):
                expected[:rows, index * width : (index + 1) * width].copy_(
                    source[:rows]
                )
            expected[rows:capacity].zero_()
            pos = torch.full_like(positions, 99)
            pos[rows:capacity].zero_()
            slot = torch.full_like(slots, 77)
            slot[:, rows:capacity].fill_(PAD_SLOT_ID)
            torch.cuda.synchronize()
            assert torch.equal(aux.view(torch.int16), expected.view(torch.int16))
            assert torch.equal(positions, pos)
            assert torch.equal(slots, slot)

        run()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for source in sources:
            source[:rows].neg_()
        aux.fill_(7)
        positions.fill_(99)
        slots.fill_(77)
        graph.replay()
        check()


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("split", [False, True])
def test_mxfp8_unwritten_tail_replay(rows, split):
    """Use the native arithmetic/reference; initcheck must see fresh tail reads."""
    from tests.kernels.quantization.test_deepseek_v41_tilelang_linear import (
        _block32_weight,
        _gen,
        _mxfp8,
    )
    from vllm.models.deepseek_v4_1.tilelang.gemm import (
        mxfp8_gemm,
        mxfp8_gemm_decode,
        mxfp8_gemm_partials,
        split_row_tile,
        splitk_reduce,
    )
    from vllm.models.deepseek_v4_1.tilelang.scale_pack import pack_scale_words

    n, k, shards = 64, 256, 2 if split else 1
    tile = split_row_tile(rows)
    padded = -(-rows // tile) * tile
    weight, exps, _ = _block32_weight(n, k, 41)
    weight_sf = pack_scale_words(exps)
    x = torch.randn((rows, k), generator=_gen(42), device="cuda").bfloat16()
    xq, sf, _ = _mxfp8(x)
    a = torch.empty((padded, k), dtype=xq.dtype, device="cuda")
    scales = torch.empty((padded, sf.shape[1]), dtype=sf.dtype, device="cuda")
    a[:rows].copy_(xq)
    scales[:rows].copy_(sf)
    out = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
    expected = torch.empty_like(out)
    reference = mxfp8_gemm(
        n, k, block_M=64, block_N=64, block_K=64, num_stages=2, shards=shards
    )
    reference(xq, weight, sf.view(torch.uint32), weight_sf, expected)
    if split:
        partials = torch.empty((shards, rows, n), device="cuda")
        kernel = mxfp8_gemm_partials(
            n, k, shards, block_M=tile, block_N=64, block_K=64, num_stages=2
        )
        reduce = splitk_reduce(n, shards)

        def run():
            kernel(a, weight, scales.view(torch.uint32), weight_sf, partials)
            reduce(partials, out)
    else:
        kernel = mxfp8_gemm_decode(
            n, k, block_M=tile, block_N=64, block_K=64, num_stages=2, padded_rows=True
        )

        def run():
            kernel(a, weight, scales.view(torch.uint32), weight_sf, out)

    run()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    a[:rows].copy_(-xq.float())
    a[rows:].view(torch.uint8).fill_(0x7F)
    scales[rows:].fill_(0xFF)
    reference(a[:rows], weight, scales[:rows].view(torch.uint32), weight_sf, expected)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
