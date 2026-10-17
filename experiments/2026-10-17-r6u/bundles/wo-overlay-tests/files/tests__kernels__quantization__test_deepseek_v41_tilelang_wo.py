# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang WO projection (grouped WO-A, WO-B) of WO-A's input, which
the sparse MLA kernels store already rotated back and through ``act_quant``."""

import pytest
import torch
from torch import nn

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")
tile_kernels = pytest.importorskip("tile_kernels")

from tests.kernels.deepseek_v41_contracts import (  # noqa: E402
    NAN,
    assert_other_rows_equal,
    assert_rows_nan,
    mxfp8_values,
    rope_table,
    wo_input_reference,
)
from vllm.models.deepseek_v4_1.tilelang.linear import (  # noqa: E402
    _block32_linear,
    _block32_scale_words,
    _declare_block32,
)
from vllm.models.deepseek_v4_1.tilelang.wo import TileLangWOProjection  # noqa: E402
from vllm.v1.worker.workspace import use_preallocated_workspace  # noqa: E402

DEVICE = "cuda"
# TP4: 16 heads of 512 in 2 groups, rank 1024, hidden 5120.
HEADS, HEAD_DIM, GROUPS, RANK, HIDDEN = 16, 512, 2, 1024, 5120
ROWS = (1, 7, 16, 33, 64, 65, 96, 97, 128, 129, 200)


def _gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _block32(n, k, seed):
    gen = _gen(seed)
    layer = nn.Module()
    layer.weight = (torch.randn((n, k), generator=gen, device=DEVICE) * 2).to(
        torch.float8_e4m3fn
    )
    exps = torch.randint(118, 123, (n // 32, k // 32), generator=gen, device=DEVICE)
    layer.weight_scale_inv = exps.to(torch.uint8).view(torch.float8_e8m0fnu)
    return layer


def _dequant(layer):
    exps = layer.weight_scale_inv.view(torch.uint8).double()
    scale = torch.exp2(exps - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)
    return layer.weight.double() * scale


def test_wo_projection():
    wo_a = _block32(GROUPS * RANK, HEADS // GROUPS * HEAD_DIM, 1)
    wo_b = _block32(HIDDEN, GROUPS * RANK, 2)
    wo = TileLangWOProjection(wo_a, wo_b, groups=GROUPS)
    wo.declare_kernels()  # standalone: prepared and bound now
    rows = max(ROWS)
    gen = _gen(3)
    o = torch.randn((rows, HEADS, HEAD_DIM), generator=gen, device=DEVICE).bfloat16()
    positions = torch.randint(0, 4096, (rows,), generator=gen, device=DEVICE)
    # WO-A's input as the sparse MLA kernels store it: rotated back, act_quant values.
    prepared = wo_input_reference(o, positions, rope_table())
    scratch = torch.empty(32 << 20, dtype=torch.uint8, device=DEVICE)

    def project(count):
        out = torch.empty((count, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
        with use_preallocated_workspace(scratch):
            wo(prepared[:count].clone(), out)
        return out

    full = project(rows)
    # Reference: FP64 GEMMs, WO-A on WO-A's input (DeepSeek's BF16 einsum on the
    # dequantized weights), WO-B on MXFP8 activations (its linear), the intermediate
    # rounded to BF16 as the kernel stores it.
    x = prepared.view(rows, -1).double()
    w_a, width = _dequant(wo_a), HEADS // GROUPS * HEAD_DIM
    a = torch.cat(
        [
            x[:, g * width : (g + 1) * width] @ w_a[g * RANK : (g + 1) * RANK].T
            for g in range(GROUPS)
        ],
        dim=1,
    ).bfloat16()
    ref = mxfp8_values(a) @ _dequant(wo_b).T
    torch.testing.assert_close(
        full.double(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item()
    )
    for count in ROWS:
        assert torch.equal(project(count), full[:count]), f"rows={count}"


@pytest.mark.parametrize("count", [7, 97])
def test_wo_projection_nan_row(count):
    """A NaN in one head of one row's input makes that row's whole projection NaN
    (WO-B mixes both groups) and leaves the other rows' bits."""
    wo = TileLangWOProjection(
        _block32(GROUPS * RANK, HEADS // GROUPS * HEAD_DIM, 1),
        _block32(HIDDEN, GROUPS * RANK, 2),
        groups=GROUPS,
    )
    wo.declare_kernels()
    gen = _gen(13)
    o = torch.randn((count, HEADS, HEAD_DIM), generator=gen, device=DEVICE).bfloat16()
    scratch = torch.empty(32 << 20, dtype=torch.uint8, device=DEVICE)

    def project(values):
        out = torch.empty((count, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
        with use_preallocated_workspace(scratch):
            wo(values.clone(), out)
        return out

    clean = project(o)
    o[3, 5, 17] = NAN
    out = project(o)
    assert_rows_nan(out, [3])
    assert_other_rows_equal(out, clean, [3])


@pytest.mark.parametrize("count", ROWS)
def test_grouped_gemm_equals_groups(count, default_vllm_config):
    """One grouped launch gives each group's ordinary GEMM bit for bit."""
    width = HEADS // GROUPS * HEAD_DIM
    grouped = _block32(GROUPS * RANK, width, 4)
    _block32_scale_words(grouped)
    _declare_block32(grouped, GROUPS)
    x = torch.randn(
        (count, GROUPS * width), generator=_gen(5), device=DEVICE
    ).bfloat16()
    scratch = torch.empty(32 << 20, dtype=torch.uint8, device=DEVICE)
    out = torch.empty((count, GROUPS * RANK), dtype=torch.bfloat16, device=DEVICE)
    with use_preallocated_workspace(scratch):
        _block32_linear(x, out, grouped.tilelang_key, scratch)
        for g in range(GROUPS):
            part = nn.Module()
            part.weight = grouped.weight[g * RANK : (g + 1) * RANK]
            part.weight_scale_inv = grouped.weight_scale_inv[
                g * RANK // 32 : (g + 1) * RANK // 32
            ]
            _block32_scale_words(part)
            _declare_block32(part)
            single = torch.empty((count, RANK), dtype=torch.bfloat16, device=DEVICE)
            _block32_linear(
                x[:, g * width : (g + 1) * width].contiguous(),
                single,
                part.tilelang_key,
                scratch,
            )
            assert torch.equal(out[:, g * RANK : (g + 1) * RANK], single), f"group {g}"


def _wo_a_bf16(layer, x):
    """WO-A's BF16 output through the unfused kernels with the WO's tile per bucket:
    the value the fused path rounds to BF16 before its cast."""
    from vllm.models.deepseek_v4_1.tilelang import gemm
    from vllm.models.deepseek_v4_1.tilelang.wo import (
        _WO_A_COMMON,
        _WO_A_SHARDS,
        _WO_A_TILES,
        _GroupedBF16,
    )

    rows, (n, k) = x.shape[0], layer.weight.shape
    tile = _WO_A_TILES[_GroupedBF16.bucket(rows)]
    e = layer.weight_scale_inv.view(torch.uint8)
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=DEVICE)
    if tile["split"]:
        partials = torch.empty(
            (_WO_A_SHARDS, rows, n), dtype=torch.float32, device=DEVICE
        )
        gemm.fp8w_bf16_gemm_partials(
            n // GROUPS,
            k,
            GROUPS,
            _WO_A_SHARDS,
            block_M=tile["block_M"],
            **_WO_A_COMMON,
        )(x, layer.weight, e, partials)
        gemm.splitk_reduce(n, _WO_A_SHARDS)(partials, out)
    else:
        gemm.fp8w_bf16_gemm(
            n // GROUPS,
            k,
            GROUPS,
            shards=_WO_A_SHARDS,
            block_M=tile["block_M"],
            **_WO_A_COMMON,
        )(x, layer.weight, e, out)
    return out


def test_wo_a_on_act_quant_values():
    """WO-A within one BF16 rounding of the FP64 product of its input (act_quant's MXFP8
    values, as the sparse MLA kernels store them, below the amax floor too) and the
    exactly dequantized weights, with the same bits at every row count (decode rows
    split the K shards over CTAs and longer steps add them inside the kernel, in the
    same order)."""
    width = HEADS // GROUPS * HEAD_DIM
    layer = _block32(GROUPS * RANK, width, 7)
    x = torch.randn((300, GROUPS * width), generator=_gen(8), device=DEVICE).bfloat16()
    x[::5] *= 1e-7  # some blocks fall below the amax floor
    quantized = mxfp8_values(x)
    x = quantized.bfloat16()
    w = _dequant(layer)

    full = _wo_a_bf16(layer, x)
    ref = torch.cat(
        [
            quantized[:, g * width : (g + 1) * width] @ w[g * RANK : (g + 1) * RANK].T
            for g in range(GROUPS)
        ],
        dim=1,
    )
    error = (full.double() - ref).abs()
    assert (error <= ref.abs() * 2**-8 + 1e-3 * ref.abs().max()).all()
    for rows in (1, 7, 16, 17, 33, 64, 65, 96, 97, 100, 128, 129, 200):
        assert torch.equal(_wo_a_bf16(layer, x[:rows]), full[:rows]), rows


def test_wo_a_casts_as_per_token_cast():
    """WO-A's fused output is the reference's activation cast of its BF16 output, byte
    for byte (E4M3 values and UE8M0 bytes), on the split decode path and the in-kernel
    path, including blocks below the amax floor."""
    from vllm.models.deepseek_v4_1.tilelang.wo import _GroupedBF16

    width = HEADS // GROUPS * HEAD_DIM
    layer = _block32(GROUPS * RANK, width, 9)
    wo_a = _GroupedBF16(layer, GROUPS)
    wo_a.declare_kernels()
    x = torch.randn((300, GROUPS * width), generator=_gen(10), device=DEVICE).bfloat16()
    x[::5] *= 1e-7  # some rows' outputs fall below the amax floor
    for rows in (1, 7, 16, 17, 33, 64, 65, 96, 97, 100, 128, 129, 200, 300):
        a = _wo_a_bf16(layer, x[:rows])
        ref_q, ref_sf = tile_kernels.quant.per_token_cast(
            a, "e4m3", 32, round_sf=True, use_packed_ue8m0=True
        )
        xq = torch.empty(
            (rows, GROUPS * RANK), dtype=torch.float8_e4m3fn, device=DEVICE
        )
        sf = torch.empty((rows, GROUPS * RANK // 32), dtype=torch.uint8, device=DEVICE)
        specs = wo_a.partial_specs(rows)
        partials = (
            torch.empty(specs[0][0], dtype=specs[0][1], device=DEVICE)
            if specs
            else None
        )
        wo_a(x[:rows], xq, sf, partials)
        assert torch.equal(xq.view(torch.uint8), ref_q.view(torch.uint8)), rows
        assert torch.equal(
            sf, ref_sf.contiguous().view(torch.uint8)[:, : sf.shape[1]]
        ), rows
