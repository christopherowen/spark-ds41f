# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang WO projection (inverse RoPE, grouped WO-A, WO-B)."""

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

from vllm.models.deepseek_v4_1.tilelang.linear import (  # noqa: E402
    _block32_linear,
    _prepare_block32,
)
from vllm.models.deepseek_v4_1.tilelang.wo import TileLangWOProjection  # noqa: E402
from vllm.v1.worker.workspace import use_preallocated_workspace  # noqa: E402

DEVICE = "cuda"
# TP4: 16 heads of 512 in 2 groups, rank 1024, hidden 5120.
HEADS, HEAD_DIM, GROUPS, RANK, HIDDEN, ROPE = 16, 512, 2, 1024, 5120, 64
ROWS = (1, 7, 16, 33, 64, 65, 200)


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


def _mxfp8_values(x):
    xq, sf = tile_kernels.quant.per_token_cast(
        x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True
    )
    exps = sf.contiguous().view(torch.uint8)[:, : x.shape[1] // 32].double()
    return xq.double() * torch.exp2(exps - 127).repeat_interleave(32, -1)


def _table(positions=4096):
    inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=DEVICE).double() / ROPE))
    angles = torch.arange(positions, device=DEVICE).double()[:, None] * inv
    return torch.cat((angles.cos(), angles.sin()), dim=-1).float()


def test_wo_projection():
    wo_a = _block32(GROUPS * RANK, HEADS // GROUPS * HEAD_DIM, 1)
    wo_b = _block32(HIDDEN, GROUPS * RANK, 2)
    wo = TileLangWOProjection(wo_a, wo_b, groups=GROUPS)
    rows = max(ROWS)
    gen = _gen(3)
    o = torch.randn((rows, HEADS, HEAD_DIM), generator=gen, device=DEVICE).bfloat16()
    positions = torch.randint(0, 4096, (rows,), generator=gen, device=DEVICE)
    table = _table()
    scratch = torch.empty(32 << 20, dtype=torch.uint8, device=DEVICE)

    def project(count):
        out = torch.empty((count, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
        with use_preallocated_workspace(scratch):
            wo(o[:count].clone(), positions[:count], table, out)
        return out

    full = project(rows)
    # Reference: TileKernels' inverse rotation in BF16, then FP64 GEMMs on the
    # MXFP8 activations, the intermediate rounded to BF16 as the kernel stores it.
    rotated = o.clone()
    tile_kernels.transform.apply_rotary(
        rotated[..., -ROPE:],
        table,
        positions=positions,
        interleaved=True,
        conjugate=True,
    )
    x = _mxfp8_values(rotated.view(rows, -1))
    w_a, width = _dequant(wo_a), HEADS // GROUPS * HEAD_DIM
    a = torch.cat(
        [
            x[:, g * width : (g + 1) * width] @ w_a[g * RANK : (g + 1) * RANK].T
            for g in range(GROUPS)
        ],
        dim=1,
    ).bfloat16()
    ref = _mxfp8_values(a) @ _dequant(wo_b).T
    torch.testing.assert_close(
        full.double(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item()
    )
    for count in ROWS:
        assert torch.equal(project(count), full[:count]), f"rows={count}"


@pytest.mark.parametrize("count", ROWS)
def test_grouped_gemm_equals_groups(count):
    """One grouped launch gives each group's ordinary GEMM bit for bit."""
    width = HEADS // GROUPS * HEAD_DIM
    grouped = _block32(GROUPS * RANK, width, 4)
    _prepare_block32(grouped, GROUPS)
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
            _prepare_block32(part)
            single = torch.empty((count, RANK), dtype=torch.bfloat16, device=DEVICE)
            _block32_linear(
                x[:, g * width : (g + 1) * width].contiguous(),
                single,
                part.tilelang_key,
                scratch,
            )
            assert torch.equal(out[:, g * RANK : (g + 1) * RANK], single), f"group {g}"
