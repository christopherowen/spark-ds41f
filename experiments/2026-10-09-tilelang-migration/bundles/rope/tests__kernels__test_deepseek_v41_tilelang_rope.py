# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang RoPE against TileKernels' apply_rotary and FP64."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")
tile_kernels = pytest.importorskip("tile_kernels")

from vllm.models.deepseek_v4_1.tilelang.rope import rope_  # noqa: E402

DEVICE = "cuda"
ROPE, POSITIONS = 64, 4096


def _table():
    inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=DEVICE).double() / ROPE))
    angles = torch.arange(POSITIONS, device=DEVICE).double()[:, None] * inv
    return torch.cat((angles.cos(), angles.sin()), dim=-1).float()


# q (TP4), index query, kv, latent and index key at compression ratio 4.
@pytest.mark.parametrize(
    "heads, dim, ratio",
    [(16, 512, 1), (64, 128, 1), (1, 512, 1), (1, 512, 4), (1, 128, 4)],
)
@pytest.mark.parametrize("inverse", [False, True])
def test_rope(heads, dim, ratio, inverse, rows=97):
    gen = torch.Generator(device=DEVICE).manual_seed(heads + dim + ratio)
    x = torch.randn((rows, heads, dim), generator=gen, device=DEVICE).bfloat16()
    positions = torch.randint(0, POSITIONS, (rows,), generator=gen, device=DEVICE)
    table = _table()
    out = x.clone()
    rope_(out if heads > 1 else out.view(rows, dim), positions, table, ratio, inverse)
    assert torch.equal(out[..., :-ROPE], x[..., :-ROPE]), "leading columns moved"
    floored = positions // ratio * ratio
    expected = x.clone()
    tile_kernels.transform.apply_rotary(
        expected[..., -ROPE:],
        table,
        positions=floored,
        interleaved=True,
        conjugate=inverse,
    )
    assert torch.equal(out, expected), "differs from TileKernels' apply_rotary"
    # FP64: interleaved pairs (x0, x1) -> (x0 cos - x1 sin, x0 sin + x1 cos).
    cos, sin = table.double()[floored].chunk(2, dim=-1)
    sin = -sin if inverse else sin
    pairs = x[..., -ROPE:].double().unflatten(-1, (ROPE // 2, 2))
    first, second = pairs[..., 0], pairs[..., 1]
    cos, sin = cos[:, None], sin[:, None]
    ref = torch.stack((first * cos - second * sin, first * sin + second * cos), -1)
    torch.testing.assert_close(
        out[..., -ROPE:].double(), ref.flatten(-2), rtol=1e-2, atol=1e-2
    )
    # Rows are independent: a prefix rotates to the same bits.
    part = x[:5].clone()
    rope_(part, positions[:5], table, ratio, inverse)
    assert torch.equal(part, out[:5])
