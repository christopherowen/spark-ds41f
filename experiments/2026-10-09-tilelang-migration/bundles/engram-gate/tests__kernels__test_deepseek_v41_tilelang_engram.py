# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang Engram gate against FP64 and TileKernels' engram_gate_fwd."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")
tile_kernels = pytest.importorskip("tile_kernels")

from vllm.models.deepseek_v4_1.tilelang.engram import CLAMP, engram_gate  # noqa: E402

DEVICE = "cuda"
HC, DIM, EPS = 4, 5120, 1e-6
ROWS = (1, 6, 17, 96, 300)


def _inputs(rows, seed=3):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn((rows, HC * DIM), generator=gen, device=DEVICE).bfloat16()
    kv = torch.randn((rows, (HC + 1) * DIM), generator=gen, device=DEVICE).bfloat16()
    weight = torch.rand((HC * DIM,), generator=gen, device=DEVICE) + 0.5
    image = torch.rand((rows,), generator=gen, device=DEVICE) < 0.25
    return x, kv, weight, image


def _reference(x, kv, weight, image):
    rows = x.shape[0]
    xs = x.double().view(rows, HC, DIM)
    kvs = kv.double().view(rows, HC + 1, DIM)
    k, v = kvs[:, :HC], kvs[:, HC:]
    w = weight.double().view(HC, DIM)
    dot = (xs * w * k).sum(-1)
    dot = dot * torch.rsqrt(xs.square().mean(-1) + EPS)
    dot = dot * torch.rsqrt(k.square().mean(-1) + EPS) * DIM**-0.5
    gate = torch.sigmoid(dot.sign() * dot.abs().clamp(min=CLAMP).sqrt())
    gate = torch.where(image[:, None], 0.0, gate)
    return (xs + gate[..., None] * v).view(rows, HC * DIM)


@pytest.mark.parametrize("masked", [True, False])
def test_engram_gate(masked):
    x, kv, weight, image = _inputs(max(ROWS))
    mask = image if masked else None
    full = torch.empty_like(x)
    engram_gate(x, kv, weight, mask, full, EPS, HC)
    ref = _reference(x, kv, weight, image if masked else torch.zeros_like(image))
    torch.testing.assert_close(full.double(), ref, rtol=1e-2, atol=1e-2)
    if masked:
        assert torch.equal(full[image], x[image]), "image tokens must pass through"
    # TileKernels' reference kernel: the same formula, its own summation order.
    expected, *_ = tile_kernels.engram.engram_gate_fwd(
        x.view(-1, HC, DIM),
        kv.view(-1, HC + 1, DIM),
        weight.view(HC, DIM),
        EPS,
        CLAMP,
        save_for_backward=False,
        image_token_mask=mask,
    )
    torch.testing.assert_close(full.view_as(expected), expected, rtol=1e-2, atol=1e-2)
    for rows in ROWS:
        out = torch.full_like(x, float("nan"))
        engram_gate(
            x[:rows],
            kv[:rows],
            weight,
            None if mask is None else mask[:rows],
            out[:rows],
            EPS,
            HC,
        )
        assert torch.equal(out[:rows], full[:rows]), f"rows={rows}"
        assert out[rows:].isnan().all(), f"rows={rows} wrote past its rows"
