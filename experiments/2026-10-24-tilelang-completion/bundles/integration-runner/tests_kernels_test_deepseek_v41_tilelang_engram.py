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

from tests.kernels.deepseek_v41_contracts import (  # noqa: E402
    NAN,
    assert_other_rows_equal,
    assert_same_nonfinite,
)
from vllm.models.deepseek_v4_1.tilelang.engram import (  # noqa: E402
    CLAMP,
    engram_gate,
    engram_gate_blocked,
)
from vllm.models.deepseek_v4_1.tilelang.engram_hash import (  # noqa: E402
    IMAGE_SENTINEL,
)

DEVICE = "cuda"
HC, DIM, EPS = 4, 5120, 1e-6
ROWS = (1, 6, 17, 96, 300)


def _inputs(rows, seed=3, images=True, ids_dtype=torch.int32):
    """The gate's inputs and the step's token ids, a quarter of them the image
    sentinel when ``images``, the rest its neighbours (the compare is exact)."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn((rows, HC * DIM), generator=gen, device=DEVICE).bfloat16()
    kv = torch.randn((rows, (HC + 1) * DIM), generator=gen, device=DEVICE).bfloat16()
    weight = torch.rand((HC * DIM,), generator=gen, device=DEVICE) + 0.5
    ids = torch.randint(
        IMAGE_SENTINEL - 1, IMAGE_SENTINEL + 2, (rows,), generator=gen, device=DEVICE
    )
    ids[ids == IMAGE_SENTINEL] += 1
    if images:
        ids[torch.rand((rows,), generator=gen, device=DEVICE) < 0.25] = IMAGE_SENTINEL
    return x, kv, weight, ids.to(ids_dtype)


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


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("images", [True, False])
def test_engram_gate(images, ids_dtype):
    x, kv, weight, ids = _inputs(max(ROWS), images=images, ids_dtype=ids_dtype)
    image = ids == IMAGE_SENTINEL
    assert bool(image.any()) == images
    full = torch.empty_like(x)
    engram_gate(x, kv, weight, ids, full, EPS, HC)
    ref = _reference(x, kv, weight, image)
    torch.testing.assert_close(full.double(), ref, rtol=1e-2, atol=1e-2)
    assert torch.equal(full[image], x[image]), "image tokens must pass through"
    # TileKernels' reference kernel: the same formula, its own summation order.
    expected, *_ = tile_kernels.engram.engram_gate_fwd(
        x.view(-1, HC, DIM),
        kv.view(-1, HC + 1, DIM),
        weight.view(HC, DIM),
        EPS,
        CLAMP,
        save_for_backward=False,
        image_token_mask=image if images else None,
    )
    torch.testing.assert_close(full.view_as(expected), expected, rtol=1e-2, atol=1e-2)
    for rows in ROWS:
        out = torch.full_like(x, float("nan"))
        engram_gate(x[:rows], kv[:rows], weight, ids[:rows], out[:rows], EPS, HC)
        assert torch.equal(out[:rows], full[:rows]), f"rows={rows}"
        assert out[rows:].isnan().all(), f"rows={rows} wrote past its rows"


@pytest.mark.parametrize("where", ["residual", "key", "value"])
def test_nonfinite_like_the_reference(where):
    """A NaN in a stream's residual or key makes its gate NaN, so the reference's
    whole stream is NaN (``clamp_min`` passes it, unlike ``max(|dot|, CLAMP)``);
    a NaN in the shared value reaches every stream at that channel."""
    rows, t, h, d = 17, 5, 2, 100
    x, kv, weight, ids = _inputs(rows, seed=5, images=False)
    clean = torch.empty_like(x)
    engram_gate(x, kv, weight, ids, clean, EPS, HC)
    if where == "residual":
        x[t, h * DIM + d] = NAN
    elif where == "key":
        kv[t, h * DIM + d] = NAN
    else:
        kv[t, HC * DIM + d] = NAN
    out = torch.empty_like(x)
    engram_gate(x, kv, weight, ids, out, EPS, HC)
    ref = _reference(x, kv, weight, torch.zeros(rows, dtype=torch.bool, device=DEVICE))
    assert_same_nonfinite(out, ref)
    assert_other_rows_equal(out, clean, [t])


@pytest.mark.parametrize("images", [False, True])
@pytest.mark.parametrize("blocks, block_cols", [(3, 8544), (4, 6400)])
@pytest.mark.parametrize("rows", [6, 300])
def test_blocked_kv_gives_the_packed_bits(images, blocks, block_cols, rows):
    """kv as prefill SP's exchanged column blocks [N, L, C] (TP3's and TP4's widths),
    read in place, gives the packed gate's bits; rows past the tokens are not read."""
    x, kv, weight, ids = _inputs(rows, seed=7, images=images)
    width = (HC + 1) * DIM
    padded = torch.zeros((rows + 3, blocks * block_cols), device=DEVICE).bfloat16()
    padded[:rows, :width] = kv
    padded[rows:] = NAN  # rows a rank does not own: never read
    blocked = padded.view(rows + 3, blocks, block_cols).transpose(0, 1).contiguous()
    packed_out = torch.empty_like(x)
    blocked_out = torch.empty_like(x)
    engram_gate(x, kv, weight, ids, packed_out, EPS, HC)
    engram_gate_blocked(x, blocked, weight, ids, blocked_out, EPS, HC)
    torch.cuda.synchronize()
    assert torch.equal(packed_out.view(torch.int16), blocked_out.view(torch.int16))
