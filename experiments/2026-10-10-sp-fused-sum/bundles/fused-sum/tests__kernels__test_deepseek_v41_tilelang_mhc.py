# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 lagged mHC on TileKernels against the FP32 reference, with
results independent of the batch."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")
pytest.importorskip("tile_kernels")

from vllm.models.deepseek_v4_1.tilelang.collectives import (  # noqa: E402
    RankOrderParts,
)
from vllm.models.deepseek_v4_1.tilelang.mhc import (  # noqa: E402
    BLOCK_H,
    MIXES,
    TileKernelsMHC,
    project_streams,
)

DEVICE = "cuda"
HIDDEN = 5120
CONFIG = SimpleNamespace(
    hidden_size=HIDDEN,
    hc_mult=4,
    rms_norm_eps=1e-20,
    hc_eps=1e-6,
    hc_sinkhorn_iters=20,
)


def _reference(residual, fn, scale, base, incoming, weight, eps=1e-20, hc_eps=1e-6):
    """V4.1's lagged mHC pre in FP32: (post, comb, y, new pre-mix)."""
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + base[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    collapsed = (incoming.unsqueeze(-1) * residual.float()).sum(1).bfloat16().float()
    y = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + eps)
    return post, comb, (y * weight.float()).bfloat16(), pre


def _post_reference(x, residual, post, comb):
    return (
        post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).to(x.dtype)


def _inputs(tokens, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=DEVICE) * scale

    residual = randn(tokens, 4, HIDDEN).bfloat16()
    fn = randn(MIXES, 4 * HIDDEN, scale=0.02)
    scale = randn(3, scale=0.1) + 1.0
    base = randn(MIXES, scale=0.1)
    weight = (randn(HIDDEN, scale=0.1) + 1.0).bfloat16()
    incoming = torch.softmax(randn(tokens, 4), dim=-1) + 1e-6
    return residual, fn, scale, base, weight, incoming


@pytest.fixture(scope="module")
def mhc():
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device(DEVICE))
    return TileKernelsMHC(CONFIG)


def _close_bf16(actual, expected, ulps=1):
    """Within ``ulps`` BF16 ulps; near zero, where the stream sums cancel, the
    rounding of fused multiply-adds may differ from torch's by a few 1e-8."""
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=1e-5, rtol=ulps * 2**-7
    )


def _check(actual, expected, tokens):
    post, comb, y, pre = expected
    residual_out, a_post, a_comb, a_y, a_pre = actual
    assert residual_out.shape == (tokens, 4, HIDDEN)
    # The projection runs on TF32 tensor cores, as in DeepGEMM.
    torch.testing.assert_close(a_post, post, atol=2e-3, rtol=0)
    torch.testing.assert_close(a_comb, comb, atol=2e-3, rtol=0)
    torch.testing.assert_close(a_pre, pre, atol=2e-3, rtol=0)
    # y takes no projection. The collapse is rounded to BF16 before the norm,
    # so an ulp of difference there can become two in y.
    _close_bf16(a_y, y, ulps=2)


@pytest.mark.parametrize("tokens", [1, 7, 16, 33, 64, 200])
def test_pre_matches_reference(mhc, tokens):
    residual, fn, scale, base, weight, incoming = _inputs(tokens, seed=tokens)
    actual = mhc.pre(residual, fn, scale, base, weight, incoming)
    _check(actual, _reference(residual, fn, scale, base, incoming, weight), tokens)
    assert actual[0] is residual


@pytest.mark.parametrize("tokens", [1, 9, 64])
def test_post_pre_matches_reference(mhc, tokens):
    residual, fn, scale, base, weight, incoming = _inputs(tokens, seed=100 + tokens)
    g = torch.Generator(device=DEVICE).manual_seed(tokens)
    x = torch.randn(tokens, HIDDEN, generator=g, device=DEVICE).bfloat16()
    post = torch.rand(tokens, 4, generator=g, device=DEVICE) * 2
    comb = torch.softmax(torch.randn(tokens, 4, 4, generator=g, device=DEVICE), -1)
    actual = mhc.post_pre(x, residual, post, comb, fn, scale, base, weight, incoming)
    updated = _post_reference(x, residual, post, comb)
    _close_bf16(actual[0], updated)
    _check(actual, _reference(actual[0], fn, scale, base, incoming, weight), tokens)


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("tokens", [1, 9, 64, 300])
def test_post_pre_adds_rank_order_parts(mhc, world, tokens):
    """Unreduced partials give exactly the result of their rank-order sum."""
    residual, fn, scale, base, weight, incoming = _inputs(tokens, seed=200 + tokens)
    g = torch.Generator(device=DEVICE).manual_seed(world * 1000 + tokens)
    parts = torch.randn(world, tokens, HIDDEN, generator=g, device=DEVICE)
    parts = parts.bfloat16()
    parts[0, 0, :8] = -0.0  # a signed zero, added to nothing, survives
    post = torch.rand(tokens, 4, generator=g, device=DEVICE) * 2
    comb = torch.softmax(torch.randn(tokens, 4, 4, generator=g, device=DEVICE), -1)
    summed = RankOrderParts(parts).sum()
    expected = mhc.post_pre(
        summed, residual, post, comb, fn, scale, base, weight, incoming
    )
    actual = mhc.post_pre(
        RankOrderParts(parts), residual, post, comb, fn, scale, base, weight, incoming
    )
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)
    assert torch.equal(
        mhc.post(RankOrderParts(parts), residual, post, comb),
        mhc.post(summed, residual, post, comb),
    )


def test_first_layer_broadcast_matches_expanded(mhc):
    tokens = 13
    residual, fn, scale, base, weight, _ = _inputs(tokens, seed=5)
    embedding = residual[:, 0].contiguous()
    broadcast = fn.view(MIXES, 4, HIDDEN).sum(1)
    out = mhc.pre(embedding, broadcast, scale, base, weight, None)
    expanded = embedding[:, None].expand(-1, 4, -1).contiguous()
    first = torch.zeros(tokens, 4, device=DEVICE)
    first[:, 0] = 1
    assert torch.equal(out[0], expanded)
    _check(out, _reference(expanded, fn, scale, base, first, weight), tokens)


def test_rows_independent_of_batch(mhc):
    residual, fn, scale, base, weight, incoming = _inputs(130, seed=9)
    full = mhc.pre(residual, fn, scale, base, weight, incoming)
    for rows in (1, 2, 7, 16, 17, 33, 64):
        part = mhc.pre(residual[:rows], fn, scale, base, weight, incoming[:rows])
        for a, b in zip(part[1:], full[1:]):
            assert torch.equal(a, b[:rows]), rows
    part = mhc.pre(residual[70:71], fn, scale, base, weight, incoming[70:71])
    for a, b in zip(part[1:], full[1:]):
        assert torch.equal(a, b[70:71])


@pytest.mark.parametrize("block_M", [16, 32])
def test_project_streams_partials(block_M):
    tokens, k, splits = 37, 4 * HIDDEN, HIDDEN // BLOCK_H
    residual, fn, _, _, _, incoming = _inputs(tokens, seed=3)
    flat = residual.view(tokens, k)
    P = torch.empty(splits, tokens, 1, MIXES, device=DEVICE)
    S = torch.empty(splits, tokens, 1, device=DEVICE)
    C = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    Q = torch.empty(splits, tokens, device=DEVICE)
    y = torch.empty_like(C)
    comb = torch.empty(tokens, 4, 4, device=DEVICE)
    project_streams(HIDDEN, "pre", block_M=block_M)(
        y.unsqueeze(0), flat, incoming, comb, incoming, fn, flat, P, S, C, Q
    )
    # The projection runs on TF32 tensor cores, as in DeepGEMM.
    reference = flat.float() @ fn.T
    torch.testing.assert_close(P.sum(0)[:, 0], reference, atol=2e-2, rtol=2e-3)
    squares = flat.float().square().sum(-1)
    torch.testing.assert_close(S.sum(0)[:, 0], squares, atol=0, rtol=1e-5)
    collapsed = (incoming.unsqueeze(-1) * residual.float()).sum(1).bfloat16()
    _close_bf16(C, collapsed)
    torch.testing.assert_close(Q.sum(0), C.float().square().sum(-1), atol=0, rtol=1e-5)


def test_post_and_collapse(mhc):
    tokens = 21
    residual, *_ = _inputs(tokens, seed=4)
    g = torch.Generator(device=DEVICE).manual_seed(4)
    x = torch.randn(tokens, HIDDEN, generator=g, device=DEVICE).bfloat16()
    post = torch.rand(tokens, 4, generator=g, device=DEVICE) * 2
    comb = torch.softmax(torch.randn(tokens, 4, 4, generator=g, device=DEVICE), -1)
    updated = mhc.post(x, residual, post, comb)
    _close_bf16(updated, _post_reference(x, residual, post, comb))
    mix = torch.rand(tokens, 4, generator=g, device=DEVICE)
    weighted = (residual.float() * mix[:, :, None]).sum(1).bfloat16()
    _close_bf16(mhc.collapse(residual, mix), weighted)
    mean = residual.float().mean(1).bfloat16()
    _close_bf16(mhc.collapse(residual), mean)
