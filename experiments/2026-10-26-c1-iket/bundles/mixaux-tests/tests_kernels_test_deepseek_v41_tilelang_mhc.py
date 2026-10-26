# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 lagged mHC on TileKernels against the FP32 reference, with
results independent of the batch; the standalone post and the stream collapse bit
for bit against TileKernels' CUDA kernels they replace (``mhc_post_fwd``,
``mhc_pre_apply_mix_fwd``)."""

import functools
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
tile_kernels = pytest.importorskip("tile_kernels")

from tests.kernels.deepseek_v41_contracts import (  # noqa: E402
    INF,
    NAN,
    assert_other_rows_equal,
    assert_rows_nan,
    assert_same_nonfinite,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import mhc as mhc_module  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang import prepare  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.collectives import (  # noqa: E402
    RankOrderParts,
)
from vllm.models.deepseek_v4_1.tilelang.gemm import scale_words  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.linear import (  # noqa: E402
    empty_mxfp8,
    mxfp8_rows,
)
from vllm.models.deepseek_v4_1.tilelang.mhc import (  # noqa: E402
    BLOCK_H,
    MIXES,
    SIDE_ROWS,
    TileKernelsMHC,
    collapse_streams,
    finalize,
    post_streams,
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


class _Joined(TileKernelsMHC):
    """The tests read an entry's mixes as soon as it returns; in the model the next
    mHC operation joins the side stream that computes them first."""

    def pre(self, *args, **kwargs):
        out = super().pre(*args, **kwargs)
        mhc_module.join_mixes(out[1].device)
        return out


@pytest.fixture(scope="module")
def mhc():
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device(DEVICE))
    return _Joined(CONFIG)


def _close_bf16(actual, expected, ulps=1):
    """Within ``ulps`` BF16 ulps; near zero, where the stream sums cancel, the
    rounding of fused multiply-adds may differ from torch's by a few 1e-8."""
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=1e-5, rtol=ulps * 2**-7
    )


def _per_token_cast(y):
    """TileKernels' MXFP8 cast of BF16 rows, as the projections cast their input
    before the entry wrote it: E4M3 bytes and packed UE8M0 bytes, whole words a row."""
    rows, hidden = y.shape
    values = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=DEVICE)
    scales = torch.empty(
        (rows, 4 * scale_words(hidden)), dtype=torch.uint8, device=DEVICE
    )
    tile_kernels.quant.per_token_cast(
        y, "e4m3", 32, round_sf=True, use_packed_ue8m0=True, out=(values, scales)
    )
    return values, scales


def _assert_bytes_equal(actual, expected, what=""):
    actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
    assert actual.shape == expected.shape
    differ = actual != expected
    if differ.any():
        where = differ.nonzero()[:4].tolist()
        raise AssertionError(
            f"{what}{int(differ.sum())} of {differ.numel()} bytes differ; first at "
            f"{where}"
        )


def _assert_cast(cast, y, what=""):
    """An entry's ``MXFP8Rows`` hold ``per_token_cast``'s bytes of the ``y`` it wrote,
    in buffers of whole decode tiles."""
    tokens = y.shape[0]
    assert cast.rows is y
    assert cast.values.shape == (mxfp8_rows(tokens), y.shape[1])
    assert cast.scales.shape == (mxfp8_rows(tokens), 4 * scale_words(y.shape[1]))
    values, scales = _per_token_cast(y)
    _assert_bytes_equal(cast.values[:tokens], values, f"{what}E4M3: ")
    _assert_bytes_equal(cast.scales[:tokens], scales, f"{what}scales: ")


def _check(actual, expected, tokens):
    post, comb, y, pre = expected
    residual_out, a_post, a_comb, a_y, a_pre, a_cast = actual
    assert residual_out.shape == (tokens, 4, HIDDEN)
    # The projection runs on TF32 tensor cores, as in DeepGEMM.
    torch.testing.assert_close(a_post, post, atol=2e-3, rtol=0)
    torch.testing.assert_close(a_comb, comb, atol=2e-3, rtol=0)
    torch.testing.assert_close(a_pre, pre, atol=2e-3, rtol=0)
    # y takes no projection. The collapse is rounded to BF16 before the norm,
    # so an ulp of difference there can become two in y.
    _close_bf16(a_y, y, ulps=2)
    _assert_cast(a_cast, a_y)


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
    for a, b in zip(actual[:5], expected[:5]):
        assert torch.equal(a, b)
    _assert_bytes_equal(actual[5].values[:tokens], expected[5].values[:tokens])
    _assert_bytes_equal(actual[5].scales[:tokens], expected[5].scales[:tokens])
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

    def same(part, rows):
        for a, b in zip(part[1:5], full[1:5]):
            assert torch.equal(a, b[rows]), rows
        cast, whole = part[5], full[5]
        n = cast.rows.shape[0]
        _assert_bytes_equal(cast.values[:n], whole.values[rows], f"{rows}: ")
        _assert_bytes_equal(cast.scales[:n], whole.scales[rows], f"{rows}: ")

    for rows in (1, 2, 7, 16, 17, 33, 64):
        same(
            mhc.pre(residual[:rows], fn, scale, base, weight, incoming[:rows]),
            slice(0, rows),
        )
    same(
        mhc.pre(residual[70:71], fn, scale, base, weight, incoming[70:71]),
        slice(70, 71),
    )


def test_pre_nan_token(mhc):
    """A NaN in one token's residual makes all of that token's mixes and its y NaN,
    as the FP32 reference's softmax, Sinkhorn and norm do, and moves no other
    token."""
    tokens, t = 33, 7
    residual, fn, scale, base, weight, incoming = _inputs(tokens, seed=11)
    clean = [
        a.clone() for a in mhc.pre(residual, fn, scale, base, weight, incoming)[:5]
    ]
    residual = residual.clone()
    residual[t, 2, 100] = NAN
    actual = mhc.pre(residual, fn, scale, base, weight, incoming)
    for name, got, before in zip(("post", "comb", "y", "pre"), actual[1:], clean[1:]):
        assert_rows_nan(got, [t], f"{name}: ")
        assert_other_rows_equal(got, before, [t], f"{name}: ")


# The entries' MXFP8 cast of y, for the sublayer's block-32 projections: TileKernels'
# per_token_cast's bytes (E4M3 and packed UE8M0), which the projections no longer write.


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("tokens", [1, 6, 64, 65, 96, 129, 300, 2048])
def test_entry_cast_is_per_token_cast(mhc, tokens, first):
    """At decode and prefill sizes (decode sizes in whole tiles), from both finalize
    widths (the first layer's broadcast, four streams), and in every scale regime the
    norm weight gives y: zeroed blocks (the 1e-4 amax floor), blocks far below it,
    blocks above 448 (positive exponents) and block maxima a power of two apart."""
    residual, fn, scale, base, weight, incoming = _inputs(tokens, seed=1300 + tokens)
    weight[:64] = 0.0
    weight[64:128] = 1e-30
    weight[128:192] *= 4096.0
    weight[192:256] = 2.0 ** torch.arange(-16, 48, device=DEVICE, dtype=torch.float32)
    if first:
        embedding = residual[:, 0].contiguous()
        broadcast = fn.view(MIXES, 4, HIDDEN).sum(1)
        out = mhc.pre(embedding, broadcast, scale, base, weight, None)
    else:
        out = mhc.pre(residual, fn, scale, base, weight, incoming)
    _assert_cast(out[5], out[3])


def test_entry_cast_of_a_nan_token(mhc):
    """A NaN token's y row casts as per_token_cast casts it; no other row moves."""
    tokens, t = 33, 7
    residual, fn, scale, base, weight, incoming = _inputs(tokens, seed=17)
    clean = mhc.pre(residual, fn, scale, base, weight, incoming)
    residual = residual.clone()
    residual[t, 2, 100] = NAN
    actual = mhc.pre(residual, fn, scale, base, weight, incoming)
    assert_rows_nan(actual[3], [t])
    _assert_cast(actual[5], actual[3])
    others = [r for r in range(tokens) if r != t]
    for name in ("values", "scales"):
        _assert_bytes_equal(
            getattr(actual[5], name).view(torch.uint8)[others],
            getattr(clean[5], name).view(torch.uint8)[others],
            f"{name}: ",
        )


@pytest.mark.parametrize("tokens", [1, 70, 300])
def test_cast_rows_is_per_token_cast(mhc, tokens):
    """Rows regrouped after their entry (a CED boundary's compact rows, prefill SP's
    gathered rows) are cast by per_token_cast into the same padded buffers."""
    _, state, _, _ = _post_inputs(tokens, seed=1400 + tokens)
    rows = state[:, 1].contiguous()
    _assert_cast(mhc.cast_rows(rows), rows)


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
    # FP32 precision on TF32 tensor cores: fn split into three exact TF32 parts.
    # FP32 accumulation over 20480 products stays near 3e-5; fn cut to TF32's 11
    # bits errs by about 5e-3 here, some twenty times this bound.
    reference = flat.double() @ fn.double().T
    error = (P.sum(0)[:, 0].double() - reference).abs().max().item()
    assert error <= 3e-5 * reference.abs().max().item(), error
    squares = flat.float().square().sum(-1)
    torch.testing.assert_close(S.sum(0)[:, 0], squares, atol=0, rtol=1e-5)
    collapsed = (incoming.unsqueeze(-1) * residual.float()).sum(1).bfloat16()
    _close_bf16(C, collapsed)
    torch.testing.assert_close(Q.sum(0), C.float().square().sum(-1), atol=0, rtol=1e-5)


@pytest.mark.parametrize("k", [HIDDEN, 4 * HIDDEN])
@pytest.mark.parametrize("tokens", [1, 4, 6, 16, 17, 24, 48, SIDE_ROWS])
def test_finalize_parts_give_the_whole_bits(k, tokens):
    """The rows part (``y`` and its cast) and the mixes part of ``finalize`` write the
    whole kernel's outputs bit for bit, at both widths (the broadcast entry's ``rms``
    group is one stream)."""
    residual, fn, scale, base, weight, incoming = _inputs(tokens, seed=1700 + tokens)
    flat = residual.view(tokens, 4 * HIDDEN)
    splits = HIDDEN // BLOCK_H
    P = torch.empty(splits, tokens, 1, MIXES, device=DEVICE)
    S = torch.empty(splits, tokens, 1, device=DEVICE)
    C = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    Q = torch.empty(splits, tokens, device=DEVICE)
    comb = torch.empty(tokens, 4, 4, device=DEVICE)
    project_streams(HIDDEN, "pre", block_M=16)(
        C.unsqueeze(0), flat, incoming, comb, incoming, fn, flat, P, S, C, Q
    )
    mix = (HIDDEN, k, CONFIG.rms_norm_eps, CONFIG.hc_eps, CONFIG.hc_sinkhorn_iters)

    def outputs():
        y = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
        cast = empty_mxfp8(y)
        out = {
            "post": torch.empty(tokens, 4, device=DEVICE),
            "comb": torch.empty(tokens, 4, 4, device=DEVICE),
            "pre": torch.empty(tokens, 4, device=DEVICE),
            "y": y,
            "values": cast.values,
            "scales": cast.scales,
        }
        for tensor in out.values():
            tensor.view(torch.uint8).fill_(0xFF)  # NaN in every format here
        return out

    whole, parts = outputs(), outputs()
    finalize(*mix, CONFIG.rms_norm_eps)(
        P, S, C, Q, scale, base, weight, whole["post"], whole["comb"], whole["pre"],
        whole["y"], whole["values"], whole["scales"],
    )  # fmt: skip
    finalize(*mix, CONFIG.rms_norm_eps, part="rows")(
        C, Q, weight, parts["y"], parts["values"], parts["scales"]
    )
    finalize(*mix, CONFIG.rms_norm_eps, part="mixes")(
        P, S, scale, base, parts["post"], parts["comb"], parts["pre"]
    )
    for name in whole:
        _assert_bytes_equal(parts[name], whole[name], f"{name}: ")


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


# The standalone post and the stream collapse: TileLang ports of TileKernels'
# CUDA kernels, held to their bits (BF16 bit patterns, so signed zeros count).

SERVING_TOKENS = [1, 6, 96, 384, 2048, 16384]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _assert_bits_equal(actual, expected, what=""):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    differ = _bits(actual) != _bits(expected)
    if differ.any():
        where = differ.nonzero()[:4].tolist()
        raise AssertionError(
            f"{what}{int(differ.sum())} of {differ.numel()} values differ in their "
            f"bits; first at {where}"
        )


def _tk_post(x, residual, post, comb):
    """TileKernels' ``mhc_post_fwd`` (``mhc_post_fwd_cuda_kernel``), as the forward
    called it before the port."""
    tokens, hidden = x.shape
    out = torch.empty_like(residual)
    tile_kernels.mhc.mhc_post_fwd(
        x.view(1, tokens, hidden),
        residual.view(1, tokens, 4, hidden),
        post.view(1, tokens, 4, 1),
        comb.view(1, tokens, 4, 4),
        out=out.view(1, tokens, 4, hidden),
    )
    return out


def _tk_collapse(state, mix):
    """TileKernels' ``mhc_pre_apply_mix_fwd`` (``mhc_pre_apply_mix_fwd_cuda_kernel``),
    as the forward called it before the port."""
    out = torch.empty(
        (state.shape[0], state.shape[-1]), dtype=state.dtype, device=state.device
    )
    tile_kernels.mhc.mhc_pre_apply_mix_fwd(state, mix, out)
    return out


def _post_inputs(tokens, seed):
    """A post's operands at serving magnitudes: the sublayer's BF16 output and
    streams, the post-mix in (0, 2), combination rows summing to one; the first
    eight columns are negative zeros (TileKernels keeps the sign)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(tokens, HIDDEN, generator=g, device=DEVICE, dtype=torch.bfloat16)
    residual = torch.randn(
        tokens, 4, HIDDEN, generator=g, device=DEVICE, dtype=torch.bfloat16
    )
    post = 2 * torch.sigmoid(torch.randn(tokens, 4, generator=g, device=DEVICE))
    comb = torch.softmax(torch.randn(tokens, 4, 4, generator=g, device=DEVICE), -1)
    x[:, :8] = -0.0
    residual[:, :, :8] = -0.0
    return x, residual, post, comb


def _pre_mix(tokens, seed):
    """An incoming pre-mix: sigmoid values plus ``hc_eps``."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.sigmoid(torch.randn(tokens, 4, generator=g, device=DEVICE)) + 1e-6


@pytest.mark.parametrize("tokens", SERVING_TOKENS)
def test_post_bits_match_tile_kernels(mhc, tokens):
    x, residual, post, comb = _post_inputs(tokens, seed=500 + tokens)
    _assert_bits_equal(
        mhc.post(x, residual, post, comb), _tk_post(x, residual, post, comb)
    )


@pytest.mark.parametrize("tokens", SERVING_TOKENS)
def test_collapse_bits_match_tile_kernels(mhc, tokens):
    """The weighted collapse and the mean; columns whose streams are all negative
    zeros collapse to positive zero, as TileKernels' do."""
    _, state, _, _ = _post_inputs(tokens, seed=600 + tokens)
    mix = _pre_mix(tokens, seed=700 + tokens)
    _assert_bits_equal(mhc.collapse(state, mix), _tk_collapse(state, mix), "mix: ")
    quarter = torch.full((tokens, 4), 0.25, device=DEVICE)
    _assert_bits_equal(mhc.collapse(state), _tk_collapse(state, quarter), "mean: ")
    assert (_bits(mhc.collapse(state, mix)[:, :8]) == 0).all()


@pytest.mark.parametrize("tokens", [1, 6, 96, 2048])
@pytest.mark.parametrize("column", [0, 7, 1029, HIDDEN - 1])
def test_collapse_writes_the_sink_column(mhc, tokens, column):
    """The last collapse's sink-column variant: the same collapse, bit for bit, and
    each stream's ``column`` as FP32 (the detector's buffer, written whole)."""
    _, state, _, _ = _post_inputs(tokens, seed=800 + tokens)
    state[0, 2, column] = float("nan")
    state[-1, 1, column] = -0.0
    mix = _pre_mix(tokens, seed=900 + tokens)
    columns = torch.full((tokens, 4), -7.0, device=DEVICE)
    out = mhc.collapse(state, mix, columns=columns, column=column)
    _assert_bits_equal(out, mhc.collapse(state, mix), "collapse: ")
    _assert_bits_equal(columns, state[:, :, column].float(), "columns: ")


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("tokens", [1, 6, 96, 2048])
def test_post_of_rank_order_parts_matches_tile_kernels(mhc, world, tokens):
    """The ranks' unreduced partials are summed in rank order first (RankOrderParts),
    then folded: TileKernels' post of that sum, bit for bit."""
    _, residual, post, comb = _post_inputs(tokens, seed=800 + tokens)
    g = torch.Generator(device=DEVICE).manual_seed(world * 1000 + tokens)
    parts = torch.randn(
        world, tokens, HIDDEN, generator=g, device=DEVICE, dtype=torch.bfloat16
    )
    parts[0, 0, :8] = -0.0  # a signed zero, added to nothing, survives
    summed = RankOrderParts(parts).sum()
    _assert_bits_equal(
        mhc.post(RankOrderParts(parts), residual, post, comb),
        _tk_post(summed, residual, post, comb),
    )


def _bf16(values, shape):
    return torch.tensor(values, device=DEVICE).bfloat16().expand(shape).clone()


def test_post_order_of_operations(mhc):
    """Inputs on which another order of the terms, another fused product or another
    starting value gives other bits. TileKernels' arithmetic per element (nvcc's
    contraction of ``post * x + comb[0] * residual[0] + ...``):
    ``p = comb[0, n] * residual[0]`` rounded to FP32, ``p = fma(post[n], x, p)``,
    then ``p = fma(comb[m, n], residual[m], p)`` for m = 1..3, then one rounding to
    BF16. The port gives those bits."""
    big, eps, u = 2.0**127, 2.0**-23, 2.0**-7
    x = torch.zeros(3, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    residual = torch.zeros(3, 4, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    post = torch.zeros(3, 4, device=DEVICE)
    comb = torch.zeros(3, 4, 4, device=DEVICE)
    # Token 0, output 0: big + big overflows before -big is added; stream 1 before
    # x would leave big.
    x[0], residual[0, 0], residual[0, 1] = big, big, -big
    post[0, 0], comb[0, 0, 0], comb[0, 1, 0] = 1.0, 1.0, 1.0
    # Token 1, x = -(1 + u) and stream 0 = 1 + u. Output 0: the first stream's
    # product (1 + eps)(1 + u) rounded to 1 + u + eps, then x added, gives eps; with
    # that product fused, eps (1 + u). Output 1: x's product (1 + eps) x fused
    # with the first stream's 1 + u gives -eps (1 + u); rounded first, -eps.
    x[1], residual[1, 0] = -(1 + u), 1 + u
    post[1, 0], comb[1, 0, 0] = 1.0, 1 + eps
    post[1, 1], comb[1, 0, 1] = 1 + eps, 1.0
    # Token 2: negative zeros from the first product on stay negative.
    x[2], residual[2] = -0.0, -0.0
    post[2], comb[2] = 1.0, 1.0
    expected = torch.zeros(3, 4, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    expected[0, 0] = INF
    expected[1, 0] = eps
    expected[1, 1] = -eps * (1 + u)
    expected[2] = -0.0
    ours = mhc.post(x, residual, post, comb)
    _assert_bits_equal(ours, _tk_post(x, residual, post, comb), "TileKernels: ")
    _assert_bits_equal(ours, expected, "expected: ")


def test_collapse_order_of_operations(mhc):
    """TileKernels' collapse per element: ``s = +0``, then ``s = fma(mix[m],
    state[m], s)`` for m = 0..3, then one rounding to BF16."""
    big, eps = 2.0**127, 2.0**-23
    state = torch.zeros(3, 4, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    mix = torch.zeros(3, 4, device=DEVICE)
    # Token 0: big + big overflows before -big is added.
    state[0, 0], state[0, 1], state[0, 2] = big, big, -big
    mix[0, :3] = 1.0
    # Token 1: the second stream's product fused with the first.
    state[1, 0], state[1, 1] = -(1 + 2.0**-7), 1 + 2.0**-7
    mix[1, 0], mix[1, 1] = 1.0, 1 + eps
    # Token 2: four negative-zero products added to the positive zero start.
    state[2] = -0.0
    mix[2] = 0.25
    expected = torch.zeros(3, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    expected[0] = INF
    expected[1] = eps * (1 + 2.0**-7)
    ours = mhc.collapse(state, mix)
    _assert_bits_equal(ours, _tk_collapse(state, mix), "TileKernels: ")
    _assert_bits_equal(ours, expected, "expected: ")


def test_post_matches_the_fused_entry(mhc):
    """The standalone post and a sublayer entry's fused post write the same
    streams, bit for bit."""
    for tokens in (1, 96, 300):
        residual, fn, scale, base, weight, incoming = _inputs(tokens, seed=900 + tokens)
        x, _, post, comb = _post_inputs(tokens, seed=950 + tokens)
        fused = mhc.post_pre(x, residual, post, comb, fn, scale, base, weight, incoming)
        _assert_bits_equal(mhc.post(x, residual, post, comb), fused[0], f"{tokens}: ")


def test_post_and_collapse_rows_independent_of_batch(mhc):
    x, residual, post, comb = _post_inputs(384, seed=31)
    mix = _pre_mix(384, seed=32)
    full_post = mhc.post(x, residual, post, comb)
    full_collapse = mhc.collapse(residual, mix)
    for rows in (slice(0, 1), slice(0, 6), slice(0, 96), slice(0, 97), slice(200, 201)):
        _assert_bits_equal(
            mhc.post(x[rows], residual[rows], post[rows], comb[rows]),
            full_post[rows],
            f"post {rows}: ",
        )
        _assert_bits_equal(
            mhc.collapse(residual[rows], mix[rows]),
            full_collapse[rows],
            f"collapse {rows}: ",
        )


def test_post_and_collapse_repeatable(mhc):
    x, residual, post, comb = _post_inputs(2048, seed=41)
    mix = _pre_mix(2048, seed=42)
    first_post = mhc.post(x, residual, post, comb)
    first_collapse = mhc.collapse(residual, mix)
    for _ in range(3):
        _assert_bits_equal(mhc.post(x, residual, post, comb), first_post)
        _assert_bits_equal(mhc.collapse(residual, mix), first_collapse)


def test_post_nonfinite_like_tile_kernels(mhc):
    """NaNs and infinities in any operand reach exactly TileKernels' outputs, with
    its bits; no other token moves."""
    tokens = 33
    x, residual, post, comb = _post_inputs(tokens, seed=51)
    clean = mhc.post(x, residual, post, comb)
    x[2, 100] = NAN
    residual[5, 1, 7] = INF
    residual[9, 0, 300], residual[9, 3, 300] = INF, -INF  # inf - inf
    post[12, 2] = NAN
    comb[17, 3, 1] = -INF
    residual[20, 2, 4000], comb[20, 2] = INF, 0.0  # inf * 0
    x[25, 4500] = -INF
    poisoned = [2, 5, 9, 12, 17, 20, 25]
    ours = mhc.post(x, residual, post, comb)
    theirs = _tk_post(x, residual, post, comb)
    assert_same_nonfinite(ours, theirs)
    _assert_bits_equal(ours, theirs)
    assert_other_rows_equal(ours, clean, poisoned)


def test_collapse_nonfinite_like_tile_kernels(mhc):
    tokens = 33
    _, state, _, _ = _post_inputs(tokens, seed=61)
    mix = _pre_mix(tokens, seed=62)
    clean = mhc.collapse(state, mix)
    state[3, 0, 11] = NAN
    state[8, 2, 2048] = INF
    state[14, 1, 77], state[14, 3, 77] = INF, -INF  # inf - inf
    mix[19, 1] = NAN
    state[23, 2, 5000], mix[23, 2] = -INF, 0.0  # inf * 0
    mix[29, 0] = INF
    poisoned = [3, 8, 14, 19, 23, 29]
    ours = mhc.collapse(state, mix)
    theirs = _tk_collapse(state, mix)
    assert_same_nonfinite(ours, theirs)
    _assert_bits_equal(ours, theirs)
    assert_other_rows_equal(ours, clean, poisoned)
    assert_rows_nan(ours, [19])


def test_no_tokens_launch_nothing(mhc):
    x, residual, post, comb = _post_inputs(0, seed=0)
    assert mhc.post(x, residual, post, comb).shape == (0, 4, HIDDEN)
    assert mhc.collapse(residual).shape == (0, HIDDEN)


def test_declared_calls_are_the_forwards(mhc, monkeypatch):
    """The post and the collapse are declared exactly as the forward launches them,
    and nothing is left to warm: TileKernels' kernels no longer run."""
    registry = JitWarmupRegistry(vllm_config=None)
    with registry.activate():
        mhc.declare_kernels()
    declared = prepare.declared_calls(registry)
    assert prepare.warm_declared(registry) == 0
    factories = (post_streams, collapse_streams)
    ours = [call for call in declared if call[0] in factories]
    assert ours == [(post_streams, (HIDDEN,), {}), (collapse_streams, (HIDDEN,), {})]

    def by_factory(call):
        # TileLang's JIT objects have no __name__; order by the factory itself.
        found = (i for i, factory in enumerate(factories) if factory is call[0])
        return next(found, len(factories))

    # The forward's calls, recorded as its cached lookups make them.
    launched = []
    for name in ("post_streams", "collapse_streams"):
        factory = getattr(mhc_module, name)

        def record(*args, _factory=factory, **kwargs):
            launched.append((_factory, args, kwargs))
            return _factory(*args, **kwargs)

        monkeypatch.setattr(mhc_module, name, record)
    for name in ("_post", "_collapse"):
        lookup = getattr(mhc_module, name)
        monkeypatch.setattr(mhc_module, name, functools.cache(lookup.__wrapped__))
    x, residual, post, comb = _post_inputs(6, seed=71)
    mhc.post(x, residual, post, comb)
    mhc.post(RankOrderParts(torch.stack((x, x))), residual, post, comb)
    mhc.collapse(residual, _pre_mix(6, seed=72))
    mhc.collapse(residual)
    assert sorted(launched, key=by_factory) == ours
