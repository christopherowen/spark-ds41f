# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang routed experts against an FP32 torch reference."""

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
    assert_rows_nan,
)
from vllm.models.deepseek_v4_1.tilelang import prepare  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.linear import _calls  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.moe import (  # noqa: E402
    MoeDispatchV41Plan,
    MoeExpertsV41Plan,
    carve,
    moe_swiglu_cast,
    workspace_bytes,
)

# Declaring a plan warms TileKernels' token cast for the scheduler's batch buckets.
pytestmark = pytest.mark.usefixtures("default_vllm_config")

DEVICE = "cuda"
E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
LIMIT = 10.0


def _plan(*args, **kwargs):
    """A plan with its kernels declared: outside a runner, prepared and bound now."""
    plan = MoeExpertsV41Plan(*args, **kwargs)
    plan.declare_kernels()
    return plan


def _gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _experts(num_experts, n, k, seed):
    """Packed E2M1 ``[E, N, K / 2]``, UE8M0 ``[E, N, K / 32]`` and their FP32 values."""
    gen = _gen(seed)
    codes = torch.randint(0, 16, (num_experts, n, k), generator=gen, device=DEVICE)
    exps = torch.randint(
        118, 122, (num_experts, n, k // 32), generator=gen, device=DEVICE
    )
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    lut = torch.tensor(E2M1, device=DEVICE)
    values = lut[codes] * torch.exp2(exps.float() - 127).repeat_interleave(32, -1)
    return packed.contiguous(), exps.to(torch.uint8).contiguous(), values


def _routes(rows, num_experts, topk, seed, dtype):
    gen = _gen(seed)
    ids = torch.stack(
        [
            torch.randperm(num_experts, generator=gen, device=DEVICE)[:topk]
            for _ in range(rows)
        ]
    ).to(dtype)
    weights = torch.rand((rows, topk), generator=gen, device=DEVICE) + 0.1
    return ids.contiguous(), weights.float().contiguous()


def _mxfp8_values(x):
    """TileKernels' MXFP8 cast (a UE8M0 scale per 32 values), dequantized."""
    xq, sf = tile_kernels.quant.per_token_cast(
        x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True
    )
    # Rows of packed exponents are padded to whole words.
    exps = sf.contiguous().view(torch.uint8)[:, : x.shape[1] // 32]
    return xq.float() * torch.exp2(exps.float() - 127).repeat_interleave(32, -1)


def _reference(x, ids, weights, w13, w2, intermediate):
    """FP32 experts over the kernels' MXFP8 tokens; the down input stays unquantized."""
    xd = _mxfp8_values(x)
    out = torch.zeros(x.shape, dtype=torch.float32, device=DEVICE)
    for slot in range(ids.shape[1]):
        e = ids[:, slot].long()
        h = torch.einsum("tk,tnk->tn", xd, w13[e]).bfloat16().float()
        gate = h[:, :intermediate].clamp(max=LIMIT)
        up = h[:, intermediate:].clamp(-LIMIT, LIMIT)
        act = torch.nn.functional.silu(gate) * up * weights[:, slot : slot + 1]
        out += torch.einsum("tn,tkn->tk", act, w2[e])
    return out


# Down projection K blocks of 128, 192 (TP4's 576-wide rows) and 64.
@pytest.mark.parametrize("hidden, intermediate", [(512, 256), (384, 576), (384, 320)])
@pytest.mark.parametrize("rows", [1, 12, 300])
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_moe_experts(rows, idx_dtype, hidden, intermediate, num_experts=32, topk=6):
    w13_packed, w13_sf, w13 = _experts(num_experts, 2 * intermediate, hidden, 1)
    w2_packed, w2_sf, w2 = _experts(num_experts, hidden, intermediate, 2)
    plan = _plan(num_experts, topk, hidden, intermediate, swiglu_limit=LIMIT)
    weights = plan.prepare_weights(w13_packed, w13_sf, w2_packed, w2_sf)
    x = (torch.randn((rows, hidden), generator=_gen(3), device=DEVICE) * 0.5).bfloat16()
    ids, route_weights = _routes(rows, num_experts, topk, 4, idx_dtype)
    out = torch.empty_like(x)
    workspace = torch.empty(
        workspace_bytes(plan.workspace_specs(rows, idx_dtype)),
        dtype=torch.uint8,
        device=DEVICE,
    )
    plan.run(x, ids, route_weights, weights, out, workspace)
    ref = _reference(x, ids, route_weights, w13, w2, intermediate)
    scale = ref.abs().max().item()
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2 * scale)

    # A token's output does not depend on the rest of the batch.
    for r in {0, rows - 1}:
        solo = torch.empty_like(x[r : r + 1])
        plan.run(x[r : r + 1], ids[r : r + 1], route_weights[r : r + 1], weights, solo)
        assert torch.equal(solo[0], out[r]), f"row {r} changed with batch composition"


@pytest.mark.parametrize("rows", [1, 12, 300])
def test_moe_experts_fp32_output(
    rows, hidden=384, intermediate=576, num_experts=32, topk=6
):
    """An FP32 combine keeps the slot-order FP32 sum the BF16 combine rounds: the
    same sum, unrounded, and as independent of the batch."""
    w13_packed, w13_sf, _ = _experts(num_experts, 2 * intermediate, hidden, 1)
    w2_packed, w2_sf, _ = _experts(num_experts, hidden, intermediate, 2)
    plans = {
        dtype: _plan(
            num_experts, topk, hidden, intermediate, swiglu_limit=LIMIT, out_dtype=dtype
        )
        for dtype in (torch.bfloat16, torch.float32)
    }
    weights = plans[torch.bfloat16].prepare_weights(
        w13_packed, w13_sf, w2_packed, w2_sf
    )
    x = (torch.randn((rows, hidden), generator=_gen(3), device=DEVICE) * 0.5).bfloat16()
    ids, route_weights = _routes(rows, num_experts, topk, 4, torch.int32)
    outs = {}
    for dtype, plan in plans.items():
        assert plan.out_dtype == dtype
        outs[dtype] = torch.empty((rows, hidden), dtype=dtype, device=DEVICE)
        plan.run(x, ids, route_weights, weights, outs[dtype])
    assert torch.equal(outs[torch.float32].bfloat16(), outs[torch.bfloat16])
    assert not torch.equal(outs[torch.float32], outs[torch.bfloat16].float())
    for r in {0, rows - 1}:
        solo = torch.empty((1, hidden), dtype=torch.float32, device=DEVICE)
        plans[torch.float32].run(
            x[r : r + 1], ids[r : r + 1], route_weights[r : r + 1], weights, solo
        )
        assert torch.equal(solo[0], outs[torch.float32][r])


@pytest.mark.parametrize("rows", [1, 12, 300])
def test_moe_experts_fold_shared(
    rows, hidden=384, intermediate=576, num_experts=32, topk=6
):
    """The FP32 plan's combine with the shared expert's output folded in (BF16 out)
    gives the runner's bits: the FP32 partial plus the shared output in FP32, rounded
    once, as the all-reduce path and the reduce-scatter buffer compute it. A row's
    result does not depend on which of them a step takes."""
    w13_packed, w13_sf, _ = _experts(num_experts, 2 * intermediate, hidden, 1)
    w2_packed, w2_sf, _ = _experts(num_experts, hidden, intermediate, 2)
    plan = _plan(
        num_experts, topk, hidden, intermediate, swiglu_limit=LIMIT,
        out_dtype=torch.float32,
    )  # fmt: skip
    weights = plan.prepare_weights(w13_packed, w13_sf, w2_packed, w2_sf)
    x = (torch.randn((rows, hidden), generator=_gen(3), device=DEVICE) * 0.5).bfloat16()
    shared = (torch.randn((rows, hidden), generator=_gen(5), device=DEVICE)).bfloat16()
    ids, route_weights = _routes(rows, num_experts, topk, 4, torch.int32)
    wide = torch.empty((rows, hidden), dtype=torch.float32, device=DEVICE)
    plan.run(x, ids, route_weights, weights, wide)
    folded = torch.empty((rows, hidden), dtype=torch.bfloat16, device=DEVICE)
    plan.run(x, ids, route_weights, weights, folded, shared=shared)
    assert torch.equal(folded, (shared + wide).to(torch.bfloat16))  # all-reduce path
    buffer = torch.empty_like(folded)
    torch.add(shared, wide, out=buffer)  # reduce-scatter buffer
    assert torch.equal(folded, buffer)
    with pytest.raises(AssertionError):
        plan.run(x, ids, route_weights, weights, folded)  # BF16 out needs the shared


@pytest.mark.parametrize("rows", [1, 12, 96])
def test_moe_experts_fold_deferred_shared(
    rows, hidden=384, intermediate=576, num_experts=32, topk=6
):
    """A callable shared addend, as the runner passes for a shared expert on the aux
    stream: the plan calls it once, just before its combine, and the folded output
    has the bits of the shared rows passed directly."""
    w13_packed, w13_sf, _ = _experts(num_experts, 2 * intermediate, hidden, 1)
    w2_packed, w2_sf, _ = _experts(num_experts, hidden, intermediate, 2)
    plan = _plan(
        num_experts, topk, hidden, intermediate, swiglu_limit=LIMIT,
        out_dtype=torch.float32,
    )  # fmt: skip
    weights = plan.prepare_weights(w13_packed, w13_sf, w2_packed, w2_sf)
    x = (torch.randn((rows, hidden), generator=_gen(3), device=DEVICE) * 0.5).bfloat16()
    source = torch.randn((rows, hidden), generator=_gen(5), device=DEVICE)
    ids, route_weights = _routes(rows, num_experts, topk, 4, torch.int32)
    direct = torch.empty((rows, hidden), dtype=torch.bfloat16, device=DEVICE)
    plan.run(x, ids, route_weights, weights, direct, shared=source.bfloat16())

    side, ready = torch.cuda.Stream(), torch.cuda.Event()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        shared = source.bfloat16()
        ready.record(side)
    calls = []

    def resolve():
        calls.append(1)
        ready.wait(torch.cuda.current_stream())
        return shared

    deferred = torch.empty_like(direct)
    plan.run(x, ids, route_weights, weights, deferred, shared=resolve)
    assert calls == [1]
    assert torch.equal(deferred, direct)


def _case(rows, hidden=384, intermediate=576, num_experts=32, topk=6):
    w13_packed, w13_sf, w13 = _experts(num_experts, 2 * intermediate, hidden, 1)
    w2_packed, w2_sf, w2 = _experts(num_experts, hidden, intermediate, 2)
    plan = _plan(num_experts, topk, hidden, intermediate, swiglu_limit=LIMIT)
    weights = plan.prepare_weights(w13_packed, w13_sf, w2_packed, w2_sf)
    x = (torch.randn((rows, hidden), generator=_gen(3), device=DEVICE) * 0.5).bfloat16()
    ids, route_weights = _routes(rows, num_experts, topk, 4, torch.int32)

    def run(tokens):
        out = torch.empty_like(tokens)
        plan.run(tokens, ids, route_weights, weights, out)
        return out

    return run, x, ids, route_weights, w13, w2, intermediate


@pytest.mark.parametrize("rows", [12, 300])
def test_moe_experts_nan_token_stays_in_its_row(rows):
    """A NaN in one token's input changes no other token's bits, though the expert
    tiles mix tokens."""
    run, x, *_ = _case(rows)
    clean = run(x)
    x[5, 17] = NAN
    assert_other_rows_equal(run(x), clean, [5])


def test_moe_experts_nan_token_is_nan(rows=12):
    """A NaN in a token's input reaches every output of that token, as in DeepSeek's
    ``Expert.forward``."""
    run, x, *_ = _case(rows)
    x[5, 17] = NAN
    assert_rows_nan(run(x), [5])


def _arithmetic(x, ids, weights, w13, w2, intermediate, defect=None):
    """The plan's arithmetic in torch, optionally with one injected defect.

    MXFP8 tokens; the gate/up output rounded to BF16; the clamped SwiGLU times the
    route weight in FP32, cast to MXFP8 straight from FP32 (TileKernels' fused
    cast); the down output rounded to BF16; the routes summed in slot order in
    FP32 and rounded once.
    """
    xd = _mxfp8_values(x)
    if defect == "skipped K tile":  # gate/up never loads its first 128 columns
        xd[:, :128] = 0
    lowest = weights.argmin(1)
    out = torch.zeros(x.shape, dtype=torch.float32, device=DEVICE)
    for slot in range(ids.shape[1]):
        e = ids[:, slot].long()
        if defect == "wrong expert" and slot == 0:
            e = (e + 1) % w13.shape[0]
        h = torch.einsum("tk,tnk->tn", xd, w13[e]).bfloat16().float()
        gate = h[:, :intermediate].clamp(max=LIMIT)
        up = h[:, intermediate:].clamp(-LIMIT, LIMIT)
        w = weights[:, slot : slot + 1]
        if defect == "unweighted":
            w = torch.ones_like(w)
        act = _mxfp8_values(gate / (1 + torch.exp(-gate)) * up * w)
        y = torch.einsum("tn,tkn->tk", act, w2[e]).bfloat16().float()
        if defect == "dropped route":  # each token's lowest-weight route
            y = torch.where((lowest == slot)[:, None], 0.0, y)
        out = out + y
        if defect == "BF16 combine":
            out = out.bfloat16().float()
    return out.bfloat16()


DEFECTS = (
    "dropped route",
    "wrong expert",
    "unweighted",
    "skipped K tile",
    "BF16 combine",
)


def _error(a, b):
    """Mean absolute difference relative to the mean magnitude, in FP64."""
    a, b = a.double(), b.double()
    return ((a - b).abs().mean() / b.abs().mean()).item()


@pytest.mark.parametrize("rows", [12, 300])
def test_moe_reference_separates_defects(rows):
    """Calibration of the MoE check (arXiv 2608.12700, section 3.3): a reference must
    put the kernel well inside every defect it is meant to catch. The kernel must
    be at least four times closer to the reference that follows its arithmetic than
    each injected defect. ``test_moe_experts``'s FP32 reference leaves the down
    input unquantized, so the kernel already differs from it by MXFP8 rounding,
    more than a BF16 combine adds; the error table shows both."""
    run, x, ids, route_weights, w13, w2, intermediate = _case(rows)
    out = run(x)
    exact = _arithmetic(x, ids, route_weights, w13, w2, intermediate)
    loose = _reference(x, ids, route_weights, w13, w2, intermediate)
    kernel = _error(out, exact)
    defects = {
        d: _error(_arithmetic(x, ids, route_weights, w13, w2, intermediate, d), exact)
        for d in DEFECTS
    }
    table = "; ".join(
        [f"kernel {kernel:.2e} (unquantized-down reference {_error(out, loose):.2e})"]
        + [f"{d} {e:.2e}" for d, e in defects.items()]
    )
    print(f"rows={rows}: {table}")
    for d, e in defects.items():
        assert 4 * kernel < e, f"{d} is within 4x of the kernel: {table}"


def _tile_kernels_swiglu(intermediate, num_experts, block_M):
    """TileKernels' ``swiglu_forward_and_per_token_cast`` kernel as vLLM called it
    until ``moe_swiglu_cast`` (the image's TileKernels carries spark-ds41f's patch
    0001: the clamp keeps a NaN)."""
    from tile_kernels.quant.common import get_cast_output_config
    from tile_kernels.quant.swiglu_forward_cuda import (
        get_swiglu_forward_and_per_token_cast_kernel_cuda,
    )
    from tilelang import language as T

    tile_kernels.config.set_token_alignment(block_M)
    return get_swiglu_forward_and_per_token_cast_kernel_cuda(
        intermediate,
        num_experts,
        block_M,
        True,
        False,
        LIMIT,
        False,
        in_dtype=T.dtype(torch.bfloat16),
        out_config=get_cast_output_config("e4m3", (1, 32), False, True, True),
        num_sms=tile_kernels.config.get_num_sms(),
        use_pdl=False,
    )


def _swiglu_case(rows, intermediate, num_experts, topk, block_M=32):
    """A real dispatch layout of ``rows`` tokens and a gate/up output over it whose
    values reach the clamp, the certificate's fallback (gates below -16) and the
    smallest scales."""
    dispatch = MoeDispatchV41Plan(
        num_experts, topk, block_M, 512, torch.int32, torch.bfloat16
    )
    prepare.declare(lambda: _calls(dispatch.kernels()), bind=dispatch.bind)
    ids, weights = _routes(rows, num_experts, topk, 11, torch.int32)
    specs = dispatch.workspace_specs(rows)
    layout = dispatch.dispatch(
        ids,
        weights,
        carve(
            torch.empty(workspace_bytes(specs), dtype=torch.uint8, device=DEVICE), specs
        ),
    )
    layout_rows = layout["order"].shape[0]
    gen = _gen(12)
    h = torch.randn((layout_rows, 2 * intermediate), generator=gen, device=DEVICE) * 6
    h[::7, : intermediate // 4] = -20.0  # gates below the fast exp's domain
    h[3::11, 64:96] *= 1e-30  # blocks at the smallest scales
    return layout, h.bfloat16().contiguous()


def _swiglu_outputs(layout_rows, intermediate):
    out = torch.zeros(
        (layout_rows, intermediate), dtype=torch.float8_e4m3fn, device=DEVICE
    )
    sf = torch.zeros(
        (layout_rows, (intermediate // 32 + 3) // 4 * 4),
        dtype=torch.uint8,
        device=DEVICE,
    )
    return out, sf


@pytest.mark.parametrize("intermediate", [576, 768, 256])
@pytest.mark.parametrize("rows", [1, 6, 16, 96, 700])
def test_moe_swiglu_cast_equals_tile_kernels(
    rows, intermediate, num_experts=384, topk=6
):
    """Every live row's E4M3 bytes and UE8M0 scales equal TileKernels' (TP4's 576 and
    TP3's 768 columns a rank, decode and prefill sizes), and a NaN gate value reaches
    its row's output in both."""
    layout, h = _swiglu_case(rows, intermediate, num_experts, topk)
    live = layout["order"] >= 0
    ours, theirs = (
        moe_swiglu_cast(intermediate, 32, LIMIT),
        _tile_kernels_swiglu(intermediate, num_experts, 32),
    )
    for case in ("finite", "nan"):
        if case == "nan":
            first = int(torch.nonzero(live)[0])
            h[first, 5] = NAN
        a, a_sf = _swiglu_outputs(h.shape[0], intermediate)
        b, b_sf = _swiglu_outputs(h.shape[0], intermediate)
        ours(
            h, a, a_sf, layout["order"], layout["layout_weights"], layout["tile_expert"]
        )
        theirs(h, b, b_sf, layout["psum"], layout["layout_weights"], 1.0, None)
        cols = intermediate // 32
        assert torch.equal(a[live].view(torch.uint8), b[live].view(torch.uint8)), case
        assert torch.equal(a_sf[live, :cols], b_sf[live, :cols]), case
        if case == "nan":
            assert a[first].float().isnan().any()
