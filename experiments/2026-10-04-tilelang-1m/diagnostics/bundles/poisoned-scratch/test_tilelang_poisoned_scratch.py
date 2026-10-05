# SPDX-License-Identifier: Apache-2.0
"""TileLang projections and routed experts with poisoned scratch.

Scratch comes from vLLM's workspace, which nothing clears between kernels.
Every byte a kernel reads must have been written by the step that feeds it,
so outputs must not change when the scratch starts as 0xFF (NaN as E4M3 and
as a UE8M0 scale) instead of zeros. Shapes are DeepSeek-V4.1-Flash at TP4:
the shared expert (5120 -> 1152 -> 576 -> 5120) and the drafter's routed
experts (128 experts, top 3, intermediate 576), at row counts on both sides
of the 64-row decode tile.
"""

import pytest
import torch

pytest.importorskip("tilelang")
tile_kernels = pytest.importorskip("tile_kernels")

from vllm.models.deepseek_v4_1.tilelang.gemm import (  # noqa: E402
    DECODE_ROWS,
    default_config,
    mxfp8_gemm,
    pack_scale_words,
)
from vllm.models.deepseek_v4_1.tilelang.linear import (  # noqa: E402
    _activation_specs,
    _carve,
    _spec_bytes,
)
from vllm.models.deepseek_v4_1.tilelang.moe import (  # noqa: E402
    MoeExpertsV41Plan,
    workspace_bytes,
)

DEVICE = "cuda"
ROWS = (48, 60, 64, 65, 80, 96)
POISON = (0x00, 0xFF)


def _gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _block32_linear(x, weight, weight_sf, n, k, gemms, fill):
    """vllm::dsv41_tilelang_block32_linear with scratch that starts as ``fill``."""
    rows = x.shape[0]
    specs = _activation_specs(rows, k)
    scratch = torch.full((_spec_bytes(specs),), fill, dtype=torch.uint8, device=DEVICE)
    xq, sf = _carve(scratch, specs)
    tile_kernels.quant.per_token_cast(
        x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True, out=(xq[:rows], sf[:rows])
    )
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=DEVICE)
    gemm = gemms[0 if rows <= DECODE_ROWS else 1]
    gemm(xq, weight, sf.view(torch.uint32), weight_sf, out)
    return out


@pytest.mark.parametrize("n, k", [(1152, 5120), (5120, 576), (320, 576)])
def test_mxfp8_linear_ignores_scratch(n, k):
    gen = _gen(1)
    weight = (torch.randn((n, k), generator=gen, device=DEVICE) * 2).to(
        torch.float8_e4m3fn
    )
    exps = torch.randint(120, 125, (n // 32, k // 32), generator=gen, device=DEVICE)
    weight_sf = pack_scale_words(exps.to(torch.uint8), rows=n)
    gemms = (
        mxfp8_gemm(n, k, **default_config(DECODE_ROWS, k), padded_rows=True),
        mxfp8_gemm(n, k, **default_config(DECODE_ROWS + 1, k)),
    )
    x = torch.randn((max(ROWS), k), generator=_gen(2), device=DEVICE).bfloat16()
    failures = []
    for rows in ROWS:
        outs = [
            _block32_linear(x[:rows], weight, weight_sf, n, k, gemms, fill)
            for fill in POISON
        ]
        finite = bool(torch.isfinite(outs[1].float()).all())
        if not finite or not torch.equal(outs[0], outs[1]):
            bad = (~torch.isfinite(outs[1].float())).any(-1).nonzero().flatten()
            failures.append(f"rows={rows}: finite={finite}, non-finite rows {bad.tolist()[:12]}")
    assert not failures, "; ".join(failures)


def _random_experts(num_experts, n, k, seed):
    gen = _gen(seed)
    packed = torch.randint(
        0, 256, (num_experts, n, k // 2), generator=gen, device=DEVICE, dtype=torch.uint8
    )
    exps = torch.randint(
        118, 122, (num_experts, n, k // 32), generator=gen, device=DEVICE
    ).to(torch.uint8)
    return packed, exps


@pytest.mark.parametrize(
    "num_experts, topk, hidden, intermediate",
    [(128, 3, 5120, 576), (384, 6, 5120, 576)],
)
def test_moe_experts_ignore_workspace(num_experts, topk, hidden, intermediate):
    w13_packed, w13_sf = _random_experts(num_experts, 2 * intermediate, hidden, 1)
    w2_packed, w2_sf = _random_experts(num_experts, hidden, intermediate, 2)
    plan = MoeExpertsV41Plan(
        num_experts, topk, hidden, intermediate, swiglu_limit=10.0
    )
    weights = plan.prepare_weights(w13_packed, w13_sf, w2_packed, w2_sf)
    x = (torch.randn((max(ROWS), hidden), generator=_gen(3), device=DEVICE) * 0.5).bfloat16()
    gen = _gen(4)
    ids = torch.stack(
        [torch.randperm(num_experts, generator=gen, device=DEVICE)[:topk] for _ in range(max(ROWS))]
    ).to(torch.int64)
    route_weights = torch.rand((max(ROWS), topk), generator=gen, device=DEVICE) + 0.1
    failures = []
    for rows in ROWS:
        outs = []
        for fill in POISON:
            workspace = torch.full(
                (workspace_bytes(plan.workspace_specs(rows, ids.dtype)),),
                fill,
                dtype=torch.uint8,
                device=DEVICE,
            )
            out = torch.empty((rows, hidden), dtype=torch.bfloat16, device=DEVICE)
            plan.run(x[:rows], ids[:rows], route_weights[:rows], weights, out, workspace)
            outs.append(out)
        finite = bool(torch.isfinite(outs[1].float()).all())
        if not finite or not torch.equal(outs[0], outs[1]):
            bad = (~torch.isfinite(outs[1].float())).any(-1).nonzero().flatten()
            failures.append(f"rows={rows}: finite={finite}, non-finite rows {bad.tolist()[:12]}")
    assert not failures, "; ".join(failures)
