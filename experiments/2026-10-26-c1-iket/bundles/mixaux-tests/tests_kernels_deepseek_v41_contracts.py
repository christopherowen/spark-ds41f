# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tolerance-free contracts for the DeepSeek-V4.1 kernels: non-finite values.

DeepSeek's reference lets a NaN or an infinity through (``torch.clamp``, ``relu_``,
``softmax``), so the error that produced it stays visible. A kernel that turns one
into an ordinary number hides that error; among generated kernels that a loose
``allclose`` accepted, it is the most common defect (Shah and Shrestha, "A
Contract-Grade Verifier for LLM-Generated GPU Kernels", arXiv 2608.12700). These
checks compare positions and bits, so no tolerance can explain a failure away:

- a non-finite input gives non-finite outputs in exactly the reference's positions,
  NaN for NaN and each infinity with its sign;
- a poisoned row changes no other row's bits.

It also holds the references several kernels' tests share: the reference's
``act_quant`` values, the RoPE table and WO-A's input.
"""

import torch

NAN, INF = float("nan"), float("inf")

_KINDS = (("NaN", torch.isnan), ("+inf", torch.isposinf), ("-inf", torch.isneginf))


def assert_same_nonfinite(got: torch.Tensor, want: torch.Tensor, what: str = ""):
    """NaNs and signed infinities in exactly ``want``'s positions."""
    assert got.shape == want.shape, (got.shape, want.shape)
    got, want = got.float(), want.float()
    for name, test in _KINDS:
        g, w = test(got), test(want)
        if not torch.equal(g, w):
            where = (g != w).nonzero()[:4].tolist()
            raise AssertionError(
                f"{what}{name} in {int(g.sum())} positions, the reference in "
                f"{int(w.sum())}; first differences at {where}"
            )


def assert_finite_equal(got: torch.Tensor, want: torch.Tensor, what: str = ""):
    """Equal wherever ``want`` is finite (bit equality, as ``torch.equal``)."""
    finite = torch.isfinite(want.float())
    assert torch.equal(got[finite], want[finite]), f"{what}finite values differ"


def assert_rows_nan(out: torch.Tensor, rows, what: str = ""):
    """Every value of ``out[rows]`` is NaN."""
    block = out[rows].float()
    bad = int((~block.isnan()).sum())
    assert bad == 0, f"{what}rows {rows}: {bad} of {block.numel()} values are not NaN"


def assert_other_rows_equal(out: torch.Tensor, clean: torch.Tensor, rows, what=""):
    """Rows outside ``rows`` keep the bits of the unpoisoned run."""
    keep = torch.ones(out.shape[0], dtype=torch.bool, device=out.device)
    keep[rows] = False
    if not torch.equal(out[keep], clean[keep]):
        changed = (out[keep] != clean[keep]).flatten(1).any(1).nonzero().flatten()
        others = keep.nonzero().flatten()[changed][:8].tolist()
        raise AssertionError(f"{what}poisoning rows {rows} changed rows {others}")


def mxfp8_values(x: torch.Tensor) -> torch.Tensor:
    """DeepSeek's ``act_quant(x, 32, "ue8m0")`` values of the rows of ``x``, in FP64:
    TileKernels' ``per_token_cast`` (byte-identical to ``act_quant``), E4M3 times
    2^e."""
    import tile_kernels

    xq, sf = tile_kernels.quant.per_token_cast(
        x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True
    )
    exps = sf.contiguous().view(torch.uint8)[:, : x.shape[1] // 32].double()
    return xq.double() * torch.exp2(exps - 127).repeat_interleave(32, -1)


def rope_table(positions: int = 4096, rope_dim: int = 64) -> torch.Tensor:
    """A V4.1 RoPE table on the GPU: FP32 rows of ``cos | sin`` of each position times
    ``10000^(-2i / rope_dim)``."""
    exponents = torch.arange(0, rope_dim, 2, device="cuda").double() / rope_dim
    angles = torch.arange(positions, device="cuda").double()[:, None] * (
        1.0 / 10000**exponents
    )
    return torch.cat((angles.cos(), angles.sin()), dim=-1).float()


def wo_input_reference(
    o: torch.Tensor, positions: torch.Tensor, table: torch.Tensor
) -> torch.Tensor:
    """WO-A's input from attention output ``o`` (``[rows, heads, 512]`` BF16) as the
    reference prepares it: the trailing RoPE columns rotated back (TileKernels'
    ``apply_rotary``, conjugate), then ``act_quant``'s values, in BF16."""
    import tile_kernels

    rows, heads, dim = o.shape
    rotated = o.clone()
    tile_kernels.transform.apply_rotary(
        rotated[..., -table.shape[1] :],
        table,
        positions=positions,
        interleaved=True,
        conjugate=True,
    )
    return mxfp8_values(rotated.view(rows, -1)).bfloat16().view(rows, heads, dim)
