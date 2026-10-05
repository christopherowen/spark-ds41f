# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang GEMMs and RMSNorm against torch references."""

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")
tile_kernels = pytest.importorskip("tile_kernels")

from vllm.models.deepseek_v4_1.tilelang.gemm import (  # noqa: E402
    DECODE_ROWS,
    DECODE_TILE_ROWS,
    bf16_gemm,
    bf16_gemm_partials,
    bf16_shards,
    fp8_decode_config,
    fp8_prefill_config,
    mxfp8_gemm,
    mxfp8_gemm_decode,
    pack_scale_words,
    splitk_reduce,
)
from vllm.models.deepseek_v4_1.tilelang.norm import TileLangRMSNorm  # noqa: E402

DEVICE = "cuda"
ROWS = (1, 7, 16, 17, 32, 33, 64, 65, 200)


def _gen(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _block32_weight(n, k, seed):
    """E4M3 ``[N, K]`` with one UE8M0 scale per 32x32 block, and its FP32 values."""
    gen = _gen(seed)
    weight = (torch.randn((n, k), generator=gen, device=DEVICE) * 2).to(
        torch.float8_e4m3fn
    )
    exps = torch.randint(120, 125, (n // 32, k // 32), generator=gen, device=DEVICE).to(
        torch.uint8
    )
    scale = (
        torch.exp2(exps.float() - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)
    )
    return weight, exps, weight.float() * scale


def _mxfp8(x):
    xq, sf = tile_kernels.quant.per_token_cast(
        x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True
    )
    sf = sf.contiguous().view(torch.uint8)
    # Rows of packed exponents are padded to whole words.
    exps = sf[:, : x.shape[1] // 32]
    values = xq.float() * torch.exp2(exps.float() - 127).repeat_interleave(32, -1)
    return xq, sf, values


# K blocks of 128 and, for TP4's 576-wide shared expert rows, 192 and 64.
@pytest.mark.parametrize(
    "n, k", [(256, 512), (1536, 1024), (224, 384), (320, 576), (1152, 576)]
)
def test_mxfp8_gemm(n, k):
    weight, exps, weight_values = _block32_weight(n, k, 1)
    weight_sf = pack_scale_words(exps, rows=n)
    x = torch.randn((max(ROWS), k), generator=_gen(2), device=DEVICE).bfloat16()
    xq, sf, x_values = _mxfp8(x)
    ref = x_values @ weight_values.T
    words = sf.contiguous().view(torch.uint32)
    decode = {
        block_M: mxfp8_gemm_decode(
            n, k, **fp8_decode_config(n, k, block_M), padded_rows=True
        )
        for block_M in DECODE_TILE_ROWS
    }
    large = mxfp8_gemm(n, k, **fp8_prefill_config(n, k))
    swizzled = mxfp8_gemm(n, k, **dict(fp8_prefill_config(n, k), swizzle_panel=2))
    outs = {}
    for rows in ROWS:
        out = torch.empty((rows, n), dtype=torch.bfloat16, device=DEVICE)
        if rows <= DECODE_ROWS:
            # Decode rows read whole tiles of the padded workspace and store
            # only live rows; every tile tall enough gives the same bits.
            for block_M, kernel in decode.items():
                if rows <= block_M:
                    kernel(xq[:DECODE_ROWS], weight, words[:DECODE_ROWS], weight_sf, out)
                    outs[rows, block_M] = out.clone()
        else:
            large(xq[:rows], weight, words[:rows], weight_sf, out)
            outs[rows, "prefill"] = out.clone()
            swizzled(xq[:rows], weight, words[:rows], weight_sf, out)
            outs[rows, "swizzled"] = out.clone()
    full = outs[max(ROWS), "prefill"]
    torch.testing.assert_close(
        full.float(), ref, rtol=1e-2, atol=1e-2 * ref.abs().max().item()
    )
    # Rows agree bit for bit across batch sizes, padding and tile configurations.
    for (rows, tile), out in outs.items():
        assert torch.equal(out, full[:rows]), (
            f"rows={rows} on the {tile} tile differs from the full batch"
        )


def test_mxfp8_decode_tile_repeats(n=2048, k=2048, runs=100):
    """Narrow 16-row decode tiles give the same bits on every run.

    With TileLang's warp-specialized pipeline, this tile's cp.async scale
    staging raced on SM121 and returned wrong results; decode tiles run the
    unspecialized pipeline.
    """
    weight, exps, _ = _block32_weight(n, k, 5)
    weight_sf = pack_scale_words(exps, rows=n)
    x = torch.randn((DECODE_ROWS, k), generator=_gen(6), device=DEVICE).bfloat16()
    xq, sf, _ = _mxfp8(x)
    words = sf.contiguous().view(torch.uint32)
    tile = dict(block_M=16, block_N=32, block_K=128, num_stages=2, threads=128)
    kernel = mxfp8_gemm_decode(n, k, **tile, padded_rows=True)
    full = torch.empty((DECODE_ROWS, n), dtype=torch.bfloat16, device=DEVICE)
    mxfp8_gemm(n, k, **fp8_prefill_config(n, k))(xq, weight, words, weight_sf, full)
    want = full[:16]
    out = torch.empty_like(want)
    for run in range(runs):
        kernel(xq, weight, words, weight_sf, out)
        assert torch.equal(out, want), f"run {run} differs"


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
# The router (384 x 5120) splits K eight ways; (100, 200) cannot split.
@pytest.mark.parametrize("n, k", [(32, 4096), (384, 512), (384, 5120), (100, 200)])
def test_bf16_gemm(n, k, out_dtype):
    gen = _gen(3)
    weight = torch.randn((n, k), generator=gen, device=DEVICE).bfloat16()
    x = torch.randn((max(ROWS), k), generator=gen, device=DEVICE).bfloat16()
    tl_dtype = {torch.bfloat16: "bfloat16", torch.float32: "float32"}[out_dtype]
    shards = bf16_shards(n, k)
    gemm = bf16_gemm(n, k, out_dtype=tl_dtype, shards=shards)
    ref = x.float() @ weight.float().T
    full = torch.empty((max(ROWS), n), dtype=out_dtype, device=DEVICE)
    gemm(x, weight, full)
    torch.testing.assert_close(
        full.float(), ref, rtol=1e-2, atol=1e-2 * ref.abs().max().item()
    )
    if shards > 1:
        from vllm.models.deepseek_v4_1.tilelang.linear import PARTIAL_ROW_TILES

        partials = {
            block_M: bf16_gemm_partials(n, k, shards, block_M=block_M)
            for block_M in PARTIAL_ROW_TILES
        }
        reduce = splitk_reduce(n, shards, out_dtype=tl_dtype)
    for rows in ROWS:
        out = torch.empty((rows, n), dtype=out_dtype, device=DEVICE)
        if shards > 1 and rows <= DECODE_ROWS:
            # Decode rows: split-K partials, reduced in shard order. Every row
            # tile height gives the full batch's result bit for bit.
            p = torch.empty((shards, rows, n), dtype=torch.float32, device=DEVICE)
            for block_M, kernel in partials.items():
                kernel(x[:rows], weight, p)
                reduce(p, out)
                assert torch.equal(out, full[:rows]), (
                    f"rows={rows} with {block_M}-row tiles differs from the full batch"
                )
        else:
            gemm(x[:rows], weight, out)
            assert torch.equal(out, full[:rows]), (
                f"rows={rows} differs from the full batch"
            )


@pytest.mark.parametrize("hidden", [128, 512, 4096])
def test_rmsnorm(hidden, rows=33, eps=1e-6):
    gen = _gen(4)
    norm = TileLangRMSNorm(hidden, eps).to(DEVICE)
    norm.weight.data.copy_(torch.rand((hidden,), generator=gen, device=DEVICE) + 0.5)
    x = torch.randn((rows, hidden), generator=gen, device=DEVICE).bfloat16()
    out = norm(x)
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * norm.weight.float()
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
