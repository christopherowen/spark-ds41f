# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek-V4.1 post-load weight preparation, on-disk cache, and startup optimizations."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

from vllm.models.deepseek_v4_1.tilelang.gemm import (
    TILE_WEIGHT_K,
    TILE_WEIGHT_N,
    pack_weight_tiles,
    unpack_weight_tiles,
)
from vllm.models.deepseek_v4_1.tilelang.scale_codes import (
    _pack_crumbs,
    _pack_nibbles,
    encode_expert_scale_codes,
    narrow_code_offsets,
    overlay_byte_codes,
    pack_expert_scale_codes,
    tile_byte_codes,
)


def _legacy_encode_expert_scale_codes(
    exponents: torch.Tensor, block_N: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = pack_expert_scale_codes(exponents)
    e, n, words = packed.codes.shape
    tiled_4bit = (
        packed.codes.view(torch.int16)
        .view(e, n // block_N, block_N, words)
        .permute(0, 1, 3, 2)
        .reshape(-1, block_N)
        .contiguous()
        .view(torch.uint16)
    )
    tiled_2bit = tile_byte_codes(packed.codes_2bit, block_N)
    codes, codes_2bit = overlay_byte_codes(tiled_4bit, tiled_2bit, packed.narrow)
    return codes, codes_2bit, narrow_code_offsets(packed.narrow), packed.base


def test_swar_pack_crumbs_and_nibbles_exhaustive():
    vals4 = torch.arange(4, dtype=torch.uint8)
    grid4 = torch.cartesian_prod(vals4, vals4, vals4, vals4).contiguous()
    ref_crumbs = (
        grid4[:, 0] | (grid4[:, 1] << 2) | (grid4[:, 2] << 4) | (grid4[:, 3] << 6)
    ).to(torch.uint8)
    assert torch.equal(_pack_crumbs(grid4), ref_crumbs)

    vals16 = torch.arange(16, dtype=torch.uint8)
    grid16 = torch.cartesian_prod(vals16, vals16, vals16, vals16).contiguous()
    lo = (grid16[:, 0] | (grid16[:, 1] << 4)).to(torch.int16)
    hi = (grid16[:, 2] | (grid16[:, 3] << 4)).to(torch.int16)
    ref_nibbles = (lo | (hi << 8)).view(torch.uint16)
    assert torch.equal(_pack_nibbles(grid16), ref_nibbles)


@pytest.mark.parametrize("mode", ["all_narrow", "mixed", "all_wide", "zero_padded"])
def test_encode_expert_scale_codes_matches_legacy(mode: str):
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e, n, k_scales = 32, 256, 56
    base = torch.randint(115, 128, (e, n, 1), dtype=torch.uint8, device=device)
    if mode == "all_narrow":
        deltas = torch.randint(0, 4, (e, n, k_scales), dtype=torch.uint8, device=device)
        scales = base + deltas
    elif mode == "all_wide":
        deltas = torch.randint(0, 12, (e, n, k_scales), dtype=torch.uint8, device=device)
        deltas[:, 0, 0] = 9
        scales = base + deltas
    elif mode == "mixed":
        deltas = torch.randint(0, 4, (e, n, k_scales), dtype=torch.uint8, device=device)
        deltas[::3, 0, 0] = 11
        scales = base + deltas
    else:
        deltas = torch.randint(0, 4, (e, n, k_scales), dtype=torch.uint8, device=device)
        deltas[5, 0, 0] = 8
        scales = base + deltas
        scales[:, :, -8:] = 0

    ref_codes, ref_2bit, ref_narrow, ref_base = _legacy_encode_expert_scale_codes(
        scales, 128
    )
    codes, codes_2bit, narrow, out_base = encode_expert_scale_codes(scales, 128)
    assert torch.equal(codes, ref_codes)
    assert torch.equal(codes_2bit, ref_2bit)
    assert torch.equal(narrow, ref_narrow)
    assert torch.equal(out_base, ref_base)


def test_pack_weight_tiles_and_direct_prefix_slice():
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, q_rows, k = 1024, 640, 512
    raw = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    packed = pack_weight_tiles(raw)
    ref_packed = (
        raw.reshape(n // TILE_WEIGHT_N, TILE_WEIGHT_N, k // TILE_WEIGHT_K, TILE_WEIGHT_K)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(n * (k // TILE_WEIGHT_K), TILE_WEIGHT_K)
    )
    assert torch.equal(packed.view(torch.uint8), ref_packed.view(torch.uint8))
    unpacked = unpack_weight_tiles(packed, n, k)
    assert torch.equal(unpacked.view(torch.uint8), raw.view(torch.uint8))

    # Direct packed slice for trimmed group rows (TP3 rank 2 wq_b optimization)
    direct_slice = packed[: q_rows * (k // TILE_WEIGHT_K)].contiguous()
    repacked_slice = pack_weight_tiles(unpacked[:q_rows].contiguous())
    assert torch.equal(direct_slice.view(torch.uint8), repacked_slice.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for TileLang")
def test_pack_bf16_weight_in_place_direct_rows_and_tile_expert_weights():
    from vllm.models.deepseek_v4_1.tilelang.moe import FP4_TMA_K, tile_expert_weights
    from vllm.models.deepseek_v4_1.tilelang.vocab import (
        pack_bf16_weight,
        unpack_bf16_weight,
    )

    torch.manual_seed(19)
    # > 3072 rows exercises the overlap-safe direct write tail in pack_bf16_weight(in_place=True).
    w = torch.randn(4352, 256, device="cuda", dtype=torch.bfloat16)
    w[0, 3] = 1e-38
    w[1024, 17] = -1e-37
    w[3072, 129] = 1e35
    w[4351, 255] = -1e35

    ref = pack_bf16_weight(w.clone(), in_place=False)
    inp = pack_bf16_weight(w.clone(), in_place=True)
    assert inp.exponent_base == ref.exponent_base
    assert inp.exceptions == ref.exceptions
    assert torch.equal(inp.values, ref.values)
    assert torch.equal(inp.row_offsets, ref.row_offsets)
    assert torch.equal(inp.exception_rows, ref.exception_rows)
    assert torch.equal(inp.exception_entries, ref.exception_entries)
    assert torch.equal(
        unpack_bf16_weight(inp).view(torch.int16), w.view(torch.int16)
    )

    # tile_expert_weights with and without 64-column tail
    for half_k in (256, 288):
        b_packed = torch.randint(
            -128, 127, (16, 256, half_k), dtype=torch.int8, device="cuda"
        )
        tiles, tails = tile_expert_weights(b_packed, 128)
        k_full, tail = divmod(2 * half_k, FP4_TMA_K)
        b_view = b_packed.view(16, 2, 128, half_k)
        ref_tiles = (
            b_view[..., : k_full * (FP4_TMA_K // 2)]
            .reshape(16, 2, 128, k_full, FP4_TMA_K // 2)
            .permute(0, 1, 3, 2, 4)
            .reshape(-1, 128, FP4_TMA_K // 2)
            .contiguous()
        )
        assert torch.equal(tiles, ref_tiles)
        if tail:
            ref_tails = (
                b_view[..., k_full * (FP4_TMA_K // 2) :]
                .reshape(-1, 128, tail // 2)
                .contiguous()
                .view(torch.uint8)
            )
            assert torch.equal(tails, ref_tails)
        else:
            assert tails is None


def test_lazy_deepseek_v41_resolution_avoids_eager_dspark_and_model_import():
    code = """
import sys
import vllm.models.deepseek_v41 as ds41
cls = ds41.DeepseekV41ForCausalLM
assert "vllm.models.deepseek_v4_1.nvidia.dspark" not in sys.modules
assert "vllm.models.deepseek_v4_1.nvidia.model" not in sys.modules
assert "vllm.models.deepseek_v41.nvidia.model" not in sys.modules
"""
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
