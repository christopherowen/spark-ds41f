# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang vocabulary heads: the exact packed BF16 head and BF16
heads, against FP32 references, with logits independent of the batch."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")
packed_format = pytest.importorskip("b12x.gemm.packed_bf16_vocab_projection")

from vllm.models.deepseek_v4_1.tilelang.vocab import (  # noqa: E402
    ROW_TILES,
    packed_vocab_projection,
    prepare_bf16_head,
    prepare_packed_head,
    project_head,
    row_tile,
    tile_config,
)

DEVICE = "cuda"
SHAPES = [(1000, 512), (32320, 5120)]


def _head(n, k, seed):
    """Head-like BF16 weights with values outside any exponent window, and zeros."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    weight = (torch.randn(n, k, generator=generator, device=DEVICE) * 0.02).bfloat16()
    flat = weight.view(-1)
    picks = torch.randint(0, flat.numel(), (64,), generator=generator, device=DEVICE)
    flat[picks[:32]] = 300.0
    flat[picks[32:48]] = -1.5e-7
    flat[picks[48:]] = 0.0
    return weight


def _project(packed, n, k, x, block_M, **config):
    kernel = packed_vocab_projection(
        n,
        k,
        packed.exponent_base,
        block_M=block_M,
        **(config or tile_config(block_M, k)),
    )
    out = torch.empty(x.shape[0], n, dtype=torch.bfloat16, device=DEVICE)
    kernel(
        x,
        packed.values,
        packed.exception_rows,
        packed.exception_entries,
        packed.row_offsets,
        out,
    )
    return out


def _close_to_fp32(out, x, weight):
    reference = x.float() @ weight.float().T
    error = (out.float() - reference).abs()
    bound = reference.abs() * 2.0**-7 + 1e-3 * reference.abs().max()
    return bool((error <= bound).all())


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("rows", [1, 5, 16, 33, 64, 100])
def test_packed_head_matches_fp32_reference(n, k, rows):
    weight = _head(n, k, seed=n + k)
    packed = packed_format.pack(weight.clone())
    assert packed.exceptions > 0
    x = (torch.randn(rows, k, device=DEVICE) * 0.5).bfloat16()
    assert _close_to_fp32(_project(packed, n, k, x, row_tile(rows)), x, weight)


@pytest.mark.parametrize("n,k", SHAPES)
def test_packed_head_rows_independent_of_batch_and_tile(n, k):
    weight = _head(n, k, seed=7)
    packed = packed_format.pack(weight.clone())
    x = (torch.randn(64, k, device=DEVICE) * 0.5).bfloat16()
    full = _project(packed, n, k, x, 64)
    for block_M in ROW_TILES:
        assert torch.equal(_project(packed, n, k, x, block_M), full), block_M
    for rows in (1, 2, 6, 17, 40):
        part = _project(packed, n, k, x[:rows].contiguous(), row_tile(rows))
        assert torch.equal(part, full[:rows]), rows
    assert torch.equal(_project(packed, n, k, x[13:14].contiguous(), 16), full[13:14])
    # Other tiles, step widths and pipeline depths give the same bits.
    for block_M, config in (
        (16, dict(block_N=32, num_stages=2, step_groups=1)),
        (16, dict(block_N=16, threads=64, num_stages=4, step_groups=2)),
        (64, dict(block_N=32, threads=128, num_stages=3)),
    ):
        rows = 9 if block_M == 16 else 64
        part = _project(packed, n, k, x[:rows].contiguous(), block_M, **config)
        assert torch.equal(part, full[:rows]), config


def test_packed_head_exceptions_are_exact():
    n, k = 512, 256
    weight = torch.full((n, k), 0.01, dtype=torch.bfloat16, device=DEVICE)
    weight[5, 17] = 1.0e30  # far outside any window holding 0.01
    packed = packed_format.pack(weight.clone())
    assert packed.exceptions == 1
    x = torch.zeros(3, k, dtype=torch.bfloat16, device=DEVICE)
    x[:, 17] = 1.0
    out = _project(packed, n, k, x, 16)
    assert torch.equal(out[:, 5], weight[5, 17].expand(3))
    assert torch.equal(out[:, 4], weight[4, 17].expand(3))


def test_packed_head_without_exceptions():
    n, k = 640, 384
    weight = (torch.rand(n, k, device=DEVICE) * 0.5 + 0.5).bfloat16()
    packed = packed_format.pack(weight.clone())
    assert packed.exceptions == 0
    x = (torch.randn(7, k, device=DEVICE) * 0.5).bfloat16()
    assert _close_to_fp32(_project(packed, n, k, x, 16), x, weight)


def _packed_layer(weight):
    packed = packed_format.pack(weight.clone())
    return SimpleNamespace(
        weight=packed.values,
        weight_exception_rows=packed.exception_rows,
        weight_exception_entries=packed.exception_entries,
        weight_row_offsets=packed.row_offsets,
        packed_bf16_in_features=weight.shape[1],
        packed_bf16_exponent_base=packed.exponent_base,
    )


@pytest.mark.parametrize("rows", [1, 6, 48, 130])
def test_prepared_packed_head(rows):
    n, k = 1000, 512
    weight = _head(n, k, seed=3)
    layer = _packed_layer(weight)
    prepare_packed_head(layer)
    assert set(layer.tilelang_vocab) == set(ROW_TILES)
    x = (torch.randn(rows, k, device=DEVICE) * 0.5).bfloat16()
    logits = project_head(layer, x)
    assert logits.shape == (rows, n) and logits.dtype == torch.bfloat16
    assert _close_to_fp32(logits, x, weight)


@pytest.mark.parametrize("n", [32320, 43092])
def test_bf16_head_matches_reference_and_is_batch_invariant(n):
    k = 256
    generator = torch.Generator(device=DEVICE).manual_seed(n)
    weight = (torch.randn(n, k, generator=generator, device=DEVICE) * 0.05).bfloat16()
    layer = SimpleNamespace(weight=weight)
    prepare_bf16_head(layer)
    x = torch.randn(130, k, device=DEVICE).bfloat16()
    full = project_head(layer, x)
    assert _close_to_fp32(full, x, weight)
    for rows in (1, 5, 16, 17, 33, 64):
        assert torch.equal(project_head(layer, x[:rows]), full[:rows]), rows
    assert torch.equal(project_head(layer, x[70:71]), full[70:71])


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n", [32320, 43136])
def test_markov_head_runs_its_16_row_tile_on_the_rows_gemm(n, out_dtype, k=256):
    """A wide, short BF16 head (the DSpark Markov bias projection at a TP4 or TP3
    vocabulary shard) runs up to 16 rows on the persistent few-row GEMM, with the
    bits of the BF16 head's own 16-row tile; more rows take the usual tiles."""
    from vllm.models.deepseek_v4_1.tilelang.gemm import bf16_gemm
    from vllm.models.deepseek_v4_1.tilelang.vocab import _OUT, BF16_HEAD_CONFIGS

    weight = _head(n, k, seed=31)
    layer = SimpleNamespace(weight=weight)
    prepare_bf16_head(layer, out_dtype)
    tile = bf16_gemm(
        n,
        k,
        block_M=16,
        num_stages=2,
        out_dtype=_OUT[out_dtype],
        **BF16_HEAD_CONFIGS[16],
    )
    generator = torch.Generator(device=DEVICE).manual_seed(32)
    x = torch.randn(20, k, generator=generator, device=DEVICE).bfloat16()
    for rows in (1, 2, 5, 8, 16):
        want = torch.empty(rows, n, dtype=out_dtype, device=DEVICE)
        tile(x[:rows], weight, want)
        assert torch.equal(project_head(layer, x[:rows]), want), rows
    full = project_head(layer, x)  # 20 rows: the 32-row tile
    assert torch.equal(project_head(layer, x[:16]), full[:16])


def test_logits_processor_projects_prepared_heads():
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.head_dtype = None
    processor.use_tilelang_vocab_projection = True
    n, k = 1000, 512
    layer = _packed_layer(_head(n, k, seed=11))
    prepare_packed_head(layer)
    hidden = (torch.randn(2, 3, k, device=DEVICE) * 0.5).bfloat16()
    logits = processor._apply_head(layer, hidden, None)
    expected = project_head(layer, hidden.reshape(-1, k)).reshape(2, 3, n)
    assert torch.equal(logits, expected)


@pytest.mark.parametrize("rows", [1, 6, 48, 130])
def test_packed_head_writes_fp32_logits(rows):
    """FP32 logits are the BF16 head's accumulator unrounded: rounding them gives the
    BF16 head's bits, and they sit closer to an FP64 reference."""
    n, k = 1000, 512
    weight = _head(n, k, seed=5)
    bf16, fp32 = _packed_layer(weight), _packed_layer(weight)
    prepare_packed_head(bf16)
    prepare_packed_head(fp32, torch.float32)
    x = (torch.randn(rows, k, device=DEVICE) * 0.5).bfloat16()
    low, full = project_head(bf16, x), project_head(fp32, x)
    assert full.dtype == torch.float32 and full.shape == (rows, n)
    assert torch.equal(full.bfloat16(), low)
    reference = x.double() @ weight.double().T
    assert (full.double() - reference).abs().max() <= (
        low.double() - reference
    ).abs().max()


def test_bf16_head_writes_fp32_logits_batch_invariant():
    n, k = 32320, 256
    generator = torch.Generator(device=DEVICE).manual_seed(9)
    weight = (torch.randn(n, k, generator=generator, device=DEVICE) * 0.05).bfloat16()
    bf16, fp32 = SimpleNamespace(weight=weight), SimpleNamespace(weight=weight)
    prepare_bf16_head(bf16)
    prepare_bf16_head(fp32, torch.float32)
    x = torch.randn(130, k, device=DEVICE).bfloat16()
    full = project_head(fp32, x)
    assert full.dtype == torch.float32
    assert torch.equal(full.bfloat16(), project_head(bf16, x))
    for rows in (1, 5, 16, 17, 33, 64):
        assert torch.equal(project_head(fp32, x[:rows]), full[:rows]), rows


def test_logits_processor_projects_fp32_heads():
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.head_dtype = torch.float32
    processor.use_tilelang_vocab_projection = True
    n, k = 1000, 512
    layer = _packed_layer(_head(n, k, seed=13))
    prepare_packed_head(layer, torch.float32)
    hidden = (torch.randn(2, 3, k, device=DEVICE) * 0.5).bfloat16()
    logits = processor._apply_head(layer, hidden, None)
    assert logits.dtype == torch.float32
    expected = project_head(layer, hidden.reshape(-1, k)).reshape(2, 3, n)
    assert torch.equal(logits, expected)
