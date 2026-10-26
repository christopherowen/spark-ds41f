# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 TileLang vocabulary heads: the exact packed BF16 head (its packer and
its projection) and BF16 heads, against FP32 references, with logits independent of
the batch."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(120)
):
    pytest.skip("TileLang V4.1 kernels target SM120/SM121", allow_module_level=True)

pytest.importorskip("tilelang")

from vllm.models.deepseek_v4_1.tilelang import vocab_pack as packed_format  # noqa: E402
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


def _format_weight(rows, k, *, seed=0, infinities=False):
    """Vocabulary-like weights plus the value classes the format must keep (B12X's
    packer tests): zeros, subnormals, values far below and above the window, +-1,
    infinities and a NaN, and rows with many out-of-window values."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    weight = torch.randn(rows, k, device=DEVICE, generator=generator) * 0.02
    weight = weight.bfloat16()
    bits = weight.view(torch.int16)
    specials = [0x0000, -0x8000, 0x0001, -0x7FFF, 0x007F, 0x0080, 0x2000, 0x1000]
    specials += [0x3F80, -0x4080, 0x4500]
    if infinities:
        specials += [0x7F7F, 0x7F80, -0x0080, 0x7FC0]
    specials = torch.tensor(specials, dtype=torch.int16, device=DEVICE)
    bits[3, : specials.numel()] = specials
    bits[7, 64 : 64 + specials.numel()] = specials
    bits[rows - 2, ::3] = 0x4100
    bits[rows - 5, 1::4] = 0x2000
    return weight


def _bits(weight):
    return weight.view(torch.int16)


def test_pack_round_trips_every_bit():
    weight = _format_weight(512, 640, infinities=True)
    packed = packed_format.pack(weight)
    assert packed.values.shape == (512, 960) and packed.values.dtype == torch.uint8
    assert packed.exceptions > 300
    assert torch.equal(_bits(packed_format.unpack(packed)), _bits(weight))
    offsets = packed.row_offsets.cpu()
    assert offsets[0] == 0 and offsets[-1] == packed.exceptions
    assert torch.all(offsets[1:] >= offsets[:-1])


def test_pack_lists_each_rows_exceptions_in_column_order():
    weight = _format_weight(300, 1024, seed=4, infinities=True)
    packed = packed_format.pack(weight)
    rows = packed.exception_rows[: packed.exceptions].cpu()
    entries = packed.exception_entries[: packed.exceptions].cpu()
    offsets = packed.row_offsets.cpu()
    for row in range(weight.shape[0]):
        start, end = int(offsets[row]), int(offsets[row + 1])
        assert torch.all(rows[start:end] == row)
        columns = entries[start:end] >> 16
        assert torch.all(columns[1:] > columns[:-1]), row
    # Every out-of-window value's slot holds +0: a code of 0 and a zero byte.
    columns = (entries >> 16).long()
    assert torch.all(packed.values[rows.long().cuda(), columns.cuda()] == 0)


def test_pack_window_is_the_lowest_of_the_fullest():
    # Exponents 112..126 hold every value but two: the window starts at 112.
    exponents = torch.arange(-15, 0, device=DEVICE).repeat(64, 18)[:, :256]
    weight = torch.ldexp(torch.ones(64, 256, device=DEVICE), exponents).bfloat16()
    weight[0, 0], weight[1, 1] = 2.0**20, 2.0**-40
    packed = packed_format.pack(weight)
    assert packed.exponent_base == 112 and packed.exceptions == 2
    assert torch.equal(_bits(packed_format.unpack(packed)), _bits(weight))
    # Only exponents 120 and 121 (and zeros, which take code 0): every window from
    # 107 to 120 holds them all, and the lowest is taken.
    tied = torch.where(exponents % 2 == 0, 2.0**-7, -(2.0**-6)).bfloat16()
    tied[5, 9] = 0.0
    packed = packed_format.pack(tied)
    assert packed.exponent_base == 107 and packed.exceptions == 0
    assert torch.equal(_bits(packed_format.unpack(packed)), _bits(tied))


def test_pack_allocates_only_its_result():
    # Packing runs at load, beside the whole model: it must not need transient
    # copies of the weight, which unified-memory GPUs do not hand back cleanly.
    weight = _format_weight(4096, 5120, seed=11)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    packed = packed_format.pack(weight)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    result = sum(
        t.numel() * t.element_size()
        for t in (
            packed.values,
            packed.exception_rows,
            packed.exception_entries,
            packed.row_offsets,
        )
    )
    assert peak <= result + 64 * 1024, (peak, result)
    assert torch.equal(_bits(packed_format.unpack(packed)), _bits(weight))


def test_pack_in_place_matches_and_allocates_one_staging_block():
    # 2,500 rows: two full 1,024-row blocks and a partial one.
    weight = _format_weight(2_500, 5_120, seed=13, infinities=True)
    original = weight.clone()
    expected = packed_format.pack(original.clone())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    packed = packed_format.pack(weight, in_place=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    staging = 1_024 * (5_120 + 2_560)
    small = sum(
        t.numel() * t.element_size()
        for t in (packed.exception_rows, packed.exception_entries, packed.row_offsets)
    )
    assert peak <= staging + small + 64 * 1024, (peak, staging, small)
    assert packed.values.data_ptr() == weight.data_ptr()
    assert packed.values.is_contiguous() and packed.values.shape == (2_500, 7_680)
    assert packed.exponent_base == expected.exponent_base
    assert packed.exceptions == expected.exceptions
    for name in ("values", "exception_rows", "exception_entries", "row_offsets"):
        assert torch.equal(getattr(packed, name), getattr(expected, name)), name
    assert torch.equal(_bits(packed_format.unpack(packed)), _bits(original))


def test_pack_without_exceptions_and_rejects_bad_weights():
    weight = torch.full((16, 128), 0.5, device=DEVICE, dtype=torch.bfloat16)
    packed = packed_format.pack(weight)
    assert packed.exceptions == 0 and packed.exception_rows.numel() == 1
    assert torch.equal(packed_format.unpack(packed), weight)
    with pytest.raises(ValueError):
        packed_format.pack(torch.zeros((16, 96), device=DEVICE, dtype=torch.bfloat16))
    with pytest.raises(TypeError):
        packed_format.pack(torch.zeros((16, 128), device=DEVICE, dtype=torch.float16))
    transposed = torch.zeros((128, 16), device=DEVICE, dtype=torch.bfloat16).t()
    with pytest.raises(ValueError):
        packed_format.pack(transposed)


def test_pack_refuses_a_weight_that_does_not_round_trip(monkeypatch):
    # The write pass counts the values whose decoded bits differ from the source;
    # any count refuses the pack.
    weight = _format_weight(64, 256, seed=2)
    real = packed_format.pack_rows_kernel

    def corrupting(k):
        kernel = real(k)

        def run(bits, values, offsets, rows, entries, mismatches, base, start):
            kernel(bits, values, offsets, rows, entries, mismatches, base, start)
            mismatches.add_(1)

        return run

    monkeypatch.setattr(packed_format, "pack_rows_kernel", corrupting)
    with pytest.raises(ValueError, match="round-trip"):
        packed_format.pack(weight)


def test_is_supported_on_this_device():
    assert packed_format.is_supported()
    assert packed_format.is_supported(torch.device("cuda", 0))
    assert not packed_format.is_supported("cpu")


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


@torch.no_grad()
def test_packed_lm_head_packs_in_its_storage_and_projects(
    default_vllm_config, dist_init, monkeypatch
):
    """A packed BF16 LM head from load to logits: ParallelLMHead's packed method packs
    the loaded head in its own storage, declares the TileLang projection, and the
    shared target and drafter processors project it, close to the BF16 head, with a
    token's logits the same bits in any batch and on graph replay."""
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    monkeypatch.setattr(default_vllm_config.kernel_config, "linear_backend", "tilelang")
    torch.manual_seed(412)
    with torch.device(DEVICE):
        head = ParallelLMHead(
            32768, 5120, params_dtype=torch.bfloat16, lm_head_quantization="packed_bf16"
        )
    assert head.runtime_lm_head_quantization == "packed_bf16"
    head.weight.normal_(std=0.02)
    head.weight[5, :7] = torch.tensor(
        [0.0, -0.0, 1e-30, 3.0, -2.5, 1e-7, 6e-39], dtype=torch.bfloat16
    )
    original = head.weight.detach().clone()
    storage = head.weight.data_ptr()
    head.quant_method.process_weights_after_loading(head)
    assert head.weight.dtype == torch.uint8 and head.weight.shape == (32768, 7680)
    assert head.weight.data_ptr() == storage
    packed = head.quant_method.packed_weight(head)
    assert packed.exceptions > 0
    assert torch.equal(_bits(packed_format.unpack(packed)), _bits(original))
    head.quant_method.declare_layer_kernels(head)
    assert set(head.tilelang_vocab) == set(ROW_TILES)
    target, draft = LogitsProcessor(32768), LogitsProcessor(32768)
    for processor in (target, draft):
        processor.prepare_b12x_vocab_projection(head)
        assert not processor._tilelang_vocab_heads  # the packed method owns its kernels
    source = (torch.randn(8, 5120, device=DEVICE) * 0.5).bfloat16()
    first = target(head, source[:1])
    for rows in (1, 7, 8):
        for processor in (target, draft):
            actual = processor(head, source[:rows])
            assert _close_to_fp32(actual, source[:rows], original)
            assert torch.equal(_bits(actual[:1]), _bits(first))
    expected = draft(head, source).clone()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            actual = draft(head, source)
        actual.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(actual), _bits(expected))
    finally:
        graph.reset()
