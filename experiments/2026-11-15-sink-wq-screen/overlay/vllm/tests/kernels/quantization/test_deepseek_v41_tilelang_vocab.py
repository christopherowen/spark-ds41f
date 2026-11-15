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

from vllm.models.deepseek_v4_1.tilelang import vocab as packed_format  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.vocab import (  # noqa: E402
    ROW_TILES,
    nvfp4_vocab_projection,
    pack_bf16_weight,
    packed_vocab_projection,
    prepare_bf16_head,
    prepare_nvfp4_head,
    prepare_packed_head,
    project_head,
    project_head_greedy_pairs,
    project_head_markov_greedy_pairs,
    row_tile,
    tile_config,
    unpack_bf16_weight,
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
    x = (torch.randn(128, k, device=DEVICE) * 0.5).bfloat16()
    full = _project(packed, n, k, x, 64)
    for block_M in ROW_TILES:
        assert torch.equal(_project(packed, n, k, x, block_M), full), block_M
    for rows in (1, 2, 6, 17, 40, 65, 96, 128):
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


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_packed_head_greedy_pairs_match_greedy_shard_pair(n, k, out_dtype):
    from vllm.v1.worker.gpu.spec_decode.greedy_verification import (
        vocab_parallel_pairs,
        vocab_parallel_pairs_from_head,
    )

    weight = _head(n, k, seed=n + 19)
    # Include identical rows inside and across vocabulary tiles to exercise tie-breaking.
    weight[100] = weight[5]
    weight[500] = weight[5]
    layer = _packed_layer(weight)
    prepare_packed_head(layer, out_dtype)
    assert set(layer.tilelang_vocab_greedy) == set(ROW_TILES)
    x = (torch.randn(130, k, device=DEVICE) * 0.5).bfloat16()
    for vocab, vocab_start in ((n, 0), (n - 37, 32320)):
        for rows in (1, 6, 24, 48, 65, 96, 128, 130):
            part = x[:rows].contiguous()
            logits = project_head(layer, part)
            expected = vocab_parallel_pairs(logits, vocab, vocab_start, tp_size=1)
            actual = project_head_greedy_pairs(
                layer, part, vocab=vocab, vocab_start=vocab_start
            )
            from_head = vocab_parallel_pairs_from_head(
                layer, part, vocab, vocab_start, tp_size=1
            )
            assert torch.equal(actual[:, :2], expected[:, :2]), (rows, vocab)
            assert torch.equal(from_head[:, :2], expected[:, :2]), (rows, vocab)


def test_project_head_greedy_pairs_borrows_workspace_and_reuses_layer_scratch(
    monkeypatch,
):
    import vllm.v1.worker.gpu.spec_decode.greedy_verification as gv
    import vllm.v1.worker.workspace as ws_mod
    from vllm.v1.worker.gpu.spec_decode.greedy_verification import (
        PAIR_WIDTH,
        vocab_parallel_pairs,
        vocab_parallel_pairs_from_head,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    n, k = 1000, 512
    weight = _head(n, k, seed=77)
    layer = _packed_layer(weight)
    prepare_packed_head(layer)
    x = (torch.randn(24, k, device=DEVICE) * 0.5).bfloat16()
    logits = project_head(layer, x)
    expected = vocab_parallel_pairs(logits, n, 0, tp_size=1)

    # 1. Without WorkspaceManager initialized: persistent layer scratch is reused
    # across tile sizes (16 -> 24 -> 6) and when scratch_pairs=True.
    monkeypatch.setattr(ws_mod, "_manager", None)
    p6_a = project_head_greedy_pairs(layer, x[:6], vocab=n, vocab_start=0)
    scratch_6 = layer._tilelang_vocab_greedy_scratch
    assert scratch_6 is not None
    p6_b = project_head_greedy_pairs(layer, x[:6], vocab=n, vocab_start=0)
    assert layer._tilelang_vocab_greedy_scratch[0].data_ptr() == scratch_6[0].data_ptr()
    assert p6_a.data_ptr() != p6_b.data_ptr()
    assert torch.equal(p6_a[:, :2], expected[:6, :2])
    assert torch.equal(p6_b[:, :2], expected[:6, :2])

    p24 = project_head_greedy_pairs(
        layer, x, vocab=n, vocab_start=0, scratch_pairs=True
    )
    assert p24.data_ptr() == layer._tilelang_vocab_greedy_scratch[2].data_ptr()
    assert torch.equal(p24[:, :2], expected[:, :2])

    # 2. With WorkspaceManager initialized: partial_vals, partial_idxs, and
    # TP>1 local pairs are borrowed directly from the active workspace.
    init_workspace_manager(torch.device(DEVICE), num_ubatches=1)
    manager = ws_mod.current_workspace_manager()
    monkeypatch.setattr(
        gv,
        "tensor_model_parallel_all_gather",
        lambda t, dim=-1: t.repeat(1, 3),
    )
    tp3_pairs = vocab_parallel_pairs_from_head(layer, x[:6], n, 0, tp_size=3)
    assert tp3_pairs.shape == (6, PAIR_WIDTH * 3)
    assert torch.equal(tp3_pairs[:, :2], expected[:6, :2])
    ws_base = manager._current_workspaces[0].data_ptr()
    ws_end = ws_base + manager.available_bytes()
    p6_ws = project_head_greedy_pairs(
        layer, x[:6], vocab=n, vocab_start=0, scratch_pairs=True
    )
    assert ws_base <= p6_ws.data_ptr() < ws_end
    assert torch.equal(p6_ws[:, :2], expected[:6, :2])


@pytest.mark.parametrize("n,k", [(512, 256), (2500, 512), (32320, 5120)])
@pytest.mark.parametrize("in_place", [False, True])
def test_tilelang_pack_bf16_weight_round_trips_and_matches_b12x(n, k, in_place):
    weight = _head(n, k, seed=n + k + 41)
    weight[0, :7] = torch.tensor(
        [0.0, -0.0, 1e-30, 3.0, -2.5, 1e-7, 6e-39], dtype=torch.bfloat16, device=DEVICE
    )
    original = weight.clone()
    packed = pack_bf16_weight(weight.clone() if in_place else weight, in_place=in_place)
    assert packed.out_features == n and packed.in_features == k
    assert packed.exceptions > 0
    unpacked = unpack_bf16_weight(packed)
    assert torch.equal(unpacked.view(torch.int16), original.view(torch.int16))

    b12x_packed_mod = pytest.importorskip("b12x.gemm.packed_bf16_vocab_projection")
    ref = b12x_packed_mod.pack(original.clone(), in_place=False)
    assert packed.exponent_base == ref.exponent_base
    assert packed.exceptions == ref.exceptions
    assert torch.equal(packed.values, ref.values)
    assert torch.equal(packed.row_offsets, ref.row_offsets)
    assert torch.equal(packed.exception_rows, ref.exception_rows)
    assert torch.equal(packed.exception_entries, ref.exception_entries)


def test_packed_bf16_lm_head_method_tilelang_without_b12x(monkeypatch):
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.online import packed_bf16 as pbf16_mod
    from vllm.model_executor.layers.quantization.online.packed_bf16 import (
        PackedBf16LmHeadMethod,
        packed_bf16_supported,
    )

    config = VllmConfig(kernel_config=KernelConfig(linear_backend="tilelang"))
    monkeypatch.setattr(
        pbf16_mod, "get_b12x_packed_bf16_vocab_projection", lambda: None
    )
    with set_current_vllm_config(config):
        assert packed_bf16_supported(512, torch.bfloat16)
        method = PackedBf16LmHeadMethod()
        assert method.tilelang
        layer = torch.nn.Module()
        with torch.device(DEVICE):
            method.create_weights(layer, 512, [1000], 512, 1000, torch.bfloat16)
        original = _head(1000, 512, seed=77)
        layer.weight.data.copy_(original)
        method.process_weights_after_loading(layer)
        packed = method.packed_weight(layer)
        assert torch.equal(
            unpack_bf16_weight(packed).view(torch.int16), original.view(torch.int16)
        )
        method.declare_layer_kernels(layer)
        x = (torch.randn(9, 512, device=DEVICE) * 0.5).bfloat16()
        assert _close_to_fp32(project_head(layer, x), x, original)



def _nvfp4_layer(weight: torch.Tensor):
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        FLOAT4_E2M1_MAX,
        dequantize_to_dtype,
    )

    float8_e4m3_max = torch.finfo(torch.float8_e4m3fn).max
    amax = weight.abs().amax().float().clamp_min(1e-8)
    global_scale = (FLOAT4_E2M1_MAX * float8_e4m3_max) / amax
    packed, scales = scaled_fp4_quant(
        weight.contiguous(), global_scale, is_sf_swizzled_layout=False
    )
    weight_global_scale = global_scale.reciprocal().reshape(1).to(torch.float32)
    dequant = dequantize_to_dtype(
        packed,
        scales.view(torch.float8_e4m3fn),
        weight_global_scale,
        dtype=torch.float32,
        block_size=16,
        swizzle=False,
    )
    layer = SimpleNamespace(
        weight=packed,
        weight_scale=scales,
        weight_global_scale=weight_global_scale,
        nvfp4_in_features=int(weight.shape[1]),
    )
    return layer, dequant


@pytest.mark.parametrize("n,k", [(1000, 256), (32320, 256), (1000, 512)])
def test_nvfp4_head_matches_dequantized_reference_and_is_batch_invariant(n, k):
    generator = torch.Generator(device=DEVICE).manual_seed(n + k + 23)
    weight = (torch.randn(n, k, generator=generator, device=DEVICE) * 0.05).bfloat16()
    layer, dequant_weight = _nvfp4_layer(weight)
    prepare_nvfp4_head(layer)
    assert set(layer.tilelang_vocab) == set(ROW_TILES)
    x = (torch.randn(130, k, generator=generator, device=DEVICE) * 0.5).bfloat16()
    full = project_head(layer, x)
    assert _close_to_fp32(full, x, dequant_weight)
    for rows in (1, 5, 16, 17, 33, 64, 96, 128):
        assert torch.equal(project_head(layer, x[:rows].contiguous()), full[:rows]), rows
    assert torch.equal(project_head(layer, x[70:71].contiguous()), full[70:71])


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_nvfp4_head_greedy_pairs_match_greedy_shard_pair(out_dtype):
    from vllm.v1.worker.gpu.spec_decode.greedy_verification import (
        vocab_parallel_pairs,
        vocab_parallel_pairs_from_head,
    )

    n, k = 1000, 256
    generator = torch.Generator(device=DEVICE).manual_seed(109)
    weight = (torch.randn(n, k, generator=generator, device=DEVICE) * 0.05).bfloat16()
    weight[100] = weight[5]
    weight[500] = weight[5]
    layer, _ = _nvfp4_layer(weight)
    prepare_nvfp4_head(layer, out_dtype)
    assert set(layer.tilelang_vocab_greedy) == set(ROW_TILES)
    x = (torch.randn(96, k, generator=generator, device=DEVICE) * 0.5).bfloat16()
    for vocab, vocab_start in ((n, 0), (n - 37, 32320)):
        for rows in (1, 6, 24, 48, 65, 96):
            part = x[:rows].contiguous()
            logits = project_head(layer, part)
            expected = vocab_parallel_pairs(logits, vocab, vocab_start, tp_size=1)
            actual = project_head_greedy_pairs(
                layer, part, vocab=vocab, vocab_start=vocab_start
            )
            from_head = vocab_parallel_pairs_from_head(
                layer, part, vocab, vocab_start, tp_size=1
            )
            assert torch.equal(actual[:, :2], expected[:, :2]), (rows, vocab)
            assert torch.equal(from_head[:, :2], expected[:, :2]), (rows, vocab)


def test_nvfp4_online_linear_method_tilelang_without_b12x(monkeypatch):
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear.nvfp4.b12x import B12xNvFp4LinearKernel
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.quantization.online.nvfp4 import (
        Nvfp4OnlineLinearMethod,
    )

    monkeypatch.setattr(
        B12xNvFp4LinearKernel,
        "is_supported",
        classmethod(lambda cls, *_args, **_kwargs: (False, "b12x disabled")),
    )
    config = VllmConfig(kernel_config=KernelConfig(linear_backend="tilelang"))
    with set_current_vllm_config(config):
        method = Nvfp4OnlineLinearMethod()
        assert method.tilelang and method.kernel is None
        layer = torch.nn.Module()
        generator = torch.Generator(device=DEVICE).manual_seed(83)
        original = (
            torch.randn(1000, 256, generator=generator, device=DEVICE) * 0.05
        ).bfloat16()
        layer.register_parameter(
            "weight", torch.nn.Parameter(original.clone(), requires_grad=False)
        )
        method.process_weights_after_loading(layer)
        method.declare_layer_kernels(layer)

        _, dequant = _nvfp4_layer(original)
        x = (torch.randn(2, 5, 256, generator=generator, device=DEVICE) * 0.5).bfloat16()
        out = method.apply(layer, x)
        assert out.shape == (2, 5, 1000) and out.dtype == torch.bfloat16
        assert _close_to_fp32(out.reshape(-1, 1000), x.reshape(-1, 256), dequant)

        processor = LogitsProcessor.__new__(LogitsProcessor)
        torch.nn.Module.__init__(processor)
        processor.head_dtype = None
        processor.use_tilelang_vocab_projection = True
        logits = processor._apply_head(layer, x, None)
        assert torch.equal(logits, out)


# DS4.1's Markov projection (K = 256) shards: padded rows, then (real rows, first
# global index) of a first and a last rank; "odd" ends in a partial tile.
MARKOV_SHARDS = {
    "tp4": (32320, ((32320, 0), (32320, 96960))),
    "tp3": (43136, ((43136, 0), (43008, 86272))),
    "odd": (1000, ((1000, 0), (963, 32320))),
}
MARKOV_ROWS = (1, 2, 3, 5, 8, 16, 17, 31, 33, 48, 64)


def _markov_greedy_inputs(n, k, dtype, seed):
    """Markov embeddings and a strided draft position of base logits whose first
    rows hold exact ties (in a 2048-column part, across parts, and through
    duplicate weight rows), NaNs and a row of -inf."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(64, k, generator=generator, device=DEVICE, dtype=torch.bfloat16)
    base_3d = torch.randn(64, 5, n, generator=generator, device=DEVICE, dtype=dtype)
    base = base_3d[:, 2]

    def put(row, columns, values):
        for column, value in zip(columns, values):
            if column < n:
                base[row, column] = value

    x[:3] = 0  # a zero bias: the base logits decide
    base[0] = -1.0
    put(0, (7, 13, 2053, 4105), (5.0,) * 4)
    base[1] = -1.0
    put(1, (40, 41, 6145), (6.0, 5.5, 5.0))  # the runner-up shares the max's part
    put(2, (900, n - 20), (float("nan"),) * 2)
    base[3] = -float("inf")
    put(4, (5, 100, 500), (20.0,) * 3)  # duplicate weight rows: equal biases
    return x, base_3d


def _assert_markov_greedy_pairs_exact(layer, x, base_3d, shards, top2):
    """The fused pairs (with ``top2``, and runner-ups) of every row count are the
    bits ``greedy_markov_shard_pair`` writes from the materialized bias, and
    neither writes the slack columns without ``top2``."""
    from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
        greedy_markov_shard_pair,
        scratch_shape,
    )

    for vocab, vocab_start in shards:
        for rows in MARKOV_ROWS:
            part = x[:rows].contiguous()
            base_step = base_3d[:rows, 2]
            bias = project_head(layer, part)
            expected = torch.full((rows, 4), -7.0, dtype=torch.float32, device=DEVICE)
            shape = scratch_shape(rows, vocab)
            greedy_markov_shard_pair(
                base_step[:, :vocab],
                bias[:, :vocab],
                vocab_start,
                expected,
                torch.empty(shape, dtype=torch.float32, device=DEVICE),
                torch.empty(shape, dtype=torch.int32, device=DEVICE),
                top2=top2,
            )
            actual = torch.full_like(expected, -7.0)
            project_head_markov_greedy_pairs(
                layer,
                part,
                base_step,
                vocab=vocab,
                vocab_start=vocab_start,
                out=actual,
                top2=top2,
            )
            assert torch.equal(
                actual.view(torch.int32), expected.view(torch.int32)
            ), (rows, vocab, top2)


@pytest.mark.parametrize("top2", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", sorted(MARKOV_SHARDS))
def test_packed_head_markov_greedy_pairs_match_greedy_markov_shard_pair(
    shape, out_dtype, top2
):
    from vllm.v1.worker.gpu.spec_decode.dspark import greedy

    assert packed_format.GREEDY_PART == greedy._BLOCK
    n, shards = MARKOV_SHARDS[shape]
    k = 256
    weight = _head(n, k, seed=n + 303)
    weight[100] = weight[5]
    weight[500] = weight[5]
    layer = _packed_layer(weight)
    layer.tilelang_max_rows = 64
    layer.tilelang_markov_head = True
    prepare_packed_head(layer, out_dtype)
    assert set(layer.tilelang_vocab_markov_greedy) == {16, 32, 64}
    x, base_3d = _markov_greedy_inputs(n, k, out_dtype, seed=n + 404)
    _assert_markov_greedy_pairs_exact(layer, x, base_3d, shards, top2)


@pytest.mark.parametrize("top2", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", sorted(MARKOV_SHARDS))
def test_bf16_head_markov_greedy_pairs_match_greedy_markov_shard_pair(
    shape, out_dtype, top2
):
    n, shards = MARKOV_SHARDS[shape]
    k = 256
    generator = torch.Generator(device=DEVICE).manual_seed(n + 505)
    weight = (torch.randn(n, k, generator=generator, device=DEVICE) * 0.05).bfloat16()
    weight[100] = weight[5]
    weight[500] = weight[5]
    layer = SimpleNamespace(
        weight=weight, tilelang_max_rows=64, tilelang_markov_head=True
    )
    prepare_bf16_head(layer, out_dtype)
    assert set(layer.tilelang_vocab_markov_greedy) == {16, 32, 64}
    x, base_3d = _markov_greedy_inputs(n, k, out_dtype, seed=n + 606)
    _assert_markov_greedy_pairs_exact(layer, x, base_3d, shards, top2)


def test_only_markov_heads_declare_markov_greedy_pairs():
    weight = _head(1000, 256, seed=17)
    packed, bf16 = _packed_layer(weight), SimpleNamespace(weight=weight)
    prepare_packed_head(packed)
    prepare_bf16_head(bf16)
    assert hasattr(packed, "tilelang_vocab_greedy")
    assert not hasattr(packed, "tilelang_vocab_markov_greedy")
    assert not hasattr(bf16, "tilelang_vocab_markov_greedy")



@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_packed_head_certified_screen_edge_cases(out_dtype):
    from vllm.v1.worker.gpu.spec_decode.greedy_verification import vocab_parallel_pairs

    n, k = 1000, 512
    weight = _head(n, k, seed=919)
    weight[77] = 0.0
    weight[100] = weight[5]
    weight[500] = weight[5]
    weight[200] = weight[5].clone()
    weight[200, 0] = (weight[200, 0].float() + 1e-4).bfloat16()
    weight[300, 7] = (2.0 * weight[300].abs().max().float()).bfloat16()
    weight[301, 11] = 300.0
    layer = _packed_layer(weight)
    prepare_packed_head(layer, out_dtype)

    generator = torch.Generator(device=DEVICE).manual_seed(920)
    x = (torch.randn(65, k, generator=generator, device=DEVICE) * 0.5).bfloat16()
    x[0] = 0.0
    x[1, 13] = float("nan")
    x[2] = weight[200]
    x[3] = weight[300]
    x[4] = weight[301]
    for vocab, vocab_start in ((n, 0), (n - 37, 32320)):
        for rows in (1, 2, 6, 16, 17, 32, 33, 48, 65):
            part = x[:rows].contiguous()
            logits = project_head(layer, part)
            expected = vocab_parallel_pairs(logits, vocab, vocab_start, tp_size=1)
            actual = project_head_greedy_pairs(
                layer, part, vocab=vocab, vocab_start=vocab_start
            )
            assert torch.equal(
                actual[:, :2].view(torch.int32), expected[:, :2].view(torch.int32)
            ), (rows, vocab, out_dtype)


def test_packed_bf16_lm_head_carveout_shares_single_buffer_with_screen_wq(monkeypatch):
    import os
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.online.packed_bf16 import (
        PackedBf16LmHeadMethod,
    )
    from vllm.v1.worker import display_carveout
    from vllm.v1.worker.gpu.spec_decode.greedy_verification import vocab_parallel_pairs

    if not os.access("/dev/dri/card0", os.R_OK | os.W_OK):
        pytest.skip("no DRM node")

    class _HeadLayer(torch.nn.Module):
        pass

    monkeypatch.setattr(display_carveout, "VocabParallelEmbedding", _HeadLayer)
    monkeypatch.setattr(display_carveout, "_REQUIRED", True)
    monkeypatch.setattr(display_carveout, "_carveouts", [])
    monkeypatch.setattr(display_carveout, "_integrity", None)
    monkeypatch.setattr(display_carveout, "_CHECK_SECONDS", 300.0)

    n, k = 16384, 4096
    config = VllmConfig(kernel_config=KernelConfig(linear_backend="tilelang"))
    with set_current_vllm_config(config):
        method = PackedBf16LmHeadMethod()
        layer = _HeadLayer()
        with torch.device(DEVICE):
            method.create_weights(layer, k, [n], k, n, torch.bfloat16)
        original = _head(n, k, seed=42)
        layer.weight.data.copy_(original)
        method.process_weights_after_loading(layer)
        method.declare_layer_kernels(layer)

        assert len(display_carveout._carveouts) == 1
        carveout = display_carveout._carveouts[0]
        w_q = layer._tilelang_vocab_screen[0]
        assert display_carveout._holds(carveout, layer.weight.data)
        assert display_carveout._holds(carveout, w_q)

        model = torch.nn.Module()
        model.lm_head = layer
        display_carveout.confirm_streamed_weights(model, torch.device(DEVICE))
        assert display_carveout._integrity is not None
        display_carveout._integrity.check()
        assert display_carveout._integrity.failure is None

        x = (torch.randn(6, k, device=DEVICE) * 0.5).bfloat16()
        logits = project_head(layer, x)
        expected = vocab_parallel_pairs(logits, n, 0, tp_size=1)
        actual = project_head_greedy_pairs(layer, x, vocab=n, vocab_start=0)
        assert torch.equal(actual[:, :2], expected[:, :2])


def test_tile_config_128_uses_matched_two_stage_pipeline():
    cfg = tile_config(128, 5120)
    assert cfg == dict(block_N=32, threads=256, step_groups=1, num_stages=2)
    assert "act_stages" not in cfg
    assert (
        packed_format.shared_memory_bytes(
            128,
            cfg["block_N"],
            cfg["num_stages"],
            cfg["step_groups"],
            act_stages=cfg.get("act_stages"),
        )
        == 86_016
    )
    assert (
        packed_format.nvfp4_shared_memory_bytes(
            128,
            cfg["block_N"],
            cfg["num_stages"],
            cfg["step_groups"],
            act_stages=cfg.get("act_stages"),
        )
        == 78_336
    )


def test_screen_tile_config_128_uses_matched_two_stage_pipeline():
    """The certified screen's 128-row tile stages its activations and 8-bit weights
    together (no ``act_stages`` offset), as the projection's 128-row tile does."""
    cfg = packed_format.screen_tile_config(128, 5120)
    assert cfg == dict(block_N=32, threads=256, step_groups=1, num_stages=2)
    assert "act_stages" not in cfg


@pytest.mark.parametrize("n,k", [(64, 256), (1000, 512), (32320, 5120)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_packed_head_greedy_pairs_sink_cut_and_topk_exact(n, k, out_dtype):
    from vllm.v1.sample.sink import SinkConfig
    from vllm.v1.worker.gpu.sample.sink import SinkSampler
    from vllm.v1.worker.gpu.spec_decode.greedy_verification import (
        greedy_rejection_sample,
        vocab_parallel_pairs,
        vocab_parallel_pairs_from_head,
    )

    weight = _head(n, k, seed=n + 701)
    weight[6] = weight[5]
    weight[40] = weight[5]
    if n > 500:
        weight[500] = weight[5]
    weight[30 if n <= 64 else 300, 7] = 300.0
    layer = _packed_layer(weight)
    prepare_packed_head(layer, out_dtype)

    generator = torch.Generator(device=DEVICE).manual_seed(n + 702)
    max_rows = 65
    x = (torch.randn(max_rows, k, generator=generator, device=DEVICE) * 0.5).bfloat16()
    x[0] = 0.0
    x[1, 13] = float("nan")
    if max_rows > 2 and n > 40:
        x[2] = weight[5]

    sink = SinkSampler(
        SinkConfig(column=0, threshold=1.0, rejection=True, max_retries=3),
        max_num_reqs=32,
        hidden_size=k,
        vocab_size=n + 32320,
        num_speculative_steps=5,
        device=torch.device(DEVICE),
        use_fp64_gumbel=False,
    )

    for vocab, vocab_start in ((n, 0), (n - 7, 32320)):
        for rows in (1, 6, 24, 65):
            part = x[:rows].contiguous()
            cu = torch.tensor(
                [0, 1, rows] if rows > 1 else [0, 1], dtype=torch.int32, device=DEVICE
            )
            num_reqs = cu.numel() - 1
            logits_ref = project_head(layer, part)
            arg0 = logits_ref[:, :vocab].argmax(dim=-1).to(torch.int32)
            drafts = torch.empty(rows, dtype=torch.int32, device=DEVICE)
            drafts[0] = 5 + vocab_start
            if rows > 1:
                drafts[1:] = arg0[:-1] + vocab_start
                if rows > 3:
                    drafts[3] = 5 if vocab_start > 0 else n + 100
            flags = torch.zeros(rows, dtype=torch.int8, device=DEVICE)
            flags[0] = 1
            if rows > 1:
                flags[1] = 1
            if rows > 2:
                flags[2] = 1
            if rows > 4:
                flags[4] = 1

            logits_cut = logits_ref.clone()
            sink.cut(logits_cut, flags, drafts, cu, num_reqs, vocab_start, vocab)
            expected_pairs = vocab_parallel_pairs(logits_cut, vocab, vocab_start, tp_size=1)
            exp_sampled, exp_num_sampled = greedy_rejection_sample(
                expected_pairs, drafts, cu, 5
            )
            expected_alts = sink.alternates_of_greedy_shards(
                logits_cut, cu, exp_num_sampled, exp_sampled, vocab_start, vocab, 1
            )

            cut_cols = sink.cut_cols(flags, drafts, cu, num_reqs, vocab_start, vocab)
            actual_pairs, row_topk = vocab_parallel_pairs_from_head(
                layer,
                part,
                vocab,
                vocab_start,
                tp_size=1,
                cut_cols=cut_cols,
                top_k=sink.num_candidates,
            )
            act_sampled, act_num_sampled = greedy_rejection_sample(
                actual_pairs, drafts, cu, 5
            )
            actual_alts = sink.alternates_of_greedy_top_candidates(
                row_topk, cu, act_num_sampled, act_sampled, tp_size=1
            )

            assert torch.equal(
                actual_pairs.view(torch.int32)[:, :2],
                expected_pairs.view(torch.int32)[:, :2],
            ), (n, out_dtype, vocab, vocab_start, rows)
            assert torch.equal(act_num_sampled, exp_num_sampled)
            for req_idx in range(num_reqs):
                n_tok = int(act_num_sampled[req_idx])
                assert torch.equal(
                    act_sampled[req_idx, :n_tok], exp_sampled[req_idx, :n_tok]
                )
            assert torch.equal(actual_alts, expected_alts), (
                n,
                out_dtype,
                vocab,
                vocab_start,
                rows,
            )


def test_packed_head_greedy_pairs_sink_tp4_and_cudagraph(monkeypatch):
    import vllm.v1.worker.gpu.sample.sink as sink_mod
    import vllm.v1.worker.gpu.spec_decode.greedy_verification as gv
    from vllm.v1.sample.sink import SinkConfig
    from vllm.v1.worker.gpu.sample.sink import SinkSampler
    from vllm.v1.worker.gpu.spec_decode.greedy_verification import (
        greedy_rejection_sample,
        vocab_parallel_pairs,
        vocab_parallel_pairs_from_head,
    )

    n, k, tp_size = 512, 256, 4
    layer = _packed_layer(_head(n, k, seed=811))
    prepare_packed_head(layer, torch.float32)
    x = (torch.randn(12, k, device=DEVICE) * 0.5).bfloat16()
    cu = torch.tensor([0, 6, 12], dtype=torch.int32, device=DEVICE)
    drafts = torch.arange(12, dtype=torch.int32, device=DEVICE) + 32320
    flags = torch.tensor([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.int8, device=DEVICE)

    sink = SinkSampler(
        SinkConfig(column=0, threshold=1.0, rejection=True, max_retries=3),
        max_num_reqs=8,
        hidden_size=k,
        vocab_size=n * tp_size,
        num_speculative_steps=5,
        device=torch.device(DEVICE),
        use_fp64_gumbel=False,
    )

    def _fake_gather(t, dim=-1):
        chunks = [t.clone() for _ in range(tp_size)]
        for r in range(1, tp_size):
            if t.shape[-1] == 4:
                chunks[r][:, 1] += r * n
            else:
                half = t.shape[-1] // 2
                chunks[r][:, half:] += r * n
        return torch.cat(chunks, dim=dim)

    monkeypatch.setattr(gv, "tensor_model_parallel_all_gather", _fake_gather)
    monkeypatch.setattr(sink_mod, "tensor_model_parallel_all_gather", _fake_gather)

    logits_ref = project_head(layer, x)
    sink.cut(logits_ref, flags, drafts, cu, 2, 32320, n)
    exp_pairs = vocab_parallel_pairs(logits_ref, n, 32320, tp_size=tp_size)
    exp_sampled, exp_n = greedy_rejection_sample(exp_pairs, drafts, cu, 5)
    exp_alts = sink.alternates_of_greedy_shards(logits_ref, cu, exp_n, exp_sampled, 32320, n, tp_size)

    cut_cols = sink.cut_cols(flags, drafts, cu, 2, 32320, n)
    act_pairs, row_topk = vocab_parallel_pairs_from_head(
        layer, x, n, 32320, tp_size=tp_size, cut_cols=cut_cols, top_k=sink.num_candidates
    )
    act_sampled, act_n = greedy_rejection_sample(act_pairs, drafts, cu, 5)
    act_alts = sink.alternates_of_greedy_top_candidates(row_topk, cu, act_n, act_sampled, tp_size=tp_size)

    assert torch.equal(
        act_pairs.view(-1, tp_size, 4)[..., :2].view(torch.int32),
        exp_pairs.view(-1, tp_size, 4)[..., :2].view(torch.int32),
    )
    assert torch.equal(act_n, exp_n)
    for req_idx in range(2):
        n_tok = int(act_n[req_idx])
        assert torch.equal(act_sampled[req_idx, :n_tok], exp_sampled[req_idx, :n_tok])
    assert torch.equal(act_alts, exp_alts)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        g_cut = sink.cut_cols(flags, drafts, cu, 2, 32320, n)
        g_pairs, g_topk = vocab_parallel_pairs_from_head(
            layer, x, n, 32320, tp_size=tp_size, cut_cols=g_cut, top_k=sink.num_candidates
        )
        g_sampled, g_n = greedy_rejection_sample(g_pairs, drafts, cu, 5)
        g_alts = sink.alternates_of_greedy_top_candidates(g_topk, cu, g_n, g_sampled, tp_size=tp_size)
    graph.replay()
    assert torch.equal(
        g_pairs.view(-1, tp_size, 4)[..., :2].view(torch.int32),
        exp_pairs.view(-1, tp_size, 4)[..., :2].view(torch.int32),
    )
    assert torch.equal(g_n, exp_n)
    for req_idx in range(2):
        n_tok = int(g_n[req_idx])
        assert torch.equal(g_sampled[req_idx, :n_tok], exp_sampled[req_idx, :n_tok])
    assert torch.equal(g_alts, exp_alts)
