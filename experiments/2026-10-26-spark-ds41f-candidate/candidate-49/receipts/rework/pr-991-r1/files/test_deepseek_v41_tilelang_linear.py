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

from tests.kernels.deepseek_v41_contracts import (  # noqa: E402
    INF,
    NAN,
    assert_other_rows_equal,
    assert_rows_nan,
    assert_same_nonfinite,
)
from vllm.models.deepseek_v4_1.tilelang.gemm import (  # noqa: E402
    DECODE_ROWS,
    DECODE_TILE_ROWS,
    TILE_CONTIGUOUS_FP8,
    bf16_gemm,
    bf16_gemm_partials,
    bf16_shards,
    fp8_decode_config,
    fp8_prefill_config,
    mxfp8_gemm,
    mxfp8_gemm_decode,
    pack_scale_words,
    pack_weight_tiles,
    splitk_reduce,
    unpack_weight_tiles,
)
from vllm.models.deepseek_v4_1.tilelang.norm import TileLangRMSNorm  # noqa: E402

DEVICE = "cuda"
ROWS = (1, 7, 16, 17, 32, 33, 64, 65, 200)


@pytest.fixture
def default_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config


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
    "n, k",
    [
        (256, 512),
        (1536, 1024),
        (224, 384),
        (320, 576),
        (1152, 576),
        (1280, 15360),
        (1728, 15360),
    ],
)
def test_mxfp8_gemm(n, k):
    weight, exps, weight_values = _block32_weight(n, k, 1)
    # Packed block grid. The GEMM loads SFB[row // 32].
    weight_sf = pack_scale_words(exps)
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
                    kernel(
                        xq[:DECODE_ROWS], weight, words[:DECODE_ROWS], weight_sf, out
                    )
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
    # Packed block grid. The GEMM loads SFB[row // 32].
    weight_sf = pack_scale_words(exps)
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


@pytest.mark.parametrize(
    "rows",
    [1, 3, 5, 7, 11, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 95, 96, 97, 127, 128],
)
@pytest.mark.parametrize("split", [False, True])
def test_mxfp8_decode_reads_only_live_rows(rows, split):
    """Also run with compute-sanitizer initcheck: padding is never initialized.

    Comparing live results alone cannot detect an unused MMA lane's uninitialized
    read. Initcheck must see the first call before the later poisoned-tail replay.
    Both paths must preserve the prefill path's exact per-row accumulation.
    """
    from vllm.models.deepseek_v4_1.tilelang.gemm import (
        mxfp8_gemm_partials,
        split_row_tile,
    )

    n, k, shards = 64, 256, 2 if split else 1
    tile = split_row_tile(rows)
    padded = -(-rows // tile) * tile
    weight, exps, _ = _block32_weight(n, k, 41)
    weight_sf = pack_scale_words(exps)
    x = torch.randn((rows, k), generator=_gen(42), device=DEVICE).bfloat16()
    xq, sf, _ = _mxfp8(x)
    a = torch.empty((padded, k), dtype=xq.dtype, device=DEVICE)
    scales = torch.empty((padded, sf.shape[1]), dtype=sf.dtype, device=DEVICE)
    a[:rows].copy_(xq)
    scales[:rows].copy_(sf)
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=DEVICE)
    expected = torch.empty_like(out)
    mxfp8_gemm(n, k, block_M=64, block_N=64, block_K=64, num_stages=2, shards=shards)(
        xq, weight, sf.view(torch.uint32), weight_sf, expected
    )
    if split:
        partials = torch.empty((shards, rows, n), device=DEVICE)
        kernel = mxfp8_gemm_partials(
            n, k, shards, block_M=tile, block_N=64, block_K=64, num_stages=2
        )
        reduce = splitk_reduce(n, shards)

        def run():
            kernel(a, weight, scales.view(torch.uint32), weight_sf, partials)
            reduce(partials, out)
    else:
        kernel = mxfp8_gemm_decode(
            n,
            k,
            block_M=tile,
            block_N=64,
            block_K=64,
            num_stages=2,
            padded_rows=True,
        )

        def run():
            kernel(a, weight, scales.view(torch.uint32), weight_sf, out)

    run()
    assert torch.equal(out, expected)
    # Reuse the captured graph with different live input and NaN padding. The
    # exact reference is a separate unpadded prefill call on the new live rows.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    a[:rows].copy_(-xq.float())
    a[rows:].view(torch.uint8).fill_(0x7F)
    scales[rows:].fill_(0xFF)
    mxfp8_gemm(n, k, block_M=64, block_N=64, block_K=64, num_stages=2, shards=shards)(
        a[:rows], weight, scales[:rows].view(torch.uint32), weight_sf, expected
    )
    graph.replay()
    assert torch.equal(out, expected)


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


# The compressor's wkv (ratio 1, BF16 out) and wkv + wgate (ratio 2, FP32 out).
@pytest.mark.parametrize("parts, out_dtype", [(1, torch.bfloat16), (2, torch.float32)])
def test_split_linear(parts, out_dtype, k=5120):
    from torch import nn

    from vllm.models.deepseek_v4_1.tilelang.linear import (
        TileLangSplitLinearMethod,
        project_parts,
    )
    from vllm.v1.worker.workspace import use_preallocated_workspace

    from vllm.models.deepseek_v4_1.tilelang.linear import unpack_bf16_layer

    gen = _gen(5)
    layer = nn.Module()
    weight = (torch.randn((512 * parts, k), generator=gen, device=DEVICE) * 0.02).bfloat16()
    layer.weight = nn.Parameter(weight.clone(), requires_grad=False)
    layer.out_dtype = out_dtype
    method = TileLangSplitLinearMethod(parts)
    method.process_weights_after_loading(layer)
    assert hasattr(layer, "packed_bf16_exponent_base")
    assert torch.equal(unpack_bf16_layer(layer), weight)
    method.declare_layer_kernels(layer)  # standalone: prepares and binds now
    x = torch.randn((max(ROWS), k), generator=gen, device=DEVICE).bfloat16()
    scratch = torch.empty(
        method.get_workspace_size(layer, max(ROWS)), dtype=torch.uint8, device=DEVICE
    )

    def project(rows):
        outs = [
            torch.full((max(ROWS), 512), float("nan"), dtype=out_dtype, device=DEVICE)
            for _ in range(parts)
        ]
        with use_preallocated_workspace(scratch):
            project_parts(layer, x[:rows], outs, rows)
        return outs

    full = project(max(ROWS))
    for index, out in enumerate(full):
        ref = x.float() @ weight[index * 512 : (index + 1) * 512].float().T
        torch.testing.assert_close(
            out.float(), ref, rtol=1e-2, atol=1e-2 * ref.abs().max().item()
        )
    for rows in ROWS:
        for out, whole in zip(project(rows), full):
            assert torch.equal(out[:rows], whole[:rows]), f"rows={rows}"
            assert out[rows:].isnan().all(), f"rows={rows} wrote past its rows"
    # The one-launch split-K leaves its tile counters zeroed for the next call.
    assert not layer.tilelang_counters.any()


# The indexer's head weights (32 x 5120) scaled by 1/64.
def test_scaled_linear(n=32, k=5120, scale=1 / 64):
    from torch import nn

    from vllm.models.deepseek_v4_1.tilelang.linear import (
        TileLangLinearMethod,
        TileLangScaledLinearMethod,
    )
    from vllm.v1.worker.workspace import use_preallocated_workspace

    gen = _gen(8)
    weight = (torch.randn((n, k), generator=gen, device=DEVICE) * 0.02).bfloat16()
    plain, scaled = nn.Module(), nn.Module()
    plain.weight = nn.Parameter(weight.clone(), requires_grad=False)
    scaled.weight = nn.Parameter(weight.clone(), requires_grad=False)
    plain_method = TileLangLinearMethod()
    plain_method.process_weights_after_loading(plain)
    plain_method.declare_layer_kernels(plain)
    scaled_method = TileLangScaledLinearMethod(scale)
    scaled_method.process_weights_after_loading(scaled)
    scaled_method.declare_layer_kernels(scaled)
    x = torch.randn((max(ROWS), k), generator=gen, device=DEVICE).bfloat16()
    scratch = torch.empty(8 << 20, dtype=torch.uint8, device=DEVICE)
    with use_preallocated_workspace(scratch):
        for rows in ROWS:
            expected = plain_method.apply(plain, x[:rows]) * scale  # exact in BF16
            assert torch.equal(scaled_method.apply(scaled, x[:rows]), expected), rows
    with pytest.raises(ValueError):
        TileLangScaledLinearMethod(0.3)


# The DSpark drafter's context KV: rows 1280: of the fused Q-A/KV weight, a
# narrow projection (512 x 5120) that splits K for decode rows.
def test_block32_rows(default_vllm_config, n=1792, k=5120, start=1280):
    from torch import nn

    from vllm.models.deepseek_v4_1.tilelang.gemm import SPLIT_DECODE_ROWS, SPLIT_FP8
    from vllm.models.deepseek_v4_1.tilelang.linear import TileLangBlock32Rows
    from vllm.v1.worker.workspace import use_preallocated_workspace

    assert (n - start, k) in SPLIT_FP8
    weight, exps, weight_values = _block32_weight(n, k, 6)
    layer = nn.Module()
    layer.weight, layer.weight_scale_inv = weight, exps
    rows = TileLangBlock32Rows(layer, start)
    rows.declare_kernels()  # standalone: prepares and binds now
    counts = (1, 7, 16, 17, 32, 33, 64, 65, 96, SPLIT_DECODE_ROWS, 200, 300)
    x = torch.randn((max(counts), k), generator=_gen(7), device=DEVICE).bfloat16()
    _, _, x_values = _mxfp8(x)
    ref = x_values @ weight_values[start:].T
    scratch = torch.empty(16 << 20, dtype=torch.uint8, device=DEVICE)
    with use_preallocated_workspace(scratch):
        full = rows(x)  # prefill rows: the split prefill GEMM
        torch.testing.assert_close(
            full.float(), ref, rtol=1e-2, atol=1e-2 * ref.abs().max().item()
        )
        for count in counts:
            # Split-K decode rows add the shards as the prefill GEMM does.
            assert torch.equal(rows(x[:count]), full[:count]), f"rows={count}"
    with pytest.raises(ValueError):
        TileLangBlock32Rows(layer, start + 16)


# Attention's fused Q-A/KV (1792 x 5120 at TP4) and the shared expert's gate/up
# (1152 x 5120) read their input's MXFP8 cast from the mHC entry that wrote it.
@pytest.mark.parametrize("n, k", [(1792, 5120), (1152, 5120)])
def test_block32_reads_its_producers_cast(default_vllm_config, n, k):
    """A projection marked ``tilelang_mxfp8_input`` gives, from the cast handed over
    with its input (``provide_mxfp8``), the bits it gives casting the input itself:
    decode tiles over the padded rows and prefill GEMMs over the live rows. Without
    a cast, or with another tensor's, it fails rather than casting."""
    from torch import nn

    from vllm.models.deepseek_v4_1.tilelang.linear import (
        TileLangBlock32Rows,
        cast_mxfp8,
        provide_mxfp8,
    )
    from vllm.v1.worker.workspace import use_preallocated_workspace

    weight, exps, _ = _block32_weight(n, k, 6)
    layer = nn.Module()
    layer.weight, layer.weight_scale_inv = weight, exps
    casting = TileLangBlock32Rows(layer, 0)
    casting.declare_kernels()
    fed = TileLangBlock32Rows(layer, 0)
    fed.tilelang_mxfp8_input = True
    fed.declare_kernels()
    x = torch.randn((300, k), generator=_gen(8), device=DEVICE).bfloat16()
    scratch = torch.empty(16 << 20, dtype=torch.uint8, device=DEVICE)
    with use_preallocated_workspace(scratch):
        for count in (1, 7, 16, 17, 64, 65, 96, 128, 129, 300):
            rows = x[:count]
            expected = casting(rows)
            with provide_mxfp8(cast_mxfp8(rows)):
                assert torch.equal(fed(rows), expected), f"rows={count}"
            with pytest.raises(RuntimeError):
                fed(rows)
            with provide_mxfp8(cast_mxfp8(rows.clone())), pytest.raises(RuntimeError):
                fed(rows)


def test_slice_mxfp8_reuses_or_pads_rows(default_vllm_config, n=4096, k=1280):
    """``slice_mxfp8`` hands a split projection (such as ``indexer.wq_b``) the exact
    bits of casting the slice on its own, whether the slice is the full batch, an
    interior SP range with enough trailing rows, or a tail SP range that pads into a
    decode tile."""
    from torch import nn

    from vllm.models.deepseek_v4_1.tilelang.linear import (
        TileLangBlock32Rows,
        cast_mxfp8,
        provide_mxfp8,
        slice_mxfp8,
    )
    from vllm.v1.worker.workspace import use_preallocated_workspace

    weight, exps, _ = _block32_weight(n, k, 16)
    layer = nn.Module()
    layer.weight, layer.weight_scale_inv = weight, exps
    casting = TileLangBlock32Rows(layer, 0)
    casting.declare_kernels()
    fed = TileLangBlock32Rows(layer, 0)
    fed.tilelang_mxfp8_input = True
    fed.declare_kernels()
    x = torch.randn((200, k), generator=_gen(18), device=DEVICE).bfloat16()
    scratch = torch.empty(16 << 20, dtype=torch.uint8, device=DEVICE)
    with use_preallocated_workspace(scratch):
        full_cast = cast_mxfp8(x)
        assert slice_mxfp8(full_cast, 0, 200) is full_cast
        for first, last in ((0, 200), (0, 17), (32, 96), (64, 196), (180, 200)):
            sub = x[first:last]
            expected = casting(sub)
            sliced = slice_mxfp8(full_cast, first, last)
            with provide_mxfp8(sliced):
                assert torch.equal(fed(sub), expected), f"slice=({first}, {last})"


def test_block32_split_reads_its_producers_cast(default_vllm_config):
    """A split-K projection whose decode tiles divide ``DECODE_ROWS`` (DSpark's
    512 x 5120 context KV slice) reads a producer's ``MXFP8Rows`` cast bit for bit."""
    from torch import nn

    from vllm.models.deepseek_v4_1.tilelang.linear import (
        TileLangBlock32Rows,
        cast_mxfp8,
        provide_mxfp8,
    )
    from vllm.v1.worker.workspace import use_preallocated_workspace

    weight, exps, _ = _block32_weight(1792, 5120, 6)
    layer = nn.Module()
    layer.weight, layer.weight_scale_inv = weight, exps
    casting = TileLangBlock32Rows(layer, 1280)
    casting.declare_kernels()
    narrow = TileLangBlock32Rows(layer, 1280)  # 512 x 5120: splits K
    narrow.tilelang_mxfp8_input = True
    narrow.declare_kernels()
    x = torch.randn((300, 5120), generator=_gen(10), device=DEVICE).bfloat16()
    scratch = torch.empty(16 << 20, dtype=torch.uint8, device=DEVICE)
    with use_preallocated_workspace(scratch):
        for count in (1, 7, 16, 17, 32, 33, 64, 65, 96, 128, 129, 300):
            rows = x[:count]
            expected = casting(rows)
            with provide_mxfp8(cast_mxfp8(rows)):
                assert torch.equal(narrow(rows), expected), f"rows={count}"


# The router splits K eight ways; (100, 200) cannot split.
@pytest.mark.parametrize("n, k", [(384, 5120), (100, 200)])
@pytest.mark.parametrize("rows", [7, 65])
def test_bf16_gemm_nonfinite(n, k, rows):
    """NaNs and signed infinities land where the FP32 product puts them, through the
    split-K partials of decode rows too, and the other rows keep their bits."""
    gen = _gen(9)
    weight = torch.randn((n, k), generator=gen, device=DEVICE).bfloat16()
    x = torch.randn((rows, k), generator=gen, device=DEVICE).bfloat16()
    shards = bf16_shards(n, k)

    def project(a):
        out = torch.empty((rows, n), dtype=torch.float32, device=DEVICE)
        if shards > 1 and rows <= DECODE_ROWS:
            p = torch.empty((shards, rows, n), dtype=torch.float32, device=DEVICE)
            bf16_gemm_partials(n, k, shards, block_M=16)(a, weight, p)
            splitk_reduce(n, shards, out_dtype="float32")(p, out)
        else:
            bf16_gemm(n, k, out_dtype="float32", shards=shards)(a, weight, out)
        return out

    clean = project(x)
    poisoned = [1, 3, 4]
    x[1, 5], x[3, 7], x[4, 9] = NAN, INF, -INF
    out = project(x)
    assert_same_nonfinite(out, x.float() @ weight.float().T)
    assert_other_rows_equal(out, clean, poisoned)


def test_block32_rows_nan_row(default_vllm_config, n=1792, k=5120, start=1280):
    """A NaN in one row's activations survives the MXFP8 cast and reaches every output
    of that row, at split-K decode sizes and in the prefill GEMM, and no other row
    moves."""
    from torch import nn

    from vllm.models.deepseek_v4_1.tilelang.linear import TileLangBlock32Rows
    from vllm.v1.worker.workspace import use_preallocated_workspace

    weight, exps, _ = _block32_weight(n, k, 6)
    layer = nn.Module()
    layer.weight, layer.weight_scale_inv = weight, exps
    rows = TileLangBlock32Rows(layer, start)
    rows.declare_kernels()
    scratch = torch.empty(16 << 20, dtype=torch.uint8, device=DEVICE)
    for count in (7, 300):
        x = torch.randn((count, k), generator=_gen(count), device=DEVICE).bfloat16()
        with use_preallocated_workspace(scratch):
            clean = rows(x).clone()
            x[2, 11] = NAN
            out = rows(x)
        assert_rows_nan(out, [2], f"rows={count}: ")
        assert_other_rows_equal(out, clean, [2], f"rows={count}: ")


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


@pytest.mark.parametrize("width", [1280, 512])
def test_rmsnorm_of_a_fused_projections_half(width, rows=33, eps=1e-6):
    """The q and kv halves of ``fused_wqa_wkv``'s output (1280 + 512 columns),
    normalized where they lie, equal the norm of their contiguous copies bit for bit."""
    gen = _gen(5)
    fused = torch.randn((rows, 1280 + 512), generator=gen, device=DEVICE).bfloat16()
    half = fused[:, :1280] if width == 1280 else fused[:, 1280:]
    norm = TileLangRMSNorm(width, eps).to(DEVICE)
    norm.weight.data.copy_(torch.rand((width,), generator=gen, device=DEVICE) + 0.5)
    assert not half.is_contiguous()
    out = norm(half)
    assert out.is_contiguous()
    assert torch.equal(out.view(torch.int16), norm(half.contiguous()).view(torch.int16))


@pytest.mark.parametrize("n, k", sorted(TILE_CONTIGUOUS_FP8))
def test_tile_contiguous_fp8_matches_row_major(default_vllm_config, n, k):
    """Tile-contiguous [N // 128, K // 128, 128, 128] weights for Q-B (8192 x 1280)
    and indexer Q-B (4096 x 1280) match row-major [N, K] bit for bit across all decode
    tile heights and prefill."""
    from types import SimpleNamespace
    from torch import nn

    from vllm.models.deepseek_v4_1.tilelang.linear import TileLangFP8LinearMethod
    from vllm.v1.worker.workspace import use_preallocated_workspace

    weight, exps, _ = _block32_weight(n, k, 19)
    packed = pack_weight_tiles(weight)
    assert torch.equal(unpack_weight_tiles(packed, n, k), weight)

    weight_sf = pack_scale_words(exps)
    x = torch.randn((max(ROWS), k), generator=_gen(23), device=DEVICE).bfloat16()
    xq, sf, _ = _mxfp8(x)
    words = sf.contiguous().view(torch.uint32)

    ref_prefill = mxfp8_gemm(n, k, **fp8_prefill_config(n, k))
    tiled_prefill = mxfp8_gemm(n, k, **fp8_prefill_config(n, k), tile_contiguous=True)
    full_ref = torch.empty((max(ROWS), n), dtype=torch.bfloat16, device=DEVICE)
    full_tiled = torch.empty_like(full_ref)
    ref_prefill(xq, weight, words, weight_sf, full_ref)
    tiled_prefill(xq, packed, words, weight_sf, full_tiled)
    assert torch.equal(full_tiled, full_ref)

    for block_M in DECODE_TILE_ROWS:
        cfg = fp8_decode_config(n, k, block_M)
        k_ref = mxfp8_gemm_decode(n, k, **cfg, padded_rows=True)
        k_tiled = mxfp8_gemm_decode(
            n, k, **cfg, padded_rows=True, tile_contiguous=True
        )
        for rows in (1, 6, block_M):
            out_ref = torch.empty((rows, n), dtype=torch.bfloat16, device=DEVICE)
            out_tiled = torch.empty_like(out_ref)
            k_ref(xq[:DECODE_ROWS], weight, words[:DECODE_ROWS], weight_sf, out_ref)
            k_tiled(xq[:DECODE_ROWS], packed, words[:DECODE_ROWS], weight_sf, out_tiled)
            assert torch.equal(out_tiled, out_ref), f"block_M={block_M} rows={rows}"
            assert torch.equal(out_tiled, full_ref[:rows])

    layer = nn.Module()
    layer.weight = nn.Parameter(weight.clone(), requires_grad=False)
    layer.weight_scale_inv = nn.Parameter(exps.clone(), requires_grad=False)
    method = TileLangFP8LinearMethod(SimpleNamespace(weight_block_size=[32, 32]))
    method.process_weights_after_loading(layer)
    method.declare_layer_kernels(layer)
    assert getattr(layer, "tilelang_weight_packed", False) is True
    assert layer.weight.shape == (n, k)
    assert layer.tilelang_weight.shape == (n * (k // 128), 128)

    scratch = torch.empty(16 << 20, dtype=torch.uint8, device=DEVICE)
    with use_preallocated_workspace(scratch):
        for rows in ROWS:
            out = method.apply(layer, x[:rows])
            assert torch.equal(out, full_ref[:rows]), f"linear apply rows={rows}"


@pytest.mark.parametrize(
    "tp_size,part_n",
    [(2, 896), (3, 608), (4, 448), (5, 384), (6, 320), (7, 256), (8, 224)],
)
def test_sharded_wqa_wkv_and_strided_attention_front_match_replicated(tp_size, part_n):
    """Column-sharded target ``fused_wqa_wkv`` (448 rows/rank at TP4; 608 rows/rank
    padded to 1824 at TP3 with a zero-copy ``[:, :1792]`` strided slice) matches the
    replicated 1792-row projection and ``attention_front`` bit for bit."""
    from vllm.models.deepseek_v4_1.tilelang.front import attention_front
    from vllm.models.deepseek_v4_1.tilelang.linear import empty_mxfp8

    k = 5120
    weight, exps, _ = _block32_weight(1792, k, 29)
    weight_sf = pack_scale_words(exps)
    if tp_size * part_n > 1792:
        pad = tp_size * part_n - 1792
        w_pad = torch.cat(
            [
                weight,
                torch.zeros((pad, k), dtype=weight.dtype, device=DEVICE),
            ],
            dim=0,
        )
        s_pad = torch.cat(
            [
                exps,
                torch.full((pad // 32, exps.shape[1]), 0x7F, dtype=exps.dtype, device=DEVICE),
            ],
            dim=0,
        )
    else:
        w_pad, s_pad = weight, exps

    shards = [
        (
            w_pad[r * part_n : (r + 1) * part_n].contiguous(),
            pack_scale_words(s_pad[r * (part_n // 32) : (r + 1) * (part_n // 32)].contiguous()),
        )
        for r in range(tp_size)
    ]

    x = torch.randn((256, k), generator=_gen(31), device=DEVICE).bfloat16()
    xq, sf, _ = _mxfp8(x)
    words = sf.contiguous().view(torch.uint32)

    ref_prefill = mxfp8_gemm(1792, k, **fp8_prefill_config(1792, k))
    shard_prefill = mxfp8_gemm(part_n, k, **fp8_prefill_config(part_n, k))
    full_ref = torch.empty((256, 1792), dtype=torch.bfloat16, device=DEVICE)
    ref_prefill(xq, weight, words, weight_sf, full_ref)
    shard_outs = []
    for w_s, sf_s in shards:
        o = torch.empty((256, part_n), dtype=torch.bfloat16, device=DEVICE)
        shard_prefill(xq, w_s, words, sf_s, o)
        shard_outs.append(o)
    cat_prefill = torch.cat(shard_outs, dim=1)[:, :1792]
    assert torch.equal(cat_prefill, full_ref)

    for block_M in DECODE_TILE_ROWS:
        k_ref = mxfp8_gemm_decode(
            1792, k, **fp8_decode_config(1792, k, block_M), padded_rows=True
        )
        k_shard = mxfp8_gemm_decode(
            part_n, k, **fp8_decode_config(part_n, k, block_M), padded_rows=True
        )
        rows = min(6, block_M)
        out_ref = torch.empty((rows, 1792), dtype=torch.bfloat16, device=DEVICE)
        k_ref(xq[:DECODE_ROWS], weight, words[:DECODE_ROWS], weight_sf, out_ref)
        parts = []
        for w_s, sf_s in shards:
            o = torch.empty((rows, part_n), dtype=torch.bfloat16, device=DEVICE)
            k_shard(xq[:DECODE_ROWS], w_s, words[:DECODE_ROWS], sf_s, o)
            parts.append(o)
        cat_decode = torch.cat(parts, dim=1)[:, :1792]
        assert torch.equal(cat_decode, out_ref)

    # Verify attention_front on the strided TP3 view (stride 1824) matches contiguous.
    rows = 6
    gen = _gen(37)
    qw = (torch.rand((1280,), generator=gen, device=DEVICE) + 0.5).bfloat16()
    kvw = (torch.rand((512,), generator=gen, device=DEVICE) + 0.5).bfloat16()
    positions = torch.arange(rows, dtype=torch.int64, device=DEVICE)
    cos_sin = torch.randn((32, 64), generator=gen, device=DEVICE, dtype=torch.float32)
    slots = torch.arange(rows, dtype=torch.int64, device=DEVICE)
    cache_ref = torch.zeros((2, 128 * 528), dtype=torch.uint8, device=DEVICE)
    cache_shd = torch.zeros_like(cache_ref)
    qn_ref = torch.empty((rows, 1280), dtype=torch.bfloat16, device=DEVICE)
    qn_shd = torch.empty_like(qn_ref)
    kv_ref = torch.empty((rows, 512), dtype=torch.bfloat16, device=DEVICE)
    kv_shd = torch.empty_like(kv_ref)
    qc_ref = empty_mxfp8(qn_ref)
    qc_shd = empty_mxfp8(qn_shd)

    attention_front(
        out_ref, qw, kvw, 1e-6, positions, cos_sin, cache_ref, slots, 128,
        qn_ref, qc_ref.values, qc_ref.scales, kv_ref,
    )
    attention_front(
        cat_decode, qw, kvw, 1e-6, positions, cos_sin, cache_shd, slots, 128,
        qn_shd, qc_shd.values, qc_shd.scales, kv_shd,
    )
    assert torch.equal(qn_shd, qn_ref)
    assert torch.equal(kv_shd, kv_ref)
    assert torch.equal(qc_shd.values[:rows], qc_ref.values[:rows])
    assert torch.equal(qc_shd.scales[:rows], qc_ref.scales[:rows])
    assert torch.equal(cache_shd, cache_ref)



@pytest.mark.parametrize(
    "parts, part_n, out_dtype",
    [
        (2, 512, torch.float32),
        (2, 256, torch.float32),
        (2, 192, torch.float32),
        (2, 128, torch.float32),
        (2, 64, torch.float32),
        (1, 512, torch.bfloat16),
        (1, 256, torch.bfloat16),
        (1, 192, torch.bfloat16),
        (1, 128, torch.bfloat16),
        (1, 64, torch.bfloat16),
    ],
)
def test_packed_bf16_compressor_shapes_exact(parts, part_n, out_dtype, k=5120):
    """Selective 12-bit packed ``TileLangSplitLinearMethod`` matches uncompressed
    ``bf16_gemm_splitk`` and ``bf16_gemm``/``bf16_gemm_two`` bit for bit (0 ULP)
    across all replicated and TP-sharded compressor shapes, even with heavy BF16
    exponent exceptions and subnormals, while leaving ``TileLangLinearMethod`` and
    ``TileLangScaledLinearMethod`` uncompressed."""
    from torch import nn

    from vllm.models.deepseek_v4_1.tilelang.gemm import (
        SPLIT_BF16,
        bf16_gemm,
        bf16_gemm_splitk,
        bf16_gemm_two,
        packed_bf16_gemm,
        packed_bf16_gemm_two,
        packed_bf16_unpack,
    )
    from vllm.models.deepseek_v4_1.tilelang.linear import (
        TileLangLinearMethod,
        TileLangScaledLinearMethod,
        TileLangSplitLinearMethod,
        project_parts,
        unpack_bf16_layer,
    )
    from vllm.v1.worker.workspace import use_preallocated_workspace

    n = parts * part_n
    gen = _gen(41 + n + parts)
    weight = (torch.randn((n, k), generator=gen, device=DEVICE) * 0.02).bfloat16()
    # Inject out-of-window large, tiny, and subnormal BF16 values to exercise exceptions.
    weight[::7, ::97] = torch.tensor(64.0, dtype=torch.bfloat16, device=DEVICE)
    weight[1::5, 3::113] = torch.tensor(-1e-6, dtype=torch.bfloat16, device=DEVICE)
    weight[2::9, 5::127] = torch.tensor(1e-40, dtype=torch.bfloat16, device=DEVICE)

    layer = nn.Module()
    layer.weight = nn.Parameter(weight.clone(), requires_grad=False)
    layer.out_dtype = out_dtype
    method = TileLangSplitLinearMethod(parts)
    method.process_weights_after_loading(layer)
    assert hasattr(layer, "packed_bf16_exponent_base")
    assert layer.packed_bf16_exceptions > 0
    assert torch.equal(unpack_bf16_layer(layer), weight)
    method.declare_layer_kernels(layer)

    # Verify unpacked_bf16_unpack kernel also matches bit for bit.
    unpacked_gpu = torch.empty_like(weight)
    packed_bf16_unpack(n, k, layer.packed_bf16_exponent_base)(
        layer.weight,
        layer.weight_exception_entries,
        layer.weight_row_offsets,
        unpacked_gpu,
    )
    assert torch.equal(unpacked_gpu, weight)

    tl_dtype = "float32" if out_dtype == torch.float32 else "bfloat16"
    split_cfg = SPLIT_BF16[(n, k)]
    shards = split_cfg["shards"]
    blocked = split_cfg["blocked"]

    x = torch.randn((256, k), generator=gen, device=DEVICE).bfloat16()
    scratch = torch.empty(
        method.get_workspace_size(layer, 256), dtype=torch.uint8, device=DEVICE
    )

    if parts == 2:
        ref_prefill = bf16_gemm_two(
            part_n, k, out_dtype=tl_dtype, shards=shards, blocked=blocked
        )
        pack_prefill = packed_bf16_gemm_two(
            part_n,
            k,
            layer.packed_bf16_exponent_base,
            out_dtype=tl_dtype,
            shards=shards,
            blocked=blocked,
        )
        ref_full = [
            torch.empty((256, part_n), dtype=out_dtype, device=DEVICE),
            torch.empty((256, part_n), dtype=out_dtype, device=DEVICE),
        ]
        pack_full = [torch.empty_like(ref_full[0]), torch.empty_like(ref_full[1])]
        ref_prefill(x, weight, *ref_full)
        pack_prefill(
            x,
            layer.weight,
            layer.weight_exception_entries,
            layer.weight_row_offsets,
            *pack_full,
        )
        assert torch.equal(pack_full[0], ref_full[0])
        assert torch.equal(pack_full[1], ref_full[1])
    else:
        ref_prefill = bf16_gemm(
            n, k, out_dtype=tl_dtype, shards=shards, blocked=blocked
        )
        pack_prefill = packed_bf16_gemm(
            n,
            k,
            layer.packed_bf16_exponent_base,
            out_dtype=tl_dtype,
            shards=shards,
            blocked=blocked,
        )
        ref_full = [torch.empty((256, part_n), dtype=out_dtype, device=DEVICE)]
        pack_full = [torch.empty_like(ref_full[0])]
        ref_prefill(x, weight, ref_full[0])
        pack_prefill(
            x,
            layer.weight,
            layer.weight_exception_entries,
            layer.weight_row_offsets,
            pack_full[0],
        )
        assert torch.equal(pack_full[0], ref_full[0])

    for rows in (1, 7, 16, 18, 32, 36, 64, 128, 256):
        outs = [
            torch.empty((rows, part_n), dtype=out_dtype, device=DEVICE)
            for _ in range(parts)
        ]
        with use_preallocated_workspace(scratch):
            project_parts(layer, x[:rows], outs, rows)
        for p_idx in range(parts):
            assert torch.equal(outs[p_idx], ref_full[p_idx][:rows]), (
                f"parts={parts} part_n={part_n} rows={rows} part={p_idx}"
            )

    # Verify router.gate (TileLangLinearMethod) and indexer.weights_proj
    # (TileLangScaledLinearMethod) remain uncompressed BF16.
    router_layer = nn.Module()
    router_layer.weight = nn.Parameter(weight[:64].clone(), requires_grad=False)
    TileLangLinearMethod().process_weights_after_loading(router_layer)
    assert not hasattr(router_layer, "packed_bf16_exponent_base")
    assert router_layer.weight.dtype == torch.bfloat16

    scaled_layer = nn.Module()
    scaled_layer.weight = nn.Parameter(
        (torch.randn((32, k), generator=gen, device=DEVICE) * 0.02).bfloat16(),
        requires_grad=False,
    )
    TileLangScaledLinearMethod(0.5).process_weights_after_loading(scaled_layer)
    assert not hasattr(scaled_layer, "packed_bf16_exponent_base")
    assert scaled_layer.weight.dtype == torch.bfloat16


def test_mxfp8_gemm_rope_mxfp4_decode_matches_reference(default_vllm_config):
    """Fused ``project_block32_rope_mxfp4_decode`` on ``indexer.wq_b`` (4096 x 1280,
    32 heads x 128, trailing 64-dim GPT-J RoPE) matches ``wq_b`` + ``quantize_mxfp4``
    bit for bit across all decode tile heights (16, 32, 64) and ignores poisoned
    padding rows."""
    from torch import nn

    from vllm.models.deepseek_v4_1.tilelang.indexer import quantize_mxfp4
    from vllm.models.deepseek_v4_1.tilelang.linear import (
        TileLangBlock32Rows,
        cast_mxfp8,
        project_block32_rope_mxfp4_decode,
        provide_mxfp8,
    )
    from vllm.v1.worker.workspace import use_preallocated_workspace

    n, k, heads, head_dim, rope_dim = 4096, 1280, 32, 128, 64
    weight, exps, _ = _block32_weight(n, k, 41)
    layer = nn.Module()
    layer.weight, layer.weight_scale_inv = weight, exps
    proj = TileLangBlock32Rows(layer, 0)
    proj.tilelang_mxfp8_input = True
    proj.tilelang_rope_mxfp4 = (head_dim, rope_dim)
    proj.declare_kernels()

    gen = _gen(43)
    x_all = torch.randn((64, k), generator=gen, device=DEVICE).bfloat16()
    cos_sin = torch.randn((256, rope_dim), generator=gen, device=DEVICE, dtype=torch.float32)
    pos_all = torch.randint(0, 256, (64,), generator=gen, device=DEVICE, dtype=torch.int64)
    scratch = torch.empty(16 << 20, dtype=torch.uint8, device=DEVICE)

    with use_preallocated_workspace(scratch):
        for rows in (1, 2, 3, 4, 6, 8, 12, 16, 17, 24, 32, 33, 48, 64):
            x = x_all[:rows]
            pos = pos_all[:rows]
            provided = cast_mxfp8(x)
            # Poison padding rows past `rows` in the MXFP8 buffer.
            if rows < provided.values.shape[0]:
                provided.values[rows:].fill_(float("nan"))
                provided.scales[rows:].fill_(0xFF)
            with provide_mxfp8(provided):
                iq_ref = proj(x).view(rows, heads, head_dim)
            q_ref = torch.empty((rows, heads, head_dim // 2), dtype=torch.uint8, device=DEVICE)
            s_ref = torch.empty((rows, heads, head_dim // 32), dtype=torch.uint8, device=DEVICE)
            quantize_mxfp4(iq_ref, q_ref, s_ref, positions=pos, cos_sin=cos_sin)

            q_out = torch.empty_like(q_ref)
            s_out = torch.empty_like(s_ref)
            with provide_mxfp8(provided):
                project_block32_rope_mxfp4_decode(
                    proj, x, pos, cos_sin, q_out, s_out
                )
            assert torch.equal(q_out, q_ref), f"q mismatch at rows={rows}"
            assert torch.equal(s_out, s_ref), f"s mismatch at rows={rows}"



def test_hw_e2m1x2_codegen_in_gemm_and_indexer_quantizers():
    """Verify that ``mxfp8_gemm_rope_mxfp4_decode``, ``mxfp4_quantize_v41``, and
    ``mxfp4_norm_rope_quantize_v41`` lower MXFP4 byte packing through hardware
    ``cvt.rn.satfinite.e2m1x2.f32`` and vectorized stores rather than 7-comparison
    software emulation."""
    from vllm.models.deepseek_v4_1.tilelang.gemm import mxfp8_gemm_rope_mxfp4_decode
    from vllm.models.deepseek_v4_1.tilelang.indexer import (
        mxfp4_norm_rope_quantize_v41,
        mxfp4_quantize_v41,
    )

    gemm_src = mxfp8_gemm_rope_mxfp4_decode(
        4096, 1280, 128, 64, block_M=16, padded_rows=True, tile_contiguous=True
    ).get_kernel_source()
    assert "cvt.rn.satfinite.e2m1x2.f32" in gemm_src
    assert "dsv41_e2m1x2(" in gemm_src
    # The epilogue row loop stays unforced so nvcc can unroll it (decode M=1..16).
    assert "#pragma unroll 1" not in gemm_src

    for q_kernel in (
        mxfp4_quantize_v41(paged=False),
        mxfp4_quantize_v41(paged=False, rope_heads=32, rope_dim=64),
        mxfp4_quantize_v41(paged=True, page_size=64),
    ):
        src = q_kernel.get_kernel_source()
        assert "cvt.rn.satfinite.e2m1x2.f32" in src
        assert "dsv41_store_e2m1x32(" in src

    for nrq_kernel in (
        mxfp4_norm_rope_quantize_v41(1e-6, page_size=64, shards=1),
        mxfp4_norm_rope_quantize_v41(1e-6, page_size=64, shards=6),
    ):
        src = nrq_kernel.get_kernel_source()
        assert "cvt.rn.satfinite.e2m1x2.f32" in src
        assert "dsv41_store_e2m1x8(" in src
