#!/usr/bin/env python3
"""2-boot hardware benchmark for post-load weight preparation, cache hit/spot-check,
API/tokenizer startup setup, and first-request prefill latency on GB10 (SM121)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from vllm.models.deepseek_v4_1.tilelang.gemm import (
    TILE_CONTIGUOUS_FP8,
    TILE_WEIGHT_N,
    TILE_WEIGHT_K,
    pack_weight_tiles,
)
from vllm.models.deepseek_v4_1.tilelang.moe import (
    FP4_TMA_K,
    MoeExpertsV41Plan,
    tile_codes,
)
from vllm.models.deepseek_v4_1.tilelang.postload_cache import (
    configure_postload_cache,
    get_postload_cache_stats,
    loading_done,
    reset_postload_cache_stats,
)
from vllm.models.deepseek_v4_1.tilelang.scale_codes import (
    narrow_code_offsets,
    overlay_byte_codes,
    tile_byte_codes,
)
from vllm.models.deepseek_v4_1.tilelang.vocab import (
    GROUP,
    WINDOW,
    _PACK_BLOCK_ROWS,
    PackedBf16Weight,
    pack_bf16_weight,
    packed_bf16_exception_count,
    packed_bf16_histogram,
    packed_bf16_pack_rows,
)


def _baseline_tile_expert_weights(b_packed: torch.Tensor, block_N: int = 128):
    E, N, half_k = b_packed.shape
    k_full, tail = divmod(2 * half_k, FP4_TMA_K)
    b = b_packed.view(E, N // block_N, block_N, half_k)
    full = b[..., : k_full * FP4_TMA_K // 2]
    tiles = (
        full.reshape(E, N // block_N, block_N, k_full, FP4_TMA_K // 2)
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, block_N, FP4_TMA_K // 2)
        .contiguous()
    )
    if not tail:
        return tiles, None
    tails = b[..., k_full * FP4_TMA_K // 2 :].reshape(-1, block_N, tail // 2)
    return tiles, tails.contiguous().view(torch.uint8)


def _baseline_row_deltas(
    exponents: torch.Tensor, max_delta: int
) -> tuple[torch.Tensor, torch.Tensor]:
    e = exponents.view(torch.uint8)
    nonzero = e > 0
    min_nonzero = torch.where(nonzero, e, torch.full_like(e, 255)).amin(dim=-1)
    base = torch.where(
        nonzero.any(dim=-1), min_nonzero, torch.zeros_like(min_nonzero)
    ).contiguous()
    deltas = torch.where(nonzero, e - base.unsqueeze(-1), torch.zeros_like(e))
    span = int(deltas.max()) if deltas.numel() else 0
    if span > max_delta:
        bits = 2 if max_delta == 3 else 4
        raise ValueError(
            f"a row's scale exponents span {span + 1} values; {bits}-bit codes hold {max_delta + 1}"
        )
    rem = (-deltas.shape[-1]) % 4
    if rem:
        deltas = torch.nn.functional.pad(deltas, (0, rem))
    return deltas.reshape(*deltas.shape[:-1], -1, 4), base


def _baseline_pack_nibbles(d: torch.Tensor) -> torch.Tensor:
    return (
        d[..., 0].to(torch.int32)
        | (d[..., 1].to(torch.int32) << 4)
        | (d[..., 2].to(torch.int32) << 8)
        | (d[..., 3].to(torch.int32) << 12)
    ).to(torch.uint16).contiguous()


def _baseline_pack_crumbs(d: torch.Tensor) -> torch.Tensor:
    return (
        d[..., 0] | (d[..., 1] << 2) | (d[..., 2] << 4) | (d[..., 3] << 6)
    ).contiguous()


def _baseline_moe_prepare_weight(
    w: torch.Tensor,
    scale: torch.Tensor,
    block_N: int = 128,
):
    d, base = _baseline_row_deltas(scale, 15)
    expert_delta = d.reshape(d.shape[0], -1).amax(dim=1)
    narrow = (expert_delta <= 3).to(torch.uint8).contiguous()
    codes_raw = _baseline_pack_nibbles(d)
    codes_2bit_raw = torch.zeros(codes_raw.shape, dtype=torch.uint8, device=codes_raw.device)
    mask = narrow.bool()
    if mask.any().item():
        codes_2bit_raw[mask] = _baseline_pack_crumbs(d[mask])
    codes, codes_2bit = overlay_byte_codes(
        tile_codes(codes_raw, block_N),
        tile_byte_codes(codes_2bit_raw, block_N),
        narrow,
    )
    tiles, tails = _baseline_tile_expert_weights(w.view(torch.int8), block_N)
    return tiles, tails, codes, codes_2bit, narrow_code_offsets(narrow), base


def _baseline_pack_bf16_weight_in_place(
    weight: torch.Tensor,
) -> PackedBf16Weight:
    out_features, in_features = map(int, weight.shape)
    width = in_features + in_features // 2
    device = weight.device
    bits = weight.view(torch.uint16)
    histogram = torch.zeros(256, dtype=torch.int32, device=device)
    packed_bf16_histogram(in_features)(bits, histogram)
    hist = histogram.to(torch.int64).cpu()
    hist[0] = 0
    windows = hist.unfold(0, WINDOW, 1).sum(1)
    windows[0] = -1
    base = int(windows.argmax())
    row_offsets = torch.zeros(out_features + 1, dtype=torch.int32, device=device)
    packed_bf16_exception_count(in_features)(bits, row_offsets[1:], base)
    row_offsets[1:] = torch.cumsum(row_offsets[1:], 0, dtype=torch.int32)
    exceptions = int(row_offsets[-1])
    rows = torch.zeros(max(exceptions, 1), dtype=torch.int32, device=device)
    entries = torch.zeros_like(rows)
    mismatches = torch.zeros(1, dtype=torch.int32, device=device)
    pack_kernel = packed_bf16_pack_rows(in_features)
    values = (
        weight.view(torch.uint8)
        .view(-1)[: out_features * width]
        .view(out_features, width)
    )
    staging = torch.empty(
        (min(_PACK_BLOCK_ROWS, out_features), width),
        dtype=torch.uint8,
        device=device,
    )
    for start in range(0, out_features, _PACK_BLOCK_ROWS):
        count = min(_PACK_BLOCK_ROWS, out_features - start)
        pack_kernel(
            bits,
            staging[:count],
            row_offsets,
            rows,
            entries,
            mismatches,
            base,
            start,
        )
        values[start : start + count].copy_(staging[:count])
    del staging
    return PackedBf16Weight(values, base, rows, entries, row_offsets, exceptions)


def _packed_bf16_eq(a: PackedBf16Weight, b: PackedBf16Weight) -> bool:
    exc = a.exceptions
    return (
        a.exponent_base == b.exponent_base
        and a.exceptions == b.exceptions
        and torch.equal(a.values, b.values)
        and torch.equal(a.row_offsets, b.row_offsets)
        and torch.equal(a.exception_rows[:exc], b.exception_rows[:exc])
        and torch.equal(a.exception_entries[:exc], b.exception_entries[:exc])
    )


def _baseline_pack_weight_tiles(w: torch.Tensor, tile_n: int) -> torch.Tensor:
    n, k = w.shape
    return (
        w.view(n // tile_n, tile_n, k // TILE_WEIGHT_K, TILE_WEIGHT_K)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(n * (k // TILE_WEIGHT_K), TILE_WEIGHT_K)
    )


def _baseline_unpack_weight_tiles(
    w_packed: torch.Tensor, n: int, k: int, tile_n: int
) -> torch.Tensor:
    return (
        w_packed.view(n // tile_n, k // TILE_WEIGHT_K, tile_n, TILE_WEIGHT_K)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(n, k)
    )


def _make_realistic_moe_exponents(
    num_experts: int, rows: int, groups: int, device: torch.device, seed: int
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    base = torch.randint(118, 126, (num_experts, rows, 1), generator=gen, device=device, dtype=torch.int16)
    jitter = torch.randint(0, 4, (num_experts, rows, groups), generator=gen, device=device, dtype=torch.int16)
    exponents = (base + jitter).clamp(1, 254).to(torch.uint8)
    # 5% of experts have wide exponent span (4..9) matching mixed narrow/wide checkpoints
    wide_experts = torch.rand((num_experts, 1, 1), generator=gen, device=device) < 0.05
    wide_jitter = torch.randint(0, 8, (num_experts, rows, groups), generator=gen, device=device, dtype=torch.int16)
    exponents = torch.where(wide_experts, (base + wide_jitter).clamp(1, 254).to(torch.uint8), exponents)
    return exponents.contiguous()


def bench_moe_postload_58_layers(device: torch.device) -> dict:
    num_layers = 58
    num_experts = 128
    w13_n, w13_k_packed, w13_groups = 1536, 3584, 224
    w2_n, w2_k_packed, w2_groups = 7168, 384, 24

    gen = torch.Generator(device=device)
    gen.manual_seed(20261108)
    w13_template = torch.randint(
        0, 256, (num_experts, w13_n, w13_k_packed), generator=gen, device=device, dtype=torch.uint8
    )
    w13_scale_template = _make_realistic_moe_exponents(
        num_experts, w13_n, w13_groups, device, seed=101
    )
    w2_template = torch.randint(
        0, 256, (num_experts, w2_n, w2_k_packed), generator=gen, device=device, dtype=torch.uint8
    )
    w2_scale_template = _make_realistic_moe_exponents(
        num_experts, w2_n, w2_groups, device, seed=202
    )

    plan = MoeExpertsV41Plan(hidden=7168, intermediate=768, num_experts=128, topk=8)
    configure_postload_cache(None)
    _ = _baseline_moe_prepare_weight(w13_template, w13_scale_template)
    _ = _baseline_moe_prepare_weight(w2_template, w2_scale_template)
    _ = plan.prepare_weight(w13_template, w13_scale_template)
    _ = plan.prepare_weight(w2_template, w2_scale_template)
    torch.cuda.synchronize(device)

    bw13 = _baseline_moe_prepare_weight(w13_template, w13_scale_template)
    cw13 = plan.prepare_weight(w13_template, w13_scale_template)
    bw2 = _baseline_moe_prepare_weight(w2_template, w2_scale_template)
    cw2 = plan.prepare_weight(w2_template, w2_scale_template)

    def _tuple_eq(a, b) -> bool:
        return all((x is None and y is None) or torch.equal(x, y) for x, y in zip(a, b))

    bit_identical_cold = _tuple_eq(bw13, cw13) and _tuple_eq(bw2, cw2)
    assert bit_identical_cold, "MoE cold preparation output diverged from baseline!"

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(num_layers):
        _ = _baseline_moe_prepare_weight(w13_template, w13_scale_template)
        _ = _baseline_moe_prepare_weight(w2_template, w2_scale_template)
    torch.cuda.synchronize(device)
    baseline_58_compute_s = time.perf_counter() - t0

    dummy_blocks = [torch.empty((16 * 1024 * 1024,), dtype=torch.uint8, device=device) for _ in range(8)]
    for b in dummy_blocks[::2]:
        del b
    torch.cuda.synchronize(device)
    t_ec0 = time.perf_counter()
    for _ in range(num_layers):
        torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    empty_cache_58_s = time.perf_counter() - t_ec0
    del dummy_blocks

    with tempfile.TemporaryDirectory(dir="/cache/spark-ds41f", prefix="postload_bench_moe_") as tmpdir:
        ckpt_dir = Path(tmpdir) / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "model-00001-of-00001.safetensors").write_bytes(b"x" * 4096)
        cache_dir = Path(tmpdir) / "cache"

        reset_postload_cache_stats()
        configure_postload_cache(
            cache_dir,
            checkpoint_path=ckpt_dir,
            tp_rank=0,
            tp_world_size=3,
            min_free_bytes=1024,
        )
        try:
            torch.cuda.synchronize(device)
            t_cold0 = time.perf_counter()
            for layer_idx in range(num_layers):
                _ = plan.prepare_weight(
                    w13_template,
                    w13_scale_template,
                    cache_tag=f"layer.{layer_idx}.experts:0",
                )
                _ = plan.prepare_weight(
                    w2_template,
                    w2_scale_template,
                    cache_tag=f"layer.{layer_idx}.experts:1",
                )
            torch.cuda.synchronize(device)
            candidate_cold_compute_s = time.perf_counter() - t_cold0
            t_flush0 = time.perf_counter()
            loading_done()
            bg_flush_s = time.perf_counter() - t_flush0
        finally:
            configure_postload_cache(None)

        reset_postload_cache_stats()
        configure_postload_cache(
            cache_dir,
            checkpoint_path=ckpt_dir,
            tp_rank=0,
            tp_world_size=3,
            min_free_bytes=1024,
        )
        try:
            torch.cuda.synchronize(device)
            t_warm0 = time.perf_counter()
            for layer_idx in range(num_layers):
                hw13 = plan.prepare_weight(
                    w13_template,
                    w13_scale_template,
                    cache_tag=f"layer.{layer_idx}.experts:0",
                )
                hw2 = plan.prepare_weight(
                    w2_template,
                    w2_scale_template,
                    cache_tag=f"layer.{layer_idx}.experts:1",
                )
            torch.cuda.synchronize(device)
            candidate_cached_s = time.perf_counter() - t_warm0
            warm_stats = get_postload_cache_stats()
        finally:
            configure_postload_cache(None)

        bit_identical_cached = (
            _tuple_eq(bw13, hw13)
            and _tuple_eq(bw2, hw2)
            and warm_stats.hits == num_layers * 2
            and warm_stats.misses == 0
        )
        assert bit_identical_cached, f"MoE cached preparation mismatch or miss: {warm_stats}"

    return {
        "num_layers": num_layers,
        "tp3_shapes": {
            "w13": [num_experts, w13_n, w13_k_packed],
            "w2": [num_experts, w2_n, w2_k_packed],
        },
        "baseline_58_layers_compute_s": round(baseline_58_compute_s, 4),
        "baseline_58_layers_empty_cache_s": round(empty_cache_58_s, 4),
        "baseline_58_layers_total_s": round(baseline_58_compute_s + empty_cache_58_s, 4),
        "candidate_cold_58_layers_s": round(candidate_cold_compute_s, 4),
        "candidate_cold_bg_flush_s": round(bg_flush_s, 4),
        "candidate_cached_58_layers_s": round(candidate_cached_s, 4),
        "speedup_cold_vs_baseline_compute": round(baseline_58_compute_s / candidate_cold_compute_s, 2),
        "speedup_cached_vs_baseline_compute": round(baseline_58_compute_s / candidate_cached_s, 2),
        "bit_identical_cold": bit_identical_cold,
        "bit_identical_cached": bit_identical_cached,
    }


def bench_pack12_weights(device: torch.device) -> dict:
    lm_m, lm_k = 43264, 7168
    kv_m, kv_k = 2432, 7168
    num_kv_layers = 59

    gen = torch.Generator(device=device)
    gen.manual_seed(20261109)
    lm_template = (torch.randn((lm_m, lm_k), generator=gen, device=device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    kv_template = (torch.randn((kv_m, kv_k), generator=gen, device=device, dtype=torch.float32) * 0.02).to(torch.bfloat16)

    configure_postload_cache(None)
    _ = _baseline_pack_bf16_weight_in_place(kv_template.clone())
    _ = pack_bf16_weight(kv_template.clone(), in_place=True, use_cache=False)
    torch.cuda.synchronize(device)

    lm_base_in = lm_template.clone()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    alloc_before_base = torch.cuda.memory_allocated(device)
    t0 = time.perf_counter()
    b_res = _baseline_pack_bf16_weight_in_place(lm_base_in)
    torch.cuda.synchronize(device)
    lm_baseline_ms = (time.perf_counter() - t0) * 1000.0
    lm_baseline_peak_temp_mib = (torch.cuda.max_memory_allocated(device) - alloc_before_base) / (1024 * 1024)

    lm_cand_in = lm_template.clone()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    alloc_before_cand = torch.cuda.memory_allocated(device)
    t1 = time.perf_counter()
    c_res = pack_bf16_weight(lm_cand_in, in_place=True, use_cache=False)
    torch.cuda.synchronize(device)
    lm_candidate_cold_ms = (time.perf_counter() - t1) * 1000.0
    lm_candidate_peak_temp_mib = (torch.cuda.max_memory_allocated(device) - alloc_before_cand) / (1024 * 1024)

    lm_bit_identical_cold = _packed_bf16_eq(b_res, c_res)
    assert lm_bit_identical_cold, "lm_head cold in-place pack12 diverged from baseline!"

    kv_inputs_base = [kv_template.clone() for _ in range(num_kv_layers)]
    lm_input_base = lm_template.clone()
    torch.cuda.synchronize(device)
    t_b0 = time.perf_counter()
    for w in kv_inputs_base:
        _ = _baseline_pack_bf16_weight_in_place(w)
    _ = _baseline_pack_bf16_weight_in_place(lm_input_base)
    torch.cuda.synchronize(device)
    total_baseline_ms = (time.perf_counter() - t_b0) * 1000.0
    del kv_inputs_base, lm_input_base

    with tempfile.TemporaryDirectory(dir="/cache/spark-ds41f", prefix="postload_bench_pack12_") as tmpdir:
        ckpt_dir = Path(tmpdir) / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "model-00001-of-00001.safetensors").write_bytes(b"y" * 4096)
        cache_dir = Path(tmpdir) / "cache"

        kv_inputs_cold = [kv_template.clone() for _ in range(num_kv_layers)]
        lm_input_cold = lm_template.clone()
        reset_postload_cache_stats()
        configure_postload_cache(
            cache_dir,
            checkpoint_path=ckpt_dir,
            tp_rank=0,
            tp_world_size=3,
            min_free_bytes=1024,
        )
        try:
            torch.cuda.synchronize(device)
            t_c0 = time.perf_counter()
            for idx, w in enumerate(kv_inputs_cold):
                _ = pack_bf16_weight(w, in_place=True, cache_tag=f"layer.{idx}.wq_a_kv", use_cache=True)
            _ = pack_bf16_weight(lm_input_cold, in_place=True, cache_tag="lm_head", use_cache=True)
            torch.cuda.synchronize(device)
            total_candidate_cold_ms = (time.perf_counter() - t_c0) * 1000.0
            loading_done()
        finally:
            configure_postload_cache(None)
        del kv_inputs_cold, lm_input_cold

        kv_inputs_warm = [kv_template.clone() for _ in range(num_kv_layers)]
        lm_input_warm = lm_template.clone()
        reset_postload_cache_stats()
        configure_postload_cache(
            cache_dir,
            checkpoint_path=ckpt_dir,
            tp_rank=0,
            tp_world_size=3,
            min_free_bytes=1024,
        )
        try:
            torch.cuda.synchronize(device)
            t_w0 = time.perf_counter()
            for idx, w in enumerate(kv_inputs_warm):
                _ = pack_bf16_weight(w, in_place=True, cache_tag=f"layer.{idx}.wq_a_kv", use_cache=True)
            h_res = pack_bf16_weight(lm_input_warm, in_place=True, cache_tag="lm_head", use_cache=True)
            torch.cuda.synchronize(device)
            total_candidate_cached_ms = (time.perf_counter() - t_w0) * 1000.0
            warm_stats = get_postload_cache_stats()
        finally:
            configure_postload_cache(None)

        lm_bit_identical_cached = (
            _packed_bf16_eq(b_res, h_res)
            and warm_stats.hits == num_kv_layers + 1
            and warm_stats.misses == 0
        )
        assert lm_bit_identical_cached, f"pack12 cached mismatch or miss: {warm_stats}"
        del kv_inputs_warm, lm_input_warm

    return {
        "lm_head_shape": [lm_m, lm_k],
        "wq_a_kv_shape": [kv_m, kv_k],
        "num_kv_layers": num_kv_layers,
        "lm_head_baseline_ms": round(lm_baseline_ms, 2),
        "lm_head_candidate_cold_ms": round(lm_candidate_cold_ms, 2),
        "lm_head_baseline_peak_temp_mib": round(lm_baseline_peak_temp_mib, 2),
        "lm_head_candidate_peak_temp_mib": round(lm_candidate_peak_temp_mib, 2),
        "lm_head_temp_memory_saved_mib": round(lm_baseline_peak_temp_mib - lm_candidate_peak_temp_mib, 2),
        "total_60_layers_baseline_ms": round(total_baseline_ms, 2),
        "total_60_layers_candidate_cold_ms": round(total_candidate_cold_ms, 2),
        "total_60_layers_candidate_cached_ms": round(total_candidate_cached_ms, 2),
        "bit_identical_cold": lm_bit_identical_cold,
        "bit_identical_cached": lm_bit_identical_cached,
    }


def bench_tp3_rank2_wq_b_trim(device: torch.device) -> dict:
    num_layers = 59
    padded_n, trimmed_n, k = 12288, 8192, 1280
    assert (padded_n, k) in TILE_CONTIGUOUS_FP8 and (trimmed_n, k) in TILE_CONTIGUOUS_FP8
    tile_n = TILE_WEIGHT_N
    w_raw = torch.randint(0, 256, (padded_n, k), device=device, dtype=torch.uint8).view(torch.float8_e4m3fn)
    w_packed_template = pack_weight_tiles(w_raw, tile_n)

    _ = _baseline_pack_weight_tiles(
        _baseline_unpack_weight_tiles(w_packed_template, padded_n, k, tile_n)[:trimmed_n].contiguous(),
        tile_n,
    )
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(num_layers):
        unpacked = _baseline_unpack_weight_tiles(w_packed_template, padded_n, k, tile_n)
        trimmed = unpacked[:trimmed_n].contiguous()
        b_out = _baseline_pack_weight_tiles(trimmed, tile_n)
    torch.cuda.synchronize(device)
    baseline_59_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    for _ in range(num_layers):
        c_out = w_packed_template[: trimmed_n * (k // TILE_WEIGHT_K)]
    torch.cuda.synchronize(device)
    candidate_59_ms = (time.perf_counter() - t1) * 1000.0

    bit_identical = torch.equal(b_out.view(torch.uint8), c_out.view(torch.uint8))
    assert bit_identical, "wq_b direct tile slice diverged from unpack+slice+repack!"

    return {
        "num_layers": num_layers,
        "padded_shape": [padded_n, k],
        "trimmed_shape": [trimmed_n, k],
        "baseline_59_layers_ms": round(baseline_59_ms, 3),
        "candidate_59_layers_ms": round(candidate_59_ms, 4),
        "bit_identical": bit_identical,
    }


def bench_api_startup_and_first_prefill(device: torch.device) -> dict:
    cand_cmd = [
        sys.executable,
        "-c",
        (
            "import time; t0 = time.perf_counter(); "
            "from vllm.model_executor.models.registry import ModelRegistry; "
            "cls = ModelRegistry._try_load_model_cls('DeepseekV41ForCausalLM'); "
            "import sys; "
            "print(f'{(time.perf_counter() - t0)*1000.0:.2f},{int(\"tilelang\" in sys.modules)}')"
        ),
    ]
    base_cmd = [
        sys.executable,
        "-c",
        (
            "import time; t0 = time.perf_counter(); "
            "import vllm.models.deepseek_v4_1.nvidia.model; "
            "from vllm.model_executor.models.registry import ModelRegistry; "
            "cls = ModelRegistry._try_load_model_cls('DeepseekV41ForCausalLM'); "
            "import sys; "
            "print(f'{(time.perf_counter() - t0)*1000.0:.2f},{int(\"tilelang\" in sys.modules)}')"
        ),
    ]
    base_out = subprocess.check_output(base_cmd, text=True).strip().splitlines()[-1].split(",")
    cand_out = subprocess.check_output(cand_cmd, text=True).strip().splitlines()[-1].split(",")
    resolve_baseline_ms = float(base_out[0])
    resolve_candidate_ms = float(cand_out[0])
    cand_tilelang_loaded = bool(int(cand_out[1]))

    from vllm.models.deepseek_v4_1 import sp_prefill

    num_layers = 59
    ptrs_per_layer = 6
    layer_ptr_lists = [
        [int(torch.empty((64,), dtype=torch.uint8, device=device).data_ptr()) for _ in range(ptrs_per_layer)]
        for _ in range(num_layers)
    ]
    sp_prefill._SIDE_STREAMS.clear()
    torch.cuda.synchronize(device)
    t_first0 = time.perf_counter()
    _ = sp_prefill._side_stream(device)
    l2_tables = []
    for ptrs in layer_ptr_lists:
        l2_tables.append((
            torch.tensor(ptrs, dtype=torch.int64, device=device),
            torch.tensor([2 * 1024 * 1024] * len(ptrs), dtype=torch.int32, device=device),
        ))
    torch.cuda.synchronize(device)
    first_request_lazy_setup_ms = (time.perf_counter() - t_first0) * 1000.0

    t_second0 = time.perf_counter()
    _ = sp_prefill._side_stream(device)
    _ = l2_tables[0]
    torch.cuda.synchronize(device)
    prewarmed_request_setup_ms = (time.perf_counter() - t_second0) * 1000.0

    return {
        "resolve_model_cls_baseline_ms": round(resolve_baseline_ms, 2),
        "resolve_model_cls_candidate_ms": round(resolve_candidate_ms, 2),
        "resolve_model_cls_speedup": round(resolve_baseline_ms / max(resolve_candidate_ms, 0.01), 2),
        "candidate_imports_tilelang_on_resolve": cand_tilelang_loaded,
        "first_prefill_lazy_l2_and_stream_setup_ms": round(first_request_lazy_setup_ms, 3),
        "prewarmed_prefill_setup_ms": round(prewarmed_request_setup_ms, 4),
    }


def main() -> None:
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    moe_res = bench_moe_postload_58_layers(device)
    pack12_res = bench_pack12_weights(device)
    wqb_res = bench_tp3_rank2_wq_b_trim(device)
    startup_res = bench_api_startup_and_first_prefill(device)

    total_postload_baseline_s = (
        moe_res["baseline_58_layers_total_s"]
        + pack12_res["total_60_layers_baseline_ms"] / 1000.0
        + wqb_res["baseline_59_layers_ms"] / 1000.0
    )
    total_postload_candidate_cold_s = (
        moe_res["candidate_cold_58_layers_s"]
        + moe_res["candidate_cold_bg_flush_s"]
        + pack12_res["total_60_layers_candidate_cold_ms"] / 1000.0
        + wqb_res["candidate_59_layers_ms"] / 1000.0
    )
    total_postload_candidate_cached_s = (
        moe_res["candidate_cached_58_layers_s"]
        + pack12_res["total_60_layers_candidate_cached_ms"] / 1000.0
        + wqb_res["candidate_59_layers_ms"] / 1000.0
    )

    report = {
        "device_name": props.name,
        "sm": f"{props.major}.{props.minor}",
        "moe_58_layers": moe_res,
        "pack12_60_layers": pack12_res,
        "tp3_rank2_wq_b_59_layers": wqb_res,
        "api_startup_and_first_prefill": startup_res,
        "summary": {
            "total_postload_baseline_s": round(total_postload_baseline_s, 4),
            "total_postload_candidate_cold_s": round(total_postload_candidate_cold_s, 4),
            "total_postload_candidate_cached_s": round(total_postload_candidate_cached_s, 4),
            "cold_postload_speedup": round(total_postload_baseline_s / total_postload_candidate_cold_s, 2),
            "cached_postload_speedup": round(total_postload_baseline_s / total_postload_candidate_cached_s, 2),
            "postload_saved_cold_s": round(total_postload_baseline_s - total_postload_candidate_cold_s, 4),
            "postload_saved_cached_s": round(total_postload_baseline_s - total_postload_candidate_cached_s, 4),
            "startup_resolve_saved_s": round(
                (startup_res["resolve_model_cls_baseline_ms"] - startup_res["resolve_model_cls_candidate_ms"]) / 1000.0,
                4,
            ),
        },
    }
    out = json.dumps(report, indent=2)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
