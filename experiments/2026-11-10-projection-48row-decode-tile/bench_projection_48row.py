#!/usr/bin/env python3
"""Benchmark 48-row vs 64-row decode projection tiles across TP4 and TP3 shapes
for M in (36, 42, 48) rows (c6..c8 speculative decode verification)."""

import json
import sys
import torch
import tile_kernels

from vllm.models.deepseek_v4_1.tilelang.gemm import (
    DECODE_TILE_ROWS,
    SPLIT_FP8,
    TILE_CONTIGUOUS_FP8,
    bf16_gemm_partials,
    decode_tile_rows,
    fp8_decode_config,
    mxfp8_gemm_decode,
    mxfp8_gemm_partials,
    pack_scale_words,
    pack_weight_tiles,
    split_bucket,
    split_row_tile,
)
from vllm.models.deepseek_v4_1.tilelang.linear import PARTIAL_ROW_TILES


def _bench_us(fn, warmup: int = 25, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def main() -> None:
    boot = sys.argv[1] if len(sys.argv) > 1 else "boot1"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(20261110)

    assert DECODE_TILE_ROWS == (16, 32, 48, 64)
    assert PARTIAL_ROW_TILES == (16, 32, 48, 64)
    for m in (33, 36, 42, 48):
        assert decode_tile_rows(m) == 48
        assert split_bucket(m) == 48
        assert split_row_tile(m) == 48

    shapes = [
        ("wq_b_tp4", 8192, 1280, 1, 61),
        ("wq_b_tp3", 12288, 1280, 1, 61),
        ("indexer_wq_b", 4096, 1280, 1, 61),
        ("fused_wqa_wkv_full", 1792, 5120, 1, 1),
        ("fused_wqa_wkv_tp4", 448, 5120, 1, 61),
        ("fused_wqa_wkv_tp3", 608, 5120, 1, 61),
        ("shared_gate_up_tp4", 1152, 5120, 1, 61),
        ("shared_gate_up_tp3", 1536, 5120, 1, 61),
        ("shared_down_tp4", 5120, 576, 1, 61),
        ("shared_down_tp3", 5120, 768, 1, 61),
        ("wo_a_tp4", 1024, 4096, 2, 61),
        ("wo_a_tp3", 1024, 4096, 3, 61),
        ("wo_b_tp4", 5120, 2048, 1, 61),
        ("wo_b_tp3", 5120, 3072, 1, 61),
        ("dspark_eh_tp4", 1280, 15360, 1, 1),
    ]

    records = []
    for name, n, k, groups, count_in_step in shapes:
        total_n = n * groups
        w = (torch.randn((total_n, k), generator=gen, device=device) * 2).to(
            torch.float8_e4m3fn
        )
        exps = torch.randint(
            120, 125, (total_n // 32, k // 32), generator=gen, device=device
        ).to(torch.uint8)
        wsf = pack_scale_words(exps)
        tile_contig = groups == 1 and (n, k) in TILE_CONTIGUOUS_FP8
        w_in = pack_weight_tiles(w) if tile_contig else w

        x = torch.randn(
            (64, k * groups), generator=gen, device=device, dtype=torch.bfloat16
        )
        xq, sf = tile_kernels.quant.per_token_cast(
            x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True
        )
        sf_u32 = sf.contiguous().view(torch.uint32)

        cfg64 = fp8_decode_config(n, k, 64, groups)
        cfg48 = fp8_decode_config(n, k, 48, groups)
        k64 = mxfp8_gemm_decode(
            n, k, **cfg64, padded_rows=True, groups=groups, tile_contiguous=tile_contig
        )
        k48 = mxfp8_gemm_decode(
            n, k, **cfg48, padded_rows=True, groups=groups, tile_contiguous=tile_contig
        )

        for m in (36, 42, 48):
            out64 = torch.empty((m, total_n), dtype=torch.bfloat16, device=device)
            out48 = torch.empty((m, total_n), dtype=torch.bfloat16, device=device)
            k64(xq, w_in, sf_u32, wsf, out64)
            k48(xq, w_in, sf_u32, wsf, out48)
            assert torch.equal(out48, out64), f"Bit mismatch on {name} at M={m}"

        m = 42
        out = torch.empty((m, total_n), dtype=torch.bfloat16, device=device)
        t64 = _bench_us(lambda: k64(xq, w_in, sf_u32, wsf, out))
        t48 = _bench_us(lambda: k48(xq, w_in, sf_u32, wsf, out))
        rec = {
            "boot": boot,
            "name": name,
            "shape": [n, k, groups],
            "m": m,
            "layers": count_in_step,
            "baseline_bm64_us": round(t64, 3),
            "candidate_bm48_us": round(t48, 3),
            "speedup": round(t64 / t48, 3),
            "ratio": round(t48 / t64, 3),
            "step_saved_us": round((t64 - t48) * count_in_step, 2),
            "bit_exact": True,
        }
        records.append(rec)
        print(json.dumps(rec), flush=True)

    # Router bf16_gemm_partials (384, 5120)
    n, k, shards = 384, 5120, 8
    wb = torch.randn((n, k), generator=gen, device=device, dtype=torch.bfloat16)
    xb = torch.randn((42, k), generator=gen, device=device, dtype=torch.bfloat16)
    bp64 = bf16_gemm_partials(n, k, shards, block_M=64)
    bp48 = bf16_gemm_partials(n, k, shards, block_M=48)
    pb64 = torch.empty((shards, 42, n), dtype=torch.float32, device=device)
    pb48 = torch.empty((shards, 42, n), dtype=torch.float32, device=device)
    bp64(xb, wb, pb64)
    bp48(xb, wb, pb48)
    assert torch.equal(pb48, pb64)
    t64 = _bench_us(lambda: bp64(xb, wb, pb64))
    t48 = _bench_us(lambda: bp48(xb, wb, pb48))
    rec_router = {
        "boot": boot,
        "name": "router_bf16_partials",
        "shape": [384, 5120, 1],
        "m": 42,
        "layers": 58,
        "baseline_bm64_us": round(t64, 3),
        "candidate_bm48_us": round(t48, 3),
        "speedup": round(t64 / t48, 3),
        "ratio": round(t48 / t64, 3),
        "step_saved_us": round((t64 - t48) * 58, 2),
        "bit_exact": True,
    }
    records.append(rec_router)
    print(json.dumps(rec_router), flush=True)

    # SPLIT_FP8 (512, 5120)
    n, k = 512, 5120
    split_cfg = SPLIT_FP8[(n, k)]
    shards = split_cfg["shards"]
    w = (torch.randn((n, k), generator=gen, device=device) * 2).to(torch.float8_e4m3fn)
    exps = torch.randint(120, 125, (n // 32, k // 32), generator=gen, device=device).to(
        torch.uint8
    )
    wsf = pack_scale_words(exps)
    x = torch.randn((64, k), generator=gen, device=device, dtype=torch.bfloat16)
    xq, sf = tile_kernels.quant.per_token_cast(
        x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True
    )
    sf_u32 = sf.contiguous().view(torch.uint32)
    kp64 = mxfp8_gemm_partials(n, k, shards, block_M=64, **split_cfg["decode"][64])
    kp48 = mxfp8_gemm_partials(n, k, shards, block_M=48, **split_cfg["decode"][48])
    p64 = torch.empty((shards, 42, n), dtype=torch.float32, device=device)
    p48 = torch.empty((shards, 42, n), dtype=torch.float32, device=device)
    kp64(xq, w, sf_u32, wsf, p64)
    kp48(xq[:48], w, sf_u32[:48], wsf, p48)
    assert torch.equal(p48, p64)
    t64 = _bench_us(lambda: kp64(xq, w, sf_u32, wsf, p64))
    t48 = _bench_us(lambda: kp48(xq[:48], w, sf_u32[:48], wsf, p48))
    rec_split = {
        "boot": boot,
        "name": "dspark_context_kv_split_fp8",
        "shape": [512, 5120, 1],
        "m": 42,
        "layers": 1,
        "baseline_bm64_us": round(t64, 3),
        "candidate_bm48_us": round(t48, 3),
        "speedup": round(t64 / t48, 3),
        "ratio": round(t48 / t64, 3),
        "step_saved_us": round(t64 - t48, 2),
        "bit_exact": True,
    }
    records.append(rec_split)
    print(json.dumps(rec_split), flush=True)

    if out_path:
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
