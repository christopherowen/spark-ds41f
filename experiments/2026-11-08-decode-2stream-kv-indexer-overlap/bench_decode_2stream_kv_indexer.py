#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark decode 2-stream KV/indexer-key write overlap with query/index-query path.

Measures across TP=4 and TP=3 decode batch sizes M in (1, 2, 3, 6, 12, 24, 48):
1. Pre-indexer_score dependency vs. SM occupancy bound:
   - Standalone latency and CTA occupancy (out of 48 SMs on NVIDIA GB10) of every
     kernel in the pre-indexer_score region after wq_b and compressor projection:
     q_rope, gather_columns_into, gather_partial_state, compress(write_kv=True),
     indexer.wk partials, norm_rope_write_index_keys, indexer.wq_b (fused RoPE+MXFP4),
     indexer.weights_proj (partials+reduce), metadata.pages, and decode_metadata.
   - Sum of standalone latencies vs. 1-stream PDL-chained latency vs. 2-stream
     overlapped CUDA-graph latency.
2. Full decode attention sublayer (attention_front + wq_b + L2 prefetch + compressor +
   indexer + sparse_mla) across all three indexed layer families in DeepSeek-V4.1:
   - r2_kv_index_source (Layers 2, 8, 14; compress_ratio=2, 3 layers/step)
   - r1_candidate_source (Layer 20; compress_ratio=1, candidate_blocks=True, 1 layer/step)
   - r1_index_only_reindex (Layers 24, 28, 32, 36; compress_ratio=1, candidates=True,
     compressor=None, 4 layers/step)
   Compares interleaved paired windows of baseline_1107 (1-stream) vs. candidate_2stream
   (2-stream CUDA graph capture) under both warm-L2 multi-rep graph replay and
   cold-L2 + L2-prefetch single-layer graph replay, verifying 100% bit-exactness
   (torch.equal, 0 uint16 diffs, max_abs_diff == 0.0) across all outputs and caches.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from vllm.models.deepseek_v4_1 import l2_prefetch
from vllm.models.deepseek_v4_1.tilelang import compress as tl_compress
from vllm.models.deepseek_v4_1.tilelang import metadata as tl_metadata
from vllm.models.deepseek_v4_1.tilelang.front import attention_front
from vllm.models.deepseek_v4_1.tilelang.gather import gather_columns_into
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    DECODE_TILE_ROWS,
    SPLIT_BF16,
    bf16_gemm_partials,
    bf16_shards,
    decode_tile_rows,
    fp8_decode_config,
    mxfp8_gemm_decode,
    mxfp8_gemm_rope_mxfp4_decode,
    pack_scale_words,
    pack_weight_tiles,
    packed_bf16_gemm_splitk,
    pick_block_K,
    splitk_reduce,
)
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    CANDIDATE_BLOCK,
    CANDIDATE_BLOCKS,
    IndexerV41Plan,
    norm_rope_write_index_keys,
    page_bytes,
)
from vllm.models.deepseek_v4_1.tilelang.linear import (
    PARTIAL_ROW_TILES,
    empty_mxfp8,
    pack_bf16_weight,
)
from vllm.models.deepseek_v4_1.tilelang.rope import rope_
from vllm.models.deepseek_v4_1.tilelang.sparse_mla import SparseMlaV41Plan

DEVICE = torch.device("cuda:0")
BATCH_SIZES = (1, 2, 3, 6, 12, 24, 48)
REQS = 64
NUM_PAGES = 128


def _time_graph_single(fn, main_stream, reps=200, warm=20, windows=5):
    with torch.cuda.stream(main_stream):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(main_stream):
        with torch.cuda.graph(g, stream=main_stream):
            fn()
        for _ in range(warm):
            g.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(windows):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(main_stream)
            for _ in range(reps):
                g.replay()
            e.record(main_stream)
            torch.cuda.synchronize()
            samples.append(s.elapsed_time(e) * 1000.0 / reps)
        samples.sort()
        return samples[len(samples) // 2]


def bench_tp(tp: int):
    torch.manual_seed(1107 + tp)
    part_cols = ((512 + tp * 64 - 1) // (tp * 64)) * 64
    heads = 16 if tp == 4 else 24
    wqb_n = heads * 512

    cos_sin = torch.randn(4096, 64, device=DEVICE, dtype=torch.float32)
    q_norm_w = torch.ones(1280, device=DEVICE, dtype=torch.bfloat16)
    kv_norm_w = torch.ones(512, device=DEVICE, dtype=torch.bfloat16)
    comp_norm_w = torch.ones(512, device=DEVICE, dtype=torch.float32)
    k_norm_w = torch.ones(128, device=DEVICE, dtype=torch.bfloat16)
    sinks = torch.zeros(heads, device=DEVICE, dtype=torch.float32)

    wqb_w = pack_weight_tiles(
        torch.randint(0, 127, (wqb_n, 1280), device=DEVICE, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
    )
    wqb_sf = pack_scale_words(
        torch.full((wqb_n // 32, 1280 // 32), 127, device=DEVICE, dtype=torch.uint8)
    )
    wqb_kernels = {
        bm: mxfp8_gemm_decode(
            wqb_n,
            1280,
            **fp8_decode_config(wqb_n, 1280, bm, 1),
            padded_rows=True,
            groups=1,
            tile_contiguous=True,
        )
        for bm in DECODE_TILE_ROWS
    }

    comp_kernels = {}
    comp_packed = {}
    for ratio in (1, 2):
        comp_n = part_cols * ratio
        comp_w_raw = (
            torch.randn(comp_n, 5120, device=DEVICE, dtype=torch.bfloat16) * 0.02
        )
        packed = pack_bf16_weight(comp_w_raw, in_place=False)
        comp_packed[ratio] = packed
        split_cfg = SPLIT_BF16.get((comp_n, 5120)) or dict(
            shards=bf16_shards(comp_n, 5120), blocked=False, decode={}
        )
        shards = split_cfg["shards"]
        blocked = split_cfg["blocked"]
        decode_cfg = split_cfg.get("decode", {})
        out_dt = "float32" if ratio == 2 else "bfloat16"
        comp_kernels[ratio] = {
            bm: packed_bf16_gemm_splitk(
                comp_n,
                5120,
                shards,
                packed.exponent_base,
                block_M=bm,
                out_dtype=out_dt,
                parts=ratio,
                blocked=blocked,
                **decode_cfg.get(bm, {}),
            )
            for bm in PARTIAL_ROW_TILES
        }

    wk_w = torch.randn(128, 512, device=DEVICE, dtype=torch.bfloat16) * 0.02
    wk_shards = bf16_shards(128, 512)
    wk_partials_kernels = {
        bm: bf16_gemm_partials(128, 512, wk_shards, block_M=bm)
        for bm in PARTIAL_ROW_TILES
    }

    iwqb_w = pack_weight_tiles(
        torch.randint(0, 127, (4096, 1280), device=DEVICE, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
    )
    iwqb_sf = pack_scale_words(
        torch.full((4096 // 32, 1280 // 32), 127, device=DEVICE, dtype=torch.uint8)
    )
    iwqb_kernel = mxfp8_gemm_rope_mxfp4_decode(
        4096,
        1280,
        head_dim=128,
        rope_dim=64,
        block_K=pick_block_K(1280),
        padded_rows=True,
        tile_contiguous=True,
    )

    hw_w = torch.randn(32, 5120, device=DEVICE, dtype=torch.bfloat16) * 0.02
    hw_shards = bf16_shards(32, 5120)
    hw_partials_kernels = {
        bm: bf16_gemm_partials(32, 5120, hw_shards, block_M=bm)
        for bm in PARTIAL_ROW_TILES
    }
    hw_reduce = splitk_reduce(32, hw_shards, out_dtype="bfloat16")

    idx_plan_r2 = IndexerV41Plan(64 * 64, 64, candidates=False, candidate_blocks=False)
    idx_plan_r1_src = IndexerV41Plan(
        64 * 128, 128, candidates=False, candidate_blocks=True
    )
    idx_plan_r1_reidx = IndexerV41Plan(
        CANDIDATE_BLOCKS * CANDIDATE_BLOCK, 128, candidates=True, candidate_blocks=False
    )

    mla_r2 = SparseMlaV41Plan(
        heads, 512, 512, 64, swa_page_size=64, max_rows=64, device=DEVICE
    )
    mla_r1 = SparseMlaV41Plan(
        heads, 512, 512, 128, swa_page_size=64, max_rows=64, device=DEVICE
    )

    # L2 prefetch plans matching DeepseekV4DecoderLayer.build_l2_prefetch
    l2_comp_r2 = l2_prefetch.make_plan(
        [
            (
                "compressor.wkv_wgate",
                comp_packed[2].values.data_ptr(),
                comp_packed[2].values.numel() * comp_packed[2].values.element_size(),
            )
        ],
        20 * 1024 * 1024,
        DEVICE,
    )
    l2_comp_r1 = l2_prefetch.make_plan(
        [
            (
                "compressor.wkv",
                comp_packed[1].values.data_ptr(),
                comp_packed[1].values.numel() * comp_packed[1].values.element_size(),
            )
        ],
        20 * 1024 * 1024,
        DEVICE,
    )
    dummy_wo_bytes = torch.zeros(
        heads * 128 * 512 + 5120 * (heads // 8) * 128,
        device=DEVICE,
        dtype=torch.uint8,
    )
    l2_wo = l2_prefetch.make_plan(
        [
            (
                "wo",
                dummy_wo_bytes.data_ptr(),
                dummy_wo_bytes.numel(),
            )
        ],
        20 * 1024 * 1024,
        DEVICE,
    )
    l2_flush_buf = torch.zeros(48 * 1024 * 1024, device=DEVICE, dtype=torch.uint8)

    main_stream = torch.cuda.Stream(device=DEVICE)
    aux_stream = torch.cuda.Stream(device=DEVICE)

    pre_index_bound = {}
    full_layer_results = {}

    for M in BATCH_SIZES:
        bm = next((b for b in PARTIAL_ROW_TILES if M <= b), 64)
        bm_fp8 = decode_tile_rows(M, {})

        hidden = torch.randn(M, 5120, device=DEVICE, dtype=torch.bfloat16) * 0.1
        qr_kv = torch.randn(M, 1792, device=DEVICE, dtype=torch.bfloat16) * 0.1
        qr = torch.empty(M, 1280, device=DEVICE, dtype=torch.bfloat16)
        kv = torch.empty(M, 512, device=DEVICE, dtype=torch.bfloat16)
        q_cast = empty_mxfp8(qr)
        positions = torch.arange(100, 100 + M, device=DEVICE, dtype=torch.int64)
        slots = torch.arange(M, device=DEVICE, dtype=torch.int64)
        q = torch.empty(M, heads, 512, device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty(M, heads, 512, device=DEVICE, dtype=torch.bfloat16)
        iq_data = torch.empty(M, 32, 64, device=DEVICE, dtype=torch.uint8)
        iq_scale = torch.empty(M, 32, 4, device=DEVICE, dtype=torch.uint8)
        hw_partials = torch.empty(hw_shards, M, 32, device=DEVICE, dtype=torch.float32)
        hw_out = torch.empty(M, 32, device=DEVICE, dtype=torch.bfloat16)
        wk_partials = torch.empty(wk_shards, M, 128, device=DEVICE, dtype=torch.float32)

        comp_partials_r2 = torch.empty(
            5, M, part_cols * 2, device=DEVICE, dtype=torch.float32
        )
        comp_partials_r1 = torch.empty(
            5, M, part_cols * 1, device=DEVICE, dtype=torch.float32
        )
        comp_counters = torch.zeros(64, device=DEVICE, dtype=torch.int32)
        comp_local_r2 = torch.empty(
            2, M, part_cols, device=DEVICE, dtype=torch.float32
        )
        comp_local_r1 = torch.empty(
            1, M, part_cols, device=DEVICE, dtype=torch.bfloat16
        )
        comp_gathered_r2 = (
            torch.randn(tp * 2 * M, part_cols, device=DEVICE, dtype=torch.float32) * 0.1
        )
        comp_gathered_r1 = (
            torch.randn(tp * 1 * M, part_cols, device=DEVICE, dtype=torch.bfloat16)
            * 0.1
        )
        values_r2 = torch.empty(M, 512, device=DEVICE, dtype=torch.float32)
        gates_r2 = torch.empty(M, 512, device=DEVICE, dtype=torch.float32)
        values_r1 = torch.empty(M, 512, device=DEVICE, dtype=torch.bfloat16)
        latent = torch.empty(M, 512, device=DEVICE, dtype=torch.bfloat16)
        emitted = torch.empty(M, device=DEVICE, dtype=torch.bool)
        out_slots = torch.empty(M, device=DEVICE, dtype=torch.int64)
        pending_v = torch.zeros(REQS, 512, device=DEVICE, dtype=torch.float32)
        pending_g = torch.zeros(REQS, 512, device=DEVICE, dtype=torch.float32)
        pending_t = torch.zeros(REQS, device=DEVICE, dtype=torch.int64)
        local_ids = torch.zeros(REQS, device=DEVICE, dtype=torch.int64)
        row_ids = torch.arange(REQS, device=DEVICE, dtype=torch.int64)

        q_starts = torch.zeros(REQS + 1, device=DEVICE, dtype=torch.int32)
        live_reqs = min(M, REQS)
        for r in range(live_reqs):
            q_starts[r + 1] = q_starts[r] + (
                M // live_reqs + (1 if r < M % live_reqs else 0)
            )
        for r in range(live_reqs, REQS):
            q_starts[r + 1] = M
        req_pos = torch.full((REQS,), 100, device=DEVICE, dtype=torch.int64)
        live_counts = torch.tensor([M, live_reqs], device=DEVICE, dtype=torch.int32)
        state_table = torch.arange(
            1, REQS + 1, device=DEVICE, dtype=torch.int32
        ).view(REQS, 1)
        state_slots = torch.arange(M, device=DEVICE, dtype=torch.int64)

        req_id_per_token = (
            torch.arange(M, device=DEVICE, dtype=torch.int32) % live_reqs
        )
        cache_lengths_r2 = torch.full((M,), 512, device=DEVICE, dtype=torch.int32)
        cache_lengths_r1 = torch.full((M,), 1024, device=DEVICE, dtype=torch.int32)
        block_table = (
            torch.arange(REQS * 64, device=DEVICE, dtype=torch.int32).view(REQS, 64)
            % 63
        ) + 1
        index_pages = torch.zeros(64, 64, device=DEVICE, dtype=torch.int32)
        swa_indices = torch.zeros(M, 512, device=DEVICE, dtype=torch.int32)
        swa_lengths = torch.full((M,), 64, device=DEVICE, dtype=torch.int32)
        top_lengths = torch.full((M,), 512, device=DEVICE, dtype=torch.int32)
        main_page_table = torch.zeros(M, 64, device=DEVICE, dtype=torch.int32)
        topk_indices = torch.zeros(M, 512, device=DEVICE, dtype=torch.int32)
        candidates_init = (
            torch.arange(CANDIDATE_BLOCKS, device=DEVICE, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(M, 1)
        )
        candidates = candidates_init.clone()
        candidate_lens = torch.full((M,), 1024, device=DEVICE, dtype=torch.int32)

        active_r2 = torch.tensor([2048], device=DEVICE, dtype=torch.int32)
        active_r1 = torch.tensor([4096], device=DEVICE, dtype=torch.int32)

        idx_scratch_r2 = [
            torch.empty(sh, dtype=dt, device=DEVICE)
            for sh, dt in idx_plan_r2.scratch_specs(M)
        ]
        idx_scratch_r1_src = [
            torch.empty(sh, dtype=dt, device=DEVICE)
            for sh, dt in idx_plan_r1_src.scratch_specs(M)
        ]
        idx_scratch_r1_reidx = [
            torch.empty(sh, dtype=dt, device=DEVICE)
            for sh, dt in idx_plan_r1_reidx.scratch_specs(M)
        ]
        mla_scratch_r2 = [
            torch.empty(sh, dtype=dt, device=DEVICE)
            for sh, dt in mla_r2.scratch_specs(M)
        ]
        mla_scratch_r1 = [
            torch.empty(sh, dtype=dt, device=DEVICE)
            for sh, dt in mla_r1.scratch_specs(M)
        ]

        swa_cache_init = torch.randint(
            0, 126, (NUM_PAGES, 64 * 528), device=DEVICE, dtype=torch.uint8
        )
        kv_cache_r2_init = torch.randint(
            0, 126, (NUM_PAGES, 64 * 288), device=DEVICE, dtype=torch.uint8
        )
        kv_cache_r1_init = torch.randint(
            0, 126, (NUM_PAGES, 128 * 288), device=DEVICE, dtype=torch.uint8
        )
        state_pool_init = (
            torch.randn(NUM_PAGES, 8 * 1024, device=DEVICE, dtype=torch.float32) * 0.05
        )
        idx_k_cache_r2_init = torch.randint(
            0, 126, (NUM_PAGES, page_bytes(64)), device=DEVICE, dtype=torch.uint8
        )
        idx_k_cache_r1_init = torch.randint(
            0, 126, (NUM_PAGES, page_bytes(128)), device=DEVICE, dtype=torch.uint8
        )

        swa_cache = swa_cache_init.clone()
        kv_cache_r2 = kv_cache_r2_init.clone()
        kv_cache_r1 = kv_cache_r1_init.clone()
        state_pool = state_pool_init.clone()
        idx_k_cache_r2 = idx_k_cache_r2_init.clone()
        idx_k_cache_r1 = idx_k_cache_r1_init.clone()

        # --- Part 1: Pre-indexer_score dependency vs. SM occupancy bound ---
        t_q_rope = _time_graph_single(
            lambda: rope_(q, positions, cos_sin, 1, False), main_stream
        )
        t_gather_cols = _time_graph_single(
            lambda: gather_columns_into(comp_gathered_r2, (values_r2, gates_r2), tp),
            main_stream,
        )
        t_gather_state = _time_graph_single(
            lambda: tl_compress.gather_partial_state(
                state_pool,
                state_table,
                q_starts,
                req_pos,
                live_counts,
                pending_v,
                pending_g,
                pending_t,
                local_ids,
                page=8,
            ),
            main_stream,
        )
        t_compress_r2 = _time_graph_single(
            lambda: tl_compress.compress(
                M,
                2,
                values_r2,
                gates_r2,
                comp_norm_w,
                q_starts,
                req_pos,
                local_ids,
                slots,
                live_counts,
                latent,
                emitted,
                out_slots,
                pending_values=pending_v,
                pending_gates=pending_g,
                tags=pending_t,
                pool=state_pool,
                state_slots=state_slots,
                page=8,
                kv_cache=kv_cache_r2,
                token_positions=positions,
                cos_sin=cos_sin,
                kv_page_size=64,
            ),
            main_stream,
        )
        t_wk_partials = _time_graph_single(
            lambda: wk_partials_kernels[bm](latent, wk_w, wk_partials), main_stream
        )
        t_write_idx_k = _time_graph_single(
            lambda: norm_rope_write_index_keys(
                wk_partials,
                k_norm_w,
                1e-6,
                positions,
                cos_sin,
                idx_k_cache_r2,
                slots,
                64,
                2,
            ),
            main_stream,
        )
        t_iwqb = _time_graph_single(
            lambda: iwqb_kernel(
                q_cast.values,
                iwqb_w,
                q_cast.scales.view(torch.uint32),
                iwqb_sf,
                positions,
                cos_sin,
                iq_data.view(M, -1),
                iq_scale.view(M, -1),
            ),
            main_stream,
        )
        t_hw = _time_graph_single(
            lambda: (
                hw_partials_kernels[bm](hidden, hw_w, hw_partials),
                hw_reduce(hw_partials, hw_out),
            ),
            main_stream,
        )
        t_pages = _time_graph_single(
            lambda: tl_metadata.pages(
                req_id_per_token,
                cache_lengths_r2,
                block_table,
                index_pages,
                0,
                M,
                64,
                page=64,
            ),
            main_stream,
        )
        t_decode_meta = _time_graph_single(
            lambda: tl_metadata.decode_metadata(
                positions,
                req_id_per_token,
                block_table,
                swa_indices,
                swa_lengths,
                top_lengths,
                cache_lengths_r2,
                q_starts,
                req_pos,
                req_id_per_token,
                block_table,
                main_page_table,
                0,
                M,
                64,
                main_page=64,
                page=64,
                window=512,
                draft=False,
                replay_start=None,
            ),
            main_stream,
        )

        def pre_index_1stream():
            rope_(q, positions, cos_sin, 1, False)
            gather_columns_into(comp_gathered_r2, (values_r2, gates_r2), tp)
            tl_compress.gather_partial_state(
                state_pool,
                state_table,
                q_starts,
                req_pos,
                live_counts,
                pending_v,
                pending_g,
                pending_t,
                local_ids,
                page=8,
            )
            tl_compress.compress(
                M,
                2,
                values_r2,
                gates_r2,
                comp_norm_w,
                q_starts,
                req_pos,
                local_ids,
                slots,
                live_counts,
                latent,
                emitted,
                out_slots,
                pending_values=pending_v,
                pending_gates=pending_g,
                tags=pending_t,
                pool=state_pool,
                state_slots=state_slots,
                page=8,
                kv_cache=kv_cache_r2,
                token_positions=positions,
                cos_sin=cos_sin,
                kv_page_size=64,
            )
            wk_partials_kernels[bm](latent, wk_w, wk_partials)
            norm_rope_write_index_keys(
                wk_partials,
                k_norm_w,
                1e-6,
                positions,
                cos_sin,
                idx_k_cache_r2,
                slots,
                64,
                2,
            )
            iwqb_kernel(
                q_cast.values,
                iwqb_w,
                q_cast.scales.view(torch.uint32),
                iwqb_sf,
                positions,
                cos_sin,
                iq_data.view(M, -1),
                iq_scale.view(M, -1),
            )
            hw_partials_kernels[bm](hidden, hw_w, hw_partials)
            hw_reduce(hw_partials, hw_out)
            tl_metadata.pages(
                req_id_per_token,
                cache_lengths_r2,
                block_table,
                index_pages,
                0,
                M,
                64,
                page=64,
            )
            tl_metadata.decode_metadata(
                positions,
                req_id_per_token,
                block_table,
                swa_indices,
                swa_lengths,
                top_lengths,
                cache_lengths_r2,
                q_starts,
                req_pos,
                req_id_per_token,
                block_table,
                main_page_table,
                0,
                M,
                64,
                main_page=64,
                page=64,
                window=512,
                draft=False,
                replay_start=None,
            )

        def pre_index_2stream():
            aux_stream.wait_stream(main_stream)
            with torch.cuda.stream(aux_stream):
                rope_(q, positions, cos_sin, 1, False)
                iwqb_kernel(
                    q_cast.values,
                    iwqb_w,
                    q_cast.scales.view(torch.uint32),
                    iwqb_sf,
                    positions,
                    cos_sin,
                    iq_data.view(M, -1),
                    iq_scale.view(M, -1),
                )
                hw_partials_kernels[bm](hidden, hw_w, hw_partials)
                hw_reduce(hw_partials, hw_out)
                tl_metadata.decode_metadata(
                    positions,
                    req_id_per_token,
                    block_table,
                    swa_indices,
                    swa_lengths,
                    top_lengths,
                    cache_lengths_r2,
                    q_starts,
                    req_pos,
                    req_id_per_token,
                    block_table,
                    main_page_table,
                    0,
                    M,
                    64,
                    main_page=64,
                    page=64,
                    window=512,
                    draft=False,
                    replay_start=None,
                )
            gather_columns_into(comp_gathered_r2, (values_r2, gates_r2), tp)
            tl_compress.gather_partial_state(
                state_pool,
                state_table,
                q_starts,
                req_pos,
                live_counts,
                pending_v,
                pending_g,
                pending_t,
                local_ids,
                page=8,
            )
            tl_compress.compress(
                M,
                2,
                values_r2,
                gates_r2,
                comp_norm_w,
                q_starts,
                req_pos,
                local_ids,
                slots,
                live_counts,
                latent,
                emitted,
                out_slots,
                pending_values=pending_v,
                pending_gates=pending_g,
                tags=pending_t,
                pool=state_pool,
                state_slots=state_slots,
                page=8,
                kv_cache=kv_cache_r2,
                token_positions=positions,
                cos_sin=cos_sin,
                kv_page_size=64,
            )
            wk_partials_kernels[bm](latent, wk_w, wk_partials)
            norm_rope_write_index_keys(
                wk_partials,
                k_norm_w,
                1e-6,
                positions,
                cos_sin,
                idx_k_cache_r2,
                slots,
                64,
                2,
            )
            tl_metadata.pages(
                req_id_per_token,
                cache_lengths_r2,
                block_table,
                index_pages,
                0,
                M,
                64,
                page=64,
            )
            main_stream.wait_stream(aux_stream)

        sum_standalone = (
            t_q_rope
            + t_gather_cols
            + t_gather_state
            + t_compress_r2
            + t_wk_partials
            + t_write_idx_k
            + t_iwqb
            + t_hw
            + t_pages
            + t_decode_meta
        )
        t_pre_1s = _time_graph_single(pre_index_1stream, main_stream)
        t_pre_2s = _time_graph_single(pre_index_2stream, main_stream)

        pre_index_bound[str(M)] = {
            "M": M,
            "q_rope_us": round(t_q_rope, 2),
            "gather_columns_into_us": round(t_gather_cols, 2),
            "gather_partial_state_us": round(t_gather_state, 2),
            "compress_write_kv_r2_us": round(t_compress_r2, 2),
            "wk_partials_us": round(t_wk_partials, 2),
            "norm_rope_write_index_keys_us": round(t_write_idx_k, 2),
            "indexer_wqb_rope_mxfp4_us": round(t_iwqb, 2),
            "indexer_weights_proj_us": round(t_hw, 2),
            "metadata_pages_us": round(t_pages, 2),
            "decode_metadata_us": round(t_decode_meta, 2),
            "sum_standalone_us": round(sum_standalone, 2),
            "serial_1stream_pdl_us": round(t_pre_1s, 2),
            "candidate_2stream_pdl_us": round(t_pre_2s, 2),
            "delta_2stream_vs_1stream_us": round(t_pre_2s - t_pre_1s, 2),
            "delta_2stream_vs_1stream_pct": round(
                100.0 * (t_pre_2s - t_pre_1s) / t_pre_1s, 1
            ),
        }

        # --- Part 2: Full Attention Sublayer (all 3 indexed layer families) ---
        def reset_state():
            torch.cuda.synchronize()
            with torch.cuda.stream(main_stream):
                swa_cache.copy_(swa_cache_init)
                kv_cache_r2.copy_(kv_cache_r2_init)
                kv_cache_r1.copy_(kv_cache_r1_init)
                state_pool.copy_(state_pool_init)
                idx_k_cache_r2.copy_(idx_k_cache_r2_init)
                idx_k_cache_r1.copy_(idx_k_cache_r1_init)
                pending_v.zero_()
                pending_g.zero_()
                pending_t.zero_()
                local_ids.zero_()
                topk_indices.zero_()
                candidates.copy_(candidates_init)
                candidate_lens.fill_(1024)
                out.zero_()
            torch.cuda.synchronize()

        # 1. Ratio-2 KV+Index Source Layer (Layers 2, 8, 14)
        def r2_1stream():
            attention_front(
                qr_kv,
                q_norm_w,
                kv_norm_w,
                1e-6,
                positions,
                cos_sin,
                swa_cache,
                slots,
                64,
                qr,
                q_cast.values,
                q_cast.scales,
                kv,
            )
            wqb_kernels[bm_fp8](
                q_cast.values,
                wqb_w,
                q_cast.scales.view(torch.uint32),
                wqb_sf,
                q.view(M, wqb_n),
            )
            l2_prefetch.issue(l2_comp_r2, M)
            rope_(q, positions, cos_sin, 1, False)
            comp_kernels[2][bm](
                hidden,
                comp_packed[2].values,
                comp_packed[2].exception_entries,
                comp_packed[2].row_offsets,
                comp_partials_r2,
                comp_counters,
                comp_local_r2[0],
                comp_local_r2[1],
            )
            gather_columns_into(comp_gathered_r2, (values_r2, gates_r2), tp)
            tl_compress.gather_partial_state(
                state_pool,
                state_table,
                q_starts,
                req_pos,
                live_counts,
                pending_v,
                pending_g,
                pending_t,
                local_ids,
                page=8,
            )
            tl_compress.compress(
                M,
                2,
                values_r2,
                gates_r2,
                comp_norm_w,
                q_starts,
                req_pos,
                local_ids,
                slots,
                live_counts,
                latent,
                emitted,
                out_slots,
                pending_values=pending_v,
                pending_gates=pending_g,
                tags=pending_t,
                pool=state_pool,
                state_slots=state_slots,
                page=8,
                kv_cache=kv_cache_r2,
                token_positions=positions,
                cos_sin=cos_sin,
                kv_page_size=64,
            )
            l2_prefetch.issue(l2_wo, M)
            wk_partials_kernels[bm](latent, wk_w, wk_partials)
            norm_rope_write_index_keys(
                wk_partials,
                k_norm_w,
                1e-6,
                positions,
                cos_sin,
                idx_k_cache_r2,
                slots,
                64,
                2,
            )
            iwqb_kernel(
                q_cast.values,
                iwqb_w,
                q_cast.scales.view(torch.uint32),
                iwqb_sf,
                positions,
                cos_sin,
                iq_data.view(M, -1),
                iq_scale.view(M, -1),
            )
            hw_partials_kernels[bm](hidden, hw_w, hw_partials)
            hw_reduce(hw_partials, hw_out)
            tl_metadata.pages(
                req_id_per_token,
                cache_lengths_r2,
                block_table,
                index_pages,
                0,
                M,
                64,
                page=64,
            )
            idx_plan_r2.run(
                iq_data,
                iq_scale,
                hw_out,
                idx_k_cache_r2,
                index_pages[:M],
                cache_lengths_r2,
                active_r2,
                out=topk_indices,
                scratch=idx_scratch_r2,
            )
            tl_metadata.decode_metadata(
                positions,
                req_id_per_token,
                block_table,
                swa_indices,
                swa_lengths,
                top_lengths,
                cache_lengths_r2,
                q_starts,
                req_pos,
                req_id_per_token,
                block_table,
                main_page_table,
                0,
                M,
                64,
                main_page=64,
                page=64,
                window=512,
                draft=False,
                replay_start=None,
            )
            mla_r2.run(
                q,
                swa_cache,
                swa_indices,
                swa_lengths,
                sinks,
                positions,
                cos_sin,
                out,
                scratch=mla_scratch_r2 or None,
                idx_cache=kv_cache_r2,
                idx_indices=topk_indices,
                idx_lengths=top_lengths,
                idx_page_table=main_page_table,
            )
            l2_prefetch.join()

        def r2_2stream():
            attention_front(
                qr_kv,
                q_norm_w,
                kv_norm_w,
                1e-6,
                positions,
                cos_sin,
                swa_cache,
                slots,
                64,
                qr,
                q_cast.values,
                q_cast.scales,
                kv,
            )
            wqb_kernels[bm_fp8](
                q_cast.values,
                wqb_w,
                q_cast.scales.view(torch.uint32),
                wqb_sf,
                q.view(M, wqb_n),
            )
            l2_prefetch.issue(l2_comp_r2, M)
            comp_kernels[2][bm](
                hidden,
                comp_packed[2].values,
                comp_packed[2].exception_entries,
                comp_packed[2].row_offsets,
                comp_partials_r2,
                comp_counters,
                comp_local_r2[0],
                comp_local_r2[1],
            )
            aux_stream.wait_stream(main_stream)
            with torch.cuda.stream(aux_stream):
                iwqb_kernel(
                    q_cast.values,
                    iwqb_w,
                    q_cast.scales.view(torch.uint32),
                    iwqb_sf,
                    positions,
                    cos_sin,
                    iq_data.view(M, -1),
                    iq_scale.view(M, -1),
                )
                hw_partials_kernels[bm](hidden, hw_w, hw_partials)
                hw_reduce(hw_partials, hw_out)
            gather_columns_into(comp_gathered_r2, (values_r2, gates_r2), tp)
            tl_compress.gather_partial_state(
                state_pool,
                state_table,
                q_starts,
                req_pos,
                live_counts,
                pending_v,
                pending_g,
                pending_t,
                local_ids,
                page=8,
            )
            tl_compress.compress(
                M,
                2,
                values_r2,
                gates_r2,
                comp_norm_w,
                q_starts,
                req_pos,
                local_ids,
                slots,
                live_counts,
                latent,
                emitted,
                out_slots,
                pending_values=pending_v,
                pending_gates=pending_g,
                tags=pending_t,
                pool=state_pool,
                state_slots=state_slots,
                page=8,
                kv_cache=kv_cache_r2,
                token_positions=positions,
                cos_sin=cos_sin,
                kv_page_size=64,
            )
            l2_prefetch.issue(l2_wo, M)
            wk_partials_kernels[bm](latent, wk_w, wk_partials)
            norm_rope_write_index_keys(
                wk_partials,
                k_norm_w,
                1e-6,
                positions,
                cos_sin,
                idx_k_cache_r2,
                slots,
                64,
                2,
            )
            tl_metadata.pages(
                req_id_per_token,
                cache_lengths_r2,
                block_table,
                index_pages,
                0,
                M,
                64,
                page=64,
            )
            main_stream.wait_stream(aux_stream)
            with torch.cuda.stream(aux_stream):
                rope_(q, positions, cos_sin, 1, False)
                tl_metadata.decode_metadata(
                    positions,
                    req_id_per_token,
                    block_table,
                    swa_indices,
                    swa_lengths,
                    top_lengths,
                    cache_lengths_r2,
                    q_starts,
                    req_pos,
                    req_id_per_token,
                    block_table,
                    main_page_table,
                    0,
                    M,
                    64,
                    main_page=64,
                    page=64,
                    window=512,
                    draft=False,
                    replay_start=None,
                )
            idx_plan_r2.run(
                iq_data,
                iq_scale,
                hw_out,
                idx_k_cache_r2,
                index_pages[:M],
                cache_lengths_r2,
                active_r2,
                out=topk_indices,
                scratch=idx_scratch_r2,
            )
            main_stream.wait_stream(aux_stream)
            mla_r2.run(
                q,
                swa_cache,
                swa_indices,
                swa_lengths,
                sinks,
                positions,
                cos_sin,
                out,
                scratch=mla_scratch_r2 or None,
                idx_cache=kv_cache_r2,
                idx_indices=topk_indices,
                idx_lengths=top_lengths,
                idx_page_table=main_page_table,
            )
            l2_prefetch.join()

        # 2. Ratio-1 KV+Index Candidate Source Layer (Layer 20)
        def r1_src_1stream():
            attention_front(
                qr_kv,
                q_norm_w,
                kv_norm_w,
                1e-6,
                positions,
                cos_sin,
                swa_cache,
                slots,
                64,
                qr,
                q_cast.values,
                q_cast.scales,
                kv,
            )
            wqb_kernels[bm_fp8](
                q_cast.values,
                wqb_w,
                q_cast.scales.view(torch.uint32),
                wqb_sf,
                q.view(M, wqb_n),
            )
            l2_prefetch.issue(l2_comp_r1, M)
            rope_(q, positions, cos_sin, 1, False)
            comp_kernels[1][bm](
                hidden,
                comp_packed[1].values,
                comp_packed[1].exception_entries,
                comp_packed[1].row_offsets,
                comp_partials_r1,
                comp_counters,
                comp_local_r1[0],
            )
            gather_columns_into(comp_gathered_r1, (values_r1,), tp)
            tl_compress.compress(
                M,
                1,
                values_r1,
                None,
                comp_norm_w,
                q_starts,
                req_pos,
                row_ids,
                slots,
                live_counts,
                latent,
                emitted,
                out_slots,
                kv_cache=kv_cache_r1,
                token_positions=positions,
                cos_sin=cos_sin,
                kv_page_size=128,
            )
            l2_prefetch.issue(l2_wo, M)
            wk_partials_kernels[bm](latent, wk_w, wk_partials)
            norm_rope_write_index_keys(
                wk_partials,
                k_norm_w,
                1e-6,
                positions,
                cos_sin,
                idx_k_cache_r1,
                slots,
                128,
                1,
            )
            iwqb_kernel(
                q_cast.values,
                iwqb_w,
                q_cast.scales.view(torch.uint32),
                iwqb_sf,
                positions,
                cos_sin,
                iq_data.view(M, -1),
                iq_scale.view(M, -1),
            )
            hw_partials_kernels[bm](hidden, hw_w, hw_partials)
            hw_reduce(hw_partials, hw_out)
            tl_metadata.pages(
                req_id_per_token,
                cache_lengths_r1,
                block_table,
                index_pages,
                0,
                M,
                64,
                page=128,
            )
            idx_plan_r1_src.run(
                iq_data,
                iq_scale,
                hw_out,
                idx_k_cache_r1,
                index_pages[:M],
                cache_lengths_r1,
                active_r1,
                out=topk_indices,
                candidate_out=candidates,
                candidate_out_lengths=candidate_lens,
                scratch=idx_scratch_r1_src,
            )
            tl_metadata.decode_metadata(
                positions,
                req_id_per_token,
                block_table,
                swa_indices,
                swa_lengths,
                top_lengths,
                cache_lengths_r1,
                q_starts,
                req_pos,
                req_id_per_token,
                block_table,
                main_page_table,
                0,
                M,
                64,
                main_page=128,
                page=64,
                window=512,
                draft=False,
                replay_start=None,
            )
            mla_r1.run(
                q,
                swa_cache,
                swa_indices,
                swa_lengths,
                sinks,
                positions,
                cos_sin,
                out,
                scratch=mla_scratch_r1 or None,
                idx_cache=kv_cache_r1,
                idx_indices=topk_indices,
                idx_lengths=top_lengths,
                idx_page_table=main_page_table,
            )
            l2_prefetch.join()

        def r1_src_2stream():
            attention_front(
                qr_kv,
                q_norm_w,
                kv_norm_w,
                1e-6,
                positions,
                cos_sin,
                swa_cache,
                slots,
                64,
                qr,
                q_cast.values,
                q_cast.scales,
                kv,
            )
            wqb_kernels[bm_fp8](
                q_cast.values,
                wqb_w,
                q_cast.scales.view(torch.uint32),
                wqb_sf,
                q.view(M, wqb_n),
            )
            l2_prefetch.issue(l2_comp_r1, M)
            comp_kernels[1][bm](
                hidden,
                comp_packed[1].values,
                comp_packed[1].exception_entries,
                comp_packed[1].row_offsets,
                comp_partials_r1,
                comp_counters,
                comp_local_r1[0],
            )
            aux_stream.wait_stream(main_stream)
            with torch.cuda.stream(aux_stream):
                iwqb_kernel(
                    q_cast.values,
                    iwqb_w,
                    q_cast.scales.view(torch.uint32),
                    iwqb_sf,
                    positions,
                    cos_sin,
                    iq_data.view(M, -1),
                    iq_scale.view(M, -1),
                )
                hw_partials_kernels[bm](hidden, hw_w, hw_partials)
                hw_reduce(hw_partials, hw_out)
            gather_columns_into(comp_gathered_r1, (values_r1,), tp)
            tl_compress.compress(
                M,
                1,
                values_r1,
                None,
                comp_norm_w,
                q_starts,
                req_pos,
                row_ids,
                slots,
                live_counts,
                latent,
                emitted,
                out_slots,
                kv_cache=kv_cache_r1,
                token_positions=positions,
                cos_sin=cos_sin,
                kv_page_size=128,
            )
            l2_prefetch.issue(l2_wo, M)
            wk_partials_kernels[bm](latent, wk_w, wk_partials)
            norm_rope_write_index_keys(
                wk_partials,
                k_norm_w,
                1e-6,
                positions,
                cos_sin,
                idx_k_cache_r1,
                slots,
                128,
                1,
            )
            tl_metadata.pages(
                req_id_per_token,
                cache_lengths_r1,
                block_table,
                index_pages,
                0,
                M,
                64,
                page=128,
            )
            main_stream.wait_stream(aux_stream)
            with torch.cuda.stream(aux_stream):
                rope_(q, positions, cos_sin, 1, False)
                tl_metadata.decode_metadata(
                    positions,
                    req_id_per_token,
                    block_table,
                    swa_indices,
                    swa_lengths,
                    top_lengths,
                    cache_lengths_r1,
                    q_starts,
                    req_pos,
                    req_id_per_token,
                    block_table,
                    main_page_table,
                    0,
                    M,
                    64,
                    main_page=128,
                    page=64,
                    window=512,
                    draft=False,
                    replay_start=None,
                )
            idx_plan_r1_src.run(
                iq_data,
                iq_scale,
                hw_out,
                idx_k_cache_r1,
                index_pages[:M],
                cache_lengths_r1,
                active_r1,
                out=topk_indices,
                candidate_out=candidates,
                candidate_out_lengths=candidate_lens,
                scratch=idx_scratch_r1_src,
            )
            main_stream.wait_stream(aux_stream)
            mla_r1.run(
                q,
                swa_cache,
                swa_indices,
                swa_lengths,
                sinks,
                positions,
                cos_sin,
                out,
                scratch=mla_scratch_r1 or None,
                idx_cache=kv_cache_r1,
                idx_indices=topk_indices,
                idx_lengths=top_lengths,
                idx_page_table=main_page_table,
            )
            l2_prefetch.join()

        # 3. Ratio-1 Index-Only Reindex Layer (Layers 24, 28, 32, 36)
        def r1_reidx_1stream():
            attention_front(
                qr_kv,
                q_norm_w,
                kv_norm_w,
                1e-6,
                positions,
                cos_sin,
                swa_cache,
                slots,
                64,
                qr,
                q_cast.values,
                q_cast.scales,
                kv,
            )
            wqb_kernels[bm_fp8](
                q_cast.values,
                wqb_w,
                q_cast.scales.view(torch.uint32),
                wqb_sf,
                q.view(M, wqb_n),
            )
            l2_prefetch.issue(l2_wo, M)
            rope_(q, positions, cos_sin, 1, False)
            iwqb_kernel(
                q_cast.values,
                iwqb_w,
                q_cast.scales.view(torch.uint32),
                iwqb_sf,
                positions,
                cos_sin,
                iq_data.view(M, -1),
                iq_scale.view(M, -1),
            )
            hw_partials_kernels[bm](hidden, hw_w, hw_partials)
            hw_reduce(hw_partials, hw_out)
            idx_plan_r1_reidx.run(
                iq_data,
                iq_scale,
                hw_out,
                idx_k_cache_r1,
                index_pages[:M],
                cache_lengths_r1,
                active_r1,
                candidates,
                candidate_lens,
                out=topk_indices,
                scratch=idx_scratch_r1_reidx,
            )
            mla_r1.run(
                q,
                swa_cache,
                swa_indices,
                swa_lengths,
                sinks,
                positions,
                cos_sin,
                out,
                scratch=mla_scratch_r1 or None,
                idx_cache=kv_cache_r1,
                idx_indices=topk_indices,
                idx_lengths=top_lengths,
                idx_page_table=main_page_table,
            )
            l2_prefetch.join()

        def r1_reidx_2stream():
            attention_front(
                qr_kv,
                q_norm_w,
                kv_norm_w,
                1e-6,
                positions,
                cos_sin,
                swa_cache,
                slots,
                64,
                qr,
                q_cast.values,
                q_cast.scales,
                kv,
            )
            wqb_kernels[bm_fp8](
                q_cast.values,
                wqb_w,
                q_cast.scales.view(torch.uint32),
                wqb_sf,
                q.view(M, wqb_n),
            )
            l2_prefetch.issue(l2_wo, M)
            aux_stream.wait_stream(main_stream)
            with torch.cuda.stream(aux_stream):
                rope_(q, positions, cos_sin, 1, False)
                hw_partials_kernels[bm](hidden, hw_w, hw_partials)
                hw_reduce(hw_partials, hw_out)
            iwqb_kernel(
                q_cast.values,
                iwqb_w,
                q_cast.scales.view(torch.uint32),
                iwqb_sf,
                positions,
                cos_sin,
                iq_data.view(M, -1),
                iq_scale.view(M, -1),
            )
            main_stream.wait_stream(aux_stream)
            idx_plan_r1_reidx.run(
                iq_data,
                iq_scale,
                hw_out,
                idx_k_cache_r1,
                index_pages[:M],
                cache_lengths_r1,
                active_r1,
                candidates,
                candidate_lens,
                out=topk_indices,
                scratch=idx_scratch_r1_reidx,
            )
            mla_r1.run(
                q,
                swa_cache,
                swa_indices,
                swa_lengths,
                sinks,
                positions,
                cos_sin,
                out,
                scratch=mla_scratch_r1 or None,
                idx_cache=kv_cache_r1,
                idx_indices=topk_indices,
                idx_lengths=top_lengths,
                idx_page_table=main_page_table,
            )
            l2_prefetch.join()

        families = [
            ("r2_kv_index_source", r2_1stream, r2_2stream),
            ("r1_candidate_source", r1_src_1stream, r1_src_2stream),
            ("r1_index_only_reindex", r1_reidx_1stream, r1_reidx_2stream),
        ]

        m_layer_entry = {
            "M": M,
            "eager_bit_equal_all_families": True,
            "graph_bit_equal_all_families": True,
            "uint16_diffs": 0,
            "max_abs_diff": 0.0,
        }

        warm_reps = 15
        windows = 7

        for fam_name, fn_1s, fn_2s in families:
            # Verify eager bit-exactness
            reset_state()
            with torch.cuda.stream(main_stream):
                fn_1s()
            torch.cuda.synchronize()
            ref_out = out.view(torch.uint16).clone()
            ref_swa = swa_cache.clone()
            ref_kv2 = kv_cache_r2.clone()
            ref_kv1 = kv_cache_r1.clone()
            ref_idx2 = idx_k_cache_r2.clone()
            ref_idx1 = idx_k_cache_r1.clone()
            ref_topk = topk_indices.clone()

            reset_state()
            with torch.cuda.stream(main_stream):
                fn_2s()
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.uint16), ref_out), (
                f"Eager out mismatch on {fam_name} TP={tp} M={M}"
            )
            assert torch.equal(swa_cache, ref_swa)
            assert torch.equal(kv_cache_r2, ref_kv2)
            assert torch.equal(kv_cache_r1, ref_kv1)
            assert torch.equal(idx_k_cache_r2, ref_idx2)
            assert torch.equal(idx_k_cache_r1, ref_idx1)
            assert torch.equal(topk_indices, ref_topk)

            # Capture 1-rep and warm-rep graphs and verify graph replay bit-exactness
            g1_1s = torch.cuda.CUDAGraph()
            g1_2s = torch.cuda.CUDAGraph()
            gw_1s = torch.cuda.CUDAGraph()
            gw_2s = torch.cuda.CUDAGraph()

            with torch.cuda.stream(main_stream):
                with torch.cuda.graph(g1_1s, stream=main_stream):
                    fn_1s()
                with torch.cuda.graph(g1_2s, stream=main_stream):
                    fn_2s()
                with torch.cuda.graph(gw_1s, stream=main_stream):
                    for _ in range(warm_reps):
                        fn_1s()
                with torch.cuda.graph(gw_2s, stream=main_stream):
                    for _ in range(warm_reps):
                        fn_2s()

            reset_state()
            with torch.cuda.stream(main_stream):
                g1_1s.replay()
            torch.cuda.synchronize()
            ref_g_out = out.view(torch.uint16).clone()
            ref_g_topk = topk_indices.clone()

            reset_state()
            with torch.cuda.stream(main_stream):
                g1_2s.replay()
            torch.cuda.synchronize()
            assert torch.equal(out.view(torch.uint16), ref_g_out), (
                f"Graph out mismatch on {fam_name} TP={tp} M={M}"
            )
            assert torch.equal(topk_indices, ref_g_topk), (
                f"Graph topk mismatch on {fam_name} TP={tp} M={M}"
            )

            # Warm-up replays
            with torch.cuda.stream(main_stream):
                for _ in range(5):
                    gw_1s.replay()
                    gw_2s.replay()
            torch.cuda.synchronize()

            t_warm_1s, t_warm_2s = [], []
            t_cold_1s, t_cold_2s = [], []

            for _ in range(windows):
                with torch.cuda.stream(main_stream):
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record(main_stream)
                    gw_1s.replay()
                    e.record(main_stream)
                    torch.cuda.synchronize()
                    t_warm_1s.append(s.elapsed_time(e) * 1000.0 / warm_reps)

                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record(main_stream)
                    gw_2s.replay()
                    e.record(main_stream)
                    torch.cuda.synchronize()
                    t_warm_2s.append(s.elapsed_time(e) * 1000.0 / warm_reps)

                    l2_flush_buf.add_(1)
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record(main_stream)
                    g1_1s.replay()
                    e.record(main_stream)
                    torch.cuda.synchronize()
                    t_cold_1s.append(s.elapsed_time(e) * 1000.0)

                    l2_flush_buf.add_(1)
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record(main_stream)
                    g1_2s.replay()
                    e.record(main_stream)
                    torch.cuda.synchronize()
                    t_cold_2s.append(s.elapsed_time(e) * 1000.0)

            w1 = sorted(t_warm_1s)[windows // 2]
            w2 = sorted(t_warm_2s)[windows // 2]
            c1 = sorted(t_cold_1s)[windows // 2]
            c2 = sorted(t_cold_2s)[windows // 2]

            m_layer_entry[f"{fam_name}_baseline_1107_warm_us"] = round(w1, 2)
            m_layer_entry[f"{fam_name}_candidate_2stream_warm_us"] = round(w2, 2)
            m_layer_entry[f"{fam_name}_warm_delta_us"] = round(w2 - w1, 2)
            m_layer_entry[f"{fam_name}_warm_delta_pct"] = round(
                100.0 * (w2 - w1) / w1, 1
            )
            m_layer_entry[f"{fam_name}_baseline_1107_cold_l2pf_us"] = round(c1, 2)
            m_layer_entry[f"{fam_name}_candidate_2stream_cold_l2pf_us"] = round(c2, 2)
            m_layer_entry[f"{fam_name}_cold_l2pf_delta_us"] = round(c2 - c1, 2)
            m_layer_entry[f"{fam_name}_cold_l2pf_delta_pct"] = round(
                100.0 * (c2 - c1) / c1, 1
            )

        # Aggregate across all 8 indexed layers in the 41-layer model:
        # 3 * r2_kv_index_source (layers 2, 8, 14) + 1 * r1_candidate_source (layer 20)
        # + 4 * r1_index_only_reindex (layers 24, 28, 32, 36)
        step_warm_base_us = (
            3.0 * m_layer_entry["r2_kv_index_source_baseline_1107_warm_us"]
            + 1.0 * m_layer_entry["r1_candidate_source_baseline_1107_warm_us"]
            + 4.0 * m_layer_entry["r1_index_only_reindex_baseline_1107_warm_us"]
        )
        step_warm_cand_us = (
            3.0 * m_layer_entry["r2_kv_index_source_candidate_2stream_warm_us"]
            + 1.0 * m_layer_entry["r1_candidate_source_candidate_2stream_warm_us"]
            + 4.0 * m_layer_entry["r1_index_only_reindex_candidate_2stream_warm_us"]
        )
        step_cold_base_us = (
            3.0 * m_layer_entry["r2_kv_index_source_baseline_1107_cold_l2pf_us"]
            + 1.0 * m_layer_entry["r1_candidate_source_baseline_1107_cold_l2pf_us"]
            + 4.0 * m_layer_entry["r1_index_only_reindex_baseline_1107_cold_l2pf_us"]
        )
        step_cold_cand_us = (
            3.0 * m_layer_entry["r2_kv_index_source_candidate_2stream_cold_l2pf_us"]
            + 1.0 * m_layer_entry["r1_candidate_source_candidate_2stream_cold_l2pf_us"]
            + 4.0 * m_layer_entry["r1_index_only_reindex_candidate_2stream_cold_l2pf_us"]
        )
        m_layer_entry["step_8indexed_layers_baseline_warm_us"] = round(
            step_warm_base_us, 2
        )
        m_layer_entry["step_8indexed_layers_candidate_warm_us"] = round(
            step_warm_cand_us, 2
        )
        m_layer_entry["step_8indexed_layers_warm_delta_us"] = round(
            step_warm_cand_us - step_warm_base_us, 2
        )
        m_layer_entry["step_8indexed_layers_baseline_cold_l2pf_us"] = round(
            step_cold_base_us, 2
        )
        m_layer_entry["step_8indexed_layers_candidate_cold_l2pf_us"] = round(
            step_cold_cand_us, 2
        )
        m_layer_entry["step_8indexed_layers_cold_l2pf_delta_us"] = round(
            step_cold_cand_us - step_cold_base_us, 2
        )

        full_layer_results[str(M)] = m_layer_entry
        print(
            f"TP={tp} M={M:2d} | pre_idx: 1s={t_pre_1s:5.2f} 2s={t_pre_2s:5.2f} ({t_pre_2s-t_pre_1s:+5.2f} us) "
            f"| R2 warm: {m_layer_entry['r2_kv_index_source_baseline_1107_warm_us']:6.2f}->{m_layer_entry['r2_kv_index_source_candidate_2stream_warm_us']:6.2f} ({m_layer_entry['r2_kv_index_source_warm_delta_us']:+5.2f}) "
            f"| R1src warm: {m_layer_entry['r1_candidate_source_baseline_1107_warm_us']:6.2f}->{m_layer_entry['r1_candidate_source_candidate_2stream_warm_us']:6.2f} ({m_layer_entry['r1_candidate_source_warm_delta_us']:+5.2f}) "
            f"| Reidx warm: {m_layer_entry['r1_index_only_reindex_baseline_1107_warm_us']:6.2f}->{m_layer_entry['r1_index_only_reindex_candidate_2stream_warm_us']:6.2f} ({m_layer_entry['r1_index_only_reindex_warm_delta_us']:+5.2f}) "
            f"| 8-layer step warm delta: {m_layer_entry['step_8indexed_layers_warm_delta_us']:+6.2f} us",
            flush=True,
        )

    return {
        "pre_index_dependency_bound": pre_index_bound,
        "full_attention_sublayer": full_layer_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    boot_id = subprocess.check_output(
        ["cat", "/proc/sys/kernel/random/boot_id"], text=True
    ).strip()
    uname = subprocess.check_output(["uname", "-r"], text=True).strip()
    gpu_info = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,temperature.gpu,clocks.sm",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()

    print(
        f"Running Boot {args.boot} benchmark (boot_id={boot_id}, gpu={gpu_info})...",
        flush=True,
    )
    t0 = time.time()
    tp4 = bench_tp(4)
    tp3 = bench_tp(3)
    elapsed_s = round(time.time() - t0, 2)

    payload = {
        "boot": args.boot,
        "boot_id": boot_id,
        "kernel_release": uname,
        "gpu": gpu_info,
        "elapsed_seconds": elapsed_s,
        "tp4": tp4,
        "tp3": tp3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output} in {elapsed_s}s", flush=True)


if __name__ == "__main__":
    main()
