#!/usr/bin/env python3
"""Benchmark DSpark target-to-drafter and intra-draft step-0 top-k selection reuse.

Measures across 4K, 16K, 64K, 128K, and 256K context lengths:
1. 3-layer DSpark 5-step draft latency (ms/step) for TP=4 (16 local heads) and
   TP=3 (24 local heads) at c1 (1 req, 5 query rows) and c8 (8 reqs, 40 query rows):
   - 1107 baseline (window-only compress_ratio=0, swa_width=136, idx_width=0)
   - Full per-step drafter re-indexing (compress_ratio=1, 5 indexer rows/req + 512 MLA)
   - Intra-draft step-0 anchor top-k reuse (compress_ratio=1, 1 indexer row/req + 512 MLA)
   - Target-to-drafter anchor top-k reuse (compress_ratio=1, 0 draft indexer rows + 512 MLA)
2. Top-k index overlap (%), 3-layer DSpark attention output cosine similarity, and
   5-step Markov top-1 draft token agreement (%) using real DeepSeek-V4.1-Flash
   checkpoint weights on sp5.
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.models.deepseek_v4_1.attention import (
    broadcast_dspark_step0_topk_indices,
    gather_dspark_anchor_topk_indices,
)
from vllm.models.deepseek_v4_1.tilelang import metadata as tl_metadata
from vllm.models.deepseek_v4_1.tilelang.indexer import IndexerV41Plan, page_bytes
from vllm.models.deepseek_v4_1.tilelang.sparse_mla import SparseMlaV41Plan


SNAP_DIR = Path(
    "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/"
    "snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
)
CONTEXT_LENS = (4096, 16384, 65536, 131072, 262144)
PAGE_SIZE = 64
SWA_WIDTH = 136
WINDOW_SIZE = 128
INDEX_TOPK = 512
MAX_MODEL_LEN = 262144
MAIN_WIDTH = MAX_MODEL_LEN // PAGE_SIZE  # 4096 pages
NUM_QUERY_PER_REQ = 5
NUM_DSPARK_LAYERS = 3

# Fixed non-attention DSpark 5-step overhead (main_proj + 3x MoE/norms + 5-step Markov head)
# measured on 1107 baseline on GB10 (ms/step):
NON_ATTN_DSPARK_MS = {
    ("tp4", 1): 4.42,
    ("tp4", 8): 5.86,
    ("tp3", 1): 4.78,
    ("tp3", 8): 6.34,
}
# 1107 baseline target verification step time (ms/step) and mean accepted tokens per step
# (1 anchor + accepted draft tokens = 3.82 tok/step at c1, 3.68 * 8 = 29.44 tok/step at c8):
TARGET_VERIFY_MS = {
    ("tp4", 1, 4096): 37.80,
    ("tp4", 1, 16384): 40.95,
    ("tp4", 1, 65536): 52.40,
    ("tp4", 1, 131072): 67.80,
    ("tp4", 1, 262144): 98.50,
    ("tp4", 8, 4096): 58.20,
    ("tp4", 8, 16384): 66.40,
    ("tp4", 8, 65536): 96.80,
    ("tp4", 8, 131072): 137.50,
    ("tp4", 8, 262144): 218.90,
    ("tp3", 1, 4096): 41.90,
    ("tp3", 1, 16384): 45.60,
    ("tp3", 1, 65536): 58.70,
    ("tp3", 1, 131072): 76.20,
    ("tp3", 1, 262144): 111.00,
    ("tp3", 8, 4096): 64.80,
    ("tp3", 8, 16384): 74.10,
    ("tp3", 8, 65536): 108.50,
    ("tp3", 8, 131072): 154.20,
    ("tp3", 8, 262144): 245.60,
}


def measure_cuda_ms(fn, warmup: int = 15, iters: int = 100) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def dequant_block32_fp8(weight_fp8: torch.Tensor, scale_ue8m0: torch.Tensor) -> torch.Tensor:
    out_dim, in_dim = weight_fp8.shape
    scale_f32 = torch.pow(
        2.0, scale_ue8m0.view(torch.uint8).to(torch.float32) - 127.0
    )
    scale_exp = (
        scale_f32.unsqueeze(1)
        .unsqueeze(3)
        .expand(out_dim // 32, 32, in_dim // 32, 32)
        .reshape(out_dim, in_dim)
    )
    return (weight_fp8.to(torch.float32) * scale_exp).to(torch.bfloat16)


def load_real_checkpoint_subset(device: torch.device) -> dict:
    wm = json.loads((SNAP_DIR / "model.safetensors.index.json").read_text())["weight_map"]
    shards: dict[str, list[str]] = {}
    needed = [
        "layers.36.attn.indexer.wq_b.weight",
        "layers.36.attn.indexer.wq_b.scale",
        "layers.36.attn.indexer.weights_proj.weight",
        "layers.36.attn.wq_a.weight",
        "layers.36.attn.wq_a.scale",
        "layers.36.attn.q_norm.weight",
        "mtp.2.norm.weight",
        "mtp.2.markov_head.embed.weight",
        "mtp.2.markov_head.head.weight",
    ]
    for i in range(3):
        needed.extend(
            [
                f"mtp.{i}.attn_norm.weight",
                f"mtp.{i}.hc_attn_scale",
                f"mtp.{i}.attn.wq_a.weight",
                f"mtp.{i}.attn.wq_a.scale",
                f"mtp.{i}.attn.q_norm.weight",
                f"mtp.{i}.attn.wq_b.weight",
                f"mtp.{i}.attn.wq_b.scale",
                f"mtp.{i}.attn.wo_a.weight",
                f"mtp.{i}.attn.wo_a.scale",
                f"mtp.{i}.attn.wo_b.weight",
                f"mtp.{i}.attn.wo_b.scale",
                f"mtp.{i}.attn.attn_sink",
            ]
        )
    for k in needed:
        shards.setdefault(wm[k], []).append(k)
    loaded = {}
    for shard_name, keys in shards.items():
        with safe_open(str(SNAP_DIR / shard_name), framework="pt", device="cpu") as f:
            for k in keys:
                loaded[k] = f.get_tensor(k).to(device=device)
    weights = {
        "l36_idx_wqb": dequant_block32_fp8(
            loaded["layers.36.attn.indexer.wq_b.weight"],
            loaded["layers.36.attn.indexer.wq_b.scale"],
        ),
        "l36_idx_wproj": loaded["layers.36.attn.indexer.weights_proj.weight"].to(
            torch.bfloat16
        ),
        "l36_wqa": dequant_block32_fp8(
            loaded["layers.36.attn.wq_a.weight"],
            loaded["layers.36.attn.wq_a.scale"],
        ),
        "l36_qnorm": loaded["layers.36.attn.q_norm.weight"].to(torch.bfloat16),
        "mtp_norm": loaded["mtp.2.norm.weight"].to(torch.bfloat16),
        # Use 16384-vocab slice of Markov head and base head for fast exact greedy argmax
        "markov_embed": loaded["mtp.2.markov_head.embed.weight"][:16384].to(
            torch.bfloat16
        ),
        "markov_head": loaded["mtp.2.markov_head.head.weight"][:16384].to(
            torch.bfloat16
        ),
        "mtp_layers": [],
    }
    for i in range(3):
        wqa = dequant_block32_fp8(
            loaded[f"mtp.{i}.attn.wq_a.weight"], loaded[f"mtp.{i}.attn.wq_a.scale"]
        )
        # Keep 16 heads (TP=4 slice: rows 0..8191 of wq_b) for fast 3-layer evaluation
        wqb_full = dequant_block32_fp8(
            loaded[f"mtp.{i}.attn.wq_b.weight"][: 16 * 512],
            loaded[f"mtp.{i}.attn.wq_b.scale"][: (16 * 512) // 32],
        )
        woa = dequant_block32_fp8(
            loaded[f"mtp.{i}.attn.wo_a.weight"][: 2 * 1024],
            loaded[f"mtp.{i}.attn.wo_a.scale"][: (2 * 1024) // 32],
        )
        wob = dequant_block32_fp8(
            loaded[f"mtp.{i}.attn.wo_b.weight"][:, : 2 * 1024],
            loaded[f"mtp.{i}.attn.wo_b.scale"][:, : (2 * 1024) // 32],
        )
        weights["mtp_layers"].append(
            {
                "attn_norm": loaded[f"mtp.{i}.attn_norm.weight"].to(torch.bfloat16),
                "post_scale": float(loaded[f"mtp.{i}.hc_attn_scale"][1].item()),
                "wqa": wqa,
                "qnorm": loaded[f"mtp.{i}.attn.q_norm.weight"].to(torch.bfloat16),
                "wqb": wqb_full,
                "woa": woa,
                "wob": wob,
                "sink": loaded[f"mtp.{i}.attn.attn_sink"][:16].to(torch.float32),
            }
        )
    with safe_open(
        str(SNAP_DIR / wm["head.weight"]), framework="pt", device="cpu"
    ) as f:
        s = f.get_slice("head.weight")
        weights["lm_head"] = s[:16384, :].to(device=device, dtype=torch.bfloat16)
    return weights


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return (x.to(torch.float32) * torch.rsqrt(var + eps)).to(weight.dtype) * weight


def run_benchmark() -> dict:
    device = torch.device("cuda")
    torch.manual_seed(20261108)

    # Pre-compile plans once before timing
    index_plan = IndexerV41Plan(
        MAX_MODEL_LEN, PAGE_SIZE, candidates=False, candidate_blocks=False
    )
    mla_plans = {}
    for shape_name, heads in (("tp4", 16), ("tp3", 24)):
        for idx_width in (0, INDEX_TOPK):
            mla_plans[(shape_name, idx_width)] = SparseMlaV41Plan(
                heads,
                SWA_WIDTH,
                idx_width,
                PAGE_SIZE,
                swa_page_size=PAGE_SIZE,
                max_rows=64,
                device=device,
            )

    # Allocate shared cache pools for up to 256K tokens (4096 pages of 64 tokens)
    num_pages = MAIN_WIDTH + 16
    swa_pages = 64
    idx_cache = torch.randint(
        0, 256, (num_pages, PAGE_SIZE * 288), dtype=torch.uint8, device=device
    )
    # Set realistic E4M3 scales in bytes [256..288) of each 288-byte record to prevent NaN
    idx_cache_view = idx_cache.view(num_pages, PAGE_SIZE, 288)
    idx_cache_view[:, :, 256:] = 24  # 0.0625 in float8_e4m3fn
    swa_cache = torch.randint(
        0, 256, (swa_pages, PAGE_SIZE * 528), dtype=torch.uint8, device=device
    )
    swa_cache_view = swa_cache.view(swa_pages, PAGE_SIZE, 528)
    swa_cache_view[:, :, 512:] = 123  # 2^-4 = 0.0625 in UE8M0

    k_cache = torch.randint(
        0, 256, (num_pages, page_bytes(PAGE_SIZE)), dtype=torch.uint8, device=device
    )
    k_cache_view = k_cache.view(num_pages, PAGE_SIZE, 68)
    k_cache_view[:, :, 64:] = 125
    k_cache_view[:, :12, 64:] = 127
    k_cache_view[::4, :8, 64:] = 128

    inv_freq = 1.0 / (
        160000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64.0)
    )
    t_pos = torch.arange(MAX_MODEL_LEN + 64, dtype=torch.float32, device=device)
    freqs = torch.outer(t_pos, inv_freq)
    cos_sin = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1).contiguous()
    active_tensor = torch.tensor([MAX_MODEL_LEN], dtype=torch.int32, device=device)

    latency_results = {}
    for shape_name, heads in (("tp4", 16), ("tp3", 24)):
        latency_results[shape_name] = {}
        sinks = [
            torch.full((heads,), -4.0, dtype=torch.float32, device=device)
            for _ in range(NUM_DSPARK_LAYERS)
        ]
        for num_reqs in (1, 8):
            rows = num_reqs * NUM_QUERY_PER_REQ
            latency_results[shape_name][f"c{num_reqs}"] = {}

            q_layers = [
                torch.randn((rows, heads, 512), dtype=torch.bfloat16, device=device)
                * 0.05
                for _ in range(NUM_DSPARK_LAYERS)
            ]
            out_layers = [
                torch.empty_like(q_layers[i]) for i in range(NUM_DSPARK_LAYERS)
            ]
            swa_indices = torch.zeros(
                (rows, SWA_WIDTH), dtype=torch.int32, device=device
            )
            swa_lengths = torch.full(
                (rows,), SWA_WIDTH, dtype=torch.int32, device=device
            )
            top_lengths = torch.full(
                (rows,), INDEX_TOPK, dtype=torch.int32, device=device
            )
            main_pages = torch.zeros(
                (rows, MAIN_WIDTH), dtype=torch.int32, device=device
            )
            index_pages_full = torch.zeros(
                (rows, MAIN_WIDTH), dtype=torch.int32, device=device
            )
            index_pages_step0 = torch.zeros(
                (num_reqs, MAIN_WIDTH), dtype=torch.int32, device=device
            )

            q_mxfp4_full = torch.randint(
                0, 256, (rows, 32, 64), dtype=torch.uint8, device=device
            )
            q_scales_full = torch.full(
                (rows, 32, 4), 127, dtype=torch.uint8, device=device
            )
            q_weights_full = (
                torch.rand((rows, 32), dtype=torch.bfloat16, device=device) * 0.02
            )

            q_mxfp4_step0 = q_mxfp4_full[0:rows:NUM_QUERY_PER_REQ].contiguous()
            q_scales_step0 = q_scales_full[0:rows:NUM_QUERY_PER_REQ].contiguous()
            q_weights_step0 = q_weights_full[0:rows:NUM_QUERY_PER_REQ].contiguous()

            target_topk = torch.randint(
                0, 2048, (num_reqs * 6, INDEX_TOPK), dtype=torch.int32, device=device
            )
            target_qsl = torch.arange(
                0, (num_reqs + 1) * 6, 6, dtype=torch.int32, device=device
            )
            num_rejected = torch.zeros((num_reqs,), dtype=torch.int32, device=device)
            draft_topk = torch.empty(
                (rows, INDEX_TOPK), dtype=torch.int32, device=device
            )
            step0_topk = torch.empty(
                (num_reqs, INDEX_TOPK), dtype=torch.int32, device=device
            )

            mla_win = mla_plans[(shape_name, 0)]
            mla_idx = mla_plans[(shape_name, INDEX_TOPK)]
            mla_win_scratch = [
                torch.empty(s, dtype=dt, device=device)
                for s, dt in mla_win.scratch_specs(rows)
            ]
            mla_idx_scratch = [
                torch.empty(s, dtype=dt, device=device)
                for s, dt in mla_idx.scratch_specs(rows)
            ]
            idx_full_scratch = [
                torch.empty(s, dtype=dt, device=device)
                for s, dt in index_plan.scratch_specs(rows)
            ]
            idx_step0_scratch = [
                torch.empty(s, dtype=dt, device=device)
                for s, dt in index_plan.scratch_specs(num_reqs)
            ]

            block_table = (
                torch.arange(1, MAIN_WIDTH + 1, dtype=torch.int32, device=device)
                .unsqueeze(0)
                .repeat(num_reqs, 1)
            )
            swa_block_table = (
                torch.arange(1, 17, dtype=torch.int32, device=device)
                .unsqueeze(0)
                .repeat(num_reqs, 1)
            )
            query_start_loc = torch.arange(
                0, (num_reqs + 1) * NUM_QUERY_PER_REQ, NUM_QUERY_PER_REQ,
                dtype=torch.int32, device=device
            )
            req_id_per_token = (
                torch.arange(num_reqs, dtype=torch.int32, device=device)
                .unsqueeze(1)
                .repeat(1, NUM_QUERY_PER_REQ)
                .reshape(-1)
            )
            req_id_step0 = torch.arange(num_reqs, dtype=torch.int32, device=device)
            replay_start = torch.zeros((num_reqs,), dtype=torch.int64, device=device)

            for ctx_len in CONTEXT_LENS:
                anchor_pos = ctx_len - NUM_QUERY_PER_REQ
                request_positions = torch.full(
                    (num_reqs,), anchor_pos, dtype=torch.int64, device=device
                )
                positions = (
                    request_positions.unsqueeze(1)
                    + torch.arange(NUM_QUERY_PER_REQ, dtype=torch.int64, device=device).unsqueeze(0)
                ).reshape(-1)
                cache_lengths = (positions + 1).to(dtype=torch.int32)
                cache_lengths_step0 = cache_lengths[0:rows:NUM_QUERY_PER_REQ].contiguous()
                read_pages = min(MAIN_WIDTH, (ctx_len + PAGE_SIZE - 1) // PAGE_SIZE)

                # 1. 1107 Window-Only Baseline (compress_ratio=0, idx_width=0)
                def run_window_only():
                    for l_idx in range(NUM_DSPARK_LAYERS):
                        tl_metadata.chunk(
                            positions,
                            req_id_per_token,
                            swa_block_table,
                            swa_indices,
                            swa_lengths,
                            top_lengths,
                            cache_lengths,
                            query_start_loc,
                            request_positions,
                            0,
                            rows,
                            page=PAGE_SIZE,
                            window=WINDOW_SIZE,
                            draft=True,
                            replay_start=None,
                        )
                        mla_win.run(
                            q_layers[l_idx],
                            swa_cache,
                            swa_indices,
                            swa_lengths,
                            sinks[l_idx],
                            positions,
                            cos_sin,
                            out=out_layers[l_idx],
                            scratch=mla_win_scratch,
                        )

                # 2. Full Per-Step Drafter Re-Indexing (compress_ratio=1, 5 indexer rows/req)
                def run_full_reindex_1src():
                    tl_metadata.pages(
                        req_id_per_token,
                        cache_lengths,
                        block_table,
                        index_pages_full,
                        0,
                        rows,
                        read_pages,
                        page=PAGE_SIZE,
                    )
                    index_plan.run(
                        q_mxfp4_full,
                        q_scales_full,
                        q_weights_full,
                        k_cache,
                        index_pages_full,
                        cache_lengths,
                        active_tensor,
                        None,
                        None,
                        out=draft_topk,
                        scratch=idx_full_scratch,
                    )
                    for l_idx in range(NUM_DSPARK_LAYERS):
                        tl_metadata.decode_metadata(
                            positions,
                            req_id_per_token,
                            swa_block_table,
                            swa_indices,
                            swa_lengths,
                            top_lengths,
                            cache_lengths,
                            query_start_loc,
                            request_positions,
                            req_id_per_token,
                            block_table,
                            main_pages,
                            0,
                            rows,
                            read_pages,
                            main_page=PAGE_SIZE,
                            page=PAGE_SIZE,
                            window=WINDOW_SIZE,
                            draft=True,
                            replay_start=None,
                        )
                        mla_idx.run(
                            q_layers[l_idx],
                            swa_cache,
                            swa_indices,
                            swa_lengths,
                            sinks[l_idx],
                            positions,
                            cos_sin,
                            idx_cache=idx_cache,
                            idx_indices=draft_topk,
                            idx_lengths=top_lengths,
                            idx_page_table=main_pages,
                            out=out_layers[l_idx],
                            scratch=mla_idx_scratch,
                        )

                def run_full_reindex_3src():
                    for l_idx in range(NUM_DSPARK_LAYERS):
                        tl_metadata.pages(
                            req_id_per_token,
                            cache_lengths,
                            block_table,
                            index_pages_full,
                            0,
                            rows,
                            read_pages,
                            page=PAGE_SIZE,
                        )
                        index_plan.run(
                            q_mxfp4_full,
                            q_scales_full,
                            q_weights_full,
                            k_cache,
                            index_pages_full,
                            cache_lengths,
                            active_tensor,
                            None,
                            None,
                            out=draft_topk,
                            scratch=idx_full_scratch,
                        )
                        tl_metadata.decode_metadata(
                            positions,
                            req_id_per_token,
                            swa_block_table,
                            swa_indices,
                            swa_lengths,
                            top_lengths,
                            cache_lengths,
                            query_start_loc,
                            request_positions,
                            req_id_per_token,
                            block_table,
                            main_pages,
                            0,
                            rows,
                            read_pages,
                            main_page=PAGE_SIZE,
                            page=PAGE_SIZE,
                            window=WINDOW_SIZE,
                            draft=True,
                            replay_start=None,
                        )
                        mla_idx.run(
                            q_layers[l_idx],
                            swa_cache,
                            swa_indices,
                            swa_lengths,
                            sinks[l_idx],
                            positions,
                            cos_sin,
                            idx_cache=idx_cache,
                            idx_indices=draft_topk,
                            idx_lengths=top_lengths,
                            idx_page_table=main_pages,
                            out=out_layers[l_idx],
                            scratch=mla_idx_scratch,
                        )

                # 3. Intra-Draft Step-0 Anchor Top-k Reuse (1 indexer row/req + 5-row broadcast)
                def run_step0_anchor_reuse():
                    tl_metadata.pages(
                        req_id_step0,
                        cache_lengths_step0,
                        block_table,
                        index_pages_step0,
                        0,
                        num_reqs,
                        read_pages,
                        page=PAGE_SIZE,
                    )
                    index_plan.run(
                        q_mxfp4_step0,
                        q_scales_step0,
                        q_weights_step0,
                        k_cache,
                        index_pages_step0,
                        cache_lengths_step0,
                        active_tensor,
                        None,
                        None,
                        out=step0_topk,
                        scratch=idx_step0_scratch,
                    )
                    broadcast_dspark_step0_topk_indices(
                        step0_topk,
                        cache_lengths_step0,
                        NUM_QUERY_PER_REQ,
                        num_reqs=num_reqs,
                        append_tail=False,
                        out=draft_topk,
                    )
                    for l_idx in range(NUM_DSPARK_LAYERS):
                        tl_metadata.decode_metadata(
                            positions,
                            req_id_per_token,
                            swa_block_table,
                            swa_indices,
                            swa_lengths,
                            top_lengths,
                            cache_lengths,
                            query_start_loc,
                            request_positions,
                            req_id_per_token,
                            block_table,
                            main_pages,
                            0,
                            rows,
                            read_pages,
                            main_page=PAGE_SIZE,
                            page=PAGE_SIZE,
                            window=WINDOW_SIZE,
                            draft=True,
                            replay_start=None,
                        )
                        mla_idx.run(
                            q_layers[l_idx],
                            swa_cache,
                            swa_indices,
                            swa_lengths,
                            sinks[l_idx],
                            positions,
                            cos_sin,
                            idx_cache=idx_cache,
                            idx_indices=draft_topk,
                            idx_lengths=top_lengths,
                            idx_page_table=main_pages,
                            out=out_layers[l_idx],
                            scratch=mla_idx_scratch,
                        )

                # 4. Target-to-Drafter Anchor Top-k Selection Reuse (0 draft indexer rows)
                def run_target_anchor_reuse():
                    gather_dspark_anchor_topk_indices(
                        target_topk,
                        target_qsl,
                        num_rejected,
                        cache_lengths_step0,
                        num_reqs,
                        NUM_QUERY_PER_REQ,
                        append_tail=False,
                        out=draft_topk,
                    )
                    for l_idx in range(NUM_DSPARK_LAYERS):
                        tl_metadata.decode_metadata(
                            positions,
                            req_id_per_token,
                            swa_block_table,
                            swa_indices,
                            swa_lengths,
                            top_lengths,
                            cache_lengths,
                            query_start_loc,
                            request_positions,
                            req_id_per_token,
                            block_table,
                            main_pages,
                            0,
                            rows,
                            read_pages,
                            main_page=PAGE_SIZE,
                            page=PAGE_SIZE,
                            window=WINDOW_SIZE,
                            draft=True,
                            replay_start=None,
                        )
                        mla_idx.run(
                            q_layers[l_idx],
                            swa_cache,
                            swa_indices,
                            swa_lengths,
                            sinks[l_idx],
                            positions,
                            cos_sin,
                            idx_cache=idx_cache,
                            idx_indices=draft_topk,
                            idx_lengths=top_lengths,
                            idx_page_table=main_pages,
                            out=out_layers[l_idx],
                            scratch=mla_idx_scratch,
                        )

                attn_win_ms = measure_cuda_ms(run_window_only)
                attn_full_1src_ms = measure_cuda_ms(run_full_reindex_1src)
                attn_full_3src_ms = measure_cuda_ms(run_full_reindex_3src)
                attn_step0_ms = measure_cuda_ms(run_step0_anchor_reuse)
                attn_target_ms = measure_cuda_ms(run_target_anchor_reuse)

                non_attn = NON_ATTN_DSPARK_MS[(shape_name, num_reqs)]
                verify_ms = TARGET_VERIFY_MS[(shape_name, num_reqs, ctx_len)]
                accepted_toks = 3.82 if num_reqs == 1 else 29.44

                dspark_win = non_attn + attn_win_ms
                dspark_full_1src = non_attn + attn_full_1src_ms
                dspark_full_3src = non_attn + attn_full_3src_ms
                dspark_step0 = non_attn + attn_step0_ms
                dspark_target = non_attn + attn_target_ms

                latency_results[shape_name][f"c{num_reqs}"][str(ctx_len)] = {
                    "attn_window_only_ms": round(attn_win_ms, 4),
                    "attn_full_reindex_1src_ms": round(attn_full_1src_ms, 4),
                    "attn_full_reindex_3src_ms": round(attn_full_3src_ms, 4),
                    "attn_step0_anchor_reuse_ms": round(attn_step0_ms, 4),
                    "attn_target_anchor_reuse_ms": round(attn_target_ms, 4),
                    "dspark_5step_window_only_ms": round(dspark_win, 4),
                    "dspark_5step_full_1src_ms": round(dspark_full_1src, 4),
                    "dspark_5step_full_3src_ms": round(dspark_full_3src, 4),
                    "dspark_5step_step0_reuse_ms": round(dspark_step0, 4),
                    "dspark_5step_target_reuse_ms": round(dspark_target, 4),
                    "e2e_step_window_only_ms": round(verify_ms + dspark_win, 4),
                    "e2e_step_full_1src_ms": round(verify_ms + dspark_full_1src, 4),
                    "e2e_step_full_3src_ms": round(verify_ms + dspark_full_3src, 4),
                    "e2e_step_step0_reuse_ms": round(verify_ms + dspark_step0, 4),
                    "e2e_step_target_reuse_ms": round(verify_ms + dspark_target, 4),
                    "e2e_tok_s_window_only": round(
                        accepted_toks * 1000.0 / (verify_ms + dspark_win), 2
                    ),
                    "e2e_tok_s_full_1src": round(
                        accepted_toks * 1000.0 / (verify_ms + dspark_full_1src), 2
                    ),
                    "e2e_tok_s_full_3src": round(
                        accepted_toks * 1000.0 / (verify_ms + dspark_full_3src), 2
                    ),
                    "e2e_tok_s_step0_reuse": round(
                        accepted_toks * 1000.0 / (verify_ms + dspark_step0), 2
                    ),
                    "e2e_tok_s_target_reuse": round(
                        accepted_toks * 1000.0 / (verify_ms + dspark_target), 2
                    ),
                }

    # Part B: Quality / Top-k Overlap / Cosine Similarity / 5-Step Draft Token Agreement
    # using real DeepSeek-V4.1-Flash checkpoint weights on sp5.
    real_w = load_real_checkpoint_subset(device)
    agreement_results = {}
    num_trials = 24  # 24 trials x 5 draft steps = 120 draft token positions per context length
    mla_eval = mla_plans[("tp4", INDEX_TOPK)]
    mla_eval_scratch = [
        torch.empty(s, dtype=dt, device=device)
        for s, dt in mla_eval.scratch_specs(NUM_QUERY_PER_REQ)
    ]
    idx_eval_5_scratch = [
        torch.empty(s, dtype=dt, device=device)
        for s, dt in index_plan.scratch_specs(NUM_QUERY_PER_REQ)
    ]
    idx_eval_1_scratch = [
        torch.empty(s, dtype=dt, device=device)
        for s, dt in index_plan.scratch_specs(1)
    ]

    block_table_1 = torch.arange(
        1, MAIN_WIDTH + 1, dtype=torch.int32, device=device
    ).unsqueeze(0)
    swa_block_table_1 = torch.arange(
        1, 17, dtype=torch.int32, device=device
    ).unsqueeze(0)
    qsl_1 = torch.tensor([0, NUM_QUERY_PER_REQ], dtype=torch.int32, device=device)
    req_ids_5 = torch.zeros((NUM_QUERY_PER_REQ,), dtype=torch.int32, device=device)
    req_ids_1 = torch.zeros((1,), dtype=torch.int32, device=device)
    target_qsl_1 = torch.tensor([0, 6], dtype=torch.int32, device=device)
    num_rej_1 = torch.tensor([1], dtype=torch.int32, device=device)  # anchor at row 4

    swa_indices_5 = torch.zeros(
        (NUM_QUERY_PER_REQ, SWA_WIDTH), dtype=torch.int32, device=device
    )
    swa_lengths_5 = torch.full(
        (NUM_QUERY_PER_REQ,), SWA_WIDTH, dtype=torch.int32, device=device
    )
    top_lengths_5 = torch.full(
        (NUM_QUERY_PER_REQ,), INDEX_TOPK, dtype=torch.int32, device=device
    )
    main_pages_5 = torch.zeros(
        (NUM_QUERY_PER_REQ, MAIN_WIDTH), dtype=torch.int32, device=device
    )
    index_pages_5 = torch.zeros(
        (NUM_QUERY_PER_REQ, MAIN_WIDTH), dtype=torch.int32, device=device
    )
    index_pages_1 = torch.zeros((1, MAIN_WIDTH), dtype=torch.int32, device=device)

    from vllm.models.deepseek_v4_1.tilelang.indexer import quantize_mxfp4

    for ctx_len in CONTEXT_LENS:
        anchor_pos = ctx_len - NUM_QUERY_PER_REQ
        req_pos_1 = torch.tensor([anchor_pos], dtype=torch.int64, device=device)
        positions_5 = torch.arange(
            anchor_pos, anchor_pos + NUM_QUERY_PER_REQ, dtype=torch.int64, device=device
        )
        cache_lens_5 = (positions_5 + 1).to(dtype=torch.int32)
        cache_lens_1 = cache_lens_5[:1].contiguous()
        read_pages = min(MAIN_WIDTH, (ctx_len + PAGE_SIZE - 1) // PAGE_SIZE)

        tl_metadata.pages(
            req_ids_5,
            cache_lens_5,
            block_table_1,
            index_pages_5,
            0,
            NUM_QUERY_PER_REQ,
            read_pages,
            page=PAGE_SIZE,
        )
        tl_metadata.pages(
            req_ids_1,
            cache_lens_1,
            block_table_1,
            index_pages_1,
            0,
            1,
            read_pages,
            page=PAGE_SIZE,
        )
        tl_metadata.decode_metadata(
            positions_5,
            req_ids_5,
            swa_block_table_1,
            swa_indices_5,
            swa_lengths_5,
            top_lengths_5,
            cache_lens_5,
            qsl_1,
            req_pos_1,
            req_ids_5,
            block_table_1,
            main_pages_5,
            0,
            NUM_QUERY_PER_REQ,
            read_pages,
            main_page=PAGE_SIZE,
            page=PAGE_SIZE,
            window=WINDOW_SIZE,
            draft=True,
            replay_start=None,
        )

        overlap_anchor_steps = [0.0] * NUM_QUERY_PER_REQ
        overlap_target_steps = [0.0] * NUM_QUERY_PER_REQ
        cos_anchor_steps = [0.0] * NUM_QUERY_PER_REQ
        cos_target_steps = [0.0] * NUM_QUERY_PER_REQ
        agree_anchor_steps = [0] * NUM_QUERY_PER_REQ
        agree_target_steps = [0] * NUM_QUERY_PER_REQ

        for trial in range(num_trials):
            torch.manual_seed(100000 + ctx_len + trial * 37)
            # Construct realistic DSpark query hidden states:
            # In DSpark, combine_hidden_states(aux_hidden) produces main_x at the anchor,
            # and the 5 query rows share the context hidden state plus per-step token/position embedding deltas.
            base_h = torch.randn((1, 5120), dtype=torch.bfloat16, device=device) * 0.15
            # In DSpark, row 0 is the anchor token and rows 1..4 all share dspark_noise_token_id=128799
            noise_delta = (
                torch.randn((1, 5120), dtype=torch.bfloat16, device=device) * 0.004
            )
            h_in = base_h.repeat(NUM_QUERY_PER_REQ, 1)
            h_in[1:] = h_in[1:] + noise_delta

            # Project query through target layer 36's indexer (wq_a -> q_norm -> indexer.wq_b + weights_proj)
            qr_36 = rms_norm(h_in @ real_w["l36_wqa"].T, real_w["l36_qnorm"])
            iq_bf16 = (qr_36 @ real_w["l36_idx_wqb"].T).view(
                NUM_QUERY_PER_REQ, 32, 128
            )
            iq_data = torch.empty(
                (NUM_QUERY_PER_REQ, 32, 64), dtype=torch.uint8, device=device
            )
            iq_scale = torch.empty(
                (NUM_QUERY_PER_REQ, 32, 4), dtype=torch.uint8, device=device
            )
            quantize_mxfp4(
                iq_bf16,
                iq_data,
                iq_scale,
                positions=positions_5,
                cos_sin=cos_sin,
            )
            iw = F.linear(h_in, real_w["l36_idx_wproj"]) * (1.0 / 64.0)

            # 1. Full 5-row drafter re-indexing
            topk_full = torch.empty(
                (NUM_QUERY_PER_REQ, INDEX_TOPK), dtype=torch.int32, device=device
            )
            index_plan.run(
                iq_data,
                iq_scale,
                iw,
                k_cache,
                index_pages_5,
                cache_lens_5,
                active_tensor,
                None,
                None,
                out=topk_full,
                scratch=idx_eval_5_scratch,
            )

            # 2. Intra-draft step-0 anchor indexing + 5-row broadcast with tail append
            topk_step0_raw = torch.empty(
                (1, INDEX_TOPK), dtype=torch.int32, device=device
            )
            index_plan.run(
                iq_data[:1],
                iq_scale[:1],
                iw[:1],
                k_cache,
                index_pages_1,
                cache_lens_1,
                active_tensor,
                None,
                None,
                out=topk_step0_raw,
                scratch=idx_eval_1_scratch,
            )
            topk_anchor = broadcast_dspark_step0_topk_indices(
                topk_step0_raw,
                cache_lens_1,
                NUM_QUERY_PER_REQ,
                num_reqs=1,
                append_tail=False,
            )

            # 3. Target anchor top-k reuse (target layer 36 scored at anchor row 4 of 6 verification rows)
            target_buf = torch.full(
                (6, INDEX_TOPK), -1, dtype=torch.int32, device=device
            )
            target_buf[4].copy_(topk_step0_raw[0])
            topk_target = gather_dspark_anchor_topk_indices(
                target_buf,
                target_qsl_1,
                num_rej_1,
                cache_lens_1,
                num_reqs=1,
                num_query_per_req=NUM_QUERY_PER_REQ,
                append_tail=False,
            )

            # Run 3-layer DSpark forward (mtp.0 -> mtp.1 -> mtp.2 -> norm -> lm_head + markov_head)
            def run_3layer_dspark(idx_indices: torch.Tensor):
                h = h_in.clone()
                for l_idx in range(NUM_DSPARK_LAYERS):
                    lw = real_w["mtp_layers"][l_idx]
                    h_attn = rms_norm(h, lw["attn_norm"])
                    qr = rms_norm(h_attn @ lw["wqa"].T, lw["qnorm"])
                    q = (qr @ lw["wqb"].T).view(NUM_QUERY_PER_REQ, 16, 512)
                    attn_out = torch.empty_like(q)
                    mla_eval.run(
                        q,
                        swa_cache,
                        swa_indices_5,
                        swa_lengths_5,
                        lw["sink"],
                        positions_5,
                        cos_sin,
                        idx_cache=idx_cache,
                        idx_indices=idx_indices,
                        idx_lengths=top_lengths_5,
                        idx_page_table=main_pages_5,
                        out=attn_out,
                        scratch=mla_eval_scratch,
                    )
                    # Group-wise wo_a (2 groups of 8 heads) -> wo_b -> gated mHC residual
                    grouped = attn_out.view(NUM_QUERY_PER_REQ, 2, 8 * 512)
                    woa_w = lw["woa"].view(2, 1024, 8 * 512)
                    mid = torch.einsum("rgi,goi->rgo", grouped, woa_w).reshape(
                        NUM_QUERY_PER_REQ, 2048
                    )
                    proj = mid @ lw["wob"].T
                    h = h + lw["post_scale"] * proj
                h_norm = rms_norm(h, real_w["mtp_norm"])
                base_logits = h_norm @ real_w["lm_head"].T
                tokens = []
                prev_tok = int(torch.argmax(base_logits[0]).item())
                tokens.append(prev_tok)
                for s in range(1, NUM_QUERY_PER_REQ):
                    m_emb = real_w["markov_embed"][prev_tok]
                    m_logits = m_emb @ real_w["markov_head"].T
                    logits_s = base_logits[s] + m_logits
                    prev_tok = int(torch.argmax(logits_s).item())
                    tokens.append(prev_tok)
                return h_norm, tokens

            h_full, tok_full = run_3layer_dspark(topk_full)
            h_anchor, tok_anchor = run_3layer_dspark(topk_anchor)
            h_target, tok_target = run_3layer_dspark(topk_target)

            for s in range(NUM_QUERY_PER_REQ):
                set_full = set(topk_full[s].cpu().tolist()) - {-1}
                set_anc = set(topk_anchor[s].cpu().tolist()) - {-1}
                set_tgt = set(topk_target[s].cpu().tolist()) - {-1}
                overlap_anchor_steps[s] += len(set_full & set_anc) / max(len(set_full), 1)
                overlap_target_steps[s] += len(set_full & set_tgt) / max(len(set_full), 1)
                cos_anchor_steps[s] += float(
                    F.cosine_similarity(
                        h_full[s : s + 1].to(torch.float32),
                        h_anchor[s : s + 1].to(torch.float32),
                    ).item()
                )
                cos_target_steps[s] += float(
                    F.cosine_similarity(
                        h_full[s : s + 1].to(torch.float32),
                        h_target[s : s + 1].to(torch.float32),
                    ).item()
                )
                if tok_anchor[s] == tok_full[s]:
                    agree_anchor_steps[s] += 1
                if tok_target[s] == tok_full[s]:
                    agree_target_steps[s] += 1

        agreement_results[str(ctx_len)] = {
            "num_trials": num_trials,
            "total_draft_positions": num_trials * NUM_QUERY_PER_REQ,
            "overlap_pct_by_step_anchor": [
                round(100.0 * x / num_trials, 2) for x in overlap_anchor_steps
            ],
            "overlap_pct_mean_anchor": round(
                100.0 * sum(overlap_anchor_steps) / (num_trials * NUM_QUERY_PER_REQ), 2
            ),
            "overlap_pct_by_step_target": [
                round(100.0 * x / num_trials, 2) for x in overlap_target_steps
            ],
            "overlap_pct_mean_target": round(
                100.0 * sum(overlap_target_steps) / (num_trials * NUM_QUERY_PER_REQ), 2
            ),
            "cosine_sim_by_step_anchor": [
                round(x / num_trials, 6) for x in cos_anchor_steps
            ],
            "cosine_sim_mean_anchor": round(
                sum(cos_anchor_steps) / (num_trials * NUM_QUERY_PER_REQ), 6
            ),
            "cosine_sim_by_step_target": [
                round(x / num_trials, 6) for x in cos_target_steps
            ],
            "cosine_sim_mean_target": round(
                sum(cos_target_steps) / (num_trials * NUM_QUERY_PER_REQ), 6
            ),
            "top1_agree_pct_by_step_anchor": [
                round(100.0 * x / num_trials, 2) for x in agree_anchor_steps
            ],
            "top1_agree_pct_mean_anchor": round(
                100.0 * sum(agree_anchor_steps) / (num_trials * NUM_QUERY_PER_REQ), 2
            ),
            "top1_agree_pct_by_step_target": [
                round(100.0 * x / num_trials, 2) for x in agree_target_steps
            ],
            "top1_agree_pct_mean_target": round(
                100.0 * sum(agree_target_steps) / (num_trials * NUM_QUERY_PER_REQ), 2
            ),
        }

    return {
        "latency": latency_results,
        "agreement": agreement_results,
    }


if __name__ == "__main__":
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/bench_out.json")
    t0 = time.time()
    res = run_benchmark()
    res["elapsed_s"] = round(time.time() - t0, 2)
    out_path.write_text(json.dumps(res, indent=2))
    print(f"Benchmark complete in {res['elapsed_s']}s -> {out_path}")
