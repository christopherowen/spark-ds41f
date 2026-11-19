#!/usr/bin/env python3
"""2-boot GPU recall and latency evaluation for VLLM_DS41_LSA block-summary formulations."""

import argparse
import json
import math
import time
from pathlib import Path

import tilelang
import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import pdl
from vllm.models.deepseek_v4_1.tilelang.block_shortlist import (
    STATE_BYTES,
    summarize_blocks as summarize_blocks_orig,
)
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    CANDIDATE_BLOCKS,
    IndexerV41Plan,
    quantize_mxfp4,
    write_index_keys,
)
from vllm.models.deepseek_v4_1.tilelang.query_reuse import (
    QueryReusePlan,
    prepared_scores,
)
from vllm.models.deepseek_v4_1.tilelang.shortlist_indexer import decode


@tilelang.jit
def summarize_blocks_mid_blend(page: int, beta: float = 0.25, slot_dtype: str = "int64"):
    rows, pool_pages, stride = T.dynamic("rows, pool_pages, stride")
    groups = (page // 8) * 4
    assert page in (128, 256)
    mean_w = 0.125 * (1.0 - beta)
    mid_w = 0.5 * beta

    @T.prim_func
    def update(
        Pool: T.StridedTensor([pool_pages, page * STATE_BYTES], [stride, 1], T.uint8),
        Half: T.StridedTensor([pool_pages, page * STATE_BYTES // 2], [stride // 2, 1], T.bfloat16),
        Slots: T.Tensor([rows], slot_dtype),
    ):
        with T.Kernel(rows, threads=512) as row:
            pdl.chain()
            acc = T.alloc_local([1], T.float32)
            hi = T.alloc_local([1], T.float32)
            lo = T.alloc_local([1], T.float32)
            slot = Slots[row]
            if (slot >= 0) & (slot % page == page - 1) & (slot // page < pool_pages):
                physical = slot // page
                for g, i in T.Parallel(groups, 32):
                    d = (g % 4) * 32 + i
                    tok0 = (g // 4) * 8
                    v0 = decode(
                        Pool[physical, tok0 * 64 + d // 2],
                        Pool[physical, page * 64 + tok0 * 4 + g % 4],
                        d,
                    )
                    acc[0] = v0
                    hi[0] = v0
                    lo[0] = v0
                    for k in T.serial(1, 8):
                        token = tok0 + k
                        val = decode(
                            Pool[physical, token * 64 + d // 2],
                            Pool[physical, page * 64 + token * 4 + g % 4],
                            d,
                        )
                        acc[0] += val
                        hi[0] = T.max(hi[0], val)
                        lo[0] = T.min(lo[0], val)
                    Half[physical, page * 68 // 2 + g * 32 + i] = T.cast(
                        acc[0] * mean_w + (hi[0] + lo[0]) * mid_w,
                        T.bfloat16,
                    )

    return update


@tilelang.jit
def summarize_blocks_absmax_blend(page: int, beta: float = 0.25, slot_dtype: str = "int64"):
    rows, pool_pages, stride = T.dynamic("rows, pool_pages, stride")
    groups = (page // 8) * 4
    assert page in (128, 256)
    mean_w = 0.125 * (1.0 - beta)

    @T.prim_func
    def update(
        Pool: T.StridedTensor([pool_pages, page * STATE_BYTES], [stride, 1], T.uint8),
        Half: T.StridedTensor([pool_pages, page * STATE_BYTES // 2], [stride // 2, 1], T.bfloat16),
        Slots: T.Tensor([rows], slot_dtype),
    ):
        with T.Kernel(rows, threads=512) as row:
            pdl.chain()
            acc = T.alloc_local([1], T.float32)
            ext = T.alloc_local([1], T.float32)
            slot = Slots[row]
            if (slot >= 0) & (slot % page == page - 1) & (slot // page < pool_pages):
                physical = slot // page
                for g, i in T.Parallel(groups, 32):
                    d = (g % 4) * 32 + i
                    tok0 = (g // 4) * 8
                    v0 = decode(
                        Pool[physical, tok0 * 64 + d // 2],
                        Pool[physical, page * 64 + tok0 * 4 + g % 4],
                        d,
                    )
                    acc[0] = v0
                    ext[0] = v0
                    for k in T.serial(1, 8):
                        token = tok0 + k
                        val = decode(
                            Pool[physical, token * 64 + d // 2],
                            Pool[physical, page * 64 + token * 4 + g % 4],
                            d,
                        )
                        acc[0] += val
                        ext[0] = T.if_then_else(T.abs(val) > T.abs(ext[0]), val, ext[0])
                    Half[physical, page * 68 // 2 + g * 32 + i] = T.cast(
                        acc[0] * mean_w + ext[0] * beta,
                        T.bfloat16,
                    )

    return update


@tilelang.jit
def summarize_blocks_l2_blend(page: int, beta: float = 0.35, slot_dtype: str = "int64"):
    rows, pool_pages, stride = T.dynamic("rows, pool_pages, stride")
    groups = (page // 8) * 4
    assert page in (128, 256)
    mean_w = 0.125 * (1.0 - beta)

    @T.prim_func
    def update(
        Pool: T.StridedTensor([pool_pages, page * STATE_BYTES], [stride, 1], T.uint8),
        Half: T.StridedTensor([pool_pages, page * STATE_BYTES // 2], [stride // 2, 1], T.bfloat16),
        Slots: T.Tensor([rows], slot_dtype),
    ):
        with T.Kernel(rows, threads=512) as row:
            pdl.chain()
            acc = T.alloc_local([1], T.float32)
            num = T.alloc_local([1], T.float32)
            den = T.alloc_local([1], T.float32)
            slot = Slots[row]
            if (slot >= 0) & (slot % page == page - 1) & (slot // page < pool_pages):
                physical = slot // page
                for g, i in T.Parallel(groups, 32):
                    d = (g % 4) * 32 + i
                    tok0 = (g // 4) * 8
                    v0 = decode(
                        Pool[physical, tok0 * 64 + d // 2],
                        Pool[physical, page * 64 + tok0 * 4 + g % 4],
                        d,
                    )
                    av0 = T.abs(v0)
                    acc[0] = v0
                    num[0] = v0 * av0
                    den[0] = av0 + 1e-20
                    for k in T.serial(1, 8):
                        token = tok0 + k
                        val = decode(
                            Pool[physical, token * 64 + d // 2],
                            Pool[physical, page * 64 + token * 4 + g % 4],
                            d,
                        )
                        aval = T.abs(val)
                        acc[0] += val
                        num[0] += val * aval
                        den[0] += aval
                    Half[physical, page * 68 // 2 + g * 32 + i] = T.cast(
                        acc[0] * mean_w + (num[0] / den[0]) * beta,
                        T.bfloat16,
                    )

    return update


@tilelang.jit
def prepared_scores_bbox(width: int, page: int, budget: int, alpha: float = 0.125, block_c: int = 64):
    rows, pool_pages, stride, r_stride = T.dynamic("rows, pool_pages, stride, r_stride")
    n = width // 8
    ctas = min(tilelang.cdiv(n, block_c), 128)

    @T.prim_func
    def score(
        Q: T.Tensor([rows, 32, 128], T.bfloat16),
        Scale: T.Tensor([rows, 32, 4], T.uint8),
        Weight: T.Tensor([rows, 32], T.bfloat16),
        Pool: T.StridedTensor([pool_pages, page // 8, 128], [stride, 128, 1], T.bfloat16),
        RadPool: T.StridedTensor([pool_pages, page // 8, 128], [r_stride, 128, 1], T.bfloat16),
        Table: T.Tensor([rows, width // page], T.int32),
        Length: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Score: T.Tensor([rows, n], T.bfloat16),
        Count: T.Tensor([rows], T.int32),
    ):
        with T.Kernel(ctas, rows, threads=128) as (cta, row):
            pdl.chain()
            a = T.alloc_shared([block_c, 128], T.bfloat16)
            b = T.alloc_shared([32, 128], T.bfloat16)
            b_abs = T.alloc_shared([32, 128], T.bfloat16)
            acc = T.alloc_fragment([block_c, 32], T.float32)
            terms = T.alloc_shared([block_c, 32], T.float32)
            total = T.alloc_fragment([block_c], T.float32)
            extent = T.min(T.max(Length[row], 0), T.max(T.min(Active[0], width), 0))
            full = extent // page * (page // 8)
            count = T.if_then_else(full <= budget, 0, full)
            if (cta == 0) & (T.get_thread_binding() == 0):
                Count[row] = count
            if cta * block_c < count:
                T.copy(Q[row, 0:32, 0:128], b)
                for h, d in T.Parallel(32, 128):
                    b_abs[h, d] = T.cast(T.abs(T.cast(b[h, d], T.float32)) * alpha, T.bfloat16)
                for iteration in T.serial(T.ceildiv(T.max(T.ceildiv(count, block_c) - cta, 0), ctas)):
                    start = (cta + iteration * ctas) * block_c
                    for part in T.serial(block_c // (page // 8)):
                        first = start + part * (page // 8)
                        if first < count:
                            physical = Table[row, first // (page // 8)]
                            T.copy(
                                Pool[physical, 0 : page // 8, 0:128],
                                a[part * (page // 8) : (part + 1) * (page // 8), 0:128],
                            )
                        else:
                            for c, d in T.Parallel(page // 8, 128):
                                a[part * (page // 8) + c, d] = 0
                    T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
                    T.sync_threads()
                    for part in T.serial(block_c // (page // 8)):
                        first = start + part * (page // 8)
                        if first < count:
                            physical = Table[row, first // (page // 8)]
                            T.copy(
                                RadPool[physical, 0 : page // 8, 0:128],
                                a[part * (page // 8) : (part + 1) * (page // 8), 0:128],
                            )
                        else:
                            for c, d in T.Parallel(page // 8, 128):
                                a[part * (page // 8) + c, d] = 0
                    T.gemm(a, b_abs, acc, transpose_B=True, clear_accum=False)
                    for c, h in T.Parallel(block_c, 32):
                        dot = T.max(T.cast(T.cast(acc[c, h], T.bfloat16), T.float32), 0.0)
                        terms[c, h] = T.cast(
                            T.cast(T.fmul(dot, T.cast(Weight[row, h], T.float32)), T.bfloat16),
                            T.float32,
                        )
                    T.sync_threads()
                    T.fill(total, 0.0)
                    for h in T.serial(32):
                        for c in T.Parallel(block_c):
                            total[c] += terms[c, h]
                    for c in T.Parallel(block_c):
                        if start + c < count:
                            Score[row, start + c] = T.cast(total[c], T.bfloat16)

    return score


def decode_mxfp4_pages(pool: torch.Tensor, page: int) -> torch.Tensor:
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=pool.device,
    )
    pages = pool.shape[0]
    raw_bytes = pool[:, : page * 64].view(pages, page, 64).to(torch.int32)
    scales = pool[:, page * 64 : page * 68].view(pages, page, 4).to(torch.int32)
    even = raw_bytes & 15
    odd = (raw_bytes >> 4) & 15
    codes = torch.stack((even, odd), dim=-1).view(pages, page, 128)
    mag = codes & 7
    sign = torch.where((codes & 8) != 0, -1.0, 1.0)
    scale_f32 = torch.bitwise_left_shift(scales, 23).view(torch.float32)
    scale_expanded = scale_f32.repeat_interleave(32, dim=-1)
    decoded = sign * lut[mag] * scale_expanded
    return decoded.view(pages, page // 8, 8, 128)


def build_realistic_workload(
    seq_len: int,
    page: int,
    rows: int = 8,
    num_needles_per_type: int = 16,
    seed: int = 20261108,
):
    gen = torch.Generator(device="cuda").manual_seed(seed + seq_len + page)
    num_pages = seq_len // page
    num_blocks = seq_len // 8

    num_topics = 64
    topics = torch.randn((num_topics, 128), generator=gen, device="cuda", dtype=torch.float32)
    topics = topics / topics.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    block_topic_ids = torch.randint(0, num_topics, (num_blocks,), generator=gen, device="cuda")
    block_topics = topics[block_topic_ids]

    pos = torch.arange(seq_len, device="cuda", dtype=torch.float32).view(num_blocks, 8)
    freqs = (10000.0 ** (-torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64.0)).view(1, 1, 32)
    angles = pos.unsqueeze(-1) * freqs
    rope_wave = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    noise = torch.randn((num_blocks, 8, 128), generator=gen, device="cuda", dtype=torch.float32) * 0.45
    keys_f32 = 0.55 * block_topics.unsqueeze(1) + noise
    keys_f32[:, :, :64] = 0.40 * block_topics[:, None, :64] + 0.65 * rope_wave * noise[:, :, :64]

    q_f32 = torch.randn((rows, 32, 128), generator=gen, device="cuda", dtype=torch.float32) * 0.35
    for r in range(rows):
        t1 = (r * 7 + 3) % num_topics
        t2 = (r * 13 + 11) % num_topics
        q_f32[r, :16, :] += 0.85 * topics[t1].unsqueeze(0)
        q_f32[r, 16:, :] += 0.70 * topics[t2].unsqueeze(0)

    w_f32 = torch.rand((rows, 32), generator=gen, device="cuda", dtype=torch.float32) * 0.02 + 0.005
    weights = w_f32.to(torch.bfloat16)

    usable_blocks = num_blocks - (page // 8) * 2
    total_needles_per_row = 3 * num_needles_per_type
    perm = torch.randperm(usable_blocks, generator=gen, device="cuda")[: rows * total_needles_per_row]
    perm = perm.view(rows, 3, num_needles_per_type)

    needle_blocks = {"1tok": [], "2tok": [], "4tok": []}

    for r in range(rows):
        q_dir = q_f32[r] / q_f32[r].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        w_r = w_f32[r] / w_f32[r].sum()
        needle_vec = (w_r.unsqueeze(-1) * q_dir).sum(dim=0)
        needle_vec = needle_vec / needle_vec.norm().clamp_min(1e-6)

        for kind_idx, (kind_name, width_k, amp) in enumerate(
            [("1tok", 1, 3.6), ("2tok", 2, 3.1), ("4tok", 4, 2.7)]
        ):
            b_ids = perm[r, kind_idx]
            needle_blocks[kind_name].append(b_ids)
            for j in range(num_needles_per_type):
                b = int(b_ids[j].item())
                offset = (r + j) % (8 - width_k + 1)
                keys_f32[b, :, :64] -= 0.35 * needle_vec[:64].unsqueeze(0)
                for dt in range(width_k):
                    keys_f32[b, offset + dt] = amp * needle_vec + 0.15 * torch.randn(
                        128, generator=gen, device="cuda"
                    )

    for k in needle_blocks:
        needle_blocks[k] = torch.stack(needle_blocks[k], dim=0)

    pool = torch.zeros((num_pages, page * STATE_BYTES), dtype=torch.uint8, device="cuda")
    slots = torch.arange(seq_len, dtype=torch.int64, device="cuda")
    write_index_keys(keys_f32.view(seq_len, 128).to(torch.bfloat16), pool, slots, page)

    q_mxfp4 = torch.empty((rows, 32, 64), dtype=torch.uint8, device="cuda")
    q_scales_u8 = torch.empty((rows, 32, 4), dtype=torch.uint8, device="cuda")
    quantize_mxfp4(q_f32.view(rows * 32, 128).to(torch.bfloat16), q_mxfp4, q_scales_u8)
    q_scales = q_scales_u8.view(torch.float32).reshape(rows, 32)

    return pool, q_mxfp4, q_scales, weights, needle_blocks


def compute_summary_into_pool(pool: torch.Tensor, decoded: torch.Tensor, page: int, mode: str, param: float = 0.0):
    mean = decoded.mean(dim=2)
    if mode == "mean":
        summary = mean
    elif mode == "mid_blend":
        beta = param
        mid = 0.5 * (decoded.amax(dim=2) + decoded.amin(dim=2))
        summary = (1.0 - beta) * mean + beta * mid
    elif mode == "absmax_blend":
        beta = param
        idx = decoded.abs().argmax(dim=2, keepdim=True)
        ext = torch.gather(decoded, 2, idx).squeeze(2)
        summary = (1.0 - beta) * mean + beta * ext
    elif mode == "l2_blend":
        beta = param
        aval = decoded.abs()
        l2 = (decoded * aval).sum(dim=2) / aval.sum(dim=2).clamp_min(1e-20)
        summary = (1.0 - beta) * mean + beta * l2
    elif mode == "p3_blend":
        beta = param
        sq = decoded * decoded
        p3 = (sq * decoded).sum(dim=2) / sq.sum(dim=2).clamp_min(1e-20)
        summary = (1.0 - beta) * mean + beta * p3
    else:
        raise ValueError(f"Unknown mode: {mode}")

    dest = pool.view(torch.bfloat16)[:, page * 68 // 2 : page * 100 // 2].view(
        pool.shape[0], page // 8, 128
    )
    dest.copy_(summary.to(torch.bfloat16))


def run_bbox_shortlist(
    shortlist: QueryReusePlan,
    q: torch.Tensor,
    s: torch.Tensor,
    w: torch.Tensor,
    pool: torch.Tensor,
    decoded: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    active: torch.Tensor,
    center_mode: str,
    alpha: float,
    out: torch.Tensor,
    candidate_out: torch.Tensor,
    candidate_out_lengths: torch.Tensor,
    scratch: list,
):
    rows = len(q)
    specs = shortlist._specs(rows)
    full = dict(zip(specs, scratch))

    def buffers(name, plan, b):
        return [b[f"{name}_{i}"] for i in range(len(plan.scratch_specs(1)))]

    if center_mode == "mid":
        k_center = (0.5 * (decoded.amax(dim=2) + decoded.amin(dim=2))).to(torch.bfloat16).view(-1, 128)
    elif center_mode == "mean":
        k_center = decoded.mean(dim=2).to(torch.bfloat16).view(-1, 128)
    elif center_mode == "hybrid":
        mid = 0.5 * (decoded.amax(dim=2) + decoded.amin(dim=2))
        k_center = (0.5 * decoded.mean(dim=2) + 0.5 * mid).to(torch.bfloat16).view(-1, 128)
    else:
        raise ValueError(center_mode)

    k_rad = (0.5 * (decoded.amax(dim=2) - decoded.amin(dim=2))).to(torch.bfloat16).view(-1, 128)

    b = full
    shortlist.unpack(q.view(torch.uint8), s.view(torch.uint8).reshape(rows, 32, 4), lengths, active, b["prepared_q"])
    q_bf16 = b["prepared_q"]
    q_abs_bf16 = (q_bf16.to(torch.float32).abs() * alpha).to(torch.bfloat16)

    seq_len = int(lengths[0].item())
    n_blocks = seq_len // 8
    b["counts"].fill_(n_blocks if n_blocks > shortlist.block_budget else 0)
    b["scores"].zero_()

    chunk_b = 4096
    for sb in range(0, n_blocks, chunk_b):
        eb = min(sb + chunk_b, n_blocks)
        kc = k_center[sb:eb].to(torch.float32)
        kr = k_rad[sb:eb].to(torch.float32)
        dots = torch.matmul(q_bf16.to(torch.float32), kc.t()) + torch.matmul(q_abs_bf16.to(torch.float32), kr.t())
        dots_bf16 = dots.to(torch.bfloat16).to(torch.float32).clamp_min(0.0)
        terms = (dots_bf16 * w.to(torch.float32).unsqueeze(-1)).to(torch.bfloat16).to(torch.float32)
        b["scores"][:, sb:eb] = terms.sum(dim=1).to(torch.bfloat16)

    shortlist.coarse.run(b["scores"], b["counts"], active, b["selected"], b["values"], scratch=buffers("coarse", shortlist.coarse, b))
    shortlist.assemble(b["selected"], lengths, active, b["candidates"], b["candidate_lengths"])
    _, scores = shortlist.final.run(
        q, s, w, pool, table, lengths, active, b["candidates"], b["candidate_lengths"],
        out=out, scratch=buffers("final", shortlist.final, b)
    )
    if shortlist.source:
        shortlist.select_blocks.run(
            scores, b["candidate_lengths"], active, b["compact"], b["block_values"],
            counts=candidate_out_lengths, scratch=buffers("blocks", shortlist.select_blocks, b)
        )
        shortlist.remap(b["compact"], b["candidates"], candidate_out)


def eval_recall_matrix():
    results = {}
    configs = [
        ("dense_32k", 131072, 32768, 128, False, 64),
        ("dense_64k", 131072, 65536, 128, False, 64),
        ("dense_128k", 131072, 131072, 128, False, 64),
        ("source_64k", 262144, 65536, 256, True, 512),
        ("source_128k", 262144, 131072, 256, True, 1024),
        ("source_256k", 262144, 262144, 256, True, 1024),
    ]
    candidates_to_screen = [
        ("mean", "f1", "mean", 0.0),
        ("mid_blend_0.25", "f1", "mid_blend", 0.25),
        ("mid_blend_0.375", "f1", "mid_blend", 0.375),
        ("mid_blend_0.50", "f1", "mid_blend", 0.50),
        ("mid_blend_0.75", "f1", "mid_blend", 0.75),
        ("mid_blend_1.00", "f1", "mid_blend", 1.00),
        ("absmax_blend_0.15", "f1", "absmax_blend", 0.15),
        ("absmax_blend_0.25", "f1", "absmax_blend", 0.25),
        ("absmax_blend_0.35", "f1", "absmax_blend", 0.35),
        ("absmax_blend_0.50", "f1", "absmax_blend", 0.50),
        ("l2_blend_0.35", "f1", "l2_blend", 0.35),
        ("l2_blend_0.50", "f1", "l2_blend", 0.50),
        ("l2_blend_0.75", "f1", "l2_blend", 0.75),
        ("l2_blend_1.00", "f1", "l2_blend", 1.00),
        ("p3_blend_0.35", "f1", "p3_blend", 0.35),
        ("p3_blend_0.50", "f1", "p3_blend", 0.50),
        ("p3_blend_0.75", "f1", "p3_blend", 0.75),
        ("p3_blend_1.00", "f1", "p3_blend", 1.00),
        ("bbox_mid_0.125", "f2", "mid", 0.125),
        ("bbox_mid_0.25", "f2", "mid", 0.25),
        ("bbox_mid_0.3535", "f2", "mid", 0.3535),
        ("bbox_mid_0.50", "f2", "mid", 0.50),
        ("bbox_mid_1.00", "f2", "mid", 1.00),
        ("bbox_mean_0.125", "f2", "mean", 0.125),
        ("bbox_mean_0.25", "f2", "mean", 0.25),
        ("bbox_mean_0.3535", "f2", "mean", 0.3535),
        ("bbox_mean_0.50", "f2", "mean", 0.50),
        ("bbox_mean_1.00", "f2", "mean", 1.00),
        ("bbox_hyb_0.125", "f2", "hybrid", 0.125),
        ("bbox_hyb_0.25", "f2", "hybrid", 0.25),
    ]

    rows = 8
    for cfg_name, width, seq_len, page, source, budget in configs:
        pool, q, s, w, needle_blocks = build_realistic_workload(seq_len, page, rows=rows, num_needles_per_type=16)
        decoded = decode_mxfp4_pages(pool, page)
        num_pages = seq_len // page
        table = torch.full((rows, width // page), -1, dtype=torch.int32, device="cuda")
        table[:, :num_pages] = torch.arange(num_pages, dtype=torch.int32, device="cuda").unsqueeze(0)
        lengths = torch.full((rows,), seq_len, dtype=torch.int32, device="cuda")
        active = torch.tensor([width], dtype=torch.int32, device="cuda")

        native = IndexerV41Plan(width, page, candidate_blocks=source)
        shortlist = QueryReusePlan(width, page, candidate_blocks=source, budget=budget, row_cap=64)

        native_scratch = [torch.empty(sh, dtype=dt, device="cuda") for sh, dt in native.scratch_specs(rows)]
        short_scratch = [torch.empty(sh, dtype=dt, device="cuda") for sh, dt in shortlist.scratch_specs(rows)]

        native_out = torch.full((rows, 512), -1, dtype=torch.int32, device="cuda")
        native_cand = torch.full((rows, CANDIDATE_BLOCKS), -1, dtype=torch.int32, device="cuda") if source else None
        native_cand_len = torch.full((rows,), -1, dtype=torch.int32, device="cuda") if source else None

        _, native_scores = native.run(
            q, s, w, pool[:, : page * 68], table, lengths, active,
            out=native_out, candidate_out=native_cand, candidate_out_lengths=native_cand_len,
            scratch=native_scratch,
        )
        valid_scores = native_scores[:, :seq_len].to(torch.float32)
        native_top64 = torch.topk(valid_scores, k=64, dim=-1).indices
        native_top2048 = torch.topk(valid_scores, k=2048, dim=-1).indices
        block_max_scores = valid_scores.view(rows, seq_len // 8, 8).amax(dim=-1)
        native_top2048_blocks = torch.topk(block_max_scores, k=2048, dim=-1).indices

        specs_names = list(shortlist._specs(rows).keys())
        selected_idx = specs_names.index("selected")

        cfg_res = {}
        for label, fam, mode, param in candidates_to_screen:
            lsa_out = torch.full((rows, 512), -1, dtype=torch.int32, device="cuda")
            lsa_cand = torch.full((rows, CANDIDATE_BLOCKS), -1, dtype=torch.int32, device="cuda") if source else None
            lsa_cand_len = torch.full((rows,), -1, dtype=torch.int32, device="cuda") if source else None

            if fam == "f1":
                compute_summary_into_pool(pool, decoded, page, mode, param)
                shortlist.run(
                    q, s, w, pool, table, lengths, active,
                    out=lsa_out, candidate_out=lsa_cand, candidate_out_lengths=lsa_cand_len,
                    scratch=short_scratch,
                )
            else:
                run_bbox_shortlist(
                    shortlist, q, s, w, pool, decoded, table, lengths, active,
                    center_mode=mode, alpha=param,
                    out=lsa_out, candidate_out=lsa_cand, candidate_out_lengths=lsa_cand_len,
                    scratch=short_scratch,
                )

            shortlisted_blocks = short_scratch[selected_idx]

            n_rec = {}
            for kname in ("1tok", "2tok", "4tok"):
                nb = needle_blocks[kname]
                hits, total = 0, 0
                for r in range(rows):
                    sb = set(shortlisted_blocks[r].tolist())
                    s2048 = set(native_top2048_blocks[r].tolist())
                    for b in nb[r].tolist():
                        if b in s2048:
                            total += 1
                            if b in sb:
                                hits += 1
                n_rec[kname] = round(100.0 * hits / max(total, 1), 2)

            r64, r512, r2048, sw_sum, sm_sum = 0.0, 0.0, 0.0, 0.0, 0.0
            for r in range(rows):
                out_set = set(lsa_out[r].tolist())
                nat64_set = set(native_top64[r].tolist())
                nat512_set = set(native_out[r].tolist())
                r64 += len(out_set & nat64_set) / 64.0
                r512 += len(out_set & nat512_set) / 512.0

                if source:
                    cand_set = set(lsa_cand[r].tolist())
                    nat_cand_set = set(native_cand[r].tolist())
                    r2048 += len(cand_set & nat_cand_set) / 2048.0
                else:
                    sb_set = set(shortlisted_blocks[r].tolist())
                    nat2048_blks = (native_top2048[r] // 8).tolist()
                    r2048 += sum(1 for b in nat2048_blks if b in sb_set) / 2048.0

                lsa_idx = lsa_out[r].clamp(0, seq_len - 1).long()
                nat_idx = native_out[r].clamp(0, seq_len - 1).long()
                s_lsa = valid_scores[r, lsa_idx]
                s_nat = valid_scores[r, nat_idx]
                sw_sum += (s_lsa.sum() / s_nat.sum().clamp_min(1e-6)).item()
                tau = s_nat.std().clamp_min(1e-3)
                s_max = s_nat.max()
                sm_sum += (torch.exp((s_lsa - s_max) / tau).sum() / torch.exp((s_nat - s_max) / tau).sum()).clamp_max(1.0).item()

            cfg_res[label] = {
                "needle_1tok_pct": n_rec["1tok"],
                "needle_2tok_pct": n_rec["2tok"],
                "needle_4tok_pct": n_rec["4tok"],
                "recall_at_64_pct": round(100.0 * r64 / rows, 2),
                "recall_at_512_pct": round(100.0 * r512 / rows, 2),
                "recall_at_2048_pct": round(100.0 * r2048 / rows, 2),
                "score_weighted_pct": round(100.0 * sw_sum / rows, 2),
                "softmax_mass_pct": round(100.0 * sm_sum / rows, 2),
            }
        results[cfg_name] = cfg_res
    return results


def bench_fn(fn, warmup=30, rep=150):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return round((start.elapsed_time(end) * 1000.0) / rep, 2)


def eval_latency_matrix():
    summary_timings = {}
    for page in (128, 256):
        k_orig = summarize_blocks_orig(page)
        k_mid = summarize_blocks_mid_blend(page, 0.25)
        k_abs = summarize_blocks_absmax_blend(page, 0.25)
        k_l2 = summarize_blocks_l2_blend(page, 0.35)

        num_pages = 512
        pool = torch.randint(0, 256, (num_pages, page * STATE_BYTES), dtype=torch.uint8, device="cuda")
        half = pool.view(torch.bfloat16)

        for active_pages in (4, 16, 64, 256):
            slots = torch.arange(active_pages, dtype=torch.int64, device="cuda") * page + (page - 1)
            key = f"page_{page}_active_{active_pages}"
            summary_timings[key] = {
                "mean_us": bench_fn(lambda: k_orig(pool, half, slots)),
                "mid_blend_0.25_us": bench_fn(lambda: k_mid(pool, half, slots)),
                "absmax_blend_0.25_us": bench_fn(lambda: k_abs(pool, half, slots)),
                "l2_blend_0.35_us": bench_fn(lambda: k_l2(pool, half, slots)),
            }

    plan_timings = {}
    for cfg_name, width, seq_len, page, source, budget in [
        ("dense_32k", 131072, 32768, 128, False, 64),
        ("dense_64k", 131072, 65536, 128, False, 64),
        ("dense_128k", 131072, 131072, 128, False, 64),
        ("source_64k", 262144, 65536, 256, True, 512),
        ("source_128k", 262144, 131072, 256, True, 1024),
        ("source_256k", 262144, 262144, 256, True, 1024),
    ]:
        rows = 8
        num_pages = seq_len // page
        pool = torch.randint(0, 256, (num_pages, page * STATE_BYTES), dtype=torch.uint8, device="cuda")
        rad_pool = torch.zeros((num_pages, page // 8, 128), dtype=torch.bfloat16, device="cuda")
        q = torch.randint(0, 256, (rows, 32, 64), dtype=torch.uint8, device="cuda")
        s_u8 = torch.randint(120, 130, (rows, 32, 4), dtype=torch.uint8, device="cuda")
        s = s_u8.view(torch.float32).reshape(rows, 32)
        w = torch.randn((rows, 32), dtype=torch.bfloat16, device="cuda")
        table = torch.full((rows, width // page), -1, dtype=torch.int32, device="cuda")
        table[:, :num_pages] = torch.arange(num_pages, dtype=torch.int32, device="cuda").unsqueeze(0)
        lengths = torch.full((rows,), seq_len, dtype=torch.int32, device="cuda")
        active = torch.tensor([width], dtype=torch.int32, device="cuda")

        native = IndexerV41Plan(width, page, candidate_blocks=source)
        shortlist = QueryReusePlan(width, page, candidate_blocks=source, budget=budget, row_cap=64)
        s_bbox = prepared_scores_bbox(width, page, budget * 8, 0.125)

        nat_scratch = [torch.empty(sh, dtype=dt, device="cuda") for sh, dt in native.scratch_specs(rows)]
        sh_scratch = [torch.empty(sh, dtype=dt, device="cuda") for sh, dt in shortlist.scratch_specs(rows)]
        out = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
        cand = torch.empty((rows, CANDIDATE_BLOCKS), dtype=torch.int32, device="cuda") if source else None
        cand_len = torch.empty((rows,), dtype=torch.int32, device="cuda") if source else None

        specs = dict(zip(shortlist._specs(rows), sh_scratch))
        means = pool.view(torch.bfloat16)[:, page * 68 // 2 : page * 100 // 2].view(num_pages, page // 8, 128)

        t_q_f1 = bench_fn(
            lambda: (
                shortlist.unpack(q, s_u8, lengths, active, specs["prepared_q"]),
                shortlist.prepared_score(specs["prepared_q"], s_u8, w, means, table, lengths, active, specs["scores"], specs["counts"]),
            )
        )
        t_q_f2 = bench_fn(
            lambda: (
                shortlist.unpack(q, s_u8, lengths, active, specs["prepared_q"]),
                s_bbox(specs["prepared_q"], s_u8, w, means, rad_pool, table, lengths, active, specs["scores"], specs["counts"]),
            )
        )
        t_full_f1 = bench_fn(
            lambda: shortlist.run(
                q, s, w, pool, table, lengths, active,
                out=out, candidate_out=cand, candidate_out_lengths=cand_len, scratch=sh_scratch
            )
        )
        t_native = bench_fn(
            lambda: native.run(
                q, s, w, pool[:, : page * 68], table, lengths, active,
                out=out, candidate_out=cand, candidate_out_lengths=cand_len, scratch=nat_scratch
            )
        )
        plan_timings[cfg_name] = {
            "coarse_score_f1_us": t_q_f1,
            "coarse_score_f2_bbox_us": t_q_f2,
            "full_plan_f1_us": t_full_f1,
            "full_plan_f2_bbox_us": round(t_full_f1 - t_q_f1 + t_q_f2, 2),
            "native_indexer_us": t_native,
        }

    return {"summarize_blocks": summary_timings, "plans_rows8": plan_timings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "boot": args.boot,
        "device": torch.cuda.get_device_name(0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recall": eval_recall_matrix(),
        "latency": eval_latency_matrix(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Boot {args.boot} complete -> {args.out}")


if __name__ == "__main__":
    main()
