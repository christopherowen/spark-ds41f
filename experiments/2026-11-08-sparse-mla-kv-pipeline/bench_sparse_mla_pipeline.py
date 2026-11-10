#!/usr/bin/env python3
"""Two-boot benchmark and bit-exactness verification for Sparse MLA Stage-1 KV gather
and shared-memory staging pipeline optimizations on SM121 (GB10).
"""

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("TILELANG_CACHE_DIR", "/tmp/bench_mla_pipe/tl_cache")

import tilelang
import tilelang.language as T
import torch
from vllm.models.deepseek_v4_1.tilelang import pdl, prepare
from vllm.models.deepseek_v4_1.tilelang import sparse_mla as orig_mod
from vllm.models.deepseek_v4_1.tilelang.sparse_mla import (
    DECODE_MAX_WAVES,
    HEAD_DIM,
    IDX_GROUP,
    IDX_RECORD_BYTES,
    LOG2E,
    ROPE_DIM,
    SEGMENT_CHUNKS,
    SWA_GROUP,
    SWA_RECORD_BYTES,
    wo_input,
)


def make_candidate_sparse_mla(fix_s2: bool = False):
    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def candidate_sparse_mla_fwd_v41(
        heads: int,
        swa_width: int,
        idx_width: int,
        idx_page_size: int = 64,
        schedule: str = "extend",
        swa_page_size: int = 1,
        sm_scale: float = HEAD_DIM**-0.5,
        block_H: int = 32,
        block_I: int = 32,
        threads: int = 256,
        segment_chunks: int = SEGMENT_CHUNKS,
        skip_inactive: bool = True,
    ):
        rows, swa_pool, swa_stride, idx_pool, idx_stride, idx_pages = T.dynamic(
            "rows, swa_pool, swa_stride, idx_pool, idx_stride, idx_pages"
        )
        positions_max = T.dynamic("positions_max")
        swa_bytes = swa_page_size * SWA_RECORD_BYTES
        idx_bytes = idx_page_size * IDX_RECORD_BYTES

        D = HEAD_DIM
        BI = block_I
        dtype = T.bfloat16
        accum_dtype = T.float32
        scale_log2 = sm_scale * LOG2E

        head_blocks = tilelang.cdiv(heads, block_H)
        pad_heads = heads % block_H != 0
        swa_blocks = tilelang.cdiv(swa_width, BI)
        idx_blocks = tilelang.cdiv(idx_width, BI)
        chunks = swa_blocks + idx_blocks
        G = segment_chunks
        segments = tilelang.cdiv(chunks, G)
        slot_count = chunks * BI if schedule == "extend" else G * BI
        copies = tilelang.cdiv(BI * SWA_RECORD_BYTES // 16, threads)
        swa_rows = tilelang.cdiv(BI, 6) * 6
        idx_rows = swa_rows * SWA_RECORD_BYTES // IDX_RECORD_BYTES

        @T.macro
        def gather_swa(SwaIndices, SwaLengths, row, first, count, slots, base):
            for i in T.Parallel(count):
                pos = first + i
                slot = SwaIndices[row, T.min(pos, swa_width - 1)]
                slots[base + i] = T.if_then_else(
                    (pos < swa_width) & (pos < SwaLengths[row]) & (slot >= 0),
                    slot,
                    -1,
                )

        @T.macro
        def gather_idx(
            IdxIndices, IdxLengths, IdxPageTable, row, first, count, slots, base
        ):
            for i in T.Parallel(count):
                pos = first + i
                logical = IdxIndices[row, T.min(pos, idx_width - 1)]
                page = IdxPageTable[
                    row, T.min(T.max(logical, 0) // idx_page_size, idx_pages - 1)
                ]
                slots[base + i] = T.if_then_else(
                    (pos < idx_width) & (pos < IdxLengths[row]) & (logical >= 0),
                    page * idx_page_size + logical % idx_page_size,
                    -1,
                )

        @T.macro
        def load_copy_slots(chunk, slots, base, copy_slot):
            is_swa = True if not idx_blocks else chunk < swa_blocks
            tx = T.get_thread_binding()
            if is_swa:
                swa_segs = SWA_RECORD_BYTES // 16
                for k in T.unroll(tilelang.cdiv(BI * swa_segs, threads)):
                    copy_slot[k] = slots[
                        base + T.min((k * threads + tx) // swa_segs, BI - 1)
                    ]
            else:
                idx_segs = IDX_RECORD_BYTES // 16
                for k in T.unroll(tilelang.cdiv(BI * idx_segs, threads)):
                    copy_slot[k] = slots[
                        base + T.min((k * threads + tx) // idx_segs, BI - 1)
                    ]

        @T.macro
        def issue_stage_records(
            Records, record_bytes, page_size, staged, copy_slot
        ):
            segments_rec = record_bytes // 16
            tx = T.get_thread_binding()
            for k in T.unroll(tilelang.cdiv(BI * segments_rec, threads)):
                e = k * threads + tx
                if e < BI * segments_rec:
                    T.ptx_cp_async(
                        T.access_ptr(
                            staged[e // segments_rec, e % segments_rec * 16],
                            "w",
                            16,
                        ),
                        T.access_ptr(
                            Records[
                                T.max(copy_slot[k], 0) // page_size,
                                T.max(copy_slot[k], 0) % page_size * record_bytes
                                + e % segments_rec * 16,
                            ],
                            "r",
                            16,
                        ),
                        16,
                        copy_slot[k] >= 0,
                    )
            T.ptx_commit_group()

        @T.macro
        def issue_stage_chunk(
            SwaRecords, IdxRecords, chunk, staged_swa, staged_idx, copy_slot
        ):
            is_swa = True if not idx_blocks else chunk < swa_blocks
            if is_swa:
                issue_stage_records(
                    SwaRecords,
                    SWA_RECORD_BYTES,
                    swa_page_size,
                    staged_swa,
                    copy_slot,
                )
            else:
                issue_stage_records(
                    IdxRecords,
                    IDX_RECORD_BYTES,
                    idx_page_size,
                    staged_idx,
                    copy_slot,
                )

        @T.macro
        def stage_chunk(
            SwaRecords,
            IdxRecords,
            chunk,
            slots,
            base,
            staged_swa,
            staged_idx,
            copy_slot,
        ):
            load_copy_slots(chunk, slots, base, copy_slot)
            issue_stage_chunk(
                SwaRecords, IdxRecords, chunk, staged_swa, staged_idx, copy_slot
            )

        @T.macro
        def dequant_swa(staged_swa, staged_swa_fp8, KV_shared):
            for i, d in T.Parallel(BI, D):
                scale = T.reinterpret(
                    T.shift_left(
                        T.cast(staged_swa[i, D + d // SWA_GROUP], T.uint32), 23
                    ),
                    accum_dtype,
                )
                KV_shared[i, d] = T.cast(
                    T.cast(staged_swa_fp8[i, d], accum_dtype) * scale, dtype
                )

        @T.macro
        def dequant_idx(staged_idx_fp8, staged_idx_fp4, KV_shared):
            for i, d in T.Parallel(BI, D):
                scale = T.cast(
                    staged_idx_fp8[i, D // 2 + d // IDX_GROUP], accum_dtype
                )
                KV_shared[i, d] = T.cast(
                    T.cast(staged_idx_fp4[i, d], accum_dtype) * scale, dtype
                )

        @T.macro
        def dequant_chunk(
            chunk,
            staged_swa,
            staged_swa_fp8,
            staged_idx_fp8,
            staged_idx_fp4,
            KV_shared,
        ):
            is_swa = True if not idx_blocks else chunk < swa_blocks
            if is_swa:
                dequant_swa(staged_swa, staged_swa_fp8, KV_shared)
            else:
                dequant_idx(staged_idx_fp8, staged_idx_fp4, KV_shared)

        @T.macro
        def load_q(Q, row, h0, Q_shared):
            if pad_heads:
                for h, d in T.Parallel(block_H, D):
                    Q_shared[h, d] = T.if_then_else(
                        h0 + h < heads,
                        Q[row, T.min(h0 + h, heads - 1), d],
                        0,
                    )
            else:
                T.copy(Q[row, h0 : h0 + block_H, :], Q_shared)

        @T.macro
        def chunk_update(
            Q_shared,
            KV_shared,
            slots,
            base,
            P_shared,
            acc_s,
            acc_g,
            m_g,
            l_g,
            m_prev,
            alpha,
            row_sum,
        ):
            T.gemm(
                Q_shared,
                KV_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
                clear_accum=True,
            )
            for h, i in T.Parallel(block_H, BI):
                acc_s[h, i] = T.if_then_else(
                    slots[base + i] >= 0,
                    T.fmul(acc_s[h, i], scale_log2),
                    -T.infinity(accum_dtype),
                )
            T.copy(m_g, m_prev)
            T.reduce_max(acc_s, m_g, dim=1, clear=True)
            for h in T.Parallel(block_H):
                m_g[h] = T.max(m_g[h], m_prev[h])
                alpha[h] = T.if_then_else(
                    m_prev[h] == -T.infinity(accum_dtype),
                    0,
                    T.exp2(m_prev[h] - m_g[h]),
                )
            for h, i in T.Parallel(block_H, BI):
                acc_s[h, i] = T.if_then_else(
                    acc_s[h, i] != -T.infinity(accum_dtype),
                    T.exp2(acc_s[h, i] - m_g[h]),
                    0,
                )
            T.reduce_sum(acc_s, row_sum, dim=1)
            for h in T.Parallel(block_H):
                l_g[h] = T.fma(l_g[h], alpha[h], row_sum[h])
            for h, d in T.Parallel(block_H, D):
                acc_g[h, d] = T.fmul(acc_g[h, d], alpha[h])
            T.copy(acc_s, P_shared)
            T.gemm(P_shared, KV_shared, acc_g, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def clear_segment(acc_g, m_g, l_g):
            T.fill(acc_g, 0)
            T.fill(l_g, 0)
            T.fill(m_g, -T.infinity(accum_dtype))

        @T.macro
        def sink_log2(Sinks, h):
            return T.fmul(
                T.if_then_else(h < heads, Sinks[T.min(h, heads - 1)], 0), LOG2E
            )

        @T.macro
        def row_active(SwaLengths, IdxLengths, row):
            if not idx_blocks:
                return SwaLengths[row] > 0
            return (SwaLengths[row] > 0) | (IdxLengths[row] > 0)

        @T.macro
        def seg_active(seg, SwaLengths, IdxLengths, row):
            swa_active = (seg * G < swa_blocks) & (seg * G * BI < SwaLengths[row])
            if not idx_blocks:
                return swa_active
            idx_first = T.max(seg * G - swa_blocks, 0) * BI
            return swa_active | (
                (seg * G + G > swa_blocks)
                & (seg * G < chunks)
                & (idx_first < IdxLengths[row])
            )

        @T.macro
        def extend_body(
            Q,
            SwaRecords,
            SwaIndices,
            SwaLengths,
            IdxRecords,
            IdxIndices,
            IdxLengths,
            IdxPageTable,
            Sinks,
            Positions,
            CosSin,
            Output,
        ):
            with T.Kernel(head_blocks, rows, threads=threads) as (hb, row):
                Q_shared = T.alloc_shared([block_H, D], dtype)
                KV_shared = T.alloc_shared([BI, D], dtype)
                P_shared = T.alloc_shared([block_H, BI], dtype)
                cs_shared = T.alloc_shared([ROPE_DIM], accum_dtype)
                staged_swa = T.alloc_shared([swa_rows, SWA_RECORD_BYTES], T.uint8)
                staged_swa_fp8 = T.view(staged_swa, dtype=T.float8_e4m3fn)
                staged_idx = T.view(
                    staged_swa, [idx_rows, IDX_RECORD_BYTES], T.uint8
                )
                staged_idx_fp8 = T.view(
                    staged_swa, [idx_rows, IDX_RECORD_BYTES], T.float8_e4m3fn
                )
                staged_idx_fp4 = T.view(
                    staged_swa, [idx_rows, IDX_RECORD_BYTES * 2], T.float4_e2m1fn
                )
                slots = T.alloc_shared([slot_count], T.int32)
                copy_slot = T.alloc_local([copies], T.int32)
                acc_s = T.alloc_fragment([block_H, BI], accum_dtype)
                acc_g = T.alloc_fragment([block_H, D], accum_dtype)
                acc_o = T.alloc_fragment([block_H, D], accum_dtype)
                m_g = T.alloc_fragment([block_H], accum_dtype)
                l_g = T.alloc_fragment([block_H], accum_dtype)
                m_prev = T.alloc_fragment([block_H], accum_dtype)
                alpha = T.alloc_fragment([block_H], accum_dtype)
                row_sum = T.alloc_fragment([block_H], accum_dtype)
                M = T.alloc_fragment([block_H], accum_dtype)
                L = T.alloc_fragment([block_H], accum_dtype)
                a = T.alloc_fragment([block_H], accum_dtype)
                b = T.alloc_fragment([block_H], accum_dtype)
                X = T.alloc_fragment([block_H * D // 32, 32], accum_dtype)
                amax = T.alloc_fragment([block_H * D // 32], accum_dtype)
                inv = T.alloc_fragment([block_H * D // 32], accum_dtype)
                scale = T.alloc_fragment([block_H * D // 32], accum_dtype)

                h0 = hb * block_H
                for h in T.Parallel(block_H):
                    M[h] = sink_log2(Sinks, h0 + h)
                pos_ext = Positions[row]
                for i in T.Parallel(ROPE_DIM):
                    cs_shared[i] = CosSin[pos_ext, i]

                pdl.chain()

                if row_active(SwaLengths, IdxLengths, row):
                    gather_swa(
                        SwaIndices, SwaLengths, row, 0, swa_blocks * BI, slots, 0
                    )
                    if idx_blocks:
                        gather_idx(
                            IdxIndices,
                            IdxLengths,
                            IdxPageTable,
                            row,
                            0,
                            idx_blocks * BI,
                            slots,
                            swa_blocks * BI,
                        )
                    T.sync_threads()
                    stage_chunk(
                        SwaRecords,
                        IdxRecords,
                        0,
                        slots,
                        0,
                        staged_swa,
                        staged_idx,
                        copy_slot,
                    )
                    load_q(Q, row, h0, Q_shared)
                    T.fill(acc_o, 0)
                    T.fill(L, 1)

                    active_segs = T.alloc_var(T.int32)
                    active_end = T.alloc_var(T.int32)
                    if idx_blocks and skip_inactive and swa_blocks % G == 0:
                        active_segs = (swa_blocks // G) + T.min(
                            tilelang.cdiv(T.max(IdxLengths[row], 0), G * BI),
                            segments - (swa_blocks // G),
                        )
                        active_end = T.min(active_segs * G, chunks)
                    else:
                        active_segs = segments
                        active_end = chunks
                    for seg in T.serial(active_segs):
                        clear_segment(acc_g, m_g, l_g)
                        for c in T.serial(G):
                            chunk = seg * G + c
                            if chunk < chunks:
                                T.ptx_wait_group(0)
                                T.sync_threads()
                                dequant_chunk(
                                    chunk,
                                    staged_swa,
                                    staged_swa_fp8,
                                    staged_idx_fp8,
                                    staged_idx_fp4,
                                    KV_shared,
                                )
                                T.sync_threads()
                                if chunk + 1 < active_end:
                                    stage_chunk(
                                        SwaRecords,
                                        IdxRecords,
                                        chunk + 1,
                                        slots,
                                        (chunk + 1) * BI,
                                        staged_swa,
                                        staged_idx,
                                        copy_slot,
                                    )
                                chunk_update(
                                    Q_shared,
                                    KV_shared,
                                    slots,
                                    chunk * BI,
                                    P_shared,
                                    acc_s,
                                    acc_g,
                                    m_g,
                                    l_g,
                                    m_prev,
                                    alpha,
                                    row_sum,
                                )
                        for h in T.Parallel(block_H):
                            m_new = T.max(M[h], m_g[h])
                            a[h] = T.exp2(M[h] - m_new)
                            b[h] = T.exp2(m_g[h] - m_new)
                            L[h] = T.fma(l_g[h], b[h], T.fmul(L[h], a[h]))
                            M[h] = m_new
                        for h, d in T.Parallel(block_H, D):
                            acc_o[h, d] = T.fma(
                                acc_g[h, d], b[h], T.fmul(acc_o[h, d], a[h])
                            )

                    for h in T.Parallel(block_H):
                        L[h] = 1.0 / L[h]
                    T.sync_threads()
                    for h, d in T.Parallel(block_H, D):
                        Q_shared[h, d] = T.fmul(acc_o[h, d], L[h])
                    wo_input(
                        Q_shared, block_H, None, cs_shared, X, amax, inv, scale
                    )
                    for h, d in T.Parallel(block_H, D):
                        if h0 + h < heads:
                            Output[row, h0 + h, d] = Q_shared[h, d]
                else:
                    for h, d in T.Parallel(block_H, D):
                        if h0 + h < heads:
                            Output[row, h0 + h, d] = 0

        @T.macro
        def decode_body(
            Q,
            SwaRecords,
            SwaIndices,
            SwaLengths,
            IdxRecords,
            IdxIndices,
            IdxLengths,
            IdxPageTable,
            Sinks,
            Positions,
            CosSin,
            Partial,
            PartialStats,
            Output,
        ):
            with T.Kernel(segments, head_blocks, rows, threads=threads) as (
                seg,
                hb,
                row,
            ):
                pdl.chain()
                Q_shared = T.alloc_shared([block_H, D], dtype)
                KV_shared = T.alloc_shared([BI, D], dtype)
                P_shared = T.alloc_shared([block_H, BI], dtype)
                staged_swa = T.alloc_shared([swa_rows, SWA_RECORD_BYTES], T.uint8)
                staged_swa_fp8 = T.view(staged_swa, dtype=T.float8_e4m3fn)
                staged_idx = T.view(
                    staged_swa, [idx_rows, IDX_RECORD_BYTES], T.uint8
                )
                staged_idx_fp8 = T.view(
                    staged_swa, [idx_rows, IDX_RECORD_BYTES], T.float8_e4m3fn
                )
                staged_idx_fp4 = T.view(
                    staged_swa, [idx_rows, IDX_RECORD_BYTES * 2], T.float4_e2m1fn
                )
                slots = T.alloc_shared([slot_count], T.int32)
                copy_slot = T.alloc_local([copies], T.int32)
                acc_s = T.alloc_fragment([block_H, BI], accum_dtype)
                acc_g = T.alloc_fragment([block_H, D], accum_dtype)
                m_g = T.alloc_fragment([block_H], accum_dtype)
                l_g = T.alloc_fragment([block_H], accum_dtype)
                m_prev = T.alloc_fragment([block_H], accum_dtype)
                alpha = T.alloc_fragment([block_H], accum_dtype)
                row_sum = T.alloc_fragment([block_H], accum_dtype)
                h0 = hb * block_H

                if not skip_inactive or seg_active(seg, SwaLengths, IdxLengths, row):
                    if not idx_blocks:
                        gather_swa(
                            SwaIndices,
                            SwaLengths,
                            row,
                            seg * G * BI,
                            G * BI,
                            slots,
                            0,
                        )
                    elif swa_blocks % G == 0:
                        if seg * G < swa_blocks:
                            gather_swa(
                                SwaIndices,
                                SwaLengths,
                                row,
                                seg * G * BI,
                                G * BI,
                                slots,
                                0,
                            )
                        else:
                            gather_idx(
                                IdxIndices,
                                IdxLengths,
                                IdxPageTable,
                                row,
                                (seg * G - swa_blocks) * BI,
                                G * BI,
                                slots,
                                0,
                            )
                    else:
                        for c in T.serial(G):
                            chunk = seg * G + c
                            if chunk < swa_blocks:
                                gather_swa(
                                    SwaIndices,
                                    SwaLengths,
                                    row,
                                    chunk * BI,
                                    BI,
                                    slots,
                                    c * BI,
                                )
                            elif chunk < chunks:
                                gather_idx(
                                    IdxIndices,
                                    IdxLengths,
                                    IdxPageTable,
                                    row,
                                    (chunk - swa_blocks) * BI,
                                    BI,
                                    slots,
                                    c * BI,
                                )
                    T.sync_threads()
                    stage_chunk(
                        SwaRecords,
                        IdxRecords,
                        seg * G,
                        slots,
                        0,
                        staged_swa,
                        staged_idx,
                        copy_slot,
                    )
                    load_q(Q, row, h0, Q_shared)
                    clear_segment(acc_g, m_g, l_g)
                    for c in T.serial(G):
                        chunk = seg * G + c
                        if chunk < chunks:
                            if (c + 1 < G) & (chunk + 1 < chunks):
                                load_copy_slots(
                                    chunk + 1, slots, (c + 1) * BI, copy_slot
                                )
                            T.ptx_wait_group(0)
                            T.sync_threads()
                            dequant_chunk(
                                chunk,
                                staged_swa,
                                staged_swa_fp8,
                                staged_idx_fp8,
                                staged_idx_fp4,
                                KV_shared,
                            )
                            T.sync_threads()
                            if (c + 1 < G) & (chunk + 1 < chunks):
                                issue_stage_chunk(
                                    SwaRecords,
                                    IdxRecords,
                                    chunk + 1,
                                    staged_swa,
                                    staged_idx,
                                    copy_slot,
                                )
                            chunk_update(
                                Q_shared,
                                KV_shared,
                                slots,
                                c * BI,
                                P_shared,
                                acc_s,
                                acc_g,
                                m_g,
                                l_g,
                                m_prev,
                                alpha,
                                row_sum,
                            )
                    for h, d in T.Parallel(block_H, D):
                        if h0 + h < heads:
                            Partial[row, seg, h0 + h, d] = acc_g[h, d]
                    for h in T.Parallel(block_H):
                        if h0 + h < heads:
                            PartialStats[row, seg, h0 + h, 0] = m_g[h]
                            PartialStats[row, seg, h0 + h, 1] = l_g[h]
                elif fix_s2:
                    for h in T.Parallel(block_H):
                        if h0 + h < heads:
                            PartialStats[row, seg, h0 + h, 0] = -T.infinity(
                                accum_dtype
                            )
                            PartialStats[row, seg, h0 + h, 1] = T.Cast(
                                accum_dtype, 0
                            )

            with T.Kernel(heads, rows, threads=128) as (hc, rc):
                o = T.alloc_fragment([D], accum_dtype)
                O_shared = T.alloc_shared([1, D], dtype)
                cs_shared = T.alloc_shared([ROPE_DIM], accum_dtype)
                X = T.alloc_fragment([D // 32, 32], accum_dtype)
                amax = T.alloc_fragment([D // 32], accum_dtype)
                inv = T.alloc_fragment([D // 32], accum_dtype)
                scale = T.alloc_fragment([D // 32], accum_dtype)
                M = T.alloc_var(accum_dtype)
                L = T.alloc_var(accum_dtype)
                a = T.alloc_var(accum_dtype)
                b = T.alloc_var(accum_dtype)
                m_new = T.alloc_var(accum_dtype)

                M = sink_log2(Sinks, hc)
                pos_dec = Positions[rc]
                for i in T.Parallel(ROPE_DIM):
                    cs_shared[i] = CosSin[pos_dec, i]
                L = 1.0
                T.clear(o)
                pdl.chain()
                if not skip_inactive or row_active(SwaLengths, IdxLengths, rc):
                    for c in T.unroll(segments):
                        if fix_s2:
                            m_new = T.max(M, PartialStats[rc, c, hc, 0])
                            a = T.exp2(M - m_new)
                            b = T.exp2(PartialStats[rc, c, hc, 0] - m_new)
                            L = T.fma(PartialStats[rc, c, hc, 1], b, T.fmul(L, a))
                            M = m_new
                            if not skip_inactive or PartialStats[rc, c, hc, 1] > 0:
                                for d in T.Parallel(D):
                                    o[d] = T.fma(
                                        Partial[rc, c, hc, d], b, T.fmul(o[d], a)
                                    )
                        else:
                            if not skip_inactive or seg_active(
                                c, SwaLengths, IdxLengths, rc
                            ):
                                m_new = T.max(M, PartialStats[rc, c, hc, 0])
                                a = T.exp2(M - m_new)
                                b = T.exp2(PartialStats[rc, c, hc, 0] - m_new)
                                L = T.fma(
                                    PartialStats[rc, c, hc, 1], b, T.fmul(L, a)
                                )
                                M = m_new
                                for d in T.Parallel(D):
                                    o[d] = T.fma(
                                        Partial[rc, c, hc, d], b, T.fmul(o[d], a)
                                    )
                    L = 1.0 / L
                    for d in T.Parallel(D):
                        O_shared[0, d] = T.fmul(o[d], L)
                    wo_input(O_shared, 1, None, cs_shared, X, amax, inv, scale)
                    for d in T.Parallel(D):
                        Output[rc, hc, d] = O_shared[0, d]
                else:
                    for d in T.Parallel(D):
                        Output[rc, hc, d] = 0

        if schedule == "extend":
            if idx_blocks:

                @T.prim_func
                def sparse_mla(
                    Q: T.Tensor([rows, heads, D], dtype),
                    SwaRecords: T.StridedTensor(
                        [swa_pool, swa_bytes], [swa_stride, 1], T.uint8
                    ),
                    SwaIndices: T.Tensor([rows, swa_width], T.int32),
                    SwaLengths: T.Tensor([rows], T.int32),
                    IdxRecords: T.StridedTensor(
                        [idx_pool, idx_bytes], [idx_stride, 1], T.uint8
                    ),
                    IdxIndices: T.Tensor([rows, idx_width], T.int32),
                    IdxLengths: T.Tensor([rows], T.int32),
                    IdxPageTable: T.Tensor([rows, idx_pages], T.int32),
                    Sinks: T.Tensor([heads], accum_dtype),
                    Positions: T.Tensor([rows], T.int64),
                    CosSin: T.Tensor([positions_max, ROPE_DIM], accum_dtype),
                    Output: T.Tensor([rows, heads, D], dtype),
                ):
                    extend_body(
                        Q,
                        SwaRecords,
                        SwaIndices,
                        SwaLengths,
                        IdxRecords,
                        IdxIndices,
                        IdxLengths,
                        IdxPageTable,
                        Sinks,
                        Positions,
                        CosSin,
                        Output,
                    )

                return sparse_mla

            @T.prim_func
            def sparse_mla(
                Q: T.Tensor([rows, heads, D], dtype),
                SwaRecords: T.StridedTensor(
                    [swa_pool, swa_bytes], [swa_stride, 1], T.uint8
                ),
                SwaIndices: T.Tensor([rows, swa_width], T.int32),
                SwaLengths: T.Tensor([rows], T.int32),
                Sinks: T.Tensor([heads], accum_dtype),
                Positions: T.Tensor([rows], T.int64),
                CosSin: T.Tensor([positions_max, ROPE_DIM], accum_dtype),
                Output: T.Tensor([rows, heads, D], dtype),
            ):
                extend_body(
                    Q,
                    SwaRecords,
                    SwaIndices,
                    SwaLengths,
                    None,
                    None,
                    None,
                    None,
                    Sinks,
                    Positions,
                    CosSin,
                    Output,
                )

            return sparse_mla

        if idx_blocks:

            @T.prim_func
            def sparse_mla(
                Q: T.Tensor([rows, heads, D], dtype),
                SwaRecords: T.StridedTensor(
                    [swa_pool, swa_bytes], [swa_stride, 1], T.uint8
                ),
                SwaIndices: T.Tensor([rows, swa_width], T.int32),
                SwaLengths: T.Tensor([rows], T.int32),
                IdxRecords: T.StridedTensor(
                    [idx_pool, idx_bytes], [idx_stride, 1], T.uint8
                ),
                IdxIndices: T.Tensor([rows, idx_width], T.int32),
                IdxLengths: T.Tensor([rows], T.int32),
                IdxPageTable: T.Tensor([rows, idx_pages], T.int32),
                Sinks: T.Tensor([heads], accum_dtype),
                Positions: T.Tensor([rows], T.int64),
                CosSin: T.Tensor([positions_max, ROPE_DIM], accum_dtype),
                Partial: T.Tensor([rows, segments, heads, D], accum_dtype),
                PartialStats: T.Tensor([rows, segments, heads, 2], accum_dtype),
                Output: T.Tensor([rows, heads, D], dtype),
            ):
                decode_body(
                    Q,
                    SwaRecords,
                    SwaIndices,
                    SwaLengths,
                    IdxRecords,
                    IdxIndices,
                    IdxLengths,
                    IdxPageTable,
                    Sinks,
                    Positions,
                    CosSin,
                    Partial,
                    PartialStats,
                    Output,
                )

            return sparse_mla

        @T.prim_func
        def sparse_mla(
            Q: T.Tensor([rows, heads, D], dtype),
            SwaRecords: T.StridedTensor(
                [swa_pool, swa_bytes], [swa_stride, 1], T.uint8
            ),
            SwaIndices: T.Tensor([rows, swa_width], T.int32),
            SwaLengths: T.Tensor([rows], T.int32),
            Sinks: T.Tensor([heads], accum_dtype),
            Positions: T.Tensor([rows], T.int64),
            CosSin: T.Tensor([positions_max, ROPE_DIM], accum_dtype),
            Partial: T.Tensor([rows, segments, heads, D], accum_dtype),
            PartialStats: T.Tensor([rows, segments, heads, 2], accum_dtype),
            Output: T.Tensor([rows, heads, D], dtype),
        ):
            decode_body(
                Q,
                SwaRecords,
                SwaIndices,
                SwaLengths,
                None,
                None,
                None,
                None,
                Sinks,
                Positions,
                CosSin,
                Partial,
                PartialStats,
                Output,
            )

        return sparse_mla

    return candidate_sparse_mla_fwd_v41


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--boot", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(42)

    swa_width = 128
    idx_width = 512
    idx_page_size = 64
    swa_page_size = 64
    swa_pool = 32
    idx_pool = 32
    idx_pages = 32
    max_rows = 64

    SwaRecords = torch.randint(
        0, 120, (swa_pool, swa_page_size * SWA_RECORD_BYTES), dtype=torch.uint8, device=device
    )
    IdxRecords = torch.randint(
        0, 120, (idx_pool, idx_page_size * IDX_RECORD_BYTES), dtype=torch.uint8, device=device
    )
    CosSin = torch.randn((8192, ROPE_DIM), dtype=torch.float32, device=device)

    fn_s1 = make_candidate_sparse_mla(fix_s2=False)
    fn_s1_s2 = make_candidate_sparse_mla(fix_s2=True)

    results = {
        "schema_version": 1,
        "boot": args.boot,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": torch.cuda.get_device_name(0),
        "iters": args.iters,
        "all_bit_exact": True,
        "shapes": {},
    }

    for heads, tp_name in [(24, "TP3_h24"), (16, "TP4_h16")]:
        orig_mod.sparse_mla_fwd_v41 = orig_mod._orig_fn if hasattr(orig_mod, "_orig_fn") else orig_mod.sparse_mla_fwd_v41
        orig_mod._orig_fn = orig_mod.sparse_mla_fwd_v41
        plan_base = orig_mod.SparseMlaV41Plan(
            heads, swa_width, idx_width, idx_page_size, swa_page_size=swa_page_size, max_rows=max_rows
        )
        orig_mod.sparse_mla_fwd_v41 = fn_s1
        plan_s1 = orig_mod.SparseMlaV41Plan(
            heads, swa_width, idx_width, idx_page_size, swa_page_size=swa_page_size, max_rows=max_rows
        )
        orig_mod.sparse_mla_fwd_v41 = fn_s1_s2
        plan_s1_s2 = orig_mod.SparseMlaV41Plan(
            heads, swa_width, idx_width, idx_page_size, swa_page_size=swa_page_size, max_rows=max_rows
        )
        orig_mod.sparse_mla_fwd_v41 = orig_mod._orig_fn

        Sinks = torch.randn((heads,), dtype=torch.float32, device=device) * 0.1
        results["shapes"][tp_name] = {}

        for ctx_name, swa_len_val, idx_len_val in [
            ("med_ctx_128_512", 128, 512),
            ("short_ctx_128_64", 128, 64),
        ]:
            results["shapes"][tp_name][ctx_name] = {}
            print(f"\n=== {tp_name} | {ctx_name} (swa={swa_len_val}, idx={idx_len_val}) ===")
            print(f"{'M':>3} | {'Route':>6} | {'1107(us)':>9} | {'Cand_S1':>9} | {'d_S1':>15} | {'Cand_S1+S2':>10} | {'d_S1+S2':>15} | {'Exact':>5}")
            print("-" * 92)

            for M in [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]:
                Q = torch.randn((M, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
                SwaIndices = torch.randint(0, swa_pool, (M, swa_width), dtype=torch.int32, device=device)
                SwaLengths = torch.full((M,), swa_len_val, dtype=torch.int32, device=device)
                IdxIndices = torch.randint(
                    0, idx_pages * idx_page_size, (M, idx_width), dtype=torch.int32, device=device
                )
                IdxLengths = torch.full((M,), idx_len_val, dtype=torch.int32, device=device)
                IdxPageTable = torch.randint(0, idx_pool, (M, idx_pages), dtype=torch.int32, device=device)
                Positions = torch.arange(100, 100 + M, dtype=torch.int64, device=device)

                route = plan_base.kernel_for(M)
                out_base = torch.empty((M, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
                out_s1 = torch.empty_like(out_base)
                out_s12 = torch.empty_like(out_base)

                scratch_base = [
                    torch.empty(s, dtype=dt, device=device) for s, dt in plan_base.scratch_specs(M, route)
                ]
                scratch_s1 = [
                    torch.empty(s, dtype=dt, device=device) for s, dt in plan_s1.scratch_specs(M, route)
                ]
                scratch_s12 = [
                    torch.empty(s, dtype=dt, device=device) for s, dt in plan_s1_s2.scratch_specs(M, route)
                ]

                kw = dict(
                    idx_cache=IdxRecords,
                    idx_indices=IdxIndices,
                    idx_lengths=IdxLengths,
                    idx_page_table=IdxPageTable,
                    kernel=route,
                )
                plan_base.run(Q, SwaRecords, SwaIndices, SwaLengths, Sinks, Positions, CosSin, out_base, scratch=scratch_base, **kw)
                plan_s1.run(Q, SwaRecords, SwaIndices, SwaLengths, Sinks, Positions, CosSin, out_s1, scratch=scratch_s1, **kw)
                plan_s1_s2.run(Q, SwaRecords, SwaIndices, SwaLengths, Sinks, Positions, CosSin, out_s12, scratch=scratch_s12, **kw)
                torch.cuda.synchronize()

                exact_s1 = bool(torch.equal(out_base, out_s1))
                exact_s12 = bool(torch.equal(out_base, out_s12))
                all_exact = exact_s1 and exact_s12
                if not all_exact:
                    results["all_bit_exact"] = False

                def measure(plan, out, sc):
                    for _ in range(20):
                        plan.run(Q, SwaRecords, SwaIndices, SwaLengths, Sinks, Positions, CosSin, out, scratch=sc, **kw)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for _ in range(10):
                            plan.run(Q, SwaRecords, SwaIndices, SwaLengths, Sinks, Positions, CosSin, out, scratch=sc, **kw)
                    torch.cuda.synchronize()
                    for _ in range(10):
                        g.replay()
                    torch.cuda.synchronize()
                    windows = 25
                    replays_per_win = 10
                    starts = [torch.cuda.Event(enable_timing=True) for _ in range(windows)]
                    ends = [torch.cuda.Event(enable_timing=True) for _ in range(windows)]
                    for w in range(windows):
                        starts[w].record()
                        for _ in range(replays_per_win):
                            g.replay()
                        ends[w].record()
                    torch.cuda.synchronize()
                    samples_us = [
                        s.elapsed_time(e) * 1000.0 / (replays_per_win * 10)
                        for s, e in zip(starts, ends)
                    ]
                    return {
                        "mean_us": round(statistics.mean(samples_us), 3),
                        "median_us": round(statistics.median(samples_us), 3),
                        "stdev_us": round(statistics.stdev(samples_us), 3),
                    }

                b_stat = measure(plan_base, out_base, scratch_base)
                s1_stat = measure(plan_s1, out_s1, scratch_s1)
                s12_stat = measure(plan_s1_s2, out_s12, scratch_s12)

                tb = b_stat["mean_us"]
                ts1 = s1_stat["mean_us"]
                ts12 = s12_stat["mean_us"]
                d1 = ts1 - tb
                p1 = (d1 / tb) * 100.0
                d12 = ts12 - tb
                p12 = (d12 / tb) * 100.0

                results["shapes"][tp_name][ctx_name][str(M)] = {
                    "route": route,
                    "base_1107": b_stat,
                    "cand_s1": s1_stat,
                    "cand_s1_s2": s12_stat,
                    "delta_s1_us": round(d1, 3),
                    "delta_s1_pct": round(p1, 2),
                    "delta_s1_s2_us": round(d12, 3),
                    "delta_s1_s2_pct": round(p12, 2),
                    "bit_exact_s1": exact_s1,
                    "bit_exact_s1_s2": exact_s12,
                }
                print(
                    f"{M:>3} | {route:>6} | {tb:>9.2f} | {ts1:>9.2f} | {d1:>+6.2f} ({p1:>+5.1f}%) | "
                    f"{ts12:>10.2f} | {d12:>+6.2f} ({p12:>+5.1f}%) | {str(all_exact):>5}"
                )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\nSaved receipt to {args.output} (all_bit_exact={results['all_bit_exact']})")


if __name__ == "__main__":
    main()
