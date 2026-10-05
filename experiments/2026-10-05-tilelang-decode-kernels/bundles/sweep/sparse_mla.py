# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash compressed sparse MLA forward for SM120 / SM121.

Every V4.1 attention row reads two streams of 512-wide MQA latents. Each
latent serves as both K and V:

- the sliding-window (SWA) cache: 528-byte records of 512 E4M3 values and
  16 UE8M0 scales, one per 32 coordinates;
- the indexed cache, holding top-k selected compressed entries: 288-byte
  records of 256 bytes of packed E2M1 (low nibble first) and 32 E4M3 scales,
  one per 16 coordinates.

Both caches are vLLM page buffers: each page holds ``page_size`` contiguous
records, followed by any padding the allocator adds to equalize page sizes.
SWA entries are physical slots (``page * page_size + offset``). Indexed
entries are logical positions, resolved through a per-row page table as in
vLLM. All 512 coordinates are quantized, RoPE included.

Each CTA resolves the slots it needs up front, copies the next chunk's raw
records into shared memory with ``cp.async`` while the current chunk
computes, dequantizes them exactly to BF16 in shared memory, and multiplies
on the warp-level MMA path. Masked entries are zero-filled.

The arithmetic is defined over chunks of ``block_I`` entries of the
concatenated [SWA | indexed] list, grouped into segments of
``segment_chunks`` chunks:

- within a segment, chunks update a state ``(O_g, m_g, l_g)`` in order, as
  in flash attention: ``m_g' = max(m_g, max(s))`` with scores
  ``s = q.k * sm_scale * log2(e)``, ``p = exp2(s - m_g')``,
  ``l_g = fma(l_g, a, sum(p))`` and ``O_g = O_g * a + P V`` with
  ``a = exp2(m_g - m_g')``;
- the segment states then fold in order into a state that starts from the
  head's attention sink, with ``M' = max(M, m_g)``,
  ``L = fma(L, exp2(M - M'), l_g * exp2(m_g - M'))``, the same update for
  ``O``, and finally ``out = O * (1 / L)``.

Two schedules implement this contract:

- ``extend``: one CTA per row and head block computes and folds every
  segment in registers;
- ``decode``: one CTA per segment, head block and row writes its segment
  state to a caller-owned workspace, and a merge kernel folds them. This
  gives small batches up to ``segments x head_blocks`` CTAs per row instead
  of one.

Both schedules use the same segment code, and explicit ``T.fma``/``T.fmul``
keep the fold's rounding fixed, so they produce identical bits, for any head
block size. A row's output depends neither on its neighbours nor on the
kernel the batch size selects. ``segment_chunks`` and ``block_I`` are part of
the arithmetic; ``schedule`` and ``block_H`` are not. ``SparseMlaV41Plan``
compiles the kernels for one layer shape, owns the split workspace, and
picks a kernel from the row count.

SM120/SM121 have no WGMMA and allow at most 99 KiB of shared memory per
block, so the tiles are ``mma.sync`` tiles of 16..64 heads by 16..32
entries, instead of the 64x64 SM90 tiles in
``examples/deepseek_v4/sparse_attn_fwd_sm90.py``.
"""

import tilelang
import tilelang.language as T
import torch

HEAD_DIM = 512
SWA_RECORD_BYTES = 528  # 512 x E4M3 + 16 x UE8M0
SWA_GROUP = 32
IDX_RECORD_BYTES = 288  # 256 x packed E2M1 + 32 x E4M3
IDX_GROUP = 16
LOG2E = 1.44269504
# Segments of four 32-entry chunks: the 128-entry sliding window is one segment.
SEGMENT_CHUNKS = 4
# The plan runs decode while it needs at most this many waves of CTAs.
DECODE_MAX_WAVES = 3


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def sparse_mla_fwd_v41(
    heads: int,
    swa_width: int,
    idx_width: int,
    idx_page_size: int,
    schedule: str = "extend",
    swa_page_size: int = 1,
    sm_scale: float = HEAD_DIM**-0.5,
    block_H: int = 32,
    block_I: int = 32,
    threads: int = 256,
    segment_chunks: int = SEGMENT_CHUNKS,
    stage_buffers: int = 2,
    dequant_vec: int = 8,
):
    """Build one schedule of the sparse MLA kernel.

    Both schedules take:

    - ``Q``: ``[rows, heads, 512]`` BF16 queries, with RoPE applied;
    - ``SwaRecords``: ``[pages, swa_page_size * 528]`` uint8 SWA cache pages
      (``[slots, 528]`` when ``swa_page_size`` is 1), any row stride;
    - ``SwaIndices``: ``[rows, swa_width]`` int32 SWA slots, ``-1`` for padding;
    - ``SwaLengths``: ``[rows]`` int32 count of leading SWA entries to use;
    - ``IdxRecords``: ``[pages, idx_page_size * 288]`` uint8 indexed cache
      pages, any row stride (this and the next three are absent when
      ``idx_width == 0``);
    - ``IdxIndices``: ``[rows, idx_width]`` int32 logical indexed positions,
      ``-1`` for padding;
    - ``IdxLengths``: ``[rows]`` int32 count of leading indexed entries to use;
    - ``IdxPageTable``: ``[rows, pages]`` int32 physical page of each logical
      ``idx_page_size`` page, as in vLLM's indexed page table;
    - ``Sinks``: ``[heads]`` FP32 attention-sink logits.

    ``decode`` then takes the FP32 workspaces ``Partial``
    (``[rows, segments, heads, 512]``) and ``PartialStats``
    (``[rows, segments, heads, 2]``). Both write ``Output``
    (``[rows, heads, 512]`` BF16). ``rows``, the cache page counts and
    strides, and the page-table width are dynamic.
    """
    # Cache pages are rows of a vLLM page buffer, which may interleave other
    # layers' pages (BLHNC) or pad: the row stride is a runtime value.
    rows, swa_pool, swa_stride, idx_pool, idx_stride, idx_pages = T.dynamic(
        "rows, swa_pool, swa_stride, idx_pool, idx_stride, idx_pages"
    )
    swa_bytes = swa_page_size * SWA_RECORD_BYTES
    idx_bytes = idx_page_size * IDX_RECORD_BYTES

    D = HEAD_DIM
    BI = block_I
    dtype = T.bfloat16
    accum_dtype = T.float32
    scale_log2 = sm_scale * LOG2E

    assert schedule in ("extend", "decode"), schedule
    assert block_H % 16 == 0, "block_H must be a multiple of the 16-row MMA atom"
    assert BI % 16 == 0, "block_I must be a multiple of 16"
    head_blocks = tilelang.cdiv(heads, block_H)
    pad_heads = heads % block_H != 0
    swa_blocks = tilelang.cdiv(swa_width, BI)
    idx_blocks = tilelang.cdiv(idx_width, BI)
    chunks = swa_blocks + idx_blocks
    G = segment_chunks
    segments = tilelang.cdiv(chunks, G)
    # extend resolves every slot of its row up front; decode only its segment's.
    slot_count = chunks * BI if schedule == "extend" else G * BI
    copies = tilelang.cdiv(BI * SWA_RECORD_BYTES // 16, threads)
    # Q, the dequantized KV block and P in BF16, the raw record staging rows,
    # the slots, and reduction scratch.
    # decode stages the next chunk's records while the current one computes,
    # where a second staging buffer fits in shared memory.
    assert stage_buffers in (1, 2), stage_buffers
    buffers = stage_buffers if schedule == "decode" else 1
    if buffers == 2 and (
        2 * (block_H * D + BI * D + block_H * BI)
        + 2 * BI * SWA_RECORD_BYTES
        + 4 * G * BI
        + 2048
        > 99 * 1024
    ):
        buffers = 1
    assert D % dequant_vec == 0 and SWA_GROUP % dequant_vec == 0, dequant_vec
    smem_bytes = (
        2 * (block_H * D + BI * D + block_H * BI)
        + buffers * BI * SWA_RECORD_BYTES
        + 4 * slot_count
        + 2048
    )
    assert smem_bytes <= 99 * 1024, (
        f"block_H={block_H}, block_I={BI} needs ~{smem_bytes} B of shared memory; "
        "SM120 allows 99 KiB (use block_I=16 with block_H=64)"
    )

    @T.macro
    def gather_swa(SwaIndices, SwaLengths, row, first, count, slots, base):
        for i in T.Parallel(count):
            pos = first + i
            slot = SwaIndices[row, T.min(pos, swa_width - 1)]
            slots[base + i] = T.if_then_else(
                (pos < swa_width) & (pos < SwaLengths[row]) & (slot >= 0), slot, -1
            )

    @T.macro
    def gather_idx(
        IdxIndices, IdxLengths, IdxPageTable, row, first, count, slots, base
    ):
        # Map logical positions to physical slots through the row's page table.
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
    def stage_records(
        Records, record_bytes, page_size, slots, base, staged, copy_slot, row0=0
    ):
        # cp.async each record into its staging row, 16 bytes per copy; masked
        # entries are zero-filled, so they dequantize to zero. Reading the slots
        # first keeps shared-memory reads out of the issue loop.
        segments = record_bytes // 16
        tx = T.get_thread_binding()
        for k in T.unroll(tilelang.cdiv(BI * segments, threads)):
            copy_slot[k] = slots[base + T.min((k * threads + tx) // segments, BI - 1)]
        for k in T.unroll(tilelang.cdiv(BI * segments, threads)):
            e = k * threads + tx
            if e < BI * segments:
                T.ptx_cp_async(
                    T.access_ptr(
                        staged[row0 + e // segments, e % segments * 16], "w", 16
                    ),
                    T.access_ptr(
                        Records[
                            T.max(copy_slot[k], 0) // page_size,
                            T.max(copy_slot[k], 0) % page_size * record_bytes
                            + e % segments * 16,
                        ],
                        "r",
                        16,
                    ),
                    16,
                    copy_slot[k] >= 0,
                )
        T.ptx_commit_group()

    @T.macro
    def stage_chunk(
        SwaRecords, IdxRecords, chunk, slots, base, staged, copy_slot, row0=0
    ):
        # SWA-only layers compile without the indexed branch.
        is_swa = True if not idx_blocks else chunk < swa_blocks
        if is_swa:
            stage_records(
                SwaRecords,
                SWA_RECORD_BYTES,
                swa_page_size,
                slots,
                base,
                staged,
                copy_slot,
                row0,
            )
        else:
            stage_records(
                IdxRecords,
                IDX_RECORD_BYTES,
                idx_page_size,
                slots,
                base,
                staged,
                copy_slot,
                row0,
            )

    VEC = dequant_vec

    @T.macro
    def dequant_swa(staged, staged_fp8, KV_shared, row0=0):
        # Each thread converts VEC contiguous values under one scale; the
        # arithmetic per value is unchanged.
        for i, dv in T.Parallel(BI, D // VEC):
            # UE8M0 scale 2^(e - 127), built from the exponent bits.
            scale = T.reinterpret(
                T.shift_left(
                    T.cast(staged[row0 + i, D + dv * VEC // SWA_GROUP], T.uint32), 23
                ),
                accum_dtype,
            )
            for v in T.vectorized(VEC):
                KV_shared[i, dv * VEC + v] = T.cast(
                    T.cast(staged_fp8[row0 + i, dv * VEC + v], accum_dtype) * scale,
                    dtype,
                )

    @T.macro
    def dequant_idx(staged_fp8, staged_fp4, KV_shared, row0=0):
        # TileLang's SM120 code generation has no packed E2M1 vector load, so
        # the indexed records convert value by value.
        for i, d in T.Parallel(BI, D):
            scale = T.cast(staged_fp8[row0 + i, D // 2 + d // IDX_GROUP], accum_dtype)
            KV_shared[i, d] = T.cast(
                T.cast(staged_fp4[row0 + i, d], accum_dtype) * scale, dtype
            )

    @T.macro
    def dequant_chunk(chunk, staged, staged_fp8, staged_fp4, KV_shared, row0=0):
        is_swa = True if not idx_blocks else chunk < swa_blocks
        if is_swa:
            dequant_swa(staged, staged_fp8, KV_shared, row0)
        else:
            dequant_idx(staged_fp8, staged_fp4, KV_shared, row0)

    @T.macro
    def load_q(Q, row, h0, Q_shared):
        if pad_heads:
            for h, d in T.Parallel(block_H, D):
                Q_shared[h, d] = T.if_then_else(
                    h0 + h < heads, Q[row, T.min(h0 + h, heads - 1), d], 0
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
        """Add one chunk to the segment state (acc_g, m_g, l_g) of the block's heads."""
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
            # The state is empty until the first unmasked entry.
            alpha[h] = T.if_then_else(
                m_prev[h] == -T.infinity(accum_dtype), 0, T.exp2(m_prev[h] - m_g[h])
            )
        for h, i in T.Parallel(block_H, BI):
            acc_s[h, i] = T.if_then_else(
                slots[base + i] >= 0, T.exp2(acc_s[h, i] - m_g[h]), 0
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
        return T.fmul(T.if_then_else(h < heads, Sinks[T.min(h, heads - 1)], 0), LOG2E)

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
        Output,
    ):
        with T.Kernel(head_blocks, rows, threads=threads) as (hb, row):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            P_shared = T.alloc_shared([block_H, BI], dtype)
            # Raw records, one 528-byte row per entry; indexed records use 288 bytes.
            staged = T.alloc_shared([BI, SWA_RECORD_BYTES], T.uint8)
            staged_fp8 = T.view(staged, dtype=T.float8_e4m3fn)
            staged_fp4 = T.view(staged, [BI, SWA_RECORD_BYTES * 2], T.float4_e2m1fn)
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

            # Resolve the row's slots once, so record copies never wait on index loads.
            gather_swa(SwaIndices, SwaLengths, row, 0, swa_blocks * BI, slots, 0)
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
            stage_chunk(SwaRecords, IdxRecords, 0, slots, 0, staged, copy_slot)
            h0 = hb * block_H
            load_q(Q, row, h0, Q_shared)
            T.fill(acc_o, 0)
            T.fill(L, 1)
            for h in T.Parallel(block_H):
                M[h] = sink_log2(Sinks, h0 + h)

            for seg in T.serial(segments):
                clear_segment(acc_g, m_g, l_g)
                for c in T.serial(G):
                    chunk = seg * G + c
                    if chunk < chunks:
                        T.ptx_wait_group(0)
                        T.sync_threads()
                        dequant_chunk(chunk, staged, staged_fp8, staged_fp4, KV_shared)
                        T.sync_threads()
                        # The next chunk's records load while this chunk computes.
                        if chunk + 1 < chunks:
                            stage_chunk(
                                SwaRecords,
                                IdxRecords,
                                chunk + 1,
                                slots,
                                (chunk + 1) * BI,
                                staged,
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
                # Fold the segment into the running state, as the decode merge does.
                for h in T.Parallel(block_H):
                    m_new = T.max(M[h], m_g[h])
                    a[h] = T.exp2(M[h] - m_new)
                    b[h] = T.exp2(m_g[h] - m_new)
                    L[h] = T.fma(L[h], a[h], T.fmul(l_g[h], b[h]))
                    M[h] = m_new
                for h, d in T.Parallel(block_H, D):
                    acc_o[h, d] = T.fma(acc_o[h, d], a[h], T.fmul(acc_g[h, d], b[h]))

            for h in T.Parallel(block_H):
                L[h] = 1.0 / L[h]
            for h, d in T.Parallel(block_H, D):
                if h0 + h < heads:
                    Output[row, h0 + h, d] = T.fmul(acc_o[h, d], L[h])

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
        Partial,
        PartialStats,
        Output,
    ):
        with T.Kernel(segments, head_blocks, rows, threads=threads) as (seg, hb, row):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            P_shared = T.alloc_shared([block_H, BI], dtype)
            staged_all = T.alloc_shared([buffers * BI, SWA_RECORD_BYTES], T.uint8)
            slots = T.alloc_shared([slot_count], T.int32)
            copy_slot = T.alloc_local([copies], T.int32)
            acc_s = T.alloc_fragment([block_H, BI], accum_dtype)
            acc_g = T.alloc_fragment([block_H, D], accum_dtype)
            m_g = T.alloc_fragment([block_H], accum_dtype)
            l_g = T.alloc_fragment([block_H], accum_dtype)
            m_prev = T.alloc_fragment([block_H], accum_dtype)
            alpha = T.alloc_fragment([block_H], accum_dtype)
            row_sum = T.alloc_fragment([block_H], accum_dtype)

            for c in T.serial(G):
                chunk = seg * G + c
                if not idx_blocks:
                    if chunk < swa_blocks:
                        gather_swa(
                            SwaIndices, SwaLengths, row, chunk * BI, BI, slots, c * BI
                        )
                elif chunk < swa_blocks:
                    gather_swa(
                        SwaIndices, SwaLengths, row, chunk * BI, BI, slots, c * BI
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
            staged_fp8 = T.view(staged_all, dtype=T.float8_e4m3fn)
            staged_fp4 = T.view(
                staged_all, [buffers * BI, SWA_RECORD_BYTES * 2], T.float4_e2m1fn
            )
            stage_chunk(SwaRecords, IdxRecords, seg * G, slots, 0, staged_all, copy_slot)
            h0 = hb * block_H
            load_q(Q, row, h0, Q_shared)
            clear_segment(acc_g, m_g, l_g)
            for c in T.serial(G):
                chunk = seg * G + c
                if chunk < chunks:
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    if buffers == 2:
                        # The next chunk's records load into the other buffer
                        # while this one dequantizes and multiplies.
                        if (c + 1 < G) & (chunk + 1 < chunks):
                            stage_chunk(
                                SwaRecords,
                                IdxRecords,
                                chunk + 1,
                                slots,
                                (c + 1) * BI,
                                staged_all,
                                copy_slot,
                                (c + 1) % 2 * BI,
                            )
                        dequant_chunk(
                            chunk,
                            staged_all,
                            staged_fp8,
                            staged_fp4,
                            KV_shared,
                            c % 2 * BI,
                        )
                        T.sync_threads()
                    else:
                        dequant_chunk(
                            chunk, staged_all, staged_fp8, staged_fp4, KV_shared
                        )
                        T.sync_threads()
                        if (c + 1 < G) & (chunk + 1 < chunks):
                            stage_chunk(
                                SwaRecords,
                                IdxRecords,
                                chunk + 1,
                                slots,
                                (c + 1) * BI,
                                staged_all,
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

        # Fold the segment partials in order, starting from the sink.
        with T.Kernel(heads, rows, threads=128) as (hc, rc):
            o = T.alloc_fragment([D], accum_dtype)
            M = T.alloc_var(accum_dtype)
            L = T.alloc_var(accum_dtype)
            a = T.alloc_var(accum_dtype)
            b = T.alloc_var(accum_dtype)
            m_new = T.alloc_var(accum_dtype)

            M = sink_log2(Sinks, hc)
            L = 1.0
            T.clear(o)
            for c in T.unroll(segments):
                m_new = T.max(M, PartialStats[rc, c, hc, 0])
                a = T.exp2(M - m_new)
                b = T.exp2(PartialStats[rc, c, hc, 0] - m_new)
                L = T.fma(L, a, T.fmul(PartialStats[rc, c, hc, 1], b))
                M = m_new
                for d in T.Parallel(D):
                    o[d] = T.fma(o[d], a, T.fmul(Partial[rc, c, hc, d], b))
            L = 1.0 / L
            for d in T.Parallel(D):
                Output[rc, hc, d] = T.fmul(o[d], L)

    # SWA-only layers (idx_width == 0) compile without the indexed tensors.
    if schedule == "extend":
        if idx_blocks:

            @T.prim_func
            def main(
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
                    Output,
                )

            return main

        @T.prim_func
        def main(
            Q: T.Tensor([rows, heads, D], dtype),
            SwaRecords: T.StridedTensor(
                [swa_pool, swa_bytes], [swa_stride, 1], T.uint8
            ),
            SwaIndices: T.Tensor([rows, swa_width], T.int32),
            SwaLengths: T.Tensor([rows], T.int32),
            Sinks: T.Tensor([heads], accum_dtype),
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
                Output,
            )

        return main

    if idx_blocks:

        @T.prim_func
        def main(
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
                Partial,
                PartialStats,
                Output,
            )

        return main

    @T.prim_func
    def main(
        Q: T.Tensor([rows, heads, D], dtype),
        SwaRecords: T.StridedTensor([swa_pool, swa_bytes], [swa_stride, 1], T.uint8),
        SwaIndices: T.Tensor([rows, swa_width], T.int32),
        SwaLengths: T.Tensor([rows], T.int32),
        Sinks: T.Tensor([heads], accum_dtype),
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
            Partial,
            PartialStats,
            Output,
        )

    return main


class SparseMlaV41Plan:
    """The kernels for one layer shape.

    All kernels implement the same arithmetic, fixed by ``segment_chunks`` and
    ``block_I``, so choosing one by row count never changes results:

    - ``decode`` with ``split_block_H``-head blocks while it fits one wave of
      CTAs (``split_max_rows``);
    - ``decode`` with ``block_H``-head blocks while it needs at most
      ``DECODE_MAX_WAVES`` waves (``decode_max_rows``);
    - ``extend`` with ``block_H``-head blocks above that.

    The plan holds no buffers: ``scratch_specs(rows)`` lists the segment
    workspace the decode kernels need, which a caller can borrow from a shared
    workspace. ``run`` only launches kernels and can be captured in a CUDA
    graph.
    """

    def __init__(
        self,
        heads,
        swa_width,
        idx_width,
        idx_page_size=64,
        *,
        swa_page_size=1,
        max_rows,
        segment_chunks=SEGMENT_CHUNKS,
        decode_max_rows=None,
        split_max_rows=None,
        sm_scale=HEAD_DIM**-0.5,
        block_H=32,
        split_block_H=16,
        block_I=32,
        threads=256,
        stage_buffers=2,
        dequant_vec=8,
        device="cuda",
    ):
        shape = (heads, swa_width, idx_width, idx_page_size)
        config = dict(
            sm_scale=sm_scale,
            block_I=block_I,
            segment_chunks=segment_chunks,
            swa_page_size=swa_page_size,
            stage_buffers=stage_buffers,
            dequant_vec=dequant_vec,
        )
        chunks = tilelang.cdiv(swa_width, block_I) + tilelang.cdiv(idx_width, block_I)
        segments = tilelang.cdiv(chunks, segment_chunks)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        if decode_max_rows is None:
            decode_max_rows = (
                DECODE_MAX_WAVES * sms // (segments * tilelang.cdiv(heads, block_H))
            )
        if split_max_rows is None:
            split_max_rows = (
                sms // (segments * tilelang.cdiv(heads, split_block_H))
                if split_block_H
                else 0
            )
        self.max_rows = max_rows
        self.idx_width = idx_width
        self.swa_bytes = swa_page_size * SWA_RECORD_BYTES
        self.idx_bytes = idx_page_size * IDX_RECORD_BYTES
        self.decode_max_rows = min(decode_max_rows, max_rows)
        self.split_max_rows = min(split_max_rows, self.decode_max_rows)
        self.kernels = {
            "extend": sparse_mla_fwd_v41(
                *shape, "extend", block_H=block_H, threads=threads, **config
            )
        }
        if self.decode_max_rows:
            self.kernels["decode"] = sparse_mla_fwd_v41(
                *shape, "decode", block_H=block_H, threads=threads, **config
            )
        if self.split_max_rows:
            # One warp per 16x8 tile of the 16-head x block_I score block, at most four.
            split_threads = 32 * min(4, (split_block_H // 16) * (block_I // 8))
            self.kernels["split"] = sparse_mla_fwd_v41(
                *shape, "decode", block_H=split_block_H, threads=split_threads, **config
            )
        self.segments, self.heads = segments, heads

    def scratch_specs(self, rows, kernel=None):
        """``Partial`` and ``PartialStats`` for a decode launch; nothing for extend."""
        if (kernel or self.kernel_for(rows)) == "extend":
            return ()
        return (
            ((rows, self.segments, self.heads, HEAD_DIM), torch.float32),
            ((rows, self.segments, self.heads, 2), torch.float32),
        )

    def kernel_for(self, rows):
        if rows <= self.split_max_rows:
            return "split"
        return "decode" if rows <= self.decode_max_rows else "extend"

    def run(
        self,
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        sinks,
        out,
        *,
        idx_cache=None,
        idx_indices=None,
        idx_lengths=None,
        idx_page_table=None,
        kernel=None,
        scratch=None,
    ):
        """Attend over uint8 ``[pages, page_size * record_bytes]`` cache pages.

        Writes ``out``. The page buffers may have any row stride.

        The indexed tensors are required when ``idx_width > 0`` and absent
        otherwise. ``kernel`` forces ``"split"``, ``"decode"`` or ``"extend"``.
        """
        rows = q.shape[0]
        assert rows <= self.max_rows, f"{rows} rows exceed the plan ({self.max_rows})"
        kernel = kernel or self.kernel_for(rows)
        indexed = (idx_cache, idx_indices, idx_lengths, idx_page_table)
        assert all(t is not None for t in indexed) == bool(self.idx_width), (
            "indexed tensors must match idx_width"
        )
        if self.idx_width:
            # Views of each page's records; rows keep the buffer's stride.
            indexed = (idx_cache[:, : self.idx_bytes], *indexed[1:])
        args = (
            q,
            swa_cache[:, : self.swa_bytes],
            swa_indices,
            swa_lengths,
            *(indexed if self.idx_width else ()),
            sinks,
        )
        if kernel == "extend":
            self.kernels["extend"](*args, out)
        else:
            assert rows <= self.decode_max_rows, (
                f"{rows} rows exceed the decode route ({self.decode_max_rows})"
            )
            if scratch is None:
                scratch = [
                    torch.empty(shape, dtype=dtype, device=q.device)
                    for shape, dtype in self.scratch_specs(rows, kernel)
                ]
            self.kernels[kernel](*args, *scratch, out)
        return out
