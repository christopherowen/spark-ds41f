# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash sparse-attention indexer for SM120 / SM121: scores and top-k.

The V4.1 indexer has 32 heads of width 128. It keeps its keys in a paged
MXFP4 cache: each page holds ``page_size`` rows of 64 bytes of packed E2M1
(low nibble first), followed by ``page_size`` rows of 4 UE8M0 scales (one per
32 dims). Queries are quantized to MXFP4 the same way, and per-head weights
arrive as BF16, already multiplied by ``1/64`` and rounded.

The published score keeps distinct BF16 rounding stages. For each position
``j`` and head ``h``:

    dot   = bf16(q_h . k_j)                     # FP32 accumulation
    c_h   = bf16(max(dot, 0) * w_h)
    score = bf16(c_0 + c_1 + ... + c_31)         # FP32, increasing head order

``indexer_score_v41`` computes ``q . k`` straight from the packed operands with
SM120's MXFP4 block-scaled MMA
(``mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::2X.ue8m0``), then
applies the rounding stages and the ordered head sum in the epilogue. It
scores either every position of a row's context, or only the positions in a
candidate list (V4.1's reindex layers).

``IndexerTopK`` selects the top ``k`` entries of each row exactly. Each
4096-entry segment caches order-preserving 16-bit keys of its BF16 scores in
shared memory, runs a two-pass 8-bit radix select and an ordered compaction
(ties at the threshold keep the lowest positions), and keeps its own top
``k``. Long rows then merge the segment winners with the same selection, so
small batches still spread over many CTAs. Positions come back in ascending
order, padded with ``-1``.
``group=8`` ranks blocks of eight positions by their maximum, always keeping
the block that holds the last visible position, and expands them: V4.1's
candidate-domain selection at its candidate source layer.

Selection is deterministic: the same scores give the same positions
regardless of thread timing or batch composition.
"""

import tilelang
import tilelang.language as T
import torch

from .gemm import SF_BLOCK

INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_TOPK = 512
CANDIDATE_BLOCKS = 2048
CANDIDATE_BLOCK = 8
# Order-preserving uint16 key of BF16 -inf: entries at or below it are never selected.
_KEY_NEG_INF = 0x007F


def page_bytes(page_size: int) -> int:
    """Index-cache page bytes: ``[page_size, 64]`` data, ``[page_size, 4]`` UE8M0."""
    return page_size * (INDEX_DIM // 2 + INDEX_DIM // SF_BLOCK)


def _row_extent(Lengths, Active, row, width):
    """Positions a row selects from: ``min(Lengths, Active, width)``, at least zero.

    Candidate rows have no ``Active`` cap: their length counts candidates.
    """
    extent = T.max(Lengths[row], 0)
    if Active is not None:
        extent = T.min(extent, T.max(Active[0], 0))
    return T.min(extent, width)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False})
def indexer_score_v41(
    width: int,
    page_size: int,
    candidates: bool = False,
    heads: int = INDEX_HEADS,
    block_P: int = 64,
    threads: int = 128,
    max_ctas: int = 128,
):
    """Build the score kernel. Inputs:

    - ``Q``: ``[rows, heads, 128]`` E2M1 queries (``[rows, heads, 64]`` int8 storage);
    - ``QScale``: ``[rows, heads]`` uint32, the four UE8M0 scales of each head;
    - ``Weights``: ``[rows, heads]`` BF16 head weights;
    - ``KPool``: E2M1 view (``[pages, page_bytes * 2]`` logical) of the index cache;
    - ``KScale``: uint32 view (``[pages, page_bytes / 4]``) of the same cache.
      Both may keep the allocator's page stride when it pads pages;
    - ``PageTable``: ``[rows, table_width]`` int32 physical page per logical page;
    - ``Lengths``: ``[rows]`` int32 positions per row;
    - ``Active``: ``[1]`` int32 width cap;
    - candidate mode: ``Candidates`` ``[rows, width]`` int32 positions and
      ``CandidateLengths`` ``[rows]``.

    Writes ``Scores`` (``[rows, width]`` BF16) up to each row's extent, ``-inf``
    for masked candidates; entries past the extent are left unwritten and the
    top-k never reads them. Each row gets at most ``max_ctas`` CTAs, which loop
    over its live tiles, so the launch scales with the context in use rather
    than the planned width.
    """
    assert page_size % block_P == 0 or candidates, (
        "dense tiles must not straddle a page"
    )
    rows, pool_pages, table_width, pool_stride, scale_stride = T.dynamic(
        "rows, pool_pages, table_width, pool_stride, scale_stride"
    )
    D = INDEX_DIM
    fp4 = T.float4_e2m1fn
    accum_dtype = T.float32
    page_words = page_bytes(page_size) // 4
    scale_word0 = page_size * D // 2 // 4  # first scale word of a page
    ctas = min(tilelang.cdiv(width, block_P), max_ctas)

    @T.macro
    def score_row(
        Q,
        QScale,
        Weights,
        KPool,
        KScale,
        PageTable,
        row,
        cta,
        extent,
        position,
        slot_valid,
        Scores,
    ):
        K_shared = T.alloc_shared([block_P, D], fp4)
        Q_shared = T.alloc_shared([heads, D], fp4)
        SFK_shared = T.alloc_shared([block_P, 1], T.uint32)
        SFQ_shared = T.alloc_shared([heads, 1], T.uint32)
        W_shared = T.alloc_shared([heads], accum_dtype)
        contrib = T.alloc_shared([block_P, heads], accum_dtype)
        acc = T.alloc_fragment([block_P, heads], accum_dtype)
        total = T.alloc_fragment([block_P], accum_dtype)

        for h, d in T.Parallel(heads, D):
            Q_shared[h, d] = Q[row, h, d]
        for h in T.Parallel(heads):
            SFQ_shared[h, 0] = QScale[row, h]
            W_shared[h] = T.cast(Weights[row, h], accum_dtype)
        for i in T.serial(T.ceildiv(T.max(T.ceildiv(extent, block_P) - cta, 0), ctas)):
            p0 = (cta + i * ctas) * block_P
            # Invalid slots load a clamped, in-pool key and are masked at the end.
            for p, d in T.Parallel(block_P, D):
                pos = position(p0, p)
                page = T.min(
                    T.max(PageTable[row, T.min(pos // page_size, table_width - 1)], 0),
                    pool_pages - 1,
                )
                K_shared[p, d] = KPool[page, (pos % page_size) * D + d]
            for p in T.Parallel(block_P):
                pos = position(p0, p)
                page = T.min(
                    T.max(PageTable[row, T.min(pos // page_size, table_width - 1)], 0),
                    pool_pages - 1,
                )
                SFK_shared[p, 0] = KScale[page, scale_word0 + pos % page_size]
            T.mma_gemm_blockscaled(
                K_shared,
                Q_shared,
                acc,
                SFK_shared,
                SFQ_shared,
                transpose_B=True,
                clear_accum=True,
                k_start=0,
                sf_a_granularity_k=SF_BLOCK,
                sf_b_granularity_k=SF_BLOCK,
            )
            # The published rounding stages: BF16 dot, ReLU, BF16 weighted product.
            # The ReLU keeps a NaN, as the reference's relu_ does; max alone would
            # drop it. Every other value, signed zeros included, takes max's bits.
            for p, h in T.Parallel(block_P, heads):
                dot = T.cast(T.cast(acc[p, h], T.bfloat16), accum_dtype)
                dot = T.if_then_else(dot != dot, dot, T.max(dot, 0.0))
                contrib[p, h] = T.cast(
                    T.cast(T.fmul(dot, W_shared[h]), T.bfloat16), accum_dtype
                )
            T.fill(total, 0)
            for h in T.serial(heads):  # increasing head order, FP32
                for p in T.Parallel(block_P):
                    total[p] += contrib[p, h]
            for p in T.Parallel(block_P):
                if p0 + p < extent:
                    Scores[row, p0 + p] = T.if_then_else(
                        slot_valid(p0, p),
                        T.cast(total[p], T.bfloat16),
                        -T.infinity(T.bfloat16),
                    )

    if not candidates:

        @T.prim_func
        def indexer_score(
            Q: T.Tensor([rows, heads, D], fp4),
            QScale: T.Tensor([rows, heads], T.uint32),
            Weights: T.Tensor([rows, heads], T.bfloat16),
            KPool: T.StridedTensor([pool_pages, page_words * 8], [pool_stride, 1], fp4),
            KScale: T.StridedTensor(
                [pool_pages, page_words], [scale_stride, 1], T.uint32
            ),
            PageTable: T.Tensor([rows, table_width], T.int32),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            Scores: T.Tensor([rows, width], T.bfloat16),
        ):
            with T.Kernel(ctas, rows, threads=threads) as (cta, row):
                extent = _row_extent(Lengths, Active, row, width)
                score_row(
                    Q,
                    QScale,
                    Weights,
                    KPool,
                    KScale,
                    PageTable,
                    row,
                    cta,
                    extent,
                    lambda p0, p: p0 + p,
                    lambda p0, p: True,
                    Scores,
                )

        return indexer_score

    @T.prim_func
    def indexer_score(
        Q: T.Tensor([rows, heads, D], fp4),
        QScale: T.Tensor([rows, heads], T.uint32),
        Weights: T.Tensor([rows, heads], T.bfloat16),
        KPool: T.StridedTensor([pool_pages, page_words * 8], [pool_stride, 1], fp4),
        KScale: T.StridedTensor([pool_pages, page_words], [scale_stride, 1], T.uint32),
        PageTable: T.Tensor([rows, table_width], T.int32),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Candidates: T.Tensor([rows, width], T.int32),
        CandidateLengths: T.Tensor([rows], T.int32),
        Scores: T.Tensor([rows, width], T.bfloat16),
    ):
        with T.Kernel(ctas, rows, threads=threads) as (cta, row):
            extent = _row_extent(CandidateLengths, None, row, width)
            score_row(
                Q,
                QScale,
                Weights,
                KPool,
                KScale,
                PageTable,
                row,
                cta,
                extent,
                lambda p0, p: T.max(Candidates[row, T.min(p0 + p, width - 1)], 0),
                lambda p0, p: Candidates[row, T.min(p0 + p, width - 1)] >= 0,
                Scores,
            )

    return indexer_score


def _e2m1_code(x):
    """Nearest-even E2M1 code of FP32 ``x``, saturating at 6 (``cvt.rn.satfinite``)."""
    m = T.abs(x)
    # Ties go to the even code: strict at even-lower midpoints, inclusive at odd.
    code = (
        T.cast(m > 0.25, T.int32)
        + T.cast(m >= 0.75, T.int32)
        + T.cast(m > 1.25, T.int32)
        + T.cast(m >= 1.75, T.int32)
        + T.cast(m > 2.5, T.int32)
        + T.cast(m >= 3.5, T.int32)
        + T.cast(m > 5.0, T.int32)
    )
    sign = T.cast(T.shift_right(T.reinterpret(x, T.uint32), 31), T.int32)
    return code | (sign << 3)


def _mxfp4_byte(X, row, first, scale):
    """Two scaled values ``X[row, first : first + 2]`` packed low nibble first."""
    lo = _e2m1_code(T.fmul(T.cast(X[row, first], T.float32), scale))
    hi = _e2m1_code(T.fmul(T.cast(X[row, first + 1], T.float32), scale))
    return T.cast(lo | (hi << 4), T.uint8)


@tilelang.jit
def mxfp4_quantize_v41(paged: bool = False, page_size: int = 64, threads: int = 128):
    """MXFP4-quantize ``[rows, 128]`` BF16 vectors, one thread per 32-value group.

    Each group's UE8M0 scale is ``pow2_ceil(max(amax, 6 * 2^-126) * fp32(1/6))``
    and its values round to nearest even E2M1, as B12X's quantizer does, so
    both backends see identical index queries and keys. The output is either
    contiguous (``Q`` ``[rows, 64]`` packed, ``S`` ``[rows, 4]``) or, with
    ``paged``, written through ``Slots`` (``[rows]`` int64; negative slots
    skip) into index-cache pages ``Pool`` (``[pages, page bytes]``, any row
    stride).
    """
    rows, pool_pages, pool_stride = T.dynamic("rows, pool_pages, pool_stride")
    groups = INDEX_DIM // SF_BLOCK
    half = SF_BLOCK // 2

    @T.macro
    def body(X, Q, S, Slots, Pool):
        with T.Kernel(T.ceildiv(rows * groups, threads), threads=threads) as bx:
            tx = T.get_thread_binding()
            g = bx * threads + tx
            row, k = g // groups, g % groups
            amax = T.alloc_var(T.float32)
            if row < rows:
                amax = 6.0 * 2.0**-126
                for i in T.serial(SF_BLOCK):
                    amax = T.max(
                        amax, T.abs(T.cast(X[row, k * SF_BLOCK + i], T.float32))
                    )
                bits = T.reinterpret(T.fmul(amax, T.float32(1.0 / 6.0)), T.uint32)
                sf = T.cast(T.shift_right(bits, 23) & 0xFF, T.int32) + T.cast(
                    (bits & 0x7FFFFF) != 0, T.int32
                )
                scale = T.reinterpret(
                    T.cast(T.shift_left(254 - sf, 23), T.uint32), T.float32
                )
                if paged:
                    slot = Slots[row]
                    page, token = slot // page_size, slot % page_size
                    if (slot >= 0) & (page < pool_pages):
                        for i in T.serial(half):
                            Pool[page, token * (INDEX_DIM // 2) + k * half + i] = (
                                _mxfp4_byte(X, row, k * SF_BLOCK + 2 * i, scale)
                            )
                        Pool[
                            page, page_size * (INDEX_DIM // 2) + token * groups + k
                        ] = T.cast(sf, T.uint8)
                else:
                    for i in T.serial(half):
                        Q[row, k * half + i] = _mxfp4_byte(
                            X, row, k * SF_BLOCK + 2 * i, scale
                        )
                    S[row, k] = T.cast(sf, T.uint8)

    if paged:

        @T.prim_func
        def indexer_mxfp4_quantize(
            X: T.Tensor([rows, INDEX_DIM], T.bfloat16),
            Slots: T.Tensor([rows], T.int64),
            Pool: T.StridedTensor(
                [pool_pages, page_bytes(page_size)], [pool_stride, 1], T.uint8
            ),
        ):
            body(X, None, None, Slots, Pool)

        return indexer_mxfp4_quantize

    @T.prim_func
    def indexer_mxfp4_quantize(
        X: T.Tensor([rows, INDEX_DIM], T.bfloat16),
        Q: T.Tensor([rows, INDEX_DIM // 2], T.uint8),
        S: T.Tensor([rows, groups], T.uint8),
    ):
        body(X, Q, S, None, None)

    return indexer_mxfp4_quantize


def quantize_mxfp4(x, q, scales):
    """MXFP4-quantize ``[..., 128]`` BF16 into ``q`` and ``scales``.

    ``q`` is ``[..., 64]`` packed E2M1; ``scales`` is ``[..., 4]`` UE8M0.
    """
    mxfp4_quantize_v41()(
        x.reshape(-1, INDEX_DIM),
        q.view(-1, INDEX_DIM // 2),
        scales.view(-1, INDEX_DIM // SF_BLOCK),
    )
    return q, scales


def write_index_keys(keys, pool, slots, page_size):
    """Quantize ``[rows, 128]`` keys into index-cache pages at ``slots`` (-1 skips)."""
    kernel = mxfp4_quantize_v41(True, page_size)
    kernel(keys, slots.to(torch.int64), pool[:, : page_bytes(page_size)])


def _key16(x):
    """Order-preserving uint16 key of a BF16 value, as int32."""
    bits = T.cast(T.reinterpret(x, T.uint16), T.int32)
    return T.if_then_else(bits >= 0x8000, bits ^ 0xFFFF, bits | 0x8000)


def _value16(key):
    """The BF16 value of an order-preserving key."""
    return T.reinterpret(
        T.cast(T.if_then_else(key >= 0x8000, key ^ 0x8000, key ^ 0xFFFF), T.uint16),
        T.bfloat16,
    )


_TOPK_THREADS = 512
_SEGMENT_ENTRIES = _TOPK_THREADS * 8


def _select_macros(threads, topk):
    """Radix threshold and ordered compaction over int32 keys held in shared memory."""
    radix = 256
    steps = threads.bit_length() - 1

    @T.macro
    def block_scan(values, tx, n_steps):
        # Inclusive Hillis-Steele scan of values[0:threads].
        carry = T.alloc_var(T.int32)
        for s in T.serial(n_steps):
            off = 1 << s
            carry = T.if_then_else(tx >= off, values[T.max(tx - off, 0)], 0)
            T.sync_threads()
            values[tx] += carry
            T.sync_threads()

    @T.macro
    def suffix_counts(hist, scan, tx):
        # hist[b] <- number of keys whose digit is >= b; hist[radix] = 0.
        if tx < radix:
            scan[tx] = hist[radix - 1 - tx]
        T.sync_threads()
        block_scan(scan, tx, 8)
        if tx < radix:
            hist[radix - 1 - tx] = scan[tx]
        if tx == 0:
            hist[radix] = 0
        T.sync_threads()

    @T.macro
    def threshold(keys, n, tx, hist, scan, params):
        """Top ``topk`` of keys[0:n]: params[0] <- threshold, params[1] <- ties kept."""
        key = T.alloc_var(T.int32)
        if tx < radix + 1:
            hist[tx] = 0
        if tx == 0:
            params[2] = 0
            params[3] = 0
        T.sync_threads()
        for i in T.serial(T.ceildiv(n, threads)):
            if i * threads + tx < n:
                key = keys[i * threads + tx]
                if key > _KEY_NEG_INF:
                    T.atomic_add(hist[key >> 8], 1)
        T.sync_threads()
        suffix_counts(hist, scan, tx)
        if tx < radix:  # noqa: SIM102 - the guard keeps hist[tx + 1] in bounds; TIR & is not short-circuit
            if (hist[tx] >= topk) & (hist[tx + 1] < topk):
                params[2] = tx
        T.sync_threads()
        hi = T.alloc_var(T.int32)
        above = T.alloc_var(T.int32)
        selectable = T.alloc_var(T.int32)
        hi = params[2]
        above = hist[hi + 1]
        selectable = hist[0]
        T.sync_threads()
        if tx < radix + 1:
            hist[tx] = 0
        T.sync_threads()
        for i in T.serial(T.ceildiv(n, threads)):
            if i * threads + tx < n:
                key = keys[i * threads + tx]
                if (key > _KEY_NEG_INF) & ((key >> 8) == hi):
                    T.atomic_add(hist[key & 0xFF], 1)
        T.sync_threads()
        suffix_counts(hist, scan, tx)
        if tx < radix:  # noqa: SIM102 - the guard keeps hist[tx + 1] in bounds
            if (above + hist[tx] >= topk) & (above + hist[tx + 1] < topk):
                params[3] = tx
        T.sync_threads()
        if tx == 0:
            if selectable <= topk:
                # Everything selectable survives.
                params[0] = _KEY_NEG_INF + 1
                params[1] = selectable
            else:
                params[0] = (hi << 8) | params[3]
                params[1] = topk - above - hist[params[3] + 1]
        T.sync_threads()

    @T.macro
    def compact(keys, per, tx, scan, params, sel_idx, sel_key):
        """Write the selected indices and keys to sel_idx/sel_key in index order.

        Ties at the threshold keep the lowest indices. Each thread owns
        keys[tx * per : (tx + 1) * per]. Leaves the selected count in
        scan[threads - 1].
        """
        key = T.alloc_var(T.int32)
        n_gt = T.alloc_var(T.int32)
        n_eq = T.alloc_var(T.int32)
        sel_eq = T.alloc_var(T.int32)
        out_pos = T.alloc_var(T.int32)
        taken = T.alloc_var(T.int32)
        thr = T.alloc_var(T.int32)
        need = T.alloc_var(T.int32)
        thr = params[0]
        need = params[1]
        n_gt = 0
        n_eq = 0
        for v in T.serial(per):
            key = keys[tx * per + v]
            if key > thr:
                n_gt += 1
            elif key == thr:
                n_eq += 1
        scan[tx] = n_eq
        T.sync_threads()
        block_scan(scan, tx, steps)
        sel_eq = T.max(T.min(need - (scan[tx] - n_eq), n_eq), 0)
        T.sync_threads()
        scan[tx] = n_gt + sel_eq
        T.sync_threads()
        block_scan(scan, tx, steps)
        out_pos = scan[tx] - n_gt - sel_eq
        taken = 0
        for v in T.serial(per):
            key = keys[tx * per + v]
            if (key > thr) | ((key == thr) & (taken < sel_eq)):
                if key == thr:
                    taken += 1
                sel_idx[out_pos] = tx * per + v
                sel_key[out_pos] = key
                out_pos += 1
        T.sync_threads()

    return threshold, compact


def _final_writer(threads, topk, group):
    @T.macro
    def write_final(
        Indices, Values, Counts, Gather, row, extent, total, entry, sel_key, params
    ):
        """Write the ``total`` selected entries (``entry(slot)``), ``-1`` padded.

        Grouped entries expand to their positions below ``extent``; ``Counts``
        receives how many positions that leaves.
        """
        tx = T.get_thread_binding()
        count = T.alloc_var(T.int32)
        count = 0
        for i in T.serial(T.ceildiv(topk, threads)):
            slot = i * threads + tx
            if slot < topk:
                if slot < total:
                    e = entry(slot)
                    Values[row, slot] = _value16(sel_key[slot])
                    if group == 1:
                        if Gather is not None:
                            Indices[row, slot] = Gather[row, e]
                        else:
                            Indices[row, slot] = e
                    else:
                        for o in T.serial(group):
                            Indices[row, slot * group + o] = T.if_then_else(
                                e * group + o < extent, e * group + o, -1
                            )
                        count += T.min(group, extent - e * group)
                else:
                    Values[row, slot] = -T.infinity(T.bfloat16)
                    for o in T.serial(group):
                        Indices[row, slot * group + o] = -1
        if Counts is not None:
            T.sync_threads()
            if tx == 0:
                params[0] = 0
            T.sync_threads()
            T.atomic_add(params[0], count)
            T.sync_threads()
            if tx == 0:
                Counts[row] = params[0]

    return write_final


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True}
)
def indexer_topk_segment_v41(
    width: int,
    topk: int = INDEX_TOPK,
    group: int = 1,
    final: bool = True,
    candidates: bool = False,
    threads: int = _TOPK_THREADS,
):
    """Stage 1: the exact top-``topk`` of each ``_SEGMENT_ENTRIES``-entry row segment.

    Inputs: ``Scores`` (``[rows, width]`` BF16), ``Lengths`` (``[rows]`` int32)
    and, unless ``candidates``, ``Active`` (``[1]`` int32): a row selects from
    its first ``min(Lengths, Active, width)`` positions. With ``group > 1`` an
    entry is a block of ``group`` positions keyed by its maximum; the block
    holding the last position always ranks first.

    ``final`` (one segment per row) writes ``Indices`` (``[rows, topk * group]``
    int32 ascending positions, ``-1`` padded) and ``Values`` (``[rows, topk]``
    BF16); ``candidates`` maps entries through ``Gather`` (``[rows, width]``),
    and ``group > 1`` also writes ``Counts`` (``[rows]`` positions kept).
    Otherwise it writes each live segment's winners, in entry order, to
    ``WorkValues`` / ``WorkEntries`` (``[rows, segments * topk]``, ``-1``
    padded); segments past a row's extent are left unwritten.
    """
    assert not (candidates and group > 1), "candidate rows select positions, not blocks"
    rows = T.dynamic("rows")
    entries = tilelang.cdiv(width, group)
    segment = _SEGMENT_ENTRIES
    segments = tilelang.cdiv(entries, segment)
    assert not final or segments == 1, "a final segment must cover the row"
    per = segment // threads
    threshold, compact = _select_macros(threads, topk)
    write_final = _final_writer(threads, topk, group)

    @T.macro
    def select_segment(
        Scores, extent, row, seg, keys, hist, scan, params, sel_idx, sel_key
    ):
        tx = T.get_thread_binding()
        best = T.alloc_var(T.int32)
        n_entries = T.ceildiv(extent, group)
        for i in T.serial(per):
            j = i * threads + tx
            e = seg * segment + j
            if group == 1:
                keys[j] = T.if_then_else(
                    e < n_entries, _key16(Scores[row, T.min(e, width - 1)]), 0
                )
            else:
                best = 0
                for o in T.serial(group):
                    if (e < n_entries) & (e * group + o < extent):
                        best = T.max(
                            best, _key16(Scores[row, T.min(e * group + o, width - 1)])
                        )
                # The block holding the last visible position always survives.
                keys[j] = T.if_then_else(
                    (e < n_entries) & (e == (extent - 1) // group), 0xFF80, best
                )
        T.sync_threads()
        threshold(keys, segment, tx, hist, scan, params)
        compact(keys, per, tx, scan, params, sel_idx, sel_key)

    @T.macro
    def body(
        Scores,
        Lengths,
        Active,
        Gather,
        Indices,
        Values,
        Counts,
        WorkValues,
        WorkEntries,
    ):
        with T.Kernel(segments, rows, threads=threads) as (seg, row):
            tx = T.get_thread_binding()
            keys = T.alloc_shared([segment], T.int32)
            hist = T.alloc_shared([257], T.int32)
            scan = T.alloc_shared([threads], T.int32)
            params = T.alloc_shared([4], T.int32)
            sel_idx = T.alloc_shared([topk], T.int32)
            sel_key = T.alloc_shared([topk], T.int32)
            extent = _row_extent(Lengths, Active, row, width)
            if final:
                select_segment(
                    Scores, extent, row, 0, keys, hist, scan, params, sel_idx, sel_key
                )
                write_final(
                    Indices,
                    Values,
                    Counts,
                    Gather,
                    row,
                    extent,
                    scan[threads - 1],
                    lambda slot: sel_idx[slot],
                    sel_key,
                    params,
                )
            elif seg * segment < T.ceildiv(extent, group):
                # Segments past the row's extent hold nothing; the merges skip them.
                select_segment(
                    Scores, extent, row, seg, keys, hist, scan, params, sel_idx, sel_key
                )
                total = scan[threads - 1]
                for i in T.serial(T.ceildiv(topk, threads)):
                    slot = i * threads + tx
                    if slot < topk:
                        if slot < total:
                            WorkValues[row, seg * topk + slot] = _value16(sel_key[slot])
                            WorkEntries[row, seg * topk + slot] = (
                                seg * segment + sel_idx[slot]
                            )
                        else:
                            WorkValues[row, seg * topk + slot] = -T.infinity(T.bfloat16)
                            WorkEntries[row, seg * topk + slot] = -1

    if not final:
        if candidates:

            @T.prim_func
            def indexer_topk_segment(
                Scores: T.Tensor([rows, width], T.bfloat16),
                Lengths: T.Tensor([rows], T.int32),
                WorkValues: T.Tensor([rows, segments * topk], T.bfloat16),
                WorkEntries: T.Tensor([rows, segments * topk], T.int32),
            ):
                body(
                    Scores,
                    Lengths,
                    None,
                    None,
                    None,
                    None,
                    None,
                    WorkValues,
                    WorkEntries,
                )

            return indexer_topk_segment

        @T.prim_func
        def indexer_topk_segment(
            Scores: T.Tensor([rows, width], T.bfloat16),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            WorkValues: T.Tensor([rows, segments * topk], T.bfloat16),
            WorkEntries: T.Tensor([rows, segments * topk], T.int32),
        ):
            body(
                Scores, Lengths, Active, None, None, None, None, WorkValues, WorkEntries
            )

        return indexer_topk_segment

    if candidates:

        @T.prim_func
        def indexer_topk_segment(
            Scores: T.Tensor([rows, width], T.bfloat16),
            Lengths: T.Tensor([rows], T.int32),
            Gather: T.Tensor([rows, width], T.int32),
            Indices: T.Tensor([rows, topk], T.int32),
            Values: T.Tensor([rows, topk], T.bfloat16),
        ):
            body(Scores, Lengths, None, Gather, Indices, Values, None, None, None)

        return indexer_topk_segment

    if group > 1:

        @T.prim_func
        def indexer_topk_segment(
            Scores: T.Tensor([rows, width], T.bfloat16),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            Indices: T.Tensor([rows, topk * group], T.int32),
            Values: T.Tensor([rows, topk], T.bfloat16),
            Counts: T.Tensor([rows], T.int32),
        ):
            body(Scores, Lengths, Active, None, Indices, Values, Counts, None, None)

        return indexer_topk_segment

    @T.prim_func
    def indexer_topk_segment(
        Scores: T.Tensor([rows, width], T.bfloat16),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Indices: T.Tensor([rows, topk], T.int32),
        Values: T.Tensor([rows, topk], T.bfloat16),
    ):
        body(Scores, Lengths, Active, None, Indices, Values, None, None, None)

    return indexer_topk_segment


@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True}
)
def indexer_topk_merge_v41(
    width: int,
    topk: int = INDEX_TOPK,
    group: int = 1,
    lists: int = 2,
    span: int = _SEGMENT_ENTRIES,
    fan_in: int = 16,
    candidates: bool = False,
    threads: int = _TOPK_THREADS,
):
    """Merge winner lists: the exact top-``topk`` of every ``fan_in`` lists.

    Input list ``l`` of a row holds the winners, in ascending entry order, of
    entries ``[l * span, (l + 1) * span)``; lists past the row's extent
    (``Lengths`` and ``Active`` as in ``indexer_topk_segment_v41``) are never
    read. One output list leaves each group of ``fan_in`` inputs, again in
    entry order, so ties keep the lowest positions at every level exactly as
    a one-pass selection would. With one group the merge is final and writes
    ``Indices``, ``Values`` (and ``Counts``) like the final segment kernel;
    otherwise it writes ``OutValues`` / ``OutEntries`` (``[rows, groups *
    topk]``), leaving groups past the extent unwritten.
    """
    assert not (candidates and group > 1), "candidate rows select positions, not blocks"
    rows = T.dynamic("rows")
    groups = tilelang.cdiv(lists, fan_in)
    final = groups == 1
    n = min(lists, fan_in) * topk
    n_pad = tilelang.cdiv(n, threads) * threads
    per = n_pad // threads
    assert n_pad * 4 <= 64 * 1024, f"{n} merge keys exceed shared memory; lower fan_in"
    threshold, compact = _select_macros(threads, topk)
    write_final = _final_writer(threads, topk, group)

    @T.macro
    def body(WorkValues, WorkEntries, Lengths, Active, Gather, Indices, Values, Counts):
        with T.Kernel(groups, rows, threads=threads) as (g, row):
            tx = T.get_thread_binding()
            keys = T.alloc_shared([n_pad], T.int32)
            hist = T.alloc_shared([257], T.int32)
            scan = T.alloc_shared([threads], T.int32)
            params = T.alloc_shared([4], T.int32)
            sel_idx = T.alloc_shared([topk], T.int32)
            sel_key = T.alloc_shared([topk], T.int32)
            extent = _row_extent(Lengths, Active, row, width)
            live = T.min(
                T.ceildiv(T.ceildiv(extent, group), span), lists
            )  # input lists with entries
            base = g * fan_in * topk
            # A final merge always writes its row; others skip groups past the extent.
            selects = True if final else g * fan_in < live
            if selects:
                for i in T.serial(per):
                    j = i * threads + tx
                    c = T.min(base + j, lists * topk - 1)
                    keys[j] = T.if_then_else(
                        (j < n)
                        & (g * fan_in + j // topk < live)
                        & (WorkEntries[row, c] >= 0),
                        _key16(WorkValues[row, c]),
                        0,
                    )
                T.sync_threads()
                threshold(keys, n_pad, tx, hist, scan, params)
                compact(keys, per, tx, scan, params, sel_idx, sel_key)
                total = scan[threads - 1]
                if final:
                    write_final(
                        Indices,
                        Values,
                        Counts,
                        Gather,
                        row,
                        extent,
                        total,
                        lambda slot: WorkEntries[row, sel_idx[slot]],
                        sel_key,
                        params,
                    )
                else:
                    for i in T.serial(T.ceildiv(topk, threads)):
                        slot = i * threads + tx
                        if slot < topk:
                            if slot < total:
                                Values[row, g * topk + slot] = _value16(sel_key[slot])
                                Indices[row, g * topk + slot] = WorkEntries[
                                    row, base + sel_idx[slot]
                                ]
                            else:
                                Values[row, g * topk + slot] = -T.infinity(T.bfloat16)
                                Indices[row, g * topk + slot] = -1

    if not final:
        if candidates:

            @T.prim_func
            def indexer_topk_merge(
                WorkValues: T.Tensor([rows, lists * topk], T.bfloat16),
                WorkEntries: T.Tensor([rows, lists * topk], T.int32),
                Lengths: T.Tensor([rows], T.int32),
                OutValues: T.Tensor([rows, groups * topk], T.bfloat16),
                OutEntries: T.Tensor([rows, groups * topk], T.int32),
            ):
                body(
                    WorkValues,
                    WorkEntries,
                    Lengths,
                    None,
                    None,
                    OutEntries,
                    OutValues,
                    None,
                )

            return indexer_topk_merge

        @T.prim_func
        def indexer_topk_merge(
            WorkValues: T.Tensor([rows, lists * topk], T.bfloat16),
            WorkEntries: T.Tensor([rows, lists * topk], T.int32),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            OutValues: T.Tensor([rows, groups * topk], T.bfloat16),
            OutEntries: T.Tensor([rows, groups * topk], T.int32),
        ):
            body(
                WorkValues,
                WorkEntries,
                Lengths,
                Active,
                None,
                OutEntries,
                OutValues,
                None,
            )

        return indexer_topk_merge

    if candidates:

        @T.prim_func
        def indexer_topk_merge(
            WorkValues: T.Tensor([rows, lists * topk], T.bfloat16),
            WorkEntries: T.Tensor([rows, lists * topk], T.int32),
            Lengths: T.Tensor([rows], T.int32),
            Gather: T.Tensor([rows, width], T.int32),
            Indices: T.Tensor([rows, topk], T.int32),
            Values: T.Tensor([rows, topk], T.bfloat16),
        ):
            body(WorkValues, WorkEntries, Lengths, None, Gather, Indices, Values, None)

        return indexer_topk_merge

    if group > 1:

        @T.prim_func
        def indexer_topk_merge(
            WorkValues: T.Tensor([rows, lists * topk], T.bfloat16),
            WorkEntries: T.Tensor([rows, lists * topk], T.int32),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            Indices: T.Tensor([rows, topk * group], T.int32),
            Values: T.Tensor([rows, topk], T.bfloat16),
            Counts: T.Tensor([rows], T.int32),
        ):
            body(
                WorkValues, WorkEntries, Lengths, Active, None, Indices, Values, Counts
            )

        return indexer_topk_merge

    @T.prim_func
    def indexer_topk_merge(
        WorkValues: T.Tensor([rows, lists * topk], T.bfloat16),
        WorkEntries: T.Tensor([rows, lists * topk], T.int32),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Indices: T.Tensor([rows, topk], T.int32),
        Values: T.Tensor([rows, topk], T.bfloat16),
    ):
        body(WorkValues, WorkEntries, Lengths, Active, None, Indices, Values, None)

    return indexer_topk_merge


_MERGE_KEYS = 8192  # keys one merge CTA holds in shared memory (32 KiB)


def _take(scratch, specs):
    """Split caller scratch into ``specs`` tensors, allocating any that are missing."""
    if scratch is None:
        return [
            torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in specs
        ]
    assert len(scratch) == len(specs), (
        f"expected {len(specs)} scratch tensors, got {len(scratch)}"
    )
    return list(scratch)


class IndexerTopK:
    """Exact top-k over BF16 row scores: one pass, or segments plus merges.

    Rows longer than one segment select each segment's winners, then merge
    them ``fan_in`` lists at a time until one list remains, so any width fits
    SM120 shared memory. Every route returns the same positions: each merge
    sees exact winners in entry order and applies the same score-then-position
    order. Segments and merge groups past a row's extent are skipped.
    ``candidates`` rows select from their first ``Lengths`` candidates and
    return the candidates' positions.
    """

    def __init__(self, width, topk=INDEX_TOPK, group=1, candidates=False):
        self.topk, self.group, self.candidates = topk, group, candidates
        entries = tilelang.cdiv(width, group)
        self.segments = tilelang.cdiv(entries, _SEGMENT_ENTRIES)
        self.levels = []  # (kernel, input lists, output lists)
        if self.segments == 1:
            self.single = indexer_topk_segment_v41(width, topk, group, True, candidates)
            return
        self.stage1 = indexer_topk_segment_v41(width, topk, group, False, candidates)
        fan_in = max(2, _MERGE_KEYS // topk)
        lists, span = self.segments, _SEGMENT_ENTRIES
        while True:
            outputs = tilelang.cdiv(lists, fan_in)
            kernel = indexer_topk_merge_v41(
                width, topk, group, lists, span, fan_in, candidates
            )
            self.levels.append((kernel, lists, outputs))
            if outputs == 1:
                break
            lists, span = outputs, span * fan_in

    def scratch_specs(self, rows):
        """Two ping-pong winner buffers (values, entries) for the merge levels."""
        if self.segments == 1:
            return ()
        widest = (self.segments * self.topk, self.levels[0][2] * self.topk)
        return tuple(
            ((rows * cols,), dtype)
            for cols in widest
            for dtype in (torch.bfloat16, torch.int32)
        )

    def run(
        self,
        scores,
        lengths,
        active,
        indices,
        values,
        *,
        gather=None,
        counts=None,
        scratch=None,
    ):
        """Select into ``indices`` / ``values``; grouped selections fill ``counts``.

        ``active`` is the ``[1]`` width cap of dense rows; candidate rows take
        ``gather`` (their candidate positions) instead.
        """
        rows = scores.shape[0]
        cap = () if self.candidates else (active,)
        final = (
            *((gather,) if self.candidates else ()),
            indices,
            values,
            *((counts,) if self.group > 1 else ()),
        )
        if self.segments == 1:
            self.single(scores, lengths, *cap, *final)
            return indices
        flat = _take(scratch, self.scratch_specs(rows))
        buffers = ((flat[0], flat[1]), (flat[2], flat[3]))

        def lists_of(buffer, lists):
            return tuple(
                t[: rows * lists * self.topk].view(rows, lists * self.topk)
                for t in buffer
            )

        self.stage1(scores, lengths, *cap, *lists_of(buffers[0], self.segments))
        for level, (kernel, lists, outputs) in enumerate(self.levels):
            work = lists_of(buffers[level % 2], lists)
            if outputs == 1:
                kernel(*work, lengths, *cap, *final)
            else:
                kernel(
                    *work, lengths, *cap, *lists_of(buffers[(level + 1) % 2], outputs)
                )
        return indices


class IndexerV41Plan:
    """Score and top-k kernels for one context width.

    ``candidates=True`` scores only the positions in a candidate list (V4.1's
    reindex layers). ``candidate_blocks=True`` also selects the top
    ``CANDIDATE_BLOCKS`` blocks of ``CANDIDATE_BLOCK`` positions, the
    candidate list a source layer publishes for the reindex layers after it.

    The plan holds no buffers: ``scratch_specs(rows)`` lists the scratch a
    call needs, which a caller can borrow from a shared workspace.
    """

    def __init__(
        self,
        width,
        page_size,
        *,
        topk=INDEX_TOPK,
        candidates=False,
        candidate_blocks=False,
    ):
        assert not (candidates and candidate_blocks), (
            "a layer reads or publishes candidates, not both"
        )
        self.width, self.page_size, self.topk = width, page_size, topk
        self.candidates, self.candidate_blocks = candidates, candidate_blocks
        self.score = indexer_score_v41(width, page_size, candidates)
        self.select = IndexerTopK(width, topk, 1, candidates)
        self.select_blocks = (
            IndexerTopK(width, CANDIDATE_BLOCKS, CANDIDATE_BLOCK)
            if candidate_blocks
            else None
        )

    def scratch_specs(self, rows):
        specs = [
            ((rows, self.width), torch.bfloat16),  # scores
            ((rows, self.topk), torch.bfloat16),  # selected values
            *self.select.scratch_specs(rows),
        ]
        if self.select_blocks is not None:
            specs.append(((rows, CANDIDATE_BLOCKS), torch.bfloat16))
            specs.extend(self.select_blocks.scratch_specs(rows))
        return tuple(specs)

    def run(
        self,
        q_packed,
        q_scales,
        weights,
        pool,
        page_table,
        lengths,
        active,
        candidates=None,
        candidate_lengths=None,
        *,
        out=None,
        candidate_out=None,
        candidate_out_lengths=None,
        scratch=None,
    ):
        """Select each row's top positions into ``out``; returns ``(out, scores)``.

        Reindex plans take ``candidates`` / ``candidate_lengths``; source plans
        write ``candidate_out`` (``[rows, CANDIDATE_BLOCKS * CANDIDATE_BLOCK]``
        ascending positions, ``-1`` padded) and ``candidate_out_lengths``
        (its count of positions). Scores past a row's extent are unwritten.
        """
        rows = q_packed.shape[0]
        buffers = _take(scratch, self.scratch_specs(rows))
        scores, values = buffers[:2]
        select_scratch = len(self.select.scratch_specs(rows))
        select_buffers = buffers[2 : 2 + select_scratch]
        if out is None:
            out = torch.empty(
                (rows, self.topk), dtype=torch.int32, device=q_packed.device
            )
        nbytes = page_bytes(self.page_size)
        args = (
            q_packed.view(torch.int8),
            q_scales.view(torch.uint32).reshape(rows, -1),
            weights,
            # Pages may carry allocator padding past their records.
            pool.view(torch.int8)[:, :nbytes],
            pool.view(torch.uint32)[:, : nbytes // 4],
            page_table,
            lengths,
            active,
        )
        if self.candidates:
            self.score(*args, candidates, candidate_lengths, scores)
            self.select.run(
                scores,
                candidate_lengths,
                None,
                out,
                values,
                gather=candidates,
                scratch=select_buffers,
            )
            return out, scores
        self.score(*args, scores)
        self.select.run(scores, lengths, active, out, values, scratch=select_buffers)
        if self.select_blocks is not None:
            block_values, *block_buffers = buffers[2 + select_scratch :]
            self.select_blocks.run(
                scores,
                lengths,
                active,
                candidate_out,
                block_values,
                counts=candidate_out_lengths,
                scratch=block_buffers,
            )
        return out, scores
