# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Out-of-band 64-mean tile; native score arithmetic, no model/runtime adoption."""
import tilelang
import tilelang.language as T
from vllm.models.deepseek_v4_1.tilelang import pdl
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    INDEX_HEADS, INDEX_DIM, CANDIDATE_BLOCK, SF_BLOCK, page_bytes,
    _row_extent, _candidate_block, _candidate_position,
)

@tilelang.jit
def indexer_score_v41(
    width: int,
    page_size: int,
    candidates: bool = False,
    heads: int = INDEX_HEADS,
    block_P: int = 64,
    threads: int = 128,
    max_ctas: int = 128,
    summary_page: int = 0,
    summary_budget: int = 0,
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
    - candidate mode: ``Candidates`` ``[rows, width // CANDIDATE_BLOCK]`` int32
      block ids (``-1``: a masked block) and ``CandidateLengths`` ``[rows]``,
      candidate ``j`` being position ``Candidates[j // 8] * 8 + j % 8``; its
      source layer writes the ids whole, ``-1`` padded.

    Writes ``Scores`` (``[rows, width]`` BF16) up to each row's extent, ``-inf``
    for masked candidates; entries past the extent are left unwritten and the
    top-k never reads them. No key or scale is read past the visible positions
    either.
    Each row gets at most ``max_ctas`` CTAs, which loop over its live tiles, so
    the launch scales with the context in use rather than the planned width.
    """
    assert page_size % block_P == 0 or candidates or summary_page, (
        "dense tiles must not straddle a page"
    )
    assert not candidates or width % block_P == 0, "candidate rows score whole tiles"
    blocks = width // CANDIDATE_BLOCK
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
    def load_keys(
        KPool, KScale, PageTable, row, p0, position, loads, K_shared, SFK_shared
    ):
        for p, d in T.Parallel(block_P, D):
            pos = position(p0, p)
            page = T.min(
                T.max(PageTable[row, T.min(pos // page_size, table_width - 1)], 0),
                pool_pages - 1,
            )
            K_shared[p, d] = T.if_then_else(
                loads(p0, p),
                KPool[page, (pos % page_size) * D + d],
                T.cast(0, fp4),
            )
        for p in T.Parallel(block_P):
            pos = position(p0, p)
            page = T.min(
                T.max(PageTable[row, T.min(pos // page_size, table_width - 1)], 0),
                pool_pages - 1,
            )
            SFK_shared[p, 0] = T.if_then_else(
                loads(p0, p),
                KScale[page, scale_word0 + pos % page_size],
                T.cast(0, T.uint32),
            )

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
        loads,
        whole,
        whole_position,
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

        if cta * block_P < extent:
            for h, d in T.Parallel(heads, D):
                Q_shared[h, d] = Q[row, h, d]
            for h in T.Parallel(heads):
                SFQ_shared[h, 0] = QScale[row, h]
                W_shared[h] = T.cast(Weights[row, h], accum_dtype)
            for i in T.serial(
                T.ceildiv(T.max(T.ceildiv(extent, block_P) - cta, 0), ctas)
            ):
                p0 = (cta + i * ctas) * block_P
                # No load reaches past the visible positions (DeepSeek's reference reads
                # ``index_k[:, : end_pos // ratio]``). Tile rows that ``loads`` refuses
                # (a dense row's rows past its extent) load no key or scale and score
                # zeros, which the store drops. Only the key and scale loads are
                # predicated: the page read stays unconditional, so a dense tile's
                # addresses stay affine and its loads issue together. Every candidate
                # row loads: its position is always a visible one (a masked or padding
                # candidate's too, ``_candidate_position``), and its score is masked at
                # the end. Only the tile holding a candidate row's count bounds its
                # positions by it: a tile ``whole`` below the count takes
                # ``whole_position``, chosen once for the tile.
                if whole is None:
                    load_keys(
                        KPool,
                        KScale,
                        PageTable,
                        row,
                        p0,
                        position,
                        loads,
                        K_shared,
                        SFK_shared,
                    )
                elif whole(p0):
                    load_keys(
                        KPool,
                        KScale,
                        PageTable,
                        row,
                        p0,
                        whole_position,
                        loads,
                        K_shared,
                        SFK_shared,
                    )
                else:
                    load_keys(
                        KPool,
                        KScale,
                        PageTable,
                        row,
                        p0,
                        position,
                        loads,
                        K_shared,
                        SFK_shared,
                    )
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
                for p, h in T.Parallel(block_P, heads):
                    dot = T.max(
                        T.cast(T.cast(acc[p, h], T.bfloat16), accum_dtype), 0.0
                    )
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

    if summary_page:
        assert not candidates and summary_page == page_size * 8
        assert summary_budget > 0

        @T.prim_func
        def summary_score(
            Q: T.Tensor([rows, heads, D], fp4),
            QScale: T.Tensor([rows, heads], T.uint32),
            Weights: T.Tensor([rows, heads], T.bfloat16),
            KPool: T.StridedTensor([pool_pages, page_words * 8], [pool_stride, 1], fp4),
            KScale: T.StridedTensor([pool_pages, page_words], [scale_stride, 1], T.uint32),
            PageTable: T.Tensor([rows, table_width], T.int32),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            Scores: T.Tensor([rows, width], T.bfloat16),
            Counts: T.Tensor([rows], T.int32),
        ):
            with T.Kernel(ctas, rows, threads=threads) as (cta, row):
                pdl.chain()
                raw = _row_extent(Lengths, Active, row, width * 8)
                full = raw // summary_page * page_size
                extent = T.if_then_else(full <= summary_budget, 0, full)
                if (cta == 0) & (T.get_thread_binding() == 0):
                    Counts[row] = extent
                score_row(Q, QScale, Weights, KPool, KScale, PageTable, row, cta,
                          extent, lambda p0, p: T.min(p0+p, T.max(extent-1, 0)),
                          lambda p0, p: True, lambda p0, p: p0+p < extent,
                          lambda p0: p0+block_P <= extent, lambda p0, p: p0+p, Scores)
        return summary_score

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
                pdl.chain()
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
                    lambda p0, p: p0 + p < extent,
                    None,
                    None,
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
        Candidates: T.Tensor([rows, blocks], T.int32),
        CandidateLengths: T.Tensor([rows], T.int32),
        Scores: T.Tensor([rows, width], T.bfloat16),
    ):
        with T.Kernel(ctas, rows, threads=threads) as (cta, row):
            pdl.chain()
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
                lambda p0, p: _candidate_position(Candidates, row, p0 + p, extent),
                lambda p0, p: _candidate_block(Candidates, row, p0 + p) >= 0,
                lambda p0, p: True,
                lambda p0: p0 + block_P <= extent,
                lambda p0, p: _candidate_position(Candidates, row, p0 + p),
                Scores,
            )

    return indexer_score

