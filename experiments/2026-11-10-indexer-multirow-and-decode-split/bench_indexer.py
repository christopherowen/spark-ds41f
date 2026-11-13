import math
import torch
import tilelang
import tilelang.language as T
from vllm.models.deepseek_v4_1.tilelang import pdl
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    CANDIDATE_BLOCK,
    CANDIDATE_BLOCKS,
    INDEX_DIM,
    INDEX_HEADS,
    SF_BLOCK,
    IndexerV41Plan,
    _row_extent,
    indexer_score_v41,
    page_bytes,
)

DEVICE = "cuda"


@tilelang.jit
def baseline_indexer_score_1row(
    width: int,
    page_size: int,
    heads: int = INDEX_HEADS,
    block_P: int = 64,
    threads: int = 128,
    max_ctas: int = 128,
):
    rows, pool_pages, table_width, pool_stride, scale_stride = T.dynamic(
        "rows, pool_pages, table_width, pool_stride, scale_stride"
    )
    D = INDEX_DIM
    fp4 = T.float4_e2m1fn
    accum_dtype = T.float32
    page_words = page_bytes(page_size) // 4
    scale_word0 = page_size * D // 2 // 4
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
        Scores: T.Tensor([rows, width], T.bfloat16),
    ):
        with T.Kernel(ctas, rows, threads=threads) as (cta, row):
            pdl.chain()
            extent = _row_extent(Lengths, Active, row, width)
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
                    load_keys(
                        KPool,
                        KScale,
                        PageTable,
                        row,
                        p0,
                        lambda p0, p: p0 + p,
                        lambda p0, p: p0 + p < extent,
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
                    for p, h in T.Parallel(block_P, heads):
                        dot = T.max(
                            T.cast(T.cast(acc[p, h], T.bfloat16), accum_dtype), 0.0
                        )
                        contrib[p, h] = T.cast(
                            T.cast(T.fmul(dot, W_shared[h]), T.bfloat16), accum_dtype
                        )
                    T.fill(total, 0)
                    for h in T.serial(heads):
                        for p in T.Parallel(block_P):
                            total[p] += contrib[p, h]
                    for p in T.Parallel(block_P):
                        if p0 + p < extent:
                            Scores[row, p0 + p] = T.cast(total[p], T.bfloat16)

    return indexer_score


def bench_fn(fn, warmup=25, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def make_inputs(rows, width, page_size, req_span=6):
    gen = torch.Generator(device=DEVICE).manual_seed(rows * 1000 + width + page_size)
    logical_pages = width // page_size
    num_reqs = max(1, math.ceil(rows / req_span))
    pool_pages = logical_pages * num_reqs + 4
    pool = torch.randint(
        0, 256, (pool_pages, page_bytes(page_size)), generator=gen, device=DEVICE, dtype=torch.uint8
    )
    pool[:, page_size * (INDEX_DIM // 2) :] = torch.randint(
        120, 130, (pool_pages, page_size * 4), generator=gen, device=DEVICE, dtype=torch.uint8
    )
    req_tables = torch.stack(
        [
            torch.randperm(pool_pages - 1, generator=gen, device=DEVICE)[:logical_pages] + 1
            for _ in range(num_reqs)
        ]
    ).to(torch.int32)
    req_ids = torch.arange(rows, device=DEVICE) // req_span
    page_table = req_tables[req_ids]
    q = torch.randint(
        0, 256, (rows, INDEX_HEADS, INDEX_DIM // 2), generator=gen, device=DEVICE, dtype=torch.uint8
    )
    q_scales = torch.randint(
        120, 130, (rows, INDEX_HEADS, 4), generator=gen, device=DEVICE, dtype=torch.uint8
    )
    weights = (torch.randn((rows, INDEX_HEADS), generator=gen, device=DEVICE) / 64).bfloat16()
    lengths = torch.full((rows,), width - 3, dtype=torch.int32, device=DEVICE)
    active = torch.tensor([width], dtype=torch.int32, device=DEVICE)
    return q, q_scales, weights, pool, page_table, lengths, active


def main():
    print("=== Part A: Dense indexer_score_v41 (1-row per CTA vs 2-row key-tile reuse) ===")
    print(f"{'Keys':>8} | {'Page':>4} | {'Rows':>4} | {'ReqSpan':>7} | {'1-Row (us)':>11} | {'2-Row (us)':>11} | {'Speedup':>8} | {'BitExact':>8}")
    print("-" * 86)
    for width, page_size in [(16384, 64), (65536, 64), (65536, 128), (131072, 128)]:
        k_base = baseline_indexer_score_1row(width, page_size)
        k_cand = indexer_score_v41(width, page_size, candidates=False)
        for rows, req_span in [(6, 6), (12, 6), (24, 6), (48, 6), (256, 256)]:
            q, q_scales, weights, pool, page_table, lengths, active = make_inputs(
                rows, width, page_size, req_span=req_span
            )
            scores_base = torch.zeros((rows, width), dtype=torch.bfloat16, device=DEVICE)
            scores_cand = torch.zeros((rows, width), dtype=torch.bfloat16, device=DEVICE)
            args_common = (
                q.view(torch.int8),
                q_scales.view(torch.uint32).reshape(rows, -1),
                weights,
                pool.view(torch.int8),
                pool.view(torch.uint32),
                page_table,
                lengths,
                active,
            )
            k_base(*args_common, scores_base)
            k_cand(*args_common, scores_cand)
            torch.cuda.synchronize()
            ext = int(lengths[0])
            bit_exact = bool(torch.equal(scores_base[:, :ext], scores_cand[:, :ext]))
            assert bit_exact, f"Mismatch at width={width}, page_size={page_size}, rows={rows}"

            t_base = bench_fn(lambda: k_base(*args_common, scores_base))
            t_cand = bench_fn(lambda: k_cand(*args_common, scores_cand))
            print(
                f"{width:8d} | {page_size:4d} | {rows:4d} | {req_span:7d} | {t_base:11.2f} | {t_cand:11.2f} | {t_base / t_cand:7.2f}x | {str(bit_exact):>8}"
            )

    print("\n=== Part B: Full 8-Layer Decode Indexer Step (Layers 2,8,14,20,26,32,38,44) ===")
    print("  4 dense layers (3x ratio-2 compressed + 1x ratio-1 full) + 4 candidate layers")
    print(f"{'Context':>8} | {'Rows':>4} | {'Base Unsplit (ms)':>17} | {'TP4 Split+2Row (ms)':>19} | {'TP4 Saved (ms)':>14} | {'TP3 Split+2Row (ms)':>19} | {'TP3 Saved (ms)':>14}")
    print("-" * 118)
    cand_width = CANDIDATE_BLOCKS * CANDIDATE_BLOCK
    for context in [32768, 131072]:
        w_comp = context // 2
        w_full = context
        k_base_comp = baseline_indexer_score_1row(w_comp, 64)
        k_base_full = baseline_indexer_score_1row(w_full, 128)
        plan_comp = IndexerV41Plan(w_comp, 64)
        plan_full = IndexerV41Plan(w_full, 128, candidate_blocks=True)
        plan_cand = IndexerV41Plan(cand_width, 128, candidates=True)

        for rows in [6, 24, 48, 96]:
            q_c, qs_c, wt_c, pool_c, pt_c, len_c, act_c = make_inputs(rows, w_comp, 64, req_span=6)
            q_f, qs_f, wt_f, pool_f, pt_f, len_f, act_f = make_inputs(rows, w_full, 128, req_span=6)
            out_idx = torch.empty((rows, 512), dtype=torch.int32, device=DEVICE)
            cand_out = torch.empty((rows, CANDIDATE_BLOCKS), dtype=torch.int32, device=DEVICE)
            cand_lens = torch.empty((rows,), dtype=torch.int32, device=DEVICE)
            scores_c = torch.empty((rows, w_comp), dtype=torch.bfloat16, device=DEVICE)
            scores_f = torch.empty((rows, w_full), dtype=torch.bfloat16, device=DEVICE)
            vals_top = torch.empty((rows, 512), dtype=torch.bfloat16, device=DEVICE)
            vals_blk = torch.empty((rows, CANDIDATE_BLOCKS), dtype=torch.bfloat16, device=DEVICE)
            scratch_comp_top = [torch.empty(s, dtype=d, device=DEVICE) for s, d in plan_comp.select.scratch_specs(rows)]
            scratch_full_top = [torch.empty(s, dtype=d, device=DEVICE) for s, d in plan_full.select.scratch_specs(rows)]
            scratch_full_blk = [torch.empty(s, dtype=d, device=DEVICE) for s, d in plan_full.select_blocks.scratch_specs(rows)]
            scratch_comp_all = [torch.empty(s, dtype=d, device=DEVICE) for s, d in plan_comp.scratch_specs(rows)]
            scratch_full_all = [torch.empty(s, dtype=d, device=DEVICE) for s, d in plan_full.scratch_specs(rows)]
            scratch_cand_all = [torch.empty(s, dtype=d, device=DEVICE) for s, d in plan_cand.scratch_specs(rows)]

            # Warmup & populate candidate_out
            plan_full.run(
                q_f, qs_f, wt_f, pool_f, pt_f, len_f, act_f,
                out=out_idx,
                candidate_out=cand_out, candidate_out_lengths=cand_lens,
                scratch=scratch_full_all,
            )

            def run_base_step():
                for _ in range(3):
                    k_base_comp(
                        q_c.view(torch.int8), qs_c.view(torch.uint32).reshape(rows, -1),
                        wt_c, pool_c.view(torch.int8), pool_c.view(torch.uint32),
                        pt_c, len_c, act_c, scores_c,
                    )
                    plan_comp.select.run(scores_c, len_c, act_c, out_idx, vals_top, scratch=scratch_comp_top)
                k_base_full(
                    q_f.view(torch.int8), qs_f.view(torch.uint32).reshape(rows, -1),
                    wt_f, pool_f.view(torch.int8), pool_f.view(torch.uint32),
                    pt_f, len_f, act_f, scores_f,
                )
                plan_full.select.run(scores_f, len_f, act_f, out_idx, vals_top, scratch=scratch_full_top)
                plan_full.select_blocks.run(
                    scores_f, len_f, act_f, cand_out, vals_blk, counts=cand_lens, scratch=scratch_full_blk
                )
                for _ in range(4):
                    plan_cand.run(
                        q_f, qs_f, wt_f, pool_f, pt_f, len_f, act_f,
                        cand_out, cand_lens, out=out_idx, scratch=scratch_cand_all,
                    )

            def make_split_runner(tp_size):
                local = math.ceil(rows / tp_size)
                sc_c = [torch.empty(s, dtype=d, device=DEVICE) for s, d in plan_comp.scratch_specs(local)]
                sc_f = [torch.empty(s, dtype=d, device=DEVICE) for s, d in plan_full.scratch_specs(local)]
                sc_k = [torch.empty(s, dtype=d, device=DEVICE) for s, d in plan_cand.scratch_specs(local)]
                o_l = out_idx[:local]
                c_o = cand_out[:local]
                c_l = cand_lens[:local]

                def run_split():
                    for _ in range(3):
                        plan_comp.run(
                            q_c[:local], qs_c[:local], wt_c[:local],
                            pool_c, pt_c[:local], len_c[:local], act_c,
                            out=o_l, scratch=sc_c,
                        )
                    plan_full.run(
                        q_f[:local], qs_f[:local], wt_f[:local],
                        pool_f, pt_f[:local], len_f[:local], act_f,
                        out=o_l,
                        candidate_out=c_o, candidate_out_lengths=c_l,
                        scratch=sc_f,
                    )
                    for _ in range(4):
                        plan_cand.run(
                            q_f[:local], qs_f[:local], wt_f[:local],
                            pool_f, pt_f[:local], len_f[:local], act_f,
                            c_o, c_l, out=o_l, scratch=sc_k,
                        )

                return run_split

            t_base_ms = bench_fn(run_base_step, warmup=10, iters=40) / 1000.0
            t_tp4_ms = bench_fn(make_split_runner(4), warmup=10, iters=40) / 1000.0
            t_tp3_ms = bench_fn(make_split_runner(3), warmup=10, iters=40) / 1000.0
            print(
                f"{context:8d} | {rows:4d} | {t_base_ms:17.3f} | {t_tp4_ms:19.3f} | {t_base_ms - t_tp4_ms:14.3f} | {t_tp3_ms:19.3f} | {t_base_ms - t_tp3_ms:14.3f}"
            )


if __name__ == "__main__":
    main()
