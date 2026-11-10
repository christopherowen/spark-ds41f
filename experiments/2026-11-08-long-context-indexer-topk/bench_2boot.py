import importlib.util
import json
import math
import sys
import torch
from vllm.models.deepseek_v4_1.tilelang import indexer as cand_mod


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base_mod = load_module(
    "/tmp_host/indexer_baseline.py",
    "vllm.models.deepseek_v4_1.tilelang.indexer_baseline",
)


def alloc_scratch(tk, M, device="cuda"):
    return [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in tk.scratch_specs(M)]


def bench_us(fn, warmup=8, rep=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    trim = max(1, rep // 10)
    trimmed = times[trim:-trim]
    return sum(trimmed) / len(trimmed)


def make_score_args(M, width, N, page_size):
    num_pages = math.ceil(width / page_size)
    total_pages = num_pages + 4
    q_packed = torch.randint(-128, 127, (M, 32, 64), device="cuda", dtype=torch.int8)
    q_scales = torch.randint(0, 2000000, (M, 32), device="cuda", dtype=torch.int32).view(torch.uint32)
    weights = torch.randn((M, 32), device="cuda", dtype=torch.bfloat16)
    nbytes = cand_mod.page_bytes(page_size)
    pool = torch.randint(0, 256, (total_pages, nbytes), device="cuda", dtype=torch.uint8)
    page_row = torch.arange(num_pages, device="cuda", dtype=torch.int32)
    page_table = page_row.unsqueeze(0).expand(M, -1).contiguous()
    lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
    active = torch.tensor([N], device="cuda", dtype=torch.int32)
    scores = torch.empty((M, width), device="cuda", dtype=torch.bfloat16)
    args = (
        q_packed.view(torch.int8),
        q_scales.view(torch.uint32).reshape(M, -1),
        weights,
        pool.view(torch.int8)[:, :nbytes],
        pool.view(torch.uint32)[:, : nbytes // 4],
        page_table,
        lengths,
        active,
        scores,
    )
    return args, (q_packed, q_scales, weights, pool, page_table, lengths, active, scores)


def verify_exactness():
    torch.manual_seed(20261108)
    for width, topk, group in [
        (16384, 512, 1),
        (524288, 512, 1),
        (1048576, 512, 1),
        (1048576, 2048, 8),
        (1048576, 2048, 1),
    ]:
        base_tk = base_mod.IndexerTopK(width, topk, group)
        cand_tk = cand_mod.IndexerTopK(width, topk, group)
        for N in [65536, width]:
            if N > width:
                continue
            M = 8
            scores = torch.empty((M, width), device="cuda", dtype=torch.bfloat16)
            scores[:4, :N] = torch.randn((4, N), device="cuda", dtype=torch.bfloat16)
            scores[4:6, :N] = torch.randint(0, 16, (2, N), device="cuda", dtype=torch.int32).to(torch.bfloat16)
            scores[6:, :N] = 1.0
            if N < width:
                scores[:, N:] = float("-inf")
            lengths = torch.tensor([N, N - 13, 4096, 512, N, N // 2, N, 2048], device="cuda", dtype=torch.int32)
            active = torch.tensor([N], device="cuda", dtype=torch.int32)

            b_idx = torch.empty((M, topk), device="cuda", dtype=torch.int32)
            b_val = torch.empty((M, topk), device="cuda", dtype=torch.bfloat16)
            b_cnt = torch.empty((M,), device="cuda", dtype=torch.int32) if group > 1 else None
            b_scr = alloc_scratch(base_tk, M)
            base_tk.run(scores, lengths, active, b_idx, b_val, counts=b_cnt, scratch=b_scr)

            c_idx = torch.empty((M, topk), device="cuda", dtype=torch.int32)
            c_val = torch.empty((M, topk), device="cuda", dtype=torch.bfloat16)
            c_cnt = torch.empty((M,), device="cuda", dtype=torch.int32) if group > 1 else None
            c_scr = alloc_scratch(cand_tk, M)
            cand_tk.run(scores, lengths, active, c_idx, c_val, counts=c_cnt, scratch=c_scr)

            if group > 1:
                assert torch.equal(b_cnt, c_cnt), f"cnt mismatch {width} {topk} {group} {N}"
            assert torch.equal(b_idx, c_idx), f"idx mismatch {width} {topk} {group} {N}"
            assert torch.equal(b_val, c_val), f"val mismatch {width} {topk} {group} {N}"

    # Verify Layer 20 gather_scores=True vs full baseline select
    base_plan = base_mod.IndexerV41Plan(1048576, 256, candidate_blocks=True)
    cand_plan = cand_mod.IndexerV41Plan(1048576, 256, candidate_blocks=True)
    for N in [65536, 262144, 1048576]:
        M = 6
        scores = torch.randn((M, 1048576), device="cuda", dtype=torch.bfloat16)
        scores[2, :N] = torch.randint(0, 32, (N,), device="cuda", dtype=torch.int32).to(torch.bfloat16)
        scores[3, :N] = 7.5
        if N < 1048576:
            scores[:, N:] = float("-inf")
        lengths = torch.tensor([N, N - 257, N, N, 16384, 500], device="cuda", dtype=torch.int32)
        active = torch.tensor([N], device="cuda", dtype=torch.int32)

        b_cand = torch.empty((M, 2048), device="cuda", dtype=torch.int32)
        b_cand_val = torch.empty((M, 2048), device="cuda", dtype=torch.bfloat16)
        b_cand_cnt = torch.empty((M,), device="cuda", dtype=torch.int32)
        b_bscr = alloc_scratch(base_plan.select_blocks, M)
        base_plan.select_blocks.run(scores, lengths, active, b_cand, b_cand_val, counts=b_cand_cnt, scratch=b_bscr)

        b_idx = torch.empty((M, 512), device="cuda", dtype=torch.int32)
        b_val = torch.empty((M, 512), device="cuda", dtype=torch.bfloat16)
        b_scr = alloc_scratch(base_plan.select, M)
        base_plan.select.run(scores, lengths, active, b_idx, b_val, scratch=b_scr)

        c_cand = torch.empty((M, 2048), device="cuda", dtype=torch.int32)
        c_cand_val = torch.empty((M, 2048), device="cuda", dtype=torch.bfloat16)
        c_cand_cnt = torch.empty((M,), device="cuda", dtype=torch.int32)
        c_bscr = alloc_scratch(cand_plan.select_blocks, M)
        cand_plan.select_blocks.run(scores, lengths, active, c_cand, c_cand_val, counts=c_cand_cnt, scratch=c_bscr)

        c_idx = torch.empty((M, 512), device="cuda", dtype=torch.int32)
        c_val = torch.empty((M, 512), device="cuda", dtype=torch.bfloat16)
        c_scr = alloc_scratch(cand_plan.select, M)
        cand_plan.select.run(scores, c_cand_cnt, None, c_idx, c_val, gather=c_cand, scratch=c_scr)

        assert torch.equal(b_cand_cnt, c_cand_cnt)
        assert torch.equal(b_cand, c_cand)
        assert torch.equal(b_idx, c_idx)
        assert torch.equal(b_val, c_val)


def main():
    out_path = sys.argv[1]
    verify_exactness()
    print("Exactness verification: PASS", flush=True)

    Ms = [1, 6, 16, 32, 64, 256, 1024]
    Ns = [65536, 131072, 262144, 524288, 1048576]

    score_128 = cand_mod.IndexerV41Plan(524288, 128, candidate_blocks=False)
    score_256 = cand_mod.IndexerV41Plan(1048576, 256, candidate_blocks=True)

    base_dense = base_mod.IndexerTopK(524288, 512, 1)
    cand_dense = cand_mod.IndexerTopK(524288, 512, 1)

    base_src_plan = base_mod.IndexerV41Plan(1048576, 256, candidate_blocks=True)
    cand_src_plan = cand_mod.IndexerV41Plan(1048576, 256, candidate_blocks=True)

    base_1m_512 = base_mod.IndexerTopK(1048576, 512, 1)
    cand_1m_512 = cand_mod.IndexerTopK(1048576, 512, 1)

    base_1m_2048 = base_mod.IndexerTopK(1048576, 2048, 1)
    cand_1m_2048 = cand_mod.IndexerTopK(1048576, 2048, 1)

    base_reidx = base_mod.IndexerTopK(16384, 512, 1, candidates=True)
    cand_reidx = cand_mod.IndexerTopK(16384, 512, 1, candidates=True)

    results = {
        "exactness": "PASS",
        "reindex_16k": {},
        "grid": {},
    }

    for M in Ms:
        scores_16k = torch.randn((M, 16384), device="cuda", dtype=torch.bfloat16)
        cands_16k = torch.arange(2048, device="cuda", dtype=torch.int32).unsqueeze(0).expand(M, -1).contiguous()
        lens_16k = torch.full((M,), 16384, device="cuda", dtype=torch.int32)
        idx = torch.empty((M, 512), device="cuda", dtype=torch.int32)
        val = torch.empty((M, 512), device="cuda", dtype=torch.bfloat16)
        b_scr = alloc_scratch(base_reidx, M)
        c_scr = alloc_scratch(cand_reidx, M)
        t_base = bench_us(lambda: base_reidx.run(scores_16k, lens_16k, None, idx, val, gather=cands_16k, scratch=b_scr))
        t_cand = bench_us(lambda: cand_reidx.run(scores_16k, lens_16k, None, idx, val, gather=cands_16k, scratch=c_scr))
        results["reindex_16k"][str(M)] = {"base_us": round(t_base, 2), "cand_us": round(t_cand, 2)}
        del scores_16k, cands_16k, lens_16k, idx, val, b_scr, c_scr
        torch.cuda.empty_cache()

    for N in Ns:
        results["grid"][str(N)] = {}
        for M in Ms:
            row = {}
            # 1. Score kernel latency
            if N <= 524288:
                s_args, s_keep = make_score_args(M, 524288, N, 128)
                row["score_p128_us"] = round(bench_us(lambda: score_128.score(*s_args)), 2)
                del s_args, s_keep
                torch.cuda.empty_cache()

            s_args, s_keep = make_score_args(M, 1048576, N, 256)
            row["score_p256_us"] = round(bench_us(lambda: score_256.score(*s_args)), 2)
            del s_args, s_keep
            torch.cuda.empty_cache()

            # 2. Dense layer IndexerTopK(524288, 512, 1) for N <= 524288
            if N <= 524288:
                scores_524k = torch.randn((M, 524288), device="cuda", dtype=torch.bfloat16)
                lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
                active = torch.tensor([N], device="cuda", dtype=torch.int32)
                idx = torch.empty((M, 512), device="cuda", dtype=torch.int32)
                val = torch.empty((M, 512), device="cuda", dtype=torch.bfloat16)
                b_scr = alloc_scratch(base_dense, M)
                c_scr = alloc_scratch(cand_dense, M)
                row["dense_524k_k512_base_us"] = round(
                    bench_us(lambda: base_dense.run(scores_524k, lengths, active, idx, val, scratch=b_scr)), 2
                )
                row["dense_524k_k512_cand_us"] = round(
                    bench_us(lambda: cand_dense.run(scores_524k, lengths, active, idx, val, scratch=c_scr)), 2
                )
                del scores_524k, lengths, active, idx, val, b_scr, c_scr
                torch.cuda.empty_cache()

            # 3. 1M width selectors: Source layer (select_blocks + select), standalone k=512, standalone k=2048
            scores_1m = torch.randn((M, 1048576), device="cuda", dtype=torch.bfloat16)
            lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
            active = torch.tensor([N], device="cuda", dtype=torch.int32)
            idx512 = torch.empty((M, 512), device="cuda", dtype=torch.int32)
            val512 = torch.empty((M, 512), device="cuda", dtype=torch.bfloat16)
            idx2048 = torch.empty((M, 2048), device="cuda", dtype=torch.int32)
            val2048 = torch.empty((M, 2048), device="cuda", dtype=torch.bfloat16)
            cnt2048 = torch.empty((M,), device="cuda", dtype=torch.int32)

            # Source layer select_blocks (k=2048, g=8)
            b_bscr = alloc_scratch(base_src_plan.select_blocks, M)
            c_bscr = alloc_scratch(cand_src_plan.select_blocks, M)
            t_b_blk = bench_us(
                lambda: base_src_plan.select_blocks.run(
                    scores_1m, lengths, active, idx2048, val2048, counts=cnt2048, scratch=b_bscr
                )
            )
            t_c_blk = bench_us(
                lambda: cand_src_plan.select_blocks.run(
                    scores_1m, lengths, active, idx2048, val2048, counts=cnt2048, scratch=c_bscr
                )
            )
            row["src_1m_blk_k2048_g8_base_us"] = round(t_b_blk, 2)
            row["src_1m_blk_k2048_g8_cand_us"] = round(t_c_blk, 2)

            # Source layer select (k=512): base is full 1M select, cand is gather_scores=True from idx2048
            b_sscr = alloc_scratch(base_src_plan.select, M)
            c_sscr = alloc_scratch(cand_src_plan.select, M)
            t_b_sel = bench_us(
                lambda: base_src_plan.select.run(scores_1m, lengths, active, idx512, val512, scratch=b_sscr)
            )
            t_c_sel = bench_us(
                lambda: cand_src_plan.select.run(
                    scores_1m, cnt2048, None, idx512, val512, gather=idx2048, scratch=c_sscr
                )
            )
            row["src_1m_sel_k512_base_us"] = round(t_b_sel, 2)
            row["src_1m_sel_k512_cand_us"] = round(t_c_sel, 2)
            row["src_1m_total_base_us"] = round(t_b_blk + t_b_sel, 2)
            row["src_1m_total_cand_us"] = round(t_c_blk + t_c_sel, 2)
            del b_bscr, c_bscr, b_sscr, c_sscr

            # Standalone 1M k=512 g=1
            b_scr512 = alloc_scratch(base_1m_512, M)
            c_scr512 = alloc_scratch(cand_1m_512, M)
            row["raw_1m_k512_g1_base_us"] = round(
                bench_us(lambda: base_1m_512.run(scores_1m, lengths, active, idx512, val512, scratch=b_scr512)),
                2,
            )
            row["raw_1m_k512_g1_cand_us"] = round(
                bench_us(lambda: cand_1m_512.run(scores_1m, lengths, active, idx512, val512, scratch=c_scr512)),
                2,
            )
            del b_scr512, c_scr512

            # Standalone 1M k=2048 g=1
            b_scr2048 = alloc_scratch(base_1m_2048, M)
            c_scr2048 = alloc_scratch(cand_1m_2048, M)
            row["raw_1m_k2048_g1_base_us"] = round(
                bench_us(lambda: base_1m_2048.run(scores_1m, lengths, active, idx2048, val2048, scratch=b_scr2048)),
                2,
            )
            row["raw_1m_k2048_g1_cand_us"] = round(
                bench_us(lambda: cand_1m_2048.run(scores_1m, lengths, active, idx2048, val2048, scratch=c_scr2048)),
                2,
            )
            del b_scr2048, c_scr2048, scores_1m, lengths, active, idx512, val512, idx2048, val2048, cnt2048
            torch.cuda.empty_cache()

            results["grid"][str(N)][str(M)] = row
            print(f"Done N={N} M={M}: {json.dumps(row)}", flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"SAVED {out_path}", flush=True)


if __name__ == "__main__":
    main()
