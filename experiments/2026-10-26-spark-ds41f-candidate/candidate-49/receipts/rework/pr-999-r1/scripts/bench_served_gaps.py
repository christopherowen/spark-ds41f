#!/usr/bin/env python3
"""Paired 1107/candidate timings of the served top-k configurations PR #999's
bench_2boot.py grid does not time, plus a per-kernel breakdown of the one slower
grid cell (IndexerTopK(1048576, 512, 1) at N=524288, M=1).

Methodology is bench_2boot.py's own: it is imported (so base = the module mounted at
/tmp_host/indexer_baseline.py, cand = the image's indexer), with its bench_us
(8 warmups, 25 timed calls, 10% trimmed mean) and alloc_scratch, and its score
fill (torch.randn rows). Served configurations come from served_topk_configs.py.

usage: bench_served_gaps.py <out.json>
"""
import json
import sys

import torch

sys.path.insert(0, "/tmp")
import bench_2boot as b  # noqa: E402

base_mod, cand_mod = b.base_mod, b.cand_mod
bench_us, alloc_scratch = b.bench_us, b.alloc_scratch
CB, CBS = cand_mod.CANDIDATE_BLOCKS, cand_mod.CANDIDATE_BLOCK


def dense_cell(base_tk, cand_tk, width, N, M):
    scores = torch.randn((M, width), device="cuda", dtype=torch.bfloat16)
    lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
    active = torch.tensor([N], device="cuda", dtype=torch.int32)
    idx = torch.empty((M, 512), device="cuda", dtype=torch.int32)
    val = torch.empty((M, 512), device="cuda", dtype=torch.bfloat16)
    bs, cs = alloc_scratch(base_tk, M), alloc_scratch(cand_tk, M)
    base_tk.run(scores, lengths, active, idx, val, scratch=bs)
    ref = idx.clone(), val.clone()
    cand_tk.run(scores, lengths, active, idx, val, scratch=cs)
    exact = torch.equal(ref[0], idx) and torch.equal(ref[1], val)
    tb = bench_us(lambda: base_tk.run(scores, lengths, active, idx, val, scratch=bs))
    tc = bench_us(lambda: cand_tk.run(scores, lengths, active, idx, val, scratch=cs))
    return {"base_us": round(tb, 2), "cand_us": round(tc, 2), "exact": exact}


def reindex_cell(base_tk, cand_tk, N, M):
    scores = torch.randn((M, CB * CBS), device="cuda", dtype=torch.bfloat16)
    cands = torch.arange(CB, device="cuda", dtype=torch.int32).unsqueeze(0).expand(M, -1).contiguous()
    lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
    idx = torch.empty((M, 512), device="cuda", dtype=torch.int32)
    val = torch.empty((M, 512), device="cuda", dtype=torch.bfloat16)
    bs, cs = alloc_scratch(base_tk, M), alloc_scratch(cand_tk, M)
    base_tk.run(scores, lengths, None, idx, val, gather=cands, scratch=bs)
    ref = idx.clone(), val.clone()
    cand_tk.run(scores, lengths, None, idx, val, gather=cands, scratch=cs)
    exact = torch.equal(ref[0], idx) and torch.equal(ref[1], val)
    tb = bench_us(lambda: base_tk.run(scores, lengths, None, idx, val, gather=cands, scratch=bs))
    tc = bench_us(lambda: cand_tk.run(scores, lengths, None, idx, val, gather=cands, scratch=cs))
    return {"base_us": round(tb, 2), "cand_us": round(tc, 2), "exact": exact}


def source_cell(base_plan, cand_plan, width, N, M):
    """Source layer: 1107 = select_blocks + full-width select; candidate =
    select_blocks + gather_scores select from the published blocks."""
    scores = torch.randn((M, width), device="cuda", dtype=torch.bfloat16)
    lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
    active = torch.tensor([N], device="cuda", dtype=torch.int32)
    i512 = torch.empty((M, 512), device="cuda", dtype=torch.int32)
    v512 = torch.empty((M, 512), device="cuda", dtype=torch.bfloat16)
    i2k = torch.empty((M, CB), device="cuda", dtype=torch.int32)
    v2k = torch.empty((M, CB), device="cuda", dtype=torch.bfloat16)
    c2k = torch.empty((M,), device="cuda", dtype=torch.int32)
    bb, cb = alloc_scratch(base_plan.select_blocks, M), alloc_scratch(cand_plan.select_blocks, M)
    bsel, csel = alloc_scratch(base_plan.select, M), alloc_scratch(cand_plan.select, M)

    def base_blk():
        base_plan.select_blocks.run(scores, lengths, active, i2k, v2k, counts=c2k, scratch=bb)

    def cand_blk():
        cand_plan.select_blocks.run(scores, lengths, active, i2k, v2k, counts=c2k, scratch=cb)

    def base_sel():
        base_plan.select.run(scores, lengths, active, i512, v512, scratch=bsel)

    def cand_sel():
        cand_plan.select.run(scores, c2k, None, i512, v512, gather=i2k, scratch=csel)

    base_blk()
    ref_blk = i2k.clone(), c2k.clone()
    base_sel()
    ref = i512.clone(), v512.clone()
    cand_blk()
    exact_blk = torch.equal(ref_blk[0], i2k) and torch.equal(ref_blk[1], c2k)
    cand_sel()
    exact = exact_blk and torch.equal(ref[0], i512) and torch.equal(ref[1], v512)
    t = {}
    t["blk_base_us"] = round(bench_us(base_blk), 2)
    t["blk_cand_us"] = round(bench_us(cand_blk), 2)
    t["sel_base_us"] = round(bench_us(base_sel), 2)
    cand_blk()  # the gather select reads the candidate's own published blocks
    t["sel_cand_us"] = round(bench_us(cand_sel), 2)
    t["total_base_us"] = round(t["blk_base_us"] + t["sel_base_us"], 2)
    t["total_cand_us"] = round(t["blk_cand_us"] + t["sel_cand_us"], 2)
    t["exact"] = exact
    return t


def stage_breakdown(tk, width, N, M):
    """Each kernel of tk.run timed alone (inputs populated by a full run first)."""
    scores = torch.randn((M, width), device="cuda", dtype=torch.bfloat16)
    lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
    active = torch.tensor([N], device="cuda", dtype=torch.int32)
    idx = torch.empty((M, 512), device="cuda", dtype=torch.int32)
    val = torch.empty((M, 512), device="cuda", dtype=torch.bfloat16)
    scratch = alloc_scratch(tk, M)
    rows = M
    full = bench_us(lambda: tk.run(scores, lengths, active, idx, val, scratch=scratch))
    out = {"segments": tk.segments, "levels": [[l, o] for _, l, o in tk.levels], "full_run_us": round(full, 2)}
    if tk.segments == 1:
        return out
    flat = list(scratch)
    buffers = ((flat[0], flat[1]), (flat[2], flat[3]))

    def lists_of(buffer, lists):
        return tuple(t[: rows * lists * tk.topk].view(rows, lists * tk.topk) for t in buffer)

    st = lambda: tk.stage1(scores, lengths, active, *lists_of(buffers[0], tk.segments))  # noqa: E731
    out["stage1_us"] = round(bench_us(st), 2)
    for level, (kernel, lists, outputs) in enumerate(tk.levels):
        work = lists_of(buffers[level % 2], lists)
        if outputs == 1:
            fn = lambda k=kernel, w=work: k(*w, lengths, active, idx, val)  # noqa: E731
        else:
            nxt = lists_of(buffers[(level + 1) % 2], outputs)
            fn = lambda k=kernel, w=work, n=nxt: k(*w, lengths, active, *n)  # noqa: E731
        out[f"merge{level + 1}_{lists}to{outputs}_us"] = round(bench_us(fn), 2)
    return out


def main():
    out_path = sys.argv[1]
    torch.manual_seed(20261109)
    res = {"cells": {}, "breakdown": {}}
    Ms_dec = [1, 6, 16, 48]

    def put(sel, N, M, cell):
        res["cells"].setdefault(sel, {})[f"N={N},M={M}"] = cell
        print(f"{sel} N={N} M={M}: {json.dumps(cell)}", flush=True)

    # 1. prefill_short of the dense layers (2, 8, 14): IndexerTopK(16384, 512, 1), rows <= 1024
    bt, ct = base_mod.IndexerTopK(16384, 512, 1), cand_mod.IndexerTopK(16384, 512, 1)
    for N in (2048, 8192, 16384):
        for M in (1, 6, 16, 48, 256, 1024):
            put("prefill_short_16k_k512", N, M, dense_cell(bt, ct, 16384, N, M))
    del bt, ct

    # 2. reindex layers (24, 28, 32, 36) at partial candidate lengths
    bt = base_mod.IndexerTopK(16384, 512, 1, candidates=True)
    ct = cand_mod.IndexerTopK(16384, 512, 1, candidates=True)
    for N in (2048, 8192):
        for M in (*Ms_dec, 256):
            put("reindex_16k_partial", N, M, reindex_cell(bt, ct, N, M))
    del bt, ct

    # 3. 1M shapes below the grid's 64K: dense layers' IndexerTopK(524288, 512, 1)
    bt, ct = base_mod.IndexerTopK(524288, 512, 1), cand_mod.IndexerTopK(524288, 512, 1)
    for N in (4096, 16384, 32768):
        for M in (*Ms_dec, 256):
            put("dense_524k_k512_small_n", N, M, dense_cell(bt, ct, 524288, N, M))
    del bt, ct

    # 4. 1M shapes below the grid's 64K: source layer 20
    bp = base_mod.IndexerV41Plan(1048576, 256, candidate_blocks=True)
    cp = cand_mod.IndexerV41Plan(1048576, 256, candidate_blocks=True)
    for N in (4096, 16384, 32768):
        for M in (*Ms_dec, 256):
            put("source_1m_small_n", N, M, source_cell(bp, cp, 1048576, N, M))
    del bp, cp
    torch.cuda.empty_cache()

    # 5. tp3-triangle-4k (max_model_len 262144): dense layers' IndexerTopK(131072, 512, 1)
    bt, ct = base_mod.IndexerTopK(131072, 512, 1), cand_mod.IndexerTopK(131072, 512, 1)
    for N in (16384, 65536, 131072):
        for M in (*Ms_dec, 256):
            put("dense_131k_k512", N, M, dense_cell(bt, ct, 131072, N, M))
    del bt, ct

    # 6. tp3-triangle-4k: source layer 20 at width 262144
    bp = base_mod.IndexerV41Plan(262144, 256, candidate_blocks=True)
    cp = cand_mod.IndexerV41Plan(262144, 256, candidate_blocks=True)
    for N in (16384, 65536, 262144):
        for M in (*Ms_dec, 256):
            put("source_262k", N, M, source_cell(bp, cp, 262144, N, M))
    del bp, cp
    torch.cuda.empty_cache()

    # 7. Diagnosis of the slower grid cell: per-kernel times at N=524288, M=1
    for name, mod in (("base", base_mod), ("cand", cand_mod)):
        for N in (524288, 1048576):
            res["breakdown"][f"raw_1m_k512_g1_{name}_N={N}_M=1"] = stage_breakdown(
                mod.IndexerTopK(1048576, 512, 1), 1048576, N, 1
            )
        res["breakdown"][f"dense_524k_k512_{name}_N=524288_M=1"] = stage_breakdown(
            mod.IndexerTopK(524288, 512, 1), 524288, 524288, 1
        )
    for k, v in res["breakdown"].items():
        print(f"breakdown {k}: {json.dumps(v)}", flush=True)

    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"SAVED {out_path}", flush=True)


if __name__ == "__main__":
    main()
