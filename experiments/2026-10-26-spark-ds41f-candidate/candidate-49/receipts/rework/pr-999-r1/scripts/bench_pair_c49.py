#!/usr/bin/env python3
"""Paired 1107 / corrected (image module, cand) / candidate-49 (c49) timings of the
full-extent served selectors, in one process: Layer 20's select_blocks
IndexerTopK(1048576, 2048, 8) and its gather select, the dense layers'
IndexerTopK(524288, 512, 1), and the reindex IndexerTopK(16384, 512, 1, candidates).
Run through warm_wrap.py. Every arm is checked against 1107 with torch.equal.

usage: bench_pair_c49.py <out.json> c49=/path/indexer.py
"""
import json
import sys

import torch

sys.path.insert(0, "/tmp")
import bench_2boot as b  # noqa: E402

out_path = sys.argv[1]
arms = {"base": b.base_mod, "cand": b.cand_mod}
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1)
    arms[name] = b.load_module(path, f"vllm.models.deepseek_v4_1.tilelang.indexer_{name}")
bench_us, alloc_scratch = b.bench_us, b.alloc_scratch
CB = b.cand_mod.CANDIDATE_BLOCKS


def main():
    torch.manual_seed(20261112)
    res = {"arms": list(arms), "cells": {}}

    def put(sel, key, c):
        res["cells"].setdefault(sel, {})[key] = c
        print(f"{sel} {key}: {json.dumps(c)}", flush=True)

    plans = {n: m.IndexerV41Plan(1048576, 256, candidate_blocks=True) for n, m in arms.items()}
    for N in (262144, 1048576):
        for M in (1, 16, 32, 64, 256):
            scores = torch.randn((M, 1048576), device="cuda", dtype=torch.bfloat16)
            lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
            active = torch.tensor([N], device="cuda", dtype=torch.int32)
            blk, sel, ref, exact = {}, {}, None, True
            for n, p in plans.items():
                i2k = torch.empty((M, CB), device="cuda", dtype=torch.int32)
                v2k = torch.empty((M, CB), device="cuda", dtype=torch.bfloat16)
                c2k = torch.empty((M,), device="cuda", dtype=torch.int32)
                i512 = torch.empty((M, 512), device="cuda", dtype=torch.int32)
                v512 = torch.empty((M, 512), device="cuda", dtype=torch.bfloat16)
                bs, ss = alloc_scratch(p.select_blocks, M), alloc_scratch(p.select, M)
                fb = lambda p=p, i2k=i2k, v2k=v2k, c2k=c2k, bs=bs: p.select_blocks.run(scores, lengths, active, i2k, v2k, counts=c2k, scratch=bs)  # noqa: E731
                if getattr(p.select, "gather_scores", False):
                    fs = lambda p=p, i2k=i2k, c2k=c2k, i512=i512, v512=v512, ss=ss: p.select.run(scores, c2k, None, i512, v512, gather=i2k, scratch=ss)  # noqa: E731
                else:
                    fs = lambda p=p, i512=i512, v512=v512, ss=ss: p.select.run(scores, lengths, active, i512, v512, scratch=ss)  # noqa: E731
                fb()
                fs()
                got = (i2k.clone(), c2k.clone(), i512.clone(), v512.clone())
                if ref is None:
                    ref = got
                else:
                    exact &= all(torch.equal(a, c) for a, c in zip(ref, got))
                blk[f"{n}_us"] = round(bench_us(fb), 2)
                sel[f"{n}_us"] = round(bench_us(fs), 2)
            put("src_1m_select_blocks_k2048_g8", f"N={N},M={M}", {**blk, "exact": bool(exact)})
            put("src_1m_select_k512", f"N={N},M={M}", {**sel, "exact": bool(exact)})
            del scores
            torch.cuda.empty_cache()
    del plans
    tks = {n: m.IndexerTopK(524288, 512, 1) for n, m in arms.items()}
    for N in (131072, 524288):
        for M in (1, 16, 64, 256):
            scores = torch.randn((M, 524288), device="cuda", dtype=torch.bfloat16)
            lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
            active = torch.tensor([N], device="cuda", dtype=torch.int32)
            c, ref, exact = {}, None, True
            for n, tk in tks.items():
                idx = torch.empty((M, 512), device="cuda", dtype=torch.int32)
                val = torch.empty((M, 512), device="cuda", dtype=torch.bfloat16)
                scr = alloc_scratch(tk, M)
                fn = lambda tk=tk, idx=idx, val=val, scr=scr: tk.run(scores, lengths, active, idx, val, scratch=scr)  # noqa: E731
                fn()
                if ref is None:
                    ref = (idx.clone(), val.clone())
                else:
                    exact &= torch.equal(ref[0], idx) and torch.equal(ref[1], val)
                c[f"{n}_us"] = round(bench_us(fn), 2)
            put("dense_524k_k512", f"N={N},M={M}", {**c, "exact": bool(exact)})
    del tks
    tks = {n: m.IndexerTopK(16384, 512, 1, candidates=True) for n, m in arms.items()}
    for M in (1, 16, 64, 256, 1024):
        scores = torch.randn((M, 16384), device="cuda", dtype=torch.bfloat16)
        gather = torch.arange(CB, device="cuda", dtype=torch.int32).unsqueeze(0).expand(M, -1).contiguous()
        lengths = torch.full((M,), 16384, device="cuda", dtype=torch.int32)
        c, ref, exact = {}, None, True
        for n, tk in tks.items():
            idx = torch.empty((M, 512), device="cuda", dtype=torch.int32)
            val = torch.empty((M, 512), device="cuda", dtype=torch.bfloat16)
            scr = alloc_scratch(tk, M)
            fn = lambda tk=tk, idx=idx, val=val, scr=scr: tk.run(scores, lengths, None, idx, val, gather=gather, scratch=scr)  # noqa: E731
            fn()
            if ref is None:
                ref = (idx.clone(), val.clone())
            else:
                exact &= torch.equal(ref[0], idx) and torch.equal(ref[1], val)
            c[f"{n}_us"] = round(bench_us(fn), 2)
        put("reindex_16k", f"N=16384,M={M}", {**c, "exact": bool(exact)})
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"SAVED {out_path}", flush=True)


if __name__ == "__main__":
    main()
