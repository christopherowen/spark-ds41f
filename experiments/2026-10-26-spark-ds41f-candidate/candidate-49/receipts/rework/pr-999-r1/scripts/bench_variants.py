#!/usr/bin/env python3
"""Diagnostic: 1107 (base), candidate-49 (cand) and variant indexer modules on the
regime where candidate-49's prefill_short selector loses (short extents, many rows),
plus full-extent cells where #999's gains must be kept. bench_2boot.py methodology
(imported); every arm is checked against base with torch.equal.

usage: bench_variants.py <out.json> name=/path/indexer.py ...
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


def cell(tks, width, lengths, M, active_n, gather=None):
    scores = torch.randn((M, width), device="cuda", dtype=torch.bfloat16)
    active = torch.tensor([active_n], device="cuda", dtype=torch.int32)
    out, ref, exact = {}, None, True
    for name, tk in tks.items():
        idx = torch.empty((M, 512), device="cuda", dtype=torch.int32)
        val = torch.empty((M, 512), device="cuda", dtype=torch.bfloat16)
        scr = alloc_scratch(tk, M)
        if gather is None:
            fn = lambda tk=tk, idx=idx, val=val, scr=scr: tk.run(scores, lengths, active, idx, val, scratch=scr)  # noqa: E731
        else:
            fn = lambda tk=tk, idx=idx, val=val, scr=scr: tk.run(scores, lengths, None, idx, val, gather=gather, scratch=scr)  # noqa: E731
        fn()
        if ref is None:
            ref = (idx.clone(), val.clone())
        else:
            exact &= torch.equal(ref[0], idx) and torch.equal(ref[1], val)
        out[f"{name}_us"] = round(bench_us(fn), 2)
    out["exact"] = bool(exact)
    return out


def main():
    torch.manual_seed(20261111)
    res = {"arms": list(arms), "cells": {}}

    def put(sel, key, c):
        res["cells"].setdefault(sel, {})[key] = c
        print(f"{sel} {key}: {json.dumps(c)}", flush=True)

    tks = {n: m.IndexerTopK(16384, 512, 1) for n, m in arms.items()}
    for N in (1024, 2048, 3072, 4096, 6144, 16384):
        for M in (1, 256, 512, 1024):
            lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
            put("prefill_short_16k_k512", f"N={N},M={M}", cell(tks, 16384, lengths, M, 16384))
    for start in range(0, 8192, 1024):
        pos = torch.arange(start, start + 1024, device="cuda", dtype=torch.int32)
        lengths = ((pos + 1) // 2).clamp(min=1)
        put("prefill_short_causal_chunk", f"P=8192,start={start},M=1024", cell(tks, 16384, lengths, 1024, 16384))
    del tks
    tks = {n: m.IndexerTopK(16384, 512, 1, candidates=True) for n, m in arms.items()}
    for N, M in ((4096, 256), (16384, 1), (16384, 64), (16384, 1024)):
        lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
        gather = torch.arange(CB, device="cuda", dtype=torch.int32).unsqueeze(0).expand(M, -1).contiguous()
        put("reindex_16k", f"N={N},M={M}", cell(tks, 16384, lengths, M, 16384, gather=gather))
    del tks
    tks = {n: m.IndexerTopK(524288, 512, 1) for n, m in arms.items()}
    for N, M in ((4096, 256), (524288, 1), (524288, 64), (524288, 256)):
        lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
        put("dense_524k_k512", f"N={N},M={M}", cell(tks, 524288, lengths, M, N))
    del tks
    torch.cuda.empty_cache()
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"SAVED {out_path}", flush=True)


if __name__ == "__main__":
    main()
