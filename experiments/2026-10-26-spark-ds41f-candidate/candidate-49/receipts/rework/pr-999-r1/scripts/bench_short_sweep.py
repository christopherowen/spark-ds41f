#!/usr/bin/env python3
"""Short-extent sweep of the served prefill_short selector IndexerTopK(16384, 512, 1)
(dense layers 2, 8, 14; rows <= 1024) and of the other single-segment / one-merge
selectors at short extents, for 1107 (base), the candidate-49 image (cand) and
optional variant modules.

Methodology is bench_2boot.py's (imported: bench_us = 8 warmups, 25 timed calls,
10% trimmed mean; base = /tmp_host/indexer_baseline.py). Each cell checks every
arm's indices and values against base with torch.equal.

usage: bench_short_sweep.py <out.json> [name=/path/to/indexer.py ...]
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


def run_cell(tks, width, lengths, M, active_n, gather=None):
    scores = torch.randn((M, width), device="cuda", dtype=torch.bfloat16)
    active = torch.tensor([active_n], device="cuda", dtype=torch.int32)
    out = {}
    ref = None
    exact = True
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
    torch.manual_seed(20261110)
    res = {"arms": list(arms), "cells": {}}

    def put(sel, key, cell):
        res["cells"].setdefault(sel, {})[key] = cell
        print(f"{sel} {key}: {json.dumps(cell)}", flush=True)

    # prefill_short: IndexerTopK(16384, 512, 1), uniform extents
    tks = {n: m.IndexerTopK(16384, 512, 1) for n, m in arms.items()}
    for N in (520, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384):
        for M in (64, 256, 512, 1024):
            lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
            put("prefill_short_16k_k512", f"N={N},M={M}", run_cell(tks, 16384, lengths, M, 16384))
    # prefill_short: causal 1024-row chunks of a P-token prompt (compress ratio 2:
    # a row at position p sees (p + 1) // 2 index entries)
    for P in (2048, 4096, 8192, 16384, 32768):
        for start in range(0, P, 1024):
            M = min(1024, P - start)
            pos = torch.arange(start, start + M, device="cuda", dtype=torch.int32)
            lengths = ((pos + 1) // 2).clamp(min=1)
            put("prefill_short_causal_chunk", f"P={P},start={start},M={M}", run_cell(tks, 16384, lengths, M, 16384))
    del tks
    # reindex IndexerTopK(16384, 512, 1, candidates=True) at partial lengths, rows <= 256
    tks = {n: m.IndexerTopK(16384, 512, 1, candidates=True) for n, m in arms.items()}
    for N in (1024, 2048, 4096, 8192):
        for M in (64, 256):
            lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
            gather = torch.arange(CB, device="cuda", dtype=torch.int32).unsqueeze(0).expand(M, -1).contiguous()
            put("reindex_16k_partial", f"N={N},M={M}", run_cell(tks, 16384, lengths, M, 16384, gather=gather))
    del tks
    # dense layers' IndexerTopK(524288, 512, 1) at short extents, prefill rows <= 256
    tks = {n: m.IndexerTopK(524288, 512, 1) for n, m in arms.items()}
    for N in (2048, 4096, 16384, 32768, 65536):
        for M in (64, 256):
            lengths = torch.full((M,), N, device="cuda", dtype=torch.int32)
            put("dense_524k_k512", f"N={N},M={M}", run_cell(tks, 524288, lengths, M, N))
    del tks
    torch.cuda.empty_cache()
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"SAVED {out_path}", flush=True)


if __name__ == "__main__":
    main()
