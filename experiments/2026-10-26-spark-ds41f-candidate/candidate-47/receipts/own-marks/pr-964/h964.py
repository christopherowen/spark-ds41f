#!/usr/bin/env python3
"""Own-mark harness for PR #964 (ours, not the PR's): the image's SparseMlaV41Plan
and the PR bench's embedded stage-1 candidate kernel on the bench's workload
(swa=128 / idx=512 and idx=64, heads 24 and 16, M=1..48), with output hashes for
cross-image bit-exactness and CUDA-graph replay timing interleaved window by window.

Arms: image (the image's sparse_mla_fwd_v41: 1107's or candidate-47's 0769) and
bench_s1 (make_candidate_sparse_mla(fix_s2=False) from the PR bench).
usage: h964.py LABEL   (writes /out/h964-LABEL.json)
"""

import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time

import torch

spec = importlib.util.spec_from_file_location("b964", "/bench/bench_sparse_mla_pipeline.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
orig_mod = b.orig_mod

LABEL = sys.argv[1]
DEV = "cuda"
WINDOWS = 25


def summ(xs):
    xs = list(xs)
    n = len(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    return {"median": round(statistics.median(xs), 3), "mean": round(statistics.fmean(xs), 3),
            "ci95": round(1.96 * sd / math.sqrt(n), 3), "n": n}


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def main():
    out = {"label": LABEL, "device": torch.cuda.get_device_name(0), "windows": WINDOWS,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shapes": {}}
    g = torch.Generator(device=DEV).manual_seed(964)
    swa_width, idx_width, idx_page_size, swa_page_size = 128, 512, 64, 64
    swa_pool = idx_pool = idx_pages = 32
    max_rows = 64
    SwaRecords = torch.randint(0, 120, (swa_pool, swa_page_size * b.SWA_RECORD_BYTES), generator=g, dtype=torch.uint8, device=DEV)
    IdxRecords = torch.randint(0, 120, (idx_pool, idx_page_size * b.IDX_RECORD_BYTES), generator=g, dtype=torch.uint8, device=DEV)
    CosSin = torch.randn((8192, b.ROPE_DIM), generator=g, dtype=torch.float32, device=DEV)
    fn_s1 = b.make_candidate_sparse_mla(fix_s2=False)
    image_fn = orig_mod.sparse_mla_fwd_v41
    for heads, tp_name in [(24, "TP3_h24"), (16, "TP4_h16")]:
        plans = {"image": orig_mod.SparseMlaV41Plan(heads, swa_width, idx_width, idx_page_size,
                                                    swa_page_size=swa_page_size, max_rows=max_rows)}
        orig_mod.sparse_mla_fwd_v41 = fn_s1
        try:
            plans["bench_s1"] = orig_mod.SparseMlaV41Plan(heads, swa_width, idx_width, idx_page_size,
                                                          swa_page_size=swa_page_size, max_rows=max_rows)
        finally:
            orig_mod.sparse_mla_fwd_v41 = image_fn
        Sinks = torch.randn((heads,), generator=g, dtype=torch.float32, device=DEV) * 0.1
        out["shapes"][tp_name] = {}
        for ctx_name, swa_len, idx_len in [("med_ctx_128_512", 128, 512), ("short_ctx_128_64", 128, 64)]:
            rows = {}
            for M in [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]:
                Q = torch.randn((M, heads, b.HEAD_DIM), generator=g, dtype=torch.bfloat16, device=DEV)
                SwaIndices = torch.randint(0, swa_pool, (M, swa_width), generator=g, dtype=torch.int32, device=DEV)
                SwaLengths = torch.full((M,), swa_len, dtype=torch.int32, device=DEV)
                IdxIndices = torch.randint(0, idx_pages * idx_page_size, (M, idx_width), generator=g, dtype=torch.int32, device=DEV)
                IdxLengths = torch.full((M,), idx_len, dtype=torch.int32, device=DEV)
                IdxPageTable = torch.randint(0, idx_pool, (M, idx_pages), generator=g, dtype=torch.int32, device=DEV)
                Positions = torch.arange(100, 100 + M, dtype=torch.int64, device=DEV)
                route = plans["image"].kernel_for(M)
                kw = dict(idx_cache=IdxRecords, idx_indices=IdxIndices, idx_lengths=IdxLengths,
                          idx_page_table=IdxPageTable, kernel=route)
                outs, calls, graphs = {}, {}, {}
                for name, plan in plans.items():
                    o = torch.empty((M, heads, b.HEAD_DIM), dtype=torch.bfloat16, device=DEV)
                    sc = [torch.empty(s, dtype=dt, device=DEV) for s, dt in plan.scratch_specs(M, route)]

                    def call(plan=plan, o=o, sc=sc):
                        plan.run(Q, SwaRecords, SwaIndices, SwaLengths, Sinks, Positions, CosSin, o, scratch=sc, **kw)

                    call()
                    torch.cuda.synchronize()
                    outs[name] = o
                    calls[name] = call
                for name, call in calls.items():
                    for _ in range(20):
                        call()
                    torch.cuda.synchronize()
                    gg = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gg):
                        for _ in range(10):
                            call()
                    for _ in range(5):
                        gg.replay()
                    graphs[name] = gg
                torch.cuda.synchronize()
                samples = {name: [] for name in plans}
                for _ in range(WINDOWS):
                    for name, gg in graphs.items():
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record()
                        for _ in range(10):
                            gg.replay()
                        e.record()
                        torch.cuda.synchronize()
                        samples[name].append(s.elapsed_time(e) * 1e3 / 100)
                rows[str(M)] = {
                    "route": route,
                    "sha256_16": {n: sha(o) for n, o in outs.items()},
                    "image_equal_bench_s1": bool(torch.equal(outs["image"], outs["bench_s1"])),
                    "graph_us": {n: summ(v) for n, v in samples.items()},
                    "paired_bench_s1_minus_image_us": summ([a - c for a, c in zip(samples["bench_s1"], samples["image"])]),
                }
                print(tp_name, ctx_name, M, route, rows[str(M)]["graph_us"]["image"]["median"],
                      rows[str(M)]["graph_us"]["bench_s1"]["median"], rows[str(M)]["image_equal_bench_s1"], flush=True)
                del graphs
            out["shapes"][tp_name][ctx_name] = rows
    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"/out/h964-{LABEL}.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", f"/out/h964-{LABEL}.json")


if __name__ == "__main__":
    main()
