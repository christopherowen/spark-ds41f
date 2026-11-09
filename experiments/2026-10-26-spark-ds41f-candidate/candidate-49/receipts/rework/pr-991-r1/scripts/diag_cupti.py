#!/usr/bin/env python3
"""GPU kernel durations (CUPTI via torch.profiler) of the #991 variants.

``sw``: 1107 (the PR bench's re-declared kernels); ``module``: the image's kernel, or the
bind-mounted gemm.py/indexer.py; ``hw_serial``: diag_variants' hardware conversion with
1107's T.serial epilogue loop.

For each variant and M: 3 alternating passes of 100 launches, each launch
followed by a synchronize (isolated: no PDL overlap, no queueing), the
kernel's own start-to-end time read from the CUPTI activity records.

usage: diag_cupti.py --out /out/x.json
"""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

spec = importlib.util.spec_from_file_location("diag", "/tmp/diag_variants.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
bench = diag.bench


def durations(fn, n):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
            torch.cuda.synchronize()
    ds = [e.device_time for e in prof.events() if e.device_type.name == "CUDA" and "kernel" in e.name]
    return ds


def summarize(ds):
    ds = sorted(ds)
    return {"n": len(ds), "median_us": round(statistics.median(ds), 3), "q25_us": round(ds[len(ds) // 4], 3),
            "q75_us": round(ds[3 * len(ds) // 4], 3), "min_us": round(ds[0], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(20261108)
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gemm": [], "paged": []}

    n, k, hd, rd = 4096, 1280, 128, 64
    sf_words = diag.scale_words(k)
    b = torch.randn((n, k), generator=gen, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    sfb = torch.randint(120, 126, (n // 32, sf_words * 4), generator=gen, device=dev,
                        dtype=torch.int32).to(torch.uint8).view(torch.uint32)
    cos_sin = torch.randn((16384, rd), generator=gen, device=dev, dtype=torch.float32)
    b_tc = b.view(n * (k // diag.TILE_WEIGHT_K), diag.TILE_WEIGHT_K)
    gk = {}
    for bm in (16, 32, 64):
        gk[("sw", bm)] = bench._sw_mxfp8_gemm_rope_mxfp4_decode(n, k, hd, rd, block_M=bm, padded_rows=True,
                                                                tile_contiguous=True)
        gk[("module", bm)] = diag.mxfp8_gemm_rope_mxfp4_decode(n, k, hd, rd, block_M=bm, padded_rows=True,
                                                           tile_contiguous=True)
        gk[("hw_serial", bm)] = diag.gemm_variant(n, k, "hw", False, hd, rd, block_M=bm)
    bench._warmup_gpu_clock(dev)
    for M in (1, 2, 4, 8, 16, 32, 64):
        bm = 16 if M <= 16 else (32 if M <= 32 else 64)
        a = torch.randn((bm, k), generator=gen, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        sfa = torch.randint(120, 126, (bm, sf_words * 4), generator=gen, device=dev,
                            dtype=torch.int32).to(torch.uint8).view(torch.uint32)
        pos = torch.randint(0, 16384, (M,), generator=gen, device=dev, dtype=torch.int64)
        q = torch.empty((M, n // 2), dtype=torch.uint8, device=dev)
        s = torch.empty((M, n // 32), dtype=torch.uint8, device=dev)
        acc = {m: [] for m in ("sw", "module", "hw_serial")}
        for _ in range(3):
            for m in acc:
                acc[m] += durations(lambda m=m: gk[(m, bm)](a, b_tc, sfa, sfb, pos, cos_sin, q, s), args.n)
        row = {"M": M, "block_M": bm, **{m: summarize(v) for m, v in acc.items()}}
        out["gemm"].append(row)
        print("gemm", M, {m: row[m]["median_us"] for m in acc}, flush=True)

    pk = {"sw": bench._sw_mxfp4_quantize_v41(paged=True, page_size=64),
          "module": diag.mxfp4_quantize_v41(paged=True, page_size=64)}
    pool = torch.zeros((256, diag.page_bytes(64)), dtype=torch.uint8, device=dev)
    bench._warmup_gpu_clock(dev)
    for M in (1, 2, 4, 6, 8, 16, 32, 512, 2048, 8192):
        x = torch.randn((M, diag.INDEX_DIM), generator=gen, device=dev, dtype=torch.bfloat16)
        slots = torch.arange(M, device=dev, dtype=torch.int64)
        acc = {m: [] for m in pk}
        for _ in range(3):
            for m in acc:
                acc[m] += durations(lambda m=m: pk[m](x, slots, pool), args.n)
        row = {"M": M, **{m: summarize(v) for m, v in acc.items()}}
        out["paged"].append(row)
        print("paged", M, {m: row[m]["median_us"] for m in acc}, flush=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
