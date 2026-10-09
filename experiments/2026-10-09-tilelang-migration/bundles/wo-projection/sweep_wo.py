"""Decode-tile sweep for the WO projection's GEMMs at TP4: the grouped WO-A (2 groups of
1024 x 4096) and WO-B (5120 x 2048), after the context-KV sweep. B12X fuses both into
one call, so tiles are ranked against the current default tile instead: warm (same
weight), cold (distinct weights) and self (each call follows the L2 prefetch of its own
weight), whole calls with the activation cast. Each height keeps the tile with the
lowest worst ratio to the default, then repeats a 120-round bit check against the
prefill GEMM. Exits 1 only if a winner's bits differ from prefill.
"""
import concurrent.futures
import itertools
import json

import tile_kernels
import torch

from vllm.models.deepseek_v4_1.tilelang import gemm as g
from vllm.models.glm5next.nvidia import l2_prefetch as L

DEV = torch.device("cuda")
GEN = torch.Generator(device="cuda").manual_seed(13)
PF = L.L2Prefetcher.get(DEV)
SHAPES = {"wo_a": (1024, 4096, 2), "wo_b": (5120, 2048, 1)}  # (N per group, K per group, groups)
HEIGHTS = {16: 6, 32: 24, 64: 48}


def fp8(shape):
    return (torch.randn(shape, device=DEV, generator=GEN) * 0.5).to(torch.float8_e4m3fn)


def sfw(rows, k):
    return g.pack_scale_words(torch.randint(124, 131, (rows, k // 32), device=DEV, generator=GEN, dtype=torch.uint8))


def timed(calls, rounds=10):
    for c in calls:
        c()
    PF.join()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
        PF.join()
    graph.replay()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000 / (rounds * len(calls))


def name_of(cfg):
    return "n{block_N}-k{block_K}-st{num_stages}-t{threads}".format(**cfg)


summary, failures = {}, []
for shape, (N, K, groups) in SHAPES.items():
    total_N, total_K = N * groups, K * groups
    copies = max(4, -(-160 * 2**20 // (total_N * K)))
    ws_ = [fp8((total_N, K)) for _ in range(copies)]
    wsf_ = [sfw(total_N, K) for _ in range(copies)]
    plans = [L.make_plan([L.tensor_segment("w", w), L.tensor_segment("sf", s, min_bytes=0)], 20 * 2**20, DEV)[0]
             for w, s in zip(ws_, wsf_)]
    xq = torch.empty(g.DECODE_ROWS, total_K, dtype=torch.float8_e4m3fn, device=DEV)
    sf = torch.empty(g.DECODE_ROWS, 4 * g.scale_words(total_K), dtype=torch.uint8, device=DEV)
    prefill = g.mxfp8_gemm(N, K, **g.fp8_prefill_config(N, K, groups), groups=groups)
    summary[shape] = {}
    for block_M, rows in HEIGHTS.items():
        configs = [dict(block_M=block_M, block_N=n, block_K=bk, num_stages=s, threads=t)
                   for n, bk, s, t in itertools.product((32, 64, 128), (128, 256), (2, 3, 4, 6), (64, 128))
                   if N % n == 0 and K % bk == 0 and s * (block_M + n) * bk <= 92 * 1024
                   and not (block_M == 64 and n == 32)]
        default = g.fp8_decode_config(N, K, block_M, groups)
        if default not in configs:
            configs.append(default)

        def build(cfg):
            try:
                return g.mxfp8_gemm_decode(N, K, **cfg, padded_rows=True, groups=groups)
            except Exception as error:
                return error

        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            kernels = list(pool.map(build, configs))
        x = (torch.randn(rows, total_K, device=DEV, generator=GEN) * 0.5).bfloat16()
        out = torch.empty(rows, total_N, dtype=torch.bfloat16, device=DEV)

        def measure(run_one):
            warm = timed([lambda: run_one(0)] * 16)
            cold = timed([(lambda i=i: run_one(i % copies)) for i in range(2 * copies)])

            def selfcall(i):
                def run():
                    PF.issue(plans[i % copies], rows)
                    run_one(i % copies)
                return run
            return warm, cold, timed([selfcall(i) for i in range(2 * copies)])

        results = {}
        for cfg, kernel in zip(configs, kernels):
            if isinstance(kernel, Exception):
                continue

            def run_one(i, kernel=kernel):
                tile_kernels.quant.per_token_cast(x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True,
                                                  out=(xq[:rows], sf[:rows]))
                kernel(xq, ws_[i], sf.view(torch.uint32), wsf_[i], out)
            try:
                results[name_of(cfg)] = (cfg, measure(run_one))
            except Exception as error:
                print(f"  {shape} m{block_M} {cfg}: {type(error).__name__}: {str(error)[:80]}")
                torch.cuda.synchronize()
        base = results[name_of(default)][1]

        def worst(m):
            return max(a / b for a, b in zip(m, base))
        ranked = sorted(results, key=lambda k: (worst(results[k][1]), sum(results[k][1])))
        print(f"{shape} m{block_M} {rows} rows (warm/cold/self us): default {name_of(default)} "
              f"{base[0]:.1f}/{base[1]:.1f}/{base[2]:.1f}; "
              + ", ".join(f"{k} {results[k][1][0]:.1f}/{results[k][1][1]:.1f}/{results[k][1][2]:.1f}"
                          for k in ranked[:5]), flush=True)
        best = ranked[0]
        cfg = results[best][0]
        kernel = kernels[configs.index(cfg)]
        bad = 0
        for round_ in range(120):
            r = (rows, block_M, max(1, block_M // 2 - 1))[round_ % 3]
            a, asf = fp8((g.DECODE_ROWS, total_K)), sfw(g.DECODE_ROWS, total_K)
            w = round_ % copies
            o = torch.empty(r, total_N, dtype=torch.bfloat16, device=DEV)
            kernel(a, ws_[w], asf, wsf_[w], o)
            big = torch.empty(80, total_N, dtype=torch.bfloat16, device=DEV)
            prefill(torch.cat([a[:r], fp8((80 - r, total_K))]), ws_[w], torch.cat([asf[:r], sfw(80 - r, total_K)]),
                    wsf_[w], big)
            bad += not torch.equal(o.view(torch.int16), big[:r].view(torch.int16))
        if bad:
            failures.append(f"{shape} m{block_M} {best}")
        m = results[best][1]
        print(f"BEST {shape} m{block_M}: {best} {m[0]:.1f}/{m[1]:.1f}/{m[2]:.1f} vs default {base[0]:.1f}/"
              f"{base[1]:.1f}/{base[2]:.1f} worst ratio {worst(m):.3f}; bits {120 - bad}/120 "
              f"{'PASS' if not bad else 'FAIL'}", flush=True)
        summary[shape][block_M] = {"rows": rows, "best": cfg, "tilelang": m, "default": base, "bits_equal": 120 - bad}
print(json.dumps({"summary": summary, "failures": failures}))
raise SystemExit(1 if failures else 0)
