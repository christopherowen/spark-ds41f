"""Decode-tile sweep at the TP4 serving shapes (from the serving trace), three weight states.

warm: same weight every call; cold: distinct weights (DRAM); self: each call
follows the L2 prefetch of its own weight, as the NEXT window races the GEMM.
Whole calls (activation cast + GEMM) against B12X's block_fp8_linear. Each
height keeps the configuration whose worst ratio to B12X over the three
states is lowest; it then repeats a 120-round bit check against prefill.
"""
import concurrent.futures
import importlib.util
import itertools
import json
import sys

import tile_kernels
import torch
from b12x.gemm import block_fp8_linear
from vllm.models.deepseek_v4_1.l2_prefetch import object_segments
from vllm.models.glm5next.nvidia import l2_prefetch as L

spec = importlib.util.spec_from_file_location("g", "/b/gemm.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
DEV = torch.device("cuda")
GEN = torch.Generator(device="cuda").manual_seed(7)
PF = L.L2Prefetcher.get(DEV)
SHAPES = {  # (N, K) per rank at TP4: calls per step
    "q_b": (8192, 1280), "indexer_q_b": (4096, 1280), "q_a_kv": (1792, 5120),
    "shared_gate_up": (1152, 5120), "shared_down": (5120, 576), "draft_main": (6400, 6144),
}
HEIGHTS = {16: 6, 32: 24, 64: 48}
only = [a for a in sys.argv[1:] if a in SHAPES]


def fp8(shape):
    return (torch.randn(shape, device=DEV, generator=GEN) * 0.5).to(torch.float8_e4m3fn)


def sfw(rows, K):
    return g.pack_scale_words(torch.randint(124, 131, (rows, K // 32), device=DEV, generator=GEN, dtype=torch.uint8))


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


def block_Ks(K):
    return [bk for bk in (64, 128, 192, 256) if K % bk == 0 and (bk >= 128 or K % 128)]


summary, failures = {}, []
for name, (N, K) in SHAPES.items():
    if only and name not in only:
        continue
    copies = max(4, -(-160 * 2**20 // (N * K)))
    ws_ = [fp8((N, K)) for _ in range(copies)]
    block_scales = [torch.randint(124, 131, (N // 32, K // 32), device=DEV, generator=GEN, dtype=torch.uint8) for _ in range(copies)]
    wsf_ = [g.pack_scale_words(s, rows=N) for s in block_scales]
    packed = [block_fp8_linear.pack_weight(w, s.view(torch.float8_e8m0fnu), block_size=(32, 32)) for w, s in zip(ws_, block_scales)]
    tl_plans = [L.make_plan([L.tensor_segment("w", w), L.tensor_segment("sf", s, min_bytes=0)], 20 * 2**20, DEV)[0] for w, s in zip(ws_, wsf_)]
    b_plans = [L.make_plan(object_segments("b12x_weight", p), 20 * 2**20, DEV)[0] for p in packed]
    xq = torch.empty(g.DECODE_ROWS, K, dtype=torch.float8_e4m3fn, device=DEV)
    sf = torch.empty(g.DECODE_ROWS, 4 * g.scale_words(K), dtype=torch.uint8, device=DEV)
    prefill = g.mxfp8_gemm(N, K, **g.fp8_prefill_config(N, K))
    summary[name] = {}
    for block_M, rows in HEIGHTS.items():
        configs = [dict(block_M=block_M, block_N=n, block_K=bk, num_stages=s, threads=t)
                   for n, bk, s, t in itertools.product((32, 64, 128), block_Ks(K), (2, 3, 4, 6), (64, 128))
                   if N % n == 0 and s * (block_M + n) * bk <= 92 * 1024 and not (block_M == 64 and n == 32)]

        def build(cfg):
            try:
                return g.mxfp8_gemm_decode(N, K, **cfg, padded_rows=True)
            except Exception as error:
                return error

        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            kernels = list(pool.map(build, configs))
        x = (torch.randn(rows, K, device=DEV, generator=GEN) * 0.5).bfloat16()
        plan = block_fp8_linear.plan(block_fp8_linear.Caps(device=DEV, max_tokens=rows, in_features=K, out_features=N, block_size=(32, 32)))
        scratch = [torch.empty(s.shape, dtype=s.dtype, device=DEV) for s in plan.scratch_specs()]
        bout = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)
        bind = [block_fp8_linear.bind(plan, scratch=scratch, source=x, packed_weight=packed[i], output=bout.view(rows, N, 1)) for i in range(copies)]
        out = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)

        def measure(run_one, prefetch_plans):
            warm = timed([lambda: run_one(0)] * 16)
            cold = timed([(lambda i=i: run_one(i % copies)) for i in range(2 * copies)])

            def selfcall(i):
                def run():
                    PF.issue(prefetch_plans[i % copies], rows)
                    run_one(i % copies)
                return run
            return warm, cold, timed([selfcall(i) for i in range(2 * copies)])

        base = measure(lambda i: block_fp8_linear.run(binding=bind[i]), b_plans)
        results = {}
        for cfg, kernel in zip(configs, kernels):
            if isinstance(kernel, Exception):
                continue

            def run_one(i, kernel=kernel):
                tile_kernels.quant.per_token_cast(x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True, out=(xq[:rows], sf[:rows]))
                kernel(xq, ws_[i], sf.view(torch.uint32), wsf_[i], out)
            try:
                results["n{block_N}-k{block_K}-st{num_stages}-t{threads}".format(**cfg)] = (cfg, measure(run_one, tl_plans))
            except Exception as error:
                print(f"  {name} m{block_M} {cfg}: {type(error).__name__}: {str(error)[:80]}")
                torch.cuda.synchronize()

        def worst(m):
            return max(a / b for a, b in zip(m, base))
        ranked = sorted(results, key=lambda k: (worst(results[k][1]), sum(results[k][1])))
        print(f"{name} m{block_M} {rows} rows (warm/cold/self us): B12X {base[0]:.1f}/{base[1]:.1f}/{base[2]:.1f}; " + ", ".join(
            f"{k} {results[k][1][0]:.1f}/{results[k][1][1]:.1f}/{results[k][1][2]:.1f}" for k in ranked[:5]), flush=True)
        best = ranked[0]
        cfg = results[best][0]
        kernel = kernels[configs.index(cfg)]
        bad = 0
        for round_ in range(120):
            r = (rows, block_M, max(1, block_M // 2 - 1))[round_ % 3]
            a, asf = fp8((g.DECODE_ROWS, K)), sfw(g.DECODE_ROWS, K)
            w = round_ % copies
            o = torch.empty(r, N, dtype=torch.bfloat16, device=DEV)
            kernel(a, ws_[w], asf, wsf_[w], o)
            big = torch.empty(80, N, dtype=torch.bfloat16, device=DEV)
            prefill(torch.cat([a[:r], fp8((80 - r, K))]), ws_[w], torch.cat([asf[:r], sfw(80 - r, K)]), wsf_[w], big)
            bad += not torch.equal(o.view(torch.int16), big[:r].view(torch.int16))
        if bad:
            failures.append(f"{name} m{block_M} {best}")
        m = results[best][1]
        print(f"BEST {name} m{block_M}: {best} {m[0]:.1f}/{m[1]:.1f}/{m[2]:.1f} vs B12X {base[0]:.1f}/{base[1]:.1f}/{base[2]:.1f} "
              f"worst ratio {worst(m):.3f}; bits {120 - bad}/120 {'PASS' if not bad else 'FAIL'}", flush=True)
        summary[name][block_M] = {"rows": rows, "best": cfg, "tilelang": m, "b12x": base, "bits_equal": 120 - bad}
print(json.dumps({"summary": summary, "failures": failures}))
