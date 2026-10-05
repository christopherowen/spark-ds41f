"""Draft main decode tiles (no-WS, K blocks of 128 or 256) vs B12X with bit checks; prefill raster swizzle."""
import importlib.util

import torch

spec = importlib.util.spec_from_file_location("g", "/b/gemm.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
DEV = torch.device("cuda")
GEN = torch.Generator(device="cuda").manual_seed(4)


def fp8(shape):
    return (torch.randn(shape, device=DEV, generator=GEN) * 0.5).to(torch.float8_e4m3fn)


def sfw(rows, K):
    return g.pack_scale_words(torch.randint(124, 131, (rows, K // 32), device=DEV, generator=GEN, dtype=torch.uint8))


def timed(calls, rounds=20):
    for c in calls:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
    graph.replay()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000 / (rounds * len(calls))


src = g.mxfp8_gemm_decode(6400, 5120, block_M=16, block_N=32, block_K=128, num_stages=4, threads=128, padded_rows=True).get_kernel_source()
print("decode kernel mbarrier mentions:", src.count("mbarrier"), "cp_async:", src.count("cp_async"), "tma_load:", src.count("tma_load"))

# (b) prefill raster swizzle, bits against the unswizzled kernel
for name, (N, K) in {"draft_main": (6400, 5120), "q_b": (8192, 1280), "q_a_kv": (1792, 5120), "indexer_q_b": (4096, 1280)}.items():
    w, wsf = fp8((N, K)), sfw(N, K)
    for M in (4096, 16384):
        a, asf = fp8((M, K)), sfw(M, K)
        ref = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
        cells = []
        for panel in (0, 2, 4, 8, 16):
            try:
                k_ = g.mxfp8_gemm(N, K, **g.default_config(M, K), swizzle_panel=panel)
                out = ref if panel == 0 else torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
                t = timed([lambda: k_(a, w, asf, wsf, out)] * 2, rounds=5)
                same = "" if panel == 0 else ("=" if torch.equal(out.view(torch.int16), ref.view(torch.int16)) else " BITS DIFFER")
                cells.append(f"p{panel} {t:.0f}{same}")
            except Exception as error:
                cells.append(f"p{panel} {type(error).__name__}: {str(error)[:80]}")
        print(f"prefill {name} {M} rows (us): " + ", ".join(cells), flush=True)

# (a) draft main decode
import tile_kernels
from b12x.gemm import block_fp8_linear

N, K = 6400, 5120
copies = max(2, -(-160 * 2**20 // (N * K)))
ws_ = [fp8((N, K)) for _ in range(copies)]
block_scales = [torch.randint(124, 131, (N // 32, K // 32), device=DEV, generator=GEN, dtype=torch.uint8) for _ in range(copies)]
wsf_ = [g.pack_scale_words(s, rows=N) for s in block_scales]
packed = [block_fp8_linear.pack_weight(w, s.view(torch.float8_e8m0fnu), block_size=(32, 32)) for w, s in zip(ws_, block_scales)]
xq = torch.empty(g.DECODE_ROWS, K, dtype=torch.float8_e4m3fn, device=DEV)
sf = torch.empty(g.DECODE_ROWS, 4 * g.scale_words(K), dtype=torch.uint8, device=DEV)
configs = [dict(block_M=16, block_N=n, block_K=bk, num_stages=s, threads=t)
           for n in (32, 64) for bk in (128, 256) for s in (3, 4, 6, 8) for t in (64, 128)
           if s * (16 + n) * bk <= 92 * 1024]
kernels = {}
for cfg in configs:
    try:
        kernels["n{block_N}-k{block_K}-st{num_stages}-t{threads}".format(**cfg)] = (cfg, g.mxfp8_gemm_decode(N, K, **cfg, padded_rows=True))
    except Exception as error:
        print("  compile", cfg, type(error).__name__, str(error)[:80])
score = {}
for rows in (6, 12):
    x = (torch.randn(rows, K, device=DEV, generator=GEN) * 0.5).bfloat16()
    plan = block_fp8_linear.plan(block_fp8_linear.Caps(device=DEV, max_tokens=rows, in_features=K, out_features=N, block_size=(32, 32)))
    scratch = [torch.empty(s.shape, dtype=s.dtype, device=DEV) for s in plan.scratch_specs()]
    bout = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)
    bind = [block_fp8_linear.bind(plan, scratch=scratch, source=x, packed_weight=packed[i], output=bout.view(rows, N, 1)) for i in range(copies)]
    res = {"b12x": (timed([lambda: block_fp8_linear.run(binding=bind[0])] * 16),
                    timed([(lambda i=i: block_fp8_linear.run(binding=bind[i % copies])) for i in range(2 * copies)]))}
    out = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)
    for key, (cfg, k_) in kernels.items():
        def call(i, k_=k_):
            def run():
                tile_kernels.quant.per_token_cast(x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True, out=(xq[:rows], sf[:rows]))
                k_(xq, ws_[i], sf.view(torch.uint32), wsf_[i], out)
            return run
        res[key] = (timed([call(0)] * 16), timed([call(i % copies) for i in range(2 * copies)]))
        score[key] = score.get(key, 0) + res[key][0] + res[key][1]
    ranked = sorted((k for k in res if k != "b12x"), key=lambda k: res[k][0] + res[k][1])
    print(f"draft {rows} rows (warm/cold us): B12X {res['b12x'][0]:.1f}/{res['b12x'][1]:.1f}; " +
          ", ".join(f"{k} {res[k][0]:.1f}/{res[k][1]:.1f}" for k in ranked[:8]), flush=True)
prefill = g.mxfp8_gemm(N, K, **g.default_config(g.DECODE_ROWS + 1, K))
for key in sorted(score, key=score.get)[:4]:
    cfg, k_ = kernels[key]
    bad = 0
    for round_ in range(120):
        rows = (6, 12, 16)[round_ % 3]
        a, asf = fp8((g.DECODE_ROWS, K)), sfw(g.DECODE_ROWS, K)
        w = round_ % copies
        out = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)
        k_(a, ws_[w], asf, wsf_[w], out)
        big = torch.empty(80, N, dtype=torch.bfloat16, device=DEV)
        prefill(torch.cat([a[:rows], fp8((80 - rows, K))]), ws_[w], torch.cat([asf[:rows], sfw(80 - rows, K)]), wsf_[w], big)
        bad += not torch.equal(out.view(torch.int16), big[:rows].view(torch.int16))
    print(f"  bits {key}: {120 - bad}/120 equal to prefill {'PASS' if not bad else 'FAIL'}", flush=True)
