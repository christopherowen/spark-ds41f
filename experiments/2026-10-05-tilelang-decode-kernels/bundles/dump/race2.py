"""Race census: tiles where some producer threads copy no scale words, with and without warp
specialization, plus timing of the race-free variants at the decode shapes."""
import importlib.util
import sys

import tilelang
import torch

src = open("/b/gemm.py").read()


def module(name, text):
    path = f"/tmp/{name}.py"  # TileLang reads kernel source through inspect
    open(path, "w").write(text)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g = module("g", src)
nows = module("g_nows", src.replace(
    "@tilelang.jit\ndef mxfp8_gemm(",
    "@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})\ndef mxfp8_gemm("))
DEV = torch.device("cuda")
GEN = torch.Generator(device="cuda").manual_seed(2)


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


def census(mod, label, N, K, cfg, runs, rows=(6, 16, 48)):
    kernel = mod.mxfp8_gemm(N, K, **cfg, padded_rows=True)
    prefill = g.mxfp8_gemm(N, K, **g.default_config(g.DECODE_ROWS + 1, K))
    w, wsf = fp8((N, K)), sfw(N, K)
    report = []
    for r in rows:
        if r > cfg["block_M"] * (g.DECODE_ROWS // cfg["block_M"]):
            continue
        a, asf = fp8((g.DECODE_ROWS, K)), sfw(g.DECODE_ROWS, K)
        a_big = torch.cat([a[:r], fp8((80 - r, K))])
        asf_big = torch.cat([asf[:r], sfw(80 - r, K)])
        big = torch.empty(80, N, dtype=torch.bfloat16, device=DEV)
        prefill(a_big, w, asf_big, wsf, big)
        want = big[:r].view(torch.int16)
        bad = 0
        out = torch.empty(r, N, dtype=torch.bfloat16, device=DEV)
        for _ in range(runs):
            kernel(a, w, asf, wsf, out)
            bad += not torch.equal(out.view(torch.int16), want)
        report.append(f"{r} rows {bad}/{runs}")
    print(f"{label} {N}x{K} {cfg}: unequal " + ", ".join(report), flush=True)


SMALL = [dict(block_M=16, block_N=n, block_K=128, num_stages=s, threads=t)
         for n, s, t in ((32, 2, 128), (64, 2, 128), (64, 4, 128), (64, 2, 64), (128, 2, 128), (128, 2, 64))]
LARGE = [dict(block_M=64, block_N=n, block_K=128, num_stages=s, threads=128) for n, s in ((64, 3), (64, 4), (128, 2))]
for cfg in SMALL:
    census(g, "ws", 6400, 5120, cfg, 300)
    census(nows, "no-ws", 6400, 5120, cfg, 300)
for N, K in ((8192, 1280), (6400, 5120), (4096, 1280), (1792, 5120)):
    for cfg in LARGE:
        census(g, "ws", N, K, cfg, 500)
# Timing of no-WS small tiles against WS at the decode shapes (cold weights, 6 rows).
for N, K in ((8192, 1280), (4096, 1280), (1792, 5120), (6400, 5120)):
    copies = max(2, -(-160 * 2**20 // (N * K)))
    ws_ = [fp8((N, K)) for _ in range(copies)]
    wsf_ = [sfw(N, K) for _ in range(copies)]
    a, asf = fp8((g.DECODE_ROWS, K)), sfw(g.DECODE_ROWS, K)
    out = torch.empty(6, N, dtype=torch.bfloat16, device=DEV)
    cells = []
    for n, s, t in ((64, 2, 64), (128, 2, 128), (128, 2, 64), (128, 3, 128), (128, 4, 128), (64, 4, 128), (64, 2, 128)):
        cfg = dict(block_M=16, block_N=n, block_K=128, num_stages=s, threads=t)
        for label, mod in (("ws", g), ("no-ws", nows)):
            k_ = mod.mxfp8_gemm(N, K, **cfg, padded_rows=True)
            cold = timed([(lambda i=i: k_(a, ws_[i % copies], asf, wsf_[i % copies], out)) for i in range(2 * copies)])
            warm = timed([lambda: k_(a, ws_[0], asf, wsf_[0], out)] * 16)
            cells.append(f"{label} n{n}-st{s}-t{t} {warm:.1f}/{cold:.1f}")
    print(f"timing {N}x{K} 6 rows (warm/cold us): " + "; ".join(cells), flush=True)
