"""No-WS timing: large decode and prefill tiles against WS, narrow draft tiles against B12X."""
import importlib.util

import tilelang
import torch

src = open("/b/gemm.py").read()


def module(name, text):
    path = f"/tmp/{name}.py"
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
GEN = torch.Generator(device="cuda").manual_seed(3)


def fp8(shape):
    return (torch.randn(shape, device=DEV, generator=GEN) * 0.5).to(torch.float8_e4m3fn)


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


SHAPES = {"q_b": (8192, 1280), "indexer_q_b": (4096, 1280), "q_a_kv": (1792, 5120), "draft_main": (6400, 5120)}
for name, (N, K) in SHAPES.items():
    copies = max(2, -(-160 * 2**20 // (N * K)))
    ws_ = [fp8((N, K)) for _ in range(copies)]
    block_scales = [torch.randint(124, 131, (N // 32, K // 32), device=DEV, generator=GEN, dtype=torch.uint8) for _ in range(copies)]
    wsf_ = [g.pack_scale_words(s, rows=N) for s in block_scales]
    a = fp8((4096, K))
    asf = g.pack_scale_words(torch.randint(124, 131, (4096, K // 32), device=DEV, generator=GEN, dtype=torch.uint8))
    cells = []
    cfg = g.fp8_decode_config(N, K)
    cfg.pop("shards")
    for rows in (24, 48):
        out = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)
        for label, mod in (("ws", g), ("no-ws", nows)):
            k_ = mod.mxfp8_gemm(N, K, block_M=64, **cfg, padded_rows=True)
            cold = timed([(lambda i=i: k_(a, ws_[i % copies], asf, wsf_[i % copies], out)) for i in range(2 * copies)])
            cells.append(f"{rows}r {label} {cold:.1f}")
    out = torch.empty(4096, N, dtype=torch.bfloat16, device=DEV)
    for label, mod in (("ws", g), ("no-ws", nows)):
        k_ = mod.mxfp8_gemm(N, K, **g.default_config(4096, K))
        cells.append(f"prefill4096 {label} {timed([lambda: k_(a, ws_[0], asf, wsf_[0], out)] * 4, rounds=10):.0f}")
    out = torch.empty(512, N, dtype=torch.bfloat16, device=DEV)
    for label, mod in (("ws", g), ("no-ws", nows)):
        k_ = mod.mxfp8_gemm(N, K, **g.default_config(512, K))
        cells.append(f"prefill512 {label} {timed([lambda: k_(a[:512], ws_[0], asf[:512], wsf_[0], out)] * 4, rounds=10):.1f}")
    print(f"large {name} {cfg} (cold us): " + ", ".join(cells), flush=True)

# Draft main: narrow no-WS tiles at 6 and 12 rows, whole call (TileKernels cast + GEMM) vs B12X.
import tile_kernels
from b12x.gemm import block_fp8_linear

N, K = SHAPES["draft_main"]
copies = max(2, -(-160 * 2**20 // (N * K)))
ws_ = [fp8((N, K)) for _ in range(copies)]
block_scales = [torch.randint(124, 131, (N // 32, K // 32), device=DEV, generator=GEN, dtype=torch.uint8) for _ in range(copies)]
wsf_ = [g.pack_scale_words(s, rows=N) for s in block_scales]
packed = [block_fp8_linear.pack_weight(w, s.view(torch.float8_e8m0fnu), block_size=(32, 32)) for w, s in zip(ws_, block_scales)]
xq = torch.empty(g.DECODE_ROWS, K, dtype=torch.float8_e4m3fn, device=DEV)
sf = torch.empty(g.DECODE_ROWS, 4 * g.scale_words(K), dtype=torch.uint8, device=DEV)
for rows in (6, 12):
    x = (torch.randn(rows, K, device=DEV, generator=GEN) * 0.5).bfloat16()
    plan = block_fp8_linear.plan(block_fp8_linear.Caps(device=DEV, max_tokens=rows, in_features=K, out_features=N, block_size=(32, 32)))
    scratch = [torch.empty(s.shape, dtype=s.dtype, device=DEV) for s in plan.scratch_specs()]
    bout = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)
    bind = [block_fp8_linear.bind(plan, scratch=scratch, source=x, packed_weight=packed[i], output=bout.view(rows, N, 1)) for i in range(copies)]
    res = {"b12x": (timed([lambda: block_fp8_linear.run(binding=bind[0])] * 16),
                    timed([(lambda i=i: block_fp8_linear.run(binding=bind[i % copies])) for i in range(2 * copies)]))}
    out = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)
    for n in (32, 64, 128):
        for s in (2, 3, 4, 6, 8):
            for t in (32, 64, 128):
                if s * (16 + n) * 128 > 92 * 1024 or (t == 32 and n > 64):
                    continue
                try:
                    k_ = nows.mxfp8_gemm(N, K, block_M=16, block_N=n, block_K=128, num_stages=s, threads=t, padded_rows=True)

                    def call(i, k_=k_):
                        def run():
                            tile_kernels.quant.per_token_cast(x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True, out=(xq[:rows], sf[:rows]))
                            k_(xq, ws_[i], sf.view(torch.uint32), wsf_[i], out)
                        return run
                    res[f"n{n}-st{s}-t{t}"] = (timed([call(0)] * 16), timed([call(i % copies) for i in range(2 * copies)]))
                except Exception as error:
                    print(f"  n{n}-st{s}-t{t}: {type(error).__name__}: {str(error)[:100]}")
    ranked = sorted((k for k in res if k != "b12x"), key=lambda k: res[k][1])
    print(f"draft {rows} rows (warm/cold us): B12X {res['b12x'][0]:.1f}/{res['b12x'][1]:.1f}; " +
          ", ".join(f"{k} {res[k][0]:.1f}/{res[k][1]:.1f}" for k in ranked[:8]), flush=True)
