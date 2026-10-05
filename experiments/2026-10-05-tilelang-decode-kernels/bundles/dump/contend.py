"""Decode GEMMs under the serving L2 prefetch: TileLang tiles vs B12X at 6 and 12 rows.

contended: each call follows a 20 MB prefetch of unrelated weights (the WO window
           running beside the indexer Q-B), weights cold;
self:      each call follows a prefetch of its own weight (the NEXT window's
           projections), so the GEMM races the fill.
Whole calls (activation cast + GEMM), microseconds per call, CUDA graphs.
"""
import importlib.util

import tile_kernels
import torch
from b12x.gemm import block_fp8_linear
from vllm.models.glm5next.nvidia import l2_prefetch as L

spec = importlib.util.spec_from_file_location("g", "/b/gemm.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
DEV = torch.device("cuda")
GEN = torch.Generator(device="cuda").manual_seed(6)
print("prefetch launcher:", L._get_launcher() is not None, "enabled:", L.ENABLED, "allowed:", L._prefetch_allowed())
PF = L.L2Prefetcher.get(DEV)  # driven directly: the module switch is a serving default


def issue(plan, rows):
    PF.issue(plan, rows)


def fp8(shape):
    return (torch.randn(shape, device=DEV, generator=GEN) * 0.5).to(torch.float8_e4m3fn)


def timed(calls, rounds=20):
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


junk = [torch.empty(24 * 2**20, dtype=torch.uint8, device=DEV) for _ in range(4)]
junk_plans = [L.make_plan([L.tensor_segment(f"junk{i}", j)], 20 * 2**20, DEV)[0] for i, j in enumerate(junk)]
CANDIDATES = {
    "q_b": ((8192, 1280), ["n64-k256-st4-t128", "n32-k256-st4-t128", "n64-k128-st4-t128", "n128-k128-st2-t128", "n32-k256-st2-t128"]),
    "indexer_q_b": ((4096, 1280), ["n128-k128-st3-t128", "n32-k256-st3-t64", "n64-k256-st3-t64", "n32-k256-st2-t128", "n64-k256-st4-t128"]),
    "q_a_kv": ((1792, 5120), ["n64-k256-st3-t128", "n32-k256-st3-t128", "n64-k256-st4-t64"]),
    "draft_main": ((6400, 5120), ["n32-k256-st6-t64", "n32-k256-st4-t128", "n64-k256-st4-t128", "n32-k256-st3-t64"]),
}


def parse(key):
    n, k, s, t = key.split("-")
    return dict(block_M=16, block_N=int(n[1:]), block_K=int(k[1:]), num_stages=int(s[2:]), threads=int(t[1:]))


for name, ((N, K), keys) in CANDIDATES.items():
    copies = max(4, -(-160 * 2**20 // (N * K)))
    ws_ = [fp8((N, K)) for _ in range(copies)]
    block_scales = [torch.randint(124, 131, (N // 32, K // 32), device=DEV, generator=GEN, dtype=torch.uint8) for _ in range(copies)]
    wsf_ = [g.pack_scale_words(s, rows=N) for s in block_scales]
    packed = [block_fp8_linear.pack_weight(w, s.view(torch.float8_e8m0fnu), block_size=(32, 32)) for w, s in zip(ws_, block_scales)]
    tl_plans = [L.make_plan([L.tensor_segment("w", w), L.tensor_segment("sf", s, min_bytes=0)], 20 * 2**20, DEV)[0]
                for w, s in zip(ws_, wsf_)]
    def b12x_segments(p):
        from vllm.models.deepseek_v4_1.l2_prefetch import object_segments
        return object_segments("b12x_weight", p)
    b_plans = [L.make_plan(b12x_segments(p), 20 * 2**20, DEV)[0] for p in packed]
    xq = torch.empty(g.DECODE_ROWS, K, dtype=torch.float8_e4m3fn, device=DEV)
    sf = torch.empty(g.DECODE_ROWS, 4 * g.scale_words(K), dtype=torch.uint8, device=DEV)
    kernels = {key: g.mxfp8_gemm_decode(N, K, **parse(key), padded_rows=True) for key in keys}
    for rows in (6, 12):
        x = (torch.randn(rows, K, device=DEV, generator=GEN) * 0.5).bfloat16()
        plan = block_fp8_linear.plan(block_fp8_linear.Caps(device=DEV, max_tokens=rows, in_features=K, out_features=N, block_size=(32, 32)))
        scratch = [torch.empty(s.shape, dtype=s.dtype, device=DEV) for s in plan.scratch_specs()]
        bout = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)
        bind = [block_fp8_linear.bind(plan, scratch=scratch, source=x, packed_weight=packed[i], output=bout.view(rows, N, 1)) for i in range(copies)]
        out = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)

        def b12x_call(i, mode):
            def run():
                issue(junk_plans[i % 4] if mode == "contended" else b_plans[i % copies], rows)
                block_fp8_linear.run(binding=bind[i % copies])
            return run

        def tl_call(kernel, i, mode):
            def run():
                issue(junk_plans[i % 4] if mode == "contended" else tl_plans[i % copies], rows)
                tile_kernels.quant.per_token_cast(x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True, out=(xq[:rows], sf[:rows]))
                kernel(xq, ws_[i % copies], sf.view(torch.uint32), wsf_[i % copies], out)
            return run

        cells = []
        for mode in ("contended", "self"):
            b = timed([b12x_call(i, mode) for i in range(2 * copies)])
            row = [f"B12X {b:.1f}"]
            for key, kernel in kernels.items():
                row.append(f"{key} {timed([tl_call(kernel, i, mode) for i in range(2 * copies)]):.1f}")
            cells.append(f"{mode}: " + ", ".join(row))
        print(f"{name} {rows} rows | " + " | ".join(cells), flush=True)
