"""Final decode-tile sweep: unspecialized (race-free) tiles, K splits of one, whole calls vs B12X.

Small tiles (16 rows) at 6 and 12 rows, large tiles (64 rows) at 24 and 48 rows; K blocks of
128 or 256. The best configuration per shape and height then repeats a bit check against the
prefill kernel over 120 rounds of fresh inputs. Prints a JSON summary last.
"""
import importlib.util
import itertools
import json
import sys

import tile_kernels
import torch
from b12x.gemm import block_fp8_linear

spec = importlib.util.spec_from_file_location("g", "/b/gemm.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
DEV = torch.device("cuda")
GEN = torch.Generator(device="cuda").manual_seed(5)
TILES = [(int(m), tuple(int(r) for r in rows.split(","))) for m, rows in (a.split(":") for a in sys.argv[1:])] or [(16, (6, 12)), (64, (24, 48))]
SHAPES = {"q_b": (8192, 1280), "indexer_q_b": (4096, 1280), "q_a_kv": (1792, 5120), "draft_main": (6400, 5120)}


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


summary, failures = {}, []
for name, (N, K) in SHAPES.items():
    copies = max(2, -(-160 * 2**20 // (N * K)))
    ws_ = [fp8((N, K)) for _ in range(copies)]
    block_scales = [torch.randint(124, 131, (N // 32, K // 32), device=DEV, generator=GEN, dtype=torch.uint8) for _ in range(copies)]
    wsf_ = [g.pack_scale_words(s, rows=N) for s in block_scales]
    packed = [block_fp8_linear.pack_weight(w, s.view(torch.float8_e8m0fnu), block_size=(32, 32)) for w, s in zip(ws_, block_scales)]
    xq = torch.empty(g.DECODE_ROWS, K, dtype=torch.float8_e4m3fn, device=DEV)
    sf = torch.empty(g.DECODE_ROWS, 4 * g.scale_words(K), dtype=torch.uint8, device=DEV)
    prefill = g.mxfp8_gemm(N, K, **g.default_config(g.DECODE_ROWS + 1, K))
    summary[name] = {}
    for block_M, rows_list in TILES:
        configs = [dict(block_M=block_M, block_N=n, block_K=bk, num_stages=s, threads=t)
                   for n, bk, s, t in itertools.product((32, 64, 128), (128, 256), (2, 3, 4, 6), (64, 128))
                   if K % bk == 0 and N % n == 0 and s * (block_M + n) * bk <= 92 * 1024
                   and not (block_M == 64 and n == 32) and not (block_M == 32 and s == 6 and bk == 256)]
        kernels = {}
        for cfg in configs:
            try:
                kernels["n{block_N}-k{block_K}-st{num_stages}-t{threads}".format(**cfg)] = (
                    cfg, g.mxfp8_gemm_decode(N, K, **cfg, padded_rows=True))
            except Exception as error:
                print("  compile", cfg, type(error).__name__, str(error)[:80])
        score, table = {}, {}
        for rows in rows_list:
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
                try:
                    res[key] = (timed([call(0)] * 16), timed([call(i % copies) for i in range(2 * copies)]))
                except Exception as error:
                    print(f"  {name} {key}: {type(error).__name__}: {str(error)[:80]}")
                    torch.cuda.synchronize()
                    continue
                # Rank by the B12X-relative cost, cold weighted as serving reads most weights cold.
                score[key] = score.get(key, 0) + (res[key][0] / res["b12x"][0] + 3 * res[key][1] / res["b12x"][1]) / 4
            table[rows] = res
            ranked = sorted((k for k in res if k != "b12x"), key=lambda k: res[k][0] + 3 * res[k][1])
            print(f"{name} m{block_M} {rows} rows (warm/cold us): B12X {res['b12x'][0]:.1f}/{res['b12x'][1]:.1f}; " +
                  ", ".join(f"{k} {res[k][0]:.1f}/{res[k][1]:.1f}" for k in ranked[:6]), flush=True)
        best = min((k for k in score if all(k in table[r] for r in rows_list)), key=score.get)
        cfg, k_ = kernels[best]
        bad = 0
        for round_ in range(120):
            rows = (rows_list + (block_M,))[round_ % 3]
            a, asf = fp8((g.DECODE_ROWS, K)), sfw(g.DECODE_ROWS, K)
            w = round_ % copies
            out = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)
            k_(a, ws_[w], asf, wsf_[w], out)
            big = torch.empty(80, N, dtype=torch.bfloat16, device=DEV)
            prefill(torch.cat([a[:rows], fp8((80 - rows, K))]), ws_[w], torch.cat([asf[:rows], sfw(80 - rows, K)]), wsf_[w], big)
            bad += not torch.equal(out.view(torch.int16), big[:rows].view(torch.int16))
        verdict = "PASS" if not bad else "FAIL"
        if bad:
            failures.append(f"{name} m{block_M} {best}")
        print(f"BEST {name} m{block_M}: {best} " + "; ".join(
            f"{r} rows {table[r][best][0]:.1f}/{table[r][best][1]:.1f} vs B12X {table[r]['b12x'][0]:.1f}/{table[r]['b12x'][1]:.1f}"
            for r in rows_list) + f"; bits {120 - bad}/120 {verdict}", flush=True)
        summary[name][f"m{block_M}"] = {"best": cfg, "rows": {r: {"tilelang": table[r][best], "b12x": table[r]["b12x"]} for r in rows_list},
                                       "bits_equal": 120 - bad}
print(json.dumps({"summary": summary, "failures": failures}))
