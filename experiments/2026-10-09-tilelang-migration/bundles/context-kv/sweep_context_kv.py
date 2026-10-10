"""Split-K sweep for the DSpark context-KV projection (512 x 5120 block-32 FP8 per rank).

For each shard count (shards change bits, so one is chosen per shape):
- decode: whole calls (activation cast, split-K partials, reduce) at 6, 24, 48, 96 and
  128 rows over partial tiles, and at 256 and 512 rows to see how far split-K pays
  against the prefill tiles, against B12X's block_fp8_linear prepared as the drafter
  prepares it; warm (same weight), cold (distinct weights) and self (each call
  follows the L2 prefetch of its own weight);
- prefill: the shard-accumulating GEMM over prefill tiles at 512, 2048 and 8192 rows
  (warm and cold), against B12X;
- bits: the best decode tile at each height against the best prefill tile.

Prints the winners and a JSON summary; exits 1 only if a winner's bits differ.
"""
import concurrent.futures
import itertools
import json

import tile_kernels
import torch
from b12x.gemm import block_fp8_linear
from b12x.preparation import PreparationSession, PreparedCall

from vllm.models.deepseek_v4_1.l2_prefetch import object_segments
from vllm.models.deepseek_v4_1.tilelang import gemm as g
from vllm.models.glm5next.nvidia import l2_prefetch as L

DEV = torch.device("cuda", torch.cuda.current_device())
GEN = torch.Generator(device=DEV).manual_seed(7)
PF = L.L2Prefetcher.get(DEV)
N, K = 512, 5120
DECODE_ROWS = {6: 16, 24: 32, 48: 64, 96: 64, 128: 64, 256: 64, 512: 64}  # rows: partial tile height
PREFILL_ROWS = (512, 2048, 8192)
SHARDS = (5,)


def fp8(shape):
    return (torch.randn(shape, device=DEV, generator=GEN) * 0.5).to(torch.float8_e4m3fn)


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


def build(factory):
    try:
        return factory()
    except Exception as error:
        return error


copies = max(4, -(-160 * 2**20 // (N * K)))
ws_ = [fp8((N, K)) for _ in range(copies)]
exps_ = [torch.randint(124, 131, (N // 32, K // 32), device=DEV, generator=GEN, dtype=torch.uint8) for _ in range(copies)]
wsf_ = [g.pack_scale_words(e, rows=N) for e in exps_]
packed = [block_fp8_linear.pack_weight(w, e.view(torch.float8_e8m0fnu), block_size=(32, 32)) for w, e in zip(ws_, exps_)]
tl_plans = [L.make_plan([L.tensor_segment("w", w), L.tensor_segment("sf", s, min_bytes=0)], 20 * 2**20, DEV)[0]
            for w, s in zip(ws_, wsf_)]
b_plans = [L.make_plan(object_segments("b12x_weight", p), 20 * 2**20, DEV)[0] for p in packed]
x_all = (torch.randn(max(PREFILL_ROWS), K, device=DEV, generator=GEN) * 0.5).bfloat16()
xq = torch.empty(max(PREFILL_ROWS), K, dtype=torch.float8_e4m3fn, device=DEV)
sf = torch.empty(max(PREFILL_ROWS), 4 * g.scale_words(K), dtype=torch.uint8, device=DEV)
out = torch.empty(max(PREFILL_ROWS), N, dtype=torch.bfloat16, device=DEV)
session = PreparationSession(device=DEV, autotune=False, compile_workers=2)
b12x = {}
ALL_ROWS = sorted({*DECODE_ROWS, *PREFILL_ROWS})  # 512 is both; a repeated plan name is not prepared again
for rows in ALL_ROWS:
    plan = block_fp8_linear.plan(block_fp8_linear.Caps(device=DEV, max_tokens=rows, in_features=K, out_features=N,
                                                       block_size=(32, 32), output_mode="provided"))
    owned = {}
    bout = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)

    def prepare(state, rows=rows, owned=owned, bout=bout):
        owned["scratch"] = [torch.empty(s.shape, dtype=s.dtype, device=DEV) for s in state.scratch.scratch_specs()]
        binding = state.bind(scratch=owned["scratch"], source=x_all[:rows], packed_weight=packed[0],
                             output=bout.view(rows, N, 1))
        return PreparedCall(run=lambda: state.run_binding(binding))

    session.prepare((plan.request(name=f"context_kv.m{rows}", prepare_call=prepare),))
    b12x[rows] = [block_fp8_linear.bind(plan, scratch=owned["scratch"], source=x_all[:rows], packed_weight=packed[i],
                                        output=bout.view(rows, N, 1)) for i in range(copies)]


def measure(run_one, prefetch_plans, rows, with_self=True):
    warm = timed([lambda: run_one(0)] * 16)
    cold = timed([(lambda i=i: run_one(i % copies)) for i in range(2 * copies)])
    if not with_self:
        return warm, cold

    def selfcall(i):
        def run():
            PF.issue(prefetch_plans[i % copies], rows)
            run_one(i % copies)
        return run
    return warm, cold, timed([selfcall(i) for i in range(2 * copies)])


def cast(rows):
    tile_kernels.quant.per_token_cast(x_all[:rows], "e4m3", 32, round_sf=True, use_packed_ue8m0=True,
                                      out=(xq[:rows], sf[:rows]))


def name_of(cfg):
    return "-".join(f"{k[6] if k.startswith('block_') else k[:2]}{v}" for k, v in cfg.items())


base = {rows: measure(lambda i, rows=rows: block_fp8_linear.run(binding=b12x[rows][i]), b_plans, rows,
                      rows in DECODE_ROWS) for rows in ALL_ROWS}
for rows, m in base.items():
    print(f"B12X {rows} rows: " + "/".join(f"{t:.1f}" for t in m), flush=True)

summary, failures = {}, []
for shards in SHARDS:
    if (K // 128) % shards:
        continue
    reduce = g.splitk_reduce(N, shards)
    partials_buf = torch.empty(shards, max(DECODE_ROWS), N, dtype=torch.float32, device=DEV)
    entry = summary.setdefault(shards, {"decode": {}, "prefill": {}})
    winners = {}
    for rows, block_M in DECODE_ROWS.items():
        configs = [dict(block_N=n, block_K=bk, num_stages=st, threads=t)
                   for n, bk, st, t in itertools.product((32, 64, 128), (128, 256), (2, 3, 4), (64, 128))
                   if (K // shards) % bk == 0 and st * (block_M + n) * bk <= 92 * 1024]
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            kernels = list(pool.map(lambda c: build(lambda: g.mxfp8_gemm_partials(N, K, shards, block_M=block_M, **c)), configs))
        p = partials_buf.view(-1)[:shards * rows * N].view(shards, rows, N)
        padded = -(-rows // block_M) * block_M
        results = {}
        for cfg, kernel in zip(configs, kernels):
            if isinstance(kernel, Exception):
                continue

            def run_one(i, kernel=kernel):
                cast(rows)
                kernel(xq[:padded], ws_[i], sf[:padded].view(torch.uint32), wsf_[i], p)
                reduce(p, out[:rows])
            try:
                results[name_of(cfg)] = (cfg, measure(run_one, tl_plans, rows))
            except Exception as error:
                print(f"  s{shards} {rows} {cfg}: {type(error).__name__}: {str(error)[:80]}")
                torch.cuda.synchronize()

        def worst(m, rows=rows):
            return max(a / b for a, b in zip(m, base[rows]))
        ranked = sorted(results, key=lambda k: (worst(results[k][1]), sum(results[k][1])))
        best = ranked[0]
        winners[rows] = (results[best][0], kernels[configs.index(results[best][0])])
        m = results[best][1]
        print(f"shards {shards} decode {rows} rows (m{block_M}): best {best} {m[0]:.1f}/{m[1]:.1f}/{m[2]:.1f} vs B12X "
              + "/".join(f"{t:.1f}" for t in base[rows]) + f" worst ratio {worst(m):.3f}", flush=True)
        entry["decode"][rows] = {"best": results[best][0], "tilelang": m, "b12x": base[rows], "worst": worst(m)}
    prefill_configs = [dict(block_M=bm, block_N=bn, block_K=bk, num_stages=st)
                       for bm, bn, bk, st in itertools.product((64, 128), (64, 128), (128,), (2, 3))
                       if st * (bm + bn) * bk <= 92 * 1024]
    pkernels = [build(lambda c=c: g.mxfp8_gemm(N, K, **c, shards=shards)) for c in prefill_configs]
    pbest = None
    for rows in PREFILL_ROWS:
        results = {}
        for cfg, kernel in zip(prefill_configs, pkernels):
            if isinstance(kernel, Exception):
                continue

            def run_one(i, kernel=kernel, rows=rows):
                cast(rows)
                kernel(xq[:rows], ws_[i], sf[:rows].view(torch.uint32), wsf_[i], out[:rows])
            results[name_of(cfg)] = (cfg, measure(run_one, tl_plans, rows, with_self=False))

        def worst(m, rows=rows):
            return max(a / b for a, b in zip(m, base[rows]))
        ranked = sorted(results, key=lambda k: (worst(results[k][1]), sum(results[k][1])))
        m = results[ranked[0]][1]
        print(f"shards {shards} prefill {rows} rows: best {ranked[0]} {m[0]:.1f}/{m[1]:.1f} vs B12X "
              + "/".join(f"{t:.1f}" for t in base[rows]) + f" worst ratio {worst(m):.3f}; all: "
              + ", ".join(f"{k} {results[k][1][0]:.0f}/{results[k][1][1]:.0f}" for k in ranked), flush=True)
        entry["prefill"][rows] = {"best": results[ranked[0]][0], "tilelang": m, "b12x": base[rows], "worst": worst(m)}
        if rows == 2048:
            pbest = pkernels[prefill_configs.index(results[ranked[0]][0])]
    # Bits: each decode winner against the prefill winner, on fresh activations.
    bad = 0
    for rows, (cfg, kernel) in winners.items():
        block_M = DECODE_ROWS[rows]
        padded = -(-rows // block_M) * block_M
        p = partials_buf.view(-1)[:shards * rows * N].view(shards, rows, N)
        R = max(DECODE_ROWS)
        for trial in range(20):
            x_all[:R].copy_((torch.randn(R, K, device=DEV, generator=GEN) * 0.5).bfloat16())
            cast(R)
            w = trial % copies
            kernel(xq[:padded], ws_[w], sf[:padded].view(torch.uint32), wsf_[w], p)
            small = torch.empty(rows, N, dtype=torch.bfloat16, device=DEV)
            reduce(p, small)
            big = torch.empty(R, N, dtype=torch.bfloat16, device=DEV)
            pbest(xq[:R], ws_[w], sf[:R].view(torch.uint32), wsf_[w], big)
            bad += not torch.equal(small.view(torch.int16), big[:rows].view(torch.int16))
    entry["bits_bad"] = bad
    print(f"shards {shards}: decode winners vs prefill bits {'PASS' if not bad else f'FAIL ({bad})'}", flush=True)
    if bad:
        failures.append(f"shards {shards}")
session.close()
print(json.dumps({"summary": summary, "failures": failures}, default=str))
raise SystemExit(1 if failures else 0)
