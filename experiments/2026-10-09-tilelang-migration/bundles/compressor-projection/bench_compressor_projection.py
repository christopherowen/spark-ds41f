"""Port B5: the compressor's wkv/wgate projection (5120 -> 512 per part), B12X's
bf16_gemv against TileLangSplitLinearMethod (both parts in one launch: one-launch
split-K for decode rows, where the last CTA of a tile adds the shards, and a
two-output GEMM for prefill), at
the TP4 shapes: ratio 1 (wkv, BF16 out) and ratio 2 (wkv + wgate, FP32 out).

1. Error against FP64, no worse than DeepSeek's reference computation (an FP32 matmul
   of the BF16 values, TF32 off; max and RMS); B12X's error reported beside it.
2. Batch invariance: every row count gives the same rows of the 8192-row batch, and
   repeated graph replays give the same bits (the split-K arrival counters reset).
3. Time per call (both parts) under CUDA graphs, warm and cold, at the decode capture
   sizes and prefill rows, against B12X.
4. A sweep of the one-launch split-K's decode tiles (block_N, stages, threads per
   row tile) at the serving shards: bits checked against the serving tile, the
   three best by worst warm/cold ratio to B12X.

Exits 1 if a check fails or the serving configuration is slower than B12X at any
shape, warm or cold.
"""
import concurrent.futures
import sys

import torch
from torch import nn

sys.path.insert(0, "/b")
from kbench import (  # noqa: E402
    CAPACITY, DECODE, DEVICE, PREFILL, REPEAT, REPLAYS, errors, no_worse, per_call, repeatable, slower,
    timing_line,
)

from b12x.gemm import bf16_gemv  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang import gemm as g  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.linear import (  # noqa: E402
    TileLangSplitLinearMethod,
    project_parts,
)
from vllm.v1.worker.workspace import use_preallocated_workspace  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
K, PART = 5120, 512
SPLIT_BLOCKED = True  # the serving setting (SPLIT_BF16)
failures = []


def run(parts, out_dtype, session):
    name = str(out_dtype).removeprefix("torch.")
    tl_dtype = {torch.bfloat16: "bfloat16", torch.float32: "float32"}[out_dtype]
    gen = torch.Generator(device=DEVICE).manual_seed(41 + parts)
    weight = (torch.randn((PART * parts, K), generator=gen, device=DEVICE) * 0.02).bfloat16()
    x = torch.randn((CAPACITY, K), generator=gen, device=DEVICE).bfloat16()
    w_parts = [weight[i * PART:(i + 1) * PART] for i in range(parts)]

    layer = nn.Module()
    layer.weight = nn.Parameter(weight, requires_grad=False)
    layer.out_dtype = out_dtype
    TileLangSplitLinearMethod(parts).process_weights_after_loading(layer)
    scratch = torch.empty(64 << 20, dtype=torch.uint8, device=DEVICE)

    def outputs():
        return [torch.empty((CAPACITY, PART), dtype=out_dtype, device=DEVICE) for _ in range(parts)]

    tl_out, b_out, sw_out = outputs(), outputs(), outputs()

    def tilelang(rows):
        with use_preallocated_workspace(scratch):
            project_parts(layer, x[:rows], tl_out, rows)

    plans, requests = {}, []
    for rows in DECODE + (CAPACITY,):
        plans[rows] = bf16_gemv.plan(bf16_gemv.GemvQuery(
            source_dtype="bfloat16", weight_dtype="bfloat16", output_dtype=name,
            max_rows=rows, in_features=K, out_features=PART, source_contiguous=True,
            source_aligned=True, weight_contiguous=True, weight_aligned=True))

        def prepare(state, rows=rows):
            return PreparedCall(run=lambda: [state.run(x[:rows], w, out=o[:rows])
                                             for w, o in zip(w_parts, b_out)])

        requests.append(plans[rows].request(name=f"projection.{name}.m{rows}", prepare_call=prepare))
    session.prepare(tuple(requests))

    def b12x(rows):
        plan = plans[rows if rows <= DECODE[-1] else CAPACITY]
        for w, o in zip(w_parts, b_out):
            bf16_gemv.mm(x[:rows], w, out=o[:rows], output_dtype=out_dtype, plan=plan)

    # 1. Error against FP64 at the full batch, beside DeepSeek's FP32 reference.
    tilelang(CAPACITY)
    b12x(CAPACITY)
    torch.cuda.synchronize()
    full = [o.clone() for o in tl_out]
    b_full = [o.clone() for o in b_out]
    for i, w in enumerate(w_parts):
        ref = x.double() @ w.double().T
        reference = (x.float() @ w.float().T).to(out_dtype)
        port, base, ds = errors(full[i], ref), errors(b_full[i], ref), errors(reference, ref)
        ok = no_worse(port, ds)
        print(f"ratio {parts} {name} part {i}: error vs fp64 max/rms  tilelang {port[0]:.3e}/{port[1]:.3e}"
              f"  deepseek-fp32 {ds[0]:.3e}/{ds[1]:.3e}  b12x {base[0]:.3e}/{base[1]:.3e}  {'ok' if ok else 'WORSE'}",
              flush=True)
        if not ok:
            failures.append(f"ratio {parts} part {i} error")
        del ref, reference

    # 2. Batch invariance.
    varies = []
    for rows in DECODE + (g.SPLIT_DECODE_ROWS, 200) + PREFILL[:-1]:
        tilelang(rows)
        torch.cuda.synchronize()
        if not all(torch.equal(tl_out[i][:rows], full[i][:rows]) for i in range(parts)):
            varies.append(rows)
    if varies:
        failures.append(f"ratio {parts}: TileLang differs from its full batch at {varies}")
    print(f"ratio {parts}: TileLang differs from its full batch at {varies or 'no'} sizes", flush=True)
    flaky = [rows for rows in REPEAT if repeatable(lambda: tilelang(rows), lambda: [o[:rows] for o in tl_out])]
    if flaky:
        failures.append(f"ratio {parts}: TileLang not repeatable at {flaky}")
    print(f"ratio {parts}: TileLang not repeatable at {flaky or 'no'} sizes ({REPLAYS} replays each)", flush=True)

    # 3. Time, serving configuration against B12X.
    print(f"ratio {parts} {name}: us per call (both parts), warm / cold; serving shards "
          f"{layer.tilelang_shards}", flush=True)
    base_times = {}
    for rows in DECODE + PREFILL:
        base = base_times[rows] = per_call(lambda: b12x(rows), rows)
        port = per_call(lambda: tilelang(rows), rows)
        print(timing_line(rows, base, port), flush=True)
        if slower(base, port):
            failures.append(f"ratio {parts} rows {rows}: TileLang slower")

    # 4. Decode tile sweep at the serving shards: block_N, stages and threads per
    # row tile (they change speed, not bits; checked), warm / cold against B12X.
    shards, blocked = layer.tilelang_shards, SPLIT_BLOCKED
    counters = torch.zeros(8 * (PART * parts // 16), dtype=torch.int32, device=DEVICE)
    p = torch.empty((shards, g.SPLIT_DECODE_ROWS, PART * parts), dtype=torch.float32, device=DEVICE)
    print(f"ratio {parts} decode tiles (shards {shards}): best per row tile by worst ratio to B12X", flush=True)
    for block_M, rows_set in ((16, (1, 2, 6, 16)), (32, (24, 32)), (64, (48, 64, 96))):
        configs = [dict(block_N=bn, num_stages=st, threads=t)
                   for bn in (32, 64, 128) for st in (2, 3, 4, 6) for t in (64, 128, 256)
                   if (PART * parts // 2) % bn == 0]
        def build(cfg, block_M=block_M):
            try:
                return g.bf16_gemm_splitk(PART * parts, K, shards, block_M=block_M, out_dtype=tl_dtype,
                                          parts=parts, blocked=blocked, **cfg)
            except Exception as error:
                return error
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            kernels = list(pool.map(build, configs))
        results = {}
        for cfg, kernel in zip(configs, kernels):
            if isinstance(kernel, Exception):
                print(f"  m{block_M} {cfg}: {type(kernel).__name__}", flush=True)
                continue

            def call(rows, kernel=kernel):
                view = p.view(-1)[:shards * rows * PART * parts].view(shards, rows, PART * parts)
                kernel(x[:rows], weight, view, counters, *[o[:rows] for o in sw_out])
            times, ok = {}, True
            for rows in rows_set:
                try:
                    tilelang(rows)
                    call(rows)
                    torch.cuda.synchronize()
                except Exception as error:
                    print(f"  m{block_M} {cfg}: {type(error).__name__}: {str(error)[:60]}", flush=True)
                    ok = False
                    break
                if not all(torch.equal(sw_out[i][:rows], tl_out[i][:rows]) for i in range(parts)):
                    failures.append(f"ratio {parts} m{block_M} {cfg} rows {rows}: bits differ")
                times[rows] = per_call(lambda rows=rows: call(rows), rows)
            if ok:
                results["n{block_N}-st{num_stages}-t{threads}".format(**cfg)] = times

        def worst(times):
            return max(max(w / base_times[r][0], c / base_times[r][1]) for r, (w, c) in times.items())
        ranked = sorted(results, key=lambda key: (worst(results[key]), sum(sum(v) for v in results[key].values())))
        for key in ranked[:3]:
            print(f"  m{block_M} {key}: worst {worst(results[key]):.3f}  "
                  + "  ".join(f"{r}: {w:.2f}/{c:.2f}" for r, (w, c) in results[key].items()), flush=True)

for parts, out_dtype in ((1, torch.bfloat16), (2, torch.float32)):
    with PreparationSession(device=DEVICE, autotune=False, compile_workers=2) as session:
        run(parts, out_dtype, session)
for failure in failures:
    print("FAIL", failure)
sys.exit(1 if failures else 0)
