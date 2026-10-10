"""Port B5: the compressor's wkv/wgate projection (5120 -> 512 per part), B12X's
bf16_gemv against TileLangSplitLinearMethod (both parts in one launch: one-launch
split-K for decode rows, where the last CTA of a tile adds the shards, and a
two-output GEMM for prefill), at
the TP4 shapes: ratio 1 (wkv, BF16 out) and ratio 2 (wkv + wgate, FP32 out).

1. Error against FP64, no worse than DeepSeek's reference computation (an FP32 matmul
   of the BF16 values, TF32 off; max and RMS); B12X's error reported beside it.
2. Batch invariance: every row count gives the same rows of the 8192-row batch.
3. Time per call (both parts) under CUDA graphs, warm and cold, at the decode capture
   sizes and prefill rows, against B12X.
4. A sweep of shard counts and blocked accumulation (both change bits, so one
   setting is chosen per shape): time and error at each, one launch for decode rows.

Exits 1 if a check fails or the serving configuration is slower than B12X at any
shape, warm or cold.
"""
import sys

import torch
from torch import nn

sys.path.insert(0, "/b")
from kbench import (  # noqa: E402
    CAPACITY, DECODE, DEVICE, PREFILL, errors, no_worse, per_call, slower, timing_line,
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
SWEEP = (4, 5, 8, 10)
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

    def configuration(shards, blocked):
        """A shard count's kernels, called like the serving path (one launch for
        decode rows, the shard GEMM for prefill)."""
        splitk = {bm: g.bf16_gemm_splitk(PART * parts, K, shards, block_M=bm, out_dtype=tl_dtype, parts=parts,
                                         blocked=blocked) for bm in (16, 32, 64)}
        counters = torch.zeros(8 * (PART * parts // 64), dtype=torch.int32, device=DEVICE)
        if parts == 2:
            gemm = g.bf16_gemm_two(PART, K, out_dtype=tl_dtype, shards=shards, blocked=blocked)
        else:
            gemm = g.bf16_gemm(PART, K, out_dtype=tl_dtype, shards=shards, blocked=blocked)
        p = torch.empty((shards, g.SPLIT_DECODE_ROWS, PART * parts), dtype=torch.float32, device=DEVICE)

        def call(rows):
            outs = [o[:rows] for o in sw_out]
            if rows <= g.SPLIT_DECODE_ROWS:
                view = p.view(-1)[:shards * rows * PART * parts].view(shards, rows, PART * parts)
                splitk[next((bm for bm in (16, 32) if rows <= bm), 64)](x[:rows], weight, view, counters, *outs)
            else:
                gemm(x[:rows], weight, *outs)
        return call

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

    # 3. Time, serving configuration against B12X.
    print(f"ratio {parts} {name}: us per call (both parts), warm / cold; serving shards "
          f"{layer.tilelang_shards}", flush=True)
    for rows in DECODE + PREFILL:
        base = per_call(lambda: b12x(rows), rows)
        port = per_call(lambda: tilelang(rows), rows)
        print(timing_line(rows, base, port), flush=True)
        if slower(base, port):
            failures.append(f"ratio {parts} rows {rows}: TileLang slower")

    # 4. Shard sweep: warm / cold per call, and error vs FP64 of part 0.
    print(f"ratio {parts} sweep: shards, blocked -> us warm/cold at rows 1, 2, 6, 16, 48, 96, 512, 8192; "
          "error max/rms", flush=True)
    ref0 = x.double() @ w_parts[0].double().T
    for shards in SWEEP:
        for blocked in (False, True):
            if (K // 64) % shards:
                continue
            call = configuration(shards, blocked)
            times = [per_call(lambda: call(rows), rows) for rows in (1, 2, 6, 16, 48, 96, 512, CAPACITY)]
            call(CAPACITY)
            torch.cuda.synchronize()
            err = errors(sw_out[0], ref0)
            print(f"  shards {shards:2d} blocked {int(blocked)}: " + "  ".join(f"{w:.2f}/{c:.2f}" for w, c in times)
                  + f"  error {err[0]:.3e}/{err[1]:.3e}", flush=True)
    del ref0


for parts, out_dtype in ((1, torch.bfloat16), (2, torch.float32)):
    with PreparationSession(device=DEVICE, autotune=False, compile_workers=2) as session:
        run(parts, out_dtype, session)
for failure in failures:
    print("FAIL", failure)
sys.exit(1 if failures else 0)
