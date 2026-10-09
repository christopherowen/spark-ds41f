"""Port B5: the compressor's wkv/wgate projection (5120 -> 512 per part), B12X's
bf16_gemv against TileLang's BF16 GEMM through TileLangSplitLinearMethod, at the TP4
serving shapes: ratio 1 (wkv, BF16 out) and ratio 2 (wkv + wgate, FP32 out).

1. Error against an FP64 reference, no worse than B12X's (max and RMS).
2. Batch invariance: every row count gives the same rows of the 8192-row batch, bit for
   bit (TileLang gated; B12X reported).
3. Time per call (both parts) under CUDA graphs, warm and cold, at the decode capture
   sizes and prefill rows. For 72-96 rows (16 streams) it also times split-K at 32- and
   64-row tiles, which the serving path uses only up to 64 rows.

Exits 1 if a check fails or the serving path is more than 3% slower than B12X at any
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

from vllm.models.deepseek_v4_1.tilelang.linear import (  # noqa: E402
    TileLangSplitLinearMethod,
    project_parts,
)
from vllm.v1.worker.workspace import use_preallocated_workspace  # noqa: E402

K, PART = 5120, 512
failures = []


def run(parts, out_dtype, session):
    name = str(out_dtype).removeprefix("torch.")
    gen = torch.Generator(device=DEVICE).manual_seed(41 + parts)
    weight = (torch.randn((PART * parts, K), generator=gen, device=DEVICE) * 0.02).bfloat16()
    x = torch.randn((CAPACITY, K), generator=gen, device=DEVICE).bfloat16()
    w_parts = [weight[i * PART : (i + 1) * PART] for i in range(parts)]

    layer = nn.Module()
    layer.weight = nn.Parameter(weight, requires_grad=False)
    layer.out_dtype = out_dtype
    TileLangSplitLinearMethod(parts).process_weights_after_loading(layer)
    scratch = torch.empty(1 << 22, dtype=torch.uint8, device=DEVICE)

    def outputs():
        return [torch.empty((CAPACITY, PART), dtype=out_dtype, device=DEVICE) for _ in range(parts)]

    tl_out, b_out = outputs(), outputs()

    def tilelang(rows):
        with use_preallocated_workspace(scratch):
            project_parts(layer, x[:rows], tl_out, rows)

    # Serving plans every capture size; prefill chunks use the capacity plan.
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

    def splitk(block_M):
        def call(rows):
            for part, o in zip(layer.tilelang_parts, tl_out):
                p = scratch[: part.tilelang_shards * rows * PART * 4].view(torch.float32)
                p = p.view(part.tilelang_shards, rows, PART)
                part.tilelang_partials[block_M](x[:rows], part.weight, p)
                part.tilelang_reduce(p, o[:rows])
        return call

    # 1. Error against FP64 at the full batch.
    tilelang(CAPACITY)
    b12x(CAPACITY)
    torch.cuda.synchronize()
    full = [o.clone() for o in tl_out]
    b_full = [o.clone() for o in b_out]
    for i, w in enumerate(w_parts):
        ref = x.double() @ w.double().T
        port, base = errors(full[i], ref), errors(b_full[i], ref)
        ok = no_worse(port, base)
        print(f"ratio {parts} {name} part {i}: error vs fp64 max/rms  tilelang {port[0]:.3e}/{port[1]:.3e}"
              f"  b12x {base[0]:.3e}/{base[1]:.3e}  {'ok' if ok else 'WORSE'}", flush=True)
        if not ok:
            failures.append(f"ratio {parts} part {i} error")
        del ref

    # 2. Batch invariance.
    tl_varies, b12x_varies = [], set()
    for rows in DECODE + PREFILL[:-1]:
        tilelang(rows)
        b12x(rows)
        torch.cuda.synchronize()
        for i in range(parts):
            if not torch.equal(tl_out[i][:rows], full[i][:rows]):
                tl_varies.append(rows)
            if not torch.equal(b_out[i][:rows], b_full[i][:rows]):
                b12x_varies.add(rows)
    for block_M in (32, 64):
        for rows in (72, 80, 96):
            splitk(block_M)(rows)
            torch.cuda.synchronize()
            if not all(torch.equal(tl_out[i][:rows], full[i][:rows]) for i in range(parts)):
                tl_varies.append(f"{rows} (split-K {block_M})")
    if tl_varies:
        failures.append(f"ratio {parts}: TileLang differs from its full batch at {tl_varies}")
    print(f"ratio {parts}: TileLang differs from its full batch at {tl_varies or 'no'} sizes;"
          f" B12X at {sorted(b12x_varies) or 'no'}", flush=True)

    # 3. Time.
    print(f"ratio {parts} {name}: us per call (both parts), warm / cold", flush=True)
    for rows in DECODE + PREFILL:
        base = per_call(lambda: b12x(rows), rows)
        port = per_call(lambda: tilelang(rows), rows)
        line = timing_line(rows, base, port)
        if DECODE[-1] >= rows > 64:
            for block_M in (32, 64):
                w, c = per_call(lambda: splitk(block_M)(rows), rows)
                line += f"  splitk{block_M} {w:7.2f} / {c:7.2f}"
        print(line, flush=True)
        if slower(base, port):
            failures.append(f"ratio {parts} rows {rows}: TileLang slower")


for parts, out_dtype in ((1, torch.bfloat16), (2, torch.float32)):
    with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
        run(parts, out_dtype, session)
for failure in failures:
    print("FAIL", failure)
sys.exit(1 if failures else 0)
