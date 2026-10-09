"""Port B6: the DSpark drafter's context-KV projection, rows 1280: (512) of the fused
Q-A/KV block-32 FP8 weight (1792 x 5120), B12X's block_fp8_linear prepared as
_ContextKVProjection prepares it, against TileLangBlock32Rows.

1. Error against the BF16 activations times the dequantized weight in FP64, no worse
   than B12X's (max and RMS). Both quantize the activations to MXFP8 (E4M3, one UE8M0
   scale per 32); the error includes that rounding, whichever way each side does it.
2. Batch invariance: every row count gives the same rows of the 8192-row batch, bit for
   bit (TileLang gated; B12X reported).
3. Time per call under CUDA graphs, warm and cold, at the decode capture sizes and
   prefill rows.

Exits 1 if a check fails or TileLang is more than 3% slower than B12X at any shape.
"""
import sys

import torch
from torch import nn

sys.path.insert(0, "/b")
from kbench import (  # noqa: E402
    CAPACITY, DECODE, DEVICE, PREFILL, errors, no_worse, per_call, slower, timing_line,
)

from b12x.gemm import block_fp8_linear  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.linear import TileLangBlock32Rows  # noqa: E402
from vllm.v1.worker.workspace import use_preallocated_workspace  # noqa: E402

Q_LORA, KV, K = 1280, 512, 5120
failures = []

gen = torch.Generator(device=DEVICE).manual_seed(61)
weight = (torch.randn((Q_LORA + KV, K), generator=gen, device=DEVICE) * 0.5).to(torch.float8_e4m3fn)
exps = torch.randint(120, 125, ((Q_LORA + KV) // 32, K // 32), generator=gen, device=DEVICE).to(torch.uint8)
x = torch.randn((CAPACITY, K), generator=gen, device=DEVICE).bfloat16()

fused = nn.Module()
fused.weight, fused.weight_scale_inv = weight, exps.view(torch.float8_e8m0fnu)
rows_gemm = TileLangBlock32Rows(fused, Q_LORA)
scratch = torch.empty(64 << 20, dtype=torch.uint8, device=DEVICE)
tl_out = {}


def tilelang(rows):
    with use_preallocated_workspace(scratch):
        tl_out[rows] = rows_gemm(x[:rows])


# B12X as _ContextKVProjection: the KV rows packed once, one plan per capacity bound.
packed = block_fp8_linear.pack_weight(weight[Q_LORA:], fused.weight_scale_inv[Q_LORA // 32 :], block_size=(32, 32))
plans, scratches, requests = {}, {}, []
for bound in DECODE + (CAPACITY,):
    plans[bound] = block_fp8_linear.plan(block_fp8_linear.Caps(
        device=DEVICE, max_tokens=bound, in_features=K, out_features=KV, block_size=(32, 32),
        output_mode="provided"))

    def prepare(state, bound=bound):
        source = torch.zeros((bound, K), dtype=torch.bfloat16, device=DEVICE)
        output = torch.empty((bound, KV, 1), dtype=torch.bfloat16, device=DEVICE)
        scratches[bound] = [torch.empty(s.shape, dtype=s.dtype, device=DEVICE) for s in state.scratch.scratch_specs()]
        binding = state.bind(scratch=scratches[bound], source=source, packed_weight=packed, output=output)
        return PreparedCall(run=lambda: state.run_binding(binding))

    requests.append(plans[bound].request(name=f"dspark.context_kv.m{bound}", prepare_call=prepare))
b_out = {}


def b12x(rows):
    bound = rows if rows <= DECODE[-1] else CAPACITY
    out = torch.empty((rows, KV), dtype=torch.bfloat16, device=DEVICE)
    binding = block_fp8_linear.bind(plans[bound], scratch=scratches[bound], source=x[:rows],
                                    packed_weight=packed, output=out.view(rows, KV, 1))
    block_fp8_linear.run(binding=binding)
    b_out[rows] = out


with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
    session.prepare(tuple(requests))

    # 1. Error against FP64 at the full batch.
    tilelang(CAPACITY)
    b12x(CAPACITY)
    torch.cuda.synchronize()
    w_scale = torch.exp2(exps[Q_LORA // 32 :].double() - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)
    ref = x.double() @ (weight[Q_LORA:].double() * w_scale).T
    port, base = errors(tl_out[CAPACITY], ref), errors(b_out[CAPACITY], ref)
    ok = no_worse(port, base)
    print(f"error vs fp64 max/rms  tilelang {port[0]:.3e}/{port[1]:.3e}  b12x {base[0]:.3e}/{base[1]:.3e}"
          f"  {'ok' if ok else 'WORSE'}", flush=True)
    if not ok:
        failures.append("error")
    full, b_full = tl_out[CAPACITY], b_out[CAPACITY]
    del ref, w_scale

    # 2. Batch invariance.
    tl_varies, b12x_varies = [], []
    for rows in DECODE + PREFILL[:-1]:
        tilelang(rows)
        b12x(rows)
        torch.cuda.synchronize()
        if not torch.equal(tl_out[rows], full[:rows]):
            tl_varies.append(rows)
        if not torch.equal(b_out[rows], b_full[:rows]):
            b12x_varies.append(rows)
    if tl_varies:
        failures.append(f"TileLang differs from its full batch at {tl_varies}")
    print(f"TileLang differs from its full batch at {tl_varies or 'no'} sizes; B12X at {b12x_varies or 'no'}",
          flush=True)

    # 3. Time.
    print("us per call, warm / cold", flush=True)
    for rows in DECODE + PREFILL:
        base = per_call(lambda: b12x(rows), rows)
        port = per_call(lambda: tilelang(rows), rows)
        print(timing_line(rows, base, port), flush=True)
        if slower(base, port):
            failures.append(f"rows {rows}: TileLang slower")

for failure in failures:
    print("FAIL", failure)
sys.exit(1 if failures else 0)
