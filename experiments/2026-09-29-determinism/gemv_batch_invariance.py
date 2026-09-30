#!/usr/bin/env python3
"""Do the router gate and shared-expert linears give a row the same bits at every batch size?
(GPU; cluster stopped)

usage: gemv_batch_invariance.py   (cwd: the B12X checkout)

The V4.1 router gate is a B12X BF16 GEMV (5120 -> 384, FP32 out) with one
prepared plan per row capacity; serving calls the plan for the batch's padded
row count. Nineteen fixed target rows (the traced JSON prefill) run through it
at capacities 19-48 with random neighbours after them, and with neighbours
before them, compared bitwise with the 19-row result. The same sweep runs for
the block-FP8 shared-expert projections (gate_up 5120 -> 1536, down 768 -> 5120)
as a control; serving showed them matching.
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

from b12x.gemm import bf16_gemv, block_fp8_linear as bfl  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from tests.gemm.test_gemm_block_fp8_linear import _make_block_fp8_weight  # noqa: E402

CAPS = (19, 20, 24, 28, 32, 40, 48)
T = 19
device = torch.device("cuda")
torch.manual_seed(20260930)


def gemv_plans(weight):
    plans, requests = {}, []
    for cap in CAPS:
        query = bf16_gemv.GemvQuery(
            source_dtype="bfloat16", weight_dtype="bfloat16", output_dtype="float32",
            max_rows=cap, in_features=weight.shape[1], out_features=weight.shape[0],
            source_contiguous=True, source_aligned=True, weight_contiguous=True, weight_aligned=True)
        plan = bf16_gemv.plan(query)

        def make_call(state, weight=weight):
            q = state.query
            source = torch.empty((q.max_rows, q.in_features), device=device, dtype=torch.bfloat16)
            out = torch.empty((q.max_rows, q.out_features), device=device, dtype=torch.float32)
            return PreparedCall(run=lambda: state.run(source, weight, out=out),
                                produce=lambda: source.normal_(std=0.25), owners=(weight,))

        plans[cap] = plan
        requests.append(plan.request(name=f"gate-{cap}", prepare_call=make_call, benchmark_call=make_call))
    return plans, requests


def fp8_plans(n, k, packed):
    plans, requests = {}, []
    for cap in CAPS:
        caps = bfl.Caps(device=device, max_tokens=cap, in_features=k, out_features=n,
                        source_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
                        block_size=(32, 32), output_mode="provided")
        plan = bfl.plan(caps)
        src = torch.randn(cap, k, device=device, dtype=torch.bfloat16)

        def prepare(state, src=src, n=n):
            spec, = state.scratch.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            out = torch.empty((src.shape[0], n, 1), dtype=torch.bfloat16, device=device)
            binding = state.bind(scratch=scratch, source=src, packed_weight=packed, output=out)
            return PreparedCall(run=lambda: state.run_binding(binding), output=out, owners=(scratch, binding))

        plans[cap] = plan
        requests.append(plan.request(name=f"fp8-{n}-{cap}", prepare_call=prepare))
    return plans, requests


gate_w = (torch.randn(384, 5120, device=device) * 0.02).to(torch.bfloat16)
gate_plans, reqs = gemv_plans(gate_w)
fp8 = {}
for name, (n, k) in {"gate_up": (1536, 5120), "down": (5120, 768)}.items():
    w, s = _make_block_fp8_weight(n, k, block_size=32)
    packed = bfl.pack_weight(w, s, block_size=(32, 32))
    plans, r = fp8_plans(n, k, packed)
    fp8[name] = (n, k, packed, plans)
    reqs += r
with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
    session.prepare(tuple(reqs))

    def gate(x, cap):
        return bf16_gemv.mm(x, gate_w, plan=gate_plans[cap], output_dtype=torch.float32)

    def linear(name, x, cap):
        n, k, packed, plans = fp8[name]
        spec, = plans[cap].scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        out = torch.empty((x.shape[0], n, 1), dtype=torch.bfloat16, device=device)
        bfl.run(binding=bfl.bind(plans[cap], scratch=scratch, source=x, packed_weight=packed, output=out))
        return out[:, :, 0]

    for label, fn, k in (("router gate (bf16 gemv 5120->384)", gate, 5120),
                         ("shared gate_up (block-FP8 5120->1536)", lambda x, c: linear("gate_up", x, c), 5120),
                         ("shared down (block-FP8 768->5120)", lambda x, c: linear("down", x, c), 768)):
        target = (torch.randn(T, k, device=device) * 0.5).to(torch.bfloat16)
        base = fn(target.contiguous(), 19).clone()
        torch.cuda.synchronize()
        results = []
        for cap in CAPS[1:]:
            nb = (torch.randn(cap - T, k, device=device) * 0.5).to(torch.bfloat16)
            after = fn(torch.cat([target, nb]).contiguous(), cap)[:T]
            before = fn(torch.cat([nb, target]).contiguous(), cap)[cap - T:]
            torch.cuda.synchronize()
            results.append(f"{cap}: after {'=' if torch.equal(after, base) else 'DIFF'} "
                           f"before {'=' if torch.equal(before, base) else 'DIFF'}")
        print(f"{label}: " + "; ".join(results), flush=True)
