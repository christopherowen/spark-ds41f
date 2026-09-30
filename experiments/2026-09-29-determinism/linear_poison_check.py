#!/usr/bin/env python3
"""Does a block-FP8 linear's output depend on stale scratch bytes? (GPU; cluster stopped)

usage: linear_poison_check.py [--quick]   (cwd: the B12X checkout)

DeepSeek V4.1 Flash shared-expert projections at TP3 with 32x32 block-FP8
weights, at decode row counts 1-8 and the capacities decode plans use. For
each case the prepared scratch is filled with different byte patterns before
each run; the output must be identical every time and equal the reference
built from the quantized operands.
"""
import sys

sys.path.insert(0, ".")

import torch  # noqa: E402

from b12x.gemm import block_fp8_linear as bfl  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from tests.gemm.test_gemm_block_fp8_linear import (  # noqa: E402
    _make_block_fp8_weight,
    _reference_from_quantized_operands,
)

QUICK = "--quick" in sys.argv
shapes = {"down": (5120, 768), "gate_up": (1536, 5120)}
cases = [(6, 6)] if QUICK else [(r, c) for c in (6, 8, 48) for r in (1, 3, 4, 5, 6) if r <= c]
torch.manual_seed(20260930)
failures = 0
for name, (n, k) in shapes.items():
    weight, scale = _make_block_fp8_weight(n, k, block_size=32)
    packed = bfl.pack_weight(weight, scale, block_size=(32, 32))
    plans = {}
    requests = []
    for capacity in sorted({c for _, c in cases}):
        caps = bfl.Caps(device="cuda", max_tokens=capacity, in_features=k, out_features=n,
                        source_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
                        block_size=(32, 32), output_mode="provided")
        plan = bfl.plan(caps)
        src = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16)

        def prepare(state, src=src):
            spec, = state.scratch.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device="cuda")
            out = torch.empty((src.shape[0], n, 1), dtype=torch.bfloat16, device="cuda")
            binding = state.bind(scratch=scratch, source=src, packed_weight=packed, output=out)
            return PreparedCall(run=lambda: state.run_binding(binding), output=out,
                                owners=(scratch, binding))

        plans[capacity] = plan
        requests.append(plan.request(name=f"{name}-{capacity}", prepare_call=prepare))
    with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
        session.prepare(tuple(requests))
        for rows, capacity in cases:
            plan = plans[capacity]
            source = (torch.randn(rows, k, device="cuda", dtype=torch.bfloat16) * 0.25).contiguous()
            expected = _reference_from_quantized_operands(source, weight, scale, block_size=32)
            spec, = plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device="cuda")
            output = torch.empty((rows, n, 1), dtype=torch.bfloat16, device="cuda")
            binding = bfl.bind(plan, scratch=scratch, source=source, packed_weight=packed,
                               output=output)
            outs = []
            for fill in (0, 255, 127, 64, None):
                if fill is None:
                    scratch.copy_(torch.randint(0, 256, scratch.shape, dtype=torch.uint8,
                                                device="cuda"))
                else:
                    scratch.fill_(fill)
                output.fill_(float("nan"))
                bfl.run(binding=binding)
                torch.cuda.synchronize()
                outs.append(output[:, :, 0].clone())
            same = all(torch.equal(o, outs[0]) for o in outs)
            exact = torch.equal(outs[0], expected)
            worst = max((o.float() - outs[0].float()).abs().max().item() for o in outs)
            ref_err = (outs[0].float() - expected.float()).abs().max().item()
            print(f"{name:7s} rows {rows} capacity {capacity}: same across poisons {same} "
                  f"(max diff {worst:.3g}), equals reference {exact} (max err {ref_err:.3g})",
                  flush=True)
            failures += (not same) or (not exact)
print("FAIL" if failures else "OK", flush=True)
sys.exit(1 if failures else 0)
