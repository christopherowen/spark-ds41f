#!/usr/bin/env python3
"""Kernel time of the block-FP8 linear with identical inputs, alone and beside the MoE.

usage: gemm_fence_timing.py [--caps 6,48] [--calls 400]   (cwd: the B12X checkout)

Run once in the r5m image (shipped dense GEMM) and once in the r5n image
(0004's fenced GEMM): everything else is identical, including inputs (fixed
seeds). For the DS4.1 TP3 shared-expert projections (down 768 -> 5120,
gate_up 5120 -> 1536) at each capacity, the linear runs CALLS times alone and
CALLS times on a side stream while the main stream runs the prepared W4A8
routed MoE (atomic combine, r5n's). The profiler reports the dense GEMM and
quantize kernel durations (median and mean, microseconds) in both settings.
"""
import os
import statistics
import sys
from contextlib import ExitStack, contextmanager

sys.path.insert(0, os.getcwd())
os.environ.setdefault("B12X_W4A8_TINY_DECODE", "0")

import torch  # noqa: E402

from b12x.gemm import block_fp8_linear as bfl  # noqa: E402
from b12x.moe import fused_moe  # noqa: E402
from b12x.moe.fused_moe import _impl  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from benchmarks.benchmark_ds4_moe import make_synthetic_mxfp4_moe  # noqa: E402
from tests._reference.helpers import make_tp_moe_fp4_binding  # noqa: E402
from tests.gemm.test_gemm_block_fp8_linear import _make_block_fp8_weight  # noqa: E402


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


CAPS = [int(c) for c in arg("--caps", "6,48").split(",")]
CALLS = int(arg("--calls", "400"))
device = torch.device("cuda")
SHAPES = {"down": (768, 5120), "gate_up": (5120, 1536)}


@contextmanager
def plans_for(packed_by_shape):
    plans, requests = {}, []
    for shape, (k, n) in SHAPES.items():
        for cap in CAPS:
            caps = bfl.Caps(device=device, max_tokens=cap, in_features=k, out_features=n,
                            source_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
                            block_size=(32, 32), output_mode="provided")
            plan = bfl.plan(caps)
            source = torch.randn(cap, k, device=device, dtype=torch.bfloat16)
            packed = packed_by_shape[shape]

            def prepare(state, source=source, packed=packed, n=n):
                spec, = state.scratch.scratch_specs()
                scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
                out = torch.empty((source.shape[0], n, 1), dtype=torch.bfloat16, device=device)
                binding = state.bind(scratch=scratch, source=source, packed_weight=packed, output=out)
                return PreparedCall(run=lambda: state.run_binding(binding), output=out,
                                    owners=(scratch, binding))

            plans[(shape, cap)] = plan
            requests.append(plan.request(name=f"{shape}-{cap}", prepare_call=prepare))
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(requests))
        yield plans


def kernel_times(prof):
    out = {"dense_gemm": [], "quantize": []}
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        key = ("dense_gemm" if "DenseGemm" in event.name else
               "quantize" if "MXFP8RowsQuant" in event.name else None)
        if key:
            out[key].append(event.device_time if hasattr(event, "device_time") else event.cuda_time)
    return out


torch.manual_seed(20260930)
packed = {}
for shape, (k, n) in SHAPES.items():
    w, s = _make_block_fp8_weight(n, k, block_size=32)
    packed[shape] = bfl.pack_weight(w, s, block_size=(32, 32))
E, H, I, TOPK, M = 384, 5120, 768, 6, 6
weights = make_synthetic_mxfp4_moe(E, H, I, seed=7, device=device)
moe_plan = fused_moe.plan_weights(
    source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"),
                                  w13_layout=fused_moe.W13Layout("w13")),
    activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A8, nonlinearity="silu",
                                        io_dtype=torch.bfloat16),
    geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I),
)
experts = fused_moe.prepare_weights(plan=moe_plan, weights=fused_moe.PackedWeights(
    w13=weights["w13_fp4"], w2=weights["w2_fp4"], w13_block_scales=weights["w13_mx"],
    w2_block_scales=weights["w2_mx"], w13_global_scales=weights["alphas"],
    w2_global_scales=weights["alphas"], input_scale=weights["input_scale"],
    intermediate_scale=weights["input_scale"], immutable_input_scales=True))
gen = torch.Generator(device=device).manual_seed(606)
x = (torch.randn(M, H, generator=gen, device=device) * 2.0).to(torch.bfloat16)
logits = torch.randn(M, E, generator=gen, device=device)
top_logits, ids = torch.topk(logits, TOPK, dim=-1)
tw = torch.softmax(top_logits, dim=-1).float().contiguous()
ids = ids.to(torch.int32).contiguous()
moe_out = torch.zeros(M, H, dtype=torch.bfloat16, device=device)
side = torch.cuda.Stream()
main = torch.cuda.current_stream()

with ExitStack() as stack:
    plans = stack.enter_context(plans_for(packed))
    moe = stack.enter_context(make_tp_moe_fp4_binding(
        a=x, experts=experts, topk_weights=tw, topk_ids=ids, output=moe_out,
        input_scales_static=True, quant_mode="w4a8_mx"))
    for (shape, cap), plan in plans.items():
        k, n = SHAPES[shape]
        gsrc = torch.Generator(device=device).manual_seed(1000 + cap)
        source = (torch.randn(cap, k, generator=gsrc, device=device) * 0.25).to(torch.bfloat16)
        spec, = plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        out = torch.empty((cap, n, 1), dtype=torch.bfloat16, device=device)
        binding = bfl.bind(plan, scratch=scratch, source=source, packed_weight=packed[shape], output=out)
        for _ in range(20):
            bfl.run(binding=binding)
        torch.cuda.synchronize()
        result = {}
        for mode in ("alone", "beside_moe"):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                done = 0
                while done < CALLS:
                    side.wait_stream(main)
                    if mode == "beside_moe":
                        for _ in range(8):
                            _impl.b12x_moe_fp4(binding=moe)
                    with torch.cuda.stream(side):
                        for _ in range(50):
                            bfl.run(binding=binding)
                    main.wait_stream(side)
                    done += 50
                torch.cuda.synchronize()
            times = kernel_times(prof)
            result[mode] = {k2: (statistics.median(v), statistics.mean(v), len(v)) for k2, v in times.items() if v}
        line = "; ".join(
            f"{mode}: " + ", ".join(f"{k2} median {m:.2f} mean {a:.2f} us (n {c})" for k2, (m, a, c) in r.items())
            for mode, r in result.items())
        print(f"{shape} cap {cap}: {line}", flush=True)
