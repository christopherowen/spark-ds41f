#!/usr/bin/env python3
"""Routed MoE on the main stream beside the shared-expert chain on a side stream.

usage: overlap_repro.py [--reps N] [--no-slices] [--atomic]   (cwd: the B12X checkout)

Mirrors vLLM's decode overlap for DeepSeek V4.1 Flash at TP3, one decode step
(6 rows): the main stream runs the prepared B12X W4A8 routed MoE (384 experts,
hidden 5120, intermediate 768, top 6); a side stream, forked after the input
is ready, runs the shared expert: block-FP8 gate_up (5120 -> 1536), SiLU-and-mul,
block-FP8 down (768 -> 5120), allocating its intermediates as vLLM does. The
main stream joins before reading the shared output.
Modes: shared expert alone; overlapped eagerly; overlapped inside one captured
CUDA graph. Every shared output must equal the alone result bit for bit.
"""
import os
import sys
from contextlib import ExitStack, contextmanager

sys.path.insert(0, os.getcwd())
os.environ["B12X_W4A8_TINY_DECODE"] = "0"
if "--atomic" not in sys.argv:
    os.environ["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from b12x.gemm import block_fp8_linear as bfl  # noqa: E402
from b12x.moe import fused_moe  # noqa: E402
from b12x.moe.fused_moe import _impl  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from benchmarks.benchmark_ds4_moe import make_synthetic_mxfp4_moe  # noqa: E402
from tests._reference.helpers import make_tp_moe_fp4_binding  # noqa: E402
from tests.gemm.test_gemm_block_fp8_linear import _make_block_fp8_weight  # noqa: E402

if "--no-slices" in sys.argv:
    _impl._DETERMINISTIC_SLICE_PARTIAL_MAX_ROWS = 0
REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 30
E, K, N, TOPK, M, SHARED = 384, 5120, 768, 6, 6, 768
device = torch.device("cuda")


@contextmanager
def linear_plans(specs):
    """Prepare every block-FP8 linear in one session, left unfrozen.

    Freezing is process-wide; the routed MoE binding prepared afterwards
    freezes once everything has compiled.
    """
    plans, requests = [], []
    for in_features, out_features, packed in specs:
        caps = bfl.Caps(device=device, max_tokens=M, in_features=in_features,
                        out_features=out_features, source_dtype=torch.bfloat16,
                        output_dtype=torch.bfloat16, block_size=(32, 32), output_mode="provided")
        plan = bfl.plan(caps)
        source = torch.randn(M, in_features, device=device, dtype=torch.bfloat16)

        def prepare(state, source=source, packed=packed, out_features=out_features):
            spec, = state.scratch.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            output = torch.empty((M, out_features, 1), dtype=torch.bfloat16, device=device)
            binding = state.bind(scratch=scratch, source=source, packed_weight=packed, output=output)
            return PreparedCall(run=lambda: state.run_binding(binding), output=output,
                                owners=(scratch, binding))

        plans.append(plan)
        requests.append(plan.request(name=f"bfl-{out_features}", prepare_call=prepare))
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(requests))
        yield plans


def linear(plan, packed, x, scratch):
    """vLLM's V4.1 block32 apply: a fresh output, one shared scratch lease."""
    out = torch.empty((x.shape[0], packed.out_features, 1), dtype=torch.bfloat16, device=device)
    binding = bfl.bind(plan, scratch=scratch, source=x, packed_weight=packed, output=out)
    bfl.run(binding=binding)
    return out[:, :, 0]


torch.manual_seed(20260930)
w13, s13 = _make_block_fp8_weight(2 * SHARED, K, block_size=32)
w2, s2 = _make_block_fp8_weight(K, SHARED, block_size=32)
p13 = bfl.pack_weight(w13, s13, block_size=(32, 32))
p2 = bfl.pack_weight(w2, s2, block_size=(32, 32))
weights = make_synthetic_mxfp4_moe(E, K, N, seed=7, device=device)
plan = fused_moe.plan_weights(
    source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"),
                                  w13_layout=fused_moe.W13Layout("w13")),
    activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A8, nonlinearity="silu",
                                        io_dtype=torch.bfloat16),
    geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=K, intermediate_size=N),
)
experts = fused_moe.prepare_weights(plan=plan, weights=fused_moe.PackedWeights(
    w13=weights["w13_fp4"], w2=weights["w2_fp4"], w13_block_scales=weights["w13_mx"],
    w2_block_scales=weights["w2_mx"], w13_global_scales=weights["alphas"],
    w2_global_scales=weights["alphas"], input_scale=weights["input_scale"],
    intermediate_scale=weights["input_scale"], immutable_input_scales=True))

gen = torch.Generator(device=device).manual_seed(606)
x = (torch.randn(M, K, generator=gen, device=device) * 2.0).to(torch.bfloat16)
logits = torch.randn(M, E, generator=gen, device=device)
top_logits, topk_ids = torch.topk(logits, TOPK, dim=-1)
topk_weights = torch.softmax(top_logits, dim=-1).float().contiguous()
topk_ids = topk_ids.to(torch.int32).contiguous()
routed_out = torch.zeros(M, K, dtype=torch.bfloat16, device=device)
side = torch.cuda.Stream()

with ExitStack() as stack:
    plan13, plan2 = stack.enter_context(linear_plans([(K, 2 * SHARED, p13), (SHARED, K, p2)]))
    binding = stack.enter_context(make_tp_moe_fp4_binding(
        a=x, experts=experts, topk_weights=topk_weights, topk_ids=topk_ids,
        output=routed_out, input_scales_static=True, quant_mode="w4a8_mx"))
    scratch_bytes = max(plan13.scratch_specs()[0].nbytes, plan2.scratch_specs()[0].nbytes)
    shared_scratch = torch.empty(scratch_bytes, dtype=torch.uint8, device=device)

    def shared_expert():
        gate_up = linear(plan13, p13, x, shared_scratch[: plan13.scratch_specs()[0].nbytes])
        act = F.silu(gate_up[:, :SHARED]) * gate_up[:, SHARED:]
        return linear(plan2, p2, act, shared_scratch[: plan2.scratch_specs()[0].nbytes])

    def overlapped():
        ready = torch.cuda.Event()
        ready.record()
        side.wait_event(ready)
        x.record_stream(side)
        with torch.cuda.stream(side):
            shared = shared_expert()
        _impl.b12x_moe_fp4(binding=binding)          # routed MoE on the main stream
        torch.cuda.current_stream().wait_stream(side)
        shared.record_stream(torch.cuda.current_stream())
        return shared

    reference = shared_expert().clone()
    torch.cuda.synchronize()
    routed_ref = None
    results = {"alone": [], "overlap-eager": [], "overlap-graph": []}
    for _ in range(REPS):
        results["alone"].append(shared_expert().clone())
    for _ in range(REPS):
        shared = overlapped()
        torch.cuda.synchronize()
        results["overlap-eager"].append(shared.clone())
        if routed_ref is None:
            routed_ref = routed_out.clone()
    graph = torch.cuda.CUDAGraph()
    capture = torch.cuda.Stream()
    capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture):
        overlapped()  # warm the capture stream's allocations
        with torch.cuda.graph(graph, stream=capture):
            graph_shared = overlapped()
    torch.cuda.current_stream().wait_stream(capture)
    routed_equal = 0
    for _ in range(REPS):
        graph.replay()
        torch.cuda.synchronize()
        results["overlap-graph"].append(graph_shared.clone())
        routed_equal += int(torch.equal(routed_out, routed_ref))
    failures = 0
    for mode, outs in results.items():
        equal = sum(torch.equal(o, reference) for o in outs)
        worst = max((o.float() - reference.float()).abs().max().item() for o in outs)
        print(f"{mode:14s} shared equal to alone: {equal}/{len(outs)}  max diff {worst:.3g}", flush=True)
        failures += equal != len(outs)
    print(f"routed output in graph equal to eager: {routed_equal}/{REPS}", flush=True)
    print("FAIL" if failures else "OK", flush=True)
    sys.exit(1 if failures else 0)
