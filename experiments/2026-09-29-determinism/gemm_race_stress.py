#!/usr/bin/env python3
"""Does the block-FP8 linear return wrong tiles beside co-resident work? (GPU; cluster stopped)

usage: gemm_race_stress.py [--rounds R] [--caps 6,8,40,48] [--shape down|gate_up]
                           (cwd: the B12X checkout)

Serving found the shared-expert down projection (6 rows, K 768 -> N 5120,
32x32 block-FP8, MXFP8 activations) wrong on a few 64-column tiles in ~5% of
calls while the routed MoE ran on the other stream; an immediate rerun was
right. This runs the prepared linear (quantize + GEMM, as bfl.run does) many
times on a side stream while the main stream runs, in turn: nothing; a
bandwidth-bound copy; the prepared B12X W4A8 routed MoE (DS4.1 TP3, top 6,
deterministic slice partials). Every output is compared on the device with
the result computed alone; the counts of wrong calls are printed per mode.
"""
import os
import sys
from contextlib import ExitStack, contextmanager

sys.path.insert(0, os.getcwd())
os.environ.setdefault("B12X_W4A8_TINY_DECODE", "0")
os.environ.setdefault("B12X_DYNAMIC_DETERMINISTIC_OUTPUT", "1")

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


ROUNDS = int(arg("--rounds", "20"))
CAPS = [int(c) for c in arg("--caps", "6,8,40,48").split(",")]
SHAPE = arg("--shape", "down")
K, N = {"down": (768, 5120), "gate_up": (5120, 1536)}[SHAPE]
GEMMS_PER_ROUND, MOE_PER_ROUND = 200, 24
E, H, I, TOPK, MOE_TOKENS = 384, 5120, 768, 6, 6
device = torch.device("cuda")
torch.manual_seed(20260930)


@contextmanager
def linear_plans(packed):
    plans, requests = {}, []
    for cap in CAPS:
        caps = bfl.Caps(device=device, max_tokens=cap, in_features=K, out_features=N,
                        source_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
                        block_size=(32, 32), output_mode="provided")
        plan = bfl.plan(caps)
        source = torch.randn(cap, K, device=device, dtype=torch.bfloat16)

        def prepare(state, source=source):
            spec, = state.scratch.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            out = torch.empty((source.shape[0], N, 1), dtype=torch.bfloat16, device=device)
            binding = state.bind(scratch=scratch, source=source, packed_weight=packed, output=out)
            return PreparedCall(run=lambda: state.run_binding(binding), output=out,
                                owners=(scratch, binding))

        plans[cap] = plan
        requests.append(plan.request(name=f"bfl-{cap}", prepare_call=prepare))
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(requests))
        yield plans


weight, scale = _make_block_fp8_weight(N, K, block_size=32)
packed = bfl.pack_weight(weight, scale, block_size=(32, 32))
moe_weights = make_synthetic_mxfp4_moe(E, H, I, seed=7, device=device)
moe_plan = fused_moe.plan_weights(
    source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"),
                                  w13_layout=fused_moe.W13Layout("w13")),
    activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A8, nonlinearity="silu",
                                        io_dtype=torch.bfloat16),
    geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I),
)
experts = fused_moe.prepare_weights(plan=moe_plan, weights=fused_moe.PackedWeights(
    w13=moe_weights["w13_fp4"], w2=moe_weights["w2_fp4"], w13_block_scales=moe_weights["w13_mx"],
    w2_block_scales=moe_weights["w2_mx"], w13_global_scales=moe_weights["alphas"],
    w2_global_scales=moe_weights["alphas"], input_scale=moe_weights["input_scale"],
    intermediate_scale=moe_weights["input_scale"], immutable_input_scales=True))
gen = torch.Generator(device=device).manual_seed(606)
moe_x = (torch.randn(MOE_TOKENS, H, generator=gen, device=device) * 2.0).to(torch.bfloat16)
logits = torch.randn(MOE_TOKENS, E, generator=gen, device=device)
top_logits, topk_ids = torch.topk(logits, TOPK, dim=-1)
topk_weights = torch.softmax(top_logits, dim=-1).float().contiguous()
topk_ids = topk_ids.to(torch.int32).contiguous()
moe_out = torch.zeros(MOE_TOKENS, H, dtype=torch.bfloat16, device=device)
hog_src = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device)
hog_dst = torch.empty_like(hog_src)
side = torch.cuda.Stream()
main = torch.cuda.current_stream()

with ExitStack() as stack:
    plans = stack.enter_context(linear_plans(packed))
    moe_binding = stack.enter_context(make_tp_moe_fp4_binding(
        a=moe_x, experts=experts, topk_weights=topk_weights, topk_ids=topk_ids,
        output=moe_out, input_scales_static=True, quant_mode="w4a8_mx"))
    failures = 0
    for cap in CAPS:
        rows = cap
        source = (torch.randn(rows, K, device=device, dtype=torch.bfloat16) * 0.25).contiguous()
        spec, = plans[cap].scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        out = torch.empty((rows, N, 1), dtype=torch.bfloat16, device=device)
        binding = bfl.bind(plans[cap], scratch=scratch, source=source, packed_weight=packed,
                           output=out)
        bfl.run(binding=binding)
        torch.cuda.synchronize()
        ref = out.clone()
        stable = all(bfl.run(binding=binding) is not None and torch.equal(out, ref)
                     for _ in range(5))
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(200):
            bfl.run(binding=binding)
        start.record()
        for _ in range(2000):
            bfl.run(binding=binding)
        stop.record()
        torch.cuda.synchronize()
        alone_us = start.elapsed_time(stop) * 1000 / 2000
        results = {}
        for mode in ("alone", "copy", "moe"):
            wrong = torch.zeros((), dtype=torch.int64, device=device)
            tiles = torch.zeros(N // 64, dtype=torch.int64, device=device)
            for _ in range(ROUNDS):
                side.wait_stream(main)
                if mode == "copy":
                    for _ in range(4):
                        hog_dst.copy_(hog_src)
                elif mode == "moe":
                    for _ in range(MOE_PER_ROUND):
                        _impl.b12x_moe_fp4(binding=moe_binding)
                with torch.cuda.stream(side):
                    for _ in range(GEMMS_PER_ROUND):
                        bfl.run(binding=binding)
                        bad = torch.ne(out, ref)[:, :, 0]
                        wrong += bad.any()
                        tiles += bad.view(rows, -1, 64).any(2).any(0)
                main.wait_stream(side)
            torch.cuda.synchronize()
            results[mode] = (int(wrong), ROUNDS * GEMMS_PER_ROUND,
                             tiles.nonzero().flatten().tolist()[:16])
            failures += int(wrong)
        line = "; ".join(f"{m}: wrong {w}/{t}" + (f" tiles {tl}" if tl else "")
                         for m, (w, t, tl) in results.items())
        print(f"{SHAPE} cap {cap} rows {rows}: alone-stable {stable}, {alone_us:.2f} us/call alone "
              f"(quantize + GEMM, launch-bound); {line}", flush=True)
    print("FAIL" if failures else "OK", flush=True)
    sys.exit(1 if failures else 0)
