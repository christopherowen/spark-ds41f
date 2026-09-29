#!/usr/bin/env python3
"""Does one prepared W4A8 MoE launch repeat bit for bit? (GPU; cluster stopped)

usage: moe_repeat_check.py [--no-slices] [--only M] [--reps N] [--no-graph]
       (cwd: the B12X checkout)

DeepSeek V4.1 Flash per-rank geometry at TP3 (384 experts, hidden 5120,
intermediate 768, top 6), canonical prepared weights, deterministic output,
tiny decode off, as served. For each token count it prepares one binding,
launches eagerly five times and replays a CUDA graph three times, and reports
whether every output equals the first and the route-output rows planned.
Before every launch the route-output rows are poisoned with NaN, so a row the
top-k sum reads but no kernel wrote makes the output non-finite. --no-slices
disables slice partials (the round-2 detfast behaviour) for an A/B; --only,
--reps and --no-graph shrink the run for compute-sanitizer. Each token count
also runs a history check: launch input A, a different input B in the same
binding, then A again; A's two outputs must be equal, or the launch depends on
state an earlier launch left behind (as serving's varied history would).
"""
import os
import sys

os.environ["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"
os.environ["B12X_W4A8_TINY_DECODE"] = "0"
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

from b12x.moe import fused_moe  # noqa: E402
from b12x.moe.fused_moe import _impl  # noqa: E402
from benchmarks.benchmark_ds4_moe import make_synthetic_mxfp4_moe  # noqa: E402
from tests._reference.helpers import make_tp_moe_fp4_binding  # noqa: E402

if "--no-slices" in sys.argv:
    _impl._DETERMINISTIC_SLICE_PARTIAL_MAX_ROWS = 0
ONLY = int(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None
REPS = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 5
GRAPH = "--no-graph" not in sys.argv
E, K, N, TOPK = 384, 5120, 768, 6
device = torch.device("cuda")
weights = make_synthetic_mxfp4_moe(E, K, N, seed=7, device=device)
plan = fused_moe.plan_weights(
    source=fused_moe.PackedSource(
        format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"),
        w13_layout=fused_moe.W13Layout("w13"),
    ),
    activation=fused_moe.ActivationSpec(
        mode=fused_moe.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16
    ),
    geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=K, intermediate_size=N),
)
prepared = fused_moe.prepare_weights(
    plan=plan,
    weights=fused_moe.PackedWeights(
        w13=weights["w13_fp4"],
        w2=weights["w2_fp4"],
        w13_block_scales=weights["w13_mx"],
        w2_block_scales=weights["w2_mx"],
        w13_global_scales=weights["alphas"],
        w2_global_scales=weights["alphas"],
        input_scale=weights["input_scale"],
        intermediate_scale=weights["input_scale"],
        immutable_input_scales=True,
    ),
)
failures = 0
for m in (1, 3, 5, 6, 16, 48):
    if ONLY is not None and m != ONLY:
        continue
    gen = torch.Generator(device=device).manual_seed(100 + m)
    x = (torch.randn(m, K, generator=gen, device=device) * 2.0).to(torch.bfloat16)
    logits = torch.randn(m, E, generator=gen, device=device)
    top_logits, topk_ids = torch.topk(logits, TOPK, dim=-1)
    topk_weights = torch.softmax(top_logits, dim=-1).float().contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    output = torch.zeros(m, K, dtype=torch.bfloat16, device=device)
    with make_tp_moe_fp4_binding(
        a=x, experts=prepared, topk_weights=topk_weights, topk_ids=topk_ids,
        output=output, input_scales_static=True, quant_mode="w4a8_mx",
    ) as binding:
        def launch():
            _impl.b12x_moe_fp4(binding=binding)

        outs = []
        for _ in range(REPS):
            output.fill_(float("nan"))
            binding.route_output.fill_(float("nan"))
            launch()
            torch.cuda.synchronize()
            outs.append(output.clone())
        graph = None
        if GRAPH:
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.cuda.graph(graph):
                launch()
            torch.cuda.current_stream().wait_stream(stream)
            for _ in range(3):
                output.fill_(float("nan"))
                binding.route_output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                outs.append(output.clone())
        # History: A, then B (other tokens and routes), then A again.
        x_a, ids_a, w_a = x.clone(), topk_ids.clone(), topk_weights.clone()
        output.fill_(float("nan"))
        launch()
        torch.cuda.synchronize()
        history_a = output.clone()
        x.mul_(-0.75)
        topk_ids.add_(1).remainder_(E)
        topk_weights.copy_(topk_weights.flip(-1))
        output.fill_(float("nan"))
        launch()
        torch.cuda.synchronize()
        x.copy_(x_a)
        topk_ids.copy_(ids_a)
        topk_weights.copy_(w_a)
        output.fill_(float("nan"))
        launch()
        torch.cuda.synchronize()
        history_same = torch.equal(output, history_a)
        history_diff = (output.float() - history_a.float()).abs().max().item()
        first = outs[0]
        same = [torch.equal(o, first) for o in outs]
        worst = max((o.float() - first.float()).abs().max().item() for o in outs)
        finite = all(bool(torch.isfinite(o).all()) for o in outs)
        rows = tuple(binding.route_output.shape)
        print(f"m={m:3d} route_output={rows} finite={finite} repeats={same} max_diff={worst:.3g} "
              f"history_same={history_same} history_diff={history_diff:.3g}", flush=True)
        failures += (not all(same)) or (not finite) or (not history_same)
        del graph
print("FAIL" if failures else "OK", flush=True)
sys.exit(1 if failures else 0)
