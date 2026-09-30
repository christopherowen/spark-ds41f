#!/usr/bin/env python3
"""Routed-MoE combine cost at fixed verification shapes (GPU; cluster stopped).

usage: moe_combine_bench.py --mode atomic|collapsed|slices [--rows 6,12,24,36,48]
                            [--dead 0,0.25,0.5] [--ncu]   (cwd: the B12X checkout)

DS4.1 TP3 routed MoE (384 experts, hidden 5120, intermediate 768, top 6,
W4A8) at each verification row count, with a fraction of whole rows dead
(top-k ids -1, as dead verification rows arrive). Modes: atomic (r5n's combine),
collapsed (deterministic, one task per M tile), slices (deterministic, 0006's
slice partials and 36-row top-k sum). Work is held fixed by construction: the
same routing for every mode. For each shape, 20 calls are captured in a CUDA
graph and replayed; the profiler splits the time between the fused MoE kernel
and the top-k sum. --ncu makes three eager calls per shape instead, for
Nsight Compute's DRAM and L2 byte counters.
"""
import json
import os
import sys
from contextlib import ExitStack

sys.path.insert(0, os.getcwd())


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


MODE = arg("--mode", "slices")
ROWS = [int(r) for r in arg("--rows", "6,12,24,36,48").split(",")]
DEAD = [float(d) for d in arg("--dead", "0,0.25,0.5").split(",")]
NCU = "--ncu" in sys.argv
os.environ.setdefault("B12X_W4A8_TINY_DECODE", "0")
if MODE != "atomic":
    os.environ["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"

import torch  # noqa: E402

from b12x.moe import fused_moe  # noqa: E402
from b12x.moe.fused_moe import _impl  # noqa: E402
from benchmarks.benchmark_ds4_moe import make_synthetic_mxfp4_moe  # noqa: E402
from tests._reference.helpers import make_tp_moe_fp4_binding  # noqa: E402

if MODE == "collapsed":
    _impl._DETERMINISTIC_SLICE_PARTIAL_MAX_ROWS = 0
E, H, I, TOPK = 384, 5120, 768, 6
CALLS, REPLAYS = 20, 10
device = torch.device("cuda")
weights = make_synthetic_mxfp4_moe(E, H, I, seed=7, device=device)
plan = fused_moe.plan_weights(
    source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"),
                                  w13_layout=fused_moe.W13Layout("w13")),
    activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A8, nonlinearity="silu",
                                        io_dtype=torch.bfloat16),
    geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I),
)
experts = fused_moe.prepare_weights(plan=plan, weights=fused_moe.PackedWeights(
    w13=weights["w13_fp4"], w2=weights["w2_fp4"], w13_block_scales=weights["w13_mx"],
    w2_block_scales=weights["w2_mx"], w13_global_scales=weights["alphas"],
    w2_global_scales=weights["alphas"], input_scale=weights["input_scale"],
    intermediate_scale=weights["input_scale"], immutable_input_scales=True))


def routing(rows, dead, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(rows, H, generator=gen, device=device) * 2.0).to(torch.bfloat16)
    logits = torch.randn(rows, E, generator=gen, device=device)
    top_logits, ids = torch.topk(logits, TOPK, dim=-1)
    w = torch.softmax(top_logits, dim=-1).float().contiguous()
    ids = ids.to(torch.int32)
    n_dead = int(round(rows * dead))
    if n_dead:
        ids[rows - n_dead:] = -1          # dead verification rows: the batch tail
    return x, w, ids.contiguous()


results = []
for rows in ROWS:
    for dead in DEAD:
        x, w, ids = routing(rows, dead, 1000 + rows)
        out = torch.zeros(rows, H, dtype=torch.bfloat16, device=device)
        with ExitStack() as stack:
            binding = stack.enter_context(make_tp_moe_fp4_binding(
                a=x, experts=experts, topk_weights=w, topk_ids=ids, output=out,
                input_scales_static=True, quant_mode="w4a8_mx"))
            _impl.b12x_moe_fp4(binding=binding)
            torch.cuda.synchronize()
            if NCU:
                for _ in range(3):
                    _impl.b12x_moe_fp4(binding=binding)
                torch.cuda.synchronize()
                print(json.dumps({"mode": MODE, "rows": rows, "dead": dead, "ncu": True}), flush=True)
                continue
            graph = torch.cuda.CUDAGraph()
            capture = torch.cuda.Stream()
            capture.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture):
                _impl.b12x_moe_fp4(binding=binding)
                with torch.cuda.graph(graph, stream=capture):
                    for _ in range(CALLS):
                        _impl.b12x_moe_fp4(binding=binding)
            torch.cuda.current_stream().wait_stream(capture)
            graph.replay()
            torch.cuda.synchronize()
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(REPLAYS):
                graph.replay()
            stop.record()
            torch.cuda.synchronize()
            per_call = start.elapsed_time(stop) * 1000 / (REPLAYS * CALLS)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
            kernels = {}
            for event in prof.key_averages():
                if event.device_type != torch.autograd.DeviceType.CUDA:
                    continue
                name = ("topk_sum" if "TopKSum" in event.key else
                        "fused_moe" if "MoEDynamic" in event.key else "other")
                total = getattr(event, "device_time_total", None) or event.cuda_time_total
                kernels[name] = kernels.get(name, 0.0) + total / (3 * CALLS)
            record = {"mode": MODE, "rows": rows, "dead": dead, "us_per_call": round(per_call, 2),
                      **{f"{k}_us": round(v, 2) for k, v in sorted(kernels.items())}}
            results.append(record)
            print(json.dumps(record), flush=True)
            del graph
