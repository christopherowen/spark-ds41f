#!/usr/bin/env python3
"""Replay captured mHC inputs under each plan configuration: bits across batch sizes, agreement, timings.

usage: mhc_capture_replay.py CAPTURE_DIR   (GPU, cluster stopped; cwd the B12X checkout, the
       candidate b12x_layers.py (vllm-0039) mounted over vLLM's, transition_map.py importable)

Inputs are the rows rank 0 saw in one serving prefill (make_overlay_mhccap.py): residual
streams, previous output and mixes, and the layer's own mHC weights, for pre and post_pre at
three layers. Configurations:
- production: graph-size plans and the 4096-row plan, b12x defaults (the 4096 plan is TF32 TMA);
- ref2 (vllm-0033): only the 4096-row plan, its default TF32 TMA configuration;
- candidate (vllm-0039): every capacity on the lagged native route (the graph-size plans' own);
- candidate-capacity: only the 4096-row plan, on the lagged native route.
For each: rows grouped by exact bits as the last or first row of 1..N-row batches (N the
captured rows); the candidate against production at decode sizes (bit equality) and against
production and ref2 at 1024 rows (largest relative difference per output); timings (median
of 40) on pre-gathered inputs, decode sizes replayed from a CUDA graph, prefill sizes eager
(1334 and 1366 rows are one rank's share of a 4000- and 4096-token chunk).
"""
import glob
import json
import os
import sys
import types

import torch

sys.path.insert(0, "/r")
import transition_map as tm  # noqa: E402

from vllm.models.deepseek_v4_1 import b12x_layers  # noqa: E402
from vllm.utils.b12x import B12xWorkload, register_b12x_layer  # noqa: E402
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402
from b12x.preparation import PreparationSession  # noqa: E402

device = tm.device
H = 5120
GRAPH = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48)
LIMIT = 4096
TIMED = (1, 2, 4, 6, 8, 16, 32, 48, 128, 256, 1024, 1334, 1366, 2048, 4000)
init_workspace_manager(device)
b12x_layers._execution_capacities = lambda: (1, 8, LIMIT)
captures = {}
for path in sorted(glob.glob(os.path.join(sys.argv[1], "mhc-*.pt"))):
    c = torch.load(path)
    captures[(c["layer"].rsplit(".", 1)[-1], c["operation"])] = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c.items()}
print(json.dumps({"captures": {f"{k[0]} {k[1]}": v["residual"].shape[0] for k, v in captures.items()}}), flush=True)
CONFIGS = {"production": (False, GRAPH), "ref2": (False, ()), "candidate": (True, GRAPH),
           "candidate-capacity": (True, ())}


def build(name):
    invariant, fixed = CONFIGS[name]
    b12x_layers._BATCH_INVARIANT = invariant
    modules, requests = {}, []
    for (layer, op), c in captures.items():
        owner = torch.nn.Module()
        for part in ("attn", "ffn"):
            setattr(owner, f"hc_{part}_fn", c["fn"].float().contiguous())
            setattr(owner, f"hc_{part}_scale", c["scale"].float().contiguous())
            setattr(owner, f"hc_{part}_base", c["base"].float().contiguous())
            setattr(owner, f"{part}_norm", types.SimpleNamespace(weight=c["norm"]))
        owner.hc_attn_fn_broadcast = None
        module = b12x_layers.B12xMHC(types.SimpleNamespace(hidden_size=H, rms_norm_eps=1e-20, hc_eps=1e-6,
                                                           hc_sinkhorn_iters=20, hc_mult=4))
        owner._b12x_mhc = module
        key = f"mhc-replay-{name}-{layer}-{op}"
        register_b12x_layer(key, owner)
        module.bind_layer_name(key)
        workload = B12xWorkload(stage="weights", token_counts=tuple(sorted({*fixed, LIMIT})),
                                fixed_token_counts=fixed, output_dtype=torch.bfloat16,
                                max_tokens=LIMIT, max_seqs=8, max_model_len=LIMIT)
        requests += [r for u in module.get_b12x_preparation_units(owner, workload) for r in u.requests]
        modules[(layer, op)] = (owner, module)
    return modules, requests


def runner(module, c, op):
    def call(idx):
        take = lambda key: c[key][idx].contiguous()  # noqa: E731
        if op == "pre":
            out = module.pre(take("residual"), c["fn"], c["scale"], c["base"], c["norm"], take("pre"))
        else:
            out = module.post_pre(take("previous_output"), take("residual"), take("previous_post"),
                                  take("previous_comb"), c["fn"], c["scale"], c["base"], c["norm"], take("pre"))
        return [t for t in out if torch.is_tensor(t)]
    return call


def timed(module, c, op, rows, n):
    """Median microseconds of one call on pre-gathered inputs; decode sizes replay a CUDA graph."""
    idx = torch.arange(rows, device=device) % n
    inputs = {k: c[k][idx].contiguous() for k in ("residual", "pre", "previous_output", "previous_post",
                                                  "previous_comb") if c.get(k) is not None}

    def once():
        if op == "pre":
            return module.pre(inputs["residual"], c["fn"], c["scale"], c["base"], c["norm"], inputs["pre"])
        return module.post_pre(inputs["previous_output"], inputs["residual"], inputs["previous_post"],
                               inputs["previous_comb"], c["fn"], c["scale"], c["base"], c["norm"], inputs["pre"])

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            once()
    torch.cuda.current_stream().wait_stream(side)
    run = once
    if rows <= 48:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            once()
        run = graph.replay
    times = []
    for _ in range(40):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return round(sorted(times)[len(times) // 2], 1)


outputs = {}
for name in CONFIGS:
    modules, requests = build(name)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(requests))
        (layer0, op0), (owner0, module0) = next(iter(modules.items()))
        print(json.dumps({"config": name, "plans": {f"{op} {cap}": str(p.selection.config)[:160]
                                                    for (op, cap), p in sorted(module0._plans.items())
                                                    if cap in (1, 48, LIMIT)}}), flush=True)
        for (layer, op), (owner, module) in modules.items():
            c = captures[(layer, op)]
            n = c["residual"].shape[0]
            call = runner(module, c, op)
            tm.LIMIT = n
            tm.SIZES = sorted(set(range(1, 73)) | {x + d for x in (*GRAPH, 96, 128, 192, 256, 384, 512, 768, 1024,
                                                                1536, 2048) for d in (-1, 0, 1) if 1 <= x + d <= n})
            tm.transitions(f"mhc {name} layer {layer} {op}", call, 1, make=lambda m, g: torch.arange(m, device=device))
            for rows in (*GRAPH, 1024):
                if rows <= n:
                    outputs[(name, layer, op, rows)] = [t.clone() for t in call(torch.arange(rows, device=device))]
            print(json.dumps({"timing_us": f"{name} layer {layer} {op}",
                              **{str(r): timed(module, c, op, r, n) for r in TIMED}}), flush=True)
    torch.cuda.synchronize()

for (layer, op) in captures:
    equal = {rows: all(torch.equal(a, b) for a, b in zip(outputs[("candidate", layer, op, rows)],
                                                         outputs[("production", layer, op, rows)]))
             for rows in GRAPH if ("candidate", layer, op, rows) in outputs}

    def rel(a, b):
        return max(float(((x.float() - y.float()).abs().max() / y.float().abs().max().clamp(min=1e-30)))
                   for x, y in zip(a, b))
    big = {}
    for other in ("production", "ref2"):
        if ("candidate", layer, op, 1024) in outputs and (other, layer, op, 1024) in outputs:
            big[other] = rel(outputs[("candidate", layer, op, 1024)], outputs[(other, layer, op, 1024)])
    print(json.dumps({"agreement": f"layer {layer} {op}", "candidate_equals_production_at": sorted(r for r, e in equal.items() if e),
                      "differs_at": sorted(r for r, e in equal.items() if not e),
                      "max_rel_diff_at_1024": {k: f"{v:.2e}" for k, v in big.items()}}), flush=True)
print("mhc replay done", flush=True)
