#!/usr/bin/env python3
"""Replay real rows through the prefill-eligible GEMVs under candidate backends: bits and timings.

usage: gemv_capture_replay.py CAPTURE_DIR   (GPU, cluster stopped; cwd the B12X checkout, checkpoint
       at /models, transition_map.py importable)

Inputs: the mHC captures (make_overlay_mhccap.py) run through production's mHC once; pre's output
(the attention input) feeds the compressors, post_pre's (the MoE input) the router gate, so the
rows are one serving prefill's real hidden states (one rank's share, about 1275 rows; larger
batches repeat them). Weights from the checkpoint. Shapes: router gate 384x5120 (FP32 out),
ratio-2 compressor 1024x5120 (FP32 out), ratio-1 compressor 512x5120 (BF16 out). Backends per
capacity (serving's GEMV capacities):
- production: b12x defaults (SIMT up to 192 rows, the TMA prefill kernel from 256; torch for the
  BF16 output above 8 rows);
- ref2 (vllm-0032): SIMT at every capacity;
- prefill: the TMA prefill kernel at every capacity;
- mma: the MMA kernel at every capacity.
For each: rows grouped by exact bits as the last or first row of 1..4096-row batches, then
timings (median of 40; decode sizes from a CUDA graph, prefill sizes eager on pre-gathered rows;
1334 and 1366 rows are one rank's share of a 4000- and 4096-token chunk).
"""
import glob
import json
import os
import sys
import types

import torch

sys.path.insert(0, "/r")
import transition_map as tm  # noqa: E402

from b12x.gemm import bf16_gemv  # noqa: E402
from b12x.gemm.bf16_gemv._tuning import GemvConfig  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from vllm.models.deepseek_v4_1 import b12x_layers  # noqa: E402
from vllm.utils.b12x import B12xWorkload, register_b12x_layer  # noqa: E402
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402

device = tm.device
H = 5120
GRAPH = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48)
CAPS = tm.GEMV_CAPS
TIMED = (1, 2, 4, 6, 8, 16, 48, 128, 256, 1334, 1366, 2048, 4000)
init_workspace_manager(device)
b12x_layers._execution_capacities = lambda: (1, 8, 4096)
b12x_layers._BATCH_INVARIANT = False

# Real rows: production mHC on the captured inputs.
rows = {}
modules, requests = {}, []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "mhc-*.pt"))):
    c = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in torch.load(path).items()}
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
    key = f"gemv-rows-{os.path.basename(path)}"
    register_b12x_layer(key, owner)
    module.bind_layer_name(key)
    workload = B12xWorkload(stage="weights", token_counts=(*GRAPH, 4096), fixed_token_counts=GRAPH,
                            output_dtype=torch.bfloat16, max_tokens=4096, max_seqs=8, max_model_len=4096)
    requests += [r for u in module.get_b12x_preparation_units(owner, workload) for r in u.requests]
    modules[path] = (c, module)
with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
    session.prepare(tuple(requests))
    for path, (c, module) in modules.items():
        if c["operation"] == "pre":
            out = module.pre(c["residual"], c["fn"], c["scale"], c["base"], c["norm"], c["pre"])
        else:
            out = module.post_pre(c["previous_output"], c["residual"], c["previous_post"], c["previous_comb"],
                                  c["fn"], c["scale"], c["base"], c["norm"], c["pre"])
        rows.setdefault(c["operation"], []).append(out[3].clone())  # y
pools = {op: torch.cat(v) for op, v in rows.items()}
print(json.dumps({"rows": {op: p.shape[0] for op, p in pools.items()}}), flush=True)

compressor2 = torch.cat([tm.bf16("layers.2.attn.compressor.wkv.weight"),
                         tm.bf16("layers.2.attn.compressor.wgate.weight")]).contiguous()
SHAPES = {
    "router gate 384x5120 fp32": (tm.bf16("layers.2.ffn.gate.weight"), torch.float32, "post_pre"),
    "compressor ratio-2 1024x5120 fp32": (compressor2, torch.float32, "pre"),
    "compressor ratio-1 512x5120 bf16": (tm.bf16("layers.20.attn.compressor.wkv.weight"), torch.bfloat16, "pre"),
}
MODES = {"production": None, "ref2": GemvConfig(backend="simt"), "prefill": GemvConfig(backend="prefill"),
         "mma": GemvConfig(backend="mma")}
plans, requests = {}, []
for label, (weight, out_dtype, _) in SHAPES.items():
    for mode, override in MODES.items():
        for cap in CAPS:
            query = bf16_gemv.GemvQuery(source_dtype="bfloat16", weight_dtype="bfloat16",
                                        output_dtype=str(out_dtype).removeprefix("torch."), max_rows=cap,
                                        in_features=H, out_features=weight.shape[0], source_contiguous=True,
                                        source_aligned=True, weight_contiguous=True, weight_aligned=True)
            try:
                plan = bf16_gemv.plan(query, override=override)
            except Exception as error:  # noqa: BLE001
                print(json.dumps({"op": label, "mode": mode, "cap": cap, "plan_error": repr(error)[:160]}))
                continue

            def make_call(state, weight=weight):
                q = state.query
                source = torch.empty((q.max_rows, q.in_features), device=device, dtype=torch.bfloat16)
                out = torch.empty((q.max_rows, q.out_features), device=device, dtype=getattr(torch, q.output_dtype))
                return PreparedCall(run=lambda: state.run(source, weight, out=out),
                                    produce=lambda: source.normal_(std=0.25), owners=(weight,))
            plans[label, mode, cap] = plan
            requests.append(plan.request(name=f"{label}-{mode}-{cap}", prepare_call=make_call,
                                         benchmark_call=make_call))


def timed(fn, x):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn(x)
    torch.cuda.current_stream().wait_stream(side)
    run = lambda: fn(x)  # noqa: E731
    if x.shape[0] <= 48:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(x)
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


with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
    session.prepare(tuple(requests))
    for label, (weight, out_dtype, source) in SHAPES.items():
        pool = pools[source]
        n = pool.shape[0]
        big = pool[torch.arange(4096, device=device) % n].contiguous()
        for mode in MODES:
            caps = [c for c in CAPS if (label, mode, c) in plans]
            if not caps:
                continue
            configs = {}
            for c in caps:
                configs.setdefault(str(plans[label, mode, c].selection.config), []).append(c)
            print(json.dumps({"op": label, "mode": mode, "configs": configs}), flush=True)

            def fn(x, label=label, mode=mode, caps=caps, weight=weight, out_dtype=out_dtype):
                return bf16_gemv.mm(x.contiguous(), weight, plan=plans[label, mode, tm.smallest(caps, x.shape[0])],
                                    output_dtype=out_dtype)
            tm.LIMIT = 4096
            tm.SIZES = sorted(set(range(1, 73)) | {c + d for c in (*CAPS, 1334, 1366) for d in (-1, 0, 1)
                                                   if 1 <= c + d <= 4096})
            tm.transitions(f"gemv {label} [{mode}]", fn, H, make=lambda m, g, big=big: big[:m])
            print(json.dumps({"timing_us": f"{label} [{mode}]",
                              **{str(r): timed(fn, big[:r].contiguous()) for r in TIMED}}), flush=True)
print("gemv replay done", flush=True)
