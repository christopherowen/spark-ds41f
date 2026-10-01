#!/usr/bin/env python3
"""Replay captured mHC inputs under each plan configuration: bits across batch sizes, agreement, timings.

usage: mhc_capture_replay.py CAPTURE_DIR   (GPU, cluster stopped; cwd the B12X checkout, the
       candidate b12x_layers.py (vllm-0039) mounted over vLLM's, transition_map.py importable)

Inputs are the rows rank 0 saw in one serving prefill (make_overlay_mhccap.py): residual
streams, previous output and mixes, and the layer's own mHC weights, for pre and post_pre at
three layers. Configurations:
- production: graph-size plans and the 4096-row plan, b12x defaults (the 4096 plan is TF32 TMA);
- ref2 (vllm-0033): only the 4096-row plan, its default TF32 TMA configuration;
- candidate (vllm-0039): every capacity on the lagged native route (the graph-size plans' own),
  the 4096-row plan computing 13 partial sums per CTA; candidate-p4 groups them by 4 (same
  arithmetic: they must agree bit for bit);
- candidate-tN (B12X 0008, overlay mhc-mt): the capacity plan computes N tokens per CTA (13 partial
  sums each), which must agree bit for bit with the candidate;
- tf32-sN (MHC_REPLAY_TF32=1): every capacity on the TF32 TMA projection with N K slices (16-row
  tiles), one arithmetic at every count if the slices are summed in a fixed order.
MHC_REPLAY_SET=sweep replaces the set with production, the candidate, candidate-t16 and its 9- and
25-partial groupings (candidate-t16-pP, which must equal the candidate bit for bit), tf32-s40, and
tf32x-* variants: 40 K slices with one launch geometry (tile_m.tile_n.n_warps.tile_k.stages) for the
graph-size plans and another for the capacity plan, which must equal tf32-s40 bit for bit if the
geometry only partitions the work.
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
# name: (batch-invariant mode, fixed graph-size plans, capacity-plan partial sums per CTA, or a TF32 split count)
CONFIGS = {"production": (False, GRAPH, None), "ref2": (False, (), None), "candidate": (True, GRAPH, 13),
           "candidate-p4": (True, GRAPH, 4),
           **{f"candidate-t{n}": (True, GRAPH, f"mt:{n}") for n in (4, 8, 16)}}
if os.environ.get("MHC_REPLAY_TF32"):
    CONFIGS.update({f"tf32-s{n}": (True, GRAPH, f"tf32:{n}") for n in (8, 16, 32, 40, 64)})
SWEEP_TF32 = {"a": ("16.8.1.64.2", "64.24.1.64.2"), "b": ("16.24.3.64.2", "32.24.1.64.2"),
              "c": ("16.8.1.128.2", "64.24.3.64.2"), "d": ("16.24.3.128.3", "64.24.1.128.2"),
              "e": ("16.8.1.32.4", "32.8.1.64.2")}
if os.environ.get("MHC_REPLAY_SET") == "seq":
    # Sequential partial sums (overlay mhc-seq): one thread per (token, partial) summing the tile
    # in a fixed order. Every tokens-per-CTA setting must give the same bits as one per CTA.
    CONFIGS = {"production": (False, GRAPH, None), "candidate": (True, GRAPH, 13),
               **{f"seq-t{n}": (True, GRAPH, f"seq:{n}") for n in (1, 8, 16)}}
if os.environ.get("MHC_REPLAY_SET") == "sweep":
    CONFIGS = {"production": (False, GRAPH, None), "candidate": (True, GRAPH, 13),
               "candidate-t16": (True, GRAPH, "mt:16"),
               **{f"candidate-t16-p{p}": (True, GRAPH, f"mt:16:{p}") for p in (9, 25)},
               "tf32-s40": (True, GRAPH, "tf32:40"),
               **{f"tf32x-{k}": (True, GRAPH, f"tf32x:40:{d}/{c}") for k, (d, c) in SWEEP_TF32.items()}}
real_config = b12x_layers._mhc_batch_invariant_config


def tf32(splits, geometry="16.24.1.64.2"):
    """TF32 TMA projection: `splits` K slices; by default 16-row tiles, all 24 mixes per tile."""
    from b12x.norm.mhc._tuning import MhcConfig
    tm_, tn, nn, tk, st = (int(v) for v in geometry.split("."))
    return MhcConfig(backend="tf32_tma", projection_tile_m=tm_, projection_tile_n=tn, projection_tile_k=tk,
                     projection_num_stages=st, projection_num_m_warps=tm_ // 16, projection_num_n_warps=nn,
                     projection_k_splits=splits)


def build(name):
    invariant, fixed, partials = CONFIGS[name]
    b12x_layers._BATCH_INVARIANT = invariant

    def config(tokens, partials=partials):
        if isinstance(partials, str) and partials.startswith("mt:"):
            chosen = real_config(tokens)
            parts = partials.split(":")
            if chosen is not None and tokens >= 96:  # B12X 0008: several tokens per CTA at capacity
                chosen = type(chosen)(**{**chosen.to_dict(), "partials_per_cta": int(parts[2]) if len(parts) > 2 else 13,
                                         "tokens_per_cta": int(parts[1])})
            return chosen
        if isinstance(partials, str) and partials.startswith("seq:"):
            from b12x.norm.mhc._tuning import MhcConfig
            return MhcConfig(backend="native", projection_tile_m=16, projection_tile_n=8, projection_tile_k=256,
                             projection_num_stages=1, projection_num_m_warps=1, projection_num_n_warps=1,
                             projection_k_splits=1, lagged_prepare=True, partials_per_cta=4,
                             tokens_per_cta=max(1, min(int(partials.split(":")[1]), tokens)),
                             sequential_partials=True)
        if isinstance(partials, str) and partials.startswith("tf32x:"):
            _, splits, geometries = partials.split(":")
            decode, capacity = geometries.split("/")
            return tf32(int(splits), decode if tokens <= 48 else capacity)
        if isinstance(partials, str):
            return tf32(int(partials.split(":")[1]))
        chosen = real_config(tokens)
        if chosen is not None and tokens >= 96:
            chosen = type(chosen)(**{**chosen.to_dict(), "partials_per_cta": partials})
        return chosen
    b12x_layers._mhc_batch_invariant_config = config
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


def kernels(module, c, op, rows, n, calls=20):
    """Mean device microseconds per call of each kernel one call launches (CUPTI activity trace)."""
    idx = torch.arange(rows, device=device) % n
    inputs = {k: c[k][idx].contiguous() for k in ("residual", "pre", "previous_output", "previous_post",
                                                  "previous_comb") if c.get(k) is not None}

    def once():
        if op == "pre":
            return module.pre(inputs["residual"], c["fn"], c["scale"], c["base"], c["norm"], inputs["pre"])
        return module.post_pre(inputs["previous_output"], inputs["residual"], inputs["previous_post"],
                               inputs["previous_comb"], c["fn"], c["scale"], c["base"], c["norm"], inputs["pre"])

    once()
    torch.cuda.synchronize()
    run = once
    if rows <= 48:  # as timed(): decode sizes replay a CUDA graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            once()
        run = graph.replay
        run()
        torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            run()
            torch.cuda.synchronize()
    totals, spans = {}, []
    events = sorted((e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                    key=lambda e: e.time_range.start)
    for event in events:
        key = event.name.replace("kernel_cutlass_kernel_b12xnormmhc_kernels", "")[:40]
        totals[key] = totals.get(key, 0.0) + event.device_time
    per_call = len(events) // calls if calls else 0
    gaps = []
    for i in range(calls):
        group = events[i * per_call:(i + 1) * per_call]
        if group:
            spans.append(group[-1].time_range.end - group[0].time_range.start)
            gaps += [b.time_range.start - a.time_range.end for a, b in zip(group, group[1:])]
    result = {k: round(v / calls, 1) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])}
    if spans:
        result["span"] = round(sorted(spans)[len(spans) // 2], 1)
        result["gap"] = round(sorted(gaps)[len(gaps) // 2], 1) if gaps else 0.0
    return result


KERNEL_ROWS = [int(r) for r in os.environ.get("MHC_REPLAY_KERNEL_ROWS", "").split(",") if r]
outputs = {}
for name in CONFIGS:
    modules, requests = build(name)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        try:
            session.prepare(tuple(requests))
        except Exception as error:  # a configuration B12X rejects: report it, keep the others
            print(json.dumps({"config": name, "error": f"{type(error).__name__}: {error}"[:300]}), flush=True)
            continue
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
            if layer == "1" and op == "post_pre":
                for rows in KERNEL_ROWS:
                    print(json.dumps({"kernels_us": f"{name} layer {layer} {op} rows {rows}",
                                      **kernels(module, c, op, rows, n)}), flush=True)
    torch.cuda.synchronize()

for (layer, op) in captures:
    equal = {rows: all(torch.equal(a, b) for a, b in zip(outputs[("candidate", layer, op, rows)],
                                                         outputs[("production", layer, op, rows)]))
             for rows in GRAPH if ("candidate", layer, op, rows) in outputs}

    def rel(a, b):
        return max(float(((x.float() - y.float()).abs().max() / y.float().abs().max().clamp(min=1e-30)))
                   for x, y in zip(a, b))
    same = {other: all(torch.equal(a, b) for a, b in zip(outputs[(other, layer, op, 1024)],
                                                         outputs[("candidate", layer, op, 1024)]))
            for other in ("candidate-p4", "candidate-t4", "candidate-t8", "candidate-t16")
            if (other, layer, op, 1024) in outputs}
    big = {}
    for other in ("production", "ref2"):
        if ("candidate", layer, op, 1024) in outputs and (other, layer, op, 1024) in outputs:
            big[other] = rel(outputs[("candidate", layer, op, 1024)], outputs[(other, layer, op, 1024)])
    print(json.dumps({"agreement": f"layer {layer} {op}", "candidate_equals_production_at": sorted(r for r, e in equal.items() if e),
                      "differs_at": sorted(r for r, e in equal.items() if not e),
                      "max_rel_diff_at_1024": {k: f"{v:.2e}" for k, v in big.items()},
                      "candidate_bit_equal_at_1024": same}), flush=True)
for (layer, op) in captures:
    for name in CONFIGS:
        base = ("tf32-s40" if name.startswith("tf32x-") else "candidate" if name.startswith("candidate-t16")
                else "seq-t1" if name.startswith("seq-") and name != "seq-t1" else None)
        if base is None or base not in CONFIGS:
            continue
        rows = [r for r in (*GRAPH, 1024) if (name, layer, op, r) in outputs and (base, layer, op, r) in outputs]
        equal = [r for r in rows if all(torch.equal(a, b) for a, b in zip(outputs[(name, layer, op, r)],
                                                                         outputs[(base, layer, op, r)]))]
        print(json.dumps({"bits": f"{name} vs {base} layer {layer} {op}", "equal_at": equal,
                          "differs_at": [r for r in rows if r not in equal]}), flush=True)
for (layer, op) in captures:
    for name in [n for n in CONFIGS if n.startswith("seq-")]:
        diffs = {}
        for other in ("production", "candidate"):
            if (name, layer, op, 1024) in outputs and (other, layer, op, 1024) in outputs:
                a, b = outputs[(name, layer, op, 1024)], outputs[(other, layer, op, 1024)]
                diffs[other] = f"{max(float((x.float() - y.float()).abs().max() / y.float().abs().max().clamp(min=1e-30)) for x, y in zip(a, b)):.2e}"
        print(json.dumps({"numerics": f"{name} layer {layer} {op} at 1024 rows", "max_rel_diff": diffs}), flush=True)
print("mhc replay done", flush=True)
