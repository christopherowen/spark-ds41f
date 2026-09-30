#!/usr/bin/env python3
"""Does the smallest-capacity lookup make the V4.1 GEMV projections batch-invariant?
(GPU; cluster stopped)

usage: gemv_lookup_validation.py JSON_GATES PROSE_GATES GATE_WEIGHTS   (cwd: the B12X checkout,
                                                                        checkpoint at /models)

The inputs are real. JSON_GATES and PROSE_GATES are the attn-trace overlay's
captures of the first router gate call after a reset on rank 0, taken during
the solo JSON (19 rows) and prose (15 rows) prefills: every layer's gate input
rows, the FP32 logits serving returned, and the prepared capacities.
GATE_WEIGHTS holds the served gate weights.

1. Fidelity: each captured layer's rows through the plan the smallest-capacity
   lookup picks must reproduce serving's logits bitwise, and each row must
   equal the same row computed alone.
2. Sweep: the JSON prefill's rows at one layer in batches of R rows (1-64 and
   larger, around every prepared capacity), at the start, middle and end,
   the other captured rows as neighbours. Each target row is compared
   bitwise with that row alone; the old lookup (the exact row count, else the
   largest plan) runs beside the new one.
3. The same sweep for the other unquantized projections, with checkpoint
   weights and the captured rows as inputs: the index head weights (32 x
   5120, BF16 out), the ratio-2 compressor (512 x 5120, FP32 out), the
   ratio-1 compressor (512 x 5120, BF16 out) and the index key (128 x 512,
   BF16 out; the captured rows' first 512 columns stand in for latents).
3b. The block-FP8 query projection wq_b (layer 0, rank 0's 22 heads: 1280 ->
   11264), which already takes the smallest capacity: the same sweep with
   serving's capacities, and each capacity's dense GEMM configuration.
4. Kernel time per call at a few row counts, old and new plan (CUDA graph
   of 50 calls, replayed 10 times, so launch overhead is excluded).
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from b12x.gemm import bf16_gemv  # noqa: E402
from b12x.gemm import block_fp8_linear as bfl  # noqa: E402
from b12x.gemm.bf16_gemv._tuning import default_config  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402

JSON_GATES, PROSE_GATES, WEIGHTS = sys.argv[1:4]
device = torch.device("cuda")
json_gates, prose_gates = torch.load(JSON_GATES), torch.load(PROSE_GATES)
gate_weights = torch.load(WEIGHTS)
main = [n for n in json_gates if n.startswith("language_model.") and json_gates[n]["taken"]]
names = [n for n in json_gates if json_gates[n]["taken"]]
CAPS = tuple(json_gates[main[0]]["capacities"])
print(f"prepared capacities {CAPS}; lookup in serving: {json_gates[main[0]]['lookup']}; "
      f"{len(names)} gates captured (JSON rows {json_gates[main[0]]['rows']}, "
      f"prose rows {prose_gates[main[0]]['rows']})", flush=True)


def new_cap(rows):
    return next(c for c in CAPS if rows <= c)


def old_cap(rows):
    return rows if rows in CAPS else CAPS[-1]


def checkpoint(name):
    index = json.load(open("/models/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"/models/{index[name]}", framework="pt", device="cuda") as f:
        return f.get_tensor(name).to(torch.bfloat16).contiguous()


compressor2 = torch.cat([checkpoint("layers.2.attn.compressor.wkv.weight"),
                         checkpoint("layers.2.attn.compressor.wgate.weight")])
SHAPES = {  # name -> (weight, output dtype)
    "router gate (384x5120, FP32 out)": (gate_weights[main[0]].to(device), torch.float32),
    "index head weights (32x5120, BF16 out)": (checkpoint("layers.2.attn.indexer.weights_proj.weight"),
                                               torch.bfloat16),
    "compressor ratio 2 (512x5120, FP32 out)": (compressor2[:512].contiguous(), torch.float32),
    "compressor ratio 1 (512x5120, BF16 out)": (checkpoint("layers.20.attn.compressor.wkv.weight")[:512]
                                                .contiguous(), torch.bfloat16),
    "index key (128x512, BF16 out)": (checkpoint("layers.2.attn.indexer.wk.weight"), torch.bfloat16),
}
drafter = [n for n in names if not n.startswith("language_model.")]
if drafter:
    w = gate_weights[drafter[0]].to(device)
    SHAPES[f"drafter gate ({w.shape[0]}x{w.shape[1]}, FP32 out)"] = (w, torch.float32)

plans, requests = {}, []
for label, (weight, out_dtype) in SHAPES.items():
    for cap in CAPS:
        query = bf16_gemv.GemvQuery(
            source_dtype="bfloat16", weight_dtype="bfloat16",
            output_dtype=str(out_dtype).removeprefix("torch."), max_rows=cap,
            in_features=weight.shape[1], out_features=weight.shape[0],
            source_contiguous=True, source_aligned=True, weight_contiguous=True, weight_aligned=True)
        plan = bf16_gemv.plan(query)

        def make_call(state, weight=weight):
            q = state.query
            source = torch.empty((q.max_rows, q.in_features), device=device, dtype=torch.bfloat16)
            out = torch.empty((q.max_rows, q.out_features), device=device, dtype=getattr(torch, q.output_dtype))
            return PreparedCall(run=lambda: state.run(source, weight, out=out),
                                produce=lambda: source.normal_(std=0.25), owners=(weight,))

        plans[label, cap] = plan
        requests.append(plan.request(name=f"{label}-{cap}", prepare_call=make_call, benchmark_call=make_call))
    backends = {}
    for cap in CAPS:
        backend = default_config(plans[label, cap].query, device).backend
        backends.setdefault(backend, []).append(cap)
    print(f"{label}: backends by capacity {backends}", flush=True)


def checkpoint_raw(name):
    index = json.load(open("/models/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"/models/{index[name]}", framework="pt", device="cuda") as f:
        return f.get_tensor(name)


wq_b = checkpoint_raw("layers.0.attn.wq_b.weight")[:11264].contiguous()
wq_b_scale = checkpoint_raw("layers.0.attn.wq_b.scale")[:11264 // 32].float().contiguous()
wq_b_packed = bfl.pack_weight(wq_b, wq_b_scale, block_size=(32, 32))
fp8_plans = {}
for cap in CAPS:
    caps = bfl.Caps(device=device, max_tokens=cap, in_features=1280, out_features=11264,
                    source_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
                    block_size=(32, 32), output_mode="provided")
    plan = bfl.plan(caps)
    src = torch.randn(cap, 1280, device=device, dtype=torch.bfloat16)

    def prepare_fp8(state, src=src):
        spec, = state.scratch.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        out = torch.empty((src.shape[0], 11264, 1), dtype=torch.bfloat16, device=device)
        binding = state.bind(scratch=scratch, source=src, packed_weight=wq_b_packed, output=out)
        return PreparedCall(run=lambda: state.run_binding(binding), output=out, owners=(scratch, binding))

    fp8_plans[cap] = plan
    requests.append(plan.request(name=f"wq_b-{cap}", prepare_call=prepare_fp8))


def rows_of(gates, name, width):
    x = gates[name]["x"].to(device)
    return x[:, :width].contiguous()


with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
    session.prepare(tuple(requests))
    configs = {}
    for cap in CAPS:
        configs.setdefault(str(fp8_plans[cap].selection.config), []).append(cap)
    print("wq_b (block FP8 1280->11264) selected configuration by capacity:", flush=True)
    for config, caps in configs.items():
        print(f"  {caps}: {config[:400]}", flush=True)

    def run(label, x, cap):
        weight, out_dtype = SHAPES[label]
        return bf16_gemv.mm(x.contiguous(), weight, plan=plans[label, cap], output_dtype=out_dtype)

    # 1. fidelity against serving
    gate_label = "router gate (384x5120, FP32 out)"
    bad = []
    for gates, which in ((json_gates, "JSON"), (prose_gates, "prose")):
        for name in names:
            g = gates[name]
            if not g["taken"]:
                continue
            weight = gate_weights[name].to(device)
            label = gate_label if weight.shape[0] == 384 else next(k for k in SHAPES if k.startswith("drafter"))
            SHAPES[label] = (weight, torch.float32)
            x, served = g["x"].to(device), g["logits"].to(device)
            new = run(label, x, new_cap(x.shape[0]))
            old = run(label, x, old_cap(x.shape[0]))
            alone = torch.cat([run(label, x[i:i + 1], new_cap(1)) for i in range(x.shape[0])])
            if not torch.equal(new, served) or not torch.equal(new, alone):
                bad.append(f"{which} {name}: served {'=' if torch.equal(new, served) else 'DIFF'} "
                           f"alone {'=' if torch.equal(new, alone) else 'DIFF'}")
            if name == main[0]:
                print(f"fidelity {which} {name} ({x.shape[0]} rows): new-lookup replay vs served "
                      f"{'=' if torch.equal(new, served) else 'DIFF'}; vs rows alone "
                      f"{'=' if torch.equal(new, alone) else 'DIFF'}; old lookup ({old_cap(x.shape[0])}-row plan) "
                      f"vs served {'=' if torch.equal(old, served) else 'DIFF'} "
                      f"(max |d| {(old - served).abs().max().item():.3g})", flush=True)
    print(f"fidelity: {2 * len(names) - len(bad)} of {2 * len(names)} captured gate calls reproduce serving "
          f"bitwise and equal their rows alone" + (f"; mismatches: {bad[:6]}" if bad else ""), flush=True)
    SHAPES[gate_label] = (gate_weights[main[0]].to(device), torch.float32)

    # 2-3. sweeps on real rows
    R = list(range(1, 65)) + [96, 128, 255, 256, 257, 300, 512]
    target_name = main[0]
    for label, (weight, _) in SHAPES.items():
        width = weight.shape[1]
        target = rows_of(json_gates, target_name, width)
        if label.startswith("drafter"):
            target = rows_of(json_gates, drafter[0], width)
        T = target.shape[0]
        pool = torch.cat([rows_of(g, n, width) for g in (json_gates, prose_gates) for n in names
                          if g[n]["taken"] and not (g is json_gates and n == target_name)])
        alone = torch.cat([run(label, target[i:i + 1], 1) for i in range(T)])
        summary = {"new": [], "old": []}
        for rows in R:
            if rows < T:
                cases = [("start", target[:rows], 0, rows)]
            else:
                n = rows - T
                nb = pool[(rows * 7) % max(1, pool.shape[0] - n):][:n]
                cases = [(pos, torch.cat([nb[:k], target, nb[k:]]), k, T)
                         for pos, k in (("start", 0), ("middle", n // 2), ("end", n))]
            for lookup, cap_of in (("new", new_cap), ("old", old_cap)):
                cap = cap_of(rows)
                for pos, batch, k, t in cases:
                    out = run(label, batch, cap)[k:k + t]
                    diff = (out.float() - alone[:t].float()).abs()
                    if not torch.equal(out, alone[:t]):
                        summary[lookup].append((rows, cap, pos, int((diff > 0).any(1).sum()), t,
                                                diff.max().item()))
        torch.cuda.synchronize()
        for lookup in ("new", "old"):
            bad = summary[lookup]
            if not bad:
                print(f"{label}, {lookup} lookup: every target row equals the row alone at every R and position",
                      flush=True)
                continue
            by_rows = sorted({b[0] for b in bad})
            worst = max(bad, key=lambda b: b[5])
            print(f"{label}, {lookup} lookup: differs at R in {by_rows[:40]}"
                  f"{' ...' if len(by_rows) > 40 else ''}; worst R {worst[0]} ({worst[1]}-row plan, {worst[2]}): "
                  f"{worst[3]}/{worst[4]} rows, max |d| {worst[5]:.3g}", flush=True)

    # 3b. block-FP8 wq_b with serving's (smallest-capacity) lookup
    def fp8(x, cap):
        spec, = fp8_plans[cap].scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        out = torch.empty((x.shape[0], 11264, 1), dtype=torch.bfloat16, device=device)
        bfl.run(binding=bfl.bind(fp8_plans[cap], scratch=scratch, source=x.contiguous(),
                                 packed_weight=wq_b_packed, output=out))
        return out[:, :, 0]

    target = rows_of(json_gates, target_name, 1280)
    T = target.shape[0]
    pool = torch.cat([rows_of(g, n, 1280) for g in (json_gates, prose_gates) for n in names
                      if g[n]["taken"] and not (g is json_gates and n == target_name)])
    alone = torch.cat([fp8(target[i:i + 1], 1) for i in range(T)])
    bad = []
    for rows in R:
        cap = new_cap(rows)
        if rows < T:
            cases = [("start", target[:rows], 0, rows)]
        else:
            n = rows - T
            nb = pool[(rows * 7) % max(1, pool.shape[0] - n):][:n]
            cases = [(pos, torch.cat([nb[:k], target, nb[k:]]), k, T)
                     for pos, k in (("start", 0), ("middle", n // 2), ("end", n))]
        for pos, batch, k, t in cases:
            out = fp8(batch, cap)[k:k + t]
            if not torch.equal(out, alone[:t]):
                d = (out.float() - alone[:t].float()).abs()
                bad.append((rows, cap, pos, int((d > 0).any(1).sum()), t, d.max().item()))
    torch.cuda.synchronize()
    if bad:
        worst = max(bad, key=lambda b: b[5])
        print(f"wq_b (block FP8), serving lookup: differs from rows alone at R in "
              f"{sorted({b[0] for b in bad})[:40]}; worst R {worst[0]} ({worst[1]}-row plan, {worst[2]}): "
              f"{worst[3]}/{worst[4]} rows, max |d| {worst[5]:.3g}", flush=True)
        same = sorted({new_cap(r) for r in R} - {b[1] for b in bad})
        print(f"  capacities whose rows all equal the rows alone: {same}", flush=True)
    else:
        print("wq_b (block FP8), serving lookup: every target row equals the row alone at every R and position",
              flush=True)

    # 4. kernel time per call, inside CUDA graphs
    session.freeze()
    print("kernel time per call (us; graph of 50 calls x 10 replays): rows: old plan -> new plan", flush=True)
    for label, (weight, _) in SHAPES.items():
        width = weight.shape[1]
        base = torch.cat([rows_of(json_gates, target_name, width)] * 3)
        cells = []
        for rows in (1, 3, 8, 15, 19, 20, 33, 48):
            x = base[:rows].contiguous()
            times = {}
            for lookup, cap_of in (("old", old_cap), ("new", new_cap)):
                cap = cap_of(rows)
                run(label, x, cap)
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    for _ in range(50):
                        run(label, x, cap)
                graph.replay()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):
                    graph.replay()
                end.record()
                torch.cuda.synchronize()
                times[lookup] = 1000 * start.elapsed_time(end) / 500
                graph.reset()
            cells.append(f"{rows}: {times['old']:.1f}->{times['new']:.1f}")
        print(f"  {label}: " + "; ".join(cells), flush=True)
