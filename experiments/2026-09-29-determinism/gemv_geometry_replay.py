#!/usr/bin/env python3
"""TMA prefill GEMV launch geometries on real rows: bits against the production geometry, timings.

usage: gemv_geometry_replay.py CAPTURE_DIR   (GPU, cluster stopped; overlay gemv-geom mounted over
       B12X's bf16_gemv, cwd the B12X checkout, checkpoint at /models, transition_map.py importable)

Rows: the mHC captures run through production's mHC (pre's output feeds the compressors,
post_pre's the router gate), batches built by repeating them. For the router gate (384x5120,
FP32 out) and the compressors (1024x5120 FP32, 512x5120 BF16), each geometry in GEOMETRIES is
compiled directly and run at 1-4000 rows with a target row last and first; every output row must
equal the production geometry's bit for bit. Timings: median of 40, decode sizes from a CUDA
graph, larger sizes eager.
"""
import glob
import json
import os
import sys
import types

import torch

sys.path.insert(0, "/r")
import transition_map as tm  # noqa: E402

from b12x.gemm.bf16_gemv import _prefill  # noqa: E402
from b12x.preparation import PreparationSession  # noqa: E402
from vllm.models.deepseek_v4_1 import b12x_layers  # noqa: E402
from vllm.utils.b12x import B12xWorkload, register_b12x_layer  # noqa: E402
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402

device = tm.device
H = 5120
GRAPH = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48)
SIZES = (1, 2, 3, 4, 5, 6, 8, 12, 16, 19, 24, 32, 48, 49, 64, 96, 128, 255, 256, 1334, 1366, 2048, 4000)
TIMED = (1, 4, 6, 16, 48, 128, 1334, 4000)
init_workspace_manager(device)
b12x_layers._execution_capacities = lambda: (1, 8, 4096)
b12x_layers._BATCH_INVARIANT = False

pools, modules, requests = {}, {}, []
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
    key = f"geom-rows-{os.path.basename(path)}"
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
        pools.setdefault(c["operation"], []).append(out[3].clone())
pools = {op: torch.cat(v) for op, v in pools.items()}
print(json.dumps({"rows": {op: p.shape[0] for op, p in pools.items()}}), flush=True)

compressor2 = torch.cat([tm.bf16("layers.2.attn.compressor.wkv.weight"),
                         tm.bf16("layers.2.attn.compressor.wgate.weight")]).contiguous()
SHAPES = {
    "router gate 384x5120 fp32": (tm.bf16("layers.2.ffn.gate.weight"), torch.float32, "post_pre"),
    "compressor ratio-2 1024x5120 fp32": (compressor2, torch.float32, "pre"),
    "compressor ratio-1 512x5120 bf16": (tm.bf16("layers.20.attn.compressor.wkv.weight"), torch.bfloat16, "pre"),
}


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


ordinal = device.index or 0
for label, (weight, out_dtype, source) in SHAPES.items():
    pool = pools[source]
    big = pool[torch.arange(4096, device=device) % pool.shape[0]].contiguous()
    target = big[-1:]
    n = weight.shape[0]
    launchers = {}
    for geometry in sorted(_prefill.GEOMETRIES):
        try:
            launchers[geometry] = _prefill.compile_prefill(ordinal, 4096, n, H, str(out_dtype).removeprefix("torch."),
                                                           geometry)
        except Exception as error:  # noqa: BLE001
            print(json.dumps({"op": label, "geometry": geometry, "compile_error": repr(error)[:300]}), flush=True)

    def make_fn(launcher):
        def fn(x):
            out = torch.empty((x.shape[0], n), dtype=out_dtype, device=device)
            launcher(x, weight, out)
            return out
        return fn

    reference = {}
    base = make_fn(launchers[_prefill.DEFAULT_GEOMETRY])
    for m in SIZES:
        for pos in ("e", "s"):
            batch = torch.cat([big[: m - 1], target]) if pos == "e" else torch.cat([target, big[: m - 1]])
            reference[(m, pos)] = base(batch.contiguous())
    alone = base(target.contiguous())[0]
    print(json.dumps({"op": label, "default_rows_invariant": all(
        torch.equal(reference[(m, p)][m - 1 if p == "e" else 0], alone) for m in SIZES for p in ("e", "s"))}),
        flush=True)
    for geometry, launcher in launchers.items():
        fn = make_fn(launcher)
        mismatched = []
        for m in SIZES:
            for pos in ("e", "s"):
                batch = torch.cat([big[: m - 1], target]) if pos == "e" else torch.cat([target, big[: m - 1]])
                if not torch.equal(fn(batch.contiguous()), reference[(m, pos)]):
                    mismatched.append(f"{m}{pos}")
        print(json.dumps({"op": label, "geometry": geometry, "bit_equal_to_production": not mismatched,
                          "mismatched": mismatched[:12],
                          "timing_us": {str(r): timed(fn, big[:r].contiguous()) for r in TIMED}}), flush=True)
print("gemv geometry replay done", flush=True)
