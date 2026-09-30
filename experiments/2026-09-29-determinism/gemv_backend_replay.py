#!/usr/bin/env python3
"""Do the BF16-output GEMVs become batch-invariant on the SIMT backend? (GPU; cluster stopped)

usage: gemv_backend_replay.py   (cwd: the B12X checkout, vLLM importable, checkpoint at /models)

The BF16-output V4.1 projections (index head weights 32 x 5120, the ratio-1
compressor 512 x 5120 at layer 20, the index key 128 x 512) take the torch
backend (cuBLAS, which picks its kernel by M) at capacities from 10 rows. For
each, with layer-20 checkpoint weights, plans at serving's capacities run with
the default configuration and with the SIMT backend forced up to 192 rows
(the capacities where the FP32-output projections already run SIMT); target
rows (19) run alone and in batches of R rows (target at the start, middle, end)
under the smallest-capacity lookup; batch sizes are grouped by the target rows'
exact bits. Then layer 20's index-key RMSNorm (B12xRMSNorm, 128 wide), which
has one plan. Kernel time per call at a few row counts for both backends.
"""
import hashlib
import json
import os
import sys
import types

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from b12x.gemm import bf16_gemv  # noqa: E402
from b12x.gemm.bf16_gemv._tuning import GemvConfig, default_config  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from safetensors import safe_open  # noqa: E402

from vllm.models.deepseek_v4_1 import b12x_layers  # noqa: E402
from vllm.utils.b12x import B12xWorkload  # noqa: E402
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402

device = torch.device("cuda", 0)
init_workspace_manager(device)
CAPS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 21, 24, 25, 28, 30, 32, 35, 40, 42, 48, 49, 56, 72,
        96, 192, 384, 768, 1536, 3072, 4091, 4096)
index = json.load(open("/models/model.safetensors.index.json"))["weight_map"]


def tensor(name):
    with safe_open(f"/models/{index[name]}", framework="pt", device="cuda") as f:
        return f.get_tensor(name).to(torch.bfloat16).contiguous()


SHAPES = {
    "index head weights (32x5120)": tensor("layers.20.attn.indexer.weights_proj.weight"),
    "ratio-1 compressor (512x5120)": tensor("layers.20.attn.compressor.wkv.weight")[:512].contiguous(),
    "index key (128x512)": tensor("layers.20.attn.indexer.wk.weight"),
}
plans, requests = {}, []
for label, weight in SHAPES.items():
    for backend in ("default", "simt"):
        for cap in CAPS:
            query = bf16_gemv.GemvQuery(
                source_dtype="bfloat16", weight_dtype="bfloat16", output_dtype="bfloat16", max_rows=cap,
                in_features=weight.shape[1], out_features=weight.shape[0], source_contiguous=True,
                source_aligned=True, weight_contiguous=True, weight_aligned=True)
            override = GemvConfig(backend="simt") if backend == "simt" and cap <= 192 else None
            plan = bf16_gemv.plan(query, override=override)

            def make_call(state, weight=weight):
                q = state.query
                source = torch.empty((q.max_rows, q.in_features), device=device, dtype=torch.bfloat16)
                out = torch.empty((q.max_rows, q.out_features), device=device, dtype=torch.bfloat16)
                return PreparedCall(run=lambda: state.run(source, weight, out=out),
                                    produce=lambda: source.normal_(std=0.25), owners=(weight,))

            plans[label, backend, cap] = plan
            requests.append(plan.request(name=f"{label}-{backend}-{cap}", prepare_call=make_call,
                                         benchmark_call=make_call))
b12x_layers._capacity = lambda: 4096
norm = b12x_layers.B12xRMSNorm(128, 1e-20).to(device)
norm.weight.data.copy_(tensor("layers.20.attn.indexer.k_norm.weight"))
workload = B12xWorkload(stage="weights", token_counts=CAPS, fixed_token_counts=CAPS[:-1],
                        output_dtype=torch.bfloat16, max_tokens=4096, max_seqs=8, max_model_len=4096)
requests += [r for unit in norm.get_b12x_preparation_units(norm, workload) for r in unit.requests]


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:8]


def groups_for(fn, width, T=19):
    gen = torch.Generator(device=device).manual_seed(20260930)
    pool = (torch.randn(700, width, generator=gen, device=device) * 0.5).to(torch.bfloat16)
    target = pool[600:600 + T]
    solo = digest(fn(target))
    groups = {}
    for R in list(range(T, 65)) + [72, 96, 128, 192, 256, 384, 512]:
        n = R - T
        for pos, k in ((("start", 0), ("middle", n // 2), ("end", n)) if n else (("solo", 0),)):
            batch = torch.cat([pool[:k], target, pool[k:n]])
            groups.setdefault(digest(fn(batch)[k:k + T]), []).append(f"{R}{pos[0] if n else ''}")
    torch.cuda.synchronize()
    return "; ".join(f"{'*' if key == solo else ''}{m[0]}..{m[-1]} ({len(m)})" for key, m in groups.items()), groups


with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
    session.prepare(tuple(requests))
    for label, weight in SHAPES.items():
        for backend in ("default", "simt"):
            by = {}
            for cap in CAPS:
                by.setdefault(plans[label, backend, cap].selection.config.backend, []).append(cap)
            fn = lambda x, label=label, backend=backend, weight=weight: bf16_gemv.mm(  # noqa: E731
                x, weight, plan=plans[label, backend, next(c for c in CAPS if c >= x.shape[0])],
                output_dtype=torch.bfloat16)
            summary, groups = groups_for(fn, weight.shape[1])
            print(f"{label}, {backend} ({by}): {len(groups)} groups: {summary}", flush=True)
            if len(groups) > 1:
                for key, members in groups.items():
                    print(f"      {key}: {' '.join(members)}", flush=True)
    summary, groups = groups_for(lambda x: norm(x), 128)
    print(f"index-key RMSNorm (one plan): {len(groups)} groups: {summary}", flush=True)
    session.freeze()
    print("kernel time per call (us; graph of 50 calls x 10 replays): rows: default -> simt", flush=True)
    for label, weight in SHAPES.items():
        cells = []
        for rows in (10, 15, 19, 28, 48, 96, 192):
            x = (torch.randn(rows, weight.shape[1], device=device) * 0.5).to(torch.bfloat16)
            times = {}
            for backend in ("default", "simt"):
                plan = plans[label, backend, next(c for c in CAPS if c >= rows)]
                bf16_gemv.mm(x, weight, plan=plan, output_dtype=torch.bfloat16)
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    for _ in range(50):
                        bf16_gemv.mm(x, weight, plan=plan, output_dtype=torch.bfloat16)
                graph.replay()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):
                    graph.replay()
                end.record()
                torch.cuda.synchronize()
                times[backend] = 1000 * start.elapsed_time(end) / 500
                graph.reset()
            cells.append(f"{rows}: {times['default']:.1f}->{times['simt']:.1f}")
        print(f"  {label}: " + "; ".join(cells), flush=True)
