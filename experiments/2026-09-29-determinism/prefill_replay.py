#!/usr/bin/env python3
"""Are V4.1's projections and mHC batch-invariant at prefill sizes? (GPU; cluster stopped)

usage: prefill_replay.py   (cwd: the B12X checkout, vLLM importable, checkpoint at /models)

The final validation's long target (998 prompt tokens) prefilled alone in
998-row steps and, beside background decodes, in 1004-1040-row steps, and its
first token's logprob already differed. Its rows are far above the small-plan
ranges (192 rows), so this replays the large-row families with checkpoint
weights under the smallest-capacity lookup: the V4.1 BF16 GEMVs (FP32 out:
router gate, ratio-2 compressor; BF16 out: index head weights, ratio-1
compressor, index key) at serving's capacities; the block-FP8 query
projection and Engram projection (rank 0's slices) at the block-FP8
capacities; and mHC (layer 0's pre on the embedding, layer 1's pre and
post_pre) with every prepared count warm. A 998-row target runs alone and
behind (end, as serving orders new prefills after running requests) or ahead
of (start) k other rows; batch sizes are grouped by the target rows' exact
bits, and each group's capacities are printed.
"""
import hashlib
import json
import os
import sys
import types

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from b12x.gemm import bf16_gemv  # noqa: E402
from b12x.gemm import block_fp8_linear as bfl  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from safetensors import safe_open  # noqa: E402

from vllm.models.deepseek_v4_1 import b12x_layers  # noqa: E402
from vllm.utils.b12x import B12xWorkload, register_b12x_layer  # noqa: E402
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402

device = torch.device("cuda", 0)
init_workspace_manager(device)
T = int(os.environ.get("PREFILL_TARGET", "998"))
K = (0, 1, 2, 6, 12, 18, 24, 30, 36, 42, 48, 64, 128, 256, 512, 538, 600, 1000)
POOL = 2 * T + 200
GEMV_CAPS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 21, 24, 25, 28, 30, 32, 35, 40, 42, 48, 49, 56,
             72, 96, 192, 384, 768, 1536, 3072, 4091, 4096)
FP8_CAPS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 128, 256, 384, 512, 640, 768, 896, 1024, 4096)
MHC_PREPARED = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 21, 24, 25, 28, 30, 32, 35, 40, 42, 48, 49,
                56, 72, 96, 192, 384, 768, 1536, 3072, 4091)
H, LIMIT = 5120, 4096
index = json.load(open("/models/model.safetensors.index.json"))["weight_map"]


def raw(name):
    with safe_open(f"/models/{index[name]}", framework="pt", device="cuda") as f:
        return f.get_tensor(name)


def bf16(name):
    return raw(name).to(torch.bfloat16).contiguous()


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors if isinstance(tensors, (list, tuple)) else (tensors,):
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:8]


def smallest(caps, rows):
    return next(c for c in caps if c >= rows)


def report(label, call, caps):
    """call(row_index_tensor) -> outputs for those pool rows; groups by the target rows' bits."""
    target = torch.arange(POOL - T, POOL, device=device)
    solo = digest(call(target))
    groups = {}
    for k in K:
        others = torch.arange(k, device=device)
        for pos in (("end", "start") if k else ("solo",)):
            rows = torch.cat([others, target]) if pos != "start" else torch.cat([target, others])
            first = k if pos != "start" else 0
            out = call(rows)
            key = digest([t[first:first + T] for t in out] if isinstance(out, (list, tuple))
                         else out[first:first + T])
            groups.setdefault(key, []).append(f"{T + k}{pos[0] if k else ''}")
    torch.cuda.synchronize()
    print(f"{label}: {len(groups)} groups", flush=True)
    for key, members in groups.items():
        cap_set = sorted({smallest(caps, int(m.rstrip('es'))) for m in members})
        print(f"    {'solo ' if key == solo else ''}{key}: {' '.join(members)}  (capacities {cap_set})",
              flush=True)


# --- V4.1 BF16 GEMVs ---------------------------------------------------------------------------
compressor2 = torch.cat([bf16("layers.2.attn.compressor.wkv.weight"),
                         bf16("layers.2.attn.compressor.wgate.weight")])
GEMVS = {
    "router gate (384x5120, FP32 out)": (bf16("layers.2.ffn.gate.weight"), torch.float32),
    "compressor ratio 2 (512x5120, FP32 out)": (compressor2[:512].contiguous(), torch.float32),
    "index head weights (32x5120, BF16 out)": (bf16("layers.2.attn.indexer.weights_proj.weight"), torch.bfloat16),
    "compressor ratio 1 (512x5120, BF16 out)": (bf16("layers.20.attn.compressor.wkv.weight")[:512].contiguous(),
                                                torch.bfloat16),
    "index key (128x512, BF16 out)": (bf16("layers.2.attn.indexer.wk.weight"), torch.bfloat16),
}
LARGE_GEMV = tuple(c for c in GEMV_CAPS if c >= 768)
gemv_plans, requests = {}, []
for label, (weight, out_dtype) in GEMVS.items():
    for cap in LARGE_GEMV:
        query = bf16_gemv.GemvQuery(
            source_dtype="bfloat16", weight_dtype="bfloat16", output_dtype=str(out_dtype).removeprefix("torch."),
            max_rows=cap, in_features=weight.shape[1], out_features=weight.shape[0], source_contiguous=True,
            source_aligned=True, weight_contiguous=True, weight_aligned=True)
        plan = bf16_gemv.plan(query)

        def make_call(state, weight=weight):
            q = state.query
            source = torch.empty((q.max_rows, q.in_features), device=device, dtype=torch.bfloat16)
            out = torch.empty((q.max_rows, q.out_features), device=device, dtype=getattr(torch, q.output_dtype))
            return PreparedCall(run=lambda: state.run(source, weight, out=out),
                                produce=lambda: source.normal_(std=0.25), owners=(weight,))

        gemv_plans[label, cap] = plan
        requests.append(plan.request(name=f"{label}-{cap}", prepare_call=make_call, benchmark_call=make_call))

# --- block FP8: query projection (layer 2) and Engram projection (layer 1), rank 0's slices ---------
FP8 = {}
for label, prefix in (("wq_b (layer 2)", "layers.2.attn.wq_b"), ("Engram wkv (layer 1)", "layers.1.engram.wkv")):
    w, s = raw(f"{prefix}.weight"), raw(f"{prefix}.scale")
    n = -(-w.shape[0] // 3 // 32) * 32
    FP8[label] = (bfl.pack_weight(w[:n].contiguous(), s[:n // 32].float().contiguous(), block_size=(32, 32)),
                  n, w.shape[1])
LARGE_FP8 = tuple(c for c in FP8_CAPS if c >= 768)
fp8_plans = {}
for label, (packed, n, k) in FP8.items():
    for cap in LARGE_FP8:
        plan = bfl.plan(bfl.Caps(device=device, max_tokens=cap, in_features=k, out_features=n,
                                 source_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
                                 block_size=(32, 32), output_mode="provided"))
        src = torch.randn(cap, k, device=device, dtype=torch.bfloat16)

        def prepare(state, src=src, n=n, packed=packed):
            spec, = state.scratch.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            out = torch.empty((src.shape[0], n, 1), dtype=torch.bfloat16, device=device)
            binding = state.bind(scratch=scratch, source=src, packed_weight=packed, output=out)
            return PreparedCall(run=lambda: state.run_binding(binding), output=out, owners=(scratch, binding))

        fp8_plans[label, cap] = plan
        requests.append(plan.request(name=f"{label}-{cap}", prepare_call=prepare))

# --- mHC ----------------------------------------------------------------------------------------
b12x_layers._execution_capacities = lambda: (1, 8, LIMIT)


def mhc_owner(layer):
    owner = torch.nn.Module()
    for part in ("attn", "ffn"):
        for kind in ("fn", "scale", "base"):
            setattr(owner, f"hc_{part}_{kind}", raw(f"layers.{layer}.hc_{part}_{kind}").float().contiguous())
        setattr(owner, f"{part}_norm", types.SimpleNamespace(weight=bf16(f"layers.{layer}.{part}_norm.weight")))
    owner.hc_attn_fn_broadcast = (owner.hc_attn_fn.view(-1, 4, H).sum(dim=1).contiguous() if layer == 0 else None)
    module = b12x_layers.B12xMHC(types.SimpleNamespace(hidden_size=H, rms_norm_eps=1e-20, hc_eps=1e-6,
                                                       hc_sinkhorn_iters=20, hc_mult=4))
    owner._b12x_mhc = module
    name = f"mhc-prefill-replay-{layer}"
    register_b12x_layer(name, owner)
    module.bind_layer_name(name)
    workload = B12xWorkload(stage="weights", token_counts=tuple(sorted({*MHC_PREPARED, LIMIT})),
                            fixed_token_counts=MHC_PREPARED, output_dtype=torch.bfloat16,
                            max_tokens=LIMIT, max_seqs=8, max_model_len=LIMIT)
    units = module.get_b12x_preparation_units(owner, workload)
    return owner, module, [r for unit in units for r in unit.requests]


def mhc_smallest(self, operation, tokens):
    capacities = sorted(rows for op, rows in self._plans if op == operation)
    return self._plans[(operation, next(rows for rows in capacities if rows >= int(tokens)))]


MHC = {}
for layer in (0, 1):
    owner, module, reqs = mhc_owner(layer)
    module._plan_for = types.MethodType(mhc_smallest, module)
    MHC[layer] = (owner, module)
    requests += reqs

with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
    session.prepare(tuple(requests))
    gen = torch.Generator(device=device).manual_seed(20261001)
    print(f"target {T} rows; batches of {T}+k rows, k in {K}; e = target last, s = target first", flush=True)

    for label, (weight, out_dtype) in GEMVS.items():
        configs = {}
        for cap in LARGE_GEMV:
            configs.setdefault(str(gemv_plans[label, cap].selection.config)[:160], []).append(cap)
        print(f"{label} configurations: " + "; ".join(f"{c}: {cfg}" for cfg, c in configs.items()), flush=True)
        pool = (torch.randn(POOL, weight.shape[1], generator=gen, device=device) * 0.5).to(torch.bfloat16)
        report(label, lambda rows, weight=weight, out_dtype=out_dtype, label=label, pool=pool: bf16_gemv.mm(
            pool[rows].contiguous(), weight, plan=gemv_plans[label, smallest(LARGE_GEMV, rows.numel())],
            output_dtype=out_dtype), LARGE_GEMV)

    for label, (packed, n, k) in FP8.items():
        configs = {}
        for cap in LARGE_FP8:
            configs.setdefault(str(fp8_plans[label, cap].selection.config)[:200], []).append(cap)
        print(f"{label} configurations: " + "; ".join(f"{c}: {cfg}" for cfg, c in configs.items()), flush=True)
        pool = (torch.randn(POOL, k, generator=gen, device=device) * 0.5).to(torch.bfloat16)

        def fp8_call(rows, label=label, packed=packed, n=n, pool=pool):
            plan = fp8_plans[label, smallest(LARGE_FP8, rows.numel())]
            spec, = plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            out = torch.empty((rows.numel(), n, 1), dtype=torch.bfloat16, device=device)
            bfl.run(binding=bfl.bind(plan, scratch=scratch, source=pool[rows].contiguous(),
                                     packed_weight=packed, output=out))
            return out[:, :, 0]

        report(label, fp8_call, LARGE_FP8)

    mpool = {"x2": torch.randn(POOL, H, generator=gen, device=device).to(torch.bfloat16),
             "x3": torch.randn(POOL, 4, H, generator=gen, device=device).to(torch.bfloat16),
             "post": torch.rand(POOL, 4, generator=gen, device=device),
             "comb": torch.rand(POOL, 4, 4, generator=gen, device=device),
             "pre": torch.softmax(torch.randn(POOL, 4, generator=gen, device=device), dim=-1)}
    _, m1 = MHC[1]
    configs = {}
    for (op, cap), plan in sorted(m1._plans.items()):
        if cap >= 768:
            configs.setdefault((op, str(plan.selection.config)[:200]), []).append(cap)
    for (op, cfg), caps in configs.items():
        print(f"mHC {op} capacities {caps}: {cfg}", flush=True)
    mhc_caps = tuple(c for c in (*MHC_PREPARED, LIMIT) if c >= 768)
    for op in ("layer-0 pre (2-D embedding)", "pre (4-stream residual)", "post_pre"):
        owner, module = MHC[0 if op.startswith("layer-0") else 1]

        def mhc_call(rows, op=op, owner=owner, module=module):
            take = lambda key: mpool[key][rows].contiguous()  # noqa: E731
            if op.startswith("layer-0"):
                return module.pre(take("x2"), owner.hc_attn_fn_broadcast, owner.hc_attn_scale,
                                  owner.hc_attn_base, owner.attn_norm.weight, None)
            if op.startswith("pre"):
                return module.pre(take("x3"), owner.hc_attn_fn, owner.hc_attn_scale,
                                  owner.hc_attn_base, owner.attn_norm.weight, take("pre"))
            return module.post_pre(take("x2"), take("x3"), take("post"), take("comb"),
                                   owner.hc_ffn_fn, owner.hc_ffn_scale, owner.hc_ffn_base,
                                   owner.ffn_norm.weight, take("pre"))

        report(f"mHC {op}", lambda rows, f=mhc_call: [t for t in f(rows) if torch.is_tensor(t)], mhc_caps)
print("prefill replay done", flush=True)
