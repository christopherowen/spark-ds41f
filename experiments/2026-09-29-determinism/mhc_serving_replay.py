#!/usr/bin/env python3
"""Is V4.1's mHC batch-invariant under serving's plan lookup? (GPU; cluster stopped)

usage: mhc_serving_replay.py   (cwd: the B12X checkout, vLLM importable, checkpoint at /models)

vLLM's B12xMHC prepares one plan per warm token count for each operation (pre,
post_pre, post) and looked a live count up exactly, else took the largest (the
4096-token plan). This builds B12xMHC as serving does, with layer 0's and layer
1's checkpoint mHC weights, registers it for its custom ops, and runs three
serving operations: layer 0's pre on the 2-D embedding (broadcast fn), a later
layer's pre on the four-stream residual, and post_pre. Target rows (19, the
JSON prefill, and 15, the prose prefill) run alone and inside batches of R rows
(target at the start, middle and end, random neighbours); batch sizes are
grouped by the exact bits of the target rows' outputs (y, residual_out, post,
comb, pre_out). Serving's lookup and the smallest capacity at or above R, for
two warm sets: the graph sizes, and every count serving prepares.
"""
import hashlib
import json
import os
import sys
import types

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from b12x.preparation import PreparationSession  # noqa: E402
from safetensors import safe_open  # noqa: E402

from vllm.models.deepseek_v4_1 import b12x_layers  # noqa: E402
from vllm.utils.b12x import B12xWorkload, register_b12x_layer  # noqa: E402
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402

device = torch.device("cuda", 0)
init_workspace_manager(device)
H, LIMIT = 5120, 4096
GRAPH = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48)
PREPARED = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 21, 24, 25, 28, 30, 32, 35, 40, 42, 48, 49, 56,
            72, 96, 192, 384, 768, 1536, 3072, 4091)
b12x_layers._execution_capacities = lambda: (1, 8, LIMIT)
index = json.load(open("/models/model.safetensors.index.json"))["weight_map"]


def tensor(name):
    with safe_open(f"/models/{index[name]}", framework="pt", device="cuda") as f:
        return f.get_tensor(name)


def owner_for(layer, warm):
    owner = torch.nn.Module()
    for part in ("attn", "ffn"):
        for kind in ("fn", "scale", "base"):
            setattr(owner, f"hc_{part}_{kind}", tensor(f"layers.{layer}.hc_{part}_{kind}").float().contiguous())
        setattr(owner, f"{part}_norm", types.SimpleNamespace(
            weight=tensor(f"layers.{layer}.{part}_norm.weight").to(torch.bfloat16).contiguous()))
    owner.hc_attn_fn_broadcast = (owner.hc_attn_fn.view(-1, 4, H).sum(dim=1).contiguous()
                                  if layer == 0 else None)
    module = b12x_layers.B12xMHC(types.SimpleNamespace(hidden_size=H, rms_norm_eps=1e-20, hc_eps=1e-6,
                                                       hc_sinkhorn_iters=20, hc_mult=4))
    owner._b12x_mhc = module
    name = f"mhc-replay-{layer}-{len(warm)}"
    register_b12x_layer(name, owner)
    module.bind_layer_name(name)
    workload = B12xWorkload(stage="weights", token_counts=tuple(sorted({*warm, LIMIT})),
                            fixed_token_counts=tuple(warm), output_dtype=torch.bfloat16,
                            max_tokens=LIMIT, max_seqs=8, max_model_len=LIMIT)
    units = module.get_b12x_preparation_units(owner, workload)
    return owner, module, tuple(r for unit in units for r in unit.requests)


def smallest(self, operation, tokens):
    tokens = int(tokens)
    capacities = sorted(rows for op, rows in self._plans if op == operation)
    capacity = next((rows for rows in capacities if rows >= tokens), None)
    if capacity is None:
        raise RuntimeError("tokens exceed the largest mHC plan")
    return self._plans[(operation, capacity)]


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:8]


setups = {(layer, warm_name): owner_for(layer, warm)
          for layer in (0, 1) for warm_name, warm in (("graph sizes", GRAPH), ("all prepared", PREPARED))}
with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
    session.prepare(tuple(r for _, _, requests in setups.values() for r in requests))
    _, module, _ = setups[(1, "all prepared")]
    configs = {}
    for (op, cap), plan in sorted(module._plans.items()):
        configs.setdefault((op, str(plan.selection.config)), []).append(cap)
    for (op, config), caps in configs.items():
        print(f"mHC {op} capacities {caps}: {config[:240]}", flush=True)
    gen = torch.Generator(device=device).manual_seed(20260930)
    pool = {"x2": torch.randn(700, H, generator=gen, device=device).to(torch.bfloat16),
            "x3": torch.randn(700, 4, H, generator=gen, device=device).to(torch.bfloat16),
            "post": torch.rand(700, 4, generator=gen, device=device),
            "comb": torch.rand(700, 4, 4, generator=gen, device=device),
            "pre": torch.softmax(torch.randn(700, 4, generator=gen, device=device), dim=-1)}
    sizes = list(range(15, 65)) + [72, 96, 128, 192, 256, 384, 512]
    for (layer, warm_name), (owner, module, _) in setups.items():
        operations = (("layer-0 pre (2-D embedding)",) if layer == 0
                      else ("pre (4-stream residual)", "post_pre"))
        for op in operations:
            for lookup in ("serving", "smallest"):
                if lookup == "smallest":
                    module._plan_for = types.MethodType(smallest, module)
                else:
                    module.__dict__.pop("_plan_for", None)

                def call(rows):
                    take = lambda key: pool[key][rows]  # noqa: E731
                    if op.startswith("layer-0"):
                        out = module.pre(take("x2"), owner.hc_attn_fn_broadcast, owner.hc_attn_scale,
                                         owner.hc_attn_base, owner.attn_norm.weight, None)
                    elif op.startswith("pre"):
                        out = module.pre(take("x3"), owner.hc_attn_fn, owner.hc_attn_scale,
                                         owner.hc_attn_base, owner.attn_norm.weight, take("pre"))
                    else:
                        out = module.post_pre(take("x2"), take("x3"), take("post"), take("comb"),
                                              owner.hc_ffn_fn, owner.hc_ffn_scale, owner.hc_ffn_base,
                                              owner.ffn_norm.weight, take("pre"))
                    return out

                for T in (19, 15):
                    target = torch.arange(600, 600 + T, device=device)
                    solo = digest([t[:T] for t in call(target)])
                    groups = {}
                    for R in sizes:
                        if R < T:
                            continue
                        n = R - T
                        others = torch.arange(n, device=device)
                        for pos, k in ((("start", 0), ("middle", n // 2), ("end", n)) if n else (("solo", 0),)):
                            rows = torch.cat([others[:k], target, others[k:]])
                            key = digest([t[k:k + T] for t in call(rows)])
                            groups.setdefault(key, []).append(f"{R}{pos[0] if n else ''}")
                    torch.cuda.synchronize()
                    summary = "; ".join(f"{'*' if key == solo else ''}{m[0]}..{m[-1]} ({len(m)})"
                                        for key, m in groups.items())
                    print(f"[{warm_name}] {op}, {lookup} lookup, target {T}: {len(groups)} groups: {summary}",
                          flush=True)
                    if len(groups) > 1:
                        for key, members in groups.items():
                            print(f"      {'solo ' if key == solo else ''}{key}: {' '.join(members)}", flush=True)
