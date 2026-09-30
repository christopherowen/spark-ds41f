#!/usr/bin/env python3
"""Is the Engram projection (block FP8) batch-invariant across serving's capacities? (GPU; cluster stopped)

usage: engram_wkv_replay.py   (cwd: the B12X checkout, checkpoint at /models)

With all four lookup and backend fixes the traced JSON target first differs,
solo against mixed, at layer 14's MoE input in its first decode step (6 rows,
padded to 6 alone and 16 mixed), with layer 14's attention output matching.
Layer 14 is an Engram layer; its projection wkv (6144 -> 5 x 5120, column
parallel over three ranks) is a block-FP8 linear planned per capacity. This
replays rank 0's slice with the checkpoint weights of layers 1 and 14 at
serving's block-FP8 capacities (smallest-capacity lookup, as serving), prints
each capacity's dense configuration, and groups batch sizes by the exact bits
of target rows (6 and 19 rows, at the start, middle and end of each batch).
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from b12x.gemm import block_fp8_linear as bfl  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402
from safetensors import safe_open  # noqa: E402

device = torch.device("cuda", 0)
CAPS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 128, 256, 384, 512, 640, 768, 896, 1024, 4096)
index = json.load(open("/models/model.safetensors.index.json"))["weight_map"]


def raw(name):
    with safe_open(f"/models/{index[name]}", framework="pt", device="cuda") as f:
        return f.get_tensor(name)


weights = {}
for layer in (1, 14):
    w, s = raw(f"layers.{layer}.engram.wkv.weight"), raw(f"layers.{layer}.engram.wkv.scale")
    n = -(-w.shape[0] // 3 // 32) * 32  # rank 0's padded column slice
    weights[layer] = (bfl.pack_weight(w[:n].contiguous(), s[:n // 32].float().contiguous(), block_size=(32, 32)),
                      n, w.shape[1])
    print(f"layer {layer}: wkv {tuple(w.shape)}, rank-0 slice {n} x {w.shape[1]}", flush=True)
plans, requests = {}, []
for layer, (packed, n, k) in weights.items():
    for cap in CAPS:
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

        plans[layer, cap] = plan
        requests.append(plan.request(name=f"wkv{layer}-{cap}", prepare_call=prepare))


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:8]


with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
    session.prepare(tuple(requests))
    for layer, (packed, n, k) in weights.items():
        configs = {}
        for cap in CAPS:
            configs.setdefault(str(plans[layer, cap].selection.config), []).append(cap)
        print(f"layer {layer} wkv configurations:", flush=True)
        for config, caps in configs.items():
            print(f"  {caps}: {config[:260]}", flush=True)

        def run(x, layer=layer, n=n, packed=packed):
            cap = next(c for c in CAPS if c >= x.shape[0])
            spec, = plans[layer, cap].scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            out = torch.empty((x.shape[0], n, 1), dtype=torch.bfloat16, device=device)
            bfl.run(binding=bfl.bind(plans[layer, cap], scratch=scratch, source=x.contiguous(),
                                     packed_weight=packed, output=out))
            return out[:, :, 0]

        gen = torch.Generator(device=device).manual_seed(20260930)
        pool = (torch.randn(700, k, generator=gen, device=device) * 0.5).to(torch.bfloat16)
        for T in (6, 19):
            target = pool[600:600 + T]
            solo = digest(run(target))
            groups = {}
            for R in list(range(T, 65)) + [96, 128, 256, 512]:
                m = R - T
                for pos, j in ((("start", 0), ("middle", m // 2), ("end", m)) if m else (("solo", 0),)):
                    batch = torch.cat([pool[:j], target, pool[j:m]])
                    groups.setdefault(digest(run(batch)[j:j + T]), []).append(f"{R}{pos[0] if m else ''}")
            torch.cuda.synchronize()
            print(f"layer {layer}, target {T}: {len(groups)} groups: " + "; ".join(
                f"{'*' if key == solo else ''}{mem[0]}..{mem[-1]} ({len(mem)})" for key, mem in groups.items()),
                flush=True)
            if len(groups) > 1:
                for key, mem in groups.items():
                    print(f"      {key}: {' '.join(mem)}", flush=True)
