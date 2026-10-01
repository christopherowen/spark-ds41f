#!/usr/bin/env python3
"""Where does each operation's arithmetic change with the batch's row count? (GPU; cluster stopped)

usage: transition_map.py FAMILY [FAMILY...]   (cwd: the B12X checkout, vLLM importable, checkpoint at /models)
       FAMILY: gemv, fp8, mhc, norm, moe, head, wo, reference   (moe and reference need det-variant2)

One target row runs alone and as the last (e) or first (s) row of batches of
M rows (random neighbours), for every M from 1 to 72, the rows on both sides
of every prepared capacity, and up to 4096. Row counts are grouped by the
target row's exact output bits: each group is one arithmetic, and the map
lists the row counts where the group changes. Configurations per family:

- gemv: the V4.1 BF16 GEMVs (router gate and ratio-2 compressor, FP32 out;
  index head weights, ratio-1 compressor and index key, BF16 out) at serving's
  capacities with the smallest-capacity lookup: the default selection, the
  batch-invariant mode (SIMT below 256 rows), and SIMT at every capacity.
- fp8: block-FP8 linears (query a+kv, query b, index query b, shared expert
  gate/up and down, Engram projection; rank 0's slices) at the block-FP8
  capacities: the default selection and one K slice everywhere.
- mhc: layer 0's pre on the embedding, layer 1's pre, post and post_pre, with
  every prepared count warm (serving's set), under vllm-0028's smallest-capacity
  lookup and under the original exact-count-else-largest lookup.
- norm: B12xRMSNorm at the model's widths.
- moe: the deterministic routed MoE through serving's one plan (graph sizes and
  the 4096-token limit warm).
- head: the LM head's rank-0 vocabulary shard as vLLM runs it under the B12X
  linear backend: GEMV plans for the graph sizes up to 8 rows, F.linear above.
- wo: the inverse-RoPE WO projection with rank 0's three groups of layer 2's
  checkpoint weights: decode steps' static plan for each row count up to 96
  against prefill steps' dynamic 4096-row plan (one grouping over both).
"""
import hashlib
import json
import os
import sys
import types

sys.path.insert(0, os.getcwd())
os.environ.setdefault("B12X_W4A8_TINY_DECODE", "0")

import torch  # noqa: E402

device = torch.device("cuda", 0)
GEMV_CAPS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 21, 24, 25, 28, 30, 32, 35, 40, 42, 48, 49, 56,
             72, 96, 192, 384, 768, 1536, 3072, 4091, 4096)
FP8_CAPS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 128, 256, 384, 512, 640, 768, 896, 1024, 4096)
GRAPH = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48)
LIMIT = 4096
SIZES = sorted(set(range(1, 73)) | {c + d for c in (*GEMV_CAPS, *FP8_CAPS, 204, 205, 255, 256, 2048)
                                    for d in (-1, 0, 1) if 1 <= c + d <= LIMIT})
POSITIONS = ("e", "s")
_index = None


def raw(name):
    global _index
    from safetensors import safe_open
    if _index is None:
        _index = json.load(open("/models/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"/models/{_index[name]}", framework="pt", device="cuda") as f:
        return f.get_tensor(name)


def bf16(name):
    return raw(name).to(torch.bfloat16).contiguous()


def digest(out):
    h = hashlib.sha256()
    for t in out if isinstance(out, (list, tuple)) else (out,):
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:10]


def transitions(label, call, width, dtype=torch.bfloat16, make=None):
    """call(batch) -> outputs for the batch's rows; group sizes by the target row's bits."""
    gen = torch.Generator(device=device).manual_seed(20261001)
    pool = make(LIMIT, gen) if make else (torch.randn(LIMIT, width, generator=gen, device=device) * 0.5).to(dtype)
    target = pool[-1:]
    groups, order = {}, []
    for m in SIZES:
        for pos in (POSITIONS if m > 1 else ("e",)):
            others = pool[: m - 1]
            batch = torch.cat([others, target]) if pos == "e" else torch.cat([target, others])
            row = m - 1 if pos == "e" else 0
            try:
                out = call(batch)
                key = digest([t[row: row + 1] for t in out] if isinstance(out, (list, tuple)) else out[row: row + 1])
            except Exception as error:  # noqa: BLE001 - an unsupported size is a map entry too
                key = "error:" + type(error).__name__
            if key not in groups:
                order.append(key)
            groups.setdefault(key, []).append((m, pos))
    torch.cuda.synchronize()
    names = {key: chr(65 + i) if i < 26 else f"g{i}" for i, key in enumerate(order)}
    sequence, changes = [], []
    previous = None
    for m in SIZES:
        cell = "".join(names[next(k for k, v in groups.items() if (m, p) in v)] for p in
                       (POSITIONS if m > 1 else ("e",)))
        sequence.append((m, cell))
        if cell != previous:
            changes.append(f"{m}:{cell}")
        previous = cell
    position_dependent = [m for m, cell in sequence if len(set(cell)) > 1]
    result = {"op": label, "groups": len(groups), "changes": changes,
              "position_dependent": position_dependent[:20],
              "errors": sorted(k for k in groups if k.startswith("error:"))}
    print(json.dumps(result), flush=True)
    return result


def smallest(caps, rows):
    return next(c for c in caps if c >= rows)


def family_gemv():
    from b12x.gemm import bf16_gemv
    from b12x.gemm.bf16_gemv._tuning import GemvConfig
    from b12x.preparation import PreparationSession, PreparedCall

    compressor2 = torch.cat([bf16("layers.2.attn.compressor.wkv.weight"),
                             bf16("layers.2.attn.compressor.wgate.weight")])
    shapes = {
        "gemv router gate 384x5120 fp32": (bf16("layers.2.ffn.gate.weight"), torch.float32),
        "gemv compressor ratio-2 1024x5120 fp32": (compressor2.contiguous(), torch.float32),
        "gemv index weights 32x5120 bf16": (bf16("layers.2.attn.indexer.weights_proj.weight"), torch.bfloat16),
        "gemv compressor ratio-1 512x5120 bf16": (bf16("layers.20.attn.compressor.wkv.weight"), torch.bfloat16),
        "gemv index key 128x512 bf16": (bf16("layers.2.attn.indexer.wk.weight"), torch.bfloat16),
    }
    modes = ("default", "bi", "simt")
    plans, requests = {}, []
    for label, (weight, out_dtype) in shapes.items():
        for mode in modes:
            for cap in GEMV_CAPS:
                query = bf16_gemv.GemvQuery(
                    source_dtype="bfloat16", weight_dtype="bfloat16",
                    output_dtype=str(out_dtype).removeprefix("torch."), max_rows=cap,
                    in_features=weight.shape[1], out_features=weight.shape[0], source_contiguous=True,
                    source_aligned=True, weight_contiguous=True, weight_aligned=True)
                override = None
                if mode == "simt" or (mode == "bi" and cap < 256 and out_dtype == torch.bfloat16):
                    override = GemvConfig(backend="simt")
                try:
                    plan = bf16_gemv.plan(query, override=override)
                except Exception as error:  # noqa: BLE001
                    print(json.dumps({"op": label, "mode": mode, "cap": cap, "plan_error": repr(error)[:200]}))
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
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(requests))
        for label, (weight, out_dtype) in shapes.items():
            for mode in modes:
                caps = [c for c in GEMV_CAPS if (label, mode, c) in plans]
                configs = {}
                for c in caps:
                    configs.setdefault(str(plans[label, mode, c].selection.config), []).append(c)
                print(json.dumps({"op": label, "mode": mode, "configs": {k: v for k, v in configs.items()}}))
                transitions(f"{label} [{mode}]", lambda x, label=label, mode=mode, weight=weight, out_dtype=out_dtype,
                            caps=caps: bf16_gemv.mm(x.contiguous(), weight,
                                                    plan=plans[label, mode, smallest(caps, x.shape[0])],
                                                    output_dtype=out_dtype), weight.shape[1])


def family_fp8():
    from dataclasses import replace

    from b12x.gemm import block_fp8_linear as bfl
    from b12x.preparation import PreparationSession, PreparedCall

    def rank0(prefix, dim):
        w, s = raw(f"{prefix}.weight"), raw(f"{prefix}.scale")
        if dim == 0:  # column parallel: rank 0's output rows
            n = -(-w.shape[0] // 3 // 32) * 32
            return w[:n].contiguous(), s[: n // 32].float().contiguous()
        k = -(-w.shape[1] // 3 // 32) * 32  # row parallel: rank 0's input columns
        return w[:, :k].contiguous(), s[:, : k // 32].float().contiguous()

    def full(prefix):
        return raw(f"{prefix}.weight").contiguous(), raw(f"{prefix}.scale").float().contiguous()

    wq_a, wkv = full("layers.2.attn.wq_a"), full("layers.2.attn.wkv")
    w1, w3 = rank0("layers.2.ffn.shared_experts.w1", 0), rank0("layers.2.ffn.shared_experts.w3", 0)
    linears = {
        "fp8 query a+kv 1792x5120": (torch.cat([wq_a[0], wkv[0]]), torch.cat([wq_a[1], wkv[1]])),
        "fp8 query b r0 11264x1280": rank0("layers.2.attn.wq_b", 0),
        "fp8 index query b r0 1376x1280": rank0("layers.2.attn.indexer.wq_b", 0),
        "fp8 shared gate/up r0 1536x5120": (torch.cat([w1[0], w3[0]]), torch.cat([w1[1], w3[1]])),
        "fp8 shared down r0 5120x768": rank0("layers.2.ffn.shared_experts.w2", 1),
        "fp8 engram wkv r0 8544x6144": rank0("layers.1.engram.wkv", 0),
    }
    packed, plans, requests = {}, {}, []
    for label, (w, s) in linears.items():
        n, k = w.shape
        packed[label] = (bfl.pack_weight(w, s, block_size=(32, 32)), n, k)
        for mode in ("default", "one-slice"):
            for cap in FP8_CAPS:
                caps = bfl.Caps(device=device, max_tokens=cap, in_features=k, out_features=n,
                                source_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
                                block_size=(32, 32), output_mode="provided")
                plan = bfl.plan(caps)
                if mode == "one-slice":
                    try:
                        from b12x.gemm.block_fp8_linear._tuning import TUNING
                        from b12x.preparation.device import detect_device
                        default = TUNING.default_config(plan.query, detect_device(device).identity)
                        if default.split_k_slices > 1:
                            plan = bfl.plan(caps, override=replace(default, split_k_slices=1))
                    except Exception as error:  # noqa: BLE001
                        print(json.dumps({"op": label, "cap": cap, "one_slice_error": repr(error)[:200]}))
                src = torch.randn(cap, k, device=device, dtype=torch.bfloat16)

                def prepare(state, src=src, n=n, p=packed[label][0]):
                    spec, = state.scratch.scratch_specs()
                    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
                    out = torch.empty((src.shape[0], n, 1), dtype=torch.bfloat16, device=device)
                    binding = state.bind(scratch=scratch, source=src, packed_weight=p, output=out)
                    return PreparedCall(run=lambda: state.run_binding(binding), output=out, owners=(scratch, binding))

                plans[label, mode, cap] = plan
                requests.append(plan.request(name=f"{label}-{mode}-{cap}", prepare_call=prepare))
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(requests))
        for label, (p, n, k) in packed.items():
            for mode in ("default", "one-slice"):
                configs = {}
                for c in FP8_CAPS:
                    configs.setdefault(str(plans[label, mode, c].selection.config)[:170], []).append(c)
                print(json.dumps({"op": label, "mode": mode, "configs": configs}))

                def run(x, label=label, mode=mode, p=p, n=n):
                    plan = plans[label, mode, smallest(FP8_CAPS, x.shape[0])]
                    spec, = plan.scratch_specs()
                    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
                    out = torch.empty((x.shape[0], n, 1), dtype=torch.bfloat16, device=device)
                    bfl.run(binding=bfl.bind(plan, scratch=scratch, source=x.contiguous(), packed_weight=p, output=out))
                    return out[:, :, 0]

                transitions(f"{label} [{mode}]", run, k)


def family_mhc():
    from b12x.preparation import PreparationSession

    from vllm.models.deepseek_v4_1 import b12x_layers
    from vllm.utils.b12x import B12xWorkload, register_b12x_layer
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(device)
    H = 5120
    prepared = tuple(c for c in GEMV_CAPS if c != LIMIT)
    b12x_layers._execution_capacities = lambda: (1, 8, LIMIT)
    owners = {}
    requests = []
    for layer in (0, 1):
        owner = torch.nn.Module()
        for part in ("attn", "ffn"):
            for kind in ("fn", "scale", "base"):
                setattr(owner, f"hc_{part}_{kind}", raw(f"layers.{layer}.hc_{part}_{kind}").float().contiguous())
            setattr(owner, f"{part}_norm", types.SimpleNamespace(weight=bf16(f"layers.{layer}.{part}_norm.weight")))
        owner.hc_attn_fn_broadcast = owner.hc_attn_fn.view(-1, 4, H).sum(dim=1).contiguous() if layer == 0 else None
        module = b12x_layers.B12xMHC(types.SimpleNamespace(hidden_size=H, rms_norm_eps=1e-20, hc_eps=1e-6,
                                                           hc_sinkhorn_iters=20, hc_mult=4))
        owner._b12x_mhc = module
        name = f"mhc-map-{layer}"
        register_b12x_layer(name, owner)
        module.bind_layer_name(name)
        workload = B12xWorkload(stage="weights", token_counts=tuple(sorted({*prepared, LIMIT})),
                                fixed_token_counts=prepared, output_dtype=torch.bfloat16,
                                max_tokens=LIMIT, max_seqs=8, max_model_len=LIMIT)
        requests += [r for unit in module.get_b12x_preparation_units(owner, workload) for r in unit.requests]
        owners[layer] = (owner, module)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(requests))
        _, m1 = owners[1]
        configs = {}
        for (op, cap), plan in sorted(m1._plans.items()):
            configs.setdefault(f"{op} {str(plan.selection.config)[:220]}", []).append(cap)
        print(json.dumps({"op": "mhc", "configs": configs}))

        def make(n, gen):
            return {"x2": torch.randn(n, H, generator=gen, device=device).to(torch.bfloat16),
                    "x3": torch.randn(n, 4, H, generator=gen, device=device).to(torch.bfloat16),
                    "post": torch.rand(n, 4, generator=gen, device=device),
                    "comb": torch.rand(n, 4, 4, generator=gen, device=device),
                    "pre": torch.softmax(torch.randn(n, 4, generator=gen, device=device), dim=-1)}

        class Rows:  # a batch of row indices standing for rows of every pooled tensor
            def __init__(self, idx):
                self.idx = idx

        gen = torch.Generator(device=device).manual_seed(20261001)
        pool = make(LIMIT, gen)
        def smallest_plan(self, operation, tokens):
            capacities = sorted(rows for o, rows in self._plans if o == operation)
            return self._plans[(operation, next(r for r in capacities if r >= int(tokens)))]

        for lookup, op in ((lookup, op) for lookup in ("smallest", "exact-or-max")
                           for op in ("layer-0 pre", "pre", "post", "post_pre")):
            owner, module = owners[0 if op.startswith("layer-0") else 1]
            if lookup == "smallest":
                module._plan_for = types.MethodType(smallest_plan, module)
            else:
                module.__dict__.pop("_plan_for", None)

            def call(idx, op=op, owner=owner, module=module):
                take = lambda key: pool[key][idx].contiguous()  # noqa: E731
                if op == "layer-0 pre":
                    out = module.pre(take("x2"), owner.hc_attn_fn_broadcast, owner.hc_attn_scale,
                                     owner.hc_attn_base, owner.attn_norm.weight, None)
                elif op == "pre":
                    out = module.pre(take("x3"), owner.hc_attn_fn, owner.hc_attn_scale, owner.hc_attn_base,
                                     owner.attn_norm.weight, take("pre"))
                elif op == "post":
                    out = module.post(take("x2"), take("x3"), take("post"), take("comb"))
                else:
                    out = module.post_pre(take("x2"), take("x3"), take("post"), take("comb"), owner.hc_ffn_fn,
                                          owner.hc_ffn_scale, owner.hc_ffn_base, owner.ffn_norm.weight, take("pre"))
                return [t for t in (out if isinstance(out, (list, tuple)) else (out,)) if torch.is_tensor(t)]

            transitions(f"mhc {op} [{lookup}]", call, 1, make=lambda n, g: torch.arange(n, device=device))


def family_norm():
    from vllm.models.deepseek_v4_1 import b12x_layers
    from vllm.utils.b12x import B12xWorkload
    from b12x.preparation import PreparationSession

    b12x_layers._capacity = lambda: LIMIT
    norms = {"norm q 1280": (1280, "layers.2.attn.q_norm.weight"), "norm kv 512": (512, "layers.2.attn.kv_norm.weight"),
             "norm index key 128": (128, "layers.2.attn.indexer.k_norm.weight"),
             "norm final 5120": (5120, "norm.weight")}
    modules, requests = {}, []
    workload = B12xWorkload(stage="weights", token_counts=GEMV_CAPS, fixed_token_counts=GEMV_CAPS[:-1],
                            output_dtype=torch.bfloat16, max_tokens=LIMIT, max_seqs=8, max_model_len=LIMIT)
    for label, (width, name) in norms.items():
        norm = b12x_layers.B12xRMSNorm(width, 1e-6).to(device)
        norm.weight.data.copy_(bf16(name))
        modules[label] = norm
        requests += [r for unit in norm.get_b12x_preparation_units(norm, workload) for r in unit.requests]
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(requests))
        for label, norm in modules.items():
            transitions(label, lambda x, norm=norm: norm(x), norm.weight.shape[0])


def family_moe():
    os.environ["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe.workloads import TUNING_WORKLOAD_VERSION
    from b12x.preparation import FrozenMapping, PreparationSession
    from benchmarks.benchmark_ds4_moe import make_synthetic_mxfp4_moe

    from vllm.model_executor.layers.fused_moe.b12x import _prepared_moe_call_factory

    E, H, I, TOPK = 384, 5120, 768, 6
    weights = make_synthetic_mxfp4_moe(E, H, I, seed=7, device=device)
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"),
                                      w13_layout=fused_moe.W13Layout("w13")),
        activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A8, nonlinearity="silu",
                                            io_dtype=torch.bfloat16),
        geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I))
    experts = fused_moe.prepare_weights(plan=weight_plan, weights=fused_moe.PackedWeights(
        w13=weights["w13_fp4"], w2=weights["w2_fp4"], w13_block_scales=weights["w13_mx"],
        w2_block_scales=weights["w2_mx"], w13_global_scales=weights["alphas"], w2_global_scales=weights["alphas"],
        input_scale=weights["input_scale"], intermediate_scale=weights["input_scale"], immutable_input_scales=True))
    counts = tuple(sorted({LIMIT, *GRAPH}))
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=LIMIT, top_k=TOPK, warmup_token_counts=counts,
                                             route_num_experts=0),
        routing=fused_moe.RoutingSpec(apply_router_weight_on_input=False),
        invocation=FrozenMapping({"tuning_route_pattern": TUNING_WORKLOAD_VERSION}))
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        if hasattr(plan, "token_counts"):  # a composite: one variant per warm count
            calls = {c: _prepared_moe_call_factory(tokens=c, topk=TOPK, prepared=experts,
                                                   output_dtype=torch.bfloat16) for c in plan.token_counts}
            session.prepare((plan.request(name="moe-map", prepare_calls=calls, benchmark_calls=calls),))
        else:  # one warm count: a plain plan
            call = _prepared_moe_call_factory(tokens=LIMIT, topk=TOPK, prepared=experts, output_dtype=torch.bfloat16)
            session.prepare((plan.request(name="moe-map", prepare_call=call, benchmark_call=call),))
        variants = getattr(plan, "variants", None) or getattr(getattr(plan, "prepared", None), "variants", None)
        if variants is not None and hasattr(variants, "items"):
            print(json.dumps({"op": "moe", "variants": {str(k): str(getattr(getattr(v, "selection", None), "config", v))[:240]
                                                        for k, v in sorted(variants.items())}}))
        scratch = torch.empty(sum(s.nbytes for s in plan.scratch_specs()), dtype=torch.uint8, device=device)
        gen = torch.Generator(device=device).manual_seed(20261001)
        x = (torch.randn(LIMIT, H, generator=gen, device=device) * 2.0).to(torch.bfloat16)
        logits = torch.randn(LIMIT, E, generator=gen, device=device)
        top, ids = torch.topk(logits, TOPK, dim=-1)
        w, ids = torch.softmax(top, dim=-1).float().contiguous(), ids.to(torch.int32).contiguous()

        def call(idx):
            out = torch.empty((idx.numel(), H), dtype=torch.bfloat16, device=device)
            binding = fused_moe.bind(plan, scratch=scratch, a=x[idx].contiguous(), experts=experts,
                                     topk_weights=w[idx].contiguous(), topk_ids=ids[idx].contiguous(), output=out,
                                     input_scales_static=True)
            fused_moe.run(binding=binding)
            return out

        transitions("moe deterministic (graph sizes + 4096 warm)", call, 1,
                    make=lambda n, g: torch.arange(n, device=device))


def family_head():
    from b12x.gemm import bf16_gemv
    from b12x.preparation import PreparationSession, PreparedCall

    head = bf16("head.weight")
    rows = -(-head.shape[0] // 3)
    weight = head[:rows].contiguous()
    del head
    caps = tuple(c for c in GRAPH if c <= 8)
    plans, requests = {}, []
    for cap in caps:
        query = bf16_gemv.GemvQuery(source_dtype="bfloat16", weight_dtype="bfloat16", max_rows=cap,
                                    in_features=weight.shape[1], out_features=weight.shape[0], source_contiguous=True,
                                    source_aligned=True, weight_contiguous=True, weight_aligned=True)
        plan = bf16_gemv.plan(query)

        def make_call(state):
            q = state.query
            source = torch.empty((q.max_rows, q.in_features), device=device, dtype=torch.bfloat16)
            out = torch.empty((q.max_rows, q.out_features), device=device, dtype=torch.bfloat16)
            return PreparedCall(run=lambda: state.run(source, weight, out=out),
                                produce=lambda: source.normal_(std=0.25), owners=(weight,))

        plans[cap] = plan
        requests.append(plan.request(name=f"head-{cap}", prepare_call=make_call, benchmark_call=make_call))
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(requests))
        print(json.dumps({"op": "lm head", "configs": {str(c): str(plans[c].selection.config) for c in caps}}))

        def call(x):
            if x.shape[0] <= max(caps):
                return bf16_gemv.mm(x.contiguous(), weight, plan=plans[smallest(caps, x.shape[0])],
                                    output_dtype=torch.bfloat16)
            return torch.nn.functional.linear(x, weight)

        global SIZES
        SIZES = [m for m in SIZES if m <= 512]  # logit rows: decode verify rows and prefill last rows
        transitions("lm head r0 43094x5120 (b12x plans to 8 rows, F.linear above)", call, weight.shape[1])


def family_wo():
    from b12x.gemm import wo_projection
    from b12x.preparation import PreparationSession, PreparedCall

    groups, rank, group_width, hidden, heads, nope, rope = 3, 1024, 4096, 5120, 24, 448, 64
    wa, sa = raw("layers.2.attn.wo_a.weight"), raw("layers.2.attn.wo_a.scale").float()
    wb, sb = raw("layers.2.attn.wo_b.weight"), raw("layers.2.attn.wo_b.scale").float()
    weights = wo_projection.pack_weights(
        wa[: groups * rank].contiguous(), sa[: groups * rank // 32].contiguous(),
        wb[:, : groups * rank].contiguous(), sb[:, : groups * rank // 32].contiguous(),
        groups=groups, group_width=group_width, rank=rank, hidden=hidden, block_size=(32, 32))
    gen = torch.Generator(device=device).manual_seed(20261001)
    angles = torch.rand(8192, rope // 2, generator=gen, device=device) * 6.283
    table = torch.cat([angles.cos(), angles.sin()], dim=1).float().contiguous()
    invocation = dict(operation="inv_rope", heads_per_group=heads // groups, nope_dim=nope, rope_dim=rope,
                      positions_dtype="int64", cos_sin_dtype="float32")

    def prepare(state):
        rows = state.query.max_tokens
        source = torch.empty((rows, heads, 512), dtype=torch.bfloat16, device=device)
        positions = torch.arange(rows, dtype=torch.int64, device=device) % table.shape[0]
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device)
                        for spec in state._scratch_state.scratch_specs())
        binding = state.bind_inv_rope(scratch=scratch, o=source, positions=positions, cos_sin_cache=table,
                                      weights=weights, heads_per_group=heads // groups, nope_dim=nope, rope_dim=rope)
        return PreparedCall(run=lambda: state.run_inv_rope(binding), produce=lambda: source.normal_(std=0.25),
                            owners=(weights, table))

    plans = {}
    for key, rows, dynamic in [*((c, c, False) for c in range(1, 97)), ("prefill", LIMIT, True)]:
        plans[key] = wo_projection.plan(
            wo_projection.Caps(device=device, max_tokens=rows, groups=groups, group_width=group_width, rank=rank,
                               hidden=hidden), invocation=dict(invocation, dynamic_tokens=dynamic))
    requests = [p.request(name=f"wo-{k}", prepare_call=prepare, benchmark_call=prepare) for k, p in plans.items()]
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(requests))
        print(json.dumps({"op": "wo", "configs": {str(k): str(p.selection.config) for k, p in
                                                   list(plans.items())[:3] + [("prefill", plans["prefill"])]}}))
        o_pool = (torch.randn(LIMIT, heads, 512, generator=gen, device=device) * 0.25).to(torch.bfloat16)
        pos_pool = torch.randint(0, table.shape[0], (LIMIT,), generator=gen, device=device, dtype=torch.int64)

        def call(idx, key):
            plan = plans[key]
            scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in plan.scratch_specs())
            binding = wo_projection.bind_inv_rope(plan, scratch=scratch, o=o_pool[idx].contiguous(),
                                                  positions=pos_pool[idx].contiguous(), cos_sin_cache=table,
                                                  weights=weights, heads_per_group=heads // groups,
                                                  nope_dim=nope, rope_dim=rope)
            return wo_projection.run_inv_rope(binding=binding, plan=plan)

        groups_seen, order, rows_out = {}, [], []
        target = torch.tensor([LIMIT - 1], device=device)
        for m in SIZES:
            for pos in (POSITIONS if m > 1 else ("e",)):
                others = torch.arange(m - 1, device=device)
                idx = torch.cat([others, target]) if pos == "e" else torch.cat([target, others])
                row = m - 1 if pos == "e" else 0
                for kind in (("decode", m), ("prefill", "prefill")) if m <= 96 else (("prefill", "prefill"),):
                    try:
                        key = digest(call(idx, kind[1])[row: row + 1])
                    except Exception as error:  # noqa: BLE001
                        key = "error:" + type(error).__name__ + ":" + str(error)[:80]
                    if key not in groups_seen:
                        order.append(key)
                    groups_seen.setdefault(key, []).append(f"{kind[0][0]}{m}{pos}")
        torch.cuda.synchronize()
        print(json.dumps({"op": "wo inverse-RoPE (d = decode static plan, p = prefill dynamic plan)",
                          "groups": len(groups_seen),
                          "members": {k[:16]: (v[:12] + [f"... {len(v)} total"]) if len(v) > 12 else v
                                      for k, v in groups_seen.items()}}), flush=True)


REFERENCE_MHC = ("MhcConfig(backend='native', projection_tile_m=16, projection_tile_n=8, projection_tile_k=256, "
                 "projection_num_stages=1, projection_num_m_warps=1, projection_num_n_warps=1, "
                 "projection_k_splits=1, lagged_prepare={lagged}, partials_per_cta=4)")


def family_reference():
    """The candidate reference configuration, against serving's actual warm sets.

    - mHC with serving's warm counts (graph sizes and 4096): as served, and with
      every pre, post_pre and post plan forced to the native configuration that
      post already uses below 4096 rows (one K slice, no lagged prepare).
    - The MoE with only the 4096-token variant warm, so every count binds it.
    - The LM head with SIMT GEMV plans at every capacity up to 512 logit rows.
    """
    global GEMV_CAPS, GRAPH, SIZES
    from b12x.norm.mhc._tuning import MhcConfig

    from vllm.models.deepseek_v4_1 import b12x_layers

    real = b12x_layers.mhc

    class Forced:
        reference = None

        def __getattr__(self, name):
            return getattr(real, name)

        def plan(self, caps, *, invocation=None, override=None):
            return real.plan(caps, invocation=invocation, override=self.reference)

    saved = GEMV_CAPS
    GEMV_CAPS = (*GRAPH, LIMIT)  # family_mhc warms GEMV_CAPS minus the limit: serving's set
    try:
        print(json.dumps({"family": "mhc as served (graph sizes + 4096 warm)"}), flush=True)
        family_mhc()
        for lagged in (False, True):
            forced = Forced()
            forced.reference = eval(REFERENCE_MHC.format(lagged=lagged), {"MhcConfig": MhcConfig})  # noqa: S307
            b12x_layers.mhc = forced
            print(json.dumps({"family": f"mhc reference (native, one K slice, lagged_prepare={lagged})"}),
                  flush=True)
            family_mhc()
            b12x_layers.mhc = real
    finally:
        b12x_layers.mhc = real
        GEMV_CAPS = saved

    graph = GRAPH
    GRAPH = ()  # family_moe warms GRAPH plus the limit
    try:
        print(json.dumps({"family": "moe reference (only the 4096-token variant warm)"}), flush=True)
        family_moe()
    finally:
        GRAPH = graph

    from b12x.gemm import bf16_gemv
    from b12x.gemm.bf16_gemv._tuning import GemvConfig
    from b12x.preparation import PreparationSession, PreparedCall

    head = bf16("head.weight")
    weight = head[: -(-head.shape[0] // 3)].contiguous()
    del head
    caps = (*GRAPH, 64, 128, 256, 512)
    plans = {}
    for cap in caps:
        query = bf16_gemv.GemvQuery(source_dtype="bfloat16", weight_dtype="bfloat16", max_rows=cap,
                                    in_features=weight.shape[1], out_features=weight.shape[0], source_contiguous=True,
                                    source_aligned=True, weight_contiguous=True, weight_aligned=True)
        plans[cap] = bf16_gemv.plan(query, override=GemvConfig(backend="simt"))

    def make_call(state):
        q = state.query
        source = torch.empty((q.max_rows, q.in_features), device=device, dtype=torch.bfloat16)
        out = torch.empty((q.max_rows, q.out_features), device=device, dtype=torch.bfloat16)
        return PreparedCall(run=lambda: state.run(source, weight, out=out),
                            produce=lambda: source.normal_(std=0.25), owners=(weight,))

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(p.request(name=f"head-ref-{c}", prepare_call=make_call, benchmark_call=make_call)
                              for c, p in plans.items()))
        saved_sizes = SIZES
        SIZES = [m for m in SIZES if m <= 512]
        transitions("lm head as served (F.linear at every row count)",
                    lambda x: torch.nn.functional.linear(x, weight), weight.shape[1])
        transitions("lm head reference (SIMT plans to 512 rows)", lambda x: bf16_gemv.mm(
            x.contiguous(), weight, plan=plans[smallest(caps, x.shape[0])], output_dtype=torch.bfloat16),
            weight.shape[1])
        SIZES = saved_sizes


if __name__ == "__main__":
    for family in sys.argv[1:]:
        print(json.dumps({"family": family, "sizes": len(SIZES)}), flush=True)
        globals()[f"family_{family}"]()
    print("transition map done", flush=True)
