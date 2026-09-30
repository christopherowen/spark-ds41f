#!/usr/bin/env python3
"""Is the deterministic routed MoE batch-invariant through serving's one-plan path? (GPU; cluster stopped)

usage: moe_serving_replay.py   (cwd: the B12X checkout, det-masked overlay mounted)

moe_batch_invariance.py built a binding per batch size. Serving instead plans
the MoE once (vLLM's B12xExperts: fused_moe.plan_execution with the batch
token limit and the fixed graph sizes as warm counts), prepares it through
vLLM's own call factory, and binds every live batch to that plan, which
chooses a launch for the live row count. DS4.1 TP3 shapes (384 experts,
hidden 5120, intermediate 768, top 6, W4A8, synthetic weights). Nineteen fixed
target rows with fixed routing (the JSON prefill's row count) run alone and
inside batches of R rows (1-64 and larger) at the start, middle and end; each
target row is compared bitwise with the same row alone. Two warm sets: the
capture sizes only, and every prepared count below the limit seen in
serving's gate capture; 19 is in neither.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("B12X_W4A8_TINY_DECODE", "0")
os.environ["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"

import torch  # noqa: E402
from b12x.moe import fused_moe  # noqa: E402
from b12x.moe.fused_moe.workloads import TUNING_WORKLOAD_VERSION  # noqa: E402
from b12x.preparation import FrozenMapping, PreparationSession  # noqa: E402
from benchmarks.benchmark_ds4_moe import make_synthetic_mxfp4_moe  # noqa: E402

from vllm.model_executor.layers.fused_moe.b12x import _prepared_moe_call_factory  # noqa: E402

E, H, I, TOPK, T, LIMIT = 384, 5120, 768, 6, 19, 4096
CAPTURE = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48)
PREPARED = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 21, 24, 25, 28, 30, 32, 35, 40, 42, 48, 49, 56,
            72, 96, 192, 384, 768, 1536, 3072, 4091)
device = torch.device("cuda")
weights = make_synthetic_mxfp4_moe(E, H, I, seed=7, device=device)
weight_plan = fused_moe.plan_weights(
    source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"),
                                  w13_layout=fused_moe.W13Layout("w13")),
    activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A8, nonlinearity="silu",
                                        io_dtype=torch.bfloat16),
    geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I),
)
experts = fused_moe.prepare_weights(plan=weight_plan, weights=fused_moe.PackedWeights(
    w13=weights["w13_fp4"], w2=weights["w2_fp4"], w13_block_scales=weights["w13_mx"],
    w2_block_scales=weights["w2_mx"], w13_global_scales=weights["alphas"],
    w2_global_scales=weights["alphas"], input_scale=weights["input_scale"],
    intermediate_scale=weights["input_scale"], immutable_input_scales=True))


def rows(n, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(n, H, generator=gen, device=device) * 2.0).to(torch.bfloat16)
    logits = torch.randn(n, E, generator=gen, device=device)
    top, ids = torch.topk(logits, TOPK, dim=-1)
    return x, torch.softmax(top, dim=-1).float().contiguous(), ids.to(torch.int32).contiguous()


def serving_plan(warm):
    counts = tuple(sorted({LIMIT, *warm}))
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=max(counts), top_k=TOPK,
                                             warmup_token_counts=counts, route_num_experts=0),
        routing=fused_moe.RoutingSpec(apply_router_weight_on_input=False),
        invocation=FrozenMapping({"tuning_route_pattern": TUNING_WORKLOAD_VERSION}),
    )
    calls = {count: _prepared_moe_call_factory(tokens=count, topk=TOPK, prepared=experts,
                                               output_dtype=torch.bfloat16)
             for count in plan.token_counts}
    return plan, plan.request(name=f"moe-{len(counts)}", prepare_calls=calls, benchmark_calls=calls)


plans = {name: serving_plan(warm) for name, warm in (("capture sizes", CAPTURE), ("all prepared", PREPARED))}
with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
    session.prepare(tuple(request for _, request in plans.values()))
    xt, wt, it = rows(T, 1)
    pool = rows(600, 2)
    for name, (plan, _) in plans.items():
        scratch = torch.empty(sum(spec.nbytes for spec in plan.scratch_specs()), dtype=torch.uint8,
                              device=device)

        def run(x, w, ids):
            out = torch.empty_like(x)
            binding = fused_moe.bind(plan, scratch=scratch, a=x.contiguous(), experts=experts,
                                     topk_weights=w.contiguous(), topk_ids=ids.contiguous(), output=out,
                                     input_scales_static=True)
            fused_moe.run(binding=binding)
            return out

        alone = torch.cat([run(xt[i:i + 1], wt[i:i + 1], it[i:i + 1]) for i in range(T)])
        solo = run(xt, wt, it)
        torch.cuda.synchronize()
        print(f"[{name}] target alone as 19 rows vs each row alone: "
              f"{'=' if torch.equal(solo, alone) else 'DIFF'} "
              f"({int((solo != alone).any(1).sum())}/{T} rows)", flush=True)
        bad = []
        for R in list(range(1, 65)) + [72, 96, 128, 192, 256, 384, 512]:
            if R < T:
                cases = [("start", xt[:R], wt[:R], it[:R], 0, R)]
            else:
                n = R - T
                nx, nw, ni = (p[:n] for p in pool)
                cases = [(pos, torch.cat([nx[:k], xt, nx[k:]]), torch.cat([nw[:k], wt, nw[k:]]),
                          torch.cat([ni[:k], it, ni[k:]]), k, T)
                         for pos, k in (("start", 0), ("middle", n // 2), ("end", n))]
            for pos, x, w, ids, k, t in cases:
                out = run(x, w, ids)[k:k + t]
                if not torch.equal(out, alone[:t]):
                    bad.append((R, pos, int((out != alone[:t]).any(1).sum()), t))
        torch.cuda.synchronize()
        if bad:
            by_r = sorted({b[0] for b in bad})
            print(f"[{name}] target rows differ from rows alone at R in {by_r}", flush=True)
            print(f"[{name}] equal at R in {sorted(set(list(range(1, 65)) + [72, 96, 128, 192, 256, 384, 512]) - set(by_r))}",
                  flush=True)
        else:
            print(f"[{name}] every target row equals the row alone at every R and position", flush=True)
