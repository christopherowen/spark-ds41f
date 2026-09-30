#!/usr/bin/env python3
"""Does the deterministic routed MoE give a row the same bits at every position of a small batch?
(GPU; cluster stopped)

usage: moe_position_replay.py   (cwd: the B12X checkout, det-variant overlay mounted)

The position-aligned trace with all fixes found the same row (same position,
token and prefix) computed in two decode steps of equal size with different
layer-0 routed outputs and identical MoE input and router logits: the row moved
from a draft slot to the bonus slot of its verify block. This replays one
target row through serving's one-plan path (every prepared count warm) at
every position of batches of 2-8 rows and of 28, with three different sets of
neighbours each, and groups the target row's output by its exact bits. It
also checks each call repeats bitwise.

---- earlier docstring (moe_serving_replay.py) follows ----

usage: moe_serving_replay.py   (cwd: the B12X checkout, det-masked overlay mounted)

moe_batch_invariance.py built a binding per batch size. Serving instead plans
the MoE once (vLLM's B12xExperts: fused_moe.plan_execution with the batch
token limit and the fixed graph sizes as warm counts), prepares it through
vLLM's own call factory, and binds every live batch to that plan, which
chooses a launch for the live row count. DS4.1 TP3 shapes (384 experts,
hidden 5120, intermediate 768, top 6, W4A8, synthetic weights). Nineteen fixed
target rows with fixed routing (the JSON prefill's row count) run alone and
inside batches of R rows (19-64 and larger) at the start, middle and end,
and the first R target rows (R 1-18) run alone.
Batch sizes are grouped by the exact bits of the target rows: one group means
batch-invariant; the group holding the solo 19-row call is marked. (A single
row alone takes the M=1 launch, which differs from every batch.) Two warm
sets: the capture sizes only, and every prepared count below the limit seen
in serving's gate capture; 19 is in neither.
"""
import hashlib
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


plan, request = serving_plan(PREPARED)
with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
    session.prepare((request,))
    scratch = torch.empty(sum(spec.nbytes for spec in plan.scratch_specs()), dtype=torch.uint8, device=device)

    def run(x, w, ids):
        out = torch.empty_like(x)
        binding = fused_moe.bind(plan, scratch=scratch, a=x.contiguous(), experts=experts,
                                 topk_weights=w.contiguous(), topk_ids=ids.contiguous(), output=out,
                                 input_scales_static=True)
        fused_moe.run(binding=binding)
        return out

    xt, wt, it = rows(1, 1)
    groups, repeat_bad = {}, 0
    for R in (1, 2, 3, 4, 5, 6, 7, 8, 28):
        for seed in (2, 3, 4):
            nx, nw, ni = rows(R - 1, seed) if R > 1 else (xt[:0], wt[:0], it[:0])
            for i in range(R):
                x = torch.cat([nx[:i], xt, nx[i:]])
                w = torch.cat([nw[:i], wt, nw[i:]])
                ids = torch.cat([ni[:i], it, ni[i:]])
                out = run(x, w, ids)
                again = run(x, w, ids)
                repeat_bad += int(not torch.equal(out, again))
                key = hashlib.sha256(out[i].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:8]
                groups.setdefault(key, []).append(f"R{R}p{i}s{seed}")
    torch.cuda.synchronize()
    print(f"calls not repeating bitwise: {repeat_bad}", flush=True)
    print(f"{len(groups)} groups of (batch rows R, target position p, neighbour seed s):", flush=True)
    for key, members in groups.items():
        sizes = sorted({m.split("p")[0] for m in members}, key=lambda s: int(s[1:]))
        print(f"  {key}: {len(members)} calls, batch sizes {' '.join(sizes)}; e.g. {' '.join(members[:12])}", flush=True)
