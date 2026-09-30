#!/usr/bin/env python3
"""Does the routed MoE give a row the same bits whatever else is in the batch? (GPU; cluster stopped)

usage: moe_batch_invariance.py [--atomic]   (cwd: the B12X checkout; det-masked overlay for
                                             the deterministic mode)

DS4.1 TP3 routed MoE (384 experts, hidden 5120, intermediate 768, top 6, W4A8).
Nineteen fixed target rows with fixed routing (the traced JSON prefill) run
alone and inside larger batches: neighbours after or before the target,
batches of 28, 40 and 48 rows, different neighbour contents, neighbours routed
to the target's experts (sharing M tiles) or to disjoint experts, and a lone
neighbour row. Each case prints whether the target rows' output equals the
alone result bitwise, and how many rows differ. With --atomic the atomic
combine is used instead of the deterministic one.
"""
import os
import sys
from contextlib import ExitStack

sys.path.insert(0, os.getcwd())
os.environ.setdefault("B12X_W4A8_TINY_DECODE", "0")
if "--atomic" not in sys.argv:
    os.environ["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"

import torch  # noqa: E402

from b12x.moe import fused_moe  # noqa: E402
from b12x.moe.fused_moe import _impl  # noqa: E402
from benchmarks.benchmark_ds4_moe import make_synthetic_mxfp4_moe  # noqa: E402
from tests._reference.helpers import make_tp_moe_fp4_binding  # noqa: E402

E, H, I, TOPK, T = 384, 5120, 768, 6, 19
device = torch.device("cuda")
weights = make_synthetic_mxfp4_moe(E, H, I, seed=7, device=device)
plan = fused_moe.plan_weights(
    source=fused_moe.PackedSource(format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"),
                                  w13_layout=fused_moe.W13Layout("w13")),
    activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A8, nonlinearity="silu",
                                        io_dtype=torch.bfloat16),
    geometry=fused_moe.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I),
)
experts = fused_moe.prepare_weights(plan=plan, weights=fused_moe.PackedWeights(
    w13=weights["w13_fp4"], w2=weights["w2_fp4"], w13_block_scales=weights["w13_mx"],
    w2_block_scales=weights["w2_mx"], w13_global_scales=weights["alphas"],
    w2_global_scales=weights["alphas"], input_scale=weights["input_scale"],
    intermediate_scale=weights["input_scale"], immutable_input_scales=True))


def rows(n, seed, experts_from=None):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(n, H, generator=gen, device=device) * 2.0).to(torch.bfloat16)
    logits = torch.randn(n, E, generator=gen, device=device)
    if experts_from is not None:            # restrict routing to a given expert set
        mask = torch.full((E,), float("-inf"), device=device)
        mask[experts_from] = 0
        logits = logits + mask
    top, ids = torch.topk(logits, TOPK, dim=-1)
    return x, torch.softmax(top, dim=-1).float(), ids.to(torch.int32)


def run(x, w, ids):
    out = torch.zeros(x.shape[0], H, dtype=torch.bfloat16, device=device)
    with ExitStack() as stack:
        binding = stack.enter_context(make_tp_moe_fp4_binding(
            a=x.contiguous(), experts=experts, topk_weights=w.contiguous(), topk_ids=ids.contiguous(),
            output=out, input_scales_static=True, quant_mode="w4a8_mx"))
        _impl.b12x_moe_fp4(binding=binding)
        torch.cuda.synchronize()
        again = out.clone()
        _impl.b12x_moe_fp4(binding=binding)
        torch.cuda.synchronize()
        assert torch.equal(again, out), "not repeatable at fixed batch"
    return out


xt, wt, it = rows(T, 1)
target_experts = torch.unique(it).tolist()
others = [e for e in range(E) if e not in set(target_experts)]
alone = run(xt, wt, it)
print(f"mode {'atomic' if '--atomic' in sys.argv else 'deterministic'}; target {T} rows, "
      f"{len(target_experts)} distinct experts", flush=True)


def case(label, pieces, target_at):
    x = torch.cat([p[0] for p in pieces])
    w = torch.cat([p[1] for p in pieces])
    ids = torch.cat([p[2] for p in pieces])
    out = run(x, w, ids)[target_at:target_at + T]
    bad = (out != alone).any(dim=1)
    print(f"  {label:52s} batch {x.shape[0]:2d}: equal {bool(not bad.any())}, "
          f"target rows differing {int(bad.sum())}/{T}", flush=True)


target = (xt, wt, it)
case("9 neighbours after (random experts)", [target, rows(9, 2)], 0)
case("9 neighbours before (random experts)", [rows(9, 2), target], 9)
case("9 other neighbours after (random experts)", [target, rows(9, 3)], 0)
case("21 neighbours after (random experts)", [target, rows(21, 4)], 0)
case("29 neighbours after (random experts)", [target, rows(29, 5)], 0)
case("9 neighbours after, target's experts only", [target, rows(9, 6, target_experts)], 0)
case("9 neighbours after, disjoint experts only", [target, rows(9, 7, others)], 0)
case("29 neighbours after, disjoint experts only", [target, rows(29, 8, others)], 0)
case("1 neighbour after, disjoint experts", [target, rows(1, 9, others)], 0)
