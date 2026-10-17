"""moe_swiglu_cast (vLLM 0181) against TileKernels' swiglu_forward_and_per_token_cast, as the
serving image runs it (spark-ds41f patch 0001: the clamp keeps a NaN), on real dispatch
layouts from vLLM's MoE dispatch plan:

- TP4 (576 columns a rank) and TP3 (768) target experts: 384 experts, top 6;
- the drafter's experts at TP4: 128 experts, top 3.

For every row count of the serving decode captures and the prefill chunks:
1. every live row's E4M3 bytes and UE8M0 scales equal TileKernels';
2. time per call, warm and cold, in three rounds with alternating order;
3. graph replays give the eager bits (temperature-0 serving replays graphs).

Exit 0 only if every bit check passes; the timing verdict is printed per bucket, each
bucket judged by its slowest size against kbench's floor.
"""
import os
import statistics
import sys
import traceback

import torch

sys.path.insert(0, "/b")
from kbench import BUCKETS, DECODE, DEVICE, PREFILL, REPEAT, SLOWER, graph, per_call  # noqa: E402

import tile_kernels  # noqa: E402
from tilelang import language as T  # noqa: E402
from tile_kernels.quant.common import get_cast_output_config  # noqa: E402
from tile_kernels.quant.swiglu_forward_cuda import (  # noqa: E402
    get_swiglu_forward_and_per_token_cast_kernel_cuda,
)
from vllm.models.deepseek_v4_1.tilelang import prepare  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.linear import _calls  # noqa: E402
from vllm.models.deepseek_v4_1.tilelang.moe import (  # noqa: E402
    MoeDispatchV41Plan,
    carve,
    moe_swiglu_cast,
    workspace_bytes,
)

LIMIT, ALIGN = 10.0, 32
# Rows per CTA to compare (bits do not depend on it); the first is the module default.
VARIANTS = tuple(None if v == "default" else int(v)
                 for v in os.environ.get("SWIGLU_RPC", "default").split(","))
SHAPES = (("tp4", 576, 384, 6), ("tp3", 768, 384, 6), ("drafter-tp4", 576, 128, 3))
failures = 0


def theirs(inter, experts):
    tile_kernels.config.set_token_alignment(ALIGN)
    return get_swiglu_forward_and_per_token_cast_kernel_cuda(
        inter, experts, ALIGN, True, False, LIMIT, False, in_dtype=T.dtype(torch.bfloat16),
        out_config=get_cast_output_config("e4m3", (1, 32), False, True, True),
        num_sms=tile_kernels.config.get_num_sms(), use_pdl=False)


def case(rows, inter, experts, topk, dispatch, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    ids = torch.stack([torch.randperm(experts, generator=gen, device=DEVICE)[:topk]
                       for _ in range(rows)]).int().contiguous()
    weights = (torch.rand((rows, topk), generator=gen, device=DEVICE) + 0.1).contiguous()
    specs = dispatch.workspace_specs(rows)
    layout = dispatch.dispatch(
        ids, weights, carve(torch.empty(workspace_bytes(specs), dtype=torch.uint8, device=DEVICE), specs))
    n = layout["order"].shape[0]
    h = torch.randn((n, 2 * inter), generator=gen, device=DEVICE) * 6
    h[::7, : inter // 4] = -20.0
    h[3::11, 64:96] *= 1e-30
    return layout, h.bfloat16().contiguous()


def outputs(n, inter):
    out = torch.zeros((n, inter), dtype=torch.float8_e4m3fn, device=DEVICE)
    sf = torch.zeros((n, (inter // 32 + 3) // 4 * 4), dtype=torch.uint8, device=DEVICE)
    return out, sf


def same(a, a_sf, b, b_sf, live, cols):
    return torch.equal(a[live].view(torch.uint8), b[live].view(torch.uint8)) and torch.equal(
        a_sf[live, :cols], b_sf[live, :cols])


def measure(name, inter, experts, topk, dispatch, k_theirs, rpc):
    """One variant's table; returns the number of failed sizes."""
    failed = 0
    k_ours = moe_swiglu_cast(inter, ALIGN, LIMIT, rows_per_cta=rpc)
    print(f"== {name}: {inter} columns, {experts} experts, top {topk}, {rpc or "default"} rows a CTA", flush=True)
    print("rows  tiles   theirs us (warm/cold)    ours us (warm/cold)   ours/theirs (warm/cold)  bits  replay",
          flush=True)
    ratios = {}
    for rows in DECODE + PREFILL:
        try:
            layout, h = case(rows, inter, experts, topk, dispatch, rows)
            live = layout["order"] >= 0
            o1, s1 = outputs(h.shape[0], inter)
            o2, s2 = outputs(h.shape[0], inter)
            run_ours = lambda: k_ours(h, o1, s1, layout["order"], layout["layout_weights"], layout["tile_expert"])
            run_theirs = lambda: k_theirs(h, o2, s2, layout["psum"], layout["layout_weights"], 1.0, None)
            run_ours()
            run_theirs()
            torch.cuda.synchronize()
            bits = same(o1, s1, o2, s2, live, inter // 32)
            replay = "-"
            if rows in REPEAT:
                eager, eager_sf = o1.clone(), s1.clone()
                g = graph(run_ours, 1)
                ok = True
                for _ in range(2):
                    o1.zero_()
                    s1.zero_()
                    g.replay()
                    torch.cuda.synchronize()
                    ok = ok and same(o1, s1, eager, eager_sf, live, inter // 32)
                replay = "equal" if ok else "DIFFER"
                bits = bits and ok
            ours_t, theirs_t, warm, cold = [], [], [], []
            for r in range(3):
                if r % 2:
                    a, b = per_call(run_theirs, rows), per_call(run_ours, rows)
                else:
                    b, a = per_call(run_ours, rows), per_call(run_theirs, rows)
                theirs_t.append(a)
                ours_t.append(b)
                warm.append(b[0] / a[0])
                cold.append(b[1] / a[1])
            ratios[rows] = (statistics.median(warm), statistics.median(cold))
            tw = statistics.median(t[0] for t in theirs_t)
            tc = statistics.median(t[1] for t in theirs_t)
            ow = statistics.median(t[0] for t in ours_t)
            oc = statistics.median(t[1] for t in ours_t)
            failed += not bits
            print(f"{rows:5d} {layout['tile_expert'].shape[0]:6d}  {tw:9.2f} / {tc:9.2f}  {ow:9.2f} / {oc:9.2f}  "
                  f"{ratios[rows][0]:6.3f} / {ratios[rows][1]:6.3f}  {'equal' if bits else 'DIFFER'}  {replay}",
                  flush=True)
        except Exception:
            failed += 1
            print(f"{rows:5d} ERROR\n{traceback.format_exc(limit=6)}", flush=True)
    for bucket, lo, hi in BUCKETS:
        sel = [v for k, v in ratios.items() if lo <= k <= hi]
        if sel:
            worst = max(max(w, c) for w, c in sel)
            print(f"  {bucket}: worst ours/theirs {worst:.3f} "
                  f"({'within' if worst <= SLOWER else 'OVER'} the {SLOWER} floor)", flush=True)
    return failed


for name, inter, experts, topk in SHAPES:
    dispatch = MoeDispatchV41Plan(experts, topk, ALIGN, 512, torch.int32, torch.bfloat16)
    prepare.declare(lambda d=dispatch: _calls(d.kernels()), bind=dispatch.bind)
    k_theirs = theirs(inter, experts)
    for rpc in VARIANTS:
        failures += measure(name, inter, experts, topk, dispatch, k_theirs, rpc)

print(f"{len(SHAPES)} passed" if not failures else f"{failures} failed", flush=True)
sys.exit(1 if failures else 0)
