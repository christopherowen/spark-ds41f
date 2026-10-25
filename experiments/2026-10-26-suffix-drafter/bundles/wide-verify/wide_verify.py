#!/usr/bin/env python3
"""The routed experts' GEMMs for one request's wide verification block.

A suffix drafter verifies 17 to 65 rows of one request in a step. Its price at one stream is
the distinct experts those rows reach, if FC1 and FC2 keep streaming each expert's weights
once at the rate the expert-rate bundle measured at three and six rows (experiments/
2026-10-26-c1-100tps: 14.2 us + 11.41 us an expert at six rows). Wider blocks put more rows
on each expert, and a copied span's repeated tokens pile rows onto a few experts. This bundle
times FC1 and FC2 (TP4 rank shapes, the production plan with coded scales, cold weights: the
expert-rate bundle's method) at 6, 17, 33, 49 and 65 rows against the number of distinct
experts, with the rows spread evenly over the experts ("even") or with three experts in
every row and the rest spread ("skew"). One JSON line a case:
{"case", "layout", "rows", "experts", "fc1_us", "fc2_us", "fc1_gbs", "fc2_gbs", ...}.
"""
import json
import statistics
import sys

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4_1.tilelang.moe import (
    SF_BLOCK,
    MoeExpertsV41Plan,
    _present,
    carve,
    workspace_bytes,
)

E, TOPK, HIDDEN, INTER = 384, 6, 5120, 576
CALLS, REPEATS = 24, 7
FC1_BYTES = 1152 * 5120 // 2 + 1152 * 5120 // 64 + 1152  # E2M1, 4-bit scale codes, row bases
FC2_BYTES = 5120 * 576 // 2 + 5120 * 576 // 64 + 5120
ROWS = (6, 17, 33, 49, 65)
EXPERTS = (6, 12, 24, 36, 48, 72, 96, 144, 192, 246)
HOT = 3  # experts every row of a skewed block reaches
DEVICE = "cuda"


def cases():
    for rows in ROWS:
        for n in EXPERTS:
            if TOPK <= n <= rows * TOPK:
                yield "even", rows, n
            if rows > TOPK and HOT + (TOPK - HOT) <= n <= HOT + rows * (TOPK - HOT):
                yield "skew", rows, n


def experts(n, k, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    packed = torch.randint(0, 256, (E, n, k // 2), generator=gen, device=DEVICE, dtype=torch.uint8)
    exps = torch.randint(118, 122, (E, n, k // 32), generator=gen, device=DEVICE).to(torch.uint8)
    return packed, exps


def spread(pool, rows, slots):
    """``rows`` x ``slots`` picks, distinct within a row, whose union is exactly ``pool``."""
    n = len(pool)
    picks = [[] for _ in range(rows)]
    for i, e in enumerate(pool):
        picks[i % rows].append(e)
    j = 0
    for row in picks:
        while len(row) < slots:
            e = pool[j % n]
            j += 1
            if e not in row:
                row.append(e)
    return picks


def routes(layout, pool, rows):
    if layout == "even":
        picks = spread(pool, rows, TOPK)
    else:
        hot, cold = pool[:HOT], pool[HOT:]
        picks = [hot + row for row in spread(cold, rows, TOPK - HOT)]
    return torch.tensor(picks, dtype=torch.int64, device=DEVICE)


def pools(n, count, seed):
    """``count`` pools of n experts, consecutive pools disjoint while 2n <= 384."""
    gen = torch.Generator().manual_seed(seed)
    out, perm = [], []
    while len(out) < count:
        if len(perm) < n:
            perm = torch.randperm(E, generator=gen).tolist()
        out.append(perm[:n])
        perm = perm[n:]
    return out


def timed(calls, flush):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for call in calls[:2]:
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    samples = []
    for _ in range(REPEATS):
        flush.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / len(calls))
    return statistics.median(samples), min(samples)


def main():
    torch.cuda.set_device(0)
    flush = torch.empty(64 << 20, dtype=torch.uint8, device=DEVICE)
    with set_current_vllm_config(VllmConfig()):
        plan = MoeExpertsV41Plan(E, TOPK, HIDDEN, INTER, swiglu_limit=10.0)
        plan.declare_kernels()
        w13, w13_sf = experts(2 * INTER, HIDDEN, 1)
        w2, w2_sf = experts(HIDDEN, INTER, 2)
        gate_up, down = plan.prepare_weights(w13.view(torch.int8), w13_sf, w2.view(torch.int8), w2_sf)
        del w13, w2
        torch.cuda.empty_cache()
        x = (torch.randn((max(ROWS), HIDDEN), device=DEVICE) * 0.5).bfloat16()
        for layout, rows, n in cases():
            dispatch = plan.dispatch[torch.int64]
            specs = plan.workspace_specs(rows, torch.int64)
            count = len(dispatch.workspace_specs(rows))
            fc1_calls, fc2_calls = [], []
            for pool in pools(n, CALLS, 1000 * rows + n):
                ids = routes(layout, pool, rows)
                weights = torch.full((rows, TOPK), 1.0 / TOPK, device=DEVICE)
                buffers = carve(torch.empty(workspace_bytes(specs), dtype=torch.uint8, device=DEVICE), specs)
                xq, x_sf, h, hq, h_sf, y = buffers[count:]
                layout_out = dispatch.dispatch(ids, weights, buffers[:count])
                plan.tk.quant.per_token_cast(x[:rows], "e4m3", SF_BLOCK, round_sf=True,
                                             use_packed_ue8m0=True, out=(xq, x_sf))
                order, tile_expert = layout_out["order"], layout_out["tile_expert"]
                xw, hw = x_sf.view(torch.uint32), h_sf.view(torch.uint32)
                plan.gate_up(xq, xw, *_present(gate_up), order, tile_expert, h)
                plan.swiglu(h, hq, h_sf, order, layout_out["layout_weights"], tile_expert)
                fc1_calls.append(lambda a=(xq, xw, order, tile_expert, h): plan.gate_up(
                    a[0], a[1], *_present(gate_up), a[2], a[3], a[4]))
                fc2_calls.append(lambda a=(hq, hw, order, tile_expert, y): plan.down(
                    a[0], a[1], *_present(down), a[2], a[3], a[4]))
            torch.cuda.synchronize()
            fc1, fc1_min = timed(fc1_calls, flush)
            fc2, fc2_min = timed(fc2_calls, flush)
            print(json.dumps({
                "case": "experts", "layout": layout, "rows": rows, "experts": n,
                "fc1_us": round(fc1, 2), "fc2_us": round(fc2, 2),
                "fc1_min_us": round(fc1_min, 2), "fc2_min_us": round(fc2_min, 2),
                "fc1_gbs": round(n * FC1_BYTES / fc1 / 1e3, 1), "fc2_gbs": round(n * FC2_BYTES / fc2 / 1e3, 1)}),
                flush=True)
            del fc1_calls, fc2_calls
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
