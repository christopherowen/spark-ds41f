#!/usr/bin/env python3
"""The routed experts' GEMMs against the number of distinct experts the rows reach.

At one stream a decode step routes 1 + verified drafts rows, six experts a row, and
FC1 (gate/up) and FC2 (down) stream every distinct expert's weights once. The traces
give each layer's FC1 and FC2 time but not how many experts it read, so they cannot
say whether FC1 runs at GB10's attainable rate. This bundle fixes the count: at 3 and
6 rows, routes that reach exactly N distinct of the 384 experts (TP4 rank shapes,
hidden 5,120, 576 expert columns, the production plan with coded scales), timed on
cold weights:

- each graph runs CALLS launches, each on its own pool of N experts, consecutive
  pools disjoint, so no launch finds its weights in L2 (24 MiB);
- the L2 is flushed (a 64 MiB write) before every replay;
- FC1 and FC2 are timed apart, as their own graphs, on the same pools.

A plain read of the same bytes (torch.sum over FP32 views of cold buffers) gives the
rate a stream reaches with no GEMM around it. One JSON line a case:
{"case", "rows", "experts", "fc1_us", "fc2_us", "fc1_gbs", "fc2_gbs", ...}.
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
CASES = [(6, n) for n in (6, 9, 12, 15, 18, 21, 24, 30, 36)] + [(3, n) for n in (6, 9, 12, 15, 18)]
DEVICE = "cuda"


def experts(n, k, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    packed = torch.randint(0, 256, (E, n, k // 2), generator=gen, device=DEVICE, dtype=torch.uint8)
    exps = torch.randint(118, 122, (E, n, k // 32), generator=gen, device=DEVICE).to(torch.uint8)
    return packed, exps


def routes(pool, rows):
    """rows x TOPK distinct experts per row whose union is exactly ``pool``."""
    n = len(pool)
    assert TOPK <= n <= rows * TOPK
    picks = [[] for _ in range(rows)]
    for i, e in enumerate(pool):  # every expert of the pool reaches some row
        picks[i % rows].append(e)
    j = 0
    for row in picks:
        while len(row) < TOPK:
            e = pool[j % n]
            j += 1
            if e not in row:
                row.append(e)
    return torch.tensor(picks, dtype=torch.int64, device=DEVICE)


def pools(n, count, seed):
    """``count`` pools of n experts, consecutive pools disjoint."""
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
        x = (torch.randn((6, HIDDEN), device=DEVICE) * 0.5).bfloat16()
        for rows, n in CASES:
            dispatch = plan.dispatch[torch.int64]
            specs = plan.workspace_specs(rows, torch.int64)
            count = len(dispatch.workspace_specs(rows))
            fc1_calls, fc2_calls = [], []
            for i, pool in enumerate(pools(n, CALLS, 1000 * rows + n)):
                ids = routes(pool, rows)
                weights = torch.full((rows, TOPK), 1.0 / TOPK, device=DEVICE)
                buffers = carve(torch.empty(workspace_bytes(specs), dtype=torch.uint8, device=DEVICE), specs)
                xq, x_sf, h, hq, h_sf, y = buffers[count:]
                layout = dispatch.dispatch(ids, weights, buffers[:count])
                plan.tk.quant.per_token_cast(x[:rows], "e4m3", SF_BLOCK, round_sf=True,
                                             use_packed_ue8m0=True, out=(xq, x_sf))
                order, tile_expert = layout["order"], layout["tile_expert"]
                xw, hw = x_sf.view(torch.uint32), h_sf.view(torch.uint32)
                plan.gate_up(xq, xw, *_present(gate_up), order, tile_expert, h)
                plan.swiglu(h, hq, h_sf, order, layout["layout_weights"], tile_expert)
                fc1_calls.append(lambda a=(xq, xw, order, tile_expert, h): plan.gate_up(
                    a[0], a[1], *_present(gate_up), a[2], a[3], a[4]))
                fc2_calls.append(lambda a=(hq, hw, order, tile_expert, y): plan.down(
                    a[0], a[1], *_present(down), a[2], a[3], a[4]))
            torch.cuda.synchronize()
            fc1, fc1_min = timed(fc1_calls, flush)
            fc2, fc2_min = timed(fc2_calls, flush)
            print(json.dumps({
                "case": "experts", "rows": rows, "experts": n, "fc1_us": round(fc1, 2),
                "fc2_us": round(fc2, 2), "fc1_min_us": round(fc1_min, 2), "fc2_min_us": round(fc2_min, 2),
                "fc1_gbs": round(n * FC1_BYTES / fc1 / 1e3, 1), "fc2_gbs": round(n * FC2_BYTES / fc2 / 1e3, 1),
                "fc2_rate_vs_fc1": round((n * FC2_BYTES / fc2) / (n * FC1_BYTES / fc1), 3)}), flush=True)
        for mb in (18, 55, 110, 220):
            bufs = [torch.ones((mb << 20) // 4, dtype=torch.float32, device=DEVICE) for _ in range(8)]
            outs = [torch.empty((), dtype=torch.float32, device=DEVICE) for _ in bufs]
            us, us_min = timed([lambda b=b, o=o: torch.sum(b, out=o) for b, o in zip(bufs, outs)], flush)
            print(json.dumps({"case": "read", "mib": mb, "us": round(us, 2),
                              "gbs": round((mb << 20) / us / 1e3, 1),
                              "gbs_best": round((mb << 20) / us_min / 1e3, 1)}), flush=True)
            del bufs
    return 0


if __name__ == "__main__":
    sys.exit(main())
