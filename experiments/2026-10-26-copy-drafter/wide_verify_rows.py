#!/usr/bin/env python3
"""The routed experts of a verification step with and without a copy block, at the
c1, c8 and c16 shapes.

From experiments/2026-10-26-suffix-drafter/bundles/wide-verify/wide_verify.py (the
production plan with coded scales, TP4 rank shapes, cold weights, an L2 flush between
replays), timing the whole routed layer as the model runs it (``plan.run``: the
dispatch, FC1, SwiGLU, FC2 and the combine). A step is its requests' rows: DSpark's
step has every request at 6 rows; the copy step has one request at 17 rows (1 + the
lookahead of 16) and the rest at 6, as the scheduler lays out one copying request.
A request's rows reach its own experts: N(6) = 24 and N(17) = 48 distinct a layer
(the route probe's copied and tool-call text, runs/routes-summary.json of the
suffix-drafter record), drawn independently for each request, its rows spread evenly
over them. The step's experts are their union. Each case times 24 draws and reports
the per-layer time and the step's 40 layers. One JSON line a case.
"""
import json
import statistics
import sys

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4_1.tilelang.linear import cast_mxfp8
from vllm.models.deepseek_v4_1.tilelang.moe import MoeExpertsV41Plan, workspace_bytes

E, TOPK, HIDDEN, INTER, LAYERS = 384, 6, 5120, 576, 40
CALLS, REPEATS = 24, 7
EXPERTS_A_REQUEST = {6: 24, 17: 48}
DEVICE = "cuda"


def experts(n, k, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    packed = torch.randint(0, 256, (E, n, k // 2), generator=gen, device=DEVICE, dtype=torch.uint8)
    exps = torch.randint(118, 122, (E, n, k // 32), generator=gen, device=DEVICE).to(torch.uint8)
    return packed, exps


def spread(pool, rows, slots):
    """``rows`` x ``slots`` picks, distinct within a row, whose union is ``pool``."""
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
    return [row[:slots] for row in picks]


def step_routes(counts, gen):
    """Each request's rows over its own independent expert set."""
    rows = []
    for count in counts:
        pool = torch.randperm(E, generator=gen)[: EXPERTS_A_REQUEST[count]].tolist()
        rows += spread(pool, count, TOPK)
    ids = torch.tensor(rows, dtype=torch.int64, device=DEVICE)
    return ids, len(set(ids.flatten().tolist()))


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
    return statistics.median(samples)


def main():
    torch.cuda.set_device(0)
    flush = torch.empty(64 << 20, dtype=torch.uint8, device=DEVICE)
    results = {}
    with set_current_vllm_config(VllmConfig()):
        plan = MoeExpertsV41Plan(E, TOPK, HIDDEN, INTER, swiglu_limit=10.0)
        plan.declare_kernels()
        w13, w13_sf = experts(2 * INTER, HIDDEN, 1)
        w2, w2_sf = experts(HIDDEN, INTER, 2)
        prepared = plan.prepare_weights(w13.view(torch.int8), w13_sf, w2.view(torch.int8), w2_sf)
        del w13, w2
        torch.cuda.empty_cache()
        x = (torch.randn((112, HIDDEN), device=DEVICE) * 0.5).bfloat16()
        for streams in (1, 8, 16):
            for arm in ("dspark", "copy"):
                counts = ([17] if arm == "copy" else [6]) + [6] * (streams - 1)
                rows = sum(counts)
                gen = torch.Generator().manual_seed(1000 * streams + rows)
                specs = plan.workspace_specs(rows, torch.int64)
                tokens = cast_mxfp8(x[:rows])
                calls, distinct = [], []
                for _ in range(CALLS):
                    ids, n = step_routes(counts, gen)
                    distinct.append(n)
                    weights = torch.full((rows, TOPK), 1.0 / TOPK, device=DEVICE)
                    workspace = torch.empty(workspace_bytes(specs), dtype=torch.uint8, device=DEVICE)
                    out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=DEVICE)

                    def call(a=(ids, weights, workspace, out)):
                        plan.run(tokens, a[0], a[1], prepared, a[3], a[2])

                    calls.append(call)
                torch.cuda.synchronize()
                layer_us = timed(calls, flush)
                result = {
                    "case": "step", "streams": streams, "arm": arm, "rows": rows,
                    "experts_a_layer": round(statistics.fmean(distinct), 1),
                    "routed_us_a_layer": round(layer_us, 2),
                    "step_ms": round(layer_us * LAYERS / 1e3, 3),
                }
                results[(streams, arm)] = result
                print(json.dumps(result), flush=True)
                del calls
                torch.cuda.empty_cache()
    for streams in (1, 8, 16):
        a, b = results[(streams, "dspark")], results[(streams, "copy")]
        print(json.dumps({
            "case": "delta", "streams": streams,
            "rows": [a["rows"], b["rows"]],
            "experts_a_layer": [a["experts_a_layer"], b["experts_a_layer"]],
            "step_ms_added": round(b["step_ms"] - a["step_ms"], 3),
        }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
