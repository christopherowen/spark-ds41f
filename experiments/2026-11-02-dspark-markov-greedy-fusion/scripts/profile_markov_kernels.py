#!/usr/bin/env python3
"""Per-kernel GPU time of the five-position reduction loop (``bench_markov_greedy``'s
``reduce`` graph, replayed back to back, L2 hot), from a CUPTI trace of the graph
replays: each arm's kernels by name, microseconds per launch, and the loop's span
from its first kernel's start to its last kernel's end."""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_markov_greedy import (  # noqa: E402
    ARMS,
    N_SPEC,
    SHAPES,
    DraftSteps,
    capture,
    load_bf16,
    packed_head,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("/bench/weights"))
    parser.add_argument("--rows", default="1,4,16,48")
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows_list = [int(r) for r in args.rows.split(",")]
    w2 = load_bf16(args.weights / "markov_w2")
    table = load_bf16(args.weights / "markov_w1").cuda()
    results = []
    for shape, (n, ranks) in SHAPES.items():
        head, _ = packed_head(w2[:n], torch.float32, max(rows_list))
        for rows in rows_list:
            steps = DraftSteps(head, table, rows, n, ranks, torch.float32, False, rows)
            for arm in ARMS:
                graph = capture(lambda a=arm: steps.reduce_loop(a))
                for _ in range(20):
                    graph.replay()
                torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(args.replays):
                        graph.replay()
                    torch.cuda.synchronize()
                kernels = sorted(
                    (
                        e
                        for e in prof.events()
                        if e.device_type == torch.autograd.DeviceType.CUDA
                    ),
                    key=lambda e: e.time_range.start,
                )
                per_loop = len(kernels) // args.replays
                durations = defaultdict(list)
                spans = []
                for r in range(args.replays):
                    loop = kernels[r * per_loop : (r + 1) * per_loop]
                    spans.append(loop[-1].time_range.end - loop[0].time_range.start)
                    for e in loop:
                        durations[e.name].append(e.time_range.end - e.time_range.start)
                busy = statistics.median(
                    sum(
                        e.time_range.end - e.time_range.start
                        for e in kernels[r * per_loop : (r + 1) * per_loop]
                    )
                    for r in range(args.replays)
                )
                record = {
                    "shape": shape,
                    "rows": rows,
                    "arm": arm,
                    "kernels_per_loop": per_loop,
                    "loop_span_us": statistics.median(spans),
                    "loop_kernel_busy_us": busy,
                    "kernels_us": {
                        name: {
                            "per_launch_median": statistics.median(d),
                            "launches_per_loop": len(d) // args.replays,
                        }
                        for name, d in durations.items()
                    },
                }
                results.append(record)
                print(
                    f"{shape} rows={rows:2d} {arm:12s} span={record['loop_span_us']:6.1f} "
                    f"busy={busy:6.1f} "
                    + " ".join(
                        f"{name[:40]}={v['per_launch_median']:.1f}x{v['launches_per_loop']}"
                        for name, v in record["kernels_us"].items()
                    ),
                    flush=True,
                )
                del graph
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_spec": N_SPEC, "cases": results}, indent=1) + "\n")


if __name__ == "__main__":
    main()
