#!/usr/bin/env python3
"""Where a build spends the layer chain's time, kernel by kernel: the chain's CUDA graph
replayed under the torch profiler.

usage (in a build's tree, as pdl_chain_ab.py runs it):
    pdl_trace.py [--rows 16,48] [--layers 4] [--replays 30]

Each replay's kernels are grouped by the graph launch that ran them (the trace's
correlation id) and keyed by name and occurrence in the replay. For each key it prints
the median, over replays, of the kernel's start and end relative to the replay's first
kernel start, its duration, and its stream, as one JSON line shaped for pdl_chain_ab.py
("shape" is ``trace-<rows>``; ``warm_us`` is the replay's median span per layer). Under
PDL a kernel's start is its first CTA's, which may be before its predecessor's end; the
end times are what compare across builds.
"""

import argparse
import json
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from tests.kernels.deepseek_v41_pdl_chains import LayerChain
from vllm.config import VllmConfig, set_current_vllm_config


def graph(fn, layers: int) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(layers):
            fn()
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="16,48")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for rows in map(int, args.rows.split(",")):
        trace(rows, args)
        torch.cuda.empty_cache()


def trace(rows: int, args) -> None:
    with set_current_vllm_config(VllmConfig()):
        chain = LayerChain(rows, seed=args.seed)
        chain.run()
        torch.cuda.synchronize()
        eager = chain.digests()
        g = graph(chain.run, args.layers)
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.replays):
                g.replay()
                torch.cuda.synchronize()
        replayed = chain.digests()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.json"
        prof.export_chrome_trace(str(path))
        events = json.loads(path.read_text())["traceEvents"]
    replays = defaultdict(list)
    for e in events:
        if e.get("cat") == "kernel":
            replays[e["args"]["correlation"]].append(e)
    keyed = defaultdict(lambda: defaultdict(list))
    spans = []
    for kernels in replays.values():
        kernels.sort(key=lambda e: e["ts"])
        t0 = kernels[0]["ts"]
        spans.append(max(e["ts"] + e["dur"] for e in kernels) - t0)
        seen = defaultdict(int)
        for e in kernels:
            name = e["name"]
            key = (name, seen[name])
            seen[name] += 1
            keyed[key]["start"].append(e["ts"] - t0)
            keyed[key]["end"].append(e["ts"] + e["dur"] - t0)
            keyed[key]["dur"].append(e["dur"])
            keyed[key]["stream"].append(e["args"].get("stream"))
    kernels = sorted(
        (
            {
                "name": name[:80],
                "occurrence": n,
                "stream": statistics.mode(v["stream"]),
                "start_us": round(statistics.median(v["start"]), 2),
                "end_us": round(statistics.median(v["end"]), 2),
                "dur_us": round(statistics.median(v["dur"]), 2),
                "replays": len(v["end"]),
            }
            for (name, n), v in keyed.items()
        ),
        key=lambda k: k["end_us"],
    )
    print(json.dumps({
        "shape": f"trace-{rows}",
        "rows": rows,
        "layers": args.layers,
        "warm_us": round(statistics.median(spans) / args.layers, 2),
        "cold_us": round(statistics.median(spans) / args.layers, 2),
        "replays_traced": len(spans),
        "finite": bool(torch.isfinite(chain.out).all()),
        "replay_equals_eager": replayed == eager,
        "digests": replayed,
        "kernels": kernels,
    }), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
