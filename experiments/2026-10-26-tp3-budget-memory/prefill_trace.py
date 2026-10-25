#!/usr/bin/env python3
"""One prefill forward of a torch-profiler trace, split into its full-row and CED-compact
layers, with each kernel's time, launches and grids.

usage: prefill_trace.py TRACE.pt.trace.json.gz

The forward is the kernels from the first vocab_embedding_kernel to the next (the prompt's
step, then the first decode step). Layers are counted by the routed gate/up GEMM
(moe_grouped_fp8_fp4_gemm_kernel with the gate/up N tiles, grid x 9 at TP4's 1,152
columns, 12 at TP3's 1,536); the first layer whose gate/up grid is smaller than the first
layer's runs on the CED decoder's compacted rows. The L2 prefetch side stream is left out.
Run on request-inventory-5's 15,395-token prompt (TP4 rank 0, results/private/determinism/
request-inventory-5/ports/prefill-trace/dgx1), it reproduces the README's split.
"""
import collections
import gzip
import json
import sys


def main() -> None:
    with gzip.open(sys.argv[1], "rt") as handle:
        events = json.load(handle)["traceEvents"]
    kernels = sorted(
        (e["ts"], e["dur"], e["name"], tuple(e.get("args", {}).get("grid", ())))
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
        and e["name"] != "l2_prefetch_kernel"
    )
    embeds = [i for i, k in enumerate(kernels) if k[2] == "vocab_embedding_kernel"]
    forward = kernels[embeds[0]:embeds[1]]
    start, end = forward[0][0], max(k[0] + k[1] for k in forward)
    gate_up = [i for i, k in enumerate(forward) if k[2].startswith("moe_grouped") and k[3][0] < 20]
    full_tiles = forward[gate_up[0]][3][1]
    split = next(i for i in gate_up if forward[i][3][1] < full_tiles)
    busy, cur_s, cur_e = 0.0, forward[0][0], forward[0][0] + forward[0][1]
    for s, d, *_ in forward[1:]:
        if s > cur_e:
            busy, cur_s = busy + cur_e - cur_s, s
        cur_e = max(cur_e, s + d)
    busy += cur_e - cur_s
    print(f"forward {(end - start) / 1000:.1f} ms, GPU busy {busy / 1000:.1f} ms, {len(forward)} kernels; "
          f"full-row layers {sum(1 for i in gate_up if i < split)}, compact layers "
          f"{sum(1 for i in gate_up if i >= split)}")
    for label, part in (("full-row part", forward[:split]), ("compact part", forward[split:])):
        agg = collections.defaultdict(lambda: [0, 0.0, collections.Counter()])
        for _, dur, name, grid in part:
            entry = agg[name[:48]]
            entry[0] += 1
            entry[1] += dur
            entry[2][grid] += 1
        print(f"== {label}: {sum(k[1] for k in part) / 1000:.1f} ms in {len(part)} kernels")
        for name, (count, total, grids) in sorted(agg.items(), key=lambda x: -x[1][1])[:30]:
            shown = ", ".join(f"{g}x{n}" for g, n in grids.most_common(2))
            print(f"  {total / 1000:8.2f} ms {count:4d}x  {name:48s} {shown[:70]}")


if __name__ == "__main__":
    main()
