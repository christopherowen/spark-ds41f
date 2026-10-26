#!/usr/bin/env python3
"""Where the shared expert's down GEMM runs in each build's layer chain, from a
pdl-trace receipt (pdl_chain_ab.py running pdl_trace.py).

usage: trace_compare.py RECEIPT

Per row count and build, over every run and the layers after the first: the routed
down GEMM's end, the shared down GEMM's end relative to it (negative: it finished
first, beside the routed experts), the combine's end, and the layer's time (combine to
combine). In a replay the chain's kernels are, per layer, five block-32 GEMMs
(``gemm_mxfp8``: fused Q-A/KV, Q-B, WO-B, the shared gate/up, the shared down), two
grouped GEMMs (the routed gate/up and down) and one combine, so the shared down GEMM of
layer L is ``gemm_mxfp8`` occurrence 5L + 4, the routed down GEMM ``moe_grouped`` 2L + 1.
"""

import json
import statistics
import sys
from collections import defaultdict


def runs(path: str) -> dict:
    out = defaultdict(list)
    build = None
    for line in open(path):
        if line.startswith("=== round"):
            build = line.split(":")[1].strip()
        elif line.startswith('{"shape"') and '"kernels"' in line:
            d = json.loads(line)
            out[(d["rows"], build)].append(d)
    return out


def layer_points(d: dict) -> list[dict]:
    end = {(k["name"], k["occurrence"]): k["end_us"] for k in d["kernels"]}

    def at(prefix: str, n: int) -> float:
        (value,) = [
            v for (name, i), v in end.items() if name.startswith(prefix) and i == n
        ]
        return value

    points = []
    for layer in range(1, d["layers"]):
        routed = at("moe_grouped_fp8_fp4_gemm", 2 * layer + 1)
        points.append({
            "routed_down_end": routed,
            "shared_down_after_routed": at("gemm_mxfp8", 5 * layer + 4) - routed,
            "layer_us": at("moe_combine", layer) - at("moe_combine", layer - 1),
        })  # fmt: skip
    return points


def main() -> None:
    data = runs(sys.argv[1])
    for rows in sorted({r for r, _ in data}):
        builds = [b for r, b in data if r == rows]
        print(f"rows {rows}")
        for build in sorted(builds, key=lambda b: (b != "base", b != "pdl", b)):
            points = [p for d in data[(rows, build)] for p in layer_points(d)]
            after = [p["shared_down_after_routed"] for p in points]
            print(f"  {build:12} layer {statistics.median(p['layer_us'] for p in points):8.1f} us   "
                  f"shared down ends {statistics.median(after):+8.1f} us from the routed one "
                  f"(after it in {sum(a > 0 for a in after)}/{len(after)} layers)")  # fmt: skip


if __name__ == "__main__":
    main()
