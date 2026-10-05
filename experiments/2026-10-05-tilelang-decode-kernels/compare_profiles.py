#!/usr/bin/env python3
"""Per-module decode step time from rank-0 torch profiles, B12X and TileLang side by side.

usage: compare_profiles.py LABEL=TRACE.json.gz ...

Takes the main stream of each trace, splits it into target steps at the shared
SWA cache writer (40 per step), keeps steady six-row steps and reports the
median per module: kernel time per step, calls per step and time per call.
TileLang kernels are all named ``main_kernel``; they are identified by grid and
block, which this experiment's tile choices fix.
"""
import gzip
import json
import statistics
import sys
from collections import Counter, defaultdict


def module(name, grid, block):
    g, b = tuple(grid), tuple(block)
    if "ConcatAndCacheGlmNextMla" in name:
        return "cache writers"
    if "RoceOneshot" in name or "RoceAllGather" in name:
        return "collectives"
    if "moe_sharedkernels" in name or "W4A16TopKSum" in name:
        return "routed experts"
    if name.startswith("swiglu_forward"):
        return "routed experts"
    if "mhc" in name.lower() or "GroupedRmsNorm" in name and False:
        return "mHC"
    if "_dsv4_topk" in name or ("SmallNGemv" in name and g == (384, 2, 1)):
        return "router"
    if "SmallNGemv" in name:
        return "BF16 GEMV (B12X, both)"
    if "DenseGemm" in name:
        return {(1, 1, 48): "Q-B", (1, 1, 14): "fused Q-A/KV", (1, 4, 100): "draft main projection",
                (1, 1, 32): "WO and indexer Q-B (B12X)", (1, 1, 40): "WO (B12X)"}.get(g, "dense other (B12X)")
    if "sharedmlakernelUnified" in name or "SparseMLASplitDecode" in name or "map_indexed_pages" in name:
        return "sparse MLA"
    if "dsa_indexer" in name:
        return "indexer"
    if "sparse_mlarota" in name:
        return "rotary (B12X, both)"
    if "MXFP8RowsQuant" in name or "per_token_cast" in name or "norm_forward" in name or "GroupedRmsNorm" in name:
        return "quantization and norms"
    if name.startswith(("main_kernel", "main_kernel_1")):
        tl = {
            ((9, 36, 1), (128, 1, 1)): "routed experts", ((40, 36, 1), (128, 1, 1)): "routed experts",
            ((1, 1, 1), (512, 1, 1)): "routed experts", ((5, 6, 1), (128, 1, 1)): "routed experts",
            ((40, 1, 1), (128, 1, 1)): "mHC", ((6, 1, 1), (128, 1, 1)): "mHC",
            ((128, 1, 1), (256, 1, 1)): "Q-B", ((64, 1, 1), (256, 1, 1)): "indexer Q-B",
            ((32, 1, 1), (256, 1, 1)): "indexer Q-B", ((28, 1, 1), (256, 1, 1)): "fused Q-A/KV",
            ((14, 2, 1), (256, 1, 1)): "fused Q-A/KV", ((14, 6, 1), (128, 1, 1)): "fused Q-A/KV",
            ((100, 1, 1), (256, 1, 1)): "draft main projection",
            ((5, 1, 6), (128, 1, 1)): "sparse MLA", ((16, 6, 1), (128, 1, 1)): "sparse MLA",
            ((1, 1, 6), (128, 1, 1)): "sparse MLA",
            ((6, 8, 1), (256, 1, 1)): "router", ((3, 1, 1), (64, 1, 1)): "router",
        }.get((g, b))
        if tl:
            return tl
        if len(g) == 3 and g[1] == 6 and g[0] in (128, 8, 1, 4, 256, 16, 32, 2) or g in ((1, 40, 1), (2, 4, 1)):
            return "indexer"
        return f"TileLang other {g} {b}"
    if name.startswith(("_quantize_attention", "_pages", "_chunk")):
        return "Triton (both)"
    return "other"


def steps(path):
    events = json.load(gzip.open(path, "rt"))["traceEvents"]
    k = sorted((e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"), key=lambda e: e["ts"])
    main = Counter(e["args"].get("stream") for e in k).most_common(1)[0][0]
    seq = [e for e in k if e["args"]["stream"] == main]
    marks = [i for i, e in enumerate(seq) if "ConcatAndCacheGlmNextMla" in e["name"]]
    out = []
    for i in range(0, len(marks) - 40, 40):
        a, b = marks[i], marks[i + 40]
        if seq[a]["args"]["grid"][0] != 6:
            continue
        t, c = defaultdict(float), Counter()
        for e in seq[a:b]:
            m = module(e["name"], e["args"]["grid"], e["args"]["block"])
            t[m] += e["dur"] / 1000
            c[m] += 1
        wall = (seq[b - 1]["ts"] + seq[b - 1]["dur"] - seq[a]["ts"]) / 1000
        out.append((wall, t, c))
    return out


def main():
    arms = {}
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        s = steps(path)
        mods = set().union(*(x[1] for x in s))
        arms[label] = {
            "steps": len(s), "wall": statistics.median(x[0] for x in s),
            "busy": statistics.median(sum(x[1].values()) for x in s),
            "modules": {m: (statistics.median(x[1].get(m, 0) for x in s), statistics.median(x[2].get(m, 0) for x in s))
                        for m in mods},
        }
    labels = list(arms)
    mods = sorted(set().union(*(a["modules"] for a in arms.values())),
                  key=lambda m: -max(a["modules"].get(m, (0, 0))[0] for a in arms.values()))
    print(f"{'ms per six-row step':34}" + "".join(f"{lab:>22}" for lab in labels))
    print(f"{'steps / wall / busy':34}" + "".join(
        f"{arms[lab]['steps']:>5} {arms[lab]['wall']:7.2f} {arms[lab]['busy']:7.2f} " for lab in labels))
    for m in mods:
        cells = []
        for lab in labels:
            ms, n = arms[lab]["modules"].get(m, (0, 0))
            cells.append(f"{ms:7.3f} ({n:3.0f}x {1000 * ms / n if n else 0:6.1f}us)" if n else f"{'-':>22}")
        print(f"{m[:34]:34}" + "".join(f"{c:>22}" for c in cells))
    print(json.dumps({lab: {"wall": a["wall"], "busy": a["busy"], "modules": a["modules"]} for lab, a in arms.items()}))


if __name__ == "__main__":
    main()
