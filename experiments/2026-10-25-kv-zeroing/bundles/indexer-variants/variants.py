#!/usr/bin/env python3
"""The indexer read bound's diagnosis: candidate 6's score kernel and three bounded
variants, compiled, bit-compared and timed against candidate 6's.

- v1: 0503 as committed (the row index clamped into every load's position and page);
- v2: v1 with a dense tile's page read once a tile;
- v3: candidate 6's loads, guarded by the extent (rows past it skip the loads);
- v4: candidate 6's loads for whole tiles, v1's clamped loads only in the tile the
  extent ends inside (a branch per tile, uniform across the CTA);
- v5: candidate 6's addresses, with only the key and scale loads predicated on the
  extent (rows past it score zeros).

v1-v4 were measured in earlier runs of this bundle; this run measures v5 and v4.

For each: the generated CUDA of the dense score kernel (between markers), bits against
candidate 6's in dense and candidate mode, and paired CUDA-graph timings (eight inputs,
alternating order) at decode shapes in both modes. Each compile prints its wall time
first, so the compile cache's cubins can be matched to kernels by mtime. Exits 0; this
run only measures.
"""
import importlib.util
import json
import math
import statistics
import time

import torch

PACKAGE = "vllm.models.deepseek_v4_1.tilelang"
DEVICE = "cuda"
CALLS, INPUTS, T_95 = 20, 8, 2.365
VARIANTS = ("v5", "v4")
TIMED = [(False, 6, 128, 524288, 4133), (False, 6, 128, 524288, 524261),
         (False, 96, 128, 524288, 524261), (False, 96, 256, 1048576, 1048549),
         (True, 6, 128, 524288, 524261), (True, 96, 256, 1048576, 1048549)]
BITS = [(6, 128, 524288, 4133), (96, 256, 1048576, 8229)]


def load(name, path):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}._variant_{name}", path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = PACKAGE
    spec.loader.exec_module(module)
    return module


def inputs(mod, rows, page_size, width, extent, seed, candidates=False):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    pages = -(-width // page_size)
    pool = torch.randint(0, 256, (pages, mod.page_bytes(page_size)), generator=gen,
                         device=DEVICE, dtype=torch.uint8)
    pool[:, page_size * 64:] = torch.randint(120, 130, (pages, page_size * 4), generator=gen,
                                             device=DEVICE, dtype=torch.uint8)
    table = torch.stack([torch.randperm(pages, generator=gen, device=DEVICE)
                         for _ in range(rows)]).to(torch.int32)
    heads, dim = mod.INDEX_HEADS, mod.INDEX_DIM
    q = torch.randint(0, 256, (rows, heads, dim // 2), generator=gen, device=DEVICE,
                      dtype=torch.uint8)
    qs = torch.randint(120, 130, (rows, heads, 4), generator=gen, device=DEVICE,
                       dtype=torch.uint8)
    w = (torch.randn((rows, heads), generator=gen, device=DEVICE) / 64).bfloat16()
    args = [q.view(torch.int8), qs.view(torch.uint32).reshape(rows, -1), w,
            pool.view(torch.int8), pool.view(torch.uint32), table,
            torch.full((rows,), extent, dtype=torch.int32, device=DEVICE),
            torch.tensor([width], dtype=torch.int32, device=DEVICE)]
    if candidates:
        cwidth = mod.CANDIDATE_BLOCKS * mod.CANDIDATE_BLOCK
        chosen = torch.full((rows, cwidth), -1, dtype=torch.int32, device=DEVICE)
        n = min(cwidth - 3, extent)
        for r in range(rows):
            chosen[r, :n] = torch.randperm(extent, generator=gen, device=DEVICE)[:n].sort().values.int()
            chosen[r, n // 3] = -1
        lengths = [max(1, n - 37 * r - 1) for r in range(rows)]
        args += [chosen, torch.tensor(lengths, dtype=torch.int32, device=DEVICE)]
    return args


def plan(mod, page_size, width, candidates=False):
    if candidates:
        width = mod.CANDIDATE_BLOCKS * mod.CANDIDATE_BLOCK
    print(json.dumps({"plan": mod.__name__.rsplit("_", 1)[-1], "page_size": page_size,
                      "candidates": candidates, "time": time.time()}), flush=True)
    return mod.IndexerV41Plan(width, page_size, candidates=candidates).score, width


def scores_of(score, args, rows, width):
    out = torch.zeros((rows, width), dtype=torch.bfloat16, device=DEVICE)
    score(*args, out)
    torch.accelerator.synchronize()
    return out


def timed(g):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / CALLS


def graph(score, args, out):
    score(*args, out)
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(CALLS):
            score(*args, out)
    return g


def main():
    modules = {"release": load("release", "/b/baseline/indexer.py")}
    for name in VARIANTS:
        modules[name] = load(name, f"/b/variants/{name}.py")
    for name, mod in modules.items():
        score, _ = plan(mod, 128, 524288)
        print(f"===== CUDA {name} (dense, 128-row pages)")
        print(score.get_kernel_source())
        print(f"===== END CUDA {name}", flush=True)
    release = modules["release"]
    for name in VARIANTS:
        mod = modules[name]
        for candidates in (False, True):
            for rows, page_size, width, extent in BITS:
                mine, w = plan(mod, page_size, width, candidates)
                theirs, _ = plan(release, page_size, width, candidates)
                args = inputs(release, rows, page_size, width, extent, 7, candidates)
                limit = extent if not candidates else int(args[-1].min())
                a = scores_of(mine, args, rows, w)[:, :limit]
                b = scores_of(theirs, args, rows, w)[:, :limit]
                same = torch.equal(a.view(torch.int16), b.view(torch.int16))
                print(json.dumps({"bits": name, "candidates": candidates, "rows": rows,
                                  "page_size": page_size, "extent": extent, "equal": same}),
                      flush=True)
        for candidates, rows, page_size, width, extent in TIMED:
            mine, w = plan(mod, page_size, width, candidates)
            theirs, _ = plan(release, page_size, width, candidates)
            diffs, times = [], {"variant": [], "release": []}
            for i in range(INPUTS):
                args = inputs(release, rows, page_size, width, extent, 100 * i + extent,
                              candidates)
                outs = {"variant": torch.zeros((rows, w), dtype=torch.bfloat16, device=DEVICE),
                        "release": torch.zeros((rows, w), dtype=torch.bfloat16, device=DEVICE)}
                graphs = {"variant": graph(mine, args, outs["variant"]),
                          "release": graph(theirs, args, outs["release"])}
                order = ("variant", "release") if i % 2 == 0 else ("release", "variant")
                measured = {k: timed(graphs[k]) for k in order}
                for k, v in measured.items():
                    times[k].append(v)
                diffs.append(measured["variant"] - measured["release"])
            mean = statistics.fmean(diffs)
            half = T_95 * statistics.stdev(diffs) / math.sqrt(INPUTS)
            print(json.dumps({"timing": name, "candidates": candidates, "rows": rows,
                              "page_size": page_size, "extent": extent,
                              "variant_us": statistics.median(times["variant"]),
                              "release_us": statistics.median(times["release"]),
                              "difference_us": mean, "ci95_us": half,
                              "slower": mean - half > 0}), flush=True)


if __name__ == "__main__":
    main()
