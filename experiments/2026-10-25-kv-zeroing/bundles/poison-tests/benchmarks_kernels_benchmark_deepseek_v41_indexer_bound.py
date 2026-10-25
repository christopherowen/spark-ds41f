# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired timing of DeepSeek-V4.1's indexer score kernel with its last tile bounded to
the visible positions against the release's (``--baseline``, the release's
``tilelang/indexer.py``), which loads that tile's rows past them.

The bound predicates a dense row's key and scale loads past its extent off; addresses,
the page read and the load count are the release's. Candidate mode is the release's
kernel (its lists are -1 padded, so rows past a list load position 0), timed to show
it. Cases are decode steps (6 rows: one stream's 1 + 5 drafts; 96: sixteen streams')
at extents ending inside a tile, on the ratio-2 caches' 128-row pages and layer 20's
256-row pages, up to a 1M-token context's 524,288 and 1,048,576 compressed positions.
Candidate cases score a reindex layer's candidate list (``CANDIDATE_BLOCKS *
CANDIDATE_BLOCK`` sorted positions with -1 padding, one masked) at the deepest
extents.

For each case both kernels first score the same inputs, compared bit for bit over the
visible positions. CUDA graphs of 20 launches of each then run back to back on eight
fresh inputs, alternating which goes first; the case's 95% interval comes from the
eight per-input differences. The gate (``--gate``) is that the bounded kernel is
slower at no case: no interval lies wholly above zero.
"""

import argparse
import importlib.util
import json
import math
import statistics

import torch

from vllm.models.deepseek_v4_1.tilelang import indexer

DEVICE = "cuda"
CALLS, INPUTS = 20, 8
ROWS = {"c1": 6, "c16": 96}
# (page size, planned width, visible positions): a 1M context's compressed positions
# at ratio 2 (128-row pages) and 1 (256-row pages), at several depths.
EXTENTS = [
    (128, 524288, 4133),
    (128, 524288, 65573),
    (128, 524288, 524261),
    (256, 1048576, 8229),
    (256, 1048576, 1048549),
]
CANDIDATE_EXTENTS = [(128, 524288, 524261), (256, 1048576, 1048549)]
T_95 = 2.365  # Student's t, 7 degrees of freedom


def baseline_module(path: str):
    package = "vllm.models.deepseek_v4_1.tilelang"
    spec = importlib.util.spec_from_file_location(f"{package}._release_indexer", path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    spec.loader.exec_module(module)
    return module


def inputs(rows: int, page_size: int, width: int, extent: int, seed: int) -> list:
    """Random MXFP4 bytes; every row reads the same pages in its own order."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    pages = -(-width // page_size)
    nbytes = indexer.page_bytes(page_size)
    pool = torch.randint(
        0, 256, (pages, nbytes), generator=gen, device=DEVICE, dtype=torch.uint8
    )
    pool[:, page_size * 64 :] = torch.randint(
        120, 130, (pages, page_size * 4), generator=gen, device=DEVICE,
        dtype=torch.uint8,
    )  # fmt: skip
    table = torch.stack(
        [torch.randperm(pages, generator=gen, device=DEVICE) for _ in range(rows)]
    ).to(torch.int32)
    heads, dim = indexer.INDEX_HEADS, indexer.INDEX_DIM
    q = torch.randint(
        0, 256, (rows, heads, dim // 2), generator=gen, device=DEVICE, dtype=torch.uint8
    )
    q_scales = torch.randint(
        120, 130, (rows, heads, 4), generator=gen, device=DEVICE, dtype=torch.uint8
    )
    weights = (torch.randn((rows, heads), generator=gen, device=DEVICE) / 64).bfloat16()
    lengths = torch.full((rows,), extent, dtype=torch.int32, device=DEVICE)
    active = torch.tensor([width], dtype=torch.int32, device=DEVICE)
    return [
        q.view(torch.int8),
        q_scales.view(torch.uint32).reshape(rows, -1),
        weights,
        pool.view(torch.int8),
        pool.view(torch.uint32),
        table,
        lengths,
        active,
    ]


def candidate_inputs(
    rows: int, page_size: int, width: int, extent: int, seed: int
) -> list:
    """``inputs`` plus each row's candidate list: sorted visible positions, one of them
    masked (-1), -1 padded, with lengths ending inside a tile."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    span = indexer.CANDIDATE_BLOCKS * indexer.CANDIDATE_BLOCK
    count = span - 3
    chosen = torch.full((rows, span), -1, dtype=torch.int32, device=DEVICE)
    for r in range(rows):
        picked = torch.randperm(extent, generator=gen, device=DEVICE)[:count]
        chosen[r, :count] = picked.sort().values.int()
        chosen[r, count // 3] = -1
    lengths = torch.tensor(
        [count - 37 * r for r in range(rows)], dtype=torch.int32, device=DEVICE
    )
    return inputs(rows, page_size, width, extent, seed) + [chosen, lengths]


def graph(score, args, scores) -> torch.cuda.CUDAGraph:
    score(*args, scores)  # warm
    torch.accelerator.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(CALLS):
            score(*args, scores)
    return g


def timed(g: torch.cuda.CUDAGraph) -> float:
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / CALLS  # us a launch


def case(
    release, rows: int, page_size: int, width: int, extent: int, candidates: bool
) -> dict:
    # Candidate plans score a list as wide as the candidate blocks; both kernels leave
    # the same zeros past each row's length, so whole rows compare.
    span = indexer.CANDIDATE_BLOCKS * indexer.CANDIDATE_BLOCK if candidates else width
    compared = span if candidates else extent
    bounded = indexer.IndexerV41Plan(span, page_size, candidates=candidates).score
    unbounded = release.IndexerV41Plan(span, page_size, candidates=candidates).score
    make = candidate_inputs if candidates else inputs
    differences, times = [], {"bounded": [], "release": []}
    for i in range(INPUTS):
        args = make(rows, page_size, width, extent, seed=1000 * i + extent)
        mine = torch.zeros((rows, span), dtype=torch.bfloat16, device=DEVICE)
        theirs = torch.zeros_like(mine)
        bounded(*args, mine)
        unbounded(*args, theirs)
        torch.accelerator.synchronize()
        same = torch.equal(
            mine[:, :compared].view(torch.int16), theirs[:, :compared].view(torch.int16)
        )
        assert same, (rows, extent, candidates)
        graphs = {"bounded": graph(bounded, args, mine),
                  "release": graph(unbounded, args, theirs)}  # fmt: skip
        order = ("bounded", "release") if i % 2 == 0 else ("release", "bounded")
        measured = {name: timed(graphs[name]) for name in order}
        for name, value in measured.items():
            times[name].append(value)
        differences.append(measured["bounded"] - measured["release"])
    mean = statistics.fmean(differences)
    half = T_95 * statistics.stdev(differences) / math.sqrt(INPUTS)
    return {
        "rows": rows, "candidates": candidates, "page_size": page_size,
        "extent": extent,
        "bounded_us": statistics.median(times["bounded"]),
        "release_us": statistics.median(times["release"]),
        "difference_us": mean, "ci95_us": half,
        "slower": mean - half > 0,
    }  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, help="the release's indexer.py")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    release = baseline_module(args.baseline)
    results = []
    cases = [(False, *e) for e in EXTENTS] + [(True, *e) for e in CANDIDATE_EXTENTS]
    for name, rows in ROWS.items():
        for candidates, page_size, width, extent in cases:
            result = {
                "case": name,
                **case(release, rows, page_size, width, extent, candidates),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    slower = [r for r in results if r["slower"]]
    print(json.dumps({"summary": "indexer read bound", "cases": len(results),
                      "slower": len(slower)}), flush=True)  # fmt: skip
    return 1 if args.gate and slower else 0


if __name__ == "__main__":
    raise SystemExit(main())
