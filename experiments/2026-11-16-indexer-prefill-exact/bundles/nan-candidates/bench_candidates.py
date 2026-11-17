# SPDX-License-Identifier: Apache-2.0
"""Timing of the reindex (candidate) score kernel at decode shapes: 6 rows (one stream and five
drafts) and 96 (sixteen streams), 2,048 candidate blocks of 8 over a 1M-token context's ratio-1
cache (256-row pages), in-tree module against ``--baseline``, scores compared bit for bit."""
import argparse
import importlib.util
import json
import statistics

import torch

from vllm.models.deepseek_v4_1.tilelang import indexer

DEVICE, PAGE, CONTEXT, CALLS, INPUTS = "cuda", 256, 1_048_549, 50, 6
SPAN = indexer.CANDIDATE_BLOCKS * indexer.CANDIDATE_BLOCK


def module(path):
    package = "vllm.models.deepseek_v4_1.tilelang"
    spec = importlib.util.spec_from_file_location(f"{package}._baseline_indexer", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package
    spec.loader.exec_module(mod)
    return mod


def inputs(rows, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    pages = -(-CONTEXT // PAGE)
    nbytes = indexer.page_bytes(PAGE)
    pool = torch.randint(0, 256, (pages, nbytes), generator=gen, device=DEVICE, dtype=torch.uint8)
    pool[:, PAGE * 64:] = torch.randint(120, 130, (pages, PAGE * 4), generator=gen, device=DEVICE,
                                        dtype=torch.uint8)
    table = torch.randperm(pages, generator=gen, device=DEVICE).to(torch.int32)
    table = table.unsqueeze(0).expand(rows, pages).contiguous()
    q = torch.randint(0, 256, (rows, 32, 64), generator=gen, device=DEVICE, dtype=torch.uint8)
    qs = torch.randint(120, 130, (rows, 32, 4), generator=gen, device=DEVICE, dtype=torch.uint8)
    w = (torch.randn((rows, 32), generator=gen, device=DEVICE) / 64).bfloat16()
    blocks = CONTEXT // 8
    cand = torch.stack([torch.randperm(blocks, generator=gen, device=DEVICE)[:indexer.CANDIDATE_BLOCKS]
                        .sort().values for _ in range(rows)]).to(torch.int32)
    lengths = torch.full((rows,), SPAN - 5, dtype=torch.int32, device=DEVICE)
    return [q.view(torch.int8), qs.view(torch.uint32).reshape(rows, -1), w,
            pool.view(torch.int8)[:, :nbytes], pool.view(torch.uint32)[:, :nbytes // 4], table,
            torch.full((rows,), CONTEXT, dtype=torch.int32, device=DEVICE),
            torch.tensor([CONTEXT], dtype=torch.int32, device=DEVICE), cand, lengths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    mods = {"in-tree": indexer, "baseline": module(args.baseline)}
    failed = False
    for rows in (6, 96):
        times = {n: [] for n in mods}
        same = True
        for i in range(INPUTS):
            x = inputs(rows, 31 * i + rows)
            out = {}
            graphs = {}
            for n, m in mods.items():
                score = m.IndexerV41Plan(SPAN, PAGE, candidates=True).score
                s = torch.zeros((rows, SPAN), dtype=torch.bfloat16, device=DEVICE)
                score(*x, s)
                torch.cuda.synchronize()
                out[n] = s
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(CALLS):
                        score(*x, s)
                graphs[n] = g
            same = same and torch.equal(out["in-tree"][:, :SPAN - 5].view(torch.int16),
                                        out["baseline"][:, :SPAN - 5].view(torch.int16))
            order = list(mods) if i % 2 == 0 else list(reversed(mods))
            for n in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                graphs[n].replay()
                start.record()
                graphs[n].replay()
                end.record()
                end.synchronize()
                times[n].append(start.elapsed_time(end) * 1e3 / CALLS)
        diffs = [a - b for a, b in zip(times["in-tree"], times["baseline"])]
        print(json.dumps({"rows": rows, "in_tree_us": round(statistics.median(times["in-tree"]), 2),
                          "baseline_us": round(statistics.median(times["baseline"]), 2),
                          "mean_difference_us": round(statistics.fmean(diffs), 2),
                          "bit_identical": same}), flush=True)
        failed = failed or not same
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
