# SPDX-License-Identifier: Apache-2.0
"""Timing of DeepSeek-V4.1's dense indexer (score kernel and exact top-512) at cold-prefill
shapes: one 256-row index sub-chunk (``INDEX_CHUNK``) of one request on a ratio-2 layer's
cache (128-row pages, a 1M-token context's 524,288-position planned width), at the depths a
122K / 237K / 470K / 910K-token prompt reaches.

All rows share one block-table row, as the rows of one request do in prefill, and each row's
extent follows its causal position at ratio 2. Keys and queries are random MXFP4 bytes.

With ``--baseline PATH`` (another ``tilelang/indexer.py``), both modules score and select
the same inputs; the scores over every row's extent and the selected positions must match
bit for bit, then both are timed with CUDA graphs, alternating which goes first. Without it
only the in-tree module is timed.

Per-token cost: a 16,288-row chunk gives each of the 4 ranks about 4,072 rows, 16 sub-chunks
per dense layer, and 3 dense layers (2, 8, 14) run per chunk, so a sub-chunk's time times
3 * 16 / 16,288 is its share of a prompt token's time at that depth.
"""

import argparse
import importlib.util
import json
import statistics

import torch

from vllm.models.deepseek_v4_1.tilelang import indexer

DEVICE = "cuda"
ROWS, PAGE, WIDTH = 256, 128, 524288
CONTEXTS = [122_000, 237_000, 470_000, 910_000]
PER_TOKEN = 3 * 16 / 16288
INPUTS = 4


def module(path):
    package = "vllm.models.deepseek_v4_1.tilelang"
    spec = importlib.util.spec_from_file_location(f"{package}._baseline_indexer", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package
    spec.loader.exec_module(mod)
    return mod


def inputs(context, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    extent = context // 2
    pages = WIDTH // PAGE
    nbytes = indexer.page_bytes(PAGE)
    pool = torch.randint(0, 256, (pages, nbytes), generator=gen, device=DEVICE,
                         dtype=torch.uint8)
    pool[:, PAGE * 64:] = torch.randint(120, 130, (pages, PAGE * 4), generator=gen,
                                        device=DEVICE, dtype=torch.uint8)
    table = torch.randperm(pages, generator=gen, device=DEVICE).to(torch.int32)
    table = table.unsqueeze(0).expand(ROWS, pages).contiguous()
    heads, dim = indexer.INDEX_HEADS, indexer.INDEX_DIM
    q = torch.randint(0, 256, (ROWS, heads, dim // 2), generator=gen, device=DEVICE,
                      dtype=torch.uint8)
    q_scales = torch.randint(120, 130, (ROWS, heads, 4), generator=gen, device=DEVICE,
                             dtype=torch.uint8)
    weights = (torch.randn((ROWS, heads), generator=gen, device=DEVICE) / 64).bfloat16()
    lengths = torch.tensor([extent - (ROWS - 1 - r) // 2 for r in range(ROWS)],
                           dtype=torch.int32, device=DEVICE)
    active = torch.tensor([WIDTH], dtype=torch.int32, device=DEVICE)
    return dict(q=q.view(torch.int8), qs=q_scales.view(torch.uint32).reshape(ROWS, -1),
                w=weights, pool=pool, table=table, lengths=lengths, active=active)


class Runner:
    """One module's plan with its own scores, outputs and scratch for one input."""

    def __init__(self, mod, x):
        self.plan = mod.IndexerV41Plan(WIDTH, PAGE)
        nbytes = mod.page_bytes(PAGE)
        self.args = (x["q"], x["qs"], x["w"], x["pool"].view(torch.int8)[:, :nbytes],
                     x["pool"].view(torch.uint32)[:, :nbytes // 4],
                     x["table"], x["lengths"], x["active"])
        self.x = x
        specs = self.plan.scratch_specs(ROWS)
        bufs = [torch.zeros(s, dtype=d, device=DEVICE) for s, d in specs]
        self.scores, self.values = bufs[0], bufs[1]
        self.select_scratch = bufs[2:2 + len(self.plan.select.scratch_specs(ROWS))]
        self.out = torch.full((ROWS, self.plan.topk), -2, dtype=torch.int32, device=DEVICE)

    def score(self):
        self.plan.score(*self.args, self.scores)

    def select(self):
        self.plan.select.run(self.scores, self.x["lengths"], self.x["active"], self.out,
                             self.values, scratch=self.select_scratch)

    def both(self):
        self.score()
        self.select()


def graph(fn, calls):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    return g


def timed(g, calls):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay()  # warm
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / calls  # us per call


def compare(a, b, lengths):
    torch.cuda.synchronize()
    for r in range(ROWS):
        e = int(lengths[r])
        if not torch.equal(a.scores[r, :e].view(torch.int16), b.scores[r, :e].view(torch.int16)):
            return f"scores differ at row {r}"
    if not torch.equal(a.out, b.out):
        return "selected positions differ"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", help="another tilelang/indexer.py to compare against")
    parser.add_argument("--contexts", default=",".join(map(str, CONTEXTS)))
    args = parser.parse_args()
    mods = {"in-tree": indexer}
    if args.baseline:
        mods["baseline"] = module(args.baseline)
    failed = False
    for context in map(int, args.contexts.split(",")):
        extent = context // 2
        calls = max(2, min(20, 2_000_000 // extent))
        times = {name: {"score": [], "select": [], "both": []} for name in mods}
        mismatch = None
        for i in range(INPUTS):
            x = inputs(context, seed=7919 * i + context)
            runners = {name: Runner(m, x) for name, m in mods.items()}
            for r in runners.values():
                r.both()
            if args.baseline:
                mismatch = mismatch or compare(runners["in-tree"], runners["baseline"],
                                               x["lengths"])
            graphs = {(n, part): graph(getattr(r, part), calls)
                      for n, r in runners.items() for part in ("score", "select", "both")}
            names = list(runners) if i % 2 == 0 else list(reversed(runners))
            for part in ("score", "select", "both"):
                for n in names:
                    times[n][part].append(timed(graphs[(n, part)], calls))
            del graphs, runners
            torch.cuda.empty_cache()
        flops = ROWS * extent * indexer.INDEX_HEADS * indexer.INDEX_DIM * 2
        for n in mods:
            med = {p: statistics.median(v) for p, v in times[n].items()}
            low = {p: min(v) for p, v in times[n].items()}
            result = {
                "module": n, "context": context, "keys": extent, "rows": ROWS,
                "score_us": round(med["score"], 1), "select_us": round(med["select"], 1),
                "both_us": round(med["both"], 1),
                "score_min_us": round(low["score"], 1), "select_min_us": round(low["select"], 1),
                "score_tflops": round(flops / (med["score"] * 1e-6) / 1e12, 1),
                "score_tflops_best": round(flops / (low["score"] * 1e-6) / 1e12, 1),
                "score_us_per_token": round(med["score"] * PER_TOKEN, 3),
                "select_us_per_token": round(med["select"] * PER_TOKEN, 3),
                "both_us_per_token": round(med["both"] * PER_TOKEN, 3),
                "samples": {p: [round(t, 1) for t in v] for p, v in times[n].items()},
            }
            print(json.dumps(result), flush=True)
        if args.baseline:
            print(json.dumps({"context": context, "bit_identical": mismatch is None,
                              "mismatch": mismatch}), flush=True)
            failed = failed or mismatch is not None
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
