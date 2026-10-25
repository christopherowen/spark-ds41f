# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's candidate source and reindex indexers with candidates as block ids
against the release's (positions; its ``indexer.py`` given as ``--baseline``), at the
serving shapes: the candidate source scores a 1M-position row (index pages of 256) and
keeps 2,048 blocks of 8, the reindex layer scores those 16,384 candidates.

Each seed draws random MXFP4 keys and queries and row lengths from empty to 300,000
positions (most with a partly visible last block). Both chains run on the same inputs:
- the source's top-512 positions, scores and candidate counts must be equal;
- the release's candidate positions must equal the block ids expanded
  (``ids[j // 8] * 8 + j % 8`` below the count, ``-1`` past it);
- the reindex layer's scores (below the count) and top-512 positions must be equal.
Then CUDA graphs of each chain's source block selection and reindex (score and top-k)
run back to back, alternating which goes first, eight times; the 95% interval of the
differences gates (``--gate``): equal everywhere, and neither step slower.
"""

import argparse
import importlib.util
import json
import math
import statistics

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4_1.tilelang import indexer

PAGE, HEADS, MAX_LEN, CONTEXT, ROWS, SEEDS, SAMPLES, CALLS = (
    256,
    32,
    1048576,
    300000,
    32,
    3,
    8,
    5,
)
BLOCKS, BLOCK = indexer.CANDIDATE_BLOCKS, indexer.CANDIDATE_BLOCK


def baseline_module(path: str, name: str):
    package = "vllm.models.deepseek_v4_1.tilelang"
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    spec.loader.exec_module(module)
    return module


def inputs(seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = -(-CONTEXT // PAGE)
    pool = torch.randint(
        0, 256, (pages * ROWS, indexer.page_bytes(PAGE)), generator=g, device="cuda",
        dtype=torch.uint8,
    )  # fmt: skip
    pool[:, PAGE * 64 :] = torch.randint(
        118, 126, (pages * ROWS, PAGE * 4), generator=g, device="cuda", dtype=torch.uint8
    )  # UE8M0 scales of finite keys
    table = torch.full((ROWS, MAX_LEN // PAGE), -1, dtype=torch.int32, device="cuda")
    for r in range(ROWS):
        table[r, :pages] = torch.randperm(pages * ROWS, generator=g, device="cuda")[
            :pages
        ].to(torch.int32)
    lengths = torch.randint(1, CONTEXT + 1, (ROWS,), generator=g, device="cuda")
    lengths[0], lengths[1], lengths[2] = 0, 1, CONTEXT - 3
    q = torch.randint(0, 256, (ROWS, HEADS, 64), generator=g, device="cuda",
                      dtype=torch.uint8)  # fmt: skip
    q_scales = torch.randint(118, 126, (ROWS, HEADS, 4), generator=g, device="cuda",
                             dtype=torch.uint8)  # fmt: skip
    weights = (torch.randn((ROWS, HEADS), generator=g, device="cuda") / 64).bfloat16()
    active = torch.tensor([MAX_LEN], dtype=torch.int32, device="cuda")
    return (q, q_scales, weights, pool, table, lengths.to(torch.int32), active)


class Chain:
    """A module's source and reindex plans with their outputs."""

    def __init__(self, module, args, blocks_as_ids: bool):
        self.args = args
        self.source = module.IndexerV41Plan(MAX_LEN, PAGE, candidate_blocks=True)
        self.reindex = module.IndexerV41Plan(BLOCKS * BLOCK, PAGE, candidates=True)
        width = BLOCKS if blocks_as_ids else BLOCKS * BLOCK
        self.candidates = torch.full((ROWS, width), -7, dtype=torch.int32, device="cuda")
        self.counts = torch.full((ROWS,), -7, dtype=torch.int32, device="cuda")
        self.top = torch.empty((ROWS, indexer.INDEX_TOPK), dtype=torch.int32,
                               device="cuda")  # fmt: skip
        self.re_top = torch.empty_like(self.top)
        self.scratch = [torch.empty(s, dtype=d, device="cuda")
                        for s, d in self.source.scratch_specs(ROWS)]  # fmt: skip
        self.re_scratch = [torch.empty(s, dtype=d, device="cuda")
                           for s, d in self.reindex.scratch_specs(ROWS)]  # fmt: skip

    def run_source(self):
        return self.source.run(
            *self.args, out=self.top, candidate_out=self.candidates,
            candidate_out_lengths=self.counts, scratch=self.scratch,
        )[1]  # fmt: skip

    def run_reindex(self):
        return self.reindex.run(
            *self.args, self.candidates, self.counts, out=self.re_top,
            scratch=self.re_scratch,
        )[1]  # fmt: skip

    def _kernel_args(self):
        q, q_scales, weights, pool, table, lengths, active = self.args
        nbytes = indexer.page_bytes(PAGE)
        return (q.view(torch.int8), q_scales.view(torch.uint32).reshape(ROWS, -1),
                weights, pool.view(torch.int8)[:, :nbytes],
                pool.view(torch.uint32)[:, : nbytes // 4], table, lengths,
                active)  # fmt: skip

    def reindex_score(self):
        """The reindex layer's score kernel alone (diagnostic)."""
        self.reindex.score(*self._kernel_args(), self.candidates, self.counts,
                           self.re_scratch[0])  # fmt: skip

    def reindex_select(self):
        """The reindex layer's top-k alone (diagnostic)."""
        select = len(self.reindex.select.scratch_specs(ROWS))
        self.reindex.select.run(
            self.re_scratch[0], self.counts, None, self.re_top, self.re_scratch[1],
            gather=self.candidates, scratch=self.re_scratch[2 : 2 + select],
        )  # fmt: skip


def expanded(ids, counts):
    j = torch.arange(BLOCKS * BLOCK, device="cuda")
    block = ids[:, j // BLOCK]
    live = (j[None] < counts[:, None]) & (block >= 0)
    return torch.where(live, block * BLOCK + j % BLOCK, -1).to(torch.int32)


def graph(function):
    function()
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for _ in range(CALLS):
            function()
    return capture


def elapsed(capture) -> float:
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS


def interval(differences):
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(len(differences))
    return [mean - half, mean + half]


def compare(release_module) -> dict:
    record = {"seeds": []}
    for seed in range(SEEDS):
        args = inputs(seed)
        old, new = Chain(release_module, args, False), Chain(indexer, args, True)
        old_scores = old.run_source().clone()
        new_scores = new.run_source().clone()
        lengths = args[5].clamp(0, MAX_LEN).tolist()
        for r in range(ROWS):
            n = lengths[r]
            assert torch.equal(old_scores[r, :n].view(torch.int16),
                               new_scores[r, :n].view(torch.int16)), f"seed {seed} row {r}"
        assert torch.equal(old.top, new.top), f"seed {seed}: source top-512 differ"
        assert torch.equal(old.counts, new.counts), f"seed {seed}: counts differ"
        assert torch.equal(old.candidates, expanded(new.candidates, new.counts)), (
            f"seed {seed}: the release's positions are not the ids expanded"
        )
        old_re, new_re = old.run_reindex().clone(), new.run_reindex().clone()
        for r, n in enumerate(new.counts.tolist()):
            assert torch.equal(old_re[r, :n].view(torch.int16),
                               new_re[r, :n].view(torch.int16)), f"seed {seed} row {r}"
        assert torch.equal(old.re_top, new.re_top), f"seed {seed}: reindex top differ"
        record["seeds"].append({
            "seed": seed,
            "counts": new.counts.tolist()[:6],
            "partial_last_blocks": int((args[5] % BLOCK != 0).sum()),
        })
        print(json.dumps(record["seeds"][-1]), flush=True)
    timings = {}
    for step in ("run_source", "run_reindex", "reindex_score", "reindex_select"):
        captures = {name: graph(getattr(chain, step))
                    for name, chain in (("release", old), ("ids", new))}  # fmt: skip
        times, differences = {"release": [], "ids": []}, []
        for sample in range(SAMPLES):
            order = ["release", "ids"] if sample % 2 == 0 else ["ids", "release"]
            measured = {name: elapsed(captures[name]) for name in order}
            for name in measured:
                times[name].append(measured[name])
            differences.append(measured["ids"] - measured["release"])
        timings[step] = {
            "release_us": statistics.median(times["release"]),
            "ids_us": statistics.median(times["ids"]),
            "interval_us": interval(differences),
        }
        print(step, json.dumps(timings[step]), flush=True)
    record["timings"] = timings
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True,
                        help="the release's vllm/models/deepseek_v4_1/tilelang/indexer.py")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    release = baseline_module(args.baseline, "indexer_release")
    with set_current_vllm_config(VllmConfig()):
        record = compare(release)
    print(json.dumps(record, indent=1))
    if args.gate:
        slower = [step for step in ("run_source", "run_reindex")
                  if record["timings"][step]["interval_us"][0] > 0]
        if slower:
            raise SystemExit(f"block-id candidates slower at {slower}")


if __name__ == "__main__":
    main()
