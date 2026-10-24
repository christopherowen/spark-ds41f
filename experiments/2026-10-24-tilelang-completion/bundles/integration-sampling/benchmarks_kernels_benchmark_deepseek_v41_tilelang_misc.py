# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired per-call timing of the TileLang ports of the decode step's last CUDA and
cuBLAS kernels against the launches they replace, at serving shapes.

- The vocabulary-parallel embedding (``tilelang/embedding.py``) against vLLM's
  CUDA ``ops.vocab_parallel_embedding`` on a TP4 rank's shard (32320 x 5120 BF16)
  with int32 ids drawn from the whole vocabulary, a quarter of them the rank's.
  Cold: each call of a graph looks up its own ids and the L2 is flushed before
  the graph runs, so owned rows come from DRAM, as a freshly sampled token's do;
  warm: every call looks up the same ids (the drafter re-embeds tokens the target
  just embedded).
- DSpark's confidence head (``tilelang/confidence.py``) against cuBLAS's
  ``F.linear`` alone on the FP32 rows (the launches it replaces: the dot product
  and its reduction at one stream, the GEMV at sixteen), and against the whole
  torch head (concatenation, FP32 copy, ``F.linear``, sigmoid).

Each case is measured on eight fresh inputs; on each, CUDA graphs of 100 calls of
both run back to back, alternating which goes first, and the case's 95% interval
comes from the eight per-input differences. The gate (``--gate``) is that no port
is slower than what it replaces anywhere: the embedding cold and warm at every
decode shape and on an 8K prefill chunk, the confidence head against cuBLAS's
launches alone at one, two and sixteen streams. ``--sweep`` also times the
kernels' variants (threads a block, the embedding's load path), ungated.
"""

import argparse
import json
import math
import statistics
from collections.abc import Callable

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.models.deepseek_v4_1.tilelang import confidence, embedding

VOCAB, HIDDEN, MARKOV = 129280, 5120, 256
TP, TP_RANK = 4, 1
CALLS = 100
# Tokens a lookup embeds: the drafter's query block of 5 a request, the target's
# 1 + 5 drafts a request, and a long-prefill chunk (prefill SP embeds all of it).
EMBEDDING_SHAPES = {
    "decode c1 drafter": 5,
    "decode c1 target": 6,
    "decode c16 drafter": 80,
    "decode c16 target": 96,
    "prefill 8K": 8192,
}
# Drafts the head scores: requests x 5.
CONFIDENCE_SHAPES = {"decode c1": 5, "decode c2": 10, "decode c16": 80}

Calls = list[Callable[[], object]]


def graph(calls: Calls) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for call in calls[:3]:
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for call in calls:
            call()
    return capture


def elapsed(capture: torch.cuda.CUDAGraph, flush: torch.Tensor | None) -> float:
    if flush is not None:
        flush.zero_()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS  # microseconds a call


def measure(
    make: Callable[[int], tuple[Calls, Calls]],
    flush: torch.Tensor | None = None,
    samples: int = 8,
) -> dict:
    ours_us, theirs_us, differences = [], [], []
    for sample in range(samples):
        ours_calls, theirs_calls = make(sample)
        ours, theirs = graph(ours_calls), graph(theirs_calls)
        pair = (ours, theirs) if sample % 2 == 0 else (theirs, ours)
        times = {id(capture): elapsed(capture, flush) for capture in pair}
        ours_us.append(times[id(ours)])
        theirs_us.append(times[id(theirs)])
        differences.append(times[id(ours)] - times[id(theirs)])
        del ours, theirs, pair
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(samples)  # t(7), 95%
    return {
        "tilelang_us": statistics.median(ours_us),
        "replaced_us": statistics.median(theirs_us),
        "difference_us": mean,
        "interval_us": [mean - half, mean + half],
    }


class Embedding:
    """A TP4 rank's shard and its lookups."""

    def __init__(self) -> None:
        pad = math.lcm(64, TP)
        padded = -(-VOCAB // pad) * pad
        self.shard = VocabParallelEmbedding._get_indices(
            padded, padded, VOCAB, VOCAB, TP_RANK, TP
        )
        self.weight = torch.randn(
            padded // TP, HIDDEN, device="cuda", dtype=torch.bfloat16
        )
        # Larger than GB10's L2.
        self.flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")

    def bounds(self) -> tuple[int, int, int, int, int]:
        s = self.shard
        return (
            s.org_vocab_start_index,
            s.org_vocab_end_index,
            s.num_org_vocab_padding,
            s.added_vocab_start_index,
            s.added_vocab_end_index,
        )

    def tilelang(self, threads: int | None = None, coherent: bool = True):
        """The production op, or a variant's kernel launched on the same views."""
        start, end, padding, added_start, added_end = self.bounds()
        if threads is None:
            return lambda ids, out: embedding.vocab_embedding(
                ids, self.weight, out, start, end, padding, added_start, added_end
            )
        kernel = embedding.vocab_embedding_kernel(
            embedding.row_words(self.weight),
            "int32",
            threads=threads,
            coherent=coherent,
        )
        table = self.weight.view(torch.int32).view(-1)
        offset = end - start + padding - added_start
        return lambda ids, out: kernel(
            ids,
            table,
            out.view(torch.int32),
            start,
            end,
            added_start,
            added_end,
            offset,
        )

    def cuda(self):
        bounds = self.bounds()
        return lambda ids, out: ops.vocab_parallel_embedding(
            ids, self.weight, *bounds, out=out
        )

    def case(self, tokens: int, cold: bool, ours=None) -> dict:
        ours = ours or self.tilelang()
        theirs = self.cuda()

        def make(sample: int) -> tuple[Calls, Calls]:
            generator = torch.Generator(device="cuda").manual_seed(sample)
            ids = torch.randint(
                0,
                VOCAB,
                (CALLS if cold else 1, tokens),
                generator=generator,
                dtype=torch.int32,
                device="cuda",
            )
            out_ours = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
            out_theirs = torch.empty_like(out_ours)
            rows = [ids[j if cold else 0] for j in range(CALLS)]
            return (
                [lambda r=r: ours(r, out_ours) for r in rows],
                [lambda r=r: theirs(r, out_theirs) for r in rows],
            )

        return measure(make, self.flush if cold else None)


def confidence_case(rows: int, whole_head: bool, threads: int | None = None) -> dict:
    """The TileLang head against cuBLAS's ``F.linear`` alone, or the torch head."""
    kernel = (
        None
        if threads is None
        else confidence.dspark_confidence_kernel(HIDDEN, MARKOV, threads=threads)
    )

    def make(sample: int) -> tuple[Calls, Calls]:
        generator = torch.Generator(device="cuda").manual_seed(sample)
        hidden = torch.randn(rows, HIDDEN, generator=generator, device="cuda")
        markov = torch.randn(rows, MARKOV, generator=generator, device="cuda")
        hidden, markov = hidden.bfloat16(), markov.bfloat16()
        width = HIDDEN + MARKOV
        weight = torch.randn(width, generator=generator, device="cuda")
        weight /= math.sqrt(width)
        out = torch.empty(rows, dtype=torch.float32, device="cuda")
        x = torch.cat([hidden, markov], dim=-1).float()
        w = weight.view(1, -1)

        def ours():
            if kernel is None:
                confidence.dspark_confidence(hidden, markov, weight, out)
            else:
                kernel(hidden, markov, weight, out)

        def theirs():
            if whole_head:
                cat = torch.cat([hidden, markov], dim=-1).float()
                return torch.sigmoid(F.linear(cat, w).squeeze(-1))
            return F.linear(x, w)

        return [ours] * CALLS, [theirs] * CALLS

    return measure(make)


def slower(results: dict) -> list[str]:
    return [name for name, result in results.items() if result["interval_us"][0] > 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    table = Embedding()
    lookups = {}
    for name, tokens in EMBEDDING_SHAPES.items():
        lookups[f"{name} cold"] = table.case(tokens, cold=True)
        if name.startswith("decode"):
            lookups[f"{name} warm"] = table.case(tokens, cold=False)
    heads = {
        name: confidence_case(rows, whole_head=False)
        for name, rows in CONFIDENCE_SHAPES.items()
    }
    whole_heads = {
        name: confidence_case(rows, whole_head=True)
        for name, rows in CONFIDENCE_SHAPES.items()
    }
    results = {
        "embedding vs CUDA kernel": lookups,
        "confidence head vs cuBLAS F.linear": heads,
        "confidence head vs torch head": whole_heads,
    }
    if args.sweep:
        sweep = {}
        for threads in (64, 128, 256):
            for coherent in (True, False):
                ours = table.tilelang(threads, coherent)
                path = "coherent" if coherent else "read-only"
                for name in ("decode c1 target", "decode c16 target", "prefill 8K"):
                    tokens = EMBEDDING_SHAPES[name]
                    sweep[f"embedding {threads} threads {path}: {name} cold"] = (
                        table.case(tokens, cold=True, ours=ours)
                    )
            for name, rows in CONFIDENCE_SHAPES.items():
                sweep[f"confidence {threads} threads: {name}"] = confidence_case(
                    rows, whole_head=False, threads=threads
                )
        results["sweep (ungated)"] = sweep
    print(json.dumps(results, indent=1))
    if args.gate:
        failed = [f"embedding {name}" for name in slower(lookups)]
        failed += [f"confidence head {name}" for name in slower(heads)]
        if failed:
            raise SystemExit(f"TileLang slower at {', '.join(failed)}")


if __name__ == "__main__":
    main()
