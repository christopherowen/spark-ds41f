#!/usr/bin/env python3
"""GB10 microbenchmark of DSpark's vocabulary-parallel greedy draft steps on one
rank's Markov shard (DS4.1-Flash's own Markov W2 and W1, packed as serving packs
them), materialized bias against the fused projection:

- ``materialized``: ``project_head`` writes the shard's bias ``[rows, shard]``
  (``markov_bias_local``), then ``greedy_markov_shard_pair`` reduces base + bias
  (candidate-37's draft path);
- ``fused``: ``project_head_markov_greedy_pairs`` adds the base logits and reduces
  each vocabulary tile in the projection's epilogue, then finishes the pairs.

Two loops, each CUDA-graph captured and replayed:

- ``reduce``: the five draft positions' reductions alone (the changed kernels);
- ``chain``: the speculator's sampling loop without the network: the anchor's
  Markov embedding, then per position the reduction, the pairs copied into a
  rank slot of a gathered buffer (the all-gather's stand-in) and
  ``select_first_argmax`` with the next position's embedding.

``cold`` replays a 256 MB write before each timed loop (the drafter's forward and
the target's evict the Markov shard from L2 before sampling); ``hot`` replays the
loops back to back. Arms alternate within each round and swap order every round.
"""

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4_1.tilelang import vocab
from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
    greedy_markov_shard_pair,
    markov_anchor_embed,
    scratch_shape,
    select_first_argmax,
)

SHAPES = {"tp4": (32320, 4), "tp3": (43136, 3)}  # padded shard rows, ranks
ROWS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
N_SPEC = 5
VOCAB = 129280
ARMS = ("materialized", "fused")
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def load_bf16(path: Path) -> torch.Tensor:
    meta = json.loads(path.with_suffix(".json").read_text())
    data = bytearray(path.with_suffix(".bin").read_bytes())
    return torch.frombuffer(data, dtype=torch.bfloat16).view(*meta["shape"])


def packed_head(weight: torch.Tensor, dtype: torch.dtype, max_rows: int):
    packed = vocab.pack_bf16_weight(weight.cuda().contiguous())
    head = SimpleNamespace(
        weight=packed.values,
        weight_exception_rows=packed.exception_rows,
        weight_exception_entries=packed.exception_entries,
        weight_row_offsets=packed.row_offsets,
        packed_bf16_in_features=int(weight.shape[1]),
        packed_bf16_exponent_base=packed.exponent_base,
        tilelang_max_rows=max_rows,
        tilelang_markov_head=True,
    )
    vocab.prepare_packed_head(head, dtype)
    return head, int(packed.exceptions)


class DraftSteps:
    """One rank's buffers for ``rows`` requests' five greedy draft positions."""

    def __init__(self, head, table, rows, n, ranks, dtype, top2, seed):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.head, self.table, self.n, self.top2 = head, table, n, top2
        rank = table.shape[1]
        # compute_draft_logits_local's [rows * positions, shard] as [rows, positions, shard]
        self.base = (
            torch.randn(rows, N_SPEC, n, generator=generator, device="cuda") * 3
        ).to(dtype)
        self.ids = torch.randint(
            0, VOCAB, (rows,), generator=generator, device="cuda", dtype=torch.int32
        )
        self.anchor = torch.arange(rows, device="cuda")
        self.step_embed = torch.empty(rows, rank, dtype=torch.bfloat16, device="cuda")
        self.confidence = torch.empty(
            rows, N_SPEC, rank, dtype=torch.bfloat16, device="cuda"
        )
        self.pairs = torch.empty(rows, 4, device="cuda")
        # The other ranks' slots hold masked pairs, so this rank's selection wins.
        self.gathered = torch.full((rows, 4 * ranks), -float("inf"), device="cuda")
        self.gathered[:, 1::2] = float(1 << 30)
        self.tokens = torch.zeros(rows, N_SPEC, dtype=torch.int64, device="cuda")
        self.runner_up = torch.zeros_like(self.tokens)
        self.margins = torch.zeros(rows, N_SPEC, device="cuda")
        self.conf_weight = torch.randn(rank, generator=generator, device="cuda")
        shape = scratch_shape(rows, n)
        self.partial_values = torch.empty(shape, device="cuda")
        self.partial_indices = torch.empty(shape, dtype=torch.int32, device="cuda")

    def reduce(self, i: int, arm: str) -> None:
        if arm == "fused":
            vocab.project_head_markov_greedy_pairs(
                self.head,
                self.step_embed,
                self.base[:, i],
                vocab=self.n,
                vocab_start=0,
                out=self.pairs,
                top2=self.top2,
            )
        else:
            bias = vocab.project_head(self.head, self.step_embed)
            greedy_markov_shard_pair(
                self.base[:, i, : self.n],
                bias[:, : self.n],
                0,
                self.pairs,
                self.partial_values,
                self.partial_indices,
                top2=self.top2,
            )

    def embed_anchor(self) -> None:
        markov_anchor_embed(
            self.ids, self.anchor, self.table, self.step_embed, self.confidence[:, 0]
        )

    def reduce_loop(self, arm: str) -> None:
        for i in range(N_SPEC):
            self.reduce(i, arm)

    def chain_loop(self, arm: str) -> None:
        self.embed_anchor()
        for i in range(N_SPEC):
            self.reduce(i, arm)
            self.gathered[:, :4].copy_(self.pairs)
            has_embed = i + 1 < N_SPEC
            select_first_argmax(
                self.gathered,
                self.tokens[:, i],
                pair_width=4,
                embed=(
                    (self.table, self.step_embed, self.confidence[:, i + 1])
                    if has_embed
                    else None
                ),
                top2_branch=(
                    (
                        self.conf_weight if has_embed else None,
                        self.runner_up[:, i],
                        self.margins[:, i],
                        0.25,
                        1.0,
                    )
                    if self.top2
                    else None
                ),
            )


def capture(fn) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def time_graph(graph, iters, flush=None) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for k in range(iters):
        if flush is not None:
            flush.replay()
        starts[k].record()
        graph.replay()
        ends[k].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]  # us


def summary(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p10_us": ordered[len(ordered) // 10],
        "p90_us": ordered[(9 * len(ordered)) // 10],
    }


def exact(steps: DraftSteps) -> bool:
    """Both arms' chains draw the same bits: tokens, runner-ups, margins, embeddings."""
    outs = []
    for arm in ARMS:
        steps.tokens.fill_(-1)
        steps.runner_up.zero_()
        steps.margins.zero_()
        steps.chain_loop(arm)
        torch.cuda.synchronize()
        outs.append(
            [
                t.clone()
                for t in (
                    steps.tokens,
                    steps.runner_up,
                    steps.margins.view(torch.int32),
                    steps.step_embed.view(torch.int16),
                    steps.confidence.view(torch.int16),
                    steps.pairs.view(torch.int32),
                )
            ]
        )
    return all(torch.equal(a, b) for a, b in zip(*outs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("/bench/weights"))
    parser.add_argument("--shapes", default="tp4,tp3")
    parser.add_argument("--rows", default=",".join(map(str, ROWS)))
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--top2", action="store_true")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    dtype = DTYPES[args.dtype]
    rows_list = [int(r) for r in args.rows.split(",")]
    w2 = load_bf16(args.weights / "markov_w2")
    table = load_bf16(args.weights / "markov_w1").cuda()
    flush_buffer = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    flush = capture(lambda: flush_buffer.fill_(1))
    results = {
        "device": torch.cuda.get_device_name(),
        "dtype": args.dtype,
        "top2": args.top2,
        "iters": args.iters,
        "rounds": args.rounds,
        "n_spec": N_SPEC,
        "flush_us": summary(time_graph(flush, 50)),
        "cases": [],
    }
    for shape in args.shapes.split(","):
        n, ranks = SHAPES[shape]
        head, exceptions = packed_head(w2[:n], dtype, max(rows_list))
        for rows in rows_list:
            steps = DraftSteps(head, table, rows, n, ranks, dtype, args.top2, seed=rows)
            case = {
                "shape": shape,
                "shard_rows": n,
                "exceptions": exceptions,
                "rows": rows,
                "row_tile": vocab.row_tile(rows),
                "bit_exact": exact(steps),
                "rounds": [],
            }
            graphs = {
                (loop, arm): capture(
                    (lambda a=arm: steps.reduce_loop(a))
                    if loop == "reduce"
                    else (lambda a=arm: steps.chain_loop(a))
                )
                for loop in ("reduce", "chain")
                for arm in ARMS
            }
            for r in range(args.rounds):
                order = ARMS if r % 2 == 0 else ARMS[::-1]
                record = {}
                for loop in ("reduce", "chain"):
                    for l2 in ("cold", "hot"):
                        for arm in order:
                            record[f"{loop}/{l2}/{arm}"] = summary(
                                time_graph(
                                    graphs[loop, arm],
                                    args.iters,
                                    flush if l2 == "cold" else None,
                                )
                            )
                case["rounds"].append(record)
            for key in case["rounds"][0]:
                medians = [rnd[key]["median_us"] for rnd in case["rounds"]]
                case[key] = {
                    "median_us": statistics.median(medians),
                    "min_us": min(medians),
                    "max_us": max(medians),
                }
            line = [f"{shape} rows={rows:2d} exact={case['bit_exact']}"]
            for loop in ("reduce", "chain"):
                for l2 in ("cold", "hot"):
                    a = case[f"{loop}/{l2}/materialized"]["median_us"]
                    b = case[f"{loop}/{l2}/fused"]["median_us"]
                    line.append(f"{loop}/{l2} {a:7.1f}->{b:7.1f} ({b - a:+6.1f})")
            print(" | ".join(line), flush=True)
            results["cases"].append(case)
            del graphs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
