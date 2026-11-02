#!/usr/bin/env python3
"""Functional soak (no timing) of DSpark's vocabulary-parallel greedy draft steps
through the fused Markov projection, under serving's workspace discipline, against
the materialized path (``bench_markov_greedy.DraftSteps``' two arms).

As the V2 runner does for DSpark, a workspace manager has a draft lane (lane 1).
Each captured size is warmed eagerly, then captured, under that lane, and the
workspace is locked after the last capture. Each step:

- picks a captured size ``S`` (the padded request count) and ``a <= S`` active rows;
- refills the active rows' base logits (some with exact ties, NaNs or -inf rows) and
  anchor ids, and poisons the padding rows (NaN, +inf, -inf or huge values, garbage
  ids);
- fills the draft lane's whole workspace slot with 0xFF (NaN as FP32, -1 as int32),
  as the drafter's other borrowers may leave it, before each arm's replay;
- replays the materialized graph, then the fused one, and compares every output as
  bits over all ``S`` rows: tokens, runner-ups, margins, Markov embeddings and the
  shard pairs. Every token must lie in this rank's shard.

A negative control replays the fused graph once on a base that differs in one value
and must be flagged.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_markov_greedy import (  # noqa: E402
    SHAPES,
    VOCAB,
    DraftSteps,
    capture,
    load_bf16,
    packed_head,
)

from vllm.v1.worker.workspace import (  # noqa: E402
    current_workspace_manager,
    init_workspace_manager,
    lock_workspace,
    use_workspace_lane,
)

DRAFT_LANE = 1
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def outputs(steps: DraftSteps) -> list[torch.Tensor]:
    return [
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


def poison_draft_slot() -> None:
    with use_workspace_lane(DRAFT_LANE):
        manager = current_workspace_manager()
        slot = manager._current_workspaces[DRAFT_LANE]
        if slot is not None:
            slot.fill_(0xFF)


def fill_step(steps: DraftSteps, active: int, rng: random.Random, gen) -> None:
    rows, positions, n = steps.base.shape
    base = torch.randn(rows, positions, n, generator=gen, device="cuda") * 3
    for r in range(active):
        kind = rng.random()
        if kind < 0.05:  # exact ties across the shard
            base[r, :, [3, n - 1, n // 2]] = 40.0
        elif kind < 0.08:
            base[r, :, rng.randrange(n)] = float("nan")
        elif kind < 0.10:
            base[r] = -float("inf")
    for r in range(active, rows):  # padding rows: whatever a dummy row holds
        base[r] = rng.choice(
            (float("nan"), float("inf"), -float("inf"), 3.0e38, 0.0)
        )
    steps.base.copy_(base.to(steps.base.dtype))
    ids = torch.randint(0, VOCAB, (rows,), generator=gen, device="cuda")
    ids[active:] = torch.randint(-(1 << 20), 1 << 20, (rows - active,), generator=gen, device="cuda")
    steps.ids.copy_(ids.to(steps.ids.dtype))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("/bench/weights"))
    parser.add_argument("--shapes", default="tp4,tp3")
    parser.add_argument("--sizes", default="1,2,4,8,16,24,32,48")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--top2", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dtype = DTYPES[args.dtype]
    sizes = [int(s) for s in args.sizes.split(",")]
    init_workspace_manager(torch.device("cuda"), num_lanes=2)
    # The drafter's own borrowers size its lane before sampling does.
    with use_workspace_lane(DRAFT_LANE):
        current_workspace_manager().get_simultaneous(((4 << 20,), torch.uint8))
    w2 = load_bf16(args.weights / "markov_w2")
    table = load_bf16(args.weights / "markov_w1").cuda()
    plan = {}
    for shape in args.shapes.split(","):
        n, ranks = SHAPES[shape]
        head, _ = packed_head(w2[:n], dtype, max(sizes))
        for size in sizes:
            steps = DraftSteps(head, table, size, n, ranks, dtype, args.top2, seed=size)
            with use_workspace_lane(DRAFT_LANE):
                graphs = {
                    arm: capture(lambda a=arm: steps.chain_loop(a))
                    for arm in ("materialized", "fused")
                }
            plan[shape, size] = (steps, graphs, n)
    lock_workspace()
    slot_bytes = current_workspace_manager()._current_workspaces[DRAFT_LANE].numel()

    rng = random.Random(41)
    gen = torch.Generator(device="cuda").manual_seed(41)
    keys = sorted(plan)
    tally = {
        f"{shape}/{size}": {"steps": 0, "mismatches": 0, "out_of_shard": 0, "nonfinite_pair_rows": 0}
        for shape, size in keys
    }
    for _ in range(args.steps):
        shape, size = rng.choice(keys)
        steps, graphs, n = plan[shape, size]
        active = rng.randint(1, size)
        fill_step(steps, active, rng, gen)
        got = {}
        for arm in ("materialized", "fused"):
            poison_draft_slot()
            graphs[arm].replay()
            torch.cuda.synchronize()
            got[arm] = outputs(steps)
        cell = tally[f"{shape}/{size}"]
        cell["steps"] += 1
        if not all(torch.equal(a, b) for a, b in zip(got["materialized"], got["fused"])):
            cell["mismatches"] += 1
        tokens = got["fused"][0]
        cell["out_of_shard"] += int(((tokens < 0) | (tokens >= n)).any(dim=1).sum())
        pairs = got["fused"][5].view(torch.float32)
        cell["nonfinite_pair_rows"] += int((~torch.isfinite(pairs[:, 0])).sum())

    # Negative control: one base value differs for the fused arm only.
    shape, size = keys[-1]
    steps, graphs, n = plan[shape, size]
    fill_step(steps, size, rng, gen)
    steps.base[0, :, 7] = 100.0
    poison_draft_slot()
    graphs["materialized"].replay()
    torch.cuda.synchronize()
    reference = outputs(steps)
    steps.base[0, :, 7] = -100.0
    poison_draft_slot()
    graphs["fused"].replay()
    torch.cuda.synchronize()
    control_flagged = not all(torch.equal(a, b) for a, b in zip(reference, outputs(steps)))

    results = {
        "device": torch.cuda.get_device_name(),
        "dtype": args.dtype,
        "top2": args.top2,
        "steps": args.steps,
        "draft_lane_slot_bytes": slot_bytes,
        "negative_control_flagged": control_flagged,
        "cells": tally,
        "total_mismatches": sum(c["mismatches"] for c in tally.values()),
        "total_out_of_shard": sum(c["out_of_shard"] for c in tally.values()),
    }
    print(json.dumps({k: v for k, v in results.items() if k != "cells"}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
