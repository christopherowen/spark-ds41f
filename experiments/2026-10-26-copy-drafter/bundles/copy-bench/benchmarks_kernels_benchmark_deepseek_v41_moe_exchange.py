# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired timing of a sequence-parallel prefill layer's MoE output on its way into
the exchange buffers: the experts' deferred combine writing each rank's block
straight into its send or parts slot (``sp_prefill._project_exchange``, in
``moe_slices`` slices, the other ranks' blocks first), against vLLM 26382f5cf's
path, one combine into the whole output and a device copy of each rank's block
into its slot (``theirs`` below).

Request inventory 5's 15,395-token prompt paid 80 of those copies at TP4, 39.4 MB
and 345 us each: 1.29% of its prefill step. The exchange needs every rank and is
stubbed here; how much of it the combine now hides is measured on the cluster, by
the spark-ds41f experiment's profile arms. This measures what one rank's GPU does
around the exchanges: a combine launch per block and slice against the one combine
and the copies.

Shapes: the TP4 recipe's 16,384-token chunk, inventory 5's prompt and an 8,192-token
chunk at TP4, and the prompt at TP3; rank 0 and the last rank (whose block holds the
pad rows). DS4.1 Flash routing (384 experts, 6 a token, hidden 5120): random routes
in the dispatch layout, random BF16 expert outputs, the shared expert's output folded
in (the FP32 combine's BF16 fold kernel, as production runs it).

Each shape is measured on eight fresh inputs. On each, both paths' slots are first
checked bit for bit. Then ``CALLS`` calls of each run back to back, alternating which
goes first, timed with CUDA events (device time: the prefill path runs eagerly, its
launches hidden under milliseconds of device work). The shape's 95% interval comes
from the eight per-input differences. The gate (``--gate``) is that the new path is
slower at no shape (the interval's lower bound above zero). One slice (the blocks
alone, no slicing) is reported beside it.
"""

import argparse
import json
import math
import statistics
from functools import partial
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4_1 import sp_prefill
from vllm.models.deepseek_v4_1.tilelang import collectives, columns
from vllm.models.deepseek_v4_1.tilelang.moe import (
    MoeDispatchV41Plan,
    carve,
    workspace_bytes,
)

DEVICE = "cuda"
CALLS = 10
SAMPLES = 8
HIDDEN, EXPERTS, TOPK, BLOCK_M = 5120, 384, 6, 32
# name: (tokens, world, rank)
SHAPES = {
    "TP4 16,384 rank 0": (16384, 4, 0),
    "TP4 15,395 rank 0": (15395, 4, 0),
    "TP4 15,395 rank 3": (15395, 4, 3),
    "TP4 8,192 rank 0": (8192, 4, 0),
    "TP3 15,395 rank 0": (15395, 3, 0),
    "TP3 15,395 rank 2": (15395, 3, 2),
}


def _stub_exchange(record=None):
    """The TP group and ``exchange_into`` stubbed: the exchange does nothing, or
    records the chunks it would send."""
    sp_prefill.get_tp_group = lambda: SimpleNamespace(
        device_communicator=SimpleNamespace(pynccl_comm=None)
    )

    def exchange_into(comm, chunks, parts, rank, world):
        if record is not None:
            record.append(chunks.clone())

    collectives.exchange_into = exchange_into


def case(plan, tokens, world, rank, seed):
    """One layer's routed outputs and both paths into its exchange slots."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    scores = torch.rand((tokens, EXPERTS), generator=g, device=DEVICE)
    ids = scores.argsort(dim=1)[:, :TOPK].to(torch.int32).contiguous()
    weights = torch.rand((tokens, TOPK), generator=g, device=DEVICE) + 0.1
    specs = plan.workspace_specs(tokens)
    scratch = torch.empty(workspace_bytes(specs), dtype=torch.uint8, device=DEVICE)
    layout = plan.dispatch(ids, weights, carve(scratch, specs))
    y = torch.randn(
        (plan.tiles(tokens) * BLOCK_M, HIDDEN), generator=g, device=DEVICE
    ).bfloat16()
    shared = torch.randn((tokens, HIDDEN), generator=g, device=DEVICE).bfloat16()
    rows = sp_prefill.SPRows(tokens, world, rank, unreduced=True)
    local = rows.local_rows
    whole = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
    send = torch.empty((world, local, HIDDEN), dtype=torch.bfloat16, device=DEVICE)
    parts = torch.empty_like(send)

    def theirs():
        """26382f5cf: the combine into the whole output, then each block copied
        into its slot (the runner's ``write``), pad rows zeroed."""
        plan.combine(y, layout, whole, shared=shared)
        for owner in range(world):
            dest = parts if owner == rank else send
            start = owner * local
            real = max(0, min(local, tokens - start))
            if real:
                dest[owner, :real].copy_(whole[start : start + real])
            columns.zero_(dest[owner, real:])

    def combine(start, stop, out):
        plan.combine(y, layout, out, shared=shared, start=start, stop=stop)

    def ours(slices=None):
        return rows.project_reduce_scatter(
            combine,
            (HIDDEN,),
            torch.bfloat16,
            torch.device(DEVICE),
            slices=sp_prefill.moe_slices(tokens) if slices is None else slices,
            per_block=True,
        )

    def check():
        record = []
        _stub_exchange(record)
        theirs()
        ours_parts = ours()
        torch.cuda.synchronize()
        _stub_exchange()
        ours_send = torch.cat(record, dim=1)
        if not torch.equal(ours_parts[rank], parts[rank]):
            raise SystemExit("this rank's part differs from the copied block")
        for owner in range(world):
            if owner != rank and not torch.equal(ours_send[owner], send[owner]):
                raise SystemExit(f"rank {owner}'s send slot differs")

    return theirs, ours, check


def device_us(function) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(CALLS):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS


def interval(differences):
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(len(differences))
    return mean, [mean - half, mean + half]


def measure(plan, tokens, world, rank) -> dict:
    times = {"theirs": [], "ours": [], "one slice": []}
    for sample in range(SAMPLES):
        theirs, ours, check = case(plan, tokens, world, rank, seed=sample)
        check()
        paths = {"theirs": theirs, "ours": ours, "one slice": partial(ours, 1)}
        for function in paths.values():
            function()  # warm
        order = list(paths) if sample % 2 == 0 else list(reversed(paths))
        for name in order:
            times[name].append(device_us(paths[name]))
        del theirs, ours, check, paths
        torch.cuda.empty_cache()
    result = {
        "slices": sp_prefill.moe_slices(tokens),
        "copies_and_combine_us": statistics.median(times["theirs"]),
        "combine_into_slots_us": statistics.median(times["ours"]),
        "one_slice_us": statistics.median(times["one slice"]),
    }
    for name, key in (("ours", "difference"), ("one slice", "one_slice_difference")):
        mean, bounds = interval([a - b for a, b in zip(times[name], times["theirs"])])
        result[f"{key}_us"] = mean
        result[f"{key}_interval_us"] = bounds
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    plan = MoeDispatchV41Plan(
        EXPERTS, TOPK, BLOCK_M, HIDDEN, torch.int32, out_dtype=torch.float32
    )
    plan.bind()
    _stub_exchange()
    results = {name: measure(plan, *shape) for name, shape in SHAPES.items()}
    print(json.dumps(results, indent=1))
    if args.gate:
        slower = [
            name
            for name, result in results.items()
            if result["difference_interval_us"][0] > 0
        ]
        if slower:
            raise SystemExit(f"combine into the slots slower at {', '.join(slower)}")


if __name__ == "__main__":
    main()
