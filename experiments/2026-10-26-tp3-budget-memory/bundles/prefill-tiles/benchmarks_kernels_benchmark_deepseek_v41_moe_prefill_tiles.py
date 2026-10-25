# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired timing of DeepSeek-V4.1's routed experts with the prefill tile height (64
rows a tile from 2,048 tokens, ``MoeExpertsV41Plan.tile_height``) against the
release's plan (32 rows at every size, its ``moe.py`` given as ``--baseline``).

One layer's routed experts as serving runs them (``plan.run``: dispatch, MXFP8 cast of
the tokens, gate/up GEMM, fused SwiGLU, down GEMM, combine), at TP4's and TP3's
per-rank expert widths. Sequence-parallel prefill runs the MoE on the chunk's gathered
rows on every rank, so a call's rows are the chunk's tokens on either shape: 1,024 to
24,576 (TP3 serves 8,192, the ring 16,384), across the threshold. Two routings:
uniform top-6 over 384 experts, and a skewed one (expert popularity ~ 1 / rank^0.5),
which leaves cold experts with few routes and so more partial tiles.

Both plans read one copy of the weights and run in one workspace and output buffer, so
no difference comes from where their memory lies (GB10 reads two copies of the same
bytes up to 1.1% apart, experiments/2026-10-19-memory-placement). Each case is measured
on eight fresh inputs. On each, both plans' outputs are first compared bit for bit, then
CUDA graphs of CALLS launches of each run back to back, alternating which goes first;
the case's 95% interval comes from the eight per-input differences. The gate
(``--gate``): outputs identical everywhere; where the tile heights differ, the patched
plan not slower (interval wholly above zero fails); and faster at every size from 8,192
tokens (interval wholly below zero). Below 2,048 tokens both plans launch the release's
kernels on the same memory; those rows are reported, not gated.
"""

import argparse
import importlib.util
import json
import math
import statistics

import tile_kernels
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.deepseek_v4_1.tilelang import moe
from vllm.models.deepseek_v4_1.tilelang.gemm import pack_scale_codes

EXPERTS, TOPK, HIDDEN, LIMIT, BLOCK_N = 384, 6, 5120, 10.0, 128
SHAPES = {"TP4": 576, "TP3": 768}
TOKENS = (1024, 1536, 2048, 3072, 4096, 8192, 12288, 16384, 24576)
GATED_FASTER_FROM = 8192
ROUTINGS = ("uniform", "skewed")
SAMPLES, CALLS = 8, 3


def baseline_module(path: str, name: str):
    """The release's copy of a TileLang module, under ``name`` in the same package:
    its relative imports resolve to the package's other modules."""
    package = "vllm.models.deepseek_v4_1.tilelang"
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    spec.loader.exec_module(module)
    return module


def checkpoint_like(intermediate: int, generator):
    """Packed E2M1 weights with UE8M0 exponents 118-121 (where the checkpoint's sit)."""
    out = []
    for n, k in ((2 * intermediate, HIDDEN), (HIDDEN, intermediate)):
        packed = torch.randint(
            0, 256, (EXPERTS, n, k // 2), generator=generator, device="cuda",
            dtype=torch.uint8,
        )  # fmt: skip
        exponents = torch.randint(
            118, 122, (EXPERTS, n, k // 32), generator=generator, device="cuda",
            dtype=torch.uint8,
        )  # fmt: skip
        out.append((packed, exponents))
    return out


def routes(tokens: int, routing: str, generator):
    scores = torch.rand((tokens, EXPERTS), generator=generator, device="cuda")
    if routing == "skewed":
        rank = torch.arange(1, EXPERTS + 1, device="cuda", dtype=torch.float32)
        # Gumbel top-k: distinct experts drawn with popularity ~ rank^-0.5.
        scores = -0.5 * rank.log() - torch.log(-torch.log(scores.clamp_min(1e-20)))
    ids = scores.topk(TOPK, dim=-1).indices.to(torch.int32).contiguous()
    weights = torch.rand((tokens, TOPK), generator=generator, device="cuda") + 0.1
    return ids, weights.float().contiguous()


def graph(function):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            function()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        for _ in range(CALLS):
            function()
    return capture


def elapsed(capture) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    capture.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / CALLS  # microseconds a call


def interval(differences: list[float]) -> list[float]:
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(len(differences))
    return [mean - half, mean + half]  # t(7), 95%


def measure(plans, prepared, tokens: int, routing: str) -> dict:
    times = {name: [] for name in plans}
    differences = []
    nbytes = max(
        moe.workspace_bytes(plan.workspace_specs(tokens, torch.int32))
        for plan in plans.values()
    )
    workspace = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    for sample in range(SAMPLES):
        generator = torch.Generator(device="cuda").manual_seed(1000 * tokens + sample)
        ids, weights = routes(tokens, routing, generator)
        x = (torch.randn((tokens, HIDDEN), generator=generator, device="cuda") * 0.5)
        x = x.bfloat16()
        out = torch.empty_like(x)
        calls, outs = {}, {}
        for name, plan in plans.items():

            def call(plan=plan):
                plan.run(x, ids, weights, prepared, out, workspace)

            call()
            torch.cuda.synchronize()
            calls[name], outs[name] = call, out.clone()
        same = torch.equal(
            outs["tiles"].view(torch.int16), outs["release"].view(torch.int16)
        )
        if not same:
            raise SystemExit(f"outputs differ from the release's at {tokens} {routing}")
        captures = {name: graph(call) for name, call in calls.items()}
        order = list(plans) if sample % 2 == 0 else list(plans)[::-1]
        measured = {name: elapsed(captures[name]) for name in order}
        for name in plans:
            times[name].append(measured[name])
        differences.append(measured["tiles"] - measured["release"])
        del captures, calls, outs
    return {
        "tokens": tokens,
        "routing": routing,
        "tile_height": plans["tiles"].tile_height(tokens),
        "tiles_us": statistics.median(times["tiles"]),
        "release_us": statistics.median(times["release"]),
        "difference_us": statistics.fmean(differences),
        "interval_us": interval(differences),
        "change": statistics.fmean(differences) / statistics.median(times["release"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        required=True,
        help="the release's vllm/models/deepseek_v4_1/tilelang/moe.py",
    )
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    # Declaring a plan warms TileKernels' token cast for the scheduler's batch
    # buckets, which reads the current config.
    with set_current_vllm_config(VllmConfig()):
        results = run(args)
    print(json.dumps(results, indent=1))
    if args.gate:
        gate(results)


def run(args) -> dict:
    release = baseline_module(args.baseline, "moe_release")
    results = {}
    for shape, intermediate in SHAPES.items():
        plans = {
            "tiles": moe.MoeExpertsV41Plan(
                EXPERTS, TOPK, HIDDEN, intermediate, swiglu_limit=LIMIT
            ),
            "release": release.MoeExpertsV41Plan(
                EXPERTS, TOPK, HIDDEN, intermediate, swiglu_limit=LIMIT
            ),
        }
        for plan in plans.values():
            plan.declare_kernels()
        tile_kernels.config.set_token_alignment(32)
        generator = torch.Generator(device="cuda").manual_seed(7)
        gemms = []
        for packed, exponents in checkpoint_like(intermediate, generator):
            tiles, tails = moe.tile_expert_weights(packed.view(torch.int8), BLOCK_N)
            codes, base = pack_scale_codes(exponents)
            gemms.append((tiles, tails, moe.tile_codes(codes, BLOCK_N), base))
        prepared = tuple(gemms)  # one copy, both plans
        results[shape] = []
        for routing in ROUTINGS:
            for tokens in TOKENS:
                case = measure(plans, prepared, tokens, routing)
                results[shape].append(case)
                print(shape, json.dumps(case), flush=True)
        del prepared, plans
        torch.cuda.empty_cache()
    return results


def gate(results: dict) -> None:
    slower = [
        f"{shape} {c['tokens']} {c['routing']}"
        for shape, cases in results.items()
        for c in cases
        if c["tile_height"] != 32 and c["interval_us"][0] > 0
    ]
    not_faster = [
        f"{shape} {c['tokens']} {c['routing']}"
        for shape, cases in results.items()
        for c in cases
        if c["tokens"] >= GATED_FASTER_FROM and c["interval_us"][1] >= 0
    ]
    if slower or not_faster:
        raise SystemExit(f"slower at {slower}; not faster at {not_faster}")


if __name__ == "__main__":
    main()
