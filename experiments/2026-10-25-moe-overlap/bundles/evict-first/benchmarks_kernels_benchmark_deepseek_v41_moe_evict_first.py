# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired timing of DeepSeek-V4.1's routed-expert GEMMs with every weight byte loaded
evict-first in L2 against the same GEMMs with the default policy (the release's
``moe.py``, given as ``--baseline``), at the TP4 and TP3 decode shapes.

Two sequences per shape and decode row count:

- ``routed``: one layer's routed experts (the gate/up GEMM, the fused SwiGLU, the down
  GEMM, as the plan launches them), whose weights stream from memory: a decode
  step's experts are several times the 24 MB L2;
- ``shared after routed``: the shared expert's weights prefetched into L2 as the FFN
  window does (``l2_prefetch``'s kernel at the recipe's grid), the routed experts,
  then the shared expert's gate/up and down GEMMs. Under the default policy the routed
  stream can evict the prefetched lines; under evict-first they should outlive it.

Each case is measured on eight fresh inputs. On each, both builds' outputs are first
compared bit for bit, then CUDA graphs of 20 sequences of each run back to back,
alternating which goes first; the case's 95% interval comes from the eight per-input
differences. ``shared_saved_us`` is the shared GEMMs' part of the difference: the
second sequence's difference less the first's. The gate (``--gate``) is that neither
sequence is slower at any shape and row count.
"""

import argparse
import importlib.util
import json
import math
import statistics

import tile_kernels
import torch

from vllm.models.deepseek_v4_1.tilelang import moe, prepare
from vllm.models.deepseek_v4_1.tilelang.gemm import (
    DECODE_ROWS,
    WIDE_DECODE,
    decode_tile_rows,
    pack_scale_codes,
    pack_scale_words,
    scale_words,
)
from vllm.models.deepseek_v4_1.tilelang.linear import (
    _activation_specs,
    _block32_kernels,
    _calls,
    _launch,
)
from vllm.models.glm5next.nvidia import l2_prefetch as glm_prefetch
from vllm.models.glm5next.nvidia.tilelang.l2_prefetch import l2_prefetch_kernel

EXPERTS, TOPK, HIDDEN, LIMIT, BLOCK_N = 384, 6, 5120, 10.0, 128
# Per-rank expert widths (routed and shared): TP4's 576 columns, TP3's 768; and the
# recipe's prefetch grid (VLLM_L2_PREFETCH_GRID) on each.
SHAPES = {"TP4": (576, 2), "TP3": (768, 4)}
# Decode rows: one stream's verification step, four streams', sixteen streams'.
ROWS = {"c1": 4, "c4": 24, "c16": 96}
CALLS = 20


def baseline_module(path: str, name: str):
    """The release's copy of a TileLang module, under ``name`` in the same package:
    its relative imports resolve to the package's other modules."""
    package = "vllm.models.deepseek_v4_1.tilelang"
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    spec.loader.exec_module(module)
    return module


def routed_weights(intermediate: int, generator) -> list[tuple]:
    """One layer's routed experts as the plan holds them: random E2M1 bytes with
    UE8M0 exponents 118-121 (where the checkpoint's sit), tile-major, coded scales."""
    weights = []
    for n, k in ((2 * intermediate, HIDDEN), (HIDDEN, intermediate)):
        packed = torch.randint(
            0, 256, (EXPERTS, n, k // 2), generator=generator, device="cuda",
            dtype=torch.uint8,
        )  # fmt: skip
        exponents = torch.randint(
            118, 122, (EXPERTS, n, k // 32), generator=generator, device="cuda",
            dtype=torch.uint8,
        )  # fmt: skip
        tiles, tails = moe.tile_expert_weights(packed.view(torch.int8), BLOCK_N)
        codes, base = pack_scale_codes(exponents)
        weights.append(
            moe._present((tiles, tails, moe.tile_codes(codes, BLOCK_N), base))
        )
    return weights


class SharedExpert:
    """The shared expert's two block-32 FP8 GEMMs as its TileLang linears run them,
    with one decode step's MXFP8 inputs."""

    def __init__(self, intermediate: int, rows: int, generator):
        self.rows = rows
        self.projections = []
        for n, k in ((2 * intermediate, HIDDEN), (HIDDEN, intermediate)):
            weight = (
                torch.randn((n, k), generator=generator, device="cuda") * 0.05
            ).to(torch.float8_e4m3fn)
            exponents = torch.randint(
                118, 124, (n // 32, k // 32), generator=generator, device="cuda",
                dtype=torch.uint8,
            )  # fmt: skip
            words = pack_scale_words(exponents, rows=n)
            wide = WIDE_DECODE.get((n, k), {})
            decode_rows = max(wide, default=DECODE_ROWS)
            kernels = _launch(_block32_kernels(n, k))
            if rows > decode_rows:
                gemm = kernels["prefill"]
                xq_shape = (rows, k)
            else:
                gemm = kernels["decode"][decode_tile_rows(rows, wide)]
                xq_shape = _activation_specs(rows, k, decode_rows)[0][0]
            values = (
                torch.randn(xq_shape, generator=generator, device="cuda") * 20
            ).to(torch.float8_e4m3fn)
            act_words = torch.randint(
                118, 126, (xq_shape[0], 4 * scale_words(k)), generator=generator,
                device="cuda", dtype=torch.uint8,
            ).view(torch.uint32)  # fmt: skip
            out = torch.empty((rows, n), dtype=torch.bfloat16, device="cuda")
            self.projections.append((gemm, values, weight, act_words, words, out))

    def segments(self) -> torch.Tensor:
        """The prefetch kernel's (pointer, bytes) table of the shared weights and their
        scale words, as the FFN window lists them."""
        table = []
        for _, _, weight, _, words, _ in self.projections:
            for tensor in (weight, words):
                table += [tensor.data_ptr(), tensor.numel() * tensor.element_size()]
        return torch.tensor(table, dtype=torch.int64, device="cuda")

    def __call__(self) -> None:
        for gemm, values, weight, act_words, words, out in self.projections:
            gemm(values, weight, act_words, words, out)


class Case:
    """One decode step's routed experts with both builds' kernels and outputs, the
    shared expert and the prefetch of its weights."""

    def __init__(self, arms, weights, intermediate, grid, rows, seed):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        dispatch = moe.MoeDispatchV41Plan(
            EXPERTS, TOPK, 32, HIDDEN, torch.int32, torch.bfloat16
        )
        prepare.declare(lambda: _calls(dispatch.kernels()), bind=dispatch.bind)
        ids = torch.stack(
            [
                torch.randperm(EXPERTS, generator=generator, device="cuda")[:TOPK]
                for _ in range(rows)
            ]
        ).to(torch.int32)
        route_weights = torch.rand((rows, TOPK), generator=generator, device="cuda")
        specs = dispatch.workspace_specs(rows)
        self.layout = dispatch.dispatch(
            ids,
            route_weights + 0.1,
            moe.carve(
                torch.empty(
                    moe.workspace_bytes(specs), dtype=torch.uint8, device="cuda"
                ),
                specs,
            ),
        )
        x = torch.randn((rows, HIDDEN), generator=generator, device="cuda") * 0.5
        self.xq, x_sf = tile_kernels.quant.per_token_cast(
            x.bfloat16(), "e4m3", 32, round_sf=True, use_packed_ue8m0=True
        )
        self.x_words = x_sf.contiguous().view(torch.uint32)
        layout_rows = self.layout["order"].shape[0]
        self.h = torch.empty(
            (layout_rows, 2 * intermediate), dtype=torch.bfloat16, device="cuda"
        )
        self.hq = torch.empty(
            (layout_rows, intermediate), dtype=torch.float8_e4m3fn, device="cuda"
        )
        self.h_sf = torch.empty(
            (layout_rows, 4 * scale_words(intermediate)),
            dtype=torch.uint8,
            device="cuda",
        )
        self.arms = arms
        self.weights = weights
        self.out = {
            name: torch.empty(
                (layout_rows, HIDDEN), dtype=torch.bfloat16, device="cuda"
            )
            for name in arms
        }
        self.shared = SharedExpert(intermediate, rows, generator)
        self.segments = self.shared.segments()
        self.prefetch = l2_prefetch_kernel(
            grid, glm_prefetch._BLOCK, glm_prefetch._CHUNK_BYTES
        )

    def routed(self, name: str) -> None:
        kernels, layout = self.arms[name], self.layout
        order, tile_expert = layout["order"], layout["tile_expert"]
        gate_up, down = self.weights
        kernels["gate_up"](self.xq, self.x_words, *gate_up, order, tile_expert, self.h)
        kernels["swiglu"](
            self.h, self.hq, self.h_sf, order, layout["layout_weights"], tile_expert
        )
        kernels["down"](
            self.hq, self.h_sf.view(torch.uint32), *down, order, tile_expert,
            self.out[name],
        )  # fmt: skip

    def shared_after_routed(self, name: str) -> None:
        self.prefetch(self.segments)
        self.routed(name)
        self.shared()

    def check(self) -> None:
        for name in self.arms:
            self.routed(name)
        torch.cuda.synchronize()
        live = self.layout["order"] >= 0
        new, old = (self.out[name][live].view(torch.int16) for name in self.arms)
        if not torch.equal(new, old):
            raise SystemExit("evict-first routed outputs differ from the release's")


def graph(function):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
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
    return start.elapsed_time(end) * 1000 / CALLS  # microseconds per sequence


def interval(differences: list[float]) -> list[float]:
    mean = statistics.fmean(differences)
    half = 2.365 * statistics.stdev(differences) / math.sqrt(len(differences))
    return [mean - half, mean + half]  # t(7), 95%


def measure(arms, weights, shape: str, rows: int, samples: int = 8) -> dict:
    intermediate, grid = SHAPES[shape]
    times = {
        (sequence, name): []
        for sequence in ("routed", "shared after routed")
        for name in arms
    }
    differences = {"routed": [], "shared after routed": []}
    for sample in range(samples):
        case = Case(arms, weights, intermediate, grid, rows, seed=sample)
        case.check()
        for sequence, method in (
            ("routed", case.routed),
            ("shared after routed", case.shared_after_routed),
        ):
            captures = {
                name: graph(lambda name=name, method=method: method(name))
                for name in arms
            }
            order = list(arms) if sample % 2 == 0 else list(arms)[::-1]
            measured = {name: elapsed(captures[name]) for name in order}
            for name in arms:
                times[(sequence, name)].append(measured[name])
            differences[sequence].append(measured["evict_first"] - measured["default"])
            del captures
        del case
    result = {"rows": rows}
    for sequence, diffs in differences.items():
        result[sequence] = {
            "evict_first_us": statistics.median(times[(sequence, "evict_first")]),
            "default_us": statistics.median(times[(sequence, "default")]),
            "difference_us": statistics.fmean(diffs),
            "interval_us": interval(diffs),
        }
    result["shared_saved_us"] = (
        result["routed"]["difference_us"]
        - result["shared after routed"]["difference_us"]
    )
    return result


def plan_kernels(module, intermediate: int) -> dict:
    """The routed GEMMs and SwiGLU exactly as ``module``'s plan launches them."""
    plan = module.MoeExpertsV41Plan(
        EXPERTS, TOPK, HIDDEN, intermediate, swiglu_limit=LIMIT
    )
    return _launch(plan.kernels())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        required=True,
        help="the release's vllm/models/deepseek_v4_1/tilelang/moe.py",
    )
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    release = baseline_module(args.baseline, "moe_release")
    results = {}
    for shape, (intermediate, _) in SHAPES.items():
        arms = {
            "evict_first": plan_kernels(moe, intermediate),
            "default": plan_kernels(release, intermediate),
        }
        weights = routed_weights(
            intermediate, torch.Generator(device="cuda").manual_seed(intermediate)
        )
        results[shape] = {
            label: measure(arms, weights, shape, rows) for label, rows in ROWS.items()
        }
        del weights
        torch.cuda.empty_cache()
    print(json.dumps(results, indent=1))
    if args.gate:
        slower = [
            f"{shape} {label} {sequence}"
            for shape, cases in results.items()
            for label, case in cases.items()
            for sequence in ("routed", "shared after routed")
            if case[sequence]["interval_us"][0] > 0
        ]
        if slower:
            raise SystemExit(f"evict-first slower at {', '.join(slower)}")


if __name__ == "__main__":
    main()
