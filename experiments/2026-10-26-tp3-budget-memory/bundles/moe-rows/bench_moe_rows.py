#!/usr/bin/env python3
"""One layer's routed experts (DeepSeek-V4.1, the release's TileLang kernels) against
the prefill chunk size: what a chunk pays regardless of its size.

For each per-rank expert width (TP3's 768 columns, TP4's 576) and token count T:
uniform top-6 routing over 384 experts (distinct per token), MXFP8 tokens, random
MXFP4 weights as the plan holds them. Each kernel the plan launches for a prefill
chunk (dispatch, gate/up GEMM, fused SwiGLU, down GEMM, combine) is timed alone in a
CUDA graph of CALLS launches, REPEATS replays, median. The tile height block_M is the
release's 32 and, as a second arm, 64; both arms' combined outputs are compared bit
for bit at every T (a row's sum over K does not depend on how rows are tiled).

Per T the record holds each kernel's time and the layout's tile count. The summary
fits time = alpha + beta * T on T >= 2048 (every expert has rows): alpha is the part
a chunk pays whatever its size; compare it with stream_us, one read of the layer's
expert weights (a reduction over every weight byte, at the DRAM rate). Twenty
full-row layers a chunk (CED decoder layers run on compacted rows) turn alpha into
the routed experts' share of the per-chunk cost fitted from servebench TTFTs
(experiments/2026-10-26-tp3-budget-memory/README.md).
"""

import argparse
import json
import statistics

import tile_kernels
import torch

from vllm.models.deepseek_v4_1.tilelang import moe, prepare
from vllm.models.deepseek_v4_1.tilelang.gemm import pack_scale_codes, scale_words
from vllm.models.deepseek_v4_1.tilelang.linear import _calls, _launch

EXPERTS, TOPK, HIDDEN, LIMIT, BLOCK_N = 384, 6, 5120, 10.0, 128
SHAPES = {"TP3": 768, "TP4": 576}
TOKENS = (128, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576)
BLOCK_MS = (32, 64)
CALLS, REPEATS = 3, 5
FULL_ROW_LAYERS = 20


def routed_weights(intermediate: int, generator) -> list[tuple]:
    """One layer's routed experts as the plan holds them: random E2M1 bytes with
    UE8M0 exponents 118-121, tile-major, coded scales."""
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
        del packed, exponents
    return weights


def weight_bytes(weights) -> int:
    return sum(t.numel() * t.element_size() for w in weights for t in w)


def timed(function) -> float:
    """Median microseconds per call of ``function`` in a graph of CALLS launches."""
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
    samples = []
    for _ in range(REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        capture.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / CALLS)
    del capture
    return statistics.median(samples)


def stream_us(weights) -> float:
    """One read of every weight byte: a sum over each tensor as int32 words."""
    views = [
        t.reshape(-1).view(torch.uint8).view(torch.int32)
        if t.numel() * t.element_size() % 4 == 0
        else t.reshape(-1).view(torch.uint8)
        for w in weights
        for t in w
    ]
    sink = torch.zeros(len(views), dtype=torch.int64, device="cuda")

    def read():
        for i, v in enumerate(views):
            sink[i] = torch.sum(v, dtype=torch.int64)

    return timed(read)


class Arm:
    """The release's plan kernels at one tile height."""

    def __init__(self, intermediate: int, block_M: int):
        self.block_M = block_M
        plan = moe.MoeExpertsV41Plan(
            EXPERTS, TOPK, HIDDEN, intermediate, swiglu_limit=LIMIT, block_M=block_M
        )
        self.kernels = _launch(plan.kernels())
        self.dispatch = moe.MoeDispatchV41Plan(
            EXPERTS, TOPK, block_M, HIDDEN, torch.int32, torch.bfloat16
        )
        prepare.declare(
            lambda: _calls(self.dispatch.kernels()), bind=self.dispatch.bind
        )


def case(arm: Arm, weights, intermediate: int, ids, route_weights, xq, x_words):
    rows = ids.shape[0]
    specs = arm.dispatch.workspace_specs(rows)
    scratch = torch.empty(moe.workspace_bytes(specs), dtype=torch.uint8, device="cuda")
    buffers = moe.carve(scratch, specs)
    layout = arm.dispatch.dispatch(ids, route_weights, buffers)
    layout_rows = layout["order"].shape[0]
    h = torch.empty((layout_rows, 2 * intermediate), dtype=torch.bfloat16, device="cuda")
    hq = torch.empty((layout_rows, intermediate), dtype=torch.float8_e4m3fn, device="cuda")
    h_sf = torch.empty(
        (layout_rows, 4 * scale_words(intermediate)), dtype=torch.uint8, device="cuda"
    )
    y = torch.empty((layout_rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    gate_up, down = weights
    k = arm.kernels
    order, tile_expert = layout["order"], layout["tile_expert"]

    def run_dispatch():
        arm.dispatch.dispatch(ids, route_weights, buffers)

    def run_gate_up():
        k["gate_up"](xq, x_words, *gate_up, order, tile_expert, h)

    def run_swiglu():
        k["swiglu"](h, hq, h_sf, order, layout["layout_weights"], tile_expert)

    def run_down():
        k["down"](hq, h_sf.view(torch.uint32), *down, order, tile_expert, y)

    def run_combine():
        arm.dispatch.combine(y, layout, out)

    for f in (run_dispatch, run_gate_up, run_swiglu, run_down, run_combine):
        f()
    torch.cuda.synchronize()
    used = int((tile_expert >= 0).sum())
    times = {
        name: timed(f)
        for name, f in (
            ("dispatch", run_dispatch),
            ("gate_up", run_gate_up),
            ("swiglu", run_swiglu),
            ("down", run_down),
            ("combine", run_combine),
        )
    }
    times["total"] = sum(times.values())
    result = {"tiles_launched": tile_expert.shape[0], "tiles_used": used, "us": times}
    return result, out.clone()


def fit(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )
    alpha = my - beta * mx
    worst = max(abs(y - alpha - beta * x) for x, y in zip(xs, ys))
    return {"alpha_us": alpha, "beta_us_per_token": beta, "worst_residual_us": worst}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="also write the record here (stdout always has it)")
    args = parser.parse_args()
    results, mismatches = {}, []
    for shape, intermediate in SHAPES.items():
        generator = torch.Generator(device="cuda").manual_seed(intermediate)
        weights = routed_weights(intermediate, generator)
        arms = {bm: Arm(intermediate, bm) for bm in BLOCK_MS}
        record = {
            "weight_bytes": weight_bytes(weights),
            "stream_us": stream_us(weights),
            "tokens": {},
        }
        for tokens in TOKENS:
            g = torch.Generator(device="cuda").manual_seed(tokens)
            scores = torch.rand((tokens, EXPERTS), generator=g, device="cuda")
            ids = scores.topk(TOPK, dim=-1).indices.to(torch.int32).contiguous()
            route_weights = torch.rand((tokens, TOPK), generator=g, device="cuda") + 0.1
            x = torch.randn((tokens, HIDDEN), generator=g, device="cuda") * 0.5
            xq, x_sf = tile_kernels.quant.per_token_cast(
                x.bfloat16(), "e4m3", 32, round_sf=True, use_packed_ue8m0=True
            )
            x_words = x_sf.contiguous().view(torch.uint32)
            entry, outputs = {}, {}
            for bm, arm in arms.items():
                entry[f"block_M{bm}"], outputs[bm] = case(
                    arm, weights, intermediate, ids, route_weights, xq, x_words
                )
            same = torch.equal(
                outputs[BLOCK_MS[0]].view(torch.int16),
                outputs[BLOCK_MS[1]].view(torch.int16),
            )
            entry["outputs_identical"] = same
            if not same:
                mismatches.append(f"{shape} T={tokens}")
            record["tokens"][tokens] = entry
            print(shape, tokens, json.dumps(entry), flush=True)
            del scores, ids, route_weights, x, xq, x_words, outputs
            torch.cuda.empty_cache()
        record["fit"] = {}
        for bm in BLOCK_MS:
            key = f"block_M{bm}"
            per_kernel = {}
            for name in ("gate_up", "swiglu", "down", "combine", "dispatch", "total"):
                points = [
                    (t, record["tokens"][t][key]["us"][name])
                    for t in TOKENS
                    if t >= 2048
                ]
                per_kernel[name] = fit(points)
            alpha = per_kernel["total"]["alpha_us"]
            per_kernel["chunk_fixed_ms"] = FULL_ROW_LAYERS * alpha / 1000
            per_kernel["alpha_over_stream"] = alpha / record["stream_us"]
            record["fit"][key] = per_kernel
        results[shape] = record
        print(shape, json.dumps(record["fit"], indent=1), flush=True)
        del weights, arms
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=1)
    print("RECORD " + json.dumps(results))
    print(json.dumps({s: r["fit"] for s, r in results.items()}, indent=1))
    if mismatches:
        raise SystemExit(f"block_M 64 outputs differ from block_M 32: {mismatches}")


if __name__ == "__main__":
    main()
