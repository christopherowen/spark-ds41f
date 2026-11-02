#!/usr/bin/env python3
"""GB10 microbenchmark of the packed vocabulary head's 64-row token tile (rows
33..64; K = 5,120, DS4.1-Flash's own output head rows, packed as serving packs
them) at candidate-37's ``TILE_CONFIGS[64]`` against ``step_groups=2``:

- ``base``: ``block_N=32, threads=256, step_groups=1, num_stages=4``;
- ``step2``: ``block_N=32, threads=256, step_groups=2, num_stages=3, act_stages=1``.

Both kernels the tile configures: ``packed_vocab_projection`` (FP32 logits: the
draft position logits, and the target's when it projects them) and
``packed_vocab_greedy_partials`` + ``finish_vocab_greedy_pairs`` (greedy
verification's shard pairs). Each call is CUDA-graph captured and replayed back to
back (a shard's packed weights, 248-331 MB, never fit in L2); configs alternate
within each round and swap order every round.
"""

import argparse
import json
import statistics
from pathlib import Path

import tilelang.language as T
import torch

from vllm.models.deepseek_v4_1.tilelang import vocab

SHAPES = {"tp4": 32320, "tp3": 43136}
ROWS = (33, 36, 40, 42, 45, 48, 54, 60, 64)
CONFIGS = {
    "base": dict(block_N=32, threads=256, step_groups=1, num_stages=4),
    "step2": dict(block_N=32, threads=256, step_groups=2, num_stages=3, act_stages=1),
}


def load_bf16(path: Path) -> torch.Tensor:
    meta = json.loads(path.with_suffix(".json").read_text())
    data = bytearray(path.with_suffix(".bin").read_bytes())
    return torch.frombuffer(data, dtype=torch.bfloat16).view(*meta["shape"])


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


def time_graph(graph, iters) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for k in range(iters):
        starts[k].record()
        graph.replay()
        ends[k].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("/bench/weights"))
    parser.add_argument("--shapes", default="tp4,tp3")
    parser.add_argument("--rows", default=",".join(map(str, ROWS)))
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    head_rows = load_bf16(args.weights / "lm_head_rows")
    k = int(head_rows.shape[1])
    rows_list = [int(r) for r in args.rows.split(",")]
    results = {
        "device": torch.cuda.get_device_name(),
        "k": k,
        "configs": CONFIGS,
        "smem_bytes": {
            name: vocab.shared_memory_bytes(
                64,
                c["block_N"],
                c["num_stages"],
                c["step_groups"],
                act_stages=c.get("act_stages"),
            )
            for name, c in CONFIGS.items()
        },
        "iters": args.iters,
        "rounds": args.rounds,
        "cases": [],
    }
    generator = torch.Generator(device="cuda").manual_seed(64)
    for shape in args.shapes.split(","):
        n = SHAPES[shape]
        packed = vocab.pack_bf16_weight(head_rows[:n].cuda().contiguous())
        weights = (
            packed.values,
            packed.exception_rows,
            packed.exception_entries,
            packed.row_offsets,
        )
        kernels = {}
        for name, config in CONFIGS.items():
            tiles = (n + config["block_N"] - 1) // config["block_N"]
            kernels[name] = (
                vocab.packed_vocab_projection(
                    n, k, packed.exponent_base, block_M=64, out_dtype=T.float32, **config
                ),
                vocab.packed_vocab_greedy_partials(
                    n, k, packed.exponent_base, block_M=64, out_dtype=T.float32, **config
                ),
                vocab.finish_vocab_greedy_pairs(tiles),
                tiles,
            )
        for rows in rows_list:
            x = (torch.randn(rows, k, generator=generator, device="cuda") * 0.5).bfloat16()
            buffers = {}
            for name, (_, _, _, tiles) in kernels.items():
                buffers[name] = (
                    torch.empty(rows, n, device="cuda"),
                    torch.empty(rows, tiles, device="cuda"),
                    torch.empty(rows, tiles, dtype=torch.int32, device="cuda"),
                    torch.full((rows, 4), -7.0, device="cuda"),
                )

            def project(name):
                kernels[name][0](x, *weights, buffers[name][0])

            def pairs(name):
                _, partials, finish, _ = kernels[name]
                _, values, indices, out = buffers[name]
                partials(x, *weights, values, indices, n)
                finish(values, indices, out, 0)

            for name in CONFIGS:
                project(name)
                pairs(name)
            torch.cuda.synchronize()
            case = {
                "shape": shape,
                "rows": rows,
                "exceptions": int(packed.exceptions),
                "projection_bit_exact": torch.equal(
                    buffers["base"][0].view(torch.int32),
                    buffers["step2"][0].view(torch.int32),
                ),
                "pairs_bit_exact": torch.equal(
                    buffers["base"][3].view(torch.int32),
                    buffers["step2"][3].view(torch.int32),
                ),
                "rounds": [],
            }
            graphs = {
                (kind, name): capture(
                    (lambda c=name: project(c)) if kind == "projection" else (lambda c=name: pairs(c))
                )
                for kind in ("projection", "pairs")
                for name in CONFIGS
            }
            names = tuple(CONFIGS)
            for r in range(args.rounds):
                order = names if r % 2 == 0 else names[::-1]
                record = {}
                for kind in ("projection", "pairs"):
                    for name in order:
                        record[f"{kind}/{name}"] = statistics.median(
                            time_graph(graphs[kind, name], args.iters)
                        )
                case["rounds"].append(record)
            line = [
                f"{shape} rows={rows:2d} exact={case['projection_bit_exact']}/{case['pairs_bit_exact']}"
            ]
            for kind in ("projection", "pairs"):
                for name in CONFIGS:
                    medians = [rnd[f"{kind}/{name}"] for rnd in case["rounds"]]
                    case[f"{kind}/{name}"] = {
                        "median_us": statistics.median(medians),
                        "min_us": min(medians),
                        "max_us": max(medians),
                    }
                a = case[f"{kind}/base"]["median_us"]
                b = case[f"{kind}/step2"]["median_us"]
                line.append(f"{kind} {a:7.1f}->{b:7.1f} ({b - a:+6.1f})")
            print(" | ".join(line), flush=True)
            results["cases"].append(case)
            del graphs
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
