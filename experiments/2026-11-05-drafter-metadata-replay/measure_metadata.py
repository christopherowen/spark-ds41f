"""Small GPU screen of drafter metadata placement; never a serving benchmark.

All timed arms use the same candidate source and integer kernel. The eager arm
is an operation-level control, not a remeasurement of a released serving build.
The query consumer is one integer add, not the model: results bound overhead and
must not be reported as token throughput or end-to-end inference improvements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace

SOURCE_HEAD = "a2b0ea500dc5022e23c4d6fc667254a806c984dc"
SOURCE_TREE = "e11945731049bb810c69644d91b0740fb3ca3680"
IMAGE = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
NAMES = (
    "starts",
    "request_positions",
    "counts",
    "positions",
    "reqs",
    "slots",
    "lengths",
)
ARMS = (
    "eager_full",
    "direct_kernel",
    "separate_graph",
    "query_graph",
    "consumer_floor",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--boot", type=int, choices=(1, 2), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.trials >= 12 and args.iterations >= 32
    plan = {
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "image": IMAGE,
        "boot": args.boot,
        "trials": args.trials,
        "iterations": args.iterations,
        "groups": 1,
        "query_rows_per_request": 5,
        "request_shapes": [[1, 1, 8], [7, 8, 8], [8, 8, 8], [16, 16, 16]],
        "prefix_lengths": [256, 8192, 131072, 1048512],
        "arms": list(ARMS),
        "consumer_floor": "Static precomputed metadata only; invalid as an implementation. Bounds residual refresh cost.",
        "scope": "Metadata dispatch plus one integer consumer kernel; no model weights or serving.",
    }
    if args.describe:
        print(json.dumps(plan, indent=2))
        return
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadataBuilder
    from vllm.models.deepseek_v4_1.tilelang import metadata, prepare
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.model_states.interface import ModelState
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    assert torch.cuda.is_available(), (
        "GPU required; this program must not publish simulated timings"
    )
    torch.set_num_threads(1)
    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    result = dict(
        plan,
        host=platform.node(),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        cases=[],
        status="running",
    )
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def save():
        temp = args.output.with_suffix(".partial.json")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(args.output)

    def summary(values):
        values = np.asarray(values, dtype=np.float64)
        rng = np.random.default_rng(417)
        draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(
            axis=1
        )
        return {
            "mean_us": float(values.mean()),
            "median_us": float(np.median(values)),
            "ci95_mean_us": np.quantile(draws, [0.025, 0.975]).tolist(),
        }

    class State:
        prepare_draft_attn_metadata = ModelState.prepare_draft_attn_metadata

    class Group:
        def __init__(self, builder):
            self.builder = builder
            self.layer_names = ["draft.attn"]

        def get_metadata_builder(self, ubatch_idx=0):
            return self.builder

    def fixture(live, padded, capacity, context):
        width = 5
        tokens = padded * width
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_seqs=capacity, max_num_batched_tokens=128
            ),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
            speculative_config=SimpleNamespace(
                num_speculative_tokens=5,
                num_verification_tokens=24,
                parallel_drafting=False,
            ),
        )
        spec = SlidingWindowSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=128,
        )
        builder = DeepseekV41B12xMetadataBuilder(spec, ["draft.attn"], config, device)
        columns = (context + capacity + width + 127) // 128 + 1
        table = torch.arange(
            1, padded * columns + 1, dtype=torch.int32, device=device
        ).reshape(padded, columns)
        starts = torch.arange(padded + 1, dtype=torch.int32, device=device) * width
        seq = torch.empty(padded, dtype=torch.int32, device=device)
        slots = torch.ones(1, tokens, dtype=torch.int64, device=device)
        output = torch.empty(tokens, dtype=torch.int64, device=device)
        seq_cpu = torch.empty(padded, dtype=torch.int32)
        obj = DFlashSpeculator.__new__(DFlashSpeculator)
        obj.draft_cp_size = 1
        obj.block_tables = SimpleNamespace(
            cp_size=1, input_block_tables=[table], slot_mappings=slots
        )
        obj.input_buffers = SimpleNamespace(query_start_loc=starts, seq_lens=seq)
        obj.attn_groups = [[Group(builder)]]
        obj.model_state = State()
        obj.kv_cache_config = KVCacheConfig(
            0, [], [KVCacheGroupSpec(["draft.attn"], spec)]
        )
        obj.draft_attn_layer_names = {"draft.attn"}
        obj.idx_mapping = torch.arange(padded, dtype=torch.int32, device=device)
        obj.arange_np = np.arange(capacity + 1, dtype=np.int32)
        obj._query_start_loc_cpu = torch.empty(capacity + 1, dtype=torch.int32)
        obj._draft_seq_lens_cpu_upper_bound = torch.empty(capacity, dtype=torch.int32)
        obj.draft_is_prefilling = torch.zeros(capacity, dtype=torch.bool)
        obj.max_model_len = 1048576
        desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, tokens, padded, width)
        assert not obj._compute_full_cg_needs_attn_metadata_build()
        state = {"live": live}

        def change(n, shift):
            state["live"] = n
            seq_cpu.copy_(torch.arange(padded, dtype=torch.int32) + context + shift)
            seq_cpu[n:] = 0
            seq.copy_(seq_cpu + width)
            seq[n:] = 0
            table.copy_(
                (
                    torch.arange(
                        padded * columns, device=device, dtype=torch.int32
                    ).reshape(padded, columns)
                    + 1
                    + shift * padded * columns
                )
            )
            table[n:] = 0
            slots.fill_(1)
            slots[:, n * width :] = -1

        def eager():
            # Exactly the host work omitted by the FULL replay fast path.
            obj.draft_max_seq_len = min(
                int(seq_cpu[: state["live"]].max()) + width, obj.max_model_len
            )
            return obj._build_uniform_attn_metadata(
                num_reqs=state["live"],
                batch_desc=desc,
                num_query_per_req=width,
                seq_lens_cpu_upper_bound=seq_cpu,
                step=width,
                causal=False,
            )

        def refresh():
            builder.prepare_cudagraph_replay(
                starts, seq, table, slots[0], padded, tokens
            )

        def consume():
            torch.add(builder.positions[:tokens], builder.slots[:tokens], out=output)

        change(live, 0)
        eager()
        consume()
        torch.accelerator.synchronize()
        graph_consumer, graph_metadata, graph_combined = (
            torch.cuda.CUDAGraph() for _ in range(3)
        )
        with torch.cuda.graph(graph_consumer, stream=stream):
            consume()
        with torch.cuda.graph(graph_metadata, stream=stream):
            refresh()
        with torch.cuda.graph(graph_combined, stream=stream):
            refresh()
            consume()

        def full():
            eager()
            graph_consumer.replay()

        def direct():
            refresh()
            graph_consumer.replay()

        def separate():
            graph_metadata.replay()
            graph_consumer.replay()

        methods = dict(
            zip(
                ARMS,
                (full, direct, separate, graph_combined.replay, graph_consumer.replay),
            )
        )
        # Poison outputs between arms: stale/no-op refresh cannot pass equality.
        checks = []
        for n, shift in ((live, 0), (max(1, live - 1), 1), (live, 3)):
            change(n, shift)
            eager()
            consume()
            expected = {name: getattr(builder, name).clone() for name in NAMES}
            expected_output = output.clone()
            torch.accelerator.synchronize()
            for arm in ARMS[:-1]:
                for name in NAMES:
                    getattr(builder, name).fill_(-12345)
                output.fill_(-12345)
                methods[arm]()
                torch.accelerator.synchronize()
                for name in NAMES:
                    assert torch.equal(getattr(builder, name), expected[name]), (
                        arm,
                        name,
                        n,
                        shift,
                    )
                assert torch.equal(output, expected_output), (arm, "consumer", n, shift)
                checks.append({"arm": arm, "live": n, "shift": shift, "exact": True})
        change(live, 0)
        eager()
        consume()
        return methods, checks

    # Prepare all specialization keys once, before freezing and graph capture.
    for capacity in (8, 16):
        metadata.builder_kernel(capacity, 128, 1, False)
    prepare.freeze("drafter metadata placement microbenchmark")
    torch.accelerator.memory.reset_peak_memory_stats()
    result["driver_memory_before"] = list(torch.accelerator.memory.get_memory_info())
    save()
    for live, padded, capacity in plan["request_shapes"]:
        for context in plan["prefix_lengths"]:
            methods, checks = fixture(live, padded, capacity, context)
            for method in methods.values():
                for _ in range(32):
                    method()
            torch.accelerator.synchronize()
            rows = []
            for trial in range(args.trials):
                # Balanced cyclic order, reversed every other round.
                order = list(ARMS[trial % 5 :] + ARMS[: trial % 5])
                if trial % 2:
                    order.reverse()
                row = {"trial": trial, "order": order, "arms": {}}
                for arm in order:
                    torch.accelerator.synchronize()
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record()
                    started = time.perf_counter_ns()
                    for _ in range(args.iterations):
                        methods[arm]()
                    enqueued = time.perf_counter_ns()
                    end.record()
                    end.synchronize()
                    completed = time.perf_counter_ns()
                    row["arms"][arm] = {
                        "enqueue_us": (enqueued - started) / args.iterations / 1000,
                        "completion_us": (completed - started) / args.iterations / 1000,
                        "stream_elapsed_us": begin.elapsed_time(end)
                        * 1000
                        / args.iterations,
                    }
                rows.append(row)
            case = {
                "live": live,
                "padded": padded,
                "capacity": capacity,
                "context": context,
                "correctness": checks,
                "trials": rows,
                "summary": {},
            }
            for arm in ARMS:
                case["summary"][arm] = summary(
                    [r["arms"][arm]["completion_us"] for r in rows]
                )
            case["paired_saved_us"] = {
                arm: summary(
                    [
                        r["arms"][arm]["completion_us"]
                        - r["arms"]["query_graph"]["completion_us"]
                        for r in rows
                    ]
                )
                for arm in ARMS
                if arm != "query_graph"
            }
            result["cases"].append(case)
            result["peak_tensor_bytes"] = (
                torch.accelerator.memory.max_memory_allocated()
            )
            result["driver_memory_after"] = list(
                torch.accelerator.memory.get_memory_info()
            )
            save()
            gain = case["paired_saved_us"]["eager_full"]
            print(
                json.dumps(
                    {
                        "live": live,
                        "padded": padded,
                        "context": context,
                        "saved_us": gain["mean_us"],
                        "ci95": gain["ci95_mean_us"],
                    }
                ),
                flush=True,
            )
    result["status"] = "pass"
    result["limitations"] = [
        "Component screen only; no model execution, rank communication or serving workload.",
        "Bootstrap intervals cover paired trials within this process; compare both independent process boots.",
        "CUDA stream elapsed includes host enqueue gaps and is not intrinsic kernel duration.",
        "consumer_floor is a residual-cost bound, not a valid dynamic-metadata implementation.",
    ]
    save()


if __name__ == "__main__":
    main()
