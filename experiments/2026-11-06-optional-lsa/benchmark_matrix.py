"""Standalone single-GPU benchmark & recall harness comparing Default-Off vs Native and Opt-In BF16 LSA vs Native across 32K-1M."""
import argparse
import gc
import hashlib
import json
import os
import types
from pathlib import Path

import torch

from benchmark_runtime import graph_of, sample
from validate_retention import recall
from vllm.models.deepseek_v4_1.tilelang.block_shortlist import STATE_BYTES, summarize_blocks
from vllm.models.deepseek_v4_1.tilelang.indexer import CANDIDATE_BLOCKS, IndexerV41Plan
from vllm.models.deepseek_v4_1.tilelang.query_reuse import QueryReusePlan


def measure_plan(plan, q, s, w, pool, table, lengths, active, source):
    rows = len(q)
    out = torch.empty(rows, 512, dtype=torch.int32, device="cuda")
    blocks = torch.empty(rows, CANDIDATE_BLOCKS, dtype=torch.int32, device="cuda") if source else None
    counts = torch.empty(rows, dtype=torch.int32, device="cuda") if source else None
    scratch = [
        torch.empty(shape, dtype=dtype, device="cuda")
        for shape, dtype in plan.scratch_specs(rows)
    ]
    kwargs = {"out": out, "scratch": scratch}
    if source:
        kwargs["candidate_out"] = blocks
        kwargs["candidate_out_lengths"] = counts
    graph = graph_of(lambda: plan.run(q, s, w, pool, table, lengths, active, **kwargs))
    stats = sample(graph)
    stats["median_us"] = stats["median_ms"] * 1000.0
    del graph
    return stats, out, blocks, counts


def run_actual_captures(inputs: Path, replay_receipt: Path):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    replay = json.loads(replay_receipt.read_text())
    assert replay["passed"]
    rows_out = {}
    for path in sorted(inputs.glob("*-rank-0-*.pt")):
        assert sha(path) == replay["files"][path.name]["sha256"]
        data = torch.load(path, map_location="cpu", weights_only=True)
        page, width, rows = data["page_size"], data["width"], len(data["q"])
        source = bool(data["candidate_blocks"])
        native_pool = data["pool"].cuda()
        lsa_pool = torch.zeros(
            len(native_pool), page * STATE_BYTES + 128, dtype=torch.uint8, device="cuda"
        )[:, : page * STATE_BYTES]
        lsa_pool[:, : page * 68].copy_(native_pool)
        complete = [
            i for i, n in enumerate(data["layout"]["written_prefixes"]) if n == page
        ]
        slots = (
            torch.tensor(complete, dtype=torch.int64, device="cuda") * page + page - 1
        )
        sum_fn = summarize_blocks(page)
        sum_stats = sample(
            graph_of(lambda: sum_fn(lsa_pool, lsa_pool.view(torch.bfloat16), slots))
        )
        sum_stats["median_us"] = sum_stats["median_ms"] * 1000.0

        q, s, w = [data[k].cuda() for k in ("q", "scales", "weights")]
        table = torch.zeros(rows, width // page, dtype=torch.int32, device="cuda")
        copied = min(table.shape[1], data["table"].shape[1])
        table[:, :copied].copy_(data["table"][:, :copied])
        lengths, active = data["lengths"].cuda(), data["active"].cuda()

        native_plan = IndexerV41Plan(width, page, candidate_blocks=source)
        default_off_plan = IndexerV41Plan(width, page, candidate_blocks=source)
        budget = 1024 if source else 64
        lsa_plan = QueryReusePlan(
            width, page, candidate_blocks=source, budget=budget, row_cap=256
        )

        nat_stats, nat_out, nat_blocks, nat_counts = measure_plan(
            native_plan, q, s, w, native_pool, table, lengths, active, source
        )
        def_stats, def_out, def_blocks, def_counts = measure_plan(
            default_off_plan, q, s, w, native_pool, table, lengths, active, source
        )
        torch.testing.assert_close(def_out, nat_out, atol=0, rtol=0)
        if source:
            torch.testing.assert_close(def_blocks, nat_blocks, atol=0, rtol=0)
            torch.testing.assert_close(def_counts, nat_counts, atol=0, rtol=0)

        lsa_stats, lsa_out, lsa_blocks, lsa_counts = measure_plan(
            lsa_plan, q, s, w, lsa_pool, table, lengths, active, source
        )
        entry = {
            "layer": data["capture_context"]["layer_id"],
            "case_id": data["case_id"],
            "source": source,
            "rows": rows,
            "max_length": int(lengths.max().item()),
            "default_off_bit_identical": True,
            "native_us": nat_stats,
            "default_off_us": def_stats,
            "lsa_us": lsa_stats,
            "summary_all_pages_us": sum_stats,
            "lsa_with_amortized_summary_speedup": nat_stats["median_us"]
            / lsa_stats["median_us"],
            "top512_recall": recall(nat_out, lsa_out),
        }
        if source:
            entry["source_block_recall"] = recall(nat_blocks, lsa_blocks)
        rows_out[path.name] = entry
        print("ACTUAL_CASE " + json.dumps({"file": path.name, "speedup": entry["lsa_with_amortized_summary_speedup"]}), flush=True)
    return rows_out


def run_synthetic_length_matrix(lengths_tokens=(32768, 65536, 131072, 262144, 524288, 1048576)):
    matrix = {}
    g = torch.Generator(device="cuda").manual_seed(20261106)
    for tokens in lengths_tokens:
        entry = {"tokens": tokens, "dense_layer": {}, "source_layer": {}, "combined_4layer": {}}
        for source in (False, True):
            ratio = 1 if source else 2
            score_width = tokens // ratio
            page = 256 if source else 128
            if not source and score_width <= 16384:
                plan_width = 16384
            else:
                plan_width = 1048576 if source else 524288
            num_pages = score_width // page

            # Allocate compact physical pool of num_pages pages (reused cyclically if needed, or exact num_pages)
            lsa_pool = torch.empty(
                num_pages, page * STATE_BYTES, dtype=torch.uint8, device="cuda"
            )
            lsa_pool[:, : page * 64] = torch.randint(
                0, 256, (num_pages, page * 64), dtype=torch.uint8, device="cuda", generator=g
            )
            lsa_pool[:, page * 64 : page * 68] = torch.randint(
                123, 129, (num_pages, page * 4), dtype=torch.uint8, device="cuda", generator=g
            )
            native_pool = lsa_pool[:, : page * 68].contiguous()
            slots = (
                torch.arange(num_pages, dtype=torch.int64, device="cuda") * page
                + page
                - 1
            )
            summarize_blocks(page)(lsa_pool, lsa_pool.view(torch.bfloat16), slots)

            # Determine which plan opt-in LSA selects at this length
            native_plan = IndexerV41Plan(plan_width, page, candidate_blocks=source)
            if not source:
                lsa_prefill_plan = QueryReusePlan(
                    plan_width, page, candidate_blocks=False, budget=64, row_cap=256
                )
                prefill_route = f"QueryReusePlan(width={plan_width}, budget=64)"
            else:
                if score_width <= 33023:
                    lsa_prefill_plan = native_plan
                    prefill_route = f"native_index(width={plan_width})"
                elif score_width <= 65791:
                    lsa_prefill_plan = QueryReusePlan(
                        plan_width, page, candidate_blocks=True, budget=512, row_cap=256
                    )
                    prefill_route = f"source_short(width={plan_width}, budget=512, DirectCoarseTopK)"
                else:
                    lsa_prefill_plan = QueryReusePlan(
                        plan_width, page, candidate_blocks=True, budget=1024, row_cap=256
                    )
                    prefill_route = f"QueryReusePlan(width={plan_width}, budget=1024, RawWideCoarseTopK)"

            lsa_decode_plan = QueryReusePlan(
                1048576 if source else 524288,
                page,
                candidate_blocks=source,
                budget=1024 if source else 64,
                row_cap=256,
            )
            native_decode_plan = (
                native_plan
                if plan_width == (1048576 if source else 524288)
                else IndexerV41Plan(1048576 if source else 524288, page, candidate_blocks=source)
            )

            layer_res = {"score_width": score_width, "prefill_route": prefill_route, "modes": {}}
            for mode_name, rows, n_plan, l_plan, width_for_table in [
                ("prefill_r64", 64, native_plan, lsa_prefill_plan, plan_width),
                ("decode_r1", 1, native_decode_plan, native_decode_plan, 1048576 if source else 524288),
                ("decode_r8", 8, native_decode_plan, lsa_decode_plan, 1048576 if source else 524288),
                ("decode_r24", 24, native_decode_plan, lsa_decode_plan, 1048576 if source else 524288),
                ("decode_r96", 96, native_decode_plan, lsa_decode_plan, 1048576 if source else 524288),
            ]:
                q = torch.randint(
                    0, 256, (rows, 32, 64), dtype=torch.uint8, device="cuda", generator=g
                )
                s = torch.randint(
                    123, 129, (rows, 32, 4), dtype=torch.uint8, device="cuda", generator=g
                )
                w = (torch.randn(rows, 32, device="cuda", generator=g) / 64).bfloat16()
                table = torch.zeros(
                    rows, width_for_table // page, dtype=torch.int32, device="cuda"
                )
                table[:, :num_pages] = torch.arange(
                    num_pages, dtype=torch.int32, device="cuda"
                ).unsqueeze(0)
                lengths = torch.full(
                    (rows,), score_width, dtype=torch.int32, device="cuda"
                )
                active = torch.tensor(
                    [width_for_table], dtype=torch.int32, device="cuda"
                )

                n_stats, n_out, _, _ = measure_plan(
                    n_plan, q, s, w, native_pool, table, lengths, active, source
                )
                use_pool = native_pool if l_plan is n_plan else lsa_pool
                l_stats, l_out, _, _ = measure_plan(
                    l_plan, q, s, w, use_pool, table, lengths, active, source
                )
                if l_plan is n_plan:
                    torch.testing.assert_close(l_out, n_out, atol=0, rtol=0)
                layer_res["modes"][mode_name] = {
                    "rows": rows,
                    "native_median_us": n_stats["median_us"],
                    "lsa_median_us": l_stats["median_us"],
                    "speedup": n_stats["median_us"] / l_stats["median_us"],
                }
            del lsa_pool, native_pool
            gc.collect()
            torch.cuda.empty_cache()
            if source:
                entry["source_layer"] = layer_res
            else:
                entry["dense_layer"] = layer_res

        for mode_name in ("prefill_r64", "decode_r1", "decode_r8", "decode_r24", "decode_r96"):
            nat_total = (
                3.0 * entry["dense_layer"]["modes"][mode_name]["native_median_us"]
                + 1.0 * entry["source_layer"]["modes"][mode_name]["native_median_us"]
            )
            lsa_total = (
                3.0 * entry["dense_layer"]["modes"][mode_name]["lsa_median_us"]
                + 1.0 * entry["source_layer"]["modes"][mode_name]["lsa_median_us"]
            )
            entry["combined_4layer"][mode_name] = {
                "native_4layer_us": nat_total,
                "lsa_4layer_us": lsa_total,
                "speedup": nat_total / lsa_total,
            }
        matrix[str(tokens)] = entry
        print(
            "MATRIX_LENGTH "
            + json.dumps(
                {
                    "tokens": tokens,
                    "prefill_r64_speedup": entry["combined_4layer"]["prefill_r64"]["speedup"],
                    "decode_r8_speedup": entry["combined_4layer"]["decode_r8"]["speedup"],
                    "decode_r24_speedup": entry["combined_4layer"]["decode_r24"]["speedup"],
                }
            ),
            flush=True,
        )
    return matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--replay-receipt", type=Path, required=True)
    p.add_argument("--boot", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.cuda.reset_peak_memory_stats()
    actual_res = run_actual_captures(args.inputs, args.replay_receipt)
    matrix_res = run_synthetic_length_matrix()
    report = {
        "passed": True,
        "boot": args.boot,
        "state_bytes": STATE_BYTES,
        "downward_quantization": False,
        "actual_captures": actual_res,
        "length_matrix_32k_to_1m": matrix_res,
        "memory": {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT " + json.dumps({"passed": True, "boot": args.boot}), flush=True)


if __name__ == "__main__":
    main()
