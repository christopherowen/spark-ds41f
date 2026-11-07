"""Untimed cache, graph, native-score and actual-input runtime qualification for BF16 block8 QueryReusePlan."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from metadata import decode
from score import summary_scores
from validate_retention import recall
from vllm.models.deepseek_v4_1.tilelang.block_shortlist import STATE_BYTES, summarize_blocks
from vllm.models.deepseek_v4_1.tilelang.indexer import CANDIDATE_BLOCKS, IndexerTopK, IndexerV41Plan
from vllm.models.deepseek_v4_1.tilelang.query_reuse import QueryReusePlan


def means_check(pool, page, physical):
    packed = pool[physical, : page * 64].reshape(page, 64)
    scales = pool[physical, page * 64 : page * 68].reshape(page, 4)
    keys = decode(packed, scales).reshape(page // 8, 8, 128)
    expected = torch.zeros(page // 8, 128, dtype=torch.float32, device="cuda")
    for token in range(8):
        expected.add_(keys[:, token] * 0.125)
    actual = (
        pool[physical, page * 68 : page * STATE_BYTES]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(page // 8, 128)
    )
    torch.testing.assert_close(
        actual.view(torch.int16), expected.bfloat16().view(torch.int16), atol=0, rtol=0
    )
    return list(range(page // 8))


def run_case(
    q,
    scales,
    weights,
    pool,
    table,
    lengths,
    active,
    width,
    page,
    source,
    *,
    budget=None,
    coarse_oracle=False,
):
    rows = len(q)
    args = [q, scales, weights, pool, table, lengths, active]
    native = IndexerV41Plan(width, page, candidate_blocks=source)
    expected_blocks = torch.empty(rows, CANDIDATE_BLOCKS, dtype=torch.int32, device="cuda")
    expected_counts = torch.empty(rows, dtype=torch.int32, device="cuda")
    options = (
        {"candidate_out": expected_blocks, "candidate_out_lengths": expected_counts}
        if source
        else {}
    )
    expected, native_scores = native.run(*args, **options)
    expected = expected.clone()
    resolved_budget = budget or (1024 if source else 64)
    plan = QueryReusePlan(
        width, page, candidate_blocks=source, budget=resolved_budget, row_cap=256
    )
    scratch = [
        torch.empty(shape, dtype=dtype, device="cuda")
        for shape, dtype in plan.scratch_specs(rows)
    ]
    out = torch.empty(rows, 512, dtype=torch.int32, device="cuda")
    blocks = torch.empty(rows, CANDIDATE_BLOCKS, dtype=torch.int32, device="cuda")
    counts = torch.empty(rows, dtype=torch.int32, device="cuda")
    plan.run(
        *args,
        out=out,
        candidate_out=blocks,
        candidate_out_lengths=counts,
        scratch=scratch,
    )
    b = dict(zip(plan._specs(rows), scratch))
    assert rows <= 256
    if coarse_oracle:
        small = page // 8
        cap = int(active.item())
        visible = []
        for row, extent in enumerate(lengths.cpu().tolist()):
            full_pages = min(max(extent, 0), max(cap, 0), width) // page
            if full_pages:
                visible.append(table[row, :full_pages].long())
        physical = (
            torch.unique(torch.cat(visible))
            if visible
            else torch.empty(0, dtype=torch.long, device=pool.device)
        )
        assert bool(((physical >= 0) & (physical < len(pool))).all())
        means = torch.zeros(
            len(pool), small, 128, dtype=torch.bfloat16, device=pool.device
        )
        if len(physical):
            means[physical] = (
                pool[physical, page * 68 : page * STATE_BYTES]
                .contiguous()
                .view(torch.bfloat16)
                .reshape(len(physical), small, 128)
            )
        scores = torch.zeros_like(b["scores"])
        n = torch.empty_like(b["counts"])
        summary_scores(width, page, 8)(
            q, scales, weights, means, table, lengths, active, scores, n
        )
        if source:
            selector = IndexerTopK(width // 8, plan.block_budget)
            selected = torch.empty_like(b["selected"])
            values = torch.empty_like(b["values"])
            work = [
                torch.empty(shape, dtype=dtype, device="cuda")
                for shape, dtype in selector.scratch_specs(rows)
            ]
            selector.run(
                b["scores"], b["counts"], active, selected, values, scratch=work
            )
            torch.testing.assert_close(selected, b["selected"], atol=0, rtol=0)
            torch.testing.assert_close(
                values.view(torch.int16), b["values"].view(torch.int16), atol=0, rtol=0
            )
            for row, extent in enumerate(b["counts"].cpu().tolist()):
                keep = min(extent, plan.block_budget)
                winners = torch.argsort(
                    b["scores"][row, :extent], descending=True, stable=True
                )[:keep]
                winners = torch.sort(winners).values.to(torch.int32)
                torch.testing.assert_close(
                    b["selected"][row, :keep], winners, atol=0, rtol=0
                )
                assert bool((b["selected"][row, keep:] == -1).all())

        for row, extent in enumerate(lengths.cpu().tolist()):
            full = extent // page * (page // 8)
            if full > plan.block_budget:
                assert int(n[row]) == int(b["counts"][row]) == full
                torch.testing.assert_close(
                    scores[row, :full].view(torch.int16),
                    b["scores"][row, :full].view(torch.int16),
                    atol=0,
                    rtol=0,
                )
    for row, (extent, n) in enumerate(
        zip(lengths.cpu().tolist(), b["candidate_lengths"].cpu().tolist())
    ):
        ids = b["candidates"][row, : (n + 7) // 8]
        assert bool((ids >= 0).all()) and bool((ids[1:] > ids[:-1]).all())
        positions = torch.arange(n, device="cuda")
        positions = ids[positions // 8].long() * 8 + positions % 8
        assert bool((positions < extent).all())
        torch.testing.assert_close(
            b["final_0"][row, :n].view(torch.int16),
            native_scores[row, positions].view(torch.int16),
            atol=0,
            rtol=0,
        )
        if extent // page * (page // 8) <= plan.block_budget:
            torch.testing.assert_close(out[row], expected[row], atol=0, rtol=0)
            if source:
                torch.testing.assert_close(
                    blocks[row], expected_blocks[row], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    counts[row], expected_counts[row], atol=0, rtol=0
                )
    result = {
        "budget": resolved_budget,
        "retained_native_score_bits_equal": True,
        "ordered_valid_candidates": True,
        "covered_domain_native_outputs_equal": True,
        "coarse_oracle_checked": coarse_oracle,
        "independent_stable_topk_checked": coarse_oracle and source,
    }
    if int(lengths.min()) >= 512:
        result["native_top512_key_recall"] = recall(expected, out)
        if source and int(lengths.min()) >= 16384:
            result["native_source_block_recall"] = recall(expected_blocks, blocks)
    return result


def actual(inputs, replay_path, out):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    replay = json.loads(replay_path.read_text())
    assert replay["passed"] and replay["actual_captures"]
    paths = sorted(inputs.glob("*-rank-0-*.pt"))
    assert len(paths) == 12
    result = {
        "passed": False,
        "state_bytes": STATE_BYTES,
        "downward_quantization": False,
        "actual_captures": True,
        "performance_measured": False,
        "model_answer_quality_tested": False,
        "replay_sha256": sha(replay_path),
        "files": {},
    }
    for path in paths:
        assert sha(path) == replay["files"][path.name]["sha256"]
        data = torch.load(path, map_location="cpu", weights_only=True)
        page, width, rows = data["page_size"], data["width"], len(data["q"])
        source = bool(data["candidate_blocks"])
        pool = torch.zeros(
            len(data["pool"]), page * STATE_BYTES + 128, dtype=torch.uint8, device="cuda"
        )[:, : page * STATE_BYTES]
        pool[:, : page * 68].copy_(data["pool"])
        table = torch.zeros(rows, width // page, dtype=torch.int32, device="cuda")
        copied = min(table.shape[1], data["table"].shape[1])
        table[:, :copied].copy_(data["table"][:, :copied])
        complete = [
            i for i, n in enumerate(data["layout"]["written_prefixes"]) if n == page
        ]
        slots = torch.tensor(complete, dtype=torch.int64, device="cuda") * page + page - 1
        summarize_blocks(page)(pool, pool.view(torch.bfloat16), slots)
        for physical in complete:
            means_check(pool, page, physical)
        args = [data[k].cuda() for k in ("q", "scales", "weights")]
        args += [
            pool,
            table,
            data["lengths"].cuda(),
            data["active"].cuda(),
            width,
            page,
            source,
        ]
        row = run_case(*args, budget=1024 if source else 64, coarse_oracle=True)
        if source:
            row["source_short_512"] = run_case(*args, budget=512, coarse_oracle=True)
        row.update(
            layer=data["capture_context"]["layer_id"],
            case_id=data["case_id"],
            rows=rows,
            sha256=sha(path),
        )
        result["files"][path.name] = row
        out.write_text(json.dumps(result, indent=2) + "\n")
        print("CASE " + json.dumps({"file": path.name, "receipt": row}), flush=True)
    result["passed"] = True
    return result


def strict():
    cases = []
    for page in (128, 256):
        for source, budget, extents in [
            (False, 64, [5001, 5043]),
            (True, 512, [33025, 33041]),
            (True, 1024, [66049, 66065]),
        ]:
            width = 1048576 if source else 65536
            rows, logical = 2, width // page
            pool = torch.empty(
                rows * logical + 3,
                page * STATE_BYTES + 128,
                dtype=torch.uint8,
                device="cuda",
            )[:, : page * STATE_BYTES]
            table = torch.full((rows, logical), -1, dtype=torch.int32, device="cuda")
            complete = []
            g = torch.Generator(device="cuda").manual_seed(page + int(source) + budget)
            for row, extent in enumerate(extents):
                n = (extent + page - 1) // page
                ids = torch.arange(n, dtype=torch.int32, device="cuda") + 1 + row * logical
                table[row, :n] = ids
                positions = torch.arange(extent, device="cuda")
                physical, offset = ids[positions // page].long(), positions % page
                pool[
                    physical[:, None],
                    offset[:, None] * 64 + torch.arange(64, device="cuda"),
                ] = torch.randint(
                    0, 256, (extent, 64), dtype=torch.uint8, device="cuda", generator=g
                )
                pool[
                    physical[:, None],
                    page * 64 + offset[:, None] * 4 + torch.arange(4, device="cuda"),
                ] = torch.randint(
                    122, 130, (extent, 4), dtype=torch.uint8, device="cuda", generator=g
                )
                complete += ids[: extent // page].cpu().tolist()
            slots = (
                torch.tensor(complete, dtype=torch.int64, device="cuda") * page
                + page
                - 1
            )
            summarize_blocks(page)(pool, pool.view(torch.bfloat16), slots)
            q = torch.randint(
                0, 256, (rows, 32, 64), dtype=torch.uint8, device="cuda", generator=g
            )
            scales = torch.randint(
                122, 130, (rows, 32, 4), dtype=torch.uint8, device="cuda", generator=g
            )
            weights = (
                torch.randn(rows, 32, device="cuda", generator=g) / 64
            ).bfloat16()
            active = torch.tensor([width], dtype=torch.int32, device="cuda")
            for limit in (0, 37, page - 1, page, page + 1, max(extents)):
                print(
                    "CHECK "
                    + json.dumps(
                        {"page": page, "source": source, "budget": budget, "limit": limit}
                    ),
                    flush=True,
                )
                lengths = torch.tensor(
                    [min(n, limit) for n in extents], dtype=torch.int32, device="cuda"
                )
                run_case(
                    q,
                    scales,
                    weights,
                    pool,
                    table,
                    lengths,
                    active,
                    width,
                    page,
                    source,
                    budget=budget,
                    coarse_oracle=source,
                )
            cases.append(
                {
                    "page": page,
                    "source": source,
                    "budget": budget,
                    "unwritten_pool_and_table_suffix": True,
                    "empty_partial_and_pruned_rows": True,
                }
            )
            print("CASE " + json.dumps(cases[-1]), flush=True)
    return {
        "passed": True,
        "strict_fixture": True,
        "cases": cases,
        "performance_measured": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["strict", "actual"], required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--replay-receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    torch.cuda.reset_peak_memory_stats()
    if args.mode == "strict":
        result = strict()
    else:
        result = actual(args.inputs, args.replay_receipt, args.out)
    result["memory_observations"] = {
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)
