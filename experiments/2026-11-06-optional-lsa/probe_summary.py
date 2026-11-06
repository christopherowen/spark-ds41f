"""Out-of-band exact BF16 block8 summary checks and maintenance timings."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from benchmark_runtime import graph_of, sample
from metadata import decode
from vllm.models.deepseek_v4_1.tilelang.block_shortlist import (
    STATE_BYTES,
    summarize_blocks,
    summarize_blocks_reference,
)
from vllm.models.deepseek_v4_1.tilelang.indexer import quantize_mxfp4


def equal(pool, slots, page):
    expected = pool.clone()
    actual = pool.clone()
    summarize_blocks_reference(page)(expected, expected.view(torch.bfloat16), slots)
    summarize_blocks(page)(actual, actual.view(torch.bfloat16), slots)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    return actual


def oracle_check(pool, page, physical):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--replay-receipt", type=Path, required=True)
    p.add_argument("--timing", action="store_true")
    args = p.parse_args()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    replay = json.loads(args.replay_receipt.read_text())
    assert replay["passed"]
    result = {
        "passed": False,
        "state_bytes": STATE_BYTES,
        "downward_quantization": False,
        "request_performance_measured": False,
        "model_quality_tested": False,
        "timing": args.timing,
        "actual": [],
        "synthetic": [],
    }
    torch.cuda.reset_peak_memory_stats()
    for path in sorted(args.inputs.glob("validation-a-rank-0-*.pt")):
        assert sha(path) == replay["files"][path.name]["sha256"]
        data = torch.load(path, map_location="cpu", weights_only=True)
        page = data["page_size"]
        pool = torch.empty(
            len(data["pool"]), page * STATE_BYTES + 128, dtype=torch.uint8, device="cuda"
        )[:, : page * STATE_BYTES]
        pool.fill_(173)
        pool[:, : page * 68].copy_(data["pool"])
        complete = [
            i for i, n in enumerate(data["layout"]["written_prefixes"]) if n == page
        ]
        slots = torch.tensor(complete, dtype=torch.int64, device="cuda") * page + page - 1
        actual = equal(pool, slots, page)
        for physical in complete[:16]:
            oracle_check(actual, page, physical)
        row = {
            "file": path.name,
            "sha256": sha(path),
            "page": page,
            "complete_pages": len(complete),
            "all_pool_bytes_equal": True,
            "bf16_oracle_checked": True,
        }
        if args.timing:
            single = slots[len(slots) // 2 : len(slots) // 2 + 1]
            skipped = single - 1
            row["single_complete_page_cooperative"] = sample(
                graph_of(lambda: summarize_blocks(page)(pool, pool.view(torch.bfloat16), single))
            )
            row["single_complete_page_serial128"] = sample(
                graph_of(lambda: summarize_blocks_reference(page)(pool, pool.view(torch.bfloat16), single))
            )
            row["all_complete_pages_cooperative"] = sample(
                graph_of(lambda: summarize_blocks(page)(pool, pool.view(torch.bfloat16), slots))
            )
            row["all_complete_pages_serial128"] = sample(
                graph_of(lambda: summarize_blocks_reference(page)(pool, pool.view(torch.bfloat16), slots))
            )
            row["skip_only"] = sample(
                graph_of(lambda: summarize_blocks(page)(pool, pool.view(torch.bfloat16), skipped))
            )
        result["actual"].append(row)
    assert len(result["actual"]) == 4

    for page in (128, 256):
        pages = 4
        pool = torch.empty(
            pages, page * STATE_BYTES + 128, dtype=torch.uint8, device="cuda"
        )[:, : page * STATE_BYTES]
        pool.fill_(61)
        g = torch.Generator(device="cuda").manual_seed(page)
        x = torch.randn(pages * page, 128, device="cuda", generator=g, dtype=torch.bfloat16)
        exponent = torch.randint(
            -100, 101, (pages * page, 4, 1), device="cuda", generator=g, dtype=torch.int32
        )
        x = (x.view(-1, 4, 32) * torch.exp2(exponent.float())).bfloat16().view(-1, 128)
        x[0:8] = 0
        x[8:16] = 1
        x[16:24] = -1
        q = torch.empty(pages * page, 64, device="cuda", dtype=torch.uint8)
        sf = torch.empty(pages * page, 4, device="cuda", dtype=torch.uint8)
        quantize_mxfp4(x, q, sf)
        for i in range(pages):
            pool[i, : page * 64].copy_(q[i * page : (i + 1) * page].reshape(-1))
            pool[i, page * 64 : page * 68].copy_(sf[i * page : (i + 1) * page].reshape(-1))
        slots = torch.tensor(
            [-1, 0, page - 2, page - 1, 2 * page - 1, 3 * page - 1, 4 * page - 2, 5 * page - 1],
            device="cuda",
            dtype=torch.int64,
        )
        equal(pool, slots, page)
        pool[0, : page * 68].copy_(pool[2, : page * 68])
        equal(pool, slots, page)
        strict = torch.empty(2, page * STATE_BYTES, dtype=torch.uint8, device="cuda")
        strict[0, : page * 68].copy_(pool[0, : page * 68])
        strict[:, page * 68 :].fill_(91)
        strict_slots = torch.tensor(
            [-1, page - 2, page - 1, 2 * page - 2, 3 * page - 1],
            device="cuda",
            dtype=torch.int64,
        )
        summarize_blocks(page)(strict, strict.view(torch.bfloat16), strict_slots)
        expected = pool[0:1].clone()
        zero_slot = torch.tensor([page - 1], device="cuda", dtype=torch.int64)
        summarize_blocks_reference(page)(expected, expected.view(torch.bfloat16), zero_slot)
        torch.testing.assert_close(
            strict[0, page * 68 : page * STATE_BYTES],
            expected[0, page * 68 : page * STATE_BYTES],
            atol=0,
            rtol=0,
        )
        assert bool((strict[1, page * 68 :] == 91).all())
        result["synthetic"].append(
            {
                "page": page,
                "scales_exponent_range": [-100, 100],
                "zero_tie_sign_vectors": True,
                "skip_and_recycle_bytes_equal": True,
            }
        )
    result["memory"] = {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    result["passed"] = True
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
