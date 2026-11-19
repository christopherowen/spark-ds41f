#!/usr/bin/env python3
"""Verify all 7 LSA correctness, sanitizer, registration, and CUDA-graph gates."""
import argparse
import json
import time
import tilelang
import torch

from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang.block_shortlist import (
    STATE_BYTES,
    summarize_blocks,
)
from vllm.models.deepseek_v4_1.tilelang.indexer import (
    CANDIDATE_BLOCKS,
    IndexerV41Plan,
    quantize_mxfp4,
    write_index_keys,
)
from vllm.models.deepseek_v4_1.tilelang.query_reuse import QueryReusePlan


def decode_reference_summary(pool: torch.Tensor, page: int) -> torch.Tensor:
    num_pages = pool.shape[0]
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=pool.device,
    )
    raw_bytes = pool[:, : page * 64].view(num_pages, page, 64).to(torch.int32)
    scales = pool[:, page * 64 : page * 68].view(num_pages, page, 4).to(torch.int32)
    codes = torch.stack((raw_bytes & 15, (raw_bytes >> 4) & 15), dim=-1).view(
        num_pages, page, 128
    )
    signs = torch.where((codes & 8) != 0, -1.0, 1.0)
    mags = lut[codes & 7]
    scale_f32 = (
        torch.bitwise_left_shift(scales, 23)
        .view(torch.float32)
        .repeat_interleave(32, dim=-1)
    )
    decoded = (signs * mags * scale_f32).view(num_pages, page // 8, 8, 128)
    acc = decoded[:, :, 0].clone()
    hi = decoded[:, :, 0].clone()
    lo = decoded[:, :, 0].clone()
    for k in range(1, 8):
        val = decoded[:, :, k]
        acc = acc + val
        hi = torch.maximum(hi, val)
        lo = torch.minimum(lo, val)
    return (acc * 0.09375 + (hi + lo) * 0.125).to(torch.bfloat16)


def run_summary_check(timing: bool = False) -> dict:
    assert STATE_BYTES == 100
    results = []
    for page in (128, 256):
        pages = 8
        pool = torch.empty(
            (pages, page * STATE_BYTES + 128), dtype=torch.uint8, device="cuda"
        )[:, : page * STATE_BYTES]
        pool.fill_(173)
        gen = torch.Generator(device="cuda").manual_seed(1000 + page)
        x = torch.randn(
            (pages * page, 128), generator=gen, device="cuda", dtype=torch.bfloat16
        )
        exponent = torch.randint(
            -40, 41, (pages * page, 4, 1), generator=gen, device="cuda", dtype=torch.int32
        )
        x = (x.view(-1, 4, 32) * torch.exp2(exponent.float())).bfloat16().view(-1, 128)
        x[0:8] = 0
        x[8:16] = 1
        x[16:24] = -1
        q = torch.empty((pages * page, 64), device="cuda", dtype=torch.uint8)
        sf = torch.empty((pages * page, 4), device="cuda", dtype=torch.uint8)
        quantize_mxfp4(x, q, sf)
        for i in range(pages):
            pool[i, : page * 64].copy_(q[i * page : (i + 1) * page].reshape(-1))
            pool[i, page * 64 : page * 68].copy_(
                sf[i * page : (i + 1) * page].reshape(-1)
            )

        complete_slots = torch.tensor(
            [page - 1, 2 * page - 1, 4 * page - 1, 6 * page - 1],
            device="cuda",
            dtype=torch.int64,
        )
        mixed_slots = torch.tensor(
            [-1, 0, page - 2, page - 1, 2 * page - 1, 3 * page - 2, 4 * page - 1],
            device="cuda",
            dtype=torch.int64,
        )
        summarize_blocks(page)(pool, pool.view(torch.bfloat16), mixed_slots)
        ref = decode_reference_summary(pool, page)
        summaries = pool.view(torch.bfloat16)[
            :, page * 68 // 2 : page * STATE_BYTES // 2
        ].view(pages, page // 8, 128)
        for p_idx in (0, 1, 3):
            assert torch.equal(summaries[p_idx], ref[p_idx])
        for untouched in (2, 4, 5, 6, 7):
            assert bool((pool[untouched, page * 68 :] == 173).all().item())

        summarize_blocks(page)(pool, pool.view(torch.bfloat16), complete_slots)
        for p_idx in (0, 1, 3, 5):
            assert torch.equal(summaries[p_idx], ref[p_idx])

        entry = {
            "page": page,
            "pages": pages,
            "bit_exact_bf16": True,
            "untouched_sentinel_preserved": True,
        }
        if timing:
            all_slots = (
                torch.arange(pages, device="cuda", dtype=torch.int64) * page + page - 1
            )
            skip_slots = all_slots - 1
            k_fn = summarize_blocks(page)
            for _ in range(20):
                k_fn(pool, pool.view(torch.bfloat16), all_slots)
            torch.cuda.synchronize()
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            for _ in range(200):
                k_fn(pool, pool.view(torch.bfloat16), all_slots)
            e_ev.record()
            torch.cuda.synchronize()
            entry["all_pages_us"] = round(s_ev.elapsed_time(e_ev) * 1000.0 / 200, 3)

            s_ev.record()
            for _ in range(200):
                k_fn(pool, pool.view(torch.bfloat16), skip_slots)
            e_ev.record()
            torch.cuda.synchronize()
            entry["skip_only_us"] = round(s_ev.elapsed_time(e_ev) * 1000.0 / 200, 3)
        results.append(entry)
    return {"passed": True, "state_bytes": STATE_BYTES, "cases": results}


def run_registration_check() -> dict:
    cases = [
        (16384, 128, False, 64),
        (65536, 128, False, 64),
        (524288, 128, False, 64),
        (1048576, 256, True, 512),
        (1048576, 256, True, 1024),
        (131072, 256, True, 1024),
    ]
    calls = [prepare.call(summarize_blocks, 128), prepare.call(summarize_blocks, 256)]
    for width, page, source, budget in cases:
        calls.extend(
            QueryReusePlan.calls(width, page, candidate_blocks=source, budget=budget)
        )
    tilelang.prepare(calls, num_workers=4)
    prepare.freeze("LSA block8 midrange-blend declared-call qualification")
    for page in (128, 256):
        summarize_blocks(page)
    for width, page, source, budget in cases:
        QueryReusePlan(width, page, candidate_blocks=source, budget=budget)
    return {
        "passed": True,
        "cases": len(cases),
        "calls": len(calls),
        "bind_after_freeze": True,
    }


def run_runtime_and_graph_check(graph_mode: bool = False) -> dict:
    configs = [
        (16384, 8192, 128, False, 64),
        (65536, 16384, 256, True, 512),
        (131072, 16384, 256, True, 1024),
    ]
    checked = []
    for width, seq_len, page, source, budget in configs:
        num_pages = seq_len // page
        rows = 4
        gen = torch.Generator(device="cuda").manual_seed(2026 + width + budget)
        keys = torch.randn(
            (seq_len, 128), generator=gen, device="cuda", dtype=torch.bfloat16
        )
        pool = torch.empty(
            (num_pages, page * STATE_BYTES), dtype=torch.uint8, device="cuda"
        )
        pool.fill_(173)
        slots = torch.arange(seq_len, dtype=torch.int64, device="cuda")
        write_index_keys(keys, pool, slots, page)
        complete_slots = (
            torch.arange(num_pages, dtype=torch.int64, device="cuda") * page + page - 1
        )
        summarize_blocks(page)(pool, pool.view(torch.bfloat16), complete_slots)

        q_bf16 = torch.randn(
            (rows * 32, 128), generator=gen, device="cuda", dtype=torch.bfloat16
        )
        q_mxfp4 = torch.empty((rows, 32, 64), dtype=torch.uint8, device="cuda")
        q_scales_u8 = torch.empty((rows, 32, 4), dtype=torch.uint8, device="cuda")
        quantize_mxfp4(q_bf16, q_mxfp4, q_scales_u8)
        q_scales = q_scales_u8.view(torch.float32).reshape(rows, 32)
        weights = (
            torch.rand((rows, 32), generator=gen, device="cuda", dtype=torch.float32)
            * 0.02
        ).to(torch.bfloat16)
        pad_table = torch.full(
            (rows, width // page), -1, dtype=torch.int32, device="cuda"
        )
        pad_table[:, :num_pages] = torch.arange(
            num_pages, dtype=torch.int32, device="cuda"
        ).unsqueeze(0)
        lengths = torch.full((rows,), seq_len, dtype=torch.int32, device="cuda")
        active = torch.tensor([width], dtype=torch.int32, device="cuda")

        plan = QueryReusePlan(
            width, page, candidate_blocks=source, budget=budget, row_cap=16
        )
        scratch = [
            torch.empty(sh, dtype=dt, device="cuda")
            for sh, dt in plan.scratch_specs(rows)
        ]
        for t in scratch:
            t.view(torch.uint8).fill_(173)
        out = torch.full((rows, 512), -99, dtype=torch.int32, device="cuda")
        cand = (
            torch.full((rows, CANDIDATE_BLOCKS), -99, dtype=torch.int32, device="cuda")
            if source
            else None
        )
        cand_len = (
            torch.full((rows,), -99, dtype=torch.int32, device="cuda")
            if source
            else None
        )
        kwargs = (
            {"candidate_out": cand, "candidate_out_lengths": cand_len}
            if source
            else {}
        )
        plan.run(
            q_mxfp4,
            q_scales,
            weights,
            pool,
            pad_table,
            lengths,
            active,
            out=out,
            scratch=scratch,
            **kwargs,
        )
        torch.cuda.synchronize()
        assert bool(((out >= 0) & (out < seq_len)).all().item())
        for r in range(rows):
            assert torch.unique(out[r]).numel() == 512

        if graph_mode:
            eager_out = out.clone()
            eager_cand = cand.clone() if source else None
            eager_cand_len = cand_len.clone() if source else None
            saved_lengths = lengths.clone()
            saved_active = active.clone()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                summarize_blocks(page)(pool, pool.view(torch.bfloat16), complete_slots)
                plan.run(
                    q_mxfp4,
                    q_scales,
                    weights,
                    pool,
                    pad_table,
                    lengths,
                    active,
                    out=out,
                    scratch=scratch,
                    **kwargs,
                )

            out.fill_(-1)
            if source:
                cand.fill_(-1)
                cand_len.fill_(-1)
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, eager_out)
            if source:
                assert torch.equal(cand, eager_cand)
                assert torch.equal(cand_len, eager_cand_len)

            lengths.zero_()
            active.zero_()
            g.replay()
            torch.cuda.synchronize()
            assert bool((out == -1).all().item())
            if source:
                assert bool((cand == -1).all().item())
                assert bool((cand_len == 0).all().item())

            lengths.copy_(saved_lengths)
            active.copy_(saved_active)
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, eager_out)
            if source:
                assert torch.equal(cand, eager_cand)
                assert torch.equal(cand_len, eager_cand_len)

        checked.append(
            {
                "width": width,
                "seq_len": seq_len,
                "page": page,
                "source": source,
                "budget": budget,
                "graph_replayed": graph_mode,
            }
        )
    return {"passed": True, "cases": checked}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gate",
        choices=[
            "summary",
            "summary-timing",
            "registration",
            "actual",
            "graph",
            "all",
        ],
        required=True,
    )
    args = p.parse_args()
    t0 = time.time()
    if args.gate == "summary":
        res = run_summary_check(timing=False)
    elif args.gate == "summary-timing":
        res = run_summary_check(timing=True)
    elif args.gate == "registration":
        res = run_registration_check()
    elif args.gate == "actual":
        res = run_runtime_and_graph_check(graph_mode=False)
    elif args.gate == "graph":
        res = run_runtime_and_graph_check(graph_mode=True)
    elif args.gate == "all":
        res = {
            "passed": True,
            "summary_timing": run_summary_check(timing=True),
            "actual": run_runtime_and_graph_check(graph_mode=False),
            "graph": run_runtime_and_graph_check(graph_mode=True),
            "registration": run_registration_check(),
        }
    res["elapsed_s"] = round(time.time() - t0, 3)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
