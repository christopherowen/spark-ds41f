#!/usr/bin/env python3
"""Standalone 2-boot GPU bit-identity and latency benchmark for vLLM 0731 (SparseMlaV41Plan).

Tests SparseMlaV41Plan before (control: block_H=32, threads=256, decode_max_rows=28 on
indexed layers) vs. after vLLM 0731 (candidate: block_H=16, threads=128 on 16-head ranks
with skip_inactive=True on decode, and decode_max_rows=48 full-wave floor on indexed layers)
across:
- heads = 16 (TP4 all ranks & TP3 Rank 2) and heads = 24 (TP3 Ranks 0 & 1),
- indexed (idx_width = 512) and SWA-only (idx_width = 0) layers,
- wave-boundary / decode shapes (rows = 16, 24, 28, 29, 32, 40, 48) and
  extend prefill shapes (rows = 1024, 2048, 4096, 6144),
verifying 100% int16 bit-identity on GPU and measuring latency across two fresh container
boots (boot1 and boot2).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from vllm.models.deepseek_v4_1.tilelang import prepare
from vllm.models.deepseek_v4_1.tilelang.sparse_mla import (
    HEAD_DIM,
    IDX_RECORD_BYTES,
    ROPE_DIM,
    SWA_RECORD_BYTES,
    SparseMlaV41Plan,
    sparse_mla_fwd_v41,
)
from vllm.v1.worker.workspace import init_workspace_manager

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--boot", required=True, help="Boot identifier (e.g. boot1, boot2)")
p.add_argument("--repeats", type=int, default=15, help="Timed CUDA graph replays per case")
args = p.parse_args()
assert args.repeats >= 3


def emit(**record) -> None:
    print(
        json.dumps(
            {
                "boot": args.boot,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **record,
            }
        ),
        flush=True,
    )


def build_control_plan(
    heads: int,
    swa_width: int,
    idx_width: int,
    idx_page_size: int,
    *,
    swa_page_size: int,
    max_rows: int,
    device: torch.device,
) -> SparseMlaV41Plan:
    """Construct a SparseMlaV41Plan with pre-0731 control parameters:
    block_H=32, threads=256 for extend/decode, and decode_max_rows=28 (indexed) or 144 (SWA)."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    segments = 5 if idx_width else 1
    ctrl_decode_max_rows = min(max_rows, (3 * sms) // segments)
    ctrl_split_max_rows = min(
        ctrl_decode_max_rows, sms // (segments * ((heads + 15) // 16))
    )
    plan = SparseMlaV41Plan(
        heads,
        swa_width,
        idx_width,
        idx_page_size,
        swa_page_size=swa_page_size,
        max_rows=max_rows,
        decode_max_rows=ctrl_decode_max_rows,
        split_max_rows=ctrl_split_max_rows,
        block_H=32,
        threads=256,
        device=device,
    )
    # Force control extend and decode to pre-0731 block_H=32, threads=256 even when heads == 16
    shape = (heads, swa_width, idx_width, idx_page_size)
    cfg = dict(
        sm_scale=HEAD_DIM**-0.5,
        block_I=32,
        segment_chunks=4,
        swa_page_size=swa_page_size,
    )
    plan.kernels["extend"] = sparse_mla_fwd_v41(
        *shape, "extend", block_H=32, threads=256, **cfg
    )
    if ctrl_decode_max_rows:
        plan.kernels["decode"] = sparse_mla_fwd_v41(
            *shape, "decode", block_H=32, threads=256, skip_inactive=True, **cfg
        )
    if ctrl_split_max_rows:
        plan.kernels["split"] = sparse_mla_fwd_v41(
            *shape, "decode", block_H=16, threads=128, skip_inactive=False, **cfg
        )
    return plan


def make_rope_table(positions: int, device: torch.device) -> torch.Tensor:
    exponents = torch.arange(0, ROPE_DIM, 2, device=device, dtype=torch.float64) / ROPE_DIM
    angles = torch.arange(positions, device=device, dtype=torch.float64)[:, None] * (
        1.0 / (10000.0**exponents)
    )
    return torch.cat((angles.cos(), angles.sin()), dim=-1).float().contiguous()


def make_paged_pools(
    swa_pages: int,
    idx_pages: int,
    page_size: int,
    gen: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create strided BLHNC SWA and Indexed page buffers with finite quantized records."""
    layers = 3
    pad = 256
    swa_row_bytes = page_size * SWA_RECORD_BYTES
    swa_buf = torch.zeros((swa_pages, layers, swa_row_bytes + pad), dtype=torch.uint8, device=device)
    swa_view = swa_buf[:, 1, :swa_row_bytes].view(swa_pages, page_size, SWA_RECORD_BYTES)
    # 512 bytes E4M3 payload (avoid 0x7F/0xFF NaN) + 16 bytes UE8M0 exponent around 126..128
    swa_view[:, :, :512] = torch.randint(0, 120, (swa_pages, page_size, 512), generator=gen, device=device, dtype=torch.uint8)
    swa_view[:, :, 512:] = torch.randint(125, 128, (swa_pages, page_size, 16), generator=gen, device=device, dtype=torch.uint8)
    swa_pool = swa_buf[:, 1, :swa_row_bytes]

    idx_row_bytes = page_size * IDX_RECORD_BYTES
    idx_buf = torch.zeros((idx_pages, layers, idx_row_bytes + pad), dtype=torch.uint8, device=device)
    idx_view = idx_buf[:, 1, :idx_row_bytes].view(idx_pages, page_size, IDX_RECORD_BYTES)
    # 256 bytes packed E2M1 + 32 bytes E4M3 scales (avoid 0x7F/0xFF NaN)
    idx_view[:, :, :256] = torch.randint(0, 256, (idx_pages, page_size, 256), generator=gen, device=device, dtype=torch.uint8)
    idx_view[:, :, 256:] = torch.randint(48, 64, (idx_pages, page_size, 32), generator=gen, device=device, dtype=torch.uint8)
    idx_pool = idx_buf[:, 1, :idx_row_bytes]
    return swa_pool, idx_pool


def build_inputs(
    rows: int,
    heads: int,
    swa_width: int,
    idx_width: int,
    page_size: int,
    swa_pages: int,
    idx_pages: int,
    context_mode: str,
    gen: torch.Generator,
    device: torch.device,
) -> dict:
    q = (torch.randn((rows, heads, HEAD_DIM), generator=gen, device=device) * 0.5).bfloat16()
    q[0, 0, 0] = -0.0
    sinks = torch.randn((heads,), generator=gen, device=device, dtype=torch.float32)
    positions = torch.arange(rows, device=device, dtype=torch.int64) % 8192

    swa_total = swa_pages * page_size
    swa_idx = torch.randint(1, swa_total, (rows, swa_width), generator=gen, device=device, dtype=torch.int32)
    if context_mode == "full_640":
        swa_len = torch.full((rows,), swa_width, dtype=torch.int32, device=device)
    elif context_mode == "partial_256":
        swa_len = torch.full((rows,), swa_width, dtype=torch.int32, device=device)
    elif context_mode == "ced_25pct_dead":
        swa_len = torch.full((rows,), swa_width, dtype=torch.int32, device=device)
        swa_len[::4] = 0
    else:
        raise ValueError(context_mode)

    keep_swa = torch.arange(swa_width, device=device)[None, :] < swa_len[:, None]
    swa_idx = torch.where(keep_swa, swa_idx, -1).to(torch.int32).contiguous()

    data = {
        "q": q,
        "sinks": sinks,
        "positions": positions,
        "swa_idx": swa_idx,
        "swa_len": swa_len,
    }
    if idx_width:
        max_pages_per_row = 32
        idx_context = max_pages_per_row * page_size
        table = torch.randint(
            1, idx_pages, (rows, max_pages_per_row), generator=gen, device=device, dtype=torch.int32
        )
        idx_logical = torch.randint(
            0, idx_context, (rows, idx_width), generator=gen, device=device, dtype=torch.int32
        )
        if context_mode == "full_640":
            idx_len = torch.full((rows,), idx_width, dtype=torch.int32, device=device)
        elif context_mode == "partial_256":
            # 128 SWA + 128 Indexed = 256 keys (2 segments active out of 5)
            idx_len = torch.full((rows,), 128, dtype=torch.int32, device=device)
        elif context_mode == "ced_25pct_dead":
            idx_len = torch.full((rows,), idx_width, dtype=torch.int32, device=device)
            idx_len[::4] = 0
        keep_idx = torch.arange(idx_width, device=device)[None, :] < idx_len[:, None]
        idx_logical = torch.where(keep_idx, idx_logical, -1).to(torch.int32).contiguous()
        data.update(idx_logical=idx_logical, idx_len=idx_len, table=table)
    return data


def time_graph(call, wash_buf: torch.Tensor | None, repeats: int) -> list[float]:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if wash_buf is not None:
            wash_buf.add_(1.0)
        begin.record()
        call()
        end.record()
    samples = []
    for _ in range(repeats):
        graph.replay()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end))
    return samples


def main() -> int:
    device = torch.device("cuda")
    init_workspace_manager(device, num_ubatches=1)
    flush = torch.zeros(64 * 1024 * 1024 // 4, device=device, dtype=torch.float32)

    swa_width = 128
    page_size = 128
    max_rows = 6144
    swa_pages = 32
    idx_pages = 64

    gen = torch.Generator(device=device).manual_seed(20261106)
    rope = make_rope_table(8192, device)
    swa_pool, idx_pool = make_paged_pools(swa_pages, idx_pages, page_size, gen, device)

    emit(
        kind="identity",
        gpu=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        multi_processor_count=torch.cuda.get_device_properties(device).multi_processor_count,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        seed=20261106,
        repeats=args.repeats,
        cache_modes=["warm", "64MiB_wash"],
    )

    configs = [
        # (heads, idx_width, row_shapes, context_modes)
        (16, 512, (16, 24, 28, 29, 32, 40, 48, 1024, 2048, 4096, 6144), ("full_640", "partial_256", "ced_25pct_dead")),
        (16, 0, (16, 24, 32, 48, 1024, 2048, 6144), ("full_640",)),
        (24, 512, (28, 29, 32, 40, 48, 1024, 2048, 4096, 6144), ("full_640", "partial_256")),
    ]

    for heads, idx_width, row_shapes, context_modes in configs:
        ctrl_plan = build_control_plan(
            heads, swa_width, idx_width, page_size,
            swa_page_size=page_size, max_rows=max_rows, device=device,
        )
        cand_plan = SparseMlaV41Plan(
            heads, swa_width, idx_width, page_size,
            swa_page_size=page_size, max_rows=max_rows, device=device,
        )

        for rows in row_shapes:
            for context_mode in context_modes:
                if context_mode == "partial_256" and rows > 48:
                    continue
                if context_mode == "ced_25pct_dead" and rows <= 48:
                    continue

                inp = build_inputs(
                    rows, heads, swa_width, idx_width, page_size,
                    swa_pages, idx_pages, context_mode, gen, device,
                )
                idx_kwargs = {}
                if idx_width:
                    idx_kwargs = dict(
                        idx_cache=idx_pool,
                        idx_indices=inp["idx_logical"],
                        idx_lengths=inp["idx_len"],
                        idx_page_table=inp["table"],
                    )

                out_ctrl = torch.empty((rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
                out_cand = torch.empty((rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)

                ctrl_route = ctrl_plan.kernel_for(rows)
                cand_route = cand_plan.kernel_for(rows)

                def run_ctrl():
                    ctrl_plan.run(
                        inp["q"], swa_pool, inp["swa_idx"], inp["swa_len"],
                        inp["sinks"], inp["positions"], rope, out_ctrl,
                        **idx_kwargs,
                    )

                def run_cand():
                    cand_plan.run(
                        inp["q"], swa_pool, inp["swa_idx"], inp["swa_len"],
                        inp["sinks"], inp["positions"], rope, out_cand,
                        **idx_kwargs,
                    )

                run_ctrl()
                run_cand()
                torch.cuda.synchronize()

                differ = int((out_ctrl.view(torch.int16) != out_cand.view(torch.int16)).sum())
                assert differ == 0, (
                    f"Bit mismatch for heads={heads}, idx_width={idx_width}, rows={rows}, "
                    f"mode={context_mode}: {differ} elements differ"
                )

                # When heads == 16, also verify that all 3 candidate routes (extend, decode, split)
                # produce 100% bit-identical int16 outputs on this exact input!
                for explicit_route, max_r in (
                    ("extend", cand_plan.max_rows),
                    ("decode", cand_plan.decode_max_rows),
                    ("split", cand_plan.split_max_rows),
                ):
                    if rows <= max_r and rows <= 48:
                        out_route = torch.empty_like(out_cand)
                        cand_plan.run(
                            inp["q"], swa_pool, inp["swa_idx"], inp["swa_len"],
                            inp["sinks"], inp["positions"], rope, out_route,
                            kernel=explicit_route,
                            **idx_kwargs,
                        )
                        route_diff = int((out_route.view(torch.int16) != out_ctrl.view(torch.int16)).sum())
                        assert route_diff == 0, (
                            f"Route {explicit_route} bit mismatch for heads={heads}, idx={idx_width}, rows={rows}"
                        )

                emit(
                    kind="bit_identity",
                    heads=heads,
                    idx_width=idx_width,
                    rows=rows,
                    context_mode=context_mode,
                    ctrl_route=ctrl_route,
                    cand_route=cand_route,
                    int16_differ=differ,
                    elements=out_cand.numel(),
                    passed=True,
                )

                variants = [
                    ("control_pre0731", ctrl_route, run_ctrl),
                    ("candidate_0731", cand_route, run_cand),
                ]
                if args.boot.endswith("2"):
                    variants = list(reversed(variants))

                for cache_mode, wash_buf in (("warm", None), ("64MiB_wash", flush)):
                    for vname, vroute, fn in variants:
                        samples = time_graph(fn, wash_buf, args.repeats)
                        emit(
                            kind="timing",
                            heads=heads,
                            idx_width=idx_width,
                            rows=rows,
                            context_mode=context_mode,
                            cache_mode=cache_mode,
                            variant=vname,
                            route=vroute,
                            mean_ms=statistics.mean(samples),
                            median_ms=statistics.median(samples),
                            stdev_ms=statistics.stdev(samples),
                            min_ms=min(samples),
                            max_ms=max(samples),
                            samples_ms=samples,
                        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
