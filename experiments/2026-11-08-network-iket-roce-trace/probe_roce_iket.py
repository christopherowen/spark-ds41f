#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Multi-node RoCE + streaming IKET verification across all four sparknet collectives.

Executes oneshot allreduce, allgather, reducescatter, and exchange across a
multi-node RoCE world (sp5..sp8) with multi-CTA launch geometry (grid_x > 1),
recording both sparknet ROCE_TRACE (GPU %globaltimer + CPU CLOCK_MONOTONIC_RAW
proxy timestamps + NTP cross-rank clock alignment) and TileLang streaming IKET
warp ranges, then exports a unified Chrome/Perfetto timeline.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import socket
import time

import torch
import torch.distributed as dist


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _lat_stats(samples_us: list[float]) -> dict[str, float]:
    ordered = sorted(samples_us)
    return {
        "p5_us": round(_percentile(ordered, 5), 2),
        "p50_us": round(_percentile(ordered, 50), 2),
        "p95_us": round(_percentile(ordered, 95), 2),
        "n": len(ordered),
    }


def _analyze_iket_cta_waits(stream_dir: Path) -> dict[str, object]:
    """Verify CTA 0 emits wait.flags and non-zero CTAs emit wait.barrier."""
    from stream.export import records

    range_names: dict[tuple[int, int], str] = {}
    kernel_names: dict[int, str] = {}
    manifests: dict[str, dict] = {}
    for path in sorted((stream_dir / "sources").glob("*.json")):
        manifest = json.loads(path.read_text())
        kid = manifest["kernel_id"]
        kname = manifest["kernel"]
        kernel_names[kid] = kname
        manifests[kname] = {
            "ctas": manifest["ctas"],
            "warps": manifest["warps"],
            "granularity": manifest["granularity"],
            "ranges": [s["range"] for s in manifest["sites"] if "range" in s],
        }
        for site in manifest["sites"]:
            if "range_id" in site:
                range_names[kid, site["range_id"]] = site["range"]

    per_kernel: dict[str, dict[str, Counter]] = {}
    for (
        _seq,
        _ts,
        _launch,
        kernel,
        cta,
        _warp,
        _sm,
        event,
        _payload,
        kind,
        _,
    ) in records(stream_dir):
        if kind != 1:
            continue
        kname = kernel_names.get(kernel, f"kernel_{kernel}")
        rname = range_names.get((kernel, event), f"event_{event}")
        bucket = "cta0" if cta == 0 else "cta_nonzero"
        per_kernel.setdefault(kname, {"cta0": Counter(), "cta_nonzero": Counter()})[
            bucket
        ][rname] += 1

    summary: dict[str, object] = {"manifests": manifests, "kernels": {}}
    for kname, counts in sorted(per_kernel.items()):
        cta0_flags = counts["cta0"].get("wait.flags", 0)
        cta0_barrier = counts["cta0"].get("wait.barrier", 0)
        ctanz_flags = counts["cta_nonzero"].get("wait.flags", 0)
        ctanz_barrier = counts["cta_nonzero"].get("wait.barrier", 0)
        assert cta0_flags > 0, f"{kname}: CTA 0 recorded no wait.flags ({counts})"
        assert cta0_barrier == 0, f"{kname}: CTA 0 recorded wait.barrier ({counts})"
        assert ctanz_barrier > 0, (
            f"{kname}: non-zero CTAs recorded no wait.barrier ({counts})"
        )
        assert ctanz_flags == 0, (
            f"{kname}: non-zero CTAs recorded wait.flags ({counts})"
        )
        summary["kernels"][kname] = {
            "cta0_counts": dict(counts["cta0"]),
            "cta_nonzero_counts": dict(counts["cta_nonzero"]),
            "cta0_wait_flags": cta0_flags,
            "cta0_wait_barrier": cta0_barrier,
            "cta_nonzero_wait_flags": ctanz_flags,
            "cta_nonzero_wait_barrier": ctanz_barrier,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=24)
    parser.add_argument("--nbytes", type=int, default=65536)
    parser.add_argument("--skew-us", type=float, default=15.0)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    args.out.mkdir(parents=True, exist_ok=True)

    collector = None
    stream_dir = args.out / f"stream-r{rank}"
    if args.stream:
        if stream_dir.exists():
            shutil.rmtree(stream_dir)
        from stream.collector import enable

        collector = enable(
            stream_dir,
            ring_bytes=16 << 20,
            chunk_bytes=64 << 20,
            ctas=0,
            iters=1,
            warps=1,
            payloads=True,
            patched=False,
        )

    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    from sparknet import oneshot as roce
    from sparknet.oneshot import trace as roce_trace

    rt = roce.AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_size=1 << 20,
        max_gather_bytes=4 << 20,
    )
    rt.prepare((torch.bfloat16,), padded_gather=False)
    torch.cuda.synchronize()
    dist.barrier()

    # Construct inputs so every collective launches with nbytes = args.nbytes (grid_x = 8 at 64 KiB)
    elem_bytes = 2
    elems = args.nbytes // elem_bytes
    cols = 256
    rows = elems // cols

    ar_inp = torch.full((rows, cols), float(rank + 1), dtype=torch.bfloat16, device=device)
    ag_inp = torch.full((rows, cols), float(rank + 1), dtype=torch.bfloat16, device=device)
    rs_inp = torch.arange(world, dtype=torch.bfloat16, device=device).view(world, 1, 1).expand(world, rows, cols).contiguous() * float(rank + 1)
    ex_inp = (
        torch.arange(world, dtype=torch.bfloat16, device=device).view(world, 1, 1) * 10.0
        + float(rank + 1)
    ).expand(world, rows, cols).contiguous()

    # Warmup (1 iteration of each collective)
    out_ar = rt.all_reduce(ar_inp)
    out_ag = rt.all_gather(ag_inp, dim=0)
    out_rs = rt.reduce_scatter(rs_inp.view(world * rows, cols), dim=0)
    out_ex = rt.exchange(ex_inp, copy_self=True)
    torch.cuda.synchronize()
    dist.barrier()

    # Verify numerical accuracy against reference values
    expected_ar = torch.full_like(ar_inp, float(world * (world + 1) // 2))
    torch.testing.assert_close(out_ar, expected_ar)
    for p in range(world):
        assert torch.all(out_ag[p * rows : (p + 1) * rows] == float(p + 1))
    expected_rs = torch.full_like(out_rs, float(rank) * float(world * (world + 1) // 2))
    torch.testing.assert_close(out_rs, expected_rs)
    for p in range(world):
        assert torch.all(out_ex[p] == float(rank * 10 + (p + 1)))

    host_latencies: dict[str, list[float]] = {
        "allreduce": [],
        "allgather": [],
        "reducescatter": [],
        "exchange": [],
    }
    skew_ns = int(args.skew_us * 1000.0)

    for step in range(args.iters):
        apply_skew = step >= (args.iters // 2) and (rank in (1, 3)) and skew_ns > 0
        for op_name in ("allreduce", "allgather", "reducescatter", "exchange"):
            dist.barrier()
            torch.cuda.synchronize()
            if apply_skew:
                target = time.perf_counter_ns() + skew_ns
                while time.perf_counter_ns() < target:
                    pass
            t0 = time.perf_counter_ns()
            if op_name == "allreduce":
                rt.all_reduce(ar_inp, out=out_ar)
            elif op_name == "allgather":
                rt.all_gather(ag_inp, dim=0, out=out_ag)
            elif op_name == "reducescatter":
                rt.reduce_scatter(rs_inp.view(world * rows, cols), dim=0, out=out_rs)
            else:
                rt.exchange(ex_inp, out=out_ex, copy_self=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            host_latencies[op_name].append((t1 - t0) / 1000.0)
    dist.barrier()

    trace_path = rt._trace_path
    loaded_trace = roce_trace.read(trace_path) if trace_path else None
    if trace_path:
        shutil.copy2(trace_path, args.out / f"roce-r{rank}.bin")

    iket_cta_summary = None
    if collector is not None:
        collector.close()
        iket_cta_summary = _analyze_iket_cta_waits(stream_dir)

    rt_stats = rt.stats()
    rt.close()

    all_traces: list[object] = [None] * world
    all_host_lat: list[object] = [None] * world
    all_iket_cta: list[object] = [None] * world
    dist.all_gather_object(all_traces, loaded_trace)
    dist.all_gather_object(all_host_lat, {k: _lat_stats(v) for k, v in host_latencies.items()})
    dist.all_gather_object(all_iket_cta, iket_cta_summary)

    if rank == 0:
        receipt: dict[str, object] = {
            "world_size": world,
            "nbytes": args.nbytes,
            "iters": args.iters,
            "skew_us": args.skew_us,
            "roce_trace_enabled": bool(trace_path),
            "iket_stream_enabled": bool(args.stream),
            "host_latencies_by_rank": {r: all_host_lat[r] for r in range(world)},
            "runtime_stats_rank0": rt_stats,
        }
        if all(t is not None for t in all_traces):
            traces = [t for t in all_traces if t is not None]
            # Drop the 4 warmup ops (seq 1..4) for steady-state analysis
            steady = [roce_trace.select(t, min_seq=5) for t in traces]
            sync_End = 4 + (args.iters // 2) * 4
            sync_traces = [roce_trace.select(t, min_seq=5, max_seq=sync_End) for t in traces]
            skew_traces = [roce_trace.select(t, min_seq=sync_End + 1) for t in traces]

            receipt["roce_summary_all"] = roce_trace.summarize(steady)
            receipt["roce_summary_by_op"] = {
                op_name: roce_trace.summarize([roce_trace.select(t, op=op_name) for t in steady])
                for op_name in ("allreduce", "allgather", "reducescatter", "exchange")
            }
            receipt["roce_summary_synchronized"] = roce_trace.summarize(sync_traces)
            receipt["roce_summary_skewed"] = roce_trace.summarize(skew_traces)

            perfetto_only_path = args.out / "perfetto-roce.json"
            receipt["perfetto_roce"] = roce_trace.export_perfetto(
                steady, str(perfetto_only_path)
            )
            if args.stream:
                from stream.export import export as export_stream
                from vllm.models.deepseek_v4_1.tilelang import iket as vllm_iket

                iket_only_path = args.out / "perfetto-iket-r0.json"
                unified_path = args.out / "perfetto-unified.json"
                merged_via_iket_path = args.out / "perfetto-merged-via-iket.json"
                receipt["stream_export_iket_only"] = export_stream(
                    stream_dir, iket_only_path
                )
                receipt["stream_export_unified"] = export_stream(
                    stream_dir, unified_path, roce_traces=steady
                )
                receipt["iket_merge_roce_traces"] = vllm_iket.merge_roce_traces(
                    iket_only_path, steady, merged_via_iket_path
                )
                receipt["iket_cta_wait_verification"] = {
                    r: all_iket_cta[r] for r in range(world)
                }

        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
