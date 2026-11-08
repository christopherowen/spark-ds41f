#!/usr/bin/env python3
"""Multi-node RoCE benchmark for sparknet PDL chaining and zero-copy send_dev[slot] staging.

Measures CUDA Graph replay latency across M = 1, 4, 6, 16, 32 rows for four
DeepSeek-V4.1 collective chains on a 4-node RoCE fabric:
1. wo_b_allreduce_mhc_post:
   WO-B (gemm_mxfp8, 5120x2048) -> oneshot_allreduce -> mhc_post (post_streams)
2. moe_combine_allreduce_mhc_post:
   moe_combine (top_k=6 + shared, 5120) -> oneshot_allreduce -> mhc_post (post_streams)
3. wqa_wkv_allgather_attention_front:
   fused_wqa_wkv (gemm_mxfp8, 448x5120) -> oneshot_allgather (dim=-1) -> attention_front
4. wkv_allgather_gather_columns_into:
   compressor_proj (gemm_mxfp8, 1280x5120) -> oneshot_allgather (dim=0) -> gather_columns_into
"""

import argparse
import json
import os
import socket
import statistics
import time
from datetime import timedelta
from pathlib import Path

import tilelang
import tilelang.language as T
import torch
import torch.distributed as dist


def _build_allreduce_variant(
    mode: str,
    dtype_name: str,
    world_size: int,
    rank: int,
    threads: int,
    slots: int,
    flag_stride: int,
    hca_count: int,
    ring4: bool,
):
    """Build an oneshot_allreduce kernel for a specific PDL / direct-stage variant.

    mode:
      - "baseline_no_pdl": no pdl_sync, no pdl_trigger, unconditional copy
      - "pdl_entry": pdl_sync + pdl_trigger at kernel entry
      - "pdl_post_doorbell": pdl_sync at entry, pdl_trigger after roce_doorbell, unconditional copy
      - "pdl_post_wait": pdl_sync at entry, pdl_trigger after roce_wait_peers, unconditional copy
      - "pdl_post_doorbell_direct_send": pdl_sync at entry, pdl_trigger after roce_doorbell, skip Phase 1 when inp == send_slot
    """
    from sparknet.oneshot._device import PACK_BYTES, render_source
    from sparknet.oneshot._oneshot_tilelang import (
        _REDUCE_FUNCTION,
        _lanes,
        _pass_configs,
        _resident_blocks,
    )

    lanes, neighbor_lanes = _lanes(hca_count, ring4)
    source = render_source(
        world_size=world_size,
        rank=rank,
        slots=slots,
        flag_stride=flag_stride,
        hca_count=lanes,
        neighbor_lanes=neighbor_lanes,
        trace=None,
    )
    if mode not in ("pdl_post_doorbell_direct_send", "pdl_sync_only_direct_send"):
        source = source.replace(
            "if (src != dst) {\n        roce_st_global_pack(dst, roce_ld_global_pack(src));\n    }",
            "roce_st_global_pack(dst, roce_ld_global_pack(src));",
        )
    reduce_pack = _REDUCE_FUNCTION[dtype_name]
    resident = _resident_blocks(threads)
    pack_words = PACK_BYTES // 4
    use_pdl_sync = mode not in ("baseline_no_pdl", "pdl_trigger_only")
    use_direct = mode in ("pdl_post_doorbell_direct_send", "pdl_sync_only_direct_send")

    @tilelang.jit(pass_configs=_pass_configs())
    def program():
        n_in, n_out, n_counters = T.dynamic("n_in, n_out, n_counters")

        @T.prim_func
        def oneshot_allreduce(
            inp: T.Tensor((n_in,), T.int32),
            out: T.Tensor((n_out,), T.int32),
            counters: T.Tensor((n_counters,), T.int32),
            region_base: T.int64,
            size_packs: T.int32,
            nbytes: T.int32,
            recv_off: T.int32,
            flag_off: T.int32,
            send_off: T.int32,
            ctrl_off: T.int32,
            slot_bytes: T.int32,
            stage_index: T.int32,
            tail_index: T.int32,
            rung_index: T.int32,
            arrival_index: T.int32,
            grid_x: T.int32,
            trace_base: T.int64,
        ):
            with T.Kernel(grid_x, threads=threads) as bx:
                T.annotate_min_blocks_per_sm(resident)
                T.import_source(source)
                if use_pdl_sync:
                    T.pdl_sync()
                if mode == "pdl_entry":
                    T.pdl_trigger()
                tx = T.get_thread_binding()
                epoch = T.alloc_var(T.uint32)
                epoch = T.call_extern(
                    "roce_ld_relaxed_gpu_u32", T.address_of(counters[0]), dtype=T.uint32
                )
                seq = T.alloc_var(T.uint32)
                seq = epoch + T.uint32(1)
                slot = T.alloc_var(T.int32)
                slot = T.cast(seq & T.uint32(1), T.int32)
                send_slot = T.alloc_var(T.int32)
                send_slot = send_off + slot * slot_bytes
                index = T.alloc_var(T.int32)
                index = bx * threads + tx
                stride = T.alloc_var(T.int32)
                stride = grid_x * threads
                count = T.alloc_var(T.int32)
                count = T.max(0, (size_packs - index + stride - 1) // stride)
                if use_direct:
                    if (
                        T.call_extern(
                            "roce_same_ptr",
                            T.address_of(inp[0]),
                            T.call_extern(
                                "roce_ptr",
                                region_base + T.cast(send_slot, T.int64),
                                dtype="handle",
                            ),
                            dtype=T.int32,
                        )
                        == 0
                    ):
                        for i in T.serial(count):
                            pack = index + i * stride
                            T.call_extern(
                                "roce_copy_pack",
                                T.address_of(inp[pack * pack_words]),
                                T.call_extern(
                                    "roce_ptr",
                                    region_base
                                    + T.cast(send_slot + pack * PACK_BYTES, T.int64),
                                    dtype="handle",
                                ),
                                dtype="handle",
                            )
                        T.sync_threads()
                else:
                    for i in T.serial(count):
                        pack = index + i * stride
                        T.call_extern(
                            "roce_copy_pack",
                            T.address_of(inp[pack * pack_words]),
                            T.call_extern(
                                "roce_ptr",
                                region_base
                                + T.cast(send_slot + pack * PACK_BYTES, T.int64),
                                dtype="handle",
                            ),
                            dtype="handle",
                        )
                    T.sync_threads()
                if tx == 0:
                    rang = T.alloc_var(T.uint32)
                    rang = T.call_extern(
                        "roce_doorbell",
                        T.address_of(counters[stage_index]),
                        T.cast(grid_x, T.uint32),
                        T.call_extern(
                            "roce_ptr",
                            region_base + T.cast(ctrl_off, T.int64),
                            dtype="handle",
                        ),
                        T.cast(nbytes, T.uint32),
                        seq,
                        dtype=T.uint32,
                    )
                    if rang != T.uint32(0):
                        T.call_extern(
                            "roce_st_release_gpu_u32",
                            T.address_of(counters[rung_index]),
                            seq,
                            dtype="handle",
                        )
                if mode in (
                    "pdl_trigger_only",
                    "pdl_post_doorbell",
                    "pdl_post_doorbell_direct_send",
                ):
                    T.pdl_trigger()
                T.call_extern(
                    "roce_wait_peers",
                    bx,
                    tx,
                    T.call_extern(
                        "roce_ptr",
                        region_base + T.cast(flag_off, T.int64),
                        dtype="handle",
                    ),
                    seq,
                    T.call_extern(
                        "roce_ptr",
                        region_base + T.cast(ctrl_off, T.int64),
                        dtype="handle",
                    ),
                    T.address_of(counters[rung_index]),
                    T.address_of(counters[arrival_index]),
                    dtype="handle",
                )
                T.sync_threads()
                if bx == 0:
                    if tx == 0:
                        T.call_extern(
                            "roce_st_release_gpu_u32",
                            T.address_of(counters[arrival_index]),
                            seq,
                            dtype="handle",
                        )
                if mode == "pdl_post_wait":
                    T.pdl_trigger()
                for i in T.serial(count):
                    pack = index + i * stride
                    T.call_extern(
                        reduce_pack,
                        T.address_of(inp[0]),
                        T.call_extern(
                            "roce_ptr",
                            region_base + T.cast(recv_off, T.int64),
                            dtype="handle",
                        ),
                        T.cast(slot_bytes, T.int64),
                        seq,
                        T.address_of(out[0]),
                        T.cast(pack, T.int64) * PACK_BYTES,
                        dtype="handle",
                    )
                T.call_extern("roce_fence_sc_gpu", dtype="handle")
                T.sync_threads()
                if tx == 0:
                    T.call_extern(
                        "roce_tail",
                        T.address_of(counters[tail_index]),
                        T.cast(grid_x, T.uint32),
                        T.address_of(counters[0]),
                        seq,
                        dtype=T.uint32,
                    )

        return oneshot_allreduce

    kernel = program()

    def run(launch) -> None:
        kernel(
            launch.input,
            launch.output,
            launch.counters,
            int(launch.region.data_ptr()),
            int(launch.size_packs),
            int(launch.nbytes),
            int(launch.recv_off),
            int(launch.flag_off),
            int(launch.send_off),
            int(launch.ctrl_off),
            int(launch.slot_bytes),
            int(launch.stage_index),
            int(launch.tail_index),
            int(launch.rung_index),
            int(launch.arrival_index),
            int(launch.grid_x),
            int(launch.trace_base),
        )

    return run


def _build_allgather_variant(
    mode: str,
    world_size: int,
    rank: int,
    threads: int,
    slots: int,
    flag_stride: int,
    hca_count: int,
    ring4: bool,
):
    from sparknet.oneshot._device import render_source
    from sparknet.oneshot._oneshot_tilelang import (
        _lanes,
        _pass_configs,
        _resident_blocks,
    )

    lanes, neighbor_lanes = _lanes(hca_count, ring4)
    source = render_source(
        world_size=world_size,
        rank=rank,
        slots=slots,
        flag_stride=flag_stride,
        hca_count=lanes,
        neighbor_lanes=neighbor_lanes,
        trace=None,
    )
    if mode not in ("pdl_post_doorbell_direct_send", "pdl_sync_only_direct_send"):
        source = source.replace(
            "if (input != send) {\n        roce_st_global_pack((roce_byte *)send + offset, p);\n    }",
            "roce_st_global_pack((roce_byte *)send + offset, p);",
        )
    resident = _resident_blocks(threads)
    use_pdl_sync = mode not in ("baseline_no_pdl", "pdl_trigger_only")

    @tilelang.jit(pass_configs=_pass_configs())
    def program():
        n_in, n_out, n_counters = T.dynamic("n_in, n_out, n_counters")

        @T.prim_func
        def oneshot_allgather(
            inp: T.Tensor((n_in,), T.int32),
            out: T.Tensor((n_out,), T.int32),
            counters: T.Tensor((n_counters,), T.int32),
            region_base: T.int64,
            shard_packs: T.int32,
            nbytes: T.int32,
            row_packs: T.int32,
            dst_stride_packs: T.int32,
            recv_off: T.int32,
            flag_off: T.int32,
            send_off: T.int32,
            ctrl_off: T.int32,
            slot_bytes: T.int32,
            stage_index: T.int32,
            tail_index: T.int32,
            rung_index: T.int32,
            arrival_index: T.int32,
            grid_x: T.int32,
        ):
            with T.Kernel(grid_x, threads=threads) as bx:
                T.annotate_min_blocks_per_sm(resident)
                T.import_source(source)
                if use_pdl_sync:
                    T.pdl_sync()
                if mode == "pdl_entry":
                    T.pdl_trigger()
                tx = T.get_thread_binding()
                epoch = T.alloc_var(T.uint32)
                epoch = T.call_extern(
                    "roce_ld_relaxed_gpu_u32", T.address_of(counters[0]), dtype=T.uint32
                )
                seq = T.alloc_var(T.uint32)
                seq = epoch + T.uint32(1)
                slot = T.alloc_var(T.int32)
                slot = T.cast(seq & T.uint32(1), T.int32)
                send_slot = T.alloc_var(T.int32)
                send_slot = send_off + slot * slot_bytes
                index = T.alloc_var(T.int32)
                index = bx * threads + tx
                stride = T.alloc_var(T.int32)
                stride = grid_x * threads
                count = T.alloc_var(T.int32)
                count = T.max(0, (shard_packs - index + stride - 1) // stride)
                for i in T.serial(count):
                    T.call_extern(
                        "roce_stage_gather_pack",
                        T.address_of(inp[0]),
                        T.call_extern(
                            "roce_ptr",
                            region_base + T.cast(send_slot, T.int64),
                            dtype="handle",
                        ),
                        index + i * stride,
                        row_packs,
                        T.address_of(out[0]),
                        dst_stride_packs,
                        dtype="handle",
                    )
                T.sync_threads()
                if tx == 0:
                    rang = T.alloc_var(T.uint32)
                    rang = T.call_extern(
                        "roce_doorbell",
                        T.address_of(counters[stage_index]),
                        T.cast(grid_x, T.uint32),
                        T.call_extern(
                            "roce_ptr",
                            region_base + T.cast(ctrl_off, T.int64),
                            dtype="handle",
                        ),
                        T.cast(nbytes, T.uint32),
                        seq,
                        dtype=T.uint32,
                    )
                    if rang != T.uint32(0):
                        T.call_extern(
                            "roce_st_release_gpu_u32",
                            T.address_of(counters[rung_index]),
                            seq,
                            dtype="handle",
                        )
                if mode in (
                    "pdl_trigger_only",
                    "pdl_post_doorbell",
                    "pdl_post_doorbell_direct_send",
                ):
                    T.pdl_trigger()
                T.call_extern(
                    "roce_wait_peers",
                    bx,
                    tx,
                    T.call_extern(
                        "roce_ptr",
                        region_base + T.cast(flag_off, T.int64),
                        dtype="handle",
                    ),
                    seq,
                    T.call_extern(
                        "roce_ptr",
                        region_base + T.cast(ctrl_off, T.int64),
                        dtype="handle",
                    ),
                    T.address_of(counters[rung_index]),
                    T.address_of(counters[arrival_index]),
                    dtype="handle",
                )
                T.sync_threads()
                if bx == 0:
                    if tx == 0:
                        T.call_extern(
                            "roce_st_release_gpu_u32",
                            T.address_of(counters[arrival_index]),
                            seq,
                            dtype="handle",
                        )
                if mode == "pdl_post_wait":
                    T.pdl_trigger()
                for i in T.serial(count):
                    T.call_extern(
                        "roce_gather_pack",
                        T.call_extern(
                            "roce_ptr",
                            region_base + T.cast(recv_off, T.int64),
                            dtype="handle",
                        ),
                        T.cast(slot_bytes, T.int64),
                        seq,
                        index + i * stride,
                        row_packs,
                        T.address_of(out[0]),
                        dst_stride_packs,
                        dtype="handle",
                    )
                T.call_extern("roce_fence_sc_gpu", dtype="handle")
                T.sync_threads()
                if tx == 0:
                    T.call_extern(
                        "roce_tail",
                        T.address_of(counters[tail_index]),
                        T.cast(grid_x, T.uint32),
                        T.address_of(counters[0]),
                        seq,
                        dtype="handle",
                    )

        return oneshot_allgather

    kernel = program()

    def run(launch) -> None:
        dst_stride = (
            int(launch.dst_peer_packs)
            if launch.dst_peer_packs
            else int(launch.row_packs)
        )
        kernel(
            launch.input,
            launch.output,
            launch.counters,
            int(launch.region.data_ptr()),
            int(launch.size_packs),
            int(launch.nbytes),
            int(launch.row_packs),
            dst_stride,
            int(launch.recv_off),
            int(launch.flag_off),
            int(launch.send_off),
            int(launch.ctrl_off),
            int(launch.slot_bytes),
            int(launch.stage_index),
            int(launch.tail_index),
            int(launch.rung_index),
            int(launch.arrival_index),
            int(launch.grid_x),
        )

    return run


def _measure_graph_us(
    graph: torch.cuda.CUDAGraph,
    runtime,
    stream: torch.cuda.Stream,
    *,
    warmup: int = 30,
    iters: int = 200,
    steps_per_replay: int = 2,
) -> dict[str, float]:
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            graph.replay()
    torch.cuda.synchronize()
    runtime.check_health()
    dist.barrier()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    with torch.cuda.stream(stream):
        for i in range(iters):
            start_events[i].record(stream)
            graph.replay()
            end_events[i].record(stream)
    torch.cuda.synchronize()
    runtime.check_health()
    dist.barrier()

    samples = [
        (s.elapsed_time(e) * 1000.0) / steps_per_replay
        for s, e in zip(start_events, end_events, strict=True)
    ]
    samples_sorted = sorted(samples)
    return {
        "mean_us": round(statistics.mean(samples), 3),
        "median_us": round(statistics.median(samples), 3),
        "p10_us": round(samples_sorted[int(0.10 * len(samples_sorted))], 3),
        "p90_us": round(samples_sorted[int(0.90 * len(samples_sorted))], 3),
        "min_us": round(samples_sorted[0], 3),
        "max_us": round(samples_sorted[-1], 3),
        "stdev_us": round(statistics.stdev(samples), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--boot-label", type=str, default="boot1")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()

    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    from sparknet import oneshot as roce
    from tests.kernels.deepseek_v41_pdl_chains import HIDDEN, WO_B, _block32
    from tests.kernels.test_deepseek_v41_tilelang_pdl import PROGRAMMATIC, _edges
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.models.deepseek_v4_1.tilelang import front, gather, linear, mhc, moe

    rt = roce.AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_size=2 << 20,
        max_gather_bytes=4 << 20,
    )
    rt.prepare((torch.bfloat16, torch.float32), padded_gather=True)

    modes = (
        "baseline_no_pdl",
        "pdl_sync_only",
        "pdl_trigger_only",
        "pdl_entry",
        "pdl_post_doorbell",
        "pdl_post_wait",
        "pdl_post_doorbell_direct_send",
        "pdl_sync_only_direct_send",
    )
    ar_launchers = {}
    ag_launchers = {}
    ring4 = rt.topology == "ring4"
    for mode in modes:
        ar_launchers[mode] = _build_allreduce_variant(
            mode,
            "bfloat16",
            world,
            rank,
            rt._threads,
            rt._layout.slots,
            rt._layout.flag_stride,
            rt.stripe_count,
            ring4,
        )
        ag_launchers[mode] = _build_allgather_variant(
            mode,
            world,
            rank,
            rt._threads,
            rt._layout.slots,
            rt._layout.flag_stride,
            rt.stripe_count,
            ring4,
        )
    torch.cuda.synchronize()
    dist.barrier()

    batch_sizes = (1, 4, 6, 16, 32)
    results: dict[str, object] = {
        "boot_label": args.boot_label,
        "hostname": socket.gethostname(),
        "rank": rank,
        "world_size": world,
        "topology": rt.topology,
        "hcas": list(rt.hca_names),
        "iters": args.iters,
        "warmup": args.warmup,
        "chains": {},
    }

    stream = torch.cuda.Stream(device=device)

    with set_current_vllm_config(VllmConfig()):
        gen = torch.Generator(device=device).manual_seed(1234 + rank)
        wo_b_layer = _block32(WO_B[0], WO_B[1], gen)
        wqa_wkv_layer = _block32(448, HIDDEN, gen)
        wkv_comp_layer = _block32(1280, HIDDEN, gen)
        post_kernel = mhc.post_streams(HIDDEN)
        q_weight = (
            torch.randn(1280, generator=gen, device=device) * 0.1 + 1.0
        ).bfloat16()
        kv_weight = (
            torch.randn(512, generator=gen, device=device) * 0.1 + 1.0
        ).bfloat16()
        cos_sin = torch.randn(
            (4096, 64), generator=gen, device=device, dtype=torch.float32
        )
        swa_cache = torch.zeros(
            (64, 64 * 528), dtype=torch.uint8, device=device
        )
        moe_plan = moe.MoeDispatchV41Plan(
            384, 6, 16, HIDDEN, out_dtype=torch.bfloat16, weighted=True
        )
        moe_plan.bind()

        for chain_name in (
            "wo_b_allreduce_mhc_post",
            "moe_combine_allreduce_mhc_post",
            "wqa_wkv_allgather_attention_front",
            "wkv_allgather_gather_columns_into",
        ):
            results["chains"][chain_name] = {}

        for rows in batch_sizes:
            # Allocate inputs & outputs for the 4 chains at this row count
            xq_wo, sf_wo = [
                torch.randint(1, 120, shape, generator=gen, dtype=dtype, device=device).view(
                    dtype
                )
                if dtype == torch.uint8
                else (
                    torch.randn(shape, generator=gen, device=device) * 0.2
                ).to(dtype)
                for shape, dtype in linear._activation_specs(
                    rows, WO_B[1], wo_b_layer.tilelang_decode_rows
                )
            ]
            xq_qa, sf_qa = [
                torch.randint(1, 120, shape, generator=gen, dtype=dtype, device=device).view(
                    dtype
                )
                if dtype == torch.uint8
                else (
                    torch.randn(shape, generator=gen, device=device) * 0.2
                ).to(dtype)
                for shape, dtype in linear._activation_specs(
                    rows, HIDDEN, wqa_wkv_layer.tilelang_decode_rows
                )
            ]
            wo_bufs = [
                torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device),
                torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device),
            ]
            ar_out_bufs = [
                torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device),
                torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device),
            ]
            residual_in = torch.randn(
                (rows, 4, HIDDEN), generator=gen, device=device, dtype=torch.bfloat16
            )
            post_in = torch.rand(
                (rows, 4), generator=gen, device=device, dtype=torch.float32
            )
            comb_in = torch.softmax(
                torch.randn(
                    (rows, 4, 4), generator=gen, device=device, dtype=torch.float32
                ),
                dim=-1,
            )
            mhc_out_bufs = [
                torch.empty((rows, 4, HIDDEN), dtype=torch.bfloat16, device=device),
                torch.empty((rows, 4, HIDDEN), dtype=torch.bfloat16, device=device),
            ]

            # MoE combine inputs
            topk_idx = (
                torch.arange(rows * 6, device=device, dtype=torch.int64).view(rows, 6)
                % 384
            )
            topk_weights = torch.softmax(
                torch.randn((rows, 6), generator=gen, device=device, dtype=torch.float32),
                dim=-1,
            )
            moe_ws = [
                torch.zeros(shape, dtype=dtype, device=device)
                for shape, dtype in moe_plan.workspace_specs(rows)
            ]
            moe_layout = moe_plan.dispatch(topk_idx, topk_weights, moe_ws)
            routed_y = (
                torch.randn(
                    (moe_plan.tiles(rows) * moe_plan.block_M, HIDDEN),
                    generator=gen,
                    device=device,
                    dtype=torch.bfloat16,
                )
                * 0.1
            )
            shared_y = (
                torch.randn(
                    (rows, HIDDEN), generator=gen, device=device, dtype=torch.bfloat16
                )
                * 0.1
            )

            # AllGather + attention_front inputs
            qa_local_bufs = [
                torch.empty((rows, 448), dtype=torch.bfloat16, device=device),
                torch.empty((rows, 448), dtype=torch.bfloat16, device=device),
            ]
            qa_gathered_bufs = [
                torch.empty((rows, 1792), dtype=torch.bfloat16, device=device),
                torch.empty((rows, 1792), dtype=torch.bfloat16, device=device),
            ]
            positions = torch.arange(rows, device=device, dtype=torch.int64)
            slots_map = torch.arange(rows, device=device, dtype=torch.int64)
            qn_out = torch.empty((rows, 1280), dtype=torch.bfloat16, device=device)
            q_cast = linear.empty_mxfp8(qn_out)
            kv_out = torch.empty((rows, 512), dtype=torch.bfloat16, device=device)

            # AllGather (dim=0) + gather_columns_into inputs
            comp_local_bufs = [
                torch.empty((rows, 1280), dtype=torch.bfloat16, device=device),
                torch.empty((rows, 1280), dtype=torch.bfloat16, device=device),
            ]
            comp_gathered_bufs = [
                torch.empty((world * rows, 1280), dtype=torch.bfloat16, device=device),
                torch.empty((world * rows, 1280), dtype=torch.bfloat16, device=device),
            ]
            comp_out_bufs = [
                torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device),
                torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device),
            ]

            # Warmup non-collective kernels once at this row size
            linear._block32_gemm(wo_b_layer, xq_wo, sf_wo, wo_bufs[0], rows)
            post_kernel(wo_bufs[0], residual_in, post_in, comb_in, mhc_out_bufs[0])
            moe_plan.combine(
                routed_y, moe_layout, wo_bufs[0], topk_weights=topk_weights, shared=shared_y
            )
            linear._block32_gemm(wqa_wkv_layer, xq_qa, sf_qa, qa_local_bufs[0], rows)
            front.attention_front(
                qa_gathered_bufs[0],
                q_weight,
                kv_weight,
                1e-6,
                positions,
                cos_sin,
                swa_cache,
                slots_map,
                64,
                qn_out,
                q_cast.values,
                q_cast.scales,
                kv_out,
            )
            linear._block32_gemm(wkv_comp_layer, xq_qa, sf_qa, comp_local_bufs[0], rows)
            gather.gather_columns_into(
                comp_gathered_bufs[0], (comp_out_bufs[0],), world
            )
            torch.cuda.synchronize()
            dist.barrier()

            for chain_name in results["chains"]:
                results["chains"][chain_name][f"M={rows}"] = {}
                ref_output = None

                for mode in modes:
                    use_direct = mode in (
                        "pdl_post_doorbell_direct_send",
                        "pdl_sync_only_direct_send",
                    )
                    rt._launchers[torch.bfloat16] = ar_launchers[mode]
                    rt._gather_launcher = ag_launchers[mode]

                    def run_two_steps(chain_kind: str = chain_name, direct: bool = use_direct):
                        for step in range(2):
                            if chain_kind == "wo_b_allreduce_mhc_post":
                                buf = (
                                    rt.next_send_buffer((rows, HIDDEN), torch.bfloat16)
                                    if direct
                                    else wo_bufs[step]
                                )
                                linear._block32_gemm(wo_b_layer, xq_wo, sf_wo, buf, rows)
                                rt.all_reduce(buf, out=ar_out_bufs[step])
                                post_kernel(
                                    ar_out_bufs[step],
                                    residual_in,
                                    post_in,
                                    comb_in,
                                    mhc_out_bufs[step],
                                )
                            elif chain_kind == "moe_combine_allreduce_mhc_post":
                                buf = (
                                    rt.next_send_buffer((rows, HIDDEN), torch.bfloat16)
                                    if direct
                                    else wo_bufs[step]
                                )
                                moe_plan.combine(
                                    routed_y,
                                    moe_layout,
                                    buf,
                                    topk_weights=topk_weights,
                                    shared=shared_y,
                                )
                                rt.all_reduce(buf, out=ar_out_bufs[step])
                                post_kernel(
                                    ar_out_bufs[step],
                                    residual_in,
                                    post_in,
                                    comb_in,
                                    mhc_out_bufs[step],
                                )
                            elif chain_kind == "wqa_wkv_allgather_attention_front":
                                buf = (
                                    rt.next_send_buffer((rows, 448), torch.bfloat16)
                                    if direct
                                    else qa_local_bufs[step]
                                )
                                linear._block32_gemm(
                                    wqa_wkv_layer, xq_qa, sf_qa, buf, rows
                                )
                                rt.all_gather(buf, dim=-1, out=qa_gathered_bufs[step])
                                front.attention_front(
                                    qa_gathered_bufs[step],
                                    q_weight,
                                    kv_weight,
                                    1e-6,
                                    positions,
                                    cos_sin,
                                    swa_cache,
                                    slots_map,
                                    64,
                                    qn_out,
                                    q_cast.values,
                                    q_cast.scales,
                                    kv_out,
                                )
                            elif chain_kind == "wkv_allgather_gather_columns_into":
                                buf = (
                                    rt.next_send_buffer((rows, 1280), torch.bfloat16)
                                    if direct
                                    else comp_local_bufs[step]
                                )
                                linear._block32_gemm(
                                    wkv_comp_layer, xq_qa, sf_qa, buf, rows
                                )
                                rt.all_gather(buf, dim=0, out=comp_gathered_bufs[step])
                                gather.gather_columns_into(
                                    comp_gathered_bufs[step],
                                    (comp_out_bufs[step],),
                                    world,
                                )

                    # 1 eager warmup run (2 collectives -> preserves slot parity)
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        run_two_steps()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                    rt.check_health()
                    dist.barrier()

                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph, stream=stream), rt.capture(stream=stream):
                        run_two_steps()
                    torch.cuda.synchronize()
                    dist.barrier()

                    edges = _edges(graph)
                    prog_edges = sum(1 for _, _, k in edges if k == PROGRAMMATIC)
                    if mode in ("baseline_no_pdl", "pdl_trigger_only"):
                        assert prog_edges < len(edges), f"expected non-programmatic collective edges in {mode}, got {edges}"
                    else:
                        assert prog_edges == len(edges), f"expected all {len(edges)} edges to be programmatic in {mode}, got {edges}"

                    stats = _measure_graph_us(
                        graph,
                        rt,
                        stream,
                        warmup=args.warmup,
                        iters=args.iters,
                        steps_per_replay=2,
                    )
                    stats["programmatic_edges"] = prog_edges
                    stats["total_edges"] = len(edges)

                    # Verify bit-identical output against baseline_no_pdl
                    if chain_name in (
                        "wo_b_allreduce_mhc_post",
                        "moe_combine_allreduce_mhc_post",
                    ):
                        cur_out = mhc_out_bufs[1].clone()
                    elif chain_name == "wqa_wkv_allgather_attention_front":
                        cur_out = torch.cat([qn_out, kv_out], dim=-1).clone()
                    else:
                        cur_out = comp_out_bufs[1].clone()

                    if ref_output is None:
                        ref_output = cur_out
                        stats["bit_exact_vs_baseline"] = True
                    else:
                        exact = bool(
                            torch.equal(
                                cur_out.view(torch.uint16),
                                ref_output.view(torch.uint16),
                            )
                        )
                        assert exact, f"Bit mismatch in {chain_name} M={rows} {mode}"
                        stats["bit_exact_vs_baseline"] = exact

                    results["chains"][chain_name][f"M={rows}"][mode] = stats
                    if rank == 0:
                        base_med = results["chains"][chain_name][f"M={rows}"][
                            "baseline_no_pdl"
                        ]["median_us"]
                        delta = stats["median_us"] - base_med
                        pct = (delta / base_med) * 100.0 if base_med else 0.0
                        print(
                            f"[{args.boot_label}] {chain_name:34s} M={rows:2d} {mode:30s}: "
                            f"median={stats['median_us']:6.2f} us (mean={stats['mean_us']:6.2f} us, "
                            f"p10={stats['p10_us']:6.2f}, p90={stats['p90_us']:6.2f}) "
                            f"delta={delta:+6.2f} us ({pct:+5.1f}%) prog_edges={prog_edges}/{len(edges)}",
                            flush=True,
                        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    rt.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
