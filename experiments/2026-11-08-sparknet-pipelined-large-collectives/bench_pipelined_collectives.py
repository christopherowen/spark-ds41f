#!/usr/bin/env python3
"""Multi-node RoCE benchmark for SP prefill collectives:
Compares PyNCCL, Baseline Sparknet (multi-launch 64-block), and Candidate Sparknet
(single-kernel multi-stage pipelined RDMA with low SM occupancy) across:
  1. all_gather (dim=0)
  2. reduce_scatter (dim=0, rank-order FP32 accumulation)
  3. exchange (all-to-all peer chunks)
  4. all_to_all_rows
both standalone and overlapped on side_stream with a heavy GEMM on main_stream.
"""

import argparse
import json
import os
import time
import torch
import torch.distributed as dist


_SYNC_BUF = None
_SYNC_RT = None

def _sync_measure(fn, warmup=8, iters=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    times_us = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        if _SYNC_RT is not None and _SYNC_BUF is not None:
            _SYNC_RT.all_reduce(_SYNC_BUF, out=_SYNC_BUF)
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times_us.append(start_ev.elapsed_time(end_ev) * 1000.0)
    times_us.sort()
    n = len(times_us)
    return {
        "median_us": round(times_us[n // 2], 2),
        "mean_us": round(sum(times_us) / n, 2),
        "p10_us": round(times_us[int(n * 0.1)], 2),
        "p90_us": round(times_us[int(n * 0.9)], 2),
        "min_us": round(times_us[0], 2),
    }


def _baseline_multilaunch_allgather(runtime, inp, out):
    capturing = False
    launcher = runtime._resolve_gather_launcher(capturing)
    nbytes = inp.numel() * inp.element_size()
    local_rows = inp.shape[0]
    row_bytes = nbytes // local_rows
    max_rows = runtime._slot_bytes // row_bytes
    full_shard_packs = nbytes // 16
    runtime._order_stream(capturing)
    for lo in range(0, local_rows, max_rows):
        hi = min(local_rows, lo + max_rows)
        slice_bytes = (hi - lo) * row_bytes
        slice_packs = slice_bytes // 16
        out_slice = out[lo : (runtime.world_size - 1) * local_rows + hi]
        launcher(
            runtime._launch(
                inp[lo:hi],
                out_slice,
                slice_bytes,
                row_packs=slice_packs,
                dst_peer_packs=full_shard_packs,
                num_stages=1,
                stage_packs=slice_packs,
                max_blocks=64,
            )
        )
    runtime._mark_stream(capturing)
    return out


def _baseline_multilaunch_reducescatter(runtime, inp, out):
    capturing = False
    launcher = runtime._reducescatter_launcher(inp.dtype, capturing)
    nbytes = inp.numel() * inp.element_size()
    chunk_bytes = nbytes // runtime.world_size
    local_rows = inp.shape[0] // runtime.world_size
    row_bytes = chunk_bytes // local_rows
    max_rows = runtime._slot_bytes // (runtime.world_size * row_bytes)
    full_chunk_packs = chunk_bytes // 16
    runtime._order_stream(capturing)
    for lo in range(0, local_rows, max_rows):
        hi = min(local_rows, lo + max_rows)
        slice_chunk_bytes = (hi - lo) * row_bytes
        slice_packs = slice_chunk_bytes // 16
        inp_slice = inp[lo : (runtime.world_size - 1) * local_rows + hi]
        launcher(
            runtime._launch(
                inp_slice,
                out[lo:hi],
                slice_chunk_bytes,
                src_peer_packs=full_chunk_packs,
                num_stages=1,
                stage_packs=slice_packs,
                max_blocks=64,
            )
        )
    runtime._mark_stream(capturing)
    return out


def _baseline_multilaunch_exchange(runtime, chunks, parts, copy_self=False):
    capturing = False
    launcher = runtime._resolve_exchange_launcher(capturing)
    elem_size = chunks.element_size()
    chunk_bytes = chunks[0].numel() * elem_size
    src_peer_packs = (int(chunks.stride(0)) * elem_size) // 16
    dst_peer_packs = (int(parts.stride(0)) * elem_size) // 16
    local_rows = chunks.shape[1]
    row_bytes = chunk_bytes // local_rows
    max_rows = runtime._slot_bytes // (runtime.world_size * row_bytes)
    runtime._order_stream(capturing)
    for lo in range(0, local_rows, max_rows):
        hi = min(local_rows, lo + max_rows)
        c_slice = chunks[:, lo:hi]
        p_slice = parts[:, lo:hi]
        slice_chunk_bytes = (hi - lo) * row_bytes
        slice_packs = slice_chunk_bytes // 16
        src_words = runtime._span_words(c_slice, int(c_slice.stride(0)), c_slice[0].numel())
        dst_words = runtime._span_words(p_slice, int(p_slice.stride(0)), p_slice[0].numel())
        launcher(
            runtime._launch(
                src_words,
                dst_words,
                slice_chunk_bytes,
                src_peer_packs=src_peer_packs,
                dst_peer_packs=dst_peer_packs,
                copy_self=int(copy_self),
                raw_words=True,
                num_stages=1,
                stage_packs=slice_packs,
                max_blocks=64,
            )
        )
    runtime._mark_stream(capturing)
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/bench_pipelined_out.json")
    parser.add_argument("--stage-bytes", type=int, default=1048576)
    parser.add_argument("--collective-blocks", type=int, default=8)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    os.environ["SPARKNET_ROCE_TOPOLOGY"] = "direct"
    os.environ.pop("SPARKNET_ROCE_PEER_HCAS", None)
    os.environ["SPARKNET_ROCE_ALLGATHER_MAX_BYTES"] = "4194304"
    os.environ["SPARKNET_ROCE_STAGE_BYTES"] = str(args.stage_bytes)
    os.environ["SPARKNET_ROCE_COLLECTIVE_BLOCKS"] = str(args.collective_blocks)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    import datetime
    import torch.distributed.distributed_c10d as c10d
    c10d.default_pg_timeout = datetime.timedelta(seconds=60)
    import vllm.distributed.parallel_state as ps
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.device_communicators.pynccl import ring4_relay
    ring4_relay.cache_clear()

    ps._ENABLE_CUSTOM_ALL_REDUCE = True
    with set_current_vllm_config(VllmConfig()):
        ps.init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=0,
            distributed_init_method="env://",
            backend="gloo",
        )
        ps.initialize_model_parallel(tensor_model_parallel_size=world)

    tp_group = ps.get_tp_group()
    dev_comm = tp_group.device_communicator
    pynccl_comm = dev_comm.pynccl_comm
    roce_comm = dev_comm.b12x_ar_comm
    runtime = roce_comm._runtime
    global _SYNC_BUF, _SYNC_RT
    _SYNC_RT = runtime
    _SYNC_BUF = torch.zeros(8, dtype=torch.bfloat16, device=device)

    from vllm.models.deepseek_v4_1 import sp_prefill
    from vllm.models.deepseek_v4_1.tilelang.collectives import (
        exchange_into as tl_exchange_into,
        reduce_scatter_rank_order,
    )

    main_stream = torch.cuda.current_stream(device)
    side_stream = sp_prefill._side_stream(device)

    # Heavy compute GEMM on main_stream (~450 us on GB10: [1024, 2048] x [5120, 2048] BF16)
    gen = torch.Generator(device=device).manual_seed(999 + rank)
    gemm_a = torch.randn((1024, 2048), generator=gen, dtype=torch.bfloat16, device=device)
    gemm_w = torch.randn((5120, 2048), generator=gen, dtype=torch.bfloat16, device=device) * 0.02
    gemm_c = torch.empty((1024, 5120), dtype=torch.bfloat16, device=device)

    def run_gemm():
        torch.matmul(gemm_a, gemm_w.T, out=gemm_c)

    gemm_solo = _sync_measure(run_gemm)
    if rank == 0:
        print(
            f"[Rank 0] Initialized TP={world} slot_bytes={runtime._slot_bytes} "
            f"stage_bytes={args.stage_bytes} collective_blocks={args.collective_blocks} "
            f"gemm_solo={gemm_solo['median_us']}us",
            flush=True,
        )

    # Payload sizes (total collective bytes across world_size ranks)
    # Row width = 2048 bytes (1024 BF16 elements per row)
    row_elems = 1024  # 2048 bytes per row
    row_bytes = row_elems * 2
    # Target per-rank local_rows so world_size * local_rows * 2048 covers ~64KB to ~16MB
    local_row_counts = [8, 32, 64, 128, 256, 512, 1024, 1536, 2048]

    results = {
        "world_size": world,
        "stage_bytes": args.stage_bytes,
        "collective_blocks": args.collective_blocks,
        "gemm_solo": gemm_solo,
        "all_gather": {},
        "reduce_scatter": {},
        "exchange": {},
        "all_to_all_rows": {},
    }

    for local_rows in local_row_counts:
        total_rows = world * local_rows
        total_bytes = total_rows * row_bytes
        label = f"{total_bytes // 1024}KiB"

        # -----------------------------------------------------------------
        # 1. ALL-GATHER (shard: [local_rows, 1024] BF16 -> out: [total_rows, 1024] BF16)
        # -----------------------------------------------------------------
        ag_inp = torch.randn((local_rows, row_elems), generator=gen, dtype=torch.bfloat16, device=device)
        ag_out_nccl = torch.empty((total_rows, row_elems), dtype=torch.bfloat16, device=device)
        ag_out_base = torch.empty_like(ag_out_nccl)
        ag_out_cand = torch.empty_like(ag_out_nccl)

        def ag_nccl_solo():
            pynccl_comm.all_gather(ag_out_nccl, ag_inp)

        def ag_base_solo():
            _baseline_multilaunch_allgather(runtime, ag_inp, ag_out_base)

        def ag_cand_solo():
            runtime.all_gather(ag_inp, dim=0, out=ag_out_cand, sliced=True)

        def ag_nccl_ovlp():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                pynccl_comm.all_gather(ag_out_nccl, ag_inp, stream=side_stream)
            run_gemm()
            main_stream.wait_stream(side_stream)

        def ag_base_ovlp():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                _baseline_multilaunch_allgather(runtime, ag_inp, ag_out_base)
            run_gemm()
            main_stream.wait_stream(side_stream)

        def ag_cand_ovlp():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                runtime.all_gather(ag_inp, dim=0, out=ag_out_cand, sliced=True)
            run_gemm()
            main_stream.wait_stream(side_stream)

        ag_nccl_solo()
        ag_base_solo()
        ag_cand_solo()
        assert torch.equal(ag_out_nccl.view(torch.int16), ag_out_base.view(torch.int16)), f"AG base mismatch at {label}"
        assert torch.equal(ag_out_nccl.view(torch.int16), ag_out_cand.view(torch.int16)), f"AG cand mismatch at {label}"

        t_ag_nccl_solo = _sync_measure(ag_nccl_solo)
        t_ag_base_solo = _sync_measure(ag_base_solo)
        t_ag_cand_solo = _sync_measure(ag_cand_solo)
        t_ag_nccl_ovlp = _sync_measure(ag_nccl_ovlp)
        t_ag_base_ovlp = _sync_measure(ag_base_ovlp)
        t_ag_cand_ovlp = _sync_measure(ag_cand_ovlp)

        results["all_gather"][label] = {
            "total_bytes": total_bytes,
            "solo_pynccl": t_ag_nccl_solo,
            "solo_sparknet_base": t_ag_base_solo,
            "solo_sparknet_cand": t_ag_cand_solo,
            "ovlp_pynccl": t_ag_nccl_ovlp,
            "ovlp_sparknet_base": t_ag_base_ovlp,
            "ovlp_sparknet_cand": t_ag_cand_ovlp,
        }

        # -----------------------------------------------------------------
        # 2. REDUCE-SCATTER (inp: [total_rows, 1024] BF16 -> out: [local_rows, 1024] BF16)
        # -----------------------------------------------------------------
        rs_inp = torch.randn((total_rows, row_elems), generator=gen, dtype=torch.bfloat16, device=device)
        rs_out_nccl = torch.empty((local_rows, row_elems), dtype=torch.bfloat16, device=device)
        rs_out_base = torch.empty_like(rs_out_nccl)
        rs_out_cand = torch.empty_like(rs_out_nccl)

        def rs_nccl_solo():
            pynccl_comm.reduce_scatter(rs_out_nccl, rs_inp)

        def rs_base_solo():
            _baseline_multilaunch_reducescatter(runtime, rs_inp, rs_out_base)

        def rs_cand_solo():
            runtime.reduce_scatter(rs_inp, dim=0, out=rs_out_cand)

        def rs_nccl_ovlp():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                pynccl_comm.reduce_scatter(rs_out_nccl, rs_inp, stream=side_stream)
            run_gemm()
            main_stream.wait_stream(side_stream)

        def rs_base_ovlp():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                _baseline_multilaunch_reducescatter(runtime, rs_inp, rs_out_base)
            run_gemm()
            main_stream.wait_stream(side_stream)

        def rs_cand_ovlp():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                runtime.reduce_scatter(rs_inp, dim=0, out=rs_out_cand)
            run_gemm()
            main_stream.wait_stream(side_stream)

        rs_base_solo()
        rs_cand_solo()
        rs_ref = reduce_scatter_rank_order(pynccl_comm, rs_inp, rank, world)
        assert torch.equal(rs_ref.view(torch.int16), rs_out_base.view(torch.int16)), f"RS base mismatch at {label}"
        assert torch.equal(rs_ref.view(torch.int16), rs_out_cand.view(torch.int16)), f"RS cand mismatch at {label}"

        t_rs_nccl_solo = _sync_measure(rs_nccl_solo)
        t_rs_base_solo = _sync_measure(rs_base_solo)
        t_rs_cand_solo = _sync_measure(rs_cand_solo)
        t_rs_nccl_ovlp = _sync_measure(rs_nccl_ovlp)
        t_rs_base_ovlp = _sync_measure(rs_base_ovlp)
        t_rs_cand_ovlp = _sync_measure(rs_cand_ovlp)

        results["reduce_scatter"][label] = {
            "total_bytes": total_bytes,
            "solo_pynccl": t_rs_nccl_solo,
            "solo_sparknet_base": t_rs_base_solo,
            "solo_sparknet_cand": t_rs_cand_solo,
            "ovlp_pynccl": t_rs_nccl_ovlp,
            "ovlp_sparknet_base": t_rs_base_ovlp,
            "ovlp_sparknet_cand": t_rs_cand_ovlp,
        }

        # -----------------------------------------------------------------
        # 3. EXCHANGE (chunks: [world, local_rows, 1024] BF16 -> parts: [world, local_rows, 1024] BF16)
        # -----------------------------------------------------------------
        ex_inp = torch.randn((world, local_rows, row_elems), generator=gen, dtype=torch.bfloat16, device=device)
        ex_out_nccl = torch.empty_like(ex_inp)
        ex_out_base = torch.empty_like(ex_inp)
        ex_out_cand = torch.empty_like(ex_inp)

        def ex_nccl_solo():
            tl_exchange_into(pynccl_comm, ex_inp, ex_out_nccl, rank, world)

        def ex_base_solo():
            _baseline_multilaunch_exchange(runtime, ex_inp, ex_out_base, copy_self=False)

        def ex_cand_solo():
            runtime.exchange(ex_inp, out=ex_out_cand, copy_self=False)

        def ex_nccl_ovlp():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                tl_exchange_into(pynccl_comm, ex_inp, ex_out_nccl, rank, world)
            run_gemm()
            main_stream.wait_stream(side_stream)

        def ex_base_ovlp():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                _baseline_multilaunch_exchange(runtime, ex_inp, ex_out_base, copy_self=False)
            run_gemm()
            main_stream.wait_stream(side_stream)

        def ex_cand_ovlp():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                runtime.exchange(ex_inp, out=ex_out_cand, copy_self=False)
            run_gemm()
            main_stream.wait_stream(side_stream)

        ex_nccl_solo()
        ex_base_solo()
        ex_cand_solo()
        for p in range(world):
            if p != rank:
                assert torch.equal(ex_out_nccl[p].view(torch.int16), ex_out_base[p].view(torch.int16)), f"EX base mismatch at {label}"
                assert torch.equal(ex_out_nccl[p].view(torch.int16), ex_out_cand[p].view(torch.int16)), f"EX cand mismatch at {label}"

        t_ex_nccl_solo = _sync_measure(ex_nccl_solo)
        t_ex_base_solo = _sync_measure(ex_base_solo)
        t_ex_cand_solo = _sync_measure(ex_cand_solo)
        t_ex_nccl_ovlp = _sync_measure(ex_nccl_ovlp)
        t_ex_base_ovlp = _sync_measure(ex_base_ovlp)
        t_ex_cand_ovlp = _sync_measure(ex_cand_ovlp)

        results["exchange"][label] = {
            "total_bytes": total_bytes,
            "solo_pynccl": t_ex_nccl_solo,
            "solo_sparknet_base": t_ex_base_solo,
            "solo_sparknet_cand": t_ex_cand_solo,
            "ovlp_pynccl": t_ex_nccl_ovlp,
            "ovlp_sparknet_base": t_ex_base_ovlp,
            "ovlp_sparknet_cand": t_ex_cand_ovlp,
        }

        if rank == 0:
            print(
                f"[{label:>8}] "
                f"AG solo(nccl={t_ag_nccl_solo['median_us']:6.1f} base={t_ag_base_solo['median_us']:6.1f} cand={t_ag_cand_solo['median_us']:6.1f}) "
                f"ovlp(nccl={t_ag_nccl_ovlp['median_us']:6.1f} base={t_ag_base_ovlp['median_us']:6.1f} cand={t_ag_cand_ovlp['median_us']:6.1f}) | "
                f"RS solo(nccl={t_rs_nccl_solo['median_us']:6.1f} base={t_rs_base_solo['median_us']:6.1f} cand={t_rs_cand_solo['median_us']:6.1f}) "
                f"ovlp(nccl={t_rs_nccl_ovlp['median_us']:6.1f} base={t_rs_base_ovlp['median_us']:6.1f} cand={t_rs_cand_ovlp['median_us']:6.1f}) | "
                f"EX solo(nccl={t_ex_nccl_solo['median_us']:6.1f} base={t_ex_base_solo['median_us']:6.1f} cand={t_ex_cand_solo['median_us']:6.1f}) "
                f"ovlp(nccl={t_ex_nccl_ovlp['median_us']:6.1f} base={t_ex_base_ovlp['median_us']:6.1f} cand={t_ex_cand_ovlp['median_us']:6.1f})",
                flush=True,
            )

    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Rank 0] Saved results to {args.out}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
