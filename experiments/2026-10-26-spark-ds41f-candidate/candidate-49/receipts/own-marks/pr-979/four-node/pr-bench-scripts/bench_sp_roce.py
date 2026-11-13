#!/usr/bin/env python3
"""Multi-node 4-rank RoCE benchmark for SP prefill collectives and GEMM overlap."""

import argparse
import json
import os
import time
import torch
import torch.distributed as dist


def _sync_measure(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    times_us = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        dist.barrier()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="ring4", choices=["ring4", "direct"])
    parser.add_argument("--out", default="/tmp/bench_sp_roce_out.json")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if args.topology == "direct":
        os.environ["SPARKNET_ROCE_TOPOLOGY"] = "direct"
        os.environ.pop("SPARKNET_ROCE_PEER_HCAS", None)
        os.environ["SPARKNET_ROCE_ALLGATHER_MAX_BYTES"] = "4194304"
    else:
        os.environ["SPARKNET_ROCE_TOPOLOGY"] = "ring4"
        os.environ["SPARKNET_ROCE_PEER_HCAS"] = json.dumps(
            {
                str((rank - 1) % world): ["rocep1s0f0", "roceP2p1s0f0"],
                str((rank + 1) % world): ["rocep1s0f0", "roceP2p1s0f0"],
            }
        )
        os.environ["SPARKNET_ROCE_ALLGATHER_MAX_BYTES"] = "2097152"

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

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

    if rank == 0:
        print(
            f"[Rank 0] Initialized TP={world} topology={args.topology} "
            f"ring4_relay={ring4_relay()} "
            f"roce_disabled={roce_comm.disabled} "
            f"slot_bytes={runtime._slot_bytes} "
            f"max_gather_bytes={runtime.max_gather_bytes}",
            flush=True,
        )

    from vllm.models.deepseek_v4_1 import sp_prefill
    from vllm.models.deepseek_v4_1.tilelang.collectives import (
        RankOrderParts,
        exchange_into as tl_exchange_into,
    )
    from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows

    token_counts = [160, 256, 512, 1024, 2048, 4096, 6144]
    results = {
        "topology": args.topology,
        "ring4_relay": bool(ring4_relay()),
        "slot_bytes": int(runtime._slot_bytes),
        "max_gather_bytes": int(runtime.max_gather_bytes),
        "cases": {},
    }

    main_stream = torch.cuda.current_stream(device)
    side_stream = sp_prefill._side_stream(device)

    # -------------------------------------------------------------------------
    # CASE 1: SPRows.project_exchange_rows (Engram wkv [T, 6400] BF16)
    # -------------------------------------------------------------------------
    results["cases"]["case1_project_exchange_rows_6400"] = {}
    w_engram = torch.randn((6400, 6144), dtype=torch.bfloat16, device=device) * 0.02
    for T in token_counts:
        sp = sp_prefill.SPRows(num_tokens=T, world_size=world, rank=rank, unreduced=True)
        local = sp.local_rows
        gen = torch.Generator(device=device).manual_seed(1000 + T + rank)
        rows_in = torch.randn((T, 6144), generator=gen, dtype=torch.bfloat16, device=device)
        prev_packed = torch.randn((world, local, 4096), generator=gen, dtype=torch.bfloat16, device=device)
        then_out = torch.empty((local, 4096), dtype=torch.bfloat16, device=device)

        def project_fn(start, stop, dest):
            torch.matmul(rows_in[start:stop], w_engram.T, out=dest)

        def then_fn():
            then_out.copy_(RankOrderParts(prev_packed).sum())

        slices = sp_prefill.engram_slices(T)

        # 1a: 1107 baseline (serial full-row local_kv + then + exchange_rows)
        def run_1a_baseline():
            local_kv = torch.empty((T, 6400), dtype=torch.bfloat16, device=device)
            project_fn(0, T, local_kv)
            then_fn()
            return sp.exchange_rows(local_kv)

        # 1b: PyNCCL sliced side-stream overlap
        def run_1b_pynccl_side():
            out = torch.empty((world, local, 6400), dtype=torch.bfloat16, device=device)
            send = torch.empty((world * local, 6400), dtype=torch.bfloat16, device=device)
            bounds = [local * i // slices for i in range(slices + 1)]
            for lo, hi in zip(bounds, bounds[1:]):
                slice_rows = hi - lo
                if slice_rows == 0:
                    continue
                send_slice = send[world * lo : world * hi].view(world, slice_rows, 6400)
                slice_total = 0
                for owner in range(world):
                    real = max(0, min(hi, T - owner * local) - lo)
                    slice_total += real
                    if owner != rank and real:
                        start = owner * local + lo
                        project_fn(start, start + real, send_slice[owner, :real])
                if slice_total:
                    flat_send = send_slice.view(world * slice_rows, 6400)[:slice_total]
                    side_stream.wait_stream(main_stream)
                    with torch.cuda.stream(side_stream):
                        pynccl_comm.all_to_all_rows(
                            out[:, lo:hi], flat_send, slice_rows, stream=side_stream, copy_self=False
                        )
                real_me = max(0, min(hi, T - rank * local) - lo)
                if real_me:
                    start = rank * local + lo
                    project_fn(start, start + real_me, out[rank, lo : lo + real_me])
            then_fn()
            main_stream.wait_stream(side_stream)
            return out

        # 1c: Sparknet 1-slot + PyNCCL fallback on side-stream
        def run_1c_sparknet_1slot_side():
            out = torch.empty((world, local, 6400), dtype=torch.bfloat16, device=device)
            send = torch.empty((world * local, 6400), dtype=torch.bfloat16, device=device)
            bounds = [local * i // slices for i in range(slices + 1)]
            for lo, hi in zip(bounds, bounds[1:]):
                slice_rows = hi - lo
                if slice_rows == 0:
                    continue
                send_slice = send[world * lo : world * hi].view(world, slice_rows, 6400)
                slice_total = 0
                for owner in range(world):
                    real = max(0, min(hi, T - owner * local) - lo)
                    slice_total += real
                    if owner != rank and real:
                        start = owner * local + lo
                        project_fn(start, start + real, send_slice[owner, :real])
                if slice_total:
                    side_stream.wait_stream(main_stream)
                    with torch.cuda.stream(side_stream):
                        if slice_total == world * slice_rows and roce_comm.should_exchange(
                            send_slice, out[:, lo:hi]
                        ):
                            roce_comm.exchange_into(send_slice, out[:, lo:hi])
                        else:
                            flat_send = send_slice.view(world * slice_rows, 6400)[:slice_total]
                            pynccl_comm.all_to_all_rows(
                                out[:, lo:hi], flat_send, slice_rows, stream=side_stream, copy_self=False
                            )
                real_me = max(0, min(hi, T - rank * local) - lo)
                if real_me:
                    start = rank * local + lo
                    project_fn(start, start + real_me, out[rank, lo : lo + real_me])
            then_fn()
            main_stream.wait_stream(side_stream)
            return out

        # 1d: Sparknet multi-slice on side-stream
        def run_1d_sparknet_multislice_side():
            out = torch.empty((world, local, 6400), dtype=torch.bfloat16, device=device)
            send = torch.empty((world * local, 6400), dtype=torch.bfloat16, device=device)
            bounds = [local * i // slices for i in range(slices + 1)]
            for lo, hi in zip(bounds, bounds[1:]):
                slice_rows = hi - lo
                if slice_rows == 0:
                    continue
                send_slice = send[world * lo : world * hi].view(world, slice_rows, 6400)
                slice_total = 0
                for owner in range(world):
                    real = max(0, min(hi, T - owner * local) - lo)
                    slice_total += real
                    if owner != rank and real:
                        start = owner * local + lo
                        project_fn(start, start + real, send_slice[owner, :real])
                if slice_total:
                    side_stream.wait_stream(main_stream)
                    with torch.cuda.stream(side_stream):
                        if slice_total == world * slice_rows:
                            runtime.exchange(send_slice, out=out[:, lo:hi], copy_self=False)
                        else:
                            flat_send = send_slice.view(world * slice_rows, 6400)[:slice_total]
                            pynccl_comm.all_to_all_rows(
                                out[:, lo:hi], flat_send, slice_rows, stream=side_stream, copy_self=False
                            )
                real_me = max(0, min(hi, T - rank * local) - lo)
                if real_me:
                    start = rank * local + lo
                    project_fn(start, start + real_me, out[rank, lo : lo + real_me])
            then_fn()
            main_stream.wait_stream(side_stream)
            return out

        ref_out = run_1a_baseline()
        out_1b = run_1b_pynccl_side()
        out_1c = run_1c_sparknet_1slot_side()
        out_1d = run_1d_sparknet_multislice_side()
        real_me = sp.real_rows
        assert torch.equal(ref_out[:, :real_me].view(torch.int16), out_1b[:, :real_me].view(torch.int16))
        assert torch.equal(ref_out[:, :real_me].view(torch.int16), out_1c[:, :real_me].view(torch.int16))
        assert torch.equal(ref_out[:, :real_me].view(torch.int16), out_1d[:, :real_me].view(torch.int16))

        t_1a = _sync_measure(run_1a_baseline)
        t_1b = _sync_measure(run_1b_pynccl_side)
        t_1c = _sync_measure(run_1c_sparknet_1slot_side)
        t_1d = _sync_measure(run_1d_sparknet_multislice_side)
        results["cases"]["case1_project_exchange_rows_6400"][str(T)] = {
            "baseline_1107": t_1a,
            "pynccl_side": t_1b,
            "sparknet_1slot_side": t_1c,
            "sparknet_multislice_side": t_1d,
            "bit_exact": True,
        }
        if rank == 0:
            print(
                f"[Case 1 T={T}] 1107={t_1a['median_us']}us | "
                f"pynccl_side={t_1b['median_us']}us | "
                f"sparknet_1slot={t_1c['median_us']}us | "
                f"sparknet_ms={t_1d['median_us']}us",
                flush=True,
            )

    # -------------------------------------------------------------------------
    # CASE 2: WO-A MXFP8 (xq_send [4, L, 2048] E4M3 + sf_send [4, L, 64] uint8)
    #         Both standalone and overlapped with WO-B GEMM [L, 2048] x [5120, 2048]
    # -------------------------------------------------------------------------
    results["cases"]["case2_wo_mxfp8_exchange"] = {}
    w_wob = torch.randn((5120, 2048), dtype=torch.bfloat16, device=device) * 0.02
    for T in token_counts:
        sp = sp_prefill.SPRows(num_tokens=T, world_size=world, rank=rank, unreduced=True)
        local = sp.local_rows
        gen = torch.Generator(device=device).manual_seed(2000 + T + rank)
        xq_send = torch.randint(0, 256, (world, local, 2048), generator=gen, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
        sf_send = torch.randint(1, 200, (world, local, 64), generator=gen, dtype=torch.uint8, device=device)
        wob_in = torch.randn((local, 2048), generator=gen, dtype=torch.bfloat16, device=device)
        wob_out = torch.empty((local, 5120), dtype=torch.bfloat16, device=device)

        xq_recv_a = torch.empty_like(xq_send)
        sf_recv_a = torch.empty_like(sf_send)
        xq_recv_b = torch.empty_like(xq_send)
        sf_recv_b = torch.empty_like(sf_send)
        xq_recv_c = torch.empty_like(xq_send)
        sf_recv_c = torch.empty_like(sf_send)
        xq_recv_d = torch.empty_like(xq_send)
        sf_recv_d = torch.empty_like(sf_send)

        # Packed buffer for (xq, sf) -> shape (world, local, 2048 + 64) uint8
        packed_send = torch.empty((world, local, 2048 + 64), dtype=torch.uint8, device=device)
        packed_recv = torch.empty((world, local, 2048 + 64), dtype=torch.uint8, device=device)

        # 2a: 1107 baseline overlapped with WO-B GEMM
        def run_2a_baseline_ovlp():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                sp.exchange_into(xq_send, xq_recv_a)
                sp.exchange_into(sf_send, sf_recv_a)
            torch.matmul(wob_in, w_wob.T, out=wob_out)
            main_stream.wait_stream(side_stream)

        # 2b: Grouped PyNCCL when xq exceeds 1-slot (avoids PyNCCL + sequential Sparknet)
        def run_2b_grouped_ovlp():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                if roce_comm.should_exchange(xq_send, xq_recv_b) and roce_comm.should_exchange(sf_send, sf_recv_b):
                    roce_comm.exchange_into(xq_send, xq_recv_b)
                    roce_comm.exchange_into(sf_send, sf_recv_b)
                else:
                    pynccl_comm.group_start()
                    tl_exchange_into(pynccl_comm, xq_send, xq_recv_b, rank, world)
                    tl_exchange_into(pynccl_comm, sf_send, sf_recv_b, rank, world)
                    pynccl_comm.group_end()
            torch.matmul(wob_in, w_wob.T, out=wob_out)
            main_stream.wait_stream(side_stream)

        # 2c: Packed single-collective Sparknet when (xq+sf) <= slot_bytes, else grouped PyNCCL
        def run_2c_packed_ovlp():
            packed_send[:, :, :2048].copy_(xq_send.view(torch.uint8))
            packed_send[:, :, 2048:].copy_(sf_send)
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                if roce_comm.should_exchange(packed_send, packed_recv):
                    roce_comm.exchange_into(packed_send, packed_recv)
                else:
                    pynccl_comm.group_start()
                    tl_exchange_into(pynccl_comm, xq_send, xq_recv_c, rank, world)
                    tl_exchange_into(pynccl_comm, sf_send, sf_recv_c, rank, world)
                    pynccl_comm.group_end()
            torch.matmul(wob_in, w_wob.T, out=wob_out)
            main_stream.wait_stream(side_stream)
            if roce_comm.should_exchange(packed_send, packed_recv):
                xq_recv_c.view(torch.uint8).copy_(packed_recv[:, :, :2048])
                sf_recv_c.copy_(packed_recv[:, :, 2048:])

        # 2d: Multi-slice Sparknet on side_stream overlapped with WO-B GEMM
        def run_2d_sparknet_ms_ovlp():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                runtime.exchange(xq_send, out=xq_recv_d, copy_self=False)
                runtime.exchange(sf_send, out=sf_recv_d, copy_self=False)
            torch.matmul(wob_in, w_wob.T, out=wob_out)
            main_stream.wait_stream(side_stream)

        # Standalone versions (no GEMM)
        def run_2a_standalone():
            sp.exchange_into(xq_send, xq_recv_a)
            sp.exchange_into(sf_send, sf_recv_a)

        def run_2b_standalone():
            if roce_comm.should_exchange(xq_send, xq_recv_b) and roce_comm.should_exchange(sf_send, sf_recv_b):
                roce_comm.exchange_into(xq_send, xq_recv_b)
                roce_comm.exchange_into(sf_send, sf_recv_b)
            else:
                pynccl_comm.group_start()
                tl_exchange_into(pynccl_comm, xq_send, xq_recv_b, rank, world)
                tl_exchange_into(pynccl_comm, sf_send, sf_recv_b, rank, world)
                pynccl_comm.group_end()

        def run_2d_standalone():
            runtime.exchange(xq_send, out=xq_recv_d, copy_self=False)
            runtime.exchange(sf_send, out=sf_recv_d, copy_self=False)

        run_2a_baseline_ovlp()
        run_2b_grouped_ovlp()
        run_2c_packed_ovlp()
        run_2d_sparknet_ms_ovlp()
        for peer in range(world):
            if peer != rank:
                assert torch.equal(xq_recv_a[peer].view(torch.uint8), xq_recv_b[peer].view(torch.uint8))
                assert torch.equal(sf_recv_a[peer], sf_recv_b[peer])
                assert torch.equal(xq_recv_a[peer].view(torch.uint8), xq_recv_c[peer].view(torch.uint8))
                assert torch.equal(sf_recv_a[peer], sf_recv_c[peer])
                assert torch.equal(xq_recv_a[peer].view(torch.uint8), xq_recv_d[peer].view(torch.uint8))
                assert torch.equal(sf_recv_a[peer], sf_recv_d[peer])

        t_2a_ovlp = _sync_measure(run_2a_baseline_ovlp)
        t_2b_ovlp = _sync_measure(run_2b_grouped_ovlp)
        t_2c_ovlp = _sync_measure(run_2c_packed_ovlp)
        t_2d_ovlp = _sync_measure(run_2d_sparknet_ms_ovlp)
        t_2a_solo = _sync_measure(run_2a_standalone)
        t_2b_solo = _sync_measure(run_2b_standalone)
        t_2d_solo = _sync_measure(run_2d_standalone)

        results["cases"]["case2_wo_mxfp8_exchange"][str(T)] = {
            "ovlp_1107": t_2a_ovlp,
            "ovlp_grouped_pynccl": t_2b_ovlp,
            "ovlp_packed_sparknet": t_2c_ovlp,
            "ovlp_sparknet_ms": t_2d_ovlp,
            "solo_1107": t_2a_solo,
            "solo_grouped_pynccl": t_2b_solo,
            "solo_sparknet_ms": t_2d_solo,
            "bit_exact": True,
        }
        if rank == 0:
            print(
                f"[Case 2 T={T}] ovlp: 1107={t_2a_ovlp['median_us']}us | "
                f"grouped={t_2b_ovlp['median_us']}us | "
                f"packed={t_2c_ovlp['median_us']}us | "
                f"sparknet_ms={t_2d_ovlp['median_us']}us || "
                f"solo: 1107={t_2a_solo['median_us']}us | "
                f"grouped={t_2b_solo['median_us']}us | "
                f"sparknet_ms={t_2d_solo['median_us']}us",
                flush=True,
            )

    # -------------------------------------------------------------------------
    # CASE 3: SPRows.gather_mxfp8 / gather_many (MoE input gather)
    # -------------------------------------------------------------------------
    results["cases"]["case3_gather_mxfp8"] = {}
    for T in token_counts:
        sp = sp_prefill.SPRows(num_tokens=T, world_size=world, rank=rank, unreduced=True)
        local = sp.local_rows
        local_q = max(local, 128)
        gen = torch.Generator(device=device).manual_seed(3000 + T + rank)
        vals = torch.randint(0, 256, (local_q, 5120), generator=gen, dtype=torch.uint8, device=device)
        scales = torch.randint(1, 200, (local_q, 160), generator=gen, dtype=torch.uint8, device=device)
        weights = torch.randn((local, 8), generator=gen, dtype=torch.float32, device=device)
        ids = torch.randint(0, 256, (local, 8), generator=gen, dtype=torch.int32, device=device)
        x_q = MXFP8Rows(
            rows=torch.empty((local, 5120), dtype=torch.bfloat16, device=device),
            values=vals,
            scales=scales,
        )

        # 3a: 1107 baseline (PyNCCL grouped all_gather)
        def run_3a_baseline():
            return sp.gather_mxfp8(x_q, weights, ids)

        # 3b: Sparknet 4-call all_gather (when eligible)
        def run_3b_sparknet_4calls():
            q_rows = T if T > 128 else (64 if T <= 64 else 128)
            tensors = (vals[:local], scales[:local], weights, ids)
            padded_rows = (q_rows, q_rows)
            outputs = []
            for idx, t in enumerate(tensors):
                full_rows = max(sp.padded_rows, padded_rows[idx] if idx < len(padded_rows) else T)
                if roce_comm.should_all_gather(t, dim=0):
                    g = roce_comm.all_gather(t, dim=0)
                    if full_rows > sp.padded_rows:
                        buf = torch.empty((full_rows, *t.shape[1:]), dtype=t.dtype, device=device)
                        buf[:sp.padded_rows].copy_(g)
                        outputs.append(buf)
                    else:
                        outputs.append(g)
                else:
                    return sp.gather_mxfp8(x_q, weights, ids)
            out_vals, out_scales, out_w, out_ids = (
                out[: (padded_rows[idx] if idx < len(padded_rows) else T)]
                for idx, out in enumerate(outputs)
            )
            return (
                MXFP8Rows(
                    rows=torch.empty((T, 5120), dtype=torch.bfloat16, device=device),
                    values=out_vals,
                    scales=out_scales,
                ),
                out_w,
                out_ids,
            )

        # 3c: Sparknet packed single all_gather (when packed <= max_gather_bytes)
        def run_3c_sparknet_packed():
            q_rows = T if T > 128 else (64 if T <= 64 else 128)
            row_bytes = 5120 + 160 + 32 + 32
            total_bytes = sp.padded_rows * row_bytes
            if total_bytes <= runtime.max_gather_bytes:
                packed_local = torch.empty((local, row_bytes), dtype=torch.uint8, device=device)
                packed_local[:, :5120].copy_(vals[:local])
                packed_local[:, 5120:5280].copy_(scales[:local])
                packed_local[:, 5280:5312].copy_(weights.view(torch.uint8))
                packed_local[:, 5312:5344].copy_(ids.view(torch.uint8))
                gathered = roce_comm.all_gather(packed_local, dim=0)
                out_vals = torch.empty((q_rows, 5120), dtype=torch.uint8, device=device)
                out_scales = torch.empty((q_rows, 160), dtype=torch.uint8, device=device)
                out_vals[:T].copy_(gathered[:T, :5120])
                out_scales[:T].copy_(gathered[:T, 5120:5280])
                out_w = gathered[:T, 5280:5312].contiguous().view(torch.float32).view(T, 8)
                out_ids = gathered[:T, 5312:5344].contiguous().view(torch.int32).view(T, 8)
                return (
                    MXFP8Rows(
                        rows=torch.empty((T, 5120), dtype=torch.bfloat16, device=device),
                        values=out_vals,
                        scales=out_scales,
                    ),
                    out_w,
                    out_ids,
                )
            return sp.gather_mxfp8(x_q, weights, ids)

        q_a, w_a, i_a = run_3a_baseline()
        q_b, w_b, i_b = run_3b_sparknet_4calls()
        q_c, w_c, i_c = run_3c_sparknet_packed()
        assert torch.equal(q_a.values[:T], q_b.values[:T])
        assert torch.equal(q_a.scales[:T], q_b.scales[:T])
        assert torch.equal(w_a.view(torch.int32), w_b.view(torch.int32))
        assert torch.equal(i_a, i_b)
        assert torch.equal(q_a.values[:T], q_c.values[:T])
        assert torch.equal(q_a.scales[:T], q_c.scales[:T])
        assert torch.equal(w_a.view(torch.int32), w_c.view(torch.int32))
        assert torch.equal(i_a, i_c)

        t_3a = _sync_measure(run_3a_baseline)
        t_3b = _sync_measure(run_3b_sparknet_4calls)
        t_3c = _sync_measure(run_3c_sparknet_packed)
        results["cases"]["case3_gather_mxfp8"][str(T)] = {
            "baseline_1107_pynccl_group": t_3a,
            "sparknet_4calls": t_3b,
            "sparknet_packed": t_3c,
            "bit_exact": True,
        }
        if rank == 0:
            print(
                f"[Case 3 T={T}] 1107_pynccl={t_3a['median_us']}us | "
                f"sparknet_4calls={t_3b['median_us']}us | "
                f"sparknet_packed={t_3c['median_us']}us",
                flush=True,
            )

    # -------------------------------------------------------------------------
    # CASE 4: Disk Engram all_to_all_rows ([T, 1536] BF16)
    # -------------------------------------------------------------------------
    results["cases"]["case4_all_to_all_rows_1536"] = {}
    for T in token_counts:
        sp = sp_prefill.SPRows(num_tokens=T, world_size=world, rank=rank, unreduced=True)
        local = sp.local_rows
        gen = torch.Generator(device=device).manual_seed(4000 + T + rank)
        staged = torch.randn((T, 1536), generator=gen, dtype=torch.bfloat16, device=device)

        def run_4a_baseline():
            return dev_comm.all_to_all_rows(staged, local)

        def run_4b_sparknet_ms():
            if staged.shape[0] == world * local:
                return runtime.all_to_all_rows(staged, local)
            return dev_comm.all_to_all_rows(staged, local)

        out_4a = run_4a_baseline()
        out_4b = run_4b_sparknet_ms()
        real_me = sp.real_rows
        assert torch.equal(out_4a[:, :real_me].view(torch.int16), out_4b[:, :real_me].view(torch.int16))

        t_4a = _sync_measure(run_4a_baseline)
        t_4b = _sync_measure(run_4b_sparknet_ms)
        results["cases"]["case4_all_to_all_rows_1536"][str(T)] = {
            "baseline_1107": t_4a,
            "sparknet_ms": t_4b,
            "bit_exact": True,
        }
        if rank == 0:
            print(
                f"[Case 4 T={T}] 1107={t_4a['median_us']}us | "
                f"sparknet_ms={t_4b['median_us']}us",
                flush=True,
            )

    # -------------------------------------------------------------------------
    # CASE 5: SPRows._project_exchange ([4, L, 5120] BF16 overlapped with GEMM)
    # -------------------------------------------------------------------------
    results["cases"]["case5_project_exchange_5120"] = {}
    for T in token_counts:
        sp = sp_prefill.SPRows(num_tokens=T, world_size=world, rank=rank, unreduced=True)
        local = sp.local_rows
        gen = torch.Generator(device=device).manual_seed(5000 + T + rank)
        send_buf = torch.randn((world, local, 5120), generator=gen, dtype=torch.bfloat16, device=device)
        parts_a = torch.empty_like(send_buf)
        parts_b = torch.empty_like(send_buf)
        gemm_in = torch.randn((local, 2048), generator=gen, dtype=torch.bfloat16, device=device)
        gemm_out = torch.empty((local, 5120), dtype=torch.bfloat16, device=device)

        def run_5a_baseline():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                sp.exchange_into(send_buf, parts_a)
            torch.matmul(gemm_in, w_wob.T, out=gemm_out)
            main_stream.wait_stream(side_stream)

        def run_5b_sparknet_ms():
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                runtime.exchange(send_buf, out=parts_b, copy_self=False)
            torch.matmul(gemm_in, w_wob.T, out=gemm_out)
            main_stream.wait_stream(side_stream)

        run_5a_baseline()
        run_5b_sparknet_ms()
        for peer in range(world):
            if peer != rank:
                assert torch.equal(parts_a[peer].view(torch.int16), parts_b[peer].view(torch.int16))

        t_5a = _sync_measure(run_5a_baseline)
        t_5b = _sync_measure(run_5b_sparknet_ms)
        results["cases"]["case5_project_exchange_5120"][str(T)] = {
            "baseline_1107": t_5a,
            "sparknet_ms": t_5b,
            "bit_exact": True,
        }
        if rank == 0:
            print(
                f"[Case 5 T={T}] 1107={t_5a['median_us']}us | "
                f"sparknet_ms={t_5b['median_us']}us",
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
