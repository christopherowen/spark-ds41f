#!/usr/bin/env python3
"""Focused 2-boot 4-rank RoCE benchmark for project_exchange_rows and gather_many."""

import argparse
import json
import os
import torch
import torch.distributed as dist


def _sync_measure(fn, warmup=12, iters=40):
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


def _project_contiguous_peers(project, rank, world, local, tokens, lo, hi, send_slice):
    """Project peer rows in at most two contiguous ranges (< rank and > rank) when lo==0 and hi==local,
    or per-owner when slicing sub-blocks."""
    slice_rows = hi - lo
    slice_total = 0
    if lo == 0 and hi == local:
        # Prefix peers: 0 .. rank - 1
        prefix_end = min(tokens, rank * local)
        if prefix_end > 0:
            project(0, prefix_end, send_slice.view(world * local, -1)[:prefix_end])
        # Suffix peers: rank + 1 .. world - 1
        suffix_start = (rank + 1) * local
        if tokens > suffix_start:
            project(
                suffix_start,
                tokens,
                send_slice.view(world * local, -1)[suffix_start:tokens],
            )
        return tokens
    for owner in range(world):
        real = max(0, min(hi, tokens - owner * local) - lo)
        slice_total += real
        if owner != rank and real:
            start = owner * local + lo
            project(start, start + real, send_slice[owner, :real])
    return slice_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="ring4", choices=["ring4", "direct"])
    parser.add_argument("--out", default="/tmp/bench_sp_focused_out.json")
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

    from vllm.models.deepseek_v4_1 import sp_prefill
    from vllm.models.deepseek_v4_1.tilelang.collectives import RankOrderParts
    from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows, mxfp8_rows
    from vllm.models.deepseek_v4_1.tilelang.mhc import _mhc_post

    main_stream = torch.cuda.current_stream(device)
    side_stream = sp_prefill._side_stream(device)

    token_counts = [160, 256, 512, 1024, 2048, 4096, 6144]
    results = {
        "topology": args.topology,
        "ring4_relay": bool(ring4_relay()),
        "slot_bytes": int(runtime._slot_bytes),
        "max_gather_bytes": int(runtime.max_gather_bytes),
        "project_exchange_rows": {},
        "gather_mxfp8": {},
    }

    # Weight for Engram wkv [6400, 6144]
    w_engram = torch.randn((6400, 6144), dtype=torch.bfloat16, device=device) * 0.02

    for T in token_counts:
        sp = sp_prefill.SPRows(num_tokens=T, world_size=world, rank=rank, unreduced=True)
        local = sp.local_rows
        gen = torch.Generator(device=device).manual_seed(1000 + T + rank)
        rows_in = torch.randn((T, 6144), generator=gen, dtype=torch.bfloat16, device=device)
        prev_packed = torch.randn((world, local, 5120), generator=gen, dtype=torch.bfloat16, device=device)
        then_out = torch.empty((local, 5120), dtype=torch.bfloat16, device=device)

        def project_fn(start, stop, dest):
            torch.matmul(rows_in[start:stop], w_engram.T, out=dest)

        def then_fn():
            then_out.copy_(RankOrderParts(prev_packed).sum())

        # Baseline 1107: serial project(0, T) + serial then() + serial exchange_rows
        def run_p0_1107():
            local_kv = torch.empty((T, 6400), dtype=torch.bfloat16, device=device)
            project_fn(0, T, local_kv)
            then_fn()
            return sp.exchange_rows(local_kv)

        # P1: 1-GEMM project(0, T) on main, exchange on side_stream overlapped with then() on main
        def run_p1_1gemm_side_then():
            local_kv = torch.empty((T, 6400), dtype=torch.bfloat16, device=device)
            project_fn(0, T, local_kv)
            if roce_comm.should_all_to_all_rows(local_kv, local):
                side_stream.wait_stream(main_stream)
                with torch.cuda.stream(side_stream):
                    out = roce_comm.all_to_all_rows(local_kv, local)
                then_fn()
                main_stream.wait_stream(side_stream)
                return out
            out = torch.empty((world, local, 6400), dtype=torch.bfloat16, device=device)
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                pynccl_comm.all_to_all_rows(out, local_kv, local, stream=side_stream, copy_self=True)
            then_fn()
            main_stream.wait_stream(side_stream)
            return out

        # P2: Coalesced 1-slice (<=2 peer GEMMs + side exchange copy_self=False overlapped with self GEMM + then())
        def run_p2_coalesced_1slice():
            out = torch.empty((world, local, 6400), dtype=torch.bfloat16, device=device)
            send = torch.empty((world, local, 6400), dtype=torch.bfloat16, device=device)
            slice_total = _project_contiguous_peers(project_fn, rank, world, local, T, 0, local, send)
            if slice_total:
                side_stream.wait_stream(main_stream)
                with torch.cuda.stream(side_stream):
                    flat_send = send.view(world * local, 6400)[:slice_total]
                    pynccl_comm.all_to_all_rows(
                        out, flat_send, local, stream=side_stream, copy_self=False
                    )
            real_me = sp.real_rows
            if real_me:
                start = rank * local
                project_fn(start, start + real_me, out[rank, :real_me])
            then_fn()
            main_stream.wait_stream(side_stream)
            return out

        # P3: 2-slice overlapped (copy_self=False)
        def run_p3_sliced2():
            slices = 2
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

        ref = run_p0_1107()
        o1 = run_p1_1gemm_side_then()
        o2 = run_p2_coalesced_1slice()
        o3 = run_p3_sliced2()
        real_me = sp.real_rows
        assert torch.equal(ref[:, :real_me].view(torch.int16), o1[:, :real_me].view(torch.int16))
        if T >= 2048:
            assert torch.equal(ref[:, :real_me].view(torch.int16), o2[:, :real_me].view(torch.int16))
            assert torch.equal(ref[:, :real_me].view(torch.int16), o3[:, :real_me].view(torch.int16))
        else:
            assert torch.allclose(ref[:, :real_me], o2[:, :real_me], atol=1e-1, rtol=2e-2)
            assert torch.allclose(ref[:, :real_me], o3[:, :real_me], atol=1e-1, rtol=2e-2)

        t0 = _sync_measure(run_p0_1107)
        t1 = _sync_measure(run_p1_1gemm_side_then)
        t2 = _sync_measure(run_p2_coalesced_1slice)
        t3 = _sync_measure(run_p3_sliced2)
        results["project_exchange_rows"][str(T)] = {
            "p0_1107": t0,
            "p1_1gemm_side_then": t1,
            "p2_coalesced_1slice": t2,
            "p3_sliced2": t3,
        }
        if rank == 0:
            print(
                f"[ProjEx T={T}] 1107={t0['median_us']}us | "
                f"p1_1gemm_side={t1['median_us']}us | "
                f"p2_coal_1slice={t2['median_us']}us | "
                f"p3_sliced2={t3['median_us']}us",
                flush=True,
            )

    # -------------------------------------------------------------------------
    # GATHER_MANY / GATHER_MXFP8 (with real DeepSeek V4.1 topk=6!)
    # -------------------------------------------------------------------------
    for T in token_counts:
        sp = sp_prefill.SPRows(num_tokens=T, world_size=world, rank=rank, unreduced=True)
        local = sp.local_rows
        local_q = max(local, 128)
        gen = torch.Generator(device=device).manual_seed(3000 + T + rank)
        vals = torch.randint(0, 256, (local_q, 5120), generator=gen, dtype=torch.uint8, device=device)
        scales = torch.randint(1, 200, (local_q, 160), generator=gen, dtype=torch.uint8, device=device)
        weights = torch.randn((local, 6), generator=gen, dtype=torch.float32, device=device)
        ids = torch.randint(0, 256, (local, 6), generator=gen, dtype=torch.int32, device=device)
        x_q = MXFP8Rows(
            rows=torch.empty((local, 5120), dtype=torch.bfloat16, device=device),
            values=vals,
            scales=scales,
        )

        # G0: 1107 baseline (PyNCCL grouped all_gather)
        def run_g0_1107():
            return sp.gather_mxfp8(x_q, weights, ids)

        # G1: Direct-out roce_comm per-tensor all_gather when all inputs eligible
        def run_g1_roce_direct_out():
            rows_q = mxfp8_rows(T)
            inputs = (vals[:local], scales[:local], weights, ids)
            limits = (rows_q, rows_q, T, T)
            if all(roce_comm.should_all_gather(x, 0) for x in inputs):
                outs = []
                for x, rows in zip(inputs, limits):
                    inp = x.contiguous()
                    alloc_rows = max(sp.padded_rows, rows)
                    out = torch.empty((alloc_rows, *inp.shape[1:]), dtype=inp.dtype, device=device)
                    runtime.all_gather(inp, dim=0, out=out[: sp.padded_rows])
                    outs.append(out[:rows])
                out_vals, out_scales, out_w, out_ids = outs
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

        # G2: Hybrid: roce_comm for any eligible tensor(s), or 2-call (values + packed tail)
        def run_g2_roce_2call():
            rows_q = mxfp8_rows(T)
            v_in = vals[:local]
            if roce_comm.should_all_gather(v_in, 0):
                alloc_v = max(sp.padded_rows, rows_q)
                out_vals = torch.empty((alloc_v, 5120), dtype=torch.uint8, device=device)
                runtime.all_gather(v_in, dim=0, out=out_vals[: sp.padded_rows])
                out_vals = out_vals[:rows_q]

                # Pack (scales[:local], weights, ids) -> 160 + 24 + 24 = 208 bytes/row
                packed_tail = torch.empty((local, 208), dtype=torch.uint8, device=device)
                packed_tail[:, :160].copy_(scales[:local])
                packed_tail[:, 160:184].copy_(weights.view(torch.uint8))
                packed_tail[:, 184:208].copy_(ids.view(torch.uint8))
                gathered_tail = runtime.all_gather(packed_tail, dim=0)

                out_scales = torch.empty((rows_q, 160), dtype=torch.uint8, device=device)
                out_scales[:T].copy_(gathered_tail[:T, :160])
                out_w = gathered_tail[:T, 160:184].contiguous().view(torch.float32).view(T, 6)
                out_ids = gathered_tail[:T, 184:208].contiguous().view(torch.int32).view(T, 6)
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

        qa, wa, ia = run_g0_1107()
        qb, wb, ib = run_g1_roce_direct_out()
        qc, wc, ic = run_g2_roce_2call()
        assert torch.equal(qa.values[:T], qb.values[:T])
        assert torch.equal(qa.scales[:T], qb.scales[:T])
        assert torch.equal(wa.view(torch.int32), wb.view(torch.int32))
        assert torch.equal(ia, ib)
        assert torch.equal(qa.values[:T], qc.values[:T])
        assert torch.equal(qa.scales[:T], qc.scales[:T])
        assert torch.equal(wa.view(torch.int32), wc.view(torch.int32))
        assert torch.equal(ia, ic)

        tg0 = _sync_measure(run_g0_1107)
        tg1 = _sync_measure(run_g1_roce_direct_out)
        tg2 = _sync_measure(run_g2_roce_2call)
        results["gather_mxfp8"][str(T)] = {
            "g0_1107_pynccl": tg0,
            "g1_roce_direct_out": tg1,
            "g2_roce_2call": tg2,
        }
        if rank == 0:
            print(
                f"[Gather T={T}] 1107_pynccl={tg0['median_us']}us | "
                f"g1_roce_direct={tg1['median_us']}us | "
                f"g2_roce_2call={tg2['median_us']}us",
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
