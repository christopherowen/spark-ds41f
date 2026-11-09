#!/usr/bin/env python3
"""PR #979 own mark on the integrated stack (4 ranks, one per node).

Workload of the PR's bench_sp_focused.py (sha256 bd22e395..., left on sp2..sp4 /tmp by the
PR's run): T in {160..6144} (+8192), Engram wkv [6400, 6144] BF16 projection, then() =
RankOrderParts([N, L, 5120]).sum(), MXFP8 gather of values [L, 5120] u8, scales [L, 160] u8,
top-k weights [L, 6] f32 and ids [L, 6] i32, the same per-T/per-rank seeds, 12 warmup and 40
timed iterations with a barrier before each and CUDA events on the current stream.

Unlike the PR's script, which times in-script reimplementations beside the image's 1107
methods, this times the tree's own methods, so each arm measures its own sp_prefill.py:
  project_exchange_rows: sp.project_exchange_rows(project, (6400,), bf16, dev,
                         slices=sp_prefill.engram_slices(T), then=then)  (engram.py's call)
  gather_mxfp8:          sp.gather_mxfp8(x_q, weights, ids)               (kernels.py's call)
Beside each, an in-process anchor that is the same code in both arms (the PR's p0_1107:
serial project + then + sp.exchange_rows; and 1107's grouped PyNCCL all_gather written
inline), for drift between runs and an in-process bit-exactness check. Output hashes are
printed so the two arms can be compared bit for bit. The engram weight is seeded (the PR's
script draws it from the unseeded default generator) so both arms see the same weight.
"""

import argparse
import hashlib
import json
import os
import random
import socket
import time

import torch
import torch.distributed as dist

TOKENS = [160, 256, 512, 1024, 2048, 4096, 6144, 8192]


def stats(values):
    v = sorted(values)
    n = len(v)
    rng = random.Random(979)
    meds = sorted(sorted(rng.choice(v) for _ in range(n))[n // 2] for _ in range(2000))
    return {
        "median_us": round(v[n // 2], 2),
        "mean_us": round(sum(v) / n, 2),
        "p10_us": round(v[int(n * 0.1)], 2),
        "p90_us": round(v[int(n * 0.9)], 2),
        "min_us": round(v[0], 2),
        "median_ci95_us": [round(meds[int(0.025 * len(meds))], 2), round(meds[int(0.975 * len(meds))], 2)],
    }


def measure(fn, world, warmup=12, iters=40):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        dist.barrier()
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev) * 1000.0)
    per_rank = [None] * world
    dist.all_gather_object(per_rank, times)
    max_rank = [max(r[i] for r in per_rank) for i in range(iters)]
    return {"rank0": stats(per_rank[0]), "max_rank": stats(max_rank), "raw_us": per_rank}


def digest(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def warm_gpu(device, min_ms=200.0):
    a = torch.randn((4096, 4096), dtype=torch.bfloat16, device=device)
    b = torch.randn((4096, 4096), dtype=torch.bfloat16, device=device)
    c = torch.empty_like(a)
    torch.matmul(a, b, out=c)  # cuBLAS handle and kernel selection, outside the timed work
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Queue a batch, then time it on the GPU: only back-to-back GEMMs count as warm work.
    n = 0
    elapsed = 0.0
    while elapsed < min_ms:
        start.record()
        for _ in range(32):
            torch.matmul(a, b, out=c)
        end.record()
        torch.cuda.synchronize()
        n += 32
        elapsed += start.elapsed_time(end)
    return {"gemms_4096": n, "gpu_busy_ms": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True, choices=["ring4", "direct"])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    # The PR's bench_sp_focused.py topology settings, verbatim.
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
            world_size=world, rank=rank, local_rank=0,
            distributed_init_method="env://", backend="gloo",
        )
        ps.initialize_model_parallel(tensor_model_parallel_size=world)

    dev_comm = ps.get_tp_group().device_communicator
    pynccl_comm = dev_comm.pynccl_comm
    roce_comm = dev_comm.b12x_ar_comm
    runtime = roce_comm._runtime

    from vllm.models.deepseek_v4_1 import sp_prefill
    from vllm.models.deepseek_v4_1.tilelang.collectives import RankOrderParts
    from vllm.models.deepseek_v4_1.tilelang.linear import MXFP8Rows, mxfp8_rows

    with open(sp_prefill.__file__, "rb") as f:
        sp_sha = hashlib.sha256(f.read()).hexdigest()
    ident = {
        "host": socket.gethostname(), "rank": rank, "gpu": torch.cuda.get_device_name(device),
        "sp_prefill_sha256": sp_sha, "ENGRAM_SLICE_TOKENS": sp_prefill.ENGRAM_SLICE_TOKENS,
        "ring4_relay": bool(ring4_relay()), "slot_bytes": int(runtime._slot_bytes),
        "max_gather_bytes": int(runtime.max_gather_bytes), "roce_disabled": bool(roce_comm.disabled),
        "tp_pynccl_comm_active": sp_prefill._tp_pynccl_comm() is not None,
        "torch": torch.__version__,
    }
    idents = [None] * world
    dist.all_gather_object(idents, ident)
    warm = warm_gpu(device)
    warms = [None] * world
    dist.all_gather_object(warms, warm)
    results = {
        "label": args.label, "arm": args.arm, "topology": args.topology, "world": world,
        "ranks": idents, "gpu_warm": warms,
        "workload": "bench_sp_focused.py shapes and seeds; warmup 12, iters 40; engram weight seeded 979",
        "project_exchange_rows": {}, "gather_mxfp8": {},
    }
    if rank == 0:
        print(f"[{args.label}] arm={args.arm} topology={args.topology} ident={json.dumps(idents)} warm={warms}", flush=True)

    main_stream = torch.cuda.current_stream(device)
    wgen = torch.Generator(device=device).manual_seed(979)
    w_engram = torch.randn((6400, 6144), generator=wgen, dtype=torch.bfloat16, device=device) * 0.02

    for T in TOKENS:
        sp = sp_prefill.SPRows(num_tokens=T, world_size=world, rank=rank, unreduced=True)
        local = sp.local_rows
        real_me = sp.real_rows
        gen = torch.Generator(device=device).manual_seed(1000 + T + rank)
        rows_in = torch.randn((T, 6144), generator=gen, dtype=torch.bfloat16, device=device)
        prev_packed = torch.randn((world, local, 5120), generator=gen, dtype=torch.bfloat16, device=device)
        then_out = torch.empty((local, 5120), dtype=torch.bfloat16, device=device)
        slices = sp_prefill.engram_slices(T)

        def project_fn(start, stop, dest):
            torch.matmul(rows_in[start:stop], w_engram.T, out=dest)

        def then_fn():
            then_out.copy_(RankOrderParts(prev_packed).sum())

        def run_anchor_p0():  # the PR's p0_1107: identical code in both arms
            local_kv = torch.empty((T, 6400), dtype=torch.bfloat16, device=device)
            project_fn(0, T, local_kv)
            then_fn()
            return sp.exchange_rows(local_kv)

        def run_integrated():  # engram.py's call into this tree's SPRows
            return sp.project_exchange_rows(
                project_fn, (6400,), torch.bfloat16, device, slices=slices, then=then_fn
            )

        probe_kv = torch.empty((T, 6400), dtype=torch.bfloat16, device=device)
        route = {
            "engram_slices": slices,
            "roce_should_all_to_all_rows": bool(roce_comm.should_all_to_all_rows(probe_kv, local)),
        }
        del probe_kv
        ref = run_anchor_p0()
        then_ref = then_out.clone()
        got = run_integrated()
        torch.cuda.synchronize()
        equal = bool(torch.equal(ref[:, :real_me].view(torch.int16), got[:, :real_me].view(torch.int16)))
        then_equal = bool(torch.equal(then_ref.view(torch.int16), then_out.view(torch.int16)))
        h_out = digest(got[:, :real_me])
        h_then = digest(then_out)
        del ref, got
        t_anchor = measure(run_anchor_p0, world)
        t_int = measure(run_integrated, world)
        checks = [None] * world
        dist.all_gather_object(checks, {"equal_to_anchor": equal, "then_equal": then_equal,
                                        "sha256_out": h_out, "sha256_then": h_then, "route": route})
        results["project_exchange_rows"][str(T)] = {
            "anchor_p0_serial": t_anchor, "integrated": t_int, "per_rank": checks,
        }
        if rank == 0:
            print(f"[ProjEx T={T}] slices={slices} anchor_p0={t_anchor['max_rank']['median_us']}us "
                  f"integrated={t_int['max_rank']['median_us']}us (max-rank medians; rank0 "
                  f"{t_anchor['rank0']['median_us']}/{t_int['rank0']['median_us']}) "
                  f"equal={[c['equal_to_anchor'] for c in checks]}", flush=True)
        del rows_in, prev_packed, then_out
        torch.cuda.empty_cache()

    for T in TOKENS:
        sp = sp_prefill.SPRows(num_tokens=T, world_size=world, rank=rank, unreduced=True)
        local = sp.local_rows
        local_q = max(local, 128)
        gen = torch.Generator(device=device).manual_seed(3000 + T + rank)
        vals = torch.randint(0, 256, (local_q, 5120), generator=gen, dtype=torch.uint8, device=device)
        scales = torch.randint(1, 200, (local_q, 160), generator=gen, dtype=torch.uint8, device=device)
        weights = torch.randn((local, 6), generator=gen, dtype=torch.float32, device=device)
        ids = torch.randint(0, 256, (local, 6), generator=gen, dtype=torch.int32, device=device)
        x_q = MXFP8Rows(rows=torch.empty((local, 5120), dtype=torch.bfloat16, device=device),
                        values=vals, scales=scales)
        rows_q = mxfp8_rows(T)

        def run_integrated_g():
            return sp.gather_mxfp8(x_q, weights, ids)

        def run_anchor_g0():  # 1107's grouped PyNCCL all_gather, inline
            inputs = (vals[:local], scales[:local], weights, ids)
            limits = (rows_q, rows_q, T, T)
            outs, pairs = [], []
            for x, rows in zip(inputs, limits):
                inp = x.contiguous()
                out = torch.empty((max(sp.padded_rows, rows), *inp.shape[1:]), dtype=inp.dtype, device=device)
                outs.append(out[:rows])
                pairs.append((out[: sp.padded_rows], inp))
            pynccl_comm.group_start()
            for dst, src in pairs:
                pynccl_comm.all_gather(dst, src)
            pynccl_comm.group_end()
            return outs

        route = {
            "values_fit_roce": bool(roce_comm.should_all_gather(vals[:local], 0)),
            "all_fit_roce": all(bool(roce_comm.should_all_gather(x, 0)) for x in (vals[:local], scales[:local], weights, ids)),
        }
        qa, wa, ia = run_integrated_g()
        rv, rs, rw, ri = run_anchor_g0()
        torch.cuda.synchronize()
        equal = bool(torch.equal(qa.values[:T], rv[:T]) and torch.equal(qa.scales[:T], rs[:T])
                     and torch.equal(wa.view(torch.int32), rw.view(torch.int32)) and torch.equal(ia, ri))
        h = digest(qa.values[:T], qa.scales[:T], wa, ia)
        t_anchor = measure(run_anchor_g0, world)
        t_int = measure(run_integrated_g, world)
        checks = [None] * world
        dist.all_gather_object(checks, {"equal_to_anchor": equal, "sha256_out": h, "route": route})
        results["gather_mxfp8"][str(T)] = {
            "anchor_g0_pynccl_group": t_anchor, "integrated": t_int, "per_rank": checks,
        }
        if rank == 0:
            print(f"[Gather T={T}] fit_roce={route['all_fit_roce']} anchor_g0={t_anchor['max_rank']['median_us']}us "
                  f"integrated={t_int['max_rank']['median_us']}us (max-rank medians; rank0 "
                  f"{t_anchor['rank0']['median_us']}/{t_int['rank0']['median_us']}) "
                  f"equal={[c['equal_to_anchor'] for c in checks]}", flush=True)

    if rank == 0:
        print("=====RESULT-JSON-BEGIN=====", flush=True)
        print(json.dumps(results), flush=True)
        print("=====RESULT-JSON-END=====", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
