#!/usr/bin/env python3
"""2-boot 3-rank TP3 NCCL collective benchmark on dgx1-3 (tp3-triangle).

Screens DS41F TP3 prefill and decode collective shapes across 2 fresh boots:
- all_gather_into_tensor (Engram SP exchange BF16, MoE MXFP8 gather, Attention front compressed gather)
- all_to_all_single (WO 2-slice local exchange, MLP / MoE all_to_all)
- reduce_scatter_tensor (MoE / MLP reduce-scatter)
- all_reduce & all_reduce_rank_order (mHC / norm all-reduce + bit-identical rank-order exchange+AG)
at T in (16, 2048, 4096, 6144) rows.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

NODES = [
    ("dgx1", "10.0.1.71", 0),
    ("dgx2", "10.0.1.72", 1),
    ("dgx3", "10.0.1.73", 2),
]

BASE_NCCL_ENV = {
    "GLOO_SOCKET_IFNAME": "enP7s7",
    "NCCL_SOCKET_IFNAME": "enP7s7",
    "NCCL_IB_HCA": "=roceP2p1s0f0,roceP2p1s0f1,rocep1s0f0,rocep1s0f1",
    "NCCL_IB_ADDR_FAMILY": "AF_INET",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_MERGE_NICS": "1",
    "NCCL_IB_ROCE_VERSION_NUM": "2",
    "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
    "NCCL_CROSS_NIC": "1",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_IGNORE_CPU_AFFINITY": "1",
    "NCCL_LL128_BUFFSIZE": "262144",
    "NCCL_NET": "IB",
    "NCCL_NET_PLUGIN": "none",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_PROTO": "^LL128",
    "NCCL_SHM_DISABLE": "1",
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,GRAPH",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
}

ARMS: list[tuple[str, dict[str, str]]] = [
    (
        "control_c8_1m",
        {
            "NCCL_MIN_NCHANNELS": "8",
            "NCCL_MAX_NCHANNELS": "8",
            "NCCL_BUFFSIZE": "1048576",
        },
    ),
    (
        "cand_c4_1m",
        {
            "NCCL_ALGO": "Ring",
            "NCCL_MIN_NCHANNELS": "4",
            "NCCL_MAX_NCHANNELS": "4",
            "NCCL_BUFFSIZE": "1048576",
        },
    ),
    (
        "cand_c4_4m_bidir",
        {
            "NCCL_ALGO": "Ring",
            "NCCL_MIN_NCHANNELS": "4",
            "NCCL_MAX_NCHANNELS": "4",
            "NCCL_BUFFSIZE": "4194304",
            "NCCL_SWITCHLESS_BIDIRECTIONAL": "2",
            "NCCL_MIN_TRAFFIC_PER_CHANNEL": "512",
            "NCCL_THREAD_THRESHOLDS": "-2 -2 -2 1 1 1",
        },
    ),
    (
        "cand_c4_8m",
        {
            "NCCL_ALGO": "Ring",
            "NCCL_MIN_NCHANNELS": "4",
            "NCCL_MAX_NCHANNELS": "4",
            "NCCL_BUFFSIZE": "8388608",
            "NCCL_P2P_NET_CHUNKSIZE": "262144",
        },
    ),
    (
        "cand_c4_8m_bidir",
        {
            "NCCL_ALGO": "Ring",
            "NCCL_MIN_NCHANNELS": "4",
            "NCCL_MAX_NCHANNELS": "4",
            "NCCL_BUFFSIZE": "8388608",
            "NCCL_P2P_NET_CHUNKSIZE": "262144",
            "NCCL_SWITCHLESS_BIDIRECTIONAL": "2",
            "NCCL_MIN_TRAFFIC_PER_CHANNEL": "512",
            "NCCL_THREAD_THRESHOLDS": "-2 -2 -2 1 1 1",
        },
    ),
    (
        "cand_c6_8m",
        {
            "NCCL_ALGO": "Ring",
            "NCCL_MIN_NCHANNELS": "6",
            "NCCL_MAX_NCHANNELS": "6",
            "NCCL_BUFFSIZE": "8388608",
            "NCCL_P2P_NET_CHUNKSIZE": "262144",
        },
    ),
    (
        "cand_c6_8m_bidir",
        {
            "NCCL_ALGO": "Ring",
            "NCCL_MIN_NCHANNELS": "6",
            "NCCL_MAX_NCHANNELS": "6",
            "NCCL_BUFFSIZE": "8388608",
            "NCCL_P2P_NET_CHUNKSIZE": "262144",
            "NCCL_SWITCHLESS_BIDIRECTIONAL": "1",
            "NCCL_MIN_TRAFFIC_PER_CHANNEL": "512",
            "NCCL_THREAD_THRESHOLDS": "-2 -2 -2 1 1 1",
        },
    ),
]


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_worker(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist

    rank = args.rank
    world = args.world_size
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=world,
    )

    def emit(**payload: Any) -> None:
        if rank == 0:
            rec = {
                "boot": args.boot,
                "arm": args.arm,
                "utc": utc_now(),
                **payload,
            }
            print("JSON_RECORD:" + json.dumps(rec), flush=True)

    # Warmup communicator
    warm = torch.ones((1024,), dtype=torch.float32, device=device)
    dist.all_reduce(warm)
    torch.cuda.synchronize()
    dist.barrier()

    token_shapes = [16, 2048, 4096, 6144]
    warmup_iters = 5
    repeats = args.repeats

    for T in token_shapes:
        L = (T + world - 1) // world
        padded = world * L
        L_wo_slice = (L + 1) // 2

        # Prepare deterministic tensors per rank
        g = torch.Generator(device=device)
        g.manual_seed(20261106 + rank * 997 + T)

        # 1. Engram SP exchange / full BF16 all_gather_into_tensor: (L, 5120) -> (3*L, 5120) BF16
        ag_bf16_in = torch.randn((L, 5120), dtype=torch.bfloat16, device=device, generator=g)
        ag_bf16_out = torch.empty((padded, 5120), dtype=torch.bfloat16, device=device)

        # 2. MoE MXFP8 SP gather: (L, 5376) -> (3*L, 5376) uint8
        ag_mxfp8_in = torch.randint(0, 256, (L, 5376), dtype=torch.uint8, device=device, generator=g)
        ag_mxfp8_out = torch.empty((padded, 5376), dtype=torch.uint8, device=device)

        # 3. Attention front compressed gather: (L, 3584) -> (3*L, 3584) uint8
        ag_front_in = torch.randint(0, 256, (L, 3584), dtype=torch.uint8, device=device, generator=g)
        ag_front_out = torch.empty((padded, 3584), dtype=torch.uint8, device=device)

        # 4. WO local exchange (1 slice of 2-slice exchange): all_to_all_single on (3, L_wo_slice, 5120) BF16
        wo_a2a_in = torch.randn((world * L_wo_slice, 5120), dtype=torch.bfloat16, device=device, generator=g)
        wo_a2a_out = torch.empty_like(wo_a2a_in)

        # 5. MLP / MoE all_to_all_single: (3*L, 5120) BF16
        mlp_a2a_in = torch.randn((padded, 5120), dtype=torch.bfloat16, device=device, generator=g)
        mlp_a2a_out = torch.empty_like(mlp_a2a_in)

        # 6. MoE / MLP reduce_scatter_tensor: (3*L, 5120) -> (L, 5120) BF16
        rs_bf16_in = torch.randn((padded, 5120), dtype=torch.bfloat16, device=device, generator=g)
        rs_bf16_out = torch.empty((L, 5120), dtype=torch.bfloat16, device=device)

        # 7. mHC / norm all_reduce: (T, 5120) BF16
        ar_bf16_buf = torch.randn((T, 5120), dtype=torch.bfloat16, device=device, generator=g)
        ar_bf16_work = torch.empty_like(ar_bf16_buf)

        # Bit-identity check for unreduced collectives + rank-order sum
        dist.all_gather_into_tensor(ag_bf16_out, ag_bf16_in)
        own_slice = ag_bf16_out[rank * L : (rank + 1) * L]
        differ_ag = int((own_slice.view(torch.int16) != ag_bf16_in.view(torch.int16)).sum().item())
        assert differ_ag == 0, f"all_gather bit mismatch on rank {rank}: {differ_ag}"

        dist.all_to_all_single(mlp_a2a_out, mlp_a2a_in)
        own_a2a = mlp_a2a_out[rank * L : (rank + 1) * L]
        ref_a2a = mlp_a2a_in[rank * L : (rank + 1) * L]
        differ_a2a = int((own_a2a.view(torch.int16) != ref_a2a.view(torch.int16)).sum().item())
        assert differ_a2a == 0, f"all_to_all bit mismatch on rank {rank}: {differ_a2a}"

        # Rank-order sum bit-identity check (exact collectives.py all_reduce_rank_order semantics):
        # Gather all 3 full (padded, 5120) tensors on each rank to compute reference rank-order FP32 sum
        all_full = [torch.empty_like(mlp_a2a_in) for _ in range(world)]
        dist.all_gather(all_full, mlp_a2a_in)
        ref_sum = (
            all_full[0].float() + all_full[1].float() + all_full[2].float()
        ).to(torch.bfloat16)

        # Now execute rank-order reduce-scatter (via all_to_all_single + FP32 sum) + all_gather_into_tensor
        a2a_parts = torch.empty_like(mlp_a2a_in)
        dist.all_to_all_single(a2a_parts, mlp_a2a_in)
        part_views = a2a_parts.view(world, L, 5120)
        own_sum = (
            part_views[0].float() + part_views[1].float() + part_views[2].float()
        ).to(torch.bfloat16)
        ro_out = torch.empty_like(mlp_a2a_in)
        dist.all_gather_into_tensor(ro_out, own_sum)
        differ_ro = int((ro_out.view(torch.int16) != ref_sum.view(torch.int16)).sum().item())
        assert differ_ro == 0, f"all_reduce_rank_order bit mismatch on rank {rank}: {differ_ro}"

        emit(
            kind="bit_identity",
            T=T,
            L=L,
            differ_ag=differ_ag,
            differ_a2a=differ_a2a,
            differ_rank_order_ar=differ_ro,
            passed=True,
        )

        # Define collective benchmarks for this T
        def run_engram_ag():
            dist.all_gather_into_tensor(ag_bf16_out, ag_bf16_in)

        def run_moe_mxfp8_ag():
            dist.all_gather_into_tensor(ag_mxfp8_out, ag_mxfp8_in)

        def run_attn_front_ag():
            dist.all_gather_into_tensor(ag_front_out, ag_front_in)

        def run_wo_slice_a2a():
            dist.all_to_all_single(wo_a2a_out, wo_a2a_in)

        def run_mlp_a2a():
            dist.all_to_all_single(mlp_a2a_out, mlp_a2a_in)

        def run_moe_rs():
            dist.reduce_scatter_tensor(rs_bf16_out, rs_bf16_in)

        def run_mhc_ar():
            ar_bf16_work.copy_(ar_bf16_buf)
            dist.all_reduce(ar_bf16_work)

        def run_rank_order_ar():
            dist.all_to_all_single(a2a_parts, mlp_a2a_in)
            v = a2a_parts.view(world, L, 5120)
            s = (v[0].float() + v[1].float() + v[2].float()).to(torch.bfloat16)
            dist.all_gather_into_tensor(ro_out, s)

        specs = [
            ("engram_sp_ag_bf16", "all_gather_into_tensor", padded * 5120 * 2, run_engram_ag),
            ("moe_sp_gather_mxfp8", "all_gather_into_tensor", padded * 5376, run_moe_mxfp8_ag),
            ("attn_front_ag_u8", "all_gather_into_tensor", padded * 3584, run_attn_front_ag),
            ("wo_local_exchange_slice_a2a", "all_to_all_single", world * L_wo_slice * 5120 * 2, run_wo_slice_a2a),
            ("mlp_all_to_all_bf16", "all_to_all_single", padded * 5120 * 2, run_mlp_a2a),
            ("moe_reduce_scatter_bf16", "reduce_scatter_tensor", padded * 5120 * 2, run_moe_rs),
            ("mhc_norm_all_reduce_bf16", "all_reduce", T * 5120 * 2, run_mhc_ar),
            ("rank_order_all_reduce_bf16", "a2a_plus_ag_rank_order", padded * 5120 * 2, run_rank_order_ar),
        ]

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        lat_tensor = torch.empty((1,), dtype=torch.float32, device=device)

        for name, op_kind, payload_bytes, fn in specs:
            for _ in range(warmup_iters):
                fn()
            torch.cuda.synchronize()
            dist.barrier()

            samples_ms: list[float] = []
            for _ in range(repeats):
                dist.barrier()
                start_ev.record()
                fn()
                end_ev.record()
                torch.cuda.synchronize()
                lat_tensor[0] = start_ev.elapsed_time(end_ev)
                dist.all_reduce(lat_tensor, op=dist.ReduceOp.MAX)
                samples_ms.append(float(lat_tensor.item()))

            samples_sorted = sorted(samples_ms)
            med_ms = statistics.median(samples_sorted)
            p90_ms = samples_sorted[int(0.9 * (len(samples_sorted) - 1))]
            algbw_gbps = (payload_bytes / 1e9) / (med_ms / 1e3)
            # Bus bandwidth factor on world=3:
            # AG / RS / A2A: (world - 1) / world = 2/3
            # AR: 2 * (world - 1) / world = 4/3
            factor = (4.0 / 3.0) if op_kind == "all_reduce" else (2.0 / 3.0)
            busbw_gbps = algbw_gbps * factor

            emit(
                kind="timing",
                T=T,
                L=L,
                name=name,
                op_kind=op_kind,
                payload_bytes=payload_bytes,
                median_ms=round(med_ms, 4),
                p90_ms=round(p90_ms, 4),
                min_ms=round(samples_sorted[0], 4),
                max_ms=round(samples_sorted[-1], 4),
                algbw_gbps=round(algbw_gbps, 3),
                busbw_gbps=round(busbw_gbps, 3),
            )

    dist.barrier()
    dist.destroy_process_group()


def parse_nccl_init_log(log_text: str) -> dict[str, Any]:
    rings = re.findall(r"Ring \d+ : [0-9 ->]+", log_text)
    trees = re.findall(r"Trees \[\d+\] [0-9/\-> ]+", log_text)
    ch_match = re.search(r"(\d+) coll channels, (\d+) collnet channels, (\d+) nvls channels, (\d+) p2p channels", log_text)
    p2p_chunk = re.search(r"P2P Chunksize set to (\d+)", log_text)
    buff_match = re.findall(r"comm [0-9a-fx]+ rank 0 .* buffSize (\d+)", log_text)
    return {
        "coll_channels": int(ch_match.group(1)) if ch_match else None,
        "p2p_channels": int(ch_match.group(4)) if ch_match else None,
        "p2p_chunksize": int(p2p_chunk.group(1)) if p2p_chunk else None,
        "rings": rings,
        "trees_count": len(trees),
        "buff_sizes": [int(x) for x in buff_match],
    }


def run_orchestrator(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    script_host_path = Path(__file__).resolve()

    # Copy worker script to /tmp/screen_nccl_tp3.py on dgx1, dgx2, dgx3
    for host, ip, _ in NODES:
        subprocess.run(
            ["scp", "-q", str(script_host_path), f"{ip}:/tmp/screen_nccl_tp3.py"],
            check=True,
        )

    for boot_idx, boot in enumerate(("boot1", "boot2")):
        arm_list = list(ARMS) if boot_idx == 0 else list(reversed(ARMS))
        out_jsonl = out_dir / f"screen-nccl-tp3-{boot}.jsonl"
        if out_jsonl.exists():
            out_jsonl.unlink()

        for arm_idx, (arm_name, arm_env) in enumerate(arm_list):
            port = 29510 + boot_idx * 20 + arm_idx
            print(f"=== [{boot}] Running 3-rank NCCL arm: {arm_name} (port={port}) ===", flush=True)

            env_merged = {**BASE_NCCL_ENV, **arm_env}
            env_flags: list[str] = []
            for k, v in env_merged.items():
                env_flags.extend(["-e", f"{k}={v}"])

            procs: list[tuple[str, subprocess.Popen[str]]] = []
            try:
                for host, ip, rank in NODES:
                    cname = f"nccl-sp-screen-{boot}-{arm_name}-r{rank}".replace("_", "-")
                    docker_cmd = [
                        "docker", "run", "--rm",
                        "--name", cname,
                        "--network=host", "--ipc=host", "--gpus=all",
                        "--device=/dev/infiniband:/dev/infiniband:rwm",
                        "--cap-add=IPC_LOCK",
                        "--ulimit=memlock=-1:-1",
                        "--entrypoint", "",
                        *env_flags,
                        "-v", "/tmp/screen_nccl_tp3.py:/tmp/screen_nccl_tp3.py:ro",
                        args.image,
                        "python3", "/tmp/screen_nccl_tp3.py",
                        "--worker",
                        "--boot", boot,
                        "--arm", arm_name,
                        "--rank", str(rank),
                        "--world-size", "3",
                        "--master-addr", "10.0.1.71",
                        "--master-port", str(port),
                        "--repeats", str(args.repeats),
                    ]
                    remote_cmd = " ".join(shlex.quote(x) for x in docker_cmd)
                    p = subprocess.Popen(
                        ["ssh", "-o", "BatchMode=yes", ip, remote_cmd],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    procs.append((host, p))

                outputs: dict[str, str] = {}
                for host, p in procs:
                    out, _ = p.communicate(timeout=180)
                    outputs[host] = out
                    if p.returncode != 0:
                        raise RuntimeError(
                            f"[{boot} {arm_name}] {host} exited with {p.returncode}:\n{out}"
                        )
            finally:
                for host, ip, rank in NODES:
                    cname = f"nccl-sp-screen-{boot}-{arm_name}-r{rank}".replace("_", "-")
                    subprocess.run(
                        ["ssh", "-o", "BatchMode=yes", ip, f"docker rm -f {cname} >/dev/null 2>&1 || true"],
                        check=False,
                    )

            r0_out = outputs["dgx1"]
            nccl_log_path = out_dir / f"nccl-init-{boot}-{arm_name}-rank0.txt"
            nccl_lines = [
                ln.rstrip() for ln in r0_out.splitlines()
                if not ln.startswith("JSON_RECORD:")
            ]
            nccl_log_path.write_text("\n".join(nccl_lines) + "\n")

            topo_info = parse_nccl_init_log("\n".join(nccl_lines))
            init_rec = {
                "boot": boot,
                "arm": arm_name,
                "utc": utc_now(),
                "kind": "nccl_init",
                "env": arm_env,
                **topo_info,
            }
            with out_jsonl.open("a") as f:
                f.write(json.dumps(init_rec) + "\n")
                for ln in r0_out.splitlines():
                    if ln.startswith("JSON_RECORD:"):
                        f.write(ln[len("JSON_RECORD:") :] + "\n")

            print(
                f"  -> {arm_name}: channels={topo_info['coll_channels']}, "
                f"p2p_chunk={topo_info['p2p_chunksize']}, "
                f"rings={topo_info['rings']}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--boot", default="boot1")
    parser.add_argument("--arm", default="control_c8_1m")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--master-addr", default="10.0.1.71")
    parser.add_argument("--master-port", type=int, default=29510)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--image", default="spark-ds41f:candidate-36")
    parser.add_argument("--out-dir", default="receipts")
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
