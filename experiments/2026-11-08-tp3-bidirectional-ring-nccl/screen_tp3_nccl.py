#!/usr/bin/env python3
"""2-boot 3-rank RoCE NCCL verification benchmark on 3 nodes of sp5..sp8.

Compares on spark-ds41f:1107 across 2 fresh boots (boot1 forward order, boot2 reversed):
1. control_c8_1m: 8-channel 1 MiB unidirectional NCCL policy
   (NCCL_MIN_NCHANNELS=8, NCCL_MAX_NCHANNELS=8, NCCL_BUFFSIZE=1048576)
2. cand_c6_8m_bidir: 1107 tp3-triangle 6-channel bidirectional 8 MiB Ring policy
   (NCCL_ALGO=Ring, NCCL_MIN_NCHANNELS=6, NCCL_MAX_NCHANNELS=6,
    NCCL_BUFFSIZE=8388608, NCCL_P2P_NET_CHUNKSIZE=262144 -> p2pChunkSize=524288,
    NCCL_SWITCHLESS_BIDIRECTIONAL=1, NCCL_MIN_TRAFFIC_PER_CHANNEL=512,
    NCCL_THREAD_THRESHOLDS="-2 -2 -2 1 1 1")
3. cand_c6_8m_bidir_p2p512k: same 6-channel bidirectional 8 MiB Ring policy with
   NCCL_P2P_NET_CHUNKSIZE=524288 -> p2pChunkSize=1048576 (1 MiB)

Workloads at T in (16, 256, 1024, 6144):
- Standalone SP prefill collectives (all_gather_into_tensor, all_to_all_single,
  reduce_scatter_tensor, all_reduce, and bit-identical rank-order exchange+AG).
- Overlapped main-stream GEMM + side-stream NCCL collective wall time
  (testing SM contention on GB10's 48 SMs: 6 NCCL CTAs = 12.5% SMs vs 8 NCCL CTAs = 16.7% SMs).
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shlex
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

CANDIDATE_NODES = [
    ("sp5", "10.30.0.13"),
    ("sp6", "10.30.0.14"),
    ("sp7", "10.30.0.15"),
    ("sp8", "10.30.0.16"),
]

BASE_NCCL_ENV = {
    "GLOO_SOCKET_IFNAME": "enp1s0f0np0",
    "NCCL_SOCKET_IFNAME": "enp1s0f0np0",
    "TP_SOCKET_IFNAME": "enp1s0f0np0",
    "NCCL_IB_HCA": "=rocep1s0f0,roceP2p1s0f0",
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
    (
        "cand_c6_8m_bidir_p2p512k",
        {
            "NCCL_ALGO": "Ring",
            "NCCL_MIN_NCHANNELS": "6",
            "NCCL_MAX_NCHANNELS": "6",
            "NCCL_BUFFSIZE": "8388608",
            "NCCL_P2P_NET_CHUNKSIZE": "524288",
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
        device_id=device,
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

    # Warmup communicator and side stream
    comm_stream = torch.cuda.Stream(device=device)
    main_stream = torch.cuda.current_stream(device)

    warm = torch.ones((1024,), dtype=torch.float32, device=device)
    dist.all_reduce(warm)
    with torch.cuda.stream(comm_stream):
        dist.all_reduce(warm)
    torch.cuda.synchronize()
    dist.barrier()

    token_shapes = [16, 256, 1024, 6144]
    warmup_iters = 5
    repeats = args.repeats

    # Preallocate weight matrices for overlapped main-stream GEMMs
    g_w = torch.Generator(device=device)
    g_w.manual_seed(20261108 + rank * 131)
    w_proj_5120 = torch.randn((5120, 5120), dtype=torch.bfloat16, device=device, generator=g_w)
    w_moe_shared = torch.randn((5120, 2048), dtype=torch.bfloat16, device=device, generator=g_w)

    for T in token_shapes:
        L = (T + world - 1) // world
        padded = world * L
        L_wo_slice = (L + 1) // 2

        g = torch.Generator(device=device)
        g.manual_seed(20261108 + rank * 997 + T)

        # 1. Engram SP exchange / full BF16 all_gather_into_tensor: (L, 5120) -> (3*L, 5120) BF16
        ag_bf16_in = torch.randn((L, 5120), dtype=torch.bfloat16, device=device, generator=g)
        ag_bf16_out = torch.empty((padded, 5120), dtype=torch.bfloat16, device=device)

        # 2. MoE MXFP8 SP gather: (L, 5376) -> (3*L, 5376) uint8
        ag_mxfp8_in = torch.randint(0, 256, (L, 5376), dtype=torch.uint8, device=device, generator=g)
        ag_mxfp8_out = torch.empty((padded, 5376), dtype=torch.uint8, device=device)

        # 3. Attention front compressed gather: (L, 3584) -> (3*L, 3584) uint8
        ag_front_in = torch.randint(0, 256, (L, 3584), dtype=torch.uint8, device=device, generator=g)
        ag_front_out = torch.empty((padded, 3584), dtype=torch.uint8, device=device)

        # 4. 0737 SP WO local MXFP8 exchange: (3*L, 396) uint8 (384 FP8 + 12 scale bytes)
        wo_mxfp8_in = torch.randint(0, 256, (padded, 396), dtype=torch.uint8, device=device, generator=g)
        wo_mxfp8_out = torch.empty_like(wo_mxfp8_in)

        # 5. WO local exchange (1 slice of 2-slice BF16 exchange): (3*L_wo_slice, 5120) BF16
        wo_a2a_in = torch.randn((world * L_wo_slice, 5120), dtype=torch.bfloat16, device=device, generator=g)
        wo_a2a_out = torch.empty_like(wo_a2a_in)

        # 6. MLP / MoE all_to_all_single: (3*L, 5120) BF16
        mlp_a2a_in = torch.randn((padded, 5120), dtype=torch.bfloat16, device=device, generator=g)
        mlp_a2a_out = torch.empty_like(mlp_a2a_in)

        # 7. MoE / MLP reduce_scatter_tensor: (3*L, 5120) -> (L, 5120) BF16
        rs_bf16_in = torch.randn((padded, 5120), dtype=torch.bfloat16, device=device, generator=g)
        rs_bf16_out = torch.empty((L, 5120), dtype=torch.bfloat16, device=device)

        # 8. mHC / norm all_reduce: (T, 5120) BF16
        ar_bf16_buf = torch.randn((T, 5120), dtype=torch.bfloat16, device=device, generator=g)
        ar_bf16_work = torch.empty_like(ar_bf16_buf)

        # Tensors for overlapped main-stream GEMM + side-stream collective
        gemm_x_local = torch.randn((L, 5120), dtype=torch.bfloat16, device=device, generator=g)
        gemm_out_local = torch.empty((L, 5120), dtype=torch.bfloat16, device=device)
        gemm_x_full = torch.randn((T, 5120), dtype=torch.bfloat16, device=device, generator=g)
        gemm_out_shared = torch.empty((T, 2048), dtype=torch.bfloat16, device=device)

        # Bit-identity checks
        dist.all_gather_into_tensor(ag_bf16_out, ag_bf16_in)
        own_slice = ag_bf16_out[rank * L : (rank + 1) * L]
        differ_ag = int((own_slice.view(torch.int16) != ag_bf16_in.view(torch.int16)).sum().item())
        assert differ_ag == 0, f"all_gather bit mismatch on rank {rank}: {differ_ag}"

        dist.all_to_all_single(mlp_a2a_out, mlp_a2a_in)
        own_a2a = mlp_a2a_out[rank * L : (rank + 1) * L]
        ref_a2a = mlp_a2a_in[rank * L : (rank + 1) * L]
        differ_a2a = int((own_a2a.view(torch.int16) != ref_a2a.view(torch.int16)).sum().item())
        assert differ_a2a == 0, f"all_to_all bit mismatch on rank {rank}: {differ_a2a}"

        all_full = [torch.empty_like(mlp_a2a_in) for _ in range(world)]
        dist.all_gather(all_full, mlp_a2a_in)
        ref_sum = (
            all_full[0].float() + all_full[1].float() + all_full[2].float()
        ).to(torch.bfloat16)

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

        # Verify overlapped GEMM + side-stream collective produces identical GEMM & collective outputs
        ref_gemm_local = torch.matmul(gemm_x_local, w_proj_5120)
        mlp_a2a_out.zero_()
        comm_stream.wait_stream(main_stream)
        with torch.cuda.stream(comm_stream):
            dist.all_to_all_single(mlp_a2a_out, mlp_a2a_in)
        torch.matmul(gemm_x_local, w_proj_5120, out=gemm_out_local)
        main_stream.wait_stream(comm_stream)
        torch.cuda.synchronize()
        differ_ov_gemm = int((gemm_out_local.view(torch.int16) != ref_gemm_local.view(torch.int16)).sum().item())
        differ_ov_a2a = int((mlp_a2a_out[rank * L : (rank + 1) * L].view(torch.int16) != ref_a2a.view(torch.int16)).sum().item())
        assert differ_ov_gemm == 0 and differ_ov_a2a == 0

        emit(
            kind="bit_identity",
            T=T,
            L=L,
            differ_ag=differ_ag,
            differ_a2a=differ_a2a,
            differ_rank_order_ar=differ_ro,
            differ_overlap_gemm=differ_ov_gemm,
            differ_overlap_a2a=differ_ov_a2a,
            passed=True,
        )

        # Standalone collective functions
        def run_engram_ag():
            dist.all_gather_into_tensor(ag_bf16_out, ag_bf16_in)

        def run_moe_mxfp8_ag():
            dist.all_gather_into_tensor(ag_mxfp8_out, ag_mxfp8_in)

        def run_attn_front_ag():
            dist.all_gather_into_tensor(ag_front_out, ag_front_in)

        def run_wo_mxfp8_a2a():
            dist.all_to_all_single(wo_mxfp8_out, wo_mxfp8_in)

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

        # Overlapped main-stream GEMM + side-stream collective functions
        def run_overlap_proj_rs():
            comm_stream.wait_stream(main_stream)
            with torch.cuda.stream(comm_stream):
                dist.reduce_scatter_tensor(rs_bf16_out, rs_bf16_in)
            torch.matmul(gemm_x_local, w_proj_5120, out=gemm_out_local)
            main_stream.wait_stream(comm_stream)

        def run_overlap_engram_proj_a2a():
            comm_stream.wait_stream(main_stream)
            with torch.cuda.stream(comm_stream):
                dist.all_to_all_single(mlp_a2a_out, mlp_a2a_in)
            torch.matmul(gemm_x_local, w_proj_5120, out=gemm_out_local)
            main_stream.wait_stream(comm_stream)

        def run_overlap_moe_ag_shared_gemm():
            comm_stream.wait_stream(main_stream)
            with torch.cuda.stream(comm_stream):
                dist.all_gather_into_tensor(ag_mxfp8_out, ag_mxfp8_in)
            torch.matmul(gemm_x_full, w_moe_shared, out=gemm_out_shared)
            main_stream.wait_stream(comm_stream)

        specs = [
            ("engram_sp_ag_bf16", "all_gather_into_tensor", padded * 5120 * 2, run_engram_ag),
            ("moe_sp_gather_mxfp8", "all_gather_into_tensor", padded * 5376, run_moe_mxfp8_ag),
            ("attn_front_ag_u8", "all_gather_into_tensor", padded * 3584, run_attn_front_ag),
            ("wo_local_exchange_mxfp8_a2a", "all_to_all_single", padded * 396, run_wo_mxfp8_a2a),
            ("wo_local_exchange_slice_a2a", "all_to_all_single", world * L_wo_slice * 5120 * 2, run_wo_slice_a2a),
            ("mlp_all_to_all_bf16", "all_to_all_single", padded * 5120 * 2, run_mlp_a2a),
            ("moe_reduce_scatter_bf16", "reduce_scatter_tensor", padded * 5120 * 2, run_moe_rs),
            ("mhc_norm_all_reduce_bf16", "all_reduce", T * 5120 * 2, run_mhc_ar),
            ("rank_order_all_reduce_bf16", "a2a_plus_ag_rank_order", padded * 5120 * 2, run_rank_order_ar),
            ("overlap_proj_rs_bf16", "overlap_gemm_plus_rs", padded * 5120 * 2, run_overlap_proj_rs),
            ("overlap_engram_proj_a2a_bf16", "overlap_gemm_plus_a2a", padded * 5120 * 2, run_overlap_engram_proj_a2a),
            ("overlap_moe_ag_shared_gemm", "overlap_gemm_plus_ag", padded * 5376, run_overlap_moe_ag_shared_gemm),
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
                start_ev.record(main_stream)
                fn()
                end_ev.record(main_stream)
                torch.cuda.synchronize()
                lat_tensor[0] = start_ev.elapsed_time(end_ev)
                dist.all_reduce(lat_tensor, op=dist.ReduceOp.MAX)
                samples_ms.append(float(lat_tensor.item()))

            samples_sorted = sorted(samples_ms)
            med_ms = statistics.median(samples_sorted)
            p90_ms = samples_sorted[int(0.9 * (len(samples_sorted) - 1))]
            algbw_gbps = (payload_bytes / 1e9) / (med_ms / 1e3)
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
    ch_match = re.search(
        r"(\d+) coll channels, (\d+) collnet channels, (\d+) nvls channels, (\d+) p2p channels",
        log_text,
    )
    p2p_chunk = re.search(r"P2P Chunksize set to (\d+)", log_text)
    thresh_match = re.search(r"threadThresholds ([^\n]+)", log_text)
    return {
        "coll_channels": int(ch_match.group(1)) if ch_match else None,
        "p2p_channels": int(ch_match.group(4)) if ch_match else None,
        "p2p_chunksize": int(p2p_chunk.group(1)) if p2p_chunk else None,
        "rings": rings,
        "trees_count": len(trees),
        "thread_thresholds": thresh_match.group(1).strip() if thresh_match else None,
    }


def select_three_nodes() -> list[tuple[str, str, int]]:
    """Select 3 nodes from sp5..sp8, preferring currently unlocked nodes."""
    free_nodes: list[tuple[str, str]] = []
    busy_nodes: list[tuple[str, str]] = []
    for host, ip in CANDIDATE_NODES:
        res = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, 'flock -n /tmp/gpu.lock -c "true"'],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            free_nodes.append((host, ip))
        else:
            busy_nodes.append((host, ip))
    chosen = (free_nodes + busy_nodes)[:3]
    # Keep deterministic rank order by host name
    chosen.sort(key=lambda x: x[0])
    return [(host, ip, idx) for idx, (host, ip) in enumerate(chosen)]


def acquire_cluster_locks(nodes: list[tuple[str, str, int]]) -> list[subprocess.Popen[str]]:
    """Acquire /tmp/gpu.lock on all 3 nodes in sorted host order to prevent deadlock."""
    locks: list[subprocess.Popen[str]] = []
    try:
        for host, _, _ in nodes:
            print(f"Acquiring /tmp/gpu.lock on {host}...", flush=True)
            p = subprocess.Popen(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    host,
                    'flock /tmp/gpu.lock -c "echo LOCKED; cat"',
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert p.stdout is not None
            line = p.stdout.readline().strip()
            if line != "LOCKED":
                stderr = p.stderr.read() if p.stderr else ""
                raise RuntimeError(f"Failed to lock {host}: {line!r} {stderr}")
            locks.append(p)
        return locks
    except Exception:
        release_cluster_locks(locks)
        raise


def release_cluster_locks(locks: list[subprocess.Popen[str]]) -> None:
    for p in locks:
        try:
            if p.stdin:
                p.stdin.close()
            p.wait(timeout=5)
        except Exception:
            p.kill()


def run_orchestrator(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    script_host_path = Path(__file__).resolve()

    nodes = select_three_nodes()
    master_addr = nodes[0][1]
    print(f"Selected 3-node RoCE cluster: {nodes} (master_addr={master_addr})", flush=True)

    for host, _, _ in nodes:
        subprocess.run(
            ["scp", "-q", str(script_host_path), f"{host}:/tmp/screen_tp3_nccl.py"],
            check=True,
        )

    locks = acquire_cluster_locks(nodes)
    try:
        for boot_idx, boot in enumerate(("boot1", "boot2")):
            arm_list = list(ARMS) if boot_idx == 0 else list(reversed(ARMS))
            out_jsonl = out_dir / f"screen-tp3-nccl-{boot}.jsonl"
            if out_jsonl.exists():
                out_jsonl.unlink()

            for arm_idx, (arm_name, arm_env) in enumerate(arm_list):
                port = 29610 + boot_idx * 20 + arm_idx
                print(
                    f"=== [{boot}] Running 3-rank NCCL arm: {arm_name} on {[n[0] for n in nodes]} (port={port}) ===",
                    flush=True,
                )

                env_merged = {**BASE_NCCL_ENV, **arm_env}
                env_flags: list[str] = []
                for k, v in env_merged.items():
                    env_flags.extend(["-e", f"{k}={v}"])

                procs: list[tuple[str, subprocess.Popen[str]]] = []
                try:
                    for host, _, rank in nodes:
                        cname = f"tp3-nccl-verify-{boot}-{arm_name}-r{rank}".replace("_", "-")
                        docker_cmd = [
                            "docker",
                            "run",
                            "--rm",
                            "--name",
                            cname,
                            "--network=host",
                            "--ipc=host",
                            "--gpus=all",
                            "--device=/dev/infiniband:/dev/infiniband:rwm",
                            "--cap-add=IPC_LOCK",
                            "--ulimit=memlock=-1:-1",
                            "--entrypoint",
                            "",
                            *env_flags,
                            "-v",
                            "/tmp/screen_tp3_nccl.py:/tmp/screen_tp3_nccl.py:ro",
                            args.image,
                            "python3",
                            "/tmp/screen_tp3_nccl.py",
                            "--worker",
                            "--boot",
                            boot,
                            "--arm",
                            arm_name,
                            "--rank",
                            str(rank),
                            "--world-size",
                            "3",
                            "--master-addr",
                            master_addr,
                            "--master-port",
                            str(port),
                            "--repeats",
                            str(args.repeats),
                        ]
                        remote_cmd = " ".join(shlex.quote(x) for x in docker_cmd)
                        p = subprocess.Popen(
                            ["ssh", "-o", "BatchMode=yes", host, remote_cmd],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                        procs.append((host, p))

                    outputs: dict[str, str] = {}
                    for host, p in procs:
                        out, _ = p.communicate(timeout=240)
                        outputs[host] = out
                        if p.returncode != 0:
                            raise RuntimeError(
                                f"[{boot} {arm_name}] {host} exited with {p.returncode}:\n{out}"
                            )
                finally:
                    for host, _, rank in nodes:
                        cname = f"tp3-nccl-verify-{boot}-{arm_name}-r{rank}".replace("_", "-")
                        subprocess.run(
                            [
                                "ssh",
                                "-o",
                                "BatchMode=yes",
                                host,
                                f"docker rm -f {cname} >/dev/null 2>&1 || true",
                            ],
                            check=False,
                        )

                r0_host = nodes[0][0]
                r0_out = outputs[r0_host]
                nccl_log_path = out_dir / f"nccl-init-{boot}-{arm_name}-rank0.txt"
                nccl_lines = [
                    ln.rstrip()
                    for ln in r0_out.splitlines()
                    if not ln.startswith("JSON_RECORD:")
                ]
                nccl_log_path.write_text("\n".join(nccl_lines) + "\n")

                topo_info = parse_nccl_init_log("\n".join(nccl_lines))
                init_rec = {
                    "boot": boot,
                    "arm": arm_name,
                    "utc": utc_now(),
                    "kind": "nccl_init",
                    "nodes": [n[0] for n in nodes],
                    "env": arm_env,
                    **topo_info,
                }
                with out_jsonl.open("a") as f:
                    f.write(json.dumps(init_rec) + "\n")
                    for ln in r0_out.splitlines():
                        if ln.startswith("JSON_RECORD:"):
                            f.write(ln[len("JSON_RECORD:") :] + "\n")

                print(
                    f"  -> {arm_name}: coll_channels={topo_info['coll_channels']}, "
                    f"p2p_channels={topo_info['p2p_channels']}, "
                    f"p2p_chunk={topo_info['p2p_chunksize']}, "
                    f"rings={topo_info['rings']}",
                    flush=True,
                )
    finally:
        release_cluster_locks(locks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--boot", default="boot1")
    parser.add_argument("--arm", default="control_c8_1m")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--master-addr", default="10.30.0.14")
    parser.add_argument("--master-port", type=int, default=29610)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--image", default="spark-ds41f:1107")
    parser.add_argument("--out-dir", default="receipts")
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
