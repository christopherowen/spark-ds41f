#!/usr/bin/env python3
"""C3 on the fabric: the TP reduce-scatter of prefill sequence parallelism, NCCL's ring
against the rank-order exchange (one-shot arithmetic), model-free, one rank per node.

usage (a lab "fabric" job; the runner adds the probe's rank arguments):
    fabric_reduce_scatter.py --rank R --world-size N --master-addr A --master-port P
        [--rows 205 512 ...] [--iters 20]

For each step size (BF16 rows of 5120, padded to whole chunks as sequence parallelism
pads them): the median time per call of NCCL's reduce_scatter, of the rank-order
reduce-scatter, of its exchange alone and of its local sum alone, the slowest rank's
median after a barrier before every call. Every rank-order result is
checked bit for bit against the FP32 rank-order sum of an exact all-gather; NCCL's
differing elements are counted. Rank 0 prints one JSON line per step size, then a
summary; exits 1 if any rank-order result differs.
"""
import argparse
import json
import os
import statistics
import time
from datetime import timedelta

HIDDEN = 5120


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[205, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    set_custom_all_reduce("SPARKNET_ROCE_TOPOLOGY" in os.environ)
    init_distributed_environment(
        world_size=args.world_size, rank=args.rank, local_rank=0,
        distributed_init_method=f"tcp://{args.master_addr}:{args.master_port}",
        timeout=timedelta(seconds=120),
    )
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=args.world_size)
    group = get_tp_group()
    nccl = group.device_communicator.pynccl_comm
    if nccl is None or nccl.disabled:
        raise RuntimeError("PyNCCL communicator is unavailable")
    from vllm.models.deepseek_v4_1.tilelang.collectives import (
        exchange_rank_order,
        rank_order_sum,
        reduce_scatter_rank_order,
    )

    world, rank = args.world_size, args.rank

    def slowest(value: float) -> float:
        t = torch.tensor([value], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return round(float(t.item()), 1)

    def timed(fn) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(args.iters):
            dist.barrier(group.cpu_group)
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e6)
        return slowest(statistics.median(times))

    failures = []
    for rows in args.rows:
        local = -(-rows // world)
        padded = local * world
        gen = torch.Generator(device=device).manual_seed(1000 + rank)
        x = (torch.randn((padded, HIDDEN), generator=gen, device=device) * 4).bfloat16()
        gathered = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(gathered, x)
        own = slice(rank * local, (rank + 1) * local)
        reference = rank_order_sum([g[own].contiguous() for g in gathered])
        out = torch.empty((local, HIDDEN), dtype=x.dtype, device=device)

        def nccl_rs(out=out, x=x):
            nccl.reduce_scatter(out, x)

        nccl_rs()
        torch.cuda.synchronize()
        nccl_differ = int((out.view(torch.int16) != reference.view(torch.int16)).sum())
        row = {"rows": rows, "padded": padded, "chunk_mib": round(local * HIDDEN * 2 / 2**20, 2),
               "nccl_us": timed(nccl_rs), "nccl_elements_differ": slowest(nccl_differ)}
        parts = [g[own].contiguous() for g in gathered]
        row["sum_only_us"] = timed(lambda parts=parts: rank_order_sum(parts))
        def rank_order(x=x):
            return reduce_scatter_rank_order(nccl, x, rank, world)

        result = rank_order()
        torch.cuda.synchronize()
        bad = int(not torch.equal(result.view(torch.int16), reference.view(torch.int16)))
        if slowest(bad):
            failures.append(f"{rows} rows: bits differ")
        row["rank_order_us"] = timed(rank_order)
        row["exchange_only_us"] = timed(lambda x=x: exchange_rank_order(nccl, x, rank, world))
        row["ratio"] = round(row["rank_order_us"] / row["nccl_us"], 3)
        if rank == 0:
            print(json.dumps(row), flush=True)
    if rank == 0:
        print(json.dumps({"summary": "C3 reduce-scatter on the fabric", "failures": failures}), flush=True)
    dist.barrier(group.cpu_group)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
