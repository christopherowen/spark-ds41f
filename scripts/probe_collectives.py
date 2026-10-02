#!/usr/bin/env python3
"""Model-free qualification of the actual vLLM TP communicator and CUDA graphs.

Run one rank per node using commands rendered by `spark3 topology probe`.
The surrounding container supplies a 180-second timeout and a 4 GiB limit.
This checks uniform TP collectives, not EP all-to-all or arbitrary send/recv.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, choices=(3, 4), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        parser.error("rank must be in the configured process group")

    import torch
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    ring = os.environ.get("VLLM_ENABLE_ROCE_ALLREDUCE") == "0"
    set_custom_all_reduce(not ring)
    init_distributed_environment(
        world_size=args.world_size, rank=args.rank, local_rank=0,
        distributed_init_method=f"tcp://{args.master_addr}:{args.master_port}",
        timeout=timedelta(seconds=90),
    )
    # Match upstream's standalone communicator tests: group creation reads
    # parallel_config even when no model is loaded.
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=args.world_size)
    group = get_tp_group()
    communicator = group.device_communicator
    if ring and communicator.b12x_ar_comm is not None:
        raise RuntimeError("ring probe unexpectedly initialized a direct-peer B12X communicator")
    nccl = communicator.pynccl_comm
    if nccl is None or nccl.disabled:
        raise RuntimeError("PyNCCL communicator is unavailable")

    # Exercise the torch NCCL group as well as vLLM's separate PyNCCL group.
    total = args.world_size * (args.world_size + 1) // 2
    control = torch.tensor([args.rank + 1], device=device, dtype=torch.float32)
    dist.all_reduce(control)
    torch.testing.assert_close(control, torch.full_like(control, total), rtol=0, atol=0)
    dist.broadcast(control, src=0)
    torch.testing.assert_close(control, torch.full_like(control, total), rtol=0, atol=0)

    checks = []
    for dtype in (torch.bfloat16, torch.float32):
        for length in (1, 17, 1024, 2 * 1024 * 1024):
            pattern = (torch.arange(length, device=device) % 7).to(dtype)
            local = pattern + args.rank + 1
            scattered = torch.cat([local + 8 * peer for peer in range(args.world_size)])

            def collect():
                return (group.all_reduce(local), group.all_gather(local, dim=0),
                        group.reduce_scatter(scattered, dim=0))

            def verify(outputs, increment=0):
                ar, ag, rs = outputs
                torch.cuda.synchronize()
                torch.testing.assert_close(ar, pattern * args.world_size + total + increment * args.world_size,
                                           rtol=0, atol=0)
                torch.testing.assert_close(ag, torch.cat([pattern + peer + 1 + increment
                                                         for peer in range(args.world_size)]), rtol=0, atol=0)
                expected = pattern * args.world_size + total + (8 * args.rank + increment) * args.world_size
                torch.testing.assert_close(rs, expected, rtol=0, atol=0)

            # Warm every size before capture so connection setup is outside it.
            verify(collect())
            graph = torch.cuda.CUDAGraph()
            with graph_capture(device=device) as context:
                with torch.cuda.graph(graph, stream=context.stream):
                    outputs = collect()
            for step in range(1, 5):
                local.add_(1)
                scattered.add_(1)
                graph.replay()
                verify(outputs, step)
            checks.append({"dtype": str(dtype), "elements_per_rank": length,
                           "eager": True, "graph_replays": 4})
            del graph, outputs, local, scattered, pattern

    print(json.dumps({"rank": args.rank, "world_size": args.world_size,
                      "transport": "nccl-ring" if ring else "rocenante-direct",
                      "passed": True, "checks": checks}), flush=True)
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
