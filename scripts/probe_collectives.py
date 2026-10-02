#!/usr/bin/env python3
"""Model-free qualification of the actual vLLM TP communicator and CUDA graphs.

Run one rank per node using commands rendered by `spark3 topology probe`.
The surrounding container supplies a 600-second timeout and a 12 GiB limit (two NCCL groups plus compilation).
This checks uniform TP collectives, not EP all-to-all or arbitrary send/recv.
"""

from __future__ import annotations

from contextlib import contextmanager
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path


def rdma_error_counters():
    names = ('roce_adp_retrans', 'packet_seq_err', 'out_of_sequence', 'np_cnp_sent')
    counters = {}
    for device in Path('/sys/class/infiniband').iterdir():
        counters[device.name] = {name: int((device / 'ports/1/hw_counters' / name).read_text())
                                 for name in names}
    if not counters:
        raise RuntimeError('RDMA counter sampling requested but no devices visible')
    return counters


@contextmanager
def prepared_rocenante(adapter, device, cpu_group):
    """Prepare the adapter's real serving declaration without loading a model."""
    if adapter is None:
        yield
        return
    import torch
    import torch.distributed as dist
    from b12x.preparation import PreparationSession
    from vllm.utils.b12x import B12xWorkload

    workload = B12xWorkload(stage="weights", token_counts=(1,), fixed_token_counts=(),
                           output_dtype=torch.bfloat16, max_tokens=1, max_seqs=1,
                           max_model_len=1)
    units = adapter.get_b12x_preparation_units(adapter, workload)
    if not units:
        raise RuntimeError("RoCEnante declared no preparation work")

    def coordinate(state):
        if not state.ready_collectives:
            return None
        # All ranks prepare the same single transport. Wait outside the GPU
        # protocol until every compiler has reached the same priming request.
        key = state.ready_collectives[0].key
        keys = [None] * dist.get_world_size(cpu_group)
        dist.all_gather_object(keys, key, group=cpu_group)
        if any(other != key for other in keys):
            raise RuntimeError(f"collective preparation order mismatch: {keys}")
        return key

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(request for unit in units for request in unit.requests),
                        coordinator=coordinate)
        yield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, choices=(3, 4), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--benchmark", action="store_true",
                        help="after correctness, screen steady graph collective latency")
    parser.add_argument("--counter-samples", action="store_true",
                        help="record RDMA error deltas around each benchmark case")
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
    roce_enabled = os.environ.get("VLLM_ENABLE_ROCE_ALLREDUCE") == "1"
    roce_topology = os.environ.get("B12X_ROCE_TOPOLOGY", "direct")
    set_custom_all_reduce(roce_enabled)
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
    adapter = communicator.b12x_ar_comm
    if not roce_enabled and adapter is not None:
        raise RuntimeError("ring probe unexpectedly initialized a direct-peer B12X communicator")
    if roce_enabled and (adapter is None or adapter.disabled):
        raise RuntimeError("RoCEnante requested but unavailable; refusing a fallback-only pass")
    if roce_topology != "direct" and (adapter is None or getattr(adapter._runtime, "topology", None) != roce_topology):
        raise RuntimeError(f"the image does not contain the requested {roce_topology} transport")
    nccl = communicator.pynccl_comm
    if nccl is None or nccl.disabled:
        raise RuntimeError("PyNCCL communicator is unavailable")

    with prepared_rocenante(adapter, device, group.cpu_group):
        wave_setting = os.environ.get('B12X_ROCE_MESH_WAVE_BYTES')
        if wave_setting is not None and (adapter is None or
                adapter._runtime.stats().get('mesh_wave_bytes') != int(wave_setting)):
            raise RuntimeError('image did not apply the requested mesh wave threshold')
        # Exercise the torch NCCL group as well as vLLM's separate PyNCCL group.
        total = args.world_size * (args.world_size + 1) // 2
        control = torch.tensor([args.rank + 1], device=device, dtype=torch.float32)
        dist.all_reduce(control)
        torch.testing.assert_close(control, torch.full_like(control, total), rtol=0, atol=0)
        dist.broadcast(control, src=0)
        torch.testing.assert_close(control, torch.full_like(control, total), rtol=0, atol=0)

        checks = []
        before = adapter._runtime.stats()["ops_posted"] if adapter else 0
        for dtype in (torch.bfloat16, torch.float32):
            lengths = {1, 17, 1024, 2 * 1024 * 1024}
            if adapter:
                for limit in (adapter._runtime.max_size, adapter._runtime.max_gather_bytes):
                    # Exercise the exact dispatch boundary and both adjacent
                    # 16-byte packs for each operation/dtype.
                    lengths.update((limit // dtype.itemsize + offset) for offset in
                                   (-16 // dtype.itemsize, 0, 16 // dtype.itemsize))
            for length in sorted(lengths):
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

        timings = []
        if args.benchmark:
            # A graph contains 16 independent calls, replayed 16 times per
            # sample. This amortizes Python launch overhead. Report every rank;
            # analysis takes the slowest rank per sample, then the median.
            for dtype in (torch.bfloat16, torch.float32):
                for length in (5120, 30720, 245760, 1048576):
                    local = torch.full((length,), args.rank + 1, device=device, dtype=dtype)
                    scattered = torch.cat([local + 8 * p for p in range(args.world_size)])
                    operations = {
                        'all_reduce': lambda: group.all_reduce(local),
                        'all_gather': lambda: group.all_gather(local, dim=0),
                        'reduce_scatter': lambda: group.reduce_scatter(scattered, dim=0),
                    }
                    for name, operation in operations.items():
                        operation()
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with graph_capture(device=device) as context:
                            with torch.cuda.graph(graph, stream=context.stream):
                                outputs = [operation() for _ in range(16)]
                        for _ in range(4): graph.replay()
                        torch.cuda.synchronize()
                        counters_before = rdma_error_counters() if args.counter_samples else None
                        samples = []
                        for _ in range(5):
                            dist.barrier(group=group.cpu_group)
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(16): graph.replay()
                            end.record(); end.synchronize()
                            samples.append(start.elapsed_time(end) * 1000 / 256)
                        expected = (torch.full_like(local, total) if name == 'all_reduce'
                                    else torch.cat([torch.full_like(local, p + 1) for p in range(args.world_size)])
                                    if name == 'all_gather' else
                                    torch.full_like(local, total + 8 * args.rank * args.world_size))
                        for output in outputs:
                            torch.testing.assert_close(output, expected, rtol=0, atol=0)
                        row = {'dtype': str(dtype), 'elements_per_rank': length,
                               'operation': name, 'microseconds_per_call': samples,
                               'calls_per_graph': 16, 'replays_per_sample': 16}
                        if counters_before is not None:
                            counters_after = rdma_error_counters()
                            row['rdma_error_deltas'] = {
                                dev: {key: value - counters_before[dev][key] for key, value in values.items()}
                                for dev, values in counters_after.items()}
                        timings.append(row)
                        print(json.dumps({'rank': args.rank, 'timing': row}), flush=True)
                        del graph, outputs
                    del local, scattered

        if adapter:
            adapter.check_health()
            if adapter._runtime.stats()["ops_posted"] <= before:
                raise RuntimeError("no probe payload used the RoCEnante proxy")
            proxy_stats = adapter._runtime.stats()
        else:
            proxy_stats = None
    print(json.dumps({"rank": args.rank, "world_size": args.world_size,
                      "transport": f"rocenante-{roce_topology}" if roce_enabled else "nccl-ring",
                      "proxy": proxy_stats,
                      "passed": True, "checks": checks, "timings": timings}), flush=True)
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
