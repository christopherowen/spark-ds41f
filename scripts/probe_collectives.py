#!/usr/bin/env python3
"""Model-free qualification of the actual vLLM TP communicator and CUDA graphs.

Run one rank per node using commands rendered by `spark3 topology probe`.
The surrounding container supplies a 600-second timeout and a 12 GiB limit (two NCCL groups plus compilation).
This checks uniform TP collectives, not EP all-to-all or arbitrary send/recv.
"""

from __future__ import annotations

from contextlib import contextmanager
import argparse
import hashlib
from datetime import timedelta
import json
import os
import subprocess
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


def port_counters():
    names = ('tx_bytes_phy', 'rx_bytes_phy', 'rx_out_of_buffer',
             'tx_vport_rdma_unicast_bytes', 'rx_vport_rdma_unicast_bytes')
    counters = {}
    for hca in Path('/sys/class/infiniband').iterdir():
        for netdev in (hca / 'device/net').iterdir():
            output = subprocess.check_output(['ethtool', '-S', netdev.name], text=True, timeout=10)
            values = {key.strip(): int(value.strip()) for line in output.splitlines()
                      if ':' in line for key, value in [line.split(':', 1)]
                      if value.strip().isdigit()}
            counters[netdev.name] = {name: values[name] for name in names}
    if not counters:
        raise RuntimeError('port counter sampling requested but no RDMA netdevs visible')
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
    parser.add_argument("--port-samples", action="store_true",
                        help="sample physical NIC bytes and buffer drops around each timed case")
    parser.add_argument("--expect-paths", type=int, choices=(2, 4))
    parser.add_argument('--lengths', type=int, nargs='+', default=[5120,30720,245760,1048576])
    parser.add_argument('--numerics', action='store_true')
    args = parser.parse_args()
    # A full 4096-row, 5120-wide TP4 prefill has 5,242,880 elements
    # per shard (10 MiB BF16). Sixteen FP32 all-gather outputs at this
    # bound occupy 1.25 GiB; retain the runner's 12 GiB container limit.
    if any(n < 64 or n > 5242880 or n % 8 for n in args.lengths):
        parser.error('benchmark lengths must be aligned and between 64 and 5242880')
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
        dispatch_setting = os.environ.get('B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES')
        if dispatch_setting is not None and (adapter is None or
                adapter._runtime.stats().get('dispatch_max_bytes') != int(dispatch_setting)):
            raise RuntimeError('image did not apply the requested independent dispatch limit')
        wave_setting = os.environ.get('B12X_ROCE_MESH_WAVE_BYTES')
        if wave_setting is not None and (adapter is None or
                adapter._runtime.stats().get('mesh_wave_bytes') != int(wave_setting)):
            raise RuntimeError('image did not apply the requested mesh wave threshold')
        if args.expect_paths is not None:
            if adapter is None or adapter._runtime.stats().get('path_slots') != args.expect_paths:
                raise RuntimeError('image did not apply the requested path count')
            requested_rotate = int(os.environ.get('B12X_ROCE_MESH_ROTATE', '0'))
            if adapter._runtime.stats().get('mesh_rotate') != requested_rotate:
                raise RuntimeError('image did not apply the requested posting order')
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
                for limit in (adapter._runtime.max_size, adapter._runtime.max_gather_bytes,
                              getattr(adapter._runtime, "dispatch_max_bytes", adapter._runtime.max_size)):
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

        numerical_checks = []
        if args.numerics:
            # Identical first 64 elements at every capacity; cancellation-sensitive
            # inputs distinguish floating reduction order from transport corruption.
            for dtype in (torch.bfloat16, torch.float32):
                base = torch.arange(64, dtype=torch.float32) % 8
                inputs = [(base + 1) * 16777216, (base + 1) / 16,
                          -(base + 1) * 16777216, (base + 1) / 32]
                inputs = [v.to(dtype) for v in inputs]
                reference = inputs[0].float()
                for v in inputs[1:]: reference = reference + v.float()
                reference = reference.to(dtype)
                for length in args.lengths:
                    local = inputs[args.rank].to(device).repeat((length+63)//64)[:length].contiguous()
                    output = group.all_reduce(local)
                    torch.cuda.synchronize()
                    observed = output[:64].cpu()
                    row = {'dtype':str(dtype), 'elements_per_rank':length,
                           'prefix_sha256':hashlib.sha256(observed.view(torch.uint8).numpy().tobytes()).hexdigest(),
                           'rank_order_fp32_prefix_sha256':hashlib.sha256(reference.view(torch.uint8).numpy().tobytes()).hexdigest(),
                           'mismatched_reference_elements':int((observed != reference).sum()),
                           'max_abs_reference_difference':float((observed.float()-reference.float()).abs().max())}
                    numerical_checks.append(row)
                    print(json.dumps({'rank':args.rank,'numerical_check':row}),flush=True)
                    del local,output
        timings = []
        if args.benchmark:
            # A graph contains 16 independent calls, replayed 16 times per
            # sample. This amortizes Python launch overhead. Report every rank;
            # analysis takes the slowest rank per sample, then the median.
            for dtype in (torch.bfloat16, torch.float32):
                for length in args.lengths:
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
                        if args.counter_samples or args.port_samples:
                            dist.barrier(group=group.cpu_group)
                        counters_before = rdma_error_counters() if args.counter_samples else None
                        ports_before = port_counters() if args.port_samples else None
                        proxy_before = adapter._runtime.stats() if adapter and args.port_samples else None
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
                        if args.counter_samples or args.port_samples:
                            dist.barrier(group=group.cpu_group)
                        if counters_before is not None:
                            counters_after = rdma_error_counters()
                            row['rdma_error_deltas'] = {
                                dev: {key: value - counters_before[dev][key] for key, value in values.items()}
                                for dev, values in counters_after.items()}
                        if proxy_before is not None:
                            expected_custom = (name == 'all_reduce' and adapter.should_custom_ar(local)) or (
                                name == 'all_gather' and adapter.should_all_gather(local, 0))
                            row['expected_backend'] = 'rocenante' if expected_custom else 'nccl'
                            proxy_after = adapter._runtime.stats()
                            row['proxy_payload_bytes'] = {h: after - before for h, after, before in zip(
                                proxy_after['hcas'], proxy_after['bytes_posted_per_hca'],
                                proxy_before['bytes_posted_per_hca'])}
                            if bool(sum(row['proxy_payload_bytes'].values())) != expected_custom:
                                raise RuntimeError(f'{name}: actual transport counters disagree with dispatch policy')
                        if ports_before is not None:
                            ports_after = port_counters()
                            row['port_deltas'] = {
                                dev: {key: value - ports_before[dev][key] for key, value in values.items()}
                                for dev, values in ports_after.items()}
                        if args.counter_samples or args.port_samples:
                            # No rank starts the next case while another rank
                            # is still reading its counters for this case.
                            dist.barrier(group=group.cpu_group)
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
                      "passed": True, "checks": checks, "numerical_checks": numerical_checks, "timings": timings}), flush=True)
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
