"""Candidate-only launch-placement screen; no released build is remeasured.

The copy chain and optional SM/memory copy are controlled contention workloads,
not a model-layer performance claim. Final acceptance needs serving boot/repeat.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist
import tilelang
import tilelang.language as T
from sparknet import oneshot
from sparknet.oneshot import _oneshot_tilelang, _allgather_tilelang


@tilelang.jit
def copy_kernel(fixed_grid: int):
    n = T.dynamic("n")
    @T.prim_func
    def copy(src: T.Tensor((n,), T.bfloat16), dst: T.Tensor((n,), T.bfloat16)):
        with T.Kernel(fixed_grid if fixed_grid else T.ceildiv(n,256), threads=256) as bx:
            T.pdl_sync()
            T.pdl_trigger()
            tx = T.get_thread_binding()
            if fixed_grid:
                for step in T.serial(T.ceildiv(n, fixed_grid*256)):
                    i = bx*256+tx+step*fixed_grid*256
                    if i < n: dst[i] = src[i]
            else:
                i = bx*256+tx
                if i < n: dst[i] = src[i]
    return copy


def digest(tensor):
    return hashlib.sha256(tensor.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=("entry","staged","arrived"))
    p.add_argument("--source-commit", required=True)
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--calls-per-graph", type=int, default=16)
    args = p.parse_args()
    started = time.time()
    assert args.samples >= 20 and args.calls_per_graph > 0
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 3, "this screen is qualified only as a TP3 triangle plan"
    runtime = oneshot.AllReduce.from_exchange_group(exchange_group=dist.group.WORLD,
        device=torch.device("cuda",0), max_size=1<<20, max_gather_bytes=4<<20)
    runtime.prepare((torch.bfloat16,), padded_gather=True)
    copier, side_copy = copy_kernel(0), copy_kernel(64)
    stream, side = torch.cuda.Stream(), torch.cuda.Stream()
    scratch = torch.full((16<<20,), 0.5, dtype=torch.bfloat16, device="cuda")
    scratch_out = torch.empty_like(scratch)
    records = []
    kernels = {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
               for module in (_oneshot_tilelang,_allgather_tilelang)}
    try:
        for rows in (1,6,9,16,24,48,96):
            n = rows*5120
            seed = torch.full((n,), rank+1, dtype=torch.bfloat16, device="cuda")
            staged, reduced = torch.empty_like(seed), torch.empty_like(seed)
            gathered = torch.empty((world*n,), dtype=torch.bfloat16, device="cuda")
            observed = torch.empty_like(gathered)
            for operation in ("all_reduce","all_gather","mixed"):
                output = observed[:n] if operation == "all_reduce" else observed
                if operation == "all_gather":
                    expected = torch.cat([torch.full_like(seed, peer+1) for peer in range(world)])
                else:
                    expected = torch.full_like(output, world*(world+1)//2)
                for background_mib in (0,4,32):
                    background_elements = background_mib*(1<<19)
                    def chain():
                        copier(seed,staged)
                        if background_elements:
                            side.wait_stream(stream)
                            with torch.cuda.stream(side):
                                side_copy(scratch[:background_elements],scratch_out[:background_elements])
                        if operation in ("all_reduce","mixed"):
                            runtime.all_reduce(staged,out=reduced)
                        if operation in ("all_gather","mixed"):
                            runtime.all_gather(reduced if operation == "mixed" else staged,dim=0,out=gathered)
                        copier(reduced if operation == "all_reduce" else gathered,output)
                        if background_elements: stream.wait_stream(side)
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(4): chain()
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize();dist.barrier()
                    assert torch.equal(output,expected), (rows,operation,background_mib,"warmup")
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream),runtime.capture(stream=stream):
                        for _ in range(args.calls_per_graph): chain()
                    for _ in range(3): graph.replay()
                    torch.cuda.synchronize();dist.barrier()
                    samples=[]
                    for sample in range(args.samples):
                        dist.barrier()
                        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();graph.replay();end.record();end.synchronize()
                        samples.append(start.elapsed_time(end)*1000/args.calls_per_graph)
                    runtime.check_health()
                    assert torch.equal(output,expected), (rows,operation,background_mib,"timed")
                    if background_elements:
                        assert torch.equal(scratch_out[:background_elements],scratch[:background_elements])
                    rank_samples=[None]*world
                    dist.all_gather_object(rank_samples,samples)
                    worst=[max(values[i] for values in rank_samples) for i in range(len(samples))]
                    record={"rows":rows,"bytes_per_rank":n*2,"operation":operation,
                            "background_mib":background_mib,"rank_samples_us":rank_samples,
                            "slowest_rank_mean_us":statistics.mean(worst),
                            "within_process_ci95_us":1.96*statistics.stdev(worst)/math.sqrt(len(worst)),
                            "output_sha256":digest(output),"exact":True}
                    records.append(record)
                    if rank == 0: print(json.dumps({"case":record}),flush=True)
                    del graph
            del seed,staged,reduced,gathered,observed
        torch.cuda.synchronize()
        result={"variant":args.variant,"source_commit":args.source_commit,"run":args.run,
                "rank":rank,"world_size":world,"kernel_hashes":kernels,"cases":records,
                "process":{"pid":os.getpid(),"started_unix":started,"ended_unix":time.time()},
                "workload":{"width":5120,"dtype":"bfloat16","samples":args.samples,
                            "calls_per_graph":args.calls_per_graph,"eager_warmup":4,"graph_warmup":3,
                            "cache_policy":"fresh process; image cache plus source-hash JIT; no explicit eviction; data and side-copy buffers reused across replays",
                            "timing":"CUDA events per replay; Gloo barrier before each sample; slowest rank per sample"},
                "software":{"torch":torch.__version__,"cuda":torch.version.cuda,
                            "tilelang":getattr(tilelang,"__version__",None)},
                "device":{"name":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0)},
                "max_cuda_memory_allocated":torch.cuda.max_memory_allocated(),
                "max_cuda_memory_reserved":torch.cuda.max_memory_reserved(),
                "proxy":runtime.stats(),"passed":True,
                "scope":"synthetic launch/resource screen; not a serving performance result"}
        print("BENCH_RESULT "+json.dumps(result),flush=True)
    finally:
        runtime.close()
        dist.destroy_process_group()

if __name__ == "__main__": main()
