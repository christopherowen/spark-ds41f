"""Compile the changed source for SM121 without any GPU device access."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import resource
import time
import tilelang
import tvm
from sparknet.oneshot import _oneshot_tilelang as reduce
from sparknet.oneshot import _allgather_tilelang as gather

# Hardware constant from the GB10 source contract; no CUDA device/property query.
reduce._resident_blocks = gather._resident_blocks = lambda threads: 1536 // threads
parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=("entry","staged","arrived"), default="entry")
args = parser.parse_args()
out = Path("/out")
target = tvm.target.Target({"kind":"cuda", "arch":"sm_121"})
records = []
for world, ring in [(3,False),(4,True)]:
    for rank in range(world):
        for dtype in ["bfloat16", "float16", "float32", "gather"]:
            label = f"tp{world}-rank{rank}-{dtype}"
            started = time.monotonic()
            factory = (gather._build(world,rank,512,2,128,2,ring) if dtype == "gather" else
                       reduce._build(dtype,world,rank,512,2,128,2,ring))
            with target, tvm.transform.PassContext(config=factory.pass_configs):
                artifact = tilelang.lower(factory.get_tir(), target=target, enable_device_compile=True)
            source = artifact.kernel_source
            kernels = list(re.finditer(r"__global__\s+void\s+[^;{]+\{", source))
            assert len(kernels) == 1, label
            for kernel in kernels:
                body = source[kernel.end():]
                wait = body.index("cudaGridDependencySynchronize()")
                trigger = body.index("cudaTriggerProgrammaticLaunchCompletion()")
                epoch = body.index("roce_ld_relaxed_gpu_u32(")
                assert wait < epoch and wait < trigger, label
                if args.phase == "entry": assert trigger < epoch, label
                elif args.phase == "staged": assert trigger > body.index("roce_doorbell("), label
                else: assert trigger > body.index("roce_wait_peers("), label
                assert "roce_" not in body[:wait], label
                assert not re.search(r"\b(?:inp|out|counters)\s*\[", body[:wait]), label
            functions = list(artifact.device_mod.functions.values())
            assert functions and all(f.attrs.get("has_cuda_pdl_sync", False) for f in functions), label
            (out/f"{label}.cu").write_text(source)
            records.append({"case":label,"seconds":time.monotonic()-started,
                            "cuda_sha256":hashlib.sha256(source.encode()).hexdigest(),
                            "wait_before_epoch":True,"pdl_launch_attribute":True})
            (out/"compile-results.json").write_text(json.dumps({"phase":args.phase,"cases":records,"gpu_device_access":False,
               "resident_threads_per_sm_assumed":1536,"process_maxrss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2)+"\n")
            print(label, "compiled and ordering checked", flush=True)
            del factory,artifact,source;gc.collect()
print("PASS",len(records),"SM121 specializations; no GPU execution",flush=True)
