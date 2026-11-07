"""Isolated native BF16 projection schedule screen; no collective/serving timing."""
import argparse
import json
import math
import statistics
import time
import traceback

import vllm  # The image package has the six verified candidate file mounts.
import torch
from vllm.models.deepseek_v4_1.tilelang.gemm import bf16_gemm

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--boot", required=True)
p.add_argument("--repeats", type=int, default=15)
p.add_argument("--schedules", help="Comma-separated MxNxstagesxthreads; default is the full screen")
a = p.parse_args()
assert a.repeats >= 3
schedules = [(64, 64, 3), (32, 64, 2), (32, 64, 3), (32, 64, 4),
             (64, 64, 2), (64, 64, 4), (128, 64, 2), (128, 64, 3),
             (32, 128, 2), (32, 128, 3), (64, 128, 2), (64, 128, 3)]
schedules = [(*item, 128) for item in schedules]
schedules += [(m, n, stages, 256) for m, n in ((64, 128), (128, 64), (128, 128))
              for stages in (2, 3)]
if a.schedules:
    schedules = [tuple(map(int, item.split("x"))) for item in a.schedules.split(",")]
    schedules = [item if len(item) == 4 else (*item, 128) for item in schedules]


def emit(**record):
    print(json.dumps({"boot": a.boot, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      **record}), flush=True)


def kernel(columns, schedule=(64, 64, 3, 128)):
    m, n, stages, threads = schedule
    return bf16_gemm(columns, 5120, block_M=m, block_N=n, block_K=64,
                     num_stages=stages, threads=threads, shards=5, blocked=True,
                     out_dtype="bfloat16")


def timings(call, wash):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    # External event nodes record timestamps inside replay, removing Python
    # launch gaps. A wash, when selected, precedes the timed kernel in the graph.
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if wash:
            flush.add_(1)
        begin.record()
        call()
        end.record()
    times = []
    for _ in range(a.repeats):
        graph.replay()
        torch.cuda.synchronize()
        times.append(begin.elapsed_time(end))
    return times


gen = torch.Generator(device="cuda").manual_seed(3643)
weight = (torch.randn(512, 5120, device="cuda", generator=gen) * 0.02).bfloat16()
inputs = torch.randn(16384, 5120, device="cuda", generator=gen).bfloat16()
flush = torch.zeros(128 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
emit(kind="identity", gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
     torch=torch.__version__, cuda=torch.version.cuda, seed=3643, repeats=a.repeats,
     cache_modes=["warm", "128MiB wash before timed kernel"],
     scope="one rank's projection only; no collective, normalization, or serving time",
     timer="external CUDA event nodes inside graph replay; host launch gaps excluded",
     fixed={"K": 5120, "block_K": 64, "shards": 5, "blocked": True})
failures = []
for world, tokens in ((3, 6144), (4, 16384)):
    rows = math.ceil(tokens / world)
    cols = 128 if world == 4 else 512
    x = inputs[:tokens]
    local = x[:rows]
    native = kernel(cols)
    native_out = torch.empty(tokens, cols, device="cuda", dtype=torch.bfloat16)
    reference = []
    for start in range(0, 512, cols):
        native(x, weight[start:start + cols], native_out)
        reference.append(native_out[:rows].clone())
    expected = torch.cat(reference, dim=-1)
    native_weight = weight[:cols]
    variants = [("parent", lambda: native(x, native_weight, native_out))]
    outputs = []
    for schedule in schedules:
        name = "local-" + "x".join(map(str, schedule))
        try:
            started = time.monotonic()
            launch = kernel(512, schedule)
            out = torch.empty(rows, 512, device="cuda", dtype=torch.bfloat16)
            launch(local, weight, out)
            torch.cuda.synchronize()
            if not torch.equal(out.view(torch.int16), expected.view(torch.int16)):
                raise AssertionError("finite bits differ from native column shards")
            probe = local[:207].clone()
            probe[0, 0], probe[1, 1], probe[2, 2] = float("nan"), float("inf"), -float("inf")
            refs = []
            for start in range(0, 512, cols):
                ref = torch.empty(207, cols, device="cuda", dtype=torch.bfloat16)
                native(probe, weight[start:start + cols], ref)
                refs.append(ref)
            got = torch.empty(207, 512, device="cuda", dtype=torch.bfloat16)
            launch(probe, weight, got)
            if not torch.equal(got.view(torch.int16), torch.cat(refs, dim=-1).view(torch.int16)):
                raise AssertionError("nonfinite bits differ from native column shards")
            outputs.append(out)
            variants.append((name, lambda launch=launch, out=out: launch(local, weight, out)))
            emit(kind="gate", world=world, tokens=tokens, schedule=name, bits_equal=True,
                 compile_and_check_seconds=time.monotonic() - started)
        except Exception as exc:
            failures.append([world, name, str(exc)])
            emit(kind="error", world=world, tokens=tokens, schedule=name, error=str(exc),
                 traceback=traceback.format_exc())
    # Rotate the order between cache modes and boots; keep every raw sample.
    offset = sum(map(ord, a.boot)) % len(variants)
    variants = variants[offset:] + variants[:offset]
    for wash in (False, True):
        for name, call in (reversed(variants) if wash else variants):
            samples = timings(call, wash)
            emit(kind="timing", world=world, tokens=tokens, local_rows=rows, schedule=name,
                 wash=wash, samples_ms=samples, mean_ms=statistics.mean(samples),
                 stdev_ms=statistics.stdev(samples),
                 allocated_bytes=torch.cuda.memory_allocated(), peak_bytes=torch.cuda.max_memory_allocated())
    del variants, outputs, expected, reference
emit(kind="complete", failures=failures)
