"""One-variable screens of pitched DMA and a 128-thread scalar copy."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from cuda.bindings import runtime as cudart
from paired import classify, measure
from vllm.tilelang_utils import T, tilelang
from vllm.v1.worker.gpu.sample import tilelang_sampling as ops

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()


@tilelang.jit(pass_configs=ops._PASS)
def scalar_copy():
    rows, cols, stride = T.dynamic("rows, cols, stride")

    @T.prim_func
    def copy_fp32_scalar(
        src: T.StridedTensor((rows, cols), (stride, 1), T.float32),
        dst: T.Tensor((rows, cols), T.float32),
    ):
        with T.Kernel(T.ceildiv(cols, 256), rows, threads=128) as (block, row):
            for i in T.unroll(2):
                col = block * 256 + i * 128 + T.get_thread_binding()
                if col < cols:
                    dst[row, col] = src[row, col]

    return copy_fp32_scalar


def profile(label, launch):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as p:
        launch()
    path = args.out / (label + ".trace.json")
    p.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    return dict(kernels=[e["name"] for e in events if e.get("cat") == "kernel"],
                memory=[e["name"] for e in events if str(e.get("cat", "")).startswith("gpu_mem")],
                trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


torch.manual_seed(20261105)
flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
rows = []
started = time.monotonic()
for count, cols in ((8, 129280), (32, 129280), (96, 129280), (8, 129281)):
    source = torch.randn((count, cols + 16), device="cuda")[:, :cols]
    out = torch.empty((count, cols), device="cuda")
    expected = source.clone()
    vector = source.stride(0) % 16 == 0 and cols % 4 == 0
    span = 4096 if vector and count >= 32 else 1024
    old = ops._kernel(ops.copy_logits_kernel, "float32", vector, span)

    def reference():
        return out.copy_(source)

    def previous():
        old(source, out)
        return out

    def scalar():
        scalar_copy()(source, out)
        return out

    def dma():
        with torch.cuda.device(source.device):
            error, = cudart.cudaMemcpy2DAsync(
                out.data_ptr(), out.stride(0) * 4, source.data_ptr(), source.stride(0) * 4,
                cols * 4, count, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                torch.cuda.current_stream(source.device).cuda_stream)
        assert error == cudart.cudaError_t.cudaSuccess, error
        return out

    for backend, launch in (("scalar128", scalar), ("pitched_dma", dma)):
        measured, _ = measure(dict(reference=reference, candidate=launch,
                                    previous=previous, null=reference), flush, 20261105,
                               expected, rounds=120)
        row = dict(shape=[count, cols], stride=list(source.stride()), backend=backend,
                   source_sha256=hashlib.sha256(source.cpu().numpy().tobytes()).hexdigest(),
                   measured=measured, gate=classify(measured["log_ratios"]["candidate"]),
                   profile=profile(f"{count}-{cols}-{backend}", launch))
        rows.append(row)
        (args.out / "result.json").write_text(json.dumps(dict(cases=rows,
            seconds=time.monotonic() - started), indent=2) + "\n")
        print(count, cols, backend, row["gate"], flush=True)
