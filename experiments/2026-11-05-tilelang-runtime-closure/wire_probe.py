"""Inventory scoring through the real nonblocking CPU result conversion."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
torch.manual_seed(20261105)
rows = []
for dtype in (torch.int32, torch.int64):
    for layout in ("dense", "strided"):
        logits = torch.randn((3, 264), device="cuda")[:, :257]
        storage = torch.randint(257, (6,), device="cuda", dtype=dtype)
        sampled = storage[::2]
        if layout == "dense":
            sampled = sampled.contiguous()
        for k in (0, 5):
            for mode in ("logprobs", "logits"):
                def launch():
                    gpu = compute_topk_scores(logits, k, sampled, logits_mode=mode == "logits")
                    cpu = gpu.to_cpu_nonblocking()
                    return gpu, cpu

                launch()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA],
                                            with_stack=True) as profile:
                    gpu, cpu = launch()
                    torch.cuda.synchronize()
                assert cpu.logprob_token_ids.device.type == "cpu"
                assert torch.equal(cpu.logprob_token_ids[:, 0], sampled.cpu())
                assert cpu.logprob_token_ids.dtype == (dtype if k == 0 else torch.int64)
                label = f"{dtype}-{layout}-{k}-{mode}"
                path = args.out / (label + ".trace.json")
                profile.export_chrome_trace(str(path))
                events = json.loads(path.read_text())["traceEvents"]
                kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
                rows.append(dict(dtype=str(dtype), layout=layout, alternatives=k, mode=mode,
                    gpu_id_strides=list(gpu.logprob_token_ids.stride()),
                    cpu_ids_pinned=cpu.logprob_token_ids.is_pinned(),
                    kernels=kernels, aten_kernels=[name for name in kernels if "at::native" in name],
                    memory=[e["name"] for e in events if str(e.get("cat", "")).startswith("gpu_mem")],
                    trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                print(label, rows[-1]["aten_kernels"], flush=True)
                (args.out / "result.json").write_text(json.dumps(dict(cases=rows), indent=2) + "\n")
