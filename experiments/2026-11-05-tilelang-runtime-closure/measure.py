#!/usr/bin/env python3
"""Pinned combined-source checks and balanced, cold-cache GPU measurements."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=("tests", "pair", "bench", "iket"))
parser.add_argument("--family", choices=("sampler", "reduction", "gather", "all"), default="sampler")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--rounds", type=int, default=120)
parser.add_argument("--case-filter", type=Path)
args = parser.parse_args()
source = Path("/opt/spark-ds41f/candidate/vllm")
base = json.loads((HERE / "base.json").read_text())
candidate = json.loads((HERE / "candidate.json").read_text())
for relative, expected in base["image_files"].items():
    path = source / relative
    assert (hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None) == expected, relative
for relative, expected in candidate["files"].items():
    payload = (HERE / "overlay" / relative).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected, relative
    path = source / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)

import torch
from paired import classify, measure

torch.manual_seed(20261105)
started = time.monotonic()
identity = dict(base=base, candidate=candidate, family=args.family, operation=args.operation,
                torch=torch.__version__, gpu=torch.cuda.get_device_name())
(args.out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")

if args.operation == "tests":
    import pytest

    class Report:
        def __init__(self):
            self.outcomes = Counter()

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.failed or report.skipped:
                self.outcomes[report.outcome] += 1

    report = Report()
    tests = list(dict.fromkeys(t for family, entries in candidate["tests"].items()
                              if args.family == "all" or family == args.family for t in entries))
    code = pytest.main(["--noconftest", "-c", "/dev/null", "-q", "--maxfail=2",
                        "--junitxml=" + str(args.out / "tests.xml"),
                        *(str(source / test) for test in tests)], plugins=[report])
    result = dict(identity, pytest_exit=int(code), outcomes=dict(report.outcomes),
                  seconds=time.monotonic() - started)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(code)

assert args.operation in ("pair", "bench"), "IKET uses the combined inventory harness"
from vllm.v1.worker.gpu.sample import tilelang_sampling as sampling
from vllm.models.deepseek_v4_1.tilelang import collectives, gather

spec = importlib.util.spec_from_file_location("previous_sampling", HERE / "control/tilelang_sampling.py")
previous_sampling = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous_sampling
spec.loader.exec_module(previous_sampling)
flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")


def cuda_hashes():
    root = Path(__import__("os").environ["TILELANG_CACHE_DIR"])
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*.cu")}


def inventory(label, launch):
    launch()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        with torch.profiler.record_function(label):
            launch()
    path = args.out / (label + ".trace.json")
    profile.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    return dict(kernels=[dict(name=e["name"], duration_us=e["dur"], args=e.get("args", {}))
                         for e in kernels],
                memory=[dict(name=e["name"], category=e.get("cat")) for e in events
                        if str(e.get("cat", "")).startswith(("gpu_memcpy", "gpu_memset"))],
                trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                aten_kernels=[e["name"] for e in kernels
                              if "at::native" in e["name"] or "_ZN2at6native" in e["name"]])


def run_case(name, launches, expected, data, identical=False):
    if args.case_filter and name not in json.loads(args.case_filter.read_text()):
        return
    measured, _ = measure(launches, flush, 20261105, expected, args.rounds)
    profiles = {role: inventory(name + "-" + role, launch)
                for role, launch in launches.items() if role != "null"}
    row = dict(name=name, data=data, identical_program=identical,
               identity_reason="exact same factory, arguments, input/output bindings" if identical else None,
               measured=measured, gate=classify(measured["log_ratios"]["candidate"], identical),
               previous_gate=classify(measured["log_ratios"]["previous"]), profiles=profiles)
    rows.append(row)
    (args.out / "result.json").write_text(json.dumps(dict(identity, cases=rows,
        compiled_cuda=cuda_hashes(), seconds=time.monotonic() - started), indent=2) + "\n")
    print(name, row["gate"], flush=True)


rows = []
if args.family in ("sampler", "all"):
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for count in (1, 8, 96):
            for layout in ("dense", "padded"):
                cols = 129280
                pitch = cols + (16 if layout == "padded" else 0)
                src = torch.randn((count, pitch), dtype=dtype, device="cuda")[:, :cols]
                untouched = src.clone()
                out = torch.empty((count, cols), dtype=torch.float32, device="cuda")
                expected = src.float().clone()
                observed = []
                original_bind = sampling._kernel

                def record(factory, *key):
                    observed.append((factory, key))
                    return original_bind(factory, *key)

                sampling._kernel = record
                try:
                    actual = sampling.copy_logits(src)
                finally:
                    sampling._kernel = original_bind
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                assert actual.data_ptr() != src.data_ptr()
                reference = lambda: out.copy_(src)
                if observed:
                    assert len(observed) == 1
                    factory, key = observed[0]
                    kernel = original_bind(factory, *key)

                    def launch(kernel=kernel, src=src, out=out):
                        kernel(src, out)
                        return out

                    old = previous_sampling._kernel(previous_sampling.copy_logits_kernel, sampling._DTYPES[dtype])

                    def before(old=old, src=src, out=out):
                        old(src, out)
                        return out
                else:
                    launch = before = reference
                run_case(f"sampler-{dtype}-{count}-{layout}",
                         dict(reference=reference, candidate=launch, previous=before, null=reference),
                         expected, dict(shape=list(src.shape), strides=list(src.stride()),
                                        source_address=src.data_ptr(), output_address=out.data_ptr(),
                                        selected_key=list(observed[0][1]) if observed else "native DMA",
                                        seed=20261105), identical=not observed)
                assert torch.equal(src.view(torch.uint8), untouched.view(torch.uint8))

if args.family in ("reduction", "all"):
    for world in (3, 4, 8):
        for dtype in (torch.float32, torch.bfloat16):
            for count, width, padded in ((7, 513, True), (1, 5120, False),
                                         (96, 5120, False), (4096, 512, True)):
                parts = [torch.randn((count, width + (8 if padded else 0)), dtype=dtype,
                                    device="cuda")[:, :width] for _ in range(world)]
                out = torch.empty((count, width), dtype=dtype, device="cuda")
                packed = dtype == torch.bfloat16 and not padded
                expected = parts[0].float().clone()
                for part in parts[1:]:
                    expected.add_(part.float())
                expected = expected.to(dtype)

                def reference(parts=parts, out=out, packed=packed, world=world):
                    if packed:
                        collectives.rank_order_sum_kernel(world)(
                            *(part.view(-1, 8) for part in parts), out.view(-1, 8))
                    else:
                        total = parts[0].to(torch.float32, copy=True)
                        for part in parts[1:]:
                            total.add_(part.float())
                        out.copy_(total)
                    return out

                def launch(parts=parts, out=out):
                    collectives.rank_order_sum(parts, out)
                    return out

                run_case(f"reduction-{world}-{dtype}-{count}-{width}-{padded}",
                         dict(reference=reference, candidate=launch, previous=launch, null=reference),
                         expected, dict(world=world, shape=[count, width], padded=padded), identical=packed)

if args.family in ("gather", "all"):
    for world, parts, local, width in ((3, 1, 128, 320), (4, 2, 128, 512), (8, 2, 64, 512)):
        for dtype in (torch.bfloat16, torch.float32):
            for count in (1, 7, 128, 4096):
                src = torch.randn((world * parts * count, local), dtype=dtype, device="cuda")
                outputs = tuple(torch.empty((count, width), dtype=dtype, device="cuda") for _ in range(parts))
                expected = tuple(src.view(world, parts, count, local).permute(1, 2, 0, 3)
                                 .reshape(parts, count, world * local)[p, :, :width].contiguous()
                                 for p in range(parts))

                def reference(src=src, outputs=outputs, world=world, parts=parts, count=count, local=local, width=width):
                    full = collectives.concat_rank_shards(src, world, (parts, count, local))
                    for p, out in enumerate(outputs):
                        out.copy_(full[p, :, :width])
                    return outputs

                def launch(src=src, outputs=outputs, world=world):
                    gather.gather_columns_into(src, outputs, world)
                    return outputs

                run_case(f"gather-{world}-{parts}-{dtype}-{count}-{local}-{width}",
                         dict(reference=reference, candidate=launch, previous=launch, null=reference),
                         expected, dict(world=world, parts=parts, shape=[count, width]))
    for dtype in (torch.bfloat16, torch.float32):
        for count in (7, 129):
            src = torch.randn((3 * count, 5), dtype=dtype, device="cuda")
            out = torch.empty((count, 15), dtype=dtype, device="cuda")
            expected = src.view(3, count, 5).movedim(0, 1).reshape(count, 15)

            def reference(src=src, out=out, count=count):
                out.view(count, 3, 5).copy_(src.view(3, count, 5).movedim(0, 1))
                return out

            def launch(src=src, out=out, count=count):
                collectives.rank_rows_bytes_kernel()(src.view(-1).view(torch.uint8),
                    out.view(-1).view(torch.uint8), 3, count, 5 * src.element_size())
                return out

            run_case(f"gather-tail-{dtype}-{count}",
                     dict(reference=reference, candidate=launch, previous=launch, null=reference),
                     expected, dict(world=3, shape=[count, 15],
                                    reference="same TensorIterator copy as movedim reshape, shared output"))

assert rows
print("completed", len(rows), "cases", flush=True)
