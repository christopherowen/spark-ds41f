#!/usr/bin/env python3
"""Independent correctness, CUDA launch inventory and paired kernel timings."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=["tests", "bench"])
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
base = json.loads((HERE / "base.json").read_text())
candidate = json.loads((HERE / "candidate.json").read_text())
source = Path("/opt/spark-ds41f/candidate/vllm")
for relative, expected in base["files"].items():
    path = source / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    assert actual == expected, f"original source mismatch: {relative}"
for path in (HERE / "overlay").rglob("*.py"):
    target = source / path.relative_to(HERE / "overlay")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
for relative, expected in candidate["files"].items():
    assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected, (
        relative
    )

import torch

assert torch.cuda.is_available()
started = time.monotonic()
identity = dict(
    base=base,
    candidate=candidate,
    torch=torch.__version__,
    gpu=torch.cuda.get_device_name(),
    operation=args.operation,
)
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
    code = pytest.main(
        [
            "--noconftest",
            "-c",
            "/dev/null",
            "-q",
            "--maxfail=2",
            "--junitxml=" + str(args.out / "tests.xml"),
            *(str(source / f) for f in candidate["tests"]),
        ],
        plugins=[report],
    )
    result = dict(
        identity,
        pytest_exit=int(code),
        outcomes=dict(report.outcomes),
        seconds=time.monotonic() - started,
    )
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(code)


def inventory(label, launch):
    launch()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        with torch.profiler.record_function(label):
            value = launch()
    path = args.out / (label + ".trace.json")
    profile.export_chrome_trace(str(path))
    del value
    events = [e.name for e in profile.events() if str(e.device_type).endswith("CUDA")]
    return dict(
        events=events,
        trace=str(path),
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        aten_kernels=[n for n in events if "at::native" in n or "_ZN2at6native" in n],
    )


flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")


def timed(launch):
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with torch.cuda.graph(graph):
        launch()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    samples = []
    for _ in range(21):
        flush.zero_()
        events[0].record()
        graph.replay()
        events[1].record()
        events[1].synchronize()
        samples.append(events[0].elapsed_time(events[1]) * 1000)
    return dict(
        samples_us=samples,
        median_us=statistics.median(samples),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        capture_allocated_delta_bytes=torch.cuda.memory_allocated() - before,
    )


from vllm.v1.worker.gpu.sample import tilelang_sampling as ops


def cases():
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for rows in (1, 8, 96):
            for padded in (True, False):
                width = 129280
                src = torch.randn(
                    (rows, width + (16 if padded else 0)), device="cuda", dtype=dtype
                )[:, :width]
                before = src.clone()

                def old():
                    return torch.empty_like(src, dtype=torch.float32).copy_(src)

                def new():
                    return ops.copy_logits(src)

                def validate():
                    torch.testing.assert_close(new(), old(), rtol=0, atol=0)
                    torch.testing.assert_close(src, before, rtol=0, atol=0)

                yield (
                    dict(dtype=str(dtype), rows=rows, width=width, padded=padded),
                    old,
                    new,
                    validate,
                )


rows = []
profiles = None
for label, old, new, validate in cases():
    validate()
    # Alternate ordering across cases; both methods get 21 cold-L2 graph samples.
    if len(rows) % 2:
        timings = dict(candidate=timed(new), reference=timed(old))
    else:
        timings = dict(reference=timed(old), candidate=timed(new))
    validate()
    rows.append(dict(case=label, timings=timings))
    (args.out / "cases.json").write_text(json.dumps(rows, indent=2) + "\n")
    if profiles is None:
        profiles = dict(
            reference=inventory("reference", old), candidate=inventory("candidate", new)
        )
        assert profiles["candidate"]["events"], "CUDA inventory unavailable"
        assert not profiles["candidate"]["aten_kernels"], (
            "remaining candidate ATen kernel"
        )
result = dict(
    identity,
    cases=rows,
    launch_inventory=profiles,
    seconds=time.monotonic() - started,
    scope="isolated kernels and local TP emulation; no serving throughput or fabric qualification",
)
(args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
