#!/usr/bin/env python3
"""Independent correctness, CUDA launch inventory and paired kernel timings."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=["tests", "bench", "iket"])
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

import torch  # noqa: E402 -- frozen source is verified and overlaid before import.

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


if args.operation == "iket":
    harness = (
        HERE.parent / "2026-10-31-iket-full-43l-and-runtime-opt/harness/iket_profile.py"
    )
    cmd = [
        sys.executable,
        str(harness),
        str(args.out / "profile"),
        "--timeout",
        "240",
        "--single-pass-max-ts",
        "64",
        "--pool-gb",
        "0.25",
        "--skip-raw-data-dump",
        "--",
        sys.executable,
        str(HERE / "iket_probe.py"),
        "--out",
        str(args.out / "probe"),
    ]
    (args.out / "iket-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    subprocess.run(cmd, check=True)
    paths = list((args.out / "profile/trace").rglob("iket.decoded_results.json"))
    assert paths, "no decoded IKET data"
    launches = []
    for path in paths:
        decoded = json.loads(path.read_text())
        strings = decoded["stringTable"]
        captured = list(decoded["launches"])
        graphs = decoded.get("graphLaunches") or {}
        for value in graphs.values() if isinstance(graphs, dict) else graphs:
            items = value if isinstance(value, list) else value.get("launches", [])
            captured.extend(
                item for item in items if isinstance(item, dict) and "ranges" in item
            )
        for launch in captured:
            if "gather_logits" not in launch["kernelName"]:
                continue
            ranges = launch["ranges"]
            assert ranges and launch["warpLifetimes"], "missing phase or lifetime data"
            names = [strings[r["rangeNameIdx"]] for r in ranges]
            assert any(n.startswith("load.") for n in names), names
            assert any(n.startswith("store.") for n in names), names
            assert all(r["endTs"] >= r["startTs"] for r in ranges)
            launches.append(
                dict(
                    kernel=launch["kernelName"],
                    grid=launch["gridId"],
                    range_names=names,
                    warps=len(launch["warpLifetimes"]),
                )
            )
    assert len(launches) == 12, f"expected 12 gathered-logits launches: {len(launches)}"
    result = dict(
        identity,
        launches=launches,
        decoded_files=[str(p) for p in paths],
        seconds=time.monotonic() - started,
    )
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0)


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
    events = [
        e["name"]
        for e in json.loads(path.read_text())["traceEvents"]
        if e.get("cat") == "kernel"
    ]
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


from vllm.v1.worker.gpu.sample import tilelang_logprob as ops  # noqa: E402


def cases():
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for rows in (1, 16, 96):
            for masked in (False, True):
                for strided in (False, True):
                    width, cols = 129280, 129 if masked else 33
                    step = 2 if strided else 1
                    src = torch.randn(
                        rows, width * step + 7, device="cuda", dtype=dtype
                    )[:, : width * step : step]
                    ids = torch.randint(width, (rows, cols * step), device="cuda")[
                        :, ::step
                    ]
                    valid = (
                        (torch.rand(rows, cols * step, device="cuda") > 0.3)[:, ::step]
                        if masked
                        else None
                    )
                    before = src.clone()

                    def old():
                        scores = src.gather(-1, ids).to(torch.float32)
                        return (
                            scores
                            if valid is None
                            else scores.masked_fill(~valid, -torch.inf)
                        )

                    def new():
                        return ops.gather_logits(src, ids, valid)

                    def validate():
                        assert torch.equal(
                            new().view(torch.int32), old().view(torch.int32)
                        )
                        assert torch.equal(src, before)

                    yield (
                        dict(
                            dtype=str(dtype),
                            rows=rows,
                            width=width,
                            cols=cols,
                            masked=masked,
                            strided=strided,
                        ),
                        old,
                        new,
                        validate,
                    )


rows = []
profiles = {}
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
    profile_key = label["dtype"] + ("-masked" if label["masked"] else "-unmasked")
    if profile_key not in profiles:
        profiles[profile_key] = dict(
            reference=inventory(profile_key + "-reference", old),
            candidate=inventory(profile_key + "-candidate", new),
        )
        assert len(profiles[profile_key]["candidate"]["events"]) == 1, profiles[
            profile_key
        ]
        assert not profiles[profile_key]["candidate"]["aten_kernels"], profiles[
            profile_key
        ]
result = dict(
    identity,
    cases=rows,
    launch_inventory=profiles,
    seconds=time.monotonic() - started,
    scope="isolated selected-score kernels and runner components; no full-model serving throughput or fabric qualification",
)
(args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
