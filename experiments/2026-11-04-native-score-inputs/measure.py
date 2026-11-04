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
parser.add_argument("operation", choices=["tests", "bench", "pair", "iket"])
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
base = json.loads((HERE / "base.json").read_text())
candidate = json.loads((HERE / "candidate.json").read_text())
source = Path("/opt/spark-ds41f/candidate/vllm")


def verify(files):
    for relative, expected in files.items():
        assert (
            hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected
        ), relative


def overlay(folder):
    for path in (folder / "overlay").rglob("*.py"):
        (source / path.relative_to(folder / "overlay")).write_bytes(path.read_bytes())


verify(base["image_files"])
prior = HERE.parents[1] / base["prerequisite"]
overlay(prior)
verify(json.loads((prior / "candidate.json").read_text())["files"])
verify(base["files"])
if args.operation in ("bench", "pair"):
    reference_dir = args.out / "reference"
    reference_dir.mkdir()
    for name in ("tilelang_logprob.py", "tilelang_topk_ids.py"):
        (reference_dir / name).write_bytes(
            (source / "vllm/v1/worker/gpu/sample" / name).read_bytes()
        )
overlay(HERE)
verify(candidate["files"])

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
        "128",
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
            if not any(
                n in launch["kernelName"]
                for n in ("topk_log_softmax", "topk_ids", "ranks")
            ):
                continue
            ranges = launch["ranges"]
            assert ranges and launch["warpLifetimes"], "missing phase or lifetime data"
            names = [strings[r["rangeNameIdx"]] for r in ranges]
            assert any(n.startswith("load.") for n in names), names
            assert all(r["endTs"] >= r["startTs"] for r in ranges)
            launches.append(
                dict(
                    kernel=launch["kernelName"],
                    grid=launch["gridId"],
                    range_names=names,
                    warps=len(launch["warpLifetimes"]),
                )
            )
    assert len(launches) == 40, (
        f"expected 40 score/selection/rank launches: {len(launches)}"
    )
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


from vllm.v1.worker.gpu.sample import logprob, tilelang_topk_ids  # noqa: E402


def load_reference(name, folder=None, prefix="reference_"):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        prefix + name, (reference_dir if folder is None else folder) / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old_scores = load_reference("tilelang_logprob")
old_topk = load_reference("tilelang_topk_ids")


def cases():
    for dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 16, 96):
            for id_dtype in (torch.int32, torch.int64):
                for layout in ("dense", "padded", "sliced", "transposed"):
                    width = 129280
                    seed = int.from_bytes(
                        hashlib.sha256(
                            f"{dtype}/{rows}/{id_dtype}/{layout}".encode()
                        ).digest()[:4],
                        "little",
                    )
                    torch.manual_seed(seed)
                    logits = torch.randn(rows, width, device="cuda", dtype=dtype)
                    for masked in (False, True):
                        cols = 129 if masked else 21
                        ids = torch.randint(
                            width, (rows, cols), device="cuda", dtype=id_dtype
                        )
                        if layout == "padded":
                            store = torch.zeros(
                                rows, cols + 7, device="cuda", dtype=id_dtype
                            )
                            store[:, :cols] = ids
                            ids = store[:, :cols]
                        elif layout == "sliced":
                            ids = torch.stack((ids, ids), dim=-1)[..., 0]
                        elif layout == "transposed":
                            ids = ids.t().contiguous().t()
                        valid = (
                            torch.rand(rows, cols, device="cuda") > 0.3
                            if masked
                            else None
                        )

                        def old():
                            native = ids.to(torch.int64).contiguous()
                            out = logits.new_empty(ids.shape, dtype=torch.float32)
                            old_scores.topk_log_softmax(logits, native, out, valid)
                            return out

                        def new():
                            return logprob.compute_token_logprobs(logits, ids, valid)

                        def validate():
                            assert torch.equal(
                                new().view(torch.int32), old().view(torch.int32)
                            )

                        new.inputs = ("scores", logits, ids, valid)
                        yield (
                            dict(
                                operation="scores",
                                dtype=str(dtype),
                                rows=rows,
                                width=width,
                                id_dtype=str(id_dtype),
                                layout=layout,
                                cols=cols,
                                masked=masked,
                            ),
                            old,
                            new,
                            validate,
                        )
                    if layout in ("dense", "sliced"):
                        sampled = torch.randint(
                            width, (rows, 2), device="cuda", dtype=id_dtype
                        )
                        sampled = (
                            sampled[:, 0]
                            if layout == "sliced"
                            else sampled[:, 0].contiguous()
                        )

                        def old():
                            return old_topk.logprob_token_ids(logits, 20, sampled)

                        def new():
                            return tilelang_topk_ids.logprob_token_ids(
                                logits, 20, sampled
                            )

                        def validate():
                            assert torch.equal(new(), old())

                        new.inputs = ("topk", logits, sampled, None)
                        yield (
                            dict(
                                operation="topk",
                                dtype=str(dtype),
                                rows=rows,
                                width=width,
                                id_dtype=str(id_dtype),
                                layout=layout,
                                cols=21,
                                masked=False,
                            ),
                            old,
                            new,
                            validate,
                        )


if args.operation == "pair":
    from paired import classify, measure
    from vllm.v1.worker.gpu.sample import tilelang_logprob as current_scores
    from vllm.v1.worker.gpu.sample import tilelang_sampling as tls

    control = HERE / "control/v3"
    control_id = json.loads((control / "identity.json").read_text())
    for name, sha in control_id["files"].items():
        assert hashlib.sha256((control / name).read_bytes()).hexdigest() == sha
    previous_scores = load_reference("tilelang_logprob", control, "previous_")
    previous_topk = load_reference("tilelang_topk_ids", control, "previous_")
    source_dir = args.out / "kernel-sources"
    source_dir.mkdir()

    def compiled(factory, key):
        kernel = factory(*key)
        data = kernel.get_kernel_source().encode()
        digest = hashlib.sha256(data).hexdigest()
        (source_dir / (digest + ".cu")).write_bytes(data)
        return digest

    rows = []
    for label, old, new, validate in cases():
        validate()
        operation, logits, ids, valid = new.inputs
        run = tls._triton_run(logits)
        alias = ids.dtype == torch.int64 and ids.is_contiguous()
        native = ids.to(torch.int64).contiguous()
        alias = alias and native.data_ptr() == ids.data_ptr()
        if operation == "scores":
            reference_code = compiled(
                old_scores.topk_log_softmax_kernel,
                old_scores.topk_log_softmax_key(logits.dtype, run, valid is not None),
            )
            previous_code = compiled(
                previous_scores.topk_log_softmax_kernel,
                previous_scores.topk_log_softmax_key(
                    logits.dtype, run, valid is not None, ids.dtype, ids.is_contiguous()
                ),
            )
            candidate_code = compiled(
                current_scores.topk_log_softmax_kernel,
                current_scores.topk_log_softmax_key(
                    logits.dtype,
                    run,
                    valid is not None,
                    ids.dtype,
                    current_scores._id_layout(ids),
                ),
            )
        else:
            reference_code = compiled(
                old_topk.topk_ids_kernel,
                old_topk.topk_ids_key(logits.dtype, torch.int64, run),
            )
            previous_code = compiled(
                previous_topk.topk_ids_kernel,
                previous_topk.topk_ids_key(logits.dtype, torch.int64, run, ids.dtype),
            )
            candidate_code = compiled(
                tilelang_topk_ids.topk_ids_kernel,
                tilelang_topk_ids.topk_ids_key(
                    logits.dtype, torch.int64, run, ids.dtype, ids.is_contiguous()
                ),
            )
        same_program = alias and candidate_code == reference_code
        # CUDA events exclude host allocation. Use the same destination for all
        # captured programs so output address/color is not another variable.
        expected = old().clone()
        destination = torch.empty_like(expected)
        if operation == "scores":

            def reference_launch():
                native_ids = ids.to(torch.int64).contiguous()
                old_scores.topk_log_softmax(logits, native_ids, destination, valid)
                return destination

            def candidate_launch():
                current_scores.topk_log_softmax(logits, ids, destination, valid)
                return destination

            def previous_launch():
                previous_scores.topk_log_softmax(logits, ids, destination, valid)
                return destination
        else:

            def reference_launch():
                old_topk._launch(logits, destination, ids.to(torch.int64).contiguous())
                return destination

            def candidate_launch():
                tilelang_topk_ids._launch(logits, destination, ids)
                return destination

            def previous_launch():
                previous_topk._launch(logits, destination, ids)
                return destination

        seed = int.from_bytes(
            hashlib.sha256(json.dumps(label, sort_keys=True).encode()).digest()[:4],
            "little",
        )
        data_id = dict(
            sample_order_seed=seed,
            data_seed=int.from_bytes(
                hashlib.sha256(
                    f"{logits.dtype}/{logits.shape[0]}/{ids.dtype}/{label['layout']}".encode()
                ).digest()[:4],
                "little",
            ),
            logits_sha256=hashlib.sha256(
                logits.cpu().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            id_strides=list(ids.stride()),
            reference_preparation_aliases_input=alias,
            shared_output_address=True,
        )
        measured, outputs = measure(
            dict(
                reference=reference_launch,
                candidate=candidate_launch,
                previous=previous_launch,
                null=reference_launch,
            ),
            flush,
            seed,
            expected,
        )
        assert torch.equal(
            outputs["candidate"].view(torch.int32),
            outputs["reference"].view(torch.int32),
        )
        assert torch.equal(
            outputs["previous"].view(torch.int32),
            outputs["reference"].view(torch.int32),
        )
        summary = classify(
            measured["log_ratios"]["candidate"], identical_program=same_program
        )
        rows.append(
            dict(
                case=label,
                data=data_id,
                same_program=same_program,
                cuda_source_sha256=dict(
                    reference=reference_code,
                    previous=previous_code,
                    candidate=candidate_code,
                ),
                measured=measured,
                gate=summary,
                previous_gate=classify(
                    measured["log_ratios"]["previous"],
                    identical_program=alias and previous_code == reference_code,
                ),
                null_control=classify(
                    measured["log_ratios"]["null"], identical_program=True
                ),
            )
        )
        (args.out / "cases.json").write_text(json.dumps(rows, indent=2) + "\n")
    failed = [r["case"] for r in rows if r["gate"]["status"] == "not_proved"]
    result = dict(
        identity,
        cases=rows,
        not_proved=failed,
        gate_passed=not failed,
        protocol="120 balanced randomized mirrored rounds per case; two eight-sample replays per role/round; GPU-resident event nodes; shared output address; cold L2 before every sample; five-SE upper ratio bound with zero regression margin, or identical CUDA program and aliasing reference preparation",
        seconds=time.monotonic() - started,
    )
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(dict(operation="pair", cases=len(rows), not_proved=len(failed))),
        flush=True,
    )
    raise SystemExit(0)


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
    profile_key = "-".join(
        str(label[k]) for k in ("operation", "dtype", "id_dtype", "layout", "masked")
    )
    if label["layout"] in ("dense", "sliced") and profile_key not in profiles:
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
    scope="native ID inputs in score and top-k kernels; no full-model serving throughput or fabric qualification",
)
(args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
