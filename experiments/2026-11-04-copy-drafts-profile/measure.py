#!/usr/bin/env python3
"""Pinned CopyDrafts tests, independent references, graph timing and IKET."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
STREAM = HERE.parent / "2026-11-04-iket-streaming"
sys.path.insert(0, str(STREAM))
parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=["tests", "bench", "profile"])
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--native", action="store_true")
parser.add_argument("--samples", type=int, default=21)
args = parser.parse_args()
base = json.loads((HERE / "base.json").read_text())
source = Path("/opt/spark-ds41f/candidate/vllm")
for relative, expected in base["files"].items():
    actual = hashlib.sha256((source / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"candidate-38 source mismatch: {relative}")

import numpy as np
import torch

assert torch.cuda.is_available()
collector = None
if args.operation == "profile":
    if args.native:
        from stream.phase import load

        iket = load(args.out)
        iket.enable(
            args.out / "native-sources",
            ctas=0,
            iters=1,
            warps=0,
            granularity="phase",
            payloads=True,
        )
    else:
        from stream.collector import enable

        collector = enable(args.out / "stream", patched=True, ctas=0, warps=0, iters=1)

started = time.monotonic()
identity = {
    "operation": args.operation,
    "source": base,
    "torch": torch.__version__,
    "device": torch.cuda.get_device_name(),
    "native": args.native,
}
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
    status = pytest.main(
        [
            "--noconftest",
            "-c",
            "/dev/null",
            "-q",
            "--junitxml=" + str(args.out / "tests.xml"),
            str(source / "tests/v1/spec_decode/test_copy_drafts.py"),
        ],
        plugins=[report],
    )
    result = dict(
        identity,
        pytest_exit=int(status),
        outcomes=dict(report.outcomes),
        seconds=time.monotonic() - started,
    )
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(status)

from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
from vllm.v1.worker.gpu.spec_decode import tilelang_copy as copy

VOCAB, MIN_MATCH, INITIAL, BLOCK, LOOKAHEAD = 129280, 16, 8, 5, 16
LEVELS = (16, 32, 64, 128, 256)


def reference(tokens):
    """Filter candidate ends, then compare whole suffixes independently."""
    ends = np.flatnonzero(tokens[:-1] == tokens[-1]) + 1
    return [
        max(
            (
                int(e)
                for e in ends
                if e >= level and np.array_equal(tokens[e - level : e], tokens[-level:])
            ),
            default=-1,
        )
        for level in LEVELS
    ]


def workload(requests, context, placement, pattern):
    backing = None
    shape = (16, context + 256)
    if placement == "uva":
        backing = StagedWriteTensor(
            shape,
            dtype=torch.int32,
            device=torch.device("cuda"),
            uva_instead_of_gpu=True,
        )
        history = backing.gpu
        history.fill_(-9)
    else:
        history = torch.full(shape, -9, dtype=torch.int32, device="cuda")
    length = torch.zeros(16, dtype=torch.int32, device="cuda")
    expected_sources, tokens_by_slot = [], []
    for slot in range(requests):
        rng = np.random.default_rng(context + slot * 17)
        if pattern in ("random", "copy_tail"):
            tokens = rng.integers(0, VOCAB, context, dtype=np.int32)
            if pattern == "copy_tail":
                # Put the isolated hit in a sampled middle scan iteration.
                at = context // 2 + MIN_MATCH - 256
                tokens[-256:] = tokens[at : at + 256]
            expected = reference(tokens)
        else:
            period = 64 if pattern == "repeat64" else 1
            tokens = (np.arange(context) % period + slot * 257 + 100).astype(np.int32)
            expected = [context - period] * len(LEVELS)
        history[slot, :context].copy_(torch.from_numpy(tokens).to(history.device))
        length[slot] = context
        expected_sources.append(expected)
        tokens_by_slot.append(tokens)
    idx = torch.arange(requests, dtype=torch.int64, device="cuda").flip(0)
    sources = torch.full(
        (requests, copy.CHUNKS, len(LEVELS)), -7, dtype=torch.int32, device="cuda"
    )
    ints = torch.zeros((16, 2), dtype=torch.int32, device="cuda")
    floats = torch.zeros((16, 2), dtype=torch.float32, device="cuda")
    dspark = torch.arange(requests * BLOCK, dtype=torch.int64, device="cuda").view(
        requests, BLOCK
    )
    dconf = torch.full((requests, BLOCK), 0.6, device="cuda")
    sampled = torch.ones(requests, dtype=torch.int32, device="cuda")
    verified = torch.zeros_like(sampled)
    drafts = torch.full((requests, LOOKAHEAD), -7, dtype=torch.int64, device="cuda")
    conf = torch.zeros((requests, LOOKAHEAD), device="cuda")
    planned = torch.zeros_like(conf)
    alloc = torch.zeros_like(sampled)
    select_alloc = torch.zeros_like(sampled)

    def match():
        copy.copy_match(idx, history, length, sources, MIN_MATCH)

    def allocate():
        copy.copy_alloc(
            idx,
            length,
            sources,
            sampled,
            verified,
            verified,
            ints,
            alloc,
            MIN_MATCH,
            LOOKAHEAD,
            BLOCK,
            INITIAL,
        )

    def select():
        copy.copy_select(
            idx,
            history,
            length,
            sources,
            dspark,
            dconf,
            sampled,
            verified,
            verified,
            ints,
            floats,
            drafts,
            conf,
            planned,
            select_alloc,
            MIN_MATCH,
            INITIAL,
        )

    def pipeline():
        match()
        allocate()
        select()

    pipeline()
    torch.cuda.synchronize()

    def validate():
        np.testing.assert_array_equal(
            sources.amax(dim=1).cpu().numpy(), np.array(expected_sources[::-1])
        )
        counts = []
        for row, slot in enumerate(range(requests - 1, -1, -1)):
            found = [e for e in expected_sources[slot] if e >= 0]
            src = found[-1] if found else -1
            offered = min(INITIAL, context - src) if src >= 0 else 0
            after = min(2 * INITIAL, LOOKAHEAD, context - src) if src >= 0 else 0
            expected = np.full(LOOKAHEAD, -1, dtype=np.int64)
            if src >= 0:
                expected[:offered] = tokens_by_slot[slot][src : src + offered]
                expected_conf = np.where(np.arange(LOOKAHEAD) < offered, 0.95, 0)
                expected_planned = np.where(np.arange(LOOKAHEAD) < after, 0.95, 0)
            else:
                expected[:BLOCK] = np.arange(row * BLOCK, (row + 1) * BLOCK)
                expected_conf = np.where(np.arange(LOOKAHEAD) < BLOCK, 0.6, 0)
                expected_planned = expected_conf
            np.testing.assert_array_equal(drafts[row].cpu().numpy(), expected)
            np.testing.assert_allclose(
                conf[row].cpu().numpy(), expected_conf, atol=1e-7
            )
            np.testing.assert_allclose(
                planned[row].cpu().numpy(), expected_planned, atol=1e-7
            )
            assert int(alloc[row]) == int(select_alloc[row]) == max(after, BLOCK)
            assert int(ints[slot, 0]) == 0 and int(ints[slot, 1]) == offered
            counts.append(offered)
        return counts

    offered = validate()
    # Hold backing/state/outputs alive through all captured replays.
    keep = (
        backing,
        history,
        length,
        sources,
        ints,
        floats,
        drafts,
        conf,
        planned,
        alloc,
        select_alloc,
        dspark,
        dconf,
        sampled,
        verified,
        idx,
    )
    return (
        dict(match=match, alloc=allocate, select=select, pipeline=pipeline),
        validate,
        offered,
        keep,
    )


flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")


def timed(launch):
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    samples = []
    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for _ in range(args.samples):
        flush.zero_()
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000)
    return dict(
        samples_us=samples,
        median_us=statistics.median(samples),
        minimum_us=min(samples),
        p95_us=float(np.quantile(samples, 0.95)),
    )


rows = []
requests_set = (1, 8, 16) if args.operation == "bench" else (1, 16)
contexts = (
    (4096, 65536, 262144, 1048576)
    if args.operation == "bench"
    else ((4096,) if args.native else (4096, 1048576))
)
placements = ("uva", "device") if args.operation == "bench" else ("uva",)
try:
    for requests in requests_set:
        for context in contexts:
            for placement in placements:
                for pattern in ("random", "copy_tail", "repeat64", "repeat1"):
                    launches, validate, offered, keep = workload(
                        requests, context, placement, pattern
                    )
                    row = dict(
                        requests=requests,
                        context=context,
                        history=placement,
                        pattern=pattern,
                        offered=offered,
                    )
                    if args.operation == "bench":
                        row["timings"] = {
                            name: timed(launch) for name, launch in launches.items()
                        }
                    else:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            launches["pipeline"]()
                        for _ in range(3):
                            graph.replay()
                        torch.cuda.synchronize()
                    validate()
                    rows.append(row)
                    (args.out / "cases.json").write_text(
                        json.dumps(rows, indent=2) + "\n"
                    )
                    print(json.dumps(row), flush=True)
finally:
    if collector:
        collector.close()

result = dict(
    identity,
    cases=rows,
    seconds=time.monotonic() - started,
    scope="three real copy kernels; fixed cap=8 with no prior drafts judged; no distributed model",
)
if collector:
    from analyze_capture import analyze

    result["integrity"] = analyze(args.out / "stream")
    (args.out / "integrity.json").write_text(
        json.dumps(result["integrity"], indent=2) + "\n"
    )
(args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
