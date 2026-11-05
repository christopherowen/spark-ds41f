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
import subprocess

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("operation", choices=("tests", "pair", "bench", "iket", "identity", "copy_screen", "wire", "transfer", "activity", "embedding_bench"))
parser.add_argument("--family", choices=("sampler", "reduction", "gather", "prompt", "wire", "draft", "embedding", "all"), default="sampler")
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
                              if (args.family == "all" and family != "prompt") or family == args.family for t in entries))
    code = pytest.main(["--noconftest", "-c", "/dev/null", "-q", "--maxfail=2",
                        "--junitxml=" + str(args.out / "tests.xml"),
                        *(str(source / test) for test in tests)], plugins=[report])
    result = dict(identity, pytest_exit=int(code), outcomes=dict(report.outcomes),
                  seconds=time.monotonic() - started)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(code)

if args.operation == "embedding_bench":
    import runpy
    script = HERE / "embedding_bench.py"
    sys.argv = [str(script), "--out", str(args.out)]
    runpy.run_path(str(script), run_name="__main__")
    raise SystemExit(0)

if args.operation in ("copy_screen", "wire", "transfer"):
    import runpy
    script = HERE / ("copy_screen.py" if args.operation == "copy_screen" else "wire_probe.py" if args.operation == "wire" else "transfer_bench.py")
    sys.argv = [str(script), "--out", str(args.out)]
    runpy.run_path(str(script), run_name="__main__")
    raise SystemExit(0)

if args.operation == "identity":
    native = HERE.parent / "2026-11-04-native-score-inputs"
    original = (native / "measure.py").read_text()
    reference = args.out / "reference"
    reference.mkdir()
    for name in ("tilelang_logprob.py", "tilelang_topk_ids.py"):
        prior = HERE.parent / "2026-11-04-logits-scores/overlay/vllm/v1/worker/gpu/sample" / name
        if not prior.exists():
            # The selection source is unchanged by that prerequisite.
            prior = HERE / "overlay/vllm/v1/worker/gpu/sample" / name
        (reference / name).write_bytes(prior.read_bytes())
    body = original[original.index("\nimport torch  # noqa: E402"):]
    body = body.replace("    candidate=candidate,", "    candidate=candidate, combined_candidate=combined_candidate,")
    args.operation = "pair"
    context = dict(__name__="__main__", __file__=str(native / "measure.py"),
                   HERE=native, args=args, reference_dir=reference,
                   base=json.loads((native / "base.json").read_text()),
                   candidate=json.loads((native / "candidate.json").read_text()),
                   combined_candidate=candidate, source=source,
                   Counter=Counter, hashlib=hashlib, json=json, Path=Path,
                   subprocess=subprocess, sys=sys, time=time,
                   statistics=__import__("statistics"))
    exec(compile(body, str(native / "measure.py"), "exec"), context)
    raise AssertionError("identity proof did not terminate")

if args.operation == "iket":
    harness = HERE.parent / "2026-10-31-iket-full-43l-and-runtime-opt/harness/iket_profile.py"
    command = [sys.executable, str(harness), str(args.out / "profile"),
               "--timeout", "300", "--single-pass-max-ts", "256", "--pool-gb", "0.25",
               "--skip-raw-data-dump", "--", sys.executable, str(HERE / "iket_probe.py"),
               "--out", str(args.out / "probe")]
    (args.out / "iket-command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, check=True)
    decoded_paths = list((args.out / "profile/trace").rglob("iket.decoded_results.json"))
    assert decoded_paths
    launches = []
    for path in decoded_paths:
        decoded = json.loads(path.read_text())
        strings = decoded["stringTable"]
        captured = list(decoded["launches"])
        graphs = decoded.get("graphLaunches") or {}
        for group in graphs.values() if isinstance(graphs, dict) else graphs:
            captured.extend(group if isinstance(group, list) else group.get("launches", []))
        for launch in captured:
            if not isinstance(launch, dict) or "ranges" not in launch:
                continue
            assert launch["ranges"] and launch["warpLifetimes"], launch["kernelName"]
            assert all(r["endTs"] >= r["startTs"] for r in launch["ranges"])
            launches.append(dict(kernel=launch["kernelName"], grid=launch["gridId"],
                range_names=[strings[r["rangeNameIdx"]] for r in launch["ranges"]],
                warps=len(launch["warpLifetimes"])))
    names = {row["kernel"] for row in launches}
    required = {"copy_logits", "collectives_rank_order_strided", "collectives_rank_order_sum",
                "gather_columns_into", "collectives_rank_rows_bytes", "collectives_rank_rows_small", "topk_log_softmax_ranks",
                "token_ranks_scores", "topk_ids"}
    normalized = {name.removesuffix("_kernel") for name in names}
    assert required <= normalized, (required - normalized, names)
    result = dict(identity, launches=launches, required=sorted(required),
                  numerics=json.loads((args.out / "probe/numerics.json").read_text()),
                  seconds=time.monotonic() - started)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0)

assert args.operation in ("pair", "bench", "activity")
from vllm.v1.worker.gpu.sample import tilelang_sampling as sampling
from vllm.models.deepseek_v4_1.tilelang import collectives, gather

spec = importlib.util.spec_from_file_location("previous_sampling", HERE / "control/tilelang_sampling.py")
previous_sampling = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous_sampling
spec.loader.exec_module(previous_sampling)
spec = importlib.util.spec_from_file_location("previous_collectives", HERE / "control/collectives-v4.py")
previous_collectives = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous_collectives
spec.loader.exec_module(previous_collectives)
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
    if args.operation == "activity":
        from activity import measure_activity
        measured, _ = measure_activity(launches, flush, 20261105, expected,
                                      args.out / (name + "-activity.trace.json"), args.rounds)
    else:
        measured, _ = measure(launches, flush, 20261105, expected, args.rounds,
                              samples_per_replay=64 if name.startswith("gather-tail") else 8)
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
                observed, pitched = [], []
                original_bind = sampling._kernel
                original_pitched = sampling._pitched_logits_copy

                def record(factory, *key):
                    observed.append((factory, key))
                    return original_bind(factory, *key)

                def record_pitched(src, dst):
                    pitched.append(True)
                    return original_pitched(src, dst)

                sampling._kernel = record
                sampling._pitched_logits_copy = record_pitched
                try:
                    actual = sampling.copy_logits(src)
                finally:
                    sampling._kernel = original_bind
                    sampling._pitched_logits_copy = original_pitched
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
                elif pitched:
                    assert len(pitched) == 1

                    def launch(src=src, out=out):
                        original_pitched(src, out)
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
                                        selected_key=list(observed[0][1]) if observed else "pitched DMA" if pitched else "native DMA",
                                        seed=20261105), identical=not observed and not pitched)
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
                collectives._rank_rows_tail(src, out, 3, count, 5 * src.element_size())
                return out

            def before(src=src, out=out, count=count):
                previous_collectives._rank_rows_tail(src, out, 3, count, 5 * src.element_size())
                return out

            run_case(f"gather-tail-{dtype}-{count}",
                     dict(reference=reference, candidate=launch, previous=before, null=reference),
                     expected, dict(world=3, shape=[count, 15],
                                    reference="same TensorIterator copy as movedim reshape, shared output"))

if args.family in ("prompt", "wire", "draft", "embedding", "all"):
    from vllm.v1.worker.gpu.sample.prompt_logprob import compute_prompt_logprobs_with_chunking as prompt_scores

    spec = importlib.util.spec_from_file_location("previous_prompt", HERE / "control/prompt_logprob.py")
    old_prompt = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old_prompt
    spec.loader.exec_module(old_prompt)
    cases = [(count, 8193, k, mode) for count in (1025, 2049) for k in (0, 5)
             for mode in ("raw_logits", "raw_logprobs")]
    cases += [(1025, 129280, 5, mode) for mode in ("raw_logits", "raw_logprobs")]
    for count, vocab, k, mode in cases:
        logits = torch.randn((count, vocab), device="cuda")
        ids = torch.randint(vocab, (2 * count,), dtype=torch.int64, device="cuda")[::2]
        expected = tuple(value.clone() for value in old_prompt.compute_prompt_logprobs_with_chunking(
            ids, logits, lambda hidden: hidden, k, mode))
        outputs = tuple(torch.empty_like(value) for value in expected)
        original_empty, original_cat = torch.empty, torch.cat

        def reference():
            index = 0

            def cat(tensors, dim=0, **kwargs):
                nonlocal index
                output = outputs[index]
                index += 1
                return original_cat(tensors, dim=dim, out=output, **kwargs)

            torch.cat = cat
            try:
                result = old_prompt.compute_prompt_logprobs_with_chunking(
                    ids, logits, lambda hidden: hidden, k, mode)
                assert index == 3
                return result
            finally:
                torch.cat = original_cat

        def launch():
            pending = list(outputs if k else outputs[1:])

            def empty(*shape, **kwargs):
                output = pending.pop(0)
                declared = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], tuple) else tuple(shape)
                assert tuple(output.shape) == declared and output.dtype == kwargs["dtype"]
                return output

            torch.empty = empty
            try:
                result = prompt_scores(ids, logits, lambda hidden: hidden, k, mode)
                assert not pending
                return result
            finally:
                torch.empty = original_empty

        # Resolve all kernels before the allocation-only hooks and graph capture.
        prompt_scores(ids, logits, lambda hidden: hidden, k, mode)
        run_case(f"prompt-{count}-{vocab}-{k}-{mode}",
                 dict(reference=reference, candidate=launch, previous=reference, null=reference),
                 expected, dict(shape=[count, vocab], num_logprobs=k, mode=mode,
                                shared_output_addresses=[value.data_ptr() for value in outputs],
                                hooks="allocation and cat destinations only; every GPU kernel and intermediate unchanged",
                                selected_ids_alias_input=k == 0))

assert rows
print("completed", len(rows), "cases", flush=True)
