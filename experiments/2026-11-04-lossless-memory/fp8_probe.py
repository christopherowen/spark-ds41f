#!/usr/bin/env python3
"""GB10 exact-byte decoder microbenchmark; never loads a serving model.

Run only in a reserved lab window. This measures a shared-memory byte consumer,
not a GEMM or a serving win. Every GPU decoder is checked by materializing and
comparing all original bytes before timing. Inputs exceed L2 by construction.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import traceback

from collect_weights import collect, family
from fp8_probe_frame import TILE_BYTES, Tile, pack, unpack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lanes", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument("--mib", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    if args.rounds < 5:
        parser.error("at least five timed observations are required")
    if args.out.exists():
        parser.error("refusing to overwrite an earlier run")
    result = {"schema_version": 1, "scope": "standalone GPU byte-decoder screen",
              "serving_performance_verified": False, "gemm_performance_verified": False,
              "command": sys.argv, "lanes": args.lanes, "requested_mib": args.mib,
              "model_revision": args.revision, "gpu_byte_exact": False}
    try:
        import base64
        import torch
        from torch.utils.cpp_extension import load

        if torch.cuda.get_device_capability() != (12, 1):
            raise RuntimeError("this screen is qualified only for GB10/SM121")
        result["device"] = torch.cuda.get_device_name()
        result["torch"] = torch.__version__
        result["cuda"] = torch.version.cuda
        source = Path(__file__).with_suffix(".cu")
        result["cuda_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        result["frame_source_sha256"] = hashlib.sha256(Path(__file__).with_name("fp8_probe_frame.py").read_bytes()).hexdigest()
        dataset = collect(args.snapshot, args.revision, 24, 2)
        samples = [s for s in dataset["samples"] if family(s["name"]) != "engram_table"]
        raw_samples = [base64.b64decode(s["data_base64"], validate=True) for s in samples]
        for sample, raw in zip(samples, raw_samples):
            if hashlib.sha256(raw).hexdigest() != sample["sha256"]:
                raise ValueError("checkpoint sample hash mismatch")
        frames = [pack(raw, args.lanes) for raw in raw_samples]
        if any(unpack(frame) != raw for frame, raw in zip(frames, raw_samples)):
            raise ValueError("CPU parallel decoder failed")
        result["inputs"] = [{k: v for k, v in s.items() if k != "data_base64"} for s in samples]
        result["sample_raw_bytes"] = sum(map(len, raw_samples))
        result["sample_encoded_bytes_with_metadata"] = sum(f.stored_bytes for f in frames)
        result["sample_compressed_tiles"] = sum(f.mode == 1 for f in frames)
        count = args.mib * 1024 * 1024 // TILE_BYTES
        raw_frames = [Tile(0, args.lanes, raw) for raw in raw_samples]

        def upload(items):
            selected = [items[i % len(items)] for i in range(count)]
            offsets, stream_offsets, payload = [], [], bytearray()
            for frame in selected:
                offsets.append(len(payload))
                stream_offsets.extend(frame.offsets if frame.mode else [0] * args.lanes)
                payload.extend(frame.payload)
            tensors = [
                torch.frombuffer(payload, dtype=torch.uint8).cuda(),
                torch.tensor(offsets, dtype=torch.int64, device="cuda"),
                torch.frombuffer(bytearray(b"".join(f.low if f.mode else bytes(TILE_BYTES // 2) for f in selected)), dtype=torch.uint8).cuda(),
                torch.frombuffer(bytearray(b"".join(f.tree if f.mode else bytes(32) for f in selected)), dtype=torch.uint8).cuda(),
                torch.frombuffer(bytearray(b"".join(f.quick if f.mode else bytes(512) for f in selected)), dtype=torch.uint16).cuda(),
                torch.tensor(stream_offsets, dtype=torch.int32, device="cuda"),
                torch.tensor([f.mode for f in selected], dtype=torch.uint8, device="cuda"),
                torch.empty(count * TILE_BYTES, dtype=torch.uint8, device="cuda"),
                torch.empty(count, dtype=torch.int32, device="cuda"),
            ]
            return tensors

        module = load(name="lossless_fp8_" + result["cuda_source_sha256"][:12],
                      sources=[str(source)], extra_cflags=["-O3"],
                      extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
        cases = {"raw": upload(raw_frames), "encoded": upload(frames)}
        for values in cases.values():
            module.run(*values, args.lanes, False)
        torch.cuda.synchronize()
        if not torch.equal(cases["raw"][-2], cases["encoded"][-2]):
            raise ValueError("GPU reconstruction changed checkpoint bytes")
        # Check against the source too, so two incorrect decoder paths cannot
        # qualify each other. This untimed transfer covers the complete buffer.
        reference = b"".join(raw_samples[i % len(raw_samples)] for i in range(count))
        if cases["encoded"][-2].cpu().numpy().tobytes() != reference:
            raise ValueError("GPU decoder differs from the immutable source")
        result["gpu_byte_exact"] = True
        result["decoded_bytes_per_replay"] = count * TILE_BYTES
        result["allocated_input_bytes"] = {k: sum(t.numel() * t.element_size() for t in v[:-2])
                                           for k, v in cases.items()}
        result["raw_probe_scaffolding_bytes"] = result["allocated_input_bytes"]["raw"] - count * TILE_BYTES
        # The fixed-shape raw control has unused metadata placeholders. They
        # are reported, never counted as a real raw-weight footprint.
        graphs = {}
        for name, values in cases.items():
            for _ in range(3):
                module.run(*values, args.lanes, True)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    module.run(*values, args.lanes, True)
            graphs[name] = graph
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(cases["raw"][-1], cases["encoded"][-1]):
            raise ValueError("shared-memory byte consumers disagree")
        observations = {name: [] for name in cases}
        order = [name for _ in range(args.rounds) for name in cases]
        random.Random(41).shuffle(order)
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            observations[name].append(start.elapsed_time(end) / 10)
        result["kernel_ms"] = observations
        result["median_kernel_ms"] = {k: statistics.median(v) for k, v in observations.items()}
        result["decoded_GBs"] = {k: count * TILE_BYTES / (statistics.median(v) * 1e6)
                                 for k, v in observations.items()}
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["status"] = "measured microbenchmark; serving remains unqualified"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    with args.out.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps({k: result.get(k) for k in ("status", "gpu_byte_exact", "median_kernel_ms", "decoded_GBs", "error")}))
    # The lab archives stdout in output.txt; retain the complete native receipt
    # there as well as in the named cache file, including every failed attempt.
    print("FP8_PROBE_RESULT " + json.dumps(result, sort_keys=True))
    print("1 failed" if result["status"] == "failed" else "1 passed (exact GPU byte-consumer check)")
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
