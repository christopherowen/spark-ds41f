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
import resource
import statistics
import sys
import time
import traceback

from collect_weights import collect, family
from fp8_plane_lz4_frame import TILE_BYTES, Tile, pack, unpack
from memory_codec import LZ4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, choices=(128, 256, 512), default=128)
    parser.add_argument("--chunk-bytes", type=int, choices=(512, 2048), required=True)
    parser.add_argument("--mib", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()
    if args.rounds < 5:
        parser.error("at least five timed observations are required")
    if args.out.exists():
        parser.error("refusing to overwrite an earlier run")
    result = {"schema_version": 1, "scope": "standalone GPU byte-decoder screen",
              "serving_performance_verified": False, "gemm_performance_verified": False,
              "command": sys.argv, "threads": args.threads, "format": "independent-plane-lz4-v1", "chunk_bytes": args.chunk_bytes, "requested_mib": args.mib,
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
        result["device_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        source = Path(__file__).with_suffix(".cu")
        result["cuda_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        result["frame_source_sha256"] = hashlib.sha256(Path(__file__).with_name("fp8_plane_lz4_frame.py").read_bytes()).hexdigest()
        result["sources_sha256"] = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("fp8_plane_lz4_probe.py", "fp8_plane_lz4_probe.cu", "fp8_plane_lz4_frame.py",
                         "collect_weights.py", "memory_codec.py")
        }
        dataset = collect(args.snapshot, args.revision, 24, 2)
        samples = [s for s in dataset["samples"] if family(s["name"]) != "engram_table"]
        raw_samples = [base64.b64decode(s["data_base64"], validate=True) for s in samples]
        for sample, raw in zip(samples, raw_samples):
            if hashlib.sha256(raw).hexdigest() != sample["sha256"]:
                raise ValueError("checkpoint sample hash mismatch")
        lz4 = LZ4()
        result["lz4"] = {"version": lz4.version, "binary_sha256": lz4.binary_sha256}
        frames = [pack(raw, lz4, args.chunk_bytes) for raw in raw_samples]
        if any(unpack(frame, lz4) != raw for frame, raw in zip(frames, raw_samples)):
            raise ValueError("CPU independent-plane decoder failed")
        result["inputs"] = [{k: v for k, v in s.items() if k != "data_base64"} for s in samples]
        result["sample_raw_bytes"] = sum(map(len, raw_samples))
        result["sample_encoded_bytes_with_metadata"] = sum(f.stored_bytes for f in frames)
        result["sample_compressed_tiles"] = sum(f.mode == 1 for f in frames)
        count = args.mib * 1024 * 1024 // TILE_BYTES
        raw_frames = [Tile(0, raw) for raw in raw_samples]

        def upload(items, num_tiles=count):
            selected = [items[i % len(items)] for i in range(num_tiles)]
            offsets, payload = [], bytearray()
            for frame in selected:
                offsets.append(len(payload))
                payload.extend(frame.payload)
            tensors = [
                torch.frombuffer(payload, dtype=torch.uint8).cuda(),
                torch.tensor(offsets, dtype=torch.int64, device="cuda"),
                torch.tensor([f.mode for f in selected], dtype=torch.uint8, device="cuda"),
                torch.empty(num_tiles * TILE_BYTES, dtype=torch.uint8, device="cuda"),
                torch.empty(num_tiles, dtype=torch.int32, device="cuda"),
            ]
            return tensors

        module = load(name="lossless_fp8_plane_lz4_" + result["cuda_source_sha256"][:12],
                      sources=[str(source)], extra_cflags=["-O3"],
                      extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)
        scattered = bytearray(TILE_BYTES)
        for index in (0, 31, 32, 511, 512, 1023, 1024, 2047, 2048, 4095, TILE_BYTES - 1):
            scattered[index] = 255
        fixtures = [bytes(range(256)) * 64, bytes(TILE_BYTES), b"\xff" * TILE_BYTES, bytes(scattered)]
        fixture_frames = [pack(raw, lz4, args.chunk_bytes) for raw in fixtures]
        if any(frame.mode != 1 or unpack(frame, lz4) != raw
               for frame, raw in zip(fixture_frames, fixtures)):
            raise ValueError("GPU validation fixtures must use exact compressed streams")
        fixtures.append(bytes(range(256)) * 64)
        fixture_frames.append(Tile(0, fixtures[-1]))
        oracle = upload(fixture_frames, len(fixtures))
        module.run(*oracle, args.threads, False)
        if oracle[-2].cpu().numpy().tobytes() != b"".join(fixtures):
            raise ValueError("GPU decoder changed a special encoding or stream boundary")
        result["gpu_all_encodings_and_stream_boundaries_exact"] = True
        result["gpu_validation_fixtures"] = [hashlib.sha256(raw).hexdigest() for raw in fixtures]
        del oracle
        cases = {"raw": upload(raw_frames), "encoded": upload(frames)}
        for values in cases.values():
            module.run(*values, args.threads, False)
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
        # The raw control has only its actual offset/mode descriptors. Count
        # every compressed byte against the real raw weight size, not scaffolding.
        graphs = {}
        for name, values in cases.items():
            for _ in range(3):
                module.run(*values, args.threads, True)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    module.run(*values, args.threads, True)
            graphs[name] = graph
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(cases["raw"][-1], cases["encoded"][-1]):
            raise ValueError("shared-memory byte consumers disagree")
        import functools
        import operator
        import struct
        sample_checksums = [functools.reduce(operator.xor, struct.unpack("<4096I", raw))
                            for raw in raw_samples]
        expected = torch.tensor([sample_checksums[i % len(sample_checksums)]
                                 for i in range(count)], dtype=torch.int64)
        if not torch.equal(cases["encoded"][-1].cpu().to(torch.int64) & 0xffffffff, expected):
            raise ValueError("shared-memory consumer differs from the source checksum")
        result["gpu_consumer_checksums_exact"] = True
        warmup = time.monotonic()
        while time.monotonic() - warmup < 2:
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
        result["alternating_warmup_seconds"] = time.monotonic() - warmup
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
    result["host_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["compiler_children_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    with args.out.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps({k: result.get(k) for k in ("status", "gpu_byte_exact", "median_kernel_ms", "decoded_GBs", "error")}))
    # The lab archives stdout in output.txt; retain the complete native receipt
    # there as well as in the named cache file, including every failed attempt.
    print("FP8_PLANE_LZ4_PROBE_RESULT " + json.dumps(result, sort_keys=True))
    print("1 failed" if result["status"] == "failed" else "1 passed (exact GPU byte-consumer check)")
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
