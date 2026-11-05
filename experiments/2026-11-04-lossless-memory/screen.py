#!/usr/bin/env python3
"""Measure exact independent-block storage and CPU reference-codec costs.

Results deliberately cannot assert a serving win. GPU consumers, candidate
boots and release suites are separate gates. Raw checkpoint bytes stay local.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import time

from memory_codec import HEADER, LZ4, MODES, decode, encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--liblz4")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two CPU observations per input and mode are required")
    root = Path(__file__).resolve().parents[2]
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root)
    if dirty:
        parser.error("commit the screening code before recording measurements")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source = args.samples.read_bytes()
    dataset = json.loads(source)
    if dataset.get("kind") not in ("immutable_checkpoint_tiles", "written_cache_records"):
        parser.error("inputs must identify real checkpoint tiles or fenced written-cache records")
    lz4 = LZ4(args.liblz4) if any("lz4" in name for name in args.modes) else None
    results = []
    for sample in dataset["samples"]:
        raw = base64.b64decode(sample["data_base64"], validate=True)
        if hashlib.sha256(raw).hexdigest() != sample["sha256"]:
            raise ValueError("sample hash mismatch")
        if not raw:
            raise ValueError("empty input is not a useful performance sample")
        frames = {mode: encode(raw, mode, lz4) for mode in args.modes}
        measurements = {mode: {"encode_verify_ns": [], "decode_verify_ns": []} for mode in args.modes}
        order = [(mode, repeat) for repeat in range(args.repeats) for mode in args.modes]
        random.Random(sample["sha256"]).shuffle(order)
        for mode, _ in order:
            start = time.perf_counter_ns()
            frame = encode(raw, mode, lz4)
            middle = time.perf_counter_ns()
            restored = decode(frame, lz4)
            end = time.perf_counter_ns()
            if frame != frames[mode] or restored != raw:
                raise ValueError("codec output was nondeterministic or not byte exact")
            measurements[mode]["encode_verify_ns"].append(middle - start)
            measurements[mode]["decode_verify_ns"].append(end - middle)
        identity = {k: v for k, v in sample.items() if k != "data_base64"}
        for mode in args.modes:
            frame = frames[mode]
            saved = len(raw) - len(frame)
            results.append({"input": identity, "requested_mode": mode,
                            "stored_mode": HEADER.unpack_from(frame)[2],
                            "raw_bytes": len(raw), "stored_bytes": len(frame),
                            "payload_bytes": len(frame) - HEADER.size,
                            "saved_bytes": saved, "byte_exact": True,
                            "deterministic": True, **measurements[mode],
                            "median_decode_verify_ns": statistics.median(
                                measurements[mode]["decode_verify_ns"]),
                            "illustrative_saved_read_ns_at_266_6_GBs": max(saved, 0) / 266.6})
    groups = {}
    for row in results:
        key = row["input"].get("family", "cache") + "/" + row["requested_mode"]
        aggregate = groups.setdefault(key, {"raw_bytes": 0, "stored_bytes": 0, "blocks": 0})
        aggregate["raw_bytes"] += row["raw_bytes"]
        aggregate["stored_bytes"] += row["stored_bytes"]
        aggregate["blocks"] += 1
    result = {"schema_version": 1, "scope": "CPU independent-block screen",
              "serving_performance_verified": False, "gpu_performance_verified": False,
              "source_commit": source_commit, "command": sys.argv,
              "input_manifest_sha256": hashlib.sha256(source).hexdigest(),
              "model_revision": dataset.get("model_revision"),
              "host": {"system": platform.platform(), "machine": platform.machine(),
                       "python": platform.python_version()},
              "codec": {"liblz4": lz4.path, "version": lz4.version,
                        "binary_sha256": lz4.binary_sha256} if lz4 else None,
              "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "peak_rss_unit": "bytes" if sys.platform == "darwin" else "KiB",
              "repeats": args.repeats, "groups": groups, "runs": results}
    with args.out.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps({"results": str(args.out), "groups": groups}))


if __name__ == "__main__":
    main()
