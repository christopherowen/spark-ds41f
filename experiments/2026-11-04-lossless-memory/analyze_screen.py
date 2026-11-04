#!/usr/bin/env python3
"""Summarize native CPU receipts without changing their measured observations."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

from collect_weights import family


def analyze(paths):
    receipts = []
    identities = set()
    for path in paths:
        data = path.read_bytes()
        raw = gzip.decompress(data) if path.suffix == ".gz" else data
        receipt = json.loads(raw)
        identities.add((receipt["source_commit"], receipt["model_revision"],
                        receipt["input_manifest_sha256"]))
        receipts.append((path, raw, receipt))
    if len(identities) != 1 or len(receipts) < 2:
        raise ValueError("need at least two observations of the same source and input identity")
    source, revision, inputs = next(iter(identities))
    baseline = receipts[0][2]
    names = {(r["input"]["name"], tuple(r["input"]["tile_origin"]))
             for r in baseline["runs"]}
    summary = {"schema_version": 2, "source_commit": source, "model_revision": revision,
               "input_manifest_sha256": inputs,
               "scope": "CPU exact-byte storage screen on immutable checkpoint tiles",
               "sampled_tiles": len(names),
               "sampled_bytes": sum(r["raw_bytes"] for r in baseline["runs"]
                                    if r["requested_mode"] == "raw"),
               "independent_cpu_processes": len(receipts),
               "repeats_per_block_and_codec": baseline["repeats"],
               "serving_performance_verified": False, "gpu_performance_verified": False,
               "all_byte_exact": all(r["byte_exact"] for _, _, d in receipts for r in d["runs"]),
               "all_deterministic": all(r["deterministic"] for _, _, d in receipts for r in d["runs"]),
               "native_receipts": [{"path": str(p), "uncompressed_sha256": hashlib.sha256(raw).hexdigest()}
                                   for p, raw, _ in receipts],
               "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "classifier_sha256": hashlib.sha256(Path(__file__).with_name("collect_weights.py").read_bytes()).hexdigest(),
               "analysis_command": sys.argv, "classification_corrections": [], "groups": {}}
    # Version-1 exports grouped every Engram-named tensor together. Derive the
    # table/projection distinction from the recorded tensor name, preserving
    # the original input manifest and every native result byte.
    corrections = {r["input"]["name"] for r in baseline["runs"]
                   if family(r["input"]["name"]) != r["input"]["family"]}
    summary["classification_corrections"] = sorted(corrections)
    modes = sorted({r["requested_mode"] for r in baseline["runs"]})
    for scope in ("dense", "engram_table"):
        for mode in modes:
            observations = []
            for _, _, receipt in receipts:
                rows = [r for r in receipt["runs"] if r["requested_mode"] == mode and
                        (family(r["input"]["name"]) == "engram_table") == (scope == "engram_table")]
                if not rows:
                    continue
                raw = sum(r["raw_bytes"] for r in rows)
                stored = sum(r["stored_bytes"] for r in rows)
                observations.append({"raw_bytes": raw, "stored_bytes": stored,
                                     "saving_pct": (1 - stored / raw) * 100,
                                     "blocks": len(rows),
                                     "median_cpu_decode_verify_ns": statistics.median(
                                         r["median_decode_verify_ns"] for r in rows),
                                     "raw_representation_blocks": sum(r["stored_mode"] == 0 for r in rows)})
            if len({(o["raw_bytes"], o["stored_bytes"], o["blocks"]) for o in observations}) > 1:
                raise ValueError("repeated deterministic encodings changed size or coverage")
            summary["groups"][scope + "/" + mode] = observations
    summary["interpretation"] = {
        "lz4": "no payload reduction on sampled raw FP8 tiles; RAW selected",
        "bitplane-lz4": "retain for GPU decode-cost screening; no serving win established",
        "fp8-exponent-huffman": "retain for parallel/fused GPU consumer investigation; no serving win established",
        "cold_prefix": "storage correctness prototype only; target export/restore and eviction/reuse unqualified",
        "expert_scales": "isolated source and measurement arm prepared; target tests and two boots pending",
        "engram": "isolated source and measurement arm prepared; target tests, packing costs and two boots pending"}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(analyze(args.receipts), indent=2) + "\n")


if __name__ == "__main__":
    main()
