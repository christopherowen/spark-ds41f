#!/usr/bin/env python3
"""Validate and losslessly archive original FP8 GPU receipts; never run CUDA."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

from collect_weights import family
from fp8_results import summarize, validate

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--cpu-inputs", type=Path, required=True)
    parser.add_argument("--cpu-summary", type=Path, default=HERE / "results.json")
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.archives.exists() or args.out.exists():
        parser.error("refusing to overwrite evidence")
    base = json.loads((HERE / "base.json").read_text())
    cpu = json.loads(args.cpu_inputs.read_bytes())
    cpu_reference = json.loads(args.cpu_summary.read_text())
    if digest(args.cpu_inputs.read_bytes()) != cpu_reference["input_manifest_sha256"]:
        raise ValueError("CPU sample identity differs from the recorded screen")
    expected = {(s["name"], tuple(s["tile_origin"]), s["sha256"])
                for s in cpu["samples"] if family(s["name"]) != "engram_table"}
    controller = json.loads((args.controller / "receipt.json").read_text())
    if controller.get("completed") is not True or controller.get("restored") is not True or controller.get("error"):
        raise ValueError("GPU screen did not complete and restore its service")
    summaries = []
    for lanes in (32, 64, 128):
        pair = []
        for label in ("first", "repeat"):
            name = f"lossless-fp8-{lanes}-{label}.json"
            native = json.loads((args.originals / ".cache/spark-lab/compile" / name).read_text())
            bundle = f"bundle-{lanes}" + ("-repeat" if label == "repeat" else "")
            worker = args.originals / "spark-lab/results" / f"lossless-fp8-kernel-{lanes}-{label}" / f"{bundle}-dgx1"
            verdict = json.loads((worker / "verdict.json").read_text())
            validate(native, verdict, lanes=lanes, image=base["image_id"], revision=base["model_revision"])
            identity = {(s["name"], tuple(s["tile_origin"]), s["sha256"]) for s in native["inputs"]}
            if identity != expected:
                raise ValueError("GPU checkpoint tiles differ from the CPU screen")
            copied = args.controller / f"lossless-fp8-kernel-{lanes}-{label}.json"
            if json.loads(copied.read_text()) != native:
                raise ValueError("controller and original native observations differ")
            pair.append(native)
        summary = summarize(pair)
        if summary != controller["summaries"][len(summaries)]:
            raise ValueError("recomputed uncertainty differs from the native summary")
        summary["median_kernel_ms_first_repeat"] = {
            name: [statistics.median(r["kernel_ms"][name]) for r in pair]
            for name in ("raw", "encoded")}
        summary["actual_input_saving_pct"] = 100 * (1 - summary["encoded_input_over_real_raw_bytes"])
        summaries.append(summary)
    args.archives.mkdir(parents=True)
    archives = []
    for group, root in (("original", args.originals), ("controller", args.controller)):
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            raw = path.read_bytes()
            if len(raw) > 2 * 1024**2 or b'"data_base64"' in raw:
                raise ValueError("unexpected large file or raw checkpoint bytes in export")
            target = args.archives / group / path.relative_to(root)
            target = target.with_name(target.name + ".gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            compressed = gzip.compress(raw, mtime=0)
            with target.open("xb") as output:
                output.write(compressed)
            archives.append({"path": str(target), "uncompressed_sha256": digest(raw),
                             "gzip_sha256": digest(compressed), "uncompressed_bytes": len(raw)})
    result = {"schema_version": 1, "scope": "exact-byte GB10 decoder screen; no serving model loaded",
              "source_commit": controller["commit"], "window": controller["window"],
              "base_image_id": base["image_id"], "model_revision": base["model_revision"],
              "cpu_input_manifest_sha256": cpu_reference["input_manifest_sha256"],
              "checkpoint_tiles": len(expected), "fresh_containers": 6,
              "all_checkpoint_and_special_encoding_bytes_exact": True,
              "serving_performance_verified": False, "model_equivalence_qualified": False,
              "restored": controller["restored"], "ended": controller["ended"],
              "summaries": summaries, "archives": archives,
              "analysis_sha256": digest(Path(__file__).read_bytes()), "command": sys.argv,
              "decision": "reject all three tested hot-weight Huffman decoder layouts"}
    with args.out.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
