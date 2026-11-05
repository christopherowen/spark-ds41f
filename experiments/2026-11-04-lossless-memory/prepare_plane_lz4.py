#!/usr/bin/env python3
"""Measure independent-plane FP8 storage size only; no CUDA or inference."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

from collect_weights import family
from fp8_plane_lz4_frame import DATA, ENTRY, HEADER, RAW, TILE_BYTES, pack
from memory_codec import LZ4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, choices=(256, 512, 1024, 2048), default=2048)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite evidence")
    source = args.inputs.read_bytes()
    inputs = json.loads(source)
    lz4, rows, started = LZ4(), [], time.monotonic()
    for sample in inputs["samples"]:
        if family(sample["name"]) == "engram_table":
            continue
        raw = base64.b64decode(sample["data_base64"], validate=True)
        if len(raw) != TILE_BYTES or hashlib.sha256(raw).hexdigest() != sample["sha256"]:
            raise ValueError("sample shape or original-byte identity differs")
        tile = pack(raw, lz4, args.chunk_bytes)
        coded = sum(not (ENTRY.unpack_from(tile.payload, HEADER.size + i * ENTRY.size)[1] & RAW)
                    for i in range(TILE_BYTES // args.chunk_bytes)) if tile.mode else 0
        rows.append({"name": sample["name"], "tile_origin": sample["tile_origin"],
                     "sha256": sample["sha256"], "mode": tile.mode, "compressed_streams": coded,
                     "stored_bytes": tile.stored_bytes,
                     "payload_sha256": hashlib.sha256(tile.payload).hexdigest()})
    if not rows:
        raise ValueError("no dense tiles")
    stored, raw = sum(r["stored_bytes"] for r in rows), len(rows) * TILE_BYTES
    here = Path(__file__).parent
    result = {"schema_version": 1, "scope": "exact CPU size feasibility; no GPU or serving measurement",
              "command": sys.argv, "input_manifest_sha256": hashlib.sha256(source).hexdigest(),
              "model_revision": inputs["model_revision"], "chunk_bytes": args.chunk_bytes,
              "tiles": len(rows), "rows": rows,
              "decoded_bytes": raw, "actual_stored_bytes": stored, "saving_pct": 100 * (1 - stored / raw),
              "library": {"version": lz4.version, "binary_sha256": lz4.binary_sha256},
              "source_sha256": {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in
                                ["fp8_plane_lz4_frame.py", "memory_codec.py", "collect_weights.py", "prepare_plane_lz4.py"]},
              "preparation_seconds": time.monotonic() - started,
              "all_original_bytes_exact": True, "serving_performance_verified": False,
              "decision": "retain only for GPU decode-cost screening" if stored < raw else "reject at size gate"}
    with args.out.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
