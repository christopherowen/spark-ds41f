#!/usr/bin/env python3
"""Export a bounded set of immutable checkpoint tiles; never initialize CUDA.

Run from a clean published checkout. Output contains checkpoint bytes and must
stay outside Git. Each sample is a real rectangular N128/K128 consumer tile,
not a whole-file compression corpus. No checkpoint or page-cache state is changed.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct


def family(name):
    if "engram" in name and ".embed." in name:
        return "engram_table"
    if "engram" in name:
        return "engram_projection"
    for token in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b", "shared_experts"):
        if token in name:
            return token
    return "other_dense"


def collect(snapshot: Path, revision: str, max_tensors: int, tiles_per_tensor: int):
    if snapshot.name != revision:
        raise ValueError("snapshot directory does not match the pinned revision")
    candidates = {}
    for path in sorted(snapshot.glob("*.safetensors")):
        with path.open("rb") as src:
            count = struct.unpack("<Q", src.read(8))[0]
            if count > 64 * 1024 * 1024:
                raise ValueError("safetensors header exceeds the bounded reader contract")
            metadata = json.loads(src.read(count))
        for name, tensor in metadata.items():
            if name == "__metadata__" or tensor.get("dtype") not in ("F8_E4M3", "F8_E4M3FN"):
                continue
            shape = tensor["shape"]
            if len(shape) != 2 or min(shape) < 128:
                continue
            start, end = tensor["data_offsets"]
            if end - start != shape[0] * shape[1]:
                raise ValueError("FP8 tensor byte length does not match its shape")
            candidates.setdefault(family(name), []).append((name, path, 8 + count + start, tensor))
    # Round-robin across families and depths; avoid letting the huge disk table
    # dominate statistics for the dense GPU projections.
    selected = []
    ordered = {}
    for key, values in sorted(candidates.items()):
        values.sort(key=lambda x: x[0])
        positions = sorted(set([0, len(values) // 2, len(values) - 1]))
        ordered[key] = [values[i] for i in positions]
    while len(selected) < max_tensors and any(ordered.values()):
        for key, values in ordered.items():
            if values and len(selected) < max_tensors:
                selected.append((key, values.pop(0)))
    samples = []
    for key, (name, path, base, tensor) in selected:
        n, k = tensor["shape"]
        positions = [(0, 0), ((n // 2 // 128) * 128, (k // 2 // 128) * 128),
                     ((n - 128) // 128 * 128, (k - 128) // 128 * 128)]
        for row, col in dict.fromkeys(positions[:tiles_per_tensor]):
            row, col = min(row, n - 128), min(col, k - 128)
            with path.open("rb") as src:
                parts = []
                for i in range(128):
                    src.seek(base + (row + i) * k + col)
                    data = src.read(128)
                    if len(data) != 128:
                        raise ValueError("checkpoint tile read was short")
                    parts.append(data)
            raw = b"".join(parts)
            samples.append({"name": name, "family": key, "dtype": tensor["dtype"],
                            "kind": "checkpoint_weight_tile", "tensor_shape": [n, k],
                            "tile_shape": [128, 128], "tile_origin": [row, col],
                            "shard": path.name, "blob": path.resolve().name,
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "data_base64": base64.b64encode(raw).decode("ascii")})
    if not samples:
        raise ValueError("no supported FP8 tiles were found")
    return {"schema_version": 2, "kind": "immutable_checkpoint_tiles",
            "model_revision": revision, "tile_bytes": 16384,
            "sample_bytes": len(samples) * 16384, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-tensors", type=int, default=24, choices=range(1, 65))
    parser.add_argument("--tiles-per-tensor", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.snapshot, args.revision, args.max_tensors, args.tiles_per_tensor)
    # Refuse overwrite so an unfavorable or failed sample set is not silently replaced.
    with args.out.open("x") as out:
        json.dump(result, out)
        out.write("\n")
    print(json.dumps({"samples": len(result["samples"]), "bytes": result["sample_bytes"],
                      "out": str(args.out)}))


if __name__ == "__main__":
    main()
