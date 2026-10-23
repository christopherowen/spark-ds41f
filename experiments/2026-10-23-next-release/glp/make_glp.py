#!/usr/bin/env python3
"""Write a synthetic GLP file for qualifying steering on (the mechanism, not a vector).

usage: make_glp.py OUT [--seed 0] [--alpha 0.75] [--layers 40] [--hidden 5120]

One seeded random unit direction per layer 1..layers-1, at residual_stream_post_layer,
in the GGUF v3 layout llama.cpp's writer uses (vLLM's reader, glp.read_glp, accepts
exactly this). A random direction is one of 5,120 dimensions: projecting it off moves
the model a little, enough to change some greedy outputs, so the run shows that
steering is applied without serving a behaviour-changing vector.
"""
import argparse
import struct

import numpy as np


def _string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _value(value) -> bytes:
    if isinstance(value, int):
        return struct.pack("<II", 4, value)
    if isinstance(value, float):
        return struct.pack("<If", 6, value)
    return struct.pack("<I", 8) + _string(value)


def write_gguf(path: str, metadata: dict, tensors: dict) -> None:
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += _string(key) + _value(value)
    blobs, offset = [], 0
    for name, values in tensors.items():
        values = np.asarray(values, dtype="<f4")
        header += _string(name) + struct.pack("<IQ", 1, values.size) + struct.pack("<IQ", 0, offset)
        blob = values.tobytes()
        blob += b"\0" * (-len(blob) % 32)
        blobs.append(blob)
        offset += len(blob)
    header += b"\0" * (-len(header) % 32)
    with open(path, "wb") as handle:
        handle.write(header + b"".join(blobs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=5120)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    layers = list(range(1, args.layers))
    tensors = {}
    for layer in layers:
        vector = rng.standard_normal(args.hidden)
        tensors[f"direction.{layer}"] = (vector / np.linalg.norm(vector)).astype(np.float32)
    metadata = {
        "general.architecture": "glp",
        "general.name": f"spark-ds41f synthetic GLP (seed {args.seed})",
        "general.base_model.0.name": "DeepSeek-V4.1-Flash",
        "general.base_model.0.version": "1017",
        "glp.mode": "project",
        "glp.spec_version": 1,
        "glp.rank": 1,
        "glp.hook_point": "residual_stream_post_layer",
        "glp.alpha_default": args.alpha,
        "glp.layer_ids_zero_based": ",".join(str(layer) for layer in layers),
    }
    write_gguf(args.out, metadata, tensors)


if __name__ == "__main__":
    main()
