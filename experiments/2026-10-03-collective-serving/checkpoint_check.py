"""Bounded checkpoint metadata check; does not hash tensor payloads or touch CUDA."""

import hashlib
import json
from pathlib import Path
import struct
import sys


def inspect(snapshot: Path) -> dict:
    index_path = snapshot / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    files = sorted(set(index["weight_map"].values()))
    records = []
    for name in files:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"unexpected checkpoint filename: {name!r}")
        path = snapshot / name
        size = path.stat().st_size
        with path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            if not 2 <= length <= 16 * 1024 * 1024:
                raise ValueError(f"invalid header length: {name}: {length}")
            header = handle.read(length)
        if len(header) != length:
            raise ValueError(f"truncated header: {name}")
        tensors = {key: value for key, value in json.loads(header).items()
                   if key != "__metadata__"}
        end = max(value["data_offsets"][1] for value in tensors.values())
        if 8 + length + end != size:
            raise ValueError(f"checkpoint extent mismatch: {name}: {size}")
        expected = {key for key, shard in index["weight_map"].items() if shard == name}
        if not expected <= tensors.keys():
            raise ValueError(f"indexed tensors missing: {name}")
        records.append({"name": name, "bytes": size,
                        "header_sha256": hashlib.sha256(header).hexdigest()})
    config_bytes = (snapshot / "config.json").read_bytes()
    return {
        "snapshot": str(snapshot),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "shards": records,
        "metadata_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True).encode()).hexdigest(),
        "payload_hashes_verified": False,
    }


if __name__ == "__main__":
    print(json.dumps(inspect(Path(sys.argv[1])), indent=2))
