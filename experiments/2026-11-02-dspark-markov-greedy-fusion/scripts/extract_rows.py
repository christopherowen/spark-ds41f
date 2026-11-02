"""Copy the first ROWS rows of a BF16 safetensors tensor to <out>.bin (raw
little-endian BF16) with <out>.json (name, shape, source); CPU only."""
import json
import struct
import sys
from pathlib import Path

snapshot, name, rows, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
shard = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"][name]
path = snapshot / shard
with open(path, "rb") as f:
    (header_len,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(header_len))
    meta = header[name]
    assert meta["dtype"] == "BF16", meta
    shape = meta["shape"]
    rows = min(rows, shape[0]) if rows > 0 else shape[0]
    row_bytes = 2 * (shape[1] if len(shape) > 1 else 1)
    start = 8 + header_len + meta["data_offsets"][0]
    f.seek(start)
    data = f.read(rows * row_bytes)
out.with_suffix(".bin").write_bytes(data)
out.with_suffix(".json").write_text(
    json.dumps({"name": name, "shard": shard, "shape": [rows, *shape[1:]], "full_shape": shape})
)
print(name, shard, shape, "->", rows)
