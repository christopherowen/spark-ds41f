#!/usr/bin/env python3
"""Compare complete crash-surviving layout records, without accessing a GPU.

Record format is the layout-ring overlay at deployment c2bb2ebb. CRC fields
describe host metadata; they are not snapshots of GPU operands or NaN origins.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib


RECORD = struct.Struct("<IiiiIIIiII")
FIELDS = ("step", "tokens", "padded", "reqs", "scheduled_crc", "drafts_crc",
          "eligible_crc", "draft_budget", "reads", "stale")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rank_files", nargs=3, type=Path)
    args = parser.parse_args()
    records, sources = [], []
    for rank, path in enumerate(args.rank_files):
        data = path.read_bytes()
        if not data or len(data) % RECORD.size:
            raise ValueError(f"Empty or truncated record file: {path}")
        rows = [dict(zip(FIELDS, values)) for values in RECORD.iter_unpack(data)]
        if any(row["step"] != step for step, row in enumerate(rows)):
            raise ValueError(f"Nonconsecutive steps: {path}")
        records.append(rows)
        sources.append({"rank": rank, "path": str(path), "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(), "records": len(rows)})
    if len({len(rows) for rows in records}) != 1:
        raise ValueError("Ranks have different record counts; refusing silent truncation")
    differences = []
    for step, rows in enumerate(zip(*records)):
        if len({tuple(row[key] for key in FIELDS[1:8]) for row in rows}) != 1:
            differences.append({"step": step, "ranks": list(rows)})
    print(json.dumps({
        "format": "<IiiiIIIiII", "record_bytes": RECORD.size,
        "sources": sources, "divergence_count": len(differences),
        "divergences": differences,
        "int32_crc_reference": {str(n): zlib.crc32(struct.pack("<i", n))
                                for n in (5, 6, 16, 17)},
        "scope": "Host layout records only. Complete equal-sized files, consecutive step IDs. No claim about original GPU values or first NaN instruction.",
    }, indent=2))


if __name__ == "__main__":
    main()
