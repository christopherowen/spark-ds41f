#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stream binary chunks into Chrome Trace JSON readable by Perfetto."""

import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import zlib

HEADER = struct.Struct("<8sIIQQ4Q")
RECORD = struct.Struct("<QQQQQIIIIII")


def records(root):
    root = Path(root)
    expected = 0
    index = 0
    while (path := root / f"events-{index:08d}.bin").exists():
        with path.open("rb") as handle:
            magic, version, size, chunk, first, *_ = HEADER.unpack(
                handle.read(HEADER.size)
            )
            if (
                magic != b"SPKIKET1"
                or version not in (1, 2)
                or size != 64
                or chunk != index
                or first != expected
            ):
                raise ValueError(f"invalid or discontinuous chunk {path}")
            count, checksum = 0, 0 if version == 2 else 14695981039346656037
            while block := handle.read(1 << 20):
                if len(block) % RECORD.size:
                    raise ValueError(f"partial record in {path}")
                if version == 2:
                    checksum = zlib.crc32(block, checksum)
                else:
                    for byte in block:
                        checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64) - 1)
                for row in RECORD.iter_unpack(block):
                    if row[0] != expected:
                        raise ValueError(
                            f"sequence gap at {path}: {row[0]} != {expected}"
                        )
                    expected += 1
                    count += 1
                    yield row
        receipt = json.loads(path.with_suffix(".bin.json").read_text())
        key = "crc32" if version == 2 else "fnv1a64"
        value = f"{checksum:08x}" if version == 2 else f"{checksum:016x}"
        if receipt != {"chunk": chunk, "records": count, key: value}:
            raise ValueError(f"checksum/count mismatch in {path}")
        index += 1
    if (root / "summary.json").exists():
        summary = json.loads((root / "summary.json").read_text())
        if summary["complete"] and (
            expected != summary["written_records"] or index != summary["chunks"]
        ):
            raise ValueError("missing chunk or summary record count mismatch")


def export(
    root,
    output,
    grid=None,
    *,
    roce_traces=None,
    align_clocks=True,
    base_pid=1_000_000,
):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("capture is incomplete; retain native chunks for diagnosis")
    names, kernels = {}, {}
    for path in (root / "sources").glob("*.json"):
        manifest = json.loads(path.read_text())
        identity = manifest["kernel_id"]
        kernels[identity] = manifest["kernel"]
        for site in manifest["sites"]:
            if "range_id" in site:
                names[identity, site["range_id"]] = site["range"]
    counts = Counter()
    with Path(output).open("x") as handle:
        handle.write('{"traceEvents":[')
        comma = ""
        for (
            seq,
            timestamp,
            launch,
            kernel,
            cta,
            warp,
            sm,
            event,
            payload,
            kind,
            _,
        ) in records(root):
            counts["records"] += 1
            counts[f"kind_{kind}"] += 1
            if grid is not None and launch != grid:
                continue
            name = kernels[kernel] if kind == 0 else names.get((kernel, event), "phase")
            item = {
                "ph": "B" if kind in (0, 1) else "E",
                "pid": launch,
                "tid": cta * 32 + warp,
                "ts": timestamp / 1000,
                "args": {"sequence": seq, "sm": sm, "payload": payload},
            }
            if kind in (0, 1):
                item["name"] = name
            handle.write(comma + json.dumps(item, separators=(",", ":")))
            comma = ","
        if roce_traces:
            from sparknet.oneshot import trace as roce_trace

            loaded = [
                t if hasattr(t, "records") else roce_trace.read(str(t))
                for t in roce_traces
            ]
            for item in roce_trace.perfetto_events(
                loaded, align_clocks=align_clocks, base_pid=base_pid
            ):
                counts["roce_events"] += 1
                handle.write(comma + json.dumps(item, separators=(",", ":")))
                comma = ","
        handle.write("]}\n")
    if counts["records"] != summary["written_records"]:
        raise ValueError("summary record count mismatch")
    if counts["kind_0"] != counts["kind_3"] or counts["kind_1"] != counts["kind_2"]:
        raise ValueError("unbalanced lifetime/phase totals")
    return dict(counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("output")
    parser.add_argument("--grid", type=int)
    parser.add_argument(
        "--roce-trace",
        dest="roce_traces",
        nargs="+",
        help="merge sparknet SPKTRACE files onto the shared %globaltimer timeline",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            export(args.root, args.output, args.grid, roce_traces=args.roce_traces)
        )
    )
