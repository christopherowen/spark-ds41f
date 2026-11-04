#!/usr/bin/env python3
"""Check every packed byte against the immutable checkpoint before measurement.

Run inside the candidate container, in its reserved window, while requests are
idle. Buffers are bounded; read pages are dropped after each chunk. No CUDA call,
model inference or checkpoint/cache write is made. This is a correctness gate,
not an NVMe performance benchmark.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time
import traceback


def available_bytes():
    if not Path("/proc/meminfo").exists():
        return None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("host memory telemetry is missing")


def tensors(snapshot):
    result = {}
    for path in sorted(snapshot.glob("*.safetensors")):
        with path.open("rb") as source:
            length = struct.unpack("<Q", source.read(8))[0]
            if length > 64 * 1024**2:
                raise ValueError("oversized safetensors header")
            header = json.loads(source.read(length))
        for name, entry in header.items():
            if ".engram.embed." not in name:
                continue
            if name in result:
                raise ValueError("duplicate Engram tensor")
            start, end = entry["data_offsets"]
            if not 0 <= start <= end or 8 + length + end > path.stat().st_size:
                raise ValueError("Engram tensor exceeds checkpoint file")
            result[name] = (path, 8 + length + start, end - start, entry["shape"])
    return result


def verify_rows(weight, scale, packed, starts, lengths, *, chunk_rows=65536):
    try:
        import numpy as np
    except ImportError:
        np = None

    rows = sum(lengths)
    if packed.stat().st_size != rows * 264:
        raise ValueError("packed cache has the wrong length")
    physical = hashlib.sha256()
    selected_sources = hashlib.sha256()
    position = 0
    minimum = available_bytes()
    with weight[0].open("rb", buffering=0) as w, scale[0].open("rb", buffering=0) as s, \
            packed.open("rb", buffering=0) as p:
        for start, count in zip(starts, lengths):
            for offset in range(0, count, chunk_rows):
                available = available_bytes()
                if available is not None:
                    minimum = min(minimum, available)
                    if available < 4 * 1024**3:
                        raise ValueError("verification reached its 4 GiB host memory floor")
                n = min(chunk_rows, count - offset)
                w_offset = weight[1] + (start + offset) * 256
                s_offset = scale[1] + (start + offset) * 8
                wb = os.pread(w.fileno(), n * 256, w_offset)
                sb = os.pread(s.fileno(), n * 8, s_offset)
                actual = os.pread(p.fileno(), n * 264, position)
                if len(wb) != n * 256 or len(sb) != n * 8 or len(actual) != n * 264:
                    raise ValueError("short source/cache read")
                if np is None:
                    expected = b"".join(wb[r * 256:(r + 1) * 256] + sb[r * 8:(r + 1) * 8]
                                        for r in range(n))
                else:
                    array = np.empty((n, 264), dtype=np.uint8)
                    array[:, :256] = np.frombuffer(wb, dtype=np.uint8).reshape(n, 256)
                    array[:, 256:] = np.frombuffer(sb, dtype=np.uint8).reshape(n, 8)
                    expected = array.tobytes()
                if actual != expected:
                    raise ValueError(f"packed bytes differ at physical offset {position}")
                physical.update(actual)
                selected_sources.update(wb)
                selected_sources.update(sb)
                for fd, at, size in ((w.fileno(), w_offset, len(wb)),
                                     (s.fileno(), s_offset, len(sb)), (p.fileno(), position, len(actual))):
                    if hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(fd, at, size, os.POSIX_FADV_DONTNEED)
                position += len(actual)
    return {"rows": rows, "verified_bytes": position, "packed_sha256": physical.hexdigest(),
            "selected_source_stream_sha256": selected_sources.hexdigest(), "all_byte_exact": True,
            "minimum_host_available_bytes": minimum}


def cache_path(directory, layer, tp, rank, weight, scale, table_rows, starts, lengths):
    w_path, w_offset = weight[:2]
    s_path, s_offset = scale[:2]
    ws, ss = w_path.stat(), s_path.stat()
    mask = sum(1 << c for c in range(rank, 24, tp))
    identity = (f"engram-packed-264-v1:{w_path.resolve()}:{w_offset}:{ws.st_size}:{ws.st_mtime_ns}:"
                f"{s_path.resolve()}:{s_offset}:{ss.st_size}:{ss.st_mtime_ns}:"
                f"{table_rows}:{mask:#x}:{starts}:{lengths}")
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return directory / f"engram-layer{layer}-tp{tp}-rank{rank}-{digest}.packed"


def open_by_serving_process(paths):
    owners = {str(p): [] for p in paths}
    identities = {str(p): (p.stat().st_dev, p.stat().st_ino) for p in paths}
    for process in Path("/proc").glob("[0-9]*"):
        if process.name == str(os.getpid()):
            continue
        for fd in (process / "fd").glob("*"):
            try:
                target = os.readlink(fd)
                info = fd.stat()
            except OSError:
                continue
            # The verifier and serving process have separate mount namespaces.
            # Match the underlying file, including stale/deleted FD detection,
            # rather than assuming procfs spells both mount paths identically.
            for path, identity in identities.items():
                if (info.st_dev, info.st_ino) == identity:
                    owners[path].append({"pid": int(process.name), "fd": fd.name,
                                         "observed_path": target})
    if not owners or not all(owners.values()):
        raise ValueError("the running reader does not hold every verified packed file")
    return owners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=(3, 4), required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.snapshot.name != args.revision or not 0 <= args.rank < args.tp or args.out.exists():
        parser.error("wrong snapshot/rank or prior receipt; no operation")
    started = time.monotonic()
    result = {"schema_version": 1, "scope": "complete packed-byte verification; not I/O performance",
              "command": sys.argv, "model_revision": args.revision, "rank": args.rank, "tp": args.tp,
              "source_sha256": globals().get("SOURCE_SHA256") or hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "all_byte_exact": False, "files": []}
    try:
        result["host_available_before_bytes"] = available_bytes()
        if result["host_available_before_bytes"] is None or result["host_available_before_bytes"] < 4 * 1024**3:
            raise ValueError("verification requires Linux telemetry and 4 GiB host headroom")
        from b12x.sequence.engram.geometry import build_geometry

        config = json.loads((args.snapshot / "config.json").read_text())["text_config"]
        geometry = build_geometry(layer_ids=tuple(config["engram_layer_ids"]),
                                  base_table_size=config["engram_vocab_size"],
                                  compressed_vocab_size=config["engram_compressed_vocab_size"])
        if list(geometry.num_embeddings) != config["engram_num_embeddings"]:
            raise ValueError("hash geometry does not match checkpoint rows")
        entries, paths = tensors(args.snapshot), []
        for index, layer in enumerate(config["engram_layer_ids"]):
            weight = entries[f"layers.{layer}.engram.embed.weight"]
            scale = entries[f"layers.{layer}.engram.embed.weight_scale_inv"]
            count = int(geometry.num_embeddings[index])
            if weight[2:] != (count * 256, [count, 256]) or scale[2:] != (count * 8, [count, 8]):
                raise ValueError("weight/scale byte layout is not the frozen 264-byte layout")
            offsets = list(map(int, geometry.offsets[index])) + [count]
            starts = [offsets[c] for c in range(args.rank, 24, args.tp)]
            lengths = [offsets[c + 1] - offsets[c] for c in range(args.rank, 24, args.tp)]
            path = cache_path(args.directory, layer, args.tp, args.rank, weight, scale, count, starts, lengths)
            paths.append(path)
            record = {"layer": layer, "path": str(path), "owned_starts": starts, "owned_rows": lengths}
            result["files"].append(record)
            record.update(verify_rows(weight, scale, path, starts, lengths))
        result["open_reader_fds"] = open_by_serving_process(paths)
        result["all_byte_exact"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    result["elapsed_verification_seconds"] = time.monotonic() - started
    result["host_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with args.out.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(json.dumps(result))
    return 0 if result["all_byte_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
