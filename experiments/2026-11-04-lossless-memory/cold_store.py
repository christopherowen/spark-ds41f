#!/usr/bin/env python3
"""Experimental bounded store for already-fenced, exact prefix checkpoint bytes.

This storage screen does not register prefix hits with vLLM. A serving adapter
must reserve destinations, restore every required group and auxiliary state,
and wait for every rank before publishing a hit. Never export unwritten pool
bytes or assume a sliding-window suffix is a complete checkpoint.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile

from memory_codec import LZ4, MAX_BLOCK, decode, encode

FILE_HEADER = struct.Struct("<4sIQ32s")
MAX_MANIFEST = 4 * 1024 * 1024
MAX_RECORDS = 65536
IDENTITY_FIELDS = frozenset({"model_revision", "tokenizer_revision", "vllm_tree",
                             "tp_size", "rank", "cache_layout", "steering_digest"})


def json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


class ColdStore:
    def __init__(self, root: Path, identity: dict, *, quota_bytes: int,
                 max_checkpoint_bytes: int, lz4: LZ4 | None = None):
        if not IDENTITY_FIELDS <= identity.keys():
            raise ValueError("checkpoint namespace lacks model/layout/rank identity")
        tp, rank = identity["tp_size"], identity["rank"]
        if type(tp) is not int or type(rank) is not int or not 0 <= rank < tp:
            raise ValueError("invalid tensor-parallel rank identity")
        if not 0 < max_checkpoint_bytes <= quota_bytes:
            raise ValueError("checkpoint cap must fit within the explicit disk quota")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.identity = json.loads(json_bytes(identity))
        self.namespace = hashlib.sha256(json_bytes(identity)).hexdigest()
        self.quota = quota_bytes
        self.cap = max_checkpoint_bytes
        self.lz4 = lz4

    def path(self, prefix_digest: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", prefix_digest):
            raise ValueError("prefix key must be a full SHA-256 digest")
        return self.root / f"{self.namespace}-{prefix_digest}.lmc"

    def put(self, prefix_digest: str, records: list[tuple[dict, bytes]], *,
            boundary: dict, mode: str = "lz4") -> dict:
        path = self.path(prefix_digest)
        if not records or len(records) > MAX_RECORDS:
            raise ValueError("checkpoint has no records or too many records")
        # The input producer must supply a full boundary export. This check is
        # necessary, but is not a substitute for a serving-side capture fence.
        required = {"num_tokens", "draft_prefix_len", "groups", "auxiliary_state",
                    "capture_fence", "complete"}
        if not required <= boundary.keys() or boundary["complete"] is not True:
            raise ValueError("a complete boundary export is required")
        tokens, draft = boundary["num_tokens"], boundary["draft_prefix_len"]
        if (type(tokens) is not int or type(draft) is not int or
                not 0 <= draft <= tokens or tokens <= 0 or
                boundary["capture_fence"] != "synchronized_written_regions"):
            raise ValueError("invalid boundary or capture-fence declaration")
        total = sum(len(raw) for _, raw in records)
        if total > self.cap or any(len(raw) > MAX_BLOCK for _, raw in records):
            raise ValueError("checkpoint or record exceeds the configured byte cap")
        names = [desc.get("name") for desc, _ in records]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("record names must be unique nonempty strings")
        frames, entries, offset = [], [], 0
        for descriptor, raw in records:
            frame = encode(raw, mode, self.lz4)
            frames.append(frame)
            entries.append({"descriptor": descriptor, "offset": offset,
                            "stored_bytes": len(frame), "raw_bytes": len(raw)})
            offset += len(frame)
        manifest = {"version": 1, "identity": self.identity, "prefix_digest": prefix_digest,
                    "boundary": boundary, "records": entries, "raw_bytes": total,
                    "stored_payload_bytes": offset}
        metadata = json_bytes(manifest)
        if len(metadata) > MAX_MANIFEST or offset + len(metadata) + FILE_HEADER.size > self.cap:
            raise ValueError("encoded checkpoint and metadata exceed the byte cap")
        size = FILE_HEADER.size + len(metadata) + offset
        # Serialize quota checks and publication; readers see the old complete
        # checkpoint or the new complete checkpoint through atomic replacement.
        with (self.root / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            used = sum(p.stat().st_size for p in self.root.glob("*.lmc"))
            old = path.stat().st_size if path.exists() else 0
            if used - old + size > self.quota:
                raise ValueError("cold-store disk quota would be exceeded")
            fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=self.root)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(FILE_HEADER.pack(b"LMC1", len(metadata), offset,
                                               hashlib.sha256(metadata).digest()))
                    out.write(metadata)
                    for frame in frames:
                        out.write(frame)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temporary, path)
                directory = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return {"path": str(path), "raw_bytes": total, "stored_bytes": size,
                "records": len(records), "published_to_vllm": False}

    def get(self, prefix_digest: str) -> tuple[dict, list[tuple[dict, bytes]]]:
        path = self.path(prefix_digest)
        with path.open("rb") as src:
            actual_size = os.fstat(src.fileno()).st_size
            if actual_size > self.cap:
                raise ValueError("stored checkpoint exceeds the configured byte cap")
            header = src.read(FILE_HEADER.size)
            if len(header) != FILE_HEADER.size:
                raise ValueError("truncated checkpoint header")
            magic, metadata_bytes, payload_bytes, metadata_digest = FILE_HEADER.unpack(header)
            if (magic != b"LMC1" or metadata_bytes > MAX_MANIFEST or
                    FILE_HEADER.size + metadata_bytes + payload_bytes != actual_size):
                raise ValueError("invalid checkpoint header or length")
            metadata = src.read(metadata_bytes)
            if hashlib.sha256(metadata).digest() != metadata_digest:
                raise ValueError("checkpoint metadata failed integrity verification")
            manifest = json.loads(metadata)
            if (manifest.get("version") != 1 or manifest.get("identity") != self.identity or
                    manifest.get("prefix_digest") != prefix_digest):
                raise ValueError("checkpoint namespace or prefix identity mismatch")
            entries = manifest.get("records", [])
            if not entries or len(entries) > MAX_RECORDS:
                raise ValueError("invalid checkpoint record count")
            offset, total, records = 0, 0, []
            names = set()
            for entry in entries:
                count, size = entry["stored_bytes"], entry["raw_bytes"]
                name = entry["descriptor"]["name"]
                if (entry["offset"] != offset or type(count) is not int or
                        type(size) is not int or not 0 <= size <= MAX_BLOCK or
                        not 0 < count <= payload_bytes - offset or name in names):
                    raise ValueError("invalid record extent or duplicate record")
                total += size
                if total > self.cap:
                    raise ValueError("decoded checkpoint exceeds the byte cap")
                raw = decode(src.read(count), self.lz4)
                if len(raw) != size:
                    raise ValueError("record size mismatch")
                records.append((entry["descriptor"], raw))
                names.add(name)
                offset += count
            if (offset != payload_bytes or total != manifest.get("raw_bytes") or
                    offset != manifest.get("stored_payload_bytes")):
                raise ValueError("checkpoint totals do not match")
        return manifest["boundary"], records
