"""Compress completed text evidence losslessly, retaining both SHA-256 hashes."""

import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent / "runs"
ledger = root / "compressed-evidence.json"
data = json.loads(ledger.read_text()) if ledger.exists() else dict(files=[])
records = {row["original"]: row for row in data["files"]}
count = 0
for path in sorted(root.rglob("*")):
    if not path.is_file() or path == ledger or path.suffix == ".gz":
        continue
    selected = path.suffix in (".cu", ".py") or path.name.endswith(".trace.json")
    selected |= path.suffix == ".json" and path.stat().st_size >= 100000
    if not selected:
        continue
    raw = path.read_bytes()
    target = path.with_name(path.name + ".gz")
    if not target.exists():
        with target.open("wb") as handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as writer:
                writer.write(raw)
    assert gzip.decompress(target.read_bytes()) == raw, target
    relative = str(path.relative_to(root))
    records[relative] = dict(original=relative, stored=str(target.relative_to(root)),
                            bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                            compressed_bytes=target.stat().st_size,
                            compressed_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    path.unlink()
    count += 1
ledger.write_text(json.dumps(dict(format="deterministic gzip; original bytes verified before replacement",
                                  files=list(records.values())), indent=2) + "\n")
print("Compressed", count, "additional artifacts;", len(records), "total recorded.")
