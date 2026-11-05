"""Compress completed text evidence losslessly, retaining both SHA-256 hashes."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "runs")
args = parser.parse_args()
root = args.root
ledger = root / "compressed-evidence.json"
data = json.loads(ledger.read_text()) if ledger.exists() else dict(files=[])
records = {row["original"]: row for row in data["files"]}
count = 0


def digest(path, compressed=False):
    value, size = hashlib.sha256(), 0
    with (gzip.open if compressed else open)(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            value.update(chunk)
            size += len(chunk)
    return size, value.hexdigest()


for path in sorted(root.rglob("*")):
    if not path.is_file() or path == ledger or path.suffix == ".gz":
        continue
    selected = path.suffix in (".cu", ".py") or path.name.endswith(".trace.json")
    selected |= path.suffix == ".sqlite"
    selected |= path.suffix == ".json" and path.stat().st_size >= 100000
    if not selected:
        continue
    size, original_sha = digest(path)
    target = path.with_name(path.name + ".gz")
    if not target.exists():
        with target.open("wb") as handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as writer:
                with path.open("rb") as source:
                    while chunk := source.read(1 << 20):
                        writer.write(chunk)
    assert digest(target, compressed=True) == (size, original_sha), target
    relative = str(path.relative_to(root))
    records[relative] = dict(original=relative, stored=str(target.relative_to(root)),
                            bytes=size, sha256=original_sha,
                            compressed_bytes=target.stat().st_size,
                            compressed_sha256=digest(target)[1])
    path.unlink()
    count += 1
temporary = ledger.with_name(ledger.name + ".tmp")
temporary.write_text(json.dumps(dict(format="deterministic gzip; original bytes verified before replacement",
                                     files=list(records.values())), indent=2) + "\n")
temporary.replace(ledger)
print("Compressed", count, "additional artifacts;", len(records), "total recorded.")
