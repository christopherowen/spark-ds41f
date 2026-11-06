"""Bound a remote image-transfer stream without storing it on the workstation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--receipt", type=Path, required=True)
p.add_argument("--bytes-per-second", type=int, default=10 * 1024 * 1024)
a = p.parse_args()
if a.receipt.exists() or a.bytes_per_second <= 0:
    raise SystemExit("receipt exists or invalid rate")
started = time.monotonic()
count = 0
digest = hashlib.sha256()
error = None
try:
    while data := sys.stdin.buffer.read(256 * 1024):
        count += len(data)
        digest.update(data)
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        delay = count / a.bytes_per_second - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
except BaseException as exc:
    error = type(exc).__name__ + ": " + str(exc)
    raise
finally:
    a.receipt.write_text(json.dumps({
        "bytes": count, "sha256": digest.hexdigest(),
        "seconds": time.monotonic() - started,
        "rate_limit_bytes_per_second": a.bytes_per_second,
        "error": error, "workstation_storage": False,
    }, indent=2) + "\n")
