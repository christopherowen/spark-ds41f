"""Read ext4's current/best extent counts, without reading weight payloads.

Sent over SSH by doctor. e4defrag -c is check-only: never remove -c here.
The fragmentation score is deliberately ignored: score zero is not an ideal
layout when current extents exceed best extents.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def extent_counts(output, expected):
    counts = [tuple(map(int, pair)) for pair in re.findall(
        r"^\s*Total/best extents\s+(\d+)/(\d+)\s*$", output, re.M,
    )]
    if len(counts) != expected or any(best < 1 or current < best for current, best in counts):
        raise ValueError("incomplete or inconsistent current/best extent report")
    return counts


def measure(root):
    started = time.monotonic()
    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if not names:
        raise ValueError("checkpoint index contains no shards")
    paths = [(root / name).resolve(strict=True) for name in names]
    command = ([] if os.geteuid() == 0 else ["sudo", "-n"])
    command += ["e4defrag", "-c", *map(str, paths)]
    result = subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=15,
        env=dict(os.environ, LC_ALL="C"),
    )
    counts = extent_counts(result.stdout, len(names))
    return {
        "shards": [
            {"name": name, "path": str(path), "current": current, "best": best}
            for name, path, (current, best) in zip(names, paths, counts)
        ],
        "elapsed_seconds": time.monotonic() - started,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(measure(sys.argv[1])))
    except subprocess.CalledProcessError as error:
        print(json.dumps({"error": error.stderr.strip() or "e4defrag check failed"}))
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"error": str(error)}))
