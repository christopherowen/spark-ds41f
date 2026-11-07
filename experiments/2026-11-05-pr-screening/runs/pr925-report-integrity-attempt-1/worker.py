"""Bounded node-local hash review of three saved reports; never decompress or benchmark."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import resource
import shutil
import sys
import traceback

home = Path.home()
out = Path(sys.argv[1])
receipt = {"started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "PR": 925, "head": "86d96454212ad12a343ad45f5e828f9b063561d6",
           "scope": "Three saved node-local report hashes only; no decoding, model request, GPU or serving operation",
           "files": [], "passed": False}
claim = home / "spark-lab/reservations/copy-default-pr925-report-integrity.json"
owned = None
locks = []

def clear():
    for name in ("spark-hold.json", "spark-request.json", "spark3-request.json", "spark-node-hold.json", "spark-cpu-hold.json"):
        if (home / name).exists():
            raise RuntimeError("defer to existing hold/request: " + name)

try:
    assert os.uname().nodename == "dgx1"
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    os.nice(15)
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
    cgroup = Path("/sys/fs/cgroup") / next(line.split("::", 1)[1].lstrip("/") for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    limits = {key: (cgroup / key).read_text().strip() for key in ("cpu.max", "memory.max", "memory.swap.max")}
    assert limits == {"cpu.max": "100000 100000", "memory.max": str(256 * 1024**2), "memory.swap.max": "0"}, limits
    receipt["effective_limits"] = limits
    assert shutil.disk_usage(out).free > 1024**3
    available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    assert available > 8 * 1024**2
    clear()
    for path in (home / "spark-cpu-work.lock", home / "spark-lab/offline-cpu.lock"):
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a")
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(stream)
    for path in (home / "spark-lab/reservations").glob("*.json"):
        record = json.loads(path.read_text())
        if record.get("kind") != "cpu-only":
            raise RuntimeError("preserve reservation: " + str(path))
    claim.parent.mkdir(parents=True, exist_ok=True)
    owned = {"holder": "copy-default-pr925-report-integrity", "kind": "cpu-only", "cpus": 1,
             "memory_gib": .25, "pid": os.getpid(), "since": receipt["started"],
             "rule": "Hash three preexisting saved model reports; abort on a hold/request; no decode, timing, GPU or serving change"}
    with claim.open("x") as stream:
        json.dump(owned, stream)
    inputs = json.loads((out / "inputs.json").read_text())
    for entry in inputs:
        clear()
        path = Path(entry["saved_path"])
        assert path.is_relative_to(home / "tl-logs/lossless-memory/recovery/scales-round4-snapshot")
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024**2):
                clear()
                digest.update(block)
        after = path.stat()
        actual = {"path": str(path), "bytes": after.st_size, "sha256": digest.hexdigest(),
                  "stable": (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)}
        actual["passed"] = actual["stable"] and actual["bytes"] == entry["bytes"] and actual["sha256"] == entry["sha256"]
        receipt["files"].append(actual)
        if not actual["passed"]:
            raise RuntimeError("saved report identity mismatch: " + str(path))
    clear()
    receipt["passed"] = len(receipt["files"]) == 3
except BaseException as error:
    receipt["error"] = type(error).__name__ + ": " + str(error)
    (out / "error.log").write_text(traceback.format_exc())
finally:
    if owned is not None and claim.exists() and json.loads(claim.read_text()) == owned:
        claim.unlink()
        receipt["matching_claim_removed"] = True
    receipt["ended"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt["model_comparison_repeated"] = False
    receipt["report_text_transferred"] = False
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    raise SystemExit(0 if receipt["passed"] else 1)
