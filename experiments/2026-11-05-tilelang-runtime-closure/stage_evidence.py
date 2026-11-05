"""Stage closed sp8 evidence through hardlinks; preserve original node artifacts."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket

assert socket.gethostname() == "spark-0016", "large evidence processing is remote only"
HERE = Path(__file__).resolve().parent
OUT = HERE / "runs"
LAB = Path.home() / "spark-lab"
LIVE = LAB / "tilelang-runtime-closure-20261105"
names = ["tail-activity-v19", "embedding-extension-v20", "draft-performance-v21",
         "tail-activity-v21", "draft-performance-v22", "draft-performance-repeat-v23",
         "final-iket-boot1-v23", "final-iket-boot2-v23",
         "final-tests-boot1-v23", "final-tests-boot2-v23",
         "draft-precision-boot1-v24", "draft-precision-boot2-v24"]
sources = [(LIVE / name, name, True) for name in names]
sources += [(LAB / ("tilelang-runtime-full-20261105" + suffix), name, False)
            for suffix, name in (("-initial", "full-initial"),
                                 ("-draft-origins", "full-draft-origins"),
                                 ("-native-embedding-v20", "full-native-embedding-v20"))]
ledger = OUT / "staged-evidence.json"
records = json.loads(ledger.read_text()) if ledger.exists() else {}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


for source, name, needs_closure in sources:
    target = OUT / name
    if (not source.is_dir() or target.exists()
            or (needs_closure and not (source / "closure.json").exists())):
        continue
    working = target.with_name(target.name + ".staging")
    working.mkdir(exist_ok=True)
    retained, count = [], 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".nsys-rep":
            retained.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
            continue
        if path.suffix not in (".cu", ".py", ".json", ".jsonl", ".txt", ".log", ".xml", ".sqlite", ".gz"):
            continue
        destination = working / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert sha(destination) == sha(path), destination
        else:
            try:
                os.link(path, destination)
            except PermissionError:
                # Docker-owned output may be readable without permission to
                # hardlink it. Copy remotely with a fixed-size buffer instead.
                pending = destination.with_name(destination.name + ".copying")
                with path.open("rb") as source_file, pending.open("wb") as sink:
                    shutil.copyfileobj(source_file, sink, length=1 << 20)
                assert sha(pending) == sha(path), path
                pending.rename(destination)
        count += 1
    records[name] = dict(source=str(source), staged_files=count,
                          raw_reports_retained_on_node=retained,
                          method="Hardlinks where permitted, otherwise bounded remote streaming copies; original artifacts remain intact on sp8.")
    (working / "evidence-source.json").write_text(json.dumps(records[name], indent=2) + "\n")
    working.rename(target)
    print(name, count, "files", flush=True)
temporary = ledger.with_name(ledger.name + ".tmp")
temporary.write_text(json.dumps(records, indent=2) + "\n")
temporary.replace(ledger)
