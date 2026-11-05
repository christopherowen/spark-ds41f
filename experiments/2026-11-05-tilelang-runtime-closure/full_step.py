"""Verify the published overlay, run the existing 43-layer harness, export CUDA."""

import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys

HERE = Path(__file__).resolve().parent
source = Path("/opt/spark-ds41f/candidate/vllm")
base = json.loads((HERE / "base.json").read_text())
candidate = json.loads((HERE / "candidate.json").read_text())
for relative, expected in base["image_files"].items():
    path = source / relative
    assert (hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None) == expected, relative
for relative, expected in candidate["files"].items():
    payload = (HERE / "overlay" / relative).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected
    path = source / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
out = Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "source-identity.json").write_text(json.dumps(candidate, indent=2) + "\n")
runpy.run_path("/b/harness/iket_step.py", run_name="__main__")
cache = Path(os.environ["TILELANG_CACHE_DIR"])
compiled = []
for path in sorted(cache.rglob("*.cu")):
    text = path.read_text()
    compiled.append(dict(path=str(path.relative_to(cache)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        kernels=re.findall(r"__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+(\w+)", text)))
(out / "compiled-cuda.json").write_text(json.dumps(compiled, indent=2) + "\n")
