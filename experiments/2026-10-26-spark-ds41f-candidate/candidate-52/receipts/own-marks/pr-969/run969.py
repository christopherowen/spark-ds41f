#!/usr/bin/env python3
"""Driver for PR #969's own mark (ours). In one container (one boot) it calls the PR bench's
run_cache_and_packed_benchmark REPS times, as the PR's README calls it
(b.run_cache_and_packed_benchmark(bench_dir='/mnt/nvme/engram_bench_test')), and records which Engram
reader extension the arm loaded. Rep 1 is the PR's single call; reps 2..REPS repeat it in the same boot.
usage: run969.py LABEL REPS   (writes /out/bench-cache-LABEL.json; BOOT_LABEL is set to LABEL)
"""

import hashlib
import json
import os
import sys
import time

LABEL = sys.argv[1]
REPS = int(sys.argv[2])
os.environ["BOOT_LABEL"] = LABEL
sys.path.insert(0, "/bench969")

import bench_and_verify as b  # noqa: E402
from vllm.models.deepseek_v4_1.engram_rows import _native  # noqa: E402

mod = _native.load()
srcs = sorted(_native._SOURCE.parent.glob("*.[ch]"))
info = {
    "label": LABEL,
    "reader_extension": mod.__file__,
    "reader_extension_sha256": hashlib.sha256(open(mod.__file__, "rb").read()).hexdigest()[:16],
    "reader_source_digest": _native._source_digest(),
    "reader_sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in srcs},
    "prebuilt_target": str(_native._prebuilt_target()),
    "reps": [],
}
print(json.dumps({k: v for k, v in info.items() if k != "reps"}), flush=True)
for rep in range(1, REPS + 1):
    t0 = time.time()
    res = b.run_cache_and_packed_benchmark(bench_dir="/mnt/nvme/engram_bench_test")
    res["rep"] = rep
    res["wall_s"] = round(time.time() - t0, 2)
    info["reps"].append(res)
    print(f"[{LABEL}] rep {rep} done in {res['wall_s']} s", flush=True)
with open(f"/out/bench-cache-{LABEL}.json", "w") as f:
    json.dump(info, f, indent=1)
    f.write("\n")
