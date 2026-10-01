#!/usr/bin/env python3
"""Arms for the prefill-lane and compute-share experiment (run from the repository root).

Base: the determinism experiment's r5o-pin-b2 (r5o, the pinned DSpark cost table, the boot-time
patches with logged step costs). The candidates mount overlay sched1 (r5o's scheduler.py plus the
two upstream liveness fixes in this directory) and change one scheduler flag each, or both:
  control         r5o-pin-b2 unchanged (one prefill lane, no compute sharing)
  lanes2          --max-parallel-prefills 2
  share           --prefill-compute-share auto
  lanes2-share    both
"""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-10-01-scheduler-lanes")
base = json.loads(Path("experiments/2026-09-29-determinism/cluster-r5o-pin-b2.json").read_text())
SCHEDULER = "/opt/spark3/candidate/vllm/vllm/v1/core/sched/scheduler.py"


def arm(lanes=None, share=None):
    config = copy.deepcopy(base)
    if lanes or share:
        mounts = [m for m in config["container"]["mounts"] if m[1] != SCHEDULER]
        mounts.append(["{home}/spark3-overlay/sched1/vllm/v1/core/sched/scheduler.py", SCHEDULER, "ro"])
        config["container"]["mounts"] = mounts
    args = config["serve_args"]
    if lanes:
        args[args.index("--max-parallel-prefills") + 1] = str(lanes)
    if share:
        args += ["--prefill-compute-share", share]
    return config


for name, config in (("control", arm()), ("lanes2", arm(lanes=2)), ("share", arm(share="auto")),
                     ("lanes2-share", arm(lanes=2, share="auto"))):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    print("wrote", path)
