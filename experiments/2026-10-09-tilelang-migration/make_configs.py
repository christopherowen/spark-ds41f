#!/usr/bin/env python3
"""Write the window's arms from the r6c TP4 recipe.

- control.json: r6c TP4 as it serves, with a torch profiler directory;
- <port>.json: control plus one port's overlay files and environment;
- sparknet-cute.json / sparknet-tilelang.json: control with sparknet main mounted,
  CuTe and TileLang kernel families.

Every arm keeps the control's pinned DSpark cost curves, so the kernel is the only
variable.
"""
import copy
import json
from pathlib import Path

from ports import PORTS, SPARKNET_MAIN

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT).as_posix()
BASE = "experiments/2026-10-08-lil-rebase/r6c-tp4.json"
IMAGE_VLLM = "/opt/spark3/candidate/vllm/vllm"


def mount(port, path):
    return [f"{{home}}/projects/spark-ds41f/{REL}/overlay/{port}/vllm/{path}", f"{IMAGE_VLLM}/{path}", "ro"]


def set_profiler(cluster, name):
    args = cluster["serve_args"]
    profiler = {"profiler": "torch", "torch_profiler_dir": f"/cache/kkref/profiles/{name}",
                "torch_profiler_with_stack": False, "torch_profiler_record_shapes": True,
                "ignore_frontend": True, "torch_profiler_use_gzip": True}
    value = json.dumps(profiler, separators=(",", ":"))
    if "--profiler-config" in args:
        args[args.index("--profiler-config") + 1] = value
    else:
        args += ["--profiler-config", value]


def main():
    base = json.loads((ROOT / BASE).read_text())
    arms = {}
    control = copy.deepcopy(base)
    set_profiler(control, "migration-control")
    arms["control"] = control
    for name, port in PORTS.items():
        arm = copy.deepcopy(control)
        arm["container"]["mounts"] += [mount(name, f) for f in port["files"]]
        arm["environment"].update(port["environment"])
        set_profiler(arm, f"migration-{name}")
        arms[name] = arm
    for family in ("cute", "tilelang"):
        arm = copy.deepcopy(control)
        arm["container"]["mounts"].append(list(SPARKNET_MAIN["mount"]))
        arm["environment"]["SPARKNET_ROCE_KERNELS"] = family
        set_profiler(arm, f"migration-sparknet-{family}")
        arms[f"sparknet-{family}"] = arm
    for name, cluster in arms.items():
        (HERE / f"{name}.json").write_text(json.dumps(cluster, indent=2) + "\n")
        print(f"wrote {REL}/{name}.json")


if __name__ == "__main__":
    main()
