#!/usr/bin/env python3
"""Write the window's arms from the r6c TP4 recipe (run sync_overlay.py first).

- control.json: r6c TP4 with the migration's modules mounted (they change nothing the
  model runs) and a torch profiler directory;
- <port>.json: the modules plus one port's switch (overlay/<port>) and environment;
- sparknet-cute.json / sparknet-tilelang.json: the control with sparknet main
  mounted, CuTe and TileLang kernel families.

Every arm keeps the control's pinned DSpark cost curves, so the switch is the only
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


def mounts(arm):
    """Every file of overlay/<arm>/vllm, over the image's vLLM tree."""
    root = HERE / "overlay" / arm / "vllm"
    return [[f"{{home}}/projects/spark-ds41f/{REL}/overlay/{arm}/vllm/{path}", f"{IMAGE_VLLM}/{path}", "ro"]
            for path in sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py"))]


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

    def arm(overlay, profile, environment=()):
        """r6c TP4 plus one overlay (each overlay holds the modules)."""
        cluster = copy.deepcopy(base)
        cluster["container"]["mounts"] += mounts(overlay)
        cluster["environment"].update(environment)
        set_profiler(cluster, profile)
        return cluster

    arms = {"control": arm("modules", "migration-control")}
    for name, port in PORTS.items():
        arms[name] = arm(name, f"migration-{name}", port["environment"])
    for family in ("cute", "tilelang"):
        cluster = arm("modules", f"migration-sparknet-{family}", {"SPARKNET_ROCE_KERNELS": family})
        cluster["container"]["mounts"].append(list(SPARKNET_MAIN["mount"]))
        arms[f"sparknet-{family}"] = cluster
    for name, cluster in arms.items():
        (HERE / f"{name}.json").write_text(json.dumps(cluster, indent=2) + "\n")
        print(f"wrote {REL}/{name}.json")


if __name__ == "__main__":
    main()
