#!/usr/bin/env python3
"""Write the window's arms from the r6c TP4 recipe (run sync_overlay.py first).

- control.json: r6c TP4 with the migration's modules mounted (they change nothing the
  model runs) and a torch profiler directory;
- <port>.json: the modules with one port's switch files over them (overlay/<port>),
  its environment and its serving arguments;
- <combo>.json: the modules with several ports' switch files and environments
  (ports.COMBOS; their switches touch different files);
- sparknet-cute.json / sparknet-tilelang.json: the control with sparknet main
  mounted, CuTe and TileLang kernel families.

Every arm keeps the control's pinned DSpark cost curves, so the switch is the only
variable.
"""
import copy
import itertools
import json
from pathlib import Path

from ports import COMBOS, PORTS, SPARKNET_MAIN

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT).as_posix()
BASE = "experiments/2026-10-08-lil-rebase/r6c-tp4.json"
IMAGE_VLLM = "/opt/spark3/candidate/vllm/vllm"


def mounts(*overlays):
    """The files of each overlay over the image's vLLM tree, later overlays winning."""
    chosen = {}
    for overlay in overlays:
        root = HERE / "overlay" / overlay / "vllm"
        for path in sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py")):
            chosen[path] = f"{{home}}/projects/spark-ds41f/{REL}/overlay/{overlay}/vllm/{path}"
    return [[source, f"{IMAGE_VLLM}/{path}", "ro"] for path, source in chosen.items()]


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

    def arm(overlay, profile, environment=(), *more, serve_args=None):
        """r6c TP4 plus the modules and one port's switch files (or several ports')."""
        cluster = copy.deepcopy(base)
        cluster["container"]["mounts"] += mounts("modules", overlay, *more)
        cluster["environment"].update(environment)
        for flag, value in (serve_args or {}).items():
            args = cluster["serve_args"]
            if flag not in args:
                raise SystemExit(f"{profile}: {flag} is not a serving argument of the recipe")
            args[args.index(flag) + 1] = value
        set_profiler(cluster, profile)
        return cluster

    arms = {"control": arm("modules", "migration-control")}
    for name, port in PORTS.items():
        arms[name] = arm(name, f"migration-{name}", port["environment"], serve_args=port.get("serve_args"))
    for name, ports in COMBOS.items():
        files = [{p.relative_to(HERE / "overlay" / port) for p in (HERE / "overlay" / port).rglob("*.py")}
                 for port in ports]
        shared = {f for a, b in itertools.combinations(files, 2) for f in a & b}
        if shared:
            raise SystemExit(f"{name}: {', '.join(ports)} switch the same files: {sorted(map(str, shared))}")
        environment = {k: v for port in ports for k, v in PORTS[port]["environment"].items()}
        serve_args = {k: v for port in ports for k, v in PORTS[port].get("serve_args", {}).items()}
        arms[name] = arm(ports[0], f"migration-{name}", environment, *ports[1:], serve_args=serve_args)
    for family in ("cute", "tilelang"):
        cluster = arm("modules", f"migration-sparknet-{family}", {"SPARKNET_ROCE_KERNELS": family})
        cluster["container"]["mounts"].append(list(SPARKNET_MAIN["mount"]))
        arms[f"sparknet-{family}"] = cluster
    for name, cluster in arms.items():
        (HERE / f"{name}.json").write_text(json.dumps(cluster, indent=2) + "\n")
        print(f"wrote {REL}/{name}.json")


if __name__ == "__main__":
    main()
