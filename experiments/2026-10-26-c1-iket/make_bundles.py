#!/usr/bin/env python3
"""Write this experiment's kernel-lab bundles from the vLLM branch.

usage: make_bundles.py VLLM_REPO [--only NAME ...]

The mechanics are the IKET experiment's (../2026-10-26-iket-debug/make_bundles.py,
imported here): each bundle runs on candidate 6's image (vLLM 26382f5cf), with every
package file its ref changes mounted over the image's tree and deleted modules mounted
as ones that fail to import; ``run.py`` runs the steps in order and fails if any step
failed; ``{python}`` is the image's interpreter and ``{out}`` the output root,
/c/iket/c1-<bundle>-<UTC stamp>.

Refs (each bundle's candidate.json records the commit):
- ``c1-iket``: next-candidate with the IKET mode merged (iket-debug 52862f2ca);
- ``next-candidate``: itself, for the off build's bits and times.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IKET_DIR = HERE.parent / "2026-10-26-iket-debug"
_spec = importlib.util.spec_from_file_location("iket_make_bundles", IKET_DIR / "make_bundles.py")
iket = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iket)

BASE, IMAGE, ROOT, MISSING, RUN = iket.BASE, iket.IMAGE, iket.ROOT, iket.MISSING, iket.RUN
PROFILE, NSYS_PROFILE, NSYS, NVDISASM = iket.PROFILE, iket.NSYS_PROFILE, iket.NSYS, iket.NVDISASM
STUDY = "c1-iket"
NEXT = "next-candidate"
LAYER = "/b/c1_layer.py"
TABLE = "/b/iket_table.py"
TIMELINE = "/b/cta_timeline.py"
ROUTES = "/b/ref-woa1.json"
CHAINS = "tests/kernels/deepseek_v41_pdl_chains.py"
NSYS_GRAPH = [*NSYS_PROFILE[:2], "--cuda-graph-trace=node", *NSYS_PROFILE[2:]]


def layer(mode, tp, out, *extra):
    return ["{python}", LAYER, mode, "--tp", str(tp), "--routes", ROUTES, "--out", out, *extra]


def chain_steps(tp, live, tag):
    """One rank's shapes: the off build's times (graph, and its nsys kernel times),
    then run-iket twice: every kernel uninstrumented (exact warp lifetimes, from which
    cta_timeline.py rebuilds each launch's SM occupancy and weight stream), and every
    kernel instrumented with no sampling (the phase table)."""
    shape = ["--live", str(live), "--mla", "--head"]
    out = f"{{out}}/tp{tp}-{tag}"
    return [
        (f"TP{tp} {tag}: off, graph", layer("off", tp, f"{out}/off", *shape, "--graph", "--replays", "30"), 3600),
        (f"TP{tp} {tag}: nsys, off, graph",
         [*NSYS_GRAPH, f"{out}/nsys-graph",
          *layer("off", tp, f"{out}/nsys-graph-run", *shape, "--graph", "--replays", "10")], 3600),
        (f"TP{tp} {tag}: nsys, off, eager",
         [*NSYS_PROFILE, f"{out}/nsys-eager",
          *layer("off", tp, f"{out}/nsys-eager-run", *shape, "--replays", "10")], 3600),
        (f"TP{tp} {tag}: lifetimes, run-iket, every kernel uninstrumented",
         [*PROFILE, f"{out}/lifetimes", "profile", "--postprocess", "json", "--",
          *layer("on", tp, f"{out}/lifetimes-run", *shape, "--iket", f"{out}/lifetimes-run/iket",
                 "--skip", ".", "--layers", "4", "--replays", "3")], 3600),
        (f"TP{tp} {tag}: phases, run-iket, every kernel instrumented",
         [*PROFILE, f"{out}/trace", "profile", "--postprocess", "json", "--",
          *layer("on", tp, f"{out}/trace-run", *shape, "--iket", f"{out}/trace-run/iket",
                 "--layers", "4", "--replays", "2")], 5400),
        (f"TP{tp} {tag}: phase table",
         ["{python}", TABLE, f"{out}/trace", "--sources", f"{out}/trace-run/iket/sources",
          "--nsys", f"graph={out}/nsys-graph.sqlite", f"eager={out}/nsys-eager.sqlite",
          "--json", f"{out}/table.json"], 1800),
        (f"TP{tp} {tag}: CTA timelines",
         ["{python}", TIMELINE, f"{out}/lifetimes", "--tp", str(tp), "--routes", ROUTES, "--live", str(live),
          "--layers", "4", "--json", f"{out}/timeline.json"], 1800),
    ]


BUNDLES = {
    "layer": dict(
        ref=STUDY,
        what="c1 decode layers at a TP4 and a TP3 rank's shapes (reference routes, 3 of 6 rows "
             "live): times and kernel times of the off build, warp lifetimes of every launch, "
             "and the per-phase table",
        steps=[*chain_steps(4, 3, "live3"), *chain_steps(3, 3, "live3")],
    ),
    "layer-live6": dict(
        ref=STUDY,
        what="c1 decode layers at a TP4 rank's shapes with all 6 rows live (24 distinct experts "
             "a layer): the routed experts' second operating point",
        steps=chain_steps(4, 6, "live6"),
    ),
    "layer-base": dict(
        ref=NEXT,
        what="next-candidate itself: the layer harness's bits and times at both ranks' shapes, "
             "for the IKET build's off identity and the prototypes' baseline",
        steps=[(f"TP{tp}: next-candidate, graph",
                layer("off", tp, f"{{out}}/tp{tp}", "--live", "3", "--mla", "--head", "--graph", "--replays", "30"),
                3600) for tp in (4, 3)],
    ),
}

SCRIPTS = {
    "c1_layer.py": HERE / "harness" / "c1_layer.py",
    "cta_timeline.py": HERE / "cta_timeline.py",
    "iket_hot.py": IKET_DIR / "harness" / "iket_hot.py",
    "iket_table.py": IKET_DIR / "iket_table.py",
    "nsys_window.py": IKET_DIR / "nsys_window.py",
    "ref-woa1.json": HERE.parent / "2026-10-18-json-count" / "runs" / "ref-woa1.json",
}


def main():
    repo = sys.argv[1]
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else []
    for name, spec in BUNDLES.items():
        if only and name not in only:
            continue
        ref = spec["ref"]
        head = iket.git(repo, "rev-parse", "--short=9", ref).strip()
        status = [line.split("\t") for line in
                  iket.git(repo, "diff", "--no-renames", "--name-status", BASE, ref).splitlines() if line]
        package = [(kind, path) for kind, path in status if path.startswith("vllm/")]
        tests = [CHAINS]
        bundle = HERE / "bundles" / name
        planned = {"run.py", "steps.json", "candidate.json", *SCRIPTS}
        planned |= {f"overlay/{path}" for _, path in package}
        planned |= {path.replace("/", "_") for path in tests}
        # Files are overwritten in place, never deleted: a file this run would not
        # write stops it, for a person to remove.
        stale = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*")
                       if p.is_file() and str(p.relative_to(bundle)) not in planned)
        if stale:
            raise SystemExit(f"{bundle} holds files this run would not write: {stale}")
        for kind, path in package:
            target = bundle / "overlay" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(MISSING.encode() if kind == "D" else iket.show(repo, ref, path))
        for path in tests:
            (bundle / path.replace("/", "_")).write_bytes(iket.show(repo, ref, path))
        for target, source in SCRIPTS.items():
            shutil.copy(source, bundle / target)
        (bundle / "run.py").write_text(RUN)
        (bundle / "steps.json").write_text(json.dumps(spec["steps"], indent=1) + "\n")
        candidate = {
            "description": f"{spec['what']} (vLLM {ref} {head}, {len(package)} package files over the "
                           f"release's {BASE}) on candidate 6's image.",
            "image": IMAGE,
            "env": {
                "HF_HUB_OFFLINE": "1",
                "IKET_BUNDLE": f"c1-{name}",
                "TILELANG_CACHE_DIR": f"/c/tilelang-c1-{name}",
                "TRITON_CACHE_DIR": f"/c/triton-c1-{name}",
                "VLLM_CACHE_ROOT": f"/c/vllm-c1-{name}",
                "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm",
                **spec.get("env", {}),
            },
            "workdir": "vllm",
            "argv": ["/b/run.py"],
            "verdict": "status",
            "mounts": [["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"],
                       *([target, f"/b/{target}"] for target in SCRIPTS), NSYS, *spec.get("mounts", []),
                       *([path.replace("/", "_"), f"{ROOT}/{path}"] for path in tests)],
        }
        (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
        print(f"{name}: {ref} {head}, {len(package)} package files, {len(spec['steps'])} steps")


if __name__ == "__main__":
    main()
