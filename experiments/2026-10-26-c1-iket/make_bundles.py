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
- ``next-candidate``: itself, for the off build's bits and times;
- ``c1-stream``: next-candidate with the stream grouped GEMM (the first prototype);
- ``c1-stream-iket``: that with the IKET mode merged, for its traces.

An A/B bundle (``builds``) mounts its base ref's files over the image's tree at
/b/base/vllm, with the layer chain's test helper, and each prototype's files over the
base ref at /b/<name>/vllm; ab.py builds and alternates the trees.
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
LEVER2_BASE = "c579b3fe1"  # next-candidate when lever 2's prototypes branched from it
STREAM = "c1-stream"
STREAM_IKET = "c1-stream-iket"
FUSION = "c1-fusion"
FUSION2 = "c1-fusion2"
MHCFOLD = "c1-mhcfold"
MIXAUX = "c1-mixaux"
MHC_TESTS = "tests/kernels/test_deepseek_v41_tilelang_mhc.py"
MOE_TESTS = "tests/kernels/moe/test_deepseek_v41_tilelang_moe.py"
PDL_TESTS = "tests/kernels/test_deepseek_v41_tilelang_pdl.py"
# The PDL experiment's test invocation: with vLLM's conftest (the layer-chain tests take
# its default_vllm_config), the release-tests bundle's two plugins, and the PDL source
# check of every launched kernel.
RELEASE_TESTS = HERE.parent / "2026-10-25-spark-ds41f-candidate" / "rc6" / "bundles" / "release-tests"
PYTEST_CHECKED = ["{python}", "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "-p", "tblib_stub",
                  "-p", "offline_skip", "-p", "tests.kernels.deepseek_v41_pdl_sources",
                  "-o", "faulthandler_timeout=600"]
LAYER = "/b/c1_layer.py"
BOUND = "/b/stream_bound.py"
TABLE = "/b/iket_table.py"
TIMELINE = "/b/cta_timeline.py"
ROUTES = "/b/ref-woa1.json"
CHAINS = "tests/kernels/deepseek_v41_pdl_chains.py"
NSYS_GRAPH = [*NSYS_PROFILE[:2], "--cuda-graph-trace=node", *NSYS_PROFILE[2:]]


def layer(mode, tp, out, *extra):
    return ["{python}", LAYER, mode, "--tp", str(tp), "--routes", ROUTES, "--out", out, *extra]


# run-iket records warp lifetimes only for instrumented kernels (smoke runs 2 and 3: none
# with every kernel skipped, the combine's alone with only it instrumented). With one CTA
# a launch recording ranges (VLLM_DS41_IKET_CTAS=1) and loops sampled once, every warp's
# lifetime is recorded and the other CTAs pass a uniform branch at each event.
LIFETIMES_SAMPLING = ["--ctas", "1", "--iters", "1"]


def lifetime_steps(tp, live, tag):
    """run-iket with one CTA a launch recording ranges (every warp's lifetime), and
    cta_timeline.py's SM occupancy and weight stream from them."""
    shape = ["--live", str(live), "--mla", "--head"]
    out = f"{{out}}/tp{tp}-{tag}"
    return [
        (f"TP{tp} {tag}: lifetimes, run-iket, one CTA a launch recording",
         [*PROFILE, f"{out}/lifetimes", "profile", "--postprocess", "json", "--",
          *layer("on", tp, f"{out}/lifetimes-run", *shape, "--iket", f"{out}/lifetimes-run/iket",
                 *LIFETIMES_SAMPLING, "--layers", "4", "--replays", "3")], 3600),
        (f"TP{tp} {tag}: CTA timelines",
         ["{python}", TIMELINE, f"{out}/lifetimes", "--tp", str(tp), "--routes", ROUTES, "--live", str(live),
          "--layers", "4", "--json", f"{out}/timeline.json"], 1800),
    ]


def chain_steps(tp, live, tag, phases=True):
    """One rank's shapes: the off build's times (graph, and its nsys kernel times),
    then run-iket twice: with only the combine instrumented (lifetime_steps), and with
    ``phases`` every kernel instrumented with no sampling (the phase table)."""
    shape = ["--live", str(live), "--mla", "--head"]
    out = f"{{out}}/tp{tp}-{tag}"
    steps = [
        (f"TP{tp} {tag}: off, graph", layer("off", tp, f"{out}/off", *shape, "--graph", "--replays", "30"), 3600),
        (f"TP{tp} {tag}: nsys, off, graph",
         [*NSYS_GRAPH, f"{out}/nsys-graph",
          *layer("off", tp, f"{out}/nsys-graph-run", *shape, "--graph", "--replays", "10")], 3600),
        (f"TP{tp} {tag}: nsys, off, eager",
         [*NSYS_PROFILE, f"{out}/nsys-eager",
          *layer("off", tp, f"{out}/nsys-eager-run", *shape, "--replays", "10")], 3600),
        lifetime_steps(tp, live, tag)[0],
        (f"TP{tp} {tag}: phases, run-iket, every kernel instrumented",
         [*PROFILE, f"{out}/trace", "profile", "--postprocess", "json", "--",
          *layer("on", tp, f"{out}/trace-run", *shape, "--iket", f"{out}/trace-run/iket",
                 "--layers", "4", "--replays", "2")], 5400),
        (f"TP{tp} {tag}: phase table",
         ["{python}", TABLE, f"{out}/trace", "--sources", f"{out}/trace-run/iket/sources",
          "--nsys", f"graph={out}/nsys-graph.sqlite", f"eager={out}/nsys-eager.sqlite",
          "--json", f"{out}/table.json"], 1800),
        lifetime_steps(tp, live, tag)[1],
    ]
    if not phases:
        steps = [step for step in steps if not step[0].split(": ")[1].startswith("phase")]
    return steps


SMOKE = ["--layers", "2", "--sets", "2", "--mla", "--head"]

BUNDLES = {
    "smoke": dict(
        ref=STUDY,
        what="the layer harness in brief: two layers at each rank's shapes, a graph and eager, "
             "and one run-iket lifetimes trace with its CTA timelines",
        steps=[
            *((f"TP{tp}: off, graph", layer("off", tp, f"{{out}}/tp{tp}", *SMOKE, "--graph", "--replays", "5"), 1800)
              for tp in (4, 3)),
            ("TP4: off, eager", layer("off", 4, "{out}/tp4-eager", *SMOKE, "--replays", "3"), 1800),
            ("TP4: lifetimes, run-iket",
             [*PROFILE, "{out}/lifetimes", "profile", "--postprocess", "json", "--",
              *layer("on", 4, "{out}/lifetimes-run", *SMOKE, "--iket", "{out}/lifetimes-run/iket",
                     *LIFETIMES_SAMPLING, "--replays", "1")], 1800),
            ("TP4: CTA timelines", ["{python}", TIMELINE, "{out}/lifetimes", "--tp", "4", "--json",
                                    "{out}/timeline.json"], 600),
        ],
    ),
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
    "layer-stream": dict(
        ref=STREAM_IKET,
        what="the stream grouped GEMM on the c1 layer harness: times, nsys kernel times and warp "
             "lifetimes at TP4 (3 and 6 live rows) and TP3 (3)",
        steps=[*chain_steps(4, 3, "live3", phases=False), *chain_steps(4, 6, "live6", phases=False),
               *chain_steps(3, 3, "live3", phases=False)],
    ),
    "lifetimes": dict(
        ref=STUDY,
        what="warp lifetimes of every launch (only the combine instrumented) and their CTA "
             "timelines: TP4 at 3 and 6 live rows, TP3 at 3",
        steps=[*lifetime_steps(4, 3, "live3"), *lifetime_steps(4, 6, "live6"), *lifetime_steps(3, 3, "live3")],
    ),
    "lifetimes-stream": dict(
        ref=STREAM_IKET,
        what="the stream grouped GEMM's warp lifetimes and CTA timelines: TP4 at 3 and 6 live "
             "rows, TP3 at 3",
        steps=[*lifetime_steps(4, 3, "live3"), *lifetime_steps(4, 6, "live6"), *lifetime_steps(3, 3, "live3")],
    ),
    "bound": dict(
        ref=NEXT,
        what="the routed experts' chain against an ideal read of the same bytes (one launch a "
             "layer, two, and one for all layers), at both ranks' shapes and 3 and 6 live rows",
        steps=[(f"TP{tp}, {live} live rows",
                ["{python}", BOUND, "--tp", str(tp), "--live", str(live), "--routes", ROUTES,
                 "--out", f"{{out}}/tp{tp}-live{live}"], 3600)
               for tp in (4, 3) for live in (3, 6)],
    ),
    "bound-stream": dict(
        ref=STREAM,
        what="the bound harness on the stream grouped GEMM's build: its routed chain against the "
             "same ideal reads",
        steps=[(f"TP{tp}, {live} live rows",
                ["{python}", BOUND, "--tp", str(tp), "--live", str(live), "--routes", ROUTES,
                 "--out", f"{{out}}/tp{tp}-live{live}"], 3600)
               for tp in (4, 3) for live in (3, 6)],
    ),
    "stream-tests": dict(
        ref=STREAM,
        what="the stream grouped GEMM: its bits against the grid GEMM's at both ranks' expert "
             "shapes and 1-15 rows, the routed-expert tests, and the PDL tests and source check",
        tests=[MOE_TESTS, PDL_TESTS, "tests/kernels/deepseek_v41_pdl_sources.py",
               "tests/kernels/deepseek_v41_contracts.py"],
        steps=[("routed-expert and PDL tests, every launched kernel source-checked",
                [*PYTEST_CHECKED, "--basetemp", "{out}/pytest", MOE_TESTS, PDL_TESTS], 3600)],
    ),
    "stages-ab": dict(
        ref=NEXT,
        builds={"stages4": "c1-stages4", "stream48": STREAM, "stream48s4": "c1-stream-s4",
                "prologue": "c1-prologue"},
        what="one weight-stream variable an arm against next-candidate on the c1 layer harness, in "
             "rotation, four rounds: the grid GEMMs at four stages, the stream GEMM on all 48 SMs, "
             "that at four stages, and the grid GEMMs loading base words after the first stages; TP4 "
             "at 3 and 6 live rows, TP3 at 3",
        steps=[("A/B, four rounds", ["{python}", "/b/ab.py", "4", "tp=4 live=3", "tp=4 live=6", "tp=3 live=3"],
                14400)],
    ),
    "fusion-tests": dict(
        ref=FUSION,
        what="the mHC entry with every stream's fn block loaded at once: the mHC and PDL tests, "
             "every launched kernel source-checked",
        tests=[MHC_TESTS, PDL_TESTS, "tests/kernels/deepseek_v41_pdl_sources.py",
               "tests/kernels/deepseek_v41_contracts.py"],
        steps=[("mHC and PDL tests, every launched kernel source-checked",
                [*PYTEST_CHECKED, "--basetemp", "{out}/pytest", MHC_TESTS, PDL_TESTS], 3600)],
    ),
    "fusion-ab": dict(
        ref=LEVER2_BASE,
        builds={"mhcfn": FUSION},
        what="the mHC entry with every stream's fn block loaded at once against next-candidate on the "
             "layer harness with all-reduce stand-ins, in alternation, four rounds: c1 (6 rows, 3 live), "
             "c8 (24 rows) and c16 (48 rows) at TP4 and TP3 shapes",
        steps=[("A/B, four rounds", ["{python}", "/b/ab.py", "4",
                                     *(f"tp={tp} rows={rows} live={live} ar=on"
                                       for tp in (4, 3) for rows, live in ((6, 3), (24, 24), (48, 48)))],
                14400)],
    ),
    "fusion2-tests": dict(
        ref=FUSION2,
        what="the mHC entry loading the next stream's fn block during the current stream's GEMMs: "
             "the mHC and PDL tests, every launched kernel source-checked",
        tests=[MHC_TESTS, PDL_TESTS, "tests/kernels/deepseek_v41_pdl_sources.py",
               "tests/kernels/deepseek_v41_contracts.py"],
        steps=[("mHC and PDL tests, every launched kernel source-checked",
                [*PYTEST_CHECKED, "--basetemp", "{out}/pytest", MHC_TESTS, PDL_TESTS], 3600)],
    ),
    "fusion2-ab": dict(
        ref=LEVER2_BASE,
        builds={"mhcfn2": FUSION2},
        what="the mHC entry loading the next stream's fn block during the current stream's GEMMs "
             "against next-candidate on the layer harness with all-reduce stand-ins, in alternation, "
             "four rounds: c1, c8 and c16 at TP4 and TP3 shapes",
        steps=[("A/B, four rounds", ["{python}", "/b/ab.py", "4",
                                     *(f"tp={tp} rows={rows} live={live} ar=on"
                                       for tp in (4, 3) for rows, live in ((6, 3), (24, 24), (48, 48)))],
                14400)],
    ),
    "mhcfold-tests": dict(
        ref=MHCFOLD,
        what="the mHC entry in one launch at decode sizes (finalize folded into project_streams "
             "past a grid barrier): the mHC and PDL tests, every launched kernel source-checked",
        tests=[MHC_TESTS, PDL_TESTS, "tests/kernels/deepseek_v41_pdl_sources.py",
               "tests/kernels/deepseek_v41_contracts.py"],
        steps=[("mHC and PDL tests, every launched kernel source-checked",
                [*PYTEST_CHECKED, "--basetemp", "{out}/pytest", MHC_TESTS, PDL_TESTS], 3600)],
    ),
    "mhcfold-ab": dict(
        ref=LEVER2_BASE,
        builds={"mhcfold": MHCFOLD},
        what="the mHC entry in one launch at decode sizes against next-candidate on the layer "
             "harness with all-reduce stand-ins, in alternation, four rounds: c1, c8 and c16 at "
             "TP4 and TP3 shapes",
        steps=[("A/B, four rounds", ["{python}", "/b/ab.py", "4",
                                     *(f"tp={tp} rows={rows} live={live} ar=on"
                                       for tp in (4, 3) for rows, live in ((6, 3), (24, 24), (48, 48)))],
                14400)],
    ),
    "mixaux-tests": dict(
        ref=MIXAUX,
        what="the mHC entry's new mixes on a side stream at decode sizes, joined at the next "
             "mHC operation: the mHC and PDL tests, every launched kernel source-checked",
        tests=[MHC_TESTS, PDL_TESTS, "tests/kernels/deepseek_v41_pdl_sources.py",
               "tests/kernels/deepseek_v41_contracts.py"],
        steps=[("mHC and PDL tests, every launched kernel source-checked",
                [*PYTEST_CHECKED, "--basetemp", "{out}/pytest", MHC_TESTS, PDL_TESTS], 3600)],
    ),
    "mixaux-ab": dict(
        ref=LEVER2_BASE,
        builds={"mixaux": MIXAUX},
        what="the mHC entry's new mixes on a side stream against next-candidate on the layer "
             "harness with all-reduce stand-ins, in alternation, four rounds: c1, c8 and c16 at "
             "TP4 and TP3 shapes",
        steps=[("A/B, four rounds", ["{python}", "/b/ab.py", "4",
                                     *(f"tp={tp} rows={rows} live={live} ar=on"
                                       for tp in (4, 3) for rows, live in ((6, 3), (24, 24), (48, 48)))],
                14400)],
    ),
    "stream-ab": dict(
        ref=NEXT,
        builds={"stream": STREAM},
        what="the stream grouped GEMM against next-candidate on the c1 layer harness, in "
             "alternation, six rounds: TP4 at 3 and 6 live rows, TP3 at 3",
        steps=[("A/B, six rounds", ["{python}", "/b/ab.py", "6", "tp=4 live=3", "tp=4 live=6", "tp=3 live=3"],
                14400)],
    ),
}

SCRIPTS = {
    "ab.py": HERE / "ab.py",
    "c1_layer.py": HERE / "harness" / "c1_layer.py",
    "stream_bound.py": HERE / "harness" / "stream_bound.py",
    "cta_timeline.py": HERE / "cta_timeline.py",
    "iket_hot.py": IKET_DIR / "harness" / "iket_hot.py",
    "iket_table.py": IKET_DIR / "iket_table.py",
    "nsys_window.py": IKET_DIR / "nsys_window.py",
    "ref-woa1.json": HERE.parent / "2026-10-18-json-count" / "runs" / "ref-woa1.json",
    "tblib_stub.py": RELEASE_TESTS / "tblib_stub.py",
    "offline_skip.py": RELEASE_TESTS / "offline_skip.py",
}


def main():
    repo = sys.argv[1]
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else []
    for name, spec in BUNDLES.items():
        if only and name not in only:
            continue
        ref = spec["ref"]
        head = iket.git(repo, "rev-parse", "--short=9", ref).strip()
        package = changed(repo, BASE, ref)
        builds = {build: (build_ref, changed(repo, ref, build_ref))
                  for build, build_ref in spec.get("builds", {}).items()}
        bundle = HERE / "bundles" / name
        # An A/B bundle carries its builds' trees; any other bundle overlays the image.
        files = {}
        if builds:
            for kind, path in package:
                files[f"base/{path}"] = (kind, ref, path)
            files[f"base/{CHAINS}"] = ("M", ref, CHAINS)
            for build, (build_ref, build_package) in builds.items():
                for kind, path in build_package:
                    files[f"{build}/{path}"] = (kind, build_ref, path)
                # The harness's mHC entries and projections come from the chain module.
                files[f"{build}/{CHAINS}"] = ("M", build_ref, CHAINS)
            tests = []
        else:
            for kind, path in package:
                files[f"overlay/{path}"] = (kind, ref, path)
            tests = [CHAINS, *spec.get("tests", [])]
            for path in tests:
                files[path.replace("/", "_")] = ("M", ref, path)
        planned = {"run.py", "steps.json", "candidate.json", *SCRIPTS, *files}
        # Files are overwritten in place, never deleted: a file this run would not
        # write stops it, for a person to remove.
        stale = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*")
                       if p.is_file() and str(p.relative_to(bundle)) not in planned)
        if stale:
            raise SystemExit(f"{bundle} holds files this run would not write: {stale}")
        for target_path, (kind, at, path) in files.items():
            target = bundle / target_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(MISSING.encode() if kind == "D" else iket.show(repo, at, path))
        for target, source in SCRIPTS.items():
            shutil.copy(source, bundle / target)
        (bundle / "run.py").write_text(RUN)
        (bundle / "steps.json").write_text(json.dumps(spec["steps"], indent=1) + "\n")
        build_text = "".join(f"; build {build}: {build_ref} {iket.git(repo, 'rev-parse', '--short=9', build_ref).strip()}, "
                             f"{len(build_package)} package files over it"
                             for build, (build_ref, build_package) in builds.items())
        candidate = {
            "description": f"{spec['what']} (vLLM {ref} {head}, {len(package)} package files over the "
                           f"release's {BASE}{build_text}) on candidate 6's image.",
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
                       *([path.replace("/", "_"), f"{ROOT}/{path}"] for path in tests),
                       *([tree, f"/b/{tree}"] for tree in (["base", *builds] if builds else []))],
        }
        (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
        print(f"{name}: {ref} {head}, {len(package)} package files{build_text}, {len(spec['steps'])} steps")


def changed(repo, old, new):
    """(kind, path) of every vLLM package file ``new`` changes from ``old``."""
    status = [line.split("\t") for line in
              iket.git(repo, "diff", "--no-renames", "--name-status", old, new).splitlines() if line]
    return [(kind, path) for kind, path in status if path.startswith("vllm/")]


if __name__ == "__main__":
    main()
