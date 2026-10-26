#!/usr/bin/env python3
"""Write the PDL experiment's kernel-lab bundles and serving arms from the vLLM branch.

usage: make_bundles.py [VLLM_REPO] [IMAGE_TREE] [BASE] [HEAD] [NAME=REF...]
       (default ~/projects/vllm-pdl-next 26382f5cf 577091997 pdl-next)

Each NAME=REF is a variant of the PDL build (a fix candidate on HEAD): its package
files over BASE, those that differ from HEAD's written to overlay-NAME/. With variants,
bundles/pdl-fix, bundles/pdl-fix-focus (fifteen rounds at six row counts) and
bundles/pdl-fix-trace time and trace every build side by side.

IMAGE_TREE is the tree image -next-6 carries (candidate 6's, 26382f5cf); BASE is the
build PDL is measured against (next-candidate); HEAD is the PDL branch on it. Every file
HEAD changes over IMAGE_TREE (package files, tests, benchmarks) is written once to
overlay/, and BASE's version of each package file PDL changes to overlay-base/. The
bundles and serving arms mount them from the node's checkout over the image's tree: a
build is the image's tree with its changed files in place. A package module deleted
since IMAGE_TREE is replaced in overlay/ by one that fails to import as a missing module
does (as retire-conditionals' bundles do), so nothing in the image reaches it.

- bundles/pdl-tests: the PDL tests, then the DS4.1 TileLang kernel tests of every
  module PDL changes, on the PDL build, all with the source check
  (tests.kernels.deepseek_v41_pdl_sources);
- bundles/pdl-chain: the layer-chain benchmark on the base build and on the PDL
  build in alternation (pdl_chain_ab.py), bits compared;
- bundles/pdl-mechanism: the synthetic PDL benchmark and the traced layer
  (layer-c1.json, from critical_path.py --layer-sequence);
- bundles/pdl-rows: the same A/B over pdl_rows.py, every row count the engine grid
  judges with the shared expert beside the routed experts and inline;
  bundles/pdl-rows-focus: 15 rounds at the row counts where PDL was level or slower;
- bundles/pdl-trace: the 16- and 48-row chains under the torch profiler, kernel by kernel;
- bundles/pdl-sources: sources_cpu.py, every chain kernel lowered at its decode shapes
  without device compilation and checked as the plugin checks launched kernels;
- pdl-tp4.json, pdl-tp4-dev.json (PDL) and base-tp4.json, base-tp4-dev.json (BASE):
  candidate 6's arms with the overlays mounted.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(HERE.parents[1])
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
RC6 = "experiments/2026-10-25-spark-ds41f-candidate/rc6"
RELEASE_BUNDLE = (
    HERE.parent / "2026-10-25-spark-ds41f-candidate/rc6/bundles/release-tests"
)
ENV = {
    "HF_HUB_OFFLINE": "1",
    "TILELANG_CACHE_DIR": "/c/tilelang-pdl",
    "PYTHONPATH": f"/b:{ROOT}:/opt/spark-ds41f/candidate/b12x",
}
PYTEST = [
    "-m",
    "pytest",
    "-q",
    "-rfEs",
    "-p",
    "no:cacheprovider",
    "-p",
    "tblib_stub",
    "-p",
    "offline_skip",
    "-p",
    "tests.kernels.deepseek_v41_pdl_sources",
    "-o",
    "faulthandler_timeout=600",
]
# The DS4.1 TileLang kernel tests of the modules the branch changes (config/release-tests.json
# gates each of them), run on the PDL build with every launched kernel source-checked.
KERNEL_TESTS = [
    "tests/kernels/test_deepseek_v41_tilelang_mhc.py",
    "tests/kernels/test_deepseek_v41_tilelang_rope.py",
    "tests/kernels/test_deepseek_v41_tilelang_cache_writer.py",
    "tests/kernels/test_deepseek_v41_tilelang_metadata.py",
    "tests/kernels/test_deepseek_v41_tilelang_compress.py",
    "tests/kernels/test_deepseek_v41_tilelang_activation.py",
    "tests/kernels/attention/test_deepseek_v41_tilelang.py",
    "tests/kernels/moe/test_deepseek_v41_tilelang_moe.py",
    "tests/kernels/moe/test_deepseek_v41_tilelang_router.py",
    "tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py",
    "tests/kernels/quantization/test_deepseek_v41_tilelang_wo.py",
]


MISSING = """# Deleted in vLLM since the image's tree ({image_tree}..{head}); mounted over the
# image's copy so that importing it fails as importing a missing module does.
raise ModuleNotFoundError(f"No module named {{__name__!r}}", name=__name__)
"""


def changed(repo: str, base: str, head: str, status: str = "AM") -> list[str]:
    out = subprocess.run(
        [
            "git",
            "-C",
            repo,
            "diff",
            "--name-only",
            f"--diff-filter={status}",
            base,
            head,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return out.split()


def copy(repo: str, head: str, path: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = subprocess.run(
        ["git", "-C", repo, "show", f"{head}:{path}"], check=True, capture_output=True
    ).stdout
    target.write_bytes(blob)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1) + "\n")


def bundle(name: str, description: str, argv: list, mounts: list, verdict: str) -> Path:
    path = HERE / "bundles" / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    write_json(
        path / "candidate.json",
        {
            "description": description,
            "image": IMAGE,
            "env": ENV,
            "workdir": "vllm",
            "argv": argv,
            "verdict": verdict,
            "mounts": mounts,
        },
    )
    return path


def rev(repo: str, name: str) -> str:
    return subprocess.run(
        ["git", "-C", repo, "rev-parse", "--short=9", name],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def packages(files: list[str]) -> list[str]:
    return [f for f in files if f.startswith("vllm/")]


def main() -> None:
    args = sys.argv[1:]
    repo = args[0] if args else str(Path.home() / "projects/vllm-pdl-next")
    image_tree = args[1] if len(args) > 1 else "26382f5cf"
    base = args[2] if len(args) > 2 else "577091997"
    head = args[3] if len(args) > 3 else "pdl-next"
    variants = [arg.split("=", 1) for arg in args[4:]]
    commit, base_commit = rev(repo, head), rev(repo, base)
    files = changed(repo, image_tree, head)
    package = packages(files)
    extra = [f for f in files if not f.startswith("vllm/")]
    # Package modules deleted since the image's tree (by next-candidate; PDL deletes none).
    gone = [
        f for f in packages(changed(repo, image_tree, head, "D")) if f.endswith(".py")
    ]
    if gone != [
        f for f in packages(changed(repo, image_tree, base, "D")) if f.endswith(".py")
    ]:
        raise SystemExit("the PDL branch deletes modules the base build keeps")
    base_package = packages(changed(repo, image_tree, base))
    pdl_files = changed(repo, base, head)
    pdl_package = packages(pdl_files)
    # The PDL branch's own tests and benchmarks: all the chain and mechanism bundles run.
    pdl_extra = [f for f in pdl_files if not f.startswith("vllm/")]
    print(
        f"vLLM {head} {commit} over {image_tree}: {len(package)} package files, "
        f"{len(extra)} others; base {base_commit}: {len(base_package)} package files; "
        f"PDL over base: {len(pdl_package)}; {len(gone)} deleted modules fail to import"
    )

    # Each file once: HEAD's files over the image's tree in overlay/, and in
    # overlay-base/ BASE's version of the package files PDL changes. The bundles mount
    # them from the lab node's checkout (lab.py resolves relative mount sources there).
    shared = [f for f in base_package if f in pdl_package]
    for name, ref, paths in (("overlay", head, files), ("overlay-base", base, shared)):
        shutil.rmtree(HERE / name, ignore_errors=True)
        for f in paths:
            copy(repo, ref, f, HERE / name / f)
    for f in gone:
        (HERE / "overlay" / f).parent.mkdir(parents=True, exist_ok=True)
        (HERE / "overlay" / f).write_text(
            MISSING.format(image_tree=image_tree, head=base)
        )
    package += gone
    base_package += gone

    def source(f: str, build: str = "pdl") -> str:
        """Where ``build``'s version of ``f`` is, relative to the checkout."""
        name = "overlay-base" if build == "base" and f in shared else "overlay"
        return f"{REL}/{name}/{f}"

    def into_tree(paths, build: str = "pdl") -> list:
        """``build``'s version of each file, mounted over the image's tree."""
        return [[source(f, build), f"{ROOT}/{f}"] for f in paths]

    # Variants: their package files over the base, mounted at /b/NAME; a file the
    # variant changes from HEAD comes from overlay-NAME/, the others from overlay/.
    variant_mounts, variant_notes, owns = [], [], {}
    for name, ref in variants:
        own = owns[name] = set(packages(changed(repo, head, ref)))
        shutil.rmtree(HERE / f"overlay-{name}", ignore_errors=True)
        for f in sorted(own):
            copy(repo, ref, f, HERE / f"overlay-{name}" / f)
        mine = packages(changed(repo, base, ref))
        variant_mounts += [
            [f"{REL}/overlay-{name}/{f}" if f in own else source(f), f"/b/{name}/{f}"]
            for f in mine
        ]
        variant_notes.append(f"{name} (vLLM {ref} {rev(repo, ref)})")
        print(f"variant {name}: {ref} {rev(repo, ref)}, {len(mine)} package files over the base, "
              f"{len(own)} differing from {head}")  # fmt: skip

    # PDL tests and the kernel tests with the source check.
    steps = [
        [
            "PDL tests",
            PYTEST
            + [
                "--basetemp",
                "/c/pytest-pdl",
                "tests/kernels/test_deepseek_v41_tilelang_pdl.py",
            ],
            3600,
        ],
        [
            "DS4.1 kernel tests on the PDL build, sources checked",
            PYTEST + ["--basetemp", "/c/pytest-pdl-kernels", *KERNEL_TESTS],
            7200,
        ],
    ]
    tests = bundle(
        "pdl-tests",
        f"PDL along the decode chain (vLLM {head} {commit}: next-candidate {base_commit} with PDL) on the "
        f"-next-6 image with its package files over {image_tree}'s: the PDL "
        "tests (programmatic edges are live in graphs, a waiting reader never reads early, the layer chain "
        "bit-equal over poisoned replays), then every changed module's TileLang kernel tests with each launched "
        "kernel's CUDA source checked (wait first, trigger after; no PDL in warp-specialized GEMMs).",
        ["/b/run.py"],
        [
            ["run.py", "/b/run.py"],
            ["steps.json", "/b/steps.json"],
            ["tblib_stub.py", "/b/tblib_stub.py"],
            ["offline_skip.py", "/b/offline_skip.py"],
        ],
        "exit",
    )
    candidate = json.loads((tests / "candidate.json").read_text())
    candidate["mounts"] += into_tree(files + gone)
    candidate["io_uring"] = True
    write_json(tests / "candidate.json", candidate)
    write_json(tests / "steps.json", steps)
    shutil.copyfile(HERE / "run_steps.py", tests / "run.py")
    for shim in ("tblib_stub.py", "offline_skip.py"):
        shutil.copyfile(RELEASE_BUNDLE / shim, tests / shim)

    # The layer chain on both builds.
    chain = bundle(
        "pdl-chain",
        f"One decode layer's kernel chain (20 kernels at TP4's shapes: mHC entries with their MXFP8 casts, "
        f"Q-A/KV, q norm, Q-B, WO-A/B, shared expert on its own stream, routed experts, combine) in CUDA graphs "
        f"at 6, 24 and 48 rows, on next-candidate's build (vLLM {base} {base_commit}) and on it with PDL "
        f"(vLLM {head} {commit}), both as package files over the -next-6 image's, three rounds alternating: "
        "time per layer warm and L2-cold, and every written buffer's bits compared.",
        ["/b/pdl_chain_ab.py", "3"],
        [["pdl_chain_ab.py", "/b/pdl_chain_ab.py"]],
        "status",
    )
    both_builds = (
        [[source(f, "base"), f"/b/base/{f}"] for f in base_package]
        + [[source(f), f"/b/pdl/{f}"] for f in pdl_package]
        + into_tree(pdl_extra)
    )
    candidate = json.loads((chain / "candidate.json").read_text())
    candidate["mounts"] += both_builds
    write_json(chain / "candidate.json", candidate)
    shutil.copyfile(HERE / "pdl_chain_ab.py", chain / "pdl_chain_ab.py")

    # The chain at every row count the engine grid judges, with the shared expert beside
    # the routed experts and inline.
    by_rows = bundle(
        "pdl-rows",
        f"The decode layer's chain at the engine grid's rows (2-96), the shared expert on its side stream and "
        f"inline, on next-candidate's build (vLLM {base} {base_commit}) and on vLLM {head} {commit}, five rounds "
        "alternating (pdl_chain_ab.py running pdl_rows.py): time per layer warm and L2-cold, and every written "
        "buffer's bits compared.",
        ["/b/pdl_chain_ab.py", "5", "/b/pdl_rows.py"],
        [["pdl_chain_ab.py", "/b/pdl_chain_ab.py"], ["pdl_rows.py", "/b/pdl_rows.py"]],
        "status",
    )
    candidate = json.loads((by_rows / "candidate.json").read_text())
    candidate["mounts"] += both_builds
    write_json(by_rows / "candidate.json", candidate)
    for script in ("pdl_chain_ab.py", "pdl_rows.py"):
        shutil.copyfile(HERE / script, by_rows / script)

    # The row counts where PDL was level or slower, with the power to see 0.4%: a fresh
    # process a build and round sets the larger chains' time to about +-1%.
    focus = bundle(
        "pdl-rows-focus",
        f"pdl-rows at 6, 16, 32, 48 and 72 rows, the shared expert on its side stream, fifteen rounds "
        f"alternating next-candidate's build (vLLM {base} {base_commit}) and vLLM {head} {commit}: the "
        "per-process spread at the larger row counts is about 1%, so fifteen rounds resolve a 0.4% "
        "difference.",
        [
            "/b/pdl_chain_ab.py",
            "15",
            "/b/pdl_rows.py",
            "--rows",
            "6,16,32,48,72",
            "--modes",
            "aux",
        ],
        [["pdl_chain_ab.py", "/b/pdl_chain_ab.py"], ["pdl_rows.py", "/b/pdl_rows.py"]],
        "status",
    )
    candidate = json.loads((focus / "candidate.json").read_text())
    candidate["mounts"] += both_builds
    write_json(focus / "candidate.json", candidate)
    for script in ("pdl_chain_ab.py", "pdl_rows.py"):
        shutil.copyfile(HERE / script, focus / script)

    # Kernel by kernel: the 16- and 48-row chains under the torch profiler.
    traced = bundle(
        "pdl-trace",
        f"The layer chain's CUDA graph at 16 and 48 rows replayed under the torch profiler on next-candidate's "
        f"build (vLLM {base} {base_commit}) and on vLLM {head} {commit}, two rounds alternating "
        "(pdl_chain_ab.py running pdl_trace.py): each kernel's median start, end and duration in a replay.",
        ["/b/pdl_chain_ab.py", "2", "/b/pdl_trace.py", "--rows", "16,48"],
        [
            ["pdl_chain_ab.py", "/b/pdl_chain_ab.py"],
            ["pdl_trace.py", "/b/pdl_trace.py"],
        ],
        "status",
    )
    candidate = json.loads((traced / "candidate.json").read_text())
    candidate["mounts"] += both_builds
    write_json(traced / "candidate.json", candidate)
    for script in ("pdl_chain_ab.py", "pdl_trace.py"):
        shutil.copyfile(HERE / script, traced / script)

    # Each variant's own run of the PDL and kernel tests with the source check.
    for name, ref in variants:
        variant_tests = HERE / "bundles" / f"pdl-tests-{name}"
        shutil.rmtree(variant_tests, ignore_errors=True)
        shutil.copytree(tests, variant_tests)
        candidate = json.loads((variant_tests / "candidate.json").read_text())
        candidate["description"] = candidate["description"].replace(
            f"vLLM {head} {commit}", f"vLLM {ref} {rev(repo, ref)}"
        )
        candidate["mounts"] = [
            [f"{REL}/overlay-{name}/{f}", target] if f in owns[name] else [src, target]
            for (src, target), f in (
                (m, m[1].removeprefix(f"{ROOT}/")) for m in candidate["mounts"]
            )
        ]
        write_json(variant_tests / "candidate.json", candidate)

    # The fix candidates beside the base and PDL builds.
    if variants:
        builds = (
            f"next-candidate (vLLM {base} {base_commit}), vLLM {head} {commit}, "
            + ", ".join(variant_notes)
        )
        for name, description, argv in (
            ("pdl-fix", "The decode layer's chain at the engine grid's rows (2-96), the shared expert on its "
             "side stream, ten rounds rotating", ["/b/pdl_chain_ab.py", "10", "/b/pdl_rows.py", "--modes", "aux"]),
            ("pdl-fix-trace", "The 16- and 48-row chains under the torch profiler, two rounds rotating",
             ["/b/pdl_chain_ab.py", "2", "/b/pdl_trace.py", "--rows", "16,48"]),
            ("pdl-fix-trace-small", "The 3- and 6-row chains under the torch profiler, three rounds rotating",
             ["/b/pdl_chain_ab.py", "3", "/b/pdl_trace.py", "--rows", "3,6"]),
            ("pdl-fix-grid", "The decode layer's chain at the engine grid's rows (2-96), the shared expert on its "
             "side stream, fifteen rounds rotating", ["/b/pdl_chain_ab.py", "15", "/b/pdl_rows.py", "--modes", "aux"]),
            ("pdl-fix-focus", "The decode layer's chain at 2, 3, 6, 16, 48 and 96 rows, the shared expert on its "
             "side stream, fifteen rounds rotating",
             ["/b/pdl_chain_ab.py", "15", "/b/pdl_rows.py", "--rows", "2,3,6,16,48,96", "--modes", "aux"]),
        ):  # fmt: skip
            script = argv[2].removeprefix("/b/")
            fixed = bundle(
                name,
                f"{description} over {builds}: time per layer (or each kernel's place in a replay) and every "
                "written buffer's bits compared.",
                argv,
                [["pdl_chain_ab.py", "/b/pdl_chain_ab.py"], [script, f"/b/{script}"]],
                "status",
            )
            candidate = json.loads((fixed / "candidate.json").read_text())
            candidate["mounts"] += both_builds + variant_mounts
            write_json(fixed / "candidate.json", candidate)
            for each in ("pdl_chain_ab.py", script):
                shutil.copyfile(HERE / each, fixed / each)

    # The chain kernels' sources, lowered at their decode shapes.
    sources = bundle(
        "pdl-sources",
        f"The PDL chain kernels' CUDA sources (vLLM {head} {commit}), lowered for sm_121 at TP4 decode shapes "
        "with each factory's pass configs and no device compilation, each checked as the source-check plugin "
        "checks launched kernels: the wait outside any block before any pointer parameter, the trigger after "
        "it, no PDL in warp-specialized GEMMs.",
        ["/b/sources_cpu.py"],
        [["sources_cpu.py", "/b/sources_cpu.py"]],
        "status",
    )
    candidate = json.loads((sources / "candidate.json").read_text())
    candidate["mounts"] += into_tree(package + pdl_extra)
    write_json(sources / "candidate.json", candidate)
    shutil.copyfile(HERE / "sources_cpu.py", sources / "sources_cpu.py")

    # The synthetic mechanism and the traced layer.
    mechanism = bundle(
        "pdl-mechanism",
        "PDL on GB10 in CUDA graphs with synthetic dependent kernels: per-boundary time in four launch modes "
        "(plain, wait only, wait then trigger, trigger after the store) at 1-192 CTAs, a fork/join like "
        "moe_combine's, and one traced c1 decode layer's critical path (layer-c1.json) replayed plain and chained.",
        [
            "benchmarks/kernels/benchmark_deepseek_v41_pdl_mechanism.py",
            "--layer",
            "/b/layer-c1.json",
        ],
        [["layer-c1.json", "/b/layer-c1.json"]],
        "status",
    )
    candidate = json.loads((mechanism / "candidate.json").read_text())
    candidate["mounts"] += into_tree(package + pdl_extra)
    write_json(mechanism / "candidate.json", candidate)
    shutil.copyfile(HERE / "layer-c1.json", mechanism / "layer-c1.json")

    # Serving arms: candidate 6's recipe with each build's package files over the image's.
    for arm, ref, ref_commit, paths, what in (
        ("pdl", head, commit, package, "next-candidate with PDL along the decode chain"),
        ("base", base, base_commit, base_package, "next-candidate without PDL"),
    ):  # fmt: skip
        mounts = [["{home}/projects/spark-ds41f/cache", "/cache", "rw"]] + [
            [f"{{home}}/projects/spark-ds41f/{source(f, arm)}", f"{ROOT}/{f}", "ro"]
            for f in paths
        ]
        write_json(
            HERE / f"{arm}-tp4.json",
            {
                "notes": f"Candidate 6's tp4-ring recipe on {what}: vLLM {ref} {ref_commit}'s {len(paths)} "
                f"package files over the image's ({image_tree}'s), mounted from overlay/ (and "
                "overlay-base/ for the base build's version of a file PDL changes).",
                "extends": {"config": f"{RC6}/rc-tp4.json"},
                "set": {"container": {"mounts": mounts}},
            },
        )
        write_json(
            HERE / f"{arm}-tp4-dev.json",
            {
                "notes": f"{arm}-tp4.json with the dev-mode collective RPC, for the suites' engine grid.",
                "extends": {"config": f"{REL}/{arm}-tp4.json"},
                "set": {"environment": {"VLLM_SERVER_DEV_MODE": "1"}},
            },
        )
    print(
        "wrote overlay/, overlay-base/, bundles/pdl-tests, bundles/pdl-chain, bundles/pdl-rows, "
        "bundles/pdl-rows-focus, bundles/pdl-trace, "
        "bundles/pdl-sources, "
        "bundles/pdl-mechanism, "
        "pdl-tp4.json, pdl-tp4-dev.json, base-tp4.json, base-tp4-dev.json"
    )


if __name__ == "__main__":
    main()
