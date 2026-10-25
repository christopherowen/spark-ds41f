#!/usr/bin/env python3
"""Write the PDL experiment's kernel-lab bundles and serving arms from the vLLM branch.

usage: make_bundles.py [VLLM_REPO] [BASE] [HEAD]
       (default ~/projects/vllm-pdl 26382f5cf pdl)

BASE is the release tree candidate 6 serves (image -next-6 carries its package files);
HEAD is the PDL branch. Every package file the branch changes goes to overlay/vllm (the
serving arms mount it over the image) and into each bundle; the branch's new tests and
benchmarks are mounted into the image's tree.

- bundles/pdl-tests: the PDL tests, then the DS4.1 TileLang kernel tests of every
  changed module on the PDL build, all with the source check
  (tests.kernels.deepseek_v41_pdl_sources);
- bundles/pdl-chain: the layer-chain benchmark on the image's build and on the PDL
  build in alternation (pdl_chain_ab.py), bits compared;
- bundles/pdl-mechanism: the synthetic PDL benchmark and the traced layer
  (layer-c1.json, from critical_path.py --layer-sequence);
- pdl-tp4.json, pdl-tp4-dev.json: candidate 6's arms with the overlay mounted.
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
    "VLLM_DS41_KERNEL_BACKEND": "tilelang",
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


def changed(repo: str, base: str, head: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", repo, "diff", "--name-only", "--diff-filter=AM", base, head],
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


def main() -> None:
    repo = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "projects/vllm-pdl")
    base = sys.argv[2] if len(sys.argv) > 2 else "26382f5cf"
    head = sys.argv[3] if len(sys.argv) > 3 else "pdl"
    commit = subprocess.run(
        ["git", "-C", repo, "rev-parse", head],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    files = changed(repo, base, head)
    package = [f for f in files if f.startswith("vllm/")]
    extra = [f for f in files if not f.startswith("vllm/")]
    print(
        f"vLLM {head} {commit[:9]} over {base}: {len(package)} package files, {len(extra)} others"
    )

    shutil.rmtree(HERE / "overlay", ignore_errors=True)
    for f in package:
        copy(repo, head, f, HERE / "overlay" / f)

    def extras(path: Path) -> list:
        """The branch's tests and benchmarks, mounted into the image's tree."""
        mounts = []
        for f in extra:
            flat = f.replace("/", "_")
            copy(repo, head, f, path / flat)
            mounts.append([flat, f"{ROOT}/{f}"])
        return mounts

    def overlay(path: Path, under: str = "overlay") -> None:
        for f in package:
            copy(repo, head, f, path / under / f)

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
        f"PDL along the decode chain (vLLM {head} {commit[:9]} over {base}) on candidate 6's image: the PDL "
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
    mounts = extras(tests)
    candidate = json.loads((tests / "candidate.json").read_text())
    candidate["mounts"] += mounts
    candidate["io_uring"] = True
    write_json(tests / "candidate.json", candidate)
    overlay(tests)
    write_json(tests / "steps.json", steps)
    shutil.copyfile(HERE / "run_steps.py", tests / "run.py")
    for shim in ("tblib_stub.py", "offline_skip.py"):
        shutil.copyfile(RELEASE_BUNDLE / shim, tests / shim)

    # The layer chain on both builds.
    chain = bundle(
        "pdl-chain",
        f"One decode layer's kernel chain (23 kernels at TP4's shapes: mHC entries, Q-A/KV, q norm, Q-B, WO-A/B, "
        f"shared expert on its own stream, routed experts, combine) in CUDA graphs at 6, 24 and 48 rows, on "
        f"candidate 6's build and on it with the PDL branch's package files (vLLM {head} {commit[:9]}), three "
        "rounds alternating: time per layer warm and L2-cold, and every written buffer's bits compared.",
        ["/b/pdl_chain_ab.py", "3"],
        [["pdl_chain_ab.py", "/b/pdl_chain_ab.py"]]
        + [[f"pdl/{f}", f"/b/pdl/{f}"] for f in package],
        "status",
    )
    candidate = json.loads((chain / "candidate.json").read_text())
    candidate["mounts"] += extras(chain)
    write_json(chain / "candidate.json", candidate)
    overlay(chain, "pdl")
    shutil.copyfile(HERE / "pdl_chain_ab.py", chain / "pdl_chain_ab.py")

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
    candidate["mounts"] += extras(mechanism)
    write_json(mechanism / "candidate.json", candidate)
    overlay(mechanism)
    shutil.copyfile(HERE / "layer-c1.json", mechanism / "layer-c1.json")

    # Serving arms: candidate 6 with the overlay.
    overlay_mounts = [["{home}/projects/spark-ds41f/cache", "/cache", "rw"]] + [
        [f"{{home}}/projects/spark-ds41f/{REL}/overlay/{f}", f"{ROOT}/{f}", "ro"]
        for f in package
    ]
    write_json(
        HERE / "pdl-tp4.json",
        {
            "notes": f"Candidate 6 at tp4-ring with PDL along the decode chain: vLLM {head} {commit[:9]}'s "
            f"{len(package)} package files over the image's (26382f5cf's), mounted from overlay/.",
            "extends": {"config": f"{RC6}/rc-tp4.json"},
            "set": {"container": {"mounts": overlay_mounts}},
        },
    )
    write_json(
        HERE / "pdl-tp4-dev.json",
        {
            "notes": "pdl-tp4.json with the dev-mode collective RPC, for the suites' engine grid.",
            "extends": {"config": f"{REL}/pdl-tp4.json"},
            "set": {"environment": {"VLLM_SERVER_DEV_MODE": "1"}},
        },
    )
    print(
        "wrote overlay/, bundles/pdl-tests, bundles/pdl-chain, bundles/pdl-mechanism, pdl-tp4.json, "
        "pdl-tp4-dev.json"
    )


if __name__ == "__main__":
    main()
