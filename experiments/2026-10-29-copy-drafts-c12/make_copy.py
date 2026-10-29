#!/usr/bin/env python3
"""Copy drafts on candidate 12, fixed: the release tests with copy drafts in, the bit gates,
the copy probe and the lean suite.

usage: make_copy.py COPY_WORKTREE

COPY_WORKTREE is a vLLM checkout over candidate 12's head (d984b2e66) holding the copy
commits candidate 12 left out (candidate 11's 0565-0570) and their fixes; its vllm/ files
become an overlay arm on candidate 12's image and, with its tests, the bundles' overlay.

Bundles: the copy tests (dgx2), TileLang verification at the copy lookahead (dgx3), and the
release gate with copy drafts' two test files back in (dgx4), the gate whose 40 copy failures
took copy drafts out of candidate 12. The arms: copy-control (candidate 12 at the copy
recipe's ladder and budget, DSpark alone) and copy (the same with copy drafts: min match 16,
8 to 16 drafts), through the bit gates and the copy probe, then the lean suite on
copy-control, copy and copy-control again.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FOLDER = "experiments/2026-10-29-copy-drafts-c12"
CANDIDATE = "experiments/2026-10-26-spark-ds41f-candidate/rc12"
BASE = f"{CANDIDATE}/rc-tp4.json"
LEAN = "experiments/2026-10-27-one-at-a-time/lean-suite.json"
HEAD = "d984b2e66"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-12"
SOURCE_ROOT = "/opt/spark-ds41f/candidate"
ROOT = f"{SOURCE_ROOT}/vllm"
ENV = {"HF_HUB_OFFLINE": "1", "VLLM_DS41_KERNEL_BACKEND": "tilelang"}
PYTEST = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "-p", "prepare_fixtures", "-rfEs"]
GATES = [
    ["experiments/2026-10-12-r6g/token_determinism_text.py",
     ["--tokens", "192", "--long-lines", "700", "--wide"], "determinism"],
    ["experiments/2026-10-12-r6g/token_determinism_text.py",
     ["--tokens", "192", "--long-lines", "700", "--wide", "--no-logprobs"], "greedy"],
    ["experiments/2026-10-24-tilelang-completion/sampled_digest.py", ["--tokens", "192"], "sampled"],
]
COPY_PROBE = ["experiments/2026-10-23-next-release/acceptance/copy_probe.py", ["--prompts", "12"], "copy-probe"]
COPY_LADDER = {
    "--max-num-batched-tokens": "16395",
    "--max-cudagraph-capture-size": "107",
    "--compilation-config": json.dumps(
        {"cudagraph_mode": "FULL_AND_PIECEWISE", "custom_ops": ["all"],
         "cudagraph_capture_sizes": [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 107]},
        separators=(",", ":")),
}
COPY_DRAFTS = {"min_match": 16, "initial_drafts": 8, "max_drafts": 16}
COPY_TESTS = {
    "tests/v1/core/test_copy_drafts_scheduler.py":
        "copy drafts in the scheduler: placeholders widened within the row envelope, nothing "
        "shared between requests (changed by the series)",
    "tests/v1/spec_decode/test_copy_drafts.py":
        "copy drafts: the copy_match and copy_select kernels and the proposer, per request "
        "(changed by the series)",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def copy_files(repo: Path) -> tuple[list[str], list[str]]:
    """(vllm/ files, tests/ files) the worktree changes or adds over candidate 12's head."""
    if git(repo, "merge-base", HEAD, "HEAD").strip() != git(repo, "rev-parse", HEAD).strip():
        raise SystemExit(f"{repo}: HEAD is not over {HEAD}")
    if git(repo, "status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit(f"{repo} has local changes")
    paths = []
    for line in filter(None, git(repo, "diff", "--name-status", HEAD, "HEAD").splitlines()):
        status, path = line.split("\t", 1)
        if status not in ("M", "A"):
            raise SystemExit(f"{repo}: {line!r}: an overlay can only replace or add files")
        paths.append(path)
    return ([p for p in paths if p.startswith("vllm/") and p.endswith(".py")],
            [p for p in paths if p.startswith("tests/") and p.endswith(".py")])


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def overlay_mounts(target: Path, repo: Path, sources: list[str], tests: list[str]) -> list[list[str]]:
    """Copy ``sources`` and ``tests`` from ``repo`` into the bundle; their mounts over the
    image's tree."""
    mounts = []
    for path in sources:
        (target / "overlay" / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / path, target / "overlay" / path)
        mounts.append([f"overlay/{path}", f"{ROOT}/{path}"])
    for path in tests:
        local = "tests-" + path.removeprefix("tests/").replace("/", "-")
        shutil.copy(repo / path, target / local)
        mounts.append([local, f"{ROOT}/{path}"])
    return mounts


def bundle(name: str, description: str, tests: list[str], repo: Path, sources: list[str],
           mounted: list[str], env: dict | None = None) -> str:
    """A kernel-lab bundle running ``tests`` on candidate 12's image with the copy files."""
    target = HERE / "bundles" / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy(HERE / "prepare_fixtures.py", target / "prepare_fixtures.py")
    mounts = [["prepare_fixtures.py", "/b/prepare_fixtures.py"],
              *overlay_mounts(target, repo, sources, mounted)]
    write_json(target / "candidate.json", {
        "description": description, "image": IMAGE,
        "env": {**ENV, **(env or {}), "PYTHONPATH": f"/b:{ROOT}:{SOURCE_ROOT}/b12x",
                "TILELANG_CACHE_DIR": f"/c/tilelang-{name}"},
        "source_root": SOURCE_ROOT, "workdir": "vllm",
        "argv": [*PYTEST, "--basetemp", f"/c/pytest-{name}", *tests], "verdict": "exit",
        "mounts": mounts,
    })
    return f"{FOLDER}/bundles/{name}"


def release_tests(repo: Path, sources: list[str], mounted: list[str]) -> str:
    """Candidate 12's release gate with the copy test files back in, on the copy overlay."""
    config = json.loads((REPO / "config/release-tests.json").read_text())
    config["gate"] = dict(sorted({**config["gate"], **COPY_TESTS}.items()))
    config_path = HERE / "release-tests-copy.json"
    write_json(config_path, config)
    target = HERE / "bundles" / "release-tests-copy"
    if target.exists():
        shutil.rmtree(target)
    script = str(REPO / "scripts/release_tests.py")
    manifest = HERE / "release-tests-copy-manifest.json"
    subprocess.run([sys.executable, script, "gate", str(repo), "HEAD", str(manifest), str(config_path)], check=True)
    subprocess.run([sys.executable, script, "bundle", str(manifest), IMAGE, str(target), "copy-drafts-c12"],
                   check=True)
    candidate = json.loads((target / "candidate.json").read_text())
    candidate["description"] = ("Release tests (copy-drafts-c12): candidate 12's gate with copy drafts' "
                                "test files back in, on the copy overlay; per-test outcomes under "
                                "/c/release-tests.")
    candidate["env"]["PYTHONPATH"] = f"/b:{ROOT}"
    candidate["mounts"] += overlay_mounts(target, repo, sources, mounted)
    write_json(target / "candidate.json", candidate)
    return f"{FOLDER}/bundles/release-tests-copy"


def candidate_args() -> list[str]:
    """Candidate 12's resolved serve_args at tp4-ring."""
    from importlib.machinery import SourceFileLoader
    from importlib.util import module_from_spec, spec_from_loader

    loader = SourceFileLoader("spark_cli", str(REPO / "bin" / "spark"))
    module = module_from_spec(spec_from_loader("spark_cli", loader))
    loader.exec_module(module)
    return list(module.read_cluster(BASE)["serve_args"])


def arm(label: str, notes: str, repo: Path | None = None, files: list[str] = (),
        serve_options: dict | None = None) -> str:
    """Candidate 12 with ``files`` from ``repo`` mounted over its tree and its serving
    changed, and the arm's dev twin for the engine grid. Returns the arm's path."""
    target = HERE / "overlay" / label
    if target.exists():
        shutil.rmtree(target)
    mounts = [["{checkout}/cache", "/cache", "rw"]]
    for path in files:
        (target / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / path, target / path)
        mounts.append([f"{{checkout}}/{FOLDER}/overlay/{label}/{path}", f"{ROOT}/{path}", "ro"])
    data = {"notes": f"Candidate 12 with {notes} (make_copy.py).", "extends": {"config": BASE}}
    if files:
        data["set"] = {"container": {"mounts": mounts}}
    if serve_options:
        data["serve_options"] = serve_options
    write_json(HERE / f"{label}-tp4.json", data)
    write_json(HERE / f"{label}-tp4-dev.json", {
        "notes": f"{label}-tp4.json with the dev-mode collective RPC, for the engine grid.",
        "extends": {"config": f"{FOLDER}/{label}-tp4.json"},
        "set": {"environment": {"VLLM_SERVER_DEV_MODE": "1"}},
    })
    return f"{FOLDER}/{label}-tp4.json"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    repo = Path(sys.argv[1]).resolve()
    args = candidate_args()
    speculative = json.loads(args[args.index("--speculative-config") + 1])
    if "copy_drafts" in speculative:
        raise SystemExit("candidate 12 already drafts copies")
    sources, tests = copy_files(repo)
    head = git(repo, "rev-parse", "--short=9", "HEAD").strip()

    copy_bundle = bundle("copy", "Candidate 12 with copy drafts: the scheduler, copy kernel and proposer, "
                         "adaptive verification, aligned-chunk and attention-layer tests.",
                         ["tests/v1/core/test_copy_drafts_scheduler.py", "tests/v1/spec_decode/test_copy_drafts.py",
                          "tests/v1/spec_decode/test_adaptive_verification.py",
                          "tests/v1/core/test_aligned_prefill_chunks.py",
                          "tests/models/test_deepseek_v4_1_attention.py"],
                         repo, sources, tests)
    copy_wide = bundle("copy-wide", "Candidate 12 with copy drafts: TileLang verification at the copy "
                       "lookahead (16 steps), the ratio-2 compressor ring and kernel preparation.",
                       ["tests/v1/spec_decode/test_tilelang_rejection.py",
                        "tests/kernels/test_deepseek_v41_tilelang_compress.py",
                        "tests/v1/worker/test_prepare_kernels.py"],
                       repo, sources, tests, {"VLLM_TEST_VERIFICATION_STEPS": "16"})
    gate = release_tests(repo, sources, tests)

    control = arm("copy-control", "the copy recipe's ladder (107 rows) and budget (16395 tokens), "
                  "DSpark alone: the copy arm's control", serve_options=COPY_LADDER)
    arm("copy", f"copy drafts beside DSpark (vLLM copy-drafts-c12 {head}; min match 16, 8 to 16 drafts) "
        "at the copy recipe's ladder and budget", repo, sources,
        serve_options={**COPY_LADDER, "--speculative-config": json.dumps(
            {**speculative, "copy_drafts": COPY_DRAFTS}, separators=(",", ":"))})
    dev = control.replace("-tp4.json", "-tp4-dev.json")
    write_json(HERE / "copy.json", {
        "experiment": FOLDER,
        "run": "c12-copy",
        "jobs": [
            {"kind": "sync"},
            {"kind": "kernel", "bundles": [{"bundle": copy_bundle, "node": "dgx2"},
                                           {"bundle": copy_wide, "node": "dgx3"},
                                           {"bundle": gate, "node": "dgx4"}]},
            {"kind": "measure", "bench": False, "bracket": False, "extras": [*GATES, COPY_PROBE],
             "arms": [{"config": control, "label": "copy-control"},
                      {"config": f"{FOLDER}/copy-tp4.json", "label": "copy"}]},
            {"kind": "measure", "suite": LEAN,
             "arms": [{"config": dev, "label": "copy-control"},
                      {"config": f"{FOLDER}/copy-tp4-dev.json", "label": "copy"},
                      {"config": dev, "label": "copy-control-again"}]},
        ],
    })
    print(f"copy: {len(sources)} vllm files, {len(tests)} test files at {head}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
