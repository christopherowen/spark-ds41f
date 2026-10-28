#!/usr/bin/env python3
"""The cluster program's ready levers on candidate 10, one spec each: the port's tests, the
bit gates, the lean suite.

usage: make_levers.py FRONT_WORKTREE COPY_WORKTREE YIELD_WORKTREE

Each worktree is a vLLM checkout over candidate 10's head (58938a074) holding one lever's
commits cherry-picked unchanged; its vllm/ files become an overlay arm on candidate 10's
image and its tests the bundles' mounted tests.

  front  the decode attention front in one kernel (c1-front 8a70cb9a9, 0060836b4): q/kv
         norms, the Q-B cast, the kv RoPE and SWA record; always on.
  copy   copy drafts beside DSpark (copy-drafts 05d5db22d..63082e9b9), min match 16, 8 to
         16 drafts. Its control is candidate 10 at the copy recipe's ladder and budget, so
         the A/B isolates the drafter.
  yield  prefill yield (prefill-yield 4316d0fc6), `--prefill-yield short`, 10 s limit; the
         lean suite with #620's 131K stall probe.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDER = "experiments/2026-10-28-candidate-10-levers"
CANDIDATE = "experiments/2026-10-26-spark-ds41f-candidate/rc10"
BASE = f"{CANDIDATE}/rc-tp4.json"
LEAN = "experiments/2026-10-27-one-at-a-time/lean-suite.json"
STALLS = "experiments/2026-10-26-concurrency-stalls/stall-suite-tp4.json"
HEAD = "58938a074"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-10"
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


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def lever_files(repo: Path) -> tuple[list[str], list[str]]:
    """(vllm/ files, tests/ files) the worktree changes or adds over candidate 10's head."""
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


def candidate_args() -> list[str]:
    """Candidate 10's resolved serve_args at tp4-ring."""
    from importlib.machinery import SourceFileLoader
    from importlib.util import module_from_spec, spec_from_loader

    loader = SourceFileLoader("spark_cli", str(HERE.parents[1] / "bin" / "spark"))
    module = module_from_spec(spec_from_loader("spark_cli", loader))
    loader.exec_module(module)
    return list(module.read_cluster(BASE)["serve_args"])


def bundle(name: str, description: str, tests: list[str], repo: Path, sources: list[str],
           mounted: list[str], env: dict | None = None) -> str:
    """A kernel-lab bundle running ``tests`` on candidate 10's image with ``sources`` from
    ``repo`` as its overlay and ``mounted`` tests from ``repo`` over the image's."""
    target = HERE / "bundles" / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy(HERE / "prepare_fixtures.py", target / "prepare_fixtures.py")
    mounts = [["prepare_fixtures.py", "/b/prepare_fixtures.py"]]
    for path in sources:
        (target / "overlay" / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / path, target / "overlay" / path)
    for path in mounted:
        local = "tests-" + path.removeprefix("tests/").replace("/", "-")
        shutil.copy(repo / path, target / local)
        mounts.append([local, f"{ROOT}/{path}"])
    write_json(target / "candidate.json", {
        "description": description, "image": IMAGE,
        "env": {**ENV, **(env or {}), "PYTHONPATH": f"/b:{ROOT}:{SOURCE_ROOT}/b12x",
                "TILELANG_CACHE_DIR": f"/c/tilelang-{name}"},
        "source_root": SOURCE_ROOT, "workdir": "vllm",
        "argv": [*PYTEST, "--basetemp", f"/c/pytest-{name}", *tests], "verdict": "exit",
        "mounts": mounts,
    })
    return f"{FOLDER}/bundles/{name}"


def arm(label: str, notes: str, repo: Path | None = None, files: list[str] = (),
        serve_options: dict | None = None, serve_args: list[str] | None = None) -> str:
    """Candidate 10 with ``files`` from ``repo`` mounted over its tree and its serving
    changed, and the arm's dev twin for the engine grid. Returns the arm's path."""
    target = HERE / "overlay" / label
    if target.exists():
        shutil.rmtree(target)
    mounts = [["{checkout}/cache", "/cache", "rw"]]
    for path in files:
        (target / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / path, target / path)
        mounts.append([f"{{checkout}}/{FOLDER}/overlay/{label}/{path}", f"{ROOT}/{path}", "ro"])
    data = {"notes": f"Candidate 10 with {notes} (make_levers.py).", "extends": {"config": BASE}}
    sets = {}
    if files:
        sets["container"] = {"mounts": mounts}
    if serve_args is not None:
        sets["serve_args"] = serve_args
    if sets:
        data["set"] = sets
    if serve_options:
        data["serve_options"] = serve_options
    write_json(HERE / f"{label}-tp4.json", data)
    write_json(HERE / f"{label}-tp4-dev.json", {
        "notes": f"{label}-tp4.json with the dev-mode collective RPC, for the engine grid.",
        "extends": {"config": f"{FOLDER}/{label}-tp4.json"},
        "set": {"environment": {"VLLM_SERVER_DEV_MODE": "1"}},
    })
    return f"{FOLDER}/{label}-tp4.json"


def spec(label: str, base: str, base_label: str, kernels: list[dict], suite: str = LEAN,
         extras: list = ()) -> None:
    """Kernel bundles, then ``base`` and the arm through the bit gates (and ``extras``),
    then ``suite`` on ``base``, the arm and ``base`` again."""
    dev = base.replace("-tp4.json", "-tp4-dev.json")
    write_json(HERE / f"{label}.json", {
        "experiment": FOLDER,
        "run": f"c10-{label}",
        "jobs": [
            {"kind": "sync"},
            {"kind": "kernel", "bundles": kernels},
            {"kind": "measure", "bench": False, "bracket": False, "extras": [*GATES, *extras],
             "arms": [{"config": base, "label": base_label}, {"config": f"{FOLDER}/{label}-tp4.json", "label": label}]},
            {"kind": "measure", "suite": suite,
             "arms": [{"config": dev, "label": base_label},
                      {"config": f"{FOLDER}/{label}-tp4-dev.json", "label": label},
                      {"config": dev, "label": f"{base_label}-again"}]},
        ],
    })


def yield_suite() -> str:
    """The lean suite with the stall suite's 131K probe after its single stream."""
    lean = json.loads((HERE.parents[1] / LEAN).read_text())
    stalls = json.loads((HERE.parents[1] / STALLS).read_text())
    probe = next(step for step in stalls["steps"] if step["id"] == "stall-probe")
    steps = [step for step in lean["steps"] if step["id"] != "prefill"]
    steps += [probe] + [step for step in lean["steps"] if step["id"] == "prefill"]
    write_json(HERE / "yield-suite.json", {
        "name": "program-lean-stall",
        "version": 1,
        "description": lean["description"] + " Then the concurrency-stalls suite's 131K stall probe "
                       "(500-token probes 2-14 s into a cold 131K prefill beside 11 decoding streams, "
                       "five repeats) before the prefill step. Built for " + FOLDER + ".",
        "steps": steps,
    })
    return f"{FOLDER}/yield-suite.json"


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    front, copy, yields = (Path(a).resolve() for a in sys.argv[1:])
    args = candidate_args()
    for flag in ("--prefill-yield", "--prefill-lanes", "--prefill-compute-share", "--prefill-schedule-interval"):
        if flag in args:
            raise SystemExit(f"candidate 10 already passes {flag}")
    speculative = json.loads(args[args.index("--speculative-config") + 1])
    if "copy_drafts" in speculative:
        raise SystemExit("candidate 10 already drafts copies")

    front_files, front_tests = lever_files(front)
    front_bundle = bundle("front", "Candidate 10 with the attention front in one kernel: every output byte "
                          "against the five kernels, and the forward's routing.",
                          ["tests/kernels/test_deepseek_v41_tilelang_front.py",
                           "tests/models/test_deepseek_v4_1_attention.py"],
                          front, front_files, front_tests)
    arm("front", "the decode attention front in one kernel (q/kv norms, the Q-B cast, the kv RoPE and "
        "SWA record; vLLM c1-front 8a70cb9a9 and 0060836b4)", front, front_files)
    spec("front", BASE, "candidate-10", [{"bundle": front_bundle, "node": "dgx1"}])

    copy_files, copy_tests = lever_files(copy)
    copy_bundle = bundle("copy", "Candidate 10 with copy drafts: the scheduler, copy kernel and proposer, "
                         "adaptive verification and aligned-chunk tests.",
                         ["tests/v1/core/test_copy_drafts_scheduler.py", "tests/v1/spec_decode/test_copy_drafts.py",
                          "tests/v1/spec_decode/test_adaptive_verification.py",
                          "tests/v1/core/test_aligned_prefill_chunks.py"],
                         copy, copy_files, copy_tests)
    copy_wide = bundle("copy-wide", "Candidate 10 with copy drafts: TileLang verification at the copy "
                       "lookahead (16 steps), the ratio-2 compressor ring and kernel preparation.",
                       ["tests/v1/spec_decode/test_tilelang_rejection.py",
                        "tests/kernels/test_deepseek_v41_tilelang_compress.py",
                        "tests/v1/worker/test_prepare_kernels.py"],
                       copy, copy_files, copy_tests, {"VLLM_TEST_VERIFICATION_STEPS": "16"})
    control = arm("copy-control", "the copy recipe's ladder (107 rows) and budget (16395 tokens), "
                  "DSpark alone: the copy arm's control", serve_options=COPY_LADDER)
    arm("copy", "copy drafts beside DSpark (vLLM copy-drafts 05d5db22d..63082e9b9; min match 16, 8 to "
        "16 drafts) at the copy recipe's ladder and budget", copy, copy_files,
        serve_options={**COPY_LADDER, "--speculative-config": json.dumps(
            {**speculative, "copy_drafts": COPY_DRAFTS}, separators=(",", ":"))})
    spec("copy", control, "copy-control",
         [{"bundle": copy_bundle, "node": "dgx2"}, {"bundle": copy_wide, "node": "dgx3"}],
         extras=[COPY_PROBE])

    yield_files, yield_tests = lever_files(yields)
    yield_bundle = bundle("yield", "Candidate 10 with prefill yield: its scheduler tests and the aligned "
                          "chunks'.", ["tests/v1/core/test_prefill_yield.py",
                                       "tests/v1/core/test_aligned_prefill_chunks.py"],
                          yields, yield_files, yield_tests)
    arm("yield", "prefill yield (vLLM prefill-yield 4316d0fc6), `--prefill-yield short`, 10 s limit",
        yields, yield_files, serve_args=[*args, "--prefill-yield", "short", "--prefill-yield-limit-s", "10"])
    spec("yield", BASE, "candidate-10", [{"bundle": yield_bundle, "node": "dgx4"}], suite=yield_suite())

    print(f"front: {len(front_files)} files; copy: {len(copy_files)} files; yield: {len(yield_files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
