#!/usr/bin/env python3
"""Write the softmax fix's kernel-lab bundles, its serving arm and the serving check.

usage: make_bundles.py VLLM_REPO BASE HEAD [--candidate-arm ARM]

Every file the vLLM branch changes between BASE (spark-ds41f-1025's package tree,
tilelang-ports 26382f5cf, which image -next-6 holds with two test patches) and HEAD
goes into each bundle: package files under overlay/vllm (the lab mounts them over the
image's tree), tests and benchmarks as mounts. Each bundle also carries BASE's
tilelang_logprob.py and tilelang_rejection.py at /b/release, the baseline the gated
benchmarks time the fix against (--baseline /b/release), besides Triton's.

softmax-serving.json compares a candidate with spark-ds41f-1025 at tp4-ring
(experiments/2026-10-25-spark-ds41f-candidate/rc6/rc-tp4.json, labelled "release").
With ``--candidate-arm``, the candidate is that arm (a release candidate that carries
the fix); without it, softmax-tp4.json, which mounts the package files
(bundles/serving/overlay) over spark-ds41f-1025: the fix alone.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT = "experiments/2026-10-25-softmax-fix"
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
RELEASE_ARM = "experiments/2026-10-25-spark-ds41f-candidate/rc6/rc-tp4.json"
# The serving check's extras: the determinism gate with and without logprobs, the
# seeded temperature-1 digest, and the logprobs near 0 position by position.
EXTRAS = [
    ["experiments/2026-10-12-r6g/token_determinism_text.py",
     ["--tokens", "192", "--long-lines", "700", "--wide"], "determinism"],
    ["experiments/2026-10-12-r6g/token_determinism_text.py",
     ["--tokens", "192", "--long-lines", "700", "--wide", "--no-logprobs"], "greedy"],
    ["experiments/2026-10-24-tilelang-completion/sampled_digest.py", ["--tokens", "192"],
     "sampled"],
    ["logprob_margins.py", ["--tokens", "256"], "logprobs"],
]
FIXTURES = HERE.parent / "2026-10-24-tilelang-completion/bundles/sampling/prepare_fixtures.py"
RELEASE = {
    "release_tilelang_logprob.py": "vllm/v1/worker/gpu/sample/tilelang_logprob.py",
    "release_tilelang_rejection.py": "vllm/v1/worker/gpu/spec_decode/tilelang_rejection.py",
}

PYTEST = ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
          "-p", "prepare_fixtures"]

# name: (node, description, [(label, argv after python, timeout seconds)]).
PARTS = {
    "logprob": (
        "dgx1",
        "the logprob log-softmax (sampled tokens, top-k, prompt logprobs, verified "
        "tokens, the sink's alternates)",
        [
            ("logprob tests", PYTEST + [
                "--basetemp", "/c/pytest-softmax-logprob",
                "tests/v1/worker/test_gpu_tilelang_logprob_softmax.py",
                "tests/v1/worker/test_gpu_verified_logprob_token_ids.py",
                "tests/v1/worker/test_gpu_logprob_ranks.py",
                "tests/v1/worker/test_gpu_tilelang_topk_ids.py",
                "tests/v1/worker/test_prepare_kernels.py"], 3600),
            # The sink's alternates take their logprobs from the same launch.
            ("sink tests", PYTEST + [
                "--basetemp", "/c/pytest-softmax-sink",
                "tests/v1/worker/test_gpu_sink.py"], 3600),
            # tests/v1/sample/test_logprobs.py's kernel tests (against torch's
            # log-softmax, within 1e-5) import tests/conftest.py, which needs tblib,
            # absent from the serving image: the release tests run them.
            ("logprob softmax benchmark, against the release",
             ["benchmarks/kernels/benchmark_tilelang_logprob_softmax.py", "--gate",
              "--baseline", "/b/release"], 1800),
            ("logprob softmax benchmark, against Triton",
             ["benchmarks/kernels/benchmark_tilelang_logprob_softmax.py", "--gate"], 1800),
        ],
    ),
    "verification": (
        "dgx2",
        "block verification (block statistics, draft log probabilities, cumulative "
        "ratios and thresholds)",
        [
            ("verification tests", PYTEST + [
                "--basetemp", "/c/pytest-softmax-verification",
                "tests/v1/spec_decode/test_tilelang_rejection.py"], 5400),
            # Unbiasedness of block verification through rejection_sample (the TileLang
            # path for one-hot drafts), chunking, int64 offsets, adaptive verification.
            ("rejection sampler tests", PYTEST + [
                "--basetemp", "/c/pytest-softmax-rejection",
                "tests/v1/spec_decode/test_rejection_sampler_utils.py",
                "tests/v1/worker/test_gpu_rejection_sampler_chunking.py",
                "tests/v1/worker/test_gpu_rejection_sampler_i64.py",
                "tests/v1/spec_decode/test_adaptive_verification.py"], 3600),
            ("rejection-sampler benchmark, against the release",
             ["benchmarks/kernels/benchmark_tilelang_rejection.py", "--gate",
              "--baseline", "/b/release"], 1800),
            ("rejection-sampler benchmark, against Triton",
             ["benchmarks/kernels/benchmark_tilelang_rejection.py", "--gate"], 1800),
        ],
    ),
}

RUN = '''#!/usr/bin/env python3
"""Run this bundle's steps in order; each step's exit status is printed, and the bundle
fails if any step failed. Every step runs whatever came before it."""
import json
import os
import subprocess
import sys

STEPS = json.load(open("/b/steps.json"))
codes = []
for label, argv, timeout, *rest in STEPS:
    print(f"=== {label}", flush=True)
    env = dict(os.environ)
    for name, value in (rest[0] if rest else {}).items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    try:
        code = subprocess.run([sys.executable, *argv], timeout=timeout, env=env).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    print(f"=== {label}: exit {code}", flush=True)
    codes.append((label, code))
print(json.dumps({"steps": codes}), flush=True)
sys.exit(0 if all(code == 0 for _, code in codes) else 1)
'''


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def show(repo: str, rev: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", repo, "show", f"{rev}:{path}"], check=True,
                          capture_output=True).stdout


def write_overlay(repo: str, head: str, changed: list[str], bundle: Path) -> None:
    overlay = bundle / "overlay"
    if overlay.exists():
        shutil.rmtree(overlay)
    for path in changed:
        if path.startswith("vllm/"):
            target = overlay / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(show(repo, head, path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo")
    parser.add_argument("base")
    parser.add_argument("head")
    parser.add_argument("--candidate-arm", metavar="ARM",
                        help="a release candidate's arm that carries the fix")
    args = parser.parse_args()
    repo, base, head = args.repo, args.base, args.head
    head_commit = git(repo, "rev-parse", "--short=9", head).strip()
    base_commit = git(repo, "rev-parse", "--short=9", base).strip()
    status = git(repo, "diff", "--name-status", base, head).split("\n")
    changed = [line.split("\t")[-1] for line in status if line]
    assert all(line.split("\t")[0] in ("A", "M") for line in status if line), status

    for part, (node, what, steps) in PARTS.items():
        bundle = HERE / "bundles" / f"softmax-{part}"
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True)
        write_overlay(repo, head, changed, bundle)
        mounts = [["prepare_fixtures.py", "/b/prepare_fixtures.py"],
                  ["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"]]
        for path in changed:
            if not path.startswith("vllm/"):
                name = path.replace("/", "_")
                (bundle / name).write_bytes(show(repo, head, path))
                mounts.append([name, f"{ROOT}/{path}"])
        for name, path in RELEASE.items():
            (bundle / name).write_bytes(show(repo, base, path))
            mounts.append([name, f"/b/release/{Path(path).name}"])
        shutil.copy(FIXTURES, bundle / "prepare_fixtures.py")
        (bundle / "run.py").write_text(RUN)
        (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")
        candidate = {
            "description": (
                f"The softmax fix (vLLM branch softmax-fix {head_commit}, {len(changed)} "
                f"files over spark-ds41f-1025's {base_commit}) on image -next-6: "
                f"{what}; GPU tests against FP64 references, then the gated benchmarks "
                f"against the release's TileLang kernels and against Triton's."),
            "image": IMAGE,
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TILELANG_CACHE_DIR": f"/c/tilelang-softmax-{part}",
                "VLLM_DS41_KERNEL_BACKEND": "tilelang",
                "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x",
            },
            "workdir": "vllm",
            "argv": ["/b/run.py"],
            "verdict": "exit",
            "mounts": mounts,
        }
        (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
        print(f"{bundle.name} ({node}): {len(changed)} files, {len(steps)} steps")

    spec = {
        "experiment": EXPERIMENT,
        "run": "softmax-fix-kernels",
        "jobs": [
            {"kind": "sync"},
            {"kind": "kernel", "bundles": [
                {"node": node, "bundle": f"{EXPERIMENT}/bundles/softmax-{part}"}
                for part, (node, _, _) in PARTS.items()]},
        ],
    }
    (HERE / "softmax-kernels.json").write_text(json.dumps(spec, indent=1) + "\n")

    write_overlay(repo, head, changed, HERE / "bundles" / "serving")
    experiment = f"{{home}}/projects/spark-ds41f/{EXPERIMENT}"
    mounts = [["{home}/projects/spark-ds41f/cache", "/cache", "rw"]]
    mounts += [[f"{experiment}/bundles/serving/overlay/{path}", f"{ROOT}/{path}", "ro"]
               for path in changed if path.startswith("vllm/")]
    arm = {
        "notes": (f"The softmax fix (vLLM branch softmax-fix {head_commit}) mounted over "
                  "spark-ds41f-1025 (tilelang-ports 26382f5cf) at tp4-ring: logprobs by "
                  "Barron et al.'s Eq. 5, block verification from logs. Text at "
                  "temperature 0 must equal the release's; logprobs move by design."),
        "extends": {"config": RELEASE_ARM},
        "set": {"container": {"mounts": mounts}},
    }
    (HERE / "softmax-tp4.json").write_text(json.dumps(arm, indent=1) + "\n")
    print(f"serving: {len(mounts) - 1} package files of {head_commit}")

    # The candidate first (heat soak never counts against it), then the release.
    candidate_arm = (
        {"config": args.candidate_arm, "label": "candidate"} if args.candidate_arm
        else {"config": "softmax-tp4.json", "label": "softmax"})
    serving = {
        "experiment": EXPERIMENT,
        "run": "softmax-fix-serving",
        "jobs": [
            {"kind": "sync"},
            {"kind": "measure", "profile": "lean", "bench": False, "extras": EXTRAS,
             "arms": [candidate_arm, {"config": RELEASE_ARM, "label": "release"}]},
        ],
    }
    (HERE / "softmax-serving.json").write_text(json.dumps(serving, indent=1) + "\n")
    print(f"serving check: {candidate_arm['config']} against {RELEASE_ARM}")


if __name__ == "__main__":
    main()
