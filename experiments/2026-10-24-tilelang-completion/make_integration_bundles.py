#!/usr/bin/env python3
"""Write the kernel-lab bundles that run every TileLang port together.

usage: make_integration_bundles.py VLLM_REPO BASE HEAD [--overlay BUNDLE... | --serving]

Every file the vLLM branch changes between BASE (candidate 4's source, which image
-next-4 holds) and HEAD goes into each bundle: package files under overlay/vllm (the lab
mounts them over the image's tree), tests and benchmarks as mounts. Each bundle then runs
one share of the ports' GPU tests and gated benchmarks (bundles/integration-*), so each
family is tested with all the other ports in place. With ``--overlay``, only the package
files are written, as the overlay of each named bundle (a variant sweep that times the
branch's production kernels). With ``--serving``, the package files go to
bundles/serving/overlay and ports-tp4.json is the serving arm that mounts them over
candidate 4 (rc4/rc-tp4.json, image -next-4).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-4"

PYTEST = ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest",
          "-p", "prepare_fixtures"]

# name: (description, [(label, argv after python, timeout seconds)])
PARTS = {
    "runner": (
        "the V2 runner's input preparation and token counts",
        [
            ("runner and sink tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-runner",
                "tests/v1/worker/test_gpu_tilelang_inputs.py",
                "tests/v1/worker/test_gpu_block_table.py",
                "tests/v1/worker/test_kv_block_zeroer.py",
                "tests/v1/worker/test_dead_rows.py",
                "tests/v1/worker/test_gpu_input_batch_v2.py",
                "tests/v1/worker/test_gpu_sink.py",
                "tests/v1/spec_decode/test_dflash_prepare_inputs.py",
                "tests/v1/worker/test_prepare_kernels.py"], 3600),
            ("runner benchmark", ["benchmarks/kernels/benchmark_runner_inputs_tilelang.py",
                                  "--gate"], 3600),
            ("token-count benchmark", ["benchmarks/kernels/benchmark_token_counts.py", "--gate",
                                       "--seed", "61027"], 1200),
        ],
    ),
    "sampling": (
        "the sampling family: Gumbel, temperature, sink detection, greedy reductions",
        [
            ("sampling tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-sampling",
                "tests/v1/worker/test_gpu_tilelang_sampling.py",
                "tests/v1/worker/test_gpu_gumbel_sample.py",
                "tests/v1/worker/test_gpu_sink.py",
                "tests/v1/worker/test_dspark_greedy_markov.py",
                "tests/v1/worker/gpu/test_greedy_verification.py",
                "tests/v1/worker/test_gpu_sampler_flags.py",
                "tests/v1/worker/test_prepare_kernels.py"], 5400),
            ("sampling benchmark", ["benchmarks/kernels/benchmark_tilelang_sampling.py",
                                    "--gate"], 3600),
        ],
    ),
    "mhc": (
        "the mHC post and stream collapse",
        [
            ("mHC tests", PYTEST + ["--basetemp", "/c/pytest-integration-mhc",
                                    "tests/kernels/test_deepseek_v41_tilelang_mhc.py"], 3600),
            ("mHC benchmark", ["benchmarks/kernels/benchmark_deepseek_v41_mhc_post.py",
                               "--gate"], 3600),
        ],
    ),
    "misc": (
        "the embedding, DSpark's confidence head, logprob ranks and the resampled token",
        [
            ("embedding, confidence, ranks and verification tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-misc",
                "tests/kernels/test_deepseek_v41_tilelang_embedding.py",
                "tests/kernels/test_deepseek_v41_tilelang_confidence.py",
                "tests/v1/worker/test_gpu_logprob_ranks.py",
                "tests/v1/spec_decode/test_rejection_sampler_utils.py"], 3600),
            # The unchanged image fails eight of these (vocabulary-parallel draft seeds;
            # experiments/2026-10-24-resampled-token): their own step, so they are told
            # apart from the ports'.
            ("vocabulary-parallel drafting tests", PYTEST + [
                "--basetemp", "/c/pytest-integration-vocab-parallel",
                "tests/v1/spec_decode/test_vocab_parallel_drafting.py"], 1800),
            ("embedding and confidence benchmark",
             ["benchmarks/kernels/benchmark_deepseek_v41_tilelang_misc.py", "--gate"], 3600),
            ("logprob-rank benchmark", ["benchmarks/kernels/benchmark_logprob_ranks.py",
                                        "--gate", "--seed", "40391"], 1200),
            ("resampled-token benchmark", ["benchmarks/kernels/benchmark_resampled_token.py",
                                           "--gate", "--seed", "52807"], 600),
            ("resampled-token benchmark, mixed temperatures",
             ["benchmarks/kernels/benchmark_resampled_token.py", "--gate", "--seed", "18443",
              "--mixed"], 600),
        ],
    ),
}

RUN = '''#!/usr/bin/env python3
"""Run this bundle's steps in order; each step's exit status is printed, and the bundle
fails if any step failed. Every step runs whatever came before it."""
import json
import subprocess
import sys

STEPS = json.load(open("/b/steps.json"))
codes = []
for label, argv, timeout in STEPS:
    print(f"=== {label}", flush=True)
    try:
        code = subprocess.run([sys.executable, *argv], timeout=timeout).returncode
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


def write_overlay(repo: str, head: str, changed: list[str], bundle: Path) -> None:
    overlay = bundle / "overlay"
    if overlay.exists():
        shutil.rmtree(overlay)
    for path in changed:
        if path.startswith("vllm/"):
            target = overlay / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.run(["git", "-C", repo, "show", f"{head}:{path}"],
                                              check=True, capture_output=True).stdout)


def main() -> None:
    repo, base, head = sys.argv[1:4]
    head_commit = git(repo, "rev-parse", "--short=9", head).strip()
    status = git(repo, "diff", "--name-status", base, head).split("\n")
    changed = [line.split("\t")[-1] for line in status if line]
    assert all(line.split("\t")[0] in ("A", "M") for line in status if line), status
    if sys.argv[4:5] == ["--serving"]:
        write_overlay(repo, head, changed, HERE / "bundles" / "serving")
        experiment = "{home}/projects/spark-ds41f/experiments/2026-10-24-tilelang-completion"
        mounts = [["{home}/projects/spark-ds41f/cache", "/cache", "rw"]]
        mounts += [[f"{experiment}/bundles/serving/overlay/{path}", f"{ROOT}/{path}", "ro"]
                   for path in changed if path.startswith("vllm/")]
        arm = {
            "notes": (f"Every TileLang port (vLLM branch tilelang-ports {head_commit}) mounted "
                      "over candidate 4 at tp4-ring: the serving check (outputs identical to "
                      "candidate 4 at temperature 0 and seeded temperature 1, no undeclared "
                      "kernel)."),
            "extends": {"config": "experiments/2026-10-23-next-release/rc4/rc-tp4.json"},
            "set": {"container": {"mounts": mounts}},
        }
        (HERE / "ports-tp4.json").write_text(json.dumps(arm, indent=1) + "\n")
        print(f"serving: {len(mounts) - 1} package files of {head_commit}")
        return
    if sys.argv[4:5] == ["--overlay"]:
        for name in sys.argv[5:]:
            write_overlay(repo, head, changed, HERE / "bundles" / name)
            print(f"{name}: overlay of {head_commit}")
        return
    fixtures = HERE / "bundles/sampling/prepare_fixtures.py"
    for part, (what, steps) in PARTS.items():
        bundle = HERE / "bundles" / f"integration-{part}"
        if bundle.exists():
            shutil.rmtree(bundle)
        (bundle / "overlay").mkdir(parents=True)
        mounts = [["prepare_fixtures.py", "/b/prepare_fixtures.py"],
                  ["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"]]
        for path in changed:
            content = subprocess.run(["git", "-C", repo, "show", f"{head}:{path}"], check=True,
                                     capture_output=True).stdout
            if path.startswith("vllm/"):
                target = bundle / "overlay" / path
            else:
                name = path.replace("/", "_")
                target = bundle / name
                mounts.append([name, f"{ROOT}/{path}"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        shutil.copy(fixtures, bundle / "prepare_fixtures.py")
        (bundle / "run.py").write_text(RUN)
        (bundle / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")
        candidate = {
            "description": (
                f"Every TileLang port together (vLLM branch tilelang-ports {head_commit}, "
                f"{len(changed)} files over candidate 4's source) on image -next-4: {what}, "
                "GPU tests then the gated benchmarks against the kernels the ports replace."),
            "image": IMAGE,
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TILELANG_CACHE_DIR": f"/c/tilelang-integration-{part}",
                "VLLM_DS41_KERNEL_BACKEND": "tilelang",
                "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm:/opt/spark-ds41f/candidate/b12x",
            },
            "workdir": "vllm",
            "argv": ["/b/run.py"],
            "verdict": "exit",
            "mounts": mounts,
        }
        (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
        print(f"{bundle.name}: {len(changed)} files, {len(steps)} steps")


if __name__ == "__main__":
    main()
