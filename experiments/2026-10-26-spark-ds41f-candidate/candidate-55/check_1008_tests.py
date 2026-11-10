#!/usr/bin/env python3
"""#1008's own tests and its in-process bench in one container of the candidate image: the runner of
bundles/mark-1008-boot1 and -boot2 (kernel-lab bundles, verdict "status"), as the PR ran them.

usage (in the container): check_1008_tests.py --label boot<k> --sparknet DIR --sparknet-head SHA
                          --sparknet-tree SHA --bench PATH --out DIR

1. The sparknet tree at --sparknet must be the candidate's prepared one: HEAD --sparknet-head and
   tree --sparknet-tree (candidate-55's lock; the image's sparknet label is that tree), and clean.
2. sparknet's unit tests, as the PR ran them: PYTHONPATH=<tree>/src python3 -m unittest discover
   -s <tree>/tests -v, from the image's working directory. They import the image's sparknet and read
   the tree's _roce_proxy.c and simulate.c, so the tree is mounted. The PR's run: 97 tests, OK.
3. vLLM's, in the image's vLLM tree: python3 -m pytest --noconftest -v
   tests/v1/worker/test_roce_cpu_affinity.py tests/kernels/test_deepseek_v41_engram_host_placement.py
   (the PR's run: 14 passed)
4. The PR's bench_roce_core_affinity.py --boot-label <label> --warmup-steps 60 --bench-steps 500
   --output <out>/bench-<label>.json: its three placements simulated in one process. It is the PR's
   own comparison, kept as it ran; it does not compare arms (check_1008_bench.py does, on the ring).

Each step's output is printed and saved in --out (<step>-<label>.txt); a closing JSON line lists
the steps. Exits 0 only when every step passes.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

IMAGE_WORKDIR = "/vllm-workspace"  # the image's WorkingDir, where the PR's docker run started
VLLM_TESTS = ["tests/v1/worker/test_roce_cpu_affinity.py", "tests/kernels/test_deepseek_v41_engram_host_placement.py"]


def run(name: str, argv: list[str], out: Path, label: str, env: dict | None = None, cwd: str | None = None) -> dict:
    started = time.time()
    process = subprocess.run(argv, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / f"{name}-{label}.txt").write_text(process.stdout)
    print(f"===== {name} ({shlex.join(argv)}): exit {process.returncode}", flush=True)
    print(process.stdout, flush=True)
    return {"step": name, "exit": process.returncode, "seconds": round(time.time() - started, 1),
            "output": str(out / f"{name}-{label}.txt"), "tail": process.stdout.strip().splitlines()[-1:]}


def git_ref(tree: Path) -> tuple[str | None, str | None, str | None]:
    """HEAD, its tree and the porcelain status by git when the image has it, else HEAD from .git."""
    if shutil.which("git"):
        def git(*args: str) -> str:
            return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(tree), *args], text=True,
                                  capture_output=True).stdout.strip()
        return git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}"), git("status", "--porcelain")
    head = (tree / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head[5:]
        path = tree / ".git" / ref
        if path.exists():
            head = path.read_text().strip()
        else:
            packed = (tree / ".git" / "packed-refs").read_text().splitlines()
            head = next((line.split()[0] for line in packed if line.endswith(" " + ref)), "")
    return head, None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--sparknet", required=True)
    parser.add_argument("--sparknet-head", required=True)
    parser.add_argument("--sparknet-tree", required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tree = Path(args.sparknet)
    steps, problems = [], []

    head, tree_hash, status = git_ref(tree)
    identity = {"head": head, "tree": tree_hash, "clean": None if status is None else status == ""}
    if head != args.sparknet_head:
        problems.append(f"sparknet HEAD {head}, expected {args.sparknet_head}")
    if tree_hash is not None and tree_hash != args.sparknet_tree:
        problems.append(f"sparknet tree {tree_hash}, expected {args.sparknet_tree}")
    if status:
        problems.append("the sparknet tree has local changes")
    if problems:
        print(json.dumps({"summary": "#1008 tests", "label": args.label, "passed": False, "sparknet": identity,
                          "problems": problems}), flush=True)
        return 1

    # As the PR ran them: PYTHONPATH replaced, from the image's working directory.
    env = {**os.environ, "PYTHONPATH": str(tree / "src")}
    workdir = IMAGE_WORKDIR if os.path.isdir(IMAGE_WORKDIR) else "/"
    steps.append(run("sparknet-unittest", [sys.executable, "-m", "unittest", "discover", "-s", str(tree / "tests"), "-v"],
                     out, args.label, env=env, cwd=workdir))
    ran = re.search(r"^Ran (\d+) tests?", (out / f"sparknet-unittest-{args.label}.txt").read_text(), re.M)
    steps[-1]["tests"] = int(ran.group(1)) if ran else 0
    steps.append(run("vllm-pytest", [sys.executable, "-m", "pytest", "--noconftest", "-v", *VLLM_TESTS], out, args.label))
    bench_out = out / f"bench-{args.label}.json"
    steps.append(run("pr-bench", [sys.executable, args.bench, "--boot-label", args.label, "--warmup-steps", "60",
                                  "--bench-steps", "500", "--output", str(bench_out)], out, args.label))
    if bench_out.exists():
        report = json.loads(bench_out.read_text())
        steps[-1]["regimes"] = {name: {"step_116_collectives_ms": regime["step_116_collectives_ms"],
                                       "proxy_nivcsw": regime["proxy_nivcsw"],
                                       "worker_main_nivcsw": regime["worker_main_nivcsw"]}
                                for name, regime in report.get("regimes", {}).items()}
    for step in steps:
        if step["exit"]:
            problems.append(f"{step['step']} exited {step['exit']}")
    if steps[0].get("tests", 0) == 0:
        problems.append("sparknet's unittest ran no tests")
    print(json.dumps({"summary": "#1008 tests", "label": args.label, "passed": not problems, "sparknet": identity,
                      "steps": steps, "problems": problems}), flush=True)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
