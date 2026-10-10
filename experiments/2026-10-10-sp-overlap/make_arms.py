#!/usr/bin/env python3
"""Write the overlap arm over r6d: the vLLM files the overlap commits change, taken from
the vLLM branch, mounted over the r6d image; and the kernel bundle of its test.

usage: make_arms.py [--vllm ~/projects/vllm-ds41-tilelang-migration]
"""
import argparse
import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT).as_posix()
BASE = ROOT / "experiments/2026-10-10-r6d-deterministic/r6d-tp4.json"
R6D = "r6d-candidate"  # the vLLM branch of the r6d image's tree
OVERLAP = "0cd9a8d9c"  # the switch on top of its module commit
TREE = "/opt/spark3/candidate/vllm"
TEST = "tests/models/test_deepseek_v41_sp_project_reduce_scatter.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    args = parser.parse_args()

    def git(*command):
        return subprocess.run(["git", "-C", args.vllm, *command], capture_output=True, check=True).stdout

    files = git("diff", "--name-only", R6D, OVERLAP, "--", "vllm").decode().split()
    overlay = HERE / "overlay" / "overlap"
    for path in files:
        out = overlay / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(git("show", f"{OVERLAP}:{path}"))
    cluster = json.loads(BASE.read_text())
    cluster["container"]["mounts"] += [
        [f"{{home}}/projects/spark-ds41f/{REL}/overlay/overlap/{path}", f"{TREE}/{path}", "ro"] for path in files]
    (HERE / "overlap.json").write_text(json.dumps(cluster, indent=2) + "\n")
    bundle = HERE / "bundles" / "sp-overlap"
    bundle.mkdir(parents=True, exist_ok=True)
    mounts = []
    for path in [*files, TEST]:
        name = path.replace("/", "__")
        (bundle / name).write_bytes(git("show", f"{OVERLAP}:{path}"))
        mounts.append([name, f"{TREE}/{path}"])
    candidate = {"description": "Sliced SP projection: its layout test, in the r6d image.",
                 "image": cluster["container"]["image"], "env": {"HF_HUB_OFFLINE": "1"}, "workdir": TREE,
                 "argv": ["-m", "pytest", "-q", "--noconftest", "-p", "no:cacheprovider", "-rfE", TEST],
                 "verdict": "exit", "mounts": mounts}
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(f"overlay/overlap: {len(files)} files ({', '.join(files)}); overlap.json; bundles/sp-overlap")


if __name__ == "__main__":
    main()
