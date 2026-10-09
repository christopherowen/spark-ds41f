#!/usr/bin/env python3
"""Copy each port's files from the vLLM migration branch into overlay/<port>/vllm/.

  sync_overlay.py [--vllm ~/projects/vllm-ds41-tilelang-migration] [port...]

Files are taken from the branch's committed tree (git show), never the working tree.
"""
import argparse
import pathlib
import subprocess

from ports import PORTS, VLLM_BRANCH

HERE = pathlib.Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    parser.add_argument("ports", nargs="*")
    args = parser.parse_args()
    commit = subprocess.run(["git", "-C", args.vllm, "rev-parse", "--short=12", VLLM_BRANCH],
                            capture_output=True, text=True, check=True).stdout.strip()
    for name in args.ports or PORTS:
        for path in PORTS[name]["files"]:
            data = subprocess.run(["git", "-C", args.vllm, "show", f"{VLLM_BRANCH}:vllm/{path}"],
                                  capture_output=True, check=True).stdout
            out = HERE / "overlay" / name / "vllm" / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            print(f"{name}: vllm/{path} @ {commit}")
        (HERE / "overlay" / name / "COMMIT").write_text(commit + "\n")


if __name__ == "__main__":
    main()
