#!/usr/bin/env python3
"""Copy each port's files from the vLLM migration branch into overlay/<port>/vllm/, and
into bundles/<port>/ (with its tests and kbench.py) when the port has a kernel bundle.

  sync_overlay.py [--vllm ~/projects/vllm-ds41-tilelang-migration] [port...]

Files are taken from the branch's committed tree (git show), never the working tree.
"""
import argparse
import pathlib
import shutil
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
    def show(path):
        return subprocess.run(["git", "-C", args.vllm, "show", f"{VLLM_BRANCH}:{path}"],
                              capture_output=True, check=True).stdout

    for name in args.ports or PORTS:
        port = PORTS[name]
        bundle = HERE / "bundles" / name
        for path in port["files"]:
            data = show(f"vllm/{path}")
            out = HERE / "overlay" / name / "vllm" / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            if bundle.is_dir():
                (bundle / pathlib.Path(path).name).write_bytes(data)
            print(f"{name}: vllm/{path} @ {commit}")
        (HERE / "overlay" / name / "COMMIT").write_text(commit + "\n")
        if bundle.is_dir():
            for path in port.get("tests", ()):
                (bundle / pathlib.Path(path).name).write_bytes(show(path))
                print(f"{name}: bundle {path}")
            shutil.copyfile(HERE / "kbench.py", bundle / "kbench.py")


if __name__ == "__main__":
    main()
