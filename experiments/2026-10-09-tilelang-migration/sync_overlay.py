#!/usr/bin/env python3
"""Copy each port's files, at the port's commit on the vLLM migration branch, into
overlay/<port>/vllm/; for a port with a kernel bundle, also into bundles/<port>/ with
its tests and kbench.py, and write the bundle's candidate.json.

  sync_overlay.py [--vllm ~/projects/vllm-ds41-tilelang-migration] [port...]

Files are taken from committed trees (git show), never the working tree.
"""
import argparse
import json
import pathlib
import shutil
import subprocess

from ports import PORTS, VLLM_BRANCH

HERE = pathlib.Path(__file__).resolve().parent
IMAGE = "vllm-ds41f-kkref:19f2c20ed4d6-r6c"
TREE = "/opt/spark3/candidate/vllm"


def bundled(path: str) -> str:
    """A vLLM tree file's name in a bundle: its path, flattened (names repeat across
    directories, such as tilelang/engram.py and common/engram.py)."""
    return path.replace("/", "__")


def candidate(port: dict) -> dict:
    """The bundle's kernel-lab spec: the selected vLLM tests, then each bench script."""
    bundle = port["bundle"]
    select = "" if bundle["select"] is None else f"'-k', {bundle['select']!r}, "
    calls = [f"subprocess.call([sys.executable, '-m', 'pytest', '-q', '--noconftest', '-p', "
             f"'no:cacheprovider', '-rfE', {select}"
             + ", ".join(repr(t) for t in bundle["tests"]) + "])"]
    calls += [f"subprocess.call([sys.executable, '/b/{script}'])" for script in bundle["scripts"]]
    argv = ("import subprocess, sys; codes = [" + ", ".join(calls) + "]; "
            "print('exit codes', codes); sys.exit(max(codes))")
    mounts = [[bundled(f"vllm/{f}"), f"{TREE}/vllm/{f}"] for f in port["files"]]
    mounts += [[bundled(t), f"{TREE}/{t}"] for t in bundle["tests"]]
    mounts += [[name, f"/b/{name}"] for name in ("kbench.py", *bundle["scripts"])]
    return {"description": bundle["description"], "image": IMAGE, "env": {"HF_HUB_OFFLINE": "1"},
            "workdir": TREE, "argv": ["-c", argv], "verdict": "exit", "mounts": mounts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    parser.add_argument("ports", nargs="*")
    args = parser.parse_args()
    for name in args.ports or PORTS:
        port = PORTS[name]
        commit = subprocess.run(["git", "-C", args.vllm, "rev-parse", "--short=12", port["commit"]],
                                capture_output=True, text=True, check=True).stdout.strip()
        # A port's commit must be on the branch.
        subprocess.run(["git", "-C", args.vllm, "merge-base", "--is-ancestor", commit, VLLM_BRANCH],
                       check=True)

        def show(path, commit=commit):
            return subprocess.run(["git", "-C", args.vllm, "show", f"{commit}:{path}"],
                                  capture_output=True, check=True).stdout

        for path in port["files"]:
            out = HERE / "overlay" / name / "vllm" / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(show(f"vllm/{path}"))
            print(f"{name}: vllm/{path} @ {commit}")
        (HERE / "overlay" / name / "COMMIT").write_text(commit + "\n")
        if "bundle" not in port:
            continue
        # The bundle keeps its bench scripts; everything else is regenerated.
        bundle = HERE / "bundles" / name
        bundle.mkdir(parents=True, exist_ok=True)
        for stale in bundle.iterdir():
            if stale.is_file() and stale.name not in port["bundle"]["scripts"]:
                stale.unlink()
        for path in [f"vllm/{f}" for f in port["files"]] + list(port["bundle"]["tests"]):
            (bundle / bundled(path)).write_bytes(show(path))
        shutil.copyfile(HERE / "kbench.py", bundle / "kbench.py")
        (bundle / "candidate.json").write_text(json.dumps(candidate(port), indent=2) + "\n")
        print(f"{name}: bundle {bundle.relative_to(HERE)}")


if __name__ == "__main__":
    main()
