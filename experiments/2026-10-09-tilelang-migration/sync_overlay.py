#!/usr/bin/env python3
"""Build every arm's overlay and every kernel bundle from the vLLM migration branch.

  sync_overlay.py [--vllm ~/projects/vllm-ds41-tilelang-migration]

- overlay/modules/vllm/: the files the modules change from BASE (the control's arm);
- overlay/<port>/vllm/: the modules plus the port's switch, applied on its own;
- bundles/<port>/: the module files (under flattened path names, since names repeat
  across directories), the port's tests, kbench.py and candidate.json, around its
  bench scripts.

Everything comes from commits, through a scratch worktree, never a working tree.
The previous overlays are replaced.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import tempfile

from ports import BASE, MODULES, PORTS, VLLM_BRANCH

HERE = pathlib.Path(__file__).resolve().parent
IMAGE = "vllm-ds41f-kkref:19f2c20ed4d6-r6c"
TREE = "/opt/spark3/candidate/vllm"


def bundled(path: str) -> str:
    """A vLLM tree file's name in a bundle: its path, flattened (names repeat across
    directories, such as tilelang/engram.py and common/engram.py)."""
    return path.replace("/", "__")


def candidate(bundle: dict, files: list[str]) -> dict:
    """The bundle's kernel-lab spec: the selected vLLM tests, then each bench script."""
    select = "" if bundle["select"] is None else f"'-k', {bundle['select']!r}, "
    calls = [f"subprocess.call([sys.executable, '-m', 'pytest', '-q', '--noconftest', '-p', "
             f"'no:cacheprovider', '-rfE', {select}"
             + ", ".join(repr(t) for t in bundle["tests"]) + "])"]
    calls += [f"subprocess.call([sys.executable, '/b/{script}'])" for script in bundle["scripts"]]
    argv = ("import subprocess, sys; codes = [" + ", ".join(calls) + "]; "
            "print('exit codes', codes); sys.exit(max(codes))")
    mounts = [[bundled(f), f"{TREE}/{f}"] for f in files + list(bundle["tests"])]
    mounts += [[name, f"/b/{name}"] for name in ("kbench.py", *bundle["scripts"])]
    return {"description": bundle["description"], "image": IMAGE, "env": {"HF_HUB_OFFLINE": "1"},
            "workdir": TREE, "argv": ["-c", argv], "verdict": "exit", "mounts": mounts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", default=str(pathlib.Path.home() / "projects/vllm-ds41-tilelang-migration"))
    args = parser.parse_args()

    def git(*command, cwd=args.vllm):
        return subprocess.run(["git", *command], cwd=cwd, capture_output=True, text=True, check=True).stdout

    def rev(name):
        return git("rev-parse", "--short=12", name).strip()

    for commit in (MODULES, *(p["switch"] for p in PORTS.values() if p["switch"])):
        git("merge-base", "--is-ancestor", commit, VLLM_BRANCH)  # on the branch
    module_files = git("diff", "--name-only", BASE, MODULES, "--", "vllm").split()
    with tempfile.TemporaryDirectory() as scratch:
        work = pathlib.Path(scratch) / "tree"
        git("worktree", "add", "-q", "--detach", str(work), MODULES)
        try:
            arms = {"modules": None, **{name: port["switch"] for name, port in PORTS.items()}}
            for arm, switch in arms.items():
                git("checkout", "-q", "--detach", MODULES, cwd=work)
                if switch:
                    git("cherry-pick", "--no-commit", switch, cwd=work)
                files = set(git("diff", "--name-only", BASE, "--", "vllm", cwd=work).split())
                files |= set(git("diff", "--cached", "--name-only", BASE, "--", "vllm", cwd=work).split())
                overlay = HERE / "overlay" / arm
                shutil.rmtree(overlay, ignore_errors=True)
                for path in sorted(files):
                    out = overlay / path
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(work / path, out)
                (overlay / "COMMIT").write_text(
                    f"modules {rev(MODULES)}\nswitch {rev(switch) if switch else '-'}\n")
                git("reset", "-q", "--hard", cwd=work)
                print(f"overlay/{arm}: {len(files)} files (switch {switch or '-'})")
        finally:
            git("worktree", "remove", "--force", str(work))
    for name, port in PORTS.items():
        bundle = HERE / "bundles" / name
        bundle.mkdir(parents=True, exist_ok=True)
        for stale in bundle.iterdir():  # keep only the bench scripts
            if stale.is_file() and stale.name not in port["bundle"]["scripts"]:
                stale.unlink()
        for path in module_files + list(port["bundle"]["tests"]):
            data = subprocess.run(["git", "show", f"{MODULES}:{path}"], cwd=args.vllm,
                                  capture_output=True, check=True).stdout
            (bundle / bundled(path)).write_bytes(data)
        shutil.copyfile(HERE / "kbench.py", bundle / "kbench.py")
        (bundle / "candidate.json").write_text(json.dumps(candidate(port["bundle"], module_files), indent=2) + "\n")
        print(f"bundles/{name}: {len(module_files)} module files")


if __name__ == "__main__":
    main()
