#!/usr/bin/env python3
"""The vLLM tests a release runs: selection, and the bundle that runs them.

usage:
  release_tests.py select VLLM_REPO BASE HEAD OUT.json
  release_tests.py bundle MANIFEST.json IMAGE BUNDLE_DIR LABEL

``select`` lists every test file of the deployment's stack (DeepSeek V4.1, B12X, TileLang,
Engram, DSpark/DFlash, CED, startup preparation, the JIT monitor, RoCE collectives), every test
file the release's patch series (BASE..HEAD) adds or changes, and every test file that imports
a module the series changes, except modules imported only incidentally (INCIDENTAL). Each file
records why it was selected.

``bundle`` writes a kernel-lab bundle that runs the manifest's files in the image with
``release_test_runner.py``: chunks of files per pytest process, each with a timeout, a chunk
that times out or crashes rerun file by file, every test's outcome kept as JSON lines under
/c/release-tests/LABEL, and a pytest-style summary for the lab's exit verdict.
"""
import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STACK = re.compile(
    r"v4_1|v41|ds41|deepseek_v4|b12x|tilelang|engram|dspark|dflash|test_ced|_ced_|"
    r"startup_rendezvous|prepare_kernels|jit_monitor|sparknet|roce_|_roce",
    re.IGNORECASE,
)
# Imported by hundreds of tests for unrelated reasons; the series only adds to them.
INCIDENTAL = {"vllm.envs", "vllm._custom_ops", "vllm.v1.attention.backends.registry"}


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def imported_modules(text: str) -> set[str]:
    """Every vllm module a file imports, at any depth; for ``from M import N`` also M.N,
    which is the module when N is a submodule."""
    modules = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names if alias.name.startswith("vllm"))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module \
                and node.module.startswith("vllm"):
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def select(repo: str, base: str, head: str, out: str) -> None:
    changed = git(repo, "diff", "--name-only", "--diff-filter=AMR", base, head).split()
    modules = set()
    for path in changed:
        if path.startswith("vllm/") and path.endswith(".py"):
            module = path[:-3].replace("/", ".")
            modules.add(module.removesuffix(".__init__"))
    modules -= INCIDENTAL
    tests = [line for line in git(repo, "ls-tree", "-r", "--name-only", head, "tests").split()
             if Path(line).name.startswith("test_") and line.endswith(".py")]
    files: dict[str, list[str]] = {}
    for path in tests:
        reasons = []
        if STACK.search(path):
            reasons.append("stack")
        if path in changed:
            reasons.append("changed by the series")
        hits = sorted(imported_modules(git(repo, "show", f"{head}:{path}")) & modules)
        if hits:
            reasons.append("imports " + ", ".join(hits))
        if reasons:
            files[path] = reasons
    manifest = {"repo": repo, "base": base, "head": git(repo, "rev-parse", head).strip(),
                "incidental": sorted(INCIDENTAL), "files": files}
    Path(out).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    counts = {"stack": 0, "changed by the series": 0, "imports": 0}
    for reasons in files.values():
        for reason in reasons:
            counts[reason.split(" ")[0] if reason.startswith("imports") else reason] += 1
    print(f"{len(files)} test files: {counts}")


def bundle(manifest: str, image: str, bundle_dir: str, label: str) -> None:
    target = Path(bundle_dir)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest, target / "manifest.json")
    for name in ("release_test_runner.py", "tblib_stub.py"):
        shutil.copyfile(HERE / "release_tests" / name, target / name)
    candidate = {
        "description": f"Release tests ({label}): the manifest's vLLM test files in the image, "
                       "chunked with timeouts; per-test outcomes under /c/release-tests.",
        "image": image,
        "env": {"HF_HUB_OFFLINE": "1", "PYTHONPATH": "/b",
                "TILELANG_CACHE_DIR": "/c/tilelang-release-tests",
                "VLLM_DS41_KERNEL_BACKEND": "tilelang"},
        "workdir": "/opt/spark3/candidate/vllm",
        "argv": ["/b/release_test_runner.py", "/b/manifest.json", f"/c/release-tests/{label}"],
        "verdict": "exit",
        "mounts": [[name, f"/b/{name}"]
                   for name in ("manifest.json", "release_test_runner.py", "tblib_stub.py")],
    }
    (target / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(f"bundle {target}: {len(json.loads(Path(manifest).read_text())['files'])} files")


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    {"select": select, "bundle": bundle}[command](*arguments)
