#!/usr/bin/env python3
"""The vLLM tests a release runs: selection, and the bundle that runs them.

usage:
  release_tests.py select VLLM_REPO BASE HEAD OUT.json
  release_tests.py check VLLM_REPO BASE HEAD [CONFIG]
  release_tests.py gate VLLM_REPO HEAD OUT.json [CONFIG]
  release_tests.py bundle MANIFEST.json IMAGE BUNDLE_DIR LABEL

``select`` lists the DeepSeek V4.1 Flash deployment's test files: those of its stack, by path
(DeepSeek V4.1, TileLang, B12X, Engram, DSpark and the DFlash base it uses, CED, kernel
preparation and startup, the JIT monitor, RoCE collectives), and every test file the release's
patch series (BASE..HEAD) adds or changes. Each file records why it was selected. Upstream
tests outside the stack are not this deployment's to run.

``check`` classifies every selected file with config/release-tests.json: ``gate`` (exact
paths: the release runs them and must pass), ``periodic`` or ``excluded`` (paths or globs, each
with its reason). It fails on a selected file the config does not classify, a gate file that
also matches a periodic or excluded pattern, and a gate file missing at HEAD, so a new or
changed test is triaged rather than silently skipped.

``gate`` writes the manifest of the config's gate files at HEAD, with the config's
``deselect`` node ids (tests of gate files left out, each with its reason).

``bundle`` writes a kernel-lab bundle that runs the manifest's files in the image with
``release_test_runner.py``: chunks of files per pytest process, each with a timeout, a chunk
that times out or crashes rerun file by file, every test's outcome kept as JSON lines under
/c/release-tests/LABEL, and a pytest-style summary for the lab's exit verdict.
"""
import fnmatch
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "config/release-tests.json"
STACK = re.compile(
    r"v4_1|v41|ds41|deepseek_v4|b12x|tilelang|engram|dspark|dflash|test_ced|_ced_|"
    r"startup_rendezvous|prepare_kernels|jit_monitor|sparknet|roce_|_roce",
    re.IGNORECASE,
)


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def test_files(repo: str, head: str) -> list[str]:
    return [line for line in git(repo, "ls-tree", "-r", "--name-only", head, "tests").split()
            if Path(line).name.startswith("test_") and line.endswith(".py")]


def candidates(repo: str, base: str, head: str) -> dict[str, list[str]]:
    """Each selected test file at HEAD with the reasons it was selected."""
    changed = set(git(repo, "diff", "--name-only", "--diff-filter=AMR", base, head).split())
    files: dict[str, list[str]] = {}
    for path in test_files(repo, head):
        reasons = (["stack"] if STACK.search(path) else []) + (
            ["changed by the series"] if path in changed else [])
        if reasons:
            files[path] = reasons
    return files


def select(repo: str, base: str, head: str, out: str) -> None:
    files = candidates(repo, base, head)
    manifest = {"repo": repo, "base": base, "head": git(repo, "rev-parse", head).strip(),
                "files": files}
    Path(out).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    counts = {"stack": 0, "changed by the series": 0}
    for reasons in files.values():
        for reason in reasons:
            counts[reason] += 1
    print(f"{len(files)} test files: {counts}")


def load_config(path: Path | str = CONFIG) -> dict[str, dict[str, str]]:
    config = json.loads(Path(path).read_text())
    for kind in ("gate", "periodic", "excluded"):
        entries = config.get(kind)
        if not isinstance(entries, dict):
            raise ValueError(f"{path}: {kind} must map paths to reasons")
        for key, reason in entries.items():
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{path}: {kind} {key} has no reason")
            if kind == "gate" and any(c in key for c in "*?["):
                raise ValueError(f"{path}: gate entries are exact paths, not {key}")
    deselect = config.setdefault("deselect", {})
    for node, reason in deselect.items():
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{path}: deselect {node} has no reason")
        if node.split("::")[0] not in config["gate"] or "::" not in node:
            raise ValueError(f"{path}: deselect {node} is not a test of a gate file")
    return config


def classify(path: str, config: dict[str, dict[str, str]]) -> list[str]:
    """The classes whose entries match ``path`` (exactly one when it is triaged)."""
    found = ["gate"] if path in config["gate"] else []
    found += [kind for kind in ("periodic", "excluded")
              if any(fnmatch.fnmatchcase(path, pattern) for pattern in config[kind])]
    return found


def check(repo: str, base: str, head: str, config_path: str = str(CONFIG)) -> int:
    config = load_config(config_path)
    selected = candidates(repo, base, head)
    at_head = set(test_files(repo, head))
    problems = []
    for path, reasons in sorted(selected.items()):
        classes = classify(path, config)
        if not classes:
            problems.append(f"unclassified: {path} ({'; '.join(reasons)})")
        elif len(classes) > 1:
            problems.append(f"ambiguous: {path} matches {', '.join(classes)}")
    problems += [f"gate file missing at {head}: {path}" for path in sorted(config["gate"])
                 if path not in at_head]
    counts = {kind: sum(classify(p, config)[:1] == [kind] for p in selected)
              for kind in ("gate", "periodic", "excluded")}
    print(f"{len(selected)} selected test files: {counts}; {len(problems)} problems")
    for problem in problems:
        print("  " + problem)
    return 1 if problems else 0


def gate(repo: str, head: str, out: str, config_path: str = str(CONFIG)) -> int:
    config = load_config(config_path)
    at_head = set(test_files(repo, head))
    missing = sorted(set(config["gate"]) - at_head)
    if missing:
        raise SystemExit("gate files missing at " + head + ": " + ", ".join(missing))
    manifest = {"repo": repo, "head": git(repo, "rev-parse", head).strip(),
                "files": {path: [reason] for path, reason in sorted(config["gate"].items())},
                "deselect": dict(sorted(config["deselect"].items()))}
    Path(out).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"{len(manifest['files'])} gate files")
    return 0


def bundle(manifest: str, image: str, bundle_dir: str, label: str) -> None:
    target = Path(bundle_dir)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest, target / "manifest.json")
    for name in ("release_test_runner.py", "tblib_stub.py", "offline_skip.py"):
        shutil.copyfile(HERE / "release_tests" / name, target / name)
    candidate = {
        "description": f"Release tests ({label}): the manifest's vLLM test files in the image, "
                       "chunked with timeouts; per-test outcomes under /c/release-tests.",
        "image": image,
        # Tests set what they need: the deployment's environment (the kernel family,
        # DSpark cost curves, carve-out) changes what modules do at import.
        "env": {"HF_HUB_OFFLINE": "1", "PYTHONPATH": "/b",
                "TILELANG_CACHE_DIR": "/c/tilelang-release-tests",
                # Disk Engram rows need O_DIRECT on a real filesystem.
                "ENGRAM_ROWS_TEST_DIR": "/c/engram-rows-test"},
        "workdir": "/opt/spark3/candidate/vllm",
        # The disk Engram reader and the direct loader use io_uring, which Docker's
        # default seccomp profile refuses; serving allows it.
        "io_uring": True,
        "argv": ["/b/release_test_runner.py", "/b/manifest.json", f"/c/release-tests/{label}"],
        "verdict": "exit",
        "mounts": [[name, f"/b/{name}"] for name in
                   ("manifest.json", "release_test_runner.py", "tblib_stub.py", "offline_skip.py")],
    }
    (target / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    print(f"bundle {target}: {len(json.loads(Path(manifest).read_text())['files'])} files")


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    code = {"select": select, "check": check, "gate": gate, "bundle": bundle}[command](*arguments)
    sys.exit(code or 0)
