#!/usr/bin/env python3
"""Candidate 9 with one fix left out, for each fix that can be isolated on its image.

usage: make_arms.py VLLM_WORKTREE

VLLM_WORKTREE is a clean vLLM checkout; it is moved to candidate 9's head (e2b2b0b59), each
fix is reverted there without committing, the files it touches are copied to
overlay/<label>/, and the checkout is reset to the head again. A fix whose revert conflicts
with later commits is not built here; "no-0504" is an edit instead of a revert (below).

Writes, per fix: overlay/<label>/vllm/..., <label>-tp4.json (candidate 9's arm with the
overlay mounted), <label>-tp4-dev.json (the same with the dev-mode collective RPC, for the
engine grid) and <label>.json (the lab spec: candidate 9, the arm, candidate 9 again).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FOLDER = "experiments/2026-10-27-one-at-a-time"
CANDIDATE = "experiments/2026-10-26-spark-ds41f-candidate/rc9"
HEAD = "e2b2b0b59"
IMAGE_ROOT = "/opt/spark-ds41f/candidate/vllm"

# label: (vLLM commits reverted, newest first; what candidate 9 claims the fix does)
REVERTS = {
    "no-0505": (["a0dc662f5"], "drops two budget-sized buffers nothing reads: more free memory, same bits"),
    "no-0506": (["410338b04"], "TileLang attention scratch in its own workspace lane: same bits"),
    "no-0511": (["4209a1903"], "SP prefill MoE: the combine writes the exchange slots and the exchange runs under it: faster prefill"),
    "no-0526": (["324ab8752"], "CED replays the prompt's last 128 rows wherever its chunks end: correct CED at any chunking"),
    "no-0528": (["f6daa5aaf"], "block verification ends each block at the dead-row cut: correct sampled verification"),
}
# A 64-row prefill tile (0504) cannot be reverted alone: PDL's late-trigger kernels were
# built on it. prefill_block_M=32 sends prefill calls through the same late-trigger kernels
# candidate 9 uses for every call of 16 rows or more, which is candidate 9 without 0504.
EDITS = {
    "no-0504": ("vllm/models/deepseek_v4_1/tilelang/moe.py", "        prefill_block_M=64,\n",
                "        prefill_block_M=32,\n",
                "routed experts at 64-row tiles for prefill calls: faster prefill, same bits"),
}


def git(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], check=check, capture_output=True, text=True)


def changed_files(repo: str, commits: list[str]) -> list[str]:
    files = set()
    for commit in commits:
        lines = git(repo, "show", "--name-status", "--format=", commit, "--", "vllm").stdout.split("\n")
        for line in filter(None, lines):
            status, path = line.split("\t", 1)
            if status != "M":
                raise SystemExit(f"{commit} {status} {path}: an overlay can only replace files")
            files.add(path)
    return sorted(files)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def arms(label: str, files: list[str], notes: str) -> None:
    mounts = [["{checkout}/cache", "/cache", "rw"]]
    for path in files:
        mounts.append([f"{{checkout}}/{FOLDER}/overlay/{label}/{path}", f"{IMAGE_ROOT}/{path}", "ro"])
    write_json(HERE / f"{label}-tp4.json", {
        "notes": f"Candidate 9 without {label[3:]}: {notes} (overlay/{label}, make_arms.py).",
        "extends": {"config": f"{CANDIDATE}/rc-tp4.json"},
        "set": {"container": {"mounts": mounts}},
    })
    write_json(HERE / f"{label}-tp4-dev.json", {
        "notes": f"Candidate 9 without {label[3:]} with the dev-mode collective RPC, for the engine grid.",
        "extends": {"config": f"{FOLDER}/{label}-tp4.json"},
        "set": {"environment": {"VLLM_SERVER_DEV_MODE": "1"}},
    })
    gates = [
        ["experiments/2026-10-12-r6g/token_determinism_text.py",
         ["--tokens", "192", "--long-lines", "700", "--wide"], "determinism"],
        ["experiments/2026-10-12-r6g/token_determinism_text.py",
         ["--tokens", "192", "--long-lines", "700", "--wide", "--no-logprobs"], "greedy"],
        ["experiments/2026-10-24-tilelang-completion/sampled_digest.py", ["--tokens", "192"], "sampled"],
    ]
    write_json(HERE / f"{label}.json", {
        "experiment": FOLDER,
        "run": f"one-at-a-time-{label}",
        "jobs": [
            {"kind": "sync"},
            {"kind": "measure", "bench": False, "bracket": False, "extras": gates,
             "arms": [{"config": f"{CANDIDATE}/rc-tp4.json", "label": "candidate-9"},
                      {"config": f"{label}-tp4.json", "label": label}]},
            {"kind": "measure", "suite": f"{FOLDER}/lean-suite.json",
             "arms": [{"config": f"{CANDIDATE}/rc-tp4-dev.json", "label": "candidate-9"},
                      {"config": f"{label}-tp4-dev.json", "label": label},
                      {"config": f"{CANDIDATE}/rc-tp4-dev.json", "label": "candidate-9-again"}]},
        ],
    })


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    repo = sys.argv[1]
    if git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise SystemExit(f"{repo} has local changes")
    git(repo, "checkout", "-q", "--detach", HEAD)
    for label, (commits, notes) in REVERTS.items():
        files = changed_files(repo, commits)
        reverted = git(repo, "-c", "user.name=make_arms", "-c", "user.email=make_arms",
                       "revert", "--no-commit", *commits, check=False)
        try:
            if reverted.returncode:
                raise SystemExit(f"{label}: {commits} do not revert cleanly at {HEAD}:\n{reverted.stderr}")
            target = HERE / "overlay" / label
            shutil.rmtree(target, ignore_errors=True)
            for path in files:
                (target / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(repo) / path, target / path)
        finally:
            git(repo, "revert", "--abort", check=False)
            git(repo, "reset", "-q", "--hard", HEAD)
        arms(label, files, notes)
        print(f"{label}: {len(files)} files")
    for label, (path, before, after, notes) in EDITS.items():
        text = git(repo, "show", f"{HEAD}:{path}").stdout
        if text.count(before) != 1:
            raise SystemExit(f"{label}: {before.strip()!r} is not in {path} exactly once")
        target = HERE / "overlay" / label / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.replace(before, after), encoding="utf-8")
        arms(label, [path], notes)
        print(f"{label}: 1 file (edit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
