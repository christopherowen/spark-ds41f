#!/usr/bin/env python3
"""candidate-38's vLLM series: candidate-37's 0001..0676, then 0677..0681 (566 patches).

usage: make_series.py [DEPLOYMENT_REPO]

Candidate 37's series (../candidate-37/vllm/series) ends at 5b7cfa37f. Each patch
listed below is a `git format-patch` commit made on that head, taken from its pull
request's branch in this deployment repository (DEPLOYMENT_REPO, default: the
repository this file is in) at the commit named. It is written to vllm/ renumbered
in application order, with its Subject's [PATCH n/m] prefix re-sequenced; the
patch is otherwise byte for byte the pull request's. vllm/series lists
candidate-37's patches by their paths relative to this folder, then these.
Nothing that exists is overwritten.
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "candidate-37/vllm/series"
BASE = "5b7cfa37f"  # vLLM candidate-37 head (tree af0fb152)
HEAD = "753354cf4"  # vLLM candidate-38 head (tree 37b34b8f)
# (number, pull request, branch, its head commit here, path on the branch, vLLM commit on BASE)
PATCHES = [
    (677, "#860", "copy-drafter-shared-budget", "0c9ae9fa",
     "experiments/2026-11-03-copy-drafter-shared-budget/vllm/"
     "0001-fix-copy-drafts-budget-verification-from-what-every-.patch", "72abb31ff"),
    (678, "#860", "copy-drafter-shared-budget", "0c9ae9fa",
     "experiments/2026-11-03-copy-drafter-shared-budget/vllm/"
     "0002-cleanup-copy-drafts-remove-the-per-slot-draft-limits.patch", "88982f34e"),
    (679, "#848", "loop-handling", "4b42ec57",
     "experiments/2026-11-02-loop-handling/"
     "0001-feat-v1-act-on-a-loop-by-where-it-is-and-follow-the-reasoning-parser.patch", "050e8b1cc"),
    (680, "#853", "dspark-context-capture-ladder", "b35cd488",
     "experiments/2026-11-02-dspark-context-capture-ladder/"
     "0677-perf-dspark-capture-context-graphs-at-the-target-s-d.patch", "5340c9e2f"),
    (681, "#854", "dspark-fused-markov-greedy", "1149bd2b",
     "experiments/2026-11-02-dspark-markov-greedy-fusion/"
     "0677-perf-dspark-reduce-vocabulary-parallel-greedy-drafts.patch", "3b7884972"),
]


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def subject(patch: str) -> str:
    """The unwrapped Subject header without its [PATCH ...] prefix."""
    lines = patch.split("\n")
    for index, line in enumerate(lines):
        if line.startswith("Subject: "):
            text = line[len("Subject: "):]
            for continuation in lines[index + 1:]:
                if not continuation.startswith(" "):
                    break
                text += continuation
            return re.sub(r"^\[PATCH[^\]]*\]\s*", "", text)
    raise ValueError("no Subject header")


def filename(number: int, text: str) -> str:
    """git format-patch's name: the subject with runs of other characters as one dash,
    at most 52 characters."""
    out: list[str] = []
    for char in text:
        if char.isalnum():
            out.append(char)
        elif out and out[-1] != "-":
            out.append("-")
    return f"{number:04d}-{''.join(out)[:52]}.patch"


def main() -> int:
    if len(sys.argv) > 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    repo = sys.argv[1] if len(sys.argv) == 2 else str(HERE.parents[2])
    out = HERE / "vllm"
    out.mkdir(exist_ok=True)
    names = []
    last = PATCHES[-1][0]
    for number, _pr, _branch, commit, path, vllm_commit in PATCHES:
        patch = git(repo, "show", f"{commit}:{path}")
        if not patch.startswith(f"From {vllm_commit}"):
            print(f"{path} at {commit} is not vLLM {vllm_commit}", file=sys.stderr)
            return 1
        patch = re.sub(r"^Subject: \[PATCH[^\]]*\]", f"Subject: [PATCH {number}/{last}]", patch,
                       count=1, flags=re.MULTILINE)
        name = filename(number, subject(patch))
        target = out / name
        if target.exists():
            print(f"{target} exists", file=sys.stderr)
            return 1
        target.write_text(patch)
        names.append(name)
    lines = [
        "# candidate-38: candidate-37's series (../../candidate-37/vllm/series), then vLLM",
        f"# 0677..0681 from {BASE} to {HEAD} (../README.md).",
        "# Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        lines.append(line if line.startswith("../") else f"../../candidate-37/vllm/{line}")
    lines += names
    series = out / "series"
    if series.exists():
        print(f"{series} exists", file=sys.stderr)
        return 1
    series.write_text("\n".join(lines) + "\n")
    print(f"{series}: {len(lines) - 3} patches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
