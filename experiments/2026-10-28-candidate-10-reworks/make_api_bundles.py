#!/usr/bin/env python3
"""The DeepSeek API prompt semantics' tests in candidate 10's image (kernel-lab bundles).

usage: make_api_bundles.py API_WORKTREE

API_WORKTREE is a vLLM checkout whose commits over candidate 10's head (58938a074) are the
DeepSeek API work: the effort aliases, then response_format, tool_choice none and the
assistant prefix. Its changed vllm/ files become the overlay and its changed tests are
mounted over the image's. Bundles, on candidate 10's image:

  api-c10        candidate 10's own copies of those tests, the control;
  api-semantics  the same test files from the branch, with the branch's sources.

The tests are CPU-only (a fake tokenizer and the chat-completion protocol); they run in
the image because the laptop has no environment with vLLM's dependencies.
"""

import subprocess
import sys
from pathlib import Path

import make_reworks

SPEC = "api-semantics.json"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    repo = Path(sys.argv[1]).resolve()
    if make_reworks.git(repo, "status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit(f"{repo} has local changes")
    changed = subprocess.run(["git", "-C", str(repo), "diff", "--name-status", make_reworks.HEAD, "HEAD"],
                             check=True, capture_output=True, text=True).stdout.splitlines()
    paths = []
    for line in changed:
        status, path = line.split("\t", 1)
        if status != "M":
            raise SystemExit(f"{line!r}: an overlay can only replace files")
        paths.append(path)
    sources = [p for p in paths if p.startswith("vllm/")]
    tests = [p for p in paths if p.startswith("tests/")]
    control = make_reworks.bundle("api-c10", "Candidate 10 unchanged: its own copies of the DeepSeek API "
                                  "tests the branch changes, the control.", tests)
    branch = make_reworks.bundle("api-semantics", "Candidate 10 with the DeepSeek API prompt semantics "
                                 "(effort aliases, response_format, tool_choice none, the assistant prefix): "
                                 "the branch's tests.", tests, repo, sources, tests)
    make_reworks.write_json(make_reworks.HERE / SPEC, {
        "experiment": make_reworks.FOLDER,
        "run": "c10-api-semantics",
        "jobs": [{"kind": "sync"},
                 {"kind": "kernel", "bundles": [{"bundle": control, "node": "dgx1"},
                                                {"bundle": branch, "node": "dgx2"}]}],
    })
    print(f"api-semantics: {len(sources)} sources, {len(tests)} tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
