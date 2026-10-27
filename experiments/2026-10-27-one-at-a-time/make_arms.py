#!/usr/bin/env python3
"""Candidate 9 with one fix left out, for each fix that can be isolated on its image.

usage: make_arms.py VLLM_WORKTREE

VLLM_WORKTREE is a clean vLLM checkout; it is moved to candidate 9's head (e2b2b0b59), each
fix is reverted there without committing, the files it touches are copied to
overlay/<label>/, and the checkout is reset to the head again. "no-0504" and "no-0502" are
edits instead of reverts; the "nopdl-" arms are reverts on candidate 9 without PDL (below).
The checkout is left at the head.

Writes, per fix: overlay/<label>/vllm/..., <label>-tp4.json (candidate 9's arm with the
overlay mounted), <label>-tp4-dev.json (the same with the dev-mode collective RPC, for the
engine grid) and <label>.json (the lab spec: candidate 9, the arm, candidate 9 again).
"""

import json
import re
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
#
# label: ({path: (text before, text after, occurrences)}, claim)
EDITS = {
    "no-0504": ({"vllm/models/deepseek_v4_1/tilelang/moe.py":
                 ("        prefill_block_M=64,\n", "        prefill_block_M=32,\n", 1)},
                "routed experts at 64-row tiles for prefill calls: faster prefill, same bits"),
    # 0502's effect is the flag; with it False the KV cache manager zeroes a block again
    # on every allocation, as before 0502.
    "no-0502": ({"vllm/models/deepseek_v4_1/attention.py":
                 ("            uses_only_written_slots=True,\n", "            uses_only_written_slots=False,\n", 2),
                 "vllm/models/deepseek_v4_1/compressor.py":
                 ("            uses_only_written_slots=True,\n", "            uses_only_written_slots=False,\n", 1)},
                "no per-allocation KV zeroing: faster block allocation, same bits"),
}

# Fixes PDL (577091997..97e204cd9) was built on, isolated on candidate 9's no-PDL arm
# (rc9/rc-tp4-nopdl.json: PDL's 13 files at 577091997) and compared with that arm.
PDL_BEFORE, PDL_AFTER = "577091997", "97e204cd9"
NOPDL = {
    "nopdl-no-0510": "mHC finalize writes its output's MXFP8 cast, the projections it feeds stop casting: "
                     "shorter layers, same bits",
    "nopdl-no-pagetables": "sparse-MLA page tables over each row's live pages (0513-0519): cheaper attention "
                           "metadata at long context, same bits",
}
PAGE_TABLES = ["16c1b8cae", "967a508b2", "cf7120d64", "18f7760c9", "8c6c5fe14", "c36f686d4", "412777ce1"]


def git(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], check=check, capture_output=True, text=True)


def changed_files(repo: str, commits: list[str], range_from: str | None = None) -> list[str]:
    """The vLLM files ``commits`` change (or ``range_from..commits[0]``), all modified."""
    files = set()
    for commit in commits:
        if range_from:
            lines = git(repo, "diff", "--name-status", range_from, commit, "--", "vllm").stdout.split("\n")
        else:
            lines = git(repo, "show", "--name-status", "--format=", commit, "--", "vllm").stdout.split("\n")
        for line in filter(None, lines):
            status, path = line.split("\t", 1)
            if status != "M":
                raise SystemExit(f"{commit} {status} {path}: an overlay can only replace files")
            files.add(path)
    return sorted(files)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def arms(label: str, files: list[str], notes: str, base: str = "rc-tp4",
         base_label: str = "candidate-9", adds: bool = False,
         kernels: list[dict] | None = None) -> None:
    """The arm (candidate 9's, with ``files`` mounted from overlay/<label>), its dev twin,
    and the spec comparing it with ``base`` (an arm in rc9/) before and after it. ``adds``:
    the overlay adds a change rather than reverting one. ``kernels``: kernel-lab bundles
    the spec runs before any boot."""
    mounts = [["{checkout}/cache", "/cache", "rw"]]
    for path in files:
        mounts.append([f"{{checkout}}/{FOLDER}/overlay/{label}/{path}", f"{IMAGE_ROOT}/{path}", "ro"])
    what = label.removeprefix("nopdl-").removeprefix("no-")
    against = "candidate 9 without PDL" if base != "rc-tp4" else "candidate 9"
    write_json(HERE / f"{label}-tp4.json", {
        "notes": f"{against[0].upper()}{against[1:]}, {'with' if adds else 'without'} {what}: {notes} "
                 f"(overlay/{label}, {'make_bundles.py' if adds else 'make_arms.py'}).",
        "extends": {"config": f"{CANDIDATE}/rc-tp4.json"},
        "set": {"container": {"mounts": mounts}},
    })
    write_json(HERE / f"{label}-tp4-dev.json", {
        "notes": f"{label}-tp4.json with the dev-mode collective RPC, for the engine grid.",
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
            *([{"kind": "kernel", "bundles": kernels}] if kernels else []),
            {"kind": "measure", "bench": False, "bracket": False, "extras": gates,
             "arms": [{"config": f"{CANDIDATE}/{base}.json", "label": base_label},
                      {"config": f"{label}-tp4.json", "label": label}]},
            {"kind": "measure", "suite": f"{FOLDER}/lean-suite.json",
             "arms": [{"config": f"{CANDIDATE}/{base}-dev.json", "label": base_label},
                      {"config": f"{label}-tp4-dev.json", "label": label},
                      {"config": f"{CANDIDATE}/{base}-dev.json", "label": f"{base_label}-again"}]},
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
    for label, (edits, notes) in EDITS.items():
        shutil.rmtree(HERE / "overlay" / label, ignore_errors=True)
        for path, (before, after, count) in edits.items():
            text = git(repo, "show", f"{HEAD}:{path}").stdout
            if text.count(before) != count:
                raise SystemExit(f"{label}: {before.strip()!r} is not in {path} {count} times")
            target = HERE / "overlay" / label / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text.replace(before, after), encoding="utf-8")
        arms(label, sorted(edits), notes)
        print(f"{label}: {len(edits)} files (edit)")
    # PDL's modified files go back to PDL_BEFORE; the module it adds (tilelang/pdl.py)
    # stays, unused, as in the no-PDL arm.
    pdl_files = [line.split("\t", 1)[1] for line in
                 git(repo, "diff", "--name-status", PDL_BEFORE, PDL_AFTER, "--", "vllm").stdout.split("\n")
                 if line.startswith("M\t")]
    for label, notes in NOPDL.items():
        git(repo, "reset", "-q", "--hard", HEAD)
        for path in pdl_files:
            (Path(repo) / path).write_text(git(repo, "show", f"{PDL_BEFORE}:{path}").stdout, encoding="utf-8")
        git(repo, "-c", "user.name=make_arms", "-c", "user.email=make_arms", "commit", "-qam",
            "candidate 9 without PDL")
        try:
            touched = (_revert_0510 if label == "nopdl-no-0510" else _revert_page_tables)(repo)
            files = sorted(set(pdl_files) | set(touched))
            target = HERE / "overlay" / label
            shutil.rmtree(target, ignore_errors=True)
            for path in files:
                (target / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(repo) / path, target / path)
        finally:
            git(repo, "revert", "--abort", check=False)
            git(repo, "reset", "-q", "--hard", HEAD)
        arms(label, files, notes, base="rc-tp4-nopdl", base_label="nopdl")
        print(f"{label}: {len(files)} files on the no-PDL arm")
    return 0


def _resolve(path: Path, pick) -> None:
    """Replace each conflict block of ``path`` with ``pick(index, ours, theirs)``."""
    text = path.read_text(encoding="utf-8")
    blocks = list(re.finditer(r"<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n", text, flags=re.S))
    pieces, last = [], 0
    for index, block in enumerate(blocks):
        pieces += [text[last:block.start()], pick(index, block.group(1), block.group(2))]
        last = block.end()
    resolved = "".join(pieces) + text[last:]
    if "<<<<<<<" in resolved or ">>>>>>>" in resolved:
        raise SystemExit(f"{path}: conflicts left")
    path.write_text(resolved, encoding="utf-8")


def _replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise SystemExit(f"{before!r} is not in the conflict exactly once")
    return text.replace(before, after)


def _revert_0510(repo: str) -> list[str]:
    """0510 reverted on the no-PDL tree. Its three conflicts are with 0504's tile heights
    and 0511's deferred combine: keep those, and take the activations unquantized again
    (the MoE casts them itself, as before 0510)."""
    commit = "a7ba079c7"
    git(repo, "-c", "user.name=make_arms", "-c", "user.email=make_arms", "revert", "--no-commit", commit,
        check=False)
    fused = Path(repo) / "vllm/model_executor/layers/fused_moe/tilelang.py"
    moe = Path(repo) / "vllm/models/deepseek_v4_1/tilelang/moe.py"
    _resolve(fused, lambda i, ours, theirs: _replace_once(ours, "_provided_mxfp8(hidden_states)", "hidden_states"))

    def pick(index: int, ours: str, theirs: str) -> str:
        if index == 0:
            return _replace_once(ours, " The tokens arrive cast to MXFP8 by the\n    kernel that produced them "
                                       "(``linear.MXFP8Rows``: the mHC entry).", "")
        if index == 1:
            return _replace_once(ours, "rows = topk_ids.shape[0]", "rows = x.shape[0]")
        ours = _replace_once(ours, "x.scales.view(torch.uint32)", "x_sf.view(torch.uint32)")
        return _replace_once(ours, "gate_up_kernel(x.values, x_words", "gate_up_kernel(xq, x_words")

    _resolve(moe, pick)
    unmerged = git(repo, "diff", "--name-only", "--diff-filter=U", "--", "vllm").stdout.split()
    if set(unmerged) - {str(fused.relative_to(repo)), str(moe.relative_to(repo))}:
        raise SystemExit(f"0510: unexpected conflicts in {unmerged}")
    return changed_files(repo, [commit])


def _revert_page_tables(repo: str) -> list[str]:
    """0513-0519 out of the no-PDL tree: metadata.py and sparse_mla.py as before 0513
    (nothing after 0519 but PDL changes them), attention.py with 0513's lines reversed
    (0505 and 0507 changed it later)."""
    first = PAGE_TABLES[-1]
    files = changed_files(repo, PAGE_TABLES)
    for path in files:
        if path == "vllm/models/deepseek_v4_1/attention.py":
            patch = git(repo, "show", first, "--", path).stdout
            subprocess.run(["git", "-C", repo, "apply", "-R"], input=patch, text=True, check=True)
            continue
        later = (git(repo, "log", "--format=%h", f"{PAGE_TABLES[0]}..{PDL_BEFORE}", "--", path).stdout.split()
                 + git(repo, "log", "--format=%h", f"{PDL_AFTER}..{HEAD}", "--", path).stdout.split())
        if later:
            raise SystemExit(f"{path} changed after the page tables by {later}")
        (Path(repo) / path).write_text(git(repo, "show", f"{first}~1:{path}").stdout, encoding="utf-8")
    return files


if __name__ == "__main__":
    sys.exit(main())
