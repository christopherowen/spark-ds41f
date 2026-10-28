#!/usr/bin/env python3
"""Candidate 11's vLLM series: candidate 10's, then the candidate-11 commits after it.

usage: make_series.py VLLM_REPO

Candidate 10's series (../rc10/vllm/series) ends at 58938a074. Every commit
from there to the candidate head is written to vllm/ as one patch, numbered from 0538 in
application order. vllm/series lists candidate 10's patches, their paths relative to this
folder, then these. Nothing that exists is overwritten.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "rc10/vllm/series"
BASE = "58938a074"  # candidate 10's head (tree 24313745)
HEAD = "69f1f158b"  # vLLM candidate-11
PATCHES = [
    ("03dfd6bc7", 538),  # [DS4.1] Grouped MoE GEMM: stage arrivals wait for the thread's copies, so the 
    ("68682ce55", 539),  # DeepSeek V4.1 tokenizer: the DeepSeek API's effort aliases
    ("ea2da0786", 540),  # DeepSeek V4.1 API prompt semantics: response_format, tool_choice none and the 
    ("668e0db3a", 541),  # DeepSeek-V4.1: GLP steering per request, under a second served model name
    ("a37440cf0", 542),  # DeepSeek-V4.1 GLP: the steering launch joins the decode chain's PDL
    ("665b4520b", 543),  # [DS4.1] Attention front in one launch: q/kv norms, the Q-B cast, the kv RoPE a
    ("a62b6dbf7", 544),  # [DS4.1] Attention: the decode front's one kernel in the forward
    ("6c835c788", 545),  # [Scheduler] Prefill yield: a long prompt's aligned chunk gives its step to sho
    ("86fb465a9", 546),  # [Spec] Copy drafts: a verification lookahead apart from DSpark's block
    ("73d996c55", 547),  # [DS4.1] Decode rows for the verification lookahead
    ("5a74d9837", 548),  # [Spec] copy_match and copy_select (TileLang), the copy drafter
    ("6071aadd5", 549),  # [V2] Copy drafts in the runner: wide verification and the copy state's report
    ("44325f431", 550),  # [Spec] Copy drafts benchmark: the copy kernels and verification at the lookahe
    ("e28f985f5", 551),  # [DS4.1] IKET debug mode: phase ranges in every TileLang kernel (VLLM_DS41_IKET
    ("398cd8b41", 552),  # [DS4.1] IKET mode: sparknet's release stores signal; tests run their kernels f
    ("86b39c0c1", 553),  # [DS4.1] IKET mode: optional CTA sampling (VLLM_DS41_IKET_CTAS); a source file 
    ("9defc5e88", 554),  # [DS4.1] IKET mode: range ends wait for a deferred barrier (a shared load befor
    ("6567e76e6", 555),  # [DS4.1] IKET mode: the settle load takes its shared address from a register
    ("4e38a08bb", 556),  # [DS4.1] IKET mode test: off adds neither IKET module (CuTe DSL's own loads wit
    ("1f5794fa4", 557),  # [DS4.1] IKET mode: optional loop-iteration sampling (VLLM_DS41_IKET_ITERS)
    ("1a9c9302b", 558),  # [DS4.1] IKET mode: each manifest bounds a warp's events under the sampling
    ("ff79802bb", 559),  # [DS4.1] IKET mode: VLLM_DS41_IKET_SKIP leaves named kernels uninstrumented
    ("2b664117d", 560),  # [DS4.1] IKET mode: sampled iterations sit in the middle of their share of the 
    ("2f67fff78", 561),  # [DS4.1] IKET mode: with sampled iterations only lane 0 records (divergent lane
    ("fe514dfa0", 562),  # [DS4.1] IKET mode: a line within the style limit
    ("942384353", 563),  # DS4.1 IKET: a range in a while loop has no bound on its records
    ("d08befbc0", 564),  # DS4.1 IKET: warp-uniform guards only; lane 0's left ptxas predicating events, 
    ("33908324e", 565),  # DS4.1 IKET: sampled events in loops of zero or one iterations, which ptxas doe
    ("2e6733dd4", 566),  # DS4.1 IKET: each site's loop trips in the manifest, to scale sampled iteration
    ("04d945fe9", 567),  # DS4.1 IKET: a sampled range's name carries the factor to its whole loops (*N)
    ("b4ae21b63", 568),  # [DS4.1] Grouped MoE GEMM: 2-bit scale codes for experts whose block scales span
    ("69f1f158b", 569),  # [DS4.1] Expert scale-code tests follow v2: the pad lives in scale_pack, each th
]


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    repo = sys.argv[1]
    order = git(repo, "rev-list", "--reverse", f"{BASE}..{HEAD}").split()
    if order != [git(repo, "rev-parse", commit) for commit, _ in PATCHES]:
        raise SystemExit(f"{BASE}..{HEAD} is not the listed commits in the listed order")
    out = HERE / "vllm"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(exist_ok=True)
    names = []
    for commit, number in PATCHES:
        # format-patch prints the path of the file it writes.
        path = Path(git(repo, "format-patch", "--zero-commit", "--no-signature", "--numbered",
                        "--start-number", str(number), "-1", commit, "-o", str(out)))
        if path.parent != out or not path.name.startswith(f"{number:04d}-"):
            raise SystemExit(f"{commit}: unexpected patch file {path}")
        names.append(path.name)
    lines = [
        "# Candidate 11: candidate 10's series (../../rc10/vllm/series), then vLLM candidate-11",
        f"# from {BASE} to {HEAD} (../README.md, Candidate 11). Paths relative to this file.",
    ]
    for line in PREVIOUS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        target = (PREVIOUS.parent / line.strip()).resolve()
        if not target.is_file():
            raise SystemExit(f"candidate 10's series names a missing patch: {line}")
        lines.append(os.path.relpath(target, out))
    lines += names
    (out / "series").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines) - 2} patches, {len(names)} new: vllm/series")
    return 0


if __name__ == "__main__":
    sys.exit(main())
