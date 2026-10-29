#!/usr/bin/env python3
"""Candidate 12's vLLM series: candidate 10's, then the candidate-12 commits after it.

usage: make_series.py VLLM_REPO

Candidate 10's series (../rc10/vllm/series) ends at 58938a074. Every commit
from there to the candidate head (candidate 11's without copy drafts) is written to vllm/ as one patch, numbered from 0538 in
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
HEAD = "d984b2e66"  # vLLM candidate-12
PATCHES = [
    ("35b098bd1", 538),  # [DS4.1] Grouped MoE GEMM: stage arrivals wait for the thread's copies, so the co
    ("930ad934e", 539),  # DeepSeek V4.1 tokenizer: the DeepSeek API's effort aliases
    ("0acbe219f", 540),  # DeepSeek V4.1 API prompt semantics: response_format, tool_choice none and the as
    ("870096912", 541),  # DeepSeek-V4.1: GLP steering per request, under a second served model name
    ("feffd018e", 542),  # DeepSeek-V4.1 GLP: the steering launch joins the decode chain's PDL
    ("4e73cb91c", 543),  # [DS4.1] Attention front in one launch: q/kv norms, the Q-B cast, the kv RoPE and
    ("449013c55", 544),  # [DS4.1] Attention: the decode front's one kernel in the forward
    ("b0f6bfa74", 545),  # [Scheduler] Prefill yield: a long prompt's aligned chunk gives its step to short
    ("167eba1ee", 546),  # [DS4.1] IKET debug mode: phase ranges in every TileLang kernel (VLLM_DS41_IKET)
    ("a56238dc6", 547),  # [DS4.1] IKET mode: sparknet's release stores signal; tests run their kernels fro
    ("a8d7b0ae6", 548),  # [DS4.1] IKET mode: optional CTA sampling (VLLM_DS41_IKET_CTAS); a source file pe
    ("bbf5ab557", 549),  # [DS4.1] IKET mode: range ends wait for a deferred barrier (a shared load before 
    ("822a1d067", 550),  # [DS4.1] IKET mode: the settle load takes its shared address from a register
    ("8b028618e", 551),  # [DS4.1] IKET mode test: off adds neither IKET module (CuTe DSL's own loads with 
    ("3793486ee", 552),  # [DS4.1] IKET mode: optional loop-iteration sampling (VLLM_DS41_IKET_ITERS)
    ("4e8e7be06", 553),  # [DS4.1] IKET mode: each manifest bounds a warp's events under the sampling
    ("62f5eb0d1", 554),  # [DS4.1] IKET mode: VLLM_DS41_IKET_SKIP leaves named kernels uninstrumented
    ("d234f87f7", 555),  # [DS4.1] IKET mode: sampled iterations sit in the middle of their share of the lo
    ("73c9cf61d", 556),  # [DS4.1] IKET mode: with sampled iterations only lane 0 records (divergent lanes 
    ("de81eb34e", 557),  # [DS4.1] IKET mode: a line within the style limit
    ("166ab94f8", 558),  # DS4.1 IKET: a range in a while loop has no bound on its records
    ("3b4bb91e8", 559),  # DS4.1 IKET: warp-uniform guards only; lane 0's left ptxas predicating events, ea
    ("cf4b7a654", 560),  # DS4.1 IKET: sampled events in loops of zero or one iterations, which ptxas does 
    ("62a44fb7e", 561),  # DS4.1 IKET: each site's loop trips in the manifest, to scale sampled iterations
    ("ef90a111e", 562),  # DS4.1 IKET: a sampled range's name carries the factor to its whole loops (*N)
    ("0084669a2", 563),  # [DS4.1] Grouped MoE GEMM: 2-bit scale codes for experts whose block scales span 
    ("d718b3c42", 564),  # [DS4.1] Expert scale-code tests follow v2: the pad lives in scale_pack, each thr
    ("d984b2e66", 565),  # [DS4.1] IKET mode tests: the subprocesses import the vLLM under test
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
        "# Candidate 12: candidate 10's series (../../rc10/vllm/series), then vLLM candidate-12",
        f"# from {BASE} to {HEAD} (../README.md, Candidate 12). Paths relative to this file.",
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
