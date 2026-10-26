#!/usr/bin/env python3
"""Candidate 7's vLLM series: spark-ds41f-1025's, then vLLM next-candidate's commits.

usage: make_series.py VLLM_REPO

1025's series is candidate 6's (../2026-10-25-spark-ds41f-candidate/rc6/vllm/series),
whose tree next-candidate's e0ccc837d has. Every commit from there to the candidate head
is written to vllm/ as one patch. A commit the records already name keeps its number
(0491-0507); the others are numbered from 0508 in application order. vllm/series lists
1025's patches, their paths relative to this folder, then these in application order.
Nothing that exists is overwritten.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RC6_SERIES = HERE.parent / "2026-10-25-spark-ds41f-candidate/rc6/vllm/series"
BASE = "e0ccc837d"  # 1025's tree (6f8e062f), candidate 6's 0490
HEAD = "c579b3fe1"
# Application order (next-candidate's history) and each commit's patch number.
PATCHES = [
    ("f630c32cb", 492),  # drafter window retention
    ("9b65ced6c", 500),  # VLLM_KV_CACHE_POISON (debug)
    ("c35ac6c69", 501),  # DSpark draft windows start at CED's replay start
    ("84e3ce800", 502),  # no per-allocation KV zeroing
    ("7a2125421", 503),  # the indexer reads no key past the visible positions
    ("fa4c25e88", 508),  # the softmax fix (Barron et al., Eq. 5)
    ("615f22255", 491),  # the KV capacity figure
    ("89e657883", 498),  # the weight digest
    ("d5dbfac3f", 499),  # --load-format direct
    ("024d69e82", 493),  # TileLang the only kernel family
    ("cfa67a05a", 494),  # VLLM_DS41_KERNEL_BACKEND goes
    ("306d4eb38", 495),  # the ported kernels' Triton originals go
    ("ae81a73b2", 496),  # B12X remnants
    ("9730c7b62", 509),  # test: the draft-window replay case on TileLang's chunk
    ("b6e396951", 497),  # names: SPARK_DS41F_INDEXER_SPLIT_CHECK
    ("a7ba079c7", 510),  # mHC finalize writes its output's MXFP8 cast
    ("4209a1903", 511),  # SP prefill MoE exchange under the combine
    ("e20c47020", 512),  # test: the window reference bounds draft rows by the replay start
    ("412777ce1", 513),  # kernel audit: live-page tables
    ("c36f686d4", 514),  # kernel audit: one-launch metadata builder
    ("8c6c5fe14", 515),
    ("18f7760c9", 516),
    ("cf7120d64", 517),
    ("967a508b2", 518),
    ("16c1b8cae", 519),
    ("9ad4a1fdb", 504),  # 64-row prefill MoE tiles
    ("a0dc662f5", 505),  # two budget-sized buffers dropped
    ("410338b04", 506),  # the attention scratch lane
    ("577091997", 507),  # candidates published as block ids
    ("0ccbe30e5", 520),  # PDL along the decode chain
    ("3a0ac1afa", 521),
    ("b2bae34c9", 522),
    ("b3f659d9a", 523),
    ("1bc005688", 524),
    ("97e204cd9", 525),  # PDL: the routed grouped GEMMs trigger late from 16 rows
    ("324ab8752", 526),  # CED prompt-end anchor
    ("b56eef3d3", 527),
    ("f6daa5aaf", 528),  # block verification at the dead-row cut
    ("2a31dd474", 529),
    ("7fa9b5238", 530),
    ("3e246a2eb", 531),
    ("c579b3fe1", 532),
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
    listed = [git(repo, "rev-parse", commit) for commit, _ in PATCHES]
    if order != listed:
        raise SystemExit(f"{BASE}..{HEAD} is not the listed commits in the listed order")
    out = HERE / "vllm"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(exist_ok=True)
    series = out / "series"
    names = []
    for commit, number in PATCHES:
        # format-patch prints the path of the file it writes.
        path = Path(git(repo, "format-patch", "--zero-commit", "--no-signature", "--numbered",
                        "--start-number", str(number), "-1", commit, "-o", str(out)))
        if path.parent != out or not path.name.startswith(f"{number:04d}-"):
            raise SystemExit(f"{commit}: unexpected patch file {path}")
        names.append(path.name)
    lines = [
        "# Candidate 7: spark-ds41f-1025's series (candidate 6,",
        "# ../../2026-10-25-spark-ds41f-candidate/rc6/vllm/series), then vLLM next-candidate",
        f"# from {BASE} to {HEAD} in application order (README.md). Paths relative to this file.",
    ]
    for line in RC6_SERIES.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        target = (RC6_SERIES.parent / line.strip()).resolve()
        if not target.is_file():
            raise SystemExit(f"1025's series names a missing patch: {line}")
        lines.append(os.path.relpath(target, out))
    lines += names
    series.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines) - 3} patches, {len(names)} new: {series.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
