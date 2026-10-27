"""Kernel-lab bundles for the GPU bit-match of two byte levers on candidate 9.

    make_bundles.py BLOCK_SCALES_WORKTREE EXPERT_CODES_WORKTREE

Each worktree is a vLLM checkout at candidate 9's commit with the change uncommitted.
Its changed and new files under vllm/ become the bundle's overlay; its changed tests
are mounted over the image's. Bundles:

  block32-base     candidate 9 unchanged: block-32 GEMM output digests.
  block32-scales   SFB[row // 32] block-scale reads: the same digests (must equal
                   block32-base's), then the branch's GEMM tests.
  expert-codes     2-bit expert scale codes: the branch's MoE tests, which compare the
                   mixed 2-bit/4-bit launch bit for bit with the uncoded GEMM.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import make_arms

FOLDER = Path(__file__).resolve().parent
BUNDLES = FOLDER / "bundles"
HEAD = "e2b2b0b59"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-9"
ROOT = "/opt/spark-ds41f/candidate/vllm"
FIXTURES = Path(__file__).resolve().parents[1] / "2026-10-03-packed-bf16-head/bundles/vllm-tests/ds41_fixtures.py"
ENV = {"HF_HUB_OFFLINE": "1", "VLLM_DS41_KERNEL_BACKEND": "tilelang"}
PYTEST = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "-p", "ds41_fixtures", "-rfE"]


def changes(repo: Path) -> tuple[list[str], list[str]]:
    """(vllm/ files, tests/ files) changed or added against HEAD."""
    if subprocess.check_output(["git", "-C", repo, "rev-parse", "--short=9", "HEAD"], text=True).strip() != HEAD:
        raise SystemExit(f"{repo} is not at {HEAD}")
    lines = subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True).splitlines()
    paths = []
    for line in lines:
        status, path = line[:2], line[3:]
        if status.strip() not in ("M", "??"):
            raise SystemExit(f"{repo}: {line!r} is neither modified nor new")
        paths.append(path)
    return ([p for p in paths if p.startswith("vllm/") and p.endswith(".py")],
            [p for p in paths if p.startswith("tests/") and p.endswith(".py")])


def bundle(name: str, description: str, argv: list[str], repo: Path | None = None,
           extra: tuple[str, ...] = ()) -> None:
    target = BUNDLES / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    mounts = []
    for path in extra:
        shutil.copy(BUNDLES / path, target / path)
        mounts.append([path, f"/b/{path}"])
    shutil.copy(FIXTURES, target / "ds41_fixtures.py")
    mounts.append(["ds41_fixtures.py", "/b/ds41_fixtures.py"])
    if repo is not None:
        sources, tests = changes(repo)
        for path in sources:
            destination = target / "overlay" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(repo / path, destination)
        for path in tests:
            local = "tests-" + path.removeprefix("tests/").replace("/", "-")
            shutil.copy(repo / path, target / local)
            mounts.append([local, f"{ROOT}/{path}"])
    candidate = {"description": description, "image": IMAGE,
                 "env": {**ENV, "PYTHONPATH": f"/b:{ROOT}:/opt/spark-ds41f/candidate/b12x",
                         "TILELANG_CACHE_DIR": f"/c/tilelang-{name}"},
                 "source_root": "/opt/spark-ds41f/candidate", "workdir": "vllm",
                 "argv": argv, "verdict": "status" if argv[0].startswith("/b/") else "exit",
                 "mounts": mounts}
    (target / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
    print(name, len(list((target / "overlay").rglob("*.py"))) if (target / "overlay").exists() else 0, "overlay files")


def serving_arm(label: str, repo: Path, notes: str, kernels: list[tuple[str, str]]) -> None:
    """``label``'s serving arm and spec (make_arms.arms): the change's vllm/ files under
    overlay/<label>, the kernel bundles first, then the bit gates and the lean suite."""
    sources, _ = changes(repo)
    target = FOLDER / "overlay" / label
    if target.exists():
        shutil.rmtree(target)
    for path in sources:
        (target / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo / path, target / path)
    folder = FOLDER.relative_to(FOLDER.parents[1])
    make_arms.arms(label, sources, notes, adds=True,
                   kernels=[{"bundle": f"{folder}/bundles/{name}", "node": node} for name, node in kernels])


def main() -> None:
    scales, codes = (Path(arg).resolve() for arg in sys.argv[1:3])
    bundle("block32-base",
           "Candidate 9 unchanged: block-32 FP8 GEMM output digests on seeded TP4-shaped "
           "weights, 1-16 decode rows and prefill rows. block32-scales must print the same.",
           ["/b/block32_digest.py"], extra=("block32_digest.py",))
    bundle("block32-scales",
           "MXFP8 block-scale reads (SFB[row // 32]) on candidate 9: the block-32 digests, "
           "which must equal block32-base's line for line.",
           ["/b/block32_digest.py"], repo=scales, extra=("block32_digest.py",))
    bundle("block32-scales-tests",
           "MXFP8 block-scale reads on candidate 9: the branch's GEMM and scale-word tests.",
           [*PYTEST, "tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py",
            "tests/kernels/quantization/test_deepseek_v41_block_scale_words.py"], repo=scales)
    bundle("expert-codes",
           "2-bit expert scale codes on candidate 9: the branch's MoE tests, including the "
           "mixed 2-bit/4-bit launch against the uncoded GEMM bit for bit.",
           [*PYTEST, "tests/kernels/moe/test_deepseek_v41_tilelang_moe.py",
            "tests/kernels/quantization/test_deepseek_v41_expert_scale_codes.py"], repo=codes)
    serving_arm("block-scales", scales,
                "the MXFP8 GEMMs load the packed 32x32 block-scale grid at SFB[row // 32] "
                "instead of a per-row broadcast; same bits, fewer scale bytes per decode step",
                [("block32-base", "dgx1"), ("block32-scales", "dgx2"), ("block32-scales-tests", "dgx3")])
    serving_arm("expert-codes", codes,
                "experts whose block scales span at most 4 values carry 2-bit scale codes; "
                "the grouped MoE GEMM expands them to the uncoded bits",
                [("expert-codes", "dgx4")])


if __name__ == "__main__":
    main()
