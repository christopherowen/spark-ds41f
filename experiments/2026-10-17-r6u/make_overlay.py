#!/usr/bin/env python3
"""Overlay arms for iterating r6u on the r6s image, without an image build.

usage: make_overlay.py VLLM_REPO VLLM_BASE VLLM_HEAD TILELANG_TREE TAG [MONITOR]

Copies every non-test file VLLM_HEAD changes against VLLM_BASE (r6s's head) into
overlay/vllm/, and TileLang 0004's Python files from TILELANG_TREE (the pinned revision
with patches 0001-0004 applied) into overlay/tilelang/, then writes the arms:

- r6u-overlay-cold-TAG.json: an empty TileLang cache on every rank;
- r6u-overlay-coldrank-TAG.json: a TileLang cache seeded on dgx1 and dgx3 (copies of
  production's, which hold r6s's kernels, the same as r6u's) and empty on dgx2.

Both mount the overlay over the r6s image, trace every JIT compile
(kernel-prepare/jittrace) and run the JIT monitor in MONITOR mode (default error, so a
compile after preparation fails the boot by name; warn logs it instead).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = HERE.relative_to(ROOT).as_posix()
TILELANG_FILES = ("tilelang/__init__.py", "tilelang/jit/__init__.py",
                  "tilelang/cache/__init__.py", "tilelang/cache/kernel_cache.py")
TILELANG_SITE = "/usr/local/lib/python3.12/dist-packages"
VLLM_SITE = "/opt/spark3/candidate/vllm"
JITTRACE = "experiments/2026-10-17-kernel-prepare/jittrace"


def main(repo: str, base: str, head: str, tilelang_tree: str, tag: str, monitor: str = "error") -> None:
    changed = subprocess.run(
        ["git", "-C", repo, "diff", "--name-only", "--diff-filter=AMR", base, head, "--", "vllm/"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    overlay = HERE / "overlay"
    shutil.rmtree(overlay, ignore_errors=True)
    mounts = []
    for path in changed:
        target = overlay / "vllm" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        content = subprocess.run(["git", "-C", repo, "show", f"{head}:{path}"],
                                 check=True, capture_output=True).stdout
        target.write_bytes(content)
        mounts.append([f"{{home}}/projects/spark-ds41f/{EXP}/overlay/vllm/{path}",
                       f"{VLLM_SITE}/{path}", "ro"])
    for path in TILELANG_FILES:
        target = overlay / "tilelang" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(tilelang_tree) / path, target)
        mounts.append([f"{{home}}/projects/spark-ds41f/{EXP}/overlay/tilelang/{path}",
                       f"{TILELANG_SITE}/{path}", "ro"])
    recipe = json.loads((ROOT / "config/cluster.json").read_text())
    mounts = recipe["container"]["mounts"] + [
        [f"{{home}}/projects/spark-ds41f/{JITTRACE}", "/opt/spark3/jittrace", "ro"]] + mounts
    serve = list(recipe["serve_args"])
    serve[serve.index("--jit-monitor-mode") + 1] = monitor
    pythonpath = "/opt/spark3/jittrace:" + recipe["environment"]["PYTHONPATH"]
    for kind, cache in (("cold", f"/cache/kkref/jit-r6u/{tag}-cold"),
                        ("coldrank", f"/cache/kkref/jit-r6u/{tag}-coldrank")):
        arm = {
            "notes": f"r6u's vLLM and TileLang Python files over the r6s image ({tag}); "
                     + ("an empty TileLang cache on every rank" if kind == "cold" else
                        "a TileLang cache seeded on dgx1 and dgx3, empty on dgx2")
                     + f"; every JIT compile traced; JIT monitor mode {monitor}.",
            "extends": {"config": "config/cluster.json", "shape": "tp3-triangle"},
            "set": {
                "container": {"mounts": mounts},
                "serve_args": serve,
                "environment": {
                    "PYTHONPATH": pythonpath,
                    "JITTRACE_DIR": f"/cache/kkref/jittrace/r6u-{tag}-{kind}",
                    "TILELANG_CACHE_DIR": cache,
                },
            },
        }
        (HERE / f"r6u-overlay-{kind}-{tag}.json").write_text(json.dumps(arm, indent=1) + "\n")
    print(f"{len(changed)} vLLM files and {len(TILELANG_FILES)} TileLang files in overlay/; arms for {tag}")


if __name__ == "__main__":
    main(*sys.argv[1:])
