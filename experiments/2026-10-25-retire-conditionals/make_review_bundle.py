#!/usr/bin/env python3
"""A kernel-lab bundle that runs named vLLM test files of HEAD over image -next-6, whose
tree is BASE's: HEAD's package files under overlay/vllm, its deleted package files as
modules that fail to import, its test and benchmark files mounted over the image's.

usage: make_review_bundle.py VLLM_REPO BASE HEAD OUT_BUNDLE PLUGINS_DIR TEST...
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
MISSING = '''# Deleted at the reviewed head; importing it fails as importing a missing module does.
raise ModuleNotFoundError(f"No module named {__name__!r}", name=__name__)
'''

repo, base, head, out, plugins_dir, *tests = sys.argv[1:]
bundle = Path(out)
if bundle.exists():
    shutil.rmtree(bundle)
bundle.mkdir(parents=True)


def git(*args, text=True):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=text).stdout


changes = [line.split("\t") for line in git("diff", "--name-status", base, head).splitlines()]
assert all(status in ("A", "M", "D") for status, _ in changes), changes
mounts = []
for status, path in changes:
    if path.startswith("vllm/"):
        target = bundle / "overlay" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if status == "D":
            target.write_text(MISSING)
        else:
            target.write_bytes(git("show", f"{head}:{path}", text=False))
    elif status != "D" and path.startswith(("tests/", "benchmarks/")):
        name = path.replace("/", "_")
        (bundle / name).write_bytes(git("show", f"{head}:{path}", text=False))
        mounts.append([name, f"{ROOT}/{path}"])
for plugin in ("tblib_stub.py", "offline_skip.py"):
    shutil.copyfile(Path(plugins_dir) / plugin, bundle / plugin)
    mounts.append([plugin, f"/b/{plugin}"])
head_commit = git("rev-parse", "--short=9", head).strip()
candidate = {
    "description": f"vLLM {head_commit} over image -next-6 (tree of {base}): " + ", ".join(tests),
    "image": IMAGE,
    "env": {"HF_HUB_OFFLINE": "1", "PYTHONPATH": "/b",
            "TILELANG_CACHE_DIR": "/c/tilelang-retire-conditionals-review"},
    "workdir": "vllm",
    "argv": ["-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "-p", "tblib_stub",
             "-p", "offline_skip", *tests],
    "verdict": "exit",
    "mounts": mounts,
}
(bundle / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
print(f"{bundle}: {head_commit}, {len(changes)} changed files, {len(tests)} test files")
