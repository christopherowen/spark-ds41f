#!/usr/bin/env python3
"""RigMark (github.com/alexellis/rigmark, pinned) against the serving cluster, for the
release notes: the community's coding-agent serving benchmark, with the request body its
published DGX Spark comparisons use (``reasoning_effort`` low).

usage: rigmark_run.py URL LABEL METADATA COMPARISON_ID

Clones the pinned revision into ~/spark-lab/rigmark-<revision> once and checks it is clean,
writes METADATA with the image and deployment commit filled in, runs ``./rigmark bench``
and copies the JSON receipt and share card into results/private/rigmark/LABEL/ (RigMark's
own output goes to rigmark.log there). Prints one JSON line for the lab's table.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = "https://github.com/alexellis/rigmark"
REVISION = "770c3106729d5b73543afa2d9b498cebda6456f3"  # 2026-10-14, protocol 1.3.0
EXTRA_BODY = {"chat_template_kwargs": {"reasoning_effort": "low"}}
ROOT = Path(__file__).resolve().parents[2]
# bin/spark's serving names: the container since the rename to spark-ds41f, every dsv41-* before it.
SERVING = ("^spark-ds41f$", "^dsv41-")


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def checkout() -> Path:
    target = Path.home() / "spark-lab" / f"rigmark-{REVISION[:12]}"
    if not (target / ".git").exists():
        staging = Path(tempfile.mkdtemp(prefix=".rigmark-", dir=target.parent))
        subprocess.run(["git", "clone", "--quiet", REPOSITORY, str(staging / "r")], check=True)
        git("checkout", "--quiet", "--detach", REVISION, cwd=staging / "r")
        (staging / "r").rename(target)
    if git("rev-parse", "HEAD", cwd=target) != REVISION:
        raise SystemExit(f"{target} is not at {REVISION}")
    if git("status", "--porcelain", "--untracked-files=no", cwd=target):
        raise SystemExit(f"{target} has local changes")
    return target


def serving_container() -> str:
    """The running serving container: a stopped one is production's, kept while an arm serves."""
    names = subprocess.run(["docker", "ps", *(f"--filter=name={pattern}" for pattern in SERVING),
                            "--format", "{{.Names}}"], check=True, capture_output=True, text=True).stdout.split()
    if len(names) != 1:
        raise SystemExit(f"expected one running serving container, found {', '.join(names) or 'none'}")
    return names[0]


def main(url: str, label: str, metadata: str, comparison: str) -> None:
    rigmark = checkout()
    out = ROOT / "results" / "private" / "rigmark" / label
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT / metadata).read_text())
    image = subprocess.run(["docker", "inspect", "-f", "{{.Config.Image}} {{.Image}}", serving_container()],
                           check=True, capture_output=True, text=True).stdout.split()
    meta["serving_image"] = f"{image[0]} ({image[1]}, built from the spark-ds41f recipe; not published)"
    meta["recipe_revision"] = git("rev-parse", "HEAD", cwd=ROOT)
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    before = set((rigmark / "results").glob(f"{label}-*"))
    env = dict(os.environ)
    env.setdefault("OPENAI_API_KEY", "EMPTY")  # vLLM serves without a key
    with (out / "rigmark.log").open("w") as log:
        code = subprocess.run(
            [sys.executable, "rigmark", "bench", "--base-url", url, "--model", "auto",
             "--label", label, "--comparison-id", comparison, "--metadata", str(meta_path),
             "--extra-body", json.dumps(EXTRA_BODY)],
            cwd=rigmark, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    produced = sorted(set((rigmark / "results").glob(f"{label}-*")) - before)
    for path in produced:
        (out / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"summary": "rigmark", "exit": code, "revision": REVISION,
                      "files": [p.name for p in produced]}))
    sys.exit(code)


if __name__ == "__main__":
    main(*sys.argv[1:5])
