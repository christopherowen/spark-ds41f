"""Check canonical source identity and syntax without importing TileLang."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f.build import patchset_sha256, submodule_commits  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent / ".work/source")
    parser.add_argument("--require-submodules", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    candidate = ROOT / "experiments/2026-10-26-spark-ds41f-candidate/candidate-42"
    lock = json.loads((candidate / "upstreams.lock.json").read_text())
    record = json.loads((candidate / "source.json").read_text())["tilelang"]
    git = lambda *argv: subprocess.check_output(["git", "-C", str(source), *argv], text=True).strip()
    assert git("rev-parse", "HEAD") == record["patch_head"]
    assert git("rev-parse", "HEAD^{tree}") == record["expected_tree"]
    assert not git("status", "--porcelain")
    assert patchset_sha256(lock["sources"]["tilelang"]) == record["patchset_sha256"]
    changed = git("diff", "--name-only", record["base_revision"], "HEAD", "--", "*.py").splitlines()
    for path in changed:
        ast.parse((source / path).read_bytes(), filename=path, feature_version=(3, 12))
    subprocess.run(["git", "-C", str(source), "diff", "--check", record["base_revision"], "HEAD"], check=True)
    provenance = json.loads((Path(__file__).parent / "runs/preflight/authored-provenance.json").read_text())
    for row in provenance["files"]:
        assert hashlib.sha256((ROOT / row["copied_to"]).read_bytes()).hexdigest() == row["sha256"], row["copied_to"]
    if args.require_submodules:
        assert submodule_commits(source) == record["submodules"], "recursive submodule identity differs"
    print(json.dumps({"head": record["patch_head"], "tree": record["expected_tree"],
                      "patchset": record["patchset_sha256"], "Python312_parsed": len(changed),
                      "exact_authored_copy_hashes": len(provenance["files"]),
                      "recursive_submodules_checked": args.require_submodules,
                      "compiler_or_GPU_execution": False}))


if __name__ == "__main__":
    main()
