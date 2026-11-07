"""Reproduce the complete ordered vLLM series in a new CPU-only checkout."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from spark_ds41f.build import patchset_sha256, read_series

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache", type=Path, required=True)
parser.add_argument("--destination", type=Path, required=True)
args = parser.parse_args()
if args.destination.exists():
    parser.error("destination must not exist")
experiment = Path(__file__).resolve().parent
lock = json.loads((experiment / "upstreams.lock.json").read_text())["sources"]["vllm"]
manifest = json.loads((experiment / "source.json").read_text())["vllm"]
assert patchset_sha256(lock) == manifest["patchset_sha256"]
series = ROOT / lock["patch_series"]
patches = read_series(series)
assert len(patches) == manifest["patch_count"]
subprocess.run(["git", "-c", "gc.auto=0", "clone", "--shared", "--no-checkout",
                str(args.cache.resolve()), str(args.destination)], check=True)


def git(*arguments):
    subprocess.run(["git", "-c", "gc.auto=0", *arguments],
                   cwd=args.destination, check=True)


git("checkout", "-b", "verify/ced-local-projection", lock["revision"])
for index, name in enumerate(patches, 1):
    git("-c", "user.name=Christopher Owen", "-c",
        "user.email=3221756+christopherowen@users.noreply.github.com",
        "am", "--quiet", "--committer-date-is-author-date",
        str((series.parent / name).resolve()))
    if index % 50 == 0:
        print(f"{index}/{len(patches)} patches applied", flush=True)
for expression, expected in (("HEAD", manifest["patch_head"]),
                             ("HEAD^{tree}", manifest["expected_tree"])):
    actual = subprocess.check_output(["git", "rev-parse", expression],
                                     cwd=args.destination, text=True).strip()
    assert actual == expected, (expression, actual, expected)
    print(expression, actual)
git("diff", "--check", lock["revision"], "HEAD")
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=args.destination)
print(f"All {len(patches)} patches reproduce the locked candidate in a clean tree")
