#!/usr/bin/env python3
"""Write the loader A/B's overlay bundles and serving arms from the vLLM loader branch.

usage: make_arms.py VLLM_REPO [--branch direct-loader] [--digest REV] [--shape tp3-triangle]

The branch is the release's source (BASE, tilelang-ports 26382f5cf, which candidate 6's
image holds with two test-only patches) with two kinds of commit (README, "Loader plan"):

- REV (``--digest``, default the branch's first commit): the weight digest, a debug check
  that logs a SHA-256 of every parameter after loading when SPARK_DS41F_WEIGHT_DIGEST=1;
- the rest: the direct loader (``--load-format direct``).

Every package file BASE..REV changes goes into bundles/digest, every one BASE..branch
changes into bundles/direct, under overlay/<path in the vLLM tree>; the arms mount them
over the image's tree. For the shape (the recipe's, config/cluster.json):

- b12x-<shape>.json: B12X's loader (production's until the direct loader replaces it)
  with the digest;
- direct-<shape>.json: the direct loader with the digest, VLLM_PLUGINS naming no plugin;
- safetensors-<shape>.json: vLLM's own safetensors loader with the digest (the evidence
  arm), VLLM_PLUGINS naming no plugin.

An unset VLLM_PLUGINS would load every installed general plugin, B12X's b12x_fp6 among
them; an empty value loads none. Each arm names its load format in ``serve_options``, so
it changes that one argument and follows every other argument of the recipe at run time;
the mounts are the recipe's own (the cache) and the bundle's files. The bundles mount
over SOURCE_ROOT, the vLLM tree of spark-ds41f-1025's image and later ones.
"""
import argparse
import importlib.machinery
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPERIMENT = HERE.relative_to(ROOT).as_posix()
RECIPE = "config/cluster.json"
BASE = "26382f5cf"
SOURCE_ROOT = "/opt/spark-ds41f/candidate/vllm"
DIGEST_ENV = "SPARK_DS41F_WEIGHT_DIGEST"


def spark_module():
    loader = importlib.machinery.SourceFileLoader("spark_loader_arms", str(ROOT / "bin" / "spark"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def write_bundle(repo: Path, revision: str, name: str) -> list[str]:
    """The package files BASE..revision changes, copied under bundles/<name>/overlay."""
    changed = [path for path in git(repo, "diff", "--name-only", "--diff-filter=AM", BASE, revision).split()
               if path.startswith("vllm/")]
    if not changed:
        raise SystemExit(f"{revision} changes no package file from {BASE}")
    overlay = HERE / "bundles" / name / "overlay"
    shutil.rmtree(overlay.parent, ignore_errors=True)
    for path in changed:
        target = overlay / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.run(["git", "-C", str(repo), "show", f"{revision}:{path}"],
                                          check=True, capture_output=True).stdout)
    (overlay.parent / "source.json").write_text(json.dumps(
        {"repository": str(repo), "base": BASE, "revision": git(repo, "rev-parse", revision).strip(),
         "files": changed}, indent=1) + "\n")
    return changed


def mounts(recipe: dict, bundle: str, files: list[str]) -> list[list[str]]:
    """The recipe's own mounts (the cache), then the bundle's files over the image's tree."""
    source = f"{{checkout}}/{EXPERIMENT}/bundles/{bundle}/overlay"
    return [*recipe["container"]["mounts"],
            *([f"{source}/{path}", f"{SOURCE_ROOT}/{path}", "ro"] for path in files)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm", type=Path)
    parser.add_argument("--branch", default="direct-loader")
    parser.add_argument("--digest", help="the commit that adds the weight digest (default: the branch's first)")
    parser.add_argument("--shape", default="tp3-triangle")
    args = parser.parse_args()
    digest = args.digest or git(args.vllm, "rev-list", "--reverse", f"{BASE}..{args.branch}").split()[0]
    digest_files = write_bundle(args.vllm, digest, "digest")
    direct_files = write_bundle(args.vllm, args.branch, "direct")
    spark = spark_module()
    recipe = spark.read_json(RECIPE)
    spark.read_cluster(RECIPE, args.shape)  # the shape is the recipe's
    short = {"tp3-triangle": "tp3", "tp4-ring": "tp4"}.get(args.shape, args.shape)
    arms = {
        f"b12x-{short}.json": (
            f"B12X's loader (--load-format b12x, VLLM_PLUGINS=b12x_loader) at {args.shape} with "
            f"the weight digest ({DIGEST_ENV}=1, bundles/digest).",
            {"environment": {DIGEST_ENV: "1", "VLLM_PLUGINS": "b12x_loader"},
             "container": {"mounts": mounts(recipe, "digest", digest_files)}},
            "b12x"),
        f"direct-{short}.json": (
            f"The direct loader (--load-format direct, bundles/direct) at {args.shape} with the weight "
            "digest; VLLM_PLUGINS names no plugin.",
            {"environment": {DIGEST_ENV: "1", "VLLM_PLUGINS": ""},
             "container": {"mounts": mounts(recipe, "direct", direct_files)}},
            "direct"),
        f"safetensors-{short}.json": (
            f"vLLM's safetensors loader (memory-mapped reads through the page cache) at {args.shape} "
            "with the weight digest; VLLM_PLUGINS names no plugin. The evidence arm.",
            {"environment": {DIGEST_ENV: "1", "VLLM_PLUGINS": ""},
             "container": {"mounts": mounts(recipe, "digest", digest_files)}},
            "safetensors"),
    }
    for name, (notes, values, load_format) in arms.items():
        arm = {"notes": notes, "extends": {"config": RECIPE, "shape": args.shape}, "set": values,
               "serve_options": {"--load-format": load_format}}
        (HERE / name).write_text(json.dumps(arm, indent=1) + "\n")
        print(f"wrote {EXPERIMENT}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
