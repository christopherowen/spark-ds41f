"""Upstream source preparation, build-context hashing, Docker image build and GPU smoke."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import fcntl
import functools
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTER_CONFIG = "config/cluster.json"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import source_labels
import kernel_backend
import kernel_readiness
import topology
import recipe as cluster_recipe
from .config import *
from .doctor import *
from .cluster import *

def read_series(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def checked(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def command_upstream(args: argparse.Namespace) -> int:
    _, _, upstreams = configuration(args)
    sources = upstreams["sources"]
    git_sources = {key: value for key, value in sources.items() if value["kind"] == "git"}
    if args.upstream_command == "list":
        for name, source in git_sources.items():
            print(f"{name}\t{source['revision']}\t{source['upstream']}")
        return 0

    source = git_sources.get(args.name)
    if source is None:
        raise SystemExit(f"unknown Git upstream {args.name!r}")
    destination = ROOT / ".work" / "upstreams" / args.name
    if destination.exists():
        raise SystemExit(f"{destination} already exists; remove it explicitly to recreate")
    destination.parent.mkdir(parents=True, exist_ok=True)
    checked(["git", "clone", source["upstream"], str(destination)])
    checked(["git", "remote", "rename", "origin", "upstream"], cwd=destination)
    contribution = source.get("contribution_remote")
    if contribution:
        checked(["git", "remote", "add", "origin", contribution], cwd=destination)
    # Release commits are not always reachable from the canonical default branch.
    # Fetch the full pinned object explicitly before detaching at it.
    checked(["git", "fetch", "upstream", source["revision"]], cwd=destination)
    checked(["git", "checkout", "--detach", source["revision"]], cwd=destination)
    if source.get("submodules"):
        checked(["git", "submodule", "update", "--init", "--recursive"], cwd=destination)
    for commit in source.get("applied_upstream_commits", []):
        checked(["git", "fetch", "upstream", commit["revision"]], cwd=destination)
        checked(
            ["git", "cherry-pick", "--no-commit", commit["revision"]],
            cwd=destination,
        )
    pending: list[str] = []
    series = source.get("patch_series")
    if series:
        series_path = ROOT / series
        for patch_name in read_series(series_path):
            if patch_name.startswith("TODO-"):
                pending.append(patch_name)
                continue
            checked(["git", "apply", str(series_path.parent / patch_name)], cwd=destination)
    print(f"prepared {args.name} in {destination}")
    print("remotes:")
    checked(["git", "remote", "-v"], cwd=destination)
    if pending:
        print("pending local patches not yet exported:", file=sys.stderr)
        for patch_name in pending:
            print(f"  {patch_name}", file=sys.stderr)
        return 2
    return 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        digest.update(str(relative).encode())
        if path.is_symlink():
            digest.update(b"L")
            digest.update(path.readlink().as_posix().encode())
        elif path.is_file():
            digest.update(b"F")
            digest.update(oct(path.stat().st_mode & 0o777).encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        elif path.is_dir():
            digest.update(b"D")
    return digest.hexdigest()


def expected_wheel_hashes() -> dict[str, str]:
    lock = ROOT / "requirements/cutlass-dsl-aarch64.txt"
    expected: dict[str, str] = {}
    package = ""
    for raw_line in lock.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            package = line.split("==", 1)[0].replace("-", "_")
        if "--hash=sha256:" in line and package:
            expected[package] = line.split("--hash=sha256:", 1)[1].rstrip(" \\")
    return expected


_PATCHSET_SHA256_CACHE: dict[tuple[Path, str | None, int | None, tuple[str, ...]], str] = sys.__dict__.setdefault(
    "_spark_patchset_sha256_cache", {}
)


def patchset_sha256(source: dict) -> str:
    commits = tuple(commit["revision"] for commit in source.get("applied_upstream_commits", []))
    series = source.get("patch_series")
    series_path = ROOT / series if series else None
    cacheable = getattr(read_series, "__module__", None) == __name__
    mtime_ns = series_path.stat().st_mtime_ns if cacheable and series_path and series_path.is_file() else None
    key = (ROOT, series, mtime_ns, commits)
    if cacheable and key in _PATCHSET_SHA256_CACHE:
        return _PATCHSET_SHA256_CACHE[key]
    digest = hashlib.sha256()
    for revision in commits:
        digest.update(revision.encode())
        digest.update(b"\0")
    if series_path is not None:
        digest.update(series_path.read_bytes())
        for patch_name in read_series(series_path):
            if patch_name.startswith("TODO-"):
                continue
            digest.update(patch_name.encode())
            digest.update(b"\0")
            digest.update((series_path.parent / patch_name).read_bytes())
    result = digest.hexdigest()
    if cacheable:
        _PATCHSET_SHA256_CACHE[key] = result
    return result


BUILD_ROOT = ROOT / ".work/build"
# Bump when the directory layout or context export changes, so older
# directories are re-prepared.
BUILD_LAYOUT = 2
BUILD_PROJECTS = ("vllm", "nccl")
# Built while the lock lists it: B12X, which spark-ds41f-1025's lock and earlier
# ones list. A lock without it builds an image without B12X (the TileLang runtime
# on Dockerfile stage runtime-core).
LEGACY_BUILD_PROJECTS = ("b12x",)
# Built only when the lock lists them (the TileLang kernel backend); each adds
# its own <name>-source context.
OPTIONAL_BUILD_PROJECTS = ("tilelang", "tile_kernels")
# Built when the lock lists it, independently: sparknet's one-shot RoCE
# collectives, installed on the TileLang runtime.
INDEPENDENT_BUILD_PROJECTS = ("sparknet",)
# The vLLM capability of images whose TileLang family runs sparknet's collectives.
SPARKNET_CAPABILITY = "tilelang-sparknet-collectives"
BUILD_PARTS = ("vllm", "nccl", "cutlass", "wheels")
BUILD_CONTEXTS = ("vllm-source", "nccl-source", "cutlass-source", "cutlass-dsl-wheels", "vllm-deletions")
# The projects a lock may list or leave out, each with a <name>-source context.
LISTED_BUILD_PROJECTS = (*LEGACY_BUILD_PROJECTS, *OPTIONAL_BUILD_PROJECTS, *INDEPENDENT_BUILD_PROJECTS)
# git am with a fixed committer and committer date reproduces the recorded
# patch heads byte for byte.
PATCH_COMMITTER = ("Christopher Owen", "3221756+christopherowen@users.noreply.github.com")
GIB = 1024**3
BUILD_MIN_AVAILABLE_GIB = 64
BUILD_RESERVE_GIB = 24
BUILD_MEMORY_PER_JOB_GIB = 8
BUILD_NVCC_THREADS = 2
BUILD_CPU_RESERVE = 4
BUILD_MAX_JOBS = 10
SMOKE_MEMORY_GIB = 16
BUILD_LOCK = Path("/tmp/spark-build.lock")
SMOKE_SCRIPT = """\
import torch

assert torch.cuda.is_available()
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
import vllm.distributed.device_communicators.b12x_roce_all_reduce  # noqa: F401
import vllm.models.deepseek_v4_1.nvidia.dspark  # noqa: F401
import vllm.models.deepseek_v4_1.nvidia.model  # noqa: F401
import vllm.models.deepseek_v4_1.nvidia.vl_model  # noqa: F401
import vllm.v1.attention.backends.mla.b12x_mla_sparse  # noqa: F401
from vllm.models.deepseek_v41 import DeepseekV41ForCausalLM

print("DS4.1 SM12x class:", DeepseekV41ForCausalLM.__module__)
assert DeepseekV41ForCausalLM.__module__.startswith("vllm.models.deepseek_v4_1")
print("GPU import smoke passed")
"""
# A TileLang-only vLLM series (kernel_backend.TILELANG_ONLY) deletes B12X's RoCE
# collectives with the kernel-family switch.
B12X_ROCE_IMPORT = "import vllm.distributed.device_communicators.b12x_roce_all_reduce  # noqa: F401\n"
B12X_MLA_IMPORT = "import vllm.v1.attention.backends.mla.b12x_mla_sparse  # noqa: F401\n"
# An image whose lock does not list B12X: the package must be absent, and the DS4.1
# modules that used it (Engram, the packed head's packer) import without it.
WITHOUT_B12X_SMOKE = """\
import importlib.util

assert importlib.util.find_spec("b12x") is None, "B12X is installed"
import vllm.models.deepseek_v4_1.common.engram  # noqa: F401
import vllm.models.deepseek_v4_1.tilelang.vocab_pack  # noqa: F401

print("B12X is not installed; the DS4.1 modules import without it")
"""


def build_projects(sources: dict) -> tuple[str, ...]:
    """The required projects, then each the lock or inputs list (B12X, TileLang, sparknet)."""
    return BUILD_PROJECTS + tuple(name for name in LISTED_BUILD_PROJECTS if name in sources)


def build_contexts(inputs: dict) -> tuple[str, ...]:
    return BUILD_CONTEXTS + tuple(f"{name}-source" for name in LISTED_BUILD_PROJECTS if name in inputs)


def runtime_stage(inputs: dict) -> str | None:
    """The Dockerfile stage the image ends with; None keeps the B12X runtime."""
    if "b12x" not in inputs and "tilelang" not in inputs:
        raise SystemExit("an image without B12X is the TileLang runtime; the lock must list the TileLang sources")
    if "sparknet" in inputs:
        if "tilelang" not in inputs:
            raise SystemExit("sparknet is installed on the TileLang runtime; the lock must list the TileLang sources")
        return "runtime-tilelang-sparknet"
    return "runtime-tilelang" if "tilelang" in inputs else None


def tilelang_base_stage(inputs: dict) -> str:
    """The Dockerfile stage the TileLang runtime builds on: B12X's while the lock lists it."""
    return "runtime" if "b12x" in inputs else "runtime-core"


def missing_patches(source: dict) -> list[str]:
    """Repository paths of a source's series and patches that do not exist yet."""
    series = source.get("patch_series")
    if not series:
        return []
    series_path = ROOT / series
    if not series_path.is_file():
        return [series]
    return [
        (series_path.parent / name).resolve().relative_to(ROOT.resolve()).as_posix()
        for name in read_series(series_path)
        if not name.startswith("TODO-") and not (series_path.parent / name).is_file()
    ]


def smoke_script(upstreams: dict) -> str:
    """The GPU import smoke, extended for an image with the TileLang sources; for a lock
    without B12X (a TileLang image), it imports no B12X module and checks B12X is absent."""
    if "tilelang" not in upstreams["sources"]:
        return SMOKE_SCRIPT
    version = upstreams["sources"]["tilelang"]["version"]
    manifest = read_json(upstreams["source_manifest"])
    capabilities = manifest["vllm"].get("capabilities", [])
    b12x = "b12x" in upstreams["sources"]
    script = SMOKE_SCRIPT
    if kernel_backend.TILELANG_ONLY in capabilities or not b12x:
        script = script.replace(B12X_ROCE_IMPORT, "")
    if not b12x:
        script = script.replace(B12X_MLA_IMPORT, "")
    script += (
        "import importlib.metadata\n\n"
        "import tilelang\n"
        "import tile_kernels  # noqa: F401\n\n"
        f"assert importlib.metadata.version(\"tilelang\") == {version!r}, tilelang.__version__\n"
        "print(\"TileLang\", tilelang.__version__)\n"
    )
    if kernel_backend.CAPABILITY in capabilities:
        script += "import vllm.models.deepseek_v4_1.tilelang  # noqa: F401\n"
    script += "print(\"TileLang import smoke passed\")\n"
    if "sparknet" in upstreams["sources"]:
        version = upstreams["sources"]["sparknet"]["version"]
        script += (
            "from sparknet.integration.vllm import SparknetOneShotAllReduce  # noqa: F401\n"
            "from sparknet.oneshot._proxy import load\n\n"
            f"assert importlib.metadata.version(\"dgx-spark-networking\") == {version!r}\n"
            "print(\"sparknet proxy ABI\", load().roce_abi_version())\n"
            "print(\"sparknet import smoke passed\")\n"
        )
    if not b12x:
        script += WITHOUT_B12X_SMOKE
    return script


def git_output(directory: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=directory, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def build_inputs(upstreams: dict, parts: list[str] | None = None) -> dict:
    """Everything that determines the build contexts, read from the lock.

    A project whose series lists a patch that does not exist yet stops the
    build. With an explicit list of parts, an unlisted project's inputs are
    recorded as incomplete instead, so the other series can still be prepared.
    """
    manifest = read_json(upstreams["source_manifest"])
    sources = upstreams["sources"]
    listed = [name for name in OPTIONAL_BUILD_PROJECTS if name in sources]
    if listed and len(listed) != len(OPTIONAL_BUILD_PROJECTS):
        raise SystemExit(f"{' and '.join(OPTIONAL_BUILD_PROJECTS)} are built together; the lock lists only {listed[0]}")
    base = sources["vllm_base_image"]
    inputs = {
        "base_image": f"{base['upstream']}@{base['recorded_base_digest']}",
        "base_source_revision": base["source_revision"],
        "cutlass": {key: sources["cutlass"][key] for key in ("upstream", "revision", "release_tag")},
        "cutlass_dsl_wheels": expected_wheel_hashes(),
    }
    for name in build_projects(sources):
        source = sources[name]
        missing = missing_patches(source)
        if missing and (parts is None or name in parts):
            raise SystemExit(
                f"{name}: missing patch {', '.join(missing)}; add it and record its patch head, "
                f"tree and fingerprint in {upstreams['source_manifest']}"
            )
        if missing:
            inputs[name] = {
                "upstream": source["upstream"],
                "tracking_ref": source["tracking_ref"],
                "revision": source["revision"],
                "patch_series": source.get("patch_series"),
                "missing_patches": missing,
            }
            continue
        record = manifest.get(name)
        if record is None:
            raise SystemExit(f"{name}: {upstreams['source_manifest']} has no record")
        patchset = patchset_sha256(source)
        if patchset != record["patchset_sha256"]:
            raise SystemExit(
                f"{name}: the patch series differs from {upstreams['source_manifest']}; "
                "record its new patch head and tree there first"
            )
        inputs[name] = {
            "upstream": source["upstream"],
            "tracking_ref": source["tracking_ref"],
            "revision": source["revision"],
            "patch_series": source.get("patch_series"),
            "patchset_sha256": patchset,
            "patch_head": record["patch_head"],
            "expected_tree": record["expected_tree"],
        }
        if name in (*OPTIONAL_BUILD_PROJECTS, *INDEPENDENT_BUILD_PROJECTS):
            if not source.get("version"):
                raise SystemExit(f"{name}: record its package version in the upstreams lock")
            inputs[name]["version"] = source["version"]
        if source.get("submodules"):
            if not isinstance(record.get("submodules"), dict):
                raise SystemExit(f"{name}: record its submodule commits in {upstreams['source_manifest']}")
            inputs[name]["submodules"] = record["submodules"]
    if sources["vllm"].get("canonical_base"):
        inputs["vllm"]["canonical_upstream"] = sources["vllm"]["canonical_base"]["upstream"]
    return inputs


def build_directory(inputs: dict) -> Path:
    """One directory per set of build inputs: same lock, same directory (a lock that
    lists B12X keeps the name it always had)."""
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    b12x = f"-b12x-{inputs['b12x']['revision'][:12]}" if "b12x" in inputs else ""
    return BUILD_ROOT / f"vllm-{inputs['vllm']['revision'][:12]}{b12x}-{digest[:8]}"


def project_problems(source: Path, record: dict) -> list[str]:
    if not (source / ".git").exists():
        return [f"{source.name}: source checkout is missing"]
    problems = []
    if git_output(source, "status", "--porcelain", "--untracked-files=all"):
        problems.append(f"{source.name}: source checkout has local changes")
    head = git_output(source, "rev-parse", "HEAD")
    tree = git_output(source, "rev-parse", "HEAD^{tree}")
    if tree != record["expected_tree"]:
        problems.append(f"{source.name}: tree {tree}, expected {record['expected_tree']}")
    if head != record["patch_head"]:
        problems.append(f"{source.name}: patch head {head}, expected {record['patch_head']}")
    if record.get("submodules") is not None and submodule_commits(source) != record["submodules"]:
        problems.append(f"{source.name}: submodule checkouts differ from the source manifest")
    return problems


def submodule_commits(source: Path) -> dict[str, str] | None:
    """Each submodule's checked-out commit, or None when one is missing or
    differs from the commit its superproject records."""
    output = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=source, check=True, text=True, stdout=subprocess.PIPE,
    ).stdout
    commits = {}
    for line in output.splitlines():
        if not line.startswith(" "):
            return None
        commit, path = line[1:].split()[:2]
        commits[path] = commit
    return commits


def replace_directory(temporary: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)


def prepare_project(directory: Path, name: str, record: dict) -> None:
    source = directory / "src" / name
    if source.is_dir() and not project_problems(source, record):
        print(f"{name}: reusing {source.relative_to(ROOT)}")
        return
    source.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=source.parent))
    try:
        checked(["git", "init", "--quiet"], cwd=temporary)
        checked(["git", "remote", "add", "upstream", record["upstream"]], cwd=temporary)
        checked(
            ["git", "fetch", "--quiet", "--filter=blob:none", "--no-tags", "upstream", record["revision"]],
            cwd=temporary,
        )
        checked(["git", "switch", "--quiet", "--create", "spark/build", "FETCH_HEAD"], cwd=temporary)
        if record.get("patch_series"):
            series = ROOT / record["patch_series"]
            for patch_name in read_series(series):
                checked(
                    [
                        "git", "-c", f"user.name={PATCH_COMMITTER[0]}",
                        "-c", f"user.email={PATCH_COMMITTER[1]}",
                        "am", "--quiet", "--committer-date-is-author-date",
                        str(series.parent / patch_name),
                    ],
                    cwd=temporary,
                )
        checked(["git", "diff", "--check", record["revision"], "HEAD"], cwd=temporary)
        if record.get("submodules") is not None:
            # The commits the patched superproject records, nested ones
            # included; project_problems compares them with the manifest.
            checked(
                ["git", "submodule", "update", "--quiet", "--init", "--recursive",
                 "--depth", "1", "--jobs", "4"],
                cwd=temporary,
            )
        problems = project_problems(temporary, record)
        if problems:
            raise SystemExit("; ".join(problems))
        replace_directory(temporary, source)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"{name}: prepared {record['patch_head'][:12]} (tree {record['expected_tree'][:12]})")


def prepare_cutlass(directory: Path, record: dict) -> None:
    source = directory / "src/cutlass"
    if (source / ".git").exists() and git_output(source, "rev-parse", "HEAD") == record["revision"]:
        print(f"cutlass: reusing {source.relative_to(ROOT)}")
        return
    source.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".cutlass.", dir=source.parent))
    try:
        checked(["git", "init", "--quiet"], cwd=temporary)
        checked(["git", "remote", "add", "upstream", record["upstream"]], cwd=temporary)
        checked(["git", "fetch", "--quiet", "--depth", "1", "upstream", record["revision"]], cwd=temporary)
        checked(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=temporary)
        if git_output(temporary, "rev-parse", "HEAD") != record["revision"]:
            raise SystemExit("cutlass: fetched revision differs from the lock")
        replace_directory(temporary, source)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"cutlass: prepared {record['release_tag']} ({record['revision'][:12]})")


def wheel_problems(wheel_dir: Path) -> list[str]:
    if not wheel_dir.is_dir():
        return ["cutlass-dsl-wheels: missing"]
    problems = []
    expected = expected_wheel_hashes()
    found: set[str] = set()
    for wheel in sorted(wheel_dir.iterdir()):
        normalized = wheel.name.replace("-", "_").lower()
        package = next(
            (name for name in sorted(expected, key=len, reverse=True)
             if normalized.startswith(name.lower() + "_")),
            None,
        )
        if package is None or not wheel.name.endswith(".whl"):
            problems.append(f"cutlass-dsl-wheels: unlocked file {wheel.name}")
            continue
        found.add(package)
        if sha256(wheel) != expected[package]:
            problems.append(f"cutlass-dsl-wheels: {wheel.name} differs from the lock hash")
    for package in sorted(set(expected) - found):
        problems.append(f"cutlass-dsl-wheels: missing {package}")
    return problems


def prepare_wheels(directory: Path, upstreams: dict) -> None:
    destination = directory / "context/cutlass-dsl-wheels"
    if not wheel_problems(destination):
        print("cutlass-dsl wheels: reusing verified wheels")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".wheels.", dir=destination.parent))
    try:
        checked([
            sys.executable, "-m", "pip", "download", "--quiet", "--dest", str(temporary),
            "--require-hashes", "--no-deps", "--only-binary=:all:",
            "--platform", "manylinux_2_28_aarch64", "--implementation", "cp",
            "--python-version", "3.12", "--abi", "cp312",
            "--requirement", str(ROOT / upstreams["sources"]["cutlass_dsl"]["lockfile"]),
        ])
        problems = wheel_problems(temporary)
        if problems:
            raise SystemExit("; ".join(problems))
        replace_directory(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print("cutlass-dsl wheels: downloaded and verified")


def export_source(source: Path, destination: Path) -> None:
    """Copy a checkout without Git metadata or bytecode, replacing atomically."""
    temporary = destination.with_name(f".{destination.name}.export")
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.copytree(
        source, temporary, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__")
    )
    replace_directory(temporary, destination)


def normalize_modes(root: Path) -> None:
    """Git's two file modes, so contexts do not depend on the host umask."""
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o755)
        else:
            path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)


def vllm_deletions(directory: Path, inputs: dict) -> list[str]:
    """Files the pinned vLLM deletes relative to the base image's revision."""
    source = directory / "src/vllm"
    base = inputs["base_source_revision"]
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{base}^{{commit}}"], cwd=source, stderr=subprocess.DEVNULL
    )
    if present.returncode:
        checked(
            ["git", "fetch", "--quiet", "--filter=blob:none", "--no-tags",
             inputs["vllm"]["canonical_upstream"], base],
            cwd=source,
        )
    output = git_output(source, "diff", "--name-only", "--diff-filter=D", base, "HEAD", "--", "vllm")
    return [path.removeprefix("vllm/") for path in output.splitlines()]


def build_problems(directory: Path, inputs: dict) -> list[str]:
    record_path = directory / "inputs.json"
    if not record_path.is_file():
        return [f"{directory.relative_to(ROOT)} is not prepared; run bin/spark build prepare"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    problems = []
    if record.get("schema") != BUILD_LAYOUT:
        problems.append(f"prepared with build layout {record.get('schema')}, now {BUILD_LAYOUT}")
    if record.get("inputs") != inputs:
        problems.append("inputs.json does not match the lock")
    for name in build_projects(inputs):
        problems.extend(project_problems(directory / "src" / name, inputs[name]))
    for name, digest in record.get("contexts", {}).items():
        context = directory / "context" / name
        if not context.is_dir():
            problems.append(f"{name}: context is missing")
        elif tree_digest(context) != digest:
            problems.append(f"{name}: context changed since it was prepared")
    if sorted(record.get("contexts", {})) != sorted(build_contexts(inputs)):
        problems.append("inputs.json does not record every build context")
    problems.extend(wheel_problems(directory / "context/cutlass-dsl-wheels"))
    if not (directory / "context/cutlass-source/include/cutlass/cutlass.h").is_file():
        problems.append("cutlass-source: include/cutlass/cutlass.h is missing")
    return problems


def command_build_prepare(args: argparse.Namespace) -> int:
    _, _, upstreams = configuration(args)
    parts = args.only.split(",") if args.only else None
    if parts:
        optional = LISTED_BUILD_PROJECTS
        unknown = sorted(set(parts) - set(BUILD_PARTS) - set(optional))
        if unknown:
            raise SystemExit(f"unknown build parts: {', '.join(unknown)}")
        unlisted = [part for part in parts if part in optional and part not in upstreams["sources"]]
        if unlisted:
            raise SystemExit(f"{', '.join(unlisted)}: not listed in the upstreams lock")
    inputs = build_inputs(upstreams, parts)
    directory = build_directory(inputs)
    parts = parts or [*BUILD_PARTS, *build_projects(inputs)]
    if not args.only and not build_problems(directory, inputs):
        print(f"build inputs ready: {directory.relative_to(ROOT)}")
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inputs.json").unlink(missing_ok=True)
    print(f"preparing {directory.relative_to(ROOT)}", flush=True)
    for name in build_projects(inputs):
        if name in parts:
            prepare_project(directory, name, inputs[name])
    if "cutlass" in parts:
        prepare_cutlass(directory, inputs["cutlass"])
    if "wheels" in parts:
        prepare_wheels(directory, upstreams)
    if args.only:
        print("partial preparation; run bin/spark build prepare for the complete inputs")
        return 0
    context = directory / "context"
    for name in (*build_projects(inputs), "cutlass"):
        export_source(directory / "src" / name, context / f"{name}-source")
    deleted = vllm_deletions(directory, inputs)
    deletions = context / "vllm-deletions"
    deletions.mkdir(exist_ok=True)
    (deletions / "deleted.txt").write_text("".join(f"{path}\n" for path in deleted), encoding="utf-8")
    (context / "empty").mkdir(exist_ok=True)
    normalize_modes(context)
    record = {
        "schema": BUILD_LAYOUT,
        "inputs": inputs,
        "contexts": {name: tree_digest(context / name) for name in build_contexts(inputs)},
        "vllm_deletions": len(deleted),
    }
    temporary = directory / ".inputs.json"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.rename(directory / "inputs.json")
    problems = build_problems(directory, inputs)
    if problems:
        for problem in problems:
            print(f"WARN: {problem}")
        return 1
    print(f"build inputs ready: {directory.relative_to(ROOT)} ({len(deleted)} vLLM deletions)")
    return 0


def command_build_check(args: argparse.Namespace) -> int:
    _, _, upstreams = configuration(args)
    inputs = build_inputs(upstreams)
    directory = build_directory(inputs)
    problems = build_problems(directory, inputs)
    if problems:
        for problem in problems:
            print(f"WARN: {problem}")
        return 1
    print(f"build inputs OK: {directory.relative_to(ROOT)}")
    return 0


def available_memory_gib() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / GIB
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read host MemAvailable: {error}") from error
    raise SystemExit("cannot read host MemAvailable")


def compile_jobs(available_gib: float) -> int:
    if available_gib < BUILD_MIN_AVAILABLE_GIB:
        raise SystemExit(
            f"the build needs {BUILD_MIN_AVAILABLE_GIB} GiB MemAvailable to start; "
            f"found {available_gib:.1f} GiB"
        )
    cpu_jobs = max(1, (len(os.sched_getaffinity(0)) - BUILD_CPU_RESERVE) // BUILD_NVCC_THREADS)
    memory_jobs = int((available_gib - BUILD_RESERVE_GIB) // BUILD_MEMORY_PER_JOB_GIB)
    return min(BUILD_MAX_JOBS, cpu_jobs, memory_jobs)


def cancel_process_group(process: subprocess.Popen) -> None:
    for signal_number, seconds in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal_number)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=seconds)
            return
        except subprocess.TimeoutExpired:
            pass


def run_memory_guarded(command: list[str], reserve_gib: float) -> float:
    """Run a command, cancelling it if MemAvailable falls below the reserve."""
    minimum = available_memory_gib()
    process = subprocess.Popen(command, start_new_session=True)
    try:
        while True:
            try:
                result = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                available = available_memory_gib()
                minimum = min(minimum, available)
                if available < reserve_gib:
                    cancel_process_group(process)
                    raise SystemExit(
                        f"canceled: MemAvailable fell to {available:.1f} GiB, "
                        f"below the {reserve_gib} GiB reserve"
                    )
                continue
            if result:
                raise SystemExit(f"{command[0]} {command[1]} failed with exit status {result}")
            return minimum
    except KeyboardInterrupt:
        cancel_process_group(process)
        raise SystemExit("canceled") from None
    except BaseException:
        cancel_process_group(process)
        raise


# Serving containers. A profile names its own (container.name): spark-ds41f from the release
# after spark-ds41f-1022, dsv41-<name> before it (dsv41-karmic-kraken). A guard that keeps the
# GPU from serving, and a start that must not run beside another engine, recognise every such
# name, whatever the profile at hand calls its own.
SERVING_CONTAINER = "spark-ds41f"
LEGACY_SERVING_PREFIX = "dsv41-"


def serving_container(name: str, cluster: dict) -> bool:
    return name in (cluster["container"]["name"], SERVING_CONTAINER) or name.startswith(LEGACY_SERVING_PREFIX)


def serving_filters(cluster: dict) -> list[str]:
    """``docker ps`` filters matching every serving container's name (filters on one key are ORed)."""
    patterns = dict.fromkeys((f"^{cluster['container']['name']}$", f"^{SERVING_CONTAINER}$",
                              f"^{LEGACY_SERVING_PREFIX}"))
    return [part for pattern in patterns for part in ("--filter", f"name={pattern}")]


def require_idle_build_host(cluster: dict) -> None:
    running = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"], check=True, text=True, stdout=subprocess.PIPE
    ).stdout.split()
    if any(serving_container(name, cluster) for name in running):
        raise SystemExit("a DS4.1 service is running on this host; stop it before building")


def image_labels(tag: str) -> dict | None:
    result = subprocess.run(
        ["docker", "image", "inspect", tag], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    if result.returncode:
        return None
    inspection = json.loads(result.stdout)[0]
    return {"id": inspection["Id"], "labels": inspection["Config"].get("Labels") or {}}


def smoke_image(cluster: dict, tag: str, script: str = SMOKE_SCRIPT) -> None:
    require_idle_build_host(cluster)
    available = available_memory_gib()
    if available < SMOKE_MEMORY_GIB + 8:
        raise SystemExit(
            f"the GPU import smoke needs {SMOKE_MEMORY_GIB + 8} GiB MemAvailable; found {available:.1f} GiB"
        )
    subprocess.run(
        [
            "docker", "run", "--rm", "-i", "--gpus", "all", "--ulimit=core=0",
            f"--memory={SMOKE_MEMORY_GIB}g", f"--memory-swap={SMOKE_MEMORY_GIB}g",
            "--entrypoint", "/usr/bin/python3", tag, "-",
        ],
        input=script,
        text=True,
        check=True,
    )


def command_build_image(args: argparse.Namespace) -> int:
    require_named_profile(args)
    cluster, _, upstreams = configuration(args)
    inputs = build_inputs(upstreams)
    directory = build_directory(inputs)
    problems = build_problems(directory, inputs)
    if problems:
        for problem in problems:
            print(f"WARN: {problem}")
        return 1
    if repository_dirty():
        print("WARN: the image records the deployment commit; commit changes first")
        return 1
    tag = args.tag or cluster["container"]["image"]
    context = directory / "context"
    base = upstreams["sources"]["vllm_base_image"]
    deployment_commit = repository_revision()

    def command(jobs: int | str) -> list[str]:
        rendered = [
            "docker", "buildx", "build", "--platform", "linux/arm64", "--network", "host", "--load",
            "--file", str(ROOT / "docker/Dockerfile"), "--tag", tag,
        ]
        for name in build_contexts(inputs):
            rendered += ["--build-context", f"{name}={context / name}"]
        build_args = {
            "BASE_IMAGE": inputs["base_image"],
            "BASE_IMAGE_NAME": base["upstream"],
            "BASE_IMAGE_DIGEST": base["recorded_base_digest"],
            "DEPLOYMENT_COMMIT": deployment_commit,
            "VLLM_COMPILE_JOBS": str(jobs),
            "VLLM_NVCC_THREADS": str(BUILD_NVCC_THREADS),
        }
        for name in build_projects(inputs):
            record = inputs[name]
            prefix = name.upper()
            build_args.update({
                f"{prefix}_UPSTREAM": record["upstream"].removesuffix(".git"),
                f"{prefix}_REF": record["tracking_ref"].removeprefix("refs/heads/"),
                f"{prefix}_REVISION": record["revision"],
                f"{prefix}_PATCH_HEAD": record["patch_head"],
                f"{prefix}_TREE": record["expected_tree"],
            })
            if "version" in record:
                build_args[f"{prefix}_VERSION"] = record["version"]
        stage = runtime_stage(inputs)
        if stage is not None:
            build_args.update(RUNTIME_STAGE=stage, TILELANG_COMPILE_JOBS=str(jobs),
                              TILELANG_BASE_STAGE=tilelang_base_stage(inputs))
        for key, value in build_args.items():
            rendered += ["--build-arg", f"{key}={value}"]
        if os.environ.get("GITHUB_TOKEN"):
            rendered += ["--secret", "id=github_token,env=GITHUB_TOKEN"]
        return rendered + [str(context / "empty")]

    if not args.apply:
        print(shlex.join(command("<sized from MemAvailable>")))
        print("plan only; pass --apply on an idle build host to build and smoke-test")
        return 0
    lock = BUILD_LOCK.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another image build is running on this host") from None
    require_idle_build_host(cluster)
    if image_labels(tag) is not None:
        raise SystemExit(f"{tag} already exists here; pass --tag with a new tag rather than overwrite it")
    available = available_memory_gib()
    jobs = compile_jobs(available)
    print(
        f"building {tag} from {directory.relative_to(ROOT)}: {jobs} compile jobs x "
        f"{BUILD_NVCC_THREADS} NVCC threads, {available:.1f} GiB available, "
        f"cancel below {BUILD_RESERVE_GIB} GiB",
        flush=True,
    )
    started = time.monotonic()
    minimum = run_memory_guarded(command(jobs), BUILD_RESERVE_GIB)
    elapsed = time.monotonic() - started
    image = image_labels(tag)
    if image is None:
        raise SystemExit(f"the build finished but {tag} is not present")
    for name in build_projects(inputs):
        label = source_labels.get(image["labels"], f"{name}.tree")
        if label != inputs[name]["expected_tree"]:
            raise SystemExit(f"{tag}: label {source_labels.key(image['labels'], f'{name}.tree')} is {label!r}")
    print(f"built {tag} {image['id']} in {elapsed:.0f}s; lowest MemAvailable {minimum:.1f} GiB", flush=True)
    smoke_image(cluster, tag, smoke_script(upstreams))
    receipts = directory / "images"
    receipts.mkdir(exist_ok=True)
    receipt = {
        "tag": tag,
        "image_id": image["id"],
        "deployment_commit": deployment_commit,
        "compile_jobs": jobs,
        "elapsed_seconds": round(elapsed),
        "min_available_gib": round(minimum, 1),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (receipts / f"{image['id'].removeprefix('sha256:')[:16]}.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def command_build_smoke(args: argparse.Namespace) -> int:
    cluster, _, upstreams = configuration(args)
    smoke_image(cluster, args.image or cluster["container"]["image"], smoke_script(upstreams))
    return 0
