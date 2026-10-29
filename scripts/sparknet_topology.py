"""sparknet's ``topology`` package at the revision a cluster's upstream lock pins.

The fabric is rendered and validated by sparknet's own code, the same revision the
image builds, so the recipe and the runtime read one version. ``sparknet.topology``
needs only the standard library. The first use of a revision fetches it from the
lock's upstream into ``~/.cache/spark-ds41f/sparknet/<revision>``; later uses import
it from there without the network.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "spark-ds41f" / "sparknet"
PATHS = ("sparknet/__init__.py", "sparknet/topology")
_LOADED: dict[str, ModuleType] = {}


def locked_source(upstreams: dict) -> tuple[str, str]:
    """The lock's sparknet ``(upstream, revision)``."""
    entry = upstreams["sources"]["sparknet"]
    return entry["upstream"], entry["revision"]


def checkout(upstream: str, revision: str) -> Path:
    """The cached tree of ``revision`` (the package and its topology modules), fetched once."""
    target = CACHE / revision
    if (target / "sparknet" / "topology" / "__init__.py").is_file():
        return target
    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CACHE, prefix=".fetch-") as tmp:
        repo, tree = Path(tmp) / "repo", Path(tmp) / "tree"
        git = ["git", "-c", "advice.detachedHead=false", "-C", str(repo)]
        try:
            subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True, text=True)
            subprocess.run([*git, "fetch", "-q", "--depth", "1", upstream, revision],
                           check=True, capture_output=True, text=True, timeout=120)
            fetched = subprocess.run([*git, "rev-parse", "FETCH_HEAD"], check=True,
                                     capture_output=True, text=True).stdout.strip()
            if fetched != revision:
                raise SystemExit(f"sparknet fetch from {upstream} returned {fetched}, not the locked {revision}")
            archive = subprocess.run([*git, "archive", "--format=tar", "FETCH_HEAD", *PATHS],
                                     check=True, capture_output=True).stdout
            tree.mkdir()
            subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            detail = getattr(error, "stderr", "") or error
            raise SystemExit(f"cannot fetch sparknet {revision} from {upstream} into {CACHE} "
                             f"(the fabric is rendered by the locked sparknet): {str(detail).strip()}") from None
        try:
            os.replace(tree, target)
        except OSError:
            if not (target / "sparknet" / "topology" / "__init__.py").is_file():
                raise
    return target


def package(upstreams: dict) -> ModuleType:
    """``sparknet.topology`` at the lock's revision, imported under its own name."""
    upstream, revision = locked_source(upstreams)
    if revision in _LOADED:
        return _LOADED[revision]
    root = checkout(upstream, revision) / "sparknet" / "topology"
    name = f"_sparknet_topology_{revision[:12]}"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for submodule in ("nodes", "render"):
        importlib.import_module(f"{name}.{submodule}")
    _LOADED[revision] = module
    return module
