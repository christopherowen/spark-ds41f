"""Sonar, the benchmark client every measurement here runs through.

Sonar (pinned in config/sonar.json) is a separate repository: steady-state
load, the engine grid, model behaviour, prefill and stalls, suites and verdict tables, the
determinism gate, the quick probes and the measurement matrix. This deployment pins one
commit of it and keeps only what is its own: the suites sized to its shapes
(config/suites/), the A/A floors measured on its testbeds (scripts/sonar_table.py), the
chat template's reasoning switch (REASONING_OFF), and the conditions a bench runs under
(cooling, fan curve, memory and temperature watch, the deployment's identity).

The pinned commit is checked out once per host under .work/sonar/<revision>, from the pin's
URL, and verified there; nothing else is fetched. The pin is the checkout's own, not part of
an image or a release's upstreams lock: the benchmark client is the same for every arm a run
compares, whatever image or upstreams_config each arm has. SPARK_DS41F_SONAR names a local
Sonar checkout to use instead, for developing Sonar itself; a run records the revision it
used either way.

Imports stay inside functions: only commands that measure need Sonar, so cluster, doctor
and build commands never fetch it.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "config" / "sonar.json"
OVERRIDE_ENV = "SPARK_DS41F_SONAR"
CACHE = ROOT / ".work" / "sonar"
# Deployment suites (their digests are the recorded references' own); a name that is not
# one of them is a Sonar built-in suite or a JSON file.
SUITES = ROOT / "config" / "suites"
# The deployment's chat template turns reasoning off with this; Sonar sends nothing
# model-specific on its own.
REASONING_OFF = {"chat_template_kwargs": {"thinking": False}}
REASONING_OFF_JSON = json.dumps(REASONING_OFF)
# The bench client's stream collector, by the deployment's name for it (recorded in a run's
# identity as client_stream_policy) and Sonar's.
STREAM_POLICY_ENV = "SPARK_DS41F_BENCH_STREAM_POLICY"
STREAM_POLICIES = {"current": "current", "recorded-1030": "text-chunks"}
# servebench's model barrier read this; Sonar's reads SONAR_MODEL_CONTRACT with a contract of
# its own format. A run that still sets the old name is refused rather than run unguarded.
LEGACY_CONTRACT_ENV = "SPARK_DS41F_SUITE_MODEL_CONTRACT"


def legacy_contract_problem(env: dict | None = None) -> str | None:
    env = os.environ if env is None else env
    if env.get(LEGACY_CONTRACT_ENV) or env.get(LEGACY_CONTRACT_ENV + "_SHA256"):
        return (f"{LEGACY_CONTRACT_ENV} is servebench's model barrier, which Sonar does not read: write a "
                "sonar-model-evidence/1 contract and set SONAR_MODEL_CONTRACT and SONAR_MODEL_CONTRACT_SHA256 "
                "(Sonar's sonar/model_evidence.py), or run the experiment at its base commit")
    return None


class SonarUnavailable(RuntimeError):
    """The pinned Sonar commit could not be checked out."""


def pin(lock: Path = LOCK) -> dict:
    """The pin: upstream URL and full revision."""
    try:
        source = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SonarUnavailable(f"{lock}: {error}") from error
    if lock_problems(source):
        raise SonarUnavailable(f"{lock.relative_to(ROOT) if lock.is_relative_to(ROOT) else lock}: "
                               + "; ".join(lock_problems(source)))
    return source


def lock_problems(source: dict) -> list[str]:
    """What doctor reports about the pin (no network)."""
    problems = []
    if source.get("kind") != "git":
        problems.append("kind must be git")
    if not str(source.get("upstream", "")).startswith("https://"):
        problems.append("upstream must be an https URL")
    revision = str(source.get("revision", ""))
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        problems.append("revision must be a full commit SHA")
    if not source.get("tracking_ref"):
        problems.append("tracking_ref is not recorded")
    return problems


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _verified(path: Path, revision: str) -> bool:
    try:
        return (_git("rev-parse", "HEAD", cwd=path) == revision
                and not _git("status", "--porcelain", "--untracked-files=no", cwd=path))
    except (OSError, subprocess.CalledProcessError):
        return False


# Checkouts this process has verified, by (cache, revision): verified once, not per call.
_VERIFIED: dict[tuple[str, str], Path] = {}


def checkout(lock: Path = LOCK) -> Path:
    """A verified checkout of the pinned commit (fetched on first use), or the override's."""
    override = os.environ.get(OVERRIDE_ENV)
    if override:
        path = Path(override).resolve()
        if not (path / "sonar" / "__main__.py").is_file():
            raise SonarUnavailable(f"{OVERRIDE_ENV}={override} is not a Sonar checkout")
        return path
    source = pin(lock)
    revision = source["revision"]
    path = CACHE / revision
    if _VERIFIED.get((str(CACHE), revision)) == path:
        return path
    if path.is_dir() and _verified(path, revision):
        _VERIFIED[(str(CACHE), revision)] = path
        return path
    if path.exists():
        raise SonarUnavailable(f"{path} is not a clean checkout of {revision}; remove it to fetch again")
    CACHE.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{revision[:12]}-", dir=CACHE))
    try:
        _git("init", "-q", cwd=staging)
        _git("fetch", "-q", "--depth", "1", source["upstream"], revision, cwd=staging)
        _git("checkout", "-q", "--detach", "FETCH_HEAD", cwd=staging)
        if not _verified(staging, revision):
            raise SonarUnavailable(f"{source['upstream']} did not give revision {revision}")
        try:
            staging.rename(path)
        except OSError:
            if not _verified(path, revision):  # another process put a different tree there
                raise
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise SonarUnavailable(f"cannot check out Sonar {revision[:12]} from {source['upstream']}: "
                               f"{str(detail).strip()}") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    _VERIFIED[(str(CACHE), revision)] = path
    return path


def revision(path: Path | None = None) -> dict:
    """The Sonar a run used: its revision, whether it was the lock's, and a dirty flag."""
    path = path or checkout()
    head = _git("rev-parse", "HEAD", cwd=path) if (path / ".git").exists() else None
    pinned = pin()["revision"]
    return {"revision": head, "pinned": head == pinned,
            "dirty": bool(_git("status", "--porcelain", cwd=path)) if head else None}


def module(name: str = "sonar"):
    """Import ``sonar`` (or ``sonar.<name>``) from the pinned checkout."""
    path = str(checkout())
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(name if name == "sonar" or name.startswith("sonar.") else f"sonar.{name}")


def environment(base: dict | None = None) -> dict:
    """An environment for ``python3 -m sonar``: the checkout on PYTHONPATH and the stream
    collector the deployment's policy names."""
    env = dict(os.environ if base is None else base)
    problem = legacy_contract_problem(env)
    if problem:
        raise SonarUnavailable(problem)
    env["PYTHONPATH"] = os.pathsep.join([str(checkout()), *filter(None, [env.get("PYTHONPATH")])])
    policy = env.get(STREAM_POLICY_ENV, "current")
    if policy not in STREAM_POLICIES:
        raise ValueError(f"unsupported stream collector policy: {policy}")
    env["SONAR_STREAM_COLLECTOR"] = STREAM_POLICIES[policy]
    return env


def stream_policy() -> str:
    """The deployment's name for the stream collector this run uses."""
    policy = os.environ.get(STREAM_POLICY_ENV, "current")
    if policy not in STREAM_POLICIES:
        raise ValueError(f"unsupported stream collector policy: {policy}")
    return policy


def argv(*args: str) -> list[str]:
    return [sys.executable, "-m", "sonar", *args]


def run(*args: str, **kwargs) -> int:
    """Run a Sonar command in a child process; its exit code."""
    return subprocess.run(argv(*args), env=environment(), cwd=ROOT, **kwargs).returncode


def suite_name(name: str) -> str:
    """The suite argument Sonar gets: a deployment suite's file, else the name as given."""
    path = SUITES / f"{name}.json"
    return str(path) if path.is_file() else name


def load_suite(name: str) -> dict:
    return module("suite").load(suite_name(name))


def suite_arguments(args: list[str]) -> list[str]:
    """``--suite NAME`` (or ``--suite=NAME``) of config/suites/ becomes that file."""
    out = list(args)
    for index, value in enumerate(out):
        if value == "--suite" and index + 1 < len(out):
            out[index + 1] = suite_name(out[index + 1])
        elif value.startswith("--suite="):
            out[index] = "--suite=" + suite_name(value.split("=", 1)[1])
    return out


# The lean profile's mixed-traffic long prompts have always been this directory's sources.
MIXED_LONG_SOURCE = ROOT / "experiments" / "2026-09-29-determinism"


def mixed_arguments(args: list[str]) -> list[str]:
    if any(value == "--long-source" or value.startswith("--long-source=") for value in args):
        return list(args)
    return [*args, "--long-source", str(MIXED_LONG_SOURCE)]


def exec_command(command: str, args: list[str], *, reasoning_off: bool = False) -> int:
    """A lab shim: ``python3 scripts/sonar_<command>.py URL ARGS...`` runs the pinned Sonar's
    command, with the deployment's reasoning switch unless the arguments name their own."""
    if reasoning_off and "--extra-body" not in args:
        args = [*args, "--extra-body", REASONING_OFF_JSON]
    os.environ.update(environment())
    sys.path.insert(0, str(checkout()))
    main = importlib.import_module("sonar.__main__").main
    return main([command, *args])
