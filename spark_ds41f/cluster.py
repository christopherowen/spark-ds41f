"""Coordinated cluster sync, start, stop, readiness polling and memory guards."""

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

def deployment_order(nodes: dict) -> list[dict]:
    workers = sorted(
        (node for node in nodes["nodes"] if not node["head"]),
        key=lambda node: node["rank"],
    )
    heads = [node for node in nodes["nodes"] if node["head"]]
    return workers + heads


def require_local_deployment_state(cluster: dict, nodes: dict, upstreams: dict) -> str:
    problems, warnings = split_findings(local_doctor(cluster, nodes, upstreams))
    for warning in warnings:
        print(f"WARN: {warning}", file=sys.stderr)
    dirty = repository_dirty()
    if dirty:
        problems.append("deployment requires a clean local repository")
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        raise SystemExit(1)
    return repository_revision()


# The repository was renamed from spark3-vllm-ds41f on 2026-10-05. Configurations
# recorded before then name the former URL, which now holds a pointer repository.
FORMER_REPOSITORY_URLS = {
    "https://github.com/christopherowen/spark3-vllm-ds41f": "https://github.com/christopherowen/spark-ds41f",
}


def normalized_repository_url(url: str) -> str:
    url = url.rstrip("/").removesuffix(".git")
    return FORMER_REPOSITORY_URLS.get(url, url)


def remote_status_command(cluster: dict) -> str:
    repo = Path(repository_path(cluster))
    pathspecs = ["."]
    for source, _, mode in cluster["container"]["mounts"]:
        if mode != "rw":
            continue
        source_path = Path(host_path(cluster, source))
        try:
            relative = source_path.relative_to(repo)
        except ValueError:
            continue
        pathspecs.append(f":(exclude){relative.as_posix()}")
    return shlex.join(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *pathspecs]
    )


def remote_repository_state(
    cluster: dict, nodes: dict, node: dict
) -> tuple[str, str, str]:
    path = shlex.quote(repository_path(cluster))
    result = remote_shell(
        nodes,
        node,
        f"cd {path} && printf '%s\\n' \"$(git rev-parse HEAD)\" && "
        "git remote get-url origin && "
        f"{remote_status_command(cluster)}",
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{node['name']}: cannot inspect checkout: {detail}")
    lines = result.stdout.splitlines()
    if len(lines) < 2:
        raise RuntimeError(f"{node['name']}: checkout returned incomplete identity")
    return lines[0], lines[1], "\n".join(lines[2:]).strip()


def unrecorded_subnet_problems(nodes: dict, node: dict, hcas: list[str]) -> list[str]:
    """A multi-node start checks every RoCE interface against its cable's subnet; an
    interface without a recorded subnet would skip that check, so it blocks the start."""
    if len(nodes["nodes"]) < 2:
        return []
    missing = [hca for hca in hcas if hca not in node.get("roce_subnets", {})]
    if not missing:
        return []
    return [f"{node['name']}: the node map records no cable subnet for {', '.join(missing)}, so a start "
            "cannot check that the cabling is the one this profile was written for; add roce_subnets "
            "(docs/switchless-topology.md)"]


def remote_runtime_problems(
    cluster: dict,
    nodes: dict,
    revision: str,
    *,
    allow_existing: bool,
    cluster_config: str,
) -> list[str]:
    """Pre-launch checks of every node, the nodes concurrently; findings keep node order."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes["nodes"])) as pool:
        findings = pool.map(
            lambda node: node_runtime_problems(
                cluster, nodes, node, revision,
                allow_existing=allow_existing, cluster_config=cluster_config,
            ),
            nodes["nodes"],
        )
        return [problem for node_problems in findings for problem in node_problems]


def mount_test_script(cluster: dict) -> str:
    """One remote script testing every mount source (writable for rw, readable for ro)."""
    lines = []
    for source, _, mode in cluster["container"]["mounts"]:
        source = host_path(cluster, source)
        test = "-w" if mode == "rw" else "-r"
        lines.append(
            f"test {test} {shlex.quote(source)} || printf '%s %s\\n' {mode} {shlex.quote(source)}"
        )
    return "\n".join(lines) + "\ntrue\n"


def node_runtime_problems(
    cluster: dict,
    nodes: dict,
    node: dict,
    revision: str,
    *,
    allow_existing: bool,
    cluster_config: str,
) -> list[str]:
    problems: list[str] = []
    repo = repository_path(cluster)
    name = cluster["container"]["name"]
    try:
        remote_revision, remote_origin, dirty = remote_repository_state(cluster, nodes, node)
    except RuntimeError as error:
        return [str(error)]
    if remote_revision != revision:
        problems.append(
            f"{node['name']}: checkout {remote_revision} differs from deployment {revision}"
        )
    if normalized_repository_url(remote_origin) != normalized_repository_url(
        cluster["deployment"]["repository"]
    ):
        problems.append(
            f"{node['name']}: origin {remote_origin!r} differs from deployment repository"
        )
    if dirty:
        problems.append(f"{node['name']}: checkout has uncommitted files")
    if cluster["host"].get("kernel"):
        policy = json.loads(kernel_policy_path(cluster).read_text())
        inventory = node_kernel_inventory(nodes, node, policy)
        problems.extend(kernel_readiness.profile_problems(
            node["name"], inventory, cluster["host"]["kernel"], policy))
    gid_index, hcas = roce_gid_setting(cluster, node)
    gids = remote_shell(nodes, node, roce_gid_query(hcas))
    problems.extend(roce_gid_problems(node["name"], gid_index, hcas, gids.stdout))
    problems.extend(topology.gid_subnet_problems(cluster, node, gid_index, gids.stdout))
    problems.extend(unrecorded_subnet_problems(nodes, node, hcas))
    doctor = run_ssh(
        nodes,
        node,
        f"{repo}/bin/spark",
        "--cluster-config",
        cluster_config,
        "doctor",
    )
    if doctor.returncode:
        detail = doctor.stdout.strip() or doctor.stderr.strip()
        problems.append(f"{node['name']}: repository doctor failed: {detail}")
    mounts = run_ssh(nodes, node, "sh", "-c", mount_test_script(cluster))
    for line in mounts.stdout.splitlines():
        mode, _, source = line.partition(" ")
        problems.append(f"{node['name']}: {mode} mount source is unavailable: {source}")
    problems.extend(node_glp_problems(cluster, nodes, node))
    image = run_ssh(
        nodes,
        node,
        "docker",
        "image",
        "inspect",
        cluster["container"]["image"],
    )
    if image.returncode:
        problems.append(
            f"{node['name']}: image is unavailable: {cluster['container']['image']}"
        )
    else:
        labels = json.loads(image.stdout)[0]["Config"].get("Labels") or {}
        for key, wanted in cluster["container"]["expected_labels"].items():
            if labels.get(key) != wanted:
                problems.append(
                    f"{node['name']}: image label {key}={labels.get(key)!r}, "
                    f"expected {wanted!r}"
                )
    existing = run_ssh(nodes, node, "docker", "inspect", name)
    if existing.returncode == 0 and not allow_existing:
        problems.append(
            f"{node['name']}: container {name} already exists; use --replace explicitly"
        )
    # --replace replaces only a container of this profile's name: an engine serving under
    # another (an arm from before the rename to spark-ds41f, or after it) would keep the GPU.
    others = run_ssh(nodes, node, "docker", "ps", *serving_filters(cluster),
                     "--format", f'{{{{.Names}}}}\t{{{{.Label "{PROFILE_LABEL}"}}}}')
    for line in others.stdout.splitlines():
        other, _, profile = line.partition("\t")
        if other and other != name:
            problems.append(
                f"{node['name']}: {other} is serving; stop it with its own profile before starting {name}: "
                f"{stop_own_profile({'profile': profile or None})}"
            )
    sudo = run_ssh(nodes, node, "sudo", "-n", "true")
    if sudo.returncode:
        problems.append(f"{node['name']}: passwordless sudo is unavailable for memguard")
    return problems


def command_cluster_sync(args: argparse.Namespace) -> int:
    require_named_profile(args)
    cluster, nodes, upstreams = configuration(args)
    _, config_relative = cluster_config_path(args)
    if args.revision:
        # A named commit or ref may be newer than this checkout's last fetch (a hash just
        # merged, or origin/main): fetch before resolving it.
        checked(["git", "fetch", "--quiet", "--prune", "origin"], cwd=ROOT)
    revision = repository_revision(args.revision or "HEAD")
    path = repository_path(cluster)
    for node in nodes["nodes"]:
        print(f"{node['name']}: fetch origin and detach {path} at {revision}")
    if not args.apply:
        print("plan only; pass --apply to synchronize")
        return 0

    local_revision = require_local_deployment_state(cluster, nodes, upstreams)
    if revision != local_revision and args.revision is None:
        raise SystemExit("resolved deployment revision changed unexpectedly")
    # Only published commits deploy. A configured branch (the promoted config
    # uses main) restricts deployment to that branch; without one, any commit
    # reachable from a branch on origin qualifies, so experiment configs keep
    # working after their branch is merged and deleted.
    branch = cluster["deployment"].get("branch")
    checked(["git", "fetch", "--quiet", "--prune", "origin"], cwd=ROOT)
    containing = [
        ref.removeprefix("origin/")
        for ref in subprocess.run(
            ["git", "branch", "--remotes", "--contains", revision,
             "--format=%(refname:short)"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.split()
        if ref.startswith("origin/") and ref != "origin/HEAD"
    ]
    if branch is not None and branch not in containing:
        raise SystemExit(
            f"refusing to deploy {revision}: it is not published on origin/{branch}"
        )
    if not containing:
        raise SystemExit(
            f"refusing to deploy {revision}: it is not published on any origin branch"
        )

    previous: dict[str, str] = {}
    for node in nodes["nodes"]:
        try:
            old_revision, remote_origin, dirty = remote_repository_state(
                cluster, nodes, node
            )
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        if dirty:
            raise SystemExit(
                f"{node['name']}: checkout is dirty; preserve or remove its changes explicitly"
            )
        if normalized_repository_url(remote_origin) != normalized_repository_url(
            cluster["deployment"]["repository"]
        ):
            raise SystemExit(
                f"{node['name']}: origin {remote_origin!r} is not the configured repository"
            )
        previous[node["name"]] = old_revision

    # Read before any checkout: on the head node the checkout can be ROOT, and
    # moving it off a commit that still tracked the file removes it.
    site_paths = {NODES_FILE, cluster.get("nodes_config", NODES_FILE)}
    if (ROOT / cluster_recipe.SITE_FILE).is_file():
        site_paths.add(cluster_recipe.SITE_FILE)
    site_texts = {relative: repository_config_path(relative).read_text(encoding="utf-8") for relative in site_paths}
    quoted_path = shlex.quote(path)
    fetch_script = (
        f"cd {quoted_path} && git fetch --quiet --prune origin && "
        f"git cat-file -e {shlex.quote(revision)}^{{commit}}"
    )
    for node in nodes["nodes"]:
        result = remote_shell(nodes, node, fetch_script)
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise SystemExit(f"{node['name']}: fetch failed: {detail}")

    changed: list[dict] = []
    try:
        for node in nodes["nodes"]:
            checkout = remote_shell(
                nodes,
                node,
                f"cd {quoted_path} && git checkout --quiet --detach {shlex.quote(revision)}",
            )
            if checkout.returncode:
                detail = checkout.stderr.strip() or checkout.stdout.strip()
                raise RuntimeError(f"{node['name']}: checkout failed: {detail}")
            changed.append(node)
            for relative, site_text in site_texts.items():
                copied = write_remote_file(nodes, node, f"{path}/{relative}", site_text)
                if copied.returncode:
                    detail = copied.stderr.strip() or copied.stdout.strip()
                    raise RuntimeError(f"{node['name']}: cannot copy {relative}: {detail}")
            doctor = run_ssh(
                nodes,
                node,
                f"{path}/bin/spark",
                "--cluster-config",
                config_relative,
                "doctor",
            )
            if doctor.returncode:
                detail = doctor.stdout.strip() or doctor.stderr.strip()
                raise RuntimeError(f"{node['name']}: doctor failed after checkout: {detail}")
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print("rolling changed checkouts back", file=sys.stderr)
        for node in reversed(changed):
            remote_shell(
                nodes,
                node,
                f"cd {quoted_path} && git checkout --quiet --detach "
                f"{shlex.quote(previous[node['name']])}",
            )
        return 1

    print(f"all nodes now use deployment commit {revision}")
    return 0


def stop_memguard(cluster: dict, nodes: dict, node: dict) -> subprocess.CompletedProcess[str]:
    unit = f"{cluster['container']['name']}-memguard.service"
    return run_ssh(nodes, node, "sudo", "-n", "systemctl", "stop", unit)


def memguard_is_active(cluster: dict, nodes: dict, node: dict) -> bool:
    unit = f"{cluster['container']['name']}-memguard.service"
    result = run_ssh(nodes, node, "systemctl", "is-active", "--quiet", unit)
    return result.returncode == 0


MEMORY_SETTLE_INTERVAL_SECONDS = 5
MEMORY_SETTLE_GIB = 0.5
MEMORY_RECOVERY_TIMEOUT_SECONDS = 180


def available_gib(nodes: dict, node: dict) -> float:
    result = run_ssh(nodes, node, "awk", "/^MemAvailable:/{print $2}", "/proc/meminfo")
    try:
        return int(result.stdout.strip()) / 1048576
    except ValueError:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(f"{node['name']}: cannot read MemAvailable: {detail}") from None


def wait_for_memory_recovery(
    cluster: dict,
    nodes: dict,
    stopped: list[dict],
    *,
    sleep=time.sleep,
    clock=time.monotonic,
) -> None:
    """Wait until each node's MemAvailable stops rising after its old engine stopped.

    A stopped engine returns its memory over seconds, not at `docker stop`'s exit: a rank
    launched before then meets a startup memguard armed against memory still being
    returned, and the guard kills a healthy boot (2026-10-25, a TP3 triangle at 1M, twice).
    """
    line = cluster["host"]["startup_memguard_gib"]
    previous: dict[str, float] = {}
    pending = list(stopped)
    deadline = clock() + MEMORY_RECOVERY_TIMEOUT_SECONDS
    while pending:
        waiting = []
        for node in pending:
            now = available_gib(nodes, node)
            before = previous.get(node["name"])
            if before is None or now - before >= MEMORY_SETTLE_GIB:
                previous[node["name"]] = now
                waiting.append(node)
                continue
            if now <= line:
                raise RuntimeError(
                    f"{node['name']}: MemAvailable settled at {now:.2f} GiB after the old engine "
                    f"stopped, at or below the {line} GiB startup memguard; a launch would be killed"
                )
            print(f"{node['name']}: MemAvailable {now:.1f} GiB after the old engine stopped", flush=True)
        pending = waiting
        if not pending:
            return
        if clock() >= deadline:
            still = ", ".join(f"{node['name']} {previous[node['name']]:.1f} GiB" for node in pending)
            raise RuntimeError(
                f"MemAvailable still rising {MEMORY_RECOVERY_TIMEOUT_SECONDS}s after the old engine "
                f"stopped ({still}); refusing to arm the startup memguard"
            )
        sleep(MEMORY_SETTLE_INTERVAL_SECONDS)


def wait_for_api(cluster: dict, nodes: dict) -> None:
    head = next(node for node in nodes["nodes"] if node["head"])
    name = cluster["container"]["name"]
    url = cluster["deployment"]["ready_url"]
    timeout = int(cluster["deployment"]["ready_timeout_seconds"])
    deadline = time.monotonic() + timeout
    unreachable_since: dict[str, float] = {}
    print(f"waiting up to {timeout}s for {head['name']} API readiness", flush=True)
    while time.monotonic() < deadline:
        unavailable = False
        for node in nodes["nodes"]:
            state = run_ssh(
                nodes,
                node,
                "docker",
                "inspect",
                "--format",
                "{{.State.Running}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}",
                name,
            )
            if state.returncode:
                unavailable = True
                if node["name"] not in unreachable_since:
                    unreachable_since[node["name"]] = time.monotonic()
                    detail = state.stderr.strip() or "SSH unavailable"
                    print(
                        f"WARN: {node['name']} temporarily unreachable during "
                        f"startup ({detail}); waiting for an inspectable state",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            if node["name"] in unreachable_since:
                elapsed = int(time.monotonic() - unreachable_since.pop(node["name"]))
                print(
                    f"{node['name']}: control path recovered after {elapsed}s",
                    flush=True,
                )
            if not state.stdout.startswith("true "):
                state_detail = state.stdout.strip() or state.stderr.strip() or "absent"
                logs = run_ssh(nodes, node, "docker", "logs", "--tail", "20", name)
                log_detail = (logs.stdout or logs.stderr).strip()
                if len(log_detail) > 4000:
                    log_detail = log_detail[-4000:]
                raise RuntimeError(
                    f"{node['name']}: {name} stopped before readiness "
                    f"({state_detail})\n{log_detail}"
                )
            if not memguard_is_active(cluster, nodes, node):
                raise RuntimeError(
                    f"{node['name']}: startup memguard is no longer active; "
                    "refusing to continue an unguarded launch"
                )
        if unavailable:
            time.sleep(10)
            continue
        result = run_ssh(
            nodes,
            head,
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            "3",
            url,
        )
        if result.returncode == 0:
            print("API ready; JIT/autotuning startup has completed", flush=True)
            return
        time.sleep(1)
    unreachable = ", ".join(sorted(unreachable_since))
    suffix = f"; unreachable nodes: {unreachable}" if unreachable else ""
    raise RuntimeError(f"API did not become ready within {timeout}s{suffix}")


def memguard_settings(cluster: dict, phase: str, mode: str = "running") -> list[str]:
    """The guard's container, phase, threshold (GiB), interval, container wait
    (seconds), mode and warn line (GiB, 0 for none), from the resolved configuration:
    an arm overlay or a recipe has no such fields of its own, so the guard never reads
    the file. The warn line belongs to the steady guard: the startup guard runs at the
    launch line, above it."""
    host = cluster["host"]
    if phase == "startup":
        values = (host["startup_memguard_gib"], host["startup_memguard_interval_seconds"],
                  host["startup_memguard_wait_seconds"])
    elif phase == "steady":
        values = (host["memguard_gib"], host["memguard_interval_seconds"], 0)
    else:
        raise ValueError(f"memguard phase must be startup or steady, not {phase!r}")
    warn_gib = host.get("memguard_warn_gib", 0) if phase == "steady" else 0
    return [cluster["container"]["name"], phase, *(str(value) for value in values), mode, str(warn_gib)]


def start_memguard(
    cluster: dict,
    nodes: dict,
    node: dict,
    phase: str,
    *,
    prelaunch: bool = False,
) -> None:
    script = f"{repository_path(cluster)}/scripts/memguard-start.sh"
    arguments = [script, *memguard_settings(cluster, phase, "prelaunch" if prelaunch else "running")]
    result = run_ssh(nodes, node, *arguments)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"{node['name']}: cannot arm {phase} memguard: {detail}"
        )
    print(f"{node['name']}: {result.stdout.strip()}", flush=True)


def start_memguards(cluster: dict, nodes: dict, phase: str) -> None:
    for node in nodes["nodes"]:
        start_memguard(cluster, nodes, node, phase)


def preserve_failed_start_logs(
    cluster: dict,
    nodes: dict,
    started: list[dict],
    cluster_config: str,
) -> Path | None:
    """Copy complete container logs locally before rollback removes containers."""
    if not started:
        return None

    config_path = ROOT / cluster_config
    if config_path.parent.parent == ROOT / "experiments":
        base = config_path.parent / "runs" / "failed-starts"
    else:
        base = ROOT / "results" / "private" / "failed-starts"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = base / f"{stamp}-{cluster['container']['name']}"
    destination.mkdir(parents=True, exist_ok=False)

    name = cluster["container"]["name"]
    for node in started:
        state = run_ssh(nodes, node, "docker", "inspect", name)
        logs = remote_shell(
            nodes,
            node,
            f"docker logs --timestamps {shlex.quote(name)} 2>&1",
        )
        content = (
            f"node: {node['name']}\n"
            f"container: {name}\n"
            f"inspect_returncode: {state.returncode}\n"
            "\n[docker inspect]\n"
            f"{state.stdout}{state.stderr}"
            "\n[docker logs]\n"
            f"{logs.stdout}{logs.stderr}"
        )
        (destination / f"{node['name']}.log").write_text(content, encoding="utf-8")
    return destination


def restore_containers(
    cluster: dict,
    nodes: dict,
    backups: dict[str, str],
    started: list[dict],
) -> None:
    name = cluster["container"]["name"]
    for node in nodes["nodes"]:
        stop_memguard(cluster, nodes, node)
    for node in started:
        run_ssh(nodes, node, "docker", "rm", "--force", name)
    for node in deployment_order(nodes):
        backup = backups.get(node["name"])
        if not backup:
            continue
        renamed = run_ssh(nodes, node, "docker", "rename", backup, name)
        if renamed.returncode:
            print(f"WARN: {node['name']}: could not restore container name", file=sys.stderr)
            continue
        try:
            start_memguard(cluster, nodes, node, "startup", prelaunch=True)
        except RuntimeError as error:
            print(
                f"WARN: {error}; did not start restored container unguarded",
                file=sys.stderr,
            )
            continue
        start_result = run_ssh(nodes, node, "docker", "start", name)
        if start_result.returncode:
            stop_memguard(cluster, nodes, node)
            print(f"WARN: {node['name']}: could not restart previous container", file=sys.stderr)
            continue


# The label naming the profile a container was started from, so a later plan can show
# the live engine's own profile and the command that stops it.
PROFILE_LABEL = "spark.cluster-config"
# What "the engine's shape" means: a start that changes any of these is a different
# deployment, not a restart.
SHAPE_FLAGS = (("nodes", "--nnodes"), ("tp", "--tensor-parallel-size"),
               ("context", "--max-model-len"), ("kv", "--kv-cache-memory-bytes"))


def require_named_profile(args: argparse.Namespace) -> None:
    """Every --apply acts on a profile named on the command line, never on the default."""
    if getattr(args, "apply", False) and getattr(args, "cluster_config", None) is None:
        raise SystemExit(
            "--apply changes the nodes, so name the profile: bin/spark --cluster-config PATH ... --apply "
            f"({DEFAULT_CLUSTER_CONFIG} is the default only for read-only commands)"
        )


def engine_shape(command: list[str]) -> dict[str, str | None]:
    """Node count, tensor parallelism, context and KV bytes of a vLLM command."""
    def value(flag: str) -> str | None:
        return command[command.index(flag) + 1] if flag in command[:-1] else None

    shape = {key: value(flag) for key, flag in SHAPE_FLAGS}
    shape["nodes"] = shape["nodes"] or "1"
    shape["tp"] = shape["tp"] or "1"
    return shape


def engine_text(shape: dict[str, str | None]) -> str:
    context, kv = shape.get("context"), shape.get("kv")
    context_text = f"{int(context):,}" if context and context.isdigit() else context or "default"
    kv_text = f"{int(kv) / 2**30:.1f} GiB" if kv and kv.isdigit() else kv or "default"
    return f"{shape['nodes']} nodes, TP{shape['tp']}, context {context_text}, KV {kv_text}"


def head_node(nodes: dict) -> dict:
    return next(node for node in nodes["nodes"] if node["head"])


def live_engine(cluster: dict, nodes: dict) -> dict | None:
    """The engine container on the profile's head node, running or stopped; None if absent.

    Raises RuntimeError when the head node cannot be reached."""
    head = head_node(nodes)
    result = run_ssh(nodes, head, "docker", "inspect", cluster["container"]["name"])
    if result.returncode == 255:
        raise RuntimeError(f"cannot reach {head['name']} to inspect the live engine: {result.stderr.strip()}")
    if result.returncode:
        return None
    inspection = json.loads(result.stdout)[0]
    return {
        "shape": engine_shape(inspection["Config"]["Cmd"] or []),
        "image": inspection["Config"]["Image"],
        "running": bool(inspection["State"]["Running"]),
        "profile": (inspection["Config"].get("Labels") or {}).get(PROFILE_LABEL),
    }


def describe_live(live: dict | None) -> str:
    if live is None:
        return "no engine container on the head node"
    state = "running" if live["running"] else "stopped"
    return (f"{state}: {engine_text(live['shape'])}, image {live['image']}, "
            f"started from {live['profile'] or 'an unrecorded profile'}")


def shape_changes(live: dict | None, planned: dict[str, str | None]) -> list[str]:
    if live is None:
        return []
    return [key for key, _ in SHAPE_FLAGS if live["shape"].get(key) != planned.get(key)]


def stop_own_profile(live: dict) -> str:
    return f"bin/spark --cluster-config {live['profile'] or 'PATH'} cluster stop --apply"


def start_refusal(live: dict | None, planned: dict[str, str | None], change_shape: bool) -> str | None:
    """Why --apply must not start this profile over the live engine, or None."""
    if live is None:
        return None
    if live["running"] and live["shape"]["nodes"] != planned["nodes"]:
        return (f"a {live['shape']['nodes']}-node engine is running and this profile has {planned['nodes']} "
                "nodes; replacing it here would leave its other nodes on a broken engine. Stop it with its "
                f"own profile first: {stop_own_profile(live)}")
    changes = shape_changes(live, planned)
    if changes and not change_shape:
        return (f"this start changes the engine's {', '.join(changes)} ({engine_text(live['shape'])} -> "
                f"{engine_text(planned)}); pass --change-shape if that is intended")
    return None


def launch_command(cluster: dict, node: dict, config_relative: str) -> list[str]:
    """The rank's docker run, labelled with the profile it was started from."""
    command = rendered_docker_command(cluster, node)
    return [*command[:5], "--label", f"{PROFILE_LABEL}={config_relative}", *command[5:]]


def command_cluster_start(args: argparse.Namespace) -> int:
    require_named_profile(args)
    cluster, nodes, upstreams = configuration(args)
    _, config_relative = cluster_config_path(args)
    if args.apply and "shape" in cluster and "benchmark_reference" not in cluster and not args.unqualified_shape:
        raise SystemExit(
            f"no release benchmark qualifies this recipe's image on {cluster['shape']}: bench it in a lab "
            "window first (experiments/2026-10-14-one-recipe/README.md), or pass --unqualified-shape"
        )
    if args.apply and not cluster["deployment"]["launch_enabled"]:
        print(
            "ERROR: this cluster configuration is not authorized for launch",
            file=sys.stderr,
        )
        return 1
    name = cluster["container"]["name"]
    order = deployment_order(nodes)
    planned = engine_shape(expected_command(cluster, head_node(nodes)))
    try:
        live, unreachable = live_engine(cluster, nodes), None
    except RuntimeError as error:
        live, unreachable = None, str(error)
    changes = shape_changes(live, planned)
    print(f"profile: {config_relative} on {', '.join(node['name'] for node in nodes['nodes'])}")
    print(f"live:    {unreachable or describe_live(live)}")
    print(f"plan:    {engine_text(planned)}, image {cluster['container']['image']}")
    print("shape:   " + (f"CHANGES {', '.join(changes)}" if changes else "new engine" if live is None else "unchanged"))
    print("start plan (fail-closed startup memory guards):")
    for node in order:
        print(f"  {node['name']}: pre-arm startup memguard, launch rank {node['rank']}")
        if getattr(args, "verbose", False):
            print(f"    {shlex.join(launch_command(cluster, node, config_relative))}")
    print("  wait for API/JIT readiness, then switch every node to steady memguard")
    refusal = unreachable or start_refusal(live, planned, getattr(args, "change_shape", False))
    if not args.apply:
        if refusal:
            print(f"--apply would refuse: {refusal}")
        print("plan only; pass --apply to start the cluster (--verbose prints each docker command)")
        return 0
    if refusal:
        print(f"ERROR: {refusal}", file=sys.stderr)
        return 1

    revision = require_local_deployment_state(cluster, nodes, upstreams)
    problems = remote_runtime_problems(
        cluster,
        nodes,
        revision,
        allow_existing=args.replace,
        cluster_config=config_relative,
    )
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    backup_suffix = f"pre-{int(time.time())}"
    backups: dict[str, str] = {}
    started: list[dict] = []
    launch_began = time.monotonic()

    def elapsed() -> str:
        return f"+{time.monotonic() - launch_began:.1f}s"

    try:
        for node in nodes["nodes"]:
            stop_memguard(cluster, nodes, node)
            if memguard_is_active(cluster, nodes, node):
                raise RuntimeError(
                    f"{node['name']}: stale memguard is still active; refusing startup"
                )
        if args.replace:
            def replace(node: dict) -> tuple[str | None, str | None]:
                """Stop and keep one node's old container; return (backup, error)."""
                exists = run_ssh(nodes, node, "docker", "inspect", name)
                if exists.returncode:
                    return None, None
                backup = f"{name}-{backup_suffix}"
                stopped = run_ssh(nodes, node, "docker", "stop", "--time", "60", name)
                if stopped.returncode:
                    return None, f"{node['name']}: could not stop {name}"
                renamed = run_ssh(nodes, node, "docker", "rename", name, backup)
                if renamed.returncode:
                    run_ssh(nodes, node, "docker", "start", name)
                    return None, f"{node['name']}: could not preserve {name}"
                return backup, None

            # The nodes stop their old containers concurrently (a vLLM shutdown takes
            # several seconds per node); any failure rolls back every stopped node.
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes["nodes"])) as pool:
                outcomes = list(pool.map(replace, nodes["nodes"]))
            errors = []
            for node, (backup, error) in zip(nodes["nodes"], outcomes):
                if backup:
                    backups[node["name"]] = backup
                if error:
                    errors.append(error)
            if errors:
                raise RuntimeError("; ".join(errors))
            wait_for_memory_recovery(
                cluster, nodes, [node for node in nodes["nodes"] if node["name"] in backups]
            )
        for node in order:
            start_memguard(cluster, nodes, node, "startup", prelaunch=True)
            if not memguard_is_active(cluster, nodes, node):
                raise RuntimeError(
                    f"{node['name']}: pre-armed startup memguard is not active; "
                    "refusing launch"
                )
            result = run_ssh(nodes, node, *launch_command(cluster, node, config_relative))
            if result.returncode:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"{node['name']}: launch failed: {detail}")
            started.append(node)
            print(f"{node['name']}: launched {result.stdout.strip()} ({elapsed()})", flush=True)
        wait_for_api(cluster, nodes)
        print(f"API ready at {elapsed()}", flush=True)
        start_memguards(cluster, nodes, "steady")
    except (RuntimeError, KeyboardInterrupt) as error:
        detail = str(error) if str(error) else "cluster start interrupted"
        print(f"ERROR: {detail}", file=sys.stderr)
        log_directory = preserve_failed_start_logs(
            cluster, nodes, started, config_relative
        )
        if log_directory is not None:
            print(
                f"preserved complete failed-start logs in "
                f"{log_directory.relative_to(ROOT)}",
                file=sys.stderr,
            )
        print("rolling back containers started by this command", file=sys.stderr)
        restore_containers(cluster, nodes, backups, started)
        if backups:
            try:
                wait_for_api(cluster, nodes)
                start_memguards(cluster, nodes, "steady")
            except RuntimeError as rollback_error:
                print(f"WARN: rollback health check failed: {rollback_error}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1

    def remove_backup(node: dict) -> str | None:
        backup = backups.get(node["name"])
        if backup and run_ssh(nodes, node, "docker", "rm", backup).returncode:
            return f"WARN: {node['name']}: retained backup container {backup}"
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes["nodes"])) as pool:
        for warning in pool.map(remove_backup, nodes["nodes"]):
            if warning:
                print(warning)
    print(f"cluster ready; memory guards are active on all nodes ({elapsed()})")
    return 0


def command_cluster_stop(args: argparse.Namespace) -> int:
    require_named_profile(args)
    cluster, nodes, _ = configuration(args)
    name = cluster["container"]["name"]
    action = "stop and remove" if args.remove else "stop"
    covered = engine_shape(expected_command(cluster, head_node(nodes)))["nodes"]
    refusal = None
    try:
        live = live_engine(cluster, nodes)
        print(f"live: {describe_live(live)}")
        if live and live["running"] and live["shape"]["nodes"] != covered:
            refusal = (f"the running engine spans {live['shape']['nodes']} nodes and this profile covers "
                       f"{covered}; stopping only these would leave the rest on a broken engine. Stop it with "
                       f"its own profile: {stop_own_profile(live)}")
    except RuntimeError as error:
        # Stopping is the safe direction: an unreachable head does not block it.
        print(f"WARN: {error}")
    for node in nodes["nodes"]:
        print(f"{node['name']}: {action} exact container {name} and stop its memguard")
    if not args.apply:
        if refusal:
            print(f"--apply would refuse: {refusal}")
        print("plan only; pass --apply to stop the cluster")
        return 0
    if refusal:
        print(f"ERROR: {refusal}", file=sys.stderr)
        return 1

    def stop_node(node: dict) -> tuple[list[str], list[str]]:
        """Stop one node's memguard and container; return (messages, errors)."""
        guard = stop_memguard(cluster, nodes, node)
        if memguard_is_active(cluster, nodes, node):
            detail = guard.stderr.strip() or "unit remains active"
            return [], [f"{node['name']}: could not stop memguard: {detail}"]
        exists = run_ssh(nodes, node, "docker", "inspect", name)
        if exists.returncode:
            return [f"{node['name']}: {name} is already absent"], []
        stopped = run_ssh(nodes, node, "docker", "stop", "--time", "60", name)
        if stopped.returncode:
            return [], [f"{node['name']}: {stopped.stderr.strip()}"]
        if args.remove:
            removed = run_ssh(nodes, node, "docker", "rm", name)
            if removed.returncode:
                return [], [f"{node['name']}: {removed.stderr.strip()}"]
        return [], []

    if getattr(args, "parallel", False):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes["nodes"])) as pool:
            outcomes = list(pool.map(stop_node, nodes["nodes"]))
    else:
        outcomes = [stop_node(node) for node in nodes["nodes"]]
    failed = False
    for messages, errors in outcomes:
        for message in messages:
            print(message)
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
            failed = True
    return int(failed)


def command_render(args: argparse.Namespace) -> int:
    cluster, nodes, _ = configuration(args)
    node = node_by_name(nodes, args.node)
    print(shlex.join(rendered_docker_command(cluster, node)))
    return 0
