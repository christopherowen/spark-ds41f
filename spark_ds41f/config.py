"""Configuration loading, merging, validation and command rendering for spark-ds41f."""

from __future__ import annotations

import argparse
import ast
import base64
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
import struct
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
sys.path.insert(0, str(ROOT / "scripts"))
import source_labels
import kernel_backend
import kernel_readiness
import topology
import recipe as cluster_recipe

def read_json(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        raise SystemExit(f"configuration file does not exist: {relative}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid JSON in {relative}: {error}") from error


def cluster_config_path(args: argparse.Namespace | None = None) -> tuple[Path, str]:
    raw = getattr(args, "cluster_config", None) or DEFAULT_CLUSTER_CONFIG
    relative = Path(raw)
    if relative.is_absolute():
        raise SystemExit("--cluster-config must be relative to the repository root")
    resolved = (ROOT / relative).resolve()
    try:
        canonical = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise SystemExit("--cluster-config must remain inside the repository") from error
    if not resolved.is_file():
        raise SystemExit(f"cluster configuration does not exist: {canonical}")
    return resolved, canonical


# The site's nodes: names, ranks, management IPs and RoCE cabling. Each site
# keeps its own copy (git-ignored), started from config/nodes.example.json;
# cluster sync copies the head node's file to every node.
NODES_FILE = "config/nodes.json"
NODE_MAP_EXAMPLES = {
    NODES_FILE: "config/nodes.example.json",
    "config/nodes-ring4.local.json": "config/examples/nodes-ring4.json",
    "config/nodes-switched.local.json": "config/examples/nodes-switched.json",
    "config/nodes-switched3.local.json": "config/examples/nodes-switched3.json",
    "config/nodes-switched5.local.json": "config/examples/nodes-switched5.json",
    "config/nodes-switched6.local.json": "config/examples/nodes-switched6.json",
    "config/nodes-switched7.local.json": "config/examples/nodes-switched7.json",
    "config/nodes-switched8.local.json": "config/examples/nodes-switched8.json",
}


def missing_nodes_message(relative: str | None = None) -> str:
    if relative is None:
        return (
            "no site node map is configured: copy one of the site templates and edit it for your nodes "
            "(docs/setup.md, Site configuration):\n"
            "  4-node switched: cp config/examples/nodes-switched.json config/nodes-switched.local.json\n"
            "  4-node ring:     cp config/examples/nodes-ring4.json config/nodes-ring4.local.json\n"
            "  3-node triangle: cp config/nodes.example.json config/nodes.json"
        )
    example = NODE_MAP_EXAMPLES.get(relative)
    hint = (
        f"run `cp {example} {relative}` and edit it for your site "
        if example
        else "create the node map for your site "
    )
    return f"{relative} is missing: {hint}(docs/setup.md, Site configuration)"


def site_nodes() -> dict:
    if not (ROOT / NODES_FILE).is_file():
        raise SystemExit(missing_nodes_message(NODES_FILE))
    return read_json(NODES_FILE)


def configuration(
    args: argparse.Namespace | None = None,
) -> tuple[dict, dict, dict]:
    """The concrete cluster configuration, its node map and its upstream lock.

    A recipe (``config/cluster.json``) resolves to the shape the site is cabled
    as (``config/site.json``), or to ``--shape``.
    """
    cluster_path, canonical = cluster_config_path(args)
    try:
        cluster = json.loads(cluster_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid JSON in {canonical}: {error}") from error
    shape_nodes = None
    is_recipe_config = cluster_recipe.is_recipe(cluster)
    is_overlay_config = cluster_recipe.is_overlay(cluster)
    overlay_sets_master = is_overlay_config and "master_addr" in cluster.get("set", {}).get("distributed", {})
    explicit_shape = getattr(args, "shape", None) or cluster_recipe.site_shape(ROOT)
    nodes_override = getattr(args, "nodes_config", None)
    auto_switched = False
    if is_overlay_config:
        cluster = read_cluster(canonical)
    elif is_recipe_config:
        shape = cluster_recipe.chosen_shape(cluster, ROOT, getattr(args, "shape", None))
        if not explicit_shape and not nodes_override and not (ROOT / "config/nodes-ring4.local.json").is_file():
            if (ROOT / "config/nodes-switched.local.json").is_file():
                if "tp4-switched" in cluster.get("shapes", {}):
                    shape = "tp4-switched"
                else:
                    nodes_override = "config/nodes-switched.local.json"
                    auto_switched = True
            elif (ROOT / NODES_FILE).is_file() and "tp3-triangle" in cluster.get("shapes", {}):
                shape = "tp3-triangle"
            else:
                raise SystemExit(missing_nodes_message())
        if shape is None:
            raise SystemExit(
                f"{canonical} serves {', '.join(sorted(cluster['shapes']))}; name this site's cabling with "
                f"`bin/spark site shape NAME --apply` ({cluster_recipe.SITE_FILE}) or pass --shape"
            )
        shape_nodes = int(cluster_recipe.shape_entry(cluster, shape)["nodes"])
        cluster = cluster_recipe.resolve(cluster, shape)
    nodes_relative = nodes_override or cluster.get("nodes_config", NODES_FILE)
    nodes_path = repository_config_path(nodes_relative)
    upstreams_relative = cluster.get("upstreams_config", "upstreams.lock.json")
    upstreams_path = repository_config_path(upstreams_relative)
    if nodes_path == ROOT / NODES_FILE:
        nodes = site_nodes()
    else:
        try:
            nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise SystemExit(missing_nodes_message(nodes_relative)) from None
        except json.JSONDecodeError as error:
            raise SystemExit(f"invalid JSON in {nodes_relative}: {error}") from error
    if not upstreams_path.is_file():
        raise SystemExit(f"upstreams lock does not exist: {upstreams_relative}")
    if shape_nodes is not None and not nodes_override and len(nodes["nodes"]) != shape_nodes:
        raise SystemExit(
            f"shape {cluster['shape']} needs {shape_nodes} nodes and its node map "
            f"{cluster.get('nodes_config', NODES_FILE)} has {len(nodes['nodes'])}"
        )
    if auto_switched:
        launch_enabled = cluster.get("deployment", {}).get("launch_enabled", True)
        branch = cluster.get("deployment", {}).get("branch")
        try:
            cluster = topology.candidate(cluster, nodes, "config/nodes-switched.local.json")
        except (ValueError, StopIteration) as error:
            raise SystemExit(f"invalid topology: {error}") from error
        cluster["deployment"]["launch_enabled"] = launch_enabled
        if branch is not None:
            cluster["deployment"]["branch"] = branch
    ssh_user = nodes.get("ssh_user")
    if cluster.get("host", {}).get("home") == "/home/swank" and ssh_user not in (None, "", "swank", "u"):
        cluster["host"]["home"] = f"/home/{ssh_user}"
    if (is_recipe_config or (is_overlay_config and not overlay_sets_master)) and cluster.get("distributed", {}).get("master_addr") == "10.0.1.71":
        heads = [n for n in nodes.get("nodes", []) if n.get("head") and (n.get("control_ip") or n.get("management_ip"))]
        if len(heads) == 1:
            cluster["distributed"]["master_addr"] = heads[0].get("control_ip") or heads[0]["management_ip"]
    return cluster, nodes, json.loads(upstreams_path.read_text(encoding="utf-8"))


def read_cluster(relative: str, shape: str | None = None) -> dict:
    """A cluster configuration file as concrete settings: a recipe resolves to
    ``shape``, else to the site's; an arm overlay resolves its base first."""
    cluster = read_json(relative)
    if cluster_recipe.is_overlay(cluster):
        base = cluster["extends"]
        if cluster_recipe.is_recipe(read_json(base["config"])) and not base.get("shape"):
            raise SystemExit(f"{relative} extends the recipe {base['config']}: name its shape, an arm must not "
                             "follow the site's cabling")
        return cluster_recipe.overlay(cluster, read_cluster(base["config"], base.get("shape")))
    if not cluster_recipe.is_recipe(cluster):
        return cluster
    name = cluster_recipe.chosen_shape(cluster, ROOT, shape)
    if name is None:
        raise SystemExit(f"{relative} is a recipe: name a shape (this site names none in {cluster_recipe.SITE_FILE})")
    return cluster_recipe.resolve(cluster, name)


def repository_config_path(raw: str) -> Path:
    path = Path(raw)
    resolved = (ROOT / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(ROOT.resolve()):
        raise SystemExit("configuration paths must remain relative to the repository root")
    return resolved


def node_by_name(nodes: dict, name: str) -> dict:
    for node in nodes["nodes"]:
        if node["name"] == name:
            return node
    raise SystemExit(f"unknown node {name!r}")


def expected_environment(cluster: dict, node: dict) -> dict[str, str]:
    return topology.node_environment(cluster, node)


def expected_command(cluster: dict, node: dict) -> list[str]:
    command = list(cluster["serve_args"])
    command.extend(
        [
            "--node-rank",
            str(node["rank"]),
            "--master-addr",
            cluster["distributed"]["master_addr"],
            "--master-port",
            cluster["distributed"]["master_port"],
        ]
    )
    if not node["head"]:
        command.append("--headless")
    return command


def expected_binds(cluster: dict) -> list[str]:
    """The --volume values; docker inspect keeps them verbatim in HostConfig.Binds."""
    return [
        f"{host_path(cluster, source)}:{destination}:{mode}"
        for source, destination, mode in cluster["container"]["mounts"]
    ]


def mount_problems(node_name: str, wanted: list[str], actual: list[str] | None) -> list[str]:
    """Extra, missing or different bind mounts, matched by container path."""
    # Docker's Linux bind syntax is source:destination[:mode], with no colons in either path.
    wanted_by_path = {bind.split(":")[1]: bind for bind in wanted}
    actual_by_path = {bind.split(":")[1]: bind for bind in actual or []}
    problems = []
    for path, bind in actual_by_path.items():
        if path not in wanted_by_path:
            problems.append(f"{node_name}: extra mount {bind!r}")
        elif bind != wanted_by_path[path]:
            problems.append(f"{node_name}: mount {bind!r}, expected {wanted_by_path[path]!r}")
    for path, bind in wanted_by_path.items():
        if path not in actual_by_path:
            problems.append(f"{node_name}: missing mount {bind!r}")
    return problems


def environment_problems(
    node_name: str, wanted: dict[str, str], actual: list[str], image: list[str] | None
) -> list[str]:
    """Changed, missing or extra variables in Config.Env.

    docker run adds the --env values to the image's Config.Env, so an entry the
    profile does not set must be the image's own, with the image's value.
    """
    actual_env = dict(item.partition("=")[::2] for item in actual)
    image_env = dict(item.partition("=")[::2] for item in image or [])
    problems = [
        f"{node_name}: {key}={actual_env.get(key)!r}, expected {value!r}"
        for key, value in wanted.items()
        if actual_env.get(key) != value
    ]
    for key, value in actual_env.items():
        if key in wanted or image_env.get(key) == value:
            continue
        if key in image_env:
            problems.append(f"{node_name}: {key}={value!r}, expected the image's {image_env[key]!r}")
        else:
            problems.append(f"{node_name}: extra variable {key}={value!r}")
    return problems


def configured_cpuset(cluster: dict) -> str | None:
    """The CPUs a profile pins serving to (``--cpuset-cpus=`` in docker_run_args), if any."""
    for argument in cluster["container"]["docker_run_args"]:
        if argument.startswith("--cpuset-cpus="):
            return argument.split("=", 1)[1]
    return None


def cpu_list(text: str) -> set[int]:
    """'5-9,15-19' as a set of CPU numbers; empty text is the empty set."""
    cpus: set[int] = set()
    for part in filter(None, (piece.strip() for piece in text.split(","))):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def fast_cpus(capacities: dict[int, int]) -> set[int]:
    """CPUs whose cpu_capacity is above the midpoint of the smallest and largest: GB10's
    ten Cortex-X925 (sparknet's "big" rule)."""
    if not capacities:
        return set()
    middle = (min(capacities.values()) + max(capacities.values())) / 2
    return {cpu for cpu, capacity in capacities.items() if capacity > middle}


def cpu_capacities(text: str) -> dict[int, int]:
    """``grep -H . /sys/devices/system/cpu/cpu*/cpu_capacity`` output as {cpu: capacity}."""
    out = {}
    for line in text.splitlines():
        match = re.search(r"cpu(\d+)/cpu_capacity:(\d+)$", line.strip())
        if match:
            out[int(match.group(1))] = int(match.group(2))
    return out


def cpuset_problems(node_name: str, configured: str | None, live: str, capacities: dict[int, int]) -> list[str]:
    """A pinned profile must run pinned, on exactly the node's fast cores."""
    if configured is None:
        return []
    problems = []
    if cpu_list(live) != cpu_list(configured):
        problems.append(f"{node_name}: container runs on CPUs {live or 'all'}, the profile pins {configured}")
    fast = fast_cpus(capacities)
    if fast and cpu_list(configured) != fast:
        problems.append(f"{node_name}: --cpuset-cpus={configured} is not this node's fast cores "
                        f"({','.join(map(str, sorted(fast)))})")
    return problems


def rendered_docker_command(cluster: dict, node: dict) -> list[str]:
    container = cluster["container"]
    command = ["docker", "run", "-d", "--name", container["name"]]
    command.extend(container["docker_run_args"])
    for key, value in sorted(expected_environment(cluster, node).items()):
        command.extend(["--env", f"{key}={value}"])
    for bind in expected_binds(cluster):
        command.extend(["--volume", bind])
    command.append(container["image"])
    command.extend(expected_command(cluster, node))
    return command


def ssh_target(nodes: dict, node: dict) -> str:
    return f"{nodes['ssh_user']}@{node['name']}"


def transfer_target(nodes: dict, node: dict) -> str:
    """Where bulk copies to and from a node go: the node map's ``transfer_host`` (the
    node's name on a CX7 link to the head, such as ``dgxN-internal``, so large copies
    stay off the LAN switch) when it names one, else the node's ``name``."""
    host = node.get("transfer_host") or node["name"]
    return f"{nodes['ssh_user']}@{host}"


SSH_CONTROL_DIR = Path(tempfile.gettempdir()) / f"spark-ssh-{os.getuid()}"


def ssh_options() -> list[str]:
    """Batch-mode SSH options that reuse one connection per node.

    A fresh connection between the Sparks takes about 0.3 s and a
    multiplexed call about 6 ms; a cluster start makes dozens of calls.
    """
    SSH_CONTROL_DIR.mkdir(mode=0o700, exist_ok=True)
    return [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={SSH_CONTROL_DIR}/%C",
        "-o", "ControlPersist=120",
    ]


def run_ssh(nodes: dict, node: dict, *command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            *ssh_options(),
            ssh_target(nodes, node),
            shlex.join(command),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def remote_shell(nodes: dict, node: dict, script: str) -> subprocess.CompletedProcess[str]:
    return run_ssh(nodes, node, "bash", "-lc", script)


def write_remote_file(nodes: dict, node: dict, path: str, text: str) -> subprocess.CompletedProcess[str]:
    """Replace a file on a node atomically with the given text."""
    quoted = shlex.quote(path)
    script = f"cat > {quoted}.tmp && mv {quoted}.tmp {quoted}"
    return subprocess.run(
        ["ssh", *ssh_options(), ssh_target(nodes, node), shlex.join(["bash", "-c", script])],
        input=text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def repository_path(cluster: dict) -> str:
    path = cluster["deployment"]["path"]
    if "{checkout}" in path:
        raise SystemExit("deployment.path is the checkout itself: it may name {home}, not {checkout}")
    return path.format(home=cluster["host"]["home"])


def host_path(cluster: dict, template: str) -> str:
    """A node path from the profile: ``{home}`` is ``host.home`` and ``{checkout}`` the
    deployment checkout (``deployment.path``), so what a site keeps in its checkout, such
    as the JIT cache, follows the checkout wherever the site clones it."""
    fields = {"home": cluster["host"]["home"]}
    if "{checkout}" in template:
        fields["checkout"] = repository_path(cluster)
    return template.format(**fields)


# GLP steering: the recipe's ``glp`` block (scripts/recipe.py renders it), the steered model
# name and the GLP file vLLM reads at startup, which every node needs in its own hub cache.
GLP_FIELDS = {
    "repo": re.compile(r"[\w.-]+/[\w.-]+"),
    "revision": re.compile(r"[0-9a-f]{40}"),
    "file": re.compile(r"[\w.-]+\.gguf"),
    "sha256": re.compile(r"[0-9a-f]{64}"),
}


def served_model_names(cluster: dict) -> list[str]:
    arguments = cluster["serve_args"]
    if "--served-model-name" not in arguments:
        return []
    names = []
    for argument in arguments[arguments.index("--served-model-name") + 1:]:
        if argument.startswith("--"):
            break
        names.append(argument)
    return names


def glp_problems(cluster: dict) -> list[str]:
    """The profile's ``glp`` block: its fields well formed, and the steered name served
    after the stock one. Steering set by hand, without the block, is refused."""
    glp = cluster.get("glp")
    environment = cluster.get("environment", {})
    steer = (cluster_recipe.STEER_PATH, cluster_recipe.STEER_MODEL)
    if not glp:
        written = [key for key in steer if key in environment]
        return [f"{', '.join(written)} set without a glp block, which configures GLP steering"] if written else []
    problems = [f"glp.{key} is {glp.get(key)!r}, not a {key} (docs/setup.md, section 5)"
                for key, pattern in GLP_FIELDS.items() if not pattern.fullmatch(str(glp.get(key, "")))]
    name, names = glp.get("served_model_name"), served_model_names(cluster)
    if not name or names.count(name) != 1 or names.index(name) == 0:
        problems.append(
            f"glp.served_model_name {name!r} must be served once, after the stock name "
            f"(--served-model-name {' '.join(names)}): the first is the stock model every tool "
            "and probe uses")
    return problems


# The profile image's own GLP reader (vllm/models/deepseek_v4_1/glp.py: "Check a GLP file
# against this runtime's reader without serving"), run against the file mounted where the
# engine reads it. The engine reads the file while it builds the model, so a vector its
# reader refuses stops the whole engine, the stock model name with it. The reader needs no
# GPU and no network; its --num-layers and --hidden-size defaults are DS4.1-Flash's (40
# layers of 5120, the checkpoint's text_config). The timeout runs in the container, so a
# stuck reader ends its own container.
GLP_READER = ["python3", "-m", "vllm.models.deepseek_v4_1.glp", cluster_recipe.GLP_TARGET]
GLP_READER_TIMEOUT = "120s"
# A GGUF file's metadata precedes its tensors; a GLP file's fits in its first 64 KiB.
GLP_HEADER_BYTES = 65536
_GGUF_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                 10: "<Q", 11: "<q", 12: "<d"}
_GGUF_STRING, _GGUF_ARRAY = 8, 9


def glp_reader_command(cluster: dict, path: str) -> list[str]:
    """The ``docker run`` of the profile image's GLP reader on the host file ``path``: never
    pulled, without GPU or network."""
    return ["docker", "run", "--rm", "--pull=never", "--network=none",
            "--entrypoint=/usr/bin/timeout",
            "--volume", f"{path}:{cluster_recipe.GLP_TARGET}:ro",
            cluster["container"]["image"],
            "--signal=TERM", "--kill-after=10s", GLP_READER_TIMEOUT, *GLP_READER]


def gguf_metadata(data: bytes) -> dict[str, object]:
    """The metadata of a GGUF file's leading bytes (version 2 or 3), as far as they reach."""
    if data[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    offset = 16  # magic, version and tensor count

    def take(fmt: str):
        nonlocal offset
        (value,) = struct.unpack_from(fmt, data, offset)
        offset += struct.calcsize(fmt)
        return value

    def value(kind: int):
        nonlocal offset
        if kind == _GGUF_STRING:
            size = take("<Q")
            if offset + size > len(data):
                raise struct.error("truncated string")
            offset += size
            return data[offset - size:offset].decode("utf-8", "replace")
        if kind == _GGUF_ARRAY:
            item, count = take("<I"), take("<Q")
            return [value(item) for _ in range(count)]
        return take(_GGUF_SCALARS[kind])

    metadata: dict[str, object] = {}
    try:
        for _ in range(take("<Q")):
            key = value(_GGUF_STRING)
            metadata[key] = value(take("<I"))
    except (struct.error, KeyError):
        pass
    return metadata


def glp_file_fields(nodes: dict, node: dict, path: str) -> str:
    """The file's ``glp.spec_version`` and ``glp.hook_point``, as a refusal names them."""
    result = run_ssh(nodes, node, "sh", "-c",
                     f"head -c {GLP_HEADER_BYTES} {shlex.quote(path)} | base64 -w0")
    try:
        metadata = gguf_metadata(base64.b64decode(result.stdout.strip(), validate=True))
    except ValueError:
        return "its GGUF metadata is unreadable"
    return ", ".join(f"{key} {metadata[key]!r}" if key in metadata else f"no {key}"
                     for key in ("glp.spec_version", "glp.hook_point"))


def glp_reader_problems(cluster: dict, nodes: dict, node: dict, path: str) -> list[str]:
    """The profile image's GLP reader on the node's file. An image the node lacks, or a
    Docker the node's user cannot reach, is left to the image checks that follow."""
    image = cluster["container"]["image"]
    result = run_ssh(nodes, node, *glp_reader_command(cluster, path))
    if result.returncode == 0:
        return []
    detail = result.stderr.strip() or result.stdout.strip()
    lines = detail.splitlines() or [f"exit status {result.returncode}"]
    # Docker's and SSH's own errors lead with their reason; the reader's traceback ends with it.
    first, last = lines[0], lines[-1]
    if (result.returncode == 125 and "no such image" in detail.lower()) or "docker.sock" in detail:
        return []
    if result.returncode == 255:
        return [f"{node['name']}: cannot check the GLP steering file with {image}'s reader: {first}"]
    if result.returncode in (125, 126, 127):
        return [f"{node['name']}: cannot run {image}'s GLP reader: {first}"]
    if result.returncode == 124:
        return [f"{node['name']}: {image}'s GLP reader did not finish within {GLP_READER_TIMEOUT}"]
    return [f"{node['name']}: {image} cannot read the GLP steering file {path} "
            f"({glp_file_fields(nodes, node, path)}): {last}; the engine reads the file while it "
            "builds the model, so it would not start, the stock model name included: pin a "
            "vector this image reads, or an image that reads this one"]


def node_glp_problems(cluster: dict, nodes: dict, node: dict) -> list[str]:
    """The ``glp`` file on one node: present at its pinned revision, with the block's sha256,
    and, on the head, one the profile image's own GLP reader accepts."""
    source = cluster_recipe.glp_file(cluster)
    if source is None:
        return []
    path = host_path(cluster, source)
    result = run_ssh(nodes, node, "sha256sum", path)
    if result.returncode == 255:
        return [f"{node['name']}: cannot check the GLP steering file: {result.stderr.strip()}"]
    fix = f"run {cluster_recipe.glp_download(cluster)} (docs/setup.md, section 5)"
    if result.returncode:
        return [f"{node['name']}: GLP steering file is missing or unreadable: {path}; {fix}"]
    digest = result.stdout.split()[0] if result.stdout.split() else ""
    if digest != cluster["glp"]["sha256"]:
        return [f"{node['name']}: GLP steering file {path} has sha256 {digest}, not glp.sha256 "
                f"{cluster['glp']['sha256']}; remove it and {fix}"]
    # Every rank holds the same image and file, so the head's reader speaks for all.
    return glp_reader_problems(cluster, nodes, node, path) if node.get("head") else []


def repository_revision(revision: str = "HEAD") -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise SystemExit(result.stderr.strip() or f"cannot resolve {revision!r}")
    return result.stdout.strip()


def repository_dirty() -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise SystemExit(result.stderr.strip() or "cannot inspect repository status")
    return result.stdout.strip()


_TRACKED_SECRET_CACHE: dict[Path, list[str]] = sys.__dict__.setdefault("_spark_tracked_secret_cache", {})


def tracked_secret_problems() -> list[str]:
    cacheable = getattr(subprocess.run, "__module__", None) == "subprocess"
    if cacheable and ROOT in _TRACKED_SECRET_CACHE:
        return list(_TRACKED_SECRET_CACHE[ROOT])
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        return ["cannot enumerate tracked files for secret checks"]
    paths = [Path(item.decode()) for item in result.stdout.split(b"\0") if item]
    forbidden_names = {".env", "id_rsa", "id_ed25519", "credentials.json"}
    forbidden_suffixes = (".pem", ".key", ".local.env", ".local.json")
    signatures = (
        b"-----BEGIN " + b"PRIVATE KEY-----",
        b"-----BEGIN OPENSSH " + b"PRIVATE KEY-----",
    )
    token_hints = (b"github_pat_", b"ghp_", b"hf_", b"tskey-")
    token_patterns = (
        re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
        re.compile(rb"\bghp_[A-Za-z0-9]{20,}\b"),
        re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
        re.compile(rb"\btskey-[A-Za-z0-9-]{20,}\b"),
    )
    problems: list[str] = []
    for relative in paths:
        if relative.name in forbidden_names or relative.name.endswith(forbidden_suffixes):
            problems.append(f"tracked secret-like path: {relative}")
            continue
        candidate = ROOT / relative
        if not candidate.is_file() or candidate.stat().st_size > 2 * 1024 * 1024:
            continue
        contents = candidate.read_bytes()
        if any(signature in contents for signature in signatures) or (
            any(hint in contents for hint in token_hints)
            and any(pattern.search(contents) for pattern in token_patterns)
        ):
            problems.append(f"tracked file contains a secret-like signature: {relative}")
    if cacheable:
        _TRACKED_SECRET_CACHE[ROOT] = problems
    return list(problems)
