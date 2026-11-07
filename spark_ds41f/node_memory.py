"""Warn when a node's CUDA free memory at boot has fallen below its own best.

On GB10 heads, device memory accumulates across container restarts and a
container stop does not release it. The loss is invisible while the host runs:
process RSS stays flat, the container cgroup stays flat, and pinned
(``VM_PFNMAP``/``VM_IO``) mappings account for a couple of megabytes. The first
place it appears is the engine's own ``Initial free memory`` line at the next
boot, and only a reboot clears it.

Measured on one head: every boot over a week reported 109.5-110.4 GiB, and after
``sudo reboot`` with nothing else changed the same node reported 113.28 GiB. The
worker nodes, which start only the TP worker, were unchanged at 115.0 GiB.

Left undetected this makes a healthy node look like a broken candidate: the
startup memory guard kills the engine, and the failure is attributed to the
build that happened to be under test.

The detector is deliberately self-calibrating. It compares each node against the
best value that node itself has reported for the same image, so it needs no
table of expected figures and cannot fire simply because a different candidate
or a different context length was deployed.
"""

import json
import re
import sys
from pathlib import Path

from .config import run_ssh

ROOT = Path(__file__).resolve().parents[1]

BASELINE_RELATIVE = Path("results/private/node-memory-baseline.json")
THRESHOLD_KEY = "node_memory_drop_gib"
DEFAULT_THRESHOLD_GIB = 1.0

_INITIAL_FREE = re.compile(r"Initial free memory ([0-9]+(?:\.[0-9]+)?) GiB")


def baseline_path() -> Path:
    return ROOT / BASELINE_RELATIVE


def read_initial_free_memory(nodes: dict, node: dict, container: str) -> float | None:
    """The last `Initial free memory` a node's container reported, or None."""
    logs = run_ssh(nodes, node, "docker", "logs", container)
    if logs.returncode:
        return None
    found = _INITIAL_FREE.findall(logs.stdout + logs.stderr)
    if not found:
        return None
    try:
        return float(found[-1])
    except ValueError:
        return None


def node_hostname(nodes: dict, node: dict) -> str:
    result = run_ssh(nodes, node, "hostname")
    return result.stdout.strip() if not result.returncode else node["name"]


def load_baseline(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save_baseline(path: Path, baseline: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(baseline, indent=1, sort_keys=True) + "\n")
    except OSError as error:
        print(f"WARN: could not record the node memory baseline: {error}", file=sys.stderr)


def threshold_gib(cluster: dict) -> float:
    value = cluster.get("host", {}).get(THRESHOLD_KEY, DEFAULT_THRESHOLD_GIB)
    return float(value) if isinstance(value, (int, float)) else DEFAULT_THRESHOLD_GIB


def check_nodes(cluster: dict, nodes: dict, stamp: str) -> list[str]:
    """Compare each node's boot-time free memory with its own recorded best.

    Returns human-readable warnings. Never raises: this is advice, not a gate.
    """
    if cluster.get("host", {}).get("node_memory_watch", True) is False:
        return []
    container = cluster["container"]["name"]
    image = cluster.get("container", {}).get("image", "")
    path = baseline_path()
    baseline = load_baseline(path)
    limit = threshold_gib(cluster)
    warnings: list[str] = []
    changed = False
    for node in nodes["nodes"]:
        value = read_initial_free_memory(nodes, node, container)
        if value is None:
            continue
        host = node_hostname(nodes, node)
        record = baseline.get(node["name"]) or {}
        best = record.get("best_gib")
        comparable = record.get("image") == image and record.get("host") == host
        if not comparable or best is None:
            baseline[node["name"]] = {
                "host": host,
                "image": image,
                "best_gib": value,
                "at": stamp,
            }
            changed = True
            continue
        if value > best:
            record.update(best_gib=value, at=stamp)
            baseline[node["name"]] = record
            changed = True
            continue
        drop = best - value
        if drop < limit:
            continue
        warnings.append(
            f"WARN: {node['name']} ({host}): CUDA free memory at boot is "
            f"{value:.2f} GiB, {drop:.2f} GiB below the best seen on this node "
            f"for {image or 'this image'} ({best:.2f} GiB, {record.get('at', 'unknown')}). "
            "A GB10 head accumulates device memory across container restarts that a stop "
            "does not release; it is invisible on a running host and a reboot clears it. "
            "If this node has run many engine restarts, reboot it and start again."
        )
    if changed:
        save_baseline(path, baseline)
    return warnings
