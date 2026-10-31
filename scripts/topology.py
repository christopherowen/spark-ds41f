"""Serving rules for one GPU per Spark, over a fabric sparknet renders.

Three nodes use sparknet's direct-peer one-shot collectives
(``oneshot-direct``). Four nodes use NCCL's neighbour ring (``nccl-ring``), or
sparknet's one-shot collectives over that ring (``oneshot-ring4``) with NCCL
for larger operations. A switch carries ``oneshot-switched`` or ``nccl-switched``.

The node map, each rank's fabric environment and the live GID check come from
``sparknet.topology`` at the revision the cluster's upstream lock pins
(``scripts/sparknet_topology.py``): one renderer for the recipe and the runtime.
This module adds what serving needs on top: the vLLM settings, the node count
the kernels support, and the parallelism the transport allows.

A configuration names its transport in ``fabric.transport``. Configurations
recorded without one ran B12X's RoCEnante direct-peer collectives
(``rocenante-direct``), retired with RoCEnante's ring4 and mesh4 transports;
doctor refuses them.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import ModuleType

import sparknet_topology

ROOT = Path(__file__).resolve().parents[1]

# vLLM's side of a neighbour ring; sparknet's RING_ENV holds NCCL's.
VLLM_RING_ENV = {
    "VLLM_ENABLE_PCIE_ALLREDUCE": "0",
    "VLLM_DISABLE_PYNCCL": "0",
}
# Node-map fields for this repository's access to a node, not the fabric.
SITE_FIELDS = ("transfer_host",)


def transport(cluster: dict) -> str | None:
    """The configured transport; None when the configuration names none."""
    return cluster.get("fabric", {}).get("transport")


# sparknet's one-shot transports and their one-shot topology. They are always
# on (no enable switch).
ONE_SHOT = {"oneshot-direct": "direct", "oneshot-ring4": "ring4", "oneshot-switched": "direct"}
TRANSPORTS = (*ONE_SHOT, "nccl-ring", "nccl-switched")

def switched(mode: str | None) -> bool:
    return mode in ("oneshot-switched", "nccl-switched")


def one_shot_topology(mode: str | None) -> str | None:
    return ONE_SHOT.get(mode)


def neighbour_ring(mode: str | None) -> bool:
    """The four-node cable loop: each rank reaches only its two neighbours."""
    return mode == "nccl-ring" or one_shot_topology(mode) == "ring4"


def is_sparknet(mode: str | None) -> bool:
    return mode in ONE_SHOT


def argument(cluster: dict, flag: str) -> str | None:
    args = cluster.get("serve_args", [])
    if flag not in args:
        return None
    i = args.index(flag) + 1
    return args[i] if i < len(args) else None


def set_argument(cluster: dict, flag: str, value: str) -> None:
    args = cluster["serve_args"]
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args.extend([flag, value])


def fabric(cluster: dict) -> ModuleType:
    """``sparknet.topology`` at the revision the cluster's upstream lock pins."""
    path = ROOT / cluster.get("upstreams_config", "upstreams.lock.json")
    return sparknet_topology.package(json.loads(path.read_text()))


def fabric_node(node: dict) -> dict:
    return {key: value for key, value in node.items() if key not in SITE_FIELDS}


def fabric_map(nodes: dict) -> dict:
    return {**nodes, "nodes": [fabric_node(node) for node in nodes.get("nodes", [])]}


def node_environment(cluster: dict, node: dict) -> dict[str, str]:
    base = {key: str(value) for key, value in cluster["environment"].items()}
    mode = transport(cluster)
    if mode in TRANSPORTS:
        env = fabric(cluster).render.node_environment({"nodes": [fabric_node(node)]}, fabric_node(node),
                                                      transport=mode, base=base)
    else:
        env = base
    env["VLLM_HOST_IP"] = node["management_ip"]
    if switched(mode):
        env["SPARK_DEPLOYMENT_TRANSPORT"] = str(mode)
    return env


def hcas(cluster: dict, node: dict) -> list[str]:
    """The node's RDMA devices for the configured transport."""
    return fabric(cluster).nodes.nccl_hcas(fabric_node(node), transport(cluster))


def problems(cluster: dict, nodes: dict) -> list[str]:
    errors = []
    entries = nodes.get("nodes", [])
    count = len(entries)
    if count not in (3, 4):
        errors.append(f"deployment requires 3 or 4 nodes, got {count}")
    mode = transport(cluster)
    settings = cluster.get("environment", {})
    if mode is None:
        errors.append("fabric.transport is required; a configuration without one ran B12X's "
                      "rocenante-direct, which is retired (docs/switchless-topology.md)")
    elif mode not in TRANSPORTS:
        errors.append(f"unknown fabric transport {mode!r}")
    else:
        sparknet = fabric(cluster)
        errors.extend(sparknet.nodes.problems(fabric_map(nodes), mode))
        errors.extend(sparknet.render.environment_problems(settings, mode, count))
        if hasattr(sparknet.render, "base_problems"):
            errors.extend(sparknet.render.base_problems(settings))
    heads = [n for n in entries if n.get("head")]
    if len(heads) == 1 and cluster.get("distributed", {}).get("master_addr") != heads[0].get("management_ip"):
        errors.append("distributed.master_addr must match the API head's management_ip")
    if switched(mode):
        if cluster.get("fabric", {}).get("topology") != "switched":
            errors.append(f"{mode} requires explicit fabric.topology=switched")
        for key in ("NCCL_SWITCHLESS_BIDIRECTIONAL", "NCCL_MIN_TRAFFIC_PER_CHANNEL",
                    "NCCL_THREAD_THRESHOLDS", "NCCL_ALGO", "NCCL_MIN_NCHANNELS",
                    "NCCL_ALGO_PLUGIN", "NCCL_TUNER_PLUGIN", "NCCL_GRAPH_FILE", "NCCL_TOPO_FILE"):
            if key in settings:
                errors.append(f"{mode} must omit stale ring/topology control {key}")
        disabled = "--disable-custom-all-reduce" in cluster.get("serve_args", [])
        if mode == "oneshot-switched" and disabled:
            errors.append(f"{mode} requires custom all-reduce enabled")
        if mode == "nccl-switched" and not disabled:
            errors.append("nccl-switched runs plain NCCL; it requires --disable-custom-all-reduce")
    elif cluster.get("fabric", {}).get("topology") == "switched":
        errors.append("fabric.topology=switched requires an explicit switched transport")
    if mode is not None and not is_sparknet(mode) and any(k.startswith("SPARKNET_ROCE_") for k in settings):
        errors.append(f"{mode} does not run sparknet; SPARKNET_ROCE_* settings do not apply")
    for flag in ("--tensor-parallel-size", "--nnodes"):
        if argument(cluster, flag) != str(count):
            errors.append(f"{flag} must equal the configured node count ({count})")
    try:
        spec = json.loads(argument(cluster, "--speculative-config") or "{}")
        if not isinstance(spec, dict):
            raise ValueError("expected object")
        if spec and spec.get("draft_tensor_parallel_size") != count:
            errors.append(f"draft_tensor_parallel_size must equal the node count ({count})")
    except (ValueError, AttributeError):
        errors.append("--speculative-config must be a JSON object")
    if neighbour_ring(mode):
        for key, value in VLLM_RING_ENV.items():
            if str(settings.get(key)) != value:
                errors.append(f"{mode} requires {key}={value}")
        disabled = "--disable-custom-all-reduce" in cluster.get("serve_args", [])
        if mode == "nccl-ring" and not disabled:
            errors.append("nccl-ring requires --disable-custom-all-reduce")
        if is_sparknet(mode) and disabled:
            errors.append(f"{mode} requires custom all-reduce enabled")
        if "--enable-expert-parallel" in cluster.get("serve_args", []):
            errors.append(f"{mode} does not support expert-parallel all-to-all")
        for flag in ("--pipeline-parallel-size", "--data-parallel-size",
                     "--decode-context-parallel-size", "--prefill-context-parallel-size"):
            if argument(cluster, flag) not in (None, "1"):
                errors.append(f"{mode} requires {flag}=1 (one TP group in cable order)")
    return errors


def gid_subnet_problems(cluster: dict, node: dict, index: int, output: str) -> list[str]:
    """Compare the map's subnets with the live RoCE GIDs at ``index``."""
    return fabric(cluster).nodes.gid_subnet_problems(fabric_node(node), index, output)


def candidate(base: dict, nodes: dict, nodes_path: str) -> dict:
    result = copy.deepcopy(base)
    count = len(nodes["nodes"])
    result["nodes_config"] = nodes_path
    result["deployment"]["launch_enabled"] = False
    result["deployment"].pop("branch", None)
    ssh_user = nodes.get("ssh_user")
    if result.get("host", {}).get("home") == "/home/swank" and ssh_user not in (None, "", "swank", "u"):
        result["host"]["home"] = f"/home/{ssh_user}"
    result["distributed"]["master_addr"] = next(n["management_ip"] for n in nodes["nodes"] if n.get("head"))
    for flag in ("--tensor-parallel-size", "--nnodes"):
        set_argument(result, flag, str(count))
    spec = json.loads(argument(result, "--speculative-config") or "{}")
    if spec:
        spec["draft_tensor_parallel_size"] = count
        set_argument(result, "--speculative-config", json.dumps(spec, separators=(",", ":")))
    switched_map = any("roce_hcas" in n and "roce_peer_hcas" not in n for n in nodes.get("nodes", []))
    if switched_map and not switched(transport(result)):
        result["fabric"] = {"transport": "oneshot-switched", "topology": "switched"}
        result.pop("benchmark_reference", None)
        for key in ("NCCL_SWITCHLESS_BIDIRECTIONAL", "NCCL_MIN_TRAFFIC_PER_CHANNEL",
                    "NCCL_THREAD_THRESHOLDS", "NCCL_ALGO", "NCCL_MIN_NCHANNELS",
                    "NCCL_ALGO_PLUGIN", "NCCL_TUNER_PLUGIN", "NCCL_GRAPH_FILE", "NCCL_TOPO_FILE"):
            result["environment"].pop(key, None)
        if "--disable-custom-all-reduce" in result["serve_args"]:
            result["serve_args"].remove("--disable-custom-all-reduce")
    elif count == 4 and not switched(transport(result)):
        result["fabric"] = {"transport": "nccl-ring"}
        result["environment"].update({**fabric(result).render.RING_ENV, **VLLM_RING_ENV,
                                      "NCCL_MIN_NCHANNELS": "1", "NCCL_MAX_NCHANNELS": "1"})
        # The NCCL ring runs no one-shot collectives: the base's sparknet settings go.
        for key in [k for k in result["environment"] if k.startswith("SPARKNET_ROCE_")]:
            del result["environment"][key]
        if "--disable-custom-all-reduce" not in result["serve_args"]:
            result["serve_args"].append("--disable-custom-all-reduce")
    errors = problems(result, nodes)
    if errors:
        raise ValueError("\n".join(errors))
    return result
