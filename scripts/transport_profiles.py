"""Materialize named tuning recipes into ordinary, launch-disabled configs.

Native environment names are the tuning API. There is no runtime inheritance:
the generated cluster JSON is the only input to render, doctor and deployment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import topology


DEFAULT_PROFILES = "experiments/2026-10-03-transport-profiles/profiles.json"
GROUPS = {
    "rocenante": {
        "VLLM_ROCE_ALLREDUCE_MAX_SIZE", "VLLM_ROCE_ALLGATHER_MAX_SIZE",
        "B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES", "B12X_ROCE_SPIN_LIMIT",
    },
    "nccl": {
        "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_BUFFSIZE",
        "NCCL_LL128_BUFFSIZE", "NCCL_PROTO", "NCCL_SWITCHLESS_BIDIRECTIONAL",
        "NCCL_MIN_TRAFFIC_PER_CHANNEL", "NCCL_THREAD_THRESHOLDS",
    },
}


def repository_path(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError("profile paths must remain relative to the repository")
    return resolved


def size_bytes(value: str) -> int:
    # The pinned vLLM size options interpret KB/MB/GB as binary units.
    match = re.fullmatch(r"([1-9][0-9]*)(B|KB|MB|GB)?", value)
    if not match:
        raise ValueError(f"invalid positive byte size: {value!r}")
    return int(match[1]) * {None: 1, "B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[match[2]]


def resolve(root: Path, catalog_path: str, name: str) -> tuple[dict, dict]:
    catalog = json.loads(repository_path(root, catalog_path).read_text())
    if catalog.get("schema_version") != 1 or set(catalog) != {"schema_version", "profiles"}:
        raise ValueError("expected tuning catalog schema_version 1 and profiles")
    if name not in catalog["profiles"]:
        raise ValueError(f"unknown tuning profile {name!r}; choose {', '.join(catalog['profiles'])}")
    profile = catalog["profiles"][name]
    fields = {"base_config", "base_sha256", "status", "node_count", "transport", *GROUPS}
    if set(profile) != fields:
        raise ValueError(f"profile fields must be {sorted(fields)}")
    raw = repository_path(root, profile["base_config"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != profile["base_sha256"]:
        raise ValueError("base config changed: review its image, sources and settings before repinning")
    result = json.loads(raw)
    count = profile["node_count"]
    if type(count) is not int or count not in (3, 4):
        raise ValueError("node_count must be 3 or 4")
    if topology.argument(result, "--tensor-parallel-size") != str(count) or topology.transport(result) != profile["transport"]:
        raise ValueError("profile topology must match its pinned base")
    env = result["environment"]
    for group, keys in GROUPS.items():
        if not isinstance(profile[group], dict) or set(profile[group]) != keys:
            raise ValueError(f"{group} must contain exactly {sorted(keys)}")
        for key, value in profile[group].items():
            if value is None:
                env.pop(key, None)
            elif isinstance(value, str) and value.strip():
                env[key] = value
            else:
                raise ValueError(f"{key} must be a nonempty string or null (upstream default)")
    required = {"VLLM_ROCE_ALLREDUCE_MAX_SIZE", "VLLM_ROCE_ALLGATHER_MAX_SIZE",
                "B12X_ROCE_SPIN_LIMIT", "NCCL_BUFFSIZE", "NCCL_LL128_BUFFSIZE", "NCCL_MAX_NCHANNELS"}
    if not required <= env.keys():
        raise ValueError(f"explicit values required for {sorted(required - env.keys())}")
    capacity = size_bytes(env["VLLM_ROCE_ALLREDUCE_MAX_SIZE"])
    gather = size_bytes(env["VLLM_ROCE_ALLGATHER_MAX_SIZE"])
    dispatch = int(env.get("B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES", capacity))
    if capacity % 16 or gather % 16 or dispatch <= 0 or dispatch % 16 or dispatch > capacity:
        raise ValueError("collective limits must be positive multiples of 16; dispatch must not exceed capacity")
    for key in ("NCCL_BUFFSIZE", "NCCL_LL128_BUFFSIZE", "NCCL_MIN_TRAFFIC_PER_CHANNEL",
                "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "B12X_ROCE_SPIN_LIMIT"):
        if key in env and (not env[key].isdigit() or int(env[key]) <= 0):
            raise ValueError(f"{key} must be a positive integer")
    if int(env.get("NCCL_MIN_NCHANNELS", "1")) > int(env["NCCL_MAX_NCHANNELS"]):
        raise ValueError("NCCL minimum channels exceeds maximum")
    if "NCCL_THREAD_THRESHOLDS" in env and not re.fullmatch(r"-?\d+( -?\d+){5}", env["NCCL_THREAD_THRESHOLDS"]):
        raise ValueError("NCCL_THREAD_THRESHOLDS requires six space-separated integers")
    if "NCCL_PROTO" in env and env["NCCL_PROTO"] != "^LL128":
        raise ValueError("these profiles retain NCCL_PROTO=^LL128")
    # These custom controls are absent from the old TP3 image. Never silently
    # pass an ignored experiment variable to that image.
    custom = {"B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES", "NCCL_SWITCHLESS_BIDIRECTIONAL",
              "NCCL_MIN_TRAFFIC_PER_CHANNEL"}
    if count == 3 and custom & env.keys():
        raise ValueError("TP3 base image does not provide the custom TP4 tuning controls")
    if count == 4 and env.get("NCCL_SWITCHLESS_BIDIRECTIONAL") != "2":
        raise ValueError("balanced TP4 requires NCCL_SWITCHLESS_BIDIRECTIONAL=2")
    return profile, result


def materialize(profile: dict, resolved: dict, nodes: dict, nodes_path: str) -> dict:
    if len(nodes.get("nodes", [])) != profile["node_count"]:
        raise ValueError(f"{profile['node_count']}-node tuning profile does not match the node map")
    result = copy.deepcopy(resolved)
    result["nodes_config"] = nodes_path
    heads = [node for node in nodes["nodes"] if node.get("head")]
    if len(heads) != 1:
        raise ValueError("node map must contain exactly one API head")
    result["distributed"]["master_addr"] = heads[0]["management_ip"]
    result["deployment"]["launch_enabled"] = False
    result["deployment"].pop("branch", None)
    errors = topology.problems(result, nodes)
    if errors:
        raise ValueError("\n".join(errors))
    return result
