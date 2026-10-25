"""Configuration and read-only fabric checks for one GPU per Spark.

Three nodes use sparknet's direct-peer one-shot collectives
(``oneshot-direct``). Four nodes use NCCL's neighbour ring (``nccl-ring``), or
sparknet's one-shot collectives over that ring (``oneshot-ring4``) with NCCL
for larger operations. sparknet reads its own ``SPARKNET_ROCE_*`` names.

A configuration names its transport in ``fabric.transport``. Configurations
recorded without one ran B12X's RoCEnante direct-peer collectives
(``rocenante-direct``), retired with RoCEnante's ring4 and mesh4 transports;
doctor refuses them.
"""

from __future__ import annotations

import copy
import ipaddress
import json


RING_ENV = {
    "VLLM_ENABLE_PCIE_ALLREDUCE": "0",
    "VLLM_DISABLE_PYNCCL": "0",
    "NCCL_ALGO": "Ring",
    "NCCL_MIN_NCHANNELS": "1",
    "NCCL_MAX_NCHANNELS": "1",
    # NCCL 2.30.7 init.cc gates runtimeConn on cuMemSupport. Without this,
    # communicator initialization eagerly connects non-neighbour trees/PAT.
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_CUMEM_HOST_ENABLE": "0",
    "NCCL_RUNTIME_CONNECT": "1",
    "NCCL_PAT_ENABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_COLLNET_ENABLE": "0",
    "NCCL_MNNVL_ENABLE": "0",
    "NCCL_RMA_DISABLE": "1",
    "NCCL_GIN_ENABLE": "0",
    "NCCL_NET": "IB",
    "NCCL_NET_PLUGIN": "none",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_MERGE_NICS": "1",
    "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
    "NCCL_IB_SUBNET_PREFIX_LEN": "24",
    "NCCL_IB_ADDR_FAMILY": "AF_INET",
    "NCCL_IB_ROCE_VERSION_NUM": "2",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "1",
}


def transport(cluster: dict) -> str | None:
    """The configured transport; None when the configuration names none."""
    return cluster.get("fabric", {}).get("transport")


# sparknet's one-shot transports and their one-shot topology. They are always
# on (no enable switch).
ONE_SHOT = {"oneshot-direct": "direct", "oneshot-ring4": "ring4"}
TRANSPORTS = (*ONE_SHOT, "nccl-ring")


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


def node_environment(cluster: dict, node: dict) -> dict[str, str]:
    env = {key: str(value) for key, value in cluster["environment"].items()}
    env["VLLM_HOST_IP"] = node["management_ip"]
    mode = transport(cluster)
    if is_sparknet(mode):
        env["SPARKNET_ROCE_PEER_HCAS"] = json.dumps(node["roce_peer_hcas"], separators=(",", ":"))
        env["SPARKNET_ROCE_TOPOLOGY"] = one_shot_topology(mode)
    if neighbour_ring(mode):
        hcas = sorted({h for route in node["roce_peer_hcas"].values() for h in route})
        # '=' makes NCCL match device names exactly rather than by prefix.
        env["NCCL_IB_HCA"] = "=" + ",".join(hcas)
    if "roce_gid_index" in node:
        env["NCCL_IB_GID_INDEX"] = str(node["roce_gid_index"])
        if is_sparknet(mode):
            env["SPARKNET_ROCE_GID_INDEX"] = str(node["roce_gid_index"])
    return env


def problems(cluster: dict, nodes: dict) -> list[str]:
    errors = []
    entries = nodes.get("nodes", [])
    count = len(entries)
    ranks = [n.get("rank") for n in entries]
    if count not in (3, 4):
        errors.append(f"switchless deployment requires 3 or 4 nodes, got {count}")
    if any(type(r) is not int for r in ranks) or sorted(ranks) != list(range(count)):
        return errors + [f"node ranks must be contiguous 0..{count - 1}, got {ranks}"]
    by_rank = {n["rank"]: n for n in entries}
    for field in ("name", "management_ip"):
        values = [n.get(field) for n in entries]
        if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != count:
            errors.append(f"nodes must have distinct nonempty {field} values")
    heads = [n for n in entries if n.get("head")]
    if len(heads) != 1 or heads[0]["rank"] != 0:
        errors.append("exactly one API head is required, at rank 0")
    elif cluster.get("distributed", {}).get("master_addr") != heads[0].get("management_ip"):
        errors.append("distributed.master_addr must match the API head's management_ip")
    mode = transport(cluster)
    if mode is None:
        errors.append("fabric.transport is required; a configuration without one ran B12X's "
                      "rocenante-direct, which is retired (docs/switchless-topology.md)")
    elif mode not in TRANSPORTS:
        errors.append(f"unknown fabric transport {mode!r}")
    if count == 4 and not neighbour_ring(mode):
        errors.append("four-node switchless fabric requires nccl-ring or oneshot-ring4")
    if one_shot_topology(mode) == "ring4" and count != 4:
        errors.append(f"{mode} requires exactly four nodes")
    settings = cluster.get("environment", {})
    if is_sparknet(mode):
        if settings.get("SPARKNET_ROCE_TOPOLOGY", one_shot_topology(mode)) != one_shot_topology(mode):
            errors.append("SPARKNET_ROCE_TOPOLOGY must match the selected transport")
    elif mode == "nccl-ring" and any(k.startswith("SPARKNET_ROCE_") for k in settings):
        errors.append("nccl-ring does not run sparknet; SPARKNET_ROCE_* settings do not apply")
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
        env = cluster.get("environment", {})
        required_env = dict(RING_ENV)
        # Four-node qualification covers these initialized channel counts on
        # neighbour rings. NCCL may use fewer channels for an individual call.
        # Keep one source of truth: the native NCCL environment settings.
        channel_keys = ("NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS")
        allowed_channels = ("1", "2", "4", "8") if count == 4 else ("1",)
        channel_values = [str(env.get(key)) for key in channel_keys]
        if channel_values[0] not in allowed_channels or channel_values[0] != channel_values[1]:
            errors.append(
                f"{mode} requires matching NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS "
                f"in {', '.join(allowed_channels)}"
            )
        for key in channel_keys:
            required_env.pop(key)
        for key, value in required_env.items():
            if str(env.get(key)) != value:
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
        for key in ("NCCL_ALGO_PLUGIN", "NCCL_TUNER_PLUGIN", "NCCL_GRAPH_FILE", "NCCL_TOPO_FILE"):
            if env.get(key):
                errors.append(f"{mode} cannot override topology/algorithm through {key}")
    networks = {}
    for rank, node in by_rank.items():
        expected = {(rank - 1) % count, (rank + 1) % count} if neighbour_ring(mode) else set(ranks) - {rank}
        routes = node.get("roce_peer_hcas")
        if not isinstance(routes, dict) or set(routes) != {str(p) for p in expected}:
            errors.append(f"{node.get('name')}: roce_peer_hcas must name peers {sorted(expected)} in cable/rank order")
            continue
        all_hcas = []
        for peer, hcas in routes.items():
            if not isinstance(hcas, list) or len(hcas) not in (1, 2) or any(not isinstance(h, str) or not h for h in hcas):
                errors.append(f"{node['name']}: peer {peer} needs one or two HCA names")
                continue
            all_hcas.extend(hcas)
            reverse = by_rank[int(peer)].get("roce_peer_hcas", {})
            back = reverse.get(str(rank)) if isinstance(reverse, dict) else None
            if not isinstance(back, list) or len(back) != len(hcas):
                errors.append(f"{node['name']}: link to rank {peer} must have reciprocal stripe counts")
            if neighbour_ring(mode):
                for lane, hca in enumerate(hcas):
                    raw = node.get("roce_subnets", {}).get(hca)
                    try:
                        net = ipaddress.IPv4Network(raw, strict=True)
                        if net.prefixlen != 24:
                            raise ValueError("expected /24")
                    except (ValueError, TypeError, ipaddress.AddressValueError):
                        errors.append(f"{node['name']}: roce_subnets[{hca}] must name its cable's IPv4 /24 network")
                        continue
                    networks.setdefault(str(net), []).append((rank, int(peer), hca, lane))
        if len(set(all_hcas)) != len(all_hcas):
            errors.append(f"{node['name']}: switchless links must use distinct local HCAs")
        if mode in ONE_SHOT and len({len(v) for v in routes.values() if isinstance(v, list)}) != 1:
            errors.append(f"{node['name']}: one-shot collectives require equal stripe counts for every peer")
        gid = node.get("roce_gid_index", 3)
        if type(gid) is not int or gid < 0:
            errors.append(f"{node['name']}: roce_gid_index must be a nonnegative integer")
    if mode in ONE_SHOT:
        widths = {len(v) for n in entries if isinstance(n.get("roce_peer_hcas"), dict) for v in n["roce_peer_hcas"].values() if isinstance(v, list)}
        if len(widths) != 1:
            errors.append("one-shot collectives require one common stripe count across all ranks")
    for net, ends in networks.items():
        if len(ends) != 2 or ends[0][:2] != ends[1][:2][::-1]:
            errors.append(f"fabric subnet {net} must connect exactly the two declared neighbour endpoints")
        elif is_sparknet(mode) and ends[0][3] != ends[1][3]:
            errors.append(f"fabric subnet {net}: one-shot endpoints must use the same stripe position")
    return errors


def gid_subnet_problems(node: dict, index: int, output: str) -> list[str]:
    """Compare configured cable subnets with the actual selected RoCE GIDs."""
    actual = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[1] == str(index) and fields[2] == "RoCEv2":
            try:
                actual[fields[0]] = ipaddress.IPv6Address(fields[3]).ipv4_mapped
            except ipaddress.AddressValueError:
                pass
    errors = []
    for hca, subnet in node.get("roce_subnets", {}).items():
        ip = actual.get(hca)
        if ip is None or ip not in ipaddress.IPv4Network(subnet):
            errors.append(f"{node['name']}: {hca} GID {index} is {ip}, expected cable subnet {subnet}; check the link's address and GID index")
    return errors


def candidate(base: dict, nodes: dict, nodes_path: str) -> dict:
    result = copy.deepcopy(base)
    count = len(nodes["nodes"])
    result["nodes_config"] = nodes_path
    result["deployment"]["launch_enabled"] = False
    result["deployment"].pop("branch", None)
    result["distributed"]["master_addr"] = next(n["management_ip"] for n in nodes["nodes"] if n.get("head"))
    for flag in ("--tensor-parallel-size", "--nnodes"):
        set_argument(result, flag, str(count))
    spec = json.loads(argument(result, "--speculative-config") or "{}")
    if spec:
        spec["draft_tensor_parallel_size"] = count
        set_argument(result, "--speculative-config", json.dumps(spec, separators=(",", ":")))
    if count == 4:
        result["fabric"] = {"transport": "nccl-ring"}
        result["environment"].update(RING_ENV)
        # The NCCL ring runs no one-shot collectives: the base's sparknet settings go.
        for key in [k for k in result["environment"] if k.startswith("SPARKNET_ROCE_")]:
            del result["environment"][key]
        if "--disable-custom-all-reduce" not in result["serve_args"]:
            result["serve_args"].append("--disable-custom-all-reduce")
    errors = problems(result, nodes)
    if errors:
        raise ValueError("\n".join(errors))
    return result
