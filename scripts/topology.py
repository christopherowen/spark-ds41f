"""Configuration and read-only fabric checks for one GPU per Spark.

Three nodes use direct-peer RoCEnante. Four nodes use NCCL's neighbour ring
or an explicitly built RoCEnante relay or NIC-forwarded mesh, with NCCL for larger operations.
"""

from __future__ import annotations

import copy
import ipaddress
import json


RING_ENV = {
    "VLLM_ENABLE_ROCE_ALLREDUCE": "0",
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


def transport(cluster: dict) -> str:
    return cluster.get("fabric", {}).get("transport", "rocenante-direct")


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


def mesh_path_specs(rank: int, paths: int = 2) -> list[tuple[int, int]]:
    """Return (interface stripe, intermediate rank) in reciprocal QP path order."""
    if paths not in (2, 4):
        raise ValueError("mesh_paths must be 2 or 4")
    opposite = (rank + 2) % 4
    via = ((rank + 1) % 4, (rank - 1) % 4) if rank < opposite else ((rank - 1) % 4, (rank + 1) % 4)
    specs = [(0, via[0]), (1, via[1])]
    if paths == 4:
        specs += [(0, via[1]), (1, via[0])]
    return specs


def logical_peer_hcas(cluster: dict, node: dict) -> dict:
    """Physical cable routes remain authoritative; add only the opposite QP path."""
    routes = copy.deepcopy(node["roce_peer_hcas"])
    if transport(cluster) == "rocenante-mesh4":
        rank = node["rank"]
        opposite = (rank + 2) % 4
        specs = mesh_path_specs(rank, cluster.get("fabric", {}).get("mesh_paths", 2))
        routes[str(opposite)] = [routes[str(peer)][lane] for lane, peer in specs]
    return routes


def node_environment(cluster: dict, node: dict) -> dict[str, str]:
    env = {key: str(value) for key, value in cluster["environment"].items()}
    env["VLLM_HOST_IP"] = node["management_ip"]
    if transport(cluster) in ("nccl-ring", "rocenante-ring4", "rocenante-mesh4"):
        if transport(cluster) == "nccl-ring":
            env.pop("B12X_ROCE_PEER_HCAS", None)
        else:
            env["B12X_ROCE_PEER_HCAS"] = json.dumps(logical_peer_hcas(cluster, node), separators=(",", ":"))
        hcas = sorted({h for route in node["roce_peer_hcas"].values() for h in route})
        # '=' makes NCCL match device names exactly rather than by prefix.
        env["NCCL_IB_HCA"] = "=" + ",".join(hcas)
    else:
        env["B12X_ROCE_PEER_HCAS"] = json.dumps(logical_peer_hcas(cluster, node), separators=(",", ":"))
    if "roce_gid_index" in node:
        env["NCCL_IB_GID_INDEX"] = str(node["roce_gid_index"])
        env["B12X_ROCE_GID_INDEX"] = str(node["roce_gid_index"])
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
    paths = cluster.get("fabric", {}).get("mesh_paths", 2)
    if paths not in (2, 4) or (paths == 4 and mode != "rocenante-mesh4"):
        errors.append("mesh_paths must be 2, or 4 for rocenante-mesh4")
    if mode not in ("rocenante-direct", "nccl-ring", "rocenante-ring4", "rocenante-mesh4"):
        errors.append(f"unknown fabric transport {mode!r}")
    if count == 4 and mode not in ("nccl-ring", "rocenante-ring4", "rocenante-mesh4"):
        errors.append("four-node switchless fabric requires nccl-ring, rocenante-ring4 or rocenante-mesh4")
    if mode in ("rocenante-ring4", "rocenante-mesh4") and count != 4:
        errors.append(f"{mode} requires exactly four nodes")
    if mode not in ("rocenante-ring4", "rocenante-mesh4") and cluster.get("environment", {}).get("B12X_ROCE_TOPOLOGY", "direct") != "direct":
        errors.append("B12X_ROCE_TOPOLOGY must match the selected transport")
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
    if mode in ("nccl-ring", "rocenante-ring4", "rocenante-mesh4"):
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
        if mode in ("rocenante-ring4", "rocenante-mesh4"):
            required_env.update(VLLM_ENABLE_ROCE_ALLREDUCE="1", B12X_ROCE_TOPOLOGY=mode.removeprefix("rocenante-"))
        for key, value in required_env.items():
            if str(env.get(key)) != value:
                errors.append(f"{mode} requires {key}={value}")
        disabled = "--disable-custom-all-reduce" in cluster.get("serve_args", [])
        if mode == "nccl-ring" and not disabled:
            errors.append("nccl-ring requires --disable-custom-all-reduce")
        if mode in ("rocenante-ring4", "rocenante-mesh4") and disabled:
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
        expected = {(rank - 1) % count, (rank + 1) % count} if mode in ("nccl-ring", "rocenante-ring4", "rocenante-mesh4") else set(ranks) - {rank}
        routes = node.get("roce_peer_hcas")
        if not isinstance(routes, dict) or set(routes) != {str(p) for p in expected}:
            errors.append(f"{node.get('name')}: roce_peer_hcas must name peers {sorted(expected)} in cable/rank order")
            continue
        if mode == "rocenante-mesh4" and any(len(v) != 2 for v in routes.values() if isinstance(v, list)):
            errors.append(f"{node['name']}: rocenante-mesh4 requires two stripes per cable")
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
            if mode in ("nccl-ring", "rocenante-ring4", "rocenante-mesh4"):
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
        if mode in ("rocenante-direct", "rocenante-ring4", "rocenante-mesh4") and len({len(v) for v in routes.values() if isinstance(v, list)}) != 1:
            errors.append(f"{node['name']}: RoCEnante requires equal stripe counts for every peer")
        gid = node.get("roce_gid_index", 3)
        if type(gid) is not int or gid < 0:
            errors.append(f"{node['name']}: roce_gid_index must be a nonnegative integer")
    if mode in ("rocenante-direct", "rocenante-ring4", "rocenante-mesh4"):
        widths = {len(v) for n in entries if isinstance(n.get("roce_peer_hcas"), dict) for v in n["roce_peer_hcas"].values() if isinstance(v, list)}
        if len(widths) != 1:
            errors.append("RoCEnante requires one common stripe count across all ranks")
    for net, ends in networks.items():
        if len(ends) != 2 or ends[0][:2] != ends[1][:2][::-1]:
            errors.append(f"fabric subnet {net} must connect exactly the two declared neighbour endpoints")
        elif mode in ("rocenante-ring4", "rocenante-mesh4") and ends[0][3] != ends[1][3]:
            errors.append(f"fabric subnet {net}: RoCEnante endpoints must use the same stripe position")
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
        result["environment"].pop("B12X_ROCE_PEER_HCAS", None)
        result["environment"].pop("B12X_ROCE_TOPOLOGY", None)
        if "--disable-custom-all-reduce" not in result["serve_args"]:
            result["serve_args"].append("--disable-custom-all-reduce")
    errors = problems(result, nodes)
    if errors:
        raise ValueError("\n".join(errors))
    return result
