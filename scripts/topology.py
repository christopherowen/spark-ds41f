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
import ipaddress
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
SITE_FIELDS = ("transfer_host", "control_ip")
GID_PINS = ("NCCL_IB_GID_INDEX", "SPARKNET_ROCE_GID_INDEX")


def transport(cluster: dict) -> str | None:
    """The configured transport; None when the configuration names none."""
    return cluster.get("fabric", {}).get("transport")


# sparknet's one-shot transports and their one-shot topology. They are always
# on (no enable switch).
LOCAL = "local"
SOCKET = "nccl-socket"
PP2_RING4 = "oneshot-pp2-ring4"
ONE_SHOT = {
    "oneshot-direct": "direct",
    "oneshot-ring4": "ring4",
    PP2_RING4: "direct",
    "oneshot-switched": "direct",
}
TRANSPORTS = (*ONE_SHOT, "nccl-ring", "nccl-switched", SOCKET, LOCAL)

def switched(mode: str | None) -> bool:
    return mode in ("oneshot-switched", "nccl-switched")


def pp2_ring4(mode: str | None) -> bool:
    """Four-node switchless ring split into PP=2 x TP=2 (direct TP pairs 0-1, 2-3; direct PP pairs 0-2, 1-3)."""
    return mode == PP2_RING4


def one_shot_topology(mode: str | None) -> str | None:
    return ONE_SHOT.get(mode)


def neighbour_ring(mode: str | None) -> bool:
    """The four-node cable loop: each rank reaches only its two neighbours."""
    return mode == "nccl-ring" or one_shot_topology(mode) == "ring4"


def is_sparknet(mode: str | None) -> bool:
    return mode in ONE_SHOT


def pipeline_parallel_size(cluster: dict) -> int:
    raw = argument(cluster, "--pipeline-parallel-size")
    return int(raw) if raw and str(raw).isdigit() else 1


def tensor_parallel_size(cluster: dict) -> int | None:
    raw = argument(cluster, "--tensor-parallel-size")
    return int(raw) if raw and str(raw).isdigit() else None


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


def gid_pinned(cluster: dict) -> bool:
    """Whether the configuration pins a RoCE GID slot instead of finding GIDs by address."""
    env = cluster.get("environment", {})
    return any(env.get(key) for key in GID_PINS)


def fabric_node(node: dict, *, strip_gid_index: bool = False) -> dict:
    excluded = (*SITE_FIELDS, "ssh_host", *(("roce_gid_index",) if strip_gid_index else ()))
    return {key: value for key, value in node.items() if key not in excluded}


def fabric_map(nodes: dict, *, strip_gid_index: bool = False) -> dict:
    return {**nodes, "nodes": [fabric_node(node, strip_gid_index=strip_gid_index) for node in nodes.get("nodes", [])]}


def _pp2_ring4_hcas(node: dict) -> list[str]:
    """All local RDMA HCAs of a node in a 4-node switchless PP=2 x TP=2 ring (TP + PP cables)."""
    peers = node.get("roce_peer_hcas") or {}
    seen: list[str] = []
    for peer in sorted(peers, key=lambda key: int(key) if str(key).isdigit() else str(key)):
        for hca in peers[peer]:
            if hca not in seen:
                seen.append(hca)
    return sorted(seen)


def pp2_ring4_node_map(ring4_nodes: dict) -> dict:
    """Derive a PP=2 x TP=2 switchless ring node map from a 4-node TP=4 ring map.

    Swaps ranks 2 and 3 (and their ``roce_peer_hcas`` keys) so Stage 0 (ranks 0, 1)
    and Stage 1 (ranks 2, 3) each sit on a direct ConnectX-7 cable while PP pairs
    (0-2 and 1-3) also sit on direct cables with zero recabling.
    """
    entries = ring4_nodes.get("nodes", [])
    if len(entries) != 4:
        raise ValueError(f"pp2_ring4_node_map requires a 4-node ring map, got {len(entries)} nodes")
    remap = {0: 0, 1: 1, 2: 3, 3: 2}
    remapped = []
    for entry in entries:
        old_rank = int(entry["rank"])
        if old_rank not in remap:
            raise ValueError(f"unexpected ring rank {old_rank!r}; expected 0..3")
        node = json.loads(json.dumps(entry))
        node["rank"] = remap[old_rank]
        peers = node.get("roce_peer_hcas")
        if isinstance(peers, dict):
            node["roce_peer_hcas"] = {
                str(remap[int(k)]) if str(k).isdigit() and int(k) in remap else k: v
                for k, v in peers.items()
            }
        remapped.append(node)
    remapped.sort(key=lambda item: int(item["rank"]))
    out = {k: json.loads(json.dumps(v)) for k, v in ring4_nodes.items() if k != "nodes"}
    out["nodes"] = remapped
    return out


def _pp2_ring4_tp_hcas(node: dict) -> list[str]:
    """Local RDMA HCAs reaching this node's TP=2 peer (rank ^ 1) in a PP=2 x TP=2 ring."""
    rank = int(node["rank"])
    peers = node.get("roce_peer_hcas") or {}
    return list(peers.get(str(rank ^ 1), ()))


def _pp2_ring4_pp_hcas(node: dict) -> list[str]:
    """Local RDMA HCAs reaching this node's PP=2 peer (rank ^ 2) in a PP=2 x TP=2 ring."""
    rank = int(node["rank"])
    peers = node.get("roce_peer_hcas") or {}
    return list(peers.get(str(rank ^ 2), ()))


def _pp2_ring4_node_environment(cluster: dict, node: dict, base: dict[str, str]) -> dict[str, str]:
    """Per-node fabric environment for PP=2 x TP=2 on a 4-node switchless ring.

    Each TP=2 stage ({0,1} and {2,3}) runs sparknet one-shot direct over its TP
    cable (keyed by TP-local peer rank 0/1), each PP=2 pair ({0,2} and {1,3})
    runs sparknet one-shot P2P over its PP cable (keyed by PP-local peer rank
    0/1), and NCCL sees all four local HCAs with subnet-aware routing.
    """
    sparknet = fabric(cluster)
    pinned = gid_pinned(cluster)
    env = dict(base)
    for key in ("SPARKNET_ROCE_PEER_HCAS", "SPARKNET_ROCE_PP_PEER_HCAS",
                "SPARKNET_ROCE_TOPOLOGY", "SPARKNET_ROCE_HCA", "SPARKNET_ROCE_PP_HCA",
                "SPARKNET_ROCE_SUBNETS", "SPARKNET_ROCE_TRAFFIC_CLASS"):
        env.pop(key, None)
    if not pinned:
        for key in GID_PINS:
            env.pop(key, None)
    env.update(sparknet.render.RING_ENV)
    all_hcas = _pp2_ring4_hcas(node)
    env["NCCL_IB_HCA"] = "=" + ",".join(all_hcas)
    rank = int(node["rank"])
    tp_peer_local = str((rank % 2) ^ 1)
    tp_hcas = _pp2_ring4_tp_hcas(node)
    pp_peer_local = str((rank // 2) ^ 1)
    pp_hcas = _pp2_ring4_pp_hcas(node)
    env["SPARKNET_ROCE_TOPOLOGY"] = "direct"
    env["SPARKNET_ROCE_PEER_HCAS"] = json.dumps({tp_peer_local: tp_hcas}, separators=(",", ":"))
    if pp_hcas:
        env["SPARKNET_ROCE_PP_PEER_HCAS"] = json.dumps({pp_peer_local: pp_hcas}, separators=(",", ":"))
    if pinned:
        if "NCCL_IB_GID_INDEX" in base:
            env["NCCL_IB_GID_INDEX"] = base["NCCL_IB_GID_INDEX"]
        if "roce_gid_index" in node:
            env["NCCL_IB_GID_INDEX"] = str(node["roce_gid_index"])
        env.pop("SPARKNET_ROCE_GID_INDEX", None)
    declared = node.get("roce_subnets", {})
    subnets = {h: declared[h] for h in all_hcas if h in declared}
    if subnets:
        env["SPARKNET_ROCE_SUBNETS"] = json.dumps(subnets, separators=(",", ":"))
    iface = node.get("management_interface")
    if iface:
        env.update(NCCL_SOCKET_IFNAME="=" + iface, GLOO_SOCKET_IFNAME=iface, TP_SOCKET_IFNAME=iface)
    return env


def _pp2_ring4_problems(cluster: dict, nodes: dict) -> list[str]:
    """Validate a 4-node switchless ring mapped to PP=2 x TP=2 (TP pairs 0-1 & 2-3, PP pairs 0-2 & 1-3)."""
    errors: list[str] = []
    entries = nodes.get("nodes", [])
    if len(entries) != 4:
        return [f"{PP2_RING4} requires 4 nodes, got {len(entries)}"]
    by_rank: dict[int, dict] = {}
    heads = 0
    for node in entries:
        rank = node.get("rank")
        if type(rank) is not int or rank not in (0, 1, 2, 3) or rank in by_rank:
            errors.append(f"{node.get('name')}: {PP2_RING4} requires unique integer ranks 0..3")
            continue
        by_rank[rank] = node
        if node.get("head"):
            heads += 1
            if rank != 0:
                errors.append(f"{node.get('name')}: {PP2_RING4} requires head=true on rank 0")
    if heads != 1:
        errors.append(f"{PP2_RING4} requires exactly one head node (rank 0)")
    if len(by_rank) != 4:
        return errors
    for rank, node in sorted(by_rank.items()):
        name = node.get("name", f"rank {rank}")
        peers = node.get("roce_peer_hcas")
        if not isinstance(peers, dict):
            errors.append(f"{name}: roce_peer_hcas is required for {PP2_RING4}")
            continue
        expected_peers = {str(rank ^ 1), str(rank ^ 2)}
        if set(peers) != expected_peers:
            errors.append(
                f"{name} (rank {rank}): {PP2_RING4} requires direct peers "
                f"{sorted(expected_peers)} (TP peer {rank ^ 1}, PP peer {rank ^ 2}), got {sorted(peers)}"
            )
            continue
        seen_local: set[str] = set()
        subnets = node.get("roce_subnets", {})
        for peer_key in sorted(expected_peers):
            route = peers[peer_key]
            if not isinstance(route, list) or not 1 <= len(route) <= 2 or len(set(route)) != len(route):
                errors.append(f"{name}: peer {peer_key} must list 1 or 2 unique HCAs")
                continue
            for hca in route:
                if hca in seen_local:
                    errors.append(f"{name}: HCA {hca} cannot belong to more than one peer")
                seen_local.add(hca)
                if hca not in subnets:
                    errors.append(f"{name}: HCA {hca} lacks a roce_subnets entry")
    # Check symmetry and matching cable subnets on every TP and PP edge.
    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3)):
        pa = (by_rank[a].get("roce_peer_hcas") or {}).get(str(b), [])
        pb = (by_rank[b].get("roce_peer_hcas") or {}).get(str(a), [])
        if len(pa) != len(pb):
            errors.append(f"rank {a}<->{b}: asymmetric stripe count ({len(pa)} vs {len(pb)})")
            continue
        sa = by_rank[a].get("roce_subnets", {})
        sb = by_rank[b].get("roce_subnets", {})
        nets_a = {sa[h] for h in pa if h in sa}
        nets_b = {sb[h] for h in pb if h in sb}
        if nets_a != nets_b:
            errors.append(f"rank {a}<->{b}: mismatched roce_subnets ({sorted(nets_a)} vs {sorted(nets_b)})")
    settings = cluster.get("environment", {})
    sparknet = fabric(cluster)
    pinned = gid_pinned(cluster)
    check_settings = {k: v for k, v in settings.items() if k not in GID_PINS} if pinned else settings
    if hasattr(sparknet.render, "base_problems"):
        errors.extend(sparknet.render.base_problems(check_settings))
    for key, value in {**sparknet.render.RING_ENV, **VLLM_RING_ENV}.items():
        if str(settings.get(key)) != value:
            errors.append(f"{PP2_RING4} requires {key}={value}")
    for key in (*sparknet.render.FORBIDDEN_RING_KEYS, "NCCL_SWITCHLESS_BIDIRECTIONAL"):
        if key in settings:
            errors.append(f"{PP2_RING4} forbids {key}")
    if "--disable-custom-all-reduce" in cluster.get("serve_args", []):
        errors.append(f"{PP2_RING4} requires custom all-reduce enabled")
    if "--enable-expert-parallel" in cluster.get("serve_args", []):
        errors.append(f"{PP2_RING4} does not support expert-parallel all-to-all")
    return errors


def node_environment(cluster: dict, node: dict) -> dict[str, str]:
    base = {key: str(value) for key, value in cluster["environment"].items()}
    mode = transport(cluster)
    if pp2_ring4(mode):
        env = _pp2_ring4_node_environment(cluster, node, base)
    elif mode == SOCKET:
        env = {
            k: v
            for k, v in base.items()
            if not k.startswith(("SPARKNET_ROCE_", "NCCL_IB_"))
            and k != "NCCL_SWITCHLESS_BIDIRECTIONAL"
        }
        env["NCCL_IB_DISABLE"] = "1"
        env["NCCL_NET"] = "Socket"
        iface = node.get("management_interface")
        if iface:
            env.update(NCCL_SOCKET_IFNAME=iface, GLOO_SOCKET_IFNAME=iface, TP_SOCKET_IFNAME=iface)
    elif mode in TRANSPORTS and mode != LOCAL:
        sparknet = fabric(cluster)
        pinned = gid_pinned(cluster)
        clean_node = fabric_node(node, strip_gid_index=True)
        clean_base = {k: v for k, v in base.items() if k not in GID_PINS} if not pinned else base
        env = sparknet.render.node_environment({"nodes": [clean_node]}, clean_node,
                                               transport=mode, base=clean_base)
        if pinned:
            env.pop("SPARKNET_ROCE_SUBNETS", None)
            for key in GID_PINS:
                if key in base:
                    env[key] = base[key]
            if "roce_gid_index" in node:
                env["NCCL_IB_GID_INDEX"] = str(node["roce_gid_index"])
                if is_sparknet(mode):
                    env["SPARKNET_ROCE_GID_INDEX"] = str(node["roce_gid_index"])
            elif not is_sparknet(mode):
                env.pop("SPARKNET_ROCE_GID_INDEX", None)
        elif is_sparknet(mode) and "SPARKNET_ROCE_SUBNETS" not in env:
            # Each one-shot HCA's cable subnet, in which sparknet finds the HCA's GID.
            routed = sorted(hcas(cluster, node))
            declared = node.get("roce_subnets", {})
            subnets = {h: declared[h] for h in routed if h in declared}
            if subnets:
                env["SPARKNET_ROCE_SUBNETS"] = json.dumps(subnets, separators=(",", ":"))
    else:
        env = base
    env["VLLM_HOST_IP"] = control_ip(node)
    if switched(mode):
        env["SPARK_DEPLOYMENT_TRANSPORT"] = str(mode)
    return env


def control_ip(node: dict) -> str:
    """The node's control-plane IP (defaults to management_ip)."""
    return node.get("control_ip") or node["management_ip"]


def hcas(cluster: dict, node: dict) -> list[str]:
    """The node's RDMA devices for the configured transport."""
    mode = transport(cluster)
    if mode in (LOCAL, SOCKET):
        return []
    if pp2_ring4(mode):
        return _pp2_ring4_hcas(node)
    return fabric(cluster).nodes.nccl_hcas(fabric_node(node), mode)


def problems(cluster: dict, nodes: dict) -> list[str]:
    errors = []
    entries = nodes.get("nodes", [])
    mode = transport(cluster)
    pinned = gid_pinned(cluster)
    for node in entries:
        if "ssh_host" in node:
            errors.append(
                f"{node.get('name')}: ssh_host is no longer used because name is now the SSH "
                "target; either rename the node to its SSH target or add an SSH Host alias"
            )
        if "roce_gid_index" in node and mode not in (LOCAL, SOCKET):
            if not pinned:
                errors.append(
                    f"{node.get('name')}: roce_gid_index is not read by a configuration that finds GIDs by "
                    "address (NCCL and sparknet take each HCA's IPv4 RoCE v2 GID in its roce_subnets network); "
                    "delete the field"
                )
            elif type(node["roce_gid_index"]) is not int or node["roce_gid_index"] < 0:
                errors.append(f"{node.get('name')}: roce_gid_index must be a nonnegative integer")
    count = len(entries)
    if mode == LOCAL:
        if count != 1:
            errors.append(f"local deployment requires 1 node, got {count}")
    elif switched(mode) or mode == SOCKET:
        if not 2 <= count <= 8:
            errors.append(f"{mode} deployment requires 2 to 8 nodes, got {count}")
    elif count not in (2, 3, 4):
        errors.append(f"deployment requires 2, 3 or 4 nodes, got {count}")
    settings = cluster.get("environment", {})
    if mode is None:
        errors.append("fabric.transport is required; a configuration without one ran B12X's "
                      "rocenante-direct, which is retired (docs/switchless-topology.md)")
    elif mode not in TRANSPORTS:
        errors.append(f"unknown fabric transport {mode!r}")
    elif mode == LOCAL:
        if count == 1:
            head_node = entries[0]
            if not head_node.get("head"):
                errors.append("single-node deployment requires head=true on rank 0")
            if head_node.get("rank") != 0:
                errors.append("single-node deployment requires rank=0")
            if not head_node.get("name") or not head_node.get("management_ip"):
                errors.append("single-node deployment requires name and management_ip")
    elif mode == SOCKET:
        ranks = [node.get("rank") for node in entries]
        if sorted(r for r in ranks if isinstance(r, int)) != list(range(count)):
            errors.append(f"{SOCKET} requires ranks 0..{count - 1}")
        for node in entries:
            if not node.get("name") or not node.get("management_ip"):
                errors.append(f"{SOCKET} requires name and management_ip on every node")
        if "--disable-custom-all-reduce" not in cluster.get("serve_args", []):
            errors.append(f"{SOCKET} runs plain NCCL over Ethernet; it requires --disable-custom-all-reduce")
    elif pp2_ring4(mode):
        errors.extend(_pp2_ring4_problems(cluster, nodes))
    else:
        sparknet = fabric(cluster)
        errors.extend(sparknet.nodes.problems(fabric_map(nodes, strip_gid_index=True), mode))
        check_settings = {k: v for k, v in settings.items() if k not in GID_PINS} if pinned else settings
        errors.extend(sparknet.render.environment_problems(check_settings, mode, count))
        if hasattr(sparknet.render, "base_problems"):
            errors.extend(sparknet.render.base_problems(check_settings))
    heads = [n for n in entries if n.get("head")]
    if len(heads) == 1:
        allowed_master = {heads[0].get("management_ip"), heads[0].get("control_ip")} - {None}
        if cluster.get("distributed", {}).get("master_addr") not in allowed_master:
            errors.append("distributed.master_addr must match the API head's management_ip")
    if switched(mode):
        if cluster.get("fabric", {}).get("topology") != "switched":
            errors.append(f"{mode} requires explicit fabric.topology=switched")
        for key in ("NCCL_SWITCHLESS_BIDIRECTIONAL", "NCCL_MIN_TRAFFIC_PER_CHANNEL",
                    "NCCL_THREAD_THRESHOLDS", "NCCL_ALGO", "NCCL_MIN_NCHANNELS",
                    "NCCL_P2P_NET_CHUNKSIZE",
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
    else:
        bidirectional = settings.get("NCCL_SWITCHLESS_BIDIRECTIONAL")
        if bidirectional is not None:
            if str(bidirectional) not in ("1", "2"):
                errors.append("NCCL_SWITCHLESS_BIDIRECTIONAL must be 1 or 2")
            channels = str(settings.get("NCCL_MAX_NCHANNELS", ""))
            if str(bidirectional) == "2" and channels.isdigit() and int(channels) > 2 and int(channels) % 4 != 0:
                errors.append("NCCL_SWITCHLESS_BIDIRECTIONAL=2 needs two or a multiple of four channels")
    if mode is not None and not is_sparknet(mode) and any(k.startswith("SPARKNET_ROCE_") for k in settings):
        errors.append(f"{mode} does not run sparknet; SPARKNET_ROCE_* settings do not apply")
    if argument(cluster, "--nnodes") != str(count):
        errors.append(f"--nnodes must equal the configured node count ({count})")
    pp = pipeline_parallel_size(cluster)
    tp = tensor_parallel_size(cluster)
    if pp < 1:
        errors.append(f"--pipeline-parallel-size must be >= 1, got {pp}")
    elif pp == 1:
        if argument(cluster, "--tensor-parallel-size") != str(count):
            errors.append(f"--tensor-parallel-size must equal the configured node count ({count})")
    else:
        if tp is None or tp * pp != count:
            errors.append(
                f"--tensor-parallel-size ({tp}) x --pipeline-parallel-size ({pp}) "
                f"must equal the configured node count ({count})"
            )
        elif pp2_ring4(mode):
            if (pp, tp) != (2, 2):
                errors.append(f"{PP2_RING4} requires --pipeline-parallel-size 2 and --tensor-parallel-size 2")
        elif switched(mode) or mode == SOCKET:
            if pp != 2 or tp not in (2, 3, 4):
                errors.append(
                    f"{mode} pipeline parallelism supports PP=2 with TP in 2..4, got PP={pp} TP={tp}"
                )
        elif not neighbour_ring(mode):
            errors.append(f"{mode} requires --pipeline-parallel-size=1")
    expected_draft_tp = tp if (pp > 1 and tp is not None) else count
    try:
        spec = json.loads(argument(cluster, "--speculative-config") or "{}")
        if not isinstance(spec, dict):
            raise ValueError("expected object")
        if spec and spec.get("draft_tensor_parallel_size") != expected_draft_tp:
            errors.append(
                f"draft_tensor_parallel_size must equal the tensor-parallel size ({expected_draft_tp})"
            )
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


def ipv4_roce_v2_gids(output: str) -> dict[str, list[tuple[int, ipaddress.IPv4Address]]]:
    """Each HCA's IPv4 RoCE v2 GIDs, (index, address) by index, from ``hca index type gid ndev`` lines."""
    found: dict[str, list[tuple[int, ipaddress.IPv4Address]]] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 4 or not fields[1].isdigit():
            continue
        found.setdefault(fields[0], [])
        if fields[2] != "RoCEv2":
            continue
        try:
            address = ipaddress.IPv6Address(fields[3]).ipv4_mapped
        except ipaddress.AddressValueError:
            continue
        if address is not None:
            found[fields[0]].append((int(fields[1]), address))
    return {hca: sorted(gids) for hca, gids in found.items()}


def gid_subnet_problems(*args: object) -> list[str]:
    """Each cable HCA's port must carry exactly one IPv4 RoCE v2 GID, in its cable's subnet, at any index.

    sparknet needs one in the subnet. NCCL scans without a per-port range and
    takes the lowest-index IPv4 RoCE v2 GID, so a second one on the port could
    become NCCL's source address.
    """
    if len(args) == 2:
        node, output = args  # type: ignore[misc]
    elif len(args) == 3 and isinstance(args[1], dict):
        _, node, output = args  # type: ignore[misc]
    elif len(args) == 3 and isinstance(args[0], dict):
        node, _, output = args  # type: ignore[misc]
    elif len(args) == 4:
        _, node, _, output = args  # type: ignore[misc]
    else:
        raise TypeError(f"gid_subnet_problems expects (node, output), got {len(args)} arguments")
    assert isinstance(node, dict) and isinstance(output, str)
    gids = ipv4_roce_v2_gids(output)
    missing = {line.split()[0] for line in output.splitlines() if line.split()[1:] == ["missing"]}
    errors = []
    for hca, subnet in node.get("roce_subnets", {}).items():
        if hca in missing:
            continue  # reported with the devices
        network = ipaddress.IPv4Network(subnet)
        present = gids.get(hca, [])
        inside = [(i, a) for i, a in present if a in network]
        outside = ", ".join(f"{a} at index {i}" for i, a in present if a not in network)
        label = f"{node['name']}: {hca}"
        if not inside:
            errors.append(f"{label} has no IPv4 RoCE v2 GID in its cable subnet {subnet}"
                          + (f" (it has {outside})" if outside else "") + "; check the link's address and cable")
        elif len(inside) > 1:
            errors.append(f"{label} has {len(inside)} IPv4 RoCE v2 GIDs in {subnet} (indices "
                          f"{', '.join(str(i) for i, _ in inside)}); its port needs one address there")
        elif outside:
            errors.append(f"{label} also has an IPv4 RoCE v2 GID outside {subnet} ({outside}); NCCL takes the "
                          "lowest-index one, so give the port only its cable's address")
    return errors


def candidate(base: dict, nodes: dict, nodes_path: str) -> dict:
    result = copy.deepcopy(base)
    count = len(nodes["nodes"])
    result["nodes_config"] = nodes_path
    result["deployment"]["launch_enabled"] = False
    result["deployment"].pop("branch", None)
    ssh_user = nodes.get("ssh_user")
    if result.get("host", {}).get("home") == "/home/swank" and ssh_user not in (None, "", "swank", "u"):
        result["host"]["home"] = f"/home/{ssh_user}"
    result["distributed"]["master_addr"] = next(control_ip(n) for n in nodes["nodes"] if n.get("head"))
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
                    "NCCL_P2P_NET_CHUNKSIZE",
                    "NCCL_ALGO_PLUGIN", "NCCL_TUNER_PLUGIN", "NCCL_GRAPH_FILE", "NCCL_TOPO_FILE",
                    "NCCL_COLLNET_ENABLE", "NCCL_CUMEM_HOST_ENABLE", "NCCL_GIN_ENABLE",
                    "NCCL_IB_SUBNET_PREFIX_LEN", "NCCL_MNNVL_ENABLE", "NCCL_PAT_ENABLE",
                    "NCCL_RMA_DISABLE", "NCCL_RUNTIME_CONNECT"):
            result["environment"].pop(key, None)
        result["environment"].update({
            "NCCL_BUFFSIZE": "1048576",
            "NCCL_CUMEM_ENABLE": "0",
            "NCCL_MAX_NCHANNELS": "8",
            "SPARKNET_ROCE_ALLGATHER_MAX_BYTES": "4194304",
        })
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
