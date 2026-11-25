"""Doctor, preflight, cooling, fan control, RoCE/GID and runtime checks for spark-ds41f."""

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
sys.path.insert(0, str(ROOT / "scripts"))
import source_labels
import kernel_backend
import kernel_readiness
import topology
import recipe as cluster_recipe

from .config import *  # noqa: F401,F403

class Warn(str):
    """A doctor finding that does not fail doctor or block a benchmark.

    Errors are what breaks serving, safety or a required capability now: the
    display carve-out and the kept text console (modeset, fbdev, DRM master,
    console mode), RoCE GIDs, the GPU clock and any difference from
    the promoted configuration. Warnings are latent (effective at the next
    boot), cost a little, could not be read, or only matter for bench
    (fan-floor control).
    """


def without_missing_glp(cluster: dict, nodes: dict) -> tuple[dict, list[str]]:
    """The profile as it can be served here, and a warning when that is without GLP.

    The ``glp`` block serves a steered variant beside the stock model, from a small file of a
    gated repository that every node needs. A site that has not installed it still serves:
    when any node lacks the file, every rank starts without the block (ranks must not
    differ), so only the stock name is served. A file that is present but differs from the
    pinned one stays an error (node_glp_problems).
    """
    source = cluster_recipe.glp_file(cluster)
    if source is None or not nodes.get("nodes"):
        return cluster, []
    path = host_path(cluster, source)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes["nodes"])) as pool:
        results = list(pool.map(lambda node: run_ssh(nodes, node, "test", "-e", path), nodes["nodes"]))
    missing = [node["name"] for node, result in zip(nodes["nodes"], results) if result.returncode == 1]
    if not missing:
        return cluster, []
    steered = cluster["glp"]["served_model_name"]
    stock = ", ".join(name for name in served_model_names(cluster) if name != steered)
    return cluster_recipe.without_glp(cluster), [Warn(
        f"GLP steering file is not installed on {', '.join(missing)} ({path}): the cluster serves "
        f"{stock} only, and a request for {steered} is refused as an unknown model. To serve it "
        f"too, on each of those nodes run {cluster_recipe.glp_download(cluster)} "
        "(docs/setup.md, section 5), then start again"
    )]


def fan_control_enabled(cluster: dict) -> bool:
    """Whether the site runs the fan-floor control (host.fan_control, default true).

    The driver binds NVIDIA's P4242 board only. Sites on other GB10 boards set
    host.fan_control false: doctor then records the absence instead of erroring,
    and benchmarks skip the curve instead of refusing to measure.
    """
    return cluster.get("host", {}).get("fan_control", True) is not False


# Every key of the container section that a command reads.
_SHAPE_TO_PROFILE_STEM = {"tp4-ring": "tp4", "tp3-triangle": "tp3"}
_L2_PREFETCH_TARGET = "/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/l2_prefetch.py"
_L2_PREFETCH_DEFAULT_MB = (
    ("VLLM_DS41_L2_PREFETCH_WO_MB", "15"),
    ("VLLM_DS41_L2_PREFETCH_FFN_MB", "12"),
    ("VLLM_DS41_L2_PREFETCH_NEXT_MB", "12"),
)
_ROCE_MAX_COLLECTIVE_BY_SHAPE = {
    "tp4-ring": "2097152",
    "tp3-triangle": "4194304",
    "tp3-switched": "4194304",
    "tp3-triangle-4k": "4194304",
    "tp2-direct": "4194304",
    "tp2-switched": "4194304",
}
_RENDERED_ENV_KEYS = frozenset((
    "GLOO_SOCKET_IFNAME",
    "NCCL_SOCKET_IFNAME",
    "UCX_NET_DEVICES",
    "SPARKNET_ROCE_IFACES",
    "VLLM_HOST_IP",
)) | frozenset(cluster_recipe.SWITCHED_DIRECT_ENV)
_REPORTING_SERVE_ARGS = frozenset(("--enable-prompt-tokens-details", "--enable-per-request-metrics"))
# Diagnostic arguments with a value: a profiler arm (the lab's `profile` job reads its trace
# directory from --profiler-config) is its base profile plus this one argument.
_DIAGNOSTIC_SERVE_OPTIONS = frozenset(("--port", "--profiler-config"))
_L2_PREFETCH_DEFAULT_RE = {
    key: re.compile(rf'_mb\("{re.escape(key)}",\s*"(\d+)"\)')
    for key, _ in _L2_PREFETCH_DEFAULT_MB
}


def _series_patch_names(upstreams: dict, source_name: str) -> set[str]:
    source = upstreams.get("sources", {}).get(source_name, {})
    rel = source.get("patch_series") if isinstance(source, dict) else None
    if not isinstance(rel, str) or not rel:
        return set()
    series_path = ROOT / rel
    if not series_path.is_file():
        return set()
    return {Path(entry).name for entry in read_series(series_path)}


def _overlay_has_sane_l2_defaults(cluster: dict) -> bool:
    for mount in cluster.get("container", {}).get("mounts", []):
        if not isinstance(mount, (list, tuple)) or len(mount) < 2:
            continue
        if mount[1] != _L2_PREFETCH_TARGET:
            continue
        src = str(mount[0])
        path = ROOT / src[len("{checkout}/"):] if src.startswith("{checkout}/") else Path(src)
        if not path.is_file():
            return False
        text = path.read_text(encoding="utf-8")
        return all(
            (match := _L2_PREFETCH_DEFAULT_RE[key].search(text)) is not None
            and match.group(1) == expected
            for key, expected in _L2_PREFETCH_DEFAULT_MB
        )
    return False


def _expected_wo_band(shape: str | None) -> str | None:
    if not isinstance(shape, str) or not shape:
        return None
    if re.match(r"^tp[1-3](?:-|$)", shape):
        return "17-32=12"
    if re.match(r"^tp[4-8](?:-|$)", shape):
        return "17-32=10"
    return None


def _cluster_shape(cluster: dict) -> str | None:
    shape = cluster.get("shape")
    if isinstance(shape, str) and shape:
        return shape
    tp_raw = topology.argument(cluster, "--tensor-parallel-size")
    if not tp_raw or not str(tp_raw).isdigit():
        return None
    tp = int(tp_raw)
    fabric = cluster.get("fabric", {})
    transport = fabric.get("transport") if isinstance(fabric, dict) else None
    topo = fabric.get("topology") if isinstance(fabric, dict) else None
    if transport == "local" or tp == 1:
        return "tp1"
    if topo == "switched" or (isinstance(transport, str) and transport.endswith("-switched")):
        return f"tp{tp}-switched"
    if tp == 4 and transport in ("oneshot-ring4", "nccl-ring"):
        return "tp4-ring"
    if tp == 2 and transport in ("oneshot-direct", "nccl-direct"):
        return "tp2-direct"
    if tp == 3 and transport in ("oneshot-direct", "nccl-direct"):
        max_len = topology.argument(cluster, "--max-model-len")
        return "tp3-triangle-4k" if str(max_len) == "262144" else "tp3-triangle"
    return None


def _qualified_reference(cluster: dict, shape: str, stem: str) -> dict | None:
    upstreams_cfg = cluster.get("upstreams_config")
    if isinstance(upstreams_cfg, str) and upstreams_cfg:
        ref_rel = Path(upstreams_cfg).parent / f"{stem}.json"
        if (ROOT / ref_rel).is_file():
            return read_cluster(str(ref_rel))
        return None
    prod = read_json(DEFAULT_CLUSTER_CONFIG)
    if (
        shape in prod.get("shapes", {})
        and cluster.get("promoted_baseline") == prod.get("promoted_baseline")
        and cluster.get("container", {}).get("image") == prod.get("container", {}).get("image")
    ):
        return cluster_recipe.resolve(prod, shape)
    return None


def _comparable_serve_args(args: list) -> list[str]:
    out: list[str] = []
    skip_next = False
    for item in args:
        if skip_next:
            skip_next = False
            continue
        if item in _DIAGNOSTIC_SERVE_OPTIONS:
            skip_next = True
            continue
        if item in _REPORTING_SERVE_ARGS:
            continue
        out.append(str(item))
    return out


def qualified_recipe_problems(cluster: dict, nodes: dict, upstreams: dict) -> list[str]:
    """Warn when a cluster configuration drifts from its stack's qualified settings."""
    if len(nodes.get("nodes", [])) <= 1:
        return []

    warnings: list[str] = []
    shape_str = _cluster_shape(cluster)
    stem = _SHAPE_TO_PROFILE_STEM.get(shape_str, shape_str) if shape_str else None
    env = cluster.get("environment", {})
    checked_keys: set[str] = set()

    vllm_patches = _series_patch_names(upstreams, "vllm")
    sparknet_patches = _series_patch_names(upstreams, "sparknet")
    has_l2pf = any(name.startswith("0781-") for name in vllm_patches)
    has_wo_bands = any(name.startswith("0806-") for name in vllm_patches)
    has_roce_cutoff = any(name.startswith("0002-") for name in sparknet_patches)
    l2pf_enabled = str(env.get("VLLM_DS41_L2_PREFETCH", "1")) != "0"

    if has_l2pf and l2pf_enabled:
        overlay_sane = _overlay_has_sane_l2_defaults(cluster)
        for key, expected in _L2_PREFETCH_DEFAULT_MB:
            checked_keys.add(key)
            if key not in env:
                if not overlay_sane:
                    warnings.append(Warn(
                        f"{key} is unset without a 15/12/12 l2_prefetch.py overlay; "
                        f"expected qualified {expected!r}"
                    ))
            elif str(env[key]) != expected:
                warnings.append(Warn(
                    f"{key}={str(env[key])!r}, expected qualified {expected!r}"
                ))

    if has_wo_bands and l2pf_enabled:
        expected_band = _expected_wo_band(shape_str)
        if expected_band is not None:
            checked_keys.add("VLLM_DS41_L2_PREFETCH_WO_BANDS")
            actual_band = env.get("VLLM_DS41_L2_PREFETCH_WO_BANDS")
            if actual_band != expected_band:
                warnings.append(Warn(
                    f"VLLM_DS41_L2_PREFETCH_WO_BANDS={actual_band!r}, "
                    f"expected qualified {expected_band!r} for {shape_str}"
                ))

    if has_roce_cutoff and shape_str in _ROCE_MAX_COLLECTIVE_BY_SHAPE:
        checked_keys.add("SPARKNET_ROCE_MAX_COLLECTIVE_BYTES")
        expected_cutoff = _ROCE_MAX_COLLECTIVE_BY_SHAPE[shape_str]
        actual_cutoff = env.get("SPARKNET_ROCE_MAX_COLLECTIVE_BYTES")
        if actual_cutoff != expected_cutoff:
            warnings.append(Warn(
                f"SPARKNET_ROCE_MAX_COLLECTIVE_BYTES={actual_cutoff!r}, "
                f"expected qualified {expected_cutoff!r} for {shape_str}"
            ))

    if shape_str and stem:
        ref = _qualified_reference(cluster, shape_str, stem)
        if ref is not None:
            ignored = _RENDERED_ENV_KEYS | checked_keys
            actual_env = {
                str(k): str(v)
                for k, v in env.items()
                if k not in ignored and not (k == "VLLM_SERVER_DEV_MODE" and str(v) == "1")
            }
            ref_env = {
                str(k): str(v)
                for k, v in ref.get("environment", {}).items()
                if k not in ignored and not (k == "VLLM_SERVER_DEV_MODE" and str(v) == "1")
            }
            for key in sorted(set(actual_env) | set(ref_env)):
                if actual_env.get(key) != ref_env.get(key):
                    warnings.append(Warn(
                        f"qualified config drift for {shape_str}: environment[{key!r}]="
                        f"{actual_env.get(key)!r}, expected {ref_env.get(key)!r}"
                    ))
            if _comparable_serve_args(cluster.get("serve_args", [])) != _comparable_serve_args(ref.get("serve_args", [])):
                warnings.append(Warn(
                    f"qualified config drift for {shape_str}: serve_args differ from qualified profile"
                ))

    return warnings


CONTAINER_KEYS = frozenset(("name", "image", "expected_labels", "docker_run_args", "mounts"))


def local_doctor(cluster: dict, nodes: dict, upstreams: dict) -> list[str]:
    problems: list[str] = topology.problems(cluster, nodes)
    problems.extend(kernel_backend.problems(cluster))
    capture_errors, capture_warnings = cluster_recipe.capture_problems(cluster)
    problems.extend(capture_errors)
    problems.extend(Warn(warning) for warning in capture_warnings)
    problems.extend(qualified_recipe_problems(cluster, nodes, upstreams))
    problems.extend(benchmark_reference_problems(cluster, nodes))
    if cluster.get("promoted_baseline") != upstreams.get("generated_from_baseline"):
        problems.append("cluster and upstream locks name different baselines")
    # A key no command reads is a setting that silently does nothing: an arm that names
    # its overlay under container.extra_mounts boots the image unchanged.
    for key in sorted(set(cluster.get("container", {})) - CONTAINER_KEYS):
        problems.append(
            f"container.{key} is not a container setting (known: {', '.join(sorted(CONTAINER_KEYS))}); "
            "files are mounted through container.mounts, as [source, target, mode]"
        )
    host = cluster.get("host", {})
    for key in (
        "memguard_gib",
        "startup_memguard_gib",
        "startup_memguard_wait_seconds",
    ):
        value = host.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            problems.append(f"host.{key} must be a positive integer")
    warn_gib = host.get("memguard_warn_gib", 0)
    if not isinstance(warn_gib, int) or isinstance(warn_gib, bool) or warn_gib < 0:
        problems.append("host.memguard_warn_gib must be a non-negative integer")
    elif warn_gib and isinstance(host.get("memguard_gib"), int) and warn_gib <= host["memguard_gib"]:
        problems.append("host.memguard_warn_gib must exceed host.memguard_gib")
    if not isinstance(host.get("fan_control", True), bool):
        problems.append("host.fan_control must be a boolean")
    if not isinstance(host.get("defragment_before_start", True), bool):
        problems.append("host.defragment_before_start must be a boolean")
    for key in ("memguard_interval_seconds", "startup_memguard_interval_seconds"):
        value = host.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
        ):
            problems.append(f"host.{key} must be a positive number")
    if (
        isinstance(host.get("startup_memguard_gib"), int)
        and isinstance(host.get("memguard_gib"), int)
        and host["startup_memguard_gib"] <= host["memguard_gib"]
    ):
        problems.append(
            "host.startup_memguard_gib must exceed host.memguard_gib"
        )
    deployment = cluster.get("deployment", {})
    for key in ("repository", "path", "ready_url", "ready_timeout_seconds"):
        if not deployment.get(key):
            problems.append(f"deployment.{key} is not configured")
    if "branch" in deployment and not deployment["branch"]:
        problems.append("deployment.branch must be omitted or name a branch")
    if not isinstance(deployment.get("launch_enabled"), bool):
        problems.append("deployment.launch_enabled must be true or false")
    for name, source in upstreams["sources"].items():
        if not source.get("upstream"):
            problems.append(f"{name}: canonical upstream is not recorded")
        if source.get("kind") == "git":
            if len(source["revision"]) != 40:
                problems.append(f"{name}: revision is not a full commit SHA")
            if not source.get("tracking_ref"):
                problems.append(f"{name}: upstream tracking ref is not recorded")
        series = source.get("patch_series")
        if series:
            series_path = ROOT / series
            if not series_path.is_file():
                problems.append(f"{name}: missing patch series {series}")
            else:
                for patch_name in read_series(series_path):
                    if not patch_name.startswith("TODO-") and not (
                        series_path.parent / patch_name
                    ).is_file():
                        problems.append(f"{name}: missing patch {patch_name} (listed in {series})")
        lockfile = source.get("lockfile")
        if lockfile and not (ROOT / lockfile).is_file():
            problems.append(f"{name}: missing dependency lock {lockfile}")
    problems.extend(cuda_toolkit_problems(upstreams["sources"]))
    problems.extend(torch_problems(upstreams["sources"]))
    # The benchmark client (Sonar) is the checkout's tool, pinned apart from any image's lock.
    from . import sonar as sonar_client
    try:
        sonar_pin = json.loads(sonar_client.LOCK.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        problems.append(f"config/sonar.json: the benchmark client is not pinned: {error}")
    else:
        problems.extend(f"config/sonar.json: {problem}" for problem in sonar_client.lock_problems(sonar_pin))
    watchlist_path = ROOT / "docs/inspiration.md"
    watchlist = (
        watchlist_path.read_text(encoding="utf-8") if watchlist_path.is_file() else ""
    )
    if not watchlist:
        problems.append(Warn("missing improvement watchlist docs/inspiration.md"))
    for name, reference in upstreams.get("references", {}).items():
        if reference.get("kind") != "git-watch":
            problems.append(Warn(f"{name}: reference kind must be git-watch"))
        if not reference.get("url") or not reference.get("role"):
            problems.append(Warn(f"{name}: reference URL and role are required"))
        elif reference["url"] not in watchlist:
            problems.append(Warn(f"{name}: reference URL is absent from the watchlist"))
    source_manifest = read_json(
        upstreams.get("source_manifest", "manifests/sources/2026-09-20-active-source.json")
    )
    problems.extend(arm_source_problems(cluster, upstreams, source_manifest))
    listed = [name for name in OPTIONAL_BUILD_PROJECTS if name in upstreams["sources"]]
    if listed and len(listed) != len(OPTIONAL_BUILD_PROJECTS):
        problems.append(f"{' and '.join(OPTIONAL_BUILD_PROJECTS)} are built together; the lock lists only {listed[0]}")
    for name in (*listed, *(n for n in INDEPENDENT_BUILD_PROJECTS if n in upstreams["sources"])):
        if not upstreams["sources"][name].get("version"):
            problems.append(f"{name}: package version is not recorded in the lock")
    # sparknet's transports run only on an image built with sparknet's collectives.
    sparknet_image = SPARKNET_CAPABILITY in source_manifest.get("vllm", {}).get("capabilities", [])
    if topology.transport(cluster) == 'nccl-switched' and sparknet_image:
        problems.append('nccl-switched has no runtime support: this image replaces NCCL with '
                        'sparknet, and a switch-aware NCCL build is separate work')
    if topology.is_sparknet(topology.transport(cluster)):
        if not sparknet_image or "sparknet" not in upstreams["sources"] or "sparknet" not in source_manifest:
            problems.append(f"{topology.transport(cluster)} requires an image with {SPARKNET_CAPABILITY} and the lock's sparknet source")
        else:
            label = source_labels.get(cluster.get("container", {}).get("expected_labels", {}), "sparknet.tree")
            expected_tree = source_manifest["sparknet"]["expected_tree"]
            if label != expected_tree:
                manifest_name = upstreams.get("source_manifest", "source.json")
                hint = (
                    ""
                    if cluster.get("upstreams_config")
                    else "; a profile for a patched image must set its own upstreams_config lock and "
                         "source_manifest (then run bin/spark --cluster-config <profile> build prepare --lock)"
                )
                problems.append(
                    f"sparknet image tree does not match its source manifest "
                    f"({manifest_name}: expected {expected_tree!r}, profile has {label!r}){hint}"
                )
            # Before 0.5.0 sparknet reads one GID slot (3 when none is set), so a
            # configuration that pins none must run one that finds GIDs by address.
            version = upstreams["sources"]["sparknet"].get("version") or "0"
            if not topology.gid_pinned(cluster) and tuple(int(p) for p in version.split(".")) < (0, 5, 0):
                problems.append(f"a configuration that pins no GID slot needs sparknet 0.5.0 or later, which finds "
                                f"each HCA's GID by address; the lock's sparknet is {version}")
    problems.extend(source_identifier_problems(upstreams, source_manifest))
    for relative, wanted in source_manifest.get("deployment_artifacts", {}).items():
        candidate = ROOT / relative
        if not candidate.is_file():
            problems.append(f"missing deployment artifact {relative}")
        elif sha256(candidate) != wanted:
            problems.append(f"deployment artifact differs from manifest: {relative}")
    problems.extend(glp_problems(cluster))
    problems.extend(tracked_secret_problems())
    return problems


DESKTOP_QUERY = "systemctl get-default; systemctl is-active display-manager.service"


def desktop_problems(node_name: str, output: str) -> list[str]:
    """Problems when a node boots to a desktop or runs its display manager.

    The display stack's processes and GPU buffers come out of the same unified
    memory as the model, and dgx1 has little to spare above its startup guard.
    """
    default_target, manager = (output.split() + ["", ""])[:2]
    problems = []
    if default_target != "multi-user.target":
        problems.append(
            Warn(
                f"{node_name}: boots to {default_target or 'an unknown target'}, "
                "expected multi-user.target (no desktop); run "
                "sudo systemctl set-default multi-user.target"
            )
        )
    if manager in ("active", "activating", "reloading"):
        problems.append(
            f"{node_name}: display manager is {manager}; run "
            "sudo systemctl disable --now display-manager.service"
        )
    return problems


def boot_order_problems(node_name: str, output: str) -> list[str]:
    """Problems when UEFI tries another boot entry before the installed OS.

    A network (PXE) entry first costs about a minute of DHCP timeouts on every
    boot; a firmware update left dgx3 that way.
    """
    order: list[str] = []
    entries: dict[str, str] = {}
    for line in output.splitlines():
        if line.startswith("BootOrder:"):
            order = [item.strip().upper() for item in line.split(":", 1)[1].split(",") if item.strip()]
        elif match := re.match(r"Boot([0-9A-Fa-f]{4})\*?\s+(.*)", line):
            entries[match.group(1).upper()] = match.group(2)
    if not order:
        return [Warn(f"{node_name}: cannot read the UEFI boot order")]
    def installed(entry: str) -> bool:
        return (
            entry.split("\t")[0].strip().lower() == "ubuntu"
            or "\\efi\\ubuntu\\" in entry.lower()
        )

    first = entries.get(order[0], "")
    label = first.split("\t")[0].strip()
    if installed(first):
        return []
    ubuntu = next((number for number in order if installed(entries.get(number, ""))), None)
    ubuntu = ubuntu or next((number for number, entry in entries.items() if installed(entry)), None)
    fix = (
        "run sudo efibootmgr --bootorder "
        + ",".join([ubuntu] + [number for number in order if number != ubuntu])
        if ubuntu
        else "no installed-OS boot entry exists; reinstall the boot loader with grub-install"
    )
    return [
        Warn(
            f"{node_name}: UEFI boots {label or 'entry ' + order[0]!r} before the "
            f"installed OS (BootOrder {','.join(order)}); {fix}"
        )
    ]


# Desktop services that serve nothing on a headless inference node:
# scripts/host-recovery disables them (the display manager has its own check).
IDLE_SERVICES = (
    "bluetooth.service",
    "cups.service",
    "cups.socket",
    "cups.path",
    "cups-browsed.service",
    "snapd.service",
    "snapd.socket",
    "snapd.seeded.service",
    "snapd.snap-repair.timer",
    "fwupd-refresh.timer",
    "dgx-dashboard.service",
    "dgx-dashboard-admin.service",
)

HOST_FACTS_QUERY = """
printf 'modeset=%s\\n' "$(sudo -n cat /sys/module/nvidia_drm/parameters/modeset 2>/dev/null)"
printf 'fbdev=%s\\n' "$(sudo -n cat /sys/module/nvidia_drm/parameters/fbdev 2>/dev/null)"
printf 'cmdline_splash=%s\\n' "$(grep -cw splash /proc/cmdline)"
printf 'grub_splash=%s\\n' "$(sudo -n grep -E '^[[:space:]]*linux[[:space:]]' /boot/grub/grub.cfg 2>/dev/null | grep -cw splash)"
card=""
for entry in /sys/class/drm/card[0-9]*; do
  [ "$(basename "$(readlink "$entry/device/driver")")" = nvidia ] && card=${entry##*/card}
done
printf 'drm_card=%s\\n' "$card"
clients=/sys/kernel/debug/dri/$card/clients
if sudo -n test -r "$clients"; then
  printf 'drm_masters=%s\\n' "$(sudo -n cat "$clients" | awk 'NR > 1 && $4 == "y" {print $1 "/" $2}' | paste -sd, -)"
else
  printf 'drm_masters=unreadable\\n'
fi
vt=$(cat /sys/class/tty/tty0/active 2>/dev/null)
printf 'console=%s:%s\\n' "$vt" "$(sudo -n python3 -c 'import fcntl, struct, sys; f = open(sys.argv[1], "rb"); print(struct.unpack("i", fcntl.ioctl(f, 0x4B3B, struct.pack("i", 0)))[0])' "/dev/$vt" 2>/dev/null)"
printf 'kernel=%s\\n' "$(uname -r)"
printf 'fan_dkms=%s\\n' "$(dkms status -m dgx-spark-fan-control -k "$(uname -r)" 2>/dev/null | head -1)"
printf 'fan_module=%s\\n' "$(grep -c '^dgx_ec_fan_control ' /proc/modules)"
printf 'fan_cooling_device=%s\\n' "$(grep -lx dgx_ec_fan_floor /sys/class/thermal/cooling_device*/type 2>/dev/null | wc -l)"
printf 'fan_service=%s\\n' "$(systemctl cat dgx-fan-control.service >/dev/null 2>&1 && echo installed || echo absent)"
printf 'lldpd=%s\\n' "$(systemctl is-active lldpd.service 2>/dev/null || true)"
printf 'gpu=%s\\n' "$(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)"
"""
HOST_FACTS_QUERY += (
    r"""printf 'idle_services=%s\n' "$(for unit in """
    + " ".join(IDLE_SERVICES)
    + r"""; do
  state=$(systemctl is-enabled "$unit" 2>/dev/null)
  if systemctl is-active --quiet "$unit" || [ "$state" = enabled ] || [ "$state" = alias ]; then
    printf '%s\n' "$unit"
  fi
done | paste -sd, -)"
"""
)
HOST_FACTS_QUERY += r"""printf 'nvme_coalescing=%s\n' "$(for c in /dev/nvme[0-9]*; do
  [ -c "$c" ] || continue
  printf '%s:%s\n' "${c#/dev/}" "$(sudo -n nvme get-feature "$c" -f 8 2>/dev/null | awk -F'value:' 'NF > 1 {print $2}')"
done | paste -sd, -)"
printf 'nvme_coalescing_service=%s\n' "$(systemctl is-enabled nvidia-nvme-interrupt-coalescing.service 2>/dev/null)"
printf 'nvme_apst=%s\n' "$(for c in /dev/nvme[0-9]*; do
  [ -c "$c" ] || continue
  table=$(sudo -n nvme get-feature "$c" -f 0x0c -H 2>/dev/null | awk '
    /APSTE/ {enabled = ($0 ~ /Enabled/) ? 1 : 0}
    /Entry\[/ {entry = $0; gsub(/[^0-9]/, "", entry)}
    /\(ITPT\)/ {idle = $(NF - 1)}
    /\(ITPS\)/ {if (idle > 0) chain = chain entry ">" $NF "@" idle " "}
    END {printf "en=%s|%s", enabled, chain}')
  exits=$(sudo -n nvme id-ctrl "$c" 2>/dev/null | awk '
    /^ps +[0-9]+ :/ {for (f = 1; f <= NF; f++) if ($f ~ /^exlat:/) {sub("exlat:", "", $f); printf "%s:%s ", $2, $f}}')
  printf '%s|%s|%s\n' "${c#/dev/}" "$table" "$exits"
done | paste -sd, -)"
printf 'wifi_radio=%s\n' "$(nmcli radio wifi 2>/dev/null)"
printf 'wireless_ifaces=%s\n' "$(for d in /sys/class/net/*; do
  [ -e "$d/wireless" -o -e "$d/phy80211" ] || continue
  ifname=${d##*/}
  ips=$(ip -4 -o addr show dev "$ifname" 2>/dev/null | awk '{sub(/\/.*/, "", $4); print $4}' | paste -sd/ -)
  printf '%s:%s\n' "$ifname" "$ips"
done | paste -sd, -)"
"""
HOST_FACTS_QUERY += r"""printf 'buddy=%s\n' "$(awk '{
  for (i = 5; i <= NF; i++) free[i - 5] += $i
  if (NF - 5 > top) top = NF - 5
} END {for (i = 0; i <= top; i++) printf "%s%s", free[i] + 0, (i < top ? "," : "")}' /proc/buddyinfo 2>/dev/null)"
printf 'page_kb=%s\n' "$(( $(getconf PAGESIZE) / 1024 ))"
printf 'gpu_apps=%s\n' "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"
"""


def host_facts(output: str) -> dict[str, str]:
    facts = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            facts[key.strip()] = value.strip()
    return facts


def carveout_weights_enabled(cluster: dict) -> bool:
    return cluster.get("environment", {}).get("SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS", "1") != "0"


def drm_carveout_enabled(cluster: dict) -> bool:
    """True when a legacy profile enables carve-out weights AND mounts /dev/dri/card0.

    Since spark-ds41f-1120 (vLLM patch 0821), display carve-out allocations go
    directly through NVIDIA RM on /dev/nvidiactl (exposed via --gpus=all) and
    no longer require /dev/dri/card0 or nvidia-drm dumb buffers.
    """
    if not carveout_weights_enabled(cluster):
        return False
    args = cluster.get("container", {}).get("docker_run_args", [])
    return any("/dev/dri/card0" in str(arg) for arg in args)


def modeset_problems(node_name: str, facts: dict[str, str],
                     *, carveout_weights: bool = True) -> list[str]:
    """Problems when nvidia-drm does not drive displays (kernel mode setting).

    Without it only the firmware framebuffer shows a console, and only for a
    display attached at boot: a monitor or KVM connected later stays dark.
    Legacy pre-1120 profiles that mount /dev/dri/card0 for DRM dumb-buffer
    carve-out weights also require mode setting and refuse to start without it.
    On 1120+ (RM /dev/nvidiactl carve-out) or when carve-out weights are off,
    modeset=N only affects an attached monitor after boot and is a warning.
    """
    value = facts.get("modeset", "")
    if value == "Y":
        return []
    if not value:
        return [
            Warn(
                f"{node_name}: nvidia-drm modeset is unreadable "
                "(/sys/module/nvidia_drm/parameters/modeset); doctor needs "
                "passwordless sudo -n to read it"
            )
        ]
    if not carveout_weights:
        return [
            Warn(
                f"{node_name}: nvidia-drm modeset is {value}, expected Y "
                "(a display attached after boot stays dark; serving does not use "
                "DRM dumb buffers for the display carve-out on this profile); run "
                "sudo apt purge nvidia-drm-options-modeset0 and reboot with the "
                "service stopped"
            )
        ]
    return [
        f"{node_name}: nvidia-drm modeset is {value}, expected Y "
        "(a display attached after boot stays dark and the display carve-out cannot "
        "be allocated); run sudo apt purge nvidia-drm-options-modeset0 and reboot "
        "with the service stopped"
    ]


def fbdev_problems(node_name: str, facts: dict[str, str]) -> list[str]:
    """Problems when nvidia-drm provides no framebuffer console (required).

    With fbdev off, a display attached to the node shows no text console. The
    console framebuffer takes 8 MiB of the display carve-out at 1920x1080,
    which the serving configuration leaves free.
    """
    value = facts.get("fbdev", "")
    if value == "Y":
        return []
    if not value:
        return [
            Warn(
                f"{node_name}: nvidia-drm fbdev is unreadable "
                "(/sys/module/nvidia_drm/parameters/fbdev); doctor needs "
                "passwordless sudo -n to read it"
            )
        ]
    return [
        f"{node_name}: nvidia-drm fbdev is {value}, expected Y "
        "(an attached display gets no text console); remove the nvidia-drm fbdev=0 "
        "option (grep -rn fbdev /etc/modprobe.d), run sudo update-initramfs -u, and "
        "reboot with the service stopped"
    ]


def drm_card_problems(node_name: str, facts: dict[str, str],
                      *, carveout_weights: bool = True) -> list[str]:
    """Problems when no DRM card belongs to the nvidia driver.

    Legacy pre-1120 profiles mount that card by its PCI path
    (/dev/dri/by-path/pci-000f:01:00.0-card) for DRM dumb-buffer carve-out
    allocations and fail without it. On 1120+ (RM /dev/nvidiactl carve-out) or
    when carve-out weights are off, a missing nvidia DRM card only affects an
    attached display and is a warning.
    """
    if facts.get("drm_card"):
        return []
    if not carveout_weights:
        return [
            Warn(
                f"{node_name}: no DRM card belongs to the nvidia driver "
                "(an attached display may stay dark; serving does not use "
                "/dev/dri/card0 on this profile); check that nvidia-drm is loaded "
                "with modeset=Y (ls -l /dev/dri/by-path)"
            )
        ]
    return [
        f"{node_name}: no DRM card belongs to the nvidia driver, so the display "
        "carve-out cannot be allocated; check that nvidia-drm is loaded with "
        "modeset=Y (ls -l /dev/dri/by-path)"
    ]


def drm_master_problems(node_name: str, facts: dict[str, str]) -> list[str]:
    """Problems when a userspace process holds DRM master.

    The kernel's framebuffer console stops drawing while a userspace master
    exists. The serving worker closes its DRM file right after importing the
    display carve-out, so nothing should hold master on these headless nodes.
    """
    value = facts.get("drm_masters", "unreadable")
    if not value:
        return []
    if value == "unreadable":
        return [
            Warn(
                f"{node_name}: cannot read the DRM clients "
                "(/sys/kernel/debug/dri/<card>/clients)"
            )
        ]
    return [
        f"{node_name}: {value} holds DRM master, so the text console cannot draw; stop "
        "or restart that process (the serving worker releases the device right after "
        "startup)"
    ]


def console_mode_problems(node_name: str, facts: dict[str, str]) -> list[str]:
    """Warnings when the active virtual console is in graphics mode.

    The boot splash can leave it there after Plymouth quits; the kernel then
    draws no text on an attached display although fbcon is available. Serving
    does not depend on the text console.
    """
    console, _, mode = facts.get("console", "").partition(":")
    if mode == "0":
        return []
    if mode == "1":
        return [
            Warn(
                f"{node_name}: the active console ({console}) is in graphics mode, so an "
                "attached display shows no text; run scripts/host-recovery apply on the node "
                "(it switches the console to text mode now and removes splash from later boots)"
            )
        ]
    return [Warn(f"{node_name}: cannot read the active console mode")]


def splash_problems(node_name: str, facts: dict[str, str]) -> list[str]:
    """Warnings when the kernel boots with the Plymouth splash.

    Latent until the next boot: console_mode_problems reports a console that
    is in graphics mode now.

    Plymouth leaves the active console in graphics mode after it quits, so an
    attached display shows no text console after a boot. host-recovery
    installs /etc/default/grub.d/zz-spark-console.cfg, which drops splash from
    the command line that dgxstation-grub's menu.cfg sets.
    """
    if facts.get("cmdline_splash", "0") == "0":
        return []
    if facts.get("grub_splash") == "0":
        return [
            Warn(
                f"{node_name}: booted with splash, but the boot configuration omits it "
                "now; reboot with the service stopped to apply"
            )
        ]
    return [
        Warn(
            f"{node_name}: the kernel command line has splash, which leaves the console "
            "in graphics mode after boot; run scripts/host-recovery apply on the node (it "
            "installs /etc/default/grub.d/zz-spark-console.cfg and runs update-grub), "
            "then reboot with the service stopped"
        )
    ]


def idle_service_warnings(node_name: str, facts: dict[str, str]) -> list[str]:
    """Warnings when desktop services run on a headless inference node.

    Bluetooth, CUPS, cups-browsed and snapd (every installed snap is a desktop
    application) each hold memory on every node, up to about 50 MiB, and
    cups-browsed listens on the network. fwupd's daily refresh timer keeps its
    daemon resident; fwupdmgr starts the daemon on demand without it. NVIDIA's
    DGX Dashboard runs periodic update checks (spark-ota-check) that peak near
    850 MiB and wake fwupd and PackageKit.
    host-recovery disables them. These are warnings: they cost a little memory
    and do not fail doctor or block a benchmark.
    """
    units = [unit for unit in facts.get("idle_services", "").split(",") if unit]
    if not units:
        return []
    return [
        Warn(
            f"{node_name}: {', '.join(units)} enabled or running, though unused on a "
            f"headless inference node; run sudo systemctl disable --now "
            f"{' '.join(units)} (scripts/host-recovery apply does the same)"
        )
    ]


NVME_COALESCING_FIX = (
    "sudo systemctl mask --now nvidia-nvme-interrupt-coalescing.service && "
    "sudo /usr/bin/nvidia-nvme-interrupt-coalescing.sh disable"
)


# APST may put an idle drive to sleep, but not on the Engram reader's critical path:
# no state with this exit latency (us) or more before this much idle (ms).
NVME_APST_LIMITS = ((5000, 1000), (20000, 30000))
NVME_APST_FIX = (
    "set nvme_core apst_primary_timeout_ms=2000 and apst_secondary_timeout_ms=60000: now "
    "with echo 2000 | sudo tee /sys/module/nvme_core/parameters/apst_primary_timeout_ms, "
    "echo 60000 | sudo tee /sys/module/nvme_core/parameters/apst_secondary_timeout_ms "
    "and rewriting each controller's power/pm_qos_latency_tolerance_us to re-arm APST; "
    "at boot with 'options nvme_core apst_primary_timeout_ms=2000 "
    "apst_secondary_timeout_ms=60000' in /etc/modprobe.d/nvme-apst.conf and "
    "sudo update-initramfs -u"
)


def nvme_apst_warnings(node_name: str, facts: dict[str, str]) -> list[str]:
    """NVMe autonomous power state transitions (APST, feature 0x0c) that sleep too soon.

    With the kernel's defaults (apst_primary_timeout_ms=100) the fleet's drives enter
    PS3 (11 ms exit latency) after 100 ms idle and PS4 (46 ms) 2 s later. The Engram
    reader's first read after a pause then waits for the drive to wake: on TP3 at
    eight streams, whose steps exceed 100 ms, the GPU waited 1.5 s for Engram rows
    in a 62 s run and throughput was 3% lower than with the drive kept awake
    (experiments/2026-10-16-r6p). Each controller's chain of transitions from PS0
    must reach no state of NVME_APST_LIMITS' exit latency before its idle time.
    These are warnings: a sleeping drive costs latency, not correctness.
    """
    warnings = []
    for entry in filter(None, facts.get("nvme_apst", "").split(",")):
        # controller|en=<0|1>|<state>><target>@<idle ms> ...|<state>:<exit us> ...
        controller, enabled, chain, exits = (entry.split("|") + ["", "", ""])[:4]
        enabled = enabled.removeprefix("en=")
        exit_us = {}
        for item in exits.split():
            state, _, latency = item.partition(":")
            if state.isdigit() and latency.isdigit():
                exit_us[int(state)] = int(latency)
        if enabled not in ("0", "1") or not exit_us:
            warnings.append(Warn(
                f"{node_name}: cannot read NVMe APST on {controller} (nvme get-feature -f 0x0c "
                "and nvme id-ctrl need passwordless sudo and nvme-cli)"
            ))
            continue
        if enabled == "0":
            continue
        transitions = {}
        for item in chain.split():
            source, _, rest = item.partition(">")
            target, _, idle = rest.partition("@")
            if source.isdigit() and target.isdigit() and idle.isdigit() and int(idle) > 0:
                transitions[int(source)] = (int(target), int(idle))
        state, idle, seen = 0, 0, {0}
        while state in transitions:
            target, wait = transitions[state]
            state, idle = target, idle + wait
            if state in seen:
                break
            seen.add(state)
            latency = exit_us.get(state, 0)
            for limit_us, limit_ms in NVME_APST_LIMITS:
                if latency >= limit_us and idle < limit_ms:
                    warnings.append(Warn(
                        f"{node_name}: {controller} sleeps in PS{state} ({latency / 1000:g} ms "
                        f"to wake) after {idle} ms idle, and the Engram reader's first read "
                        f"after a pause waits for it (no state of {limit_us / 1000:g} ms or "
                        f"more before {limit_ms} ms idle); {NVME_APST_FIX}"
                    ))
                    break
    return warnings


def nvme_coalescing_warnings(node_name: str, facts: dict[str, str]) -> list[str]:
    """NVMe interrupt coalescing (feature 0x08), which this site keeps off.

    DGX OS's nvidia-nvme-interrupt-coalescing.service sets it to 0x107 at every
    boot: one interrupt per 8 completions or 100 us. Engram's disk rows are
    small direct reads, and a lone read then waits for the timer. The drives
    default to 0 (off) and cannot save the feature, so masking the service and
    running its disable action keeps it off across boots. These are warnings:
    coalescing costs read latency, not correctness.
    """
    values = {}
    for entry in filter(None, facts.get("nvme_coalescing", "").split(",")):
        controller, _, value = entry.partition(":")
        values[controller] = value.strip()
    if not values:
        return []
    warnings = []
    unreadable = sorted(c for c, v in values.items() if not v)
    if unreadable:
        warnings.append(Warn(
            f"{node_name}: cannot read NVMe interrupt coalescing on {', '.join(unreadable)} "
            "(nvme get-feature -f 8 needs passwordless sudo and nvme-cli)"
        ))
    enabled = sorted(f"{c}={v}" for c, v in values.items() if v and int(v, 16) != 0)
    if enabled:
        warnings.append(Warn(
            f"{node_name}: NVMe interrupt coalescing is on ({', '.join(enabled)}), "
            f"but this site keeps it off; run {NVME_COALESCING_FIX}"
        ))
    service = facts.get("nvme_coalescing_service", "")
    if service not in ("masked", "masked-runtime", "not-found", ""):
        warnings.append(Warn(
            f"{node_name}: nvidia-nvme-interrupt-coalescing.service is {service} and turns "
            f"coalescing back on at the next boot; run {NVME_COALESCING_FIX}"
        ))
    return warnings


def wireless_warnings(
    node_name: str,
    facts: dict[str, str],
    node: dict | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Warnings when the onboard Wi-Fi radio is enabled or carries control traffic."""
    warnings = []
    if facts.get("wifi_radio") == "enabled":
        warnings.append(
            Warn(
                f"{node_name}: Wi-Fi radio is enabled; run nmcli radio wifi off "
                "to keep the management port off wireless"
            )
        )
    wireless: dict[str, set[str]] = {}
    for entry in filter(None, facts.get("wireless_ifaces", "").split(",")):
        ifname, _, ips = entry.partition(":")
        if ifname:
            wireless[ifname] = {ip for ip in ips.split("/") if ip}
    if wireless and (node or env):
        ctrl_ip = (node or {}).get("control_ip") or (node or {}).get("management_ip")
        sock_ifaces = {
            iface.strip()
            for key in ("GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", "TP_SOCKET_IFNAME")
            for iface in (env or {}).get(key, "").split(",")
            if iface.strip() and not iface.strip().startswith("^")
        }
        if not sock_ifaces and (node or {}).get("management_interface"):
            sock_ifaces.add(str(node["management_interface"]))
        hit_ifaces = sorted(
            ifname
            for ifname, ips in wireless.items()
            if ifname in sock_ifaces or (ctrl_ip and ctrl_ip in ips)
        )
        if hit_ifaces:
            warnings.append(
                Warn(
                    f"{node_name}: control plane uses wireless interface {', '.join(hit_ifaces)}; "
                    "move master_addr / VLLM_HOST_IP (control_ip) and *_SOCKET_IFNAME onto wired "
                    "or ConnectX-7 links and run nmcli radio wifi off (#712)"
                )
            )
    return warnings


def lldp_cabled(cluster: dict, nodes: dict) -> bool:
    """Whether sparknet reads this profile's cables over LLDP: two or more nodes cabled
    to each other. It reads a switched fabric's rails from sysfs instead."""
    return len(nodes["nodes"]) > 1 and not topology.switched(topology.transport(cluster))


def lldpd_warnings(node_name: str, facts: dict[str, str], cabled: bool = True) -> list[str]:
    """A warning when a directly cabled node does not run lldpd.

    Nothing that serves reads LLDP. ``sparknet topology discover`` does: it asks
    each node's lldpd which port every cable ends on, and writes the node map and
    the netplan addresses from the answer. Without the daemon it finds no cables.
    """
    if not cabled or facts.get("lldpd") in (None, "active"):
        return []
    return [
        Warn(
            f"{node_name}: lldpd is not running ({facts['lldpd'] or 'not installed'}), so "
            "sparknet topology discover cannot read this node's cables: it reports every port as "
            "unresolved and cannot write the node map or the netplan addresses, which you then "
            "write by hand, and a later recabling is not detected. Serving, the launch preflight "
            "and bench do not use it. To install: sudo apt install lldpd (docs/setup.md, section 3)"
        )
    ]


FAN_CONTROL_VERSION = "0.1.4"
FAN_CONTROL_UPGRADE = ("https://github.com/christopherowen/dgx-spark-fan-control/blob/"
                       f"v{FAN_CONTROL_VERSION}/docs/maintenance.md")


def fan_control_problems(node_name: str, facts: dict[str, str],
                         fan_control: bool = True) -> list[str]:
    """The first missing layer of the DGX Spark additive fan-floor control.

    Each layer needs the one before it: the signed DKMS module for the running
    kernel, the loaded module and its dgx_ec_fan_floor cooling device. Those are
    what bench and kernel-lab cooling use to select the maximum floor.

    The dgx-fan-control service must be installed too: bench runs its
    performance curve while it measures, so every benchmark uses the same fan
    policy. Whether the service also runs while serving is the site's choice,
    so only its installation is checked.
    """
    if not fan_control:
        return []
    kernel = facts.get("kernel") or "the running kernel"
    if "installed" not in facts.get("fan_dkms", ""):
        return [
            Warn(
                f"{node_name}: DKMS dgx-spark-fan-control is not installed for {kernel}; "
                "serving does not need it, bench does; run "
                f"sudo dkms autoinstall -k {facts.get('kernel') or '$(uname -r)'}"
            )
        ]
    installed = facts["fan_dkms"].split(",", 1)[0].rpartition("/")[2]
    if installed != FAN_CONTROL_VERSION:
        return [
            Warn(
                f"{node_name}: DKMS dgx-spark-fan-control is {installed} for {kernel}, and this recipe "
                f"uses {FAN_CONTROL_VERSION}; upgrade from a checkout at v{FAN_CONTROL_VERSION} "
                f"({FAN_CONTROL_UPGRADE})"
            )
        ]
    if facts.get("fan_module") != "1":
        return [
            Warn(
                f"{node_name}: dgx_ec_fan_control is not loaded; serving does not need it, "
                "bench does; run sudo modprobe dgx_ec_fan_control (if it is refused, check "
                "that the fleet signing key is enrolled with mokutil --list-enrolled)"
            )
        ]
    if facts.get("fan_cooling_device") != "1":
        return [
            Warn(
                f"{node_name}: dgx_ec_fan_control is loaded but its dgx_ec_fan_floor cooling device "
                "is missing (the driver refused it at probe; see dmesg)"
            )
        ]
    if facts.get("fan_service") == "absent":
        return [
            Warn(
                f"{node_name}: dgx-fan-control.service is not installed, and bench runs its "
                "performance curve while measuring; from a dgx-spark-fan-control checkout run "
                "sudo install -D -m 0644 systemd/dgx-fan-control.service "
                "/etc/systemd/system/dgx-fan-control.service && sudo systemctl daemon-reload "
                "(enabling it for serving is optional)"
            )
        ]
    return []


def roce_hcas(cluster_or_node: dict, node: dict | None = None) -> list[str]:
    """The node's cable HCAs."""
    if node is None:
        target = cluster_or_node
        if "roce_hcas" in target and "roce_peer_hcas" not in target:
            return sorted(target["roce_hcas"])
        return sorted({hca for peers in target.get("roce_peer_hcas", {}).values() for hca in peers})
    return sorted(topology.hcas(cluster_or_node, node))


def roce_gid_setting(cluster: dict, node: dict) -> tuple[int, list[str]]:
    """The GID index sparknet and NCCL are configured to use, and the node's HCAs."""
    env = expected_environment(cluster, node)
    key = "SPARKNET_ROCE_GID_INDEX" if topology.is_sparknet(topology.transport(cluster)) else "NCCL_IB_GID_INDEX"
    index = int(env.get(key) or env.get("NCCL_IB_GID_INDEX") or 3)
    return index, sorted(topology.hcas(cluster, node))


def roce_gid_pins(cluster: dict, node: dict) -> dict[int, list[str]]:
    """GID slots a library of this configuration is fixed to, and the settings that fix each.

    Empty when the configuration finds GIDs by address (``topology.gid_pinned``):
    NCCL with NCCL_IB_GID_INDEX unset and sparknet 0.5.0. A configuration of an
    earlier sparknet pins NCCL_IB_GID_INDEX and SPARKNET_ROCE_GID_INDEX, per node
    through ``roce_gid_index``.
    """
    env = expected_environment(cluster, node)
    pins: dict[int, list[str]] = {}
    for key in ("SPARKNET_ROCE_GID_INDEX", "NCCL_IB_GID_INDEX"):
        if env.get(key):
            pins.setdefault(int(env[key]), []).append(key)
    return pins


def roce_gid_query(hcas: list[str]) -> str:
    """Every populated GID slot of each HCA as `hca index type gid ndev`."""
    return (
        f"for d in {' '.join(shlex.quote(hca) for hca in hcas)}; do "
        'p=/sys/class/infiniband/$d/ports/1; '
        '[ -d $p ] || { echo "$d missing"; continue; }; '
        'for g in $p/gids/*; do i=${g##*/}; '
        't=$(cat $p/gid_attrs/types/$i 2>/dev/null | tr -d " "); [ -n "$t" ] || continue; '
        'echo "$d $i $t $(cat $g) $(cat $p/gid_attrs/ndevs/$i 2>/dev/null)"; done; done'
    )


def roce_gid_findings(cluster: dict, node: dict, output: str,
                      hcas: list[str] | None = None) -> list[str]:
    """The node's RoCE GIDs (``roce_gid_query`` output) against what its libraries will select.

    Every cable HCA must carry exactly one IPv4 RoCE v2 GID, in its cable's
    subnet, at any index; a slot a library is still fixed to must hold it.
    """
    if hcas is None:
        hcas = roce_hcas(cluster, node)
    missing = {line.split()[0] for line in output.splitlines() if line.split()[1:] == ["missing"]}
    problems = [f"{node['name']}: RDMA device {hca} is missing" for hca in hcas if hca in missing]
    problems += topology.gid_subnet_problems(node, output)
    for index, settings in sorted(roce_gid_pins(cluster, node).items()):
        problems += roce_gid_problems(node["name"], index, [h for h in hcas if h not in missing], output, settings)
    return problems


def roce_gid_problems(node_name: str, index: int, hcas: list[str], output: str,
                      pinned_by: list[str] | None = None) -> list[str]:
    """Problems when an HCA's fixed GID slot is not its IPv4 RoCE v2 address.

    Only a library fixed to one slot needs this (``roce_gid_pins``). The kernel
    files a re-added address in the lowest free slot, so a link flap while RDMA
    resources still hold the old entry moves the IPv4 RoCE v2 GID (dgx3 moved
    from 3 to 4 when its peers rebooted under a starting service), and a
    library fixed to the old slot then fails (NCCL with an unhandled system
    error).
    """
    slots: dict[str, dict[int, tuple[str, str, str]]] = {hca: {} for hca in hcas}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == "missing" and fields[0] in slots:
            slots.pop(fields[0])
        elif len(fields) >= 4 and fields[0] in slots and fields[1].isdigit():
            ndev = fields[4] if len(fields) > 4 else ""
            slots[fields[0]][int(fields[1])] = (fields[2], fields[3], ndev)
    problems = []
    for hca in hcas:
        if hca not in slots:
            problems.append(f"{node_name}: RDMA device {hca} is missing")
            continue
        entries = slots[hca]
        kind, gid, _ = entries.get(index, ("", "", ""))
        if kind == "RoCEv2" and gid.startswith("0000:0000:0000:0000:0000:ffff:"):
            continue
        found = [
            (slot, ndev) for slot, (slot_kind, slot_gid, ndev) in sorted(entries.items())
            if slot_kind == "RoCEv2" and slot_gid.startswith("0000:0000:0000:0000:0000:ffff:")
        ]
        where = (
            f"its IPv4 RoCE v2 GID is at index {found[0][0]}; re-activate {found[0][1]} "
            "with nmcli while nothing uses RDMA"
            if found else "it has no IPv4 RoCE v2 GID"
        )
        pinned = f" (fixed by {', '.join(pinned_by)})" if pinned_by else ""
        problems.append(
            f"{node_name}: {hca} GID index {index}{pinned} is "
            f"{kind + ' ' + gid if kind else 'empty'}; {where}"
        )
    return problems


# Free memory by block size (/proc/buddyinfo). A freshly booted GB10 has almost all of
# its free memory in the largest blocks (dgx4, 2026-11-18: 233 of 512 MiB, 96% of 124
# GiB); after ten days of engine boots dgx1..dgx4 had none there. Memory taken from such
# a node as it comes (a kernel-lab bundle, the read probe below) read 7 to 9% slower. The
# engine is mostly spared: its loader soaks the small fragments first and places the
# weights in blocks of 2 MiB or more, and a rebooted node's rank was 2% faster in two
# kernels, at most 0.6% of a single-stream step (experiments/2026-11-17-c1-floor-now).
# The share below which this warns is between the two states measured, not a measured
# threshold, and it does not predict the read rate: two hours and an image build after
# its reboot dgx4 still had most of its free memory in 512 MiB blocks and read 238 to 245.
MEMORY_BLOCKS_MIN_SHARE = 0.5
MEMORY_BLOCKS_MIN_FREE_GIB = 32.0

# The plain read of experiments/2026-10-26-c1-100tps's expert-rate bundle: torch.sum
# over cold 220 MiB FP32 buffers, the L2 flushed before each replay. A freshly booted
# GB10 reads 249 to 254 GB/s; the nodes up ten days read 229 to 237, and dgx4 two hours
# and an image build after its reboot 238 to 245. It is the rate of memory taken now, as
# it comes: what a kernel-lab timing on this node will see, not what a running engine
# has (see above).
STREAM_RATE_MIN_GBS = 245.0
STREAM_RATE_PROBE = r"""
import json, statistics, torch
MIB, COUNT, REPEATS = 220, 8, 7
flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
buffers = [torch.ones((MIB << 20) // 4, dtype=torch.float32, device="cuda") for _ in range(COUNT)]
sums = [torch.empty((), dtype=torch.float32, device="cuda") for _ in buffers]
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for buffer, out in list(zip(buffers, sums))[:2]:
        torch.sum(buffer, dim=0, out=out)
torch.cuda.current_stream().wait_stream(side)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for buffer, out in zip(buffers, sums):
        torch.sum(buffer, dim=0, out=out)
samples = []
for _ in range(REPEATS):
    flush.zero_()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    samples.append(start.elapsed_time(end) * 1e3 / COUNT)
print(json.dumps({"stream_gbs": round((MIB << 20) / statistics.median(samples) / 1e3, 1)}))
"""


def free_memory_blocks(facts: dict[str, str]) -> tuple[float, float, int] | None:
    """(GiB free in the buddy allocator, the share of it in the largest block size, that
    size in MiB), from the node's free blocks by order; None when they cannot be read."""
    try:
        counts = [int(count) for count in facts.get("buddy", "").split(",")]
        page_kb = int(facts.get("page_kb", ""))
    except ValueError:
        return None
    sizes = [count * page_kb * (1 << order) for order, count in enumerate(counts)]
    if page_kb <= 0 or not sum(sizes):
        return None
    return sum(sizes) / 2**20, sizes[-1] / sum(sizes), page_kb * (1 << (len(counts) - 1)) // 1024


def memory_block_warnings(node_name: str, facts: dict[str, str]) -> list[str]:
    """Warn when an idle node's free memory is no longer in its largest blocks.

    A node whose GPU a process holds is not judged: its free memory is what the engine
    left. ``cluster start`` drops caches and compacts before a launch
    (``host.defragment_before_start``), which does not rebuild blocks that kernel
    allocations have split; a reboot does. The warning is about timings taken on the
    node, not a reason to reboot a serving node: the engine's loader places its weights
    in blocks of 2 MiB or more either way.
    """
    if facts.get("gpu_apps", "0") not in ("", "0"):
        return []
    blocks = free_memory_blocks(facts)
    if blocks is None:
        return [Warn(f"{node_name}: cannot read free memory by block size (/proc/buddyinfo)")]
    free_gib, share, block_mib = blocks
    if free_gib < MEMORY_BLOCKS_MIN_FREE_GIB or share >= MEMORY_BLOCKS_MIN_SHARE:
        return []
    return [
        Warn(
            f"{node_name}: {share:.0%} of the {free_gib:.0f} GiB free is in {block_mib} MiB blocks, "
            "the largest (a 64 KiB node after a boot: 96% in 512 MiB blocks). Kernel-lab timings "
            "on such nodes read memory 7 to 9% slower; the engine lost at most 0.6% of a step "
            "(experiments/2026-11-17-c1-floor-now). Measure the node with doctor --live "
            "--stream-rate; a reboot restores the blocks until the memory is used again"
        )
    ]


def stream_rate_script(image: str) -> str:
    """The node's command for the read probe: nothing when a process holds the GPU or the
    kernel lab's GPU lock is taken, else one container of ``image`` for about a second of
    GPU time."""
    return "\n".join((
        'if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then',
        "  echo '{\"skipped\": \"a process holds the GPU\"}'; exit 0",
        "fi",
        "exec 9>>/tmp/spark-lab-gpu.lock",
        "if ! flock -n 9; then echo '{\"skipped\": \"the kernel lab holds the GPU lock\"}'; exit 0; fi",
        f"docker run --rm --gpus all --ipc=host --entrypoint python3 {shlex.quote(image)} -c {shlex.quote(STREAM_RATE_PROBE)}",
    ))


def stream_rate_findings(node_name: str, output: str, stderr: str = "") -> tuple[str, list[str]]:
    """(the line to print, warnings) for one node's probe output."""
    record = None
    for line in reversed(output.splitlines()):
        if line.startswith("{"):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            break
    if not isinstance(record, dict) or not ({"stream_gbs", "skipped"} & set(record)):
        lines = [line.strip() for line in (stderr or output).splitlines() if line.strip()]
        detail = next((line for line in lines if "rror" in line), lines[-1] if lines else "no output")
        return f"{node_name}: memory read not measured", [
            Warn(f"{node_name}: the memory read probe failed ({detail[:160]})")
        ]
    if "skipped" in record:
        return f"{node_name}: memory read not measured ({record['skipped']})", []
    rate = float(record["stream_gbs"])
    line = f"{node_name}: memory reads at {rate:.1f} GB/s"
    if rate >= STREAM_RATE_MIN_GBS:
        return line, []
    return line, [
        Warn(
            f"{node_name}: memory reads at {rate:.1f} GB/s, below {STREAM_RATE_MIN_GBS:.0f} "
            "(a freshly booted GB10 reads 249 to 254): kernel-lab timings here will read low "
            "against a fresh node's; the engine is mostly spared (experiments/2026-11-17-c1-floor-now)"
        )
    ]


def stream_rates(cluster: dict, nodes: dict) -> tuple[list[str], list[str]]:
    """Each node's read probe, run at once: (lines to print, warnings)."""
    script = stream_rate_script(cluster["container"]["image"])

    def probe(node: dict) -> tuple[str, list[str]]:
        result = remote_shell(nodes, node, script)
        return stream_rate_findings(node["name"], result.stdout, result.stderr)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes["nodes"])) as pool:
        results = list(pool.map(probe, nodes["nodes"]))
    return [line for line, _ in results], [warning for _, warnings in results for warning in warnings]


LATCHED_CLOCK_MHZ = 1000


def clock_latch_problems(node_name: str, facts: dict[str, str], serving: bool) -> list[str]:
    """Problems when a serving GB10 GPU is held far below its normal clock.

    With the service running, a healthy GB10 stays in P0 at about 2,400 MHz
    even between requests. The known GB10 clock latch holds it at about
    500-720 MHz and 10-14 W with no throttle reason; nvidia-smi -lgc and a warm
    reboot do not clear it, removing AC power does (docs/recovery.md).
    """
    if not serving:
        return []
    clock, _, power = facts.get("gpu", "").partition(",")
    try:
        clock_mhz = float(clock)
    except ValueError:
        return [Warn(f"{node_name}: cannot read the GPU clock")]
    if clock_mhz >= LATCHED_CLOCK_MHZ:
        return []
    return [
        f"{node_name}: GPU clock is {clock_mhz:.0f} MHz at {power.strip() or '?'} W while serving; "
        "the GB10 clock latch holds it there until AC power is removed (docs/recovery.md)"
    ]


def node_kernel_inventory(nodes: dict, node: dict, policy: dict) -> dict:
    query = shlex.join(["sudo", "-n", "python3", "-c",
                        (ROOT / "scripts/kernel_readiness.py").read_text(), json.dumps(policy)])
    result = remote_shell(nodes, node, query)
    try:
        inventory = json.loads(result.stdout)
        if result.returncode or not isinstance(inventory, dict):
            raise ValueError(result.stderr.strip() or "kernel inventory failed")
        return inventory
    except (ValueError, TypeError) as error:
        return {"errors": [str(error)]}


def kernel_policy_path(cluster: dict) -> Path:
    relative = Path(cluster.get("host", {}).get("kernel_policy", "config/kernel-trial.json"))
    path = (ROOT / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError("host.kernel_policy must remain inside the repository")
    return path


def docker_error_detail(node_name: str, stderr: str) -> str:
    detail = stderr.strip()
    if "permission denied" in detail.lower() and "docker.sock" in detail:
        return (
            f"{detail}; add the SSH user to the docker group on {node_name} "
            "(sudo usermod -aG docker $USER) and reconnect"
        )
    return detail


def ssh_resolution_problem(nodes: dict, node: dict) -> str | None:
    """Return an error if the node's SSH target resolves via neither SSH config nor DNS."""
    target = ssh_target(nodes, node)
    host = node["name"]
    proxy = False
    config = subprocess.run(
        ["ssh", "-G", target],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if config.returncode == 0:
        for line in config.stdout.splitlines():
            key, _, value = line.partition(" ")
            value = value.strip()
            if key == "hostname" and value:
                host = value
            elif key in ("proxycommand", "proxyjump") and value and value != "none":
                proxy = True
    if proxy:
        return None
    try:
        socket.getaddrinfo(host, None)
    except socket.gaierror:
        ip = node.get("management_ip", "<management_ip>")
        return (
            f"{node['name']}: cannot resolve SSH target {target!r} via SSH config or DNS; "
            f"add a Host {node['name']} entry (HostName {ip}) in ~/.ssh/config, add it to "
            "DNS or /etc/hosts, or set the node's name to a resolvable hostname or IP"
        )
    return None


def site_node_problems(cluster: dict, nodes: dict) -> list[str]:
    if any(str(node.get("management_ip", "")).startswith("192.0.2.") for node in nodes.get("nodes", [])):
        nodes_file = cluster.get("nodes_config", NODES_FILE)
        return [
            f"{nodes_file} still contains example placeholder addresses (192.0.2.x); "
            "edit it with your nodes' names, ssh_user, and management IPs"
        ]
    return [
        problem
        for node in nodes.get("nodes", [])
        if (problem := ssh_resolution_problem(nodes, node)) is not None
    ]


def image_content_identity(image: dict) -> dict:
    """Store-independent identity of the filesystem and runtime configuration.

    Docker's containerd and overlay2 stores can give a faithful copy different
    image IDs. The ordered layer digests and image Config survive that copy.
    """
    rootfs = image.get("RootFS")
    layers = rootfs.get("Layers") if isinstance(rootfs, dict) else None
    if not isinstance(layers, list) or not layers or any(
        not isinstance(layer, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", layer)
        for layer in layers
    ):
        raise ValueError("running image has no valid ordered layer digests")
    config = image.get("Config")
    if not isinstance(config, dict):
        raise ValueError("running image has no runtime Config")
    if config.get("Labels") is not None and not isinstance(config["Labels"], dict):
        raise ValueError("running image has invalid source labels")
    content = json.dumps({"layers": layers, "config": config}, sort_keys=True, separators=(",", ":"))
    return {
        "image_layers": layers,
        "image_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def read_running_image(nodes: dict, node: dict, image_id: str | None) -> dict:
    """Inspect only the immutable image named by the container, never its tag."""
    if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("cannot read the serving container's immutable image ID")
    result = run_ssh(nodes, node, "docker", "image", "inspect", image_id)
    if result.returncode:
        raise ValueError(f"cannot inspect running image {image_id}")
    try:
        images = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("cannot decode the running image inspection") from error
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise ValueError("running image inspection must contain one image")
    image_content_identity(images[0])
    return images[0]


def running_image_problems(cluster: dict, images: dict[str, dict]) -> list[str]:
    """Validate captured running images against the profile and one another."""
    problems = []
    reference = None
    for node, image in images.items():
        content = image_content_identity(image)["image_content_sha256"]
        labels = image["Config"].get("Labels") or {}
        for key, wanted in cluster["container"]["expected_labels"].items():
            if labels.get(key) != wanted:
                problems.append(f"{node}: running image label {key}={labels.get(key)!r}, expected {wanted!r}")
        if reference is None:
            reference = (node, content)
        elif content != reference[1]:
            problems.append(f"{node}: running image layers or runtime Config differ from {reference[0]}")
    return problems


def live_doctor(cluster: dict, nodes: dict) -> list[str]:
    """Findings about the live cluster: errors, and Warn for the minor ones."""
    site_problems = site_node_problems(cluster, nodes)
    if site_problems:
        return site_problems
    cluster, problems = without_missing_glp(cluster, nodes)
    running_images = {}
    trial_path = kernel_policy_path(cluster)
    trial = json.loads(trial_path.read_text()) if trial_path.exists() else None
    kernel_inventories = {}
    corrections = []
    name = cluster["container"]["name"]
    kernel_profile = cluster.get("host", {}).get("kernel")
    drm_carveout = drm_carveout_enabled(cluster)
    for node in nodes["nodes"]:
        desktop = remote_shell(nodes, node, DESKTOP_QUERY)
        if desktop.returncode == 255:
            detail = desktop.stderr.strip() or f"cannot connect to {ssh_target(nodes, node)}"
            problems.append(f"{node['name']}: SSH failed ({detail})")
            continue
        if trial:
            inventory = node_kernel_inventory(nodes, node, trial)
            kernel_inventories[node["name"]] = inventory
            warnings, fixes = kernel_readiness.findings(
                node["name"], inventory, trial
            )
            profile_warnings = kernel_readiness.profile_problems(
                node["name"], inventory, kernel_profile, trial
            )
            problems.extend(
                Warn(warning)
                for warning in dict.fromkeys([*warnings, *profile_warnings])
            )
            corrections.extend(fixes)
        problems.extend(desktop_problems(node["name"], desktop.stdout))
        boot = run_ssh(nodes, node, "efibootmgr")
        problems.extend(boot_order_problems(node["name"], boot.stdout))
        facts = host_facts(remote_shell(nodes, node, HOST_FACTS_QUERY).stdout)
        problems.extend(modeset_problems(node["name"], facts, carveout_weights=drm_carveout))
        problems.extend(fbdev_problems(node["name"], facts))
        problems.extend(drm_card_problems(node["name"], facts, carveout_weights=drm_carveout))
        problems.extend(drm_master_problems(node["name"], facts))
        problems.extend(console_mode_problems(node["name"], facts))
        problems.extend(splash_problems(node["name"], facts))
        problems.extend(idle_service_warnings(node["name"], facts))
        problems.extend(nvme_coalescing_warnings(node["name"], facts))
        problems.extend(nvme_apst_warnings(node["name"], facts))
        problems.extend(wireless_warnings(node["name"], facts, node=node, env=cluster.get("environment")))
        problems.extend(lldpd_warnings(node["name"], facts, cabled=lldp_cabled(cluster, nodes)))
        problems.extend(fan_control_problems(
            node["name"], facts, fan_control=fan_control_enabled(cluster)))
        problems.extend(memory_block_warnings(node["name"], facts))
        _, hcas = roce_gid_setting(cluster, node)
        gids = remote_shell(nodes, node, roce_gid_query(hcas))
        problems.extend(roce_gid_findings(cluster, node, gids.stdout, hcas=hcas))
        problems.extend(node_glp_problems(cluster, nodes, node))
        result = run_ssh(nodes, node, "docker", "inspect", name)
        if result.returncode:
            detail = docker_error_detail(node["name"], result.stderr)
            if "sudo usermod -aG docker" in detail:
                problems.append(Warn(f"{node['name']}: cannot inspect {name}: {detail}"))
                continue
            if "no such object" in detail.lower():
                label_result = run_ssh(
                    nodes, node, "docker", "image", "inspect", cluster["container"]["image"]
                )
                if label_result.returncode:
                    problems.append(
                        Warn(f"{node['name']}: runtime image {cluster['container']['image']} is not present")
                    )
                else:
                    labels = json.loads(label_result.stdout)[0]["Config"].get("Labels") or {}
                    for key, wanted in cluster["container"]["expected_labels"].items():
                        if labels.get(key) != wanted:
                            problems.append(
                                f"{node['name']}: image label {key}={labels.get(key)!r}, expected {wanted!r}"
                            )
                    problems.append(Warn(f"{node['name']}: serving container {name} is not running"))
                continue
            problems.append(f"{node['name']}: cannot inspect {name}: {detail}")
            continue
        inspection = json.loads(result.stdout)[0]
        if not inspection["State"]["Running"]:
            problems.append(f"{node['name']}: container is not running")
        problems.extend(
            clock_latch_problems(node["name"], facts, bool(inspection["State"]["Running"]))
        )
        if inspection["Config"]["Image"] != cluster["container"]["image"]:
            problems.append(
                f"{node['name']}: image tag {inspection['Config']['Image']!r} differs"
            )
        actual_command = inspection["Config"]["Cmd"]
        wanted_command = expected_command(cluster, node)
        if actual_command != wanted_command:
            problems.append(f"{node['name']}: vLLM command differs from promoted config")
        try:
            image = read_running_image(nodes, node, inspection.get("Image"))
        except ValueError as error:
            problems.append(f"{node['name']}: {error}")
            continue
        running_images[node["name"]] = image
        problems.extend(
            environment_problems(
                node["name"],
                expected_environment(cluster, node),
                inspection["Config"]["Env"],
                image["Config"].get("Env"),
            )
        )
        problems.extend(
            mount_problems(node["name"], expected_binds(cluster), inspection["HostConfig"]["Binds"])
        )
        if configured_cpuset(cluster) is not None:
            capacities = cpu_capacities(run_ssh(
                nodes, node, "sh", "-c", "grep -H . /sys/devices/system/cpu/cpu[0-9]*/cpu_capacity").stdout)
            problems.extend(cpuset_problems(node["name"], configured_cpuset(cluster),
                                            inspection["HostConfig"].get("CpusetCpus") or "", capacities))
    problems.extend(running_image_problems(cluster, running_images))
    if trial:
        problems.extend(kernel_readiness.alignment(kernel_inventories))
        if corrections:
            problems.append(Warn("Kernel corrections (not executed):\n  " + "\n  ".join(dict.fromkeys(corrections))))
    return problems


def split_findings(findings: list[str]) -> tuple[list[str], list[str]]:
    """Errors and warnings, each in their original order."""
    errors = [finding for finding in findings if not isinstance(finding, Warn)]
    warnings = [finding for finding in findings if isinstance(finding, Warn)]
    return errors, warnings


LIVE_RECOMMENDATION_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "install 64kb page kernel",
        "sudo scripts/install-64k-kernel",
        (
            "64 KiB package versions differ",
            "missing or empty /boot/",
            "candidate headers or CONFIG_ARM64_64K_PAGES=y missing",
            "candidate cpupower tool is missing",
            "nvidia is not prepared for",
            "nvidia_uvm is not prepared for",
            "nvidia_drm is not prepared for",
            "mlx5_core is not prepared for",
            "mlx5_ib is not prepared for",
            "dgx_ec_fan_control is not prepared for",
            "fan-control DKMS is not installed for",
            "candidate fan signature",
            "driver version differs from",
            "loaded driver differs from",
            "loaded NVIDIA RM build",
            "candidate is missing from GRUB",
            "normal boot is not explicitly pinned",
            "GRUB next-boot/recovery override",
            "running kernel/page size is outside the prepared pair",
            "-byte pages; running",
            "qualified NVIDIA RM build",
            "NVIDIA system memory pool setting differs",
            "serving profile recommends NVIDIA",
        ),
    ),
    (
        "install memory-saver",
        "scripts/install-memory-saver",
        (
            "memory-saver",
            "uvm_leaf_packing",
        ),
    ),
    (
        "configure swap",
        "sudo scripts/configure-swap",
        (
            "4 KiB swap",
            "64 KiB swap",
            "64 KiB THP/reserve policy",
            "boot memory service",
        ),
    ),
    (
        "configure nvme sleep settings",
        "sudo scripts/configure-nvme-sleep",
        (
            "sleeps in PS",
            "NVMe APST",
        ),
    ),
    (
        "disable nvme interrupt coalescing",
        "sudo scripts/disable-nvme-coalescing",
        (
            "NVMe interrupt coalescing",
            "nvidia-nvme-interrupt-coalescing.service",
        ),
    ),
    (
        "management port off wireless",
        "sudo nmcli radio wifi off",
        (
            "Wi-Fi radio is enabled",
            "wireless",
        ),
    ),
    (
        "configure docker permissions",
        "sudo usermod -aG docker $USER",
        (
            "sudo usermod -aG docker",
            "docker.sock",
        ),
    ),
    (
        "install fan-control",
        "scripts/install-fan-control",
        (
            "dgx-spark-fan-control",
            "dgx_ec_fan_control is not loaded",
            "dgx_ec_fan_floor cooling device",
            "dgx-fan-control.service",
        ),
    ),
    (
        "set cpu governor to performance",
        "sudo systemctl restart nv-cpu-governor.service",
        (
            "CPU governors differ from performance",
            "running cpupower tool is missing",
        ),
    ),
    (
        "disable idle desktop services and boot splash",
        "scripts/host-recovery apply",
        (
            "enabled or running, though unused on a headless inference node",
            "boots to graphical.target",
            "active console",
            "splash",
        ),
    ),
    (
        "enable nvidia-drm modeset",
        "sudo apt-get purge -y nvidia-drm-options-modeset0 && sudo update-initramfs -u -k all",
        (
            "nvidia-drm modeset",
            "nvidia-drm fbdev",
            "DRM clients",
        ),
    ),
    (
        "put installed OS first in UEFI boot order",
        "efibootmgr && sudo efibootmgr --bootorder <ubuntu-entry>,<remaining-entries>",
        (
            "UEFI boots",
            "UEFI boot order",
        ),
    ),
    (
        "defragment model weight shards",
        "bin/spark doctor --fragmentation-commands",
        (
            "weight shards are not at ext4's best extent count",
        ),
    ),
    (
        "build and distribute runtime image",
        "bin/spark build prepare && bin/spark --cluster-config <profile> build image --apply && bin/spark --cluster-config <profile> build distribute --apply",
        (
            "runtime image",
        ),
    ),
    (
        "start serving container",
        "bin/spark --cluster-config <profile> cluster start --apply",
        (
            "serving container",
        ),
    ),
)


def summarize_live_warnings(
    warnings: list[str],
    cluster_config: str = DEFAULT_CLUSTER_CONFIG,
) -> list[str]:
    """Group low-level live warnings into concise user recommendations with commands."""
    grouped: dict[str, dict[str, None]] = {label: {} for label, _, _ in LIVE_RECOMMENDATION_RULES}
    matched_labels: dict[str, None] = {}
    passthrough: list[str] = []
    for warning in warnings:
        if warning.startswith(("Kernel corrections (not executed):", "weight fragmentation:")):
            continue
        node, separator, detail = warning.partition(": ")
        text = detail if separator else warning
        matched = False
        for label, _, patterns in LIVE_RECOMMENDATION_RULES:
            if any(pattern in warning or pattern in text for pattern in patterns):
                matched_labels[label] = None
                if separator and node != "weight fragmentation":
                    grouped[label][node] = None
                matched = True
                break
        if not matched:
            passthrough.append(Warn(warning))
    summary: list[str] = []
    for label, command, _ in LIVE_RECOMMENDATION_RULES:
        if label not in matched_labels:
            continue
        nodes = list(grouped[label])
        suffix = f" ({', '.join(nodes)})" if nodes else ""
        rendered_command = command.replace("<profile>", cluster_config)
        summary.append(Warn(f"recommend: {label}{suffix} -> run: {rendered_command}"))
    summary.extend(passthrough)
    return summary


def report_doctor(problems: list[str], warnings: list[str], live: bool) -> int:
    """Print warnings and problems; only problems fail."""
    for warning in warnings:
        print(f"WARN: {warning}")
    for problem in problems:
        print(f"ERROR: {problem}")
    if problems:
        return 1
    summary = "configuration OK" + ("; live cluster matches" if live else "")
    if warnings:
        summary += f" with {len(warnings)} warning" + ("s" if len(warnings) > 1 else "")
    print(summary)
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    _, canonical = cluster_config_path(args)
    if (
        getattr(args, "cluster_config", None) is None
        and not (ROOT / "config/nodes-ring4.local.json").is_file()
        and (ROOT / "config/switched.local.json").is_file()
    ):
        canonical = "config/switched.local.json"
    cluster, nodes, upstreams = configuration(args)
    if getattr(args, "fragmentation_commands", False):
        for finding in weight_fragmentation_findings(cluster, nodes, commands=True):
            print(finding)
        return 0
    if getattr(args, "stream_rate", False) and not args.live:
        raise SystemExit("--stream-rate measures the nodes: use it with --live")
    findings = local_doctor(cluster, nodes, upstreams)
    if args.live and not split_findings(findings)[0]:
        findings.extend(live_doctor(cluster, nodes))
        if not split_findings(findings)[0]:
            findings.extend(weight_fragmentation_findings(cluster, nodes))
        if getattr(args, "stream_rate", False):
            lines, warnings = stream_rates(cluster, nodes)
            for line in lines:
                print(line)
            findings.extend(warnings)
    problems, warnings = split_findings(findings)
    if args.live and not getattr(args, "verbose", False):
        warnings = summarize_live_warnings(warnings, canonical)
    return report_doctor(problems, warnings, args.live)


def command_site_shape(args: argparse.Namespace) -> int:
    recipe = read_json(DEFAULT_CLUSTER_CONFIG)
    current = cluster_recipe.site_shape(ROOT)
    if args.name is None:
        default = recipe.get("default_shape")
        if current:
            print(current)
        elif default:
            print(f"{default} (the recipe's default; {cluster_recipe.SITE_FILE} is missing)")
        else:
            print(f"no shape named ({cluster_recipe.SITE_FILE} is missing)")
        return 0
    entry = cluster_recipe.shape_entry(recipe, args.name)
    if not args.apply:
        print(f"would name this site {args.name} ({entry['nodes']} nodes, {entry['fabric']['transport']}, node map "
              f"{entry.get('nodes_config', NODES_FILE)}); pass --apply, then `cluster sync` and `doctor --live`")
        return 0
    (ROOT / cluster_recipe.SITE_FILE).write_text(json.dumps({"shape": args.name}, indent=2) + "\n", encoding="utf-8")
    print(f"this site is cabled as {args.name}; run `cluster sync` to copy it and `doctor --live` to check the links")
    return 0


def command_topology_create(args: argparse.Namespace) -> int:
    cluster, _, _ = configuration(args)
    nodes_path = repository_config_path(args.nodes_config)
    output = repository_config_path(args.output)
    nodes_relative = nodes_path.relative_to(ROOT.resolve()).as_posix()
    if not nodes_path.is_file():
        raise SystemExit(missing_nodes_message(nodes_relative))
    nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
    try:
        candidate = topology.candidate(cluster, nodes, nodes_relative)
    except (ValueError, StopIteration) as error:
        raise SystemExit(f"invalid topology: {error}") from error
    # A profile that names $HOME keeps it, so the candidate resolves it against its own
    # node map when it is read.
    try:
        written_home = read_json(cluster_config_path(args)[1]).get("host", {}).get("home", HOME_VARIABLE)
    except SystemExit:
        written_home = None
    if written_home and HOME_VARIABLE in written_home:
        candidate["host"]["home"] = written_home
    output.parent.mkdir(parents=True, exist_ok=True)
    # A new candidate must never overwrite the promoted config or another run.
    with output.open("x") as handle:
        json.dump(candidate, handle, indent=2)
        handle.write("\n")
    print(f"created {output.relative_to(ROOT.resolve())} ({len(nodes['nodes'])} nodes, "
          f"{topology.transport(candidate)}, launch disabled)")
    return 0


def collective_probe_command(cluster: dict, nodes: dict, node: dict, port: int) -> list[str]:
    env = expected_environment(cluster, node)
    env.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH,NET")
    command = ["docker", "run", "--rm", "--name=spark-collective-probe",
               "--network=host", "--gpus=all", "--memory=12g", "--memory-swap=12g",
               "--device=/dev/infiniband:/dev/infiniband:rwm", "--cap-add=IPC_LOCK",
               "--ulimit=memlock=-1:-1", "--ulimit=core=0", "--shm-size=256m",
               "--entrypoint=/usr/bin/timeout"]
    for key, value in sorted(env.items()):
        command.extend(["--env", f"{key}={value}"])
    command.extend(["--volume", f"{repository_path(cluster)}/scripts/probe_collectives.py:/probe.py:ro",
                    cluster["container"]["image"], "--signal=TERM", "--kill-after=10s", "600s",
                    "python3", "/probe.py", "--rank", str(node["rank"]),
                    "--world-size", str(len(nodes["nodes"])), "--master-addr",
                    cluster["distributed"]["master_addr"], "--master-port", str(port)])
    return command


def command_topology_probe(args: argparse.Namespace) -> int:
    cluster, nodes, upstreams = configuration(args)
    errors, warnings = split_findings(local_doctor(cluster, nodes, upstreams))
    if errors:
        return report_doctor(errors, warnings, False)
    if not 1 <= args.master_port <= 65535 or str(args.master_port) == str(cluster["distributed"]["master_port"]):
        raise SystemExit("probe port must be valid and different from the serving rendezvous port")
    node = node_by_name(nodes, args.node)
    print("# Plan only. Run on this node during a coordinated window with serving stopped.")
    print(shlex.join(collective_probe_command(cluster, nodes, node, args.master_port)))
    return 0


def report_weight_fragmentation(reports: dict, commands: bool = False) -> list[str]:
    """Report every shard above ext4's best count, regardless of severity score."""
    findings = []
    has_fragmentation = False
    for name, report in reports.items():
        if report.get("error"):
            findings.append(Warn(f"{name}: cannot check weight fragmentation: {report['error']}"))
            continue
        shards = report["shards"]
        fragmented = [shard for shard in shards if shard["current"] > shard["best"]]
        if not fragmented:
            continue
        has_fragmentation = True
        current = sum(f["current"] for f in shards)
        best = sum(f["best"] for f in shards)
        findings.append(Warn(
            f"{name}: {len(fragmented)}/{len(shards)} weight shards are not at ext4's best "
            f"extent count ({current} current / {best} best across the checkpoint)"
        ))
        if commands:
            findings.append(f"# On {name}, during a coordinated maintenance window:")
            findings.extend(shlex.join(["sudo", "e4defrag", "-v", f["path"]]) for f in fragmented)
    if has_fragmentation and not commands:
        findings.append(Warn(
            "weight fragmentation: use doctor --fragmentation-commands to print targeted "
            "correction commands; no defragmentation is performed by doctor"
        ))
    return findings


def weight_fragmentation_findings(cluster: dict, nodes: dict, commands: bool = False) -> list[str]:
    """Read extent metadata and report only non-ideal layouts or unavailable checks."""
    source = cluster_recipe.model_snapshot(cluster) if "models" in cluster else next(
        (source for source, target, _ in cluster["container"]["mounts"] if target == "/models"),
        None,
    )
    if source is None:
        return [Warn("cannot check weight fragmentation: no model mount")]
    root = host_path(cluster, source)
    code = (ROOT / "scripts/weight_fragmentation.py").read_text()

    def inspect(node: dict) -> dict:
        try:
            result = subprocess.run(
                ["ssh", *ssh_options(), ssh_target(nodes, node),
                 shlex.join(["python3", "-c", code, root])],
                capture_output=True, text=True, timeout=30, check=True,
            )
            return json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            # CalledProcessError/TimeoutExpired include the entire SSH script.
            return {"error": type(error).__name__}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        reports = dict(zip(
            (node["name"] for node in nodes["nodes"]),
            pool.map(inspect, nodes["nodes"]),
        ))
    return report_weight_fragmentation(reports, commands=commands)


def command_status(args: argparse.Namespace) -> int:
    cluster, nodes, _ = configuration(args)
    name = cluster["container"]["name"]
    failed = False
    for node in nodes["nodes"]:
        result = run_ssh(
            nodes,
            node,
            "docker",
            "ps",
            "--filter",
            f"name=^{name}$",
            "--format",
            "{{.Names}}|{{.Image}}|{{.Status}}",
        )
        guard = run_ssh(
            nodes,
            node,
            "systemctl",
            "is-active",
            f"{name}-memguard.service",
        )
        output = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        guard_state = guard.stdout.strip() if guard.returncode == 0 else "inactive"
        print(f"{node['name']}\t{output or 'not running'}\tmemguard={guard_state}")
        failed |= result.returncode != 0 or not bool(result.stdout.strip())
    return int(failed)
