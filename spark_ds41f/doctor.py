"""Repository and live-node doctor checks, host facts, RoCE GIDs and status reporting."""

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

class Warn(str):
    """A doctor finding that does not fail doctor or block a benchmark.

    Errors are what breaks serving, safety or a required capability now: the
    display carve-out and the kept text console (modeset, fbdev, DRM master,
    console mode), fan control, RoCE GIDs, the GPU clock and any difference from
    the promoted configuration. Warnings are latent (effective at the next
    boot), cost a little, or could not be read.
    """


def fan_control_enabled(cluster: dict) -> bool:
    """Whether the site runs the fan-floor control (host.fan_control, default true).

    The driver binds NVIDIA's P4242 board only. Sites on other GB10 boards set
    host.fan_control false: doctor then records the absence instead of erroring,
    and benchmarks skip the curve instead of refusing to measure.
    """
    return cluster.get("host", {}).get("fan_control", True) is not False


def local_doctor(cluster: dict, nodes: dict, upstreams: dict) -> list[str]:
    problems: list[str] = topology.problems(cluster, nodes)
    problems.extend(kernel_backend.problems(cluster))
    capture_errors, capture_warnings = cluster_recipe.capture_problems(cluster)
    problems.extend(capture_errors)
    problems.extend(Warn(warning) for warning in capture_warnings)
    problems.extend(benchmark_reference_problems(cluster, nodes))
    if cluster.get("promoted_baseline") != upstreams.get("generated_from_baseline"):
        problems.append("cluster and upstream locks name different baselines")
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
        if source["kind"] == "git":
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
    problems.extend(kernel_backend.source_problems(cluster, upstreams, source_manifest))
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
            if label != source_manifest["sparknet"]["expected_tree"]:
                problems.append("sparknet image tree does not match its source manifest")
    for name in build_projects(upstreams["sources"]):
        source = upstreams["sources"][name]
        if missing_patches(source):
            continue  # reported above
        record = source_manifest.get(name)
        if record is None:
            problems.append(f"{name}: source manifest has no record")
            continue
        if record.get("patchset_sha256") != patchset_sha256(source):
            problems.append(f"{name}: patch-set fingerprint differs from source manifest")
        if source.get("submodules") and not isinstance(record.get("submodules"), dict):
            problems.append(f"{name}: source manifest does not record its submodule commits")
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
"""


def host_facts(output: str) -> dict[str, str]:
    facts = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            facts[key.strip()] = value.strip()
    return facts


def modeset_problems(node_name: str, facts: dict[str, str]) -> list[str]:
    """Problems when nvidia-drm does not drive displays (kernel mode setting, required).

    Without it only the firmware framebuffer shows a console, and only for a
    display attached at boot: a monitor or KVM connected later stays dark.
    nvidia-drm also offers DRM dumb buffers only with mode setting, so the
    display carve-out that holds the embedding and output head (vLLM patch
    0021) cannot be allocated and the worker refuses to start.
    """
    value = facts.get("modeset", "")
    if value == "Y":
        return []
    return [
        f"{node_name}: nvidia-drm modeset is {value or 'unreadable'}, expected Y "
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
    return [
        f"{node_name}: nvidia-drm fbdev is {value or 'unreadable'}, expected Y "
        "(an attached display gets no text console); remove the nvidia-drm fbdev=0 "
        "option (grep -rn fbdev /etc/modprobe.d), run sudo update-initramfs -u, and "
        "reboot with the service stopped"
    ]


def drm_card_problems(node_name: str, facts: dict[str, str]) -> list[str]:
    """Problems when no DRM card belongs to the nvidia driver.

    The display carve-out (vLLM patch 0021) allocates through that card, which
    the container bind-mounts by its PCI path (Docker device flags split on its colons)
    (/dev/dri/by-path/pci-000f:01:00.0-card): a node that boots with a display
    attached registers simpledrm first, so the GPU can be card1 there.
    """
    if facts.get("drm_card"):
        return []
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
    """Problems when the active virtual console is in graphics mode.

    The boot splash can leave it there after Plymouth quits; the kernel then
    draws no text on an attached display although fbcon is available.
    """
    console, _, mode = facts.get("console", "").partition(":")
    if mode == "0":
        return []
    if mode == "1":
        return [
            f"{node_name}: the active console ({console}) is in graphics mode, so an "
            "attached display shows no text; run scripts/host-recovery apply on the node "
            "(it switches the console to text mode now and removes splash from later boots)"
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
            f"{node_name}: DKMS dgx-spark-fan-control is not installed for {kernel}; run "
            f"sudo dkms autoinstall -k {facts.get('kernel') or '$(uname -r)'}"
        ]
    installed = facts["fan_dkms"].split(",", 1)[0].rpartition("/")[2]
    if installed != FAN_CONTROL_VERSION:
        return [
            f"{node_name}: DKMS dgx-spark-fan-control is {installed} for {kernel}, and this recipe "
            f"uses {FAN_CONTROL_VERSION}; upgrade from a checkout at v{FAN_CONTROL_VERSION} "
            f"({FAN_CONTROL_UPGRADE})"
        ]
    if facts.get("fan_module") != "1":
        return [
            f"{node_name}: dgx_ec_fan_control is not loaded; run sudo modprobe "
            "dgx_ec_fan_control (if it is refused, check that the fleet signing key is "
            "enrolled with mokutil --list-enrolled)"
        ]
    if facts.get("fan_cooling_device") != "1":
        return [
            f"{node_name}: dgx_ec_fan_control is loaded but its dgx_ec_fan_floor cooling device "
            "is missing (the driver refused it at probe; see dmesg)"
        ]
    if facts.get("fan_service") == "absent":
        return [
            f"{node_name}: dgx-fan-control.service is not installed, and bench runs its "
            "performance curve while measuring; from a dgx-spark-fan-control checkout run "
            "sudo install -D -m 0644 systemd/dgx-fan-control.service "
            "/etc/systemd/system/dgx-fan-control.service && sudo systemctl daemon-reload "
            "(enabling it for serving is optional)"
        ]
    return []


def roce_gid_setting(cluster: dict, node: dict) -> tuple[int, list[str]]:
    """The GID index sparknet and NCCL are configured to use, and the node's HCAs."""
    env = expected_environment(cluster, node)
    key = "SPARKNET_ROCE_GID_INDEX" if topology.is_sparknet(topology.transport(cluster)) else "NCCL_IB_GID_INDEX"
    index = int(env.get(key) or env.get("NCCL_IB_GID_INDEX") or 3)
    return index, sorted(topology.hcas(cluster, node))


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


def roce_gid_problems(node_name: str, index: int, hcas: list[str], output: str) -> list[str]:
    """Problems when an HCA's configured GID slot is not its IPv4 RoCE v2 address.

    sparknet and NCCL both use one fixed GID index. The kernel files a re-added
    address in the lowest free slot, so a link flap while RDMA resources still
    hold the old entry moves the IPv4 RoCE v2 GID (dgx3 moved from 3 to 4 when
    its peers rebooted under a starting service), and NCCL then fails with an
    unhandled system error.
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
        problems.append(
            f"{node_name}: {hca} GID index {index} is "
            f"{kind + ' ' + gid if kind else 'empty'}; {where}"
        )
    return problems


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


def live_doctor(cluster: dict, nodes: dict) -> list[str]:
    """Findings about the live cluster: errors, and Warn for the minor ones."""
    problems: list[str] = []
    trial_path = kernel_policy_path(cluster)
    trial = json.loads(trial_path.read_text()) if trial_path.exists() else None
    kernel_inventories = {}
    corrections = []
    name = cluster["container"]["name"]
    for node in nodes["nodes"]:
        if trial:
            inventory = node_kernel_inventory(nodes, node, trial)
            kernel_inventories[node["name"]] = inventory
            warnings, fixes = kernel_readiness.findings(node["name"], inventory, trial)
            problems.extend(Warn(warning) for warning in warnings)
            corrections.extend(fixes)
            problems.extend(kernel_readiness.profile_problems(
                node["name"], inventory, cluster["host"].get("kernel"), trial))
        desktop = remote_shell(nodes, node, DESKTOP_QUERY)
        problems.extend(desktop_problems(node["name"], desktop.stdout))
        boot = run_ssh(nodes, node, "efibootmgr")
        problems.extend(boot_order_problems(node["name"], boot.stdout))
        facts = host_facts(remote_shell(nodes, node, HOST_FACTS_QUERY).stdout)
        problems.extend(modeset_problems(node["name"], facts))
        problems.extend(fbdev_problems(node["name"], facts))
        problems.extend(drm_card_problems(node["name"], facts))
        problems.extend(drm_master_problems(node["name"], facts))
        problems.extend(console_mode_problems(node["name"], facts))
        problems.extend(splash_problems(node["name"], facts))
        problems.extend(idle_service_warnings(node["name"], facts))
        problems.extend(nvme_coalescing_warnings(node["name"], facts))
        problems.extend(nvme_apst_warnings(node["name"], facts))
        problems.extend(fan_control_problems(
            node["name"], facts, fan_control=fan_control_enabled(cluster)))
        gid_index, hcas = roce_gid_setting(cluster, node)
        gids = remote_shell(nodes, node, roce_gid_query(hcas))
        problems.extend(roce_gid_problems(node["name"], gid_index, hcas, gids.stdout))
        problems.extend(topology.gid_subnet_problems(cluster, node, gid_index, gids.stdout))
        problems.extend(node_glp_problems(cluster, nodes, node))
        result = run_ssh(nodes, node, "docker", "inspect", name)
        if result.returncode:
            problems.append(f"{node['name']}: cannot inspect {name}: {result.stderr.strip()}")
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
        # The container's own image, not the configured tag: its Config.Env is
        # what the container inherited.
        image_result = run_ssh(nodes, node, "docker", "image", "inspect", inspection["Image"])
        if image_result.returncode:
            problems.append(f"{node['name']}: cannot inspect the container's image environment")
            continue
        problems.extend(
            environment_problems(
                node["name"],
                expected_environment(cluster, node),
                inspection["Config"]["Env"],
                json.loads(image_result.stdout)[0]["Config"].get("Env"),
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
        label_result = run_ssh(
            nodes,
            node,
            "docker",
            "image",
            "inspect",
            cluster["container"]["image"],
        )
        if label_result.returncode:
            problems.append(f"{node['name']}: cannot inspect image labels")
            continue
        labels = json.loads(label_result.stdout)[0]["Config"].get("Labels") or {}
        for key, wanted in cluster["container"]["expected_labels"].items():
            if labels.get(key) != wanted:
                problems.append(
                    f"{node['name']}: image label {key}={labels.get(key)!r}, expected {wanted!r}"
                )
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
    cluster, nodes, upstreams = configuration(args)
    if getattr(args, "fragmentation_commands", False):
        for finding in weight_fragmentation_findings(cluster, nodes, commands=True):
            print(finding)
        return 0
    findings = local_doctor(cluster, nodes, upstreams)
    if args.live and not split_findings(findings)[0]:
        findings.extend(live_doctor(cluster, nodes))
        findings.extend(weight_fragmentation_findings(cluster, nodes))
    problems, warnings = split_findings(findings)
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
    nodes = json.loads(nodes_path.read_text())
    try:
        candidate = topology.candidate(cluster, nodes, nodes_path.relative_to(ROOT.resolve()).as_posix())
    except (ValueError, StopIteration) as error:
        raise SystemExit(f"invalid topology: {error}") from error
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
