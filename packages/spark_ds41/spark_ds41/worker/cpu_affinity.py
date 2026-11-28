# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic CPU core affinity partitioning for heterogeneous ARMv9 / GB10 nodes.

On NVIDIA GB10 (20 ARMv9 cores across two L3 clusters: 10x Cortex-A725 efficiency
cores cpu0..4, cpu10..14 at 2.808 GHz / 512 KiB L2 and 10x Cortex-X925 performance
cores cpu5..9, cpu15..19 at 3.900 GHz / 2048 KiB L2), running the Rank-0 API server,
EngineCore, GPU worker main thread, 25+ worker background threads (gloo_tcp_loop,
pt_nccl_watchdg, engram-reader, OpenMP), and the sparknet RoCE proxy thread in a
shared unpartitioned pool causes frequent context-switch preemption of both
sparknet-proxy and VLLM::Worker_TP0 across the 116 per-step one-shot RoCE
collectives.

This module discovers fast performance cores and efficiency cores dynamically from
sysfs (`cpufreq/cpuinfo_max_freq`, `cpu_capacity`, `cache/index2/size`) intersected
with the container's allowed cpuset (`sched_getaffinity`), and partitions them into
mutually disjoint core sets for:
- `proxy_cpus`: dedicated fast core (CPU 19 on GB10) for `sparknet-proxy`
- `worker_main_cpus`: dedicated fast core in the same L3 cluster (CPU 18 on GB10)
  for the latency-critical `VLLM::Worker_TP*` main thread
- `worker_aux_cpus`: disjoint fast cores (CPUs 5..9 on GB10) for worker background
  threads (`engram-reader`, OpenMP, `gloo_tcp_loop`, `pt_nccl_watchdg`)
- `engine_core_cpus`: disjoint core (CPU 17 on GB10) for `VLLM::EngineCore`
- `api_server_cpus`: disjoint cores (CPUs 15..16 plus any allowed efficiency cores
  0..4, 10..14 on GB10) for the Rank-0 API server, ZMQ frontend, and tokenizers.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterable, Set
from dataclasses import dataclass
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_ROCE_CPU_AFFINITY = "VLLM_ROCE_CPU_AFFINITY"
ENV_CONTAINER_CPUSET = "VLLM_CONTAINER_CPUSET"
ENV_SPARKNET_PROXY_CPU = "SPARKNET_ROCE_PROXY_CPU"
ENV_ENGRAM_CPU_AFFINITY = "SPARK_DS41F_ENGRAM_CPU_AFFINITY"
_SYMBOLIC_PROXY_VALUES = {"", "big", "fast", "auto", "dedicated"}


@dataclass(frozen=True)
class CpuAffinityPlan:
    allowed_cpus: tuple[int, ...]
    fast_cpus: tuple[int, ...]
    efficiency_cpus: tuple[int, ...]
    proxy_cpus: tuple[int, ...]
    worker_main_cpus: tuple[int, ...]
    worker_aux_cpus: tuple[int, ...]
    engine_core_cpus: tuple[int, ...]
    api_server_cpus: tuple[int, ...]


def parse_cpu_list(spec: str) -> list[int]:
    """Parse a Linux CPU list string such as `'5-9,15-19'` into sorted unique ints."""
    cpus: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, stop_s = part.split("-", 1)
            start, stop = int(start_s), int(stop_s)
            if start < 0 or stop < start:
                raise ValueError(f"invalid CPU range {part!r} in {spec!r}")
            cpus.update(range(start, stop + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"invalid CPU number {part!r} in {spec!r}")
            cpus.add(cpu)
    return sorted(cpus)


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Format a set/sequence of CPU IDs into compact range notation (`'5-9,15-19'`)."""
    ordered = sorted(set(int(c) for c in cpus))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(ranges)


def _parse_cache_kib(raw: str) -> int:
    text = raw.strip().upper()
    if not text:
        return 0
    if text.endswith("K"):
        return int(text[:-1])
    if text.endswith("M"):
        return int(text[:-1]) * 1024
    if text.endswith("G"):
        return int(text[:-1]) * 1024 * 1024
    return max(1, int(text) // 1024)


def _read_int(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return -1


def _cpu_metric_map(root: Path) -> dict[int, int]:
    """Return a host-wide performance metric per CPU from sysfs.

    Prefers `cpufreq/cpuinfo_max_freq` (kHz), then `cpu_capacity`, then
    `cache/index2/size` (KiB).
    """
    if not root.is_dir():
        return {}
    cpus: list[int] = []
    for entry in root.iterdir():
        name = entry.name
        if name.startswith("cpu") and name[3:].isdigit():
            cpus.append(int(name[3:]))
    cpus.sort()
    if not cpus:
        return {}

    freqs = {
        cpu: _read_int(root / f"cpu{cpu}" / "cpufreq" / "cpuinfo_max_freq")
        for cpu in cpus
    }
    if any(v > 0 for v in freqs.values()):
        return {cpu: val for cpu, val in freqs.items() if val > 0}

    caps = {
        cpu: _read_int(root / f"cpu{cpu}" / "cpu_capacity")
        for cpu in cpus
    }
    if any(v > 0 for v in caps.values()):
        return {cpu: val for cpu, val in caps.items() if val > 0}

    l2s: dict[int, int] = {}
    for cpu in cpus:
        size_path = root / f"cpu{cpu}" / "cache" / "index2" / "size"
        try:
            kib = _parse_cache_kib(size_path.read_text(encoding="utf-8"))
        except OSError:
            continue
        if kib > 0:
            l2s[cpu] = kib
    return l2s


def get_container_cpuset(*, record_env: bool = True) -> set[int]:
    """Return the container/process initial allowed CPU set before role narrowing."""
    env_spec = os.environ.get(ENV_CONTAINER_CPUSET, "").strip()
    if env_spec:
        with contextlib.suppress(ValueError):
            parsed = parse_cpu_list(env_spec)
            if parsed:
                return set(parsed)
    allowed: set[int] = set()
    if hasattr(os, "sched_getaffinity"):
        with contextlib.suppress(OSError):
            allowed = set(os.sched_getaffinity(0))
    if not allowed:
        count = os.cpu_count() or 1
        allowed = set(range(count))
    if record_env and ENV_CONTAINER_CPUSET not in os.environ:
        os.environ[ENV_CONTAINER_CPUSET] = format_cpu_list(allowed)
    return allowed


def _cpu_l3_topology(root: Path) -> tuple[dict[int, int], dict[int, frozenset[int]]]:
    """Return `(l3_size_kib_by_cpu, l3_shared_siblings_by_cpu)` from sysfs `cache/index3`."""
    if not root.is_dir():
        return {}, {}
    l3_kib: dict[int, int] = {}
    l3_group: dict[int, frozenset[int]] = {}
    for entry in root.iterdir():
        name = entry.name
        if not (name.startswith("cpu") and name[3:].isdigit()):
            continue
        cpu = int(name[3:])
        idx3 = root / name / "cache" / "index3"
        with contextlib.suppress(OSError, ValueError):
            kib = _parse_cache_kib((idx3 / "size").read_text(encoding="utf-8"))
            if kib > 0:
                l3_kib[cpu] = kib
        with contextlib.suppress(OSError, ValueError):
            grp = frozenset(parse_cpu_list((idx3 / "shared_cpu_list").read_text(encoding="utf-8")))
            if grp:
                l3_group[cpu] = grp
    return l3_kib, l3_group


def discover_cpu_topology(
    sysfs_root: str | os.PathLike[str] = "/sys/devices/system/cpu",
    allowed: Set[int] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Classify allowed CPUs into `(fast_cpus, efficiency_cpus)`.

    The threshold is computed across all online CPUs in `sysfs_root` first so that
    containers started with `--cpuset-cpus=5-9,15-19` on GB10 still recognize all
    10 Cortex-X925 cores as fast performance cores.
    """
    root = Path(sysfs_root)
    allowed_set = (
        set(int(c) for c in allowed)
        if allowed is not None
        else get_container_cpuset()
    )
    metrics = _cpu_metric_map(root)
    if not metrics:
        ordered = tuple(sorted(allowed_set))
        return ordered, ()

    best = max(metrics.values())
    worst = min(metrics.values())
    threshold = (best + worst) // 2

    candidates = sorted(allowed_set) if allowed_set else sorted(metrics)
    fast: list[int] = []
    efficiency: list[int] = []
    for cpu in candidates:
        score = metrics.get(cpu)
        if score is None:
            continue
        if best == worst or score > threshold:
            fast.append(cpu)
        else:
            efficiency.append(cpu)

    if not fast and efficiency:
        # Allowed cpuset contains only efficiency cores; treat them as the usable pool.
        return tuple(efficiency), ()
    if not fast and not efficiency:
        ordered = tuple(sorted(allowed_set))
        return ordered, ()
    l3_kib, _ = _cpu_l3_topology(root)
    if l3_kib:
        # Order fast cores ascending by (L3 cache size, perf metric, CPU ID) so the
        # highest-ranked cores (selected for proxy, worker_main, engine_core) always
        # reside in the largest L3 cache cluster (16 MiB Cluster 1 on GB10).
        fast.sort(key=lambda c: (l3_kib.get(c, 0), metrics.get(c, 0), c))
    return tuple(fast), tuple(efficiency)


def partition_cpu_affinity(
    allowed: Set[int] | None = None,
    sysfs_root: str | os.PathLike[str] = "/sys/devices/system/cpu",
    proxy_cpu_spec: str | None = None,
) -> CpuAffinityPlan:
    """Compute a deterministic, non-overlapping CPU core partition across roles."""
    allowed_set = (
        set(int(c) for c in allowed)
        if allowed is not None
        else get_container_cpuset()
    )
    fast_cpus, efficiency_cpus = discover_cpu_topology(
        sysfs_root=sysfs_root, allowed=allowed_set
    )
    ordered_allowed = tuple(sorted(allowed_set)) if allowed_set else fast_cpus
    if not fast_cpus:
        fast_cpus = ordered_allowed

    if proxy_cpu_spec is None:
        proxy_cpu_spec = os.environ.get(ENV_SPARKNET_PROXY_CPU, "").strip()
    else:
        proxy_cpu_spec = proxy_cpu_spec.strip()

    explicit_proxy: list[int] = []
    if proxy_cpu_spec and proxy_cpu_spec not in _SYMBOLIC_PROXY_VALUES and proxy_cpu_spec != "none":
        with contextlib.suppress(ValueError):
            parsed = [c for c in parse_cpu_list(proxy_cpu_spec) if c in fast_cpus or c in ordered_allowed]
            if parsed:
                explicit_proxy = [parsed[-1]]

    if explicit_proxy:
        proxy_cpus = (explicit_proxy[0],)
        remaining_fast = tuple(c for c in fast_cpus if c != proxy_cpus[0]) or fast_cpus
    else:
        proxy_cpus = (fast_cpus[-1],)
        remaining_fast = fast_cpus[:-1] if len(fast_cpus) > 1 else fast_cpus

    _, l3_groups = _cpu_l3_topology(Path(sysfs_root))
    hot_l3_siblings: set[int] = set()
    for c in (*proxy_cpus, *(remaining_fast[-2:] if len(remaining_fast) >= 2 else remaining_fast)):
        hot_l3_siblings.update(l3_groups.get(c, ()))
    isolated_eff = tuple(c for c in efficiency_cpus if c not in hot_l3_siblings)
    bg_eff_cpus = isolated_eff if isolated_eff else efficiency_cpus

    n_rem = len(remaining_fast)
    if n_rem >= 5:
        # GB10 (10 fast cores -> 9 remaining after proxy=19):
        # worker_main = (18,) (same L3 cluster 15..19 as proxy=19)
        # engine_core = (17,)
        # api_server_fast = (15, 16) (when n_rem >= 6) or (16,)
        # worker_aux = (5, 6, 7, 8, 9) (Cluster 0 fast cores)
        worker_main_cpus = (remaining_fast[-1],)
        engine_core_cpus = (remaining_fast[-2],)
        api_fast_count = 2 if n_rem >= 6 else 1
        api_fast = remaining_fast[-(2 + api_fast_count):-2]
        worker_aux_cpus = remaining_fast[:-(2 + api_fast_count)]
        api_server_cpus = tuple(sorted(set(bg_eff_cpus) | set(api_fast)))
    elif n_rem == 4:
        worker_main_cpus = (remaining_fast[-1],)
        engine_core_cpus = (remaining_fast[-2],)
        api_server_cpus = tuple(sorted(set(bg_eff_cpus) | {remaining_fast[-3]}))
        worker_aux_cpus = (remaining_fast[0],)
    elif n_rem == 3:
        worker_main_cpus = (remaining_fast[-1],)
        engine_core_cpus = (remaining_fast[-2],)
        worker_aux_cpus = (remaining_fast[0],)
        api_server_cpus = (
            tuple(bg_eff_cpus)
            if bg_eff_cpus
            else (remaining_fast[-2],)
        )
    elif n_rem == 2:
        worker_main_cpus = (remaining_fast[-1],)
        worker_aux_cpus = (remaining_fast[0],)
        engine_core_cpus = (
            tuple(bg_eff_cpus) if bg_eff_cpus else (remaining_fast[0],)
        )
        api_server_cpus = (
            tuple(bg_eff_cpus) if bg_eff_cpus else (remaining_fast[0],)
        )
    else:
        worker_main_cpus = remaining_fast
        worker_aux_cpus = remaining_fast
        engine_core_cpus = tuple(bg_eff_cpus) if bg_eff_cpus else remaining_fast
        api_server_cpus = tuple(bg_eff_cpus) if bg_eff_cpus else remaining_fast

    return CpuAffinityPlan(
        allowed_cpus=ordered_allowed,
        fast_cpus=fast_cpus,
        efficiency_cpus=efficiency_cpus,
        proxy_cpus=proxy_cpus,
        worker_main_cpus=worker_main_cpus,
        worker_aux_cpus=worker_aux_cpus,
        engine_core_cpus=engine_core_cpus,
        api_server_cpus=api_server_cpus,
    )


def is_affinity_enabled() -> bool:
    return os.environ.get(ENV_ROCE_CPU_AFFINITY, "1").strip() not in ("0", "false", "none")


def configure_worker_cpu_affinity(
    local_rank: int = 0,
    sysfs_root: str | os.PathLike[str] = "/sys/devices/system/cpu",
) -> CpuAffinityPlan | None:
    """Prepare worker CPU environment variables and default thread affinity before
    distributed/communicator initialization."""
    del local_rank
    if not is_affinity_enabled() or not hasattr(os, "sched_setaffinity"):
        return None
    plan = partition_cpu_affinity(sysfs_root=sysfs_root)
    if not plan.fast_cpus:
        return None

    raw_proxy = os.environ.get(ENV_SPARKNET_PROXY_CPU, "").strip()
    if raw_proxy in _SYMBOLIC_PROXY_VALUES and plan.proxy_cpus:
        os.environ[ENV_SPARKNET_PROXY_CPU] = format_cpu_list(plan.proxy_cpus)

    if ENV_ENGRAM_CPU_AFFINITY not in os.environ and plan.worker_aux_cpus:
        os.environ[ENV_ENGRAM_CPU_AFFINITY] = format_cpu_list(plan.worker_aux_cpus)

    # Set the worker process mask to worker_aux_cpus while distributed groups and
    # helper threads (gloo_tcp_loop, pt_nccl_watchdg, pt_tcpstore_uv) spawn so
    # they inherit worker_aux_cpus instead of the main worker core or proxy core.
    if plan.worker_aux_cpus:
        with contextlib.suppress(OSError):
            os.sched_setaffinity(0, set(plan.worker_aux_cpus))
    return plan


def apply_worker_thread_affinity(
    plan: CpuAffinityPlan | None = None,
    *,
    sysfs_root: str | os.PathLike[str] = "/sys/devices/system/cpu",
    task_root: str | os.PathLike[str] = "/proc/self/task",
    main_tid: int | None = None,
) -> dict[str, tuple[int, ...]]:
    """Pin the worker main thread, sparknet-proxy, and auxiliary worker threads to
    their disjoint core partitions."""
    if not is_affinity_enabled() or not hasattr(os, "sched_setaffinity"):
        return {}
    if plan is None:
        plan = partition_cpu_affinity(sysfs_root=sysfs_root)
    if not plan.fast_cpus:
        return {}

    if main_tid is None:
        main_tid = threading.main_thread().native_id or os.getpid()

    applied: dict[str, tuple[int, ...]] = {}
    task_dir = Path(task_root)
    if task_dir.is_dir():
        for entry in sorted(task_dir.iterdir(), key=lambda p: p.name):
            if not entry.name.isdigit():
                continue
            tid = int(entry.name)
            comm = ""
            with contextlib.suppress(OSError):
                comm = (entry / "comm").read_text(encoding="utf-8").strip()
            if comm == "sparknet-proxy":
                target = plan.proxy_cpus
                key = f"sparknet-proxy:{tid}"
            elif tid == main_tid:
                target = plan.worker_main_cpus
                key = f"worker-main:{tid}"
            else:
                target = plan.worker_aux_cpus
                key = f"{comm or 'worker-aux'}:{tid}"
            if target:
                with contextlib.suppress(OSError):
                    os.sched_setaffinity(tid, set(target))
                    applied[key] = target
    else:
        with contextlib.suppress(OSError):
            os.sched_setaffinity(0, set(plan.worker_main_cpus))
            applied[f"worker-main:{main_tid}"] = plan.worker_main_cpus
    return applied


def apply_role_cpu_affinity(
    role: str,
    *,
    sysfs_root: str | os.PathLike[str] = "/sys/devices/system/cpu",
    task_root: str | os.PathLike[str] = "/proc/self/task",
) -> tuple[int, ...]:
    """Pin an `engine_core` or `api_server` process (and its threads) to its
    disjoint CPU core set."""
    if not is_affinity_enabled() or not hasattr(os, "sched_setaffinity"):
        return ()
    plan = partition_cpu_affinity(sysfs_root=sysfs_root)
    if role == "engine_core":
        target = plan.engine_core_cpus
    elif role == "api_server":
        target = plan.api_server_cpus
    else:
        raise ValueError(f"unknown CPU affinity role: {role!r}")
    if not target:
        return ()

    target_set = set(target)
    task_dir = Path(task_root)
    if task_dir.is_dir():
        for entry in task_dir.iterdir():
            if entry.name.isdigit():
                with contextlib.suppress(OSError):
                    os.sched_setaffinity(int(entry.name), target_set)
    with contextlib.suppress(OSError):
        os.sched_setaffinity(0, target_set)
    logger.debug(
        "Pinned %s (pid=%d) to CPUs %s (proxy=%s, worker_main=%s, worker_aux=%s)",
        role,
        os.getpid(),
        format_cpu_list(target),
        format_cpu_list(plan.proxy_cpus),
        format_cpu_list(plan.worker_main_cpus),
        format_cpu_list(plan.worker_aux_cpus),
    )
    return target


__all__ = [
    "CpuAffinityPlan",
    "ENV_CONTAINER_CPUSET",
    "ENV_ENGRAM_CPU_AFFINITY",
    "ENV_ROCE_CPU_AFFINITY",
    "ENV_SPARKNET_PROXY_CPU",
    "apply_role_cpu_affinity",
    "apply_worker_thread_affinity",
    "configure_worker_cpu_affinity",
    "discover_cpu_topology",
    "format_cpu_list",
    "get_container_cpuset",
    "is_affinity_enabled",
    "parse_cpu_list",
    "partition_cpu_affinity",
]
