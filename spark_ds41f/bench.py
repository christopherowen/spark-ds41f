"""Benchmark conditions, identity and reference rules for spark-ds41f; Sonar measures.

Every measurement runs through Sonar (spark_ds41f/sonar.py, the commit upstreams.lock.json
pins): ``bench`` drives Sonar's measurement matrix in-process, ``bench --suite`` runs a Sonar
suite (config/suites/ or a Sonar built-in) as a child process, and ``workload`` is Sonar's
counters-only watch. What stays here is the deployment's: the nodes cooled and the fan
performance curve running before measuring, memory and temperatures watched while it runs
(a breach stops it), the deployment's identity recorded in the run, and which recorded run a
profile compares with.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTER_CONFIG = "config/cluster.json"
sys.path.insert(0, str(ROOT / "scripts"))
import source_labels
import kernel_backend
import kernel_readiness
import topology
import recipe as cluster_recipe

from .config import *  # noqa: F401,F403
from .doctor import *  # noqa: F401,F403
from .cluster import *  # noqa: F401,F403
from .build import *  # noqa: F401,F403
from . import sonar

BENCH_SCHEMA = 1


class BenchAbort(RuntimeError):
    """Stop issuing requests; the partial report is still written."""




THERMAL_FIELDS = (
    "temperature.gpu",
    "temperature.gpu.tlimit",
    "clocks.sm",
    "power.draw",
    "clocks_event_reasons_counters.sw_thermal_slowdown",
    "clocks_event_reasons_counters.hw_thermal_slowdown",
    "clocks_event_reasons_counters.sw_power_cap",
    "clocks_event_reasons_counters.hw_power_brake_slowdown",
)
# Stop measuring when a GPU comes within this many degrees of its slowdown limit.
THERMAL_HEADROOM_FLOOR_C = 5.0
# DGX Spark board zones trip a critical shutdown at 104 C; stop well before it.
BOARD_CEILING_C = 98.0


def parse_thermal(text: str) -> dict:
    """One node sample: nvidia-smi fields, then the hottest thermal zone (m°C)."""

    def number(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    values = [number(value) for value in text.strip().split(",")]
    values += [None] * (len(THERMAL_FIELDS) + 1 - len(values))
    gpu, headroom, clock, power, sw_thermal, hw_thermal, sw_power, hw_brake, zone = values[: len(THERMAL_FIELDS) + 1]

    def total(*parts: float | None) -> float | None:
        known = [part for part in parts if part is not None]
        return sum(known) if known else None

    return {
        "complete": all(value is not None and math.isfinite(value) for value in values[:len(THERMAL_FIELDS) + 1]),
        "gpu_c": gpu,
        "headroom_c": headroom,
        "sm_mhz": clock,
        "power_w": power,
        "thermal_slowdown_us": total(sw_thermal, hw_thermal),
        "power_cap_us": total(sw_power, hw_brake),
        "system_c": None if zone is None else zone / 1000,
    }


class MemoryWatch:
    """Per-node memory each second and GPU/board temperatures every fifteen.

    Temperatures change slowly; polling nvidia-smi less often keeps its GPU
    queries out of more of the timed steps.
    """

    SCRIPT = (
        "i=0; while :; do "
        "awk '/^(MemAvailable|SwapTotal|SwapFree):/{v[$1]=$2} "
        'END{print "M", v["MemAvailable:"], v["SwapTotal:"]-v["SwapFree:"]}\' /proc/meminfo; '
        'if [ $((i % 15)) -eq 0 ]; then echo "T $(nvidia-smi --query-gpu='
        + ",".join(THERMAL_FIELDS)
        + " --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '),"
        + '$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)"; fi; '
        "i=$((i + 1)); sleep 1; done"
    )

    def __init__(self, nodes: dict, floor_gib: float, halt) -> None:
        self.floor_kib = floor_gib * 1048576
        self.floor_gib = floor_gib
        self.halt = halt
        self.lock = threading.Lock()
        self.state: dict[str, dict] = {}
        self.thermal: dict[str, dict] = {}
        self.strict_thermal = os.environ.get("SPARK_DS41F_STRICT_TELEMETRY") == "1"
        self.thermal_last: dict[str, float | None] = {}
        self.thermal_errors: dict[str, str] = {}
        self.processes: list[subprocess.Popen] = []
        for node in nodes["nodes"]:
            process = subprocess.Popen(
                [
                    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    "-o", "ServerAliveInterval=5", ssh_target(nodes, node),
                    shlex.join(["bash", "-c", self.SCRIPT]),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self.processes.append(process)
            self.state[node["name"]] = {"last": None}
            self.thermal[node["name"]] = {}
            self.thermal_last[node["name"]] = None
            threading.Thread(
                target=self._read, args=(node["name"], process), daemon=True
            ).start()

    def _read(self, name: str, process: subprocess.Popen) -> None:
        for line in process.stdout:
            kind, _, rest = line.strip().partition(" ")
            if kind == "M":
                self._memory(name, rest)
            elif kind == "T":
                self._temperature(name, parse_thermal(rest))

    def _memory(self, name: str, text: str) -> None:
        try:
            available, swap = (int(float(value)) for value in text.split())
        except ValueError:
            return
        with self.lock:
            state = self.state[name]
            state["last"] = time.monotonic()
            state["min_available_kib"] = min(state.get("min_available_kib", available), available)
            state.setdefault("swap_start_kib", swap)
            state["swap_max_kib"] = max(state.get("swap_max_kib", swap), swap)
        if available < self.floor_kib:
            self.halt(
                f"{name}: MemAvailable {available / 1048576:.2f} GiB fell below "
                f"the {self.floor_gib:g} GiB bench floor"
            )

    def _temperature(self, name: str, sample: dict) -> None:
        if self.strict_thermal:
            required = ("gpu_c", "headroom_c", "sm_mhz", "power_w", "system_c",
                        "thermal_slowdown_us", "power_cap_us")
            valid = sample.get("complete") is True and all(
                type(sample.get(key)) in (int, float) and math.isfinite(sample[key]) for key in required)
            with self.lock:
                if valid:
                    self.thermal_last[name] = time.monotonic()
                else:
                    self.thermal_errors.setdefault(name, f"{name}: thermal telemetry is missing, malformed or non-finite")
                problem = self.thermal_errors.get(name)
            if problem:
                self.halt(problem)
                return
        with self.lock:
            state = self.thermal[name]
            for key, better in (("gpu_c", max), ("system_c", max), ("power_w", max), ("headroom_c", min)):
                if sample[key] is not None:
                    state[key] = better(state.get(key, sample[key]), sample[key])
            for key in ("thermal_slowdown_us", "power_cap_us"):
                if sample[key] is not None:
                    state.setdefault(f"{key}_start", sample[key])
                    state[f"{key}_last"] = sample[key]
        board = sample["system_c"]
        if board is not None and board >= BOARD_CEILING_C:
            self.halt(
                f"{name}: board thermal zone at {board:g} C reached the {BOARD_CEILING_C:g} C "
                "bench ceiling (critical trip 104 C)"
            )
        headroom = sample["headroom_c"]
        if headroom is not None and headroom <= THERMAL_HEADROOM_FLOOR_C:
            self.halt(
                f"{name}: GPU at {sample['gpu_c']:g} C is within {headroom:g} C of its "
                "thermal slowdown limit"
            )

    def wait_ready(self, timeout: float = 30) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.thermal_errors:
                    raise BenchAbort(next(iter(self.thermal_errors.values())))
                if (all(state["last"] is not None for state in self.state.values())
                        and (not self.strict_thermal or all(value is not None for value in self.thermal_last.values()))):
                    return
            time.sleep(0.5)
        raise BenchAbort(("memory/thermal" if self.strict_thermal else "memory") + " telemetry did not start on every node")

    def check(self) -> None:
        now = time.monotonic()
        with self.lock:
            if self.thermal_errors:
                raise BenchAbort(next(iter(self.thermal_errors.values())))
            for name, state in self.state.items():
                if state["last"] is None or now - state["last"] > 15:
                    raise BenchAbort(f"{name}: memory telemetry stopped reporting")
                if self.strict_thermal and (self.thermal_last[name] is None or now - self.thermal_last[name] > 45):
                    reason = f"{name}: thermal telemetry stopped reporting"
                    self.thermal_errors.setdefault(name, reason)
                    raise BenchAbort(reason)

    def thermal_summary(self) -> dict:
        summary = {}
        with self.lock:
            for name, state in self.thermal.items():
                if not state:
                    continue
                entry = {
                    "max_gpu_c": state.get("gpu_c"),
                    "min_headroom_c": state.get("headroom_c"),
                    "max_system_c": state.get("system_c"),
                    "max_power_w": state.get("power_w"),
                }
                for key, label in (("thermal_slowdown_us", "thermal_slowdown_ms"), ("power_cap_us", "power_cap_ms")):
                    if f"{key}_last" in state:
                        entry[label] = round((state[f"{key}_last"] - state[f"{key}_start"]) / 1000, 1)
                summary[name] = entry
        return summary

    def close(self) -> dict:
        for process in self.processes:
            process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        with self.lock:
            return {
                name: {
                    "min_available_gib": round(state["min_available_kib"] / 1048576, 2),
                    "swap_growth_gib": round(
                        (state["swap_max_kib"] - state["swap_start_kib"]) / 1048576, 2
                    ),
                }
                for name, state in self.state.items()
                if state["last"] is not None
            }



# Every benchmark or kernel run starts with each node's hottest thermal zone
# below COOL_BELOW_C. If any node is above it, all of them run at the maximum
# fan floor of dgx-spark-fan-control until the last one is below it, then
# their fans return to their usual control (the dgx-fan-control service's
# curve, or firmware automatic) before measuring.
COOL_BELOW_C = 55.0
COOL_TIMEOUT_S = 600.0
COOL_POLL_S = 5.0
FAN_SERVICE = "dgx-fan-control.service"
HOTTEST_ZONE_QUERY = "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1"


def hottest_zone_c(nodes: dict, node: dict) -> float | None:
    result = run_ssh(nodes, node, "sh", "-c", HOTTEST_ZONE_QUERY)
    try:
        return int(result.stdout.strip()) / 1000
    except ValueError:
        return None


def take_fan_control(nodes: dict, node: dict) -> dict:
    """Pause the dgx-fan-control service and select the maximum fan floor."""
    service = run_ssh(nodes, node, "systemctl", "is-active", "--quiet", FAN_SERVICE).returncode == 0
    if service:
        stopped = run_ssh(nodes, node, "sudo", "-n", "systemctl", "stop", FAN_SERVICE)
        if stopped.returncode:
            return {"service": service, "floor": False}
    floor = run_ssh(nodes, node, "sudo", "-n", "dgx-fan-control", "set-state", "12").returncode == 0
    return {"service": service, "floor": floor}


def restore_fan_control(nodes: dict, node: dict, control: dict) -> str | None:
    """Return the node to the fan control it had before take_fan_control."""
    if control["service"]:
        # These intentional cooling cycles can exhaust systemd's start-rate
        # budget even though the daemon exited successfully each time.
        reset = run_ssh(nodes, node, "sudo", "-n", "systemctl", "reset-failed", FAN_SERVICE)
        if reset.returncode:
            return f"FAILED: reset start-rate limit for {FAN_SERVICE}"
        restored = run_ssh(nodes, node, "sudo", "-n", "systemctl", "start", FAN_SERVICE)
        if restored.returncode:
            return f"FAILED: {FAN_SERVICE}"
        active = run_ssh(nodes, node, "systemctl", "is-active", "--quiet", FAN_SERVICE)
        return FAN_SERVICE if active.returncode == 0 else f"FAILED: {FAN_SERVICE} is not active"
    if control["floor"]:
        restored = run_ssh(nodes, node, "sudo", "-n", "dgx-fan-control", "automatic")
        return "automatic" if restored.returncode == 0 else "FAILED: automatic"
    return None


def cool_nodes(
    nodes: dict,
    targets: list[dict],
    below_c: float,
    timeout_s: float,
    poll_s: float = COOL_POLL_S,
    fan_control: bool = True,
) -> list[dict]:
    """Bring every target node's hottest zone below below_c, cooling them together.

    When any node is at or above the threshold, every target gets the maximum
    fan floor until the last one is below it, so the wait cools them all. Each
    node's usual fan control then returns. Raises RuntimeError when a node
    cannot be read or does not cool within timeout_s; the fans are restored on
    every path.
    """
    if below_c <= 0 or not targets or not fan_control:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as pool:
        def readings() -> list[float | None]:
            return list(pool.map(lambda node: hottest_zone_c(nodes, node), targets))

        temperatures = readings()
        unreadable = [node["name"] for node, value in zip(targets, temperatures) if value is None]
        if unreadable:
            raise RuntimeError(f"cannot read the thermal zones of {', '.join(unreadable)}")
        records = [
            {"node": node["name"], "start_c": value, "cooled": False, "seconds": 0.0, "final_c": value}
            for node, value in zip(targets, temperatures)
        ]
        if all(value < below_c for value in temperatures):
            return records
        hot = ", ".join(f"{r['node']} {r['start_c']:.1f} C" for r in records if r["start_c"] >= below_c)
        print(f"cooling: {hot}; every node at fan floor 12 until all are below {below_c:g} C", flush=True)
        controls = list(pool.map(lambda node: take_fan_control(nodes, node), targets))
        began = time.monotonic()
        try:
            missing = [node["name"] for node, control in zip(targets, controls) if not control["floor"]]
            if missing and any(control["floor"] or control["service"] for control in controls):
                raise RuntimeError(f"cannot select maximum fan floor on {', '.join(missing)}")
            while not all(value is not None and value < below_c for value in temperatures):
                if time.monotonic() - began > timeout_s:
                    still = ", ".join(
                        f"{node['name']} {value} C"
                        for node, value in zip(targets, temperatures)
                        if value is None or value >= below_c
                    )
                    raise RuntimeError(f"still at or above {below_c:g} C after {timeout_s:g} s: {still}")
                time.sleep(poll_s)
                temperatures = readings()
        finally:
            restored = list(pool.map(
                lambda pair: restore_fan_control(nodes, pair[0], pair[1]), zip(targets, controls)
            ))
            seconds = round(time.monotonic() - began, 1)
            for record, value, control, result in zip(records, temperatures, controls, restored):
                record.update(cooled=True, seconds=seconds, final_c=value,
                              fan_floor=12 if control["floor"] else None, restored=result)
            failed = [f"{node['name']}: {result}" for node, result in zip(targets, restored)
                      if str(result).startswith("FAILED")]
            if failed:
                raise RuntimeError("cannot restore fan control: " + "; ".join(failed))
    print("cooling: " + ", ".join(f"{r['node']} {r['final_c']:.1f} C" for r in records)
          + f" after {seconds:g} s; fans restored", flush=True)
    return records


# A benchmark measures under one fan policy whatever a site serves with: the
# dgx-fan-control service's performance curve, started where it is installed but
# not running and stopped again afterwards. A node that cannot run it stops the
# benchmark, because its numbers would not be comparable.
def start_fan_curve(nodes: dict, targets: list[dict],
                    fan_control: bool = True) -> list[dict]:
    """Run the dgx-fan-control performance curve on every target for a benchmark."""
    if not fan_control:
        return []

    def start(node: dict) -> dict:
        record = {"node": node["name"], "service": "active", "started": False}
        if run_ssh(nodes, node, "systemctl", "is-active", "--quiet", FAN_SERVICE).returncode == 0:
            return record
        if run_ssh(nodes, node, "systemctl", "cat", FAN_SERVICE).returncode:
            record["service"] = "absent"
            return record
        record["service"] = "inactive"
        # Cooling may just have cycled the service; clear its start-rate limit.
        run_ssh(nodes, node, "sudo", "-n", "systemctl", "reset-failed", FAN_SERVICE)
        started = run_ssh(nodes, node, "sudo", "-n", "systemctl", "start", FAN_SERVICE)
        record["started"] = started.returncode == 0
        if started.returncode:
            record["error"] = (started.stderr or "").strip() or f"systemctl start exited {started.returncode}"
        return record

    if not targets:
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as pool:
        records = list(pool.map(start, targets))
    states = []
    for record in records:
        if record["started"]:
            states.append(f"{record['node']} started")
        elif record["service"] == "active":
            states.append(f"{record['node']} already running")
        elif record["service"] == "absent":
            states.append(f"{record['node']} not installed")
        else:
            states.append(f"{record['node']} failed to start: {record['error']}")
    print("fan curve: " + ", ".join(states), flush=True)
    return records


def fan_curve_problems(records: list[dict]) -> list[str]:
    """The nodes start_fan_curve could not put on the performance curve."""
    problems = []
    for record in records:
        if record["service"] == "absent":
            problems.append(f"{record['node']}: dgx-fan-control.service is not installed (see doctor --live)")
        elif record["service"] == "inactive" and not record["started"]:
            problems.append(f"{record['node']}: dgx-fan-control.service did not start: {record['error']}")
    return problems


def stop_fan_curve(nodes: dict, targets: list[dict], records: list[dict]) -> list[str]:
    """Stop the curve where start_fan_curve started it; return what could not be stopped.

    The service returns its node to firmware automatic control when it stops.
    """
    started = [(node, record) for node, record in zip(targets, records) if record.get("started")]
    if not started:
        return []

    def stop(pair: tuple[dict, dict]) -> str | None:
        node, record = pair
        stopped = run_ssh(nodes, node, "sudo", "-n", "systemctl", "stop", FAN_SERVICE)
        record["stopped"] = stopped.returncode == 0
        return None if stopped.returncode == 0 else f"{node['name']}: cannot stop {FAN_SERVICE}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(started)) as pool:
        failures = [failure for failure in pool.map(stop, started) if failure]
    print("fan curve: stopped on " + ", ".join(node["name"] for node, _ in started), flush=True)
    return failures



def serve_arg(cluster: dict, flag: str) -> str | None:
    arguments = cluster["serve_args"]
    return arguments[arguments.index(flag) + 1] if flag in arguments else None


def bench_identity(cluster: dict, nodes: dict, config_name: str, config_path: Path) -> dict:
    name = cluster["container"]["name"]
    images = {}
    inspection_problems = []
    identity = {
        "cluster_config": config_name,
        "cluster_config_sha256": sha256(config_path),
        "nodes_config": cluster.get("nodes_config", "config/nodes.json"),
        "topology": nodes,
        "transport": topology.transport(cluster),
        "kernel_backend": kernel_backend.backend(cluster),
        "promoted_baseline": cluster.get("promoted_baseline"),
        "image": cluster["container"]["image"],
        "client_commit": repository_revision(),
        "client_dirty": bool(repository_dirty()),
        "client_host": socket.gethostname(),
        "client_stream_policy": sonar.stream_policy(),
        "sonar": sonar.revision(),
        "nodes": {},
    }
    for node in nodes["nodes"]:
        container = run_ssh(nodes, node, "docker", "inspect", "--format", "{{.Image}} {{.State.StartedAt}}", name)
        checkout = run_ssh(nodes, node, "git", "-C", repository_path(cluster), "rev-parse", "HEAD")
        image_id, _, started_at = (container.stdout.strip() if container.returncode == 0 else "").partition(" ")
        identity["nodes"][node["name"]] = {
            "image_id": image_id or None,
            "container_started_at": started_at or None,
            "checkout": checkout.stdout.strip() or None,
        }
        try:
            image = read_running_image(nodes, node, image_id)
        except ValueError as error:
            inspection_problems.append(f"{node['name']}: {error}")
        else:
            images[node["name"]] = image
            identity["nodes"][node["name"]].update(image_content_identity(image))
            identity["nodes"][node["name"]]["image_labels"] = image["Config"].get("Labels") or {}
    inspection_problems.extend(running_image_problems(cluster, images))
    problems, warnings = split_findings(live_doctor(cluster, nodes))
    identity["live_problems"] = list(dict.fromkeys(str(problem) for problem in [*inspection_problems, *problems]))
    identity["live_warnings"] = [str(warning) for warning in warnings]
    return identity


# What a reference run must share with the run compared against it: across node
# counts, transports or kernel families the numbers describe different systems.
BENCH_SHAPE = ("nodes", "transport", "kernel_backend")


def bench_shape(identity: dict) -> dict:
    """The node count, transport and kernel family a report's identity records.
    None marks a field it does not record; reports from before r6 omit
    kernel_backend, which then means B12X."""
    nodes = (identity.get("topology") or {}).get("nodes")
    return {
        "nodes": len(nodes) if nodes else None,
        "transport": identity.get("transport"),
        "kernel_backend": identity.get("kernel_backend", kernel_backend.DEFAULT),
    }


def profile_shape(cluster: dict, nodes: dict) -> dict:
    return bench_shape(
        {"topology": nodes, "transport": topology.transport(cluster), "kernel_backend": kernel_backend.backend(cluster)}
    )


def shape_differences(run: dict, reference: dict) -> tuple[list[str], list[str]]:
    """The fields whose recorded values differ, and those either side does not record."""
    unrecorded = [key for key in BENCH_SHAPE if run[key] is None or reference[key] is None]
    return [key for key in BENCH_SHAPE if key not in unrecorded and run[key] != reference[key]], unrecorded


def shape_text(shape: dict, keys: list[str]) -> str:
    return ", ".join(f"{key} {shape[key]}" for key in keys)


def reference_identity(path: Path) -> dict:
    """The identity a reference run records: a bench report's, or a suite run's
    (the metadata bench --suite stores in the run directory's suite.json)."""
    if path.is_dir():
        index = json.loads((path / "suite.json").read_text(encoding="utf-8"))
        return (index.get("metadata") or {}).get("identity") or {}
    return json.loads(path.read_text(encoding="utf-8")).get("identity") or {}


def is_suite_run(path: Path) -> bool:
    return (path / "suite.json").is_file()


def benchmark_reference(cluster: dict) -> tuple[Path, bool]:
    """A profile's default reference run, and whether the profile names it. A
    profile whose topology differs from its baseline's reference run names its
    own with benchmark_reference."""
    if "benchmark_reference" in cluster:
        return repository_config_path(cluster["benchmark_reference"]), True
    return ROOT / "manifests/benchmarks" / f"{cluster.get('promoted_baseline')}.json", False


def benchmark_reference_problems(cluster: dict, nodes: dict) -> list[str]:
    """A benchmark_reference must name an existing run of the profile's own shape."""
    raw = cluster.get("benchmark_reference")
    if raw is None:
        return []
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or not (ROOT / raw).resolve().is_relative_to(ROOT):
        return ["benchmark_reference must be a bench.json or suite run directory relative to the repository root"]
    path = ROOT / raw
    if not (path.is_file() or is_suite_run(path)):
        return [f"benchmark_reference does not exist: {raw}"]
    reference = bench_shape(reference_identity(path))
    profile = profile_shape(cluster, nodes)
    differing, unrecorded = shape_differences(profile, reference)
    problems: list[str] = []
    if differing:
        problems.append(
            f"benchmark_reference {raw} was measured with {shape_text(reference, differing)}; "
            f"this profile has {shape_text(profile, differing)}"
        )
    if unrecorded:
        problems.append(Warn(f"benchmark_reference {raw} does not record {', '.join(unrecorded)}; bench cannot check them"))
    return problems




def bench_rows(report: dict, reference: dict | None, tolerance: float) -> tuple[list[str], list[str]]:
    """Summary lines and regressions against the reference run: Sonar's rows of every suite,
    then each node's memory and temperatures (a node that spent time in thermal slowdown makes
    the run incomparable)."""
    lines, regressions = sonar.module("matrix").rows(report, reference, tolerance)
    for name, memory in report.get("memory", {}).items():
        lines.append(
            f"{name}: min MemAvailable {memory['min_available_gib']} GiB, "
            f"swap growth {memory['swap_growth_gib']} GiB"
        )
    for name, thermal in report.get("thermal", {}).items():
        lines.append(
            f"{name}: GPU max {thermal.get('max_gpu_c')} C "
            f"({thermal.get('min_headroom_c')} C below its limit at the closest), "
            f"board max {thermal.get('max_system_c')} C, power max {thermal.get('max_power_w')} W, "
            f"thermal slowdown {thermal.get('thermal_slowdown_ms')} ms"
        )
        if thermal.get("thermal_slowdown_ms"):
            regressions.append(
                f"{name} spent {thermal['thermal_slowdown_ms']} ms in thermal slowdown; "
                "results are not comparable"
            )
    return lines, regressions


def command_sonar(args: argparse.Namespace) -> int:
    """The pinned Sonar, as a command: ``bin/spark sonar run URL ...``."""
    try:
        path = sonar.checkout()
    except sonar.SonarUnavailable as error:
        print(f"ERROR: {error}")
        return 2
    if args.path:
        print(path)
        return 0
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if arguments[:1] == ["suite"]:
        arguments = sonar.suite_arguments(arguments)
    return sonar.run(*arguments) if arguments else sonar.run("--help")


def command_workload(args: argparse.Namespace) -> int:
    """What the live service did over a window, from its counters alone (Sonar's watch)."""
    cluster, nodes, _ = configuration(args)
    head = next(node for node in nodes["nodes"] if node["head"])
    base = (args.url or f"http://{head['management_ip']}:{serve_arg(cluster, '--port')}").rstrip("/")
    run = args.run[1:] if args.run and args.run[0] == "--" else args.run
    workload = sonar.module("workload")
    summary, returncode = workload.watch(base, duration=args.duration, interval=args.interval, run=run or None)
    summary["label"] = args.label
    summary["command"] = run or None
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(workload.format_workload(summary))
    return returncode


# bench --suite exits 2 when it measured nothing (it refused), 1 when a step failed or the
# run was halted (the other steps' results stand), 0 otherwise.
SUITE_REFUSED, SUITE_FAILED = 2, 1


def suite_reference(args: argparse.Namespace, cluster: dict, nodes: dict, definition: dict,
                    digest: str) -> tuple[Path | None, str | None]:
    """The suite run to compare with, or a reason to refuse. The default is the profile's
    benchmark_reference when it is a run of the same suite and shape; a default of another
    suite or shape is refused, an explicit one of another shape only warns."""
    if args.compare == "none":
        return None, None
    if args.compare is None:
        path, _ = benchmark_reference(cluster)
        if not is_suite_run(path):
            print(f"no suite run to compare with ({path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} "
                  "is not one)")
            return None, None
    else:
        path = Path(args.compare) if Path(args.compare).is_absolute() else ROOT / args.compare
        if not is_suite_run(path):
            return None, f"--compare {args.compare} is not a suite run directory (no suite.json)"
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    index = json.loads((path / "suite.json").read_text(encoding="utf-8"))
    if index["suite_sha256"] != digest:
        return None, (f"reference {shown} is a run of suite {index['suite']['name']} {index['suite_sha256'][:12]}, "
                      f"this run's is {definition['name']} {digest[:12]}; a changed suite is a new suite: "
                      "pass --compare with a run of this suite, or --compare none")
    run, ref = profile_shape(cluster, nodes), bench_shape(reference_identity(path))
    differing, unrecorded = shape_differences(run, ref)
    if unrecorded:
        print(f"WARN: cannot compare {', '.join(unrecorded)} with reference {shown}: not recorded")
    if differing:
        mismatch = f"reference {shown} was measured with {shape_text(ref, differing)}; this run has {shape_text(run, differing)}"
        if args.compare is None:
            return None, mismatch + "; name the profile's own run in benchmark_reference, or pass --compare"
        print(f"WARN: {mismatch}")
    return path, None


def bench_suite(args: argparse.Namespace, cluster: dict, nodes: dict, config_name: str, config_path: Path) -> int:
    """Run a Sonar suite (config/suites/<name>.json, a Sonar built-in, or a JSON file) on the
    live service under every bench's conditions: the nodes cooled, the fan performance curve
    running, memory and temperatures watched (a breach stops the run), and the deployment's
    identity stored in the run's suite.json. conditions.json beside it records the rest."""
    if sonar.legacy_contract_problem():
        print(f"ERROR: {sonar.legacy_contract_problem()}")
        return SUITE_REFUSED
    try:
        sonar_suite = sonar.module("suite")
        definition = sonar.load_suite(args.suite)
    except sonar.SonarUnavailable as error:
        print(f"ERROR: {error}")
        return SUITE_REFUSED
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        return SUITE_REFUSED
    if sonar_suite.needs_dev_rpc(definition) and cluster.get("environment", {}).get("VLLM_SERVER_DEV_MODE") != "1":
        print(f"ERROR: suite {definition['name']} times the engine grid, which needs the dev-mode collective RPC "
              "(VLLM_SERVER_DEV_MODE=1); production does not expose it, so run the suite on a development arm "
              "in a lab window")
        return SUITE_REFUSED
    digest = sonar_suite.digest(definition)
    reference_path, refusal = suite_reference(args, cluster, nodes, definition, digest)
    if refusal:
        print(f"ERROR: {refusal}")
        return SUITE_REFUSED
    head = next(node for node in nodes["nodes"] if node["head"])
    base = (args.url or f"http://{head['management_ip']}:{serve_arg(cluster, '--port')}").rstrip("/")
    identity = bench_identity(cluster, nodes, config_name, config_path)
    for warning in identity["live_warnings"]:
        print(f"WARN: {warning}")
    if identity["live_problems"]:
        for problem in identity["live_problems"]:
            print(f"ERROR: {problem}")
        if not args.allow_mismatch:
            print("refusing to benchmark a cluster that differs from its configuration")
            return SUITE_REFUSED
    output = Path(args.output) if args.output else ROOT / "results/private/suites" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    conditions: dict = {"cooling": None, "fan_curve": []}
    halted: list[str] = []
    child: dict[str, subprocess.Popen] = {}

    def halt(reason: str) -> None:
        if halted:
            return
        halted.append(reason)
        print(f"HALT: {reason}", flush=True)
        if "process" in child and child["process"].poll() is None:
            child["process"].send_signal(signal.SIGINT)

    memory = MemoryWatch(nodes, float(cluster["host"]["memguard_gib"]) + 1, halt)
    code = SUITE_REFUSED
    try:
        try:
            conditions["cooling"] = cool_nodes(nodes, nodes["nodes"], args.cool_below, args.cool_timeout,
                                               fan_control=fan_control_enabled(cluster))
        except RuntimeError as error:
            print(f"ERROR: cooling failed: {error}")
            return SUITE_REFUSED
        conditions["fan_curve"] = start_fan_curve(nodes, nodes["nodes"], fan_control=fan_control_enabled(cluster))
        missing_curve = fan_curve_problems(conditions["fan_curve"])
        if missing_curve:
            if not args.allow_mismatch and not all(r["service"] == "absent" for r in conditions["fan_curve"]):
                print("ERROR: the fan performance curve is not running everywhere: " + "; ".join(missing_curve))
                return SUITE_REFUSED
            for problem in missing_curve:
                print(f"WARN: {problem}", flush=True)
        try:
            memory.wait_ready()
        except BenchAbort as error:
            print(f"ERROR: {error}")
            return SUITE_REFUSED
        handle, metadata = tempfile.mkstemp(suffix=".json")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump({"identity": identity}, stream)
        argv = sonar.argv("suite", base, "--suite", sonar.suite_name(args.suite), "--out", str(output),
                          "--label", args.label or definition["name"], "--metadata", metadata)
        if args.corpus:
            argv += ["--corpus", args.corpus]
        print(f"benchmarking {base} ({config_name}) with suite {definition['name']} {digest[:12]}; writing {output}",
              flush=True)
        try:
            child["process"] = subprocess.Popen(argv, cwd=ROOT, env=sonar.environment())
            try:
                returncode = child["process"].wait()
            except KeyboardInterrupt:
                halt("interrupted")
                returncode = child["process"].wait()
        finally:
            os.unlink(metadata)
        code = SUITE_FAILED if returncode or halted else 0
    finally:
        conditions["fan_stop_failures"] = stop_fan_curve(nodes, nodes["nodes"], conditions["fan_curve"])
        conditions["thermal"] = memory.thermal_summary()
        conditions["memory"] = memory.close()
        conditions["halted"] = halted[0] if halted else None
        (output / "conditions.json").write_text(json.dumps(conditions, indent=1) + "\n", encoding="utf-8")
    for name, thermal in conditions["thermal"].items():
        if thermal.get("thermal_slowdown_ms"):
            print(f"FAIL: {name} spent {thermal['thermal_slowdown_ms']} ms in thermal slowdown; results are not comparable")
            code = code or SUITE_FAILED
    for failure in conditions["fan_stop_failures"]:
        print(f"WARN: {failure}")
    if reference_path:
        sonar.run("compare", str(reference_path), str(output))
        print(f"reference: {reference_path}")
    return code


def command_bench(args: argparse.Namespace) -> int:
    cluster, nodes, _ = configuration(args)
    config_path, config_name = cluster_config_path(args)
    if args.suite:
        return bench_suite(args, cluster, nodes, config_name, config_path)
    try:
        matrix = sonar.module("matrix")
    except sonar.SonarUnavailable as error:
        raise SystemExit(f"ERROR: {error}") from error
    head = next(node for node in nodes["nodes"] if node["head"])
    base = (args.url or f"http://{head['management_ip']}:{serve_arg(cluster, '--port')}").rstrip("/")
    args.model = serve_arg(cluster, "--served-model-name")
    args.max_model_len = int(serve_arg(cluster, "--max-model-len"))
    args.reasoning_off = sonar.REASONING_OFF
    max_seqs = int(serve_arg(cluster, "--max-num-seqs") or 1)
    suites = matrix.prepare(args, max_seqs)
    floor_gib = float(cluster["host"]["memguard_gib"]) + 1

    reference_path = None
    if args.compare is None:
        reference_path, named = benchmark_reference(cluster)
        if is_suite_run(reference_path):
            # A shape released on a suite has no bench report to compare with;
            # bench --suite compares with its run.
            print(f"NOTE: benchmark_reference {cluster['benchmark_reference']} is a suite run; "
                  "this bench compares with nothing (bench --suite compares with it, or pass --compare PATH)")
            reference_path = None
        elif not reference_path.is_file():
            if named:
                raise SystemExit(f"benchmark_reference does not exist: {cluster['benchmark_reference']}")
            reference_path = None
    elif args.compare != "none":
        reference_path = Path(args.compare)
        if not reference_path.is_absolute():
            reference_path = ROOT / reference_path
        if not reference_path.is_file():
            raise SystemExit(f"reference does not exist: {args.compare}")
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path else None
    report = json.loads(Path(args.report).read_text(encoding="utf-8")) if args.report else None

    if reference:
        # The run's shape: the report's own record, or the profile that doctor
        # --live holds the cluster to.
        run = bench_shape(report.get("identity") or {}) if report else profile_shape(cluster, nodes)
        ref = bench_shape(reference.get("identity") or {})
        differing, unrecorded = shape_differences(run, ref)
        shown = reference_path.relative_to(ROOT) if reference_path.is_relative_to(ROOT) else reference_path
        if unrecorded:
            print(f"WARN: cannot compare {', '.join(unrecorded)} with reference {shown}: not recorded")
        if differing:
            mismatch = (
                f"reference {shown} was measured with {shape_text(ref, differing)}; "
                f"this run has {shape_text(run, differing)}"
            )
            if args.compare is None:
                print(f"ERROR: {mismatch}")
                print(
                    "refusing to compare with a default reference of another shape; name the profile's own "
                    "run in benchmark_reference, or pass --compare PATH (or --compare none) explicitly"
                )
                return 1
            print(f"WARN: {mismatch}")

    if report:
        lines, regressions = bench_rows(report, reference, args.tolerance)
        print("\n".join(lines))
        if reference_path:
            print(f"reference: {reference_path}")
        for regression in regressions:
            print(f"FAIL: regression: {regression}")
        return 1 if regressions else 0

    identity = bench_identity(cluster, nodes, config_name, config_path)
    for warning in identity["live_warnings"]:
        print(f"WARN: {warning}")
    if identity["live_problems"]:
        for problem in identity["live_problems"]:
            print(f"ERROR: {problem}")
        if not args.allow_mismatch:
            print("refusing to benchmark a cluster that differs from its configuration")
            return 1
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = Path(args.output) if args.output else ROOT / "results/private/bench" / stamp
    output.mkdir(parents=True, exist_ok=True)
    args.output_dir = output
    report_path = output / "bench.json"
    report = {
        "schema": BENCH_SCHEMA,
        "started_utc": stamp,
        "identity": identity,
        "workload": matrix.workload(args, base, suites, memory_floor_gib=floor_gib),
        "reference": str(reference_path.relative_to(ROOT)) if reference_path and reference_path.is_relative_to(ROOT) else (str(reference_path) if reference_path else None),
        "suites": {},
    }

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"benchmarking {base} ({config_name}); writing {report_path}", flush=True)
    bench = matrix.Sampler(base, guard=lambda halt: MemoryWatch(nodes, floor_gib, halt))
    memory = bench.guard
    fan_failures: list[str] = []
    try:
        try:
            report["cooling"] = cool_nodes(nodes, nodes["nodes"], args.cool_below,
                                           args.cool_timeout, fan_control=fan_control_enabled(cluster))
        except RuntimeError as error:
            raise BenchAbort(f"cooling failed: {error}") from error
        report["fan_curve"] = start_fan_curve(nodes, nodes["nodes"],
                                              fan_control=fan_control_enabled(cluster))
        save()
        missing_curve = fan_curve_problems(report["fan_curve"])
        if missing_curve:
            if not args.allow_mismatch and not all(r["service"] == "absent" for r in report["fan_curve"]):
                raise BenchAbort("the fan performance curve is not running everywhere: " + "; ".join(missing_curve))
            for problem in missing_curve:
                print(f"WARN: {problem}", flush=True)
        memory.wait_ready()
        matrix.run_suites(bench, args, suites, report, save)
    except (BenchAbort, matrix.Abort) as abort:
        report["aborted"] = str(abort)
    except OSError as error:
        bench.halt(str(error))
        report["aborted"] = f"API or telemetry error: {error}"
    except KeyboardInterrupt:
        bench.halt("interrupted")
        report["aborted"] = "interrupted"
    finally:
        fan_failures = stop_fan_curve(nodes, nodes["nodes"], report.get("fan_curve", []))
        report["thermal"] = memory.thermal_summary()
        report["memory"] = memory.close()
        report["discarded_samples"] = bench.discarded
        report["finished_utc"] = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        save()

    lines, regressions = bench_rows(report, reference, args.tolerance)
    print("\n".join(lines))
    failures = matrix.failures(report)
    failures.extend(f"regression: {item}" for item in regressions)
    failures.extend(f"fan curve: {item}" for item in fan_failures)
    if reference_path:
        print(f"reference: {report['reference']}")
    for failure in failures:
        print(f"FAIL: {failure}")
    print(f"report: {report_path}")
    return 1 if failures else 0
