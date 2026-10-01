#!/usr/bin/env python3
"""Experiment windows, runs and kernel-lab jobs on the three-Spark cluster.

usage (on dgx1, from the deployment checkout):
  scripts/lab.py window open [--minutes M] [--note TEXT]
  scripts/lab.py window status
  scripts/lab.py window close
  scripts/lab.py run SPEC.json [--dry-run] [--keep-open]
  scripts/lab.py kernel-local BUNDLE_DIR [--out DIR] [--dry-run]   (on any node, cluster stopped)
  scripts/lab.py watchdog [--max-age SECONDS] [--once]

A window holds the cluster for one experiment session. While it is open the hold file
(~/spark3-hold.json on dgx1) names this runner as holder; other agents must not stop,
restart, sync or benchmark the cluster, and this runner refuses to open a window over a
hold someone else wrote. Jobs inside a window run back to back with no restore of r5o in
between; the window closes (r5o booted, doctor --live, hold removed) when the run ends,
when a job fails, when someone writes ~/spark3-request.json, or at its time cap. A
watchdog started with the window closes it if the runner stops refreshing the heartbeat.

A run spec is JSON: {"experiment": "experiments/<dir>", "run": "rec6", "jobs": [...]} with
jobs of kind "measure" (arms booted in turn and measured; profile "lean" or "full";
"bracket" re-measures the first arm at the end), "validate" (a trace arm: boot A, scenario
traces, optional restart, per-node analysis in parallel while stopped) and "kernel" (a
kernel-lab bundle per node, run concurrently on the named nodes while the cluster is
stopped). Results keep the layout tables_arms.py reads.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_loader = importlib.machinery.SourceFileLoader("spark3", str(ROOT / "bin" / "spark3"))
_spec = importlib.util.spec_from_loader("spark3", _loader)
spark3 = importlib.util.module_from_spec(_spec)
_loader.exec_module(spark3)

HOLD = Path.home() / "spark3-hold.json"
LAB_HOME = Path.home() / "spark3-lab"
REQUEST = Path.home() / "spark3-request.json"
HOLDER_PREFIX = "spark3-lab"
CONTAINER = "dsv41-karmic-kraken"
CANDIDATE = "/opt/spark3/candidate"
DEFAULT_MINUTES = 120
HEARTBEAT_MAX_AGE = 900
TRACE_LOG_DIR = "/cache/kkref/moe-checksums"

# Measurement profiles: (script, extra arguments, output stem). The bench command is built apart.
PROFILES = {
    "full": {
        "bench": ["--min-samples", "5", "--max-samples", "5", "--prefill-sizes", "1024,4096,16384,65536",
                  "--prefill-repeats", "3"],
        "extras": [("c1_distinct.py", ["--tokens", "256"], "c1-distinct"),
                   ("c8_distinct.py", ["--samples", "3", "--tokens", "256"], "c8-distinct"),
                   ("ttft_short.py", [], "ttft"),
                   ("mixed_latency.py", ["--rounds", "3"], "mixed")],
    },
    "lean": {
        "bench": ["--min-samples", "3", "--max-samples", "3", "--prefill-sizes", "1024,16384",
                  "--prefill-repeats", "2"],
        "extras": [("c1_distinct.py", ["--tokens", "256", "--prompts", "12"], "c1-distinct"),
                   ("c8_distinct.py", ["--samples", "2", "--tokens", "256"], "c8-distinct"),
                   ("ttft_short.py", ["--reps", "3"], "ttft"),
                   ("mixed_latency.py", ["--rounds", "3"], "mixed")],
    },
}
BENCH_BASE = ["--allow-mismatch", "--compare", "none", "--suites", "decode,prefill",
              "--decode-cases", "prose,json-nothink", "--concurrency", "1", "--prefill-text", "source"]
# Validation tiers: scenario sets for boot A and the restart (None = every scenario).
TIERS = {
    "full": {"scenarios": None, "restart_scenarios": None, "mixes": True},
    "quick": {"scenarios": "mixed,chunked,chunked_end,cache_long,distinct",
              "restart_scenarios": "cache_long,distinct", "mixes": False},
}


# ---------------------------------------------------------------- small helpers

def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"{now()} {message}", flush=True)


def parse_time(text: str) -> float:
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def write_json_atomic(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def nodes_config() -> dict:
    return spark3.read_json("config/nodes.json")


def head_url() -> str:
    head = next(node for node in nodes_config()["nodes"] if node["head"])
    return f"http://{head['management_ip']}:8000"


# ---------------------------------------------------------------- hold file and guards

def read_hold(path: Path = HOLD) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return {"holder": "unreadable hold file"}


def hold_is_ours(hold: dict | None) -> bool:
    return bool(hold) and str(hold.get("holder", "")).startswith(HOLDER_PREFIX)


def new_hold(minutes: int, note: str, at: float | None = None) -> dict:
    at = time.time() if at is None else at
    stamp = dt.datetime.fromtimestamp(at, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = dt.datetime.fromtimestamp(at + 60 * minutes, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "holder": f"{HOLDER_PREFIX}: {note}" if note else HOLDER_PREFIX,
        "window": f"{socket.gethostname()}-{int(at)}",
        "since": stamp,
        "expected_end": end,
        "heartbeat": stamp,
        "rule": "no GPU experiments, restarts, cluster sync or serving changes while this file exists",
        "request": f"to ask for the cluster, write {REQUEST}; the window closes after the current job",
    }


def heartbeat_age(hold: dict, at: float | None = None) -> float:
    at = time.time() if at is None else at
    return at - parse_time(hold["heartbeat"])


def beat(note: str | None = None) -> None:
    hold = read_hold()
    if not hold_is_ours(hold):
        raise SystemExit("the hold file is not ours; refusing to continue")
    hold["heartbeat"] = now()
    if note:
        hold["current"] = note
    write_json_atomic(HOLD, hold)


def window_should_close(hold: dict, at: float | None = None) -> str | None:
    at = time.time() if at is None else at
    if REQUEST.exists():
        return f"cluster requested ({REQUEST})"
    if at > parse_time(hold["expected_end"]):
        return "time cap reached"
    return None


def requests_in_flight(url: str | None = None) -> float | None:
    """Running plus waiting requests, or None when no server answers."""
    try:
        with urllib.request.urlopen((url or head_url()) + "/metrics", timeout=5) as response:
            text = response.read().decode()
    except OSError:
        return None
    total = 0.0
    for line in text.splitlines():
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
            total += float(line.rsplit(" ", 1)[1])
    return total


def idle_for(seconds: int = 30, interval: int = 5) -> bool:
    deadline = time.time() + seconds
    while True:
        load = requests_in_flight()
        if load not in (None, 0.0):
            log(f"cluster busy ({load:g} requests)")
            return False
        if time.time() >= deadline:
            return True
        time.sleep(interval)


def published_problems() -> list[str]:
    problems = []
    subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", "origin"], check=False)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                          capture_output=True).stdout.strip()
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", "HEAD", "origin/main"]).returncode:
        problems.append(f"deployment commit {head[:8]} is not on origin/main")
    cluster, nodes, _ = spark3.configuration()
    repo = spark3.repository_path(cluster)
    for node in nodes["nodes"]:
        if node["head"]:
            continue
        theirs = spark3.run_ssh(nodes, node, "git", "-C", repo, "rev-parse", "HEAD").stdout.strip()
        if theirs != head:
            problems.append(f"{node['name']} checkout {theirs[:8]} differs from {head[:8]}")
    return problems


# ---------------------------------------------------------------- cluster actions

def spark3_cli(*arguments: str, dry: bool = False) -> int:
    command = [sys.executable, str(ROOT / "bin" / "spark3"), *arguments]
    if dry:
        print("  $ " + shlex.join(command[1:]))
        return 0
    process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in process.stdout.splitlines():
        if "docker run" not in line:
            print(line, flush=True)
    return process.returncode


def boot(config: str, dry: bool = False) -> bool:
    log(f"start {config}")
    return spark3_cli("--cluster-config", config, "cluster", "start", "--replace", "--apply", dry=dry) == 0


def stop_cluster(dry: bool = False) -> bool:
    log("stop")
    return spark3_cli("cluster", "stop", "--remove", "--apply", "--parallel", dry=dry) == 0


def production_live() -> bool:
    process = subprocess.run([sys.executable, str(ROOT / "bin" / "spark3"), "doctor", "--live"], cwd=ROOT,
                             text=True, capture_output=True)
    return process.returncode == 0 and "live cluster matches" in process.stdout


def restore_production(dry: bool = False) -> bool:
    if not dry and production_live():
        log("r5o already serving")
        return True
    ok = boot(spark3.DEFAULT_CLUSTER_CONFIG, dry=dry)
    if not dry:
        ok = production_live() and ok
        log("doctor --live " + ("OK" if ok else "FAILED"))
    return ok


def container_running(node: dict | None = None) -> bool:
    command = ["docker", "ps", "-q", "--filter", f"name=^{CONTAINER}$"]
    if node is None:
        output = subprocess.run(command, text=True, capture_output=True).stdout
    else:
        output = spark3.run_ssh(nodes_config(), node, *command).stdout
    return bool(output.strip())


# ---------------------------------------------------------------- windows

def window_open(minutes: int, note: str, dry: bool = False) -> None:
    hold = read_hold()
    if hold and not hold_is_ours(hold):
        raise SystemExit(f"cluster held by {hold.get('holder')!r} since {hold.get('since')}; not opening")
    if hold_is_ours(hold):
        log(f"window {hold['window']} already open")
        return
    if dry:
        print(f"  would open a {minutes}-minute window after the publish and idle guards")
        return
    problems = published_problems()
    if problems:
        raise SystemExit("not opening: " + "; ".join(problems))
    if not idle_for(30):
        raise SystemExit("not opening: the cluster is serving requests")
    hold = new_hold(minutes, note)
    write_json_atomic(HOLD, hold)
    log(f"window {hold['window']} open until {hold['expected_end']}")
    watchdog_log = ROOT / "results" / "private" / "lab" / f"watchdog-{hold['window']}.log"
    watchdog_log.parent.mkdir(parents=True, exist_ok=True)
    with watchdog_log.open("a") as handle:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watchdog"], cwd=ROOT,
                         stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)


def window_close(reason: str, dry: bool = False) -> bool:
    hold = read_hold()
    if hold and not hold_is_ours(hold):
        log(f"hold belongs to {hold.get('holder')!r}; leaving the cluster alone")
        return False
    log(f"closing window ({reason})")
    ok = restore_production(dry=dry)
    if not dry and ok and hold_is_ours(read_hold()):
        HOLD.unlink()
        log("window closed; hold removed")
    elif not ok:
        log("restore FAILED; hold kept so nobody assumes production is up")
    return ok


def watchdog(max_age: int, once: bool) -> None:
    log(f"watchdog: heartbeat limit {max_age} s")
    while True:
        hold = read_hold()
        if not hold_is_ours(hold):
            log("watchdog: no window of ours; exiting")
            return
        age = heartbeat_age(hold)
        if age > max_age:
            log(f"watchdog: heartbeat {age:.0f} s old; closing the window")
            window_close("watchdog: stale heartbeat")
            return
        if once:
            return
        time.sleep(60)


# ---------------------------------------------------------------- overlay checks

def image_file(image: str, path: str, cache: Path) -> Path | None:
    """The image's copy of PATH, extracted once with docker create/cp (no process is started)."""
    image_id = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True,
                              capture_output=True).stdout.strip().split(":")[-1][:16]
    target = cache / image_id / path.lstrip("/")
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    created = subprocess.run(["docker", "create", image], text=True, capture_output=True)
    if created.returncode:
        return None
    container = created.stdout.strip()
    try:
        copied = subprocess.run(["docker", "cp", f"{container}:{path}", str(target)], capture_output=True)
    finally:
        subprocess.run(["docker", "rm", container], capture_output=True)
    return target if copied.returncode == 0 else None


def fence_problems(mounts: list, image: str, home: str, read_image=None) -> list[str]:
    """Mounted B12X sources must not carry fewer TMA proxy fences than the image's own copy.

    An overlay built from an older B12X tree silently drops fences the image gained; the
    gemv-geom and mhc-mt overlays did exactly that before r5o.
    """
    read_image = read_image or (lambda path: image_file(image, path, Path.home() / ".cache" / "spark3-lab"))
    problems = []
    for source, destination, _mode in mounts:
        if not (destination.startswith(f"{CANDIDATE}/b12x/") and destination.endswith(".py")):
            continue
        local = Path(source.format(home=home))
        reference = read_image(destination)
        if reference is None or not local.is_file():
            continue
        mine = local.read_text(encoding="utf-8").count("fence_proxy")
        theirs = Path(reference).read_text(encoding="utf-8").count("fence_proxy")
        if mine < theirs:
            problems.append(f"{local}: {mine} fence_proxy calls, image has {theirs} ({destination})")
    return problems


def overlay_manifest(config: dict, home: str) -> list[str]:
    lines = []
    for source, destination, _mode in config["container"]["mounts"]:
        path = Path(source.format(home=home))
        if "spark3-overlay" in source and path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path} -> {destination}")
    return lines


# ---------------------------------------------------------------- plans

def arm_config_path(experiment: str, config: str) -> str:
    return config if "/" in config else f"{experiment}/{config}"


def measure_steps(spec: dict, job: dict) -> list[dict]:
    experiment, run = spec["experiment"], spec["run"]
    profile = PROFILES[job.get("profile", "lean")]
    arms = list(job["arms"])
    if job.get("bracket") and len(arms) > 1:
        first = dict(arms[0])
        first["label"] = f"{first['label']}-end"
        arms.append(first)
    steps = []
    for arm in arms:
        config, label = arm_config_path(experiment, arm["config"]), arm["label"]
        out = f"results/private/determinism/{run}"
        steps.append({"kind": "boot", "config": config, "label": label})
        steps.append({"kind": "cli", "label": label,
                      "argv": ["--cluster-config", config, "bench", *BENCH_BASE, *profile["bench"],
                               "--output", f"results/private/bench/{run}-{label}"]})
        for script, extra, stem in profile["extras"]:
            steps.append({"kind": "script", "label": label,
                          "argv": [f"{experiment}/{script}", head_url(), *extra],
                          "out": f"{out}/{stem}-{label}.jsonl"})
    return steps


def validate_steps(spec: dict, job: dict) -> list[dict]:
    experiment = spec["experiment"]
    tier = TIERS[job.get("tier", "full")]
    config = arm_config_path(experiment, job["config"])
    out = f"results/private/determinism/{job['name']}"
    url = head_url()
    chunk = str(job.get("chunk", 4000))
    steps = [{"kind": "boot", "config": config, "label": job["name"]},
             {"kind": "fresh_inventory"},
             {"kind": "script", "argv": [f"{experiment}/c8_trace.py", url, f"{out}/c8", "--rounds", "1",
                                         "--tokens", "16"], "out": f"{out}/c8-runs.jsonl"}]
    scenarios = [f"{experiment}/scenario_trace.py", url, f"{out}/scenarios", "--repeats", "1", "--chunk", chunk]
    if tier["scenarios"]:
        scenarios += ["--scenarios", tier["scenarios"]]
    steps.append({"kind": "script", "argv": scenarios, "out": f"{out}/scenario-runs.jsonl"})
    dirs = ["c8", "scenarios"]
    if tier["mixes"]:
        steps.append({"kind": "script", "argv": [f"{experiment}/trace_mixes.py", url, f"{out}/mixes",
                                                 "--repeats", "1", "--tokens", "64", "--prompts",
                                                 "json,prose,long"], "out": f"{out}/mixes-runs.jsonl"})
        dirs.append("mixes")
    if job.get("restart", True):
        restart = [f"{experiment}/scenario_trace.py", url, f"{out}/scenarios", "--repeats", "1", "--chunk",
                   chunk, "--suffix", "-bootB"]
        if tier["restart_scenarios"]:
            restart += ["--scenarios", tier["restart_scenarios"]]
        steps.append({"kind": "boot", "config": config, "label": f"{job['name']} boot B"})
        steps.append({"kind": "script", "argv": restart, "out": f"{out}/scenario-runs-bootB.jsonl"})
    steps.append({"kind": "stop"})
    steps.append({"kind": "analyze", "out": out, "dirs": dirs, "config": config})
    return steps


def kernel_steps(spec: dict, job: dict) -> list[dict]:
    return [{"kind": "stop"},
            {"kind": "kernel", "bundles": job["bundles"], "out": f"results/private/lab/{spec['run']}"}]


def plan(spec: dict) -> list[dict]:
    builders = {"measure": measure_steps, "validate": validate_steps, "kernel": kernel_steps}
    steps = []
    for job in spec["jobs"]:
        if job["kind"] not in builders:
            raise SystemExit(f"unknown job kind {job['kind']!r}")
        steps.extend(builders[job["kind"]](spec, job))
    return steps


def describe_step(step: dict) -> str:
    if step["kind"] == "boot":
        return f"boot {step['config']} ({step['label']})"
    if step["kind"] == "cli":
        return "bin/spark3 " + shlex.join(step["argv"])
    if step["kind"] == "script":
        return "python3 " + shlex.join(step["argv"]) + f" > {step['out']}"
    if step["kind"] == "analyze":
        return f"analyze {step['out']} {','.join(step['dirs'])} on every node, in parallel"
    if step["kind"] == "kernel":
        return "kernel-lab " + ", ".join(f"{b['bundle']}@{b['node']}" for b in step["bundles"])
    return step["kind"]


# ---------------------------------------------------------------- execution

def run_script(step: dict) -> int:
    out = ROOT / step["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        return subprocess.run([sys.executable, *step["argv"]], cwd=ROOT, stdout=handle,
                              stderr=subprocess.STDOUT).returncode


def fresh_inventory() -> None:
    nodes = nodes_config()
    for node in nodes["nodes"]:
        spark3.run_ssh(nodes, node, "docker", "exec", CONTAINER, "sh", "-c",
                       f"rm -f {TRACE_LOG_DIR}/inventory-* {TRACE_LOG_DIR}/plans-*")


def analyze(step: dict) -> dict:
    """analyze_trace4.py per node, the three nodes concurrently (cluster stopped: memory is free)."""
    if container_running():
        raise SystemExit("analysis needs the cluster stopped")
    config = json.loads((ROOT / step["config"]).read_text())
    image = config["container"]["image"]
    experiment = str(Path(step["config"]).parent)
    out = ROOT / step["out"]

    def one_node(node: str) -> None:
        for d in step["dirs"]:
            with (out / f"analysis4-{d}-{node}.jsonl").open("w") as handle:
                process = subprocess.run(
                    ["docker", "run", "--rm", "--memory=16g", "-e", "CUDA_VISIBLE_DEVICES=",
                     "-v", f"{ROOT / experiment / 'analyze_trace4.py'}:/a.py:ro", "-v", f"{out / d}:/t:ro",
                     "--entrypoint", "python3", image, "/a.py", "/t", "--node", node, "--chain", "12",
                     "--all-pairs"], text=True, capture_output=True)
                handle.write("".join(line + "\n" for line in (process.stdout + process.stderr).splitlines()
                                     if "Warn" not in line))

    names = [node["name"] for node in nodes_config()["nodes"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as pool:
        list(pool.map(one_node, names))
    summary = summarize_analysis(out)
    write_json_atomic(out / "analysis-summary.json", summary)
    return summary


def summarize_analysis(out: Path) -> dict:
    totals = {}
    for path in sorted(out.glob("analysis4-*.jsonl")):
        entry = {"pairs": 0, "rows": 0, "differing": 0, "outputs_differ": 0, "cross_boot_pairs": 0,
                 "cross_boot_rows": 0, "recompute_mismatches": 0}
        for line in path.read_text().splitlines():
            if not line.startswith("{"):
                continue
            record = json.loads(line)
            if "rows_compared" in record:
                entry["pairs"] += 1
                entry["rows"] += record["rows_compared"]
                entry["differing"] += record["rows_differing"]
                entry["outputs_differ"] += not record["outputs_equal"]
                if ("bootB" in record["request"]) != ("bootB" in record["against"]):
                    entry["cross_boot_pairs"] += 1
                    entry["cross_boot_rows"] += record["rows_compared"]
            entry["recompute_mismatches"] += len(record.get("recomputed_mismatch") or [])
        totals[path.stem] = entry
    totals["passed"] = all(e["differing"] == 0 and e["outputs_differ"] == 0 and e["recompute_mismatches"] == 0
                           for k, e in totals.items() if k != "passed") and len(totals) > 0
    return totals


def run_kernel_bundles(step: dict) -> list[dict]:
    """Ship each bundle to ~/spark3-lab on its node and run kernel-local there, nodes concurrently.

    Bundles, their inputs and outputs stay outside the deployment checkout: a bundle under
    development never dirties a node's checkout (a dirty checkout blocks cluster start), and
    ignored files never enter a checkout by copy. Paths listed under "sync" (inputs such as
    captures, kept out of git) go to ~/spark3-lab/inputs/<path> on the other nodes.
    """
    nodes = nodes_config()
    cluster, _, _ = spark3.configuration()
    repo = spark3.repository_path(cluster)
    run = Path(step["out"]).name
    home = cluster["host"]["home"]

    def rsync(node: dict, source: Path, destination: str) -> None:
        spark3.run_ssh(nodes, node, "mkdir", "-p", str(Path(destination).parent))
        subprocess.run(["rsync", "-a", "--delete" if source.is_dir() else "--checksum",
                        "-e", "ssh " + " ".join(spark3.ssh_options()),
                        f"{source}/" if source.is_dir() else str(source),
                        f"{spark3.ssh_target(nodes, node)}:{destination}{'/' if source.is_dir() else ''}"],
                       check=False)

    def one(entry: dict) -> dict:
        node = spark3.node_by_name(nodes, entry["node"])
        bundle = ROOT / entry["bundle"]
        candidate = json.loads((bundle / "candidate.json").read_text())
        remote = f"{home}/spark3-lab/bundles/{run}/{bundle.name}"
        out = f"{home}/spark3-lab/results/{run}/{bundle.name}-{entry['node']}"
        rsync(node, bundle, remote)
        if not node["head"]:
            for relative in candidate.get("sync", []):
                rsync(node, ROOT / relative, f"{home}/spark3-lab/inputs/{relative}")
        process = spark3.run_ssh(nodes, node, "bash", "-lc",
                                 f"cd {shlex.quote(repo)} && python3 scripts/lab.py kernel-local "
                                 f"{shlex.quote(remote)} --out {shlex.quote(out)}")
        verdict = {"node": entry["node"], "bundle": entry["bundle"], "exit": process.returncode}
        fetched = spark3.run_ssh(nodes, node, "cat", f"{out}/verdict.json")
        if fetched.returncode == 0:
            verdict.update(json.loads(fetched.stdout))
        else:
            verdict.update({"passed": False, "errors": [process.stdout[-2000:], process.stderr[-2000:]]})
        return verdict

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(step["bundles"])) as pool:
        verdicts = list(pool.map(one, step["bundles"]))
    local = ROOT / step["out"]
    local.mkdir(parents=True, exist_ok=True)
    write_json_atomic(local / "verdicts.json", {"verdicts": verdicts, "agree": verdicts_agree(verdicts)})
    return verdicts


def verdicts_agree(verdicts: list[dict]) -> bool:
    """The same bundle on different nodes must give the same row groups and bit-equality sets."""
    by_bundle = {}
    for verdict in verdicts:
        key = (tuple((g["op"], g["groups"]) for g in verdict.get("groups", [])),
               tuple((b["bits"], tuple(b.get("differs_at", []))) for b in verdict.get("bits", [])))
        by_bundle.setdefault(verdict["bundle"], set()).add(key)
    return all(len(keys) == 1 for keys in by_bundle.values())


def execute(spec: dict, dry: bool, keep_open: bool) -> int:
    steps = plan(spec)
    if dry:
        print(f"run {spec['run']}: {len(steps)} steps")
        for index, step in enumerate(steps, 1):
            print(f"{index:3d}. {describe_step(step)}")
        return 0
    hold = read_hold()
    if not hold_is_ours(hold):
        window_open(spec.get("minutes", DEFAULT_MINUTES), f"run {spec['run']}")
    home = spark3.configuration()[0]["host"]["home"]
    checked = set()
    failed = None
    current = {"note": "starting"}
    stop_beating = threading.Event()

    def heartbeat() -> None:
        # The heartbeat proves the runner is alive, not that a step is quick; the time cap
        # bounds a hung step, the watchdog a dead runner.
        while not stop_beating.wait(60):
            try:
                beat(current["note"])
            except SystemExit:
                return

    threading.Thread(target=heartbeat, daemon=True).start()
    for step in steps:
        hold = read_hold()
        if not hold_is_ours(hold):
            failed = "hold file lost"
            break
        reason = window_should_close(hold)
        if reason:
            failed = reason
            break
        current["note"] = describe_step(step)
        beat(current["note"])
        if step["kind"] == "boot":
            if step["config"] not in checked:
                config = json.loads((ROOT / step["config"]).read_text())
                problems = fence_problems(config["container"]["mounts"], config["container"]["image"], home)
                if problems:
                    failed = "stale overlay: " + "; ".join(problems)
                    break
                manifest = ROOT / "results" / "private" / "lab" / f"overlays-{spec['run']}.txt"
                manifest.parent.mkdir(parents=True, exist_ok=True)
                with manifest.open("a") as handle:
                    handle.write(f"# {step['config']}\n" + "".join(l + "\n" for l in overlay_manifest(config, home)))
                checked.add(step["config"])
            if not boot(step["config"]):
                failed = f"boot of {step['config']} failed"
                break
        elif step["kind"] == "cli":
            log(f"bench {step['label']} exit {spark3_cli(*step['argv'])}")
        elif step["kind"] == "script":
            log(f"{Path(step['argv'][0]).name} {step.get('label', '')} exit {run_script(step)}")
        elif step["kind"] == "fresh_inventory":
            fresh_inventory()
        elif step["kind"] == "stop":
            if container_running() and not stop_cluster():
                failed = "stop failed"
                break
        elif step["kind"] == "analyze":
            summary = analyze(step)
            log(f"analysed {step['out']}: {'PASSED' if summary['passed'] else 'DIFFERENCES'}")
        elif step["kind"] == "kernel":
            verdicts = run_kernel_bundles(step)
            for verdict in verdicts:
                log(f"kernel {verdict['bundle']}@{verdict['node']}: "
                    f"{'pass' if verdict.get('passed') else 'FAIL'} ({verdict.get('seconds', '?')} s)")
            log(f"kernel verdicts {'agree' if verdicts_agree(verdicts) else 'DISAGREE'} across nodes")
    stop_beating.set()
    if failed:
        log(f"run stopped: {failed}")
    if failed or not keep_open:
        window_close(failed or "run complete")
    log("done")
    return 1 if failed else 0


# ---------------------------------------------------------------- kernel lab (node-local)

def compile_cache(candidate: dict) -> Path:
    """Per-node compile caches shared by every bundle; the GPU lock keeps jobs one at a time."""
    return Path(os.path.expanduser(candidate.get("cache", "~/.cache/spark3-lab/compile")))


def bundle_mounts(bundle: Path, candidate: dict) -> list[tuple[Path, str]]:
    """Mount sources: ~-paths and absolute paths as given; otherwise the bundle's own file, else a
    synced input under ~/spark3-lab/inputs, else the path in this node's checkout. A source that
    exists nowhere is reported by kernel-local; nothing falls back silently to an empty mount."""
    mounts = []
    for source, destination in candidate.get("mounts", []):
        resolved = Path(os.path.expanduser(source))
        if not resolved.is_absolute():
            for base in (bundle, LAB_HOME / "inputs", ROOT):
                if (base / source).exists():
                    resolved = base / source
                    break
            else:
                resolved = ROOT / source
        mounts.append((resolved, destination))
    return mounts


def bundle_command(bundle: Path, candidate: dict, out: Path) -> list[str]:
    """docker run for one kernel-lab bundle: overlay/{vllm,b12x}/... mounted at the candidate tree."""
    image = candidate["image"]
    command = ["docker", "run", "--rm", "--gpus", "all", "--ipc=host"]
    for key, value in sorted(candidate.get("env", {}).items()):
        command += ["-e", f"{key}={value}"]
    overlay = bundle / "overlay"
    if overlay.is_dir():
        for path in sorted(p for p in overlay.rglob("*") if p.is_file()):
            relative = path.relative_to(overlay)
            package = relative.parts[0]
            if package not in ("vllm", "b12x"):
                continue
            command += ["-v", f"{path}:{CANDIDATE}/{package}/{package}/{Path(*relative.parts[1:])}:ro"]
    for source, destination in bundle_mounts(bundle, candidate):
        command += ["-v", f"{source}:{destination}:ro"]
    command += ["-v", f"{compile_cache(candidate)}:/c"]
    command += ["-w", candidate.get("workdir", f"{CANDIDATE}/b12x"), "--entrypoint", "python3", image,
                *candidate["argv"]]
    return command


def verdict_from_lines(lines: list[str], candidate: dict) -> dict:
    """Pass when every invariant configuration has one row group and every expected-equal pair matches."""
    exempt = tuple(candidate.get("variant_configs", ["production", "ref2"]))
    groups, bits, timings, errors = [], [], [], []
    for line in lines:
        if "Traceback" in line or line.startswith(("RuntimeError", "ValueError", "AssertionError")):
            errors.append(line.strip())
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "groups" in record:
            groups.append(record)
        elif "bits" in record:
            bits.append(record)
        elif "timing_us" in record:
            timings.append(record)
        elif "error" in record:
            errors.append(f"{record.get('config')}: {record['error']}")
    variant = []
    for record in groups:
        name = record["op"].split()[1] if len(record["op"].split()) > 1 else record["op"]
        if record["groups"] != 1 and not name.startswith(exempt):
            variant.append(record["op"])
    unequal = [record["bits"] for record in bits if record.get("differs_at")]
    return {
        "passed": not (variant or unequal or errors) and bool(groups or bits),
        "batch_variant": variant,
        "bits_unequal": unequal,
        "errors": errors[:20],
        "groups": groups,
        "bits": bits,
        "timings": timings,
    }


def kernel_local(bundle_dir: str, out_dir: str | None, dry: bool) -> int:
    bundle = Path(bundle_dir).resolve()
    candidate = json.loads((bundle / "candidate.json").read_text())
    out = Path(out_dir).resolve() if out_dir else bundle / "out"
    command = bundle_command(bundle, candidate, out)
    if dry:
        print(shlex.join(command))
        return 0
    if container_running():
        raise SystemExit(f"{CONTAINER} is running on this node; the kernel lab needs the GPU to itself")
    missing = [str(source) for source, _ in bundle_mounts(bundle, candidate) if not source.exists()]
    if missing:
        raise SystemExit("missing bundle inputs: " + ", ".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    compile_cache(candidate).mkdir(parents=True, exist_ok=True)
    with open("/tmp/spark3-lab-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.time()
        process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
    (out / "output.txt").write_text(process.stdout)
    verdict = verdict_from_lines(process.stdout.splitlines(), candidate)
    image_id = subprocess.run(["docker", "image", "inspect", candidate["image"], "--format", "{{.Id}}"],
                              text=True, capture_output=True).stdout.strip()
    inputs = sorted(p for p in bundle.rglob("*") if p.is_file() and out not in p.parents)
    verdict.update({
        "node": socket.gethostname(), "image": image_id, "exit": process.returncode,
        "seconds": round(elapsed, 1),
        "inputs_sha256": {str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    })
    if process.returncode:
        verdict["passed"] = False
    write_json_atomic(out / "verdict.json", verdict)
    log(f"kernel {bundle.name}: {'pass' if verdict['passed'] else 'FAIL'} in {elapsed:.0f} s")
    return 0 if verdict["passed"] else 1


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    window = commands.add_parser("window")
    window.add_argument("action", choices=("open", "status", "close"))
    window.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    window.add_argument("--note", default="")
    window.add_argument("--dry-run", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("spec")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--keep-open", action="store_true", help="leave the window open after the run")
    kernel = commands.add_parser("kernel-local")
    kernel.add_argument("bundle")
    kernel.add_argument("--out")
    kernel.add_argument("--dry-run", action="store_true")
    dog = commands.add_parser("watchdog")
    dog.add_argument("--max-age", type=int, default=HEARTBEAT_MAX_AGE)
    dog.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "window":
        if args.action == "open":
            window_open(args.minutes, args.note, dry=args.dry_run)
        elif args.action == "close":
            return 0 if window_close("closed by hand", dry=args.dry_run) else 1
        else:
            hold = read_hold()
            print(json.dumps(hold, indent=2) if hold else "no hold")
            if hold and "heartbeat" in hold:
                print(f"heartbeat age {heartbeat_age(hold):.0f} s")
        return 0
    if args.command == "run":
        spec = json.loads(Path(args.spec).read_text())
        return execute(spec, args.dry_run, args.keep_open)
    if args.command == "kernel-local":
        return kernel_local(args.bundle, args.out, args.dry_run)
    watchdog(args.max_age, args.once)
    return 0


if __name__ == "__main__":
    sys.exit(main())
