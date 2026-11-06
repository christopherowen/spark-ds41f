"""C46 private scope/resource adapter; generic semantic validation lives elsewhere."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
import build_resources

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESTORE_IMAGE = "sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11"
RESTORE_PROFILE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json"
NODES = ["dgx1", "dgx2", "dgx3", "dgx4"]
OWNER = "copy-default-screening-c46"
CLAIM_PROBE = r'''
import json
from pathlib import Path
home = Path.home()
paths = [home / p for p in ("spark-hold.json", "spark-node-hold.json", "spark-cpu-hold.json", "spark-gpu-hold.json", "spark-lab/hold.json")]
for name in ("spark-lab/reservations", "spark-claims", "spark-node-claims", "spark-cpu-claims", "spark-gpu-claims"):
    directory = home / name
    if directory.is_dir():
        paths.extend(sorted(directory.glob("*.json")))
records = {}
for path in paths:
    if not path.exists():
        continue
    if not path.is_file() or path.stat().st_size > 65536:
        raise RuntimeError("unreadable/oversized local hold or claim: " + str(path))
    records[str(path)] = json.loads(path.read_text())
print(json.dumps(records))
'''
PREP_LIMITS = {"cpus": 1, "memory_gib": 8, "extra_swap_gib": 0, "threads": 1}
GATE_LIMITS = {"cpus": 2, "memory_gib": 32, "memory_plus_swap_gib": 32, "pids": 512,
               "threads": 1, "max_jobs": 2}
INITCHECK_LIMITS = {**GATE_LIMITS, "memory_gib": 16, "memory_plus_swap_gib": 16}


def load_plan():
    return json.loads((HERE / "control-readiness.json").read_text())


def launch_blockers(plan=None):
    plan = load_plan() if plan is None else plan
    blockers = []
    worker = plan.get("builder_worker", {})
    if not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", worker.get("image") or ""):
        blockers.append("published immutable BuildKit worker image is missing")
    if worker.get("dispatch_status") != "reviewed_inside_owned_gate":
        blockers.append("inside-window owned-worker create/inspect/validate/stop dispatch is pending review")
    if plan.get("semantic_validator", {}).get("integration_status") != "reviewed_and_connected":
        blockers.append("published declared-evidence API and C46 semantic barrier await remote verification")
    if plan.get("model_barrier", {}).get("integration_status") != "reviewed_and_connected":
        blockers.append("existing model-step barrier awaits integration and remote verification")
    if plan.get("window_lifecycle", {}).get("integration_status") != "reviewed_and_connected":
        blockers.append("exact-window lifecycle and restoration controls await remote verification")
    return blockers


def require_launch_ready():
    blockers = launch_blockers()
    if blockers:
        raise RuntimeError("C46 source/control packet is not launch-ready: " + "; ".join(blockers))
    from c46_functional import validate
    if not callable(validate):
        raise RuntimeError("C46 semantic callback is not connected")


def _validate_process(value, commit, role):
    suffix = "/experiments/2026-11-06-worker-test-isolation-rework/" + ("run_round.py" if role == "controller" else "c46_window.py")
    if (not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit) or not isinstance(value, dict) or value.get("host") != "dgx1"
            or type(value.get("pid")) is not int or value["pid"] < 1
            or type(value.get("start_ticks")) is not int or value["start_ticks"] < 1
            or not value.get("argv") or not str(value.get("path", "")).endswith(suffix)
            or value.get("commit") != commit or value.get("published_commit") != commit or value.get("start_commit") != commit):
        raise ValueError("actual C46 " + role + " identity is incomplete")


def process_identity(pid, commit, script, proc=Path("/proc")):
    """Read only process birth/argv/cwd; called immediately after this controller opens the lease."""
    directory = proc / str(pid)
    stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    argv = [x.decode() for x in (directory / "cmdline").read_bytes().split(b"\0") if x]
    cwd = (directory / "cwd").resolve()
    path = (ROOT / script).resolve()
    if stat[0] == "Z" or cwd != ROOT.resolve() or not any((cwd / arg).resolve() == path for arg in argv[1:] if not arg.startswith("-")):
        raise RuntimeError("process is not the expected live C46 controller/watchdog")
    return {"host": "dgx1", "pid": pid, "ppid": int(stat[1]), "argv": argv,
            "path": str(path), "script": script, "cwd": str(cwd), "start_ticks": int(stat[19]),
            "commit": commit, "published_commit": commit, "start_commit": commit,
            "commit_observation": "Published checkout at immediate post-window-open capture; no checkout mutation is performed."}


def capture_control_processes(commit, watchdog_argv, watchdog_pid, proc=Path("/proc")):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("published execution commit is required")
    pid = os.getpid()
    controller = process_identity(pid, commit, str(HERE.relative_to(ROOT) / "run_round.py"), proc)
    expected = watchdog_argv[1:]
    until = time.monotonic() + 2
    while True:
        matches = []
        for child in (proc / str(pid) / "task" / str(pid) / "children").read_text().split():
            try:
                item = process_identity(int(child), commit, str(HERE.relative_to(ROOT) / "c46_window.py"), proc)
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                continue
            if item["pid"] == watchdog_pid and item["ppid"] == pid and item["argv"][1:] == expected:
                matches.append(item)
        if len(matches) == 1:
            return controller, matches[0]
        if len(matches) > 1 or time.monotonic() >= until:
            raise RuntimeError("one actual direct-child named-profile watchdog was not observed")
        time.sleep(.05)


def _watchdog_binding(watchdog, token, holder, window, commit):
    for flag, value in (("--window", window), ("--holder", holder), ("--owner-token", token), ("--commit", commit),
                        ("--restore-profile", RESTORE_PROFILE)):
        argv = watchdog["argv"]
        if argv.count(flag) != 1 or argv[argv.index(flag) + 1:argv.index(flag) + 2] != [value]:
            raise ValueError("actual watchdog is not bound to this exact lease and restoration commit")


def scope(token, holder, window, controller, restore, watchdog):
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise ValueError("one unique C46 owner token is required")
    if restore != RESTORE_PROFILE:
        raise ValueError("C46 must name the independently verified C36 restoration profile")
    commit = controller.get("commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("published execution commit is required")
    _validate_process(controller, commit, "controller")
    _validate_process(watchdog, commit, "watchdog")
    if watchdog.get("ppid") != controller["pid"]:
        raise ValueError("watchdog must be this controller's actual direct child")
    _watchdog_binding(watchdog, token, holder, window, commit)
    return {"owner": OWNER, "token": window, "owner_token": token, "holder": holder, "window": window,
            "controller": controller, "nodes": NODES,
            "allocation": {"dgx1-3": "TP3 functional and two complete metric boots",
                           "dgx4": "sole builder/gates; no concurrent GPU work during TP3 timing"},
            "source_prepare_limits": PREP_LIMITS,
            "build_worker_limits": {"cpus": 2, "memory_gib": 32, "extra_swap_gib": 0,
                                    "compile_jobs": 2, "parallel_build_steps": 1},
            "restore_profile": restore, "restore_image": RESTORE_IMAGE,
            "restore": {"profile": restore, "commit": commit, "image": RESTORE_IMAGE, "nodes": NODES[:3]},
            "watchdog": watchdog,
            "pool": "DGX shared window only; no authority over independently leased Spaceport groups"}


def assert_owned(hold, window, holder, token):
    expected = hold.get("resource_scope", {}) if isinstance(hold, dict) else {}
    if not isinstance(expected, dict):
        raise RuntimeError("C46 resource scope is unreadable")
    controller = expected.get("controller", {})
    if not isinstance(controller, dict):
        raise RuntimeError("C46 controller identity is unreadable")
    if (not isinstance(hold, dict) or hold.get("window") != window or hold.get("holder") != holder
            or expected.get("owner") != OWNER or expected.get("owner_token") != token
            or expected.get("holder") != holder or expected.get("window") != window or expected.get("token") != window
            or expected.get("nodes") != NODES or expected.get("restore_profile") != RESTORE_PROFILE
            or expected.get("restore_image") != RESTORE_IMAGE
            or expected.get("restore") != {"profile": RESTORE_PROFILE, "commit": controller.get("commit"),
                                            "image": RESTORE_IMAGE, "nodes": NODES[:3]}):
        raise RuntimeError("exact C46 holder/window/token/resource scope changed")
    try:
        _validate_process(controller, controller.get("commit", ""), "controller")
        _validate_process(expected.get("watchdog"), controller.get("commit", ""), "watchdog")
        if expected["watchdog"].get("ppid") != controller["pid"]:
            raise ValueError("watchdog parent differs")
        _watchdog_binding(expected["watchdog"], token, holder, window, controller["commit"])
    except ValueError as error:
        raise RuntimeError("exact C46 process metadata changed") from error
    return hold


def validate_local_records(records, token, window):
    """No availability is inferred from absent local records: the owned lease is required too."""
    for path, value in records.items():
        if value is None:
            continue
        if not isinstance(value, dict):
            raise RuntimeError("unreadable local hold/claim: " + path)
        own = value.get("resource_scope", value)
        if (not isinstance(own, dict) or own.get("owner") != OWNER
                or own.get("owner_token") != token or own.get("window") != window):
            raise RuntimeError("foreign local node/CPU/GPU hold or claim: " + path)
    return {"passed": True, "records": records,
            "limit": "Explicit records only; real exact owned DGX lease is separately mandatory."}


def container_limits(kind):
    if kind not in ("release", "initcheck"):
        raise ValueError("unknown C46 gate-container role")
    limits = GATE_LIMITS if kind == "release" else INITCHECK_LIMITS
    result = ["--cpus", str(limits["cpus"]), "--memory", f'{limits["memory_gib"]}g',
              "--memory-swap", f'{limits["memory_plus_swap_gib"]}g',
              "--pids-limit", str(limits["pids"])]
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        result += ["-e", key + "=1"]
    return result + ["-e", "MAX_JOBS=2"]


def audit_native_files(manifest, lines):
    rows = [json.loads(line) for line in lines.splitlines() if line.strip()]
    names = [row.get("file") for row in rows]
    wanted = list(manifest["files"])
    passed = (len(wanted) == 138 and len(names) == len(set(names)) == len(wanted)
              and set(names) == set(wanted)
              and all(row.get("status") == "ok" and row.get("failed") == 0
                      and row.get("error") == 0 and not row.get("failures") for row in rows)
              and sum(row.get("passed", 0) for row in rows) > 0)
    return {"passed": passed, "files": len(rows), "expected_files": len(wanted),
            "passed_cases": sum(row.get("passed", 0) for row in rows),
            "skipped_cases": sum(row.get("skipped", 0) for row in rows),
            "failed_cases": sum(row.get("failed", 0) for row in rows),
            "error_cases": sum(row.get("error", 0) for row in rows),
            "results_sha256": hashlib.sha256(lines.encode()).hexdigest()}


def validate_functional_semantics(*args, **kwargs):
    from c46_functional import validate
    return validate(*args, **kwargs)


def worker_name(commit, token):
    if (not isinstance(commit, str) or not isinstance(token, str)
            or not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"[0-9a-f]{32}", token)):
        raise ValueError("exact execution commit and unique owner token are required")
    return "c46-" + commit[:12] + "-" + token[:12]


def validate_worker_receipt(worker, stopped):
    pin = load_plan().get("builder_worker", {}).get("image") or ""
    name = worker_name(worker.get("commit", ""), worker.get("owner_token", ""))
    plan = build_resources.builder_plan(name, pin)
    validation = worker.get("validation", {})
    actual_image = worker.get("actual_image", {})
    if (worker.get("owner") != OWNER or worker.get("passed") is not True or worker.get("created") is not True
            or worker.get("builder") != name or worker.get("requested_image") != pin
            or not worker.get("window") or validation.get("passed") is not True
            or not re.fullmatch(r"[0-9a-f]{64}", worker.get("container_id", ""))
            or validation.get("container_id") != worker["container_id"]
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", validation.get("image_id", ""))
            or actual_image.get("Id") != validation.get("image_id")
            or actual_image.get("Os") != "linux" or actual_image.get("Architecture") != "arm64"
            or pin not in actual_image.get("RepoDigests", [])
            or validation.get("plan", {}).get("limits") != plan["limits"]
            or validation.get("plan", {}).get("config_sha256") != plan["config_sha256"]
            or validation.get("effective_host_config") != {"CpuQuota": 200000, "CpuPeriod": 100000,
                                                            "Memory": build_resources.MEMORY, "MemorySwap": build_resources.MEMORY}
            or str(validation.get("cgroup", {}).get("cpu.max", "")).split() != ["200000", "100000"]
            or str(validation.get("cgroup", {}).get("memory.max", "")) != str(build_resources.MEMORY)
            or str(validation.get("cgroup", {}).get("memory.swap.max", "")) != "0"):
        raise ValueError("complete owned-worker aggregate resource evidence is required")
    if (stopped.get("passed") is not True or stopped.get("owner_token") != worker["owner_token"]
            or stopped.get("builder") != name or stopped.get("container_id") != worker["container_id"]
            or stopped.get("cache_state_preserved") is not True or stopped.get("global_builder_selection_changed") is not False):
        raise ValueError("exact owned-worker stop must precede GPU gates")
    return worker


def _worker_save(out, record):
    path = Path(out) / "worker-control" / "receipt.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def _worker_command(record, out, argv, label, allow_failure=False, timeout=60):
    directory = Path(out) / "worker-control"
    prefix = f'{len(record["commands"]):02d}-{label}'
    stdout, stderr = directory / (prefix + ".stdout"), directory / (prefix + ".stderr")
    item = {"argv": argv, "stdout": stdout.name, "stderr": stderr.name}
    record["commands"].append(item)
    try:
        with stdout.open("xb") as output, stderr.open("xb") as errors:
            result = subprocess.run(argv, cwd=ROOT, stdout=output, stderr=errors, timeout=timeout)
        item["exit"] = result.returncode
    except BaseException as error:
        item["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _worker_save(out, record)
    if result.returncode and not allow_failure:
        raise RuntimeError("owned worker " + label + f" exited {result.returncode}")
    if stdout.stat().st_size > 131072:
        if label == "bootstrap":
            return result.returncode, ""
        raise RuntimeError("oversized worker metadata; original output retained")
    return result.returncode, stdout.read_text()


def _worker_scope(window, holder, token):
    raw = subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "dgx1",
                                   "cat ~/spark-hold.json"], text=True, timeout=15)
    assert_owned(json.loads(raw), window, holder, token)
    raw = subprocess.check_output([sys.executable, "-c", CLAIM_PROBE], text=True, timeout=10)
    return validate_local_records(json.loads(raw), token, window)


def worker_cgroup(pid, proc=Path("/proc"), cgroups=Path("/sys/fs/cgroup")):
    if type(pid) is not int or pid < 1:
        raise ValueError("actual worker host PID is required")
    entries = [line.split(":", 2) for line in (proc / str(pid) / "cgroup").read_text().splitlines()]
    paths = [entry[2] for entry in entries if entry[:2] == ["0", ""]]
    if len(paths) != 1 or not paths[0].startswith("/") or ".." in Path(paths[0]).parts:
        raise RuntimeError("one actual unified worker cgroup was not observed")
    directory = (cgroups / paths[0].lstrip("/")).resolve()
    if not directory.is_relative_to(cgroups.resolve()):
        raise RuntimeError("worker cgroup escaped the host hierarchy")
    return {"host_pid": pid, "host_cgroup_path": paths[0],
            **{field: (directory / field).read_text().strip()
               for field in ("cpu.max", "memory.max", "memory.swap.max")}}


def create_validate_owned_worker(plan, out, commit, window, holder, token):
    """Only the guarded C46 builder calls this, after its existing idle/cold/lock guards."""
    if os.uname().nodename != "dgx4":
        raise RuntimeError("owned worker dispatch is DGX4-only")
    image = plan.get("image") or ""
    name = worker_name(commit, token)
    expected = build_resources.builder_plan(name, image)
    _worker_scope(window, holder, token)
    directory = Path(out) / "worker-control"
    directory.mkdir(exist_ok=False)
    free = shutil.disk_usage(ROOT).free
    minimum_gib = plan.get("minimum_free_gib", 64)
    if type(minimum_gib) is not int or minimum_gib < 1:
        raise ValueError("published minimum builder free space must be positive GiB")
    minimum = minimum_gib * 1024**3
    record = {"owner": OWNER, "owner_token": token, "commit": commit, "window": window, "holder": holder,
              "builder": name, "requested_image": image, "created": False, "passed": False,
              "plan": expected, "free_space": {"free_bytes": free, "required_bytes": minimum}, "commands": []}
    _worker_save(out, record)
    try:
        if free < minimum:
            raise RuntimeError("insufficient free builder storage; no duplicate source/image copy attempted")
        code, text = _worker_command(record, out, ["docker", "buildx", "inspect", name], "absence", True)
        if code == 0:
            raise RuntimeError("unique builder already exists; preserve it")
        error = (directory / record["commands"][-1]["stderr"]).read_text().lower()
        if not any(term in error for term in ("not found", "no builder", "no such")):
            raise RuntimeError("builder absence was not established")
        _worker_scope(window, holder, token)
        _worker_command(record, out, expected["create_argv"], "create")
        record["created"] = True
        _worker_save(out, record)
        _worker_scope(window, holder, token)
        _worker_command(record, out, expected["bootstrap_argv"], "bootstrap", timeout=300)
        _, raw = _worker_command(record, out, ["docker", "inspect", "buildx_buildkit_" + name + "0"], "container-inspect")
        inspection = json.loads(raw)
        record["container_id"] = inspection[0]["Id"]
        _worker_save(out, record)
        _, raw = _worker_command(record, out, ["docker", "image", "inspect", inspection[0]["Image"]], "image-inspect")
        actual_image = json.loads(raw)[0]
        if (actual_image.get("Os") != "linux" or actual_image.get("Architecture") != "arm64"
                or ("@" in image and image not in actual_image.get("RepoDigests", []))):
            raise RuntimeError("actual BuildKit platform or portable repository digest differs")
        record["actual_image"] = {key: actual_image.get(key) for key in ("Id", "RepoDigests", "Os", "Architecture")}
        _, raw = _worker_command(record, out, ["docker", "buildx", "inspect", name, "--format", "{{json .}}"], "buildx-inspect")
        buildx = json.loads(raw)
        _, config = _worker_command(record, out, ["docker", "exec", record["container_id"], "cat", "/etc/buildkit/buildkitd.toml"], "actual-config")
        # Read the actual worker PID's host hierarchy, not /sys/fs/cgroup's
        # namespace root inside a privileged container (which can be the host root).
        cgroup = worker_cgroup(inspection[0].get("State", {}).get("Pid"))
        record["validation"] = build_resources.validate_worker(name, image, buildx, inspection, config.encode(), cgroup)
        _worker_scope(window, holder, token)
        record["passed"] = True
        _worker_save(out, record)
        return record
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        if record["created"]:
            try:
                if not record.get("container_id"):
                    _, raw = _worker_command(record, out, ["docker", "inspect", "buildx_buildkit_" + name + "0"], "failed-create-inspect", True)
                    found = json.loads(raw)[0] if raw.strip() else {}
                    if found.get("Name") == "/buildx_buildkit_" + name + "0" and found.get("Config", {}).get("Image") == image:
                        record["container_id"] = found["Id"]
                if record.get("container_id"):
                    record["cleanup"] = stop_owned_worker(record, out, token)
            except BaseException as cleanup_error:
                record["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        _worker_save(out, record)
        raise


def stop_owned_worker(worker, out, token):
    if (worker.get("owner") != OWNER or worker.get("owner_token") != token
            or not worker.get("created") or worker.get("builder") != worker_name(worker.get("commit", ""), token)
            or not re.fullmatch(r"[0-9a-f]{64}", worker.get("container_id", ""))):
        raise RuntimeError("refusing to stop a worker without this exact C46 creation identity")
    identity = worker["container_id"]
    _, raw = _worker_command(worker, out, ["docker", "inspect", identity], "stop-identity")
    actual = json.loads(raw)[0]
    if (actual.get("Id") != identity or actual.get("Name") != "/buildx_buildkit_" + worker["builder"] + "0"
            or actual.get("Config", {}).get("Image") != worker["requested_image"]):
        raise RuntimeError("worker identity changed; no stop attempted")
    if actual.get("State", {}).get("Running"):
        _worker_command(worker, out, ["docker", "stop", "--time", "20", identity], "stop", timeout=40)
    _, raw = _worker_command(worker, out, ["docker", "inspect", identity], "stopped-inspect")
    after = json.loads(raw)[0]
    if after.get("Id") != identity or after.get("State", {}).get("Running") is not False:
        raise RuntimeError("owned worker did not stop")
    return {"passed": True, "owner_token": token, "builder": worker["builder"], "container_id": identity,
            "cache_state_preserved": True, "global_builder_selection_changed": False}
