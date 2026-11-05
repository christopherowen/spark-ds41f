"""Render or run bounded TP3 source gates inside an already owned lab window."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import re
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
PHASE = "entry"
SOURCE = Path("/home/swank/spark-lab/collective-pdl/sparknet")
PIN = "fb027b4b80e5262416bb6db2b95b671498f6de74"
TREE = "1e1720f0bbeeded1efcb9ce02b535d2981b38ca6"
IMAGE = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
RESTORE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json"
HOLDER_NOTE = "collective-pdl TP3 qualification"
ENV = {"MAX_JOBS":"2", "CMAKE_BUILD_PARALLEL_LEVEL":"2", "OMP_NUM_THREADS":"2",
       "TVM_NUM_THREADS":"2", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1",
       "NCCL_DEBUG":"WARN"}


def configure(phase):
    global SOURCE, PIN, TREE, PHASE
    PHASE = phase
    row = next(row for row in json.loads((HERE/"alternatives.json").read_text())["phases"] if row["phase"] == phase)
    PIN, TREE = row["source_commit"], row["source_tree"]
    SOURCE = Path("/home/swank/spark-lab/collective-pdl")/("sparknet" if phase == "entry" else f"sparknet-{phase}")


def validate_gate(path, map_hash):
    receipt = json.loads(path.read_text())
    expected = {"passed":True, "source_commit":PIN, "source_tree":TREE,
                "image":IMAGE, "map_sha256":map_hash, "mode":"gates"}
    if any(receipt.get(key) != value for key,value in expected.items()):
        raise RuntimeError("the prerequisite gate receipt does not qualify this source/image/map")
    if [job["kind"] for job in receipt["jobs"]] != ["tests","probe"]:
        raise RuntimeError("the prerequisite gate receipt is incomplete")
    for job in receipt["jobs"]:
        outcomes = job["outcomes"]
        if len(outcomes) != 3 or len({row["node"] for row in outcomes}) != 3:
            raise RuntimeError("the prerequisite gate receipt does not cover three distinct ranks")
        for row in outcomes:
            log = path.parent/row["log"]
            if row["exit"] or row["problems"] or hashlib.sha256(log.read_bytes()).hexdigest() != row["sha256"]:
                raise RuntimeError("a prerequisite gate failed or its log changed")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class ManagementFailure(RuntimeError):
    pass


def ssh(node, command, timeout=60):
    try:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                 "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3",
                                 node["name"], shlex.join(command)], text=True,
                                capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ManagementFailure(f"{node['name']}: management command timed out\n{exc.stdout or ''}\n{exc.stderr or ''}") from exc
    if result.returncode == 255:
        raise ManagementFailure(f"{node['name']}: SSH failed: {result.stdout}\n{result.stderr}")
    return result


def plans(nodes, run=1):
    sys.path.insert(0, str(SOURCE))
    from sparknet.cli import rendered_environment
    from sparknet.probe.container import docker_probe_command
    from sparknet.topology import render
    head = next(n for n in nodes["nodes"] if n["head"])
    assert len(nodes["nodes"]) == 3
    result = []
    for node in sorted(nodes["nodes"], key=lambda n: n["rank"]):
        env = rendered_environment(nodes, node, transport="oneshot-direct", profile="tp3-triangle")
        assert not render.environment_problems(env, "oneshot-direct", 3)
        env.update(ENV)
        shared = ["docker", "run", "--rm", "--name=collective-pdl-gpu-test", "--network=host",
                  "--gpus=all", "--memory=16g", "--memory-swap=16g", "--cpus=2", "--pids-limit=512",
                  "--device=/dev/infiniband:/dev/infiniband:rwm", "--cap-add=IPC_LOCK",
                  "--ulimit=memlock=-1:-1", "--shm-size=2g", "--label", f"collective-pdl.source={PIN}"]
        shared += [v for k, val in sorted(env.items()) for v in ("--env", f"{k}={val}")]
        shared += ["--volume", f"{SOURCE}/sparknet:/usr/local/lib/python3.12/dist-packages/sparknet:ro",
                   "--volume", f"{SOURCE}/tests:/sparknet-tests:ro", "--volume", f"{HERE}:/collective:ro",
                   "--entrypoint=/usr/bin/timeout", IMAGE]
        tests = shared + ["--signal=TERM", "--kill-after=15s", "1500s", "python3", "-m",
                          "torch.distributed.run", "--nnodes=3", "--nproc-per-node=1",
                          f"--node-rank={node['rank']}", f"--master-addr={head['management_ip']}",
                          "--master-port=29651", "-m", "pytest", "-x", "-v", "-p", "no:cacheprovider",
                          "/sparknet-tests/gpu/test_oneshot_gpu.py"]
        probe = docker_probe_command(image=IMAGE, environment=env, rank=node["rank"], world_size=3,
                    master_addr=head["management_ip"], master_port=29650, transport="oneshot-direct",
                    package_source=str(SOURCE/"sparknet"), extra_args=("--benchmark", "--counter-samples", "--port-samples"))
        probe[2:2] = ["--cpus=2", "--pids-limit=512", "--label", f"collective-pdl.source={PIN}"]
        benchmark = shared + ["--signal=TERM", "--kill-after=15s", "1200s", "python3", "-m",
                    "torch.distributed.run", "--nnodes=3", "--nproc-per-node=1",
                    f"--node-rank={node['rank']}", f"--master-addr={head['management_ip']}",
                    "--master-port=29652", "/collective/benchmark_launch_chain.py",
                    "--variant",PHASE,"--source-commit",PIN,"--run",str(run)]
        result.append({"node": node, "tests": tests, "probe": probe, "benchmark":benchmark})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deployment-root", type=Path, default=Path("/home/swank/projects/spark-ds41f"))
    p.add_argument("--map", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=("entry","staged","arrived"), default="entry")
    p.add_argument("--mode", choices=("gates","benchmark"), default="gates")
    p.add_argument("--run", type=int, choices=(1,2), default=1)
    p.add_argument("--gates-receipt", type=Path)
    p.add_argument("--window")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    configure(args.phase)
    sys.path.insert(0, str(SOURCE))
    from sparknet import topology
    nodes = topology.load(args.map)
    work = plans(nodes, args.run)
    map_hash = hashlib.sha256(args.map.read_bytes()).hexdigest()
    if not args.apply:
        print(json.dumps({"plan_only":True, "source_commit":PIN, "source_tree":TREE,
                          "map_sha256":map_hash, "mode":args.mode, "phase":PHASE, "plans":work}, indent=2))
        return
    if not args.window:
        p.error("--apply requires the exact owned --window")
    sys.path.insert(0, str(args.deployment_root))
    import spark_ds41f as spark
    from spark_ds41f import lab
    lab.PRODUCTION_CONFIG = RESTORE
    def owned():
        hold = lab.read_hold()
        if not hold or hold.get("window") != args.window or HOLDER_NOTE not in hold.get("holder", ""):
            raise RuntimeError("the expected owned window is absent")
        if hold.get("production_config") != RESTORE:
            raise RuntimeError("restore profile differs from the reviewed plan")
        return hold
    owned()
    gate_hash = None
    if args.mode == "benchmark":
        if not args.gates_receipt: p.error("benchmark --apply requires a passing --gates-receipt")
        gate_hash = validate_gate(args.gates_receipt,map_hash)
    harness_commit = subprocess.check_output(["git","-C",str(HERE),"rev-parse","HEAD"],text=True).strip()
    if subprocess.check_output(["git","-C",str(HERE),"status","--porcelain"],text=True).strip():
        raise RuntimeError("publish the clean harness before hardware execution")
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {"started":now(), "source_commit":PIN, "source_tree":TREE, "image":IMAGE,
               "map_sha256":map_hash, "window":args.window, "mode":args.mode, "phase":PHASE,
               "process_run":args.run, "harness_commit":harness_commit, "gates_receipt_sha256":gate_hash,
               "benchmark_sha256":hashlib.sha256((HERE/"benchmark_launch_chain.py").read_bytes()).hexdigest(),
               "plans":work, "jobs":[], "passed":False}
    def save():
        (args.out/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    halt = threading.Event()
    heartbeat_errors = []
    current = ["source gate preflight"]
    def heartbeat():
        while not halt.wait(30):
            try:
                owned(); lab.beat(current[0])
            except BaseException as exc:
                heartbeat_errors.append(str(exc)); return
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    fan = None
    watch = None
    guard_errors = []
    management_failure = False
    def guard(reason):
        guard_errors.append(str(reason))
    def cleanup():
        errors = []
        for node in nodes["nodes"]:
            try:
                found = ssh(node, ["docker", "ps", "-aq", "--filter", f"label=collective-pdl.source={PIN}"])
                if found.returncode: raise RuntimeError(f"{node['name']}: test container inventory failed")
                if found.stdout.strip():
                    cleaned = ssh(node, ["docker", "rm", "-f", *found.stdout.split()], timeout=40)
                    if cleaned.returncode: raise RuntimeError(f"{node['name']}: test container cleanup failed")
            except Exception as exc: errors.append(str(exc))
        return errors
    try:
        for node in nodes["nodes"]:
            check = ssh(node,["git","-C",str(HERE.parents[1]),"rev-parse","HEAD"])
            assert check.returncode == 0 and check.stdout.strip() == harness_commit, check
            check = ssh(node,["git","-C",str(HERE.parents[1]),"status","--porcelain"])
            assert check.returncode == 0 and not check.stdout.strip(), check
            check = ssh(node, ["git", "-C", str(SOURCE), "rev-parse", "HEAD", "HEAD^{tree}"])
            assert check.returncode == 0 and check.stdout.splitlines() == [PIN,TREE], check
            check = ssh(node, ["git", "-C", str(SOURCE), "status", "--porcelain"])
            assert check.returncode == 0 and not check.stdout.strip(), check
            check = ssh(node, ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"])
            assert check.returncode == 0 and check.stdout.strip() == IMAGE, check
            check = ssh(node, ["docker", "ps", "-q", "--filter", "name=spark-ds41f"])
            assert check.returncode == 0 and not check.stdout.strip(), "serving still runs"
        from spark_ds41f.bench import MemoryWatch
        class RecordedWatch(MemoryWatch):
            def __init__(self,*args):
                self.samples=[]
                super().__init__(*args)
            def _memory(self,name,text):
                super()._memory(name,text)
                with self.lock:
                    self.samples.append({"at":now(),"node":name,"kind":"memory","available_swap_kib":text})
            def _temperature(self,name,sample):
                super()._temperature(name,sample)
                with self.lock:
                    self.samples.append({"at":now(),"node":name,"kind":"thermal",**sample})
        watch = RecordedWatch(nodes,24,guard)
        watch.wait_ready()
        if guard_errors: raise RuntimeError(guard_errors)
        with watch.lock:
            if any(row["min_available_kib"] < 40*1048576 for row in watch.state.values()):
                raise RuntimeError("less than 40 GiB available before the capped 16 GiB test")
        receipt["cooling"] = spark.cool_nodes(nodes, nodes["nodes"], 55, 600)
        fan = spark.start_fan_curve(nodes, nodes["nodes"])
        assert not spark.fan_curve_problems(fan), fan
        for kind in (("tests", "probe") if args.mode == "gates" else ("benchmark",)):
            hold = owned()
            if heartbeat_errors or guard_errors: raise RuntimeError(heartbeat_errors+guard_errors)
            watch.check()
            reason = lab.window_should_close(hold)
            if reason: raise RuntimeError(f"yielding before {kind}: {reason}")
            current[0] = f"collective PDL source {kind}"
            lab.beat(current[0])
            began = time.monotonic()
            def run(item):
                command = ["flock", "--nonblock", "/tmp/spark-lab-gpu.lock", *item[kind]]
                path = args.out/f"{kind}-{item['node']['name']}.log"
                problems = []
                try:
                    result = ssh(item["node"], command, timeout={"tests":1560,"probe":660,"benchmark":1260}[kind])
                    output = result.stdout+result.stderr
                    code = result.returncode
                except subprocess.TimeoutExpired as exc:
                    output = str(exc.stdout or "")+str(exc.stderr or "")+"\n"+str(exc)
                    code = 124
                except Exception as exc:
                    output = f"{type(exc).__name__}: {exc}"
                    code = 255
                path.write_text(output)
                detail = {}
                if kind == "tests":
                    passed = [name.split("::")[-1] for name in re.findall(r"(test_[^\s]+) PASSED", output)]
                    detail["passed_test_ids"] = passed
                    if len([x for x in passed if x.startswith("test_pdl_")]) != 5:
                        problems.append("not all five PDL cases passed")
                    for name in ("test_fail_stop_on_timeout_eager", "test_fail_stop_on_timeout_graph_replay"):
                        if name not in passed: problems.append(f"{name} has no pass")
                elif kind == "benchmark" and code == 0:
                    final = [line.removeprefix("BENCH_RESULT ") for line in output.splitlines() if line.startswith("BENCH_RESULT ")]
                    if len(final) != 1:
                        problems.append("no unique benchmark result")
                    else:
                        data = json.loads(final[0])
                        detail["benchmark_result"] = data
                        expected = {"passed":True,"source_commit":PIN,"variant":PHASE,"run":args.run,
                                    "rank":item["node"]["rank"],"world_size":3}
                        if any(data.get(key) != value for key,value in expected.items()):
                            problems.append("benchmark identity differs")
                        matrix = {(r,o,b) for r in (1,6,9,16,24,48,96) for o in ("all_reduce","all_gather","mixed") for b in (0,4,32)}
                        cases = data.get("cases",[])
                        if len(cases) != len(matrix) or {(r["rows"],r["operation"],r["background_mib"]) for r in cases} != matrix:
                            problems.append("benchmark matrix incomplete")
                        if any(row.get("exact") is not True or len(row.get("rank_samples_us",[])) != 3
                               or any(len(samples) != 40 for samples in row.get("rank_samples_us",[])) for row in cases):
                            problems.append("benchmark output or samples incomplete")
                elif kind == "probe" and code == 0:
                    from sparknet.probe.fleet import _result_from_log
                    data = _result_from_log(path)
                    detail["probe_result"] = data
                    if not data or data.get("passed") is not True or data.get("rank") != item["node"]["rank"]:
                        problems.append("no successful result for the expected rank")
                    else:
                        if data.get("world_size") != 3 or data.get("transport") != "oneshot-direct":
                            problems.append("probe topology identity differs")
                        timings = data.get("timings", [])
                        if not timings: problems.append("no benchmark cases")
                        for row in timings:
                            if row.get("expected_backend") == "oneshot":
                                counters = row.get("proxy_payload_bytes", {})
                                if not counters or any(value <= 0 for value in counters.values()):
                                    problems.append("a one-shot lane posted no payload")
                        def leaves(value):
                            if isinstance(value, dict):
                                for child in value.values(): yield from leaves(child)
                            elif isinstance(value, (list,tuple)):
                                for child in value: yield from leaves(child)
                            else: yield value
                        if any(value != 0 for row in timings for value in leaves(row.get("rdma_error_deltas", {}))):
                            problems.append("RDMA error counter changed")
                if code: problems.append(f"exit {code}")
                return {"node":item["node"]["name"], "exit":code, "problems":problems, **detail,
                        "log":path.name, "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(run,item) for item in work]
                killed = False
                while not all(future.done() for future in futures):
                    try: watch.check()
                    except Exception as exc:
                        management_failure = True
                        guard(str(exc))
                    failed_rank = any(future.done() and (future.exception() or future.result()["exit"] or future.result()["problems"]) for future in futures)
                    if failed_rank and not killed: guard("a rank failed; stopping the remaining test ranks")
                    if (guard_errors or heartbeat_errors) and not killed:
                        receipt.setdefault("cleanup_errors",[]).extend(cleanup())
                        killed = True
                    time.sleep(1)
                outcomes = [future.result() for future in futures]
            if any(row["exit"] == 255 for row in outcomes): management_failure = True
            receipt["guard_errors"] = guard_errors
            if guard_errors or heartbeat_errors:
                receipt["jobs"].append({"kind":kind,"seconds":time.monotonic()-began,"outcomes":outcomes})
                raise RuntimeError(guard_errors+heartbeat_errors)
            receipt["jobs"].append({"kind":kind, "seconds":time.monotonic()-began, "outcomes":outcomes})
            save()
            if any(x["exit"] or x["problems"] for x in outcomes):
                raise RuntimeError(f"{kind} failed; retain all rank logs")
            owned()
            watch.check()
            if heartbeat_errors or guard_errors: raise RuntimeError(heartbeat_errors+guard_errors)
        receipt["passed"] = True
    except BaseException as exc:
        if isinstance(exc, ManagementFailure) or "telemetry" in str(exc): management_failure = True
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Never restore service if management or cleanup is uncertain.
        receipt.setdefault("cleanup_errors",[]).extend(cleanup())
        if watch is not None:
            receipt["thermal"] = watch.thermal_summary()
            receipt["memory"] = watch.close()
            conditions = args.out/"conditions.json"
            conditions.write_text(json.dumps(watch.samples,indent=2)+"\n")
            receipt["conditions"]={"path":conditions.name,"sha256":hashlib.sha256(conditions.read_bytes()).hexdigest()}
        if management_failure:
            receipt.setdefault("cleanup_errors",[]).append("management telemetry or command failed; no new hardware launch")
        if fan is not None:
            try:
                fan_errors = spark.stop_fan_curve(nodes, nodes["nodes"], fan)
                if fan_errors: raise RuntimeError(fan_errors)
            except Exception as exc: receipt.setdefault("cleanup_errors",[]).append(str(exc))
        halt.set(); thread.join(timeout=35)
        receipt["ended"] = now()
        save()
        if receipt.get("cleanup_errors"):
            receipt["passed"] = False
            receipt["restored"] = False
            receipt["attention"] = "Cleanup could not be verified. Hold retained; watchdog remains active. Do not retry hardware work."
            save()
            raise RuntimeError(receipt["attention"])
        if not receipt["passed"]:
            owned()
            receipt["restored"] = lab.window_close("collective source gates ended without a pass")
            save()
        else:
            owned(); lab.beat(f"{PHASE} {args.mode} passed; awaiting reviewed next step")
            print(f"{PHASE} {args.mode} passed. Window retained for the next step; watchdog remains active.", flush=True)

if __name__ == "__main__":
    main()
