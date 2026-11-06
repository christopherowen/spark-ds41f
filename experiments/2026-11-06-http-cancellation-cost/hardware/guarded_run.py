"""Run one published HTTP experiment arm under the repository's bench guards."""

import argparse
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from spark_ds41f import bench
from spark_ds41f.config import configuration, cluster_config_path, run_ssh

IMAGE = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
HOLDER = "http-cancellation-hardware"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-config", required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--arm", choices=("parent", "candidate"), required=True)
    parser.add_argument("--boot", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    cluster, nodes, _ = configuration(args)
    config_path, config_name = cluster_config_path(args)
    hardware = Path(__file__).resolve().parent
    output = args.output.resolve()
    client_name = "http-cancellation-client-" + args.boot
    argv = [
        "docker",
        "run",
        "--rm",
        "--name",
        client_name,
        "--runtime=runc",
        "--network=host",
        "--cpuset-cpus=0-3",
        "--cpus=2",
        "--memory=4g",
        "--memory-swap=4g",
        "--pids-limit=256",
        "-e",
        "CUDA_VISIBLE_DEVICES=",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=none",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-v",
        f"{hardware}:/work:ro",
        "-v",
        f"{output}:/results",
        "--entrypoint",
        "/usr/bin/timeout",
        IMAGE,
        "2400",
        "/usr/bin/python3",
        "/work/http-cancellation-hardware-probe.py",
        "--base",
        "http://127.0.0.1:18001",
        "--arm",
        args.arm,
        "--boot",
        args.boot,
        "--trials",
        str(args.trials),
        "--output",
        "/results/data",
    ]
    if not args.apply:
        print(
            json.dumps(
                {
                    "config": config_name,
                    "nodes": [n["name"] for n in nodes["nodes"]],
                    "cool_below_c": 55,
                    "memory_floor_gib": cluster["host"]["memguard_gib"] + 1,
                    "client": argv,
                },
                indent=2,
            )
        )
        return 0
    hold_path = Path.home() / "spark-hold.json"

    def hold():
        value = json.loads(hold_path.read_text())
        if value.get("holder") != HOLDER or value.get("window") != args.window:
            raise RuntimeError("this test does not own the window")
        if value.get("restoring") or value.get("management_loss_latched"):
            raise RuntimeError("restoration or management-loss review has started")
        return value

    hold()
    if os.uname().nodename != "dgx1":
        raise RuntimeError("guarded serving client must run on dgx1")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise RuntimeError("deployment checkout must be clean")
    output.mkdir(parents=True, exist_ok=False)
    expected = json.loads((hardware / "identity.json").read_text())
    identity = bench.bench_identity(cluster, nodes, config_name, config_path)
    if identity["live_problems"]:
        raise RuntimeError(str(identity["live_problems"]))
    actual = {}
    for node in nodes["nodes"]:
        image = run_ssh(
            nodes,
            node,
            "docker",
            "inspect",
            cluster["container"]["name"],
            "--format",
            "{{.Image}}",
        ).stdout.strip()
        if image != IMAGE:
            raise RuntimeError(
                f"{node['name']} image differs from the immutable control"
            )
        paths = (
            "/opt/spark-ds41f/candidate/vllm/vllm/entrypoints/serve/utils/api_utils.py",
            "/opt/spark-ds41f/candidate/vllm/vllm/entrypoints/serve/middleware/http_cancellation_deadline.py",
        )
        hashes = []
        for path in paths:
            value = run_ssh(
                nodes,
                node,
                "docker",
                "exec",
                cluster["container"]["name"],
                "sha256sum",
                path,
            )
            if value.returncode:
                raise RuntimeError(f"{node['name']} cannot read mounted source")
            hashes.append(value.stdout.split()[0])
        if hashes != [expected[args.arm]["api_sha256"], expected["middleware_sha256"]]:
            raise RuntimeError(f"{node['name']} source overlay differs")
        actual[node["name"]] = {"image": image, "overlay_sha256": hashes}
    (output / "identity.json").write_text(
        json.dumps(
            {
                "native": identity,
                "actual": actual,
                "deployment_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "runner_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "client": argv,
            },
            indent=2,
        )
        + "\n"
    )
    conditions = {"cooling": None, "fan_curve": [], "halted": []}
    process = None
    done = threading.Event()

    def halt(reason):
        if conditions["halted"]:
            return
        conditions["halted"].append(reason)
        if (
            "memory telemetry stopped reporting" in reason
            or "memory telemetry did not start" in reason
        ):
            with hold_path.with_suffix(".http.lock").open("a") as edit:
                fcntl.flock(edit, fcntl.LOCK_EX)
                value = hold()
                value["management_loss_latched"] = True
                temporary = hold_path.with_suffix(".http-cancellation.tmp")
                temporary.write_text(json.dumps(value, indent=2) + "\n")
                temporary.replace(hold_path)
        if process is not None and process.poll() is None:
            subprocess.run(
                ["docker", "kill", "--signal=SIGINT", client_name], capture_output=True
            )

    def heartbeat():
        while not done.wait(30):
            try:
                with hold_path.with_suffix(".http.lock").open("a") as edit:
                    fcntl.flock(edit, fcntl.LOCK_EX)
                    value = hold()
                    value["heartbeat"] = datetime.now(timezone.utc).isoformat()
                    temporary = hold_path.with_suffix(".http-cancellation.tmp")
                    temporary.write_text(json.dumps(value, indent=2) + "\n")
                    hold()
                    temporary.replace(hold_path)
            except BaseException as error:
                halt(f"window heartbeat failed: {error}")
                return

    heart = threading.Thread(target=heartbeat, daemon=True)
    heart.start()
    memory = bench.MemoryWatch(nodes, float(cluster["host"]["memguard_gib"]) + 1, halt)
    code = 1
    try:
        conditions["cooling"] = bench.cool_nodes(
            nodes,
            nodes["nodes"],
            55,
            600,
            fan_control=bench.fan_control_enabled(cluster),
        )
        conditions["fan_curve"] = bench.start_fan_curve(
            nodes, nodes["nodes"], fan_control=bench.fan_control_enabled(cluster)
        )
        problems = bench.fan_curve_problems(conditions["fan_curve"])
        if problems:
            raise RuntimeError("fan curve refused: " + "; ".join(problems))
        memory.wait_ready()
        hold()
        with (output / "client.log").open("w") as log:
            process = subprocess.Popen(
                argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            while process.poll() is None:
                if conditions["halted"]:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        subprocess.run(
                            ["docker", "kill", client_name], capture_output=True
                        )
                    break
                time.sleep(1)
            code = process.wait()
        if conditions["halted"]:
            code = 1
    except BaseException as error:
        conditions["error"] = repr(error)
        halt(repr(error))
        raise
    finally:
        done.set()
        heart.join(timeout=2)
        conditions["fan_stop_failures"] = bench.stop_fan_curve(
            nodes, nodes["nodes"], conditions["fan_curve"]
        )
        conditions["thermal"] = memory.thermal_summary()
        conditions["memory"] = memory.close()
        conditions["exit_code"] = code
        (output / "conditions.json").write_text(json.dumps(conditions, indent=2) + "\n")
    if conditions["fan_stop_failures"] or any(
        v.get("thermal_slowdown_ms") for v in conditions["thermal"].values()
    ):
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
