"""Supervise the owner-requested cooling interval and unchanged TP3 workload.

Run locally with TMPDIR=/tmp after taking the owned hold and saving fan controls
and cooldown.json in results/private/tp3-high-fans. Uses the published 6f71c3c
qualification checkout on the nodes; no source or configuration is deployed.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import shlex
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/private/tp3-high-fans"
CONFIG = "experiments/2026-10-03-tp3-revalidation/control.json"
REMOTE = "/home/swank/projects/spark3-ring4-qualification"
HOLDER = "tp3-high-fans"
REVISION = "6f71c3c8ffe5c9dd7867a0116f7bcd4a301c55d1"


def now():
    return datetime.now(timezone.utc).isoformat()


def ssh(host, command):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "swank@" + host, command]


def heartbeat():
    code = (
        "import pathlib,json,datetime;"
        "p=pathlib.Path.home()/'spark3-hold.json';d=json.loads(p.read_text());"
        f"assert d['holder']=={HOLDER!r};"
        "d['heartbeat']=datetime.datetime.now(datetime.timezone.utc).isoformat();"
        "p.write_text(json.dumps(d,indent=2))"
    )
    subprocess.run(ssh("dgx1", shlex.join(["python3", "-c", code])),
                   check=True, timeout=15)


def observations(stage):
    def one(host):
        result = subprocess.run(ssh(host,
            "dgx-fan-control status; systemctl is-active dgx-fan-control; "
            "cat /sys/class/hwmon/hwmon*/fan*_input 2>/dev/null; "
            "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader"),
            capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError(f"telemetry failed on {host}: {result.stderr}")
        if "state=12/12" not in result.stdout:
            raise RuntimeError(f"maximum fan state lost on {host}: {result.stdout}")
        return {"node": host, "output": result.stdout}
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(one, ["dgx1", "dgx2", "dgx3"]))
    with (RAW / "fans.jsonl").open("a") as file:
        file.write(json.dumps({"utc": now(), "stage": stage, "nodes": records}) + "\n")
    return records


def command(name, arguments, timeout):
    argv = ssh("dgx1", "cd " + shlex.quote(REMOTE) + " && " + shlex.join(arguments))
    receipt = {"argv": argv, "started_utc": now()}
    deadline = time.monotonic() + timeout
    with (RAW / (name + ".log")).open("x") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while True:
                try:
                    result = process.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    heartbeat()
                    observations(name)
                    if time.monotonic() > deadline:
                        raise TimeoutError(name)
        except BaseException:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
            raise
        finally:
            receipt.update(finished_utc=now(), exit_code=process.poll())
            (RAW / (name + "-command.json")).write_text(json.dumps(receipt, indent=2))
    print(f"{name}: exit {result}", flush=True)
    return result


def main():
    cli = runpy.run_path(str(ROOT / "bin/spark3"))
    nodes = json.loads((ROOT / "config/nodes-tp3.local.json").read_text())
    controls = json.loads((RAW / "fan-controls.json").read_text())
    cooldown = json.loads((RAW / "cooldown.json").read_text())
    start_time = datetime.fromisoformat(cooldown["all_fans_high_since"])
    deadline = time.monotonic() + max(0, 1200 - (datetime.now(timezone.utc) - start_time).total_seconds())
    result = 1
    restore_errors = []
    launched = False
    try:
        heartbeat()
        for node in nodes["nodes"]:
            p = subprocess.run(ssh(node["name"], f"git -C {REMOTE} rev-parse HEAD; docker ps -q"),
                               text=True, capture_output=True, check=True, timeout=15)
            assert p.stdout.strip() == REVISION, (node["name"], p.stdout)
        while time.monotonic() < deadline:
            heartbeat()
            snapshot = observations("cooldown")
            remaining = max(0, deadline - time.monotonic())
            print(f"cooldown: {remaining / 60:.1f} min remaining; " +
                  "; ".join(r["node"] + " " + r["output"].splitlines()[0] for r in snapshot), flush=True)
            time.sleep(min(30, remaining))
        (RAW / "cooldown-complete.json").write_text(json.dumps({
            "utc": now(), "elapsed_seconds": (datetime.now(timezone.utc) - start_time).total_seconds()}, indent=2))
        launched = True
        assert command("start", ["env", "TMPDIR=/tmp", "bin/spark3", "--cluster-config", CONFIG,
                                 "cluster", "start", "--replace", "--apply"], 3600) == 0
        # Preserve maximum fans through the check. The CLI's ordinary cooling
        # routine would restore automatic control when it owns no fan service.
        cold_deadline = time.monotonic() + 600
        while True:
            temperatures = [cli["hottest_zone_c"](nodes, n) for n in nodes["nodes"]]
            assert all(t is not None for t in temperatures), temperatures
            if max(temperatures) < 55:
                break
            if time.monotonic() >= cold_deadline:
                raise RuntimeError(f"post-start cooling failed: {temperatures}")
            heartbeat()
            observations("post-start cooling")
            time.sleep(20)
        (RAW / "bench-start-temperatures.json").write_text(json.dumps({"utc": now(), "hottest_c": temperatures}))
        result = command("bench", ["env", "TMPDIR=/tmp", "bin/spark3", "--cluster-config", CONFIG,
            "bench", "--suites", "quality,decode,prefill,prefix", "--min-samples", "3", "--max-samples", "3",
            "--seed", "0", "--prefill-text", "source", "--prefill-sizes", "1024,32768,65536,262144",
            "--prefill-repeats", "2", "--cool-below", "0", "--output", "results/private/tp3-high-fans/bench"], 2400)
        subprocess.run(["scp", "-q", "swank@dgx1:" + REMOTE + "/results/private/tp3-high-fans/bench/bench.json",
                        str(RAW / "bench.json")], check=True, timeout=30)
    finally:
        if launched:
            try:
                if command("stop", ["env", "TMPDIR=/tmp", "bin/spark3", "--cluster-config", CONFIG,
                                    "cluster", "stop", "--apply", "--parallel"], 180):
                    restore_errors.append("coordinated stop failed")
            except BaseException as error:
                restore_errors.append(str(error))
        for node in nodes["nodes"]:
            try:
                message = cli["restore_fan_control"](nodes, node, controls[node["name"]])
                if message and message.startswith("FAILED"):
                    restore_errors.append(node["name"] + ": " + message)
            except BaseException as error:
                restore_errors.append(node["name"] + ": " + str(error))
        (RAW / "restoration.json").write_text(json.dumps({"utc": now(), "errors": restore_errors}, indent=2))
        if not restore_errors:
            code = ("import pathlib,json;p=pathlib.Path.home()/'spark3-hold.json';"
                    f"assert json.loads(p.read_text())['holder']=={HOLDER!r};p.unlink()")
            subprocess.run(ssh("dgx1", shlex.join(["python3", "-c", code])), check=True, timeout=15)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
