#!/usr/bin/env python3
"""One lone request against a fresh boot; on a hang, every vLLM thread's native stack.

usage: stall_probe.py URL --out DIR [--timeout S] [--cluster-config PATH]

A lab extra (the lab passes the head URL first). Sends one short chat completion
(thinking off) and waits up to --timeout seconds. If it has not finished by then, it
gives the Engram stall watcher time to report, then on every node of the cluster's node
map saves ``sudo gdb -batch`` backtraces of every thread of the serving container's
vLLM engine and worker processes to ``DIR/<node>-<pid>.gdb.txt`` and the serving log
to ``DIR/<node>.log``. Prints one JSON line. Report only: exits 0 whether or not the
request hung.
"""

import argparse
import importlib.machinery
import importlib.util
import json
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_loader = importlib.machinery.SourceFileLoader("spark", str(ROOT / "bin" / "spark"))
_spec = importlib.util.spec_from_loader("spark", _loader)
spark = importlib.util.module_from_spec(_spec)
_loader.exec_module(spark)

GDB = ("set pagination off", "set print thread-events off", "thread apply all bt 40")


def ssh(target: str, command: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", target, command],
                          text=True, capture_output=True, timeout=timeout)


def request(url: str, outcome: dict) -> None:
    started = time.monotonic()
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=30) as response:
            model = json.load(response)["data"][0]["id"]
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "Name three primary colours."}],
            "max_tokens": 64,
            "temperature": 0,
            "chat_template_kwargs": {"thinking": False},
        }).encode()
        req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as response:
            reply = json.load(response)
        outcome["tokens"] = reply["usage"]["completion_tokens"]
    except Exception as error:  # the report names it; a hang is the case of interest
        outcome["error"] = f"{type(error).__name__}: {error}"
    outcome["seconds"] = round(time.monotonic() - started, 3)


def dump_node(nodes: dict, node: dict, container: str, out: Path) -> dict:
    target = spark.ssh_target(nodes, node)
    name = node["name"]
    listing = ssh(target, f"docker top {shlex.quote(container)} -eo pid,args")
    # vLLM retitles its engine and workers (VLLM::EngineCore, VLLM::Worker_TP<n>).
    pids = [line.split(None, 1)[0] for line in listing.stdout.splitlines()[1:]
            if "VLLM::" in line]
    gdb_args = " ".join(f"-ex {shlex.quote(c)}" for c in GDB)
    for pid in pids:
        dump = ssh(target, f"sudo -n timeout 120 gdb -p {pid} -batch {gdb_args} 2>&1", timeout=180)
        (out / f"{name}-{pid}.gdb.txt").write_text(dump.stdout + dump.stderr)
    log = ssh(target, f"docker logs --tail 6000 {shlex.quote(container)} 2>&1")
    (out / f"{name}.log").write_text(log.stdout)
    return {"pids": pids}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="the head URL the lab passes")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--watch-grace", type=float, default=15,
                        help="seconds to let the stall watcher report before the dumps")
    parser.add_argument("--cluster-config", default="config/cluster-tp4.json")
    args = parser.parse_args(argv)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    outcome: dict = {}
    worker = threading.Thread(target=request, args=(args.url, outcome), daemon=True)
    worker.start()
    worker.join(args.timeout)
    summary = {"out": args.out, "hung": worker.is_alive(), "request": outcome}
    if worker.is_alive():
        time.sleep(args.watch_grace)
        cluster, nodes, _ = spark.configuration(argparse.Namespace(cluster_config=args.cluster_config))
        container = cluster["container"]["name"]
        summary["nodes"] = {node["name"]: dump_node(nodes, node, container, out)
                            for node in nodes["nodes"]}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
