#!/usr/bin/env python3
"""Run published contract checks on idle Spaceport sp8 without GPU or serving changes."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import time

EXPERIMENT = "experiments/2026-11-04-tilelang-serving-contract"


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True,
                          timeout=120, **kwargs)


def available_gib():
    line = next(line for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:"))
    return int(line.split()[1]) / 1048576


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.commit):
        raise ValueError("a full published commit is required")
    if socket.gethostname() != "spark-0016":
        raise RuntimeError("this controller is restricted to Spaceport sp8")
    if not args.apply:
        print(json.dumps({"commit": args.commit, "cpu_only": True,
                          "memory_limit_gib": 8, "cpu_limit": 2}))
        return 0
    if (Path.home() / "spark-hold.json").exists():
        raise RuntimeError("Spaceport reservation exists")
    if command(["docker", "ps", "-q"]).stdout.strip():
        raise RuntimeError("sp8 has a running container")
    if available_gib() < 16:
        raise RuntimeError("insufficient host memory for the 8 GiB limit")
    repo = Path.home() / "projects/spark-ds41f"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path.home() / "tmp/serving-contract" / args.commit[:12] / stamp
    root.mkdir(parents=True, exist_ok=False)
    receipt = {"node": socket.gethostname(), "commit": args.commit, "root": str(root),
               "cpu_only": True, "runs": []}
    created = []
    hold = Path.home() / "spark-hold.json"
    holder = "serving-contract-cpu-" + stamp
    held = False
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        reservation = {"holder": holder, "since": now.isoformat(),
                       "heartbeat": now.isoformat(),
                       "expected_end": (now + datetime.timedelta(minutes=10)).isoformat(),
                       "scope": ["sp8"],
                       "rule": "exclusive bounded CPU source checks; no GPU or serving changes"}
        with hold.open("x") as handle:
            json.dump(reservation, handle)
        held = True
        command(["git", "-C", str(repo), "fetch", "--quiet", "origin", args.commit])
        archive = subprocess.Popen(["git", "-C", str(repo), "archive", args.commit,
                                    EXPERIMENT], stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", str(root)], stdin=archive.stdout, check=True)
        archive.stdout.close()
        if archive.wait():
            raise RuntimeError("published export failed")
        inputs = root / EXPERIMENT
        base = json.loads((inputs / "base.json").read_text())
        image = receipt["image"] = base["test_image"]
        tree = command(["docker", "image", "inspect", "--format",
                        '{{index .Config.Labels "local.spark-ds41f.vllm.tree"}}', image]).stdout.strip()
        receipt["image_tree"] = tree
        if tree != base["image_source_tree"]:
            raise ValueError("image source label mismatch")
        export = "contract-export-" + stamp.lower()
        command(["docker", "create", "--name", export, "--entrypoint", "true", image])
        created.append(export)
        generated = inputs / "generated"
        generated.mkdir()
        for name, record in base["files"].items():
            if record["original_sha256"] is None:
                continue
            target = generated / name
            target.parent.mkdir(parents=True, exist_ok=True)
            command(["docker", "cp", export + ":/opt/spark-ds41f/candidate/vllm/" + name,
                     str(target)])
            if sha(target) != record["original_sha256"]:
                raise ValueError("exported image file mismatch: " + name)
        command(["docker", "rm", export])
        created.remove(export)
        for name, expected in base["patches"].items():
            patch = inputs / "vllm" / name
            if sha(patch) != expected:
                raise ValueError("patch mismatch: " + name)
            command(["git", "apply", "--check", str(patch)], cwd=generated)
            command(["git", "apply", str(patch)], cwd=generated)
        for name, record in base["files"].items():
            if sha(generated / name) != record["corrected_sha256"]:
                raise ValueError("patched file mismatch: " + name)
        for mode in ("control", "corrected"):
            output = root / mode
            output.mkdir()
            name = "contract-" + mode + "-" + stamp.lower()
            argv = ["docker", "run", "--name", name, "--network=none",
                    "--memory=8g", "--memory-swap=8g", "--cpus=2",
                    "-e", "CUDA_VISIBLE_DEVICES=-1", "-e", "VLLM_TARGET_DEVICE=cpu",
                    "-e", "OMP_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2",
                    "--mount", f"type=bind,src={inputs},dst=/input,readonly",
                    "--mount", f"type=bind,src={output},dst=/output",
                    "--entrypoint", "python3", image, "/input/run_tests.py", "--mode", mode]
            created.append(name)
            print(json.dumps({"event": "start", "mode": mode, "command": argv}), flush=True)
            start, minimum = time.time(), available_gib()
            with (output / "controller.txt").open("w") as log:
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    minimum = min(minimum, available_gib())
                    if time.time() - start > 360:
                        command(["docker", "kill", name])
                        break
                    time.sleep(1)
                process.wait(timeout=20)
            verdict = output / "verdict.json"
            verdict = json.loads(verdict.read_text()) if verdict.exists() else None
            state = json.loads(command(["docker", "inspect", "--format",
                                        "{{json .State}}", name]).stdout)
            result = {"mode": mode, "command": argv, "exit": process.returncode,
                      "seconds": time.time() - start, "min_available_gib": minimum,
                      "state": state, "verdict": verdict}
            receipt["runs"].append(result)
            command(["docker", "rm", name])
            created.remove(name)
            print(json.dumps({"event": "done", "mode": mode, "exit": result["exit"],
                              "counts": verdict.get("counts") if verdict else None}), flush=True)
    except Exception as error:
        receipt["error"] = str(error)
    finally:
        for name in created:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
        if held and hold.is_file() and json.loads(hold.read_text()).get("holder") == holder:
            hold.unlink()
        request = Path.home() / "spark-request.json"
        if held and request.is_file():
            if json.loads(request.read_text()).get("who") == "tilelang-serving-contract":
                request.unlink()
        receipt["final_running_containers"] = command(
            ["docker", "ps", "--format", "{{.Names}}"]
        ).stdout.splitlines()
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"event": "receipt", "path": str(root / "receipt.json"),
                          "error": receipt.get("error")}), flush=True)
    return 0 if len(receipt["runs"]) == 2 and all(
        run["verdict"] and run["verdict"].get("expected_result_observed")
        for run in receipt["runs"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
