#!/usr/bin/env python3
"""Bounded CPU checks on Spaceport sp8; no serving or GPU operation."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time


EXPERIMENT = "experiments/2026-11-04-request-submission-cleanup"


def command(args, **kwargs):
    return subprocess.run(
        args, check=True, text=True, capture_output=True, timeout=120, **kwargs
    )


def available_gib():
    return int(next(
        line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )) / 1048576


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--repo", type=Path, default=Path.home() / "projects/spark-ds41f")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{40}", args.commit):
        raise ValueError("a full published commit is required")
    if socket.gethostname() != "spark-0016":
        raise RuntimeError("this CPU test controller is restricted to Spaceport sp8")
    if not args.apply:
        print(json.dumps({"commit": args.commit, "node": socket.gethostname(),
                          "cpu_only": True, "modes": ["control", "corrected"],
                          "memory_limit_gib": 8, "cpu_limit": 2}))
        return 0
    hold = Path.home() / "spark-hold.json"
    if hold.exists():
        raise RuntimeError(f"Spaceport reservation exists: {hold}")
    if command(["docker", "ps", "-q"]).stdout.strip():
        raise RuntimeError("sp8 already has a running container")
    if available_gib() < 16:
        raise RuntimeError("insufficient headroom for an 8 GiB CPU container")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path.home() / "tmp/request-submission" / args.commit[:12] / stamp
    root.mkdir(parents=True, exist_ok=False)
    receipt = {"node": socket.gethostname(), "commit": args.commit, "root": str(root),
               "cpu_only": True, "runs": []}
    created = []
    try:
        # Fetch/export a published commit without changing the site's checkout.
        command(["git", "-C", str(args.repo), "fetch", "--quiet", "origin", args.commit])
        archive = subprocess.Popen(
            ["git", "-C", str(args.repo), "archive", args.commit, EXPERIMENT],
            stdout=subprocess.PIPE,
        )
        subprocess.run(["tar", "-x", "-C", str(root)], stdin=archive.stdout, check=True)
        archive.stdout.close()
        if archive.wait():
            raise RuntimeError("published input export failed")
        inputs = root / EXPERIMENT
        base = json.loads((inputs / "base.json").read_text())
        image = receipt["image"] = base["test_image"]
        tree = command(["docker", "image", "inspect", "--format",
                        '{{index .Config.Labels "local.spark-ds41f.vllm.tree"}}', image]).stdout.strip()
        receipt["image_tree"] = tree
        if tree != base["source_base_tree"]:
            raise RuntimeError("image source label mismatch")
        # The runtime image has no Git client. Generate the exact patched files
        # outside it from its original file and the published Git patch.
        export_name = "submission-export-" + stamp.lower()
        command(["docker", "create", "--name", export_name, "--entrypoint", "true", image])
        created.append(export_name)
        source = root / "generated-source"
        target = source / "vllm/v1/engine/async_llm.py"
        target.parent.mkdir(parents=True)
        command(["docker", "cp", export_name + ":/opt/spark-ds41f/candidate/vllm/"
                 "vllm/v1/engine/async_llm.py", str(target)])
        if sha(target) != base["original_async_llm_sha256"]:
            raise RuntimeError("exported original source mismatch")
        patch = inputs / "vllm/0001-rollback-failed-submissions.patch"
        for argv in (["git", "apply", "--check", str(patch)], ["git", "apply", str(patch)]):
            command(argv, cwd=source)
        generated = inputs / "generated"
        generated.mkdir()
        shutil.copyfile(target, generated / "async_llm.py")
        shutil.copyfile(source / "tests/v1/engine/test_request_submission_cleanup.py",
                        generated / "test_request_submission_cleanup.py")
        command(["docker", "rm", export_name])
        created.remove(export_name)
        for mode in ("control", "corrected"):
            output = root / mode
            output.mkdir()
            name = f"submission-{mode}-{stamp.lower()}"
            argv = ["docker", "run", "--name", name, "--network=none",
                    "--memory=8g", "--memory-swap=8g", "--cpus=2",
                    "-e", "CUDA_VISIBLE_DEVICES=-1", "-e", "VLLM_TARGET_DEVICE=cpu",
                    "-e", "OMP_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2",
                    "--mount", f"type=bind,src={inputs},dst=/input,readonly",
                    "--mount", f"type=bind,src={output},dst=/output",
                    "--entrypoint", "python3", image,
                    "/input/run_tests.py", "--mode", mode]
            created.append(name)
            print(json.dumps({"event": "start", "mode": mode, "command": argv}), flush=True)
            start, minimum, timed_out = time.time(), available_gib(), False
            with (output / "controller.log").open("w") as log:
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    minimum = min(minimum, available_gib())
                    if time.time() - start > 240:
                        timed_out = True
                        command(["docker", "kill", name])
                        break
                    time.sleep(1)
                process.wait(timeout=20)
            state = command(["docker", "inspect", "--format", "{{json .State}}", name]).stdout
            verdict = output / f"{mode}.json"
            verdict = json.loads(verdict.read_text()) if verdict.exists() else None
            result = {"mode": mode, "command": argv, "exit": process.returncode,
                      "seconds": time.time()-start, "min_available_gib": minimum,
                      "timed_out": timed_out, "state": json.loads(state), "verdict": verdict}
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
