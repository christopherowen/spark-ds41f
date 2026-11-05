#!/usr/bin/env python3
"""Lab post-boot gate: verify each rank's entire packed cache and open reader FDs."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import spark_ds41f as spark
from spark_ds41f import lab

HOLDER = "spark-lab: owner-authorized lossless memory qualification"
RESTORE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
SNAPSHOT = "/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/" + REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    hold = lab.read_hold()
    if (not hold or hold.get("holder") != HOLDER or hold.get("production_config") != RESTORE
            or not all(re.fullmatch(r"[A-Za-z0-9._-]{1,100}", v) for v in (args.run, args.label))):
        raise SystemExit("outside the exact authorized lossless window; no node operation")
    if lab.requests_in_flight(args.url) != 0:
        raise SystemExit("verification requires an idle candidate")
    config, nodes, _ = spark.configuration(argparse.Namespace(cluster_config=args.config))
    directory = config["environment"].get("SPARK_DS41F_ENGRAM_PACKED_DIR")
    if not directory or len(nodes["nodes"]) != 3:
        raise SystemExit("not the TP3 packed-row candidate")
    doctor = subprocess.run([sys.executable, "bin/spark", "--cluster-config", args.config, "doctor", "--live"],
                            cwd=ROOT, text=True, capture_output=True)
    print(json.dumps({"doctor_exit": doctor.returncode, "stdout": doctor.stdout, "stderr": doctor.stderr}), flush=True)
    if doctor.returncode:
        return 1
    source = Path(__file__).with_name("verify_cache.py").read_text()
    expected_sha = hashlib.sha256(source.encode()).hexdigest()
    # Hash the actual stdin bytes. A separate CPU-only, memory-limited container
    # reads the checkpoint/cache and joins the serving PID namespace only to
    # inspect reader FDs. It cannot charge verification buffers to serving's cap.
    wrapper = ("import sys,hashlib; s=sys.stdin.read(); "
               "g={'__name__':'__main__','__file__':'<published-cache-verifier>',"
               "'SOURCE_SHA256':hashlib.sha256(s.encode()).hexdigest()}; "
               "exec(compile(s,g['__file__'],'exec'),g)")

    def one(node):
        rank = node["rank"]
        inspection = spark.run_ssh(nodes, node, "docker", "image", "inspect", config["container"]["image"],
                                   "--format", "{{.Id}}")
        image = inspection.stdout.strip()
        if inspection.returncode or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise ValueError("candidate image identity missing")
        mounts = []
        for bind in spark.expected_binds(config):
            source_path, target, _ = bind.rsplit(":", 2)
            if target in ("/hub", "/cache"):
                mounts += ["--volume", f"{source_path}:{target}:ro"]
        if len(mounts) != 4:
            raise ValueError("expected exactly the checkpoint hub and cache mounts")
        # The serving root process has IPC_LOCK. /proc FD inspection requires
        # matching its UID and permitted capability set, even for read access.
        remote = ["docker", "run", "--rm", "-i", "--user=0", "--cap-add=IPC_LOCK",
                  "--memory=1g", "--memory-swap=1g",
                  "--network=none", f"--pid=container:{config['container']['name']}",
                  "--env", "CUDA_VISIBLE_DEVICES=", "--env", "NVIDIA_VISIBLE_DEVICES=void",
                  *mounts, "--entrypoint", "python3", image, "-c", wrapper,
                  SNAPSHOT, "--revision", REVISION, "--directory", directory, "--tp", "3", "--rank", str(rank),
                  "--out", f"/tmp/lossless-cache-{args.run}-{args.label}-rank{rank}.json"]
        argv = ["ssh", *spark.ssh_options(), spark.ssh_target(nodes, node), shlex.join(remote)]
        process = subprocess.run(argv, input=source, text=True, capture_output=True, timeout=1800)
        receipts = []
        for line in process.stdout.splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("scope") == "complete packed-byte verification; not I/O performance":
                receipts.append(value)
        passed = (process.returncode == 0 and len(receipts) == 1
                  and receipts[0].get("all_byte_exact") is True
                  and receipts[0].get("source_sha256") == expected_sha
                  and receipts[0].get("model_revision") == REVISION
                  and receipts[0].get("rank") == rank and receipts[0].get("tp") == 3
                  and len(receipts[0].get("files", [])) == 2)
        return {"node": node["name"], "argv": argv, "exit": process.returncode, "passed": passed,
                "raw_stdout": process.stdout, "stderr": process.stderr, "receipts": receipts}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(one, node): node["name"] for node in nodes["nodes"]}
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                result = {"node": futures[future], "passed": False, "error": f"{type(error).__name__}: {error}"}
            results.append(result)
            print(json.dumps(result), flush=True)
    passed = len(results) == 3 and all(r["passed"] for r in results)
    print(json.dumps({"summary": "complete packed cache", "all_ranks_byte_exact": passed,
                      "source_sha256": expected_sha, "label": args.label}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
