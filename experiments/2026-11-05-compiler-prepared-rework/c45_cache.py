"""Read-only C45 serving-cache first-use mapping; no compiler import or launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = "0.1.15_snapshot_35f15411cf3af0fc06a1cb0097215b1372a40464/linux-aarch64"


def mapped_namespace(cluster, host_path):
    container_root = Path(cluster["environment"]["TILELANG_CACHE_DIR"])
    if not container_root.is_absolute():
        raise ValueError("actual TILELANG_CACHE_DIR must be absolute")
    matches = []
    for source, destination, mode in cluster["container"]["mounts"]:
        target = Path(destination)
        if target.is_absolute() and container_root.is_relative_to(target):
            matches.append((len(target.parts), Path(host_path(cluster, source)), target, mode))
    if not matches:
        raise ValueError("compiler cache is not mapped to an explicit host mount")
    matches.sort(key=lambda row: row[0], reverse=True)
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        raise ValueError("ambiguous compiler-cache mount")
    _, host_mount, container_mount, mode = matches[0]
    if mode != "rw" or not host_mount.is_absolute():
        raise ValueError("compiler-cache mount must be an absolute writable host path")
    host_root = host_mount / container_root.relative_to(container_mount)
    return {"container_root": str(container_root), "host_mount": str(host_mount),
            "host_root": str(host_root), "namespace": NAMESPACE,
            "host_namespace": str(host_root / NAMESPACE)}


def probe(profile, image, commit):
    sys.path.insert(0, str(ROOT))
    from spark_ds41f import config
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("passed new-image and published commit identities are mandatory")
    cluster = config.read_cluster(profile)
    if Path(config.repository_path(cluster)).resolve() != ROOT.resolve():
        raise ValueError("serving cache mapping must use this actual clean node checkout")
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()
    if head != commit or dirty:
        raise ValueError("serving first-use probe source differs or is dirty")
    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
    if inspected["Id"] != image or any(inspected["Config"].get("Labels", {}).get(k) != v
            for k, v in cluster["container"]["expected_labels"].items()):
        raise ValueError("distributed image/source differs from the passed new-image authority")
    mapping = mapped_namespace(cluster, config.host_path)
    mapping.update(namespace_absent=not os.path.lexists(mapping["host_namespace"]),
                   mount_present=Path(mapping["host_mount"]).is_dir())
    return {"passed": mapping["namespace_absent"] and mapping["mount_present"], "mapping": mapping,
            "image": image, "commit": head, "profile": profile,
            "profile_sha256": hashlib.sha256((ROOT / profile).read_bytes()).hexdigest(),
            "operation": "read-only metadata; no compiler import/cache deletion/directory creation"}


def collect(management, require_owned, profile, image, commit, output):
    result = {"passed": False, "namespace": NAMESPACE, "ranks": {}}
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve prior serving first-use evidence: " + str(output))
    try:
        for node in ("dgx1", "dgx2", "dgx3"):
            require_owned()
            command = ["python3", str(Path(__file__).relative_to(ROOT)), "--profile", profile,
                       "--image", image, "--commit", commit]
            try:
                raw = management.ssh(node, ["sh", "-c", "cd /home/swank/projects/spark-ds41f && "
                    + __import__("shlex").join(command)], "C45-before-first-serving-import", timeout=30)
            except subprocess.CalledProcessError as error:
                result["ranks"][node] = {"exit": error.returncode, "stdout": error.output, "stderr": error.stderr}
                raise
            result["ranks"][node] = json.loads(raw)
            if result["ranks"][node].get("passed") is not True:
                raise RuntimeError(node + ": C45 serving namespace is not absent; preserve and review")
        result["passed"] = True
        return result
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        with output.open("x") as stream:
            json.dump(result, stream, indent=2); stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        result = probe(args.profile, args.image, args.commit)
    except BaseException as error:
        result = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
