#!/usr/bin/env python3
"""Fill spark-ds41f-1025's baseline manifest from candidate 6's built image, once its window
(experiments/2026-10-25-spark-ds41f-candidate/rc6/candidate-tp4.json) has passed. Run from dgx1
in a checkout of the promotion branch, or pass the manifest path. Reads each node's image ID and
the image's creation time and revision label; refuses unless every node holds one image ID.

usage: fill_baseline.py [MANIFEST] [--image spark-ds41f:19f2c20ed4d6-next-6]
"""
import argparse
import datetime as dt
import json
import shlex
import subprocess
from pathlib import Path

NODES = ("dgx1", "dgx2", "dgx3", "dgx4")


def inspect(host: str, image: str, template: str) -> str:
    # ssh joins its arguments into one remote command line: quote each.
    remote = " ".join(shlex.quote(a) for a in ("docker", "image", "inspect", image, "--format", template))
    out = subprocess.run(["ssh", "-n", host, remote], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="manifests/baselines/2026-10-25-spark-ds41f-1025.json")
    parser.add_argument("--image", default="spark-ds41f:19f2c20ed4d6-next-6")
    args = parser.parse_args()
    path = Path(args.manifest)
    manifest = json.loads(path.read_text())
    ids = {host: inspect(host, args.image, "{{.Id}}") for host in NODES}
    if len(set(ids.values())) != 1:
        raise SystemExit(f"image IDs differ across nodes: {ids}")
    image_id = ids["dgx1"]
    created = inspect("dgx1", args.image, "{{.Created}}")
    revision = inspect("dgx1", args.image, '{{index .Config.Labels "org.opencontainers.image.revision"}}')
    tree = inspect("dgx1", args.image, '{{index .Config.Labels "local.spark-ds41f.vllm.tree"}}')
    if tree != manifest["source_identity"]["vllm_tree"]:
        raise SystemExit(f"image vLLM tree {tree} != manifest {manifest['source_identity']['vllm_tree']}")
    built = dt.datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    manifest["captured_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest["image_id"] = image_id
    for node in manifest["nodes"].values():
        node["image_id"] = image_id
    manifest["source_identity"]["image_build_deployment_commit"] = revision
    manifest["build"]["image_id"] = image_id
    manifest["build"]["built_utc"] = built.strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    left = path.read_text().count("PENDING")
    print(f"{path}: image {image_id[:19]}, built {manifest['build']['built_utc']}, revision {revision[:9]}; "
          f"{left} PENDING field(s) left (the evidence summary is written by hand)")


if __name__ == "__main__":
    main()
