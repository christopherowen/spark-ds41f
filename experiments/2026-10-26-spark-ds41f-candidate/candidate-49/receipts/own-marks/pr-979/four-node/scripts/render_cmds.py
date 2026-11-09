#!/usr/bin/env python3
"""Render one rank's container command for the #979 four-node bench.

The container is the repository's collective-probe container (spark_ds41f.doctor.
collective_probe_command: the release profile's per-node environment, RDMA devices, limits)
for releases/spark-ds41f-1107/cluster.json at tp4-switched (sp1..sp4), without the probe's
NCCL_DEBUG=INFO override, with the shared JIT cache bind (spark.expected_binds), the arm's
sp_prefill-stack files bound read-only over the image's vLLM tree, and the bench script.

usage: render_cmds.py <repo> <workdir> <node> <arm> <topology> <label> <port>
prints the shell-quoted docker command.
"""
import argparse
import shlex
import sys

repo, work, node_name, arm, topology, label, port = sys.argv[1:8]
sys.path.insert(0, repo)
import os  # noqa: E402
os.chdir(repo)
from spark_ds41f import lab  # noqa: E402

spark = lab.spark
cluster, nodes, _ = spark.configuration(argparse.Namespace(
    cluster_config="releases/spark-ds41f-1107/cluster.json"))
assert cluster["container"]["image"] == "spark-ds41f:1107", cluster["container"]["image"]
node = spark.node_by_name(nodes, node_name)
env = spark.expected_environment(cluster, node)
env.update(RANK=str(node["rank"]), WORLD_SIZE=str(len(nodes["nodes"])), LOCAL_RANK="0",
           MASTER_ADDR=cluster["distributed"]["master_addr"], MASTER_PORT=str(port))
files = [
    "vllm/models/deepseek_v4_1/attention.py",
    "vllm/models/deepseek_v4_1/sp_prefill.py",
    "vllm/models/deepseek_v4_1/tilelang/cache_writer.py",
    "vllm/models/deepseek_v4_1/tilelang/front.py",
    "vllm/models/deepseek_v4_1/tilelang/gemm.py",
    "vllm/models/deepseek_v4_1/tilelang/indexer.py",
    "vllm/models/deepseek_v4_1/tilelang/router.py",
]
cmd = ["docker", "run", "--rm", f"--name=c49om979-{label}-r{node['rank']}",
       "--network=host", "--gpus=all", "--memory=12g", "--memory-swap=12g",
       "--device=/dev/infiniband:/dev/infiniband:rwm", "--cap-add=IPC_LOCK",
       "--ulimit=memlock=-1:-1", "--ulimit=core=0", "--shm-size=256m",
       "--entrypoint=/usr/bin/timeout"]
for key, value in sorted(env.items()):
    cmd += ["--env", f"{key}={value}"]
cache = [b for b in spark.expected_binds(cluster) if b.endswith(":/cache:rw")]
assert len(cache) == 1, spark.expected_binds(cluster)
cmd += ["--volume", cache[0]]
for f in files:
    cmd += ["--volume", f"{work}/ov/{arm}/{f}:/opt/spark-ds41f/candidate/vllm/{f}:ro"]
cmd += ["--volume", f"{work}/bench_979_integrated.py:/bench.py:ro",
        cluster["container"]["image"], "--signal=TERM", "--kill-after=10s", "900s",
        "python3", "/bench.py", "--topology", topology, "--arm", arm, "--label", label]
print(shlex.join(cmd))
