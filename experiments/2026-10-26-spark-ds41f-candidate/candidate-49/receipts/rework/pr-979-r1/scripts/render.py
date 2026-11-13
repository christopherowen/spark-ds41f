#!/usr/bin/env python3
"""Render one rank's container command for the #979 rework bench.

The own-mark run's rendered command for that node (templates/ring4-c49-boot1-<node>.sh,
from scripts/render_cmds.py of the own-mark: the repository's collective-probe container for
releases/spark-ds41f-1107/cluster.json at tp4-switched, the shared JIT cache bind, the arm's
7 vLLM files bound read-only, the bench) with only what varies between runs substituted:
container name, MASTER_PORT, the arm's overlay directory, --topology/--arm/--label, and this
workdir's copy of the same bench file. Each substitution must match exactly as often as
expected, so nothing else can change.

usage: render.py <workdir> <node> <arm> <topology> <label> <port>
"""
import sys

work, node, arm, topo, label, port = sys.argv[1:7]
OWN = "/home/chris/spark-lab/agents/candidate-49-own-marks-979"
rank = {"sp1": 0, "sp2": 1, "sp3": 2, "sp4": 3}[node]
cmd = open(f"{work}/templates/ring4-c49-boot1-{node}.sh").read().strip()
subs = [
    (f"--name=c49om979-ring4-c49-boot1-r{rank}", f"--name=c49rw979-{label}-r{rank}", 1),
    ("MASTER_PORT=29631", f"MASTER_PORT={port}", 1),
    (f"{OWN}/ov/c49/", f"{work}/ov/{arm}/", 7),
    (f"{OWN}/bench_979_integrated.py:/bench.py:ro", f"{work}/bench_979_integrated.py:/bench.py:ro", 1),
    ("--topology ring4 --arm c49 --label ring4-c49-boot1", f"--topology {topo} --arm {arm} --label {label}", 1),
]
for old, new, n in subs:
    assert cmd.count(old) == n, (old, cmd.count(old))
    cmd = cmd.replace(old, new)
assert OWN not in cmd
print(cmd)
