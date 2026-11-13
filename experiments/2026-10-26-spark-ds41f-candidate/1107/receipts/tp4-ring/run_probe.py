#!/usr/bin/env python3
import json, subprocess, concurrent.futures, sys

nodes = ["dgx1", "dgx2", "dgx3", "dgx4"]
repo = "/home/swank/projects/spark-ds41f"
cfg = "experiments/2026-10-26-spark-ds41f-candidate/1107/tp4.json"

counter_script = """import json, subprocess
from pathlib import Path
hw = {}
for dev in sorted(Path("/sys/class/infiniband").iterdir()):
    p = dev / "ports/1/hw_counters"
    if p.exists():
        hw[dev.name] = {f.name: int(f.read_text().strip()) for f in sorted(p.iterdir()) if f.is_file()}
eth = {}
for dev in sorted(Path("/sys/class/infiniband").iterdir()):
    netdir = dev / "device/net"
    if netdir.exists():
        for nd in sorted(netdir.iterdir()):
            out = subprocess.check_output(["ethtool", "-S", nd.name], text=True)
            vals = {}
            for line in out.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k, v = k.strip(), v.strip()
                    if v.isdigit() and any(w in k for w in ("oob", "out_of_buffer", "discard", "pause", "prio", "err", "drop", "cnp", "retrans", "seq")):
                        vals[k] = int(v)
            eth[nd.name] = vals
print(json.dumps({"hw_counters": hw, "ethtool": eth}))
"""

def get_roce_counters(node):
    cmd = ["ssh", node, "python3", "-"] if node != "dgx1" else ["python3", "-"]
    out = subprocess.check_output(cmd, input=counter_script, text=True)
    return json.loads(out)

before = {n: get_roce_counters(n) for n in nodes}

cmds = {}
for n in nodes:
    out = subprocess.check_output(["bin/spark", "--cluster-config", cfg, "topology", "probe", n], cwd=repo, text=True)
    line = [l for l in out.splitlines() if l.startswith("docker run")][0]
    cmds[n] = line + " --benchmark --counter-samples --numerics"

def run_node(n):
    c = cmds[n]
    full = ["ssh", n, c] if node_is_remote(n) else ["bash", "-lc", c]
    p = subprocess.run(full, capture_output=True, text=True)
    return n, p.returncode, p.stdout, p.stderr

def node_is_remote(n):
    return n != "dgx1"

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futs = [ex.submit(run_node, n) for n in nodes]
    results = [f.result() for f in futs]

after = {n: get_roce_counters(n) for n in nodes}

deltas = {}
for n in nodes:
    deltas[n] = {"hw_counters": {}, "ethtool": {}}
    for dev, vals in after[n]["hw_counters"].items():
        d = {k: vals[k] - before[n]["hw_counters"][dev].get(k, 0) for k in vals if vals[k] - before[n]["hw_counters"][dev].get(k, 0) != 0}
        if d:
            deltas[n]["hw_counters"][dev] = d
    for nd, vals in after[n]["ethtool"].items():
        d = {k: vals[k] - before[n]["ethtool"][nd].get(k, 0) for k in vals if vals[k] - before[n]["ethtool"][nd].get(k, 0) != 0}
        if d:
            deltas[n]["ethtool"][nd] = d

ranks = {}
for n, rc, out, err in results:
    if rc != 0:
        print(f"FAILED on {n} (rc={rc}):\nSTDOUT:\n{out}\nSTDERR:\n{err}", file=sys.stderr)
    parsed = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed.append(json.loads(line))
            except Exception:
                pass
    ranks[n] = {
        "returncode": rc,
        "final": parsed[-1] if parsed else None,
        "timings_stream": parsed[:-1] if len(parsed) > 1 else [],
    }

summary = {
    "profile": cfg,
    "passed": all(r["returncode"] == 0 and r["final"] and r["final"].get("passed") is True for r in ranks.values()),
    "roce_nonzero_deltas": deltas,
    "ranks": ranks,
}
with open("/home/swank/spark-lab/1107-tp4-ring/probe-collectives.json", "w") as f:
    json.dump(summary, f, indent=2)

print("PROBE PASSED:", summary["passed"])
print("ROCE NONZERO DELTAS:", json.dumps(deltas, indent=2))
for n in nodes:
    fin = ranks[n]["final"] or {}
    print(n, "rc=", ranks[n]["returncode"], "passed=", fin.get("passed"), "transport=", fin.get("transport"), "checks=", len(fin.get("checks", [])), "num_checks=", len(fin.get("numerical_checks", [])), "timings=", len(fin.get("timings", [])))
if not summary["passed"]:
    sys.exit(1)
