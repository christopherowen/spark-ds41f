import json, subprocess, threading, time, sys

profile = sys.argv[1]
boot_label = sys.argv[2]
out_path = sys.argv[3]
nodes = ["dgx1", "dgx2", "dgx3"]

samples = {n: [] for n in nodes}
stop_flag = False

def read_mem(node):
    cmd = ["ssh", node, "awk \"/MemTotal:|MemAvailable:|SwapTotal:|SwapFree:/ {print \\$1, \\$2}\" /proc/meminfo"] if node != "dgx1" else ["awk", "/MemTotal:|MemAvailable:|SwapTotal:|SwapFree:/ {print $1, $2}", "/proc/meminfo"]
    out = subprocess.check_output(cmd, text=True, timeout=10)
    d = {}
    for line in out.strip().splitlines():
        k, v = line.split()
        d[k.rstrip(":")] = int(v) / (1024 * 1024) # GiB
    return d

def sampler():
    while not stop_flag:
        ts = time.time()
        for n in nodes:
            try:
                m = read_mem(n)
                m["ts"] = ts
                samples[n].append(m)
            except Exception:
                pass
        time.sleep(2.0)

t = threading.Thread(target=sampler, daemon=True)
t.start()

t0 = time.time()
proc = subprocess.run(
    ["bin/spark", "--cluster-config", profile, "cluster", "start", "--replace", "--apply"],
    cwd="/home/swank/projects/spark-ds41f",
    capture_output=True,
    text=True,
)
elapsed = time.time() - t0
time.sleep(4.0)
stop_flag = True
t.join(timeout=5.0)

steady = {n: read_mem(n) for n in nodes}
summary = {
    "boot_label": boot_label,
    "profile": profile,
    "returncode": proc.returncode,
    "wall_seconds": round(elapsed, 2),
    "stdout": proc.stdout,
    "stderr": proc.stderr,
    "nodes": {},
}
for n in nodes:
    avail = [s["MemAvailable"] for s in samples[n] if "MemAvailable" in s]
    swap_used = [s.get("SwapTotal", 0.0) - s.get("SwapFree", 0.0) for s in samples[n] if "SwapTotal" in s]
    summary["nodes"][n] = {
        "startup_min_mem_available_gib": round(min(avail), 2) if avail else None,
        "steady_mem_available_gib": round(steady[n]["MemAvailable"], 2),
        "max_swap_used_gib": round(max(swap_used), 3) if swap_used else None,
        "sample_count": len(avail),
    }

with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print(proc.stdout)
if proc.stderr:
    print(proc.stderr, file=sys.stderr)
print("TELEMETRY SUMMARY:", json.dumps(summary["nodes"], indent=2))
sys.exit(proc.returncode)
