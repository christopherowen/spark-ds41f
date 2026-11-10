#!/usr/bin/env python3
"""#1008 live check: where every rank's serving threads may run, and where they ran, sampled on the
served cluster under rank-0 API load. A lab extra against a served endpoint.

usage: check_1008_affinity.py URL --cluster-config PATH [--streams 16] [--seconds 120] [--samples 6]
                              [--max-tokens 512]

- **Load:** --streams closed-loop chat streams against URL for --seconds (distinct prompts, temperature
  0, thinking off, --max-tokens a request). They keep rank 0's API server, tokenizer and engine core
  busy while the ring decodes. Each stream's gaps between content chunks are recorded on the client.
- **Placement:** --samples snapshots, spread over the load after a 15 s warm-up, on every node of the
  profile (ssh, then docker exec into its serving container). Each snapshot lists every thread of
  every process in the container: its name, its process, its allowed CPUs (Cpus_allowed_list) and the
  CPU it last ran on (/proc/<pid>/task/<tid>/stat, field 39).
- **Roles:** the sparknet-proxy thread; the worker process's main thread (tid = pid); the engine core
  and API server processes (rank 0). Every other thread is listed by name.

Prints one JSON line per node and snapshot, then one line per node with each role's allowed sets and
the CPUs it ran on, then a summary. `partitioned` is true when, on every node, the proxy and the
worker main thread each may run on one CPU of its own, and on the head the engine core's and the API
server's sets are disjoint from both (#1008's plan: proxy 19, worker main 18, engine core 17, API
server 15-16, the other worker threads 5-9, in a 5-9,15-19 container). The base runs the shared
big-core regime, so it reports false by design; the placement is evidence for the mark, not a pass.
Exits 1 when the endpoint, a node or the load fails, 0 otherwise. Imports spark_ds41f from this
checkout for the node map; otherwise stdlib only.
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import spark_ds41f as spark  # noqa: E402

WORDS = ("harbor engine valley signal garden river bridge island forest letter season planet market "
         "summer window paper music teacher history morning").split()
# Runs in the serving container: every thread of every process, as JSON.
PROBE = r"""
import json, os
rows = []
for pid in filter(str.isdigit, os.listdir('/proc')):
    try:
        pcomm = open(f'/proc/{pid}/comm').read().strip()
        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0')
        cmd = ' '.join(c.decode(errors='replace') for c in cmd[:4] if c)[:160]
        tids = os.listdir(f'/proc/{pid}/task')
    except OSError:
        continue
    for tid in tids:
        try:
            base = f'/proc/{pid}/task/{tid}'
            comm = open(base + '/comm').read().strip()
            allowed = next(l.split(':', 1)[1].strip() for l in open(base + '/status') if l.startswith('Cpus_allowed_list'))
            stat = open(base + '/stat').read()
            last = int(stat[stat.rindex(')') + 2:].split()[36])
        except (OSError, StopIteration, ValueError, IndexError):
            continue
        rows.append([int(pid), int(tid), pcomm, cmd, comm, allowed, last])
print(json.dumps(rows))
"""


def cpus(spec: str) -> set[int]:
    out = set()
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def role(row: list) -> str | None:
    pid, tid, pcomm, cmd, comm = row[:5]
    if comm == "sparknet-proxy":
        return "sparknet-proxy"
    name = f"{pcomm} {cmd}"
    if "EngineCore" in name:
        return "engine-core" if pid == tid else None
    if "Worker" in name:
        return "worker-main" if pid == tid else None
    if "APIServer" in name or (" serve" in cmd and "vllm" in cmd):
        return "api-server"
    return None


def stream(base: str, model: str, prompt: str, max_tokens: int, gaps: list, timeout: float) -> int:
    body = {"model": model, "stream": True, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"thinking": False}}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    pieces, last = 0, None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            choices = json.loads(line[6:]).get("choices") or [{}]
            if (choices[0].get("delta") or {}).get("content"):
                now = time.monotonic()
                if last is not None:
                    gaps.append(now - last)
                last, pieces = now, pieces + 1
    return pieces


def percentiles(values: list[float]) -> dict:
    if len(values) < 2:
        return {"n": len(values)}
    q = statistics.quantiles(values, n=100, method="inclusive")
    return {"n": len(values), "mean_ms": round(1e3 * statistics.fmean(values), 3), "p50_ms": round(1e3 * q[49], 3),
            "p90_ms": round(1e3 * q[89], 3), "p99_ms": round(1e3 * q[98], 3), "max_ms": round(1e3 * max(values), 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--cluster-config", required=True, help="the arm's profile: its node map and container name")
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    summary = {"summary": "#1008 thread placement under rank-0 API load", "passed": False}
    cluster, nodes, _ = spark.configuration(argparse.Namespace(cluster_config=args.cluster_config))
    container = cluster["container"]["name"]
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            model = json.load(response)["data"][0]["id"]
    except OSError as error:
        summary["error"] = f"endpoint did not answer: {error}"
        print(json.dumps(summary))
        return 1

    deadline = time.monotonic() + args.seconds
    gaps: list[float] = []
    counts = {"requests": 0, "errors": 0}
    lock = threading.Lock()

    def loop(index: int) -> None:
        turn = 0
        while time.monotonic() < deadline:
            word = WORDS[(index + turn) % len(WORDS)]
            prompt = (f"Stream {index}, request {turn}: write a long, detailed story about a {word}, "
                      f"in plain prose, with no headings.")
            local: list[float] = []
            try:
                stream(base, model, prompt, args.max_tokens, local, args.timeout)
                with lock:
                    counts["requests"] += 1
                    gaps.extend(local)
            except (OSError, ValueError) as error:
                with lock:
                    counts["errors"] += 1
                    counts["last_error"] = f"{type(error).__name__}: {error}"[:300]
            turn += 1

    workers = [threading.Thread(target=loop, args=(i,), daemon=True) for i in range(args.streams)]
    for worker in workers:
        worker.start()
    warmup = min(15.0, args.seconds / 4)
    time.sleep(warmup)
    interval = max(1.0, (args.seconds - warmup - 5) / max(1, args.samples))
    observed: dict[str, dict] = {node["name"]: {"cpuset": set(), "roles": {}, "threads": {}} for node in nodes["nodes"]}
    problems: list[str] = []
    for sample in range(args.samples):
        for node in nodes["nodes"]:
            process = spark.run_ssh(nodes, node, "docker", "exec", container, "python3", "-c", PROBE)
            if process.returncode:
                problems.append(f"{node['name']}: docker exec {container} exit {process.returncode}: "
                                f"{process.stderr.strip()[-200:]}")
                continue
            rows = json.loads(process.stdout)
            entry = observed[node["name"]]
            for row in rows:
                entry["cpuset"] |= cpus(row[5])
                name = role(row)
                key = name or row[4]
                slot = (entry["roles"] if name else entry["threads"]).setdefault(key, {"allowed": set(), "ran_on": set(), "tids": set()})
                slot["allowed"].add(row[5])
                slot["ran_on"].add(row[6])
                slot["tids"].add(row[1])
            print(json.dumps({"node": node["name"], "sample": sample, "head": node["head"], "threads": len(rows),
                              "roles": {k: sorted(v["allowed"]) for k, v in entry["roles"].items()}}), flush=True)
        time.sleep(interval)
    for worker in workers:
        worker.join(timeout=args.timeout)

    partitioned = True
    for node in nodes["nodes"]:
        entry = observed[node["name"]]
        roles = {k: {"allowed": sorted(v["allowed"]), "ran_on": sorted(v["ran_on"]), "threads": len(v["tids"])}
                 for k, v in entry["roles"].items()}
        sets = {k: set().union(*(cpus(a) for a in v["allowed"])) for k, v in entry["roles"].items()}
        proxy, main_thread = sets.get("sparknet-proxy", set()), sets.get("worker-main", set())
        node_ok = len(proxy) == 1 and len(main_thread) == 1 and not proxy & main_thread
        if node["head"]:
            for other in ("engine-core", "api-server"):
                node_ok = node_ok and bool(sets.get(other)) and not sets[other] & (proxy | main_thread)
        partitioned = partitioned and node_ok
        others = {k: sorted(v["allowed"]) for k, v in sorted(entry["threads"].items())}
        print(json.dumps({"node": node["name"], "head": node["head"],
                          "container_cpus": spark_cpu_list(entry["cpuset"]), "roles": roles,
                          "other_threads_allowed": others, "partitioned": node_ok}), flush=True)
    if counts["requests"] == 0:
        problems.append("the load completed no request")
    summary.update(passed=not problems, problems=problems, partitioned=partitioned, model=model,
                   load={"streams": args.streams, "seconds": args.seconds, **counts,
                         "client_chunk_gaps": percentiles(gaps)})
    print(json.dumps(summary), flush=True)
    return 0 if summary["passed"] else 1


def spark_cpu_list(values: set[int]) -> str:
    ordered, ranges = sorted(values), []
    for cpu in ordered:
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)


if __name__ == "__main__":
    sys.exit(main())
