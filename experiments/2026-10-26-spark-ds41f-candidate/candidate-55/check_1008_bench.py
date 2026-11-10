#!/usr/bin/env python3
"""#1008's timing mark on the serving path: the TP4 ring's per-step 116-collective time under rank-0
API load, from sparknet's own per-op trace on every rank. A lab extra against a served endpoint whose
arm sets SPARKNET_ROCE_TRACE=1.

usage: check_1008_bench.py URL --cluster-config PATH --label LABEL --trace-dir DIR
                           [--streams 16] [--tokenize-clients 2] [--warmup 15] [--seconds 60]
                           [--max-ops 60000] [--max-tokens 512] [--step-collectives 116]
       check_1008_bench.py analyze DIR --seq-start N --seq-end N [--step-collectives 116]
       check_1008_bench.py verdict RUN [--base A,A-again] [--candidate B,B-again] [--stem pr-1008-bench]

Why not the PR's bench as it is: bench_roce_core_affinity.py (copied beside this file) simulates a
proxy, a worker and their load in one process and pins them itself, three ways. It never runs the
arm's sparknet proxy or the arm's placement, so it gives the same numbers on any image. This bench
keeps its unit and its statistics (116 collectives a step; mean, p50, p95, p99, p99.9 and max by the
PR's percentile method) and takes the times from the arm's own serving path.

- **Load (the same in every arm):** --streams closed-loop chat streams against rank 0's API server,
  check_1008_affinity.py's (distinct prompts, temperature 0, thinking off, --max-tokens a request),
  and --tokenize-clients loops of /tokenize on one fixed text of about 118 KB, so rank 0's API
  server and its tokenizer are busy while the ring decodes. The client pins itself to the head's
  CPUs outside the serving container's cpuset, so it is not part of the load on the container's cores.
- **Window:** after --warmup s, rank 0's newest traced sequence opens it. It closes after --seconds,
  or once --max-ops sequences passed (the trace keeps 65,536 records a rank). Every rank's trace is
  then copied at once, and only records inside the window are read.
- **Collective time:** each traced all-reduce's kernel time on its rank's GPU, start to end (stage,
  wait for every peer's payload, reduce), from %globaltimer, so no clock is compared across nodes.
  The ring's time for an op is its slowest rank's. A step is 116 consecutive traced all-reduces (the
  decode step's roce_all_reduce count) and its time is their sum: `ring_step_116_collectives_ms`
  decides the mark (its p99 and mean). Each rank's own steps and ops are beside it.
- **Mechanism:** each rank's proxy wake (doorbell to the proxy seeing it), as excess over its running
  median (1,024 ops) so the CPU-to-GPU clock offset and its drift cancel; the proxy's post time; and
  the involuntary context switches and CPU time of every rank's proxy and worker main thread over
  the window (the PR's nivcsw), with the engine core's and API server's on the head.
- **Placement:** every serving thread's allowed CPUs at the window's start and end, and
  `partitioned` as check_1008_affinity.py defines it (expected false on candidate-54, true on
  candidate-55).

Prints a JSON line per rank, one for the ring and a closing summary; the copied traces go to
--trace-dir. Each rank's live trace leaves /dev/shm once its copy is verified (as
scripts/collect_node_traces.py does), so traces do not pile up across arms. Exits 1 when the
endpoint, a node, a trace or the load fails, 0 otherwise. `analyze` re-reads a --trace-dir.
`verdict` reads the four arms' summaries and applies the interval rule (README, Marks).
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shlex
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

STEP_COLLECTIVES = 116
STEM = "pr-1008-bench"
# sparknet.oneshot.trace, file layout version 1 (identical in candidate-54's and candidate-55's
# sparknet: #1008's 0001 does not touch trace.py). Checked against the image's module at start.
TRACE_LAYOUT = {"VERSION": 1, "HEADER_BYTES": 4096, "RECORD_WORDS": 32, "W_SEQ": 0, "W_START": 1,
                "W_DOORBELL": 2, "W_WAIT_DONE": 3, "W_END": 4, "W_META": 5, "W_PROXY": 22,
                "W_PROXY_SEQ": 26, "PROXY_WORDS": ["seen", "posted", "relayed", "drained"]}
HEADER = struct.Struct("<8sIIIIIIIIqqq64s")
TOPOLOGIES = {0: "direct", 1: "ring4", 2: "mesh4"}
WAKE_MEDIAN_OPS = 1024
# The /tokenize load's one text (about 118 KB), the same in every arm.
_TOKENIZE_WORDS = "harbor engine valley signal garden river bridge island forest letter".split()
TOKENIZE_TEXT = " ".join(f"{_TOKENIZE_WORDS[i % len(_TOKENIZE_WORDS)]} {i}" for i in range(10000))

# Runs on a node's host, pinned to CPUs outside the serving container's cpuset (not in the
# container, so it adds no thread to the serving cores): the serving container's threads from
# /proc, and the sparknet traces in /dev/shm that its live processes own.
# argv: container, mode ("probe", "seq" or "copy:<dir>"), CPUs to run on ("" for any).
PROBE = r"""
import glob, json, os, re, struct, subprocess, sys, time
from array import array
container, mode = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3 and sys.argv[3]:
    allowed = set()
    for part in sys.argv[3].split(','):
        lo, _, hi = part.partition('-')
        allowed.update(range(int(lo), int(hi or lo) + 1))
    allowed &= set(range(os.cpu_count() or 0))
    if allowed:
        os.sched_setaffinity(0, allowed)
HZ = os.sysconf('SC_CLK_TCK')
btime = next(int(l.split()[1]) for l in open('/proc/stat') if l.startswith('btime'))
listing = subprocess.run(['docker', 'top', container, '-eo', 'pid'], capture_output=True, text=True)
if listing.returncode:
    print(json.dumps({'error': 'docker top %s: %s' % (container, listing.stderr.strip()[-200:])}))
    sys.exit(0)
rows, nspid = [], {}
for pid in [int(x) for x in listing.stdout.split()[1:] if x.isdigit()]:
    base = '/proc/%d' % pid
    try:
        pcomm = open(base + '/comm').read().strip()
        cmd = open(base + '/cmdline', 'rb').read().split(b'\0')
        cmd = ' '.join(c.decode(errors='replace') for c in cmd[:4] if c)[:160]
        status = open(base + '/status').read()
        ns = int(next(l.split()[-1] for l in status.splitlines() if l.startswith('NSpid:')))
        stat = open(base + '/stat').read()
        started = btime + int(stat[stat.rindex(')') + 2:].split()[19]) / HZ
        tids = os.listdir(base + '/task')
    except (OSError, StopIteration, ValueError, IndexError):
        continue
    nspid[ns] = (pid, started)
    if mode == 'seq':
        continue
    for tid in tids:
        try:
            task = '%s/task/%s' % (base, tid)
            comm = open(task + '/comm').read().strip()
            fields = {}
            for line in open(task + '/status'):
                key, _, value = line.partition(':')
                if key in ('Cpus_allowed_list', 'voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches'):
                    fields[key] = value.strip()
            stat = open(task + '/stat').read()
            rest = stat[stat.rindex(')') + 2:].split()
        except (OSError, ValueError, IndexError):
            continue
        rows.append([pid, int(tid), pcomm, cmd, comm, fields.get('Cpus_allowed_list', ''), int(rest[36]),
                     int(fields.get('voluntary_ctxt_switches', 0)), int(fields.get('nonvoluntary_ctxt_switches', 0)),
                     int(rest[11]) + int(rest[12])])
H = struct.Struct('<8sIIIIIIIIqqq64s')
traces = []
for path in sorted(glob.glob('/dev/shm/sparknet-trace-r*-p*.bin')):
    match = re.match(r'sparknet-trace-r(\d+)-p(\d+)-([0-9a-f]+)\.bin$', os.path.basename(path))
    if not match:
        continue
    try:
        with open(path, 'rb') as handle:
            header = handle.read(4096)
        size = os.stat(path).st_size
        fields = H.unpack(header[:H.size])
    except (OSError, struct.error):
        continue
    magic, version, world, rank, lanes, topology, records, words, pid, offset, halfwidth, created, host = fields
    owner = nspid.get(pid)
    # A previous boot's worker can have had the same in-container pid: a live trace was created
    # after its owner started.
    live = owner is not None and created / 1e9 >= owner[1] - 2
    entry = {'path': path, 'rank': rank, 'world': world, 'lanes': lanes, 'topology': topology,
             'records': records, 'words': words, 'pid': pid, 'host_pid': owner[0] if owner else None,
             'created_s': created / 1e9, 'size': size, 'live': live, 'magic': magic.decode(errors='replace'),
             'version': version, 'clock_offset_ns': offset, 'clock_halfwidth_ns': halfwidth}
    if live:
        with open(path, 'rb') as handle:
            raw = handle.read()
        data = array('Q')
        data.frombytes(raw[4096:4096 + records * words * 8])
        entry['newest_seq'] = max(data[0::words]) if len(data) else 0
        if mode.startswith('copy:'):
            os.makedirs(mode[5:], exist_ok=True)
            copy = os.path.join(mode[5:], os.path.basename(path))
            with open(copy, 'wb') as handle:
                handle.write(raw)
            entry['copy'] = copy
            entry['copy_size'] = len(raw)
    traces.append(entry)
traces.sort(key=lambda t: t['created_s'])  # a rank's runtimes in creation order
print(json.dumps({'time': time.time(), 'hz': HZ, 'rows': rows, 'traces': traces}))
"""

# The image's trace layout, read once in rank 0's container before the window. trace.py is loaded
# by path: importing sparknet.oneshot would import torch in the serving container.
LAYOUT_PROBE = (
    "import importlib.util, json, os, sys\n"
    "root = list(importlib.util.find_spec('sparknet').submodule_search_locations)[0]\n"
    "spec = importlib.util.spec_from_file_location('sparknet_trace', os.path.join(root, 'oneshot', 'trace.py'))\n"
    "t = importlib.util.module_from_spec(spec)\n"
    "sys.modules['sparknet_trace'] = t\n"
    "spec.loader.exec_module(t)\n"
    "print(json.dumps({'VERSION': t.VERSION, 'HEADER_BYTES': t.HEADER_BYTES, 'RECORD_WORDS': t.RECORD_WORDS,"
    " 'W_SEQ': t.W_SEQ, 'W_START': t.W_START, 'W_DOORBELL': t.W_DOORBELL, 'W_WAIT_DONE': t.W_WAIT_DONE,"
    " 'W_END': t.W_END, 'W_META': t.W_META, 'W_PROXY': t.W_PROXY, 'W_PROXY_SEQ': t.W_PROXY_SEQ,"
    " 'PROXY_WORDS': list(t.PROXY_WORDS)}))\n")


# ---------------------------------------------------------------- statistics (the PR's)

def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _summarize(values: list[float]) -> dict:
    s = sorted(values)
    mean = sum(s) / len(s) if s else 0.0
    return {
        "n": len(s),
        "mean": round(mean, 3),
        "p50": round(_percentile(s, 50.0), 3),
        "p95": round(_percentile(s, 95.0), 3),
        "p99": round(_percentile(s, 99.0), 3),
        "p999": round(_percentile(s, 99.9), 3),
        "max": round(s[-1] if s else 0.0, 3),
    }


def steps(values: list[float], size: int, scale: float) -> list[float]:
    """Sums of ``size`` consecutive values (a step's collectives), times ``scale``; no partial step."""
    return [sum(values[i:i + size]) * scale for i in range(0, len(values) - size + 1, size)]


def cpus(spec: str) -> set[int]:
    out = set()
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def cpu_list(values) -> str:
    ordered, ranges = sorted(values), []
    for cpu in ordered:
        if ranges and cpu == ranges[-1][1] + 1:
            ranges[-1][1] = cpu
        else:
            ranges.append([cpu, cpu])
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)


# ---------------------------------------------------------------- trace analysis

def read_trace(path: Path) -> tuple[dict, dict]:
    """A trace copy's header and its columns (arrays of u64, one entry per record slot)."""
    raw = path.read_bytes()
    magic, version, world, rank, lanes, topology, records, words, pid, offset, halfwidth, created, host = \
        HEADER.unpack(raw[:HEADER.size])
    layout = TRACE_LAYOUT
    if magic != b"SPKTRACE" or version != layout["VERSION"] or words != layout["RECORD_WORDS"]:
        raise ValueError(f"{path}: not a version {layout['VERSION']} sparknet trace")
    data = array("Q")
    data.frombytes(raw[layout["HEADER_BYTES"]:layout["HEADER_BYTES"] + records * words * 8])
    if sys.byteorder != "little":
        data.byteswap()
    proxy = layout["W_PROXY"]
    columns = {name: data[word::words] for name, word in (
        ("seq", layout["W_SEQ"]), ("start", layout["W_START"]), ("doorbell", layout["W_DOORBELL"]),
        ("wait_done", layout["W_WAIT_DONE"]), ("end", layout["W_END"]), ("meta", layout["W_META"]),
        ("seen", proxy), ("posted", proxy + 1), ("relayed", proxy + 2), ("proxy_seq", layout["W_PROXY_SEQ"]))}
    header = {"rank": rank, "world": world, "lanes": lanes, "topology": TOPOLOGIES.get(topology, str(topology)),
              "records": records, "pid": pid, "clock_offset_ns": offset, "clock_halfwidth_ns": halfwidth,
              "created_s": created / 1e9, "host": host.rstrip(b"\0").decode(errors="replace"),
              "sha256": hashlib.sha256(raw).hexdigest()}
    return header, columns


def window_ops(columns: dict, seq_start: int, seq_end: int, offset_ns: int) -> dict[int, dict]:
    """Every complete traced all-reduce with seq_start < seq <= seq_end, by sequence (ns)."""
    ops = {}
    for index, seq in enumerate(columns["seq"]):
        if not seq_start < seq <= seq_end:
            continue
        start, doorbell, wait_done, end = (columns[k][index] for k in ("start", "doorbell", "wait_done", "end"))
        # A slot being rewritten can pair this op's first words with an older op's last ones.
        if not (start and doorbell and end) or not start <= doorbell <= end or end - start > 10_000_000_000:
            continue
        op = {"total": end - start, "staging": doorbell - start, "nbytes": columns["meta"][index] & 0xFFFFFFFF}
        if wait_done and doorbell <= wait_done <= end:
            op["wait"], op["reduce"] = wait_done - doorbell, end - wait_done
        marked = columns["proxy_seq"][index]
        seen, posted, relayed = (columns[k][index] for k in ("seen", "posted", "relayed"))
        if seen and (marked == 0 or marked == seq):
            op["proxy_wake"] = seen + offset_ns - doorbell
            if posted >= seen:
                op["proxy_post"] = posted - seen
                if relayed >= posted:
                    op["proxy_relay"] = relayed - posted
        ops[int(seq)] = op
    return ops


def wake_excess(ordered: list[dict]) -> list[float]:
    """Each op's proxy wake minus the median of its chunk of WAKE_MEDIAN_OPS ops (ns)."""
    out = []
    for i in range(0, len(ordered), WAKE_MEDIAN_OPS):
        wakes = [op["proxy_wake"] for op in ordered[i:i + WAKE_MEDIAN_OPS] if "proxy_wake" in op]
        if not wakes:
            continue
        middle = statistics.median(wakes)
        out.extend(w - middle for w in wakes)
    return out


def analyze(traces: dict[str, Path], seq_start: int, seq_end: int, size: int) -> dict:
    """Per-rank and ring statistics of the window. ``traces`` maps a node name to its copy."""
    ranks, out = {}, {"ranks": {}}
    for node, path in traces.items():
        header, columns = read_trace(path)
        newest = max(columns["seq"]) if len(columns["seq"]) else 0
        # The newest two sequences may still be in flight on this rank.
        ops = window_ops(columns, seq_start, min(seq_end, newest - 2), header["clock_offset_ns"])
        ordered = [ops[s] for s in sorted(ops)]
        totals = [op["total"] for op in ordered]
        sizes: dict[int, int] = {}
        for op in ordered:
            sizes[op["nbytes"]] = sizes.get(op["nbytes"], 0) + 1
        row = {
            "node": node, "rank": header["rank"], "world": header["world"], "topology": header["topology"],
            "ops": len(ordered), "sizes": dict(sorted(sizes.items(), key=lambda kv: -kv[1])[:6]),
            "collective_latency_us": _summarize([t / 1e3 for t in totals]),
            f"step_{size}_collectives_ms": _summarize(steps(totals, size, 1e-6)),
            "phases_us": {name: _summarize([op[name] / 1e3 for op in ordered if name in op])
                          for name in ("staging", "wait", "reduce", "proxy_post", "proxy_relay")},
            "proxy_wakeup_latency_us": _summarize([op["proxy_wake"] / 1e3 for op in ordered if "proxy_wake" in op]),
            "proxy_wakeup_excess_us": _summarize([w / 1e3 for w in wake_excess(ordered)]),
            "trace": {"file": path.name, "sha256": header["sha256"], "pid": header["pid"],
                      "clock_offset_ns": header["clock_offset_ns"],
                      "clock_halfwidth_ns": header["clock_halfwidth_ns"], "newest_seq": int(newest)},
        }
        excess = wake_excess(ordered)
        row["proxy_wakeup_excess_over"] = {f"{limit}us": sum(1 for w in excess if w > limit * 1e3)
                                           for limit in (10, 100, 1000)}
        ranks[header["rank"]] = ops
        out["ranks"][header["rank"]] = row
    common = sorted(set.intersection(*(set(ops) for ops in ranks.values()))) if ranks else []
    slowest_rank = {r: 0 for r in ranks}
    ring = []
    for seq in common:
        rank, total = max(((r, ranks[r][seq]["total"]) for r in ranks), key=lambda item: item[1])
        slowest_rank[rank] += 1
        ring.append(total)
    out["ring"] = {
        "ranks": sorted(ranks), "ops": len(common),
        "collective_latency_us": _summarize([t / 1e3 for t in ring]),
        f"step_{size}_collectives_ms": _summarize(steps(ring, size, 1e-6)),
        "slowest_rank_ops": {str(r): n for r, n in sorted(slowest_rank.items())},
    }
    return out


# ---------------------------------------------------------------- the serving cluster

def probe(spark, nodes: dict, node: dict, container: str, mode: str, outside: str) -> dict:
    """PROBE on a node's host, pinned to the CPUs outside the serving container's cpuset."""
    command = ["python3", "-c", PROBE, container, mode, outside]
    process = subprocess.run(["ssh", *spark.ssh_options(), spark.ssh_target(nodes, node), shlex.join(command)],
                             capture_output=True, text=True, timeout=300)
    if process.returncode:
        return {"error": f"probe exit {process.returncode}: {process.stderr.strip()[-300:]}"}
    try:
        return json.loads(process.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        return {"error": f"probe output: {error}: {process.stdout[-200:]}"}


def probe_all(spark, nodes: dict, container: str, mode: str, outside: str) -> dict[str, dict]:
    """PROBE on every node at once; "{node}" in ``mode`` becomes the node's name."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes["nodes"])) as pool:
        futures = {node["name"]: pool.submit(probe, spark, nodes, node, container,
                                             mode.replace("{node}", node["name"]), outside)
                   for node in nodes["nodes"]}
    return {name: future.result() for name, future in futures.items()}


def live_traces(snapshot: dict) -> list[dict]:
    return [t for t in snapshot.get("traces", []) if t.get("live")]


def roles(snapshot: dict, role) -> dict[str, dict]:
    """Each role's threads: allowed CPUs, CPUs run on, context switches and CPU ticks by tid."""
    out: dict[str, dict] = {}
    for row in snapshot.get("rows", []):
        name = role(row[:5])
        if not name:
            continue
        entry = out.setdefault(name, {"allowed": set(), "ran_on": set(), "tids": {}})
        entry["allowed"].add(row[5])
        entry["ran_on"].add(row[6])
        entry["tids"][row[1]] = {"nvcsw": row[7], "nivcsw": row[8], "ticks": row[9]}
    return out


def role_deltas(before: dict, after: dict, hz: int) -> dict[str, dict]:
    """Over the window, per role: involuntary and voluntary switches and CPU seconds (threads in both)."""
    out = {}
    for name, entry in after.items():
        first = before.get(name, {"tids": {}})["tids"]
        shared = [tid for tid in entry["tids"] if tid in first]
        out[name] = {
            "threads": len(shared),
            "nivcsw": sum(entry["tids"][t]["nivcsw"] - first[t]["nivcsw"] for t in shared),
            "nvcsw": sum(entry["tids"][t]["nvcsw"] - first[t]["nvcsw"] for t in shared),
            "cpu_s": round(sum(entry["tids"][t]["ticks"] - first[t]["ticks"] for t in shared) / hz, 2),
        }
    return out


def placement(entry: dict, head: bool) -> tuple[dict, bool]:
    """Roles' allowed sets and CPUs run on, and check_1008_affinity.py's partition test for a node."""
    sets = {k: set().union(*(cpus(a) for a in v["allowed"])) for k, v in entry.items()}
    proxy, main_thread = sets.get("sparknet-proxy", set()), sets.get("worker-main", set())
    ok = len(proxy) == 1 and len(main_thread) == 1 and not proxy & main_thread
    if head:
        for other in ("engine-core", "api-server"):
            ok = ok and bool(sets.get(other)) and not sets[other] & (proxy | main_thread)
    shown = {k: {"allowed": sorted(v["allowed"]), "ran_on": sorted(v["ran_on"]), "threads": len(v["tids"])}
             for k, v in entry.items()}
    return shown, ok


def container_cpuset(cluster: dict) -> set[int]:
    for argument in cluster["container"].get("docker_run_args", []):
        if argument.startswith("--cpuset-cpus="):
            return cpus(argument.split("=", 1)[1])
    return set()


def chat_loop(affinity, base: str, model: str, index: int, max_tokens: int, timeout: float,
              stop: threading.Event, counts: dict, gaps: list, lock: threading.Lock) -> None:
    turn = 0
    while not stop.is_set():
        word = affinity.WORDS[(index + turn) % len(affinity.WORDS)]
        prompt = (f"Stream {index}, request {turn}: write a long, detailed story about a {word}, "
                  f"in plain prose, with no headings.")
        local: list[float] = []
        try:
            affinity.stream(base, model, prompt, max_tokens, local, timeout)
            with lock:
                counts["requests"] += 1
                gaps.extend(local)
        except (OSError, ValueError) as error:
            with lock:
                counts["errors"] += 1
                counts["last_error"] = f"{type(error).__name__}: {error}"[:300]
            time.sleep(1)
        turn += 1


def tokenize_loop(base: str, model: str, stop: threading.Event, counts: dict, seconds: list,
                  lock: threading.Lock) -> None:
    body = json.dumps({"model": model, "prompt": TOKENIZE_TEXT}).encode()
    while not stop.is_set():
        request = urllib.request.Request(base + "/tokenize", data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                tokens = json.load(response).get("count")
            with lock:
                counts["tokenize_calls"] += 1
                counts["tokenize_tokens"] = tokens
                seconds.append(time.monotonic() - started)
        except urllib.error.HTTPError as error:
            with lock:
                counts["tokenize_errors"] += 1
                counts["tokenize_last_error"] = f"HTTP {error.code}"
            if error.code in (404, 405):
                return
            time.sleep(1)
        except (OSError, ValueError) as error:
            with lock:
                counts["tokenize_errors"] += 1
                counts["tokenize_last_error"] = f"{type(error).__name__}: {error}"[:300]
            time.sleep(1)


def run_live(args: argparse.Namespace) -> int:
    import spark_ds41f as spark
    import check_1008_affinity as affinity

    base = args.url.rstrip("/")
    summary = {"summary": "#1008 serving-path collective bench", "label": args.label, "passed": False,
               "step_collectives": args.step_collectives}
    problems: list[str] = []
    cluster, nodes, _ = spark.configuration(argparse.Namespace(cluster_config=args.cluster_config))
    container = cluster["container"]["name"]
    head = next(node for node in nodes["nodes"] if node["head"])
    serving = container_cpuset(cluster)
    here = set(range(os.cpu_count() or 0))
    outside = (here - serving) if serving and serving < here else set()
    if outside:
        os.sched_setaffinity(0, outside)
    summary["client"] = {"host": socket.gethostname(), "cpus": cpu_list(os.sched_getaffinity(0)),
                         "serving_cpuset": cpu_list(serving)}
    trace_dir = ROOT / args.trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            model = json.load(response)["data"][0]["id"]
    except OSError as error:
        summary["problems"] = [f"endpoint did not answer: {error}"]
        print(json.dumps(summary))
        return 1
    layout = spark.run_ssh(nodes, head, "docker", "exec", container, "python3", "-c", LAYOUT_PROBE)
    try:
        image_layout = json.loads(layout.stdout.strip().splitlines()[-1])
        if image_layout != TRACE_LAYOUT:
            problems.append(f"the image's trace layout {image_layout} is not the bench's {TRACE_LAYOUT}")
    except (json.JSONDecodeError, IndexError):
        problems.append(f"cannot read the image's trace layout: {layout.stderr.strip()[-200:]}")
    before = probe_all(spark, nodes, container, "probe", cpu_list(outside))
    for name, snapshot in before.items():
        if "error" in snapshot:
            problems.append(f"{name}: {snapshot['error']}")
        elif not live_traces(snapshot):
            problems.append(f"{name}: no live sparknet trace in /dev/shm (the arm must set SPARKNET_ROCE_TRACE=1)")
    if problems:
        summary["problems"] = problems
        print(json.dumps(summary))
        return 1

    stop = threading.Event()
    lock = threading.Lock()
    counts = {"requests": 0, "errors": 0, "tokenize_calls": 0, "tokenize_errors": 0}
    gaps: list[float] = []
    tokenize_seconds: list[float] = []
    workers = [threading.Thread(target=chat_loop, daemon=True,
                                args=(affinity, base, model, i, args.max_tokens, args.timeout, stop, counts, gaps, lock))
               for i in range(args.streams)]
    workers += [threading.Thread(target=tokenize_loop, daemon=True,
                                 args=(base, model, stop, counts, tokenize_seconds, lock))
                for _ in range(args.tokenize_clients)]
    for worker in workers:
        worker.start()
    time.sleep(args.warmup)
    with lock:
        gaps.clear()  # chunk gaps of the window only
    start = probe_all(spark, nodes, container, "probe", cpu_list(outside))
    rank0_node = next((name for name, snap in start.items() if any(t["rank"] == 0 for t in live_traces(snap))), None)
    if rank0_node is None:
        stop.set()
        summary["problems"] = ["no rank-0 trace at the window's start"]
        print(json.dumps(summary))
        return 1
    # One sparknet runtime a worker is expected. With more, each rank's runtimes are matched by
    # creation order, and the one that ran the most all-reduces in the window is analysed.
    seq_start = [int(t["newest_seq"]) for t in live_traces(start[rank0_node])]
    opened = time.monotonic()
    while True:
        remaining = args.seconds - (time.monotonic() - opened)
        if remaining <= 0:
            break
        time.sleep(min(2.0, remaining))
        poll = probe(spark, nodes, spark.node_by_name(nodes, rank0_node), container, "seq", cpu_list(outside))
        grown = [t.get("newest_seq", 0) - s for t, s in zip(live_traces(poll), seq_start)]
        if grown and max(grown) >= args.max_ops:
            break
    staging = f"/tmp/check_1008-{args.label}-{os.getpid()}"
    end = probe_all(spark, nodes, container, f"copy:{staging}/{{node}}", cpu_list(outside))
    window_seconds = round(time.monotonic() - opened, 1)
    stop.set()
    for worker in workers:
        worker.join(timeout=args.timeout)
    with lock:
        load = {"streams": args.streams, "tokenize_clients": args.tokenize_clients, "warmup_s": args.warmup,
                **{k: v for k, v in counts.items()},
                "client_chunk_gaps": affinity.percentiles(list(gaps)),
                "tokenize_call_s": _summarize(list(tokenize_seconds))}

    copies: dict[str, list[Path]] = {}
    for node in nodes["nodes"]:
        snapshot = end[node["name"]]
        if "error" in snapshot:
            problems.append(f"{node['name']}: {snapshot['error']}")
            continue
        mine = live_traces(snapshot)
        if not mine:
            problems.append(f"{node['name']}: no live trace at the window's end")
            continue
        for trace in mine:  # in creation order
            local = trace_dir / f"{node['name']}-{Path(trace['path']).name}"
            fetched = subprocess.run(["scp", "-q", *spark.ssh_options(),
                                      f"{spark.transfer_target(nodes, node)}:{trace['copy']}", str(local)],
                                     capture_output=True, text=True, timeout=600)
            if fetched.returncode or not local.exists() or local.stat().st_size != trace["copy_size"]:
                problems.append(f"{node['name']}: copy of {trace['path']} failed: {fetched.stderr.strip()[-200:]}")
                continue
            copies.setdefault(node["name"], []).append(local)
            # As scripts/collect_node_traces.py: the trace leaves /dev/shm once its copy is verified
            # (root's file, so removed from inside the container; the runtime keeps its mapping).
            removed = spark.run_ssh(nodes, node, "docker", "exec", container, "rm", "-f", trace["path"])
            if removed.returncode:
                problems.append(f"{node['name']}: {trace['path']} left in /dev/shm: {removed.stderr.strip()[-200:]}")
    for node in nodes["nodes"]:
        spark.run_ssh(nodes, node, "rm", "-rf", staging)

    # The analysed runtime: the one whose rank-0 sequence advanced most in the window.
    ends = [int(t.get("newest_seq", 0)) - 2 for t in live_traces(end.get(rank0_node, {}))]
    if len(ends) != len(seq_start):
        problems.append(f"rank 0 has {len(ends)} live traces at the window's end, {len(seq_start)} at its start")
    group = max(range(min(len(ends), len(seq_start))), key=lambda g: ends[g] - seq_start[g], default=0)
    first, seq_end = (seq_start[group], ends[group]) if ends and seq_start else (0, 0)
    chosen: dict[str, Path] = {}
    for name, paths in copies.items():
        if len(paths) != len(seq_start):
            problems.append(f"{name}: {len(paths)} live traces, rank 0 has {len(seq_start)}")
        if group < len(paths):
            chosen[name] = paths[group]
    result = analyze(chosen, first, seq_end, args.step_collectives) if chosen else {"ranks": {}, "ring": {}}
    records = next(iter(live_traces(before[head["name"]])), {}).get("records", 65536)
    window = {"seq_start": first, "seq_end": seq_end, "sequences": seq_end - first, "runtimes": len(seq_start),
              "runtime_analysed": group, "seconds": window_seconds, "rank0_node": rank0_node,
              "trace_records": records,
              # Older records of the window were overwritten; the analysis reads the ones still there.
              "ring_wrapped": seq_end - first >= records - 64}
    if {row["rank"] for row in result["ranks"].values()} != set(range(len(nodes["nodes"]))):
        problems.append(f"analysed ranks {sorted(result['ranks'])}, expected 0..{len(nodes['nodes']) - 1}")

    key = f"step_{args.step_collectives}_collectives_ms"
    hz = int(end.get(head["name"], {}).get("hz", 100))
    all_ok = True
    nivcsw = {"sparknet-proxy": {}, "worker-main": {}}
    for node in nodes["nodes"]:
        name = node["name"]
        first, last = roles(start.get(name, {}), affinity.role), roles(end.get(name, {}), affinity.role)
        deltas = role_deltas(first, last, hz)
        shown, ok = placement(last, node["head"])
        all_ok = all_ok and ok
        for role_name in nivcsw:
            nivcsw[role_name][name] = deltas.get(role_name, {}).get("nivcsw")
        rank_row = next((row for row in result["ranks"].values() if row["node"] == name), None)
        line = {"node": name, "head": node["head"], "label": args.label, "roles": shown, "role_window": deltas,
                "partitioned": ok}
        if rank_row:
            line.update(rank_row)
        else:
            problems.append(f"{name}: no trace analysed")
        print(json.dumps(line), flush=True)
    ring = result.get("ring", {})
    print(json.dumps({"ring": ring, "label": args.label, "window": window}), flush=True)

    rank0_row = result["ranks"].get(0, {})
    if ring.get("ops", 0) < args.step_collectives * 100:
        problems.append(f"only {ring.get('ops', 0)} ring ops in the window ({args.step_collectives * 100} needed)")
    if counts["requests"] == 0 or counts["errors"]:
        problems.append(f"load: {counts['requests']} requests, {counts['errors']} errors")
    summary.update(
        passed=not problems, problems=problems, model=model, window=window, partitioned=all_ok,
        **{f"ring_{key}": ring.get(key), "ring_collective_latency_us": ring.get("collective_latency_us"),
           f"rank0_{key}": rank0_row.get(key),
           "rank0_collective_latency_us": rank0_row.get("collective_latency_us"),
           "rank0_proxy_wakeup_excess_us": rank0_row.get("proxy_wakeup_excess_us"),
           "proxy_wakeup_excess_p99_us": {str(r): row["proxy_wakeup_excess_us"]["p99"]
                                          for r, row in sorted(result["ranks"].items())},
           "proxy_nivcsw": nivcsw["sparknet-proxy"], "worker_main_nivcsw": nivcsw["worker-main"],
           "slowest_rank_ops": ring.get("slowest_rank_ops")},
        load=load, trace_dir=args.trace_dir)
    print(json.dumps(summary), flush=True)
    return 0 if summary["passed"] else 1


# ---------------------------------------------------------------- verdict

def last_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    lines = [l for l in path.read_text().splitlines() if l.startswith("{") and '"summary"' in l]
    return json.loads(lines[-1]) if lines else None


def interval(candidate: list[float], base: list[float]) -> str:
    """The rule candidates 49-53 use: lower when the candidate's higher boot is below the base's lower
    boot, higher when its lower boot is above the base's higher boot, the same otherwise."""
    if max(candidate) < min(base):
        return "lower"
    if min(candidate) > max(base):
        return "higher"
    return "same"


def run_verdict(args: argparse.Namespace) -> int:
    directory = ROOT / args.dir / args.run
    labels = {"base": args.base.split(","), "candidate": args.candidate.split(",")}
    found = {arm: [last_summary(directory / f"{args.stem}-{label}.jsonl") for label in names]
             for arm, names in labels.items()}
    problems = [f"{label}: {'no summary' if s is None else '; '.join(s.get('problems') or ['not passed'])}"
                for arm, names in labels.items() for label, s in zip(names, found[arm])
                if s is None or not s.get("passed")]
    out = {"run": args.run, "rule": "lower beyond the interval: the candidate's higher boot below the base's "
                                     "lower boot; worse: the candidate's lower boot above the base's higher boot",
           "labels": labels, "problems": problems}
    if problems:
        out["verdict"] = "invalid"
        print(json.dumps(out, indent=1))
        return 1
    key = f"ring_step_{found['base'][0]['step_collectives']}_collectives_ms"
    rows = {}
    for metric, path in (("ring step p99 ms", (key, "p99")), ("ring step mean ms", (key, "mean")),
                         ("rank-0 step p99 ms", (key.replace("ring_", "rank0_"), "p99")),
                         ("rank-0 step mean ms", (key.replace("ring_", "rank0_"), "mean")),
                         ("ring op p99 us", ("ring_collective_latency_us", "p99")),
                         ("rank-0 proxy wake excess p99 us", ("rank0_proxy_wakeup_excess_us", "p99")),
                         ("rank-0 proxy wake excess max us", ("rank0_proxy_wakeup_excess_us", "max"))):
        values = {arm: [s[path[0]][path[1]] for s in found[arm]] for arm in found}
        rows[metric] = {**values, "repeat_margin": {arm: round(abs(v[0] - v[1]), 3) for arm, v in values.items()},
                        "pairings_pct": [round(100 * (c / b - 1), 2) for b in values["base"] for c in values["candidate"]],
                        "candidate_vs_base": interval(values["candidate"], values["base"])}
    for metric, field in (("client chunk gap p99 ms", "p99_ms"), ("client chunk gap mean ms", "mean_ms")):
        values = {arm: [s["load"]["client_chunk_gaps"].get(field) for s in found[arm]] for arm in found}
        rows[metric] = {**values, "candidate_vs_base": interval(values["candidate"], values["base"])}
    rows["proxy nivcsw (sum over ranks)"] = {arm: [sum(v or 0 for v in s["proxy_nivcsw"].values()) for s in found[arm]]
                                             for arm in found}
    rows["worker main nivcsw (sum over ranks)"] = {arm: [sum(v or 0 for v in s["worker_main_nivcsw"].values())
                                                         for s in found[arm]] for arm in found}
    rows["partitioned"] = {arm: [s["partitioned"] for s in found[arm]] for arm in found}
    p99 = rows["ring step p99 ms"]["candidate_vs_base"]
    mean = rows["ring step mean ms"]["candidate_vs_base"]
    placement_ok = rows["partitioned"]["candidate"] == [True, True] and rows["partitioned"]["base"] == [False, False]
    out.update(rows=rows, decides={"ring step p99": p99, "ring step mean": mean, "placement as declared": placement_ok})
    out["verdict"] = ("meets-mark" if p99 == "lower" and mean != "higher" and placement_ok
                      else "invalid" if not placement_ok else "misses-mark")
    print(json.dumps(out, indent=1))
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "verdict":
        parser = argparse.ArgumentParser(prog="check_1008_bench.py verdict")
        parser.add_argument("run")
        parser.add_argument("--dir", default="results/private/determinism")
        parser.add_argument("--stem", default=STEM)
        parser.add_argument("--base", default="candidate-54,candidate-54-again")
        parser.add_argument("--candidate", default="candidate-55-no978,candidate-55-no978-again")
        return run_verdict(parser.parse_args(argv[1:]))
    if argv and argv[0] == "analyze":
        parser = argparse.ArgumentParser(prog="check_1008_bench.py analyze")
        parser.add_argument("dir")
        parser.add_argument("--seq-start", type=int, required=True)
        parser.add_argument("--seq-end", type=int, required=True)
        parser.add_argument("--step-collectives", type=int, default=STEP_COLLECTIVES)
        args = parser.parse_args(argv[1:])
        files = sorted(Path(args.dir).glob("*.bin"))
        print(json.dumps(analyze({p.stem: p for p in files}, args.seq_start, args.seq_end, args.step_collectives)))
        return 0
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--cluster-config", required=True, help="the arm's profile: its node map and container")
    parser.add_argument("--label", required=True)
    parser.add_argument("--trace-dir", required=True, help="where the traces are copied (repository-relative)")
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--tokenize-clients", type=int, default=2)
    parser.add_argument("--warmup", type=float, default=15.0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--max-ops", type=int, default=60000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--step-collectives", type=int, default=STEP_COLLECTIVES)
    parser.add_argument("--timeout", type=float, default=600.0)
    return run_live(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
