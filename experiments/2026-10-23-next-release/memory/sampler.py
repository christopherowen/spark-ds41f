#!/usr/bin/env python3
"""Read-only memory sampler for one node (run as root, no CUDA context), for start-up
attribution and long soaks. From experiments/2026-10-21-startup-memory/sampler.py, with the
page-table walks of the GPU workers made rare (they hold the worker's mmap lock).

Writes JSON lines to OUT/<node>-samples.jsonl until STOP exists or --max-seconds pass:
  every 0.1 s  {"k":"m"}  /proc/meminfo (kB) and the serving container's cgroup memory
  every 0.5 s  {"k":"p"}  RSS (kB) of every process in the container cgroup (statm, no walk)
               {"k":"v"}  /proc/vmstat reclaim, swap and compaction counters
  every 1 s    {"k":"g"}  nvidia-smi per-process GPU memory (MiB), from a separate thread
  every --rollup-seconds   {"k":"s"}  smaps_rollup (kB) of the container's host-side processes
                           {"k":"h"}  the largest host processes outside the container
  every --worker-seconds   {"k":"w"}  smaps_rollup (kB) of the GPU workers (VLLM::Worker*)
  every --maps-seconds     {"k":"x"}  the host-side processes' resident memory by mapping kind
                           (/proc/<pid>/smaps): the brk heap, anonymous mappings by size
                           (1 MiB: CPython arenas; 64 MiB-aligned: glibc thread arenas; others:
                           large single allocations), file mappings
Host-side processes are the container's processes other than the GPU workers: the API server
and the engine core, where #439 saw growth. Times are UTC epoch seconds, so they line up with
`docker logs -t`.
"""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

MEMINFO = ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapCached", "Active(anon)",
           "Inactive(anon)", "Active(file)", "Inactive(file)", "Unevictable", "Mlocked", "SwapFree",
           "Dirty", "Writeback", "AnonPages", "Mapped", "Shmem", "KReclaimable", "SUnreclaim",
           "KernelStack", "PageTables", "SecPageTables", "VmallocUsed", "Percpu", "Committed_AS")
CGROUP_STAT = ("anon", "file", "kernel", "kernel_stack", "pagetables", "sec_pagetables", "percpu",
               "sock", "vmalloc", "shmem", "file_mapped", "slab_reclaimable", "slab_unreclaimable")
VMSTAT = ("nr_free_pages", "pgsteal_kswapd", "pgsteal_direct", "pgscan_kswapd", "pgscan_direct",
          "pswpin", "pswpout", "allocstall_normal", "allocstall_movable", "compact_stall",
          "compact_success", "compact_fail", "workingset_refault_file", "workingset_refault_anon",
          "pgfree", "pgalloc_normal", "thp_fault_alloc", "drop_pagecache", "kswapd_low_wmark_hit_quickly",
          "kswapd_high_wmark_hit_quickly")
PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024
CGROUPS = Path("/sys/fs/cgroup/system.slice")


def read_meminfo() -> dict:
    values = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            name, rest = line.split(":", 1)
            if name in MEMINFO:
                values[name] = int(rest.split()[0])
    return values


def container_cgroup() -> Path | None:
    """The docker scope with live processes; the newest when a replacement overlaps."""
    best = None
    for scope in CGROUPS.glob("docker-*.scope"):
        try:
            if not (scope / "cgroup.procs").read_text().strip():
                continue
            mtime = scope.stat().st_mtime
        except OSError:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, scope)
    return None if best is None else best[1]


def read_cgroup(scope: Path | None) -> dict:
    if scope is None:
        return {}
    try:
        values = {"current": int((scope / "memory.current").read_text())}
        for line in (scope / "memory.stat").read_text().splitlines():
            name, value = line.split()
            if name in CGROUP_STAT:
                values[name] = int(value)
        return values
    except (OSError, ValueError):
        return {}


def cgroup_pids(scope: Path | None) -> list[int]:
    if scope is None:
        return []
    try:
        return [int(pid) for pid in (scope / "cgroup.procs").read_text().split()]
    except OSError:
        return []


def comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return "?"


def rss_kb(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * PAGE_KB
    except (OSError, IndexError, ValueError):
        return None


def rollup(pid: int) -> dict | None:
    try:
        text = Path(f"/proc/{pid}/smaps_rollup").read_text()
    except OSError:
        return None
    wanted = ("Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Anonymous", "Swap", "SwapPss")
    values = {}
    for line in text.splitlines()[1:]:
        name, _, rest = line.partition(":")
        if name in wanted:
            values[name] = int(rest.split()[0])
    return values


ARENA = 64 << 20


def maps_by_kind(pid: int) -> dict | None:
    """Resident kB and mapping counts by kind, from one pass over /proc/<pid>/smaps."""
    try:
        text = Path(f"/proc/{pid}/smaps").read_text()
    except OSError:
        return None
    kinds: dict[str, list[int]] = {}
    kind = None
    for line in text.splitlines():
        head = line.split(None, 5)
        if "-" in head[0] and len(head) >= 5 and ":" not in head[0]:
            start, end = (int(x, 16) for x in head[0].split("-"))
            size = end - start
            path = head[5] if len(head) > 5 else ""
            if path == "[heap]":
                kind = "heap"
            elif path.startswith("["):
                kind = "special"
            elif path:
                kind = "file"
            elif head[1].startswith("---"):
                kind = "anon-reserved"
            elif size == 1 << 20:
                kind = "anon-1m"
            elif start % ARENA == 0 and size <= ARENA:
                kind = "anon-arena"
            elif size >= 32 << 20:
                kind = "anon-ge32m"
            elif size > 1 << 20:
                kind = "anon-1m-32m"
            else:
                kind = "anon-lt1m"
            entry = kinds.setdefault(kind, [0, 0])
            entry[1] += 1
        elif line.startswith("Rss:") and kind is not None:
            kinds[kind][0] += int(line.split()[1])
    return {name: {"rss": rss, "n": n} for name, (rss, n) in kinds.items()}


def read_vmstat() -> dict:
    values = {}
    with open("/proc/vmstat") as handle:
        for line in handle:
            name, value = line.split()
            if name in VMSTAT:
                values[name] = int(value)
    return values


def host_top(exclude: set[int], count: int = 12) -> list:
    rows = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in exclude:
            continue
        rss = rss_kb(pid)
        if rss:
            rows.append((rss, pid))
    rows.sort(reverse=True)
    return [[pid, comm(pid), rss] for rss, pid in rows[:count]]


def gpu_thread(emit, stop: threading.Event) -> None:
    while not stop.is_set():
        started = time.time()
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            apps = []
            for line in result.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit():
                    apps.append([int(parts[0]), comm(int(parts[0])), int(parts[1]) if parts[1].isdigit() else None])
            emit({"k": "g", "t": started, "apps": apps, "rc": result.returncode})
        except (OSError, subprocess.SubprocessError) as error:
            emit({"k": "g", "t": started, "error": str(error)})
        stop.wait(max(0.0, 1.0 - (time.time() - started)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--stop", required=True, help="stop once this file exists")
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--rollup-seconds", type=float, default=1.0)
    parser.add_argument("--worker-seconds", type=float, default=30.0)
    parser.add_argument("--maps-seconds", type=float, default=30.0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    handle = open(out / f"{args.node}-samples.jsonl", "a", buffering=1 << 16)
    lock = threading.Lock()

    def emit(record: dict) -> None:
        with lock:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    stop = threading.Event()
    if not args.no_gpu:
        threading.Thread(target=gpu_thread, args=(emit, stop), daemon=True).start()
    emit({"k": "start", "t": time.time(), "node": args.node, "page_kb": PAGE_KB, "pid": os.getpid()})
    deadline = time.time() + args.max_seconds
    scope, scope_checked, tick, seen, last = None, 0.0, 0, {}, {}
    while time.time() < deadline and not os.path.exists(args.stop):
        now = time.time()
        if now - scope_checked >= 1.0:
            current = container_cgroup()
            if current != scope:
                emit({"k": "cg", "t": now, "scope": None if current is None else current.name})
                scope = current
            scope_checked = now
        emit({"k": "m", "t": now, "mi": read_meminfo(), "cg": read_cgroup(scope)})
        if tick % 5 == 0:
            pids = cgroup_pids(scope)
            rows = []
            for pid in pids:
                if pid not in seen:
                    try:
                        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")[:300].decode(errors="replace")
                        stat = Path(f"/proc/{pid}/stat").read_text()
                        ppid = int(stat.rsplit(")", 1)[1].split()[1])
                    except (OSError, ValueError, IndexError):
                        cmd, ppid = "?", None
                    seen[pid] = True
                    emit({"k": "new", "t": now, "pid": pid, "ppid": ppid, "comm": comm(pid), "cmd": cmd})
                rows.append([pid, rss_kb(pid)])
            emit({"k": "p", "t": now, "rss": rows})
            emit({"k": "v", "t": now, "vm": read_vmstat()})
        due = [name for name, every in (("s", args.rollup_seconds), ("w", args.worker_seconds),
                                        ("x", args.maps_seconds)) if now - last.get(name, 0.0) >= every]
        if due:
            pids = cgroup_pids(scope)
            workers = {pid for pid in pids if comm(pid).startswith("VLLM::Worker")}
            host_side = [pid for pid in pids if pid not in workers]
            if "s" in due:
                emit({"k": "s", "t": now, "roll": {str(pid): rollup(pid) for pid in host_side}})
                emit({"k": "h", "t": now, "top": host_top(set(pids))})
            if "w" in due:
                emit({"k": "w", "t": now, "roll": {str(pid): rollup(pid) for pid in workers}})
            if "x" in due:
                big = [pid for pid in host_side if (rss_kb(pid) or 0) > 256 * 1024]
                emit({"k": "x", "t": now, "maps": {str(pid): maps_by_kind(pid) for pid in big}})
            for name in due:
                last[name] = now
        tick += 1
        time.sleep(max(0.0, 0.1 - (time.time() - now)))
    stop.set()
    emit({"k": "end", "t": time.time()})
    handle.close()


if __name__ == "__main__":
    main()
