"""Node I/O shared by the trace harnesses (c8_trace.py, scenario_trace.py, trace_mixes.py).

One multiplexed SSH connection per node, the three nodes in parallel, and a run's logs
fetched as one tar stream per node. The harnesses used to open a new SSH connection and a
`docker exec` for every log file of every node in turn (about 15 files x 3 nodes per run,
some 2,400 calls in a full validation), which was most of a scenario pass's 12 minutes.
"""
import concurrent.futures
import os
import subprocess
import time

NODES = ("dgx1", "dgx2", "dgx3")
CONTAINER = "dsv41-karmic-kraken"
LOGDIR = "/cache/kkref/moe-checksums"
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ControlMaster=auto",
       "-o", f"ControlPath=/tmp/spark3-trace-{os.getuid()}-%C", "-o", "ControlPersist=300"]


def parallel(function, nodes=NODES):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        return list(pool.map(function, nodes))


def in_container(node, script, **kwargs):
    """Run a shell script inside the serving container on one node (no double quotes in it)."""
    return subprocess.run([*SSH, node, f'docker exec {CONTAINER} sh -c "{script}"'], **kwargs)


def on_nodes(script):
    parallel(lambda node: in_container(node, script, check=True, capture_output=True))


def reset_logs():
    """Clear the trace logs and re-arm the watcher, which polls once a second."""
    on_nodes(f"mkdir -p {LOGDIR}; rm -f {LOGDIR}/rank*; touch {LOGDIR}/reset")
    time.sleep(2.5)


def dump_and_wait(timeout=300, marker="rank*-schedule-host-*.json"):
    """Ask every node to dump its trace logs and wait until each has written its host
    schedule, which the dump writes last (after the one-time plan inventory walk)."""
    on_nodes(f"touch {LOGDIR}/dump")
    pending = set(NODES)
    deadline = time.monotonic() + timeout
    while pending:
        if time.monotonic() > deadline:
            raise RuntimeError(f"trace logs were not dumped on {sorted(pending)}")
        time.sleep(1)

        def written(node):
            count = in_container(node, f"ls {LOGDIR}/{marker} 2>/dev/null | wc -l",
                                 capture_output=True, text=True).stdout.strip()
            return node, count not in ("", "0")

        pending -= {node for node, done in parallel(written, sorted(pending)) if done}


def fetch_logs(destinations, prefixes=("rank",)):
    """Copy LOGDIR files starting with PREFIXES into destinations[node], one tar stream per node."""
    pattern = "|".join(prefixes)

    def one(node):
        target = destinations[node]
        os.makedirs(target, exist_ok=True)
        source = subprocess.Popen(
            [*SSH, node, f"docker exec {CONTAINER} sh -c \"cd {LOGDIR} && ls | grep -E '^({pattern})' "
                         f"| tar -cf - -T -\""], stdout=subprocess.PIPE)
        extracted = subprocess.run(["tar", "-xf", "-", "-C", target], stdin=source.stdout)
        source.stdout.close()
        if source.wait() or extracted.returncode:
            raise RuntimeError(f"{node}: could not fetch trace logs")

    parallel(one, list(destinations))
