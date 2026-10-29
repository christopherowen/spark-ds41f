#!/usr/bin/env python3
"""What the headless launcher (`vllm serve --headless`, node rank 1) holds before it spawns its
worker: the production argv (argv.json) through the real CLI, with run_headless replaced by its
own steps up to the executor. After each stage, this process's anonymous memory by mapping kind
and whether the NVIDIA driver is open; then the same after gc.collect() and malloc_trim(0).

Prints one JSON line a stage and a table. Exit 0 when every stage ran.
"""
import ctypes
import gc
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAPPING = re.compile(r"^[0-9a-f]+-[0-9a-f]+$")
STAGES = []


def memory() -> dict:
    heap = small = large = 0
    name = size = None
    with open("/proc/self/smaps") as smaps:
        for line in smaps:
            fields = line.split()
            if MAPPING.match(fields[0]):
                lo, hi = (int(x, 16) for x in fields[0].split("-"))
                size, name = hi - lo, (fields[5] if len(fields) > 5 else "")
            elif fields[0] == "Rss:":
                rss = int(fields[1]) * 1024
                if name == "[heap]":
                    heap += rss
                elif name == "":
                    if size <= 1 << 20:
                        small += rss
                    else:
                        large += rss
    rollup = {}
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            key, _, rest = line.partition(":")
            if key in ("Rss", "Pss_Anon", "Pss_File"):
                rollup[key] = int(rest.split()[0]) * 1024
    fds = [os.readlink(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd")
           if os.path.exists(f"/proc/self/fd/{fd}")]
    nvidia = sorted({p for p in fds if p.startswith("/dev/nvidia")})
    return {"pss_anon": rollup.get("Pss_Anon", 0), "pss_file": rollup.get("Pss_File", 0),
            "heap": heap, "anon_small": small, "anon_large": large, "nvidia": nvidia,
            "threads": len(os.listdir("/proc/self/task")),
            "cuda_loaded": "libcuda.so" in Path("/proc/self/maps").read_text()}


def stage(name: str) -> None:
    record = {"stage": name, **memory()}
    STAGES.append(record)
    print(json.dumps(record), flush=True)


def trimmed(name: str) -> None:
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    stage(name + " +gc+trim")


def headless_until_executor(args) -> None:
    import vllm
    from vllm.usage.usage_lib import UsageContext

    stage("args parsed")
    engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
    stage("engine args")
    vllm_config = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER,
                                                   headless=True)
    stage("engine config")
    trimmed("engine config")
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor  # noqa: F401
    stage("executor imported")
    import torch
    torch.cuda.device_count()
    stage("cuda device_count")
    torch.cuda.get_device_properties(0)
    stage("cuda device properties")
    trimmed("cuda")
    globals()["kept_config"] = vllm_config


def main() -> int:
    stage("python")
    import torch  # noqa: F401
    stage("import torch")
    import vllm.entrypoints.cli.main as cli_main
    import vllm.entrypoints.cli.serve as serve
    stage("import vllm cli")
    serve.run_headless = headless_until_executor
    sys.argv = ["vllm", *json.loads((HERE / "argv.json").read_text())]
    try:
        cli_main.main()
    except SystemExit as exit:
        if exit.code not in (0, None):
            raise
    mib = 1 << 20
    print(f"\n{'stage':34} {'anon':>7} {'heap':>7} {'<=1M':>7} {'>1M':>7} {'file':>7}  cuda nvidia-fds thr")
    for s in STAGES:
        print(f"{s['stage']:34} {s['pss_anon'] / mib:7.0f} {s['heap'] / mib:7.0f} {s['anon_small'] / mib:7.0f} "
              f"{s['anon_large'] / mib:7.0f} {s['pss_file'] / mib:7.0f}  {'yes' if s['cuda_loaded'] else 'no':4} "
              f"{len(s['nvidia']):>11} {s['threads']:3}")
    return 0 if any(s["stage"] == "cuda +gc+trim" for s in STAGES) else 1


if __name__ == "__main__":
    sys.exit(main())
