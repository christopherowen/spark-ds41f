#!/usr/bin/env python3
"""What the serving container did at startup, from its log: the startup work a cleanup
removes, and how long preparation and engine initialization took.

usage: startup_lines.py BASE_URL [--container NAME]

Reads `docker logs` of the head node's serving container (the extras run there). One JSON
line: counts of B12X preparation lines, temporary KV-block lines, JIT monitor
activations and reports, Triton compiles during inference, and the TileLang preparation and
engine init durations. BASE_URL is unused (every extra takes it).
"""
import argparse
import json
import re
import subprocess

PATTERNS = {
    "b12x_preparation": r"b12x \w+ preparation uses",
    "b12x_temporary_kv_blocks": r"b12x autotuning uses \d+ temporary KV blocks",
    "b12x_progress": r"b12x (warming|ready) ",
    "jit_monitor_activated": r"Kernel JIT monitor activated",
    "jit_compile_reports": r"JIT compilation during inference",
}
DURATIONS = {
    "tilelang_prepare_s": r"Prepared (\d+) TileLang calls and the registry's other kernels in ([\d.]+) s",
    "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--container", default="spark-ds41f")
    args = parser.parse_args()
    log = subprocess.run(["docker", "logs", args.container], capture_output=True, text=True,
                         check=True)
    text = log.stdout + log.stderr
    record = {name: len(re.findall(pattern, text)) for name, pattern in PATTERNS.items()}
    prepared = re.search(DURATIONS["tilelang_prepare_s"], text)
    if prepared:
        record["tilelang_calls"] = int(prepared.group(1))
        record["tilelang_prepare_s"] = float(prepared.group(2))
    init = re.search(DURATIONS["init_engine_s"], text)
    record["init_engine_s"] = float(init.group(1)) if init else None
    print(json.dumps({"summary": "startup lines", **record}), flush=True)


if __name__ == "__main__":
    main()
