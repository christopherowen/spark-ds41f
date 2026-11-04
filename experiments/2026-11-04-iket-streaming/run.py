#!/usr/bin/env python3
"""Run one Python workload with bounded-memory TileLang phase collection."""

import argparse
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
parser.add_argument("--ring-mib", type=int, default=8)
parser.add_argument("--chunk-mib", type=int, default=64)
parser.add_argument("--ctas", type=int, default=1)
parser.add_argument("--iters", type=int, default=1)
parser.add_argument("--warps", type=int, default=1)
parser.add_argument("--export", action="store_true")
parser.add_argument("--uniform-registers", action="store_true")
split = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
args = parser.parse_args(sys.argv[1:split])
workload = sys.argv[split + 1 :]
if not workload or not Path(workload[0]).is_file():
    parser.error("provide a Python script after --")
from stream.collector import enable

collector = enable(
    args.out,
    ring_bytes=args.ring_mib << 20,
    chunk_bytes=args.chunk_mib << 20,
    ctas=args.ctas,
    iters=args.iters,
    warps=args.warps,
    patched=True,
    uniform_registers=args.uniform_registers,
)
sys.argv = workload
sys.path.insert(0, str(Path(workload[0]).resolve().parent))
try:
    runpy.run_path(workload[0], run_name="__main__")
finally:
    collector.close()
if args.export:
    from stream.export import export

    print(export(args.out, args.out / "trace.json"))
