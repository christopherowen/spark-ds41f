"""Command-line argument parser and entrypoint for bin/spark."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import fcntl
import functools
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLUSTER_CONFIG = "config/cluster.json"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import source_labels
import kernel_backend
import kernel_readiness
import topology
import recipe as cluster_recipe
from .config import *
from .doctor import *
from .cluster import *
from .build import *
from .bench import *

def int_list(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers: {value!r}") from error


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="spark")
    root.add_argument(
        "--cluster-config",
        default=None,
        metavar="PATH",
        help="repository-relative cluster config (default for read-only commands: config/cluster.json; "
        "every --apply needs it named)",
    )
    root.add_argument(
        "--shape",
        default=None,
        metavar="NAME",
        help=f"the recipe's shape to use instead of the site's cabling ({cluster_recipe.SITE_FILE})",
    )
    commands = root.add_subparsers(dest="command", required=True)
    site = commands.add_parser("site", help="this site's cabling: the recipe shape it serves")
    site_commands = site.add_subparsers(dest="site_command", required=True)
    site_shape = site_commands.add_parser("shape", help="show or set the shape this site is cabled as")
    site_shape.add_argument("name", nargs="?", help="a shape of config/cluster.json (after re-cabling)")
    site_shape.add_argument("--apply", action="store_true")
    site_shape.set_defaults(func=command_site_shape)
    doctor = commands.add_parser("doctor", help="validate repository and optional live state")
    doctor.add_argument("--live", action="store_true")
    doctor.add_argument(
        "--fragmentation-commands", action="store_true",
        help="check remote checkpoint fragmentation and print correction commands only; never execute them",
    )
    doctor.set_defaults(func=command_doctor)
    fabric = commands.add_parser("topology", help="prepare a three- or four-node switchless configuration")
    fabric_commands = fabric.add_subparsers(dest="topology_command", required=True)
    create = fabric_commands.add_parser("create", help="write a launch-disabled candidate from this profile and a node map")
    create.add_argument("--nodes-config", required=True, help="repository-relative node map in cable/rank order")
    create.add_argument("--output", required=True, help="new repository-relative candidate path")
    create.set_defaults(func=command_topology_create)
    probe = fabric_commands.add_parser("probe", help="print a bounded, model-free collective probe command for one node")
    probe.add_argument("node")
    probe.add_argument("--master-port", type=int, default=29999)
    probe.set_defaults(func=command_topology_probe)
    status = commands.add_parser("status", help="show configured live containers")
    status.set_defaults(func=command_status)
    cluster = commands.add_parser(
        "cluster", help="plan or apply coordinated node operations"
    )
    cluster_commands = cluster.add_subparsers(dest="cluster_command", required=True)
    cluster_sync = cluster_commands.add_parser(
        "sync", help="detach every clean node checkout at one published commit"
    )
    cluster_sync.add_argument("--revision", help="published commit or ref, resolved after a fetch "
                                                    "(origin/main is the latest main); defaults to local HEAD")
    cluster_sync.add_argument(
        "--apply", action="store_true", help="perform the planned synchronization"
    )
    cluster_sync.set_defaults(func=command_cluster_sync)
    cluster_start = cluster_commands.add_parser(
        "start", help="preflight and start workers, head, then memory guards"
    )
    cluster_start.add_argument(
        "--replace", action="store_true", help="replace existing exact containers"
    )
    cluster_start.add_argument(
        "--apply", action="store_true", help="perform the planned coordinated start"
    )
    cluster_start.add_argument(
        "--change-shape", action="store_true",
        help="allow a start that changes the engine's node count, tensor parallelism, context or KV size",
    )
    cluster_start.add_argument(
        "--unqualified-shape", action="store_true",
        help="start a recipe shape that no release benchmark has qualified for the recipe's image",
    )
    cluster_start.add_argument(
        "--verbose", action="store_true", help="print each rank's docker command in the plan"
    )
    cluster_start.set_defaults(func=command_cluster_start)
    cluster_stop = cluster_commands.add_parser(
        "stop", help="stop the exact promoted container on every node"
    )
    cluster_stop.add_argument(
        "--remove", action="store_true", help="remove stopped containers as well"
    )
    cluster_stop.add_argument(
        "--apply", action="store_true", help="perform the planned coordinated stop"
    )
    cluster_stop.add_argument(
        "--parallel", action="store_true", help="stop the nodes concurrently instead of in turn"
    )
    cluster_stop.set_defaults(func=command_cluster_stop)
    render = commands.add_parser("render", help="render, but do not execute, one rank launch")
    render.add_argument("node", help="node name from the selected profile's node map")
    render.set_defaults(func=command_render)
    upstream = commands.add_parser("upstream", help="inspect or prepare pinned upstreams")
    upstream_commands = upstream.add_subparsers(dest="upstream_command", required=True)
    upstream_list = upstream_commands.add_parser("list")
    upstream_list.set_defaults(func=command_upstream)
    upstream_prepare = upstream_commands.add_parser("prepare")
    upstream_prepare.add_argument("name", help="Git source key from upstreams.lock.json")
    upstream_prepare.set_defaults(func=command_upstream)
    build = commands.add_parser("build", help="prepare, verify, and build the pinned image")
    build_commands = build.add_subparsers(dest="build_command", required=True)
    build_prepare = build_commands.add_parser(
        "prepare", help="create or repair the build directory the lock determines"
    )
    build_prepare.add_argument(
        "--only",
        metavar="PARTS",
        help="comma-separated subset of: "
        + ", ".join((*BUILD_PARTS, *OPTIONAL_BUILD_PROJECTS, *INDEPENDENT_BUILD_PROJECTS))
        + " (sources only, no contexts; the last two only when the lock lists them)",
    )
    build_prepare.set_defaults(func=command_build_prepare)
    build_check = build_commands.add_parser("check", help="verify the prepared build directory")
    build_check.set_defaults(func=command_build_check)
    build_image = build_commands.add_parser(
        "image", help="plan, or build and smoke-test, the image on an idle host"
    )
    build_image.add_argument(
        "--tag", help="image tag (default: container.image from the cluster config)"
    )
    build_image.add_argument(
        "--apply", action="store_true", help="build under the memory guard, then run the GPU smoke"
    )
    build_image.set_defaults(func=command_build_image)
    build_smoke = build_commands.add_parser(
        "smoke", help="GPU import smoke of a built image on an idle host"
    )
    build_smoke.add_argument(
        "--image", help="image to test (default: container.image from the cluster config)"
    )
    build_smoke.set_defaults(func=command_build_smoke)
    bench = commands.add_parser(
        "bench", help="benchmark the live service and compare it with a reference run"
    )
    bench.add_argument(
        "--full",
        action="store_true",
        help="every suite, each decode point sampled to --precision (about 35 minutes); "
        "the default is the quality gate and decode with 3-4 samples (about 6 minutes)",
    )
    bench.add_argument(
        "--suite",
        metavar="NAME",
        help="run this servebench suite (servebench/suites/<NAME>.json, or a JSON file) instead of the "
        "bench below, under the same cooling, fan curve, memory and temperature watch, with the "
        "deployment's identity stored in the run; --output is the run directory (default "
        "results/private/suites/<UTC time>); exits 2 when refused, 1 when a step failed",
    )
    bench.add_argument("--label", help="--suite: the run's name in tables (default: the suite's)")
    bench.add_argument("--corpus", metavar="PATH",
                       help="--suite: model steps score this earlier model result's text (the first arm's)")
    bench.add_argument(
        "--suites",
        help="comma-separated subset of: " + ", ".join(BENCH_SUITES),
    )
    bench.add_argument("--url", help="API base URL (default: head node management IP)")
    bench.add_argument(
        "--output", metavar="DIR", help="report directory (default: results/private/bench/<UTC time>)"
    )
    bench.add_argument(
        "--compare",
        metavar="PATH",
        help="reference bench.json, or with --suite a suite run directory (default: the profile's benchmark_reference, else "
        "manifests/benchmarks/<promoted_baseline>.json when present; a default measured on another "
        "node count, transport or kernel backend is refused, an explicit one only warns; "
        "'none' skips the comparison)",
    )
    bench.add_argument(
        "--precision",
        type=float,
        default=2.0,
        help="sample each decode point until its 95%% CI is within this percent of the mean",
    )
    bench.add_argument(
        "--cool-below",
        type=float,
        default=COOL_BELOW_C,
        help=f"before measuring, bring every node's hottest thermal zone below this many C "
        f"(default {COOL_BELOW_C:g}), at the maximum dgx-fan-control floor when needed, then "
        "restore its usual fan control; 0 skips",
    )
    bench.add_argument(
        "--cool-timeout",
        type=float,
        default=COOL_TIMEOUT_S,
        help=f"seconds a node may take to cool before the bench aborts (default {COOL_TIMEOUT_S:g})",
    )
    bench.add_argument("--min-samples", type=int, help="default 3, or 6 with --full")
    bench.add_argument("--max-samples", type=int, help="default 4, or 40 with --full")
    bench.add_argument("--concurrency", type=int_list, default=[1, 2, 4, 8])
    bench.add_argument(
        "--decode-cases",
        type=lambda value: [case.strip() for case in value.split(",") if case.strip()],
        default=["prose", "code"],
        help=(
            "comma-separated decode cases (default prose,code): "
            f"{', '.join(DECODE_CASES)}; groups: {', '.join(DECODE_CASE_GROUPS)}"
        ),
    )
    bench.add_argument(
        "--converge-on",
        choices=CONVERGENCE_METRICS,
        help="the per-sample metric whose 95%% CI must reach --precision before a decode point stops "
        "sampling; default: the case's choice (step_ms for the precision group), otherwise tps",
    )
    bench.add_argument(
        "--prefill-text",
        choices=("filler", "novel", "source"),
        default="filler",
        help="cold-prefill prompt text: the repeating filler (default), random "
        "pseudo-words whose Engram rows are not cached, or real text (Python "
        "standard-library source)",
    )
    bench.add_argument("--sampled-requests", type=int, default=32)
    bench.add_argument("--prefill-sizes", type=int_list, default=[2048, 32768, 65536, 131072])
    bench.add_argument("--prefill-repeats", type=int, default=3)
    bench.add_argument("--prefix-tokens", type=int, default=32768)
    bench.add_argument("--admission-tokens", type=int, default=65536)
    bench.add_argument("--admission-output-tokens", type=int, default=256,
                       help="output budget per admission request; increase for overlapping long prefills")
    bench.add_argument("--admission-force-length", action="store_true",
                       help="force the full admission output budget to keep long contexts active together")
    bench.add_argument(
        "--tolerance",
        type=float,
        default=3.0,
        help="fail when a point is significantly slower than the reference by more than this percent",
    )
    bench.add_argument("--seed", type=int, default=0, help="seed for sample order and prompts")
    bench.add_argument(
        "--report",
        metavar="PATH",
        help="summarize an existing bench.json against --compare instead of measuring",
    )
    bench.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="benchmark even when doctor --live reports differences",
    )
    bench.set_defaults(func=command_bench)
    workload = commands.add_parser(
        "workload",
        help="summarize live traffic over a window (read-only; sends no requests)",
        description="Snapshot the server's metrics, wait for --duration, Ctrl-C, or the "
        "command after --, then report the traffic shape, latency, draft acceptance, and "
        "load for that window.",
    )
    workload.add_argument("--url", help="API base URL (default: head node management IP)")
    workload.add_argument("--duration", type=float, help="window length in seconds")
    workload.add_argument("--interval", type=float, default=5.0, help="gauge sampling period")
    workload.add_argument("--label", help="name recorded in the JSON report")
    workload.add_argument("--json", metavar="PATH", help="also write the summary as JSON")
    workload.add_argument("run", nargs=argparse.REMAINDER, help="-- command to time the window around")
    workload.set_defaults(func=command_workload)
    return root



def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    return int(arguments.func(arguments) or 0)
