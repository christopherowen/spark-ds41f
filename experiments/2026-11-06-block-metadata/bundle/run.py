#!/usr/bin/env python3
"""Bounded component gates across two fresh processes; no serving claims."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


BUNDLE_DIR = Path(__file__).resolve().parent
CANDIDATE_PATH = BUNDLE_DIR / "candidate.json"
EXPECTED_IMAGE = "sha256:7ae96e98e361601d8232cd99e6af1c6044b7c282bf6904c989e6ad033c147f82"
EXPECTED_TREE = "ca1496c6481dc5bc4b4234b743e4ca023a6c7597"
DEFAULT_HOLDER = "spark-lab: block-metadata component screen"
DEFAULT_RESTORE = "config/cluster.json"
MAX_CPUS = 2.0
MAX_MEMORY_BYTES = 16 * 1024 ** 3
MAX_SWAP_BYTES = 0
MAX_PIDS = 512
THREAD_LIMITS = {
    "CMAKE_BUILD_PARALLEL_LEVEL": 2,
    "MAX_JOBS": 2,
    "MKL_NUM_THREADS": 1,
    "NUMEXPR_MAX_THREADS": 1,
    "OMP_NUM_THREADS": 2,
    "OPENBLAS_NUM_THREADS": 1,
    "TORCHINDUCTOR_COMPILE_THREADS": 2,
    "TVM_NUM_THREADS": 2,
}


def load_candidate(path: Path = CANDIDATE_PATH) -> dict:
    candidate = json.loads(path.read_text())
    if candidate.get("image") != EXPECTED_IMAGE:
        raise RuntimeError(f"unexpected candidate image: {candidate.get('image')!r}")
    labels = candidate.get("expected_labels") or {}
    if labels.get("local.spark-ds41f.vllm.tree") != EXPECTED_TREE:
        raise RuntimeError("candidate expected_labels missing locked vLLM tree")
    return candidate


def require_exact_owner(
    hold: dict | None,
    *,
    holder: str = DEFAULT_HOLDER,
    window: str,
    production_config: str = DEFAULT_RESTORE,
) -> dict:
    if not isinstance(hold, dict):
        raise RuntimeError("ownership lost: missing hold record")
    if not holder or hold.get("holder") != holder:
        raise RuntimeError(f"ownership lost: holder {hold.get('holder')!r} != {holder!r}")
    if not window or hold.get("window") != window:
        raise RuntimeError(f"ownership lost: window {hold.get('window')!r} != {window!r}")
    if not production_config or hold.get("production_config") != production_config:
        raise RuntimeError(
            f"ownership lost: production_config {hold.get('production_config')!r} != {production_config!r}"
        )
    return hold


def validate_thread_limits(environ: Mapping[str, str]) -> dict[str, int]:
    verified = {}
    for key, limit in sorted(THREAD_LIMITS.items()):
        raw = environ.get(key)
        if raw is None or not str(raw).strip():
            raise RuntimeError(f"missing required worker thread/compile bound {key}")
        try:
            value = int(str(raw).strip())
        except ValueError as exc:
            raise RuntimeError(f"non-integer worker thread/compile bound {key}={raw!r}") from exc
        if value <= 0 or value > limit:
            raise RuntimeError(f"worker thread/compile bound {key}={value} Outside 1..{limit}")
        verified[key] = value
    return verified


def worker_environment(
    candidate: dict | None = None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    candidate = load_candidate() if candidate is None else candidate
    configured = {str(k): str(v) for k, v in candidate.get("env", {}).items()}
    validate_thread_limits(configured)
    merged = dict(os.environ if base_env is None else base_env)
    merged.update(configured)
    validate_thread_limits(merged)
    return merged


def render_worker_command(
    bundle_dir: Path = BUNDLE_DIR,
    candidate: dict | None = None,
    *,
    cache_dir: Path | str = "~/.cache/spark-lab/compile",
    boot: int | None = None,
) -> list[str]:
    candidate = load_candidate(bundle_dir / "candidate.json") if candidate is None else candidate
    resources = candidate.get("resources") or {}
    cpus = str(resources.get("cpus", ""))
    memory = str(resources.get("memory", ""))
    memory_swap = str(resources.get("memory_swap", ""))
    pids_limit = int(resources.get("pids_limit", 0))
    if cpus != "2" or float(cpus) > MAX_CPUS:
        raise RuntimeError(f"invalid container CPU cap: {cpus!r}")
    if memory != "16g" or int(resources.get("max_memory_bytes", 0)) != MAX_MEMORY_BYTES:
        raise RuntimeError(f"invalid container memory cap: {memory!r}")
    if memory_swap != memory or int(resources.get("max_swap_bytes", -1)) != MAX_SWAP_BYTES:
        raise RuntimeError(f"container memory_swap {memory_swap!r} must equal memory {memory!r} for no swap")
    if pids_limit <= 0 or pids_limit > MAX_PIDS:
        raise RuntimeError(f"invalid container pids_limit: {pids_limit}")
    configured_env = {str(k): str(v) for k, v in candidate.get("env", {}).items()}
    validate_thread_limits(configured_env)

    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--ulimit=core=0",
        f"--cpus={cpus}",
        f"--memory={memory}",
        f"--memory-swap={memory_swap}",
        f"--pids-limit={pids_limit}",
    ]
    for key, value in sorted(configured_env.items()):
        command += ["-e", f"{key}={value}"]
    resolved_bundle = Path(bundle_dir).resolve()
    for source, destination in candidate.get("mounts", []):
        command += ["-v", f"{resolved_bundle / source}:{destination}:ro"]
    resolved_cache = Path(os.path.expanduser(str(cache_dir)))
    command += ["-v", f"{resolved_cache}:/c"]
    source_root = candidate.get("source_root", "/opt/spark-ds41f/candidate")
    workdir = str(Path(source_root) / candidate.get("workdir", "vllm"))
    command += ["-w", workdir, "--entrypoint", "python3", candidate["image"], "/b/run.py", "--worker"]
    if boot is not None:
        if boot not in (1, 2):
            raise ValueError(f"boot must be 1 or 2, got {boot}")
        command += ["--boot", str(boot)]
    return command


def render_launch_plan(
    bundle_dir: Path = BUNDLE_DIR,
    candidate: dict | None = None,
    *,
    cache_dir: Path | str = "~/.cache/spark-lab/compile",
) -> list[dict]:
    candidate = load_candidate(bundle_dir / "candidate.json") if candidate is None else candidate
    return [
        {
            "process": boot,
            "command": render_worker_command(bundle_dir, candidate, cache_dir=cache_dir, boot=boot),
            "resources": dict(candidate["resources"]),
            "thread_limits": dict(candidate["thread_limits"]),
        }
        for boot in (1, 2)
    ]


def _read_cgroup_token(cgroup_root: Path, filename: str) -> str:
    path = Path(cgroup_root) / filename
    if not path.is_file():
        raise RuntimeError(f"missing worker cgroup control file: {path}")
    return path.read_text().strip()


def verify_worker_bounds(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    environ: Mapping[str, str] | None = None,
    candidate: dict | None = None,
) -> dict:
    candidate = load_candidate() if candidate is None else candidate
    env = os.environ if environ is None else environ
    threads = validate_thread_limits(env)

    cpu_raw = _read_cgroup_token(cgroup_root, "cpu.max")
    cpu_parts = cpu_raw.split()
    if len(cpu_parts) != 2 or cpu_parts[0] == "max":
        raise RuntimeError(f"worker container cpu.max is unbounded or malformed: {cpu_raw!r}")
    quota, period = int(cpu_parts[0]), int(cpu_parts[1])
    if quota <= 0 or period <= 0:
        raise RuntimeError(f"invalid cpu.max quota/period: {cpu_raw!r}")
    cpus = quota / period
    if cpus > MAX_CPUS:
        raise RuntimeError(f"worker container cpu.max {cpus} exceeds {MAX_CPUS} CPUs")

    mem_raw = _read_cgroup_token(cgroup_root, "memory.max")
    if mem_raw == "max":
        raise RuntimeError("worker container memory.max is unbounded")
    memory_max = int(mem_raw)
    if memory_max <= 0 or memory_max > MAX_MEMORY_BYTES:
        raise RuntimeError(f"worker container memory.max {memory_max} Outside 1..{MAX_MEMORY_BYTES}")

    swap_raw = _read_cgroup_token(cgroup_root, "memory.swap.max")
    if swap_raw == "max":
        raise RuntimeError("worker container memory.swap.max is unbounded")
    swap_max = int(swap_raw)
    if swap_max != MAX_SWAP_BYTES:
        raise RuntimeError(f"worker container memory.swap.max {swap_max} must be 0 (no swap)")

    pids_raw = _read_cgroup_token(cgroup_root, "pids.max")
    if pids_raw == "max":
        raise RuntimeError("worker container pids.max is unbounded")
    pids_max = int(pids_raw)
    if pids_max <= 0 or pids_max > MAX_PIDS:
        raise RuntimeError(f"worker container pids.max {pids_max} Outside 1..{MAX_PIDS}")

    return {
        "cgroup_root": str(cgroup_root),
        "cpu_max": f"{quota} {period}",
        "cpus": cpus,
        "memory_max_bytes": memory_max,
        "memory_swap_max_bytes": swap_max,
        "pids_max": pids_max,
        "thread_limits": threads,
    }


def steps_for_boot(out: Path, boot: int) -> list[tuple[str, list[str], int]]:
    return [
        ("host-publication", ["/b/test_host.py"], 30),
        (
            "GPU-transition-gates",
            ["-m", "pytest", "-q", "-rfEs", "--noconftest", "-p", "no:cacheprovider", "/b/test_gpu.py"],
            300,
        ),
        ("component-benchmark", ["/b/bench.py", "--output", str(out / f"boot-{boot}.json")], 600),
    ]


def run_worker(
    out: Path | None = None,
    *,
    boots: Sequence[int] = (1, 2),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    environ: Mapping[str, str] | None = None,
    candidate: dict | None = None,
    runner=subprocess.run,
) -> list[dict]:
    candidate = load_candidate() if candidate is None else candidate
    child_env = worker_environment(candidate, environ)
    initial_bounds = verify_worker_bounds(cgroup_root=cgroup_root, environ=child_env, candidate=candidate)

    if out is None:
        out = Path(child_env.get("BLOCK_METADATA_OUTPUT", "/c/block-metadata"))
    out = Path(out)
    boots_tuple = tuple(int(b) for b in boots)
    if not boots_tuple or any(b not in (1, 2) for b in boots_tuple):
        raise ValueError(f"invalid boots sequence: {boots!r}")

    if boots_tuple == (2,):
        steps_path = out / "steps.json"
        if not out.is_dir() or not steps_path.is_file():
            raise RuntimeError("boot 2 requires existing boot 1 directory and steps.json")
        if (out / "boot-2.json").exists():
            raise RuntimeError("refusing to overwrite existing boot-2.json receipt")
        steps = json.loads(steps_path.read_text())
        if len(steps) != 3 or any(s.get("process") != 1 or s.get("exit") != 0 or not s.get("bounds") for s in steps):
            raise RuntimeError("boot 2 requires three passing, bounded boot 1 steps")
    else:
        # A retry needs a new, explicitly named output directory. Never replace a failed
        # qualification's partial receipts with a later run's results.
        out.mkdir(parents=True, exist_ok=False)
        steps = []

    (out / "bounds.json").write_text(json.dumps(initial_bounds, indent=2) + "\n")
    for boot in boots_tuple:
        for label, args, timeout in steps_for_boot(out, boot):
            bounds = verify_worker_bounds(cgroup_root=cgroup_root, environ=child_env, candidate=candidate)
            print(f"=== process {boot}: {label}", flush=True)
            try:
                code = runner([sys.executable, *args], env=child_env, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            steps.append(
                {
                    "process": boot,
                    "step": label,
                    "exit": code,
                    "timeout_seconds": timeout,
                    "bounds": bounds,
                }
            )
            (out / "steps.json").write_text(json.dumps(steps, indent=2) + "\n")
            print(json.dumps(steps[-1]), flush=True)
            if code != 0:
                raise SystemExit(1)
    print("Component gates completed; native serving validation remains required", flush=True)
    return steps


def run_host(
    *,
    hold_loader,
    holder: str = DEFAULT_HOLDER,
    window: str,
    production_config: str = DEFAULT_RESTORE,
    bundle_dir: Path = BUNDLE_DIR,
    cache_dir: Path | str = "~/.cache/spark-lab/compile",
    runner=subprocess.run,
) -> list[dict]:
    require_exact_owner(hold_loader(), holder=holder, window=window, production_config=production_config)
    candidate = load_candidate(bundle_dir / "candidate.json")
    for source, _ in candidate.get("mounts", []):
        if not (Path(bundle_dir) / source).is_file():
            raise RuntimeError(f"missing bundle mount source: {source}")
    inspect_proc = runner(
        ["docker", "image", "inspect", candidate["image"], "--format", "{{.Id}} {{json .Config.Labels}}"],
        text=True,
        capture_output=True,
    )
    if inspect_proc.returncode != 0:
        raise RuntimeError(f"cannot inspect image {candidate['image']}: {inspect_proc.stderr.strip()}")
    image_id, _, raw_labels = inspect_proc.stdout.strip().partition(" ")
    labels = json.loads(raw_labels or "{}")
    if image_id != EXPECTED_IMAGE or labels.get("local.spark-ds41f.vllm.tree") != EXPECTED_TREE:
        raise RuntimeError("inspected image identity or vLLM tree label does not match locked control")

    plan = render_launch_plan(bundle_dir, candidate, cache_dir=cache_dir)
    outcomes = []
    for item in plan:
        require_exact_owner(hold_loader(), holder=holder, window=window, production_config=production_config)
        proc = runner(item["command"])
        outcomes.append({"process": item["process"], "exit": proc.returncode, "command": item["command"]})
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
    return outcomes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--render-plan", action="store_true", help="Print bounded two-process container launch plan JSON")
    mode.add_argument("--host", action="store_true", help="Launch two bounded worker containers under an owned window")
    mode.add_argument("--worker", action="store_true", help="Run inside the bounded worker container (default)")
    parser.add_argument("--boot", type=int, choices=(1, 2), help="Single boot index when launched per container")
    parser.add_argument("--output", type=Path, help="Override BLOCK_METADATA_OUTPUT directory")
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    parser.add_argument("--window", default="", help="Exact owned window ID for --host mode")
    parser.add_argument("--holder", default=DEFAULT_HOLDER, help="Exact holder string for --host mode")
    parser.add_argument("--production-config", default=DEFAULT_RESTORE, help="Exact restore config for --host mode")
    parser.add_argument(
        "--hold-file",
        type=Path,
        default=Path(os.path.expanduser("~/spark-hold.json")),
        help="Path to spark-hold.json for --host mode",
    )
    parser.add_argument("--cache-dir", default="~/.cache/spark-lab/compile")
    args = parser.parse_args(argv)

    if args.render_plan:
        print(json.dumps(render_launch_plan(cache_dir=args.cache_dir), indent=2))
        return 0
    if args.host:
        def load_hold():
            return json.loads(args.hold_file.read_text()) if args.hold_file.is_file() else None

        run_host(
            hold_loader=load_hold,
            holder=args.holder,
            window=args.window,
            production_config=args.production_config,
            cache_dir=args.cache_dir,
        )
        return 0
    boots = (args.boot,) if args.boot else (1, 2)
    run_worker(out=args.output, boots=boots, cgroup_root=args.cgroup_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
