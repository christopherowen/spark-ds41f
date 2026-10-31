"""run-iket profile wrapper with multi-process tracker aggregation, tracker reuse,
and automatic KV-cache pool deduction for GB10 unified memory.

usage: iket_profile.py OUT_DIR [--timeout SECONDS] [--log-level LEVEL]
                       [--use-tracker-dir DIR] [--warp-trace]
                       [--deduct-kv-bytes [--kv-safety-mib MIB]]
                       [--max-pool-gb GB] [--dry-only] [--retries N] -- COMMAND...

Capabilities:
1. **Multi-process tracker aggregation**:
   ``run-iket`` sizes every warp's record from ``recommended_max_ts_cnt_per_warp``,
   but its CLI loop breaks after reading the first ``pid_*/tracker.json`` directory.
   A child process that initializes CUDA without launching instrumented kernels
   records 0, causing ``run-iket`` to allocate 0 slots per warp and crash in
   ``iket::to_json`` at exit. ``iket_profile.py`` takes the maximum
   ``recommended_max_ts_cnt_per_warp`` across all ``pid_*/tracker.json`` files.
2. **Tracker directory reuse (``--use-tracker-dir DIR``)**:
   For deterministic workloads (such as ``iket_step.py`` and ``iket_hot.py``), reusing
   a previously captured ``tracker/`` directory skips the dry run completely, halving
   profiling wall time and avoiding ``run-iket``'s CUDA-graph dry-run callback hang.
3. **Single-pass warp-trace mode (``--warp-trace``)**:
   Runs ``run-iket profile --warp-trace`` in a single pass without a dry run to capture
   the uninstrumented base timeline (all kernels' warp lifetimes and SM/CTA placement).
4. **Pool size extraction (``OUT_DIR/pool.json``) & KV cache deduction (``--deduct-kv-bytes``)**:
   Parses ``Auto-computed context buffer size: <G> GB [ (<B> bytes) ]`` from the dry run,
   writes ``OUT_DIR/pool.json``, and when ``--deduct-kv-bytes`` is set, rewrites
   ``--kv-bytes`` or ``--kv-cache-memory-bytes`` in ``COMMAND`` by subtracting
   ``pool_bytes + safety_margin_bytes`` so the measured run never OOMs on GB10 unified
   memory when ``run-iket`` allocates its device trace buffer.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path


def run_iket(argv, timeout, log):
    """Run ``argv`` in a session of its own, its output in ``log``; past ``timeout``
    seconds, kill the session. Returns the exit code and the output, also printed."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as out:
        process = subprocess.Popen(argv, start_new_session=True, stdout=out, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    text = log.read_text(errors="replace")
    sys.stdout.write(text)
    if code == 124:
        print(f"run-iket timed out after {timeout} s; its session was killed", flush=True)
    sys.stdout.flush()
    return code, text


def log_tails(directory, lines=40):
    for log in sorted(Path(directory).glob("*/pid_*/smodel.injection.log")):
        text = log.read_text(errors="replace").splitlines()
        print(f"--- {log} (last {min(lines, len(text))} of {len(text)} lines)", flush=True)
        print("\n".join(text[-lines:]), flush=True)


def read_tracker_counts(tracker_dir: Path) -> dict[str, int]:
    """Read ``recommended_max_ts_cnt_per_warp`` from every ``pid_*/tracker.json`` under
    ``tracker_dir`` (or ``tracker_dir/tracker`` if ``tracker_dir`` is a dry-run root)."""
    search_root = tracker_dir / "tracker" if (tracker_dir / "tracker").is_dir() else tracker_dir
    return {
        path.parent.name: int(json.loads(path.read_text()).get("recommended_max_ts_cnt_per_warp", 0))
        for path in sorted(search_root.glob("pid_*/tracker.json"))
    }


def resolve_tracker_dir(tracker_dir: Path) -> Path:
    return tracker_dir / "tracker" if (tracker_dir / "tracker").is_dir() else tracker_dir


def parse_pool_size(text: str, safety_mib: int = 256) -> dict:
    """Parse ``Auto-computed context buffer size: <G> GB [(<B> bytes)]`` from ``run-iket``
    stdout/stderr and compute the recommended KV-cache deduction in bytes."""
    match = re.search(
        r"Auto-computed context buffer size:\s*([0-9.]+)\s*GB(?:\s*\(\s*([0-9,]+)\s*bytes\s*\))?",
        text,
    )
    safety_bytes = int(safety_mib) * (1 << 20)
    if not match:
        return {
            "pool_gb": 0.0,
            "pool_bytes": 0,
            "safety_margin_bytes": safety_bytes,
            "recommended_kv_deduction_bytes": 0,
        }
    pool_gb = float(match.group(1))
    pool_bytes = (
        int(match.group(2).replace(",", ""))
        if match.group(2)
        else int(round(pool_gb * (1 << 30)))
    )
    return {
        "pool_gb": pool_gb,
        "pool_bytes": pool_bytes,
        "safety_margin_bytes": safety_bytes,
        "recommended_kv_deduction_bytes": pool_bytes + safety_bytes,
    }


def rewrite_kv_bytes(command: list[str], deduction_bytes: int, min_kv_bytes: int = 256 << 20) -> tuple[list[str], int | None, int | None]:
    """Subtract ``deduction_bytes`` from ``--kv-bytes`` or ``--kv-cache-memory-bytes`` in
    ``command``. Returns ``(new_command, old_bytes, new_bytes)``."""
    if deduction_bytes <= 0:
        return list(command), None, None
    out = list(command)
    flags = ("--kv-bytes", "--kv-cache-memory-bytes")
    for i, token in enumerate(out):
        if token in flags and i + 1 < len(out):
            try:
                old_val = int(out[i + 1])
            except ValueError:
                continue
            new_val = max(min_kv_bytes, old_val - deduction_bytes)
            out[i + 1] = str(new_val)
            return out, old_val, new_val
        for flag in flags:
            if token.startswith(f"{flag}="):
                try:
                    old_val = int(token.split("=", 1)[1])
                except ValueError:
                    continue
                new_val = max(min_kv_bytes, old_val - deduction_bytes)
                out[i] = f"{flag}={new_val}"
                return out, old_val, new_val
    return out, None, None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--" not in argv:
        raise SystemExit("usage: iket_profile.py OUT_DIR [options] -- COMMAND...")
    split = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--log-level", default="error")
    parser.add_argument("--dry-only", action="store_true")
    parser.add_argument("--retries", type=int, default=1, help="dry runs started again after a timeout")
    parser.add_argument("--max-pool-gb", type=float, default=0,
                        help="no measured run if the dry run's pool is larger (0: no limit)")
    parser.add_argument("--use-tracker-dir", type=Path,
                        help="reuse an existing tracker directory and skip the dry run")
    parser.add_argument("--warp-trace", action="store_true",
                        help="run a single-pass uninstrumented run-iket --warp-trace capture")
    parser.add_argument("--deduct-kv-bytes", action="store_true",
                        help="subtract the dry run's trace pool (+ safety margin) from --kv-bytes / --kv-cache-memory-bytes")
    parser.add_argument("--kv-safety-mib", type=int, default=256,
                        help="extra safety margin in MiB added when --deduct-kv-bytes is set (default: 256)")
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    common = ["--log-level", args.log_level]

    if args.warp_trace:
        code, _ = run_iket(
            ["run-iket", "--output-dir", str(out / "trace"), *common, "profile",
             "--warp-trace", "--keep", "--postprocess", "none", "--", *command],
            args.timeout, out / "trace.log",
        )
        if code:
            log_tails(out / "trace")
        sys.exit(code)

    pool_info = {
        "pool_gb": 0.0,
        "pool_bytes": 0,
        "safety_margin_bytes": args.kv_safety_mib * (1 << 20),
        "recommended_kv_deduction_bytes": 0,
    }

    if args.use_tracker_dir is not None:
        tracker_dir = resolve_tracker_dir(args.use_tracker_dir)
        counts = read_tracker_counts(tracker_dir)
        print(f"reusing tracker {tracker_dir}: per-warp counts by process: {counts}", flush=True)
        prior_pool = args.use_tracker_dir.parent / "pool.json"
        if prior_pool.exists():
            pool_info.update(json.loads(prior_pool.read_text()))
    else:
        code = 1
        text = ""
        dry = out / "dry"
        for attempt in range(1, args.retries + 2):
            dry = out / ("dry" if attempt == 1 else f"dry-{attempt}")
            code, text = run_iket(
                ["run-iket", "--output-dir", str(dry), "--skip-run", *common, "profile",
                 "--keep", "--postprocess", "none", "--", *command],
                args.timeout, out / f"{dry.name}.log",
            )
            if code == 124:
                log_tails(dry)
                if attempt <= args.retries:
                    print(f"dry run {attempt} hung; starting it again", flush=True)
                    continue
            break
        if code:
            if code != 124:
                log_tails(dry)
            sys.exit(code)
        used = out / "dry-used"
        if not used.is_symlink() and not used.exists():
            used.symlink_to(dry.name)
        tracker_dir = dry / "tracker"
        counts = read_tracker_counts(tracker_dir)
        print(f"dry run's per-warp counts by process: {counts}", flush=True)
        pool_info = parse_pool_size(text, safety_mib=args.kv_safety_mib)
        if pool_info["pool_bytes"] > 0:
            print(
                f"dry run's pool for the measured run: {pool_info['pool_gb']} GB "
                f"({pool_info['pool_bytes']} bytes)",
                flush=True,
            )

    most = max(counts.values(), default=0)
    pool_record = {
        **pool_info,
        "max_ts_cnt_per_warp": most,
        "per_process_counts": counts,
        "tracker_dir": str(tracker_dir),
    }
    (out / "pool.json").write_text(json.dumps(pool_record, indent=1) + "\n")

    if args.dry_only:
        sys.exit(0)
    if args.max_pool_gb and pool_info["pool_gb"] > args.max_pool_gb:
        print(
            f"the pool, {pool_info['pool_gb']} GB, is over --max-pool-gb {args.max_pool_gb:g}: "
            "instrument fewer kernels a run (slices)",
            flush=True,
        )
        sys.exit(3)
    if not most:
        raise SystemExit("no process in the tracker recorded a count")

    measured_command = list(command)
    if args.deduct_kv_bytes and pool_info["recommended_kv_deduction_bytes"] > 0:
        measured_command, old_kv, new_kv = rewrite_kv_bytes(
            measured_command, pool_info["recommended_kv_deduction_bytes"]
        )
        if old_kv is not None:
            print(
                f"deducted {pool_info['recommended_kv_deduction_bytes']} bytes from KV cache: "
                f"{old_kv} -> {new_kv}",
                flush=True,
            )

    code, _ = run_iket(
        ["run-iket", "--output-dir", str(out / "trace"), *common, "profile", "--keep",
         "--use-tracker-dir", str(tracker_dir),
         "--max-ts-cnt-per-warp", str(most), "--postprocess", "none", "--", *measured_command],
        args.timeout, out / "trace.log",
    )
    if code:
        log_tails(out / "trace")
    sys.exit(code)


if __name__ == "__main__":
    main()
