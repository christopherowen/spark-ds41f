"""run-iket profile wrapper with multi-process tracker aggregation, tracker reuse,
single-pass NativeDump profiling, raw-data dump bypass, and KV-cache pool deduction.

usage: iket_profile.py OUT_DIR [--timeout SECONDS] [--log-level LEVEL]
                       [--use-tracker-dir DIR] [--single-pass-max-ts N]
                       [--skip-raw-data-dump] [--pool-gb GB] [--warp-trace]
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
3. **Single-pass tracker-free NativeDump mode (``--single-pass-max-ts N``)**:
   TileLang kernels use ``kNativeDump`` (``instrument_method = 3``), which does not
   require ``instrument.config.json`` instrument points (``FunctionShouldInstrument``
   discovers the embedded ``IketContext`` directly from each CUBIN). Writing a 1-entry
   stub ``instrument.config.json`` with ``maxTsCntPerWarp = N`` (e.g., ``32`` for
   ``granularity=phase, iters=1``) skips the ``tracker`` dry run completely on the
   very first run (-50% wall time, 0% ``cuGraphLaunch`` tracker hang risk).
4. **Raw ``iket.data.json`` dump bypass (``--skip-raw-data-dump``)**:
   In ``libsmodel_injection.so``, ``IKET::Finalize`` constructs two JSON dumps at exit:
   (a) ``iket.data.json``, which materializes a 120-byte ``WarpBufferJson`` struct plus
   a ``nlohmann::json`` DOM for every warp of every launch (~2.3 KB/warp = ~90 GB RAM
   for 40M warps) and holds the ``WarpBufferJson`` vectors until the end of ``Finalize``,
   and (b) ``iket.decoded_results.json``, which streams per-launch ``IketTraceData`` and
   frees each launch's vectors immediately. Nothing in ``run-iket``, ``iket_table.py``,
   or ``iket_perfetto.py`` reads ``iket.data.json``. ``--skip-raw-data-dump`` applies an
   8-byte branch patch to a local copy of ``libsmodel_injection.so`` (at ``0xaaa94`` and
   ``0xaac78``) so ``IKET::Finalize`` skips the ``WarpBufferJson`` loops and writes an
   empty 34-byte ``iket.data.json`` while producing the identical ``iket.decoded_results.json``.
5. **Single-pass warp-trace mode (``--warp-trace``)**:
   Runs ``run-iket profile --warp-trace`` in a single pass without a dry run to capture
   the uninstrumented base timeline (all kernels' warp lifetimes and SM/CTA placement).
6. **Pool size extraction (``OUT_DIR/pool.json``), explicit ``--pool-gb``, & KV cache deduction (``--deduct-kv-bytes``)**:
   Parses ``Auto-computed context buffer size: <G> GB [ (<B> bytes) ]`` from the dry run
   (or uses ``--pool-gb``), writes ``OUT_DIR/pool.json``, and when ``--deduct-kv-bytes``
   is set, rewrites ``--kv-bytes`` or ``--kv-cache-memory-bytes`` in ``COMMAND`` by
   subtracting ``pool_bytes + safety_margin_bytes`` so the measured run never OOMs on
   GB10 unified memory when ``run-iket`` allocates its device trace buffer.
"""

import argparse
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
from pathlib import Path

DEFAULT_INJECTION_LIB = Path(
    "/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/dsl_packages/iket/profiler/libsmodel_injection.so"
)
# Exact AArch64 branch instructions in IKET::Finalize (libsmodel_injection.so):
#   0xaaa94: 54000be0 (b.eq 0xaac10) -> 1400005f (b 0xaac10)  [skip regular WarpBufferJson loop]
#   0xaac78: 54001840 (b.eq 0xaaf80) -> 140000c2 (b 0xaaf80)  [skip graph WarpBufferJson loop]
_RAW_DATA_PATCH_SITES = (
    (0xAAA94, 0x54000BE0, 0x1400005F),
    (0xAAC78, 0x54001840, 0x140000C2),
)


def find_default_injection_lib() -> Path:
    if DEFAULT_INJECTION_LIB.exists():
        return DEFAULT_INJECTION_LIB
    try:
        import nvidia_cutlass_dsl  # type: ignore[import-not-found]

        pkg_root = Path(nvidia_cutlass_dsl.__file__).resolve().parent
        candidate = pkg_root / "dsl_packages" / "iket" / "profiler" / "libsmodel_injection.so"
        if candidate.exists():
            return candidate
    except Exception:
        pass
    raise FileNotFoundError(f"Could not locate libsmodel_injection.so at {DEFAULT_INJECTION_LIB}")


def patch_injection_lib_skip_raw_data(out_dir: Path, source_lib: Path | None = None) -> Path:
    """Create a patched copy of ``libsmodel_injection.so`` under
    ``<out_dir>/injection/profiler/libsmodel_injection.so`` (with ``../libiket_cubin_info.so``
    symlinked alongside it to satisfy ``DT_RUNPATH=$ORIGIN/..``) that bypasses the
    ``WarpBufferJson`` / ``iket.data.json`` loops in ``IKET::Finalize``."""
    src = Path(source_lib) if source_lib is not None else find_default_injection_lib()
    inj_root = Path(out_dir) / "injection"
    prof_dir = inj_root / "profiler"
    prof_dir.mkdir(parents=True, exist_ok=True)
    dst = prof_dir / "libsmodel_injection.so"
    shutil.copy2(src, dst)

    cubin_info_src = src.parent.parent / "libiket_cubin_info.so"
    cubin_info_dst = inj_root / "libiket_cubin_info.so"
    if cubin_info_src.exists() and not cubin_info_dst.exists() and not cubin_info_dst.is_symlink():
        cubin_info_dst.symlink_to(cubin_info_src)

    with open(dst, "r+b") as f:
        for offset, expected_orig, patched_insn in _RAW_DATA_PATCH_SITES:
            f.seek(offset)
            raw = f.read(4)
            if len(raw) != 4:
                raise RuntimeError(f"libsmodel_injection.so truncated at 0x{offset:x}")
            (actual,) = struct.unpack("<I", raw)
            if actual not in (expected_orig, patched_insn):
                raise RuntimeError(
                    f"Unexpected instruction at 0x{offset:x} in {src}: "
                    f"0x{actual:08x} (expected 0x{expected_orig:08x})"
                )
            f.seek(offset)
            f.write(struct.pack("<I", patched_insn))
    return dst


def write_single_pass_stub_config(path: Path, max_ts_cnt_per_warp: int) -> Path:
    """Write a minimal ``instrument.config.json`` setting ``maxTsCntPerWarp`` so
    ``run-iket profile --use-config <path>`` runs in a single pass without invoking the
    ``tracker`` dry run."""
    if int(max_ts_cnt_per_warp) <= 0:
        raise ValueError(f"max_ts_cnt_per_warp must be > 0, got {max_ts_cnt_per_warp}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"version": 0},
        "configs": [
            {
                "kernel": "__iket_native_dump_stub__",
                "maxTsCntPerWarp": int(max_ts_cnt_per_warp),
                "instrumentations": [],
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def max_static_events_from_sources(sources_dir: Path) -> int:
    """Return the maximum finite ``events_per_warp_bound`` across ``<sources_dir>/*.json``."""
    best = 0
    for manifest_path in sorted(Path(sources_dir).glob("*.json")):
        try:
            data = json.loads(manifest_path.read_text())
        except Exception:
            continue
        for site in data.get("sites", []):
            bound = site.get("events_per_warp_bound")
            if isinstance(bound, int) and bound > best:
                best = bound
    return best


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


def format_pool_size_arg(pool_gb: float) -> str:
    mib = max(64, int(round(pool_gb * 1024)))
    return f"{mib}M"


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
    parser.add_argument("--pool-gb", type=float, default=0.0,
                        help="explicit --context-buffer-size in GiB passed to run-iket (0: auto)")
    parser.add_argument("--use-tracker-dir", type=Path,
                        help="reuse an existing tracker directory and skip the dry run")
    parser.add_argument("--single-pass-max-ts", type=int, default=0,
                        help="skip the tracker dry run completely and profile in a single pass with "
                             "this maxTsCntPerWarp (e.g. 32 for granularity=phase, iters=1)")
    parser.add_argument("--skip-raw-data-dump", action="store_true",
                        help="patch libsmodel_injection.so to skip building and writing raw "
                             "iket.data.json (~90 GB RAM at 40M warps) while keeping iket.decoded_results.json")
    parser.add_argument("--warp-trace", action="store_true",
                        help="run a single-pass uninstrumented run-iket --warp-trace capture")
    parser.add_argument("--deduct-kv-bytes", action="store_true",
                        help="subtract the trace pool (+ safety margin) from --kv-bytes / --kv-cache-memory-bytes")
    parser.add_argument("--kv-safety-mib", type=int, default=256,
                        help="extra safety margin in MiB added when --deduct-kv-bytes is set (default: 256)")
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    common = ["--log-level", args.log_level]
    if args.pool_gb > 0:
        common.extend(["--context-buffer-size", format_pool_size_arg(args.pool_gb)])
    if args.skip_raw_data_dump:
        patched_lib = patch_injection_lib_skip_raw_data(out)
        common.extend(["--use-injection-lib", str(patched_lib)])
        print(f"using raw-data-bypass injection library: {patched_lib}", flush=True)

    safety_bytes = args.kv_safety_mib * (1 << 20)
    pool_info = {
        "pool_gb": round(args.pool_gb, 3) if args.pool_gb > 0 else 0.0,
        "pool_bytes": int(round(args.pool_gb * (1 << 30))) if args.pool_gb > 0 else 0,
        "safety_margin_bytes": safety_bytes,
        "recommended_kv_deduction_bytes": (
            int(round(args.pool_gb * (1 << 30))) + safety_bytes if args.pool_gb > 0 else 0
        ),
    }

    if args.warp_trace:
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
            ["run-iket", "--output-dir", str(out / "trace"), *common, "profile",
             "--warp-trace", "--keep", "--postprocess", "none", "--", *measured_command],
            args.timeout, out / "trace.log",
        )
        if code:
            log_tails(out / "trace")
        sys.exit(code)

    if args.single_pass_max_ts > 0:
        stub_cfg = write_single_pass_stub_config(out / "stub.config.json", args.single_pass_max_ts)
        pool_record = {
            **pool_info,
            "max_ts_cnt_per_warp": args.single_pass_max_ts,
            "per_process_counts": {"single_pass_stub": args.single_pass_max_ts},
            "stub_config": str(stub_cfg),
            "single_pass": True,
            "skip_raw_data_dump": args.skip_raw_data_dump,
        }
        (out / "pool.json").write_text(json.dumps(pool_record, indent=1) + "\n")
        if args.dry_only:
            sys.exit(0)
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
             "--use-config", str(stub_cfg), "--postprocess", "none", "--", *measured_command],
            args.timeout, out / "trace.log",
        )
        if code:
            log_tails(out / "trace")
        sys.exit(code)

    if args.use_tracker_dir is not None:
        tracker_dir = resolve_tracker_dir(args.use_tracker_dir)
        counts = read_tracker_counts(tracker_dir)
        print(f"reusing tracker {tracker_dir}: per-warp counts by process: {counts}", flush=True)
        prior_pool = args.use_tracker_dir.parent / "pool.json"
        if prior_pool.exists() and args.pool_gb <= 0:
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
        parsed_pool = parse_pool_size(text, safety_mib=args.kv_safety_mib)
        if parsed_pool["pool_bytes"] > 0:
            if args.pool_gb <= 0:
                pool_info = parsed_pool
            print(
                f"dry run's pool for the measured run: {parsed_pool['pool_gb']} GB "
                f"({parsed_pool['pool_bytes']} bytes)",
                flush=True,
            )

    most = max(counts.values(), default=0)
    pool_record = {
        **pool_info,
        "max_ts_cnt_per_warp": most,
        "per_process_counts": counts,
        "tracker_dir": str(tracker_dir),
        "single_pass": False,
        "skip_raw_data_dump": args.skip_raw_data_dump,
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
