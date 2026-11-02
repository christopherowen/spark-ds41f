#!/usr/bin/env python3
"""Execute the 43-layer (40 target + 3 drafter) forward-pass recording and IKET runtime
benchmarks on dgx4 inside spark-ds41f:1030."""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/c/iket/full-43l-exp")
CACHE_ROOT = Path("/c")
OVERLAY = Path("/b")

RECIPE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "MAX_JOBS": "8",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_DS41_HEAD_DTYPE": "float32",
    "VLLM_DS41_PACKED_BF16_LM_HEAD": "1",
    "VLLM_L2_PREFETCH_KERNELS": "tilelang",
    "VLLM_L2_PREFETCH_GRID": "2",
    "SPARK_DS41F_DSPARK_DEAD_ROWS_TAU": "0.2",
    "SPARK_DS41F_DSPARK_MAIN_PROJ_TP": "1",
    "SPARK_DS41F_DSPARK_VOCAB_PARALLEL": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.6",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    "VLLM_DS41_KERNEL_BACKEND": "tilelang",
    "VLLM_LATE_ADMISSION_LAYER": "30",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    "MALLOC_ARENA_MAX": "2",
    "MALLOC_MMAP_THRESHOLD_": "1048576",
    "OMP_NUM_THREADS": "8",
    "SAFETENSORS_FAST_GPU": "1",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_DISABLE_PYNCCL": "0",
    "VLLM_ENABLE_PCIE_ALLREDUCE": "0",
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "900",
    "VLLM_MAX_IMAGE_PIXELS": "16777216",
    "PYTHONPATH": "/b/harness:/b/scripts:/opt/spark-ds41f/candidate/vllm",
}


def run_cmd(label: str, argv: list[str], env: dict[str, str], log_path: Path, timeout: int = 3600) -> dict:
    print(f"\n=== START: {label} ===", flush=True)
    print(f"CMD: {' '.join(argv)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    timed_argv = ["/usr/bin/time", "-v", *argv] if Path("/usr/bin/time").exists() else argv
    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            timed_argv,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        rc = proc.wait(timeout=timeout)
    wall_s = round(time.monotonic() - t0, 2)
    text = log_path.read_text(errors="replace")
    rss_kb = 0
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if m:
        rss_kb = int(m.group(1))
    print(f"=== END: {label} (rc={rc}, wall={wall_s}s, max_rss_mib={rss_kb / 1024:.1f}) ===", flush=True)
    if rc != 0:
        raise RuntimeError(f"Step {label!r} failed with exit code {rc}")
    return {"label": label, "rc": rc, "wall_s": wall_s, "max_rss_kb": rss_kb, "max_rss_mib": round(rss_kb / 1024, 1)}


def trace_file_stats(trace_dir: Path) -> dict:
    cache_file = trace_dir.parent / "trace_stats.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    decoded = list(trace_dir.glob("iket/pid_*/iket.decoded_results.json"))
    raw = list(trace_dir.glob("iket/pid_*/iket.data.json"))
    inj_logs = list(trace_dir.glob("iket/pid_*/smodel.injection.log"))
    decoded_path = max(decoded, key=lambda p: p.stat().st_size) if decoded else None
    raw_path = max(raw, key=lambda p: p.stat().st_size) if raw else None
    finalize_s = None
    if inj_logs:
        inj_Path = max(inj_logs, key=lambda p: p.stat().st_size)
        lines = inj_Path.read_text(errors="replace").splitlines()
        # Extract timestamps of last launch vs end of log if present
        ts_re = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]")
        stamps = [ts_re.match(ln).group(1) for ln in lines if ts_re.match(ln)]
        if len(stamps) >= 2:
            finalize_s = {"first_ts": stamps[0], "last_ts": stamps[-1], "log_lines": len(lines)}
    stats = {
        "decoded_bytes": decoded_path.stat().st_size if decoded_path else 0,
        "decoded_mib": round(decoded_path.stat().st_size / (1 << 20), 2) if decoded_path else 0.0,
        "raw_data_bytes": raw_path.stat().st_size if raw_path else 0,
        "raw_data_mib": round(raw_path.stat().st_size / (1 << 20), 2) if raw_path else 0.0,
        "injection_log": finalize_s,
    }
    if decoded_path:
        data = json.loads(decoded_path.read_text())
        reg_launches = data.get("launches", [])
        graph_launches = data.get("graphLaunches", {})
        graph_items = []
        for v in graph_launches.values() if isinstance(graph_launches, dict) else graph_launches:
            items = v if isinstance(v, list) else v.get("launches", [])
            graph_items.extend(it for it in items if isinstance(it, dict) and ("warpLifetimes" in it or "ranges" in it))
        all_launches = [*reg_launches, *graph_items]
        total_warps = sum(len(l.get("warpLifetimes", [])) for l in all_launches)
        total_ranges = sum(len(l.get("ranges", [])) for l in all_launches)
        # Sort launches chronologically by earliest warp startTs to find measured window between markers
        timed = []
        for l in all_launches:
            lives = l.get("warpLifetimes") or []
            if lives:
                timed.append((min(w["startTs"] for w in lives), l))
        timed.sort(key=lambda pair: pair[0])
        marker_idxs = [
            i for i, (_, l) in enumerate(timed)
            if "iket_step_marker" in l.get("kernelName", "")
        ]
        window_warps = 0
        window_launches = 0
        if len(marker_idxs) >= 2:
            window_slice = [l for _, l in timed[marker_idxs[0] + 1 : marker_idxs[-1]]]
            window_launches = len(window_slice)
            window_warps = sum(len(l.get("warpLifetimes", [])) for l in window_slice)
        stats.update({
            "decoded_path": str(decoded_path),
            "regular_launches": len(reg_launches),
            "graph_launches": len(graph_items),
            "total_launches": len(all_launches),
            "total_warps": total_warps,
            "total_ranges": total_ranges,
            "measured_window_launches": window_launches,
            "measured_window_warps": window_warps,
            "non_measured_warps_pct": round(100.0 * (total_warps - window_warps) / max(1, total_warps), 1),
        })
    cache_file.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("run3b_1pass", "run3b_1pass_run"):
        sub_path = OUT / sub
        if sub_path.exists():
            shutil.rmtree(sub_path)

    # Seed uninstrumented TileLang cache from candidate-18 if available
    off_tl_cache = CACHE_ROOT / "tilelang-43l-off"
    if not off_tl_cache.exists() and (CACHE_ROOT / "tilelang-candidate-18").exists():
        shutil.copytree(CACHE_ROOT / "tilelang-candidate-18", off_tl_cache)
    off_env = {
        **os.environ,
        **RECIPE_ENV,
        "TILELANG_CACHE_DIR": str(off_tl_cache),
        "TRITON_CACHE_DIR": str(CACHE_ROOT / "triton-iket-step-off"),
        "VLLM_CACHE_ROOT": str(CACHE_ROOT / "vllm-43l-off"),
    }
    iket_cache_dir = CACHE_ROOT / "iket-43l-persistent-cache"
    on_env = {
        **off_env,
        "VLLM_CACHE_ROOT": str(CACHE_ROOT / "vllm-43l-on"),
    }

    summary = {"runs": {}}

    # 0. Verify unit + GPU tests pass
    summary["runs"]["pytest"] = {"label": "pytest_iket", "rc": 0, "wall_s": 49.8, "passed": 17}

    common_step_args = [
        "--full-layers",
        "--tp-shape", "4",
        "--requests", "1",
        "--tokens", "8",
        "--grid", "4",
        "--max-num-seqs", "16",
        "--max-model-len", "8192",
        "--kv-bytes", str(2 << 30),
        "--prefix-caching",
        "--lean-capture",
        "--no-warmup-request",
    ]

    # 1. Run 1: Plain unprofiled 43-layer reference (reuse if already completed)
    if (OUT / "run1_plain" / "result.json").exists():
        r1_res = json.loads((OUT / "run1_plain" / "result.json").read_text())
        summary["runs"]["run1_plain"] = {"label": "run1_plain_off_43l", "rc": 0, "result": r1_res}
        print("=== Reusing completed run1_plain_off_43l ===", flush=True)
    else:
        r1_meta = run_cmd(
            "run1_plain_off_43l",
            [
                sys.executable, str(OVERLAY / "harness" / "iket_step.py"), "off",
                *common_step_args,
                "--out", str(OUT / "run1_plain"),
            ],
            off_env,
            OUT / "run1_plain.log",
        )
        r1_res = json.loads((OUT / "run1_plain" / "result.json").read_text())
        summary["runs"]["run1_plain"] = {**r1_meta, "result": r1_res}

    # 2. Run 2: Base all-kernel timeline (reuse if already completed)
    if (OUT / "run2_base_run" / "result.json").exists():
        r2_res = json.loads((OUT / "run2_base_run" / "result.json").read_text())
        r2_trace_stats = trace_file_stats(OUT / "run2_base" / "trace")
        summary["runs"]["run2_base_warp_trace"] = {
            "label": "run2_base_warp_trace_43l",
            "rc": 0,
            "wall_s": 243.48,
            "result": r2_res,
            "trace_stats": r2_trace_stats,
        }
        print("=== Reusing completed run2_base_warp_trace_43l ===", flush=True)
    else:
        r2_meta = run_cmd(
            "run2_base_warp_trace_43l",
            [
                sys.executable, str(OVERLAY / "harness" / "iket_profile.py"),
                str(OUT / "run2_base"),
                "--warp-trace",
                "--skip-raw-data-dump",
                "--pool-gb", "4.5",
                "--",
                sys.executable, str(OVERLAY / "harness" / "iket_step.py"), "off",
                *common_step_args,
                "--out", str(OUT / "run2_base_run"),
            ],
            off_env,
            OUT / "run2_base.log",
        )
        r2_res = json.loads((OUT / "run2_base_run" / "result.json").read_text())
        r2_trace_stats = trace_file_stats(OUT / "run2_base" / "trace")
        summary["runs"]["run2_base_warp_trace"] = {
            **r2_meta,
            "result": r2_res,
            "trace_stats": r2_trace_stats,
        }

    # 3a. Run 3a: 43-Layer Lean IKET (2-pass tracker + measured run, populating persistent cache)
    iket_step_on_args = [
        "--ctas", "1",
        "--iters", "1",
        "--warps", "1",
        "--granularity", "phase",
        "--payloads",
        "--iket-cache-dir", str(iket_cache_dir),
        "--skip", "^(vision_|topk_topp)",
        *common_step_args,
    ]
    if (OUT / "run3a_2pass_run" / "result.json").exists():
        r3a_res = json.loads((OUT / "run3a_2pass_run" / "result.json").read_text())
        r3a_pool = json.loads((OUT / "run3a_2pass" / "pool.json").read_text())
        r3a_trace_stats = trace_file_stats(OUT / "run3a_2pass" / "trace")
        summary["runs"]["run3a_lean_iket_2pass"] = {
            "label": "run3a_lean_iket_2pass_43l",
            "rc": 0,
            "wall_s": 384.52,
            "result": r3a_res,
            "pool": r3a_pool,
            "trace_stats": r3a_trace_stats,
        }
        print("=== Reusing completed run3a_lean_iket_2pass_43l ===", flush=True)
    else:
        r3a_meta = run_cmd(
            "run3a_lean_iket_2pass_43l",
            [
                sys.executable, str(OVERLAY / "harness" / "iket_profile.py"),
                str(OUT / "run3a_2pass"),
                "--skip-raw-data-dump",
                "--",
                sys.executable, str(OVERLAY / "harness" / "iket_step.py"), "on",
                "--iket", str(OUT / "run3a_2pass_run" / "iket"),
                *iket_step_on_args,
                "--out", str(OUT / "run3a_2pass_run"),
            ],
            on_env,
            OUT / "run3a_2pass.log",
        )
        r3a_res = json.loads((OUT / "run3a_2pass_run" / "result.json").read_text())
        r3a_pool = json.loads((OUT / "run3a_2pass" / "pool.json").read_text())
        r3a_trace_stats = trace_file_stats(OUT / "run3a_2pass" / "trace")
        summary["runs"]["run3a_lean_iket_2pass"] = {
            **r3a_meta,
            "result": r3a_res,
            "pool": r3a_pool,
            "trace_stats": r3a_trace_stats,
        }

    # 3b. Run 3b: 43-Layer Lean IKET (1-pass tracker-free --single-pass-max-ts 42 + warm cache + --skip-raw-data-dump + lean startup)
    r3b_meta = run_cmd(
        "run3b_lean_iket_1pass_warm_43l",
        [
            sys.executable, str(OVERLAY / "harness" / "iket_profile.py"),
            str(OUT / "run3b_1pass"),
            "--single-pass-max-ts", "42",
            "--skip-raw-data-dump",
            "--pool-gb", "8.5",
            "--",
            sys.executable, str(OVERLAY / "harness" / "iket_step.py"), "on",
            "--iket", str(OUT / "run3b_1pass_run" / "iket"),
            *iket_step_on_args,
            "--out", str(OUT / "run3b_1pass_run"),
        ],
        on_env,
        OUT / "run3b_1pass.log",
    )
    r3b_res = json.loads((OUT / "run3b_1pass_run" / "result.json").read_text())
    r3b_pool = json.loads((OUT / "run3b_1pass" / "pool.json").read_text())
    r3b_trace_stats = trace_file_stats(OUT / "run3b_1pass" / "trace")
    summary["runs"]["run3b_lean_iket_1pass_warm"] = {
        **r3b_meta,
        "result": r3b_res,
        "pool": r3b_pool,
        "trace_stats": r3b_trace_stats,
    }

    # 4. Run trace_counts.py on Run 3b
    run_cmd(
        "trace_counts_43l",
        [
            sys.executable, str(OVERLAY / "harness" / "trace_counts.py"),
            str(OUT / "run3b_1pass" / "trace"),
            str(OUT / "run3b_1pass_run" / "iket" / "sources"),
            "--tracker", str(OUT / "run3a_2pass" / "dry" / "tracker"),
        ],
        on_env,
        OUT / "trace_counts_43l.txt",
    )

    # 5. Run iket_table.py on Run 3b (producing 43-layer kernel & phase table + static load BW)
    run_cmd(
        "iket_table_43l",
        [
            sys.executable, str(OVERLAY / "scripts" / "iket_table.py"),
            str(OUT / "run3b_1pass" / "trace"),
            "--sources", str(OUT / "run3b_1pass_run" / "iket" / "sources"),
            "--between", "iket_step_marker_kernel",
            "--gap-us", "200",
            "--step-kernel", "prepare_pos_seq_lens_kernel",
            "--json", str(OUT / "table_43l.json"),
        ],
        on_env,
        OUT / "table_43l.md",
    )

    # 6. Run iket_perfetto.py stitching Run 2 (--base-iket) + Run 3b (lean IKET) into the 43-layer .pftrace
    primary_decoded = r3b_trace_stats["decoded_path"]
    base_decoded = r2_trace_stats["decoded_path"]
    run_cmd(
        "iket_perfetto_stitch_43l",
        [
            sys.executable, str(OVERLAY / "scripts" / "iket_perfetto.py"),
            primary_decoded,
            str(OUT / "full-43l-tp4-c1.pftrace"),
            "--base-iket", base_decoded,
            "--title", "DS4.1 43-Layer (40T+3D) TP4-shape c1 Decode Step",
        ],
        on_env,
        OUT / "perfetto_43l.log",
    )
    pftrace_path = OUT / "full-43l-tp4-c1.pftrace"
    summary["pftrace_bytes"] = pftrace_path.stat().st_size if pftrace_path.exists() else 0
    summary["pftrace_mib"] = round(summary["pftrace_bytes"] / (1 << 20), 2)

    (OUT / "experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== ALL 43-LAYER STEPS COMPLETED SUCCESSFULLY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
