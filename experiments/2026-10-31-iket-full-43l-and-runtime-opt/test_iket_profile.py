import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "harness"))

import iket_profile
import iket_step
import step_bundle
import step_diff


def test_parse_pool_size_and_rewrite_kv_bytes():
    log_with_bytes = "[run-iket] Auto-computed context buffer size: 1.50 GB (1,610,612,736 bytes)\n"
    info = iket_profile.parse_pool_size(log_with_bytes, safety_mib=256)
    assert info["pool_gb"] == 1.5
    assert info["pool_bytes"] == 1610612736
    assert info["safety_margin_bytes"] == 256 * (1 << 20)
    assert info["recommended_kv_deduction_bytes"] == 1610612736 + 256 * (1 << 20)

    small_log = "[run-iket] Auto-computed context buffer size: 0.00 GB (127,180 bytes)\n"
    small_info = iket_profile.parse_pool_size(small_log, safety_mib=256)
    assert small_info["pool_bytes"] == 127180

    cmd = ["python3", "iket_step.py", "on", "--kv-bytes", str(6 << 30), "--out", "/tmp/out"]
    rewritten, old_val, new_val = iket_profile.rewrite_kv_bytes(
        cmd, info["recommended_kv_deduction_bytes"]
    )
    assert old_val == 6 << 30
    assert new_val == (6 << 30) - info["recommended_kv_deduction_bytes"]
    assert rewritten[rewritten.index("--kv-bytes") + 1] == str(new_val)

    cmd_eq = ["vllm", "serve", "--kv-cache-memory-bytes=4294967296"]
    rewritten_eq, old_eq, new_eq = iket_profile.rewrite_kv_bytes(
        cmd_eq, info["recommended_kv_deduction_bytes"]
    )
    assert old_eq == 4294967296
    assert rewritten_eq[-1] == f"--kv-cache-memory-bytes={new_eq}"


def test_multiprocess_tracker_counts_and_use_tracker_dir(tmp_path):
    tracker_root = tmp_path / "prev_dry" / "tracker"
    (tracker_root / "pid_100").mkdir(parents=True)
    (tracker_root / "pid_101").mkdir(parents=True)
    # Child process pid_100 launched no instrumented kernels (count 0); worker pid_101 recorded 48.
    (tracker_root / "pid_100" / "tracker.json").write_text(
        json.dumps({"recommended_max_ts_cnt_per_warp": 0})
    )
    (tracker_root / "pid_101" / "tracker.json").write_text(
        json.dumps({"recommended_max_ts_cnt_per_warp": 48})
    )
    (tmp_path / "prev_dry" / "pool.json").write_text(
        json.dumps({
            "pool_gb": 1.0,
            "pool_bytes": 1 << 30,
            "safety_margin_bytes": 256 << 20,
            "recommended_kv_deduction_bytes": (1 << 30) + (256 << 20),
        })
    )

    counts = iket_profile.read_tracker_counts(tracker_root)
    assert counts == {"pid_100": 0, "pid_101": 48}

    invocations = []

    def fake_run_iket(argv, timeout, log):
        invocations.append(argv)
        return 0, "ok\n"

    out_dir = tmp_path / "run_out"
    with patch.object(iket_profile, "run_iket", side_effect=fake_run_iket):
        with pytest.raises(SystemExit) as exc:
            iket_profile.main([
                str(out_dir),
                "--use-tracker-dir", str(tracker_root),
                "--deduct-kv-bytes",
                "--",
                "python3", "iket_step.py", "on", "--kv-bytes", str(4 << 30),
            ])
        assert exc.value.code == 0

    # Only 1 invocation (measured run), dry run was skipped!
    assert len(invocations) == 1
    measured_argv = invocations[0]
    assert "--skip-run" not in measured_argv
    assert "--max-ts-cnt-per-warp" in measured_argv
    assert measured_argv[measured_argv.index("--max-ts-cnt-per-warp") + 1] == "48"
    expected_kv = (4 << 30) - ((1 << 30) + (256 << 20))
    assert measured_argv[measured_argv.index("--kv-bytes") + 1] == str(expected_kv)
    pool_json = json.loads((out_dir / "pool.json").read_text())
    assert pool_json["max_ts_cnt_per_warp"] == 48


def test_default_grid_rows_and_step_spec():
    assert iket_step.default_grid_rows(requests=1, drafts=5, grid_rows=0, eager=False) == 6
    assert iket_step.default_grid_rows(requests=8, drafts=5, grid_rows=0, eager=False) == 2
    assert iket_step.default_grid_rows(requests=8, drafts=5, grid_rows=0, eager=True) == 6
    assert iket_step.default_grid_rows(requests=8, drafts=5, grid_rows=4, eager=False) == 4

    spec = step_bundle.step_spec(
        "HEAD", tp=4, requests=8, tokens=16, ctas=1, iters=1, warps=2,
        granularity="phase", payloads=True, base_mode="iket",
    )
    step0_argv = spec["steps"][0][1]
    step1_argv = spec["steps"][1][1]
    assert "--deduct-kv-bytes" in step0_argv
    assert "--granularity" in step0_argv and step0_argv[step0_argv.index("--granularity") + 1] == "phase"
    assert "--warps" in step0_argv and step0_argv[step0_argv.index("--warps") + 1] == "2"
    assert "--payloads" in step0_argv
    assert "--grid-rows" in step0_argv and step0_argv[step0_argv.index("--grid-rows") + 1] == "2"
    assert "--capture-ladder" in step0_argv and "--capture-ladder" in step1_argv
    assert "--warp-trace" in step1_argv


def test_step_diff_supports_iket_json(tmp_path):
    def launch(name, s, e):
        return {
            "kernelName": name,
            "warpLifetimes": [{"locIdx": 0, "startTs": s, "endTs": e}],
        }

    trace_path = tmp_path / "warp_trace.json"
    trace_path.write_text(json.dumps({
        "graphLaunches": {
            "graph_exec_0:0": [
                # Request decode step (between iket_step_marker_kernel at 1000 and 50000)
                launch("sparse_mla_kernel", 10_000, 25_000),
                launch("moe_grouped_fp8_fp4_gemm_kernel", 26_000, 46_000),
                # Grid runs (2 runs between iket_grid_marker_kernel at 60000 and 150000)
                launch("sparse_mla_kernel", 70_000, 84_000),
                launch("moe_grouped_fp8_fp4_gemm_kernel", 85_000, 103_000),
                launch("sparse_mla_kernel", 110_000, 124_000),
                launch("moe_grouped_fp8_fp4_gemm_kernel", 125_000, 143_000),
            ]
        },
        "launches": [
            launch("iket_step_marker_kernel", 1_000, 2_000),
            launch("prepare_pos_seq_lens_kernel", 5_000, 6_000),
            launch("iket_step_marker_kernel", 50_000, 51_000),
            launch("iket_grid_marker_kernel", 60_000, 61_000),
            launch("iket_grid_marker_kernel", 150_000, 151_000),
        ],
    }))
    rows = step_diff.load_rows(trace_path)
    req = step_diff.per_step(
        step_diff.decode_steps(
            step_diff.window(rows, "iket_step_marker_kernel"),
            "prepare_pos_seq_lens_kernel",
            "prepare_prefill_inputs_kernel",
        )
    )
    assert req["steps"] == 1
    assert abs(req["kernels"]["sparse_mla_kernel"]["us"] - 15.0) < 1e-6
    assert abs(req["kernels"]["moe_grouped_fp8_fp4_gemm_kernel"]["us"] - 20.0) < 1e-6
