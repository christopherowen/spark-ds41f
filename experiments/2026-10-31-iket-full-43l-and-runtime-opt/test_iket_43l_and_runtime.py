"""Unit tests for the 43-layer full-pass harness and IKET runtime optimizations."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "scripts"))

import iket_profile  # noqa: E402
import iket_step  # noqa: E402
import step_bundle  # noqa: E402


def test_full_text_overrides_matches_43_layer_checkpoint():
    ov = iket_step.FULL_TEXT_OVERRIDES
    assert ov["num_hidden_layers"] == 40
    assert len(ov["compress_ratios"]) == 43
    # 2 SWA (0), 18 CSA (2), 20 HCA (1), 3 drafter SWA (0)
    assert ov["compress_ratios"][:2] == [0, 0]
    assert ov["compress_ratios"][2:20] == [2] * 18
    assert ov["compress_ratios"][20:40] == [1] * 20
    assert ov["compress_ratios"][40:] == [0, 0, 0]
    assert ov["kv_source_layer_ids"] == [2, 8, 14, 20]
    assert ov["index_source_layer_ids"] == [2, 8, 14, 20, 24, 28, 32, 36]
    assert ov["candidate_source_layer_id"] == 20
    assert ov["dspark_target_layer_ids"] == [37, 38, 39]


def test_write_single_pass_stub_config(tmp_path: Path):
    cfg_path = iket_profile.write_single_pass_stub_config(tmp_path / "stub.config.json", 32)
    data = json.loads(cfg_path.read_text())
    assert data == {
        "meta": {"version": 0},
        "configs": [
            {
                "kernel": "__iket_native_dump_stub__",
                "maxTsCntPerWarp": 32,
                "instrumentations": [],
            }
        ],
    }
    with pytest.raises(ValueError, match="must be > 0"):
        iket_profile.write_single_pass_stub_config(tmp_path / "bad.json", 0)


def test_patch_injection_lib_skip_raw_data(tmp_path: Path):
    fake_root = tmp_path / "orig_iket"
    prof_dir = fake_root / "profiler"
    prof_dir.mkdir(parents=True)
    fake_so = prof_dir / "libsmodel_injection.so"
    fake_cubin_so = fake_root / "libiket_cubin_info.so"
    fake_cubin_so.write_bytes(b"FAKE_CUBIN_INFO")

    buf = bytearray(0xAC700)
    for offset, expected_orig, _ in iket_profile._RAW_DATA_PATCH_SITES:
        struct.pack_into("<I", buf, offset, expected_orig)
    fake_so.write_bytes(bytes(buf))

    out_dir = tmp_path / "out"
    patched_so = iket_profile.patch_injection_lib_skip_raw_data(out_dir, source_lib=fake_so)
    assert patched_so.exists()
    assert (out_dir / "injection" / "libiket_cubin_info.so").is_symlink()

    patched_bytes = patched_so.read_bytes()
    for offset, _, patched_insn in iket_profile._RAW_DATA_PATCH_SITES:
        (actual,) = struct.unpack_from("<I", patched_bytes, offset)
        assert actual == patched_insn

    # Idempotent when given an already-patched library
    patched_again = iket_profile.patch_injection_lib_skip_raw_data(tmp_path / "out2", source_lib=patched_so)
    assert patched_again.read_bytes() == patched_bytes

    # Rejects a mismatched binary
    corrupt_so = prof_dir / "corrupt.so"
    struct.pack_into("<I", buf, 0xAAA94, 0xDEADBEEF)
    corrupt_so.write_bytes(bytes(buf))
    with pytest.raises(RuntimeError, match="Unexpected instruction at 0xaaa94"):
        iket_profile.patch_injection_lib_skip_raw_data(tmp_path / "out3", source_lib=corrupt_so)


def test_max_static_events_from_sources(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "k1.abc.json").write_text(
        json.dumps({"kernel": "k1", "sites": [{"events_per_warp_bound": 8}, {"events_per_warp_bound": 14}]})
    )
    (sources / "k2.def.json").write_text(
        json.dumps({"kernel": "k2", "sites": [{"events_per_warp_bound": 22}, {"events_per_warp_bound": None}]})
    )
    assert iket_profile.max_static_events_from_sources(sources) == 22


def test_single_pass_and_pool_gb_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_run_iket(argv, timeout, log):
        calls.append(list(argv))
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("ok\n")
        return 0, "ok\n"

    monkeypatch.setattr(iket_profile, "run_iket", fake_run_iket)
    out_dir = tmp_path / "sp_out"
    with pytest.raises(SystemExit) as exc:
        iket_profile.main(
            [
                str(out_dir),
                "--single-pass-max-ts", "32",
                "--pool-gb", "1.5",
                "--deduct-kv-bytes",
                "--kv-safety-mib", "256",
                "--",
                "python3", "iket_step.py", "on", "--kv-bytes", str(4 << 30),
            ]
        )
    assert exc.value.code == 0
    assert len(calls) == 1
    cmd = calls[0]
    assert "--context-buffer-size" in cmd
    assert cmd[cmd.index("--context-buffer-size") + 1] == "1536M"
    assert "--use-config" in cmd
    # KV cache deduction subtracted 1.5 GiB + 256 MiB = 1.75 GiB from 4 GiB -> 2.25 GiB
    kv_idx = cmd.index("--kv-bytes")
    assert int(cmd[kv_idx + 1]) == (4 << 30) - ((3 << 29) + (256 << 20))
    pool_json = json.loads((out_dir / "pool.json").read_text())
    assert pool_json["single_pass"] is True
    assert pool_json["max_ts_cnt_per_warp"] == 32


def test_step_bundle_full_layers_and_single_pass():
    spec = step_bundle.step_spec(
        "HEAD",
        tp=4,
        requests=1,
        tokens=8,
        ctas=1,
        iters=1,
        warps=1,
        granularity="phase",
        payloads=True,
        single_pass_max_ts=32,
        skip_raw_data_dump=True,
        pool_gb=1.0,
        full_layers=True,
        lean_capture=True,
        no_warmup_request=True,
    )
    assert spec["env"]["VLLM_LATE_ADMISSION_LAYER"] == "30"
    iket_cmd = spec["steps"][0][1]
    assert "--single-pass-max-ts" in iket_cmd
    assert "--skip-raw-data-dump" in iket_cmd
    assert "--full-layers" in iket_cmd
    assert "--lean-capture" in iket_cmd
    assert "--no-warmup-request" in iket_cmd


def test_same_dummy_weights_source_uses_cudamemset():
    src = Path(iket_step.__file__).read_text()
    assert "cudaMemsetAsync" in src
    assert "FULL_TEXT_OVERRIDES" in src
    assert "lean_profile_run" in src
    assert "lean_profile_dsv4_attn" in src
    assert "flashinfer_autotune" in src


