import argparse
import json
from pathlib import Path

import pytest

import iket_perfetto
import iket_table


def _make_launch(name, start, end, grid=(2, 1, 1), block=(128, 1, 1), loc_base=0, ranges=None):
    return {
        "kernelName": name,
        "gridDimX": grid[0], "gridDimY": grid[1], "gridDimZ": grid[2],
        "blockDimX": block[0], "blockDimY": block[1], "blockDimZ": block[2],
        "warpLifetimes": [
            {"locIdx": loc_base, "startTs": start, "endTs": end},
            {"locIdx": loc_base + 1, "startTs": start + 100, "endTs": end},
        ],
        "ranges": ranges or [],
    }


def _write_trace(path: Path, graph_launches, eager_launches, strings=None, locations=None):
    if locations is None:
        locations = [
            {"smId": 0, "ctaId": [0, 0, 0], "warpId": 0},
            {"smId": 0, "ctaId": [0, 0, 0], "warpId": 1},
            {"smId": 1, "ctaId": [1, 0, 0], "warpId": 0},
            {"smId": 1, "ctaId": [1, 0, 0], "warpId": 1},
        ]
    data = {
        "graphLaunches": {"graph_exec_1:0": graph_launches},
        "launches": eager_launches,
        "locationTable": locations,
        "stringTable": strings or [],
    }
    path.write_text(json.dumps(data))
    return path


def test_base_iket_and_multi_slice_stitching(tmp_path):
    # Base run-iket --warp-trace (no ranges, real kernel durations: MLA=10us, MoE=20us)
    base_path = _write_trace(
        tmp_path / "base.json",
        graph_launches=[
            _make_launch("sparse_mla_kernel", 10_000, 20_000, loc_base=0),
            _make_launch("moe_grouped_fp8_fp4_gemm_kernel", 22_000, 42_000, loc_base=2),
            _make_launch("moe_combine_kernel", 43_000, 48_000, loc_base=0),
        ],
        eager_launches=[
            _make_launch("iket_step_marker_kernel", 1_000, 2_000),
            _make_launch("prepare_pos_seq_lens_kernel", 5_000, 6_000),
            _make_launch("iket_step_marker_kernel", 90_000, 91_000),
        ],
    )

    # Slice 1: MLA instrumented (stretched 2x to 20us), MoE skipped
    slice_mla = _write_trace(
        tmp_path / "slice_mla.json",
        graph_launches=[
            _make_launch(
                "sparse_mla_kernel", 10_000, 30_000, loc_base=0,
                ranges=[
                    {"warpLocIdxs": [0], "startTs": 10_000, "endTs": 18_000, "rangeNameIdx": 0},
                    {"warpLocIdxs": [0], "startTs": 18_000, "endTs": 30_000, "rangeNameIdx": 1},
                ],
            ),
            _make_launch("moe_grouped_fp8_fp4_gemm_kernel", 32_000, 52_000, loc_base=2),
            _make_launch("moe_combine_kernel", 53_000, 58_000, loc_base=0),
        ],
        eager_launches=[
            _make_launch("iket_step_marker_kernel", 1_000, 2_000),
            _make_launch("prepare_pos_seq_lens_kernel", 5_000, 6_000),
            _make_launch("iket_step_marker_kernel", 90_000, 91_000),
        ],
        strings=["load.tma_copy", "compute.mma_ss"],
    )

    # Slice 2: MoE instrumented (stretched 1.5x to 30us), MLA skipped
    slice_moe = _write_trace(
        tmp_path / "slice_moe.json",
        graph_launches=[
            _make_launch("sparse_mla_kernel", 10_000, 20_000, loc_base=0),
            _make_launch(
                "moe_grouped_fp8_fp4_gemm_kernel", 22_000, 52_000, loc_base=2,
                ranges=[
                    {"warpLocIdxs": [2], "startTs": 22_000, "endTs": 37_000, "rangeNameIdx": 0},
                    {"warpLocIdxs": [2], "startTs": 37_000, "endTs": 52_000, "rangeNameIdx": 1},
                ],
            ),
            _make_launch("moe_combine_kernel", 53_000, 58_000, loc_base=0),
        ],
        eager_launches=[
            _make_launch("iket_step_marker_kernel", 1_000, 2_000),
            _make_launch("prepare_pos_seq_lens_kernel", 5_000, 6_000),
            _make_launch("iket_step_marker_kernel", 90_000, 91_000),
        ],
        strings=["wait.mbarrier", "compute.mma_ts"],
    )

    out_pftrace = tmp_path / "stitched.pftrace"
    args = argparse.Namespace(
        records=str(slice_mla),
        out=str(out_pftrace),
        step=0,
        layer=None,
        title="test",
        max_traced_warps=0,
        base_iket=str(base_path),
        base_nsys=None,
        slice=[str(slice_moe)],
        allow_mismatch=False,
        rank=[],
    )
    summary = iket_perfetto.write(args)
    assert out_pftrace.exists() and summary["bytes"] > 0
    assert summary["launches"] == 4  # prepare + mla + moe_gemm + moe_combine
    assert summary["traced warps"] == 2  # 1 from MLA + 1 from MoE
    assert summary["rebased"]["base kind"] == "iket-warp-trace"
    assert summary["rebased"]["slices"][0]["instrumented_launches_stitched"] == 1


def test_slice_sequence_mismatch_refuses_unless_allowed(tmp_path):
    base_path = _write_trace(
        tmp_path / "base.json",
        graph_launches=[_make_launch("sparse_mla_kernel", 10_000, 20_000)],
        eager_launches=[
            _make_launch("iket_step_marker_kernel", 1_000, 2_000),
            _make_launch("prepare_pos_seq_lens_kernel", 5_000, 6_000),
            _make_launch("iket_step_marker_kernel", 90_000, 91_000),
        ],
    )
    bad_slice = _write_trace(
        tmp_path / "bad_slice.json",
        graph_launches=[
            _make_launch("sparse_mla_kernel", 10_000, 20_000),
            _make_launch("extra_unexpected_kernel", 25_000, 30_000),
        ],
        eager_launches=[
            _make_launch("iket_step_marker_kernel", 1_000, 2_000),
            _make_launch("prepare_pos_seq_lens_kernel", 5_000, 6_000),
            _make_launch("iket_step_marker_kernel", 90_000, 91_000),
        ],
    )
    with pytest.raises(SystemExit, match="step launch sequence differs"):
        iket_perfetto.load_step(
            str(base_path), 0, None, slice_paths=[str(bad_slice)], allow_mismatch=False
        )
    # Succeeds when allow_mismatch=True
    launches, *_ = iket_perfetto.load_step(
        str(base_path), 0, None, slice_paths=[str(bad_slice)], allow_mismatch=True
    )
    assert len(launches) == 2


def test_multirank_collective_clock_alignment(tmp_path):
    # True timeline (in rank 0 ns):
    # Collective 0 exits at t = 100_000 ns (100 us)
    # Collective 1 exits at t = 1_100_000 ns (1100 us)
    # Rank 2 (dgx3) arrives 18_000 ns (18 us) late at Collective 0 (at 88_000 ns instead of 70_000 ns).
    rank_configs = [
        ("dgx1", 0, 1.0, 0),
        ("dgx2", 150_000, 1.000010, 2_000),   # +150 us offset, +10 ppm drift
        ("dgx3", -420_000, 0.999985, 18_000), # -420 us offset, -15 ppm drift, 18 us late
        ("dgx4", 800_000, 1.000020, 4_000),   # +800 us offset, +20 ppm drift
    ]
    t_ref0 = 100_000
    rank_specs = []
    for label, offset_ns, beta_inv, late_ns in rank_configs:
        # Inverse map from ref to rank clock: t_rank = t_ref0 + offset_ns + (t_ref - t_ref0) / beta_inv
        to_rank = lambda t_ref, off=offset_ns, b=beta_inv: int(round(t_ref0 + off + (t_ref - t_ref0) / b))
        p = _write_trace(
            tmp_path / f"{label}.json",
            graph_launches=[
                _make_launch(
                    "oneshot_allreduce_kernel",
                    to_rank(70_000 + late_ns),
                    to_rank(100_000),
                    ranges=[
                        {
                            "warpLocIdxs": [0],
                            "startTs": to_rank(72_000 + late_ns),
                            "endTs": to_rank(100_000),
                            "rangeNameIdx": 0,
                        }
                    ],
                ),
                _make_launch("moe_combine_kernel", to_rank(500_000), to_rank(550_000)),
                _make_launch(
                    "oneshot_allreduce_kernel",
                    to_rank(1_050_000),
                    to_rank(1_100_000),
                    ranges=[
                        {
                            "warpLocIdxs": [0],
                            "startTs": to_rank(1_060_000),
                            "endTs": to_rank(1_100_000),
                            "rangeNameIdx": 0,
                        }
                    ],
                ),
            ],
            eager_launches=[
                _make_launch("iket_step_marker_kernel", to_rank(1_000), to_rank(2_000)),
                _make_launch("prepare_pos_seq_lens_kernel", to_rank(10_000), to_rank(12_000)),
                _make_launch("iket_step_marker_kernel", to_rank(2_000_000), to_rank(2_001_000)),
            ],
            strings=["wait.flag"],
        )
        rank_specs.append(f"{label}={p}")

    out_pftrace = tmp_path / "multirank.pftrace"
    args = argparse.Namespace(
        records=None,
        out=str(out_pftrace),
        step=0,
        layer=None,
        title="tp4",
        max_traced_warps=0,
        base_iket=None,
        base_nsys=None,
        slice=[],
        allow_mismatch=False,
        rank=rank_specs,
    )
    summary = iket_perfetto.write(args)
    assert out_pftrace.exists() and summary["bytes"] > 0
    assert summary["ranks"] == 4
    assert summary["collectives_aligned"] == 2
    for fit in summary["rank_alignment"]:
        assert fit["residual_rms_us"] < 0.01

    loaded, _, coll_events = iket_perfetto.align_ranks(rank_specs, 0, None)
    assert coll_events[0]["late_rank"] == "dgx3"
    assert abs(coll_events[0]["arrival_spread_us"] - 18.0) < 0.05


def test_iket_table_slice_merge_and_payload_bandwidth(tmp_path):
    base_trace = _write_trace(
        tmp_path / "base_table.json",
        graph_launches=[
            _make_launch("sparse_mla_kernel", 10_000, 20_000, loc_base=0),
        ],
        eager_launches=[],
    )
    slice_trace = _write_trace(
        tmp_path / "slice_table.json",
        graph_launches=[
            _make_launch(
                "sparse_mla_kernel", 10_000, 30_000, loc_base=0,
                ranges=[
                    {"warpLocIdxs": [0], "startTs": 10_000, "endTs": 20_000, "rangeNameIdx": 0},
                    {"warpLocIdxs": [1], "startTs": 10_000, "endTs": 20_000, "rangeNameIdx": 0},
                ],
            ),
        ],
        eager_launches=[],
        strings=["load.tma_copy*2"],
    )
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "sparse_mla_kernel.1234.json").write_text(json.dumps({
        "kernel": "sparse_mla_kernel",
        "ctas": 1,
        "iters": 1,
        "warps": 2,
        "sites": [
            {
                "kernel": "sparse_mla_kernel",
                "range": "load.tma_copy*2",
                "phase": "load",
                "what": "tma_copy",
                "statement": "tl::tma_load(desc, mbar, smem);",
                "loops": [{"var": "k", "start": 0, "stop": 2, "step": 1, "bound": 2}],
                "tma_bytes": 16384,
            }
        ],
    }))

    base_data = json.loads(base_trace.read_text())
    slice_data = json.loads(slice_trace.read_text())
    launches = iket_table.all_launches(base_data)
    iket_table.merge_slice_launches(
        launches, base_data["stringTable"], base_data["locationTable"], [slice_data]
    )
    statements, sampling, payload_meta = iket_table.sites(sources_dir)
    groups = iket_table.summarize(
        launches, base_data["stringTable"], base_data["locationTable"],
        sampling=sampling, payload_meta=payload_meta,
    )
    assert len(groups) == 1
    g = groups[0]
    assert abs(g["span_us"] - 10.0) < 1e-6  # preserved base span (10us, not slice's 20us)
    r = g["ranges"]["load.tma_copy*2"]
    assert abs(r["us"] - 10.0) < 1e-6  # 5us rescaled * factor 2 = 10us
    assert r["tma_bytes"] == 16384
    assert r["bytes_per_warp"] == 32768.0
    assert abs(r["gbps_per_warp"] - 3.277) < 0.01
