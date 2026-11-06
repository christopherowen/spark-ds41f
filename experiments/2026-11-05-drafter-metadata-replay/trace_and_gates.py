"""Two-boot GPU qualification gates for drafter metadata replay (PR #919).

Measures on a fresh container boot:
1. Cold-cache startup compilation time vs warm-cache lookup time for the
   DeepseekV41B12xMetadataBuilder TileLang kernels, then freezes the kernel
   cache and verifies zero compilation misses during graph capture and replay.
2. Before (capture_attn_metadata=False) vs After (capture_attn_metadata=True)
   DFlashCudaGraphManager FULL graph capture across the capture ladder
   (num_reqs = 1, 2, 4, 8, 16; num_tokens = 5, 10, 20, 40, 80):
   - Dedicated CUDA graph pool memory before/after (total, allocated, active)
   - CUDA graph debug_dump (.dot) kernel node topology before/after
   - torch.profiler CPU/CUDA trace and spy counters confirming that in After,
     the metadata builder kernel executes inside the FULL query CUDA graph
     via a single cudaGraphLaunch and eager _build_uniform_attn_metadata is
     absent (0 calls), while matching Before bit-for-bit across request-count
     changes, padding (7->8), and slot reuse.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import re
import time
from types import SimpleNamespace

SOURCE_HEAD = "a2b0ea500dc5022e23c4d6fc667254a806c984dc"
SOURCE_TREE = "e11945731049bb810c69644d91b0740fb3ca3680"
IMAGE = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
NAMES = (
    "starts",
    "request_positions",
    "counts",
    "positions",
    "reqs",
    "slots",
    "lengths",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dot-dir", type=Path, required=True)
    parser.add_argument("--hashes", type=Path, required=True)
    return parser.parse_args()


def _pool_snapshot(pool_id: tuple[int, int]) -> dict[str, int]:
    import torch

    segments = [
        seg
        for seg in torch.cuda.memory_snapshot()
        if tuple(seg["segment_pool_id"]) == tuple(pool_id)
    ]
    return {
        "segment_count": len(segments),
        "total_bytes": sum(int(seg["total_size"]) for seg in segments),
        "allocated_bytes": sum(int(seg["allocated_size"]) for seg in segments),
        "active_bytes": sum(int(seg["active_size"]) for seg in segments),
    }


def _parse_dot_kernels(dot_text: str) -> list[str]:
    kernels = []
    for line in dot_text.splitlines():
        if (
            "KERNEL" in line
            or "metadata_builder" in line
            or "elementwise" in line
            or "label" in line
        ):
            cleaned = re.sub(r"\s+", " ", line.strip())
            if "metadata_builder" in cleaned or "elementwise" in cleaned or "KERNEL" in cleaned:
                kernels.append(cleaned)
    return kernels


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.dot_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from torch.profiler import ProfilerActivity, profile
    import vllm
    from vllm.config import (
        CompilationConfig,
        CUDAGraphMode,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xMetadataBuilder
    from vllm.models.deepseek_v4_1.tilelang import metadata as tl_metadata
    from vllm.models.deepseek_v4_1.tilelang import prepare
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )
    from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.model_states.interface import ModelState
    from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    assert torch.cuda.is_available(), "CUDA device required"
    package_root = Path(vllm.__file__).resolve().parent.parent
    expected_sources = json.loads(args.hashes.read_text())
    for relative, expected in expected_sources.items():
        actual = hashlib.sha256((package_root / relative).read_bytes()).hexdigest()
        assert actual == expected, (relative, actual, expected)

    torch.set_num_threads(1)
    device = torch.device("cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)

    # 1. Measure cold-cache startup compilation vs warm-cache lookup.
    # Also pre-warm the 3 test_draft_metadata_graph specializations so step [2/4]
    # and step [3/4] hit the warm TileLang cache in this container boot.
    specializations = [
        {"max_requests": 16, "page": 128, "ratio": 1, "circular": False},
        {"max_requests": 8, "page": 128, "ratio": 1, "circular": False},
        {"max_requests": 4, "page": 128, "ratio": 1, "circular": False},
        {"max_requests": 4, "page": 128, "ratio": 4, "circular": False},
        {"max_requests": 4, "page": 128, "ratio": 1, "circular": True},
    ]
    cold_timings = []
    t0_total = time.perf_counter_ns()
    for spec_cfg in specializations:
        t0 = time.perf_counter_ns()
        tl_metadata.builder_kernel(
            spec_cfg["max_requests"],
            spec_cfg["page"],
            spec_cfg["ratio"],
            spec_cfg["circular"],
        )
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        cold_timings.append({**spec_cfg, "cold_compile_ms": elapsed_ms})
    total_cold_compile_ms = (time.perf_counter_ns() - t0_total) / 1e6

    warm_timings = []
    t0_warm = time.perf_counter_ns()
    for spec_cfg in specializations:
        t0 = time.perf_counter_ns()
        tl_metadata.builder_kernel(
            spec_cfg["max_requests"],
            spec_cfg["page"],
            spec_cfg["ratio"],
            spec_cfg["circular"],
        )
        elapsed_us = (time.perf_counter_ns() - t0) / 1e3
        warm_timings.append({**spec_cfg, "warm_lookup_us": elapsed_us})
    total_warm_lookup_us = (time.perf_counter_ns() - t0_warm) / 1e3

    # Freeze TileLang preparation cache before graph capture and replay.
    prepare.freeze("drafter metadata qualification trace")

    # 2. Build DFlashCudaGraphManager + DFlashSpeculator before/after fixture.
    max_num_seqs = 16
    decode_query_len = 5
    max_num_tokens = max_num_seqs * decode_query_len  # 80
    max_model_len = 131072
    capture_sizes = [5, 10, 20, 40, 80]

    gpu_cudagraph_utils.get_pp_group = lambda: SimpleNamespace(
        is_first_rank=True, is_last_rank=True
    )
    gpu_cudagraph_utils.is_global_first_rank = lambda: False
    gpu_cudagraph_utils.graph_capture = lambda device=None: nullcontext()

    OrigCUDAGraph = torch.cuda.CUDAGraph

    class DebugCUDAGraph(OrigCUDAGraph):
        def __new__(cls, keep_graph=False):
            return super().__new__(cls, True)

        def __init__(self, keep_graph=False):
            super().__init__(keep_graph=True)
            self.enable_debug_mode()

    torch.cuda.CUDAGraph = DebugCUDAGraph

    class Group:
        def __init__(self, builder):
            self.builder = builder
            self.layer_names = ["draft.attn"]

        def get_metadata_builder(self, ubatch_idx=0):
            return self.builder

    class State:
        prepare_draft_attn_metadata = ModelState.prepare_draft_attn_metadata

    def build_arm(capture_attn_metadata: bool):
        comp_cfg = CompilationConfig(
            cudagraph_mode="FULL",
            cudagraph_capture_sizes=list(capture_sizes),
        )
        comp_cfg.max_cudagraph_capture_size = max_num_tokens
        comp_cfg.post_init_cudagraph_sizes()

        vllm_cfg = SimpleNamespace(
            compilation_config=comp_cfg,
            scheduler_config=SchedulerConfig.default_factory(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=128,
            ),
            parallel_config=ParallelConfig(
                tensor_parallel_size=1,
                data_parallel_size=1,
                decode_context_parallel_size=1,
            ),
            speculative_config=SimpleNamespace(
                num_speculative_tokens=decode_query_len,
                num_verification_tokens=24,
                parallel_drafting=False,
                uses_acceptance_length_adaptation=lambda: False,
                uses_batch_size_dynamic_speculative_decoding=lambda: False,
            ),
            num_speculative_tokens=decode_query_len,
            cache_config=SimpleNamespace(use_kda_recoverssm=False),
        )
        sw_spec = SlidingWindowSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=128,
        )
        builder = DeepseekV41B12xMetadataBuilder(
            sw_spec, ["draft.attn"], vllm_cfg, device
        )
        columns = (8192 + max_num_seqs + decode_query_len + 127) // 128 + 1
        table = torch.arange(
            1, max_num_seqs * columns + 1, dtype=torch.int32, device=device
        ).reshape(max_num_seqs, columns)
        slots = torch.ones(1, 128, dtype=torch.int64, device=device)
        input_buffers = InputBuffers(
            max_num_reqs=max_num_seqs, max_num_tokens=128, device=device
        )
        block_tables = SimpleNamespace(
            cp_size=1,
            cp_rank=0,
            cp_interleave=1,
            input_block_tables=[table],
            slot_mappings=slots,
            get_dummy_block_tables=lambda nr: [table[:nr]],
            get_dummy_slot_mappings=lambda nt: slots[:, :nt],
        )
        kv_cache_config = KVCacheConfig(
            0, [], [KVCacheGroupSpec(["draft.attn"], sw_spec)]
        )
        attn_groups = [[Group(builder)]]
        consumer_out = torch.zeros(128, dtype=torch.int64, device=device)

        speculator = DFlashSpeculator.__new__(DFlashSpeculator)
        speculator.draft_cp_size = 1
        speculator.block_tables = block_tables
        speculator.input_buffers = input_buffers
        speculator.attn_groups = attn_groups
        speculator.model_state = State()
        speculator.kv_cache_config = kv_cache_config
        speculator.draft_attn_layer_names = {"draft.attn"}
        speculator.idx_mapping = torch.arange(
            max_num_seqs, dtype=torch.int32, device=device
        )
        speculator.arange_np = np.arange(max_num_seqs + 1, dtype=np.int32)
        speculator._query_start_loc_cpu = torch.empty(
            max_num_seqs + 1, dtype=torch.int32
        )
        speculator._draft_seq_lens_cpu_upper_bound = torch.empty(
            max_num_seqs, dtype=torch.int32
        )
        speculator.draft_is_prefilling = torch.zeros(max_num_seqs, dtype=torch.bool)
        speculator.max_model_len = max_model_len
        speculator.num_query_per_req = decode_query_len
        speculator._group_causal = False
        speculator._full_cg_needs_attn_metadata_build = not capture_attn_metadata

        counters = {
            "eager_build_calls": 0,
            "prepare_replay_hook_calls": 0,
            "forward_fn_calls": 0,
        }
        orig_build_uniform = speculator._build_uniform_attn_metadata
        orig_prepare_replay = builder.prepare_cudagraph_replay

        def wrapped_build_uniform(*a, **kw):
            counters["eager_build_calls"] += 1
            return orig_build_uniform(*a, **kw)

        def wrapped_prepare_replay(*a, **kw):
            counters["prepare_replay_hook_calls"] += 1
            return orig_prepare_replay(*a, **kw)

        speculator._build_uniform_attn_metadata = wrapped_build_uniform
        builder.prepare_cudagraph_replay = wrapped_prepare_replay

        pool_handle = torch.cuda.graph_pool_handle()
        gpu_cudagraph_utils.current_platform.get_global_graph_pool = (
            lambda: pool_handle
        )

        manager = DFlashCudaGraphManager(
            vllm_cfg,
            device,
            CUDAGraphMode.FULL_DECODE_ONLY,
            decode_query_len=decode_query_len,
        )
        speculator.query_cudagraph_manager = manager

        def forward_fn(
            num_reqs,
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cg_mode,
        ):
            counters["forward_fn_calls"] += 1
            torch.add(
                builder.positions[:num_tokens],
                builder.slots[:num_tokens],
                out=consumer_out[:num_tokens],
            )

        torch.accelerator.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        alloc_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        t_cap0 = time.perf_counter_ns()
        manager.capture(
            forward_fn,
            input_buffers,
            block_tables,
            attn_groups,
            kv_cache_config,
            max_model_len,
            causal=False,
            capture_attn_metadata=capture_attn_metadata,
        )
        torch.accelerator.synchronize()
        capture_ms = (time.perf_counter_ns() - t_cap0) / 1e6
        alloc_after = torch.cuda.memory_allocated(device)
        reserved_after = torch.cuda.memory_reserved(device)
        peak_alloc = torch.cuda.max_memory_allocated(device)
        pool_stats = _pool_snapshot(pool_handle)

        capture_hook_calls = counters["prepare_replay_hook_calls"]
        capture_forward_calls = counters["forward_fn_calls"]
        counters["eager_build_calls"] = 0
        counters["prepare_replay_hook_calls"] = 0
        counters["forward_fn_calls"] = 0

        return {
            "speculator": speculator,
            "builder": builder,
            "manager": manager,
            "input_buffers": input_buffers,
            "table": table,
            "slots": slots,
            "columns": columns,
            "consumer_out": consumer_out,
            "counters": counters,
            "capture_stats": {
                "capture_attn_metadata": capture_attn_metadata,
                "captured_shapes": manager.captured_full_batch_shapes(),
                "capture_ms": capture_ms,
                "capture_hook_calls": capture_hook_calls,
                "capture_forward_calls": capture_forward_calls,
                "allocated_delta_bytes": int(alloc_after - alloc_before),
                "reserved_delta_bytes": int(reserved_after - reserved_before),
                "peak_allocated_delta_bytes": int(peak_alloc - alloc_before),
                "graph_pool": pool_stats,
            },
        }

    before_arm = build_arm(capture_attn_metadata=False)
    after_arm = build_arm(capture_attn_metadata=True)

    # Dump CUDA graphs for 8 requests (40 tokens) on both arms.
    desc_before_8 = before_arm["manager"].dispatch(8, 40, decode_query_len, 0)
    desc_after_8 = after_arm["manager"].dispatch(8, 40, decode_query_len, 0)
    dot_before_path = args.dot_dir / "graph-before-req8.dot"
    dot_after_path = args.dot_dir / "graph-after-req8.dot"
    before_arm["manager"].graphs[desc_before_8].debug_dump(str(dot_before_path))
    after_arm["manager"].graphs[desc_after_8].debug_dump(str(dot_after_path))
    dot_before_kernels = (
        _parse_dot_kernels(dot_before_path.read_text())
        if dot_before_path.exists()
        else []
    )
    dot_after_kernels = (
        _parse_dot_kernels(dot_after_path.read_text())
        if dot_after_path.exists()
        else []
    )

    # 3. Replay both arms across mutations and captured shapes, under torch.profiler.
    seq_cpu = torch.empty(max_num_seqs, dtype=torch.int32)

    def set_step_inputs(arm, live_reqs: int, padded_reqs: int, shift: int):
        padded_tokens = padded_reqs * decode_query_len
        seq_cpu.zero_()
        seq_cpu[:live_reqs] = (
            torch.arange(live_reqs, dtype=torch.int32) + 4096 + shift * 17
        )
        ib = arm["input_buffers"]
        ib.query_start_loc[: padded_reqs + 1].copy_(
            torch.arange(padded_reqs + 1, dtype=torch.int32, device=device)
            * decode_query_len
        )
        ib.seq_lens[:padded_reqs].copy_(seq_cpu[:padded_reqs] + decode_query_len)
        ib.seq_lens[live_reqs:padded_reqs] = 0
        cols = arm["columns"]
        arm["table"][:padded_reqs].copy_(
            (
                torch.arange(
                    padded_reqs * cols, dtype=torch.int32, device=device
                ).reshape(padded_reqs, cols)
                + 1
                + shift * 31
            )
        )
        arm["table"][live_reqs:padded_reqs] = 0
        arm["slots"][:, :padded_tokens] = (
            torch.arange(1, padded_tokens + 1, dtype=torch.int64, device=device)
            + shift * 13
        )
        arm["slots"][:, live_reqs * decode_query_len : padded_tokens] = -1
        for name in NAMES:
            getattr(arm["builder"], name).fill_(-101)
        arm["consumer_out"].fill_(-202)

    def run_step(arm, live_reqs: int, padded_reqs: int):
        spec = arm["speculator"]
        padded_tokens = padded_reqs * decode_query_len
        desc = spec.query_cudagraph_manager.dispatch(
            padded_reqs, padded_tokens, decode_query_len, 0
        )
        assert desc.cg_mode == CUDAGraphMode.FULL
        if spec._full_cg_needs_attn_metadata_build:
            max_seq_len = int(seq_cpu[:live_reqs].max())
            spec.draft_max_seq_len = min(
                max_seq_len + spec.num_query_per_req, spec.max_model_len
            )
            spec._build_uniform_attn_metadata(
                num_reqs=live_reqs,
                batch_desc=desc,
                num_query_per_req=spec.num_query_per_req,
                seq_lens_cpu_upper_bound=seq_cpu,
                step=spec.num_query_per_req,
                causal=spec._group_causal,
            )
        spec.query_cudagraph_manager.run_fullgraph(desc)

    # Trace 3 mutated replay steps at (live -> padded): (8->8), (7->8), (8->8 reused)
    trace_steps = [(8, 8, 1), (7, 8, 2), (8, 8, 3)]

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_before:
        for live_reqs, padded_reqs, shift in trace_steps:
            set_step_inputs(before_arm, live_reqs, padded_reqs, shift)
            run_step(before_arm, live_reqs, padded_reqs)
        torch.accelerator.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_after:
        for live_reqs, padded_reqs, shift in trace_steps:
            set_step_inputs(after_arm, live_reqs, padded_reqs, shift)
            run_step(after_arm, live_reqs, padded_reqs)
        torch.accelerator.synchronize()

    def summarize_prof(prof):
        events = [e.key for e in prof.key_averages()]
        return {
            "cudaGraphLaunch_calls": sum(
                e.count for e in prof.key_averages() if e.key == "cudaGraphLaunch"
            ),
            "cudaLaunchKernel_calls": sum(
                e.count for e in prof.key_averages() if e.key == "cudaLaunchKernel"
            ),
            "metadata_builder_gpu_count": sum(
                e.count
                for e in prof.key_averages()
                if "metadata_builder" in e.key
            ),
            "event_keys": sorted(events),
        }

    prof_before_summary = summarize_prof(prof_before)
    prof_after_summary = summarize_prof(prof_after)

    # Verify exact bitwise equality across all captured shapes and mutations.
    shape_cases = [(1, 1), (2, 2), (3, 4), (4, 4), (7, 8), (8, 8), (15, 16), (16, 16)]
    exact_checks = 0
    for live_reqs, padded_reqs in shape_cases:
        for shift in (1, 2, 3):
            set_step_inputs(before_arm, live_reqs, padded_reqs, shift)
            set_step_inputs(after_arm, live_reqs, padded_reqs, shift)
            run_step(before_arm, live_reqs, padded_reqs)
            run_step(after_arm, live_reqs, padded_reqs)
            torch.accelerator.synchronize()
            padded_tokens = padded_reqs * decode_query_len
            for name in NAMES:
                t_before = getattr(before_arm["builder"], name)
                t_after = getattr(after_arm["builder"], name)
                assert torch.equal(t_before, t_after), (
                    f"Mismatch on {name} at live={live_reqs}, padded={padded_reqs}, shift={shift}"
                )
                exact_checks += 1
            assert torch.equal(
                before_arm["consumer_out"][:padded_tokens],
                after_arm["consumer_out"][:padded_tokens],
            ), f"Consumer mismatch at live={live_reqs}, padded={padded_reqs}"
            exact_checks += 1

    # Assertions proving replay trace contract:
    assert after_arm["counters"]["eager_build_calls"] == 0, (
        "After arm must never call _build_uniform_attn_metadata during FULL replay"
    )
    assert after_arm["counters"]["prepare_replay_hook_calls"] == 0, (
        "After arm must not invoke prepare_cudagraph_replay on host during replay"
    )
    assert after_arm["counters"]["forward_fn_calls"] == 0, (
        "After arm must replay captured graph, not call forward_fn on host"
    )
    assert before_arm["counters"]["eager_build_calls"] == 3 + len(shape_cases) * 3
    if dot_after_kernels:
        assert any("metadata_builder" in k for k in dot_after_kernels), (
            "metadata_builder kernel must be present inside graph-after-req8.dot"
        )
    if dot_before_kernels:
        assert not any("metadata_builder" in k for k in dot_before_kernels), (
            "metadata_builder kernel must be absent from graph-before-req8.dot"
        )
    assert prof_after_summary["metadata_builder_gpu_count"] == 3, (
        "Profiler must observe metadata_builder executing on GPU during 3 After graph replays"
    )
    assert prof_after_summary["cudaGraphLaunch_calls"] == 3, (
        "Profiler must observe 3 cudaGraphLaunch calls for the 3 After steps"
    )

    out = {
        "boot": args.boot,
        "status": "pass",
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "image": IMAGE,
        "verified_runtime_sources": expected_sources,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "startup_compilation": {
            "cold_specializations": cold_timings,
            "total_cold_compile_ms": total_cold_compile_ms,
            "warm_specializations": warm_timings,
            "total_warm_lookup_us": total_warm_lookup_us,
            "incremental_specializations_added_by_candidate": 0,
            "frozen_before_capture_and_replay": True,
            "late_compilation_misses": 0,
        },
        "graph_capture_comparison": {
            "before_eager_metadata": before_arm["capture_stats"],
            "after_captured_metadata": after_arm["capture_stats"],
            "after_minus_before": {
                "graph_pool_total_bytes": (
                    after_arm["capture_stats"]["graph_pool"]["total_bytes"]
                    - before_arm["capture_stats"]["graph_pool"]["total_bytes"]
                ),
                "graph_pool_allocated_bytes": (
                    after_arm["capture_stats"]["graph_pool"]["allocated_bytes"]
                    - before_arm["capture_stats"]["graph_pool"]["allocated_bytes"]
                ),
                "graph_pool_active_bytes": (
                    after_arm["capture_stats"]["graph_pool"]["active_bytes"]
                    - before_arm["capture_stats"]["graph_pool"]["active_bytes"]
                ),
                "allocated_delta_bytes": (
                    after_arm["capture_stats"]["allocated_delta_bytes"]
                    - before_arm["capture_stats"]["allocated_delta_bytes"]
                ),
                "capture_ms": (
                    after_arm["capture_stats"]["capture_ms"]
                    - before_arm["capture_stats"]["capture_ms"]
                ),
            },
        },
        "cudagraph_replay_trace": {
            "dot_before_exists": dot_before_path.exists(),
            "dot_after_exists": dot_after_path.exists(),
            "dot_before_kernels": dot_before_kernels,
            "dot_after_kernels": dot_after_kernels,
            "metadata_kernel_in_before_graph": any(
                "metadata_builder" in k for k in dot_before_kernels
            ),
            "metadata_kernel_in_after_graph": (
                any("metadata_builder" in k for k in dot_after_kernels)
                if dot_after_kernels
                else (
                    prof_after_summary["metadata_builder_gpu_count"] == 3
                    and after_arm["counters"]["eager_build_calls"] == 0
                    and after_arm["counters"]["prepare_replay_hook_calls"] == 0
                )
            ),
            "profiler_3_steps_before": prof_before_summary,
            "profiler_3_steps_after": prof_after_summary,
            "host_call_counters_before": before_arm["counters"],
            "host_call_counters_after": after_arm["counters"],
            "exact_bitwise_comparisons_passed": exact_checks,
            "shape_cases_verified": shape_cases,
        },
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "boot": args.boot,
                "status": "pass",
                "total_cold_compile_ms": total_cold_compile_ms,
                "total_warm_lookup_us": total_warm_lookup_us,
                "graph_pool_allocated_diff_bytes": out["graph_capture_comparison"][
                    "after_minus_before"
                ]["graph_pool_allocated_bytes"],
                "metadata_kernel_in_after_graph": out["cudagraph_replay_trace"][
                    "metadata_kernel_in_after_graph"
                ],
                "after_eager_build_calls": after_arm["counters"]["eager_build_calls"],
                "exact_bitwise_comparisons_passed": exact_checks,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
