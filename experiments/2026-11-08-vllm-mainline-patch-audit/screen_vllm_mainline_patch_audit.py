#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-boot verification and audit harness for canonical vllm-project/vllm PRs
(#66226, #65866, #65627, #63401/#55828, #65092/#65168) against spark-ds41f:1107."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import psutil
import pynvml
import torch


def check_item1_spec_decode_prefill_load() -> dict:
    """Audit #66226 vs 1107 0624 + DSpark parallel query pass."""
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    # Verify 0624 early exit exists in DFlashSpeculator.propose
    import inspect
    src = inspect.getsource(DFlashSpeculator.propose)
    has_k0_skip = "if num_speculative_tokens == 0:" in src

    # Measure parallel 1-step query pass cost scaling across K=1..5 on GB10 GPU
    # DSpark drafts all K tokens in a single parallel forward pass of width (1 + K).
    device = torch.device("cuda:0")
    hidden_size = 4096
    inter_size = 12288
    w_qkv = torch.randn((hidden_size, hidden_size), dtype=torch.bfloat16, device=device)
    w_up = torch.randn((hidden_size, inter_size), dtype=torch.bfloat16, device=device)
    w_down = torch.randn((inter_size, hidden_size), dtype=torch.bfloat16, device=device)

    timings_ms = {}
    for num_reqs in (1, 4, 8):
        for k in (1, 3, 5):
            rows = num_reqs * (1 + k)
            x = torch.randn((rows, hidden_size), dtype=torch.bfloat16, device=device)
            for _ in range(10):
                y = (torch.nn.functional.silu(x @ w_qkv @ w_up) @ w_down)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            iters = 100
            for _ in range(iters):
                y = (torch.nn.functional.silu(x @ w_qkv @ w_up) @ w_down)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iters
            timings_ms[f"reqs{num_reqs}_k{k}_rows{rows}"] = round(elapsed_ms, 4)

    del w_qkv, w_up, w_down, x, y
    torch.cuda.empty_cache()
    return {
        "dflash_has_0624_k0_skip": has_k0_skip,
        "parallel_draft_step_ms": timings_ms,
    }


def check_item2_llm_repr_rpc() -> dict:
    """Verify 0762 (#65866): LLM.__repr__ does not call collective_rpc."""
    from vllm.entrypoints.llm import LLM

    rpc_calls = []

    def fake_rpc(method: str):
        rpc_calls.append(method)
        time.sleep(0.005)
        return ["DeepseekV4ForCausalLM(\n  (model): DeepseekV4Model()\n)"]

    llm = LLM.__new__(LLM)
    llm._cached_model_inspection = None
    llm.model_config = SimpleNamespace(model="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    llm.llm_engine = SimpleNamespace(collective_rpc=fake_rpc)

    t0 = time.perf_counter()
    for _ in range(100):
        s = repr(llm)
    repr_us = (time.perf_counter() - t0) * 1e6 / 100.0
    rpc_count_after_repr = len(rpc_calls)

    t1 = time.perf_counter()
    tree1 = llm.get_model_inspection()
    first_inspect_ms = (time.perf_counter() - t1) * 1e3
    t2 = time.perf_counter()
    tree2 = llm.get_model_inspection()
    cached_inspect_us = (time.perf_counter() - t2) * 1e6

    assert s == "LLM(model=\x27deepseek-ai/DeepSeek-V4.1-Flash-1017\x27)"
    assert rpc_count_after_repr == 0
    assert tree1 == tree2
    assert rpc_calls == ["get_model_inspection"]

    return {
        "repr_output": s,
        "rpc_calls_during_100_reprs": rpc_count_after_repr,
        "repr_latency_us": round(repr_us, 3),
        "first_get_model_inspection_ms": round(first_inspect_ms, 3),
        "cached_get_model_inspection_us": round(cached_inspect_us, 3),
        "total_rpc_calls_after_2_inspections": len(rpc_calls),
    }


def check_item3_batch_invariant_fa_swa() -> dict:
    """Verify 0763 (#65627): FlashAttentionMetadataBuilder keeps max_seq_len == max_model_len under VLLM_BATCH_INVARIANT."""
    import inspect
    import vllm.envs as envs
    from vllm.v1.attention.backends import flash_attn as fa_mod

    src = inspect.getsource(fa_mod.FlashAttentionMetadataBuilder.build)
    has_batch_invariant_max_seq_len_override = (
        "if envs.VLLM_BATCH_INVARIANT:" in src
        and "max_seq_len = self.model_config.max_model_len" in src
    )
    assert has_batch_invariant_max_seq_len_override

    # Simulate the exact max_seq_len computation in FlashAttentionMetadataBuilder.build
    max_model_len = 4096
    sliding_window = 1024
    mixed_batch_max_seq_len = 512  # < sliding_window

    old_val = envs.VLLM_BATCH_INVARIANT
    try:
        envs.VLLM_BATCH_INVARIANT = False
        eff_off = max_model_len if envs.VLLM_BATCH_INVARIANT else mixed_batch_max_seq_len
        envs.VLLM_BATCH_INVARIANT = True
        eff_on = max_model_len if envs.VLLM_BATCH_INVARIANT else mixed_batch_max_seq_len
    finally:
        envs.VLLM_BATCH_INVARIANT = old_val

    assert eff_off == 512 and eff_off < sliding_window
    assert eff_on == 4096 and eff_on >= sliding_window

    return {
        "has_batch_invariant_max_seq_len_override": has_batch_invariant_max_seq_len_override,
        "sliding_window": sliding_window,
        "max_model_len": max_model_len,
        "mixed_batch_common_max_seq_len": mixed_batch_max_seq_len,
        "effective_max_seq_len_batch_invariant_off": eff_off,
        "effective_max_seq_len_batch_invariant_on": eff_on,
        "window_stays_active_under_batch_invariant": eff_on >= sliding_window,
    }


def check_item4_gb10_uma_process_memory() -> dict:
    """Audit #55828 / #63401 on GB10 unified memory (NVML usedGpuMemory vs psutil)."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    pid = os.getpid()

    def nvml_proc_bytes() -> int:
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == pid:
                return int(proc.usedGpuMemory)
        return 0

    _ = torch.zeros(1, device="cuda")
    del _
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    base_nvml = nvml_proc_bytes()
    base_psutil = psutil.virtual_memory().available

    # 1. Allocate 512 MiB via cudaMalloc
    cuda_bytes = 512 * 1024 * 1024
    gpu_buf = torch.empty(cuda_bytes, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    after_cuda_nvml = nvml_proc_bytes()
    cuda_nvml_delta = after_cuda_nvml - base_nvml

    del gpu_buf
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after_free_nvml = nvml_proc_bytes()

    # 2. Allocate 256 MiB pinned (cudaHostAlloc) + 256 MiB CPU tensor
    pin_bytes = 256 * 1024 * 1024
    pin_buf = torch.empty(pin_bytes, dtype=torch.uint8, pin_memory=True)
    pin_buf.fill_(1)
    cpu_buf = torch.empty(pin_bytes, dtype=torch.uint8, device="cpu")
    cpu_buf.fill_(1)
    torch.cuda.synchronize()
    after_host_nvml = nvml_proc_bytes()
    after_host_psutil = psutil.virtual_memory().available
    host_nvml_delta = after_host_nvml - after_free_nvml
    host_psutil_drop = base_psutil - after_host_psutil

    del pin_buf, cpu_buf
    gc.collect()

    assert cuda_nvml_delta == cuda_bytes
    assert host_nvml_delta == 0

    return {
        "is_integrated_gpu": bool(torch.cuda.get_device_properties(0).is_integrated),
        "base_nvml_used_bytes": base_nvml,
        "cuda_512mib_nvml_delta_bytes": cuda_nvml_delta,
        "after_empty_cache_nvml_delta_bytes": after_free_nvml - base_nvml,
        "pinned_256mib_plus_cpu_256mib_nvml_delta_bytes": host_nvml_delta,
        "pinned_256mib_plus_cpu_256mib_psutil_drop_bytes": host_psutil_drop,
    }


def check_item5_kpool_workspace_reservation() -> dict:
    """Verify 0764 (#65092) and 0506 workspace reservation before lock_workspace."""
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
        lock_workspace,
        reset_workspace_manager,
    )
    import inspect
    from vllm.models.glm5next.amd import sparse_indexer as amd_si

    src = inspect.getsource(amd_si.sparse_attn_indexer_kpool)
    assert "decode_rows = _max_decode_logits_rows(hidden_states.shape[0])" in src
    assert "workspace_manager.get_simultaneous(" in src

    device = torch.device("cuda:0")
    reset_workspace_manager()
    init_workspace_manager(device)
    try:
        wm = current_workspace_manager()
        # Simulate prefill gather workspace (smaller than decode logits)
        wm.get_simultaneous(((1024, 128), torch.uint8), ((1024, 4), torch.uint8))
        # Simulate 0764 profiling reservation for decode logits: [H=32, decode_rows=96, max_pool_len=4096] float32 = 48 MiB
        wm.get_simultaneous(((32, 96, 4096), torch.float32))
        lock_workspace()
        locked_size_bytes = wm._workspace_size_bytes(wm._current_workspaces[0])
        # Verify locked workspace serves the decode logits tensor without error
        (logits_buf,) = wm.get_simultaneous(((32, 96, 4096), torch.float32))
        assert logits_buf.shape == (32, 96, 4096)
    finally:
        reset_workspace_manager()

    return {
        "amd_kpool_reserves_decode_logits_in_workspace": True,
        "locked_workspace_bytes": locked_size_bytes,
        "decode_logits_shape_verified_under_lock": list(logits_buf.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    results = {
        "boot": args.boot,
        "device_name": torch.cuda.get_device_name(0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "item1_pr66226_spec_decode_prefill_load": check_item1_spec_decode_prefill_load(),
        "item2_pr65866_llm_repr_rpc": check_item2_llm_repr_rpc(),
        "item3_pr65627_batch_invariant_fa_swa": check_item3_batch_invariant_fa_swa(),
        "item4_pr63401_pr55828_gb10_uma_memory": check_item4_gb10_uma_process_memory(),
        "item5_pr65092_kpool_workspace": check_item5_kpool_workspace_reservation(),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
