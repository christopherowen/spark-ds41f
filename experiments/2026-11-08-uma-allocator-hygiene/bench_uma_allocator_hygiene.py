import argparse
import contextlib
import ctypes
import functools
import gc
import json
import mmap
import os
import re
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch

from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.direct_loader import bind_weight_arena
from vllm.models.deepseek_v4_1.tilelang.vocab import pack_bf16_weight
from vllm.utils.mem_utils import (
    DeviceMemoryProfiler,
    MemorySnapshot,
    memory_profiling,
)


@functools.lru_cache(maxsize=1)
def _get_malloc_trim():
    try:
        fn = ctypes.CDLL('libc.so.6').malloc_trim
        fn.argtypes = [ctypes.c_size_t]
        fn.restype = ctypes.c_int
        return fn
    except (OSError, AttributeError):
        return None


@functools.lru_cache(maxsize=1)
def _get_c10_mi_collect():
    try:
        c10_path = Path(torch.__file__).resolve().parent / 'lib' / 'libc10.so'
        if not c10_path.is_file():
            return None
        with c10_path.open('rb') as f, mmap.mmap(
            f.fileno(), 0, access=mmap.ACCESS_READ
        ) as mm:
            if mm[:4] != b'\x7fELF' or mm[4] != 2 or mm[5] != 1:
                return None
            (e_shoff,) = struct.unpack_from('<Q', mm, 40)
            e_shentsize, e_shnum, _ = struct.unpack_from('<HHH', mm, 58)
            sections = [
                struct.unpack_from('<IIQQQQIIQQ', mm, e_shoff + i * e_shentsize)
                for i in range(e_shnum)
            ]
            free_cpu_val = mi_collect_val = None
            for (
                _,
                sh_type,
                _,
                _,
                sh_offset,
                sh_size,
                sh_link,
                _,
                _,
                sh_entsize,
            ) in sections:
                if (
                    sh_type not in (2, 11)
                    or sh_entsize != 24
                    or sh_link >= len(sections)
                ):
                    continue
                str_off = sections[sh_link][4]
                str_sz = sections[sh_link][5]
                strtab = mm[str_off : str_off + str_sz]
                sym_tab = mm[sh_offset : sh_offset + sh_size]
                for target_bytes, attr in (
                    (b'_ZN3c108free_cpuEPv\x00', 'free'),
                    (b'mi_collect\x00', 'mi'),
                ):
                    idx = 0
                    while True:
                        pos = strtab.find(target_bytes, idx)
                        if pos < 0:
                            break
                        idx = pos + 1
                        if pos == 0 or strtab[pos - 1] == 0:
                            packed_name = struct.pack('<I', pos)
                            spos = 0
                            while True:
                                found = sym_tab.find(packed_name, spos)
                                if found < 0:
                                    break
                                spos = found + 4
                                if found % 24 == 0:
                                    (
                                        _,
                                        _,
                                        _,
                                        st_shndx,
                                        st_value,
                                        _,
                                    ) = struct.unpack_from('<IBBHQQ', sym_tab, found)
                                    if st_shndx != 0 and st_value != 0:
                                        if attr == 'free':
                                            free_cpu_val = st_value
                                        else:
                                            mi_collect_val = st_value
            if free_cpu_val is None or mi_collect_val is None:
                return None
            c10 = ctypes.CDLL(str(c10_path))
            runtime_free = ctypes.cast(c10._ZN3c108free_cpuEPv, ctypes.c_void_p).value
            if not runtime_free:
                return None
            return ctypes.CFUNCTYPE(None, ctypes.c_bool)(
                runtime_free + (mi_collect_val - free_cpu_val)
            )
    except Exception:
        return None


def trim_host_heap() -> bool:
    trimmed = False
    mi_collect = _get_c10_mi_collect()
    if mi_collect is not None:
        try:
            mi_collect(True)
            trimmed = True
        except Exception:
            pass
    malloc_trim = _get_malloc_trim()
    if malloc_trim is not None:
        try:
            if malloc_trim(0):
                trimmed = True
        except Exception:
            pass
    return trimmed


def reclaim_uma_memory(*, synchronize: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        if synchronize:
            torch.accelerator.synchronize()
        torch.accelerator.empty_cache()
    trim_host_heap()


def get_proc_stats(device: torch.device) -> dict:
    vm_rss_kb = rss_anon_kb = 0
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                vm_rss_kb = int(line.split()[1])
            elif line.startswith('RssAnon:'):
                rss_anon_kb = int(line.split()[1])
    heap_rss_kb = 0
    anon_mmap_rss_kb = 0
    with open('/proc/self/smaps') as f:
        text = f.read()
    for b in re.split(r'\n(?=[0-9a-f]+-[0-9a-f]+ )', text):
        lines = b.splitlines()
        if not lines:
            continue
        hdr = lines[0].split()
        name = hdr[5] if len(hdr) > 5 else ''
        rss = 0
        for l in lines[1:]:
            if l.startswith('Rss:'):
                rss += int(l.split()[1])
        if name == '[heap]':
            heap_rss_kb += rss
        elif not name:
            anon_mmap_rss_kb += rss
    mem_avail_bytes = psutil.virtual_memory().available
    free_cuda, _ = torch.cuda.mem_get_info(device)
    alloc_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    mib = float(1 << 20)
    return {
        'vm_rss_mib': round(vm_rss_kb / 1024.0, 2),
        'rss_anon_mib': round(rss_anon_kb / 1024.0, 2),
        'heap_rss_mib': round(heap_rss_kb / 1024.0, 2),
        'anon_mmap_rss_mib': round(anon_mmap_rss_kb / 1024.0, 2),
        'mem_avail_mib': round(mem_avail_bytes / mib, 2),
        'cuda_free_mib': round(free_cuda / mib, 2),
        'torch_allocated_mib': round(alloc_bytes / mib, 2),
        'torch_reserved_mib': round(reserved_bytes / mib, 2),
        'torch_slack_mib': round(max(reserved_bytes - alloc_bytes, 0) / mib, 2),
    }


class _SyntheticCycleNode:
    def __init__(self, payload_cpu=None, payload_gpu=None):
        self.payload_cpu = payload_cpu
        self.payload_gpu = payload_gpu
        self.peer = self


class _DummyArenaModel(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        # 256 small materialized parameters (~512 KiB each = 128 MiB total) that
        # bind_weight_arena snapshots to CPU via param.detach().to('cpu', copy=True)
        # before binding into the unified weight arena.
        self.small_params = torch.nn.ParameterList([
            torch.nn.Parameter(
                torch.ones((256, 1024), dtype=torch.bfloat16, device=device),
                requires_grad=False,
            )
            for _ in range(256)
        ])
        self.large_blocks = torch.nn.ParameterList([
            torch.nn.Parameter(
                torch.empty((4096, 7168), dtype=torch.float8_e4m3fn, device='meta'),
                requires_grad=False,
            )
            for _ in range(16)
        ])


def find_model_snapshot() -> Path:
    base = Path(
        '/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots'
    )
    snapshots = sorted(base.glob('*'))
    if not snapshots:
        raise FileNotFoundError(f'No snapshot found under {base}')
    return snapshots[-1]


def warm_page_cache() -> None:
    snap_dir = find_model_snapshot()
    _ = (snap_dir / 'model.safetensors.index.json').read_bytes()
    for shard in sorted(snap_dir.glob('model-*.safetensors')):
        _ = weight_utils.safetensors_file_sources(str(shard))


def run_single_mode(mode: str) -> dict:
    assert mode in ('baseline', 'candidate')
    is_cand = mode == 'candidate'
    if not is_cand:
        import vllm.utils.mem_utils as _mu
        import vllm.model_executor.model_loader.direct_loader as _dl
        _mu.trim_host_heap = lambda: False
        _dl.trim_host_heap = lambda: False
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    _ = torch.empty(1, device=device)
    torch.cuda.synchronize(device)

    results = {'mode': mode}
    baseline_snapshot = MemorySnapshot(device=device)
    results['initial'] = get_proc_stats(device)

    # -------------------------------------------------------------------------
    # STAGE 1: Weight Loading & Safetensors / Arena Post-Processing
    # -------------------------------------------------------------------------
    snap_dir = find_model_snapshot()
    index_path = snap_dir / 'model.safetensors.index.json'
    shards = sorted(snap_dir.glob('model-*.safetensors'))

    with DeviceMemoryProfiler(device) as profiler:
        # 1a. Parse index and all 48 safetensors shard headers
        weight_map = json.loads(index_path.read_text())['weight_map']
        shard_sources = []
        for shard in shards:
            sources = weight_utils.safetensors_file_sources(str(shard))
            shard_sources.append(
                {k: (v.path, v.offset, v.shape, v.dtype) for k, v in sources.items()}
            )
        header_staging = [
            {f'{k}_{i}': list(range(32)) for k in list(weight_map.keys())[:4000]}
            for i in range(16)
        ]

        # 1b. Bind weight arena (materializes 128 MiB of cpu_snapshots in c10::alloc_cpu)
        model = _DummyArenaModel(device)
        arena = bind_weight_arena(model, device, order13_sponge=False)
        if is_cand:
            gc.collect()
            trim_host_heap()

        # 1c. Pack BF16 vocabulary/compressor weights to 12-bit + CPU transform staging
        cpu_weight_staging = [
            torch.ones((1024, 1024), dtype=torch.bfloat16, device='cpu')
            for _ in range(96)
        ]  # 192 MiB in c10::alloc_cpu
        raw_lm_head = torch.randn((43096, 7168), dtype=torch.bfloat16, device=device)
        packed_lm_head = pack_bf16_weight(raw_lm_head)
        del raw_lm_head, cpu_weight_staging

        loader_cycle = _SyntheticCycleNode(
            payload_cpu=[
                torch.ones((512, 512), dtype=torch.float32, device='cpu')
                for _ in range(32)
            ],
            payload_gpu=torch.empty(
                (64 * 1024 * 1024,), dtype=torch.uint8, device=device
            ),
        )
        del weight_map, shard_sources, header_staging, loader_cycle

        if is_cand:
            reclaim_uma_memory(synchronize=True)
        else:
            torch.accelerator.empty_cache()

    if is_cand:
        trim_host_heap()

    results['stage1_after_load'] = get_proc_stats(device)
    results['stage1_after_load']['weights_memory_mib'] = round(
        profiler.consumed_memory / float(1 << 20), 2
    )

    # -------------------------------------------------------------------------
    # STAGE 2: Kernel Prep & profile_run() / memory_profiling()
    # -------------------------------------------------------------------------
    with memory_profiling(
        baseline_snapshot,
        weights_memory=int(profiler.consumed_memory),
    ) as profile_result:
        if is_cand:
            trim_host_heap()
            profile_result.before_profile.measure()

        # 2a. Vision encoder cache profiling: 16 max-res dummy image inputs on CPU + GPU
        mm_cpu_staging = [
            torch.ones((3, 1024, 1024), dtype=torch.float32, device='cpu')
            for _ in range(32)
        ]  # 384 MiB CPU tensors in c10::alloc_cpu
        mm_dicts = [
            {f'patch_{i}_{j}': list(range(64)) for j in range(2000)}
            for i in range(16)
        ]
        mm_gpu_act = torch.empty(
            (256 * 1024 * 1024,), dtype=torch.uint8, device=device
        )
        del mm_cpu_staging, mm_dicts, mm_gpu_act

        # 2b. Generic _dummy_run(16384) + _dummy_sampler_run
        persistent_workspace = torch.empty(
            (512 * 1024 * 1024,), dtype=torch.uint8, device=device
        )
        generic_pass_act = torch.empty(
            (768 * 1024 * 1024,), dtype=torch.uint8, device=device
        )
        sampler_cycle = _SyntheticCycleNode(
            payload_cpu=[
                torch.ones((512, 512), dtype=torch.float32, device='cpu')
                for _ in range(32)
            ],
            payload_gpu=torch.empty(
                (256 * 1024 * 1024,), dtype=torch.uint8, device=device
            ),
        )
        del generic_pass_act, sampler_cycle

        # Transition from generic pass to _profile_deepseek_v4_attention
        if is_cand:
            reclaim_uma_memory(synchronize=True)

        # 2c. _profile_deepseek_v4_attention + profile_cudagraph_memory
        attn_prefill_act = torch.empty(
            (640 * 1024 * 1024,), dtype=torch.uint8, device=device
        )
        torch.cuda.synchronize(device)
        peak_reserved_during_attn_mib = round(
            torch.cuda.memory_reserved(device) / float(1 << 20), 2
        )
        cg_profile_host = [
            {f'graph_node_{i}_{j}': (i, j, 'kernel_launch') for j in range(3000)}
            for i in range(20)
        ]
        del attn_prefill_act, cg_profile_host

        # End of profile_run() + _release_b12x_profile_state()
        if is_cand:
            reclaim_uma_memory(synchronize=True)
        else:
            gc.collect()
            torch.accelerator.empty_cache()

    if is_cand:
        trim_host_heap()
        profile_result.after_profile.measure()
        profile_result.non_torch_increase = (
            profile_result.after_profile.non_torch_memory
            - profile_result.before_create.non_torch_memory
        )
        profile_result.total_consumed = (
            profile_result.before_create.free_memory
            - profile_result.after_profile.free_memory
        )
        profile_result.transient_peak_headroom = (
            profile_result.after_profile.torch_peak
            - profile_result.after_profile.torch_allocated
        )
        profile_result.non_kv_cache_memory = (
            profile_result.total_consumed + profile_result.transient_peak_headroom
        )

    mib = float(1 << 20)
    results['stage2_after_profile'] = get_proc_stats(device)
    results['stage2_after_profile'].update({
        'peak_reserved_during_attn_mib': peak_reserved_during_attn_mib,
        'non_torch_increase_mib': round(profile_result.non_torch_increase / mib, 2),
        'total_consumed_mib': round(profile_result.total_consumed / mib, 2),
        'transient_peak_headroom_mib': round(
            profile_result.transient_peak_headroom / mib, 2
        ),
        'non_kv_cache_memory_mib': round(profile_result.non_kv_cache_memory / mib, 2),
    })

    # -------------------------------------------------------------------------
    # STAGE 3: Warmup, CUDA Graph Capture, Post-Capture Warmup & freeze_gc_heap
    # -------------------------------------------------------------------------
    grammar_bitmasks = [
        torch.full((64, 129280 // 32), fill_value=-1, dtype=torch.int32, device='cpu')
        for _ in range(128)
    ]
    del grammar_bitmasks
    if is_cand:
        reclaim_uma_memory(synchronize=True)
    else:
        torch.accelerator.empty_cache()

    graph_pool = torch.cuda.graph_pool_handle()
    captured_graphs = []
    static_in = torch.zeros((64, 7168), dtype=torch.bfloat16, device=device)
    static_w = torch.randn((7168, 7168), dtype=torch.bfloat16, device=device)
    static_out = torch.empty_like(static_in)
    _ = torch.mm(static_in[:4], static_w)
    torch.cuda.synchronize(device)

    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        for bsz in (1, 2, 4, 8, 16, 24, 32, 48, 64):
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream(device=device)
            s.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(s):
                with torch.cuda.graph(g, pool=graph_pool, stream=s):
                    tmp = torch.mm(static_in[:bsz], static_w)
                    static_out[:bsz].copy_(tmp)
            torch.cuda.current_stream(device).wait_stream(s)
            captured_graphs.append((bsz, g))
            _ = _SyntheticCycleNode(
                payload_cpu={
                    f'capture_{bsz}_{k}': list(range(128)) for k in range(2000)
                }
            )

        for tail_tokens in (1024, 2048, 4096, 8192):
            tail_buf = torch.empty(
                (tail_tokens, 7168), dtype=torch.bfloat16, device=device
            )
            del tail_buf
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()
    finally:
        gc.unfreeze()
        gc.enable()
        gc.collect()
        if is_cand:
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()
            trim_host_heap()

    # Post-capture warmup_kernels(post_capture=True)
    post_capture_cycle = _SyntheticCycleNode(
        payload_cpu=[
            torch.ones((512, 512), dtype=torch.float32, device='cpu')
            for _ in range(64)
        ],
        payload_gpu=torch.empty(
            (384 * 1024 * 1024,), dtype=torch.uint8, device=device
        ),
    )
    del post_capture_cycle

    if is_cand:
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        reclaim_uma_memory(synchronize=True)
        gc.freeze()
        startup_snapshot = MemorySnapshot(device=device)
    else:
        torch.accelerator.empty_cache()
        startup_snapshot = MemorySnapshot(device=device)
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        gc.freeze()

    results['stage3_startup_complete'] = get_proc_stats(device)
    results['stage3_startup_complete'].update({
        'snapshot_free_mib': round(startup_snapshot.free_memory / mib, 2),
        'snapshot_non_torch_mib': round(startup_snapshot.non_torch_memory / mib, 2),
        'snapshot_torch_reserved_mib': round(startup_snapshot.torch_memory / mib, 2),
        'snapshot_torch_allocated_mib': round(
            startup_snapshot.torch_allocated / mib, 2
        ),
        'snapshot_torch_slack_mib': round(
            max(startup_snapshot.torch_memory - startup_snapshot.torch_allocated, 0)
            / mib,
            2,
        ),
    })

    # -------------------------------------------------------------------------
    # STAGE 4: Long-Prefill Eager Chunks (32,768 tokens) -> Decode Transition
    # -------------------------------------------------------------------------
    pending_post_prefill_flush = False

    # Chunk 1 of 32k prefill (16,384 tokens, CUDAGraphMode.NONE)
    chunk1_act = torch.empty((1536 * 1024 * 1024,), dtype=torch.uint8, device=device)
    chunk1_host = [
        torch.ones((512, 512), dtype=torch.float32, device='cpu') for _ in range(64)
    ]
    del chunk1_act, chunk1_host
    pending_post_prefill_flush = True
    reserved_after_chunk1 = torch.cuda.memory_reserved(device)

    # Chunk 2 of 32k prefill (16,384 tokens, CUDAGraphMode.NONE)
    reserved_before_chunk2 = torch.cuda.memory_reserved(device)
    chunk1_to_chunk2_reused = reserved_before_chunk2 == reserved_after_chunk1
    chunk2_act = torch.empty((1536 * 1024 * 1024,), dtype=torch.uint8, device=device)
    chunk2_host = [
        torch.ones((512, 512), dtype=torch.float32, device='cpu') for _ in range(64)
    ]
    del chunk2_act, chunk2_host

    # Transition Step: First CUDAGraphMode.FULL decode step after long prefill
    _, decode_graph = captured_graphs[2]  # bsz=4 full graph
    decode_graph.replay()
    if is_cand and pending_post_prefill_flush:
        pending_post_prefill_flush = False
        reclaim_uma_memory(synchronize=False)

    torch.cuda.synchronize(device)
    results['stage4_after_prefill_transition'] = get_proc_stats(device)
    results['stage4_after_prefill_transition']['chunk1_to_chunk2_reused'] = (
        chunk1_to_chunk2_reused
    )

    # Measure 200 steady-state decode steps (CUDAGraphMode.FULL)
    for _ in range(20):
        decode_graph.replay()
    torch.cuda.synchronize(device)

    step_latencies_us = []
    for _ in range(200):
        t0 = time.perf_counter_ns()
        decode_graph.replay()
        if is_cand and pending_post_prefill_flush:
            pending_post_prefill_flush = False
            reclaim_uma_memory(synchronize=False)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter_ns()
        step_latencies_us.append((t1 - t0) / 1000.0)

    results['stage4_steady_state_decode'] = {
        'num_steps': len(step_latencies_us),
        'mean_us': round(float(np.mean(step_latencies_us)), 2),
        'p50_us': round(float(np.median(step_latencies_us)), 2),
        'p99_us': round(float(np.percentile(step_latencies_us, 99)), 2),
    }

    gc.unfreeze()
    del (
        captured_graphs,
        static_in,
        static_w,
        static_out,
        persistent_workspace,
        packed_lm_head,
        arena,
        model,
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['baseline', 'candidate', 'both'], default='both'
    )
    parser.add_argument('--boot-label', default='boot1')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    if args.mode in ('baseline', 'candidate'):
        res = run_single_mode(args.mode)
        print(json.dumps(res))
        return

    warm_page_cache()
    out = {'boot_label': args.boot_label, 'gpu': torch.cuda.get_device_name(0)}
    for m in ('baseline', 'candidate'):
        proc = subprocess.run(
            [sys.executable, __file__, '--mode', m],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [l for l in proc.stdout.strip().splitlines() if l.startswith('{')]
        out[m] = json.loads(lines[-1])

    deltas = {}
    for stage in (
        'stage1_after_load',
        'stage2_after_profile',
        'stage3_startup_complete',
        'stage4_after_prefill_transition',
    ):
        b = out['baseline'][stage]
        c = out['candidate'][stage]
        deltas[stage] = {
            k: round(c[k] - b[k], 2) for k in b if isinstance(b[k], (int, float))
        }
    b_dec = out['baseline']['stage4_steady_state_decode']
    c_dec = out['candidate']['stage4_steady_state_decode']
    deltas['stage4_steady_state_decode'] = {
        k: round(c_dec[k] - b_dec[k], 2) for k in ('mean_us', 'p50_us', 'p99_us')
    }
    out['deltas_candidate_minus_baseline'] = deltas

    formatted = json.dumps(out, indent=2)
    print(formatted)
    if args.output:
        Path(args.output).write_text(formatted + '\n')


if __name__ == '__main__':
    main()
