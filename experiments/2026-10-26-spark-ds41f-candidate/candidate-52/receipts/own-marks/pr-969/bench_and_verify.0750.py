#!/usr/bin/env python3
import argparse
import ctypes
import json
import os
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from vllm.models.deepseek_v4_1.engram_rows import (
    DEFAULT_CACHE_ROWS,
    EngramDiskRows,
    column_compaction_map,
    pack_engram_columns,
    packed_table_bytes,
)
from vllm.models.deepseek_v4_1.tilelang.engram_rows import (
    CONTROL_WORDS,
    HEADS,
    ROW,
    SCALES,
    engram_rows_gather_kernel,
    engram_rows_release_kernel,
    engram_rows_step_kernel,
    engram_rows_wait_kernel,
)

libcudart = ctypes.CDLL("libcudart.so")
cudaHostAlloc = libcudart.cudaHostAlloc
cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
cudaHostAlloc.restype = ctypes.c_int
cudaHostGetDevicePointer = libcudart.cudaHostGetDevicePointer
cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
cudaHostGetDevicePointer.restype = ctypes.c_int
cudaFreeHost = libcudart.cudaFreeHost
cudaFreeHost.argtypes = [ctypes.c_void_p]
cudaFreeHost.restype = ctypes.c_int


class MappedBuf:
    def __init__(self, shape, dtype, write_combined=False):
        self.shape = shape
        self.dtype = dtype
        self.nbytes = int(torch.empty(shape, dtype=dtype).numel() * dtype.itemsize)
        flags = 0x02 | (0x04 if write_combined else 0x00)
        hp = ctypes.c_void_p()
        dp = ctypes.c_void_p()
        assert cudaHostAlloc(ctypes.byref(hp), self.nbytes, flags) == 0
        assert cudaHostGetDevicePointer(ctypes.byref(dp), hp, 0) == 0
        self.hp = hp
        self.host = torch.frombuffer(
            (ctypes.c_uint8 * self.nbytes).from_address(hp.value), dtype=dtype
        ).view(shape)
        s = torch._C._construct_storage_from_data_pointer(
            dp.value, torch.device("cuda:0"), self.nbytes
        )
        strides = [1]
        for d in reversed(shape[1:]):
            strides.append(strides[-1] * d)
        strides = list(reversed(strides))
        self.dev = torch.empty(0, dtype=dtype, device="cuda:0").set_(s, 0, shape, strides)

    def close(self):
        if self.hp and self.hp.value:
            cudaFreeHost(self.hp)
            self.hp = None


def bench_fn(fn, warmup=40, iters=200, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    medians = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        medians.append((s.elapsed_time(e) * 1000.0) / iters)
    return {
        "median_us": round(statistics.median(medians), 4),
        "mean_us": round(statistics.mean(medians), 4),
        "stdev_us": round(statistics.stdev(medians), 4) if len(medians) > 1 else 0.0,
        "min_us": round(min(medians), 4),
        "max_us": round(max(medians), 4),
        "samples_us": [round(x, 4) for x in medians],
    }


def run_step_benchmark(boot_label: str) -> dict:
    torch.manual_seed(42)
    max_m = 2048
    max_lookups = max_m * HEADS
    shard_start = 100
    shard_end = 50000

    rows_buf = MappedBuf((max_lookups, ROW), torch.uint8, write_combined=True)
    scales_buf = MappedBuf((max_lookups, SCALES), torch.uint8, write_combined=True)
    ctrl_buf = MappedBuf((CONTROL_WORDS,), torch.int32, write_combined=False)

    r_rand = torch.randint(0, 256, (max_lookups, ROW), dtype=torch.uint8)
    r_rand[(r_rand == 0x7F) | (r_rand == 0xFF)] = 0
    rows_buf.host.copy_(r_rand)
    scales_buf.host.copy_(torch.randint(1, 250, (max_lookups, SCALES), dtype=torch.uint8))
    ctrl_buf.host.zero_()
    ctrl_buf.host[0] = 42
    ctrl_buf.host[16] = 42
    ctrl_buf.host[32] = 10_000_000

    rows_dev = rows_buf.dev.view(torch.float8_e4m3fn)
    scales_dev = scales_buf.dev
    ctrl_dev = ctrl_buf.dev

    ids_full = torch.randint(0, 60000, (max_m, HEADS), dtype=torch.int64, device="cuda:0")
    batch_sizes = [1, 2, 4, 8, 16, 18, 24, 30, 36, 42, 48, 128, 2048]
    shapes = [
        ("tp4-6cols", sum(1 << (c * 4) for c in range(6)), 6),
        ("tp3-8cols", sum(1 << (c * 3) for c in range(8)), 8),
        ("tp1-24cols", (1 << 24) - 1, 24),
    ]

    results = {
        "boot": boot_label,
        "device": torch.cuda.get_device_name(0),
        "batch_sizes": batch_sizes,
        "shapes": {},
    }

    try:
        for shape_name, mask, count in shapes:
            wait_k = engram_rows_wait_kernel()
            gather_k = engram_rows_gather_kernel(max_lookups, shard_start, shard_end, mask)
            release_k = engram_rows_release_kernel()
            step_k = engram_rows_step_kernel(max_lookups, shard_start, shard_end, mask)

            shape_rows = []
            for m in batch_sizes:
                ids_m = ids_full[:m]
                num_tok = torch.tensor([m], dtype=torch.int32, device="cuda:0")
                out_ref = torch.empty((m, count * ROW), dtype=torch.bfloat16, device="cuda:0")
                out_step = torch.empty((m, count * ROW), dtype=torch.bfloat16, device="cuda:0")

                ctrl_buf.host[1] = 0
                ctrl_buf.host[56] = 0
                wait_k(ctrl_dev)
                gather_k(ids_m, num_tok, rows_dev, scales_dev, out_ref)
                release_k(ctrl_dev)
                torch.cuda.synchronize()

                ctrl_buf.host[1] = 0
                ctrl_buf.host[56] = 0
                step_k(ids_m, num_tok, rows_dev, scales_dev, ctrl_dev, out_step)
                torch.cuda.synchronize()

                bit_exact = bool(torch.equal(out_ref.view(torch.uint16), out_step.view(torch.uint16)))
                retired_zero = int(ctrl_buf.host[56].item()) == 0
                consumed_ok = int(ctrl_buf.host[1].item()) == 42
                assert bit_exact, f"Bit-exactness failed for {shape_name} M={m}"
                assert retired_zero, f"Retired CTAs nonzero for {shape_name} M={m}"
                assert consumed_ok, f"Consumed sequence mismatch for {shape_name} M={m}"

                def run_base_eager():
                    wait_k(ctrl_dev)
                    gather_k(ids_m, num_tok, rows_dev, scales_dev, out_ref)
                    release_k(ctrl_dev)

                def run_step_eager():
                    step_k(ids_m, num_tok, rows_dev, scales_dev, ctrl_dev, out_step)

                base_eager = bench_fn(run_base_eager)
                step_eager = bench_fn(run_step_eager)

                k_nodes = 20 if m <= 128 else 5
                g_base = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_base):
                    for _ in range(k_nodes):
                        run_base_eager()
                g_step = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_step):
                    for _ in range(k_nodes):
                        run_step_eager()

                base_graph_raw = bench_fn(g_base.replay, warmup=10, iters=30)
                step_graph_raw = bench_fn(g_step.replay, warmup=10, iters=30)
                base_graph = {
                    k: ([round(v / k_nodes, 4) for v in val] if isinstance(val, list) else round(val / k_nodes, 4))
                    for k, val in base_graph_raw.items()
                }
                step_graph = {
                    k: ([round(v / k_nodes, 4) for v in val] if isinstance(val, list) else round(val / k_nodes, 4))
                    for k, val in step_graph_raw.items()
                }

                # Verify graph output bit-exactness too
                out_step.zero_()
                ctrl_buf.host[1] = 0
                g_step.replay()
                torch.cuda.synchronize()
                graph_bit_exact = bool(
                    torch.equal(out_ref.view(torch.uint16), out_step.view(torch.uint16))
                ) and (int(ctrl_buf.host[56].item()) == 0)
                assert graph_bit_exact, f"Graph bit-exactness failed for {shape_name} M={m}"

                rec = {
                    "M": m,
                    "bit_exact_eager": bit_exact,
                    "bit_exact_graph": graph_bit_exact,
                    "retired_ctas_after": int(ctrl_buf.host[56].item()),
                    "base_3launch_eager_us": base_eager["median_us"],
                    "step_multicta_eager_us": step_eager["median_us"],
                    "eager_speedup_x": round(base_eager["median_us"] / step_eager["median_us"], 4),
                    "base_3launch_graph_us": base_graph["median_us"],
                    "step_multicta_graph_us": step_graph["median_us"],
                    "graph_speedup_x": round(base_graph["median_us"] / step_graph["median_us"], 4),
                    "details": {
                        "base_eager": base_eager,
                        "step_eager": step_eager,
                        "base_graph": base_graph,
                        "step_graph": step_graph,
                    },
                }
                shape_rows.append(rec)
                print(
                    f"[{boot_label}][{shape_name}] M={m:4d}: "
                    f"Eager {base_eager["median_us"]:6.2f} -> {step_eager["median_us"]:6.2f} us "
                    f"({rec["eager_speedup_x"]:.2f}x) | "
                    f"Graph {base_graph["median_us"]:6.2f} -> {step_graph["median_us"]:6.2f} us "
                    f"({rec["graph_speedup_x"]:.2f}x)",
                    flush=True,
                )
            results["shapes"][shape_name] = shape_rows
    finally:
        rows_buf.close()
        scales_buf.close()
        ctrl_buf.close()

    return results


def _wait_reader_idle(rows: EngramDiskRows, timeout_s: float = 10.0) -> dict:
    torch.cuda.synchronize()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        st = rows.stats()
        if st["queued"] == 0:
            return st
        time.sleep(0.0005)
    return rows.stats()


def run_cache_and_packed_benchmark(boot_label: str) -> dict:
    base_dir = Path("/cache/engram-bench-step-and-cache")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    table_rows = 24000
    offsets = tuple((i * table_rows) // HEADS for i in range(HEADS))
    tp3_cols = tuple(range(1, HEADS, 3))
    tp3_mask = sum(1 << c for c in tp3_cols)

    gen = torch.Generator().manual_seed(2026)
    weights = torch.randint(0, 120, (table_rows, ROW), generator=gen, dtype=torch.uint8)
    scales = torch.randint(115, 135, (table_rows, SCALES), generator=gen, dtype=torch.uint8)

    w_path = str(base_dir / "weight.bin")
    s_path = str(base_dir / "scale.bin")
    p_path = str(base_dir / "packed_tp3_rank1.bin")
    with open(w_path, "wb") as f:
        f.write(weights.numpy().tobytes())
    with open(s_path, "wb") as f:
        f.write(scales.numpy().tobytes())

    col_start, col_end, col_delta, owned_starts, owned_rows = column_compaction_map(
        offsets, table_rows, tp3_cols
    )
    pack_engram_columns(w_path, 0, s_path, 0, p_path, owned_starts, owned_rows)
    assert os.path.getsize(p_path) == packed_table_bytes(int(owned_rows.sum()))

    max_tokens = 128
    batch_sizes = [1, 4, 8, 16, 32, 48, 64, 128]
    stream = torch.cuda.current_stream()

    modes = [
        ("two_plane_nocache", False, 0),
        ("two_plane_cache64k", False, DEFAULT_CACHE_ROWS),
        ("packed_nocache", True, 0),
        ("packed_cache64k", True, DEFAULT_CACHE_ROWS),
    ]

    hot_results = []
    zipf_results = []
    cold_results = []

    try:
        # 1. Hot working-set benchmark across M = 1..128 (including M=128 > 64)
        for m in batch_sizes:
            rng = np.random.default_rng(100 + m)
            ids_np = np.zeros((m, HEADS), dtype=np.int64)
            for c in range(HEADS):
                lo = offsets[c]
                hi = offsets[c + 1] if c + 1 < HEADS else table_rows
                ids_np[:, c] = rng.integers(lo, min(lo + 64, hi), size=m)
            ids_gpu = torch.from_numpy(ids_np).cuda()
            live = torch.tensor([m], dtype=torch.int32, device="cuda:0")

            m_rec = {"M": m, "modes": {}}
            ref_out = None
            for mode_name, is_packed, cache_rows in modes:
                r = EngramDiskRows(
                    device="cuda:0",
                    table_rows=table_rows,
                    shard_start=0,
                    shard_end=table_rows,
                    max_lookups=max_tokens * HEADS,
                    columns=tp3_mask,
                    cache_rows=cache_rows,
                )
                try:
                    if is_packed:
                        r.add_packed_source(
                            p_path, 0, col_start=col_start, col_end=col_end, col_delta=col_delta
                        )
                    else:
                        r.add_source(w_path, 0, scale=False)
                        r.add_source(s_path, 0, scale=True)

                    out = torch.empty((m, 8 * ROW), dtype=torch.bfloat16, device="cuda:0")
                    # Warmup 5 steps
                    for _ in range(5):
                        r.stage(stream, ids_gpu, m * HEADS, ids_np)
                        r.gather(out, ids_gpu, live)
                    st_before = _wait_reader_idle(r)

                    if ref_out is None:
                        ref_out = out.clone()
                    else:
                        assert torch.equal(ref_out.view(torch.uint16), out.view(torch.uint16)), (
                            f"Output mismatch in {mode_name} at M={m}"
                        )

                    iters = 60
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        r.stage(stream, ids_gpu, m * HEADS, ids_np)
                        r.gather(out, ids_gpu, live)
                    torch.cuda.synchronize()
                    elapsed_us = ((time.perf_counter() - t0) * 1e6) / iters
                    st_after = _wait_reader_idle(r)

                    hits = st_after["cache_hits"] - st_before["cache_hits"]
                    misses = st_after["cache_misses"] - st_before["cache_misses"]
                    reads = st_after["read_calls"] - st_before["read_calls"]
                    m_rec["modes"][mode_name] = {
                        "step_us": round(elapsed_us, 2),
                        "cache_hits": hits,
                        "cache_misses": misses,
                        "hit_rate": round(hits / max(1, hits + misses), 4),
                        "read_calls_per_step": round(reads / iters, 2),
                    }
                finally:
                    r.close()

            tp_speedup = round(
                m_rec["modes"]["two_plane_nocache"]["step_us"]
                / m_rec["modes"]["two_plane_cache64k"]["step_us"],
                3,
            )
            pk_speedup = round(
                m_rec["modes"]["two_plane_nocache"]["step_us"]
                / m_rec["modes"]["packed_cache64k"]["step_us"],
                3,
            )
            m_rec["two_plane_cache_speedup_x"] = tp_speedup
            m_rec["packed_cache_speedup_x"] = pk_speedup
            hot_results.append(m_rec)
            print(
                f"[{boot_label}][Hot Cache] M={m:3d}: "
                f"2-plane nocache={m_rec["modes"]["two_plane_nocache"]["step_us"]:6.1f} us -> "
                f"2-plane cache={m_rec["modes"]["two_plane_cache64k"]["step_us"]:6.1f} us ({tp_speedup:.2f}x) | "
                f"packed nocache={m_rec["modes"]["packed_nocache"]["step_us"]:6.1f} us -> "
                f"packed cache={m_rec["modes"]["packed_cache64k"]["step_us"]:6.1f} us ({pk_speedup:.2f}x)",
                flush=True,
            )

        # 2. Zipfian workload benchmark (skewed n-gram frequency distribution)
        for m in [4, 16, 48, 128]:
            rng = np.random.default_rng(500 + m)
            n_steps = 80
            ranks = np.arange(1, 257, dtype=np.float64)
            probs = (ranks ** -1.35)
            probs /= probs.sum()
            step_ids_np = []
            step_ids_gpu = []
            for _ in range(n_steps):
                arr = np.zeros((m, HEADS), dtype=np.int64)
                for c in range(HEADS):
                    lo = offsets[c]
                    choices = rng.choice(256, size=m, p=probs)
                    arr[:, c] = lo + choices
                step_ids_np.append(arr)
                step_ids_gpu.append(torch.from_numpy(arr).cuda())
            live = torch.tensor([m], dtype=torch.int32, device="cuda:0")

            z_rec = {"M": m, "steps": n_steps, "modes": {}}
            for mode_name, is_packed, cache_rows in modes:
                r = EngramDiskRows(
                    device="cuda:0",
                    table_rows=table_rows,
                    shard_start=0,
                    shard_end=table_rows,
                    max_lookups=max_tokens * HEADS,
                    columns=tp3_mask,
                    cache_rows=cache_rows,
                )
                try:
                    if is_packed:
                        r.add_packed_source(
                            p_path, 0, col_start=col_start, col_end=col_end, col_delta=col_delta
                        )
                    else:
                        r.add_source(w_path, 0, scale=False)
                        r.add_source(s_path, 0, scale=True)
                    out = torch.empty((m, 8 * ROW), dtype=torch.bfloat16, device="cuda:0")
                    # Warmup cache with first 20 steps
                    for s_idx in range(20):
                        r.stage(stream, step_ids_gpu[s_idx], m * HEADS, step_ids_np[s_idx])
                        r.gather(out, step_ids_gpu[s_idx], live)
                    st_before = _wait_reader_idle(r)

                    t0 = time.perf_counter()
                    for s_idx in range(20, n_steps):
                        r.stage(stream, step_ids_gpu[s_idx], m * HEADS, step_ids_np[s_idx])
                        r.gather(out, step_ids_gpu[s_idx], live)
                    torch.cuda.synchronize()
                    measured_steps = n_steps - 20
                    elapsed_us = ((time.perf_counter() - t0) * 1e6) / measured_steps
                    st_after = _wait_reader_idle(r)
                    hits = st_after["cache_hits"] - st_before["cache_hits"]
                    misses = st_after["cache_misses"] - st_before["cache_misses"]
                    reads = st_after["read_calls"] - st_before["read_calls"]
                    z_rec["modes"][mode_name] = {
                        "step_us": round(elapsed_us, 2),
                        "cache_hits": hits,
                        "cache_misses": misses,
                        "hit_rate": round(hits / max(1, hits + misses), 4),
                        "read_calls_per_step": round(reads / measured_steps, 2),
                    }
                finally:
                    r.close()
            z_rec["two_plane_cache_speedup_x"] = round(
                z_rec["modes"]["two_plane_nocache"]["step_us"]
                / z_rec["modes"]["two_plane_cache64k"]["step_us"],
                3,
            )
            z_rec["packed_cache_speedup_x"] = round(
                z_rec["modes"]["two_plane_nocache"]["step_us"]
                / z_rec["modes"]["packed_cache64k"]["step_us"],
                3,
            )
            zipf_results.append(z_rec)
            print(
                f"[{boot_label}][Zipfian]   M={m:3d}: "
                f"2-plane nocache={z_rec["modes"]["two_plane_nocache"]["step_us"]:6.1f} us -> "
                f"2-plane cache={z_rec["modes"]["two_plane_cache64k"]["step_us"]:6.1f} us "
                f"(hit={z_rec["modes"]["two_plane_cache64k"]["hit_rate"]*100:.1f}%, {z_rec["two_plane_cache_speedup_x"]:.2f}x) | "
                f"packed cache={z_rec["modes"]["packed_cache64k"]["step_us"]:6.1f} us ({z_rec["packed_cache_speedup_x"]:.2f}x)",
                flush=True,
            )

        # 3. Cold disjoint uniform steps (0% cache hit rate: verify zero cold-miss regression & 2x fewer read_calls on packed)
        for m in [4, 16, 32]:
            n_steps = 20
            step_ids_np = []
            step_ids_gpu = []
            for s_idx in range(n_steps):
                arr = np.zeros((m, HEADS), dtype=np.int64)
                for c in range(HEADS):
                    lo = offsets[c]
                    arr[:, c] = lo + s_idx * m + np.arange(m)
                step_ids_np.append(arr)
                step_ids_gpu.append(torch.from_numpy(arr).cuda())
            live = torch.tensor([m], dtype=torch.int32, device="cuda:0")

            c_rec = {"M": m, "steps": n_steps, "modes": {}}
            for mode_name, is_packed, cache_rows in modes:
                r = EngramDiskRows(
                    device="cuda:0",
                    table_rows=table_rows,
                    shard_start=0,
                    shard_end=table_rows,
                    max_lookups=max_tokens * HEADS,
                    columns=tp3_mask,
                    cache_rows=cache_rows,
                )
                try:
                    if is_packed:
                        r.add_packed_source(
                            p_path, 0, col_start=col_start, col_end=col_end, col_delta=col_delta
                        )
                    else:
                        r.add_source(w_path, 0, scale=False)
                        r.add_source(s_path, 0, scale=True)
                    out = torch.empty((m, 8 * ROW), dtype=torch.bfloat16, device="cuda:0")
                    st_before = _wait_reader_idle(r)
                    t0 = time.perf_counter()
                    for s_idx in range(n_steps):
                        r.stage(stream, step_ids_gpu[s_idx], m * HEADS, step_ids_np[s_idx])
                        r.gather(out, step_ids_gpu[s_idx], live)
                    torch.cuda.synchronize()
                    elapsed_us = ((time.perf_counter() - t0) * 1e6) / n_steps
                    st_after = _wait_reader_idle(r)
                    hits = st_after["cache_hits"] - st_before["cache_hits"]
                    misses = st_after["cache_misses"] - st_before["cache_misses"]
                    reads = st_after["read_calls"] - st_before["read_calls"]
                    c_rec["modes"][mode_name] = {
                        "step_us": round(elapsed_us, 2),
                        "cache_hits": hits,
                        "cache_misses": misses,
                        "read_calls_per_step": round(reads / n_steps, 2),
                    }
                finally:
                    r.close()
            cold_results.append(c_rec)
            print(
                f"[{boot_label}][Cold Miss] M={m:3d}: "
                f"2-plane nocache={c_rec["modes"]["two_plane_nocache"]["step_us"]:6.1f} us "
                f"(reads={c_rec["modes"]["two_plane_nocache"]["read_calls_per_step"]:.1f}) | "
                f"2-plane cache={c_rec["modes"]["two_plane_cache64k"]["step_us"]:6.1f} us | "
                f"packed cache={c_rec["modes"]["packed_cache64k"]["step_us"]:6.1f} us "
                f"(reads={c_rec["modes"]["packed_cache64k"]["read_calls_per_step"]:.1f})",
                flush=True,
            )
    finally:
        if base_dir.exists():
            shutil.rmtree(base_dir)

    return {
        "boot": boot_label,
        "device": torch.cuda.get_device_name(0),
        "default_cache_rows": DEFAULT_CACHE_ROWS,
        "hot_working_set": hot_results,
        "zipfian_working_set": zipf_results,
        "cold_disjoint_uniform": cold_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_res = run_step_benchmark(args.boot)
    (out_dir / f"bench-step-{args.boot}.json").write_text(json.dumps(step_res, indent=2) + "\n")

    cache_res = run_cache_and_packed_benchmark(args.boot)
    (out_dir / f"bench-cache-{args.boot}.json").write_text(json.dumps(cache_res, indent=2) + "\n")


if __name__ == "__main__":
    main()
