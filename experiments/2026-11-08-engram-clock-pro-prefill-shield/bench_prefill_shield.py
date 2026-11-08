#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark 2-bit Clock + prefill probationary row cache and pinned two-plane scales."""

import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

from vllm.models.deepseek_v4_1.engram_rows.rows import (
    ALL_COLUMNS,
    PACKED_ROW_BYTES,
    ROW_BYTES,
    SCALE_BYTES,
    EngramDiskRows,
    column_compaction_map,
    pack_engram_columns,
)
from vllm.models.deepseek_v4_1.tilelang.engram_rows import engram_rows_gather

HEADS = 24
TP3_RANK1 = sum(1 << c for c in range(1, HEADS, 3))  # 8 columns


def _wait_reader_idle(rows: EngramDiskRows, timeout_s: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while True:
        st = rows.stats()
        if st["queued"] == 0:
            return st
        if time.monotonic() > deadline:
            raise TimeoutError(f"Reader did not drain within {timeout_s}s: {st}")
        time.sleep(0.0002)


def _simulate_fifo_4way_hit_rate(
    cache_rows: int,
    columns_mask: int,
    warmup_batches: list[np.ndarray],
    pre_batches: list[np.ndarray],
    prefill_chunks: list[np.ndarray],
    interleaved_batches: list[np.ndarray],
    post_batches: list[np.ndarray],
) -> dict:
    """Exact 4-way set-associative FIFO cache model matching 0750 ple_cache_insert."""
    sets = cache_rows // 4
    mask = sets - 1
    cache_ids = np.full((sets, 4), -1, dtype=np.int64)
    clock = np.zeros(sets, dtype=np.uint8)
    owned_cols = [c for c in range(HEADS) if (columns_mask >> c) & 1]

    def run_step(batch_2d: np.ndarray) -> tuple[int, int]:
        hits = 0
        misses = 0
        miss_ids = []
        for t in range(batch_2d.shape[0]):
            for c in owned_cols:
                rid = int(batch_2d[t, c])
                if rid < 0:
                    continue
                h = (rid * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
                s = int((h ^ (h >> 32)) & mask)
                row_ways = cache_ids[s]
                if rid == row_ways[0] or rid == row_ways[1] or rid == row_ways[2] or rid == row_ways[3]:
                    hits += 1
                else:
                    misses += 1
                    miss_ids.append((s, rid))
        for s, rid in miss_ids:
            row_ways = cache_ids[s]
            if rid == row_ways[0] or rid == row_ways[1] or rid == row_ways[2] or rid == row_ways[3]:
                continue
            w = int(clock[s]) & 3
            clock[s] = (w + 1) & 3
            row_ways[w] = rid
        return hits, misses

    for b in warmup_batches:
        run_step(b)
    pre_hits = 0
    pre_misses = 0
    for b in pre_batches:
        h, m = run_step(b)
        pre_hits += h
        pre_misses += m

    interleaved_hits = 0
    interleaved_misses = 0
    prefill_hits = 0
    prefill_misses = 0
    for idx, pf in enumerate(prefill_chunks):
        h, m = run_step(pf)
        prefill_hits += h
        prefill_misses += m
        if idx < len(interleaved_batches):
            ih, im = run_step(interleaved_batches[idx])
            interleaved_hits += ih
            interleaved_misses += im

    post_hits = 0
    post_misses = 0
    for b in post_batches:
        h, m = run_step(b)
        post_hits += h
        post_misses += m

    return {
        "pre_prefill_decode_hit_rate": round(pre_hits / max(1, pre_hits + pre_misses), 4),
        "prefill_hit_rate": round(prefill_hits / max(1, prefill_hits + prefill_misses), 4),
        "interleaved_decode_hit_rate": round(
            interleaved_hits / max(1, interleaved_hits + interleaved_misses), 4
        ),
        "post_prefill_decode_hit_rate": round(
            post_hits / max(1, post_hits + post_misses), 4
        ),
        "interleaved_decode_hits": interleaved_hits,
        "interleaved_decode_lookups": interleaved_hits + interleaved_misses,
        "post_prefill_decode_hits": post_hits,
        "post_prefill_decode_lookups": post_hits + post_misses,
    }


def _create_table_files(tmpdir: str, table_rows: int = 200_000):
    rng = np.random.default_rng(1234)
    w_path = os.path.join(tmpdir, "weights.bin")
    s_path = os.path.join(tmpdir, "scales.bin")
    p_path = os.path.join(tmpdir, "packed_tp3_rank1.bin")

    chunk = 20_000
    with open(w_path, "wb") as fw, open(s_path, "wb") as fs:
        for start in range(0, table_rows, chunk):
            n = min(chunk, table_rows - start)
            w = rng.integers(0, 254, size=(n, ROW_BYTES), dtype=np.uint8)
            w[w == 0x7F] = 1
            s = rng.integers(120, 134, size=(n, SCALE_BYTES), dtype=np.uint8)
            fw.write(w.tobytes())
            fs.write(s.tobytes())
        fw.flush()
        fs.flush()
        os.fsync(fw.fileno())
        os.fsync(fs.fileno())

    base = table_rows // HEADS
    rem = table_rows % HEADS
    offsets = np.zeros(HEADS, dtype=np.int64)
    cur = 0
    for c in range(HEADS):
        offsets[c] = cur
        cur += base + (1 if c < rem else 0)

    owned_cols = tuple(c for c in range(HEADS) if (TP3_RANK1 >> c) & 1)
    col_start, col_end, col_delta, owned_starts, owned_rows = column_compaction_map(
        offsets, table_rows, owned_cols
    )
    t0 = time.perf_counter()
    pack_engram_columns(w_path, 0, s_path, 0, p_path, owned_starts, owned_rows)
    pack_s = time.perf_counter() - t0
    return {
        "w_path": w_path,
        "s_path": s_path,
        "p_path": p_path,
        "offsets": offsets,
        "col_start": col_start,
        "col_end": col_end,
        "col_delta": col_delta,
        "owned_rows": int(owned_rows.sum()),
        "pack_seconds": round(pack_s, 4),
    }


def run_cache_and_packed_benchmark(meta: dict, table_rows: int = 200_000) -> dict:
    w_path = meta["w_path"]
    s_path = meta["s_path"]
    p_path = meta["p_path"]
    offsets = meta["offsets"]
    col_start = meta["col_start"]
    col_end = meta["col_end"]
    col_delta = meta["col_delta"]
    col_ends_all = np.append(offsets[1:], table_rows)

    batch_sizes = (1, 2, 4, 8, 16, 32)
    max_m = max(batch_sizes)
    stream = torch.cuda.current_stream()
    out = torch.empty((max_m, 8 * ROW_BYTES), dtype=torch.bfloat16, device="cuda")

    rng = np.random.default_rng(999)
    hot_per_col = 4096 // 8
    hot_pool = np.empty((HEADS, hot_per_col), dtype=np.int64)
    for c in range(HEADS):
        hot_pool[c] = rng.choice(
            np.arange(offsets[c], col_ends_all[c], dtype=np.int64),
            size=hot_per_col,
            replace=False,
        )

    def make_reader(mode: str) -> EngramDiskRows:
        is_packed = mode.startswith("packed")
        pin_scales = "pin_scales" in mode
        cache_rows = 65536 if "cache64k" in mode else 0
        r = EngramDiskRows(
            device="cuda",
            table_rows=table_rows,
            shard_start=0,
            shard_end=table_rows,
            max_lookups=max_m * HEADS,
            columns=TP3_RANK1,
            cache_rows=cache_rows,
            pin_scales=pin_scales,
        )
        if is_packed:
            r.add_packed_source(
                p_path, 0, col_start=col_start, col_end=col_end, col_delta=col_delta
            )
        else:
            r.add_source(w_path, 0, scale=False)
            r.add_source(s_path, 0, scale=True)
        return r

    results = {
        "pack_seconds": meta["pack_seconds"],
        "packed_file_bytes": os.path.getsize(p_path),
        "owned_rows": meta["owned_rows"],
        "modes": {},
    }

    modes = [
        "two_plane_nocache",
        "packed_nocache",
        "two_plane_cache64k",
        "two_plane_pin_scales_cache64k",
        "packed_cache64k",
    ]
    for mode in modes:
        results["modes"][mode] = {}
        for workload in ("cold", "zipf", "hot"):
            r = make_reader(mode)
            try:
                warmup = 8
                iters = 32
                total_steps = warmup + iters
                batches_cpu = []
                batches_gpu = []
                for s_idx in range(total_steps):
                    b = np.empty((max_m, HEADS), dtype=np.int64)
                    for c in range(HEADS):
                        if workload == "cold":
                            span = col_ends_all[c] - offsets[c]
                            start_idx = (s_idx * max_m * 37) % max(1, span - max_m)
                            b[:, c] = offsets[c] + start_idx + np.arange(max_m)
                        elif workload == "hot":
                            b[:, c] = rng.choice(hot_pool[c], size=max_m, replace=True)
                        else:
                            mask = rng.random(max_m) < 0.85
                            hot_pick = rng.choice(hot_pool[c], size=max_m, replace=True)
                            cold_pick = rng.integers(
                                offsets[c], col_ends_all[c], size=max_m, dtype=np.int64
                            )
                            b[:, c] = np.where(mask, hot_pick, cold_pick)
                    batches_cpu.append(b)
                    batches_gpu.append(torch.from_numpy(b).cuda())

                if "cache64k" in mode and workload in ("hot", "zipf"):
                    for c_chunk in range(0, hot_per_col, max_m):
                        end_c = min(c_chunk + max_m, hot_per_col)
                        cnt = end_c - c_chunk
                        wb = np.empty((cnt, HEADS), dtype=np.int64)
                        for c in range(HEADS):
                            wb[:, c] = hot_pool[c, c_chunk:end_c]
                        r.prefetch(wb.reshape(-1))
                    _wait_reader_idle(r)
                    # Second touch during decode warmup promotes hot working set in 2-bit Clock.
                    for c_chunk in range(0, min(hot_per_col, max_m * 8), max_m):
                        end_c = min(c_chunk + max_m, hot_per_col)
                        cnt = end_c - c_chunk
                        wb = np.empty((cnt, HEADS), dtype=np.int64)
                        for c in range(HEADS):
                            wb[:, c] = hot_pool[c, c_chunk:end_c]
                        r.prefetch(wb.reshape(-1))
                    _wait_reader_idle(r)

                results["modes"][mode][workload] = {}
                for m in batch_sizes:
                    live = torch.tensor([m], dtype=torch.int32, device="cuda")
                    out_m = out[:m]
                    for i in range(warmup):
                        early = batches_cpu[i][:m].reshape(-1)
                        r.stage(stream, batches_gpu[i][:m], m * HEADS, early)
                        engram_rows_gather(
                            batches_gpu[i][:m],
                            live,
                            r.rows_e4m3,
                            r.scales.device_view,
                            r.control.device_view,
                            out_m,
                            0,
                            table_rows,
                            columns=TP3_RANK1,
                        )
                    torch.cuda.synchronize()
                    st_before = _wait_reader_idle(r)

                    latencies_us = []
                    for i in range(warmup, total_steps):
                        early = batches_cpu[i][:m].reshape(-1)
                        t0 = time.perf_counter()
                        r.stage(stream, batches_gpu[i][:m], m * HEADS, early)
                        engram_rows_gather(
                            batches_gpu[i][:m],
                            live,
                            r.rows_e4m3,
                            r.scales.device_view,
                            r.control.device_view,
                            out_m,
                            0,
                            table_rows,
                            columns=TP3_RANK1,
                        )
                        torch.cuda.synchronize()
                        latencies_us.append((time.perf_counter() - t0) * 1e6)

                    st_after = _wait_reader_idle(r)
                    arr = np.array(latencies_us)
                    hits = st_after["cache_hits"] - st_before["cache_hits"]
                    misses = st_after["cache_misses"] - st_before["cache_misses"]
                    read_calls = st_after["read_calls"] - st_before["read_calls"]
                    read_bytes = st_after["read_bytes"] - st_before["read_bytes"]
                    results["modes"][mode][workload][f"m{m}"] = {
                        "mean_us": round(float(arr.mean()), 2),
                        "p50_us": round(float(np.percentile(arr, 50)), 2),
                        "p90_us": round(float(np.percentile(arr, 90)), 2),
                        "cache_hits": int(hits),
                        "cache_misses": int(misses),
                        "hit_rate": round(float(hits / max(1, hits + misses)), 4),
                        "read_calls_per_step": round(float(read_calls / iters), 2),
                        "read_kib_per_step": round(float(read_bytes / 1024.0 / iters), 2),
                    }
            finally:
                r.close()
    return results


def run_prefill_flood_benchmark(tmpdir: str, table_rows: int = 1_050_000) -> dict:
    """Measure decode hit rate and latency across interleaved prefill_yield and
    post-prefill decode steps when a 32K-token (8 x 4096) prefill flood sweeps
    through a 65,536-row cache warmed with a realistic decode working set."""
    meta = _create_table_files(tmpdir, table_rows=table_rows)
    p_path = meta["p_path"]
    w_path = meta["w_path"]
    s_path = meta["s_path"]
    offsets = meta["offsets"]
    col_start = meta["col_start"]
    col_end = meta["col_end"]
    col_delta = meta["col_delta"]
    col_ends_all = np.append(offsets[1:], table_rows)

    prefill_m = 4096
    num_chunks = 8  # 8 * 4096 = 32,768 tokens = 262,144 owned lookups (4x 65,536 cache)
    stream = torch.cuda.current_stream()
    out_max = torch.empty((prefill_m, 8 * ROW_BYTES), dtype=torch.bfloat16, device="cuda")

    rng = np.random.default_rng(20261108)
    hot_per_col = 4096 // 8  # 512 rows per owned col = 4096 hot decode working-set rows
    hot_pool = np.empty((HEADS, hot_per_col), dtype=np.int64)
    for c in range(HEADS):
        hot_pool[c] = offsets[c] + np.arange(hot_per_col, dtype=np.int64)

    # Pre-build 8 x 4096 prefill chunks: first 16 tokens are recurring system-prompt
    # header n-grams, remaining 4080 tokens per chunk (8 * 4080 = 32,640 rows/col)
    # are strictly disjoint one-time prompt n-grams.
    prefill_chunks_cpu = []
    prefill_chunks_gpu = []
    for chunk_idx in range(num_chunks):
        pf = np.empty((prefill_m, HEADS), dtype=np.int64)
        for c in range(HEADS):
            pf[:16, c] = offsets[c] + hot_per_col + np.arange(16, dtype=np.int64)
            start_row = offsets[c] + hot_per_col + 16 + chunk_idx * (prefill_m - 16)
            pf[16:, c] = start_row + np.arange(prefill_m - 16, dtype=np.int64)
        prefill_chunks_cpu.append(pf)
        prefill_chunks_gpu.append(torch.from_numpy(pf).cuda())

    tail_start_offset = hot_per_col + 16 + num_chunks * (prefill_m - 16)

    scenarios = {"table_rows": table_rows, "owned_rows": meta["owned_rows"]}
    for decode_m in (8, 16):
        scenarios[f"decode_m{decode_m}"] = {}
        for dist in ("hot_working_set", "zipf85_working_set"):
            warmup_batches_cpu = []
            warmup_batches_gpu = []
            steps_per_pass = hot_per_col // decode_m
            for _ in range(2):
                for s_idx in range(steps_per_pass):
                    b = np.empty((decode_m, HEADS), dtype=np.int64)
                    for c in range(HEADS):
                        b[:, c] = hot_pool[c, s_idx * decode_m : (s_idx + 1) * decode_m]
                    warmup_batches_cpu.append(b)
                    warmup_batches_gpu.append(torch.from_numpy(b).cuda())

            def sample_decode_batches(n_steps: int):
                cpu_list, gpu_list = [], []
                for _ in range(n_steps):
                    b = np.empty((decode_m, HEADS), dtype=np.int64)
                    for c in range(HEADS):
                        if dist == "hot_working_set":
                            b[:, c] = rng.choice(hot_pool[c], size=decode_m, replace=False)
                        else:
                            mask = rng.random(decode_m) < 0.85
                            hot_pick = rng.choice(hot_pool[c], size=decode_m, replace=True)
                            cold_pick = rng.integers(
                                offsets[c] + tail_start_offset,
                                col_ends_all[c],
                                size=decode_m,
                                dtype=np.int64,
                            )
                            b[:, c] = np.where(mask, hot_pick, cold_pick)
                    cpu_list.append(b)
                    gpu_list.append(torch.from_numpy(b).cuda())
                return cpu_list, gpu_list

            pre_cpu, pre_gpu = sample_decode_batches(16)
            interleaved_cpu, interleaved_gpu = sample_decode_batches(num_chunks)
            post_cpu, post_gpu = sample_decode_batches(32)

            fifo_sim = _simulate_fifo_4way_hit_rate(
                65536,
                TP3_RANK1,
                warmup_batches_cpu,
                pre_cpu,
                prefill_chunks_cpu,
                interleaved_cpu,
                post_cpu,
            )

            mode_results = {"fifo_0750_exact_sim": fifo_sim}
            for mode in ("packed_cache64k", "two_plane_pin_scales_cache64k"):
                r = EngramDiskRows(
                    device="cuda",
                    table_rows=table_rows,
                    shard_start=0,
                    shard_end=table_rows,
                    max_lookups=prefill_m * HEADS,
                    columns=TP3_RANK1,
                    cache_rows=65536,
                    pin_scales=("pin_scales" in mode),
                )
                if mode.startswith("packed"):
                    r.add_packed_source(
                        p_path, 0, col_start=col_start, col_end=col_end, col_delta=col_delta
                    )
                else:
                    r.add_source(w_path, 0, scale=False)
                    r.add_source(s_path, 0, scale=True)

                try:
                    live_dec = torch.tensor([decode_m], dtype=torch.int32, device="cuda")
                    live_pf = torch.tensor([prefill_m], dtype=torch.int32, device="cuda")
                    out_dec = out_max[:decode_m]
                    out_pf = out_max[:prefill_m]

                    for b_cpu, b_gpu in zip(warmup_batches_cpu, warmup_batches_gpu):
                        r.stage(stream, b_gpu, decode_m * HEADS, b_cpu.reshape(-1))
                        engram_rows_gather(
                            b_gpu, live_dec, r.rows_e4m3, r.scales.device_view,
                            r.control.device_view, out_dec, 0, table_rows, columns=TP3_RANK1,
                        )
                    torch.cuda.synchronize()
                    st_pre0 = _wait_reader_idle(r)

                    pre_lat_us = []
                    for b_cpu, b_gpu in zip(pre_cpu, pre_gpu):
                        t0 = time.perf_counter()
                        r.stage(stream, b_gpu, decode_m * HEADS, b_cpu.reshape(-1))
                        engram_rows_gather(
                            b_gpu, live_dec, r.rows_e4m3, r.scales.device_view,
                            r.control.device_view, out_dec, 0, table_rows, columns=TP3_RANK1,
                        )
                        torch.cuda.synchronize()
                        pre_lat_us.append((time.perf_counter() - t0) * 1e6)
                    st_pre1 = _wait_reader_idle(r)
                    pre_hits = st_pre1["cache_hits"] - st_pre0["cache_hits"]
                    pre_misses = st_pre1["cache_misses"] - st_pre0["cache_misses"]

                    interleaved_lat_us = []
                    prefill_lat_ms = []
                    interleaved_hits = 0
                    interleaved_misses = 0
                    interleaved_reads = 0
                    prefill_hits = 0
                    prefill_misses = 0

                    for c_idx in range(num_chunks):
                        st_pf0 = _wait_reader_idle(r)
                        t_pf0 = time.perf_counter()
                        r.stage(
                            stream,
                            prefill_chunks_gpu[c_idx],
                            prefill_m * HEADS,
                            prefill_chunks_cpu[c_idx].reshape(-1),
                        )
                        engram_rows_gather(
                            prefill_chunks_gpu[c_idx], live_pf, r.rows_e4m3,
                            r.scales.device_view, r.control.device_view, out_pf,
                            0, table_rows, columns=TP3_RANK1,
                        )
                        torch.cuda.synchronize()
                        prefill_lat_ms.append((time.perf_counter() - t_pf0) * 1e3)
                        st_pf1 = _wait_reader_idle(r)
                        prefill_hits += st_pf1["cache_hits"] - st_pf0["cache_hits"]
                        prefill_misses += st_pf1["cache_misses"] - st_pf0["cache_misses"]

                        t_dec0 = time.perf_counter()
                        r.stage(
                            stream,
                            interleaved_gpu[c_idx],
                            decode_m * HEADS,
                            interleaved_cpu[c_idx].reshape(-1),
                        )
                        engram_rows_gather(
                            interleaved_gpu[c_idx], live_dec, r.rows_e4m3,
                            r.scales.device_view, r.control.device_view, out_dec,
                            0, table_rows, columns=TP3_RANK1,
                        )
                        torch.cuda.synchronize()
                        interleaved_lat_us.append((time.perf_counter() - t_dec0) * 1e6)
                        st_dec1 = _wait_reader_idle(r)
                        interleaved_hits += st_dec1["cache_hits"] - st_pf1["cache_hits"]
                        interleaved_misses += st_dec1["cache_misses"] - st_pf1["cache_misses"]
                        interleaved_reads += st_dec1["read_calls"] - st_pf1["read_calls"]

                    st_post0 = _wait_reader_idle(r)
                    post_lat_us = []
                    for b_cpu, b_gpu in zip(post_cpu, post_gpu):
                        t0 = time.perf_counter()
                        r.stage(stream, b_gpu, decode_m * HEADS, b_cpu.reshape(-1))
                        engram_rows_gather(
                            b_gpu, live_dec, r.rows_e4m3, r.scales.device_view,
                            r.control.device_view, out_dec, 0, table_rows, columns=TP3_RANK1,
                        )
                        torch.cuda.synchronize()
                        post_lat_us.append((time.perf_counter() - t0) * 1e6)
                    st_post1 = _wait_reader_idle(r)
                    post_hits = st_post1["cache_hits"] - st_post0["cache_hits"]
                    post_misses = st_post1["cache_misses"] - st_post0["cache_misses"]
                    post_reads = st_post1["read_calls"] - st_post0["read_calls"]

                    pre_arr = np.array(pre_lat_us)
                    int_arr = np.array(interleaved_lat_us)
                    post_arr = np.array(post_lat_us)
                    pf_arr = np.array(prefill_lat_ms)

                    mode_results[mode] = {
                        "pre_prefill_decode_mean_us": round(float(pre_arr.mean()), 2),
                        "pre_prefill_decode_p50_us": round(float(np.percentile(pre_arr, 50)), 2),
                        "pre_prefill_decode_hit_rate": round(
                            float(pre_hits / max(1, pre_hits + pre_misses)), 4
                        ),
                        "prefill_chunk_mean_ms": round(float(pf_arr.mean()), 3),
                        "prefill_hit_rate": round(
                            float(prefill_hits / max(1, prefill_hits + prefill_misses)), 4
                        ),
                        "interleaved_decode_mean_us": round(float(int_arr.mean()), 2),
                        "interleaved_decode_p50_us": round(float(np.percentile(int_arr, 50)), 2),
                        "interleaved_decode_hit_rate": round(
                            float(
                                interleaved_hits
                                / max(1, interleaved_hits + interleaved_misses)
                            ),
                            4,
                        ),
                        "interleaved_decode_read_calls_per_step": round(
                            float(interleaved_reads / num_chunks), 2
                        ),
                        "post_prefill_decode_mean_us": round(float(post_arr.mean()), 2),
                        "post_prefill_decode_p50_us": round(float(np.percentile(post_arr, 50)), 2),
                        "post_prefill_decode_p90_us": round(float(np.percentile(post_arr, 90)), 2),
                        "post_prefill_decode_hit_rate": round(
                            float(post_hits / max(1, post_hits + post_misses)), 4
                        ),
                        "post_prefill_decode_read_calls_per_step": round(
                            float(post_reads / len(post_cpu)), 2
                        ),
                    }
                finally:
                    r.close()
            scenarios[f"decode_m{decode_m}"][dist] = mode_results
    return scenarios


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="engram_shield_bench_") as tmpdir:
        meta_200k = _create_table_files(tmpdir, table_rows=200_000)
        cache_bench = run_cache_and_packed_benchmark(meta_200k, table_rows=200_000)
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="engram_flood_bench_") as tmpdir2:
        flood_bench = run_prefill_flood_benchmark(tmpdir2, table_rows=1_050_000)
    report = {
        "device": torch.cuda.get_device_name(0),
        "cache_and_packed_benchmark": cache_bench,
        "prefill_flood_benchmark": flood_bench,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
