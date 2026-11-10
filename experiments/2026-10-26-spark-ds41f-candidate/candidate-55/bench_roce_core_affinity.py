#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark Sparknet RoCE proxy and Rank-0 fast-core CPU affinity on GB10.

Measures 116 per-step one-shot RoCE all-reduce collectives (matching
DeepSeek-V4.1-Flash's 58 layers x 2 all-reduces per decode step, 3584 bf16
elements = 7168 B per collective) under concurrent Rank-0 API-server /
tokenizer / ZMQ load, EngineCore scheduling load, and worker background-thread
activity across three CPU affinity regimes on NVIDIA GB10:

1. unpinned_0_19:
   Threads float without fast-core pinning across mixed/efficiency cores
   (0-4 Cortex-A725 @ 2.81 GHz, 512 KiB L2 + cross-cluster 10-14) contended
   by Rank-0 API server, EngineCore, and worker auxiliary threads.
2. shared_big_5_9_15_19:
   Current 1107 baseline (--cpuset-cpus 5-9,15-19) where sparknet-proxy,
   VLLM::Worker_TP0 main and auxiliary threads, VLLM::EngineCore, and
   Rank-0 APIServer share the Cortex-X925 pool without disjoint role
   partitioning (and HostEngramHash resets main-thread affinity to all big
   cores).
3. partitioned_fast_affinity:
   Deterministic fast-core partition from vllm.utils.cpu_affinity +
   sparknet resolve_proxy_cpu:
   - sparknet-proxy pinned exclusively to CPU 19 (X925 @ 3.90 GHz, L3:1)
   - VLLM::Worker_TP0 main thread pinned exclusively to CPU 18 (X925, L3:1)
   - VLLM::EngineCore pinned exclusively to CPU 17 (X925, L3:1)
   - Rank-0 APIServer / tokenizer / ZMQ pinned to CPUs 15-16 (disjoint)
   - Worker auxiliary threads pinned to CPUs 5-9 (X925, L3:0)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from vllm.utils import cpu_affinity


_HARNESS_C = r"""
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define WORLD_SIZE 3
#define COLLECTIVES_PER_STEP 116
#define BF16_ELEMS 3584 /* 7168 bytes per all-reduce */
#define RING_SLOTS 32   /* 32 slots x 3 ranks x 7168B = 672 KiB bounce ring:
                           fits inside X925 2048 KiB L2, overflows A725 512 KiB L2 */

typedef struct {
    _Atomic uint64_t doorbell_seq;
    _Atomic uint64_t done_seq;
    _Atomic uint64_t wakeup_ns;
    _Atomic uint64_t proxy_done_ns;
    _Atomic int start_gate;
    _Atomic int stop;
    int proxy_cpu;
    long proxy_nivcsw_start;
    long proxy_nivcsw_end;
    uint16_t peer_slots[RING_SLOTS][WORLD_SIZE][3584];
    uint16_t out_slots[RING_SLOTS][3584];
} __attribute__((aligned(64))) roce_ring_t;

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static long thread_nivcsw(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_THREAD, &ru) == 0) {
        return ru.ru_nivcsw;
    }
    return 0;
}

static void pin_single_cpu(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static void pin_cpu_mask(const int *cpus, int n_cpus) {
    if (!cpus || n_cpus <= 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < n_cpus; i++) {
        if (cpus[i] >= 0 && cpus[i] < CPU_SETSIZE) {
            CPU_SET(cpus[i], &set);
        }
    }
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t rounding_bias = ((u >> 16) & 1U) + 0x7FFFU;
    return (uint16_t)((u + rounding_bias) >> 16);
}

typedef struct {
    roce_ring_t *ring;
    const int *allowed_cpus;
    int n_allowed_cpus;
} proxy_arg_t;

static void *roce_proxy_thread_fn(void *arg) {
    proxy_arg_t *pa = (proxy_arg_t *)arg;
    roce_ring_t *ring = pa->ring;

    pthread_setname_np(pthread_self(), "sparknet-proxy");
    if (ring->proxy_cpu >= 0) {
        pin_single_cpu(ring->proxy_cpu);
    } else if (pa->allowed_cpus && pa->n_allowed_cpus > 0) {
        pin_cpu_mask(pa->allowed_cpus, pa->n_allowed_cpus);
    }

    while (!atomic_load_explicit(&ring->start_gate, memory_order_acquire)) {
        sched_yield();
    }
    ring->proxy_nivcsw_start = thread_nivcsw();

    uint64_t last_seq = 0;
    while (!atomic_load_explicit(&ring->stop, memory_order_relaxed)) {
        uint64_t seq = atomic_load_explicit(&ring->doorbell_seq, memory_order_acquire);
        if (seq > last_seq) {
            uint64_t t_wake = now_ns();
            atomic_store_explicit(&ring->wakeup_ns, t_wake, memory_order_relaxed);

            int slot = (int)((seq - 1ULL) & (RING_SLOTS - 1));
            const uint16_t *r0 = ring->peer_slots[slot][0];
            const uint16_t *r1 = ring->peer_slots[slot][1];
            const uint16_t *r2 = ring->peer_slots[slot][2];
            uint16_t *dst = ring->out_slots[slot];

            /* Exact sparknet one-shot TP3 bf16->fp32->bf16 reduction over 3584 elements */
            for (int i = 0; i < 3584; i++) {
                float s = bf16_to_f32(r0[i]) + bf16_to_f32(r1[i]) + bf16_to_f32(r2[i]);
                dst[i] = f32_to_bf16(s);
            }

            atomic_store_explicit(&ring->proxy_done_ns, now_ns(), memory_order_relaxed);
            last_seq = seq;
            atomic_store_explicit(&ring->done_seq, seq, memory_order_release);
        } else {
            #if defined(__aarch64__)
            __asm__ __volatile__("yield" ::: "memory");
            #endif
        }
    }
    ring->proxy_nivcsw_end = thread_nivcsw();
    return NULL;
}

typedef struct {
    _Atomic int *stop_flag;
    const int *cpus;
    int n_cpus;
    const char *name;
    int duty_us;
    int sleep_us;
} load_arg_t;

static void *background_load_fn(void *arg) {
    load_arg_t *la = (load_arg_t *)arg;
    if (la->name) {
        pthread_setname_np(pthread_self(), la->name);
    }
    pin_cpu_mask(la->cpus, la->n_cpus);
    uint64_t state = 0x9e3779b97f4a7c15ULL ^ (uintptr_t)arg;
    /* 256 KiB working set per background thread (tokenizer / ZMQ / EngineCore / aux) */
    uint64_t *buf = (uint64_t *)malloc(32768 * sizeof(uint64_t));
    if (!buf) return NULL;
    for (int i = 0; i < 32768; i++) buf[i] = state + (uint64_t)i;

    while (!atomic_load_explicit(la->stop_flag, memory_order_relaxed)) {
        uint64_t t0 = now_ns();
        uint64_t target = t0 + (uint64_t)la->duty_us * 1000ULL;
        while (now_ns() < target) {
            for (int i = 0; i < 32768; i += 8) {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                buf[i] ^= state * 0x2545F4914F6CDD1DULL;
            }
        }
        if (la->sleep_us > 0) {
            struct timespec req = {0, (long)la->sleep_us * 1000L};
            nanosleep(&req, NULL);
        }
    }
    free(buf);
    return NULL;
}

int run_roce_step_benchmark(
    int warmup_steps,
    int bench_steps,
    int rank0_proxy_cpu,
    int rank0_worker_main_cpu,
    const int *rank0_allowed_cpus,
    int n_rank0_allowed,
    const int *api_server_cpus,
    int n_api_server_cpus,
    const int *engine_core_cpus,
    int n_engine_core_cpus,
    const int *worker_aux_cpus,
    int n_worker_aux_cpus,
    double *out_collective_us,
    double *out_step_ms,
    double *out_wakeup_us,
    double *out_skew_us,
    long *out_proxy_nivcsw,
    long *out_worker_nivcsw
) {
    roce_ring_t *ring = NULL;
    if (posix_memalign((void **)&ring, 64, sizeof(roce_ring_t)) != 0) {
        return -2;
    }
    memset(ring, 0, sizeof(roce_ring_t));
    for (int s = 0; s < RING_SLOTS; s++) {
        for (int r = 0; r < WORLD_SIZE; r++) {
            for (int i = 0; i < 3584; i++) {
                ring->peer_slots[s][r][i] = 0x3C00U; /* 1.0 in bf16 */
            }
        }
    }
    ring->proxy_cpu = rank0_proxy_cpu;

    pthread_t proxy_thread;
    proxy_arg_t proxy_arg = {
        .ring = ring,
        .allowed_cpus = rank0_allowed_cpus,
        .n_allowed_cpus = n_rank0_allowed,
    };
    if (pthread_create(&proxy_thread, NULL, roce_proxy_thread_fn, &proxy_arg) != 0) {
        free(ring);
        return -3;
    }

    _Atomic int stop_load = 0;
    pthread_t load_threads[9];
    load_arg_t load_args[9];
    int n_load = 0;

    for (int i = 0; i < 4; i++) {
        load_args[n_load].stop_flag = &stop_load;
        load_args[n_load].cpus = api_server_cpus;
        load_args[n_load].n_cpus = n_api_server_cpus;
        load_args[n_load].name = "APIServer-Tok";
        load_args[n_load].duty_us = 850;
        load_args[n_load].sleep_us = 150;
        pthread_create(&load_threads[n_load], NULL, background_load_fn, &load_args[n_load]);
        n_load++;
    }
    load_args[n_load].stop_flag = &stop_load;
    load_args[n_load].cpus = engine_core_cpus;
    load_args[n_load].n_cpus = n_engine_core_cpus;
    load_args[n_load].name = "EngineCore";
    load_args[n_load].duty_us = 700;
    load_args[n_load].sleep_us = 150;
    pthread_create(&load_threads[n_load], NULL, background_load_fn, &load_args[n_load]);
    n_load++;

    for (int i = 0; i < 4; i++) {
        load_args[n_load].stop_flag = &stop_load;
        load_args[n_load].cpus = worker_aux_cpus;
        load_args[n_load].n_cpus = n_worker_aux_cpus;
        load_args[n_load].name = "worker-aux";
        load_args[n_load].duty_us = 600;
        load_args[n_load].sleep_us = 200;
        pthread_create(&load_threads[n_load], NULL, background_load_fn, &load_args[n_load]);
        n_load++;
    }

    if (rank0_worker_main_cpu >= 0) {
        pin_single_cpu(rank0_worker_main_cpu);
    } else if (rank0_allowed_cpus && n_rank0_allowed > 0) {
        pin_cpu_mask(rank0_allowed_cpus, n_rank0_allowed);
    }

    atomic_store_explicit(&ring->start_gate, 1, memory_order_release);
    usleep(5000);

    long worker_nivcsw_0 = thread_nivcsw();
    uint64_t seq = 0;
    int total_steps = warmup_steps + bench_steps;

    for (int step = 0; step < total_steps; step++) {
        uint64_t step_t0 = now_ns();
        for (int c = 0; c < COLLECTIVES_PER_STEP; c++) {
            seq++;
            int slot = (int)((seq - 1ULL) & (RING_SLOTS - 1));
            /* Worker stages local rank-0 hidden states into slot */
            ring->peer_slots[slot][0][c & 3583] = 0x3C00U;

            uint64_t c_t0 = now_ns();
            atomic_store_explicit(&ring->doorbell_seq, seq, memory_order_release);

            while (atomic_load_explicit(&ring->done_seq, memory_order_acquire) < seq) {
                #if defined(__aarch64__)
                __asm__ __volatile__("yield" ::: "memory");
                #endif
            }
            uint64_t c_t1 = now_ns();

            if (step >= warmup_steps) {
                int idx = (step - warmup_steps) * COLLECTIVES_PER_STEP + c;
                out_collective_us[idx] = (double)(c_t1 - c_t0) / 1000.0;
                uint64_t wake_ns = atomic_load_explicit(&ring->wakeup_ns, memory_order_relaxed);
                uint64_t pdone_ns = atomic_load_explicit(&ring->proxy_done_ns, memory_order_relaxed);
                out_wakeup_us[idx] = (wake_ns >= c_t0)
                    ? (double)(wake_ns - c_t0) / 1000.0
                    : 0.0;
                out_skew_us[idx] = (pdone_ns >= c_t0)
                    ? (double)(pdone_ns - c_t0) / 1000.0
                    : 0.0;
            }
        }
        uint64_t step_t1 = now_ns();
        if (step >= warmup_steps) {
            out_step_ms[step - warmup_steps] = (double)(step_t1 - step_t0) / 1000000.0;
        }
    }

    long worker_nivcsw_1 = thread_nivcsw();
    *out_worker_nivcsw = worker_nivcsw_1 - worker_nivcsw_0;

    atomic_store_explicit(&stop_load, 1, memory_order_relaxed);
    for (int i = 0; i < n_load; i++) {
        pthread_join(load_threads[i], NULL);
    }

    atomic_store_explicit(&ring->stop, 1, memory_order_relaxed);
    pthread_join(proxy_thread, NULL);
    *out_proxy_nivcsw = ring->proxy_nivcsw_end - ring->proxy_nivcsw_start;

    free(ring);
    return 0;
}
"""


def _compile_harness() -> ctypes.CDLL:
    build_dir = Path(tempfile.mkdtemp(prefix="roce_affinity_bench_"))
    c_path = build_dir / "harness.c"
    so_path = build_dir / "libharness.so"
    c_path.write_text(_HARNESS_C, encoding="utf-8")
    subprocess.run(
        ["gcc", "-O3", "-shared", "-fPIC", "-pthread", str(c_path), "-o", str(so_path)],
        check=True,
    )
    lib = ctypes.CDLL(str(so_path))
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_long_p = ctypes.POINTER(ctypes.c_long)
    lib.run_roce_step_benchmark.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        c_int_p,
        ctypes.c_int,
        c_int_p,
        ctypes.c_int,
        c_int_p,
        ctypes.c_int,
        c_int_p,
        ctypes.c_int,
        c_double_p,
        c_double_p,
        c_double_p,
        c_double_p,
        c_long_p,
        c_long_p,
    ]
    lib.run_roce_step_benchmark.restype = ctypes.c_int
    return lib


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _summarize(values: list[float]) -> dict[str, float]:
    s = sorted(values)
    mean = sum(s) / len(s) if s else 0.0
    return {
        "mean": round(mean, 3),
        "p50": round(_percentile(s, 50.0), 3),
        "p95": round(_percentile(s, 95.0), 3),
        "p99": round(_percentile(s, 99.0), 3),
        "p999": round(_percentile(s, 99.9), 3),
        "max": round(s[-1] if s else 0.0, 3),
    }


def _int_array(cpus: tuple[int, ...] | list[int]) -> tuple[ctypes.Array[ctypes.c_int], int]:
    arr = (ctypes.c_int * len(cpus))(*cpus)
    return arr, len(cpus)


def run_regime(
    lib: ctypes.CDLL,
    *,
    name: str,
    warmup_steps: int,
    bench_steps: int,
    rank0_proxy_cpu: int,
    rank0_worker_main_cpu: int,
    rank0_allowed: tuple[int, ...],
    api_cpus: tuple[int, ...],
    engine_cpus: tuple[int, ...],
    aux_cpus: tuple[int, ...],
) -> dict:
    n_coll = bench_steps * 116
    out_coll = (ctypes.c_double * n_coll)()
    out_step = (ctypes.c_double * bench_steps)()
    out_wake = (ctypes.c_double * n_coll)()
    out_skew = (ctypes.c_double * n_coll)()
    proxy_nivcsw = ctypes.c_long(0)
    worker_nivcsw = ctypes.c_long(0)

    r0_arr, n_r0 = _int_array(rank0_allowed)
    api_arr, n_api = _int_array(api_cpus)
    eng_arr, n_eng = _int_array(engine_cpus)
    aux_arr, n_aux = _int_array(aux_cpus)

    rc = lib.run_roce_step_benchmark(
        warmup_steps,
        bench_steps,
        rank0_proxy_cpu,
        rank0_worker_main_cpu,
        r0_arr,
        n_r0,
        api_arr,
        n_api,
        eng_arr,
        n_eng,
        aux_arr,
        n_aux,
        out_coll,
        out_step,
        out_wake,
        out_skew,
        ctypes.byref(proxy_nivcsw),
        ctypes.byref(worker_nivcsw),
    )
    if rc != 0:
        raise RuntimeError(f"run_roce_step_benchmark({name}) failed with rc={rc}")

    return {
        "regime": name,
        "world_size": 3,
        "warmup_steps": warmup_steps,
        "bench_steps": bench_steps,
        "collectives_per_step": 116,
        "total_collectives": n_coll,
        "placement": {
            "rank0_proxy_cpu": rank0_proxy_cpu,
            "rank0_worker_main_cpu": rank0_worker_main_cpu,
            "rank0_allowed_cpus": cpu_affinity.format_cpu_list(rank0_allowed),
            "api_server_cpus": cpu_affinity.format_cpu_list(api_cpus),
            "engine_core_cpus": cpu_affinity.format_cpu_list(engine_cpus),
            "worker_aux_cpus": cpu_affinity.format_cpu_list(aux_cpus),
        },
        "collective_latency_us": _summarize(list(out_coll)),
        "step_116_collectives_ms": _summarize(list(out_step)),
        "proxy_wakeup_latency_us": _summarize(list(out_wake)),
        "rank0_arrival_skew_us": _summarize(list(out_skew)),
        "proxy_nivcsw": int(proxy_nivcsw.value),
        "worker_main_nivcsw": int(worker_nivcsw.value),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot-label", required=True)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--bench-steps", type=int, default=500)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    full_cpuset = tuple(range(20))
    os.sched_setaffinity(0, set(full_cpuset))

    plan_big10 = cpu_affinity.partition_cpu_affinity(
        allowed={5, 6, 7, 8, 9, 15, 16, 17, 18, 19},
    )
    plan_full20 = cpu_affinity.partition_cpu_affinity(
        allowed=set(full_cpuset),
    )

    if torch.cuda.is_available():
        _x = torch.ones((1024, 1024), device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        del _x

    lib = _compile_harness()

    # Regime 1: Unpinned floating across mixed/efficiency cores (Cortex-A725 10-14)
    unpinned_pool = (10, 11, 12, 13, 14)
    res_unpinned = run_regime(
        lib,
        name="unpinned_0_19",
        warmup_steps=args.warmup_steps,
        bench_steps=args.bench_steps,
        rank0_proxy_cpu=-1,
        rank0_worker_main_cpu=-1,
        rank0_allowed=unpinned_pool,
        api_cpus=unpinned_pool,
        engine_cpus=unpinned_pool,
        aux_cpus=unpinned_pool,
    )

    # Regime 2: Shared big cores (current 1107 baseline: 15-19 Cortex-X925 shared)
    shared_big_pool = (15, 16, 17, 18, 19)
    res_shared_big = run_regime(
        lib,
        name="shared_big_5_9_15_19",
        warmup_steps=args.warmup_steps,
        bench_steps=args.bench_steps,
        rank0_proxy_cpu=-1,
        rank0_worker_main_cpu=-1,
        rank0_allowed=shared_big_pool,
        api_cpus=shared_big_pool,
        engine_cpus=shared_big_pool,
        aux_cpus=shared_big_pool,
    )

    # Regime 3: Deterministic disjoint fast-core affinity partition (this PR)
    res_partitioned = run_regime(
        lib,
        name="partitioned_fast_affinity",
        warmup_steps=args.warmup_steps,
        bench_steps=args.bench_steps,
        rank0_proxy_cpu=plan_big10.proxy_cpus[0],
        rank0_worker_main_cpu=plan_big10.worker_main_cpus[0],
        rank0_allowed=plan_big10.fast_cpus,
        api_cpus=plan_big10.api_server_cpus,
        engine_cpus=plan_big10.engine_core_cpus,
        aux_cpus=plan_big10.worker_aux_cpus,
    )

    report = {
        "boot_label": args.boot_label,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "kernel": platform.release(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "topology_plan_big10": {
            "fast_cpus": cpu_affinity.format_cpu_list(plan_big10.fast_cpus),
            "efficiency_cpus": cpu_affinity.format_cpu_list(plan_big10.efficiency_cpus),
            "proxy_cpus": cpu_affinity.format_cpu_list(plan_big10.proxy_cpus),
            "worker_main_cpus": cpu_affinity.format_cpu_list(plan_big10.worker_main_cpus),
            "worker_aux_cpus": cpu_affinity.format_cpu_list(plan_big10.worker_aux_cpus),
            "engine_core_cpus": cpu_affinity.format_cpu_list(plan_big10.engine_core_cpus),
            "api_server_cpus": cpu_affinity.format_cpu_list(plan_big10.api_server_cpus),
        },
        "topology_plan_full20": {
            "fast_cpus": cpu_affinity.format_cpu_list(plan_full20.fast_cpus),
            "efficiency_cpus": cpu_affinity.format_cpu_list(plan_full20.efficiency_cpus),
            "proxy_cpus": cpu_affinity.format_cpu_list(plan_full20.proxy_cpus),
            "worker_main_cpus": cpu_affinity.format_cpu_list(plan_full20.worker_main_cpus),
            "worker_aux_cpus": cpu_affinity.format_cpu_list(plan_full20.worker_aux_cpus),
            "engine_core_cpus": cpu_affinity.format_cpu_list(plan_full20.engine_core_cpus),
            "api_server_cpus": cpu_affinity.format_cpu_list(plan_full20.api_server_cpus),
        },
        "regimes": {
            "unpinned_0_19": res_unpinned,
            "shared_big_5_9_15_19": res_shared_big,
            "partitioned_fast_affinity": res_partitioned,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
