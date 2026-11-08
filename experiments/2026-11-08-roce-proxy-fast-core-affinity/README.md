# RoCE Proxy Fast-Core Affinity and Rank-0 Arrival Skew Partitioning

- **Date**: 2026-11-08
- **Base commit**: `8e5ea208f3d22bc69df91ace94d6975932c858a8` (`1107` candidate base: `vllm` patch head `2756478b60b90700b9918b457aeb38657da637b5`, `sparknet` revision `1fd6a1fbd8dc4884057046d1ebd93d585c9e4bcf`)
- **Candidate patches**:
  - `sparknet`: `0001-perf-oneshot-pin-RoCE-proxy-thread-to-dedicated-fast.patch` (commit `88366d5f79296e83e2715bf572114d56ae2b2b4a`, tree `dab8c4577ac1cd87f7abc59a9c44a8e9d146f579`)
  - `vllm`: `0762-perf-worker-roce-proxy-and-rank0-fast-core-affinity.patch` (commit `60cb03efd0724da97e10f88ffd10843e6846809d`, tree `606b3ecd74b3079db3985b9bea21a9fcb419a807`)
- **Intended causal variable**: Deterministic CPU core affinity partitioning on `NVIDIA GB10` across `sparknet-proxy`, `VLLM::Worker_TP*` main and auxiliary threads, `VLLM::EngineCore`, and the Rank-0 `APIServer` / ZMQ frontend / tokenizer processes (`TODO.md` item 6).

---

## 1. Hardware Topology & Live Rank-0 Thread Audit on `NVIDIA GB10`

`NVIDIA GB10` has 20 logical ARMv9.2-A CPUs organized into two 10-core L3 cache clusters (`L3:0 = CPUs 0-9`, `L3:1 = CPUs 10-19`, 16 MiB L3 per cluster), each split into 5 `Cortex-A725` efficiency cores and 5 `Cortex-X925` performance cores:

| CPU Range | Core Microarchitecture | L3 Cluster | Max Clock (`cpuinfo_max_freq`) | Per-Core L2 Cache (`index2/size`) | `cpu_capacity` |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `0-4` | ARM Cortex-A725 (efficiency) | `L3:0` (`0-9`) | 2,808.0 MHz | 512 KiB | 739 |
| `5-9` | ARM Cortex-X925 (performance) | `L3:0` (`0-9`) | 3,900.0 MHz | 2,048 KiB | 1024 |
| `10-14` | ARM Cortex-A725 (efficiency) | `L3:1` (`10-19`) | 2,808.0 MHz | 512 KiB | 739 |
| `15-19` | ARM Cortex-X925 (performance) | `L3:1` (`10-19`) | 3,900.0 MHz | 2,048 KiB | 1024 |

### Pre-Change Scheduling Flaws Identified

1. **Unpartitioned 10-Core Pool on Rank 0**:
   - In the `1107` container (`--cpuset-cpus 5-9,15-19`), `APIServer` (`vllm serve`), `VLLM::EngineCore`, and `VLLM::Worker_TP0` (29 threads) all inherited the same 10-core affinity mask `5-9,15-19`.
   - Live thread inspection on Rank 0 (`ps -L -o pid,tid,psr,pcpu,comm`) showed `VLLM::EngineCore`, `VLLM::Worker_TP0` main thread, and `sparknet-proxy` scheduled concurrently onto the same physical core (`psr=9`) while `APIServer` and `engram-reader` contended across the same pool.
2. **`HostEngramHash.__init__` Overwriting Worker Main-Thread Affinity**:
   - During model loading, `HostEngramHash.__init__` in `vllm/models/deepseek_v4_1/common/engram.py` calls `os.sched_setaffinity(0, cpus)` with `max_l2_cpus()` (`{5, 6, 7, 8, 9, 15, 16, 17, 18, 19}`) so the OpenMP thread pool inherits the big-L2 cores on first `libgomp` initialization, resetting the `VLLM::Worker_TP*` main thread's affinity back to all 10 cores.
3. **`sparknet` Proxy Core Selection Not Intersected with `sched_getaffinity`**:
   - `find_big_cpu()` in `sparknet/csrc/roce_proxy.c` scanned `/sys/devices/system/cpu` for the highest `cpuinfo_max_freq` without intersecting against `pthread_getaffinity_np`, and picked the lowest-numbered big core (`CPU 5`), which is the first core in `worker_aux` / OpenMP's mask (`5-9`).
4. **32-Slot One-Shot Bounce Ring (`672 KiB`) vs. L2 Cache Size**:
   - Across 116 per-step one-shot all-reduce collectives (`3,584` bf16 elements = `7,168 B` per rank per slot, `672 KiB` for a 32-slot × 3-rank ring), the working set fits inside `Cortex-X925`'s `2,048 KiB` L2 cache when isolated on dedicated fast cores, but overflows `Cortex-A725`'s `512 KiB` L2 cache when unpinned (`unpinned_0_19`) and suffers involuntary context switches and L2 evictions when sharing cores with `APIServer` / `EngineCore` (`shared_big_5_9_15_19`).

---

## 2. Deterministic Fast-Core Affinity Design

`vllm/utils/cpu_affinity.py` and `sparknet/csrc/roce_proxy.c` discover `Cortex-X925` performance cores dynamically via sysfs (`cpuinfo_max_freq`, `cpu_capacity`, and `cache/index2/size >= 1024K`) intersected with `os.sched_getaffinity(0)` / `pthread_getaffinity_np`:

| Role | Process / Threads | Assigned CPUs (`cpuset=5-9,15-19`) | Assigned CPUs (`cpuset=0-19`) | L3 Cluster | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`proxy_cpus`** | `sparknet-proxy` | `19` | `19` | `L3:1` | Dedicated highest-numbered `Cortex-X925` core (3.90 GHz, 2 MiB L2); zero preemption during RDMA doorbell/CQ polling and bf16 reduction. |
| **`worker_main_cpus`** | `VLLM::Worker_TP*` main (`TID == PID`) | `18` | `18` | `L3:1` | Dedicated `Cortex-X925` core on the same `L3:1` cluster as `sparknet-proxy` (`CPU 19`) for minimum cache-coherency doorbell latency. |
| **`engine_core_cpus`** | `VLLM::EngineCore` | `17` | `17` | `L3:1` | Dedicated `Cortex-X925` core; never preempts `Worker_TP0` (`18`) or `sparknet-proxy` (`19`). |
| **`api_server_cpus`** | Rank-0 `APIServer`, ZMQ, tokenizers | `15-16` | `0-4,10-16` | `L3:1` (+ efficiency) | Disjoint from `17`, `18`, `19`, and `5-9`; isolates HTTP/ZMQ/tokenizer bursts on Rank 0. |
| **`worker_aux_cpus`** | `engram-reader`, OpenMP, `gloo`, `nccl` | `5-9` | `5-9` | `L3:0` | 5 dedicated `Cortex-X925` cores (5 × 2 MiB L2) on `L3:0` for disk Engram reader, OpenMP host hash, and control loops. |

### Enforcement Lifecycle

1. **`Worker.__init__`**: Calls `configure_worker_cpu_affinity()`, setting `SPARKNET_ROCE_PROXY_CPU=19`, `SPARK_DS41F_ENGRAM_CPU_AFFINITY=5-9`, and `OMP_NUM_THREADS=5` before communicator initialization.
2. **`Worker.init_device`**: Initializes the distributed environment (spawning `sparknet-proxy` on `CPU 19`) and applies `apply_worker_thread_affinity()`.
3. **`Worker.load_model`**: Re-applies `apply_worker_thread_affinity()` after `load_model()` completes, restoring `VLLM::Worker_TP*` main thread to `CPU 18` and moving all auxiliary threads (`engram-reader`, OpenMP, `gloo_tcp_loop`, `pt_nccl_watchdg`, `pt_tcpstore_uv`, `cuda-EvtHandlr`) to `5-9` after `HostEngramHash.__init__` finishes initializing `libgomp`.
4. **`EngineCoreProc.run_engine_core`** & **`APIServer` entrypoints**: Apply `apply_role_cpu_affinity("engine_core")` (`CPU 17`) and `apply_role_cpu_affinity("api_server")` (`CPUs 15-16` or `0-4,10-16`).

---

## 3. Exact Verification & Benchmark Commands on `sp3` (`NVIDIA GB10`)

All commands were executed on `sp3` (`NVIDIA GB10`, kernel `7.0.0-1024-nvidia`) serialized under `flock /tmp/gpu.lock` across two fresh container boots (`boot1` and `boot2`):

```bash
for boot in boot1 boot2; do
  flock /tmp/gpu.lock docker run --rm --gpus all --security-opt seccomp=unconfined --ulimit memlock=-1 --entrypoint "" \
    -v /tmp/vllm-prep-roce-affinity/vllm/utils/cpu_affinity.py:/opt/spark-ds41f/candidate/vllm/vllm/utils/cpu_affinity.py:ro \
    -v /tmp/vllm-prep-roce-affinity/vllm/v1/worker/gpu_worker.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu_worker.py:ro \
    -v /tmp/vllm-prep-roce-affinity/vllm/v1/engine/core.py:/opt/spark-ds41f/candidate/vllm/vllm/v1/engine/core.py:ro \
    -v /tmp/vllm-prep-roce-affinity/vllm/entrypoints/launchers/api_server/entry.py:/opt/spark-ds41f/candidate/vllm/vllm/entrypoints/launchers/api_server/entry.py:ro \
    -v /tmp/vllm-prep-roce-affinity/vllm/entrypoints/cli/serve.py:/opt/spark-ds41f/candidate/vllm/vllm/entrypoints/cli/serve.py:ro \
    -v /tmp/vllm-prep-roce-affinity/vllm/models/deepseek_v4_1/engram_rows/_reader.c:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/engram_rows/_reader.c:ro \
    -v /tmp/vllm-prep-roce-affinity/tests/v1/worker/test_roce_cpu_affinity.py:/opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_roce_cpu_affinity.py:ro \
    -v /tmp/vllm-prep-roce-affinity/tests/kernels/test_deepseek_v41_engram_host_placement.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_engram_host_placement.py:ro \
    -v /tmp/spark-ds41f-roce-affinity/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-b00ad50c/src/sparknet:/opt/sparknet-src:ro \
    -v /tmp/spark-ds41f-roce-affinity/experiments/2026-11-08-roce-proxy-fast-core-affinity:/opt/exp \
    spark-ds41f:1107 bash -c "
      PYTHONPATH=/opt/sparknet-src/src python3 -m unittest discover -s /opt/sparknet-src/tests -v > /opt/exp/receipts/sparknet-unittest-${boot}.txt 2>&1 &&
      python3 -m pytest --noconftest -v \
        /opt/spark-ds41f/candidate/vllm/tests/v1/worker/test_roce_cpu_affinity.py \
        /opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_engram_host_placement.py \
        > /opt/exp/receipts/vllm-pytest-${boot}.txt 2>&1 &&
      python3 /opt/exp/bench_roce_core_affinity.py \
        --boot-label ${boot} \
        --warmup-steps 60 \
        --bench-steps 500 \
        --output /opt/exp/receipts/bench-${boot}.json
    "
done
```

---

## 4. Two-Boot Benchmark Results (`500 steps × 116 collectives = 58,000 collectives/boot`)

### 4.1 Per-Step 116-Collective Communication Time (`ms`)

| Regime | Placement (`proxy` / `worker` / `api` / `engine` / `aux`) | Boot | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | p99.9 (ms) | Max (ms) | `nivcsw` (`proxy` / `worker`) |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `unpinned_0_19` | Floating on `10-14` (`Cortex-A725` @ 2.81 GHz, 512 KiB L2) | Boot 1 | 0.272 | 0.271 | 0.282 | 0.286 | 0.309 | 0.329 | `7 / 0` |
| `unpinned_0_19` | Floating on `10-14` (`Cortex-A725` @ 2.81 GHz, 512 KiB L2) | Boot 2 | 0.273 | 0.272 | 0.276 | 0.289 | 0.361 | 0.413 | `7 / 13` |
| `shared_big_5_9_15_19` | Shared `15-19` (`Cortex-X925` @ 3.90 GHz, unpartitioned) | Boot 1 | 0.106 | 0.084 | 0.085 | 0.939 | 0.940 | 0.941 | `1 / 16` |
| `shared_big_5_9_15_19` | Shared `15-19` (`Cortex-X925` @ 3.90 GHz, unpartitioned) | Boot 2 | 0.084 | 0.083 | 0.084 | 0.085 | 0.096 | 0.105 | `1 / 2` |
| **`partitioned_fast_affinity`** | **`19` / `18` / `15-16` / `17` / `5-9` (Disjoint X925)** | **Boot 1** | **0.081** | **0.081** | **0.083** | **0.084** | **0.102** | **0.109** | **`2 / 2`** |
| **`partitioned_fast_affinity`** | **`19` / `18` / `15-16` / `17` / `5-9` (Disjoint X925)** | **Boot 2** | **0.082** | **0.081** | **0.084** | **0.085** | **0.096** | **0.102** | **`0 / 1`** |

### 4.2 Per-Collective One-Shot All-Reduce Latency (`µs`, `3,584 bf16 = 7,168 B`)

| Regime | Boot | Mean (µs) | p50 (µs) | p95 (µs) | p99 (µs) | p99.9 (µs) | Max (µs) | Proxy Wakeup p50 / p99 (µs) | Arrival Skew p50 / p99 (µs) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| `unpinned_0_19` | Boot 1 | 2.271 | 2.256 | 2.336 | 2.480 | 4.992 | 12.640 | `0.128 / 0.144` | `2.144 / 2.384` |
| `unpinned_0_19` | Boot 2 | 2.280 | 2.272 | 2.304 | 2.336 | 4.992 | 143.936 | `0.128 / 0.144` | `2.144 / 2.208` |
| `shared_big_5_9_15_19` | Boot 1 | 0.884 | 0.688 | 0.704 | 0.736 | 1.968 | 856.752 | `0.096 / 0.112` | `0.592 / 0.640` |
| `shared_big_5_9_15_19` | Boot 2 | 0.691 | 0.688 | 0.704 | 0.736 | 1.024 | 21.920 | `0.096 / 0.112` | `0.592 / 0.640` |
| **`partitioned_fast_affinity`** | **Boot 1** | **0.670** | **0.672** | **0.688** | **0.720** | **0.896** | **29.408** | **`0.096 / 0.096`** | **`0.592 / 0.640`** |
| **`partitioned_fast_affinity`** | **Boot 2** | **0.675** | **0.672** | **0.688** | **0.720** | **1.136** | **21.440** | **`0.096 / 0.096`** | **`0.592 / 0.640`** |

### 4.3 Two-Boot Mean & Confidence Summary

- **vs. Unpinned (`unpinned_0_19` on `Cortex-A725` efficiency cores)**:
  - Per-step 116-collective latency drops from **`0.2725 ± 0.0005 ms`** (`Boot 1: 0.272 ms`, `Boot 2: 0.273 ms`) to **`0.0815 ± 0.0005 ms`** (`Boot 1: 0.081 ms`, `Boot 2: 0.082 ms`), a **3.34× speedup (`-70.1%`)**.
  - Per-collective p50 latency drops from **`2.264 ± 0.008 µs`** to **`0.672 ± 0.000 µs`** (**3.37× faster**), and proxy wakeup p99 drops from **`0.144 µs`** to **`0.096 µs`** (**-33.3%**).
- **vs. Shared Big Pool (`shared_big_5_9_15_19` on unpartitioned `Cortex-X925` cores)**:
  - Eliminates Rank-0 `APIServer` / `EngineCore` preemption stalls on `VLLM::Worker_TP0` and `sparknet-proxy` (`856.75 µs` max collective stall and `0.939 ms` p99 step stall in `Boot 1`), holding 116-collective step p99 at **`0.0845 ± 0.0005 ms`** across both boots (**up to 11.2× lower step p99 under Rank-0 API server load**).

---

## 5. Unit Test & Sanitizer Verification Across 2 Boots

- **`sparknet` C proxy + Python test suite** (`receipts/sparknet-unittest-boot1.txt`, `receipts/sparknet-unittest-boot2.txt`):
  - **97 tests passed** in Boot 1 (`13.868s`) and Boot 2 (`13.972s`), including the 31,000-collective ASan/UBSan 3-rank C proxy simulator (`test_proxy_c.py`) and the new `resolve_proxy_cpu` (`auto`/`fast`/`big`/`dedicated`, CPU range `"18-19"`, and `pthread_getaffinity_np` cpuset intersection) tests.
- **`vLLM` CPU & GPU affinity + Engram placement/reader tests** (`receipts/vllm-pytest-boot1.txt`, `receipts/vllm-pytest-boot2.txt`, and `test_deepseek_v41_engram_rows.py`):
  - **14/14 passed** in `test_roce_cpu_affinity.py` + `test_deepseek_v41_engram_host_placement.py` across both Boot 1 and Boot 2, plus **50/50 passed** in `test_deepseek_v41_engram_rows.py` (`64 passed` total).

---

## 6. Memory Observations & Errors Encountered

- **Memory observations**: Zero host or device memory overhead (`cpu_affinity.py` uses sysfs topology inspection and `os.sched_setaffinity`; `roce_proxy.c` uses stack-allocated `cpu_set_t`).
- **Errors encountered during experimentation**:
  - Running `test_deepseek_v41_engram_rows.py` in `docker run` without `--security-opt seccomp=unconfined --ulimit memlock=-1` failed with `RuntimeError: io_uring initialization failed: Operation not permitted`; resolved by passing `--security-opt seccomp=unconfined --ulimit memlock=-1`.

---

## 7. Conclusion

Deterministic fast-core CPU affinity partitioning (`sparknet-proxy` on `CPU 19`, `VLLM::Worker_TP*` main thread on `CPU 18`, `VLLM::EngineCore` on `CPU 17`, Rank-0 `APIServer` on `CPUs 15-16` + efficiency cores, and worker auxiliary threads on `CPUs 5-9`) eliminates cross-role core contention on Rank 0 and prevents `sparknet-proxy` / `Worker_TP*` from floating onto `Cortex-A725` efficiency cores, delivering a **3.34× reduction in 116-call step collective time vs. unpinned cores** (`0.273 ms -> 0.082 ms`) and eliminating sub-millisecond preemption spikes (`0.939 ms -> 0.084 ms` p99) compared to the unpartitioned big-core pool.
