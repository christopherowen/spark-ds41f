# Unified Network IKET and `ROCE_TRACE` Across All Four `sparknet` Collectives

## Summary

This experiment closes three profiling gaps between `sparknet.oneshot` (`SPARKNET_ROCE_TRACE`) and TileLang in-kernel event tracing (`vllm/models/deepseek_v4_1/tilelang/iket.py` and `stream/export.py`):

1. **`ROCE_TRACE` coverage across all four collective families (`allreduce`, `allgather`, `reducescatter`, `exchange`)**:
   - `_allgather_tilelang.py`, `_reducescatter_tilelang.py`, and `_exchange_tilelang.py` now accept `traced: bool = False` in `_build` and `trace: bool = False` in `get_launcher` / `_process_key`, matching `_oneshot_tilelang.py`.
   - `runtime.py` includes `bool(self._trace_base)` in `_gather_launcher_key` and `_reducescatter_launcher_key` so untraced (`SPARKNET_ROCE_TRACE=0`, the default) and traced (`SPARKNET_ROCE_TRACE=1`) kernels cache separately with zero overhead when tracing is disabled.
   - `roce_trace_begin` in `_device.py` encodes the collective operation code (`OP_CODES = {"allreduce": 0, "allgather": 1, "reducescatter": 2, "exchange": 3}`) into bits `48..63` of `W_META` (`nbytes | ((grid & 0xffff) << 32) | ((op & 0xffff) << 48)`), preserving the 32-word record ABI (`RECORD_WORDS = 32`) and backward compatibility (`0` = `allreduce`).
   - `sparknet.oneshot.trace` exposes `record_op`, `record_grid`, `record_nbytes`, `op_counts`, and `select(..., op=...)`, and includes `"ops_by_kind"` per rank in `summarize(...)`.

2. **CTA-0 RDMA peer flag polling (`wait.flags`) vs. non-zero CTA local barrier spin (`wait.barrier`)**:
   - In `_device.py`, `roce_wait_cta0` is factored out of `roce_wait_peers` for non-zero CTAs (`blockIdx.x != 0`) that spin on CTA 0's local release of `counters[arrival_index]`, and both `roce_wait_peers` and `roce_wait_cta0` synchronize warp 0 (`if (tidx < 32) { __syncwarp(); }`) so lane 0 (which emits warp-level IKET events) does not exit before active `(peer, lane)` pollers in warp 0 complete.
   - All four TileLang collective kernels emit `if bx == 0: T.call_extern("roce_wait_peers", ...)` and `else: T.call_extern("roce_wait_cta0", ...)`.
   - `iket.py` classifies `roce_wait_cta0` and `roce_wait_device_eq` as `("wait", "barrier")` before `("wait", "flags", [r"\broce_wait_\w+\s*\("])`, so multi-CTA collectives record `wait.flags` exclusively on CTA 0 and `wait.barrier` exclusively on CTAs `1..grid_x-1` under both `fine` and `phase` granularities.

3. **Unified `%globaltimer` Perfetto export (`sparknet.oneshot.trace`, `iket.py`, and `stream/export.py`)**:
   - `sparknet.oneshot.trace` adds `clock_offsets(traces)`, `perfetto_events(traces, ...)`, and `export_perfetto(traces, output, ..., iket_trace=...)` (plus `--op`, `--perfetto`, and `--iket-trace` CLI flags), converting per-rank GPU collective spans (`staging`, `wait.flags`, `reduce`), CPU RoCE proxy spans (`proxy.wake`, `proxy.post`, `proxy.relay`, `proxy.drain` shifted by `clock_offset_ns`), and per-(peer, lane) RDMA flag spans (`flag.peer<P>.lane<L>` annotated with `doorbell_skew_us` and `one_way_wire_us`) onto the reference rank's GPU `%globaltimer` microsecond timeline.
   - `iket.py` adds `roce_trace_events` and `merge_roce_traces`, and `stream/export.py` accepts `--roce-trace` / `roce_traces` to stream both GPU SM warp ranges and `sparknet` RoCE/proxy records into a single Chrome/Perfetto JSON file.

## Base Commit and Environment

- `spark-ds41f` base commit: `cc20ed18bedd4072b754695a93e235c9c240fa41`
- `sparknet` base commit: `1fd6a1fbd8dc4884057046d1ebd93d585c9e4bcf`
- Container image: `spark-ds41f:1107` (`/opt/spark-ds41f/candidate/vllm` at `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` + patch series through `0757`)
- Cluster: `sp5..sp8` (4x NVIDIA GB10, switched RoCE v2 fabric over dual HCAs `rocep1s0f0` on `10.30.0.0/24` and `roceP2p1s0f0` on `10.30.1.0/24`, `topology="direct"`, `stripe_count=2`)

## Intended Variable

Enable `ROCE_TRACE` across all four `sparknet.oneshot` collective kernels (`allreduce`, `allgather`, `reducescatter`, `exchange`), separate CTA-0 RDMA flag polling (`wait.flags`) from non-zero CTA local barrier waiting (`wait.barrier`) in `_device.py` and `iket.py`, and export `sparknet` GPU/proxy/RDMA flag traces onto the shared `%globaltimer` Perfetto timeline.

## Workload Identity and Exact Commands

### Unit and Kernel Compilation Tests (`sp8`)

```bash
flock /tmp/gpu.lock docker run --rm --gpus all --ipc=host --entrypoint '' \
  -v /tmp/net-iket-test:/tmp/net-iket-test \
  -v /tmp/net-iket-test/sparknet:/usr/local/lib/python3.12/dist-packages/sparknet \
  -v /tmp/net-iket-test/vllm/iket.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/iket.py \
  -v /tmp/net-iket-test/vllm/test_deepseek_v41_tilelang_iket.py:/opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_tilelang_iket.py \
  spark-ds41f:1107 bash -lc '
    python3 -m unittest discover -s /tmp/net-iket-test/tests &&
    pytest --noconftest -o addopts="" -v /opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_tilelang_iket.py
  '
```

- Result: `99` `sparknet` host/simulator tests passed (`OK`); `25` `test_deepseek_v41_tilelang_iket.py` unit + GPU compilation tests passed (`25 passed in 64.40s`).

### 4-Node Multi-Rank GPU Correctness & Trace Verification (`sp5..sp8`)

```bash
# Executed across r=0..3 on sp5..sp8 under flock /tmp/gpu.lock:
torchrun --nnodes=4 --nproc-per-node=1 --node-rank=$r --master-addr=10.30.0.13 --master-port=29651 \
  -m pytest --noconftest -o addopts='' -v \
  -k 'test_roce_trace_records_all_four_collectives or test_reduce_scatter_matches_reference or test_exchange_matches_all_to_all' \
  /tmp/net-iket-test/tests/gpu/test_oneshot_gpu.py
```

- Result: `21 passed in 21.93s` across all 4 ranks.

### 4-Node Multi-Rank Probe Across Two Independent Boots (`sp5..sp8`)

Workload (`probe_roce_iket.py`):
- `world_size = 4` (`sp5..sp8`), `dtype = bfloat16`, `nbytes = 65536` (`64 KiB` per collective shard, `grid_x = 8` CTAs of `512` threads).
- `1` warmup iteration + `24` measured iterations of all 4 collectives (`allreduce`, `allgather`, `reducescatter`, `exchange` = `96` measured collectives + `4` warmup collectives = `100` ops per rank).
- Steps `0..11` (`48` collectives): synchronized launch after `dist.barrier(); torch.cuda.synchronize()`.
- Steps `12..23` (`48` collectives): Ranks `1` and `3` inject a `15 us` host spin delay before launching the collective to test cross-rank arrival skew (`doorbell_b_minus_a`) vs. wire transfer latency (`one_way_a_to_b` / `one_way_b_to_a`).
- Measured in two independent boots (`boot1` and `boot2`) across three arms:
  1. Untraced (`SPARKNET_ROCE_TRACE=0`, no `--stream`) -> `runs/untraced-boot{1,2}.json`
  2. `ROCE_TRACE` (`SPARKNET_ROCE_TRACE=1`, no `--stream`) -> `runs/traced-boot{1,2}.json`
  3. Unified `ROCE_TRACE` + streaming IKET (`SPARKNET_ROCE_TRACE=1 --stream`, `ctas=0, iters=1, warps=1`) -> `runs/unified-boot{1,2}.json`

## Measured Results (`sp5..sp8`, Boot 1 and Boot 2)

### 1. Host Collective Latency (`SPARKNET_ROCE_TRACE=0` vs. `SPARKNET_ROCE_TRACE=1`, Rank 0 `p50_us`)

| Collective (`64 KiB`, `grid_x=8`) | Boot 1 Untraced (`p50_us`) | Boot 1 `ROCE_TRACE=1` (`p50_us`) | Boot 2 Untraced (`p50_us`) | Boot 2 `ROCE_TRACE=1` (`p50_us`) |
| --- | ---: | ---: | ---: | ---: |
| `allreduce` | 64.41 | 65.28 | 64.34 | 64.06 |
| `allgather` | 65.42 | 66.12 | 63.56 | 64.54 |
| `reducescatter` | 67.08 | 67.37 | 67.33 | 65.82 |
| `exchange` | 73.08 | 73.85 | 69.55 | 71.98 |

 When `SPARKNET_ROCE_TRACE=0` (default), `_build(..., traced=False)` emits no `roce_trace_*` calls and `#define ROCE_TRACE 0` is compiled in. When `SPARKNET_ROCE_TRACE=1` is enabled, all four collective families record 100% of their launches (`ops_by_kind = {"allreduce": 24, "allgather": 24, "reducescatter": 24, "exchange": 24}` steady-state ops per rank) within `< 1 us` of untraced latency.

### 2. Per-Collective GPU & CPU RoCE Proxy Phase Breakdown (`ROCE_TRACE=1`, Rank 0 `p50_us`)

| Collective | Boot | `total` (`p50_us`) | `staging` (`p50_us`) | `proxy_wake` (`p50_us`) | `proxy_post` (`p50_us`) | `wait` (`p50_us`) | `reduce` / `output` (`p50_us`) |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `allreduce` | Boot 1 / Boot 2 | 42.74 / 42.74 | 1.79 / 1.92 | 0.17 / 0.17 | 1.34 / 1.18 | 38.30 / 38.22 | 2.62 / 2.59 |
| `allgather` | Boot 1 / Boot 2 | 43.17 / 42.51 | 1.95 / 2.02 | 0.15 / 0.17 | 1.18 / 1.18 | 38.64 / 37.98 | 2.59 / 2.53 |
| `reducescatter` | Boot 1 / Boot 2 | 44.18 / 42.66 | 2.59 / 2.66 | 0.15 / 0.18 | 1.33 / 1.18 | 38.90 / 37.39 | 2.62 / 2.62 |
| `exchange` | Boot 1 / Boot 2 | 43.12 / 41.33 | 2.82 / 2.78 | 0.14 / 0.17 | 1.39 / 1.20 | 37.14 / 35.98 | 2.62 / 2.58 |

### 3. Cross-Rank Arrival Skew (`doorbell_b_minus_a`) vs. Wire Transfer (`one_way_b_to_a`)

Across all 6 rank pairs (`[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]`):
- **Unstaggered Pair `[0, 2]`** (neither Rank 0 nor Rank 2 has injected delay):
  - `doorbell_b_minus_a` `p50_us`: `3.94 us` (Boot 1), `3.13 us` (Boot 2)
  - `one_way_a_to_b` `p50_us`: `14.24 us` (Boot 1), `13.85 us` (Boot 2)
  - `one_way_b_to_a` `p50_us`: `13.90 us` (Boot 1), `14.34 us` (Boot 2)
- **Staggered Pair `[0, 1]`** (Rank 1 delayed by `15 us` on steps `12..23`):
  - Synchronized slice (`steps 0..11`): `doorbell_b_minus_a` `p50_us = 10.74 us` (Boot 1) / `9.16 us` (Boot 2); `one_way_a_to_b` `p50_us = 12.39 us` (Boot 1) / `11.60 us` (Boot 2).
  - Skewed slice (`steps 12..23`): `doorbell_b_minus_a` increases to `p50_us = 19.11 us` (Boot 1) / `17.49 us` (Boot 2) (and overall `p50_us = 23.01 us` / `21.16 us` on the full-run NTP clock offset), while Rank 1 -> Rank 0 wire delay (`one_way_b_to_a`, where Rank 0 is already polling when Rank 1 posts) remains `p5_us = 11.28 us / 10.32 us`, `p50_us = 12.65 us / 12.66 us`.

### 4. CTA-0 `wait.flags` vs. Non-Zero CTA `wait.barrier` Verification in Hardware IKET Traces

Across all 4 ranks (`r=0..3`) and both boots (`unified-boot1.json` and `unified-boot2.json`), for every collective kernel (`oneshot_allreduce_kernel`, `oneshot_allgather_kernel`, `oneshot_reducescatter_kernel`, `oneshot_exchange_kernel`) launched with `grid_x = 8` CTAs over `25` launches (`1` warmup + `24` measured):
- **CTA 0 (`cta == 0`)**: `wait.flags = 25`, `wait.barrier = 0` (`100%` of CTA-0 waits are RDMA peer flag waits).
- **Non-zero CTAs (`cta in 1..7`)**: `wait.barrier = 175` (`7 CTAs * 25 launches`), `wait.flags = 0` (`100%` of non-zero CTA waits are local barrier waits on CTA 0).
- **Unified Perfetto export (`perfetto-unified.json` / `perfetto-merged-via-iket.json`)**: `41,602` balanced IKET events (`12,801` kernel lifetimes + `8,000` warp phase ranges) merged with `5,028` (Boot 1) / `5,026` (Boot 2) `sparknet` `ROCE_TRACE` events across all 4 ranks onto the shared `%globaltimer` timeline.

## Errors Encountered and Resolved

1. `test_sparknet_all_four_collectives_compile_with_iket_and_roce_trace` initially checked `"roce_trace_begin" not in src` on the full generated CUDA translation unit (which includes the `#if ROCE_TRACE` helper definitions in `_device.py`). Scoping the check to `#define ROCE_TRACE 0` in the header and `roce_trace_begin(` in the `__global__` body verified that untraced kernels emit zero trace calls.
2. Launching `test_oneshot_gpu.py` on `sp5..sp8` with `SPARKNET_ROCE_GID_INDEX=3` and key-value `SPARKNET_ROCE_SUBNETS` failed `sparknet` 0.5.0's environment validation (`SPARKNET_ROCE_GID_INDEX is no longer read` and `SPARKNET_ROCE_SUBNETS must be a JSON object`). Unsetting `SPARKNET_ROCE_GID_INDEX` and passing JSON `SPARKNET_ROCE_SUBNETS='{"rocep1s0f0":"10.30.0.0/24","roceP2p1s0f0":"10.30.1.0/24"}'` resolved the setup error.
3. In `probe_roce_iket.py`, overriding `PYTHONPATH=/tmp/net-iket-test/probe` without `/opt/spark-ds41f/candidate/vllm` caused `ModuleNotFoundError: No module named 'vllm.models.deepseek_v4_1'`; setting `PYTHONPATH=/tmp/net-iket-test/probe:/opt/spark-ds41f/candidate/vllm` and adding `torch.cuda.synchronize()` immediately after `dist.barrier()` isolated collective latency from NCCL barrier wakeup.

## Memory Observations

- When `SPARKNET_ROCE_TRACE=0` (default), zero additional host or GPU memory is allocated.
- When `SPARKNET_ROCE_TRACE=1`, each rank maps a `16,781,312`-byte (`16 MiB + 4 KiB` header) file in `/dev/shm/sparknet-trace-r<rank>-p<pid>-<id>.bin`.
- When streaming IKET is enabled (`--stream`), the collector allocates a bounded `16 MiB` managed ring buffer (`ring_bytes = 16 << 20`) with zero `producer_waits` across all `41,602` emitted records.

## Conclusion

All four `sparknet.oneshot` collective families (`allreduce`, `allgather`, `reducescatter`, `exchange`) now support `SPARKNET_ROCE_TRACE=1` with zero overhead when disabled, separate CTA-0 RDMA flag polling (`wait.flags`) from non-zero CTA barrier waiting (`wait.barrier`) in both `_device.py` and `iket.py`, and export unified GPU SM warp + CPU RoCE proxy + per-(peer, lane) RDMA flag timelines in Chrome/Perfetto format.
