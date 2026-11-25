# Candidate-73 CUPTI & IKET Decode Audit

Base commit: `24950b3eca` (`candidate-73`, frozen pre-release, `spark-ds41f:candidate-73`, `sha256:cfef8ba79ad807de00ff04a31a779759c9f94fb3f5292fee4e422aa26bb96669`).
Hardware: `dgx1..dgx4` (GB10, 64 KiB UMA, 4-node switchless RoCE ring), freshly rebooted (`O13` 512 MiB blocks: `dgx1=217`, `dgx2=210`, `dgx3=224`, `dgx4=233`).

## Goal

Investigate the decode loss previously observed on aged/unrebooted hosts (`c72-r3`) and check CUPTI (4-rank live serving) and streaming IKET (phase-level kernel breakdown on `dgx4`) for any fixable decode regressions or low-hanging TileLang kernel wins before running full rebooted switchless release qualification.

## Findings

### 1. Root cause of earlier decode gap: host fragmentation + idle profiler-config overhead

- On freshly rebooted `dgx1..dgx4` using `prof-tp4.json` (`--profiler-config` attached, even idle), the 19-shape unprofiled engine grid showed `+0.33–0.39 ms/step` in `gap_ms = step_ms - forward_ms - drafter_ms` (`1.658–1.743 ms` vs `1120`'s `1.324–1.363 ms`).
- Re-running the exact same 19-shape engine grid (`replays=100, warmup=2, settle=1`) on `candidate-73/tp4-dev.json` **without** `--profiler-config` (`receipts/grid-noprof.json`) eliminated the `+0.34–0.39 ms` gap completely (`gap_ms` delta `-0.008` to `+0.041 ms` vs `1120`), and **`candidate-73` beat `1120` on 16 of 19 decode shapes**:
  - `1x2@4K`: `-0.57%` (`-0.135 ms`)
  - `1x6@4K`: `-1.72%` (`-0.201 ms`)
  - `2x12@4K`: `-2.04%` (`-0.196 ms`)
  - `4x24@4K`: `-1.95%` (`-0.392 ms`)
  - `8x24@4K`: `-2.64%` (`-0.397 ms`)
  - `8x24@64K`: `-2.08%` (`-0.338 ms`)
  - `8x24@256K`: `-1.60%` (`-0.421 ms`)
  - `12x72@4K`: `-1.00%` (`-0.374 ms`)
  - `16x32@4K`: `-1.17%` (`-0.124 ms`)
- Unprofiled fresh-reboot real `c1` (`d001..d003`, `768` tokens, `313` steps, `receipts/c1-profile.jsonl`): **`26.558 ms/step`**, `2.4537 tok/step` (**`92.39 tok/s`** vs `1120`/`c61`'s `27.654 ms/step` / `86.23 tok/s`).

### 2. 4-rank CUPTI stream-overlap breakdown (`dgx1`, 318 steps, 510,491 kernels)

- Merged GPU timeline across all streams (`receipts/summary-dgx1.json`): span `26.239 ms/step`, GPU busy **`26.085 ms/step` (`99.41%`)**, GPU idle `0.154 ms/step`.
- Every apparent raw-duration increase vs `1120`/`c61` (`experiments/2026-11-17-c1-floor-now/receipts/trace/`) is overlapped on auxiliary streams (`0.0%–0.3%` solo):
  - `glp_steer_write_kernel` (`0.867 ms` raw -> `0.000 ms` solo, 0.0%)
  - `glp_steer_kernel` (`0.859 ms` raw -> `0.000 ms` solo, 0.1%)
  - `mhc_finalize_sinkhorn_kernel` (`0.849 ms` raw -> `0.002 ms` solo, 0.2%)
  - `mhc_finalize_norm_kernel` (`0.496 ms` raw -> `0.001 ms` solo, 0.3%)
  - `compress_gather_partial_state_kernel` (`0.154 ms` raw -> `0.000 ms` solo, 0.0%)
- Major raw-kernel reductions vs `1120`/`c61`:
  - `moe_grouped_fp8_fp4_gemm_kernel`: `-1.053 ms/step` (`12.983 -> 11.931 ms`)
  - `gemm_splitk_reduce_cast_kernel`: `-0.689 ms/step` (`1.147 -> 0.458 ms`)
  - `mhc_project_streams_kernel`: `-0.525 ms/step` (`2.406 -> 1.880 ms`)
  - `gemm_mxfp8_kernel`: `-0.313 ms/step` (`7.301 -> 6.988 ms`)
  - `oneshot_allreduce_kernel`: `-0.259 ms/step` (`4.438 -> 4.179 ms`)
  - `sparse_mla_kernel` + `sparse_mla_kernel_1`: `-0.201 ms/step`
  - `vocab_packed_projection_kernel`: `-0.071 ms/step` (`1.071 -> 1.000 ms`)
- Top exclusive (`solo`) wall-time consumers on `c1`:
  1. `moe_grouped_fp8_fp4_gemm_kernel`: `9.440 ms` share, `7.289 ms` solo
  2. `vocab_packed_projection_kernel`: `1.000 ms` share, `1.000 ms` solo (100%)
  3. `gemm_mxfp8_kernel`: `3.420 ms` share, `0.756 ms` solo
  4. `vocab_screen_greedy_partials_kernel`: `0.648 ms` share, `0.648 ms` solo (99.8%)
  5. `ncclDevKernel_AllGather_RING_LL`: `0.177 ms` share, `0.177 ms` solo (100%)

### 3. Streaming IKET phase profiling & clean DRAM-bandwidth audit (`dgx4`)

Using `experiments/2026-11-04-iket-streaming` (`receipts/bench_clean.json`, `receipts/iket_summary.json`) at exact DeepSeek-V4.1 TP4 dimensions (`hidden_size=5120`, `vocab_size=129280`, `N_shard=32320`, `E=384`, `topk=6`, `I_shard=576`):
- `vocab_packed_projection` (`M=6, N=32320, K=5120`, 248.2 MB 12-bit packed BF16): `1.060 ms` (**`234.2 GB/s`**); balanced `load.cp_async#2` (`5.55 us`) vs `compute.mma` (`4.52 us`).
- `vocab_screen_greedy_partials` (`M=5, N=32320, K=5120`, 165.5 MB 8-bit): `0.691–0.722 ms` (**`229–239 GB/s`**); balanced `load.cp_async#2` (`3.94 us`) vs `compute.mma` (`3.24 us`).
- `MoeExpertsV41Plan` (`num_stages=3, down_num_stages=3`): `M=3` `0.385 ms` (**`232.9 GB/s`**), `M=6` `0.728 ms` (**`246.2 GB/s`**), `M=24` `2.711 ms` (**`264.3 GB/s`**), `M=48` `5.321 ms` (**`269.3 GB/s`**); balanced `load#2` (`4.69 us`), `compute.mma` (`4.63 us`), `wait.mbarrier` (`4.53 us`).
- `mxfp8_gemm_decode` (`o_proj 5120x2048`): tile/thread sweep (`bN=64/128`, `threads=64/128`, `num_stages=3/4`) all tie within `2 us`; live `l2_prefetch` overlap drives it to `65.4 us` avg (`23.9 us` min).

## Conclusion

No hidden decode regression or low-hanging TileLang kernel win remains in `candidate-73`. The top exclusive decode kernels are pinned at `230–269 GB/s` against GB10's DRAM roofline and GPU busy utilization is `99.41%`. Proceeding directly to full rebooted switchless release qualification on frozen `candidate-73` (`24950b3eca`).
