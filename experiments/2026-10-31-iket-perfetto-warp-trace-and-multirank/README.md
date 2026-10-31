# Experiment: Perfetto Warp-Trace Base, Multi-Slice Stitching, and 4-Rank Collective Alignment (`iket_perfetto.py` & `iket_table.py`)

## Hypothesis & Objective

Previously, `iket_perfetto.py` and `iket_table.py` had three key gaps for production step profiling:

1. **Interim `--base-nsys` dependency**:
   - `iket_perfetto.py` required an `nsys export --type sqlite` file (`--base-nsys`) as its uninstrumented base timeline, with comments noting it was an interim stand-in for a `run-iket --warp-trace` base run.
   - `nsys` adds a separate profiling tool and provides only kernel-level start/end spans (`CUPTI_ACTIVITY_KIND_KERNEL`), lacking per-warp lifetimes and SM/CTA placement for uninstrumented or skipped kernels.
2. **No multi-slice family stitching (`--slice`)**:
   - Profiling a full 43-layer DS4.1 decode step with all families instrumented in one pass causes high range pressure and inflates kernel durations.
   - Slicing by kernel family (`VLLM_DS41_IKET_SKIP` excluding other families per slice) keeps instrumentation overhead low, but neither `iket_perfetto.py` nor `iket_table.py` could stitch multiple family-sliced traces onto a single base timeline.
3. **Single-rank only (no multi-rank collective alignment)**:
   - On the 4-rank DGX Spark ring (`dgx1..dgx4`), `iket_perfetto.py` could only render a single rank and placed static text markers where TP all-reduces would occur.
   - Because each node's GPU `%globaltimer` has a clock offset and small linear drift relative to rank 0, combining multi-rank traces requires clock alignment across collective rendezvous points to distinguish **arrival jitter wait** (waiting for a late-arriving peer rank) from **fabric transport time**.

This experiment upgrades `iket_perfetto.py` and `iket_table.py` with:

1. **`--base-iket BASE.json` (`iket_base`)**:
   - Reads a `run-iket profile --warp-trace` (`VLLM_DS41_IKET` off) `iket.decoded_results.json` as the timeline of record.
   - Pairs instrumented launches by `(source, kernel, grid, block)` ordinal, rescaling their traced warps and ranges into the base launch's span (`real_us`, `instrumented_us`, `scale`, `overhead`), while preserving exact base warp lifetimes and SM/CTA residency for uninstrumented/skipped kernels (including ATen, Triton, NCCL, and sparknet).
2. **`--slice SLICE.json [SLICE.json ...]` and `--allow-mismatch` (`stitch_slices`, `merge_slice_launches`)**:
   - Verifies that each family-sliced IKET run's decode step matches the base step's `(kernel, grid, block)` launch sequence, refusing with a unified diff on mismatch unless `--allow-mismatch` is set.
   - Merges each slice's string table and location table and rescales its instrumented launches into the base timeline in both `iket_perfetto.py` and `iket_table.py`.
3. **`--rank NODE=DECODED.json [NODE=DECODED.json ...]` (`align_ranks`, `write_multirank`)**:
   - Fits an affine clock transformation $t^{(0)}(t) = \alpha_r + \beta_r (t - t_{r,0})$ per rank via ordinary least squares on collective rendezvous completion timestamps (`oneshot_allreduce*`, `oneshot_allgather*`, `ncclDevKernel*`, `cross_device_reduce*`), reporting offset ($\mu\text{s}$), drift ($\text{ppm}$), and RMS/max fit residuals ($\mu\text{s}$).
   - Computes per-collective arrival spread, identifies the last-arriving peer (`late_rank`), and decomposes each rank's collective duration into `jitter_wait_us` vs. `fabric_us`, rendering a top-level `0 collective alignment` track above the per-rank GPU process tracks.
4. **Payload & Bandwidth Reporting in `iket_table.py`**:
   - Reads static `tma_bytes` and `cp_async_bytes_per_thread` from `--sources` site manifests (and runtime `payload` values from range records) to report `bytes_per_warp` and `gbps_per_warp` (`[GB/s/warp]`) per load range.

## Verification on `dgx4`

Verified on `dgx4` (`spark-ds41f:1030`, holding `/tmp/spark-lab-gpu.lock`):

- `pytest --noconftest -v test_iket_perfetto.py`: **4 passed in 0.03s**:
  - `test_base_iket_and_multi_slice_stitching`: verifies `--base-iket` + `--slice` stitching across `sparse_mla_kernel` and `moe_grouped_fp8_fp4_gemm_kernel` slices, confirming rescaled spans and merged traced warps.
  - `test_slice_sequence_mismatch_refuses_unless_allowed`: verifies `SystemExit` with unified diff when a slice's launch sequence diverges from the base step, and override with `--allow-mismatch`.
  - `test_multirank_collective_clock_alignment`: verifies 4-rank (`dgx1..dgx4`) OLS clock offset (`-420..+800 us`) and drift (`-15..+20 ppm`) recovery (`residual_rms_us < 0.01 us`) and late-peer detection (`late_rank == "dgx3"`, `arrival_spread_us == 18.0 us`).
  - `test_iket_table_slice_merge_and_payload_bandwidth`: verifies `iket_table.py` slice merging and static `tma_bytes` bandwidth computation (`gbps_per_warp`).

## Files

- `iket_perfetto.py`: Streaming Perfetto trace builder with `--base-iket`, `--base-nsys`, `--slice`, `--allow-mismatch`, and `--rank`.
- `iket_table.py`: Markdown/JSON per-phase and per-range table generator with `--slice` merging and payload bandwidth reporting.
- `test_iket_perfetto.py`: Unit test suite for `iket_perfetto.py` and `iket_table.py`.
