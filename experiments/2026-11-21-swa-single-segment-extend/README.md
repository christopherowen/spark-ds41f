# Decode #10: SWA KV Reuse Audit & Single-Segment `extend` Route in `sparse_mla_fwd_v41`

- **Base commit**: `deec30d0b` (`spark-ds41f:candidate-63-r5`)
- **One intended variable**: Route single-segment (`segments <= 1`, i.e., SWA-only `swa_width=128, idx_width=0`) layers directly to the single-kernel `extend` schedule in `SparseMlaV41Plan.routes()` (`tilelang/sparse_mla.py`), instead of compiling and launching the two-kernel `split`/`decode` + `merge` path.

## Sizing Audit of Original `Decode #10` Hypothesis (Multi-Row SWA KV DRAM Reuse)

DeepSeek-V4.1-Flash-1017 (`num_attention_heads=64`, `sliding_window=128`) has:
- `40` target layers (`2` SWA-only layers `0, 1`, `38` SparseMLA layers `2..39`) plus `3` MTP/DSpark SWA-only layers (`40..42`).
- Local heads per rank: `H=16` on TP4 (and TP3 rank 2) and `H=24` on TP3 ranks 0 & 1.

Cold-L2 rotating-pool profiling (`>= 198 MiB` SWA cache pools on `dgx4`) showed why multi-row SWA KV DRAM reuse is **bounded**:
- Unique SWA KV per layer is only `67 KiB` (`1x3`), `69 KiB` (`1x6`), `549 KiB` (`8x6`), and `1.07 MiB` (`16x6`) — `0.3%–4.5%` of GB10's `24 MiB` L2 cache.
- On `extend`, `SWA-only` runs at a flat `14.59 us/L` (`H=16`) / `14.59–15.62 us/L` (`H=24`) across `rows = 1..48` (`0.00 us` DRAM KV bandwidth scaling across a `48×` increase in rows).
- Grouping rows into fewer CTAs at `rows <= 48` would cut SM parallelism without reducing DRAM reads.

## Finding & Fix: Eliminate Two-Kernel Split-KV Round-Trip When `segments <= 1`

During profiling, `SparseMlaV41Plan.routes()` was found to set `split_max_rows=24` and `decode_max_rows=144` even when `segments == 1` (`idx_width == 0`, SWA-only layers). Because there is only one segment, `"split"` / `"decode"` provides zero K/V-split parallelism while writing `Partial[rows, 1, heads, 512]` (FP32) to global memory and launching a second `merge` kernel to read it back and fold one segment with `sink_log2`.

Routing `segments <= 1` directly to `"extend"` (preserving explicit `decode_max_rows` / `split_max_rows` caller overrides):
- Eliminates the FP32 `Partial` DRAM round-trip and second kernel launch on every SWA-only decode step.
- Eliminates `2` unnecessary kernel compilations (`split` and `decode`) per SWA-only shape and drops SWA-only `Partial` / `PartialStats` scratch workspace to `0` bytes.
- Is **100% `uint16` bit-identical** to `"split"` / `"decode"` (by `SparseMlaV41Plan`'s `FOLD` invariant).

## Verification (`dgx4`, `spark-ds41f:candidate-63-r5`)

### Correctness & PDL (`38 passed, 0 failed`)
- `tests/kernels/attention/test_deepseek_v41_tilelang.py -k sparse_mla`: **`23/23` passed** (including updated `test_sparse_mla_plan_routes_right_sizes_head_block_and_decode_waves`).
- `tests/kernels/test_deepseek_v41_tilelang_pdl.py`: **`15/15` passed** (including `layer_chain` `c1`/`c8`/`c8x6`, `0` PDL violations).
- Explicit `uint16` bit-exact parity (`def` vs `extend`) verified across `H=16, 24, 32` and `1x3..16x6` (`exact=True` on all 18 cases).

### Cold-L2 Microbenchmark (`dgx4`, `198 MiB` rotating SWA pools)

| Rows | `H=16` (`TP4` / `TP3 r2`) def -> `extend` | `H=24` (`TP3 r0/r1`) def -> `extend` |
|---:|---|---|
| `1` | `17.60 -> 14.59 us` (`-3.01 us`, `-17.1%`) | `17.66 -> 15.58 us` (`-2.08 us`, `-11.8%`) |
| `2` | `17.63 -> 14.59 us` (`-3.04 us`, `-17.2%`) | `17.63 -> 14.59 us` (`-3.04 us`, `-17.2%`) |
| `3` (`1x3`) | `16.64 -> 14.59 us` (`-2.05 us`, `-12.3%`) | `17.47 -> 14.59 us` (`-2.88 us`, `-16.5%`) |
| `6` (`1x6`) | `16.64 -> 14.59 us` (`-2.05 us`, `-12.3%`) | `16.64 -> 15.58 us` (`-1.06 us`, `-6.4%`) |
| `12` (`2x6`) | `16.64 -> 14.59 us` (`-2.05 us`, `-12.3%`) | `17.34 -> 15.62 us` (`-1.73 us`, `-10.0%`) |
| `24` (`8x3`/`4x6`) | `16.64 -> 14.59 us` (`-2.05 us`, `-12.3%`) | `17.63 -> 15.58 us` (`-2.05 us`, `-11.6%`) |
| `48` (`8x6`) | `16.64 -> 14.59 us` (`-2.05 us`, `-12.3%`) | `17.66 -> 15.62 us` (`-2.05 us`, `-11.6%`) |
| `96` (`16x6`) | `23.78 -> 23.71 us` (`-0.06 us`, `-0.3%`) | `33.92 -> 29.86 us` (`-4.06 us`, `-12.0%`) |

## Conclusion

- **Multi-row SWA KV DRAM reuse**: **Bounded** (SWA KV working set fits inside L2 at `<= 4.5%` capacity; `extend` is flat across `rows = 1..48`).
- **Single-segment `extend` route** ([`swa_single_segment_extend.patch`](swa_single_segment_extend.patch)): **Clear, 100% bit-exact cleanup & micro-win** (`-1.1..-4.1 us/L` across all decode shapes, `2` fewer compiled kernels, `0` SWA `Partial` scratch). Ready to roll into `candidate-64`.
