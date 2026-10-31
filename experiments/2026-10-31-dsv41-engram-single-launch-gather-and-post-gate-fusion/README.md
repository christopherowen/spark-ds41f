# Single-Launch Disk Engram Step, Fused `mhc.post` + `engram_gate`, and Native Resident Lookup (`rc21`)

## Overview
Implements Part 2 Item 10 of the TileLang kernel-fusion audit on `rc21` (`5cfcc5813fc`), while bringing along the native `@tilelang.jit` resident Engram table lookup and host geometry from PR `#802`.

## Key Changes
1. **Single-Launch Disk Engram Row Wait + Gather + Release (`vllm/models/deepseek_v4_1/tilelang/engram_rows.py`)**:
   - Added single-CTA `@tilelang.jit` `engram_rows_step_kernel` (`with T.Kernel(1, threads=256)`) that fuses `engram_wait_rows(control)` on thread 0 + `T.sync_threads()` + multi-row FP8 E4M3/E8M0 dequantize/gather across all `(token, column)` pairs + `T.sync_threads()` + `engram_release_rows(control)` on thread 0 into a single kernel launch.
   - Eliminates 2 kernel launches per disk Engram layer (**4 launches/step**) while keeping host-memory polling to a single thread and preserving timeout/failure trapping and `SPARK_DS41F_ENGRAM_TIMELINE` instrumentation.
2. **Fused `mhc.post` + `engram_gate` / `engram_gate_blocked` (`vllm/models/deepseek_v4_1/tilelang/engram.py`, `common/engram.py`, `nvidia/model.py`)**:
   - Added `@tilelang.jit` `engram_post_gate_kernel` and `engram_post_gate_blocked_kernel` (and PDL chaining on all 4 Engram gate kernels) that reconstruct each stream in registers from `(previous_output, residual, post_mix, res_mix)` using `post_streams`' exact FP32 FMA sequence (`comb[0, h] * r[0]`, then `fma(post[h], x, ...)`, then `fma(comb[m, h], r[m], ...)` for `m=1..3`, rounded to BF16) and immediately execute the signed-sqrt Engram gate in the same launch.
   - Wired `DeepseekV4DecoderLayer.forward` to pass `previous_output`, `post_mix`, and `res_mix` into `Engram.forward`, eliminating the standalone `mhc.post` launch on every Engram layer (**2 launches/step**) and avoiding writing/reading the intermediate `[tokens, 4, hidden]` (`40 KiB/token`) reconstructed residual tensor to/from GMEM.
3. **Native TileLang Resident Engram Lookup & Geometry (`vllm/models/deepseek_v4_1/tilelang/engram_rows.py`, `engram_hash.py`, `common/engram.py` — from PR `#802`)**:
   - Ported resident (`device` and mapped-host `ram`) Engram lookup (`engram_resident_lookup_kernel`) and host hash geometry (`Caps`, `Geometry`, `build_geometry`, `build_compressed_token_map`) to TileLang, decoupling `vllm/models/deepseek_v4_1/common/engram.py` from `b12x`.

## Verification on `dgx4` (`spark-ds41f:19f2c20ed4d6-next-21`)
- `pytest tests/kernels/test_deepseek_v41_tilelang_engram.py tests/kernels/test_deepseek_v41_engram_rows.py tests/models/test_deepseek_v4_1_engram.py tests/models/test_deepseek_v4_1_sp_prefill.py tests/kernels/test_deepseek_v41_tilelang_pdl.py`: **171 passed in 79.48s**.
- `pytest -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/test_deepseek_v41_tilelang_engram.py`: **27 passed, 11 kernels launched, 9 with PDL, 0 violations**.
