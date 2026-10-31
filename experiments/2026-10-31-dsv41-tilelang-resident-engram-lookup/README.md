# Native TileLang Resident Engram Lookup & Geometry (`rc21`)

## Overview
Completes item 3 of Part 1 of the TileLang migration audit on `rc21` (`5cfcc5813fc`) by porting the resident (`device` / mapped-host `ram`) Engram FP8 E4M3 + E8M0 table lookup and host hash geometry from `b12x.sequence.engram` (`_lookup` Triton kernel) to native `@tilelang.jit` in `vllm/models/deepseek_v4_1/tilelang/`.

## Key Changes
1. **`vllm/models/deepseek_v4_1/tilelang/engram_rows.py`**:
   - Added `@tilelang.jit` `engram_resident_lookup_kernel`, `resident_lookup_calls`, and `engram_resident_lookup` matching the disk gather kernel (`engram_rows_gather_kernel`) and `b12x` `_lookup` arithmetic bit-for-bit across all 24 hash heads or column-sharded masks (`columns`), in-shard/out-of-shard row ranges (`[shard_start, shard_end)`), E8M0 scale edge cases (`0 -> 2^-127`, `255 -> NaN`), and tail zeroing (`t >= num_tokens[0]`).
2. **`vllm/models/deepseek_v4_1/tilelang/engram_hash.py`**:
   - Added native `Caps`, `Geometry`, `build_geometry` (64-bit deterministic Miller-Rabin + PCG64), and `build_compressed_token_map` so `vllm/models/deepseek_v4_1/common/engram.py` has zero runtime imports from `b12x`.
3. **`vllm/models/deepseek_v4_1/common/engram.py`**:
   - Updated `ParallelEngramEmbedding.declare_kernels` to register `resident_lookup_calls` with `tilelang.prepare` for resident (`device` and `ram`) tables.
   - Updated `ParallelEngramEmbedding.lookup_native` to launch `engram_resident_lookup` directly while retaining `self` for CUDA graph capture of mapped-host allocations.
   - Removed the `b12x.sequence.engram` lookup plan and `B12xPreparationUnit("EngramLookup")` registration from `EngramLayout` and `Engram`.

## Verification on `dgx4` (`spark-ds41f:19f2c20ed4d6-next-21`)
- `pytest tests/kernels/test_deepseek_v41_engram_rows.py`: **9 passed in 10.28s** (bit-identical output against reference and `b12x` `_lookup`).
- `pytest tests/models/test_deepseek_v4_1_engram.py`: **36 passed** (including `ram` TP4 CUDA graph replay over live mapped-host views and disk/device parity).
