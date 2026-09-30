#!/usr/bin/env python3
"""Default block-FP8 linear configs per capacity (CPU; B12X_DENSE_SPLITK_TURBO as set).

usage: linear_configs.py [SM_COUNT]
Prints tile, policy (split-K slices, atomic, scheduler flags) for the DS4.1
TP3 shared-expert projections at every decode/verification capacity.
"""
import sys

from b12x.gemm.block_fp8_linear import _preparation as prep
from b12x.gemm.block_fp8_linear._tuning import BlockFp8LinearQuery, _default_config

sm = int(sys.argv[1]) if len(sys.argv) > 1 else 48


class Identity:
    sm_count = sm


class Device:
    identity = Identity()
    sm_count = sm


for name, (k, n) in {"down": (768, 5120), "gate_up": (5120, 1536)}.items():
    for cap in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 30, 32, 40, 48, 56, 64):
        q = BlockFp8LinearQuery(max_tokens=cap, in_features=k, out_features=n,
                                source_dtype="bfloat16", output_dtype="bfloat16",
                                output_mode="provided", weight_block_size=32)
        try:
            cfg = _default_config(q, Identity())
            low = prep._dense_lowering(q, cfg, Identity())
            p = low.policy
            print(f"{name:7s} cap {cap:3d}: tile {cfg.tile_m}x{cfg.tile_n} split {p.split_k_slices}"
                  f"{' atomic' if p.split_k_atomic_bf16 else ''} single_tile {p.single_work_tile_per_cta}"
                  f" direct_m1 {p.direct_one_m_tile_scheduler} m1_non_tma {p.use_m1_non_tma}"
                  f" | {cfg}", flush=True)
        except Exception as e:
            print(f"{name:7s} cap {cap:3d}: {type(e).__name__}: {e}", flush=True)
