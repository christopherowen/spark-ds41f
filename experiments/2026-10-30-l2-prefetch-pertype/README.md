# Per-Type L2 Weight Prefetch Schedule on `spark-ds41f:1030`

Ports PR #615's per-type L2 weight prefetch windows (`530205c6a`,
`experiments/2026-10-26-l2-prefetch-schedule/`) cleanly onto `spark-ds41f:1030`
(vLLM `l2-prefetch-pertype-1030`, head `70d18e1a1` over `d5df88d83`), without the 50+
pre-promotion `next-candidate` overlay files.

## Bottleneck in `spark-ds41f:1030`

1. **Owner layers (`2, 8, 14, 20`)**:
   Previously, every layer issued the `WO` prefetch window (`WO-A` and `WO-B`) immediately after
   `wq_b` (`Q-B`). On an owner layer that owns a compressor (and an indexer), the compressor's
   `fused_wkv_wgate` projection runs between `Q-B` and `WO-A`, evicting `WO`'s prefetched L2 lines
   before `WO-A` ran — so `WO-A` read from DRAM despite its prefetch having completed long before.
2. **Engram layers (`1, 14`)**:
   Previously, the layer before an Engram layer issued the `NEXT` prefetch window (`mhc` attention
   mix, `fused_wqa_wkv`, `wq_b`, and indexer weights) before its MoE all-reduce. Because the Engram
   layer runs its `engram.wkv` projection before anything in `NEXT`, `engram.wkv` evicted `NEXT`'s
   prefetched L2 lines before the attention sublayer read them.

## Change (`0568`)

Changes 4 vLLM package files (`+93 / -9` lines) and moves L2 cache hints only (every output buffer
is **100% bit-for-bit identical**):
1. **`vllm/models/deepseek_v4_1/attention.py`**:
   - Adds `self._l2pf_comp = None` and `DeepseekV4Attention._issue_wo_prefetch(self, rows, *, after)`:
     - On non-owner layers (`self.compressor is None`), issues `_l2pf_wo` at `after="q-b"`.
     - On owner layers (`self.compressor is not None`), issues `_l2pf_comp` (the compressor's
       `fused_wkv_wgate` weight) at `after="q-b"` in `WO`'s place, and issues `_l2pf_wo` right after
       `self.compressor(...)` at `after="compressor"` in `_prepare_global_kv`.
2. **`vllm/models/deepseek_v4_1/common/engram.py`**:
   - Adds `l2_prefetch_after_projection: Callable[[int], None] | None = None` on `Engram` and
     `project_rows(wkv, rows, after_projection=None)` so an Engram layer issues its `NEXT` window
     between `wkv.local_product(rows)` (the local GEMM) and `tensor_model_parallel_all_gather`.
3. **`vllm/models/deepseek_v4_1/l2_prefetch.py` & `vllm/models/deepseek_v4_1/nvidia/model.py`**:
   - `DeepseekV4DecoderLayer.build_l2_prefetch` builds `attn._l2pf_comp` on owner layers and
     `plan_engram` (`l2_prefetch.linear_segments("engram.wkv", engram.wkv)` up to `BUDGET_NEXT`
     = 20 MB) on the layer preceding an Engram layer, hooking `plan_engram` before the preceding
     layer's MoE all-reduce and `plan_next` into `engram.l2_prefetch_after_projection`.

## Measured Impact (40 Decode Layers on GB10, Bit-Identical)

| Topology | `c1` (6 rows) | `c8` (48 rows) | `c16` (96 rows) |
|---|---:|---:|---:|
| **TP4** | **`-1.60%`** (`-0.37 ms`) | **`-0.58%`** (`-0.32 ms`) | **`-0.30%`** (`-0.26 ms`) |
| **TP3** | **`-1.54%`** (`-0.45 ms`) | **`-0.50%`** (`-0.35 ms`) | **`-0.28%`** (`-0.31 ms`) |

Per-step breakdown at `c1`:
- Compressor projection: `-185 µs`
- Owner `WO-A`: `-122 µs`, `WO-B`: `-33 µs`
- Engram projection: `-94 µs`
- Indexer `Q-B`: `+48 µs` (overlapped with `WO` fill)

> Note on `c1-mhcfold` (`169744c3f`, folding `finalize` into `project_streams` in `#614`):
> In `#614`, folding `finalize` into `project_streams` was measured at `+0.17% / +0.08% / -0.28%` on
> TP4 (`c1 / c8 / c16`, regressing `c16` because 40 resident CTAs loop over 4 token tiles before the
> grid barrier) and requires compiling new TileLang kernels outside `1030`'s frozen cache, so only
> the universally positive, zero-compilation per-type L2 prefetch schedule is ported here.

## Verification on `dgx4`

Reproducible from the vLLM worktree:

```bash
python3 experiments/2026-10-30-l2-prefetch-pertype/make_bundles.py VLLM_WORKTREE
```

Ran `bundles/l2-prefetch-pertype` on `dgx4` (`spark-ds41f:1030`, `sha256:7918fd6145b5`):
- **137 passed in 35.30s** ([verdict.json](verdict.json)), including:
  - `test_deepseek_v4_1_l2_schedule.py`: `test_wo_and_idx_points_by_layer_type`,
    `test_next_window_between_engram_projection_and_all_gather`,
    `test_plans_and_hooks_by_layer_and_next_layer_type`, and
    `test_full_layer_forward_issues_windows_in_fill_order`.
  - `test_deepseek_v41_tilelang_wo.py`, `test_l2_prefetch.py`,
    `test_glm5next_tilelang_l2_prefetch.py`, `test_deepseek_v4_1_attention.py`,
    `test_deepseek_v4_1_sp_prefill.py`, and `test_deepseek_v4_1_ced_model.py`.
  - `test_prepare_kernels.py`: startup `tilelang_calls` remains 247.
