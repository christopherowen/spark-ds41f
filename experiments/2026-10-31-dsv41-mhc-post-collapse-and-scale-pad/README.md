# 2026-10-31 — Fused mHC `post_collapse` / `post_collapse_dual` and padded MXFP8 scale zeroing

## Hypothesis

1. **Eliminate `[M, 4, 5120]` BF16 stream round-trips at auxiliary hidden-state layers and final layer exits (`vllm/models/deepseek_v4_1/tilelang/mhc.py`, `nvidia/model.py`, `nvidia/dspark.py`)**:
   - In `DeepseekV4Model.forward`, intermediate auxiliary hidden-state layers (`idx + 1 in (2, 30)`) execute `aux_recon = layer.mhc.post(hidden_states, residual, post_mix, res_mix)` followed by `layer.mhc.collapse(aux_recon)`, materializing a `[M, 4, 5120]` BF16 tensor (`40 KiB/row` write + `40 KiB/row` read) that is never consumed again.
   - At the final decoder layer (`idx + 1 == 60 == self.end_layer`), when `self.end_layer in self.aux_hidden_state_layers`, `DeepseekV4Model.forward` runs `layer.mhc.post` (writing `[M, 4, 5120]`), `layer.mhc.collapse(aux_recon)` for the third auxiliary hidden state (reading `[M, 4, 5120]`), and then a second `layer.mhc.collapse(final_aux_recon, pre_mix, columns=columns, column=self._sink_column)` for the final output and sink-detector column (reading `[M, 4, 5120]` a second time) — 3 separate kernel launches and `120 KiB/row` of intermediate stream HBM traffic.
   - In `DSparkDeepseekV4Model.forward`, the draft exit similarly runs `layer.mhc.post` followed by `layer.mhc.collapse` (2 launches and `80 KiB/row` of intermediate stream HBM traffic).
   - Because `post_streams` and `collapse_streams` share the exact same `(pid, tx)` to `(token, col + v)` mapping (`blocks = hidden_size // (threads * VEC)`, `threads = 128`, `VEC = 8`), `post_collapse_streams` can fold `x` into `residual` in registers (`ov`, rounded to BF16 with the exact `T.fma` order of `post_streams`) and collapse directly from `ov`:
     - `post_collapse`: 1 launch for layers 2 and 30, non-aux final layer exits, and `DSparkDeepseekV4Model.forward` (with optional `columns` extraction at `column >= 0`).
     - `post_collapse_dual`: 1 launch for `self.end_layer in self.aux_hidden_state_layers` on the last PP rank, emitting both the `pre_mix`-collapsed final output (plus `columns` at `self._sink_column`) and the uniform-mean-collapsed auxiliary hidden state from the same register streams `ov`.
   - Across a target + draft step, this removes **5 kernel launches** (2 on layers 2 & 30, 2 on layer 60, 1 on DSpark exit) and **`360 KiB/row`** of `[M, 4, 5120]` BF16 HBM traffic.

2. **Zero padded MXFP8 scale rows in `mhc_finalize` (`vllm/models/deepseek_v4_1/tilelang/mhc.py`)**:
   - `empty_mxfp8` allocates `yq` and `ysf` with `Mq = mxfp8_rows(M)` rows (padded up to a multiple of 16) via `torch.empty`, and `mhc_finalize` only writes rows `0 .. M - 1`. Downstream `_mxfp8_gemm` (`padded_rows=True`) loads the full 16-row tile of `ysf` into `SFA_shared` and feeds it to `mma.sync.m16n8k32`. Having `pid == 0` zero `ysf[M:Mq]` inside `mhc_finalize` prevents uninitialized `UE8M0` scale bytes (`0xFF` = `NaN`) on padded rows from entering tensor-core MMA tiles.

## Verification

Unit and PDL source checks verified on `dgx4` inside `spark-ds41f:1030`:

```bash
pytest -q -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/test_deepseek_v41_tilelang_mhc.py
# 102 passed, 28 kernels launched, 21 with PDL, 0 violations
```

## Patch

- [`dsv41-mhc-post-collapse-and-scale-pad.patch`](./dsv41-mhc-post-collapse-and-scale-pad.patch) — applies cleanly to `vllm` (`d5df88d83` / `spark-ds41f:1030`).
