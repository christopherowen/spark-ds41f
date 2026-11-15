# 2026-11-15-c59-prefill (candidate-59)

- **Base commit**: `bca856d25` (`candidate-58`)
- **Branch**: `c59-prefill`
- **Cluster profiles**: `tp4-switched-c59-dev.json`, `tp3-switched-c59-dev.json`, `tp4-c59-dev.json`
- **Goal**: Stepwise prefill lift with zero prefill, decode, determinism, or quality regressions vs `candidate-58`.

## Step 1 — Targeted Host/Decompress PR Bundle + Dense Indexer Paired-Row Reduction

### Changes
1. **Bounded UTF-8 suffix detokenizer priming**:
   - `overlay/vllm/vllm/v1/engine/detokenizer.py`, `overlay/vllm/tests/detokenizer/test_check_stop_strings.py`.
2. **Zero-copy `int32` `PickleBuffer` IPC for `NewRequestData` admission**:
   - `overlay/vllm/vllm/v1/core/sched/output.py`, `overlay/vllm/vllm/v1/request.py`, `overlay/vllm/vllm/v1/worker/gpu/buffer_utils.py`, `overlay/vllm/tests/v1/core/test_new_request_data_ipc.py`.
3. **BF16x2 dequant + CUDA 13.5/PTX 9.2 `cvt.rn.bf16x2.e2m1x2` in `sparse_mla.py`**:
   - `overlay/vllm/vllm/models/deepseek_v4_1/tilelang/sparse_mla.py`, `overlay/vllm/tests/kernels/attention/test_deepseek_v41_tilelang.py`.
4. **Dense `indexer_score_v41` paired-row SMEM reduction & `2 <= rows < 64` `R=2` key-tile reuse** (without modifying `attention.py`):
   - `overlay/vllm/vllm/models/deepseek_v4_1/tilelang/indexer.py`:
     - Routed `2 <= rows < 64` to `__tl_score_rows_dense_fast<2>` (keeping `<1>` for `rows < 2` and `<4>` for `rows >= 64`).
     - Expanded `contrib` shared memory to `[2 * block_P, heads]` (`[128, 32]` float32) and paired consecutive rows inside `same_k` tiles (`rp += 2`, writing `sub=0,1` before a single `__syncthreads()`, then reducing all 128 threads concurrently in strictly increasing head order `h = 0..31` with half as many barriers and zero ULP diffs).
     - Also screened adding `__tl_score_rows_dense_fast<8>` for `rows >= 128`; discarded `<8>` because instantiating `B_reg[8]` in the single kernel increased register usage across all dispatch branches and slowed `rows=1` (`13.6 -> 19.7 us`), `rows=64` (`302.2 -> 353.3 us`), and `rows=4096` (`19987.6 -> 20145.0 us`).

### Verification & Microbenchmark (`sp8`, GB10 `sm_121a`, `spark-ds41f:candidate-55` + Step 1 overlay)

- Detokenizer + IPC unit tests (`pytest -q -p tblib_stub tests/detokenizer/test_check_stop_strings.py tests/v1/core/test_new_request_data_ipc.py`): `17 passed in 3.69s`.
- Stop string termination test (`pytest -q -p tblib_stub -p offline_skip tests/detokenizer/test_stop_string_while_stop_model_terminates.py`): `2 passed in 0.36s`.
- Sparse MLA + BF16x2 + Indexer + PDL source audit (`pytest -q -p tblib_stub -p offline_skip -p tests.kernels.deepseek_v41_pdl_sources tests/kernels/attention/test_deepseek_v41_tilelang.py -k "sparse_mla or bf16x2 or indexer or pages"`): `36 passed, 1 skipped, 34 deselected, 1 xfailed in 33.21s` (`PDL sources: 39 kernels launched, 33 with PDL, 0 violations`).
- `IndexerV41Plan` (`kv=65536`, `page_size=64`) bit-exactness check vs `candidate-58` saved reference (`idx_ref_65k.pt`): **0 diffs across all tested shapes**.

| Shape (`kv=65536`, `causal`, `span`) | `c58` score (`us`) | `c59` `<1,2,4,8>` paired (`us`, dropped) | `c59` `<1,2,4>` paired (`us`) | Speedup (`score`) |
|---|---:|---:|---:|---:|
| `rows=1, causal=0, span=1` | `13.6` | `19.7` | `13.5` | `+0.7%` |
| `rows=2, causal=0, span=2` | `21.7` | `23.8` | `15.6` | **`1.39x`** |
| `rows=6, causal=0, span=6` | `54.5` | `48.1` | `38.1` | **`1.43x`** |
| `rows=12, causal=0, span=6` | `103.2` | `91.2` | `74.3` | **`1.39x`** |
| `rows=24, causal=0, span=6` | `193.8` | `179.4` | `134.4` | **`1.44x`** |
| `rows=48, causal=0, span=6` | `380.0` | `347.3` | `261.3` | **`1.45x`** |
| `rows=64, causal=0, span=64` | `302.2` | `353.3` | `285.9` | **`1.06x`** |
| `rows=65, causal=0, span=65` | `310.4` | `363.6` | `294.1` | **`1.06x`** |
| `rows=256, causal=0, span=256` | `1221.7` | `1300.5` | `1153.3` | **`1.06x`** |
| `rows=1024, causal=0, span=1024` | `4745.3` | `4781.0` | `4517.0` | **`1.05x`** |
| `rows=1024, causal=1, span=1024` | `4711.5` | `4754.3` | `4490.5` | **`1.05x`** |
| `rows=4096, causal=1, span=4096` | `19987.6` | `20145.0` | `18815.1` | **`1.06x`** (`-1.17 ms`) |

*Pending Step 1 cluster screen & prefill bench (`tp4-switched` on `sp1`) before proceeding to Step 2.*
