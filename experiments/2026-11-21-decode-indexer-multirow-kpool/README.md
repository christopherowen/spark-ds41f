# Decode #9: Request-Aligned Multi-Row `KPool` Reuse in Dense `indexer_score_v41`

- **Base commit**: `f655bf14b` (`spark-ds41f:candidate-63-r5`)
- **One intended variable**: Replace the combined `<2>/<4>` dense `indexer_score_v41` dispatch in `IndexerV41Plan` with dedicated decode (`row_group=3`, `rows <= 96`) and prefill (`row_group=4`, `rows > 96`) kernels, enabling odd-`R` software-pipelined `same_k` reuse, dead `w_reg` removal, contiguous `loaded_page` fallback deduplication, and SM121 wave-aligned `max_ctas` (`144` for `width <= 262144`, `288` for `width > 262144`).

## Diagnosis

In `candidate-63-r5`, `indexer_score_v41(..., candidates=False)` compiled both `__tl_score_rows_dense_fast<4>` (`rows >= 64`) and `__tl_score_rows_dense_fast<2>` (`rows < 64`) into a single CUDA kernel (`165` registers, `16 B` stack frame, `max_ctas=128`):

1. **Row-group mismatch with speculative verification depths (`V=3` and `V=6`)**:
   - At `1x3` / `8x3` (`k=2`), `<2>` straddles request boundaries on every odd pair (`same_k == false`), reading `KPool` `2×` per request.
   - At `1x6` / `4x6` / `8x6` (`k=5`), `<2>` reads `KPool` `3×` per request.
   - At `16x6` (`rows=96 >= 64`, TP4), `<4>` does not divide `6`, straddling request boundaries on 2/3 of groups.
2. **Register bleed & dead `w_reg`**: Compiling `<2>` and `<4>` together forced the decode path to pay `<4>`'s `165` registers, and `float2 w_reg[R][2]` was initialized and never read (all math uses `__nv_bfloat162 w2_reg[R][2]`).
3. **Wave-tail quantization on GB10 (`48` SMs × `3` CTAs/SM = `144` concurrent CTAs/wave)**: `max_ctas=128` left 16 SMs underutilized on single-wave launches and mismatched `48`-SM wave boundaries at `1M`.

### Rejected / Measured Alternatives (`dgx4` Cold-L2 & `ptxas -v`)

| Variant | `ptxas` regs / stack / spills | Occupancy (SM121) | Verdict |
|---|---|---|---|
| Baseline (`<2>+<4>` merged, `max_ctas=128`) | `165` regs, `16 B` stack, `0` spills | `3 CTAs/SM` | Reference |
| `SMEM-Q` (`Q_shared[R]`, `R=6/12/16`) | `128` regs, `0 B` stack, `0` spills | `4 CTAs/SM` | **Rejected**: per-row `ldmatrix` + SMEM reads + dynamic `while` loop regressed `+19..+63%` at `1M` (`8x6: 4,883 us` vs `4,045 us`; `pf256: 19.7 ms` vs `12.0 ms`). |
| `R=6` in registers (`OPT R=6`) | `208` regs, `0 B` stack, `0` spills | `2 CTAs/SM` | **Superseded by `R=3`**: dropped occupancy `-33%`, regressed `1x3` (`+50..+78%`) and `8x3`/`adapt8` (`+17..+27%`) when request depth was `< 6`. |
| `R=4` in registers (`OPT R=4` w/ `!same_k` dedup) | `174` regs, `0 B` stack, `0` spills | `2 CTAs/SM` | **Superseded**: crossed the `168`-reg cliff; kept prefill `row_group=4` on its clean dedicated kernel. |
| **`OPT R=3` (`row_group=3`, `max_ctas=144/288`)** | **`158` regs, `0 B` stack, `0` spills** | **`3 CTAs/SM`** | **Winner across all 24 decode shapes (`64K–1M`) + `4K/16K`!** Exact divisor of both `V=3` and `V=6`; side-by-side `rb` scheduling keeps both halves of a `V=6` request hot in L2 while preserving `3 CTAs/SM` occupancy. |

## Change

- [`overlay/indexer.py`](overlay/indexer.py) ([`indexer_decode_multirow.patch`](indexer_decode_multirow.patch)):
  - Removes dead `float2 w_reg[R][2]` from `__tl_score_rows_dense_fast<R>`.
  - Enables odd `R` (`R=3`) on the software-pipelined `same_k` fast path (`rp < (R & ~1)` paired in `contrib` + `if constexpr ((R & 1) != 0)` for tail row `R - 1`).
  - Adds a 1-register `loaded_page` reuse check in the `!same_k` fallback loop for contiguous same-page full tiles.
  - Adds `row_group: int = 0` specialization to `indexer_score_v41` (preserving backwards-compatible direct calls).
  - Updates `IndexerV41Plan` (`__init__`, `calls`, `run`) to declare and dispatch `score_decode` (`row_group=3`, `max_ctas=288 if width > 262144 else 144`) for `rows <= 96` and `score` (`row_group=4`, `max_ctas=128`) for `rows > 96`.

## Verification (`dgx4`, `spark-ds41f:candidate-63-r5`)

### Correctness & PDL (`107 passed, 1 skipped, 0 failed`)
```bash
docker run --rm --gpus all \
  -v /tmp/p5_decode910/patched/indexer.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/indexer.py:ro \
  -v /tmp/p5_decode910:/work \
  -e PYTHONPATH=/tmp/stub:/opt/spark-ds41f/candidate/vllm \
  -v /tmp/p5_decode910/stub:/tmp/stub:ro \
  --entrypoint pytest spark-ds41f:candidate-63-r5 \
  -v /work/test_indexer_decode_multirow.py \
  /opt/spark-ds41f/candidate/vllm/tests/kernels/attention/test_deepseek_v41_tilelang.py \
  /opt/spark-ds41f/candidate/vllm/tests/kernels/test_deepseek_v41_tilelang_pdl.py
```
- `test_indexer_decode_multirow.py`: **`16/16` passed** (`uint16` bit-identical scores, candidate blocks, and top-k indices across `1x1, 1x2, 1x3, 1x5, 1x6, 4x6, 8x3, 8x6, 16x6, adapt8, dead_rows, prefill_97, prefill_130, 1M_1x3, 1M_8x3`).
- `test_deepseek_v41_tilelang.py`: **`76/76` passed** (`1 skipped`).
- `test_deepseek_v41_tilelang_pdl.py`: **`15/15` passed** (including `layer_chain` `c1`/`c8`/`c8x6`, `0` PDL violations).

### True Disjoint Per-Request Cold-L2 Microbenchmark (`dgx4`, rotating pools `>= 256 MiB` up to `1,088 MiB`)

| Shape (`rows`) | `64K` base -> `R=3 c144` | `256K` base -> `R=3 c144` | `1M` base -> `R=3 c288` | `6L` step savings @ `256K` / `1M` |
|---|---|---|---|---|
| `1x1` (`1`) | `25.9 -> 25.9 us` (`+0.1%`) | `87.1 -> 85.1 us` (`-2.4%`) | `328.8 -> 328.7 us` (`-0.0%`) | `-0.01 / -0.00 ms` |
| `1x3` (`3`) | `28.0 -> 23.9 us` (`-14.6%`) | `95.4 -> 83.2 us` (`-12.8%`) | `441.5 -> 366.8 us` (`-16.9%`) | `-0.07 / -0.45 ms` |
| `1x6` (`6`) | `36.1 -> 27.9 us` (`-22.7%`) | `122.0 -> 97.5 us` (`-20.0%`) | `591.0 -> 478.4 us` (`-19.1%`) | `-0.15 / -0.68 ms` |
| `4x6` (`24`) | `121.9 -> 95.5 us` (`-21.6%`) | `503.9 -> 421.1 us` (`-16.4%`) | `2,158.5 -> 1,632.4 us` (`-24.4%`) | `-0.50 / -3.16 ms` |
| `8x3` (`24`) | `191.6 -> 175.5 us` (`-8.4%`) | `726.0 -> 676.9 us` (`-6.8%`) | `3,018.8 -> 2,730.9 us` (`-9.5%`) | `-0.29 / -1.73 ms` |
| `8x6` (`48`) | `257.1 -> 208.1 us` (`-19.0%`) | `951.3 -> 788.5 us` (`-17.1%`) | **`4,279.5 -> 3,106.1 us` (`-27.4%`)** | **`-0.98 / -7.04 ms`** |
| `16x6` (`96`) | `449.3 -> 406.6 us` (`-9.5%`) | `1,707.0 -> 1,541.4 us` (`-9.7%`) | **`8,342.8 -> 6,324.3 us` (`-24.2%`)** | **`-0.99 / -12.11 ms`** |
| `adapt8` (`29`) | `203.6 -> 199.7 us` (`-1.9%`) | `761.0 -> 754.6 us` (`-0.8%`) | **`3,476.4 -> 3,071.5 us` (`-11.6%`)** | **`-0.04 / -2.43 ms`** |

Short-context Cold-L2 (`4K` / `16K`) also improved: `1x6 @ 4K` `-31.6%` (`13.8 -> 9.4 us`), `8x6 @ 4K` `-19.1%` (`31.9 -> 25.8 us`), `4x6 @ 16K` `-26.5%` (`46.2 -> 34.0 us`), `8x6 @ 16K` `-26.3%` (`85.0 -> 62.7 us`). Prefill (`rows > 96`) remains on its dedicated `row_group=4` kernel (`0%` regression).

## Live Serving Runs (`tp3-triangle`, `dgx1..dgx3`)

Tested inside lab window `dgx1-1795263633` on `spark-ds41f:candidate-63-r5` (`experiments/2026-11-21-decode-indexer-multirow-kpool/tp3-dev.json`) against 2-boot `ctrl-b1/b2` (`candidate-63/tp3-dev.json`), `0` failed requests on all runs:

| Arm | `model` (`128` prompts) | Grid `mad5` vs `ctrl` | Grid `fl25` vs `ctrl` | `1x3 @4K` | `8x48 @4K` | `1x3 @256K` | `1x3 @1016K` | `c1` (`tpot_p50`) | `c8` (`tpot_p50`) | Prefill `32K` / `262K` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `ctrl-b1/b2` mean | `128/128` (`b795e6d4...`) | `0.000%` (`±0.20%`) | `0.000%` (`±0.09%`) | `16.800` | `27.556` | `16.917` | `16.961` | `69.69` (`14.44`) | `169.53` (`46.96`) | `7,259 / 7,437` |
| `idx-r3-b1` | `128/128` (`b795e6d4...`, `r1=1.0`, `ppl=1.3160824`) | `+0.151%` | `+0.067%` | `16.793` | `27.520` | `16.825` | `17.014` | `69.52` (`14.47`) | `168.93` (`47.48`) | `7,196 / 7,364` |
| `idx-r3-b2` | `128/128` (`b795e6d4...`, `r1=1.0`, `ppl=1.3160824`) | `-0.015%` | `-0.084%` | `16.818` | `27.520` | `16.936` | `16.794` | `69.59` (`14.41`) | `169.17` (`46.78`) | `7,290 / 7,429` |
| **2-boot mean** | **100% bit-exact** | **`+0.068%`** | **`-0.009%`** | **`16.805`** | **`27.520` (`-0.036 ms`)** | **`16.881` (`-0.036 ms`)** | **`16.904` (`-0.057 ms`)** | **`69.56` (`14.44`)** | **`169.05` (`47.13`)** | **`7,243 / 7,397`** |

Note: `sonar engine` (`engine grid`) replays synthetic KV tables where `1x3 @4K..1016K` (`fwd_floor = 13.03 ms`) and `8x24 @4K..256K` (`fwd_floor = 17.72 ms`) stay L2-resident across all contexts, so the `engine grid` screen verifies zero short-context/L2 regression (`fl25 = -0.009%`) while the Cold-DRAM deep-context speedup (`-17.1%` / `-1.31 ms` across 8 dense layers at `8x6 @256K`, `-27.4%` / `-9.39 ms` at `8x6 @1M`) is measured on the disjoint Cold-L2 rotating-pool harness (`dgx4`) and applies to real long-context multi-stream serving.

- **Status**: **2-boot live verified zero-regression + `128/128` bit-exact (`fl25 = -0.009%`)** + `-17.1%..-27.4%` Cold-DRAM deep-context dense-indexer speedup. Composed cleanly with BlockMax-32 (`#1265`) via `_BLOCKMAX_MIN_ROWS = _DECODE_SCORE_MAX_ROWS + 1` (`97`) for **`candidate-64`**.
- Remaining optional follow-up under `Decode #9`: multi-request TP row split (`8x6` / `16x6` @ `256K–1M`) with one-shot `sparknet` RoCE gather of `topk_indices_buffer[:rows]` (`2 KiB/row`) to divide the remaining `~24.9 ms` (`8x6 @1M`, 8 layers) across TP ranks.
