# `candidate-76` — `1126` base with DSpark mask-embedding cache (`0870`), truncated-depth CUDA graph replay (`0871`), and `2 MiB -> 512 MiB` buddy defragmentation

## Base and variables

- **Base deployment commit**: `5f37f854ec` (`spark-ds41f-1126`, promoted baseline `2026-11-26-spark-ds41f-1126`).
- **Base vLLM series**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` + `candidate-74` (`0001..0869`, 742 patches, tree `f3783b0ec61851ba481f6a3a316d0a76d0e171d8`).
- **Candidate vLLM series**: `experiments/2026-10-26-spark-ds41f-candidate/candidate-76/vllm/series` (`0001..0871`, 744 patches, patch head `31e6870970eec4f16648556b4146f11c18427dd6`, tree `50dd2dac2e7c75257ed6c15136042050f5629b12`, patchset SHA-256 `ed89bf949fc0e569e16e1e86a0eeb4950a8c470dc3ee0773102cb0444c2bed4a`).
- **Built image**: `spark-ds41f:candidate-76`, OCI digest `sha256:c0ae7e09766e1d65afbdf14b23580d7631513599f999e1454932e2789472a4dd` (`dgx4`, `453 s`, `lowest MemAvailable 82.7 GiB`, identical digest verified across `dgx1..dgx4`).
- **Runtime configuration**: identical to `releases/spark-ds41f-1126/cluster.json` across all 16 shapes (`DeepSeek-V4.1-Flash-1017`, `DSpark`, `GLP-27` at `alpha=2.0`, `sparknet v0.6.0`, `tilelang`).
- **Image changes (`0870..0871`)**:
  1. `0870-dspark-cache-constant-mask-token-embedding-for-query.patch`: cache the constant `parallel_drafting_token_id` embedding once after draft weight load (`cache_mask_embed`), then embed only the `num_reqs` anchor tokens per decode step (`embed_query_block`) and broadcast the cached vector into mask offsets `1..block_size-1`. Bit-exact against full block embedding; shrinks draft embedding lookup and TP all-reduce by `5x`.
  2. `0871-dspark-retain-FULL-CUDA-graph-replay-for-truncated-d.patch`: allow greedy `DSparkSpeculator` (without a logit-reading acceptance estimator) to stay on captured `FULL` CUDA graphs when `0 < active_spec < num_speculative_steps`, masking `sample_idx_mapping[:, active_spec:]` to `-1` before replay and zeroing `draft_token_confidence_probs[:, active_spec:]` after replay. Zero behavior change at fixed `K=5`.
- **Host-side cluster defragmentation fix (`14ae2ac8f4`, `spark_ds41f/cluster.py`)**:
  - Added a `2 MiB` hugepage sweep (`hugepages-2048kB/nr_hugepages`, `(MemFree - 4 GiB) / 2 MiB`, immediately reset to `0`) **before** `compact_memory` and the `512 MiB` hugepage sweep (`hugepages-524288kB`).
  - Root cause found on aged `dgx1..dgx3` (`up 15:34`, `O13 = 0/3/0` after `cluster stop` with `124.9 GiB` `MemFree`): `cuMemCreate` left `221/252` `512 MiB` pageblocks tagged `MIGRATE_UNMOVABLE`, which `hugepages-524288kB` skips once `O13 = 0`, letting SLUB spray `SUnreclaim` (`~618 MiB`) across all `221` pageblocks and forcing `direct_loader` down to `>= 16 MiB` blocks (`+0.18..+0.48 ms` target `forward_ms` penalty).
  - Running `hugepages-2048kB` first flipped `Unmovable` pageblocks **`221 -> 9`** (`dgx1`), **`220 -> 8`** (`dgx2`), **`221 -> 9`** (`dgx3`) in **`0.55 s`**.

## Verification and gate results

- **Repo unit suite (`dgx4` / Mac)**: `955/955` passed (`40.8 s`); `tests/test_restart_guards.py` `45/45` passed (`0.024 s`).
- **Full `174`-file `release-tests` gate on `spark-ds41f:candidate-76` (`dgx4`, `5,967 s`)**: **PASS (`exit 0`): `12,082 passed, 356 skipped, 0 failed, 0 errors`** (`receipts/release-tests/release-tests.log`, `results.jsonl`).
- **Temperature-0 determinism (`--tokens 192 --long-lines 700 --wide`)**:
  - With logprobs (`receipts/tp3-triangle/determinism/wide-logprobs.txt`): **`0/0` token diffs, `0/0` logprob diffs** across all 6 scenarios (`c8`, `staggered`, `mixed`, `cached`, `again`, `mixed15`), `reference_sha256 = 5c22ce41c1b5...`, `long_sha256 = bce6092e0128...` (exact match to `1126`).
  - Without logprobs (`wide-nologprobs.txt`): **`0/0` diffs** across all 6 scenarios, `reference_sha256 = 26a739a99bca...`, `long_sha256 = 12cfee9b740e...` (exact match to `1126`).
- **Model behavior across boots (`128` prompts, `32,768` teacher-forced positions)**: **`128/128` greedy responses identical**, `100.00%` identical distributions, `KL = 0.000e+00`, `perplexity = 1.31608` (`+0.000%`).

## Measured `tp3-triangle` performance (`dgx1..dgx3` vs recorded `spark-ds41f-1126`)

### 1. Pre-reboot aged nodes (`O13 = 0`, `>= 16 MiB` weight blocks, preserved in `receipts/tp3-triangle/pre-reboot-o13-0/`)

Measured before rebooting `dgx1..dgx3` (`up 15:34`, `Unmovable = 221`, `direct_loader` soaked `5.5–7.3 GiB` and fell back to `>= 16 MiB` blocks):
- `lowest MemAvailable`: `dgx1 7.096`, `dgx2 8.768`, `dgx3 8.600 GiB` (boot `+95.5 s`, `+1` JIT file).
- `grid.json`: untouched target `forward_ms` was `+0.18..+0.48 ms` slower across all 15 shapes (`drafter_ms -0.03..+0.04 ms`).
- `c8`: `166.61 tok/s` (`-3.3%`), `single-stream c1`: `69.40 tok/s` (`-2.6%`), `prefill 32K/262K`: `7,296.8 / 7,434.0 tok/s` (`-0.4% / -1.5%`).

### 2. Post-reboot 2-boot screen with `2 MiB -> 512 MiB` defrag (`receipts/tp3-triangle/boot1/` and `boot2/`)

Nodes rebooted cleanly to `O13 = 233 / 233 / 233` (`Unmovable = 9 / 9 / 9`, `Movable = 242 / 242 / 242`), then ran **4 consecutive boots** (`1126 -> c76 Boot 1 -> c76 Boot 2 -> 1126 restored`):
- **`O13` retention across all 4 consecutive boots**: **zero** `Buddy hugepage sponge` fallback on `dgx1`, `dgx2`, or `dgx3` on any of the 4 boots (`512 MiB` superpages used directly on every restart).
- **Warm boot times & headroom**: Boot 1 `+81.9 s` (`lowest MemAvailable 7.77 / 9.63 / 9.16 GiB`), Boot 2 `+81.7 s` (`7.73 / 9.56 / 9.16 GiB`), restored `1126` `+83.1 s` (`+0/+0/+0` JIT files on all boots).

| Suite / Metric (`tp3-triangle`) | `candidate-76` Boot 1 | `candidate-76` Boot 2 | Recorded `1126` | Delta vs `1126` (B1 / B2) |
| --- | ---: | ---: | ---: | ---: |
| `grid` decode geomean (15 shapes) | `+0.60%` (`worse`) | `+0.21%` (`worse`) | ref | `+0.60%` / `+0.21%` |
| `grid 1x2 @4K` (`step_ms` / `dft`) | `16.19` / `2.40` | `16.14` / `2.38` | `16.17` / `2.39` | `+0.11%` / **`-0.21%`** |
| `grid 8x24 @4K` (`step_ms` / `dft`) | `22.53` / `3.49` | `22.47` / `3.51` | `22.53` / `3.50` | `-0.01%` / **`-0.29%`** |
| `screen-tp3 c8` (`tok/s`, `acc`) | `169.76` (`1.425`) | `169.80` (`1.426`) | `172.33` (`1.424`) | `-1.50%` / `-1.47%` (`worse`) |
| `prefill 32K` (`tok/s`) | `7,334.6` | `7,334.5` | `7,326.6` | `+0.11%` / `+0.11%` (`same`) |
| `prefill 262K` (`tok/s`) | `7,548.8` | `7,535.5` | `7,543.7` | `+0.07%` / `-0.11%` (`same`) |
| `single-stream c1` (`tok/s`, `acc`) | `70.66` (`1.474`) | `70.71` (`1.475`) | `71.28` (`1.470`) | `-0.87%` / `-0.80%` (`same`) |
| `sampled c16` (`tok/s`, `acc`) | `217.19` (`1.452`) | `215.92` (`1.440`) | `219.95` (`1.452`) | `-1.25%` / `-1.83%` (`worse`) |
| `sampled c8` (`tok/s`, `acc`) | `165.64` (`1.346`) | `166.11` (`1.360`) | `167.07` (`1.333`) | `-0.86%` / `-0.57%` (`same`) |
| `sampled c1` (`tok/s`, `acc`) | `68.53` (`1.377`) | `68.68` (`1.383`) | `69.12` (`1.374`) | `-0.86%` / `-0.65%` (`same`) |

## Conclusion

- **Host-side `defragment_buddy_memory` (`14ae2ac8f4`, already on `main`) is a verified win**: sweeping `2 MiB` hugepages before `compact_memory` and `512 MiB` hugepages prevents `MIGRATE_UNMOVABLE` pageblock ratcheting across restarts, eliminating the `O13 = 0` / `>= 16 MiB` weight-loading fallback across 4 consecutive boots, gaining `+0.67..+0.86 GiB` boot headroom, and yielding `<0.2%` boot-to-boot repeatability (`c8: -0.03%`, `prefill 32K: +0.00%`, `c1: +0.07%`).
- **`0870` (PyTorch `embed_query_block`) is neutral-to-slightly-negative on `tp3-triangle`**: shrinking the `50–400 KB` one-shot RoCE all-reduce is latency-bound on `TP3`, while the 3 extra PyTorch ops (`contiguous` + two `copy_`s + `torch.empty`) slightly offset it (`grid +0.21..+0.60%`, `c8 -1.47..-1.50%`, `sampled c16 -1.25..-1.83%`).
- **`0871` (greedy DSpark `FULL` CUDA graph replay for `active_spec < K`)** is bit-exact and passes all `12,082` release tests, ready to carry independently (without `0870`, or with a fused single-launch embedding kernel if revisited on wider TP).
- **Not promoted as-is.** `config/cluster.json` remains on `spark-ds41f:1126` (`doctor --live` OK, hold removed).
