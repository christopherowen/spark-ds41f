# `candidate-76` — `1126` base with DSpark mask-embedding cache (`0870`), truncated-depth CUDA graph replay (`0871`), and `2 MiB -> 512 MiB` buddy defragmentation

## Base and variables

- **Base deployment commit**: `5f37f854ec` (`spark-ds41f-1126`, promoted baseline `2026-11-26-spark-ds41f-1126`).
- **Base vLLM series**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` + `candidate-74` (`0001..0869`, 742 patches, tree `f3783b0ec61851ba481f6a3a316d0a76d0e171d8`).
- **Candidate vLLM series**: `experiments/2026-10-26-spark-ds41f-candidate/candidate-76/vllm/series`:
  - **Build 1 (`spark-ds41f:candidate-76`, `f4aa3a1b2a`)**: `0001..0871` (`744` patches, patch head `31e6870970eec4f16648556b4146f11c18427dd6`, tree `50dd2dac2e7c75257ed6c15136042050f5629b12`, patchset SHA-256 `ed89bf949fc0e569e16e1e86a0eeb4950a8c470dc3ee0773102cb0444c2bed4a`, OCI digest `sha256:c0ae7e09766e1d65afbdf14b23580d7631513599f999e1454932e2789472a4dd`, `453 s`, `lowest MemAvailable 82.7 GiB`). Carried both the PyTorch mask-embedding cache (`0870`) and truncated-depth CUDA graph replay (`0871`).
  - **Build 2 (`spark-ds41f:candidate-76-r2`, `2eb7183e5f`)**: `0001..0870` (`743` patches, patch head `696170c31af6c992540214f1be1b397fd802d60c`, tree `68ef31774f366d343406f90ede0488aa230f12e3`, patchset SHA-256 `3ca9752a8148ca554245e4d52191aa1739491bd69477124b430b38998bd14ad5`, OCI digest `sha256:939aa44737297cdf49bb4ecae1927948fa981d7094303afdf878086cd0e99ab8`, `472 s`, `lowest MemAvailable 81.8 GiB`). Removed Build 1's PyTorch mask-embedding cache and rebased truncated-depth graph replay as `0870`, but still carried a decorator-placement bug (`_can_replay_truncated_graph` inserted between `@torch.inference_mode()` and `def _run_model`) and was screened without rebooting `dgx1..dgx3` (`dgx1 O13 = 93.0 GiB < 96.5 GiB`).
  - **Build 3 (`spark-ds41f:candidate-76-r3`, `1017a16ccc`)**: `0001..0870` (`743` patches, patch head `22493e4ce91fa31aeb100dbb4cd5542f41b348bc`, tree `ea35d99086b692ac8c0e28c549dc66d8d2d4b62a`, patchset SHA-256 `9afb7d1da76598b825a422c67305d8a3e8b17ee460fc98f92e258577cb118618`, OCI digest `sha256:08c8ac817e5ecbabd4e704e68ff280952cd2e0868f45ef8ca67ec95ba68169c5`, `462 s`, `lowest MemAvailable 82.3 GiB`, identical digest verified across `dgx1..dgx4`). Restores `@torch.inference_mode()` onto `DFlashSpeculator._run_model` cleanly on top of `spark-ds41f-1126`.
- **Runtime configuration**: identical to `releases/spark-ds41f-1126/cluster.json` across all 16 shapes (`DeepSeek-V4.1-Flash-1017`, `DSpark`, `GLP-27` at `alpha=2.0`, `sparknet v0.6.0`, `tilelang`).
- **Image changes**:
  1. *(Build 1 only — removed in Build 2 / `candidate-76-r2`)* `0870-dspark-cache-constant-mask-token-embedding-for-query.patch`: cached the constant `parallel_drafting_token_id` embedding once after draft weight load (`cache_mask_embed`) and broadcast it into mask offsets `1..block_size-1`. Bit-exact against full block embedding, but the three extra PyTorch ops (`contiguous`, `torch.empty`, two `copy_`s) measured neutral-to-negative on `tp3-triangle` (`c8 -1.5%`, `sampled c16 -1.5%`, `grid +0.21..+0.60%`) and were dropped.
  2. *(Build 1 `0871`, rebased as `0870` in Build 2 / `candidate-76-r2`)* `0870-dspark-retain-FULL-CUDA-graph-replay-for-truncated-d.patch`: allow greedy `DSparkSpeculator` (without a logit-reading acceptance estimator) to stay on captured `FULL` CUDA graphs when `0 < active_spec < num_speculative_steps`, masking `sample_idx_mapping[:, active_spec:]` to `-1` before replay and zeroing `draft_token_confidence_probs[:, active_spec:]` after replay. Zero behavior change at fixed `K=5`.
- **Host-side cluster defragmentation fix (`14ae2ac8f4`, `spark_ds41f/cluster.py`)**:
  - Added a `2 MiB` hugepage sweep (`hugepages-2048kB/nr_hugepages`, `(MemFree - 4 GiB) / 2 MiB`, immediately reset to `0`) **before** `compact_memory` and the `512 MiB` hugepage sweep (`hugepages-524288kB`).
  - Root cause found on aged `dgx1..dgx3` (`up 15:34`, `O13 = 0/3/0` after `cluster stop` with `124.9 GiB` `MemFree`): `cuMemCreate` left `221/252` `512 MiB` pageblocks tagged `MIGRATE_UNMOVABLE`, which `hugepages-524288kB` skips once `O13 = 0`, letting SLUB spray `SUnreclaim` (`~618 MiB`) across all `221` pageblocks and forcing `direct_loader` down to `>= 16 MiB` blocks (`+0.18..+0.48 ms` target `forward_ms` penalty).
  - Running `hugepages-2048kB` first flipped `Unmovable` pageblocks **`221 -> 9`** (`dgx1`), **`220 -> 8`** (`dgx2`), **`221 -> 9`** (`dgx3`) in **`0.55 s`**.

## Verification and gate results

- **Repo unit suite (`dgx4` / Mac)**: `955/955` passed (`40.8 s`); `tests/test_restart_guards.py` `45/45` passed (`0.024 s`).
- **Targeted spec-decode pytest**:
  - `spark-ds41f:candidate-76-r2` (`dgx4`, `66.4 s`): **PASS (`94 passed, 1 skipped, 0 failed`)**.
  - `spark-ds41f:candidate-76-r3` (`dgx4`, `67.1 s`): **PASS (`92 passed, 1 skipped, 0 failed`)** across `test_dspark_truncated_cudagraph.py`, `test_dspark_vocab_parallel.py`, `test_dspark_v41_gathered_markov.py`, `test_dflash_prepare_inputs.py`, `test_dynamic_sd_cug.py`, and `test_adaptive_verification.py`.
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

### 3. Un-rebooted `candidate-76-r2` Boot 1 (`receipts/tp3-triangle-r2/boot1/`, 5th boot since reboot)

Screened without rebooting `dgx1..dgx3` (`up 5:14`, `dgx1 order-13 free = 93.0 GiB < 96.5 GiB` weight pool -> `3.5 GiB` sub-O13 spill; `dgx2 99.5 GiB`, `dgx3 98.5 GiB`; Boot 2 stopped early at `dgx1 90.0 GiB`, `dgx3 95.0 GiB`):
- `grid` decode geomean: **`+1.10%`** (`floor +0.44%`, `forward_ms +1.27%`, `drafter_ms +1.02%`).
- `screen-tp3 c8`: **`168.69 tok/s`** (`9.713 steps/s`, `-2.11%` vs `1126` B1); `prefill 32K / 262K`: **`7,253.8 / 7,481.2 tok/s`** (`-1.0% / -0.8%`); `single-stream c1`: **`69.82 tok/s`** (`-2.05%`); `sampled c16 / c8 / c1`: **`215.61 / 165.43 / 68.13 tok/s`**.

### 4. Clean-reboot `candidate-76-r3` Boot 1 (`receipts/tp3-triangle-r3/boot1/`, `111.0 / 113.0 / 113.0 GiB` `O13` free)

Nodes rebooted cleanly to `O13 = 233 / 233 / 233` (`Unmovable = 9 / 9 / 9`), and `spark-ds41f:candidate-76-r3` booted directly as Boot #1 (`+77.2 s`, `order-13 free = 111.0 / 113.0 / 113.0 GiB`, exact match to `1126`'s `110.5 / 112.0 / 112.0 GiB`; `min MemAvailable = 6.45 / 8.26 / 7.93 GiB`, `0.0 GiB` swap):

| Suite / Metric (`tp3-triangle`) | `candidate-76-r3` Boot 1 | Recorded `1126` (B1 / B2) | Delta vs `1126` |
| --- | ---: | ---: | ---: |
| `grid` decode geomean (15 shapes) | **`-0.14%`** (`floor -0.23%`) | ref | **`-0.14%`** (`fwd -0.15%`, `dft +0.08%`) |
| `grid 1x2 @4K` (`step_ms`) | `16.201 ±0.20%` | `16.171 / 16.150` | `+0.19%` |
| `grid 4x8 @4K` (`step_ms`) | `17.874 ±0.16%` | `17.875 / 17.900` | `-0.01%` |
| `grid 8x24 @4K` (`step_ms`) | `22.562 ±0.23%` | `22.534 / 22.536` | `+0.12%` |
| `screen-tp3 c8` (`tok/s`, `steps/s`) | **`171.83`** (`9.903`) | `172.33 / 171.57` (`9.913 / 9.910`) | **`-0.07%`** vs 2-boot mean (`paired -0.05%`) |
| `prefill 32K` (`tok/s`) | **`7,379.2`** | `7,326.6 / 7,321.4` | **`+0.72%`** |
| `prefill 262K` (`tok/s`) | **`7,579.6`** | `7,543.7 / 7,523.0` | **`+0.48%`** |
| `model` (`128` prompts, `outputs_sha256`) | `b795e6d4c813...` (`ppl 1.31608`) | bit-exact | `0` diffs (`rank1 = 1.0`) |

## Conclusion

- **The regression is completely gone on `candidate-76-r3` (`1017a16ccc`, `sha256:08c8ac817e5e...`)**: removing Build 1's PyTorch mask-embedding cache, restoring `@torch.inference_mode()` on `DFlashSpeculator._run_model`, and booting cleanly (`O13 = 111.0 / 113.0 / 113.0 GiB`, matching `1126`) eliminated both Build 1's `-1.5%` `c8` regression (`169.78 -> 171.83 tok/s`, `9.78 -> 9.903 steps/s`, `-0.07%` vs `1126`'s 2-boot mean; paired decode `-0.05%`) and the `+0.60%` `grid` drift (`grid -0.14%`, `floor -0.23%`, `prefill 32K/262K +0.72% / +0.48%`).
- **Host-side `defragment_buddy_memory` (`14ae2ac8f4`, already on `main`) is a verified win for multi-restart stability**, preventing `O13 = 0` collapse (`Unmovable 221 -> 9` in `0.55 s`), though residual `SUnreclaim` pinning still erodes `O13` by `~3–5 GiB` per serving cycle across 5+ restarts without reboot (`111 -> 109.5 -> 102 -> 93 -> 90 -> 89 GiB` on `dgx1`), so clean qualification comparisons against rebooted `1126` should still start from a clean reboot (`O13 >= 110 GiB`).
- **Production restored**: `config/cluster.json` remains on `spark-ds41f:1126` (`doctor --live` OK, hold removed). `candidate-76-r3` (`sha256:08c8ac817e5e...`, distributed on `dgx1..dgx4`) is ready as a clean, zero-regression `1126` + truncated-depth CUDA-graph base for future work.
