# `candidate-73` — Release Candidate (`PP=2` Support, `sparknet v0.6.0`, `GLP-27 alpha=2.0`, Reworked PR `#1323` Decode RoPE Fix)

## Summary

`candidate-73` is the clean release candidate name for the qualified `candidate-72-r5` stack (`vLLM` `0001..0868`, `741` active patches, zero local `sparknet` patches on upstream `dgx-spark-networking v0.6.0`), superseding `candidate-72`'s `r1..r5` iteration tags.

- **vLLM base revision**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354`
- **vLLM patch series**: [`vllm/series`](vllm/series) (`include ../../candidate-72/vllm/series`, `741` active patches, `0001..0868`)
- **vLLM patch head**: `32d6ab273e89923c6fea2fa8c415f8887b1da325`
- **vLLM expected tree**: `0aa091fa2b3dc5ab239816682a4b0f84e209cc58`
- **Patchset SHA-256**: `e601c2faef55777546df77f57060ae2449d8b7818f8d35a13bf84156e465c630`
- **SparkNet**: upstream `dgx-spark-networking v0.6.0` (`b01f350d061a6434c2186df73fa568ab4a9e4aa0`, tree `1739407706277142da7a12c95fca3c503c8cb11c`, `0` local patches — one-shot PP P2P + proxy ABI 10)
- **TileLang / TileKernels / NCCL / CUDA 13.5.37 / CUTLASS 4.7.1 / Model weights**: identical to `spark-ds41f-1120`.
- **Image tag**: `spark-ds41f:candidate-73` (built cleanly on `dgx4` and ring-copied across `dgx1..dgx4`).

## Included Changes Over `spark-ds41f-1120`

1. **Startup, kernel, and scheduler improvements (`0830..0851`, `0861..0865`)**:
   - Fast BF16 lookahead, bundled multidomain hotlist, TileLang `down_num_stages=3` (`0864`), engine core worker fork, step-kernel warm index, deferred MoE allocation, and ~33s faster `tp4-ring` warm startup (`+80–82s` vs `1120`'s `+113–116s`).
2. **Pipeline Parallelism `PP=2` (`0852..0860`, `pp2-tp2-ring`, `pp2-tp2-switched`, `pp2-tp3-switched`, `pp2-tp4-switched`)**:
   - Full DS4.1 `PP=2` support with upstream `sparknet v0.6.0` one-shot P2P, async zero-sync adaptive verification (`row_cost=0.25`), `VLLM_SINK_REJECTION=1`, `SPARK_DS41F_PP_USE_ENGINE_REQS=1`, intra-stage SP (`0859`), and `24.0 GiB` KV (`30.70M` tokens) on `pp2-tp2-ring`.
3. **GLP-27 dose recalibration & safe sampling defaults (`0855`, PR `#1254` by `TysAIs` + PR `#1319` by `msuiche`)**:
   - Serves `GLP-27 alpha=2.0` (`VLLM_DS41_GLP_ALPHA`), canonical boot identity, and `deepseek-v4.1-flash-weightless` per-model `0.3/0.95` defaults.
4. **DS4.1/PP env registration & chunked `y_all` scatter (`0866`)**:
   - Registers all DS4.1/PP env vars in `vllm/envs.py` and fixes `tp1`/`tp2-direct` chunked scatter.
5. **Release-test compatibility (`0867`)**:
   - Defensive `getattr(...)` guards for optional scheduler/runner stub attributes and updated unit tests; all 10 new vLLM test files classified in `config/release-tests.json` (`174` gate files).
6. **Reworked empty-index-slice decode RoPE fix (`0868`, PR `#1323` by James Meadlock)**:
   - Extracts `DeepseekV4Attention._launch_pending_decode_aux(...)` with single-shot `issue_pending_aux()` in `forward_mqa` so deferred `_rope(q)` and `_shared_decode_metadata` always run once per overlapped decode forward (even when `_decode_index_split` deals a rank `0` rows, e.g. `TP8` rank 7 at `25..28, 33..35, 41..42, 49` rows), overlap with the rank-split all-gather, never duplicate across multi-chunk or `_INDEX_SPLIT_CHECK` passes, and stay skipped on headless ranks (`n_local_heads == 0`).
   - Full diagnostic record preserved in [`experiments/2026-11-23-decode-rope-empty-index-slice/`](../../2026-11-23-decode-rope-empty-index-slice/README.md).
7. **Intentionally dropped**: PR `#1141` (opt-in buffered atomic admission).

## Qualification Scope

- **DGX (`dgx1..dgx4`)**: Full release qualification on `spark-ds41f:candidate-73` (in-image `174`-file `release-tests` bundle, `929`-test Linux repo suite, `tp4-ring` cold-boot from cleared JIT cache + warm-boot verification, strict temperature-0 determinism `--wide` + `--no-logprobs`, `80/80` `tool_gate`, and `vision-eval`), carrying over the matched 2-boot `tp4-ring` Sonar suites + same-session `1120` ABBA control and 2-boot `bin/spark bench` matrices across all four DGX shapes (`tp4-ring`, `pp2-tp2-ring`, `tp2-direct`, `tp1`) measured on the identical serving runtime in [`candidate-72`](../candidate-72/README.md) (`0868` is a strict no-op on `TP1..TP4` and `PP2-TP2`, verified bit-for-bit identical on `tp4-ring` tokens and logprobs).
- **Spaceports (`sp1..sp8`)**: Unavailable for this release (no Spaceport access).

## Receipts

- **Image**: `spark-ds41f:candidate-73` (superseded `candidate-72-r5` deleted from `dgx1..dgx4`).
  - OCI digest: `sha256:cfef8ba79ad807de00ff04a31a779759c9f94fb3f5292fee4e422aa26bb96669`
  - Labels: `local.spark-ds41f.vllm.tree=0aa091fa2b3dc5ab239816682a4b0f84e209cc58`, `local.spark-ds41f.sparknet.tree=1739407706277142da7a12c95fca3c503c8cb11c`
  - Distributed along `dgx4 -> dgx3-internal -> dgx2-internal -> dgx1-internal`; identical `56` `RootFS.Layers` (`layers_sha=9dce50cc135da4cf`) verified on all 4 nodes ([`receipts/cold-boot/image-inspect.txt`](receipts/cold-boot/image-inspect.txt)); GPU import smoke passed.
- **Full Linux repo unit suite (`dgx4`)**: **`929 passed, 0 failed`** (`36.03 s`).
- **Full `174`-file `release-tests` bundle (`dgx4`, [`receipts/release-tests/verdict.json`](receipts/release-tests/verdict.json))**: **`12,079 passed, 356 skipped, 0 failed`** (`2,976.9 s`, `passed: true`, `exit_code: 0`).
- **`tp4-ring` cold boot from cleared JIT cache (`bin/spark cache clear --apply`, [`receipts/cold-boot/`](receipts/cold-boot/))**:
  - Ready at **`+981.2 s`** (`16m 21s`), compiling `536` TileLang kernels (`528` CUDA binaries, `3,737` files) identically on all 4 nodes.
  - Minimum `MemAvailable` during cold compilation & startup: `dgx1 = 14.00 GiB`, `dgx2 = 12.64 GiB`, `dgx3 = 12.69 GiB`, `dgx4 = 15.43 GiB` (`0.00 GiB` swap growth).
- **`tp4-ring` warm boot (`spark-ds41f:candidate-73`, [`receipts/cold-boot/warm-boot.log`](receipts/cold-boot/warm-boot.log))**:
  - Ready at **`+82.2 s`** (vs `spark-ds41f-1120`'s `+113.0–116.3 s`), `0` new TileLang kernels compiled (`3,737` files unchanged across `dgx1..dgx4`), `doctor --live` OK.
  - Minimum `MemAvailable` during warm startup: `dgx1 = 13.98 GiB`, `dgx2 = 15.94 GiB`, `dgx3 = 13.32 GiB`, `dgx4 = 13.68 GiB` (`0.00 GiB` swap growth).
- **`tp4-ring` strict temperature-0 determinism ([`receipts/determinism/tp4-ring-wide{,-nologprobs}.jsonl`](receipts/determinism/tp4-ring-wide.jsonl))**:
  - **`0` token diffs, `0` logprob diffs** across all 6 scenarios (`c8`, `staggered`, `mixed`, `cached`, `again`, `mixed15`), bit-for-bit identical to `spark-ds41f-1120` golden hashes (`reference_sha256 = 18d619615e51...`, `long_sha256 = d65686498eb8...`, `reference_text_sha256 = 0a2dacdde6ff...`, `long_text_sha256 = 12fd825767d2...`).
- **Formal `tool_gate` (`T=0.0, 0.3`, stock + `deepseek-v4.1-flash-weightless` under `GLP-27 alpha=2.0`, [`receipts/tool-gate/tool-gate-tp4-ring.json`](receipts/tool-gate/tool-gate-tp4-ring.json))**:
  - **`80/80` strict PASS (`0` derailments)**.
- **Vision Eval (`2,283` items across 5 public benchmarks, `0` failed, [`receipts/vision-eval/score.txt`](receipts/vision-eval/score.txt))**:
  - `ChartQA (n=300): 86.00`, `DocVQA (n=300): 94.05`, `OCRBench (n=1,000): 81.40`, `RealWorldQA (n=383): 77.28`, `TextVQA (n=300): 86.10`, **Overall: `83.59`** (vs original TileLang vision baseline `83.36` / B12X reference `83.79`).
- **Carried-over 2-boot `tp4-ring` Sonar & 4-shape `bin/spark bench` receipts** (identical serving runtime; `0868` is a strict no-op on `TP1..TP4` and `PP2-TP2`, verified bit-identical on `tp4-ring`): see [`candidate-72/README.md`](../candidate-72/README.md#tp4-ring-two-boot-sonar-release-qualification-vs-spark-ds41f-1120).

## Conclusion

`candidate-73` (`spark-ds41f:candidate-73`, `sha256:cfef8ba79ad8...`) passed every DGX Step 9 qualification gate: `929/929` Linux repo tests, `12,079/12,079` (`174/174` files) in-image release tests, cold boot from empty JIT cache (`+981.2 s`, `>= 12.64 GiB` `MemAvailable`), warm boot (`+82.2 s`, `0` new TileLang kernels), `0/0` bit-identical `tp4-ring` determinism, `80/80` strict `tool_gate`, and `83.59` across `2,283` Vision Eval items, plus the carried-over 2-boot Sonar + 4-shape `bin/spark bench` suites from `candidate-72`. Ready for `spark-ds41f-MMDD` release promotion (with switched Spaceport shapes left to community qualification via [`docs/switched-qualification.md`](../../../docs/switched-qualification.md)).
