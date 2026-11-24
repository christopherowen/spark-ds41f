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

*(Populated upon completion of `candidate-73` build, `release-tests`, cold-boot, determinism, `tool_gate`, and `vision-eval` runs.)*
