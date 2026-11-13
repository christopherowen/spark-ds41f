# Release tests against DeepSeek, not stand-in hub models

The owner (2026-11-09): "this is a deepseek repo, we should test against deepseek. find a way to do it cheaply."

## Question
Upstream vLLM tests name small hub models when they only need a tokenizer or a config:
- `gpt2`, a structured-output tokenizer;
- `facebook/opt-125m`, `create_scheduler`'s default model;
- `Qwen/Qwen3-0.6B`, `ModelConfig`'s default model.

No node holds them, so offline those tests skip (`offline_skip`). That includes most of `tests/v1/core/test_scheduler.py`, a 1107 gate file, and every test of #1013's grammar fix. Can they run against the served DeepSeek V4.1 Flash checkpoint instead, with no download?

## Variable
`scripts/release_tests/deepseek_models.py`, a pytest plugin the release-test runner loads.
- **Stand-ins:** it builds a hub cache of symlinks in which each stand-in id resolves to the served checkpoint's `config.json` and tokenizer, never its weights.
- **Bundle:** the bundle mounts the recipe's hub read-only, as serving does, and sets `DS41F_SNAPSHOT`.
- **Refused setups:** a test whose setup DeepSeek V4.1's own config validation refuses is reported as skipped, with the guard named:
  - aligned prefill chunks with no `long_prefill_token_threshold`;
  - DBO or microbatching;
  - a classic MTP or draft model;
  - no tokenizer.
- **`offline_skip.py`:** it also skips a test that asks for more GPUs than the node's one.
- **Deselected:** `test_schedule_concurrent_partial_requests[True|False]`. It expects an unaligned 224-token chunk, which DeepSeek's aligned prefill chunks never schedule.

## Workload
- **Image:** `spark-ds41f:1107` (sha256:3f32a962ce04).
- **Manifest:** 1107's gate manifest, `experiments/2026-10-26-spark-ds41f-candidate/1107/release-tests-gate-manifest.json`, 149 files.
- **Runner:** `scripts/lab.py kernel-local`, which runs the bundle as the lab's `kernel` job does, under the GPU lock.
- **Nodes:** sp5..sp7, with the cluster stopped and no hold file.
- **Base commit:** `068328a9` (main).

## Runs

**#1013's 8 test files** (candidate-48 code: the 1107 image's tree plus 0762..0771, on sp6):

| Run | Plugins | Passed | Skipped | Failed |
| --- | --- | --- | --- | --- |
| offline | tblib_stub, offline_skip | 101 | 95 | 0 |
| deepseek1 | + deepseek_models (first version) | 191 | 0 | 5 |
| deepseek2 | + deepseek_models (first version) | 191 | 0 | 5 |

- All 80 structured-output tests pass on DeepSeek's tokenizer, including the 44 that had never run anywhere.
- The 5 failures are #1013's own prefill-lane livelock tests. DeepSeek V4.1 forces `align_prefill_chunks`, and the scheduler refuses their setup (`long_prefill_token_threshold=0`).
- Receipts: `receipts/pr-1013-sp6/`.

**1107's gate, first plugin version** (`receipts/gate-v1/`):

| Node | Bundle | Passed | Skipped | Failed | Seconds |
| --- | --- | --- | --- | --- | --- |
| sp5 | main (no plugin) | 9,507 | 357 | 0 | 5,407 |
| sp6 | + deepseek_models | 9,551 | 155 | 158 | 5,397 |
| sp7 | + deepseek_models | 9,551 | 155 | 158 | 5,405 |

The 158 failures were the same on both nodes:
- 148 in `test_scheduler.py`;
- 6 in `test_gpu_ubatch_slicing.py`;
- 3 in `test_kv_cache_utils.py`;
- 1 in `test_config.py`.

All but `test_schedule_concurrent_partial_requests` are setups the DeepSeek config refuses: aligned chunks without a threshold, DBO, a classic MTP draft, `skip_tokenizer_init`, or 4-GPU pipeline parallelism. They led to the refusal skips, the GPU-count skip and the deselect.

**The four affected files, final plugin** (`receipts/four-v2/`, `receipts/four-v3/`):

| Run | Passed | Skipped | Failed |
| --- | --- | --- | --- |
| v2, sp6 and sp7 | 509 | 304 | 1 (`[False]`, still selected) |
| v3, sp6 and sp7, both parametrizations deselected | 509 | 304 | 0 |

**1107's gate, final plugin** (`receipts/gate-final/`):

| Node | Passed | Skipped | Failed | Seconds |
| --- | --- | --- | --- | --- |
| sp6 | 9,550 | 311 | 1 | 2,689 |
| sp7 | 9,550 | 311 | 1 | 2,679 |

- The one failure is `tests/v1/worker/test_display_carveout.py::test_weights_load_in_place` on both nodes. It asserts that loading weights in place lowers free GPU memory by under 32 MiB, and it read about 57 MiB.
- It is flaky without the plugin as well (`receipts/display-carveout-sp6/runs.txt`):
  - without the plugin, run alone on sp6: 1 failure in 5 runs;
  - with the plugin, run alone: 4 of 4 runs pass.
- The tests that share its chunk (chunk 11) and their outcomes are identical with and without the plugin. So the plugin does not reach it; the reading depends on the node's memory state. GB10's GPU memory is system memory.
- Each final run took about half the time of the first runs, because the TileLang caches were warm.

## Conclusion
With the plugin, the 202 gate tests that used to skip offline go three ways:
- 44 now run and pass against DeepSeek's config and tokenizer (9,551 against 9,507 in the first runs; 9,550 in the final runs, with the carve-out flake);
- 156 are reported as skipped, each naming the DeepSeek guard that refuses its setup, or the GPUs it needs;
- 2 are deselected, with their reason.

Skips fall from 357 to 311, and nothing is downloaded.

The flaky carve-out check predates the plugin and should be fixed on its own. Measuring against a page-cache-independent baseline would do it.

## Errors
- **The first plugin version** turned 158 skips into failures. They were classified above, and none was a code failure.
- **The first deselect named only `[True]`.** The `[False]` parametrization fails for the same reason.
