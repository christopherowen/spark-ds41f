# Testing

What proves a release is correct, where each proof runs, and when. Promotion requires every
layer below (see [methodology](methodology.md#promotion)); [release-benchmark.md](release-benchmark.md)
has the commands for each.

## Layers

| Layer | Proves | Runs | When |
|---|---|---|---|
| spark-ds41f unit suite (`tests/`) | recipes, config, lab, bench and tooling logic | CPU, any machine | before every push and merge, gated on its exit status |
| vLLM release tests (`config/release-tests.json`) | the serving code paths: TileLang kernels against DeepSeek's reference math, preparation, startup, scheduling, sampling, verification, collectives, parsers | the release image, one GB10, lab window | every release candidate image |
| Determinism gate (`gate.json`) | temperature-0 and greedy outputs identical to the previous release | the cluster, recipe shape | every candidate image |
| Cold boots (`coldboots.json`) | boot with one rank's, every rank's and every JIT cache empty; nothing compiles after preparation | the cluster | every candidate whose startup or kernels change |
| Release bench (`release.json`: the shape's release suite and `single-stream`) | no slower than the previous release, on distinct prompts; the single-stream speed users see | the cluster | every candidate, on the weights that will be served |

Each layer catches what the others cannot: the bench and determinism gate see only the
paths the workload takes; the release tests reach edge cases (null blocks, padding rows,
NaN, capacity boundaries) the workload rarely hits; the cold boots exercise cache states no
warm cluster has.

## Release tests

### Selection and classification

The release tests are the DeepSeek V4.1 Flash deployment's, not upstream vLLM's.
`scripts/release_tests.py check VLLM_REPO BASE HEAD` selects:

- the stack's test files, by path (DeepSeek V4.1, TileLang, B12X, Engram, DSpark and the DFlash
  base it uses, CED, kernel preparation and startup, the JIT monitor, RoCE collectives);
- the test files the patch series (BASE..HEAD) adds or changes.

`config/release-tests.json` classifies each one, with its reason:

- **gate**: exact paths. The release runs them and every test must pass or skip.
- **periodic**: worth running, not gating (none yet).
- **excluded**: paths or globs, with the reason: a model or path this deployment does not serve
  (other models' DSpark/DFlash, DeepSeek V4, GLM, Kimi, Qwen), a B12X path production does not
  take under TileLang (its TileLang equivalent is gated), the upstream `deepseek_v41` package
  (it resolves to `deepseek_v4_1` on SM12x), an environment we do not have, or dead code.
- **deselect**: single tests of gate files left out, with the reason (B12X-only tests in a file
  whose other tests guard serving).

`check` fails on an unclassified selected file, a gate file that also matches a periodic or
excluded pattern, and a gate file missing at HEAD. A new test, or a change that makes an
existing test relevant, is therefore triaged in the release that introduces it, never
silently skipped. Hand-picked per-release bundles let three DS4.1 tests break unnoticed for
weeks (the CED metadata tests, broken by the DSpark padding mask on 10-05 and the Engram
host hash on 10-16, were first run by r6u).

### Running

`scripts/release_tests.py gate VLLM_REPO HEAD manifest.json` writes the gate manifest;
`bundle` turns it into a kernel-lab bundle. In the image, `release_test_runner.py` runs
chunks of files per pytest process with timeouts, reruns a chunk that hangs or crashes file
by file, keeps every test's outcome under `/c/release-tests/<label>` (JSON lines and JUnit),
and ends with a pytest-style summary for the lab's exit verdict.

### Environment contract

- **Clean environment.** The deployment's variables change what modules do at import:
  `VLLM_DS41_KERNEL_BACKEND=tilelang` makes every default `Scheduler` raise,
  `SPARK3_DSPARK_COST_DIR` makes adaptive verification load pinned curves. Tests set what
  they need (the kernel family through a fixture or `monkeypatch`), so the bundle exports
  only `HF_HUB_OFFLINE=1`, `PYTHONPATH` and a TileLang cache directory.
- **Offline, DS4.1 weights only, one GPU.** A test whose failure is exactly a missing resource
  (an uncached Hugging Face model or config, ray) is reported as skipped, naming it
  (`scripts/release_tests/offline_skip.py`); any other failure, any other `ModelConfig`
  validation error included, stays a failure. Files that need such resources throughout are
  excluded.
- **The serving image lacks test extras.** `tests/conftest.py` imports tblib to pickle worker
  tracebacks; a plugin (`scripts/release_tests/tblib_stub.py`) stands in, so worker failures
  still raise, with their message. ray and multiprocess are absent: tests that need them are
  excluded.

### Ownership

Most of the tree's tests come from Local Inference Lab's integration branch (LIL), which the
patch series sits on. Excluding a LIL test from the gate costs nothing; deleting it makes every
rebase conflict. So:

- LIL tests for models or paths we do not serve are **excluded**, with their reason, and
  offered upstream for removal or repair.
- LIL tests that guard DS4.1 serving and are broken against current code are **fixed** in the
  series (small, upstreamable diffs).
- Tests the series added (the TileLang kernels, preparation, startup, collectives) are ours:
  merged, rewritten or deleted as the code changes.

## Audit, 2026-10-17

Two reviews read every stack test file (114) against the code it targets; a discovery run
executed every selected file in the r6u image. `config/release-tests.json` records the outcome per
file: 115 selected, 58 gate, 57 excluded, 6 single tests deselected. In summary:

### Broken against current code

Hand-built objects (`__new__`, `SimpleNamespace`) that miss attributes the code now reads, and
environment assumptions. Fixed in the series: the CED metadata tests (model state and drafter
attributes), the DFlash `propose` fixture (`_is_padding`, `_row_index`), the greedy Markov
sampler (`_vocab_parallel_greedy`), adaptive verification under a cost directory, the CED model
forward under GB10's default L2 prefetch, and DS4.1 backend selection (B12X-only, stale error
message). Broken but excluded (they test B12X paths production does not take): the DS4.1 B12X
attention preparation tests, 43 of 50 B12X workspace cases, three GLM/V3.2 sparse-MLA cases.
DS4.1's two configuration tests moved out of the B12X sparse-MLA API file (GLM, V3.2, DeepSeek V4)
into `tests/models/test_deepseek_v4_1_config.py`.

### Coverage gaps closed

- DS4.1 kernel-family routing for TileLang (attention backend and class, linear and MoE).
- The TileLang branch of kernel preparation (declare, compile, warm, step kernels, registry,
  heap release, freeze, bind, JIT monitor).
- The custom-op boundary lint now reads the TileLang modules that define custom ops.

### Gaps still open

- 39 tests of `test_kv_cache_utils.py` skip for want of offline Hugging Face configs; the
  DS4.1-relevant ones (null blocks, packing with real configs) should build their
  `ModelConfig` from a local config.
- No DS4.1 DSpark end-to-end acceptance test (acceptance is measured only by the bench).
- The JIT monitor's TileLang hook is tested only against fakes of TileLang internals; one test
  should compile a real kernel.
- The TileLang compressor test lacks the B12X compressor test's null-block, >2^31 element
  offset and rejected-suffix scenarios.
- The metadata and compressor TileLang tests use Triton kernels as oracles that production
  runs only on the B12X path: freeze them as test-local oracles before that path is deleted.
- B12X's 4 GiB-plus workspace and replay tests drive shared code (graph replay, CED
  attention, the metadata builder) through B12X kernels; they should be rewritten on TileLang.

### Consolidation

- One shared DS4.1 reference module (growing `tests/kernels/deepseek_v41_contracts.py`): MXFP8
  and block-32 dequantization, RoPE tables and FP64 rotation, E2M1 rounding and cache records,
  clamped SwiGLU, DeepSeek RMSNorm, the Engram gate, lookup dequantization and hash. Each is
  currently copied into two to four test files.
- DSpark greedy kernels: the vocabulary-parallel, greedy verification and greedy Markov
  replay tests into one file.
- The DFlash speculator base that DSpark uses: one shared hand-built-speculator fixture, so a new
  attribute is added in one place (the cause of three of the breakages above).
- Worker startup: preparation, rendezvous, RoCE health and B12X preparation coordination as
  one group.
- `test_deepseek_v41_sp_project_reduce_scatter.py` into `test_deepseek_v4_1_sp_prefill.py`;
  `utils_/test_jit_monitor_dynamo.py` into `tests/jit_monitor/`; the reasoning-effort tests into
  `test_deepseek_v41.py`.
