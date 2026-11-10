# Expunge B12X from DS4.1 runtime, source lock, and TileLang startup path

- **Base deployment commit**: `58d988c8f` (`origin/main`)
- **Base vLLM series**: `releases/spark-ds41f-1107/vllm/series` (`641` patches, head `320021e3c4db`, tree `73faeeed477c`)
- **Candidate vLLM series**: [`vllm/series`](vllm/series) (`642` patches, head `e46a179d52bc`, tree `9ee8eda8b2dc`, patch-set SHA-256 `5c02ef57804a`)
- **Patch**: [`vllm/0762-chore-expunge-b12x.patch`](vllm/0762-chore-expunge-b12x.patch)
- **Profiles**: [`tp4.json`](tp4.json), [`tp3.json`](tp3.json)
- **Source manifests**: [`upstreams.lock.json`](upstreams.lock.json), [`source.json`](source.json), [`release-tests.json`](release-tests.json)

## Scope

### Level 1 — Expunge B12X package and DS4.1 runtime dependencies

1. **Source lock & image build**:
   - Remove `sources.b12x` from [`upstreams.lock.json`](upstreams.lock.json) and `b12x` from [`source.json`](source.json).
   - Remove `container.expected_labels.local.spark-ds41f.b12x.tree` from [`tp4.json`](tp4.json) and [`tp3.json`](tp3.json), and enforce in `scripts/kernel_backend.py` (`source_problems`) that profiles whose lock omits `b12x` do not carry `local.spark-ds41f.b12x.tree` in `container.expected_labels`.
   - Support dotted key removal in `scripts/recipe.py` (`overlay` `unset`) so overlay profiles can unset `container.expected_labels.local.spark-ds41f.b12x.tree`.
2. **DS4.1 runtime & loader**:
   - Move the DS4.0 MXFP4-CSF MoE method out of `vllm/models/deepseek_v4_1/mxfp4_csf.py` to `vllm/model_executor/layers/quantization/mxfp4_csf_moe.py` and delete `vllm/models/deepseek_v4_1/mxfp4_csf.py`.
   - Remove `vllm/model_executor/model_loader/b12x_loader.py` and `tests/model_executor/model_loader/test_b12x_loader.py`, and reject `--load-format b12x` at `LoadConfig` validation.
   - Replace remaining `b12x` calls in DS4.1 attention (`vllm/models/deepseek_v4_1/attention.py`), CED (`vllm/models/deepseek_v4_1/ced.py`), compressor (`vllm/models/deepseek_v4_1/compressor.py`), sparse MLA (`vllm/models/deepseek_v4_1/sparse_mla.py`), Engram (`vllm/models/deepseek_v4_1/common/engram.py`, `vllm/models/deepseek_v4_1/tilelang/engram_rows.py`), quant config (`vllm/models/deepseek_v4_1/quant_config.py`), and NVIDIA DS4.1 model/drafter/state modules with TileLang/PyTorch equivalents.

### Level 2 — Decouple TileLang/Sparknet startup and naming from B12X plumbing

1. **Lazy `_B12X_SUBMODULES` & startup short-circuit**:
   - Make `_B12X_SUBMODULES` in `vllm/utils/b12x.py` lazy so importing `vllm.utils.b12x` never probes `importlib.util.find_spec("b12x.*")` unless `b12x` is actually requested.
   - Guard the three `Executor._run_b12x_preparation` startup phases (`weights`, `state`, `capture`) in `vllm/v1/executor/abstract.py` and `begin_b12x_preparation` / `prepare_b12x_profile` in `vllm/model_executor/warmup/b12x_prepare.py` with `has_b12x()` so TileLang boots without `b12x` skip all three no-op collective RPC rounds and `PrefixStore` exchanges.
2. **TileLang vocabulary head binding & backend-neutral names**:
   - Bind `lm_head` directly to `logits_processor` in `vllm/models/deepseek_v4_1/tilelang/prepare.py` (`declare_modules`), and add `LogitsProcessor.prepare_vocab_projection` as the primary method name (keeping `prepare_b12x_vocab_projection` as a compatibility alias).
   - Prefer `roce_ar_comm` over `b12x_ar_comm` in `vllm/models/deepseek_v4_1/sp_prefill.py`, and rename worker RoCE health helpers in `vllm/v1/worker/gpu_worker.py` to `_roce_health_check` / `_roce_guarded` (keeping `_b12x_roce_health_check` / `_b12x_roce_guarded` aliases).

## Verification

Verified on `sp8` (NVIDIA GB10) across two fresh container boots ([`receipts/boot1.log`](receipts/boot1.log) and [`receipts/boot2.log`](receipts/boot2.log)) with `b12x` uninstalled (`pip uninstall -y b12x && rm -rf /opt/spark-ds41f/candidate/b12x`):

| Check | `boot1` | `boot2` |
| --- | --- | --- |
| `importlib.util.find_spec("b12x")` / `metadata.distribution("b12x")` | Absent | Absent |
| `WITHOUT_B12X_SMOKE` (`torch`, `vllm`, `tilelang`, `flashinfer`, `sparknet` on GB10) | Pass | Pass |
| 23 `vllm.models.deepseek_v4_1.*` runtime & TileLang module imports without `b12x` | Pass (`23/23`) | Pass (`23/23`) |
| `Executor` startup RPC short-circuit & `begin_b12x_preparation` / `prepare_b12x_profile` no-op when `not has_b12x()` | Pass | Pass |
| Container pytest suite (19 test modules covering DS4.1 Engram rows, Engram model, SP prefill, TileLang metadata, TileLang vocab, compressor, prepared projections, kernel preparation, MXFP4-CSF, CED metadata, KV packing/cache utils, startup/output/warmup, RoCE health/policy, direct loader) | `973 passed, 90 skipped, 0 failed` (`800.39s`) | `973 passed, 90 skipped, 0 failed` (`341.09s`) |

Repository checks:
- `bin/spark --cluster-config experiments/2026-11-10-expunge-b12x/tp4.json doctor` -> `configuration OK`
- `bin/spark --cluster-config experiments/2026-11-10-expunge-b12x/tp3.json doctor` -> `configuration OK`
- `python3 scripts/release_tests.py check /tmp/vllm-expunge-b12x 19f2c20ed4d64ae22bd009ceacadd85e2a81e354 HEAD experiments/2026-11-10-expunge-b12x/release-tests.json` -> `212 selected test files: {'gate': 151, 'periodic': 0, 'excluded': 61}; 0 problems`
- `python3 -m unittest tests/test_source_locks.py tests/test_recipe.py tests/test_kernel_backend.py tests/test_build.py tests/test_doctor.py tests/test_release_tests.py` -> `Ran 211 tests in 5.618s — OK`
