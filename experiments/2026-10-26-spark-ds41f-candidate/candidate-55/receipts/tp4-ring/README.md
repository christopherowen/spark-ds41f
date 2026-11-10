# candidate-55 at tp4-ring (dgx1..dgx4), 2026-11-10

These are the lab's outputs from dgx1's window (`dgx1-1794279611`, 03:00-05:23Z), copied from dgx1
`results/private/`, and the release-test output from dgx4 `~/spark-lab/results/candidate-55-image/`.

| Path | What |
| --- | --- |
| `lab/candidate-55-image/` | The image build on dgx4 (`spark-ds41f:candidate-55`, image ID `41d961dc`, the same 57 layers on every node) and the release-test kernel job: **13 failed**, 10,171 passed, 311 skipped (2,671 s). `release-tests-dgx4/output.txt` lists the failures |
| `failed-starts/20261110T04*` | The three boots (the gate's `tp4.json`, the screen's and single-stream's `tp4-dev.json`): each API server dies while constructing `AsyncLLM` |
| `lab/candidate-55-tp4/`, `lab/candidate-55-screen-tp4/`, `lab/candidate-55-single-stream-tp4/` | The serving logs the lab saved beside each failed boot |
| `lab/candidate-55-window.log` | The window's log |

**candidate-55 does not boot.** In every boot, rank 0's API server raises
`NameError: name 'Callable' is not defined` at `vllm/v1/engine/async_llm.py:890`, in `_run_output_handler`
(`failed-starts/20261110T040446Z-spark-ds41f/dgx1.log`, lines 1034-1078). #978 (0793) annotates its new
output-handler code with `Callable`, which candidate-55's `async_llm.py` does not import: the selection's context-only
rebase onto #972 did not carry that import. After each failure the lab rolled the cluster back to 1107, so
production served throughout.

**Release tests:** the 13 failures are `tests/v1/engine/test_request_submission_cleanup.py` (10:
`'AsyncLLM' object has no attribute 'renderer'`, and 3 `TimeoutError`s) and
`tests/entrypoints/serve/utils/test_api_cancellation.py::test_async_llm_generate_streaming_disconnect_aborts_engine_core`
(3, `EngineGenerateError`). #978's `_add_request` reads `self.renderer.mm_processor_cache` for every request before it
checks `request.mm_features`. candidate-54 passes all of them.

**Verdict:** #978 misses, and is in rework (round 1). #1008 and #988's dead-row part are unmeasured at the testbed:
candidate-55 boots again with #978's corrected files mounted over this image.
