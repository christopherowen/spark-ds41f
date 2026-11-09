# PR #1013 own tests on sp6: blocked by missing offline test models

**Sub-agent verdict:** `miss`, on the test run only. Both runs gave 101 passed, 51 failed and 44 errors, the same as the earlier sp8 check.

**Cause.** All 95 non-passing tests stop on a Hugging Face resource that is missing offline. None is a code failure.
- `gpt2` (80 tests): every test in `test_structured_output_manager.py` (44) and in `test_mtp_structured_output.py` (36).
- `Qwen/Qwen3-0.6B` (10): the CLI-contract tests, through `EngineArgs`' default model.
- `facebook/opt-125m` (5): `test_prefill_lane_livelock.py`, through `create_scheduler`'s default model.

sp6's cache holds only the DeepSeek V4.1 checkpoints; the premise that it had these models was wrong.

**Where the models are (2026-11-09):**
- `gpt2`: dgx2's cache, complete with config and tokenizer.
- `facebook/opt-125m`: sp7's cache, with config and tokenizer.
- `Qwen/Qwen3-0.6B`: on no node.

The tests can run once a single node holds all three models' configs and tokenizers.

## Coordinator's verdict: #1013's mark stands as passed; the test run is open

The declared mark is met on the candidate image (`../result.json`):
- the PR's bench shows the completions grammar fix working;
- the bug is present in 1107;
- the one-lane scheduling path is decision-for-decision identical to 1107's.

This run's miss is an environment gap, so #1013 carries forward. Its 44 structured-output tests are to be run once the models are available. Release qualification also carries the live `/v1/completions` grammar check.

The analysis is the test sub-agent's; the coordinator wrote the verdict.
