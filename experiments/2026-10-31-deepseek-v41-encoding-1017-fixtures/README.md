# Sync DeepSeek-V4.1 Reference Encoder Fixtures with `1017` Spec (`#62199`)

## Upstream Source

- Upstream commit: `vllm-project/vllm@ddefe40e16` (`[Bugfix][Frontend] Match DeepSeek-V4.1 encoder to the 1017 reference encoding (#62199)`)
- Note: The runtime encoder/renderer changes from `#62199` and `#60792` (`DeepseekV41Renderer` registration, `order_tool_results`, `find_last_user_index` skipping tool-result-only user messages, `IMAGE_PLACEHOLDER` validation, and `DEFAULT_REASONING_EFFORT = "high"` = `75`) are already present in `rc21` (`0539`/`0540`). This patch syncs the official `DeepSeek-V4.1-Flash-1017` fixture suite (`test_input_1..5.json`, `test_output_1..5.txt`) and `test_reference_encoder_fixtures` in `tests/tokenizers_/test_deepseek_v41.py`.

## Changes

- **`tests/tokenizers_/fixtures/deepseek_v41/`**:
  - Add `test_input_1.json` through `test_input_5.json` and `test_output_3.txt` through `test_output_5.txt` from the `DeepSeek-V4.1-Flash-1017` reference encoder test suite, and update `test_output_1.txt` and `test_output_2.txt` to match the `1017` reference outputs.
- **`tests/tokenizers_/test_deepseek_v41.py`**:
  - Expand `test_reference_encoder_fixtures` across all 5 `DeepSeek-V4.1-Flash-1017` reference test cases.

## Verification

Verified in `spark-ds41f:19f2c20ed4d6-next-21` on `dgx4`:
- `pytest -q tests/tokenizers_/test_deepseek_v41.py` (`124 passed`).
