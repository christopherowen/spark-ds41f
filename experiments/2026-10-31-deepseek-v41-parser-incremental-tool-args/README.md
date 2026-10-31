# Stream DeepSeek-V3.2 / V4 / V4.1 String Tool Arguments Incrementally (`#61689` / `#64673`)

## Upstream Source

- Upstream commits:
  - `vllm-project/vllm@974927f804` (`[Bugfix][Parser] Stream DSML string tool arguments incrementally (#61689)`)
  - `vllm-project/vllm@0aa72a450c` (`[Perf][Parser] Skip tool-argument conversions after a replaced value (#64673)`)
  - `vllm-project/vllm@44f0ddfd7b` + `vllm-project/vllm@4c5ea86f53` (`ParserEngineConfig.arg_value_extender` support in `ParserEngine`)

## Problem

1. In `DeepSeekV4Parser`, `DeepSeekV41Parser`, and `DeepSeekV32Parser`, `arg_structural_chars=frozenset(">")` skipped every delta inside a `<｜DSML｜ parameter name="..." string="true">...` value that did not contain `>`, buffering the entire string argument until its closing `</｜DSML｜ parameter>` tag arrived—and when deltas did contain `>`, `_dsml_arg_converter` re-ran regex parsing and `json.dumps` over the full accumulated argument string (`O(N^2)` CPU work for long code/file tool arguments).
2. Additionally, when a repeated parameter key replaced an already-streamed value, every subsequent delta re-ran `arg_converter` even though only the new open trailing value was growing.

## Changes

- **`vllm/parser/engine/parser_engine_config.py` & `vllm/parser/engine/parser_engine.py`**:
  - Add `ArgValueExtender` (`arg_value_extender`) support to `ParserEngineConfig` and `ParserEngine._compute_arg_delta` so deltas extending an open trailing string value stream directly in `O(1)` without re-running `arg_converter` on the whole buffer.
  - Also install the extender with `open_value_streams=False` when a repeated parameter key replaces an already-streamed value (`0aa72a450c`).
- **`vllm/parser/deepseek_v4.py`, `vllm/parser/deepseek_v41.py`, `vllm/parser/deepseek_v32.py`**:
  - Implement `_dsml_arg_value_extender` and wire it into `deepseek_v4_config`, `deepseek_v41_config`, and `deepseek_v32_config`.

## Verification

Verified in `spark-ds41f:19f2c20ed4d6-next-21` on `dgx4`:
- `pytest -q tests/parser/engine/test_parser_engine.py tests/parser/engine/test_deepseek_v4.py tests/parser/engine/test_deepseek_v41.py tests/tool_parsers/test_deepseekv32_tool_parser.py tests/tool_parsers/test_deepseekv4_tool_parser.py` (`346 passed in 1.10s`).
