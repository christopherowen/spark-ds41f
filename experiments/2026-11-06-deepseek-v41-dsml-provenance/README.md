# DeepSeek V4.1: DSML structure only from the DSML token (2026-11-06)

A proposal from the site of [2026-11-03-site-tp4-1030-1017](../2026-11-03-site-tp4-1030-1017/README.md), rebased onto
`spark-ds41f-1107`. It fixes #948 (tool-call string arguments that quote DSML closers lose text, silently) and #876
(reply text quoting DSML markup) by porting `vllm-project/vllm#65019` (and the `set_prompt_token_ids` hook from
`vllm-project/vllm#61857`) onto `spark-ds41f-1107`'s vLLM series. It is not a promotion request: nothing here has served
traffic.

## The problem

The model writes DSML structure with the `｜DSML｜` token (128825), and spells a copy it quotes in ordinary pieces.
That happens in a reply's text and inside a string tool argument, such as a file about DSML or a parser's source. The
V4.1 parser matches DSML by text, so a quoted closer ends a parameter, or the whole call block.

Captured on 1030 at temperature 0, thinking off, unforced ([`fixtures/live-1030.json`](fixtures/live-1030.json), from
[`capture_fixtures.py`](capture_fixtures.py)):

| fixture | what the live server returned |
|---|---|
| `n1_file_of_closers`: write notes.md with "DSML closers:", `</｜DSML｜ parameter>`, `</｜DSML｜ calls>` | `content` cut to `"DSML closers:\n"`, whole and streamed |
| `n2_python_line`: write consts.py with `TOOL_END = "</｜DSML｜ calls>"` | `content` missing (whole) or `""` on 1030 / `"TOOL_END = \""` on 1107 (streamed) |
| `q876_no_tools`: show the V4.1 sample in a code block, end with FINISHED, no tools | content `` ```text\n ``: the sample and FINISHED are gone |
| `ok_two_params`: write hello.md with "hello world" (control) | intact |
| `q876_tools`: the same as `q876_no_tools`, with a tool offered | intact; the model quoted the older V4 spelling `<｜DSML｜tool_calls>` this time |

In all of them, 128825 appears only at structural positions. The quoted copies are ordinary pieces.

## The change

[`vllm/0001-deepseek-v41-dsml-token-provenance.patch`](vllm/0001-deepseek-v41-dsml-token-provenance.patch), on
`1107`'s series ([`vllm/series`](vllm/series) = `1107`'s 641 patches at `b9276542af6f`, then this one). It touches 13
files (+1,830/-36, including 1,182 lines of tests).

- **Token-origin spans (`token_id_scanner.py`, `incremental_lexer.py`, `streaming_parser_engine.py`, `parser_engine.py`, `parser_engine_config.py`):**
  - `composite_token_id_terminals` maps each composite DSML terminal (`TOOL_START`, `TOOL_END`, `INVOKE_PREFIX`,
    `INVOKE_END`, `PARAM_START`, `PARAM_CLOSE`) to the `DSML_TOKEN` atom (`128825`).
  - `TokenIDScanner` (with `strict_token_spans=True`) and `IncrementalLexer` carry the character spans of token `128825`
    into `LexToken`, `_TokenOriginEvent`, and `ToolCallSlot.arg_token_id_spans`.
  - In `TOOL_BETWEEN`, pure formatting whitespace (such as the newline between `</｜DSML｜ invoke>` and
    `</｜DSML｜ calls>`) is held back unless non-whitespace text follows in the same `TOOL_BETWEEN` span, so formatting
    newlines do not append to pre-tool assistant content (as in `th_tool`) while genuine text or untrusted lookalikes
    between/after tool calls remain preserved as content.
- **Token-backed parameter conversion and incremental string streaming (`deepseek_v41.py`, `deepseek_v4.py`):**
  - `_dsml_arg_converter` and `_dsml_open_value_extender` filter parameter headers and `</｜DSML｜ parameter>` closers
    by `token_id_spans`, and hold back partial genuine parameter headers (`partial=True`) during incremental string
    argument streaming (`0643`).
  - `DeepSeekV41Parser._extend_open_value` prevents an untrusted `<｜DSML｜ parameter ...>` tag after a closed
    parameter from arming the text-based open-value extender.
- **Unified `DeepSeekV41ServingParser` (`deepseek_v41.py`, `parser_manager.py`, `parser_engine.py`, `adapters.py`):**
  - Routes both `parse()` and `parse_delta()` through a single `DeepSeekV41Parser` instance whenever either
    `--reasoning-parser deepseek_v41` or `--tool-call-parser deepseek_v41` is active (combined, tool-only, or
    reasoning-only) and tool choice is not JSON-constrained. This preserves token-ID provenance across implicit
    reasoning-to-tool transitions in streaming deltas and avoids `DelegatingParser.parse()`, which drops
    `model_output_token_ids` in non-streaming mode.
  - Initializes `ParserEngine._engine.state` to `ParserState.CONTENT` (with `_reasoning_ended=True`) when
    `skip_reasoning_parsing=True`, so tool-only configurations do not start in `ParserState.REASONING` while `THINK_END`
    is neutralized.
  - Preserves DSML markup as content when `request.tools` is empty or `request.tool_choice == "none"`, and records tool
    parser invocation metrics.
  - Includes `set_prompt_token_ids` (`#61857`) and `turn_boundary_tokens` so multi-turn and task prompts do not inherit
    an earlier turn's `</think>`.
- **Unchanged:** without token IDs, matching stays text-based. A genuine 128825 inside a value is still structure,
  because it is ambiguous by construction.

## Validation

All validation ran offline in CPU containers from `spark-ds41f:1107` on `sp5` and across two fresh container boots on
`sp8` with the patched `vllm/parser` mounted over the image's. Nothing was deployed to the serving cluster.

**Replay of the live fixtures** ([`replay_fixtures.py`](replay_fixtures.py)) runs each output token stream the way
the chat handler does:
- the parser is built per request, with its tools and its effective chat-template kwargs;
- a whole reply is `parse(text, request, enable_auto_tools=True, model_output_token_ids=ids)`, with the IDs as
  generated (end-of-sentence included) and the text decoded with `skip_special_tokens=True`;
- a stream is `parse_delta` at 1, 7 and all tokens a delta, decoded the same way, with an incomplete UTF-8 tail held
  back.

Fixtures: [`fixtures/live-1030.json`](fixtures/live-1030.json) (thinking off) and
[`fixtures/live-1030-thinking.json`](fixtures/live-1030-thinking.json) (thinking on, with effort unset or `max`: a plain
answer, two tool calls, and `th_n2_python_line`).

| | 1030 ([`runs/replay-1030.jsonl`](runs/replay-1030.jsonl)) | 1107 ([`runs/replay-1107.jsonl`](runs/replay-1107.jsonl)) | patched 1107 ([`runs/replay-patched.jsonl`](runs/replay-patched.jsonl)) |
|---|---|---|---|
| `n1_file_of_closers` | fails in 4 of 4 runs | fails in 4 of 4 runs | all 4: the three lines intact (the model ends the file with a newline) |
| `n2_python_line`, `th_n2_python_line` (thinking on) | fail in 8 of 8 | fail in 8 of 8 | all 8 intact |
| `q876_no_tools` | fails in 4 of 4 | fails in 4 of 4 | all 4: the sample and FINISHED kept |
| `ok_two_params`, `q876_tools`, `th_plain`, `th_tool`, `th_tool_max` | reasoning, content and raw arguments equal to the live server, 20 of 20 | equal to the live server, 20 of 20 | equal to the live server, byte for byte, 20 of 20 |

**vLLM tests (`spark-ds41f:1107`, 2 fresh container boots on `sp8`):**
- `tests/parser/engine/test_deepseek_v41_dsml_provenance.py` (86 passed): mock-tokenizer provenance suite covering
  quoted DSML in string values (four spellings) and reply content across chunk sizes `1..6`, mixed JSON and string
  parameters quoting DSML, open-value extender provenance after a closed parameter, split `<｜DSML｜ calls>` across
  implicit reasoning end, `tools=None` content preservation, pre-tool content whitespace preservation, and tool-only /
  reasoning-only `ParserManager.get_parser` configurations.
- `tests/tool_parsers/test_deepseekv41_tool_parser.py` (152 passed with cached tokenizer; cleanly skipped when
  `HF_HUB_OFFLINE=1` without a local tokenizer snapshot): real-tokenizer matrix across `nonstream`, `one_chunk`,
  `per_token`, and `text_lag3` schedules, including combined, tool-only, and reasoning-only parser modes.
- Existing parser and tool-parser suites (`test_deepseek_v41.py`, `test_deepseek_v4.py`, `test_token_id_scanner.py`,
  `test_parser_engine.py`, `test_deepseekv4_tool_parser.py`, `test_deepseekv32_tool_parser.py`): 389 passed (627 passed
  in total across all 8 files in Boot 1 in 13.36s and Boot 2 in 13.48s, 0 failures).

## Files

- `vllm/series`, `vllm/0001-deepseek-v41-dsml-token-provenance.patch`;
- `fixtures/live-1030.json` and `fixtures/live-1030-thinking.json`: requests, prompt and output token IDs, and live
  results, whole and streamed;
- `capture_fixtures.py`, `replay_fixtures.py`;
- `runs/replay-1030.jsonl`, `runs/replay-1107.jsonl`, `runs/replay-patched.jsonl`.
