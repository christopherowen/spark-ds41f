# DeepSeek V4.1: DSML structure only from the DSML token (2026-11-06)

A proposal from the site of [2026-11-03-site-tp4-1030-1017](../2026-11-03-site-tp4-1030-1017/README.md). It fixes #948
(tool-call string arguments that quote DSML closers lose text, silently) and #876 (reply text quoting DSML markup).
It does so with a narrow port of vllm-project/vllm#65019 onto spark-ds41f-1030's vLLM. It is not a promotion request:
nothing here has served traffic.

## The problem

The model writes DSML structure with the `｜DSML｜` token (128825), and spells a copy it quotes in ordinary pieces.
That happens in a reply's text and inside a string tool argument, such as a file about DSML or a parser's source. The
V4.1 parser matches DSML by text, so a quoted closer ends a parameter, or the whole call block.

Captured on 1030 at temperature 0, thinking off, unforced ([`fixtures/live-1030.json`](fixtures/live-1030.json), from
[`capture_fixtures.py`](capture_fixtures.py)):

| fixture | what the live server returned |
|---|---|
| `n1_file_of_closers`: write notes.md with "DSML closers:", `</｜DSML｜ parameter>`, `</｜DSML｜ calls>` | `content` cut to `"DSML closers:\n"`, whole and streamed |
| `n2_python_line`: write consts.py with `TOOL_END = "</｜DSML｜ calls>"` | `content` missing (whole) or `""` (streamed) |
| `q876_no_tools`: show the V4.1 sample in a code block, end with FINISHED, no tools | content `` ```text\n ``: the sample and FINISHED are gone |
| `ok_two_params`: write hello.md with "hello world" (control) | intact |
| `q876_tools`: the same as `q876_no_tools`, with a tool offered | intact; the model quoted the older V4 spelling `<｜DSML｜tool_calls>` this time |

In all of them, 128825 appears only at structural positions. The quoted copies are ordinary pieces.

## The change

[`vllm/0001-deepseek-v41-dsml-token-provenance.patch`](vllm/0001-deepseek-v41-dsml-token-provenance.patch), on
1030's series ([`vllm/series`](vllm/series) = rc14's 452 patches, then this one). It is 9 files, +772/-32, about 320
lines of it tests.

- **From #65019 (head `cad5796ae967`), as-is:** `token_id_scanner.py`, `incremental_lexer.py` and the scanner tests.
  A composite terminal such as `</｜DSML｜ calls>` matches only when its `｜DSML｜` is the token.
- **From #65019, adapted to this base:**
  - token spans carried through lexer tokens, events and each tool slot's arguments (`streaming_parser_engine.py`,
    `parser_engine.py`);
  - `composite_token_id_terminals` (`parser_engine_config.py`);
  - in V4.1 (`deepseek_v41.py`), the atom as a token-ID terminal, the six DSML terminals composite on it, and
    argument conversion that reads only token-backed parameter boundaries.
- **The whole-reply route, a subset of #65019's `DeepSeekV41ServingParser`:**
  - `DelegatingParser.parse` splits reasoning by text and gives the tool parser text only, so a whole reply never
    reached the guard.
  - With output token IDs, and auto or `none` tool choice, a whole reply now goes through the engine's own
    single-pass `parse`, as a stream already does. `none` still returns no calls.
  - Required and named tool choice, and streaming, keep their paths. `parser_manager` selects the class for the
    deepseek_v41 pair.
- **Not ported:**
  - the string-argument extender (vLLM main; this base has none);
  - #65019's streaming re-routing;
  - prompt turn boundaries;
  - text between tool calls returned as content;
  - metrics and docs.
- **Unchanged:** without token IDs, matching stays text-based. A genuine 128825 inside a value is still structure,
  because it is ambiguous by construction.

Candidate 31 and later carry `0643-Stream-DeepSeek-V3.2-V4-V4.1-string-tool-arguments`, which is vLLM's extender. On
that base the full #65019 fits better, and this patch would need a rebase. This is based on 1030 because that is what
was run and validated here.

## Validation

All of it ran offline, in throwaway CPU containers from the `spark-ds41f:1030` image (no network, outside the engine's
cores), with the patched `vllm/parser` mounted over the image's. Nothing was deployed.

**Replay of the live fixtures** ([`replay_fixtures.py`](replay_fixtures.py)) runs each output token stream the way
1030's chat handler does:
- the parser is built per request, with its tools and its effective chat-template kwargs;
- a whole reply is `parse(text, request, enable_auto_tools=True, model_output_token_ids=ids)`, with the IDs as
  generated (end-of-sentence included) and the text decoded with `skip_special_tokens=True`. That is the handler's
  call;
- a stream is `parse_delta` at 1, 7 and all tokens a delta, decoded the same way, with an incomplete UTF-8 tail held
  back.

Fixtures: [`fixtures/live-1030.json`](fixtures/live-1030.json) (thinking off) and
[`fixtures/live-1030-thinking.json`](fixtures/live-1030-thinking.json). The second has thinking on, with the effort
left out or max: a plain answer, two tool calls, and `th_n2_python_line`.

| | 1030 ([`runs/replay-1030.jsonl`](runs/replay-1030.jsonl)) | patched ([`runs/replay-patched.jsonl`](runs/replay-patched.jsonl)) |
|---|---|---|
| `n1_file_of_closers` | fails in 4 of 4 runs | all 4: the three lines intact (the model ends the file with a newline) |
| `n2_python_line`, `th_n2_python_line` (thinking on) | fail in 8 of 8 | all 8 intact |
| `q876_no_tools` | fails in 4 of 4 | all 4: the sample and FINISHED kept |
| `ok_two_params`, `q876_tools`, `th_plain`, `th_tool`, `th_tool_max` | reasoning, content and raw arguments equal to the live server, 20 of 20 | still equal to the live server, byte for byte, 20 of 20 |

The 1030 replay reproduces what the live server returned, thinking on and off, so the harness models the serving
path.

**vLLM tests:**
- New: `tests/parser/engine/test_deepseek_v41_dsml_provenance.py`. A mock tokenizer in which the atom is an added
  token, not a special one, as in the real tokenizer. It covers quoted DSML in string values (four spellings) and in
  reply content, with and without tools, streamed at 1/5/all and whole, plus genuine-call controls.
- DeepSeek suites (V4, V4.1, V3.2 parsers, the scanner, the V4/V3.x tool parsers) plus the new tests:
  - 1030: 245 passed, 27 failed (all of them new quoted cases; the controls pass);
  - patched: 279 passed, 0 failed.
  - Both runs have the same 19 errors: the DeepSeek V3 and V3.1 tool-parser tests need a tokenizer download, so
    they did not run.
- `tests/parser` and `tests/tool_parsers` in full:
  - 1030: 4,920 passed;
  - patched: 4,958 passed;
  - both: the same 101 other failures, test for test, and the same 781 errors, header for header. None of them is
    new, and no test that passes on 1030 fails with the patch.
  - The errors are offline downloads and modules this image does not install: Cohere's `cohere_melody` and
    Harmony's vocabulary. The root conftest's `tblib` was stubbed.

**Not done:**
- No image was built from `vllm/series`. The patch applies to rc14's head (`git apply`), but a build has not applied
  it.
- No soak, and no live traffic.

## Files

- `vllm/series`, `vllm/0001-deepseek-v41-dsml-token-provenance.patch`;
- `fixtures/live-1030.json` and `fixtures/live-1030-thinking.json`: requests, prompt and output token IDs, and live
  results, whole and streamed;
- `capture_fixtures.py`, `replay_fixtures.py`;
- `runs/replay-1030.jsonl`, `runs/replay-patched.jsonl`.
