# DSML closers inside tool arguments: keep them unless structure follows

**vLLM `0818`** on the candidate-60 patch head (through `0816`, tree `afcaac35`). It covers the case that the
token-provenance rule (`0643`, #948 / #949) leaves open: the model writes a quoted closer **as the genuine DSML
token**, so provenance treats it as structure and the string argument is cut there. #876 reports quoted markup
arriving as the special token on 1107.

## What it changes

One file, `vllm/parser/deepseek_v41.py`, TOOL_ARGS state only. Reasoning and content parsing are unchanged, and
the provenance rule still runs first: markup spelled in ordinary pieces is text before this rule is consulted.

- A parameter closer `</｜DSML｜ parameter>` ends a value only when whitespace and a new parameter head follow, or
  the end of the call's arguments.
- An invoke or calls closer ends a call only once its arguments are closed (they end with a parameter closer, or
  hold none). Otherwise the closer is held. It becomes structure if a calls closer, a new invoke head or the end of
  the stream comes next (the open parameter is then closed, not dropped), and literal text otherwise.
- Every closer kept as text is logged at INFO.
- `VLLM_DSV41_STRUCTURAL_DSML=0` restores the previous parser byte for byte. The rule wraps the streaming engine's
  hooks (`_on_terminal`, `_emit_for_state`, `finish`, `reset`, …). If those change shape it logs an error and stays
  off rather than half-applying. It is written in that defensive form because it began as a mounted file; folding it
  into the engine instead is straightforward if you prefer.

The rule is after an idea by Hector Isaza (reported by ecohash-co). This is an independent implementation; no code
was copied.

## Evidence

**Offline matrix, inside the candidate-60 image** (`spark-ds41f:candidate-60`, vLLM tree `afcaac35`, CPU only,
`--network none`, the 1017 checkpoint's tokenizer). `tests/matrix.py` has 25 cases. Each is tokenized with the
real tokenizer, so `｜DSML｜` inside a value is the genuine token, and parsed non-streaming, in four chunkings and in
two reasoning modes. `tests/compare.py` gates three arms: stock, patched and kill switch.

| gate | result |
|---|---|
| G1: every streamed call is valid JSON | pass |
| G2: stream equals non-stream | pass |
| G3: B, C, D, E, E2, J exact (values holding a calls, parameter or invoke closer; two calls; all three closers; a `string="false"` JSON value holding an invoke closer) | pass |
| G4: every other ungated case equals stock | pass, except the documented changes below |
| G5: kill switch equals stock | pass |
| G6: reasoning cases equal stock | pass |
| instrument check: stock *fails* B, C, D, E, E2, J | pass (stock gets all six wrong) |

Documented behaviour changes (printed, not gated):

| case | stock | with 0818 |
|---|---|---|
| N, N2: model omits the parameter closer before `</｜DSML｜ invoke>` | `{"path": "a.txt"}` (content dropped) | `{"path": "a.txt", "content": "hello\n"}` |
| R: value ends with a literal parameter closer | `"the tag is "` | `"the tag is </｜DSML｜ parameter>"` |
| O2, Q, U2 | same output as stock in this run | same |

Receipt: `receipts/image-gate-run.log` ("ALL GATES PASS"). That run used the file as a bind mount over the image's
parser. The patch differs from that file only in the module docstring and two comments: the AST is identical once
the docstring is removed. So the preflight sha256 in the log is the mounted file's, not the patched tree's.

To rerun on a prepared tree: `git apply vllm/0818-…patch` at the vLLM root, then
`MODEL_DIR=<checkpoint> tests/run_matrix.sh <image> <prepared-vllm-root>`. The second argument is any directory
holding `vllm/parser/deepseek_v41.py`.

**Live:** served on a 4× GB10 `tp4-switched` lane with the candidate-60 image. `tests/live_probe.py` asks for a
`write_file` call whose content names all three closers, streamed and not. It passes with 0818
(`receipts/live-probe-candidate-60-with-0818.log`), **and it also passes on stock candidate-60**
(`receipts/live-probe-candidate-60-stock.log`). On this probe the model emitted the quoted closers as ordinary
pieces, which provenance already handles. So the live probe shows no regression; it does not show the bug. The
evidence for the bug is the offline matrix plus the #876 report.

## What it does not prove

- No live capture from candidate-60 where the model wrote a quoted closer as the genuine token inside an argument.
  The six target cases are tokenized, not sampled.
- A value that contains a parameter closer followed by whitespace and a parameter head is still read as two
  arguments (case Q). Nothing in the text can tell that apart from real structure.
- A call cut mid-value by `max_tokens` behaves as before.
- The release-test bundle has not run against it. `tests/tokenizers_` and the parser unit tests in the tree were not
  extended; the matrix here is standalone. We can port it to pytest if you want it in the tree.
