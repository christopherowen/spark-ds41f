# Long-context answer coverage

Base `e9eab460`. This is CPU-only input preparation; no runtime or model setting
changes. The pinned official NVIDIA RULER generators and metrics cover all 13
tasks at generator lengths 32768, 131072, 524288 and 978944. The largest point
reserves 65536 generated tokens plus 4096 chat tokens within the 1M model limit.
Project tokenizer, temperature0, seed0 and high thinking are shared. These are
RULER tasks under the project protocol, not a published leaderboard score.

One pilot seed and three separate confirmation seeds per task/length are frozen
before any RULER model output. All task templates, answer prefixes and generator
arguments remain official. No prompt is truncated or supplied an answer key.
The held-out LongBench documents are separate and remain unseen. Synthetic cases
supply functional coverage; repeated task families are not independent books.

`requirements/ruler-generation.txt` hashes the CPU dependencies. The tokenizer
files come from the pinned public model revision and match its source identity.
The generic HF tokenizer has no chat template; an initial attempted chat render
is retained as an error. Raw tokenization and roundtrip pass. Serving uses its
own pinned deepseek_v41 encoder; actual prompt usage must confirm the input and
output allowance fits before any speed/quality acceptance.

The data lock hashes SQuAD, the official-script HotpotQA mirror, NLTK resources
and the essay corpus. The unmodified official essay parser retrieves all 218
inputs; only paulgraham HTTP transport is upgraded to HTTPS. All source URLs,
content hashes and errors are retained. The resulting corpus remains private,
outside Git, as do all generated prompts and references. Source code and third-
party notices remain in the pinned disposable checkout, not copied into serving.

The first 32K smoke input has 32617 raw tokens with its first answer around 60%
through the context; the independent metric accepts the canonical targets and
rejects a wrong sentinel. This validates preparation/oracle wiring, not model
quality. Pilot preparation records every command, output, error, prompt/artifact
hash, token length and answer depth. Final inference and grading are separate;
only naturally completed final content is eligible. Official string-match
scores measure target coverage and do not prove universal semantic correctness.

No RULER model requests or speed results are claimed. The all-task pilot and
confirmation inputs must finish and be frozen before use. Both serving arms need
fresh matched boots and a strict per-case no-loss comparison, with truncation,
empty finals and all generation time charged to the result.


All 52 pilot inputs finish successfully with no generator errors. All four
lengths and all 13 tasks are present, including both QA datasets. Canonical
references score100 and the wrong sentinel scores0 for every input. Public
receipts retain exact commands, seeds, prompt/input hashes, measured raw token
lengths and needle depths; prompts/references remain in the private cache.
This is preparation only. No RULER model output has been generated or used for
selection, and no published RULER score or performance claim is made.
