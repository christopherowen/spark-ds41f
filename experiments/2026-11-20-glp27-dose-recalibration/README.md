# GLP-27 dose recalibration: serve alpha 2.0, make alpha serve-time, gate tool calling

## Summary

GLP-27 (msuiche's Spec v3 vector for DeepSeek-V4.1-Flash-1017, layers 11–37,
`hc_fold_pre_engram`) is published at `glp.alpha_default` 3.5, a dose calibrated
on refusal suites only. Served at 3.5 in this recipe, the steered name's **auto
tool calling derails**: the model enters the DSML tool channel and never exits
it — at temperature 0 it burns the whole token budget and returns no tool call
and no content. Agent clients that never set `tool_choice` (our Hermes fleet:
non-streaming, no `tool_choice`) produced raw DSML as message text and crashed
every run; we paused the fleet.

The dose is the cause, and 2.0 is the fix. On a 76-scenario tool-calling suite,
doses 1.5 and 2.0 sit on a clean plateau (72/76 strict, zero genuine failures),
2.5 is the cliff onset (6 genuine failures), 3.0 is severe (17), 3.5 is the
production failure above. At 2.0 the steering effect is retained: the recipe's
own AdvBench runner scores 61.73% no-refusal against ~63.65% at 3.5 — **the
fix costs 1.9 points of no-refusal rate** — with no measured capability,
reasoning-length, or termination delta, and a ~3.5% shorter mean
speculative-decode acceptance length as the one persistent small cost.

This change:

- `scripts/recipe.py`: an optional `alpha` in the `glp` block, rendered as
  `WEIGHTLESS_STEER_ALPHA` (validated, refused when hand-written beside the
  block, removed with the block) — the plumbing for the two items below.
- `0001-dsv41-glp-alpha-override.patch`: the reader honours
  `WEIGHTLESS_STEER_ALPHA` as the served alpha, replacing the file's
  `glp.alpha_default` when set. The GLP spec defines alpha as a serve-time
  parameter and its own readers accept this variable; this reader deliberately
  did not (the tree's `test_the_file_is_applied_as_it_is` pins that choice for
  both ALPHA and LAYERS). The patch reverses it for ALPHA only, with the dose
  evidence below as the argument; LAYERS stays unhonoured, and a value that is
  not a positive finite number stops the boot.
- `tool_gate.py`: the 76-scenario suite as a qualification gate (stdlib only,
  endpoint-configurable). It reports the strict score and fails only on
  derailment signatures, so a GLP arm — or a dose increase — can be gated on
  tool calling the way arms are already gated on refusal suites.
- `glp27-tp4-a2.0.json`: the GLP-27 TP4 arm serving the **published, unedited
  file** at alpha 2.0 through the new key.
- `docs/setup.md`: the dose guidance (serve GLP-27 at 2.0; gate before raising
  any dose).

## Base, and the one variable

Base: `spark-ds41f-1120` (candidate-62), shape tp4-switched, four GB10 nodes,
1M context, 32 sequences, KV 16.5 GiB/rank. Model
`deepseek-ai/DeepSeek-V4.1-Flash-1017` revision `1bae0e10`; steered name
`deepseek-v4.1-flash-weightless`; GLP-27 file
`glp.deepseek-v41-flash-1017-GLP-27-L11-37-a3.5.gguf`, revision `4d570926`,
sha256 `24e97d41…`, `content_sha256` `f0aaba2f…` (independently recomputed
from the direction tensors; it matches).

The one variable is the served alpha. Dose steps were served as copies of the
published file differing **only in the 4-byte `glp.alpha_default` float**
(byte offsets 1217–1220; the 2.0 variant differs in exactly 1 byte). Every
direction tensor is bit-identical across doses and the content hash stays
valid — the ladder varies alpha and nothing else. The overlay in this change
makes the same step with the published file untouched, through the override.

The reader and kernels were re-read against the spec before any dose work:
the fold applies the projection to the stream mean and broadcasts it to every
copy, the companion writer is armed with the same stack and the same alpha
tensor, the content hash is verified over the tensors in layer order. The
served intervention is the calibrated one; what was never calibrated is the
dose against tool calling. One strictness worth a line: the spec says the v2
scaling keys remain valid under v3, and this reader refuses a file carrying
them — immaterial for GLP-27 (rank 1, no scaling keys), noted for completeness.

## The failure at the published dose

Minimal repro (steered name, `tool_choice` unset, temperature 0):

```json
{"model": "deepseek-v4.1-flash-weightless",
 "messages": [{"role": "user", "content": "What is the weather in Tokyo right now?"}],
 "tools": [{"type": "function", "function": {"name": "get_weather",
   "description": "Get current weather for a city",
   "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                  "required": ["city"]}}}],
 "max_tokens": 512, "temperature": 0}
```

At 3.5 this returns `finish_reason: "length"`, `content: "\n\n"`, no
`tool_calls` key, exactly 512 completion tokens — 5/5, and identically at
budgets 128/256/1024/2048. The stock name answers the identical request with a
correct call in 46 tokens, 3/3. `tool_choice: "required"` rescues the steered
name 10/10 at both temperatures; streaming does not (10/10 no call). At
temperature 0.7 the steered name partially recovers (2/5 call). After a forced
first call and a tool result, the steered name answers in prose and never
emits the second call (3/3). Plain QA, math and code are clean at 3.5 — the
damage is specific to the structured-emission boundary.

## Dose-response (76-scenario tool suite)

Suite: 15 single, 10 parallel, 10 select-from-16, 8 schema-stress, 8
multi-turn chains, 6 `tool_choice` modes, 5 streaming, 8 temperature-0.7
repeats, 6 agent-style. Mechanical scoring (presence, name multiset, argument
values, JSON validity). Stock control on a 20-scenario subset: 20/20.
"Genuine" is the hand-audited derailment count (vanish, no-call, empty or
dropped arguments, DSML markup in argument strings, duplicates); the
remaining strict misses are scorer artifacts — a second, different, valid
call where exactly one was demanded, read-before-patch sequencing — present
at every dose including 1.5. "Gate-flagged" is `tool_gate.py`'s mechanical
derailment count over the same recorded runs.

| dose | strict | genuine | gate-flagged | dominant failure modes |
|---|---|---|---|---|
| 1.5 | 72/76 | 0–1 | 0 | one chain re-searches instead of answering |
| **2.0** | **72/76** | **0** | 1 | see the note below |
| 2.5 | 67/76 | 6 | 6 | vanish after preamble; duplicate call; DSML markup inside argument strings ×2; dropped required argument |
| 3.0 | 56/76 | 17 | 18 | empty-args calls ×3; silent no-call stops ×6; vanish; dropped arguments ×2; streamed miss |
| 3.5 | — | catastrophic | — | the production failure above (48-request matrix; not re-run on this suite) |

At 2.0 every core category is perfect: single 18/18, parallel 13/13, select
11/11, schema 9/9, stream 5/5, modes 6/6. Forced modes pass 6/6 at **every**
dose including 3.0 — the failure lives only in the model's unconstrained
decision to enter, populate and close the DSML channel.

The gate's one flag at 2.0, disclosed because the gate is deliberately
stricter than the hand audit: in one temperature-0.7 repeat, an argument
value carried a DSML fragment (`"Austin</｜DSML｜ parameter>\n<"` as a
`from_place`). The call still parsed, scored correct, and the same scenario
is clean at temperature 0 and at 1.5. It is the only blemish at 2.0; the
hand audit counts it an artifact, the mechanical gate counts it a
derailment, and a gate that errs toward flagging is the one worth having.

The cliff onset is between 2.0 and 2.5, and the failure is graded —
boundary corruption precedes the full channel trap — which is the signature
of a distribution progressively perturbed at a low-probability structured
boundary, not of a parser or reader bug (a code bug would not be
dose-graded). 2.0 is chosen over 1.5 at identical tool performance because
retention favours it on both instruments (below). **2.5 must not ship.**

## Retention, and what else was measured at 2.0

Retention, on the recipe's own instrument
(`experiments/2026-10-28-glp-dual-serving/advbench.py`, 520 prompts, thinking
off; dataset reconstructed byte-identical to the hash the runner enforces):

| metric | stock | steered @ 2.0 | steered @ 3.5 (qualification) |
|---|---|---|---|
| no-refusal phrase | **1.35%** | **61.73%** | ~63.65% |
| judge-yes | 0.96% | 36.92% | — |
| both | 0.77% | 27.69% | — |

The stock arm reproducing the recipe's published 1.35% exactly validates the
pipeline end to end. The dose fix costs 1.9 points (10 of 520 prompts) of
no-refusal rate. The 5-prompt ladder probe agrees directionally: steered
complies 5/5 at 2.0 (4/5 at 1.5) against stock 0/5. Msuiche's own ladder in
the file's `glp.validation` agrees too: keyword refusals 20 (stock) → 8 at
2.0 → 4 at 3.5; 2.0 already captures most of the effect.

Also at 2.0, against the same endpoint:

- **Agent loop**: a scripted agent loop with real tool execution (list →
  read ×3 → compute → write → verify) completes end-to-end in 6 steps, zero
  DSML leaks — the workload shape that crashed every fleet run at 3.5.
- **Concurrency**: 16/16 concurrent auto tool calls (8/16 at 3.5).
- **Needle**: retrieved at 138,071 prompt tokens, `finish=stop`.
- **Capability slice** (40 mechanical items: QA, math, code with unit tests,
  instruction-following): stock 40/40, steered 40/40.
- **Reasoning/termination** (20 prompts × 2 temperatures, paired, budget
  4096): `finish=length` 6/40 stock vs 7/40 steered at 2.0 — noise; identical
  6/40 vs 6/40 at 2.5 and at 3.0, with reasoning-token deltas centred on
  zero. The small-budget "all reasoning, no content" pattern is a base
  checkpoint trait, stock-identical at every dose; steering's collateral is
  specific to the DSML boundary, not a general EOS degradation.
- **Speculative decoding** (10 fixed prompts × 400 tokens, `/metrics`
  deltas): token acceptance overlaps stock (steered 0.516–0.548 across
  doses, stock 0.538–0.543); mean accepted length per draft runs ~3.5%
  below stock (1.35–1.39 vs 1.42–1.43) at every dose — the one persistent,
  small speed cost. The steering kernels' own cost is ~5 µs per covered
  layer by the specv3 record's latency table.

## The change

`scripts/recipe.py` — `STEER_ALPHA` joins the rendered steering variables.
`_glp_rendering` renders it from the block's optional `alpha` after
validating a positive finite number (a string, zero, a negative or a boolean
is refused with `glp alpha is …`); `_with_glp` removes it with the previous
block's rendering, and a profile that writes `WEIGHTLESS_STEER_ALPHA` itself
beside the block is refused like the other steering variables. No `alpha`
renders nothing: existing arms are byte-identical.

`0001-dsv41-glp-alpha-override.patch` — against the pinned vLLM tree
(`8fc99ca7`, spark-ds41f-1120): `vllm/envs.py` declares
`WEIGHTLESS_STEER_ALPHA` beside PATH/MODEL (so it joins the compile key like
every variable there); the reader's new `_alpha_override` parses it
fail-closed and `glp_steering()` applies it after the file's own checks,
logging the replacement at boot. The patch also updates the tree's GLP
tests: `test_the_file_is_applied_as_it_is` now pins LAYERS only, the
autouse fixture clears the new variable, the compile-key test covers it,
and two tests pin the override's apply/refuse behaviour. It is recorded
here as a patch, in the shape of the specv3 experiment's, for the series to
pick up; it is not yet in the candidate series, and a build without it
serves the file's default — the overlay above only takes effect with it.

`tool_gate.py` — run it against a booted arm before qualifying the arm, and
before any dose increase:

```sh
python3 tool_gate.py --base http://<head>:8000/v1 \
    --model deepseek-v4.1-flash-weightless --control deepseek-v4.1-flash
```

Exit 0 = no derailment signature; 1 = at least one; 2 = the endpoint could
not be exercised. `--out` writes the full JSON record.

## Runs

Dose ladder (each step: pin the dose variant, sync, start `--replace`, run
the suites from a client host; stock name served by the same engine
throughout):

- Tool suite per dose: table above. Raw records retained with the
  measurement; the gate's mechanical counts were reproduced from them with
  `tool_gate.py`'s classifier.
- AdvBench at 2.0: `advbench.py http://<head>:8000` — stock 1.35%,
  steered 61.73% no-refusal, 520/520 replies per arm, 0 errors.
- Gate against the live arm serving 2.0 (this record's own script,
  unmodified): **GATE PASS — strict 72/76, 0 derailed**, reproducing the
  ladder run exactly (the same three agent-style scorer artifacts and the
  one chain re-search; single 18/18, parallel 13/13, select 11/11, schema
  9/9, stream 5/5, modes 6/6). Stock control subset: 20/20, 0 derailed.
  Full JSON: `tool_gate-a2.0.json` in this directory.

Tests for this change:

- `python3 -m unittest discover -s tests` (after `scripts/example_site.py`,
  the CI order): **912 tests, OK**. The first run, without the example
  site, errored 5 tests on the missing site nodes file — environmental,
  unrelated to this change; the re-run is clean. `tests.test_recipe` alone:
  43 tests, OK, including six new alpha tests and the a2.0 arm resolution.
- `python3 test_alpha_override.py --glp <patched glp.py>`: 12/12 cases
  (unset → file default; numeric values served; `high`, empty, `0`, `-1`,
  `inf`, `nan` refused). The function under test is extracted verbatim from
  the patched reader. The patch was also applied to a pristine copy of the
  pinned tree's three files and the result verified byte-identical to the
  patched sources, and all three files compile.
- The vLLM-tree GLP suite (211 tests in the specv3 record) was **not**
  re-run here: it needs the prepared tree on a GPU host. The patch's test
  changes are written for that suite and are unverified until it runs.

## What this does not prove

- **Long-horizon agent traffic.** The agent evidence is scripted loops and
  scenario suites plus the 6-step live loop. Weeks of real fleet traffic at
  2.0 is the true test; our fleet is still paused while this is reviewed.
- **Subtle prose-quality drift.** The capability slice is mechanical and
  mid-difficulty; a small open-ended quality delta either way would not be
  detected by anything run here.
- **The 2.0–2.5 interval is not subdivided.** The onset could be at 2.1 or
  2.4; nothing needs to distinguish them — 2.0 is on the plateau, with the
  same strict total at 1.5 below it.
- **Alpha 3.5 was not re-measured on the expanded suites**; it is rejected
  on the diagnosis matrix and the production failure alone.
- **AdvBench judging** is the recipe runner's own design (the stock model
  judges); its bias applies at every dose equally.
