# Loop onset and sustainment: live discrimination experiment

Started 2026-11-03 from deployment checkout `17e3c29559d8c23fde81064fc16a388735eaa7ea`
(branch `feat/short-repetition-bursts`). Owner-reported trigger: "the error
happens when reasoning is set to 100". Goal: determine, by new live
measurements, (a) what sustains the degenerate reasoning loop and (b) whether
effort 100 causes its onset.

All requests went through the documented hold protocol (identity check on all
three nodes, exclusive hold, traffic check before every call, release after;
every release receipt shows `service_idle: true`; no hold remains on dgx1).
The serving cluster was never synced, stopped, started, or restarted; no
configuration, image, or baseline changed. The service ran the promoted
candidate-36 image for every round. Detector was disabled per request
(`repetition_detection` zeros) so containment could not mask behavior; this
experiment measures, it does not qualify a candidate for promotion.

## Established by these runs

### 1. The loop is a self-sustaining attractor of the model's own distribution

The 478-token screenshot fixture (`experiments/2026-11-03-short-repetition-bursts/fixture.json`,
tokenizer-pinned) was appended as the current reasoning tail after a rendered
generation prefix, and the model was asked to continue (greedy, temperature 0,
fresh `cache_salt` every request, `/v1/completions`):

| case | prefix | effort | result |
|---|---|---|---|
| score-short-100 | 101-token rendered chat | 100 | continues loop, 64/64 tokens |
| score-short-75 | same | 75 | continues loop, 64/64 |
| continue-long-100 | 83,935-token live scale-100 prompt ids | 100 | continues loop, 64/64 |
| continue-long-75 | 83,935-token live scale-75 prompt ids | 75 | continues loop, 64/64 |

Continuations reproduce the incident texture exactly: semantic fragments
(`The read artifact://552:raw:23-23; the. The. The. The.` / `The gh api pulls;
the. The. The. The.`) escalating toward pure `The. The. The.` repetition.
Generated loop tokens carry logprobs ≈ 0.0 (p ≈ 0.99) nearly everywhere.

Teacher-forced scoring (`prompt_logprobs`) over the fixture (short cases):

- P(` The`, id 455): mean **0.91** across 78 occurrences; minimum 0.14 - and
  even at that worst position ` The` is the top-ranked token. Zero occurrences
  below p = 0.1.
- P(`.`): mean 0.55.
- All other fixture tokens average p = 0.13: the model is *most* confident
  precisely on the degenerate motif tokens.
- The effort 100 and 75 profiles are indistinguishable.

Because these continuations are greedy (no sampling), use fresh cache salts
(no prefix-cache reuse), and every request reports zero sink rejections, no
serving-side mechanism (sampling, speculative acceptance, stale cache,
numerical fault) is needed or evidenced to sustain the loop. Sustainment is a
property of the served checkpoint's next-token distribution.

### 2. Effort 100 is an exposure amplifier, not a demonstrated onset cause

`reasoning_effort: 100` (or `max`/`ultra`) renders exactly one system prefix:
`Reasoning Effort: 100 (range 1-100, the higher the value, the more thorough
the reasoning)`. Measured effect on identical prompts (greedy):

| construction | effort 100 reasoning | effort 75 reasoning | ratio |
|---|---|---|---|
| exhausted re-verification audit | 4,497 | 962 | 4.7x |
| 84K-token audit transcript | 2,755 | 207 | 13x |
| retained-reasoning feedback iter | 2,312 | 90 | 26x |
| hard self-contained puzzle | 20,000 (cap, no conclusion) | 20,000 (cap) | - |

At effort 100 the model re-reads and re-states instead of concluding; at 75 it
concludes. Onset attempts at effort 100, all negative (greedy unless noted):
hard puzzle at 6K and 20K caps; exhausted audit with tools defined and
forbidden; retained real 4.5K-token thorough reasoning fed back; seeded
degenerate fixture retained as prior-session reasoning; 84K synthetic audit
transcript; unresolvable missing-fact demand (model refuses honestly at both
efforts); six-round chained real-restatement accumulation (reasoning length
collapses 1,005 -> 60 tokens: habituation, not degeneration); temperature 1.0
x three seeds on the exhausted audit (both efforts).

So within these bounds the loop did not start from a clean state at any
effort. The incidents' real sessions (212K-380K contexts, compaction, live
tool turns) exceed what bounded HTTP probes can faithfully rebuild; the onset
transition remains unreproduced, and nothing measured here establishes effort
100 as sufficient for it - only as the setting that multiplies exposure
(time spent restating in long reasoning) by roughly 5-13x.

### 3. Causal picture consistent with all measurements

1. Effort 100 keeps the model in exhaustive-restatement reasoning for
   5-13x longer than effort 75 (measured).
2. Some event tips a long restatement trajectory into the motif attractor
   (not reproduced within probe bounds; requires the real sessions' scale or
   workload to study).
3. Once inside, the distribution itself locks the loop at p ~ 1 per token
   (measured), so generation cannot self-correct and runs to the token cap.
4. With tools defined the encoder retains prior reasoning across turns, so a
   degenerate block survives into later requests (structure confirmed in the
   2026-11-03 loop-cause records; re-triggering from retained history did not
   occur in bounded probes).

Containment must be external: the model's own distribution will not exit the
loop. Effort below 100 shrinks exposure sharply and is the one configuration
lever these measurements support; no serving-side fix is indicated because no
serving-side fault was found.

## Rounds 9-11: deepening, exits, and the control-fragment clue

A 2026-11-03 external observation noted that wild "token soup" contains
model/template control fragments (`<|DSML|parameter` with ASCII pipes, `▶`,
CJK) and asked whether they are a distinct signature. Rounds 9-11 answer from
live measurement: they are a stage of the same attractor, not a separate cause.

- `runs/round9` (`deep-short-100`, `deep-long-100`, greedy, 3,000-token cap):
  thin context: fragments -> bursts -> a real `</think>` special token emitted
  mid-degeneration -> `The. The.` soup continues *after* the thinking marker
  into the answer region, cut at 123 tokens by the completions-path default
  repetition guard (`finish_reason: repetition`). 84K context: same start ->
  `</think>` -> a well-formed DSML tool-call block (`read_artifact` 552
  23-23, `gh_api` pulls) at p ~ 1 per token, then clean stop. The semantic
  fragments inside the loop are the tool calls the model is rehearsing; real
  special-token IDs (`</think>`, `<｜DSML｜ ...`) occur in the degenerate
  stream itself.
- `runs/round10`: the chat-side `repetition_detection` override does not
  apply to `/v1/completions` (guard still fired at 123 tokens).
- `runs/round11` (`active-turn-degenerate-100`, chat, detector off, greedy):
  the fixture retained as active tool-turn reasoning followed by a fresh tool
  result. The model continued the degenerate style for 172 tokens (re-stating
  the fixture's own sentences, then 7 lines of pure `The. The. The. The.`),
  then escaped via `</think>` -> a correct `gh_api` tool call. The escape
  position is distributionally marginal: the argmax token there had
  logprob -13.7 (p ~ 1e-6). Fresh external information re-anchors the model;
  without it (round 9 thin context) the soup persists past `</think>`.

Lifecycle synthesized from rounds 5/9/11: in-style continuation at p ~ 1 per
token -> pure-burst stage -> either (a) fresh tool data / rehearsed action
emitted as a real call -> recovery, or (b) no action available -> `</think>`
crossed mid-stream -> soup continues in the answer region until a cap or the
completions-path repetition guard cuts it. The ASCII-pipe/CJK/proper-noun
widening seen in wild reports is consistent with stage (b) continuing beyond
the 123-238 tokens my guarded runs reached; that final widening was not
observed within these bounds. In agent loops, exit (a) restarts a turn whose
retained reasoning still contains the degenerate block, matching the
"grows each turn" reports.

## Runs

- `runs/round1`: plain hard task, effort 100 vs 75, 6K cap (both length-capped,
  varied text; 75 reached the answer, 100 kept re-parsing).
- `runs/round2`: exhausted no-tools audit 100/75/tools-100 (all stop, varied).
- `runs/round3`: retained real reasoning iteration 100/75 and seeded fixture
  variant (all stop, varied; seeded case summarizes around the degenerate
  text without catching it).
- `runs/round4`: 84K-token synthetic audit transcript 100/75 (stop, varied).
- `runs/round5`: sustainment scoring/continuations (the four decisive cases
  above) plus unresolvable-demand 100/75 (honest refusals).
- `runs/round6`: 20K-token long runs 100/75 (length-capped, productive tables
  and simulation throughout; no bursts).
- `runs/round7`: six-round chained real-restatement accumulation at effort 100
  (habituation: 1,005 -> 60 reasoning tokens; no bursts).
- `runs/round8`: temperature 1.0, seeds 0-2, effort 100 vs 75 on the exhausted
  audit (all stop, varied).
- `runs/round9`-`round11`: deepening/exit rounds described above.

Tools: `make_round*.py` (case construction; round 3/5 reuse actual prior
outputs), `live_probe_long.py` (copy of the loop-cause probe with the
max-token bound raised to 20,000 and 900s timeouts for round 6; identical hold
protocol), `chain_probe.py` (round 7 driver), `analyze_round.py` /
`analyze_sustainment.py` (burst detection, teacher-forced statistics).

The tokenizer environment is `.work/loop-analysis/venv` with the pinned
tokenizer (`sha256 c90dfa01...`, recorded in the fixture). No GPU memory or
throughput measurements; no performance claims. `bin/spark doctor` was not
re-run in this experiment (the loop-cause record from the same day covers the
unchanged service); node identity uniformity was asserted live before every
round.
