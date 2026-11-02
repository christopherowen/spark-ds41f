# Loop handling: act on a loop by where it is

Owner, 2026-11-02: work through the best possible handling of reasoning loops
and token soup in a new PR; review the operator's comments on #787 for
insights; make test cases from the known triggers.

**Status:** v2. On hardware, the first soak of v1 (sp5–sp7, 2026-11-02) cut
two legitimate answers: a 128-zero C array at its 292nd of 396 tokens, and
`hello` written 200 times, which the model ran past to 281. v2 replaces v1's
single 256-token horizon for a healthy answer with two (see The change). It
passes the unit tests and the offline replays on candidate-37's image on sp8.
Its probe soak is queued in the same window. Not promoted; no recipe change.

- Base: candidate-37, vLLM patch head `5b7cfa37f` (tree `af0fb152`), image
  `spark-ds41f:candidate-37` (`sha256:e1ee7b19…` on sp8).
- One variable: what `check_stop` does when its loop detector fires
  (`vllm/v1/core/sched/utils.py`, `vllm/v1/request.py`), and one log line in
  `vllm/v1/core/sched/scheduler.py`.
- Patch: [0001-feat-v1-act-on-a-loop-by-where-it-is-and-follow-the-reasoning-parser.patch](0001-feat-v1-act-on-a-loop-by-where-it-is-and-follow-the-reasoning-parser.patch),
  one commit on the base (vLLM `050e8b1cc`, branch `loop-handling` in
  `/tmp/vllm-loop-v3` on sp8), next in the series after 0676. It applies
  cleanly to a fresh `5b7cfa37f` tree. v1 was `6698d7467`; v2 changes only
  `utils.py` and the tests.

## What happened

An operator runs candidate-36 on two triangles behind an autonomous agent
(#787, comments 6559214555 to 6565237128):

1. **A 736-token run of ` Vila` in the visible answer** at thinking level max:
   2,330 tokens, 252 distinct, one token 1,171 times, interleaved with
   ` Evelyn`, ` Tripadvisor`, ` Escola`, ` Fest` and Chinese characters. No
   detector line; the request ran 154 s.
2. **A reasoning stall** (`Let me do it.` repeated) at 163 output tokens:
   #787 forced the end of reasoning, and the answer collapsed into token soup
   at once.
3. **An agent quoting a ` Vila` collapse fell into it** inside its own
   reasoning. Replayed on candidate-36 (sp5, 2026-11-02), the model continued
   ` Vila` at p 0.83 to 0.93 with zero sink flags.

Their scan of 2,041 sessions: severe runs (128 or more) are max-only, 3 against
0 at other levels, but the two with recorded context were at 212K and 380K
tokens, so effort and context length are confounded. Legitimate messages in
the same sessions carried 199-token runs of one token (separator rules) at a
distinct-token ratio of 0.82 to 0.84: run length alone cannot tell degeneration
from output. They see about three detector fires an hour against seven
messages in 124K with a run of eight or more, so the false-positive surface is
the open question.

### What candidate-37 already does, and what it gets wrong

0674 (#845) runs the detector on the answer by default (thinking on or off)
and fixes c36's premature `thinking_disabled` latch. That latch is the likely
reason incident 1 went undetected: on c36 a request whose first output tokens
did not open reasoning latched `thinking_disabled`, and `check_stop` returned
before any detection. The container log that would prove it was overwritten.

But 0674 applies the reasoning thresholds (a 24-token span) to the visible
answer, where the action is to stop the request. Replayed over legitimate code
and documentation (below), it stops 31 of 3,618 answers: a list of kernel
names, a run of `#include` lines, a test tensor literal
`[0.5, 0.5, 0.5, 0.9, …]`, `Int64(1), Int64(1), …`. That is the code an agent
writes through tool calls.

Three more defects:
- `update_reasoning_markers` reset the loop window on any start marker, even
  inside reasoning, which also cancelled a pending forced end; and a stray end
  marker in the answer moved the answer's start, so a loop containing end
  markers evaded detection.
- `if request.force_thinking_end: return False` left the loop unchecked until
  max_tokens if the forced end never landed.
- A client's `repetition_detection` replaced the built-in answer detection
  instead of adding to it.

None of the three rules sees incident 1's shape: every run of ` Vila` under 24
tokens, an irregular period, interjections of one to three tokens.

## The change

| Where the loop is | Action |
|---|---|
| inside reasoning | end reasoning at the first loop (unchanged); a forced end that has not landed within 64 output tokens stops the request |
| in the answer after a reasoning loop | stop at the first loop: the context is already degenerate (incident 2) |
| in the answer of a request whose reasoning did not loop | **token soup** (one token over 60% of the answer's last 512 tokens, in a run or more every 32 tokens): stop once it persists 256 tokens past its first detection. **Any other loop**: stop only after 2,048 tokens, and only if it leaves fewer than one distinct token in ten among the answer's last 512 tokens (the operator's condition). Detections up to 64 tokens apart count as one loop. |

Why two horizons (v2). At real token counts legitimate periodic output is
long: three tokens a zero make a 128-zero C array 396 tokens, a 256-zero
array 780 and a 16x16 matrix 792. v1's single horizon (256 tokens, gaps up to
16) cut them on hardware. Exactly periodic output can't be told from a
periodic loop by its tokens, so v2 lets it run 2,048 tokens. That bounds a
runaway at about 2,070 tokens, against running on to `max_tokens`. The
collapse seen in production is not periodic: one token in short runs broken
by irregular interjections. That shape keeps the short horizon. A long run of
one token (a separator) doesn't have it, and neither does an evenly periodic
array, where no token reaches half.

- **Token soup**, a fourth rule: one token at 60% or more of the last 32 to 64
  tokens with at most one distinct token per eight. It catches incident 1's
  shape. A pre-check (at most 5 distinct among the last 8 tokens) keeps its
  cost off prose and code.
- **Markers follow DeepSeekV4Parser** (`vllm/parser/deepseek_v4.py`): inside
  reasoning only `</think>` leaves; in the answer `<think>` opens reasoning
  again and a stray `</think>` is answer text.
- **Instrumentation:** one log line when an answer loop is first suspected,
  with the request's answer start, `thinking_disabled` and loop breaks, the
  state the operator asked for to explain a miss.
- Sampling is unchanged: `check_stop` only reads committed tokens, so a
  request that never loops gets the same tokens and finish.

## Evidence

**Unit tests** (`tests/v1/core/test_repetition_detection.py`, 40 tests).
They cover:
- token soup: incident 1's shape, interjections of new cone tokens, and bit
  strings, zero matrices and prose that must not match;
- the parser-following markers and the forced-end grace;
- the answer's two horizons and its diversity condition: an exactly periodic
  run or phrase stops at 2,048 tokens;
- the known trigger shapes at real token counts:
  - the production soup after 1,500 varied tokens stops within 512 tokens of
    its start;
  - kept: separator runs of 199 and 300 tokens in varied output, `hello` 200
    times, a 512-zero array (1,536 tokens), a 16x16 matrix, a 200-row table
    and a 100-token one-token answer;
  - the `Let me do it.` stall breaks reasoning at its 24th token.

```text
image spark-ds41f:candidate-37 on sp8, changed files mounted:
tests/v1/core/test_repetition_detection.py + test_copy_drafts_scheduler.py
  + test_sink_draft_zero.py + test_output.py        v1 121 passed, v2 122 passed
the same plus test_scheduler.py, test_async_scheduler.py, test_scheduler_e2e.py:
  candidate-37  217 failed, 116 passed, 2 errors
  v1            217 failed, 126 passed, 2 errors   (identical failing set:
                the offline hub lacks facebook/opt-125m)
  v2            217 failed, 127 passed, 2 errors   (the same 217 test ids as candidate-37)
```

**Offline replay** (`replay.py`, receipts in `runs/`): 1,500 files of this
repository, the vLLM source and docs, cut into 2,048-token chunks, each fed
token by token through the real `check_stop` once as reasoning and once as a
visible answer.

| Arm | Chunks | Tokens | Reasoning breaks | Answer stops |
|---|---:|---:|---:|---:|
| candidate-37 | 3,618 | 5.84M | 31 | 31 |
| v1 | 3,618 | 5.84M | 31 | **0** |
| v2 | 3,618 | 5.84M | 31 | **0** |

```bash
# sp8, 2026-11-02; T=/tmp/vllm-loop-v3 (vLLM loop-handling: v1 6698d7467, v2 050e8b1cc), V=/opt/spark-ds41f/candidate/vllm
docker run --rm -v /tmp/loopres:/res -v ~/.cache/huggingface/hub:/hub:ro \
  -v ~/projects/spark-ds41f:/c/repo:ro -v $T/vllm:/c/vllmsrc:ro -v $T/docs:/c/docs:ro \
  [-v $T/vllm/v1/core/sched/utils.py:$V/vllm/v1/core/sched/utils.py:ro \
   -v $T/vllm/v1/request.py:$V/vllm/v1/request.py:ro]   # this change; omitted for candidate-37
  -e HF_HUB_OFFLINE=1 --entrypoint python3 spark-ds41f:candidate-37 /res/replay.py \
  --tokenizer /hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e10…/tokenizer.json \
  --out /res/replay-ARM.json --files /c/repo /c/vllmsrc /c/docs
```

The token-soup rule adds no break. The 31 reasoning breaks are the existing
rules on code (repeated list entries, include lines); a break ends reasoning
rather than the request, but it is the false-positive surface the operator
named, and code is not reasoning: it needs real reasoning traces to measure.

**Real token counts** ([shapes.py](shapes.py); receipts
`runs/shapes-*.txt`). Legitimate repetition and loop shapes, tokenized with
the V4.1 tokenizer, go through the same `check_stop` as visible answers. Each
loop follows 1,500 tokens of prose. Cells give the token at which a kept
shape is cut, or how many tokens a loop runs before it stops.

| Shape (tokens) | candidate-37 | v1 | v2 |
|---|---:|---:|---:|
| kept: `hello` x200 (200) | cut 25 | kept | kept |
| kept: 128-zero C array (396) | cut 36 | cut 292 | kept |
| kept: 256-zero C array (780) | cut 36 | cut 292 | kept |
| kept: 16x16 matrix of ones (792) | cut 32 | cut 288 | kept |
| kept: separators and a table (453) | cut 27 | kept | kept |
| kept: 200-row table (1,407) | kept | kept | kept |
| kept: 512-zero C array (1,548) | cut 36 | cut 292 | kept |
| long: 300 identical CSV rows (2,405) | cut 36 | cut 292 | cut 2,084 |
| long: 32x32 matrix of `0.0` (5,160) | cut 33 | cut 289 | cut 2,081 |
| loop: production soup | 24 | 447 | **415** |
| loop: one token | 24 | 415 | 2,072 |
| loop: `Let me do it.` | 26 | 440 | 2,074 |
| loop: `hello` runaway | 25 | 416 | 2,073 |

What v2 gives up: a periodic runaway now runs about 2,070 tokens, where v1
stopped it at about 420 (35 s against 7 s at 60 tok/s). Legitimate
repetition past about 2,000 tokens is still cut. Candidate-37 cut every
shape here within 37 tokens.

**Detector cost** (one core of this Mac, every prefix of 144 recorded
256-token outputs): 8.1 µs a token against candidate-37's 7.7 µs; the token
count rule alone, without the pre-check, cost 10.2 µs. v2's soup-shape check
runs only once an answer loop has persisted 256 tokens, so ordinary output
never pays for it. It was not timed separately.

## Known triggers

`triggers.json` lists each known trigger with its source and expectation;
`probes.py BASE_URL` sends them and checks:

| Trigger | Expectation |
|---|---|
| `Say hi.`, `Thanks!` at effort high, temperature 0 (the sink collapse) | clean |
| a quoted ` Vila` soup at effort max, 5 seeds | bounded: no run past 600, no runaway to max_tokens |
| a quoted `Let me do it.` stall at effort max, 5 seeds | bounded |
| `hello` 200 times; a 128-zero C array; a 200-pair separator and table | kept: finish_reason stop |
| 200K tokens of repository docs at effort max, and at high (`--periodic`) | bounded; the pair separates effort from context length |

## Hardware soak

The owner authorized this soak in the sp5–sp7 lab window (tp3-switched,
2026-11-02, window `spark-0013-1793639688`), after the #842 crash
reproduction.

**v1** ([lab-soak.json](lab-soak.json), run `loop-soak-sp`, from 17:41:43Z)
boots three arms in this order: v1, candidate-37, v1 again.
- [arm-tp3-switched.json](arm-tp3-switched.json) is the profile sp5–sp7 serve
  (`config/candidate-37-tp3-switched-dev.local.json`) with `overlay/` mounted
  over the image's tree. `overlay/` holds vLLM 6698d7467's `utils.py`,
  `scheduler.py` and `request.py`. Their base files (5b7cfa37f) are
  byte-identical to the image's, and the image is digest `sha256:e1ee7b1953b1…`
  on sp5, sp6, sp7 and sp8.
- Each arm runs the lean bench, `probes.py` (the triggers below, without the
  long-context pair) and 16 distinct streams of 1,024 tokens.
- [tables_arms.py](tables_arms.py) prints each probe's verdict and the speeds
  against candidate-37.
- The SM clocks were reset by hand at about 17:42Z, during this job's first
  boot, after a failed clock-lock job in the queue before it (see #842's
  record). Its steps' telemetry shows whether any ran locked.

First arm (v1): 15 probe replies, 2 failed, both legitimate repetition
stopped with `finish_reason` repetition, and the engine stayed healthy.
- `repeat-word-200`: stopped at 281 tokens. The model was asked for 200 and
  wrote 281.
- `zero-128-array`: stopped at 292 tokens. Only about 93 of the 128 zeros
  had been written.

Every other reply passed, including the sink triggers (clean), the quoted
soup and stall at max (bounded: the longest run was 12 tokens) and the
separator table (kept). The candidate-37 and v1-repeat arms are still to
report.

**v2** ([lab-probes-v2.json](lab-probes-v2.json), run `loop-v2-probes-sp`,
queued last in the same window) boots one arm.
- [arm-tp3-switched-v2.json](arm-tp3-switched-v2.json) mounts `overlay-v2/`'s
  `utils.py` (vLLM 050e8b1cc) with `overlay/`'s `scheduler.py` and
  `request.py`, which v2 doesn't change.
- It runs `probes.py` and the 16 streams, with no bench: v2 changes only when
  an answer loop is stopped.
- The probes' `kept` verdict now means the loop handling didn't cut the
  answer (`finish_reason` is not repetition). A model that runs past what was
  asked and ends on length still counts as kept.

## Not done

- **Hardware.** v2's probe soak is queued, not yet run. The release suites and a
  thinking-on determinism check (the operator's bit-identity checks send
  `thinking: false` only) still need a window.
- **Real reasoning traces.** The reasoning-side false-positive rate needs
  recorded max-effort reasoning; `replay.py --jsonl` takes
  `{"reasoning", "content"}` lines for that.
- **Root cause.** The sink detector watches one residual column (2455).
  Watching the other near-zero `norm.weight` channels (529, 2953, 1896) is a
  hypothesis: no measurement shows them spiking at a collapse onset, and on
  the quoted-soup replay no channel needs to (the loop is in context with
  ordinary residuals). The measurement is the long-context pair above with the
  four channels logged at each sampled row.
- **Recovery.** Stopping is the last resort. A better recovery rolls the loop
  back out of the model's context (truncate the request and recompute from the
  prefix cache, as preemption does) and forces `</think>` or resamples there.
  It touches the scheduler, the async placeholders and the DSpark state, and
  needs its own experiment.
