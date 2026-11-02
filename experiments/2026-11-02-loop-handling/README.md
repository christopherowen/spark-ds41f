# Loop handling: act on a loop by where it is

Owner, 2026-11-02: work through the best possible handling of reasoning loops
and token soup in a new PR; review the operator's comments on #787 for
insights; make test cases from the known triggers.

**Status:** unit tests and an offline replay pass on candidate-37's image on
sp8 (2026-11-02). Not run on hardware: the trigger probes (`probes.py`) and the
release suites need a lab window. Not promoted; no recipe change.

- Base: candidate-37, vLLM patch head `5b7cfa37f` (tree `af0fb152`), image
  `spark-ds41f:candidate-37` (`sha256:e1ee7b19…` on sp8).
- One variable: what `check_stop` does when its loop detector fires
  (`vllm/v1/core/sched/utils.py`, `vllm/v1/request.py`), and one log line in
  `vllm/v1/core/sched/scheduler.py`.
- Patch: [0001-feat-v1-act-on-a-loop-by-where-it-is-and-follow-the-reasoning-parser.patch](0001-feat-v1-act-on-a-loop-by-where-it-is-and-follow-the-reasoning-parser.patch),
  one commit on the base (vLLM `6698d7467`, branch `loop-handling` in
  `/tmp/vllm-loop-v3` on sp8), next in the series after 0676.

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
| in the answer of a request whose reasoning did not loop | stop only a loop that persists 256 tokens past its first detection (gaps up to 16) **and** leaves fewer than one distinct token in ten among the answer's last 512 tokens (the operator's condition) |

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

**Unit tests** (`tests/v1/core/test_repetition_detection.py`, 39 tests):
token soup (incident 1's shape; interjections of new cone tokens; bit strings,
zero matrices and prose that must not match), the parser-following markers,
the forced-end grace, the answer's persistence and diversity, and the known
trigger shapes (the 736-token run inside 1,500 varied tokens stops within 512
tokens of its start; separator runs of 199 and 300 tokens in varied output,
`hello` 200 times, a 128-zero array and a 100-token one-token answer are kept;
the `Let me do it.` stall breaks reasoning at its 24th token).

```text
image spark-ds41f:candidate-37 on sp8, changed files mounted:
tests/v1/core/test_repetition_detection.py + test_copy_drafts_scheduler.py
  + test_sink_draft_zero.py + test_output.py        121 passed
the same plus test_scheduler.py, test_async_scheduler.py, test_scheduler_e2e.py:
  candidate-37  217 failed, 116 passed, 2 errors
  this change   217 failed, 126 passed, 2 errors   (identical failing set:
                the offline hub lacks facebook/opt-125m)
```

**Offline replay** (`replay.py`, receipts in `runs/`): 1,500 files of this
repository, the vLLM source and docs, cut into 2,048-token chunks, each fed
token by token through the real `check_stop` once as reasoning and once as a
visible answer.

| Arm | Chunks | Tokens | Reasoning breaks | Answer stops |
|---|---:|---:|---:|---:|
| candidate-37 | 3,618 | 5.84M | 31 | 31 |
| this change | 3,618 | 5.84M | 31 | **0** |

```bash
# sp8, 2026-11-02; T=/tmp/vllm-loop-v3 (vLLM loop-handling, 6698d7467), V=/opt/spark-ds41f/candidate/vllm
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

**Detector cost** (one core of this Mac, every prefix of 144 recorded
256-token outputs): 8.1 µs a token against candidate-37's 7.7 µs; the token
count rule alone, without the pre-check, cost 10.2 µs.

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

## Not done

- **Hardware.** The probes, the release suites and a thinking-on determinism
  check (the operator's bit-identity checks send `thinking: false` only) need
  a window.
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
