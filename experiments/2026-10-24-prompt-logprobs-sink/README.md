# Prompt logprobs kill the engine under sink rejection

One chat request with `prompt_logprobs` takes the whole serving stack down: EngineCore
dies on the scheduler's "no partial prefill outputs" assertion. The cause is sink
rejection (vLLM 0221, on by default for DeepSeek V4.1 since 0223), which holds back the
token that the prompt's last step samples. The fix is one vLLM patch,
[vllm/0266-Sink-rejection-prompt-logprobs-go-out-with-the-reque.patch](vllm/0266-Sink-rejection-prompt-logprobs-go-out-with-the-reque.patch),
on candidate 4's series. [vllm/series](vllm/series) is candidate 4's with 0266.
Production (spark-ds41f-1022) has the same code and is affected; nothing is deployed.
The lab check passed with 0266 on candidate 4 and on the promoted recipe. The crashing
request returned the model's own prompt logprobs, and every temperature-0 output was
unchanged.

## What was seen (2026-10-24 22:34 CET, run `tilelang-ports-controls`)

The `prompt-logprobs` scenario of `experiments/2026-10-24-tilelang-completion/controls_digest.py`
sent a chat completion with these settings:
- `prompt_logprobs: 2`, temperature 0, `logprobs: true`;
- `chat_template_kwargs.thinking=false`;
- the first prompt (about 30 prompt tokens), `max_tokens` 96, `ignore_eos`.

It returned HTTP 500 in both builds: candidate 4 (`spark-ds41f:19f2c20ed4d6-next-4`) and the
TileLang ports arm. The engine logs are on dgx1, under
`results/private/determinism/tilelang-ports-controls/controls-{candidate4,ports,ports-end}.jsonl.engine-dgx1.log`.

```text
File ".../vllm/v1/engine/core.py", line 867, in step_with_batch_queue
File ".../vllm/v1/core/sched/scheduler.py", line 3424, in update_from_output
    assert not prompt_logprobs_tensors
AssertionError
→ EngineDeadError, HTTP 500
```

The scenario's one-token probe, sent just before, was the same request with `max_tokens` 1.
It returned 200.

## Root cause

Line numbers are candidate 4's tree (`97fc3579`). They are the same in production's.

1. The V2 runner's `PromptLogprobsWorker.compute_prompt_logprobs`
   (`vllm/v1/worker/gpu/sample/prompt_logprob.py`) keeps the logprobs of each prompt
   chunk until the step that finishes the prompt. That step returns them all, in
   `ModelRunnerOutput.prompt_logprobs_dict`. It is also the step that samples the
   request's first token.
2. With sink rejection, `Scheduler.update_from_output` does not commit the last token a
   step samples (`scheduler.py:3192-3240`). It holds the token back as the request's
   provisional token, which its next step verifies as draft zero:
   `new_token_ids = generated_token_ids[:-1]`. The prompt's last step samples exactly one
   token, so `new_token_ids` is empty. The request has not stopped, so
   `should_emit_output` (`scheduler.py:3311`) is false and the step has no output for the
   request.
3. A step with no output must not carry prompt logprobs. That is the upstream invariant
   from the chunked-prefill days ("EngineCore returns no partial prefill outputs",
   `scheduler.py:3424`). This step carries the complete prompt logprobs, so the assertion
   fails. EngineCore exits, and the API server answers every request with
   `EngineDeadError`.
4. The one-token probe passed for another reason. Its first token reaches `max_tokens`,
   and `stops_on_next_token` commits a token that no step will follow at once
   (`scheduler.py:3227`). So that step had output, and the prompt logprobs went out with
   it. A prompt whose first token is EOS or a stop token behaves the same way.

What did not cause it:
- **Async scheduling and late admission.** The scheduler tests below fail the same way
  with synchronous scheduling and without late admission. The prompt logprobs do arrive
  in a step with no new token, as suspected, but the hold is what empties the step.
- **The ports, and the runner's prompt-logprob code.** The runner returns the right
  logprobs, once, at the right step.

**Who triggers it:** any request that asks for prompt logprobs and generates more than
one token. That includes:
- chat or completion requests with `prompt_logprobs`, 0 included;
- chat requests with `echo: true`. These set `prompt_logprobs` to `top_logprobs`, which
  defaults to 0 (`chat_completion/protocol.py:762`);
- completion requests with `echo` and `logprobs`.

**Builds:**
- **Production, spark-ds41f-1022** (image `vllm-ds41f-kkref:19f2c20ed4d6-1022-4`, vLLM tree
  `e68aa19c`): affected. That tree is commit `8fc15c18f` of the release branch. Its
  `scheduler.py`, `request.py`, `async_scheduler.py`, `prompt_logprob.py`, the engine core
  and the output processor are byte-identical to candidate 4's. `config/cluster.json` sets
  no `VLLM_SINK_REJECTION`, so `SinkConfig.for_model` turns rejection on for DeepSeek
  V4.1. Production has been exposed since the 1022 deploy (2026-10-23 02:10 CET). No
  request was sent to production to check this.
- **Candidate 4 and the TileLang ports branch:** affected (seen).
- **r6x and earlier:** not affected. They do not carry 0221.

## The fix (0266)

`vllm/v1/core/sched/scheduler.py` and `vllm/v1/request.py`. Patch tree `baa1aea6` on
candidate 4's `97fc3579`; it is vLLM commit `5acb05bc7` on branch `prompt-logprobs-sink`,
in `~/projects/vllm-ds41-prompt-logprobs`.

- In a step with no output for the request, prompt logprobs are kept on the request
  (`Request.sink_prompt_logprobs`) when sink rejection held back, or a stale output
  dropped, the token the step sampled. The scheduler emits them with the request's
  first output, beside that token. That is where they go out when no token is held.
- A request can be preempted while its first token is still held. It has committed
  nothing, so the runner computes its resumed prompt's logprobs again, and the new ones
  replace those held. They go out once.
- Every other step with no output still asserts that it carries no prompt logprobs.
- No other request is touched. The prompt logprobs are tensors the scheduler already
  holds (CPU copies from `AsyncOutput`), and the request keeps them for one more step.

## Tests

The tests are added to `tests/v1/core/test_sink_draft_zero.py`, on its scripted V2-runner
worker. The worker now returns prompt logprobs as `PromptLogprobsWorker` does: accumulated
over chunks, returned at the prompt's last step, and skipped for a request resumed after
its prompt. There are fifteen cases:

- `test_prompt_logprobs_go_out_with_the_held_first_token`: sync and async scheduling, no
  drafts or two, a whole prompt or 16-token chunks. Checks:
  - the prompt logprobs go out once, in the request's first output, beside a token, and
    they are the worker's;
  - the request's tokens, and those of a request beside it, equal the same run's without
    prompt logprobs.
- `test_prompt_logprobs_reach_the_response`: the output processor, `FINAL_ONLY` and
  `DELTA`. One response carries the prompt logprobs, one per prompt token after the first.
- `test_prompt_logprobs_held_across_a_preemption`: preemption while the first token is
  held (`reset_prefix_cache(reset_running_requests=True)`, sync and async). The resumed
  prompt's logprobs replace the held ones and go out once.
- `test_prompt_logprobs_once_across_preemptions`: KV pressure that preempts after tokens
  were committed. The prompt logprobs go out once and are not computed again.

All tests ran on CPU on a Mac (macOS arm64, torch 2.14.1). The tree comes from
`PYTHONPATH`, with no container and no cluster. The method is in
[runs/run-suites.sh](runs/run-suites.sh).

| Run | Result |
| --- | --- |
| The new tests on candidate 4's scheduler and request ([runs/tests-before-fix.txt](runs/tests-before-fix.txt)) | 15 failed, all at `scheduler.py:3424` `assert not prompt_logprobs_tensors`; 48 passed |
| 0266: `test_sink_draft_zero.py`, `test_late_admission.py`, `test_late_admission_sink.py` ([runs/tests-fix.txt](runs/tests-fix.txt)) | 87 passed |
| Those, with `test_scheduler.py`, `test_async_scheduler.py` and `test_output_processor.py`, on both trees ([runs/suites.txt](runs/suites.txt)) | candidate 4: 257 passed. 0266: 272 passed. Both: the same 72 failing ids (gated `meta-llama/Llama-3.2-1B` downloads, model configs) |

`ruff check` and `ruff format --check` (0.14.0, the tree's pre-commit pin) are clean. The
series applies to `19f2c20ed` with `git am`: 0265 gives candidate 4's tree `97fc3579`, and
0266 gives `baa1aea6`.

## Lab check

Run `prompt-logprobs-sink`, spec [serving-check.json](serving-check.json). One `measure`
job on the ring, with no bench and one extra,
[prompt_logprobs_check.py](prompt_logprobs_check.py), on four arms:

| Arm | Config | Prompt-logprob requests |
| --- | --- | --- |
| `candidate4` | `experiments/2026-10-23-next-release/rc4/rc-tp4.json` | none (its engine dies on them) |
| `candidate4-fix` | [fix-tp4.json](fix-tp4.json): candidate 4 with 0266's two files mounted over the image's tree | yes |
| `production-fix` | [production-fix-tp4.json](production-fix-tp4.json): the promoted recipe (`-1022-4`) with the same two files, a hotfix candidate | yes |
| `production` | [production-tp4.json](production-tp4.json): the promoted recipe as it serves | none |

The overlay files are [overlay/](overlay/) (candidate 4's tree with 0266, which is also
production's tree with 0266).

The check runs these steps:
1. The seven other prompts, one at a time.
2. The probe (the crashing request), without prompt logprobs.
3. In the fix arms only:
   - the probe with `prompt_logprobs: 2`;
   - the probe with `echo: true`;
   - the seven and the probe at once;
   - a teacher-forced completion of the probe's prompt and reply tokens, with prompt
     logprobs. It scores the probe's prompt again and every token it chose.

An arm passes on these:
- every request returns HTTP 200;
- one prompt-logprob entry per prompt token, the first null, each holding its prompt
  token and the top two at ranks 1-2;
- the probe's reply is the same with and without prompt logprobs, and alone or beside
  the others;
- the seven's replies are the same alone and beside the probe.

[tables_arms.py](tables_arms.py) then requires each fix arm's temperature-0 digests to be
its unfixed arm's. The teacher's agreement is recorded, not gated, because it scores the
tokens in another forward shape.

### Results (2026-10-24 23:46-23:54 CET, window `dgx1-1792878362`): pass

The records are in [runs/lab/](runs/lab/):
- the table, [prompt-logprobs-sink-table.txt](runs/lab/prompt-logprobs-sink-table.txt);
- each arm's record, `prompt-logprobs-<arm>.jsonl`;
- the run log, [serving-check.out](runs/lab/serving-check.out);
- the driver, [run.sh](runs/lab/run.sh) and [run.log](runs/lab/run.log).

The driver waited for the previous window (`tilelang-ports-serving-3`) to close at 23:41 CET,
then for two quiet minutes. It synced the nodes to `827783c`.

Boot times, to ready:

| Arm | Ready |
| --- | ---: |
| `candidate4` | +83.0 s |
| `candidate4-fix` | +92.6 s |
| `production-fix` | +77.5 s |
| `production` | +89.0 s |

Production stopped at 23:46:23 and served again from the last arm at 23:53:43. The close
found `config/cluster.json` already serving. `doctor --live` was OK, with the 5 known
weight-fragmentation warnings.

| | `candidate4` | `candidate4-fix` | `production-fix` | `production` |
| --- | --- | --- | --- | --- |
| Every request | 200 | 200 | 200 | 200 |
| The crashing request (`prompt_logprobs: 2`) | not sent | 200: 18 entries, the first null, each with its prompt token and ranks 1-2 | the same | not sent |
| `echo: true` | not sent | 200, the probe's tokens | 200, the probe's tokens | not sent |
| Prompt logprobs, alone and beside the seven | | identical | identical | |
| Teacher forcing: largest gap, prompt / 96 reply tokens | | 0.0 / 0.0, no rank differs | 0.0 / 0.0, no rank differs | |
| The seven's replies, alone = beside the probe | yes | yes | yes | yes |
| The probe's reply, plain = with prompt logprobs = beside the seven | yes | yes | yes | yes |

Every arm has the same digests: the seven's replies `2047dcbf…` and the probe's `b1e35aed…`.
So:
- 0266 changes no temperature-0 output in either build;
- prompt logprobs on one request change neither its own reply nor the replies beside it;
- candidate 4 and production agree, as #503 recorded.

The teacher-forced completion scores the probe's prompt and its 96 reply tokens again in
another forward shape (114 tokens of prefill). It gives the same logprobs to the last
digit. So the returned prompt logprobs are the model's own, and so are the reply's
logprobs.

## Production

Production is affected now. Any client that asks for prompt logprobs, or sends `echo`,
stops the service until it is restarted. This session made no production change. The
options, for the owner:

1. **Hotfix:** boot the promoted recipe with 0266's two files mounted
   (`production-fix-tp4.json`). It passed the lab check above, with outputs identical to
   production's. This means one production restart. It keeps sink rejection, and prompt
   logprobs work.
2. **Reject at validation:** answer `prompt_logprobs` and `echo` with HTTP 400 while sink
   rejection is on, as vLLM rejects `min_p` under speculative decoding. This is a code
   change, a restart and a lost feature, with no gain over option 1.
3. **Sink rejection off** (`VLLM_SINK_REJECTION=0`): a restart, no code change. It drops
   the default the owner set for DS4.1-1017.
4. **Wait for the next release** with 0266 in it. Production stays exposed until then.

The recommendation is option 1, then 0266 in the next release (candidate 4's series plus
0266, `vllm/series`). Both wait for the owner's go.

## Follow-ups

- `scripts/lab.py`: when an arm's engine dies, its containers stay up on dgx2-dgx4. The
  production restore then fails, as it did after run `tilelang-ports-controls`. Production
  was restored by hand: `bin/spark --cluster-config experiments/2026-10-24-tilelang-completion/ports-tp4.json cluster stop --apply`,
  then `scripts/lab.py window close --production-config config/cluster.json`. The lab
  should stop a dead arm with its own profile before restoring.
- `controls_digest.py` can take its `prompt-logprobs` scenario back once 0266 is in the
  build it measures.
- `logprob_token_ids` is a separate bug, reported by the TileLang ports session. Chat
  requests with custom logprob token ids return HTTP 500 under DSpark
  (`_create_chat_logprobs` IndexError); the engine survives. It is not fixed here.
