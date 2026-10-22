# Late admission and the candidate's prefill: the Engram trace, and a fix held for a decision

Owner, 2026-10-22, on spark-ds41f-1022's prefill (about 1% below r6x): "please fix it".

**Status (2026-10-22 16:10 CET):** the suspected cause is not confirmed. In the Engram debug
timeline the candidate's prefill chunks wait for their rows exactly as r6x's do. The loss is in
the cadence of the chunks instead:
- the second chunk starts 44-49 ms later after every prompt's first chunk;
- each later chunk takes 7.0 ms longer on average (0.8%).

The first fits the late decision of the next step. Whether the second does is open: it is about
the whole loss at 131K, and the trace cannot tell late admission from GPU time added by the
candidate's other pieces. As instructed for a different cause, no A/B window was queued.
vLLM 0226 (below) is written and passes its CPU tests, and its overlay arm and two run specs are
staged on dgx1-3, waiting for the owner's choice.

Base: deployment commit 6d8ff8a (main, #457).

## The question

The candidate (r6x with vLLM 0185, 0221-0225 and sparknet 0.5.0,
[2026-10-22-spark-ds41f-1022](../2026-10-22-spark-ds41f-1022/README.md)) measured prefill
about 1% below r6x: 32K −0.82% and 131K −0.74% in the trimmed promotion comparison
(candidate first), and −1.1% in the focused gate.

The suspect was late admission (0222, decided at decoder layer 30). It decides step N+1 at layer
30 of step N instead of as soon as N is under way. That delays the next prefill chunk's
preparation, including its Engram read-ahead. r6r's timeline
([2026-10-16-r6r](../2026-10-16-r6r/README.md)) had shown chunks 2-N never waiting on reads under
async scheduling. If the candidate's chunks 2-N now waited and r6x's did not, the fix was to
decide early when the step in flight continues a prompt.

## The trace (2026-10-22 15:48-15:58 CET)

Run `prefill-trace-1022`, queued as `20261022T133759-prefill-trace.json` by the coordinating
session. The window opened after ten idle minutes (`scripts/lab_idle_start.sh 10`); its release
watcher closed it once the job ended. No foreign requests.
- **Arms:** `results/private/lab/prefill-trace/t-1022.json` (the candidate's TP3 config) and
  `t-r6x.json` (`experiments/2026-10-19-r6x-tp3-1m/r6x-tp3-1m-1017.json`). Both run at the recipe's
  production sizing (1M context, 4.9 GiB KV) with `SPARK3_ENGRAM_TIMELINE=1`, staged on dgx1-3.
- **Workload:** `bin/spark bench --compare none --suites prefill --prefill-text source
  --prefill-sizes 32768,131072 --prefill-repeats 3`. That is six cold prompts (26,289, 36,341
  and 29,199 tokens for "32K"; 120,242, 116,715 and 123,163 for "131K"), the same prompts in the
  same order in both arms, one output token each. TP3 chunks are 4,048 tokens.
- **Boots:** candidate ready 15:50:28 (+90.7 s), bench to 15:52:20; r6x ready 15:53:57 (+91.7 s),
  bench to 15:55:48. Production r6x restored 15:57:53, `doctor --live` OK and hold removed at
  15:58:01.
- **Memory (min MemAvailable, dgx1 / dgx2 / dgx3):** candidate 5.72 / 7.04 / 7.43 GiB, r6x
  5.72 / 7.56 / 7.33 GiB. No swap growth, no thermal slowdown.

| | candidate | r6x |
| --- | ---: | ---: |
| 32K prefill, tok/s (n = 3) | 4549.1 ± 5.2% | 4583.6 ± 5.5% |
| 131K prefill, tok/s (n = 3) | 4445.2 ± 0.4% | 4460.6 ± 1.8% |

The timeline appended to `cache/kkref/engram-timeline/<host>-tp<rank>-layer<id>.jsonl` on every
node: one record per step and disk table (layers 1 and 14), with the reader's monotonic times
and the GPU's wait for the rows. The reader's sequence restarts at each boot, which splits the
arms. [timeline_chunks.py](timeline_chunks.py) groups each boot's prefill steps into prompts and
lines up the ranks by sequence ([runs/trace-chunks.txt](runs/trace-chunks.txt),
[runs/trace-chunks.json](runs/trace-chunks.json)). Analysis copies are on dgx1 under
`results/private/lab/prefill-trace/analysis/`.

### Result ([runs/trace-paired.txt](runs/trace-paired.txt))

Per prompt, the candidate against r6x (the same prompt). GPU waits are the largest over the
three ranks and both Engram layers:

| tokens | chunks | TTFT diff | chunk 1 → 2 | chunks 2..N−1, mean | chunk 2 wait (cand. / r6x) | chunks 3..N wait |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 120,242 | 30 | −151 ms* | −379 ms* | +9.3 ms | 23.1 / 28.2 ms | 0 / 9.7 ms |
| 26,289 | 7 | +30 ms | +44 ms | +4.9 ms | 24.5 / 22.2 ms | 0 / 0 |
| 116,715 | 29 | +209 ms | +44 ms | +7.3 ms | 31.4 / 33.2 ms | 0 / 0 |
| 36,341 | 9 | +75 ms | +48 ms | +8.5 ms | 25.5 / 26.4 ms | 0 / 0 |
| 123,163 | 31 | +218 ms | +49 ms | +6.8 ms | 29.0 / 30.8 ms | 0 / 0 |
| 29,199 | 8 | +50 ms | +48 ms | +6.5 ms | 32.5 / 34.0 ms | 0 / 0 |

\* r6x's first prefill after its boot (its chunk 1 → 2 took 1,200 ms). The pooled figure leaves
it out: chunks 2..N−1 are +7.0 ms mean, +6.7 ms median (sd 4.9 ms, n = 74).

What the trace shows:
- **Engram waits: the same in both arms.** In every prompt of both arms chunk 2 waits 22-34 ms
  for its rows, and chunks 3-N wait nothing (one 9.7 ms wait in r6x). A chunk's rows are
  published 60-70 ms after the reader receives it. Late admission does not make prefill chunks
  wait on Engram reads, so the wait explains none of the loss.
- **Chunk 1 → 2: +44 to +49 ms in all five clean pairs.** Chunk 2 is submitted that much later,
  and its wait is unchanged, so the GPU starts chunk 2 that much later. GPU time added to every
  chunk cannot account for it: that is about 7 ms a chunk (next point). This is a dispatch delay
  after a prompt's first chunk, and the step after that chunk is decided late in the candidate
  and at once in r6x.
- **Every later chunk: +7.0 ms mean (0.8%).** That is +190-210 ms at 131K, about all of that
  size's loss. It could be the next chunk's late dispatch again, or GPU time the candidate adds
  to every chunk: the sink detector's column (0221/0223), sparknet 0.5.0's wider one-shot read,
  or the RoCE-policy broadcast (0185). The Engram timeline has no GPU timestamps, so it cannot
  separate them. The late-admission screen alone measured prefill within its intervals of r6x
  (32K −0.4%, 128K −0.3%, ±1%). That leaves room for either.
- **TTFT against the chunk cadence:** each clean pair's TTFT difference is 30-40 ms smaller than
  its chunk-cadence difference. The rest of each request (before chunk 1, and from the last chunk
  to its token, sink rejection's held token included) is that much shorter in the candidate; this
  trace does not show why.

The suspected mechanism (Engram read-ahead) is ruled out. The late decision after a prompt's
first chunk is a measured cost, about 45 ms a prompt. The per-chunk 0.8% is not attributed.

## vLLM 0226, prepared and not yet measured

[vllm/0226-Late-admission-decide-the-step-after-a-continuing-pr.patch](vllm/0226-Late-admission-decide-the-step-after-a-continuing-pr.patch)
is on the candidate's series (patch head 672231a1b). It is commit 7ec49e2a0, tree 3f23caaf, on
branch `late-admission-prefill` of the prepared tree on dgx4
(`~/spark-lab/la-prefill/repo/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-16175289/src/vllm`).

- `SchedulerInterface.continues_prefill(scheduler_output)`: whether a scheduled step carries a
  prompt chunk that is not its prompt's last (`is_prefill_chunk` after scheduling). It defaults
  to False, so another scheduler keeps late admission for every step.
- `EngineCore.step_with_batch_queue`: a step in flight that continues a prompt has its successor
  decided at once, as async scheduling does. The step after a prompt's last chunk, and every
  decode step, is still decided late. That step is the one an arrival can join, so the TTFT
  gain at four and eight streams should stay.
- Nothing is lost for arrivals. Beside a whole chunk the long-prefill threshold (4,048 of 4,096)
  leaves only the decode rows, and a new prompt needs a whole aligned chunk to start.

It removes the measured late dispatch after a prompt's first chunk (about 45 ms a prompt), and
the per-chunk 0.8% only if that comes from late admission too.

### Tests ([runs/run-tests.sh](runs/run-tests.sh), [runs/tests.txt](runs/tests.txt))

Throwaway containers from image `-1022-2` on dgx4 (idle, not in the serving set), with 0226's
files mounted over the image's tree (its base). No network.

| Suite | Result |
| --- | --- |
| `tests/v1/engine/test_late_admission.py` (one new test) and `test_late_admission_sink.py` (CPU) | 24 passed |
| `tests/v1/core/test_sink_draft_zero.py` (CPU) | 48 passed |
| `test_scheduler.py`, `test_async_scheduler.py` (GPU container), with 0226 | 180 passed, 40 failed |
| the same, the image as built | 180 passed, 40 failed: the same 40 ids, all in the recorded offline failures ([offline-failures.txt](../2026-10-19-sink-rejection-v2/runs/offline-failures.txt)) |

The new test (`test_a_continuing_prompt_decides_the_next_step_at_once`) runs the real engine-core
loop and async scheduler with a 16-token budget:
- while a 40-token prompt's first and second chunks are in flight, the next steps are decided
  without waiting and nothing is admitted;
- an arrival during its last chunk joins the next step, decided late;
- that step (a decode beside the arrival's first chunk) has its successor decided at once.

## Staged next window (not queued)

[make_arms.py](make_arms.py) writes the arms and specs. `--staged results/private/lab/la-prefill`
wrote the copy now on dgx1, dgx2 and dgx3, with 0226's three files under `overlay/`. Both specs
pass `scripts/lab.py run --dry-run`.

1. **[ab-tp3.json](ab-tp3.json)**, about 40 min: 0226, the candidate as built, 0226 again. Each arm
   runs cold 32K/131K prefill (3 repeats), the traced TTFT replay
   (`experiments/2026-10-20-ttft-gap/ttft_trace.py`), and `token_determinism_text.py --wide` with
   and without log probabilities. Pass: outputs identical (62313cb5 / 9742ac2c and
   4d143977 / 12fd8257), prefill at r6x's level, TTFT at four and eight streams level with the
   candidate's.
2. **[attribution-tp3.json](attribution-tp3.json)**, about 12 min behind it in the same window: the
   same prefill on the candidate with late admission off, with sink rejection off
   (`VLLM_SINK_REJECTION=0`), and on r6x. It splits the per-chunk 0.8% among late admission,
   sink rejection and the rest (sparknet 0.5.0, 0185). Run it only if the owner wants the
   attribution beside the A/B.

To queue (after the shared-window checks in AGENTS.md):

```sh
cd ~/projects/spark-ds41f
python3 scripts/lab.py queue add results/private/lab/la-prefill/ab-tp3.json
python3 scripts/lab.py queue add results/private/lab/la-prefill/attribution-tp3.json   # optional
setsid -f scripts/lab_idle_start.sh 10 queue-la-prefill \
  > results/private/lab/queue-la-prefill-idle.log 2>&1 < /dev/null
```

## Files

| File | What it is |
| --- | --- |
| [timeline_chunks.py](timeline_chunks.py) | per-chunk Engram waits and cadence of cold prefills from the debug timeline, every rank and layer |
| [runs/trace-chunks.txt](runs/trace-chunks.txt), [runs/trace-chunks.json](runs/trace-chunks.json), [runs/trace-paired.txt](runs/trace-paired.txt) | the trace's analysis: per prompt, then paired |
| [vllm/0226-...patch](vllm/0226-Late-admission-decide-the-step-after-a-continuing-pr.patch) | the prepared fix |
| [overlay/](overlay) | 0226's changed sources, mounted over image `-1022-2` |
| [make_arms.py](make_arms.py), `*.json` | the arms (fix, current, late admission off, sink rejection off, r6x) and the two specs |
| [runs/run-tests.sh](runs/run-tests.sh), [runs/tests.txt](runs/tests.txt) | the CPU tests and their results |
