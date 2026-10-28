# Paired single-stream verdict and rows a step (2026-10-26, CET)

Items 1 and 3 of the core-metrics plan ([2026-10-26-core-metrics-plan](../2026-10-26-core-metrics-plan/README.md)):

1. a paired single-stream verdict: per-prompt decode time, paired across builds on identical text;
2. rows a step: per-position acceptance on run steps (the counters exist), and a plan for live
   and dead row counters in vLLM.

The cluster program's lean A/Bs tonight (attention front plus prefetch, about +1.1% at c1; the
copy drafter, about +4.75%; prefill yield) need a c1 verdict that resolves about 0.3%. The
window verdict's ±3.7% resolved none of them.

- **Base:** origin/main a4934480. Laptop only: no cluster or GPU time, no serving change.
- **Data:** the lab's kept suite runs, copied read-only from dgx1
  (`~/projects/spark-ds41f/results/private/suites/*/*/{c1,c4,c8,c16,suite}.json`, 76 runs).

## What changed

**servebench** (client side only; the suites and their recorded references are unchanged):
- `run` keeps a compact `request_table` in every result: one row a request (index, prompt,
  slot, sent, first, last, end, prompt and completion tokens, the first chunk's tokens,
  output chunks, error) and the window, on the run's clock. It adds about 13 KB to a c1
  result and 40 KB to a c16 one.
- `compare` pairs the requests both runs sent in their windows by index. The seed fixes the
  order, so index i is the same prompt on both builds. Two rows come from the pairs:
  - **decode tok/s, paired:** each request's output tokens after the first over the time from
    its first chunk to its last;
  - **TTFT, paired:** each request's time to first token.

  Each is the median ratio with an exact (sign-test) interval over requests. The note says
  how many pairs streamed in as many chunks on both builds (the same text and drafting give
  the same steps). A run without a table is paired window by window instead (`output tok/s,
  window-paired`).
- **Verdict block** (`suite.with_floor`): the paired decode row is judged with a recorded A/A
  floor by step, not against the run's own repeat alone (see 3 below).
  - The change is the mean over both candidate boots.
  - The floor is the A/A bound, or the repeat when that is larger, times sqrt(3/4) for two boots.
  - The margin is the interval's side toward no change and the floor, in quadrature.
  - The window row stays beside it with its old interval.
- **Rows a step:** `prom` reads the per-position counters on every scrape. Each window then
  has survival (accepted at k a draft) and coverage (verified at k a draft).
  `report.rows_a_step` derives decoding requests, verified rows, row yield and wasted rows a
  step from the counters every run already has, as `core_metrics.py` does.

**Tables:**
- `scripts/servebench_suite_table.py` passes `AA_FLOOR_PCT` and the second candidate boot to
  the verdict block, then prints rows a step for every arm and closed loop.
- `scripts/release_table.py` pairs the release profile's single stream (`c1-distinct`, twelve
  prompts in turn, by turn). It prints the paired row as the single-stream verdict, with the
  0.35% floor, above the old row.

## Validation on the kept runs ([validate.py](validate.py), [runs/validate.txt](runs/validate.txt))

The kept runs have no request tables, so they validate the rule and the floor through the
window-paired fallback, judged exactly as the per-request row is.

**1. The A/A floor.** Every run with arms X and X-again (two boots of one build in one session,
2026-10-15 to 10-25), whole-window output tok/s:

| step | A/A pairs | mean | sd | largest | 1.96 sd (floor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| c1 | 31 | +0.024% | 0.175% | 0.383% | 0.344% |
| c4 | 8 | -0.050% | 0.135% | 0.230% | 0.265% |
| c8 | 26 | -0.020% | 0.200% | 0.369% | 0.392% |
| c16 | 31 | -0.033% | 0.218% | 0.584% | 0.427% |

`AA_FLOOR_PCT` records these: c1 0.35, c4 0.27, c8 0.39, c16 0.43.

**2. A/A pairs read "same".** Each A/A pair was judged as a candidate, with the floor taken from
the other pairs (leave one out):
- c1: 31 of 31 same;
- c4: 8 of 8;
- c8: 26 of 26;
- c16: 30 of 31. The exception is overnight-release-pair's r6k, +0.586% [+0.194, +1.083]
  against a 0.539% margin, the largest c16 A/A on record (expected at 95%: about 1.5 of 31).

**3. Why a recorded floor.** A paired interval is a few hundredths of a percent, so the boot is
nearly all of a comparison's uncertainty, and one repeat is a single draw of it. Judge an A/A
change against another A/A pair's |change| as its only floor: 465 of 930 ordered c1 pairs call a
build different from itself (50%), and the same holds at every step. Today's rule (interval
plus |repeat|) is safe only because the window interval is wide.

**4. Real changes** (reference → candidate, both candidate boots, c1):

| run | window verdict (suite table) | window-paired verdict |
| --- | --- | --- |
| 1022 → 1025 TP4 (candidate-6) | +2.11%, margin 3.88%: same | +2.22% ±2.10%, floor 0.30%, margin 2.12%: **better** |
| 1022 → 1025 TP4, sampled | +2.65%, margin 4.97%: same | +2.68%, margin 2.29%: **better** |
| r6v → r6x TP4 | +4.49%, margin 4.55%: same | +3.88%, margin 1.68%: **better** |
| r6w → scale codes TP4 | +2.09%, margin 4.09%: same | +1.86%, margin 1.09%: **better** |
| r6x → broadcast TP3 | +0.09%: same | +0.06%, margin 0.36%: same |
| main → wide read TP3 | +0.16%: same | +0.21%, margin 0.35%: same |

1022 → 1025's +2.1% resolves, even through the fallback. Pairing helps at c8 and c16 too
(window-paired margins 0.4-0.8% against the window rows' 0.8-1.9%), and its verdicts agree
with the window rows wherever both resolve.

**What the kept runs cannot show:** the per-request interval itself. Window pairing drifts:
with a 2% change the faster build is about 2 prompts ahead by the window's end, so its windows
hold other prompts, and that is the ±2%. Per-request pairs have no drift. From the c1 A/A
window pairs (about ±0.13-0.2% over 10 windows of about 9 prompts), one prompt's ratio spreads
about 0.5%. The median over about 88 prompts should then sit within about ±0.15%. With two
candidate boots the margin would be hypot(0.15, 0.30), about 0.33%: the boot floor sets the
resolution. Tonight's lean A/Bs give the first per-request A/A (candidate → candidate again).
Each job's table prints it as the paired row's repeat.

### c8

- Window pairing holds at c8: margins 0.4-0.8% against 1.1-1.9%.
- Per-request pairing at c8 is weaker in principle, since a request's decode rate also
  depends on its batchmates, which drift between builds. The exact interval assumes
  independent pairs, and eight concurrent requests share step times.
- So the c8 paired row is in the verdict block with the c8 floor, but it is unconfirmed
  until a per-request A/A. It holds if tonight's candidate → candidate-again paired change
  at c8 sits inside its own interval plus the floor. If not, read c8 on the window row.
- A repeat larger than the floor widens the margin automatically.

## Rows a step

From the counters every closed loop already records:
- decoding requests a step = tokens a step / (1 + accepted a draft);
- verified rows a step = requests × (1 + verified a draft);
- row yield = tokens a step / verified rows;
- wasted rows = verified rows − tokens a step.

From this change on, the per-position counters on every run step add survival and coverage by
draft position at c1, c8 and c16, beside the model step's c16 rates. At c1 on 1025 TP4 the
table gives 4.09 rows for 2.28 tokens. Of those, 1.80 rows a step give no token.

### Live and dead rows in vLLM: the plan (not built)

Production marks dead rows (`SPARK_DS41F_DSPARK_DEAD_ROWS_TAU` 0.2,
`adaptive_verification.py`). A verification row past the first draft whose survival (the
running product of the drafts' confidences) falls below tau is marked as padding, so routed MoE
skips it. The request commits at most one token past its last live row. The cut is a
per-request device tensor (`_dead_row_cut`, int32, `dead_row_args`). The resampling kernels
read it (`tilelang_resampled._committed`: `min(count + 1, cut + 1)`).

So wasted rows split into live rows rejected and dead rows (marked, skipped by MoE, still in
attention and the dense layers). A counter with no new synchronization:

1. `AsyncOutput` (`worker/gpu/async_utils.py`) already copies `num_verified_draft_tokens` to the
   host on the output copy stream (`async_copy_to_np`). Copy the cut in the same place. It is
   one int32 a request, behind the same copy event.
2. Add `ModelRunnerOutput.num_live_draft_tokens` (`v1/outputs.py`), next to
   `num_verified_draft_tokens`.
3. In the scheduler's `update_from_output`, where it reads `num_verified_draft_tokens`
   (`v1/core/sched/scheduler.py`), pass live = min(verified, cut) to
   `SpecDecodingStats.observe_draft`. Count `num_live_draft_tokens`, plus live per position.
4. Export `vllm:spec_decode_num_live_draft_tokens` (and `_per_pos`) from `SpecDecodingProm`.
5. servebench's `prom.VLLM` adds them; `rows_a_step` splits wasted rows into rejected live and
   dead.

The counter is plain host arithmetic on a copy the step already makes, so it is free in
production rather than a debug flag. It needs a grid A/B to show that, as the plan's item 3
says, and a GPU test of the copy. It touches five files across the worker, the scheduler
and metrics, and no test box is free tonight (dgx4 serves in the ring). So it is a follow-up
branch from `next-candidate`, not part of this change.

## Using it in a lean A/B

Once this is on main and the lab has synced, nothing in a job changes. Every `run` step writes
its request table, and `scripts/servebench_suite_table.py` prints the paired verdict:
- Run the arms as usual: candidate, reference, candidate again. The single-stream suite
  (`"suite": [..., "single-stream"]`, or a suite whose closed loop is `c1`) gives the c1 row.
- The table's first block has `c1: decode tok/s, paired`: the change over both candidate boots,
  the interval over requests, the repeat, the floor (0.30% for two boots), the margin and the
  verdict, with the window row's change and interval in brackets.
- Read the per-pair blocks' note: "N requests in both windows ... M streamed in as many
  chunks". At c1 greedy, M = N means the same text and drafting. A drafter change (the copy
  drafter) changes chunks but not text, so the decode row still pairs exactly.
- The candidate → candidate-again block's `decode tok/s, paired` is the per-request A/A.
  Record it: it should fall inside its interval plus 0.35%.
- `TTFT, paired` is in each pair's block (no floor; read it there). Prefill yield's effect is
  under concurrency, so read c8 and Poisson TTFT there too.
- Runs compared against a reference kept before this change (no table) fall back to the
  window-paired row, about ±2% at c1.

## Reproduce

```bash
python3 experiments/2026-10-26-paired-verdict/validate.py <copy of results/private/suites>
```

```bash
python3 -m unittest tests.test_servebench tests.test_lab
```
