# CopyDrafter: publish the latest proposal's slot draft limits

Owner, 2026-11-02: take the operator's fix for #852 (PR #855), bring it to the
repository's standard and include it.

**Status:** patch ready, unit-tested, and measured on sp5–sp7 (window
`spark-0013-1793654131`, 2026-11-02). The stale-limit race happens on our switched
TP3 too: without the census overlay, candidate-37's copy probe loses the same tool
acceptance the operator saw, and the fix restores it. The wait costs 0.73% in paired
single-stream decode and nothing measurable at 8 or 16 streams. The crash itself has
not reproduced here: every repro arm so far carried the census, whose per-step sync
hides the race. Not promoted; no recipe change. A GPU-side alternative is #859.

- Base: candidate-37, vLLM patch head `5b7cfa37f` (tree `af0fb152`), image
  `spark-ds41f:candidate-37` (`sha256:e1ee7b1953b1…` on sp5, sp6, sp7 and sp8).
- One variable: `CopyDrafter.sync_slot_draft_limits` waits for the queued limit
  copy instead of keeping the previous limits until it lands.
- Patch: [vllm/0001-fix-copy-drafts-publish-the-latest-proposal-s-slot-draft-limits.patch](vllm/0001-fix-copy-drafts-publish-the-latest-proposal-s-slot-draft-limits.patch),
  vLLM `8bb508091` on `5b7cfa37f` (branch `copy-limit-sync` in
  `/tmp/vllm-limit-sync` on sp8). It applies cleanly to a fresh `5b7cfa37f`
  tree and goes next in the series after 0676.
- Author: Jordi Posthumus, who found the cause and wrote the fix in #852 and
  #855. This experiment adds three things to it:
  - the reconciliation with 0612's test;
  - the removal of the reset bookkeeping the fix makes dead;
  - the record and the measurements still owed.

## What happened

With copy drafts on, an operator's three-node ring (candidate-37 image, a
direct ring) lost its engine to a device assert. It happened deterministically
at `min_match=8`, and under real agent traffic at 16 (#852):

```text
Device assert failed: route_partials: gate logits contain nan or inf
  at vllm/models/deepseek_v4_1/tilelang/router.py:973 in router_partials
```

Cause, found by the operator: patch 0612 (#687, #619) made
`sync_slot_draft_limits` non-blocking, which removed an event-sync stall. The
host published the latest proposal's per-slot draft limits only once
`_limits_event.query()` reported the D2H copy complete. Until then it kept the
previous values.
- Under load the copy often hadn't landed. The operator counted 1,209 stale
  reads before the assert, with 0 alloc-ring reuses.
- A stale limit wider than the proposal makes the adaptive budget verify rows
  the proposal doesn't have: the scheduler's `-1` placeholders. Re-padding the
  copy kernel's drafts from `-1` to `0` didn't help, which places the bad rows
  in the scheduler's placeholders.
- A stale limit narrower than the proposal admits fewer of its drafts.

Why our switched TP3 didn't reproduce it is not yet measured. Our best lead:
the recipe serves with `SPARK_DS41F_DSPARK_DEAD_ROWS_TAU=0.2`
(`config/cluster.json`), while the code's default is 0, which is off
(`adaptive_verification.py`).
- With dead rows on and no batch sharder, the placeholder rows carry
  confidence 0. `combine_sampled_and_draft_tokens` marks them as padding and
  `route_partials` skips them.
- With dead rows off they are routed as live rows.
- The operator's environment list doesn't include the variable. #855's review
  asked them for it.

## The change

- `sync_slot_draft_limits` calls `_limits_event.synchronize()` when a copy is
  queued, then publishes the staged limits. The copy is small and was queued
  before the drafter ran.
- `add_request` sets the new slot's limit after that wait, so the list of resets
  to replay over a later copy (`_pending_limit_resets`) is gone.
- Tests (`tests/v1/spec_decode/test_copy_drafts.py`):
  - the host limits wait for a queued copy and are the staged ones;
  - a request added while a copy is in flight keeps its reset;
  - the stale-wide test keeps its placeholder and doubling checks. It drops
    0612's assertion that the sync never blocks, because that invariant is
    what this change gives up.

What it trades: the host now waits each step for the limit copy. The copy lands
once the previous step's target forward and sampling are done, so the host's
preparation of the next step overlaps less with the GPU. A fix that keeps
0612's no-wait design would mask rows past the device-side
`slot_draft_limits.gpu` inside the combine kernel. It is not this change: the
operator's fix is the one measured where #852 reproduces, and it also stops a
narrow stale limit from dropping drafts.

## Evidence

**Unit tests** (candidate-37 image on sp8, GPU, changed files mounted; the
operator's code was not run unreviewed: these are this experiment's files):

```text
tests/v1/spec_decode/test_copy_drafts.py tests/v1/core/test_copy_drafts_scheduler.py
tests/v1/spec_decode/test_adaptive_verification.py
  candidate-37   69 passed, 1 skipped
  this change    71 passed, 1 skipped   (the two new tests; none fails)
```

```bash
# sp8, 2026-11-02; W=/tmp/vllm-limit-sync (vLLM copy-limit-sync, 8bb508091), V=/opt/spark-ds41f/candidate/vllm
docker run --rm --gpus all --ipc=host -v /tmp/loop-test-plugins:/plugins:ro \
  [-v $W/vllm/v1/worker/gpu/spec_decode/copy_drafts.py:$V/vllm/v1/worker/gpu/spec_decode/copy_drafts.py:ro \
   -v $W/tests/v1/spec_decode/test_copy_drafts.py:$V/tests/v1/spec_decode/test_copy_drafts.py:ro]  # this change
  -e PYTHONPATH=/plugins -e HF_HUB_OFFLINE=1 -w $V --entrypoint python3 spark-ds41f:candidate-37 \
  -m pytest -q -p tblib_stub tests/v1/spec_decode/test_copy_drafts.py \
  tests/v1/core/test_copy_drafts_scheduler.py tests/v1/spec_decode/test_adaptive_verification.py -rf
```

(`tblib_stub` is the stub plugin of `docs/testing.md`.)

**The operator's ring** (#852 comments 6583970571 and 6584529514, #855; their
measurements, not repeated here):
- Image `spark-ds41f:candidate-37-af0fb152`, digest `sha256:2241b8f1ae65…`:
  the same vLLM tree, but their own build.
- Three nodes in a direct ring. SM clocks not locked: P0 at about 2.2 GHz.
  Kernel `7.0.0-1019-nvidia-64k` (64K pages), driver `580.178.04`.
- NCCL `NCCL_SOCKET_IFNAME=enP7s7`, `NCCL_MAX_NCHANNELS=8`, `NCCL_IB_DISABLE=0`,
  `NCCL_NET_PLUGIN=none`; also `CUDA_DEVICE_MAX_CONNECTIONS=32`,
  `VLLM_DS41_HEAD_DTYPE=float32`.
- #842's envelope (budget 6,155, capture 59, sizes +56,59), servebench suite
  sha `7c59b27e8503`.

| Run | Before (candidate-37) | With the fix |
|---|---|---|
| m=8: `copy_probe.py --prompts 3`, then the single-stream suite | device assert in the suite's warm-up, 2 of 2 boots | suite exit 0, no assert |
| copy probe tokens/step, m=8 (copy / edit / tool / prose) | 16.360 / 15.792 / 6.636 / 2.651 | 16.360 / 15.792 / **6.905** / 2.651 |
| copy probe tokens/step, m=16 | 16.014 / 15.315 / 6.083 / 2.651 | 16.014 / 15.315 / **6.232** / 2.651 |
| single stream, m=8 (tok/s) | 64.897 | 63.867 (−1.6%, CI95 2.77%) |
| single stream, m=16 (tok/s) | 64.683 | 63.58 (−1.7%, CI95 2.78%) |
| m=16 in their live gateway pool | engine died within minutes | healthy, no assert, from 17:36Z |

Single boots each. Their instrumentation for the stale-read count and their
`-1`→`0` control are not in this repository; #855's review asked for them.

**Our switched TP3** (sp5–sp7, window `spark-0013-1793639688`, 2026-11-02; record
in [experiments/2026-11-01-dspark-default-copy-drafts](../2026-11-01-dspark-default-copy-drafts/)):
#852's m=8 sequence on candidate-37 was clean on every arm:
- with and without 0676;
- with 8 streams;
- with clocks locked at 2200 MHz and 16 streams.

The census logged no non-finite gate row. All of it ran with the recipe's dead
rows at 0.2, which is the lead above.

## Measured on sp5–sp7

Window `spark-0013-1793654131`, queue log `results/private/lab/queue-856.log` on sp5;
image `sha256:e1ee7b1953b1…`; nodes at main `24c0e8f7`.

**[lab-repro.json](lab-repro.json), run `limit-sync-repro-sp`** (21:15 to 21:40Z):
dead rows off, #842's census, m=8, one boot each.

| Arm | copy / edit / tool tokens per step | single stream | 8 streams | census |
|---|---|---|---|---|
| candidate-37 | 16.367 / 15.745 / 6.905 | 57.85 tok/s | 153.00 tok/s | 0 non-finite, 0 asserts |
| this fix | 16.367 / 15.745 / 6.905 | 57.85 tok/s | 152.57 tok/s | 0 non-finite, 0 asserts |

**This run cannot show the race** (found by #859's session). The census overlay
reads a device value with `.item()` after the drafter every step. That waits for the
GPU, so the limit copy has always landed before the next budget read. The same holds
for every census arm of #842's window. These arms show only that a host whose
limits are current never verifies a placeholder row.

**[lab-cost.json](lab-cost.json), run `limit-sync-cost-sp`** (21:40 to 22:37Z): m=16,
the recipe's dead rows, no census, in the order fix, candidate-37, fix.

| Measure | candidate-37 | fix | fix, repeat boot |
|---|---|---|---|
| paired single-stream decode (verdict) | — | **−0.73% ±0.12%**, worse (floor 0.32%) | |
| single-stream window tok/s | 63.44 | 62.89 | 63.12 |
| TTFT, paired | 0.147 s | +0.97% ±0.75% | +1.06% ±0.43% |
| copy probe tool tokens/step | **6.083** | **6.232** | **6.232** |
| copy probe copy / edit | 16.021 / 15.322 | 16.021 / 15.322 | 16.021 / 15.322 |
| 8 distinct streams | 158.22 tok/s | 157.59 | 157.57 |
| 16 distinct streams | 162.34 tok/s | 163.34 | 163.76 |

The tool figures are the operator's m=16 numbers (6.083 to 6.232). Candidate-37's
host read stale, narrower limits often enough to drop drafts the proposal had, and
the fix's two boots agree. The single-stream suite's text is not copy-dense
(2.26 tokens per step), so it shows the wait's cost without its gain.

Why our cluster doesn't crash, still to be measured: the recipe's dead rows (0.2)
mark stale-wide placeholder rows as padding. The census-free repro of #859
(dead rows off, a host-only stale-read counter) is the run that tests it.

## Not done

- **The producer of the non-finite logits**: which kernel turns a routed `-1`
  placeholder row non-finite. The repro arm's census records it (layer, row,
  input id, origin).
- **The operator's answers**: their dead-row setting, their stale-read
  instrumentation and their control (#855's review).
- **#842**: its CopyDrafter rework changes the same code (the async copy on a
  side stream) and builds on this fix.
