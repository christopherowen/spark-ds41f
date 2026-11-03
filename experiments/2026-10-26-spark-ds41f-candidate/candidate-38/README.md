# candidate-38

Base: `candidate-37` (`0001..0676`, vLLM patch head `5b7cfa37f`, tree `af0fb152`; `sparknet` `1fd6a1f`).

Added patches (`0677..0681`). Each is the `git format-patch` commit its pull request made on
`5b7cfa37f`, taken from the pull request's branch byte for byte apart from the re-sequenced
`[PATCH n/681]` subject prefix (`make_series.py`), and applied in this order:

- `0677-fix-copy-drafts-budget-verification-from-what-every-.patch` (`#860`, vLLM `72abb31ff`):
  the fix for #852. The adaptive verification budget no longer reads CopyDrafter's per-rank host
  mirror of the slot draft limits (one argument in `model_runner.py`), so every rank budgets the
  same step from inputs the ranks hold alike. Measured on sp5–sp7 with the patch's file mounted
  over candidate-37's image: #852 reproduced on candidate-37 (`route_partials: gate logits
  contain nan or inf` in the single-stream suite after the copy probe, min match 8, dead rows
  off) and the fix ran the same sequence clean with the ranks' layout CRCs equal; paired
  single-stream −0.08% ±0.09% against candidate-37 and tool tokens per step 6.232 against 6.083
  (`lab-cost.json`'s run, as the session that queued it reports it; the experiment's README at
  `0c9ae9fa` still lists that run as queued). Unit tests in the image: 79 passed, 1 skipped.
- `0678-cleanup-copy-drafts-remove-the-per-slot-draft-limits.patch` (`#860`, vLLM `88982f34e`):
  removes what 0677 left unread: the host mirror, its staging ring and event,
  `sync_slot_draft_limits`, the limits' copy in `attach_copy_draft_alloc`, the device buffer and
  the manager's `slot_draft_limits` argument. Unit-tested only (the same 79 passed, 1 skipped
  with both patches); not measured and in no arm of #860. Included as a harmless architectural
  cleanup, under the owner's rule.
- `0679-feat-v1-act-on-a-loop-by-where-it-is-and-follow-the-.patch` (`#848`, vLLM `050e8b1cc`, v2):
  `check_stop` acts on a detected loop by where it is. Inside reasoning it ends the reasoning (a
  forced end that has not landed within 64 tokens stops the request); in the answer after a
  reasoning loop it stops at once; otherwise token soup (one token over 60% of the last 512)
  stops after 256 tokens and any other loop only after 2,048 tokens with fewer than one distinct
  token in ten. Markers follow `DeepSeekV4Parser`. Measured: 40 unit tests and the offline replays
  of the known triggers on candidate-37's image (sp8); the sp5–sp7 probe soak passed 15 of 15
  (v1 had cut two legitimate periodic answers, which v2's second horizon lets run).
- `0680-perf-dspark-capture-context-graphs-at-the-target-s-d.patch` (`#853`, vLLM `5340c9e2f`):
  `DSparkContextCudaGraphs` captures the row counts of the target's own FULL decode graphs (23
  graphs at TP3 instead of 7, 35 at TP4 instead of 8), so a full-depth decode step pads the
  context projection no further than the target does. Measured on one GB10 (dgx4), L2 flushed
  before each replay: −0.76 ±0.59 µs (TP3) and −1.65 ±0.49 µs (TP4) per context replay, at most
  −7.1 µs (TP4 c11); outputs bit-identical at every row count, no new TileLang kernels, +5–8 MiB
  driver memory per rank, +6/+10 ms capture; a 3,000-step poisoned-buffer soak per shape (sp8)
  matched bit for bit. The RoCE all_gather bytes it saves were not measured.
- `0681-perf-dspark-reduce-vocabulary-parallel-greedy-drafts.patch` (`#854`, vLLM `3b7884972`):
  the vocabulary-parallel greedy draft steps (TP > 1) reduce base logits + Markov bias in the
  projection's epilogue (the fused kernels 0671 already compiles) instead of materializing the
  bias shard; top-2 is reproduced bit for bit; no fallback (kernel preparation raises for a
  Markov head that does not fuse). Measured on sp8 with the patch's files mounted over
  candidate-37's image: 234 tests passed with the PDL sources plugin (0 violations); the
  CUDA-graph-captured draft step (`chain`, cold) is −2.0 to −12.7 µs at TP3 from 2 requests
  and −2.9 to −16.4 µs at TP4 from 3, +0.4–0.6 µs per position at one request on TP3; a
  9,000-step functional soak under the draft workspace lane had 0 mismatches. No serving soak
  with real traffic has run.

#853 and #854 both change `vllm/models/deepseek_v4_1/nvidia/dspark.py`; on `5b7cfa37f` they
apply in either order and give the same tree (#853's record). Here #854 follows #853, and the
whole series applies to a fresh `19f2c20ed` tree without a conflict.

Not included: `#842` (CopyDrafter on by default; it needs rebasing onto #860 and its own
qualification, and is staged for candidate-39), `#856` and `#859` (superseded by #860).

vLLM tree: `37b34b8f9e365cf9194b198008abf650581936a0` (patch head
`753354cf4ed0970f31ba4ef8c754a1c80a55e4e5`; 566 patches; patch set `49f728a8…`). `git diff
--check` from `5b7cfa37f` is clean. The intermediate trees are 0677 `58882ab9`, 0678
`f473423c`, 0679 `8542285f`, 0680 `77b534d1`, 0681 `37b34b8f`.

## Arms

`tp3.json` and `tp4.json` extend rc14's `rc-tp3.json` and `rc-tp4.json` (spark-ds41f-1030's
frozen recipes) with the image, its expected trees and this folder's lock, as candidate-37's
do; `tp3-dev.json`, `tp3-copy.json`, `tp3-copy-dev.json` and `tp3-8192.json` are
candidate-37's with the name changed. `candidate-tp3.json`, `release-tp3.json` and
`screen-tp3.json` are candidate-37's gates with the labels renamed. `image-dgx4.json` is the
build profile. Qualification is queued by the parent session; nothing here was started on a
serving node.

## Image

Built on dgx4 from this commit (`bin/spark --cluster-config
experiments/2026-10-26-spark-ds41f-candidate/candidate-38/image-dgx4.json build prepare`,
`build check`, `build image --apply`), tag `spark-ds41f:candidate-38`. Not copied anywhere.

- Fresh tree: pending.
- Image ID: pending.

## Release tests

Pending.

## Unit evidence

Pending.
