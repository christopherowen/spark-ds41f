# The drafter's window token in shared window groups

2026-10-25, CET. Read from vLLM `tilelang-ports` at `26382f5cf` (candidate 5's series head;
candidate 6 adds two test patches). Found while deriving the KV pool's groups
([experiments/2026-10-25-kv-capacity](../2026-10-25-kv-capacity/README.md)). No GPU or
cluster use.

## Verdict

**A real bug, now fixed in vLLM 0492.** Three of the 15 window groups each hold one of
DSpark's drafter layers behind target layers. The scheduler managed them by the first
layer's spec, so it kept 127 tokens below the settled boundary where the drafter's layers
declare 128. When a step's rows are all rejected, the drafter's block starts at that step's
first position and reads 128 tokens back, so at one position in 128 its oldest token could
sit in a block the scheduler had already freed. Outputs keep the target's distribution
(every draft is verified) and temperature-0 outputs cannot change; what can change is one of
the drafter's 128 context positions, read through a stale block id. How often this happens
on the cluster is not measured.

## How the drafter's layers lose the token

- **The specs.** `vllm/models/deepseek_v4_1/attention.py:280` gives every attention layer a
  128-token window cache. The drafter's three layers (40-42) set `extra_retained_tokens=1`;
  target layers set 0. The drafter's and the CED decoder's windows (layers 20-39) both
  replay a bounded prefix instead of caching it (`prefix_cache_enabled=False`,
  `prefill_replay_window=128`).
- **The groups.** Block-outermost packing (`_get_packed_kv_cache_groups`) buckets layers
  whose specs are uniform. `SlidingWindowMLASpec.is_uniform_with_collection` compares the
  window, DCP, decode mode and prefix policy, not `extra_retained_tokens`, so the 20 decoder
  windows and the 3 drafter windows share one bucket. Split round-robin into 8 groups of
  three, it puts layer 40 behind 24 and 32, 41 behind 25 and 33, 42 behind 26 and 34. The
  boots log the same 17 groups (`[128 x 7, 256, 8, 128 x 8]`).
- **The scheduler's spec.** `generate_scheduler_kv_cache_config` replaced each
  `UniformTypeKVCacheSpecs` group with its first layer's spec, and `SlidingWindowManager`
  takes `extra_retained_tokens` from it. Those three groups' managers retained 0.

## What the drafter reads

- DSpark's block metadata (`sparse_mla.py:_chunk`, and the TileLang port
  `tilelang/metadata.py:chunk`, both on their `DRAFT` path) starts the window at
  `prefix - window`, where `prefix` is the block's first query position (`seq_len -
  query_len`). A target row starts at `pos + 1 - window`. The drafter therefore reads 128
  context tokens back from its block, one more than a target row reads back from its own
  position.
- The block's first position is one past the last valid row of the step
  (`_prepare_dflash_inputs_kernel`, `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`).
  Usually at least one row is valid, so the block starts at `q0 + 1` or later and the
  drafter reads back only as far as the target. When every row is rejected (sink rejection
  replacing draft zero, the kernel's "No valid row" case), the block starts at the step's
  first position `q0` and reads from `q0 - 128`.
- The scheduler frees on the settled boundary (`allocate_slots`: `num_computed_tokens -
  num_in_flight_tokens`). Settled on `q0` with 127 retained, it frees every block below
  `q0 - 127`. At `q0 = 127 (mod 128)`, `q0 - 128` is the last token of the block just
  freed. The worker's block tables are append-only (`BlockTables.append_block_ids`), so
  the drafter's page table still names that block. The pool may already have handed it to
  another request, whose layers write it before the drafter runs.

## The fix (vLLM 0492)

[vllm/0492-KV-Scheduler-window-groups-retain-what-their-most-de.patch](vllm/0492-KV-Scheduler-window-groups-retain-what-their-most-de.patch),
branch `drafter-window-retention` of `~/projects/vllm-ds41-tilelang-migration`, commit
`5c5a204bf` on `26382f5cf` (worktree `~/projects/vllm-drafter-window`, local only like the
other vLLM branches). It applies on candidate 6's series alone and after 0491 (the
kv-capacity patch); [vllm/series](vllm/series) is candidate 6's with 0492.

- `UniformTypeKVCacheSpecs.scheduler_spec`: the first layer's spec, with the group's
  largest `extra_retained_tokens` for a window group. One block table serves every layer
  of a group, so it keeps what the most demanding layer reads.
- `generate_scheduler_kv_cache_config` uses it.
- The physical layout and grouping are unchanged. The three groups keep one more token:
  a decoding request still holds three blocks a group, and one request's peak is still
  258 blocks a group at 16,384 batched tokens (130 at 8,192), so capacity is unchanged.
  These groups keep no prefix-cache entries, so caching is unaffected.

Not taken: comparing `extra_retained_tokens` in the uniform-type check, which would give
the drafter's windows their own group (7 + 7 + 1 groups, still 15). It fixes the same thing
but changes the cache layout the release qualified.

## Tests

In `tests/v1/core/test_kv_cache_utils.py` (a gate file of `config/release-tests.json`; CPU
tests, no model download), [runs/tests.txt](runs/tests.txt):

- `test_ds41_scheduler_window_groups_retain_the_drafters_token`: builds DS4.1's 43 window
  caches, compressed caches and rings in registration order and groups them through
  `get_kv_cache_groups`. It checks the logged layout, the three shared groups, and that each
  window group's scheduler spec retains its layers' largest count. Before the fix:
  `assert 0 == 1` on the group of layers 24, 32 and 40.
- `test_ds41_drafter_window_survives_a_step_whose_rows_are_all_rejected`: one of those
  groups' managers, settled at `q0 = 511`. Before the fix it freed the block holding
  position 383; with the fix it keeps it and frees the two below.
- `tests/v1/core` on both trees: the same failing ids (hub ids not in the offline cache, the
  CPU build's missing modules), two more passes with the fix.
  `test_get_kv_cache_capacity_after_scheduler_unwrap`, which exercises the changed function
  and builds an uncached model's config, passes through a cached model's config on both
  trees.

## Not done

- No GPU run: neither the hazard's frequency on real traffic nor its effect on draft
  acceptance is measured. It needs a sink-rejected draft zero at one position in 128, so
  it is rare, likely too rare for a lab A/B of acceptance to resolve.
- The lab's release tests run the gate file once 0492 is in a candidate's series.
