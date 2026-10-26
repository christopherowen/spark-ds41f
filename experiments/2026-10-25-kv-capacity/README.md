# KV capacity: what the pool holds, and what vLLM's figure charged

2026-10-25, CET. Read from vLLM `tilelang-ports` at `26382f5cf` (candidate 5's series head;
candidate 6 adds two test patches) and spark-ds41f `origin/main` `8ee6ce53`. No GPU or
cluster use: the figures come from the code, the recipe, the checkpoint's `config.json`
(`state/config.json`, the same geometry as 1017's `1bae0e10`) and the boots' saved log lines.

## Answer

- **The ring's 10.5 GiB pool holds 11 full 1M contexts at once at either budget**: 11.03 at
  16,384 batched tokens and 11.49 at 8,192. All 16 sequences fit at once up to 718,848 tokens
  each at 16,384 (749,568 at 8,192). The triangle's 4.90625 GiB holds 5 at either of its
  budgets: 5.12 at the 8,192 it serves since spark-ds41f-1025 (#602), 5.35 at 4,096. Its 4 KiB
  shape holds all eight sequences at its 262,144-token limit.
- **The published 6.21 and 8.18 are vLLM's startup figure**, not the capacity. It charges
  every request the window caches' worst case, a prompt's two whole chunks in flight, as if
  all requests prefilled at once. That is why doubling the budget took the figure from 8.18
  to 6.21. The budget's real cost is 1,920 blocks (0.46 of a context's 4,142), held only while
  a long prompt is being prefilled. The figure is informational: no scheduler decision reads
  it.
- **Two chunks in flight is reachable, but only for the prompt being filled.** With async
  scheduling a prompt's next chunk is scheduled while its previous chunk executes, and its
  window blocks are freed only once that chunk settles. Across all requests the unsettled
  tokens are at most the two batches, so the pool needs that allowance once.
- **Recommendation: leave vLLM's admission bound as it is.** The recipe already turns off
  the only admission check that uses it (`--no-scheduler-reserve-full-isl`). If it were on,
  it charges the admitted request its own two-chunk peak, which that request reaches, and
  compares it with the blocks actually free; it too admits the 11th 1M context at 16,384. The
  one tightening left is 257 blocks a window group instead of 258 (the long-prefill
  threshold, 16,288, in place of the budget), worth 0.004 of a context. Fix the figure
  instead: vLLM patch 0491 makes the startup line charge the batches once (11.02x and 11.48x
  on the ring), and the docs and tooling now report the capacity.

## The pool

vLLM packs DeepSeek V4.1 Flash's caches block-outermost
(`_get_packed_kv_cache_groups`, `vllm/v1/core/kv_cache_utils.py:2303`): one pool of
equal blocks, each the size of the widest group, from which every group draws whole blocks.
The caches (`vllm/models/deepseek_v4_1/attention.py:280` and `:631`,
`compressor.py:108`) and their groups:

| Group | Layers | Tokens a block | Bytes | Groups | Blocks a request at 1M |
| --- | --- | ---: | ---: | ---: | ---: |
| Compressed | latents (288 B a state) and indexer keys (68 B) of the KV sources 2, 8, 14 (ratio 2) and 20 (ratio 1) | 256 | 3 x 128 x 356 + 256 x 356 = 227,840 | 1 | 4,096 |
| Window | every attention layer's 128-token window (528 B a token): 40 target, 3 drafter | 128 | 3 x 67,584 = 202,752 | 15 | 3 resident, 257-258 with a prompt's chunks |
| Compressor states | the ratio-2 compressors' open groups, 8 FP32 slots of 4 KiB | 8 | 3 x 32,768 | 1 | 1 |

The compressed group is the widest, so a block is 227,840 bytes and 10.5 GiB is 49,483
blocks (one is vLLM's null block). A block holds three window pages, and the window caches
split by prefix-cache policy: the 20 layers before the CED decoder (7 groups) and the 20
from it plus the drafter's 3 (8 groups). The 2026-10-05 TP4 boot logged exactly these:
`kv cache group sizes [128 x 7, 256, 8, 128 x 8]` and 49,483 serving blocks
(`results/private/tp4-500k/serving-b12x-dgx1.log`, git-ignored, in the main checkout).

## What vLLM reported

The startup line (`update_kv_cache_capacity`, `kv_cache_utils.py:2845`) prints
`concurrency x max_model_len` as "GPU KV cache size". Concurrency is the pool's blocks over
one request's blocks at `max_model_len` (`get_max_concurrency_for_kv_cache_config`, `:1116`),
and a window group's share is its `max_memory_usage_bytes`
(`vllm/v1/kv_cache_interface.py:895`): the window minus one, plus `max_in_flight_tokens`, plus
one block for alignment (`:867`). `max_in_flight_tokens` is `max_concurrent_batches x
max_num_batched_tokens` (`vllm/config/vllm.py:746`), two batches under async scheduling.

| Budget | Window group | Request | Figure | Logged |
| --- | ---: | ---: | ---: | ---: |
| 8,192 | cdiv(127 + 16,384, 128) + 1 = 130 | 4,096 + 15 x 130 + 1 = 6,047 | 49,483 / 6,047 = 8.183 | 8,580,566 |
| 16,384 | cdiv(127 + 32,768, 128) + 1 = 258 | 4,096 + 15 x 258 + 1 = 7,967 | 49,483 / 7,967 = 6.211 | 6,512,700 |

The same arithmetic gives every capacity figure the records quote, to the token: the
triangle's 4,765,898 (4.55x) at 4,096 and spark-ds41f-1025's 4,009,281 (3.82x) at 8,192, the
4 KiB shape's 1,348,708 (5.14x) and r6v's 2,845,543 at 512K (5.43x).
`tests/test_kv_capacity.py` pins all six.

## What a request holds

A window group holds, for each request, the blocks covering one contiguous span of tokens
(`allocate_slots`, `vllm/v1/core/kv_cache_manager.py:670`): the 127 tokens kept below the
processed boundary (128 for the drafter's layers, which retain one more), the tokens
scheduled but not yet settled, and the drafter's 5 lookahead slots. The processed boundary
is `num_computed_tokens - num_in_flight_tokens`: the step in flight still reads the window of
its own first token. A span of n tokens touches at most cdiv(n, 128) + 1 blocks.

- **Decoding**, as every request at the limit is: the step in flight and the one being
  scheduled carry 6 tokens each (1 + 5 drafts), so the span is 128 + 12 + 5 = 145 tokens:
  three blocks a group. A request at 1M holds 4,096 + 15 x 3 + 1 = **4,142 blocks** (about
  0.88 GiB a rank).
- **The prompt being filled**: its chunk in flight and its next chunk, each at most the
  long-prefill threshold (budget minus 16 x 6 decode rows: 16,288 at 16,384, 8,096 at 8,192),
  so the span is 128 + 2 x 16,288 + 5 tokens: 257 blocks a group (129 at 8,192), **3,810 more
  blocks** than a decoding request (1,890 at 8,192).

The full-length contexts the pool holds at once are therefore (usable blocks - the one
prompt's extra) / 4,142:

| Shape and budget | Usable blocks | Prompt in flight | Contexts at once | All sequences at once |
| --- | ---: | ---: | ---: | ---: |
| `tp4-ring`, 16,384 | 49,482 | 3,810 | 11.03 → **11** | 16 x 718,848 tokens |
| `tp4-ring`, 8,192 (through spark-ds41f-1022) | 49,482 | 1,890 | 11.49 → **11** | 16 x 749,568 tokens |
| `tp3-triangle`, 8,192 (since spark-ds41f-1025) | 23,120 | 1,905 | 5.12 → **5** | 8 x 666,880 tokens |
| `tp3-triangle`, 4,096 (through spark-ds41f-1022) | 23,120 | 945 | 5.35 → **5** | 8 x 697,600 tokens |
| `tp3-triangle-4k`, 4,096, 262,144 tokens | 10,366 | 945 | 8.80 → 8 sequences | 8 x 262,144 tokens |
| r6v TP3, 3.5 GiB, 524,288 tokens | 16,493 | 945 | 7.43 → **7** | |

At 16,384 the eleventh context leaves 110 blocks spare while it is filled (10 x 4,142 + its
own 4,096 + 3,810 + 46 = 49,372 of 49,482).

Retention does not add to this: `--prefix-cache-retention-interval` chooses which freed
window blocks stay registered for prefix hits, but freed blocks return to the pool either
way (`_remove_blocks_in_range`), and evictable cached blocks count as free.

## Are two chunks in flight reachable?

Yes, for one request at a time. The async scheduler advances `num_computed_tokens` when it
schedules a chunk, so it schedules a prompt's next chunk while the previous one executes
(late admission decides that successor at once). At that allocation the request holds the
window below its previous chunk, the previous chunk and the new one. vLLM's own
`test_swa_free_waits_for_in_flight_step` checks that those blocks are not freed early, and
this experiment's scheduler test (`test_swa_capacity_bounds_blocks_held_under_async_chunked_prefill`)
sees a lone prompt reach exactly that peak (11 blocks of 16 tokens: its 31-token window and
two 64-token chunks). So the per-request peak is right for the startup check (one request
must fit, `kv_cache_utils.py:3201`) and for the full-sequence admission gate (the admitted
request will prefill).

It is not reachable for every request at once. The scheduled but unsettled tokens of all
requests together fill at most the two batches (the step in flight and the one being
scheduled, each at most `max_num_batched_tokens`). A request at the limit has no prompt
left; only the prompt being filled carries chunks.

## The tighter bound (vLLM 0491)

[vllm/0491-KV-Capacity-figure-the-sliding-window-per-request-th.patch](vllm/0491-KV-Capacity-figure-the-sliding-window-per-request-th.patch),
branch `kv-capacity` of `~/projects/vllm-ds41-tilelang-migration`, commit `8bde1c48b` on
`26382f5cf` (worktree `~/projects/vllm-kv-capacity`, local only like the other vLLM branches).
It applies cleanly on candidate 6's series (0489-0490 touch two unrelated test files);
[vllm/series](vllm/series) is candidate 6's with 0491.

- `KVCacheSpec.max_capacity_blocks` splits a group's capacity blocks into a part every
  request holds and a part held once. By default a request holds its whole maximum usage
  (no change for full attention, Mamba, compressor rings, the kpool tail).
- `SlidingWindowSpec`: a span of n tokens touches at most (n - 2) / block + 2 blocks, so each
  request pays (kept + lookahead - 2) / 128 + 2 = 3.02 blocks a group and the pool pays
  `max_in_flight_tokens` / 128 once (256 a group at 16,384). This holds for any split of the
  unsettled tokens among requests, while every running request is allocated each step: one
  the scheduler skips keeps its last allocation's blocks. On this recipe every running
  request is scheduled every step (the 96 decode rows fit beside a chunk at the threshold, one
  prefill lane, no prefill deferral); the figure is a capacity estimate, and nothing
  depends on it.
- `get_max_concurrency_for_kv_cache_config` takes the larger of this concurrency and the
  per-request-peak one, so a pool that barely fits one request reports no less than before.
- Unchanged: `max_admission_blocks_per_request` (one request's own peak), the startup check,
  the full-sequence gate, pool sizing.

The ring's startup line becomes 11,554,122 tokens, 11.02x at 16,384 (12,040,153, 11.48x at
8,192); the triangle's 5,366,846, 5.12x at its 8,192 (5,609,861, 5.35x at 4,096). These
are with 0492 before 0491, the next
candidate's order: 0492 makes the three window groups holding a drafter layer retain one
more token ([drafter window retention](../2026-10-25-drafter-window-retention/README.md)),
which costs 65 tokens of the figure (0491 alone: 11,554,187). vLLM's own grouping and
capacity code on 0492, 0500-0503 and 0491 over the next candidate's base (e0ccc837d) gives
these numbers, and `scripts/kv_capacity.py` reproduces them. The figure sits 0.01-0.02 below
the exact model above, which counts whole blocks.

Tests, in `tests/v1/core/test_kv_cache_utils.py` (a gate file of `config/release-tests.json`,
CPU tests, no model download):

- `test_ds41_ring_capacity_charges_unsettled_tokens_once`: the ring's groups at both budgets,
  the old figure (8.18, 6.21) from the per-request peak, the unchanged single-request check,
  the new figure (11.48, 11.02), the token count and the scheduler-config shape;
- `test_kpool_tail_capacity_stays_one_block`;
- `test_swa_capacity_bounds_blocks_held_under_async_chunked_prefill`: four 300-token prompts
  through the async scheduler with two steps in flight; at every step the pool's real use
  stays within the new bound (1.6 blocks spare at the tightest step), a lone prompt reaches
  its own peak, and all four run at once;
- `test_get_max_concurrency_for_kv_cache_config`'s window cases now expect the new figure
  (they pinned 3 where the bound gives 4.91 and 3.12), and the Kimi namespace config gains
  `num_lookahead_tokens`.

All ran on CPU on a Mac ([runs/tests-v1-core.txt](runs/tests-v1-core.txt),
[runs/run-tests.sh](runs/run-tests.sh)): `tests/v1/core` on both trees fails the same 106
ids and errors the same 2 (hub ids not in the offline cache, the CPU build's missing
modules), and the branch passes four more (1,399 against 1,395; that run was `44396b03b`, whose
scheduler test bounded the lone prompt's blocks where `8bde1c48b` asserts the exact peak).
`test_get_max_concurrency_for_kv_cache_config` builds `Qwen/Qwen1.5-7B`'s config, which the
lab's offline nodes skip; run through a cached model's config it passes on both trees with
each tree's expectations. In the lab the gate file runs in the next candidate's release tests.

## Earlier records

- **`mem_snapshot.py` read a stopped container.** It named the container
  `dsv41-karmic-kraken` (a constant), so after the rename every arm's snapshot read the
  stopped production container's log: candidate 4's 8.18 came from there. It also kept the
  last six matching lines, and the periodic "GPU KV cache usage" lines match its pattern, so
  after traffic the startup figure can drop out of the excerpt. It now reads the running serving container (`spark-ds41f` or `dsv41-*`),
  names it and its image in each JSON line, and takes the startup lines.
- **Every "full contexts" figure in the docs was vLLM's.** `docs/memory-profiles.md` (now a
  [KV capacity](../../docs/memory-profiles.md#kv-capacity) section), `docs/current-state.md`
  and the README now give the capacity and name vLLM's line beside it.
  `scripts/kv_capacity.py` prints both for any recipe, shape or budget:

  ```sh
  .venv/bin/python scripts/kv_capacity.py --shape tp4-ring --max-num-batched-tokens 16384
  .venv/bin/python scripts/kv_capacity.py --cluster-config experiments/2026-10-25-spark-ds41f-candidate/cluster.json --json
  ```

- **spark-ds41f-1025's published records** keep the startup figure: its release notes' KV
  capacity row names it as the startup figure beside this record's corrected count, while
  their "The ring admits 6.21 full 1M contexts" sentences and the baseline's
  `kv_cache_tokens` note stay as published (baselines are immutable). `docs/current-state.md`
  now says what 1025's startup line reads and what the pool holds.

## Follow-ups (not done here)

- **The drafter's retained window token was dropped** in the three window groups that hold
  a drafter layer behind target layers: checked and fixed in
  [experiments/2026-10-25-drafter-window-retention](../2026-10-25-drafter-window-retention/README.md)
  (vLLM 0492, independent of 0491).
- Chunked-local attention (Llama 4) has the same per-request charge in the figure; this
  deployment has none, so 0491 leaves it alone.
