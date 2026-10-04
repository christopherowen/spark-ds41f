# Vocabulary weights streamed into the display carve-out

Base deployment commit: `3525568` (the [packed BF16 head](../2026-10-03-packed-bf16-head/README.md)
v3 candidate, packed in place, on the [native drafter heads](../2026-10-03-native-drafter-heads/README.md) change).

The display carve-out (vLLM patch 0021) holds the embedding and output head
in the firmware's scanout reserve. Until now each rank loaded both into
ordinary memory, copied them into the carve-out after loading and freed the
originals. The DSpark drafter also built its own full-size embedding and head,
which the target's replace after loading. Every packed-head boot logged
1,037–1,568 MiB of ordinary memory freed per rank at that point.

On GB10, freed GPU memory stays charged to the process: of each GiB freed,
355–753 MiB stays counted as used, and only the same process can reuse it
(found while packing the head in place). So the copy and the dropped
placeholders could cost host memory for the life of the server, while the
weights they served from were already in the carve-out. The window below
measures how much they did.

## Change

vLLM patch [0030](vllm/0030-worker-display-carveout-streamed-weights.patch), on 0028:

- With `SPARK3_DISPLAY_CARVEOUT_WEIGHTS=1`, a vocabulary-sized weight (64 MiB
  or more) is created in its own carve-out buffer when the model is built:
  the unquantized embedding's and the packed head's `create_weights` take their
  storage from `display_carveout.allocate`. The checkpoint loader writes straight
  into it, and the packed head is then packed where it lies. The head's buffer
  keeps the BF16 size, so its last quarter stays unused: 79 MiB of carve-out
  space per rank, not host memory.
- The DSpark drafter builds its placeholder embedding and head on the meta
  device. The checkpoint has no draft embedding or head, and both are replaced
  by the target's after loading.
- After loading, the worker no longer copies anything. It confirms that every
  vocabulary-sized weight of the served model is in a carve-out buffer and that
  every buffer still holds one, and refuses to start otherwise. A layer built
  in the carve-out and then dropped would strand carve-out space, so it is an
  error, not a warning. The debug integrity check baselines the weights where
  they loaded.

`scripts/lab.py` kernel-lab bundles gain `"display_carveout": true`, which
gives the bundle the serving container's DRM access (the card by PCI path as
`/dev/dri/card0`, device rule `c 226:* rw`). Without it the carve-out tests
skip.

The arm shares the packed head's DSpark cost directory: its shapes, kernels
and runtime weight placement are the packed head's, so both arms boot the same
way.

## Procedure

1. Build `-r5o-roce-contract-streamed-v1` and run the
   [carve-out tests](bundles/carveout-tests/candidate.json) and the packed
   head's [logits-head tests](bundles/vllm-tests/candidate.json) on one node.
2. One window, alternating the [packed-v3 arm](../2026-10-03-packed-bf16-head/packed-v3.json)
   and [streamed](streamed.json) twice each. Each arm runs `cluster start`, then
   `bin/spark3 bench` decode (prose and code at 1 and 8 streams, three
   samples), against the first control. Record the boot's carve-out log line,
   MemAvailable once a second on every node, and `free -m` at the end.

## Results

Both bundles passed on dgx4 with the new image. The carve-out tests passed
14 of 14 with none skipped; the in-place load test saw the two 64 MiB tables
take no ordinary memory. The logits-head tests passed 2 of 2.

Window 2026-10-04 08:57–09:15 UTC, four boots alternating control and
streamed. Every streamed rank logged both weights "in 2 buffers, loaded in
place". The control ranks freed 941–1,727 MiB of ordinary memory at that point.

| Node | Used at the end, control (MiB) | Used at the end, streamed (MiB) | Change | Lowest MemAvailable, control → streamed (GiB) |
| --- | --- | --- | ---: | --- |
| dgx1 | 95,082 / 95,030 | 94,924 / 94,901 | −143 | 30.39–30.41 → 30.58–30.59 |
| dgx2 | 93,938 / 93,952 | 93,851 / 93,843 | −98 | 31.55–31.57 → 31.76–31.95 |
| dgx3 | 93,928 / 93,909 | 93,817 / 93,841 | −90 | 31.81–31.85 → 31.97–31.98 |
| dgx4 | 94,547 / 94,527 | 94,492 / 94,491 | −45 | 31.43–31.46 → 31.49–31.50 |

Repeat boots of one arm agree within 52 MiB, so the saving is real: 45–143 MiB
of host memory per node, with the floors 0.05–0.3 GiB higher. Decode is level
in both pairs (every point within its interval, against the first control),
and quality passed 5/5 in all four boots.

The saving is a tenth of what was freed, not the third to three quarters the
packed-head stage test predicted. The serving process reuses the memory it
freed at load for its KV cache, graph capture and preparation, so most of the
old copy's cost was recovered later in the same process. What remains is
the copy's share that nothing reused. The change stays: it moves no weights
after loading, never allocates the drafter's dropped tables, and costs no speed.
