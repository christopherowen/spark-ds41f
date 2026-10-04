# TP4 end-to-end validation: TileLang and B12X

Before the default kernel family changes, both TP4 candidates run the full
validation and benchmark protocol in the same windows. TP3 follows.

| Family | Config | Image | DSpark cost directory |
| --- | --- | --- | --- |
| TileLang | [candidate.json](../2026-10-04-tilelang-1m/candidate.json) | `-tilelang-1m-v3` | `ring4-tilelang-1m-v3-20261005` |
| B12X | [b12x.json](b12x.json) | `-tp4-1m-v1` | `ring4-tp4-1m-20261005` |

Both run the [TP4 1M recipe](../2026-10-04-tp4-memory-tuning/README.md):
1,048,576-token context, 16 sequences, 8,192-token batches, CUDA graphs to
96 rows, 10.5 GiB of KV per rank, prefix-cache retention every 8,192 tokens,
vocabulary weights in the display carve-out, and the fabricated-context fix
(vLLM 0039).

- [b12x.json](b12x.json) is the recipe with launch enabled for lab windows.
- The TileLang config is the candidate itself; it adds the TileLang kernels,
  TileKernels routing and mHC with the register fold (0040), TileLang
  vocabulary heads and sparknet collectives
  ([README](../2026-10-04-tilelang-1m/README.md)).
- Each family profiles its own DSpark cost curves from its own kernels, as it
  would deploy: the B12X recipe's earlier curves came from the
  fabricated-context profile that 0039 fixes, and the TileLang curves from
  the kernels before 0040.

## Protocol

The B12X image is built and its [heads](bundles/b12x-heads/candidate.json),
[carve-out](bundles/b12x-carveout/candidate.json) and
[fabricated-context](bundles/b12x-dummy-context/candidate.json) bundles run
first. The TileLang tests bundle already passes on `-tilelang-1m-v3` (99).

**Window 1**, one boot per family, B12X first:

- `bench --full`: every suite (quality, decode sampled to the default
  precision, sampled, prefill, prefix, admission) at 1, 2, 4, 8 and 16
  streams, against the promoted baseline;
- [long_context.py](../2026-10-04-tp4-memory-tuning/long_context.py):
  ~960K-token real-text prompts with a code word at 10% and 90% depth;
- [prefix_cache.py](../2026-10-04-tp4-memory-tuning/prefix_cache.py): the
  prompt-cache probe at ~200K tokens;
- per-second MemAvailable on every node (the 10.5 GiB pool qualifies only
  with at least 8 GiB left on every node), a decode profile, and the boot
  log checked for silent fallbacks and for JIT compilation after readiness.

**Window 2**, a second boot per family, the lean screen (quality and decode,
prose and code with and without reasoning, 1 and 8 streams, three samples),
to bound boot-to-boot noise.
