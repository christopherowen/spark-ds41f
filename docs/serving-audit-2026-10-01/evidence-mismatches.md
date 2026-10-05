# Replay coverage versus the served operations

This is a source-and-artifact finding, not a new GPU result. It distinguishes **a standalone replay passing** from **the exact operation used in serving passing**. None of these mismatches by itself invalidates the later all-rank end-to-end traces. They do limit the claim that every served operation has been independently replayed with its exact operands and plan.

All source IDs resolve in [source-index.md](source-index.md). The [checkpoint header inspection](measurements/checkpoint-shapes.json) read only safetensors metadata, without loading weights or calling the GPU. Shapes below are N×K.

| Operation | Standalone replay | Frozen production serving | Consequence |
|---|---|---|---|
| Ratio-2 compressor | D06 concatenates value/gate weights and calls one 1024×5120 FP32-output GEMV | V02 `_project` executes two 512×5120 FP32-output GEMVs, one for each half | Different backend threshold: N1024 uses Q128; N512 uses Q256. Timing and rounding coverage of the concatenated call cannot simply be assigned to the served split calls. |
| Attention query-B | D05 `family_fp8` slices raw checkpoint output rows using its `rank0` helper. Its label says 11264×1280. Applying the frozen helper to the checkpoint's 32768 outputs yields **10944**, so the hardcoded label itself is unreliable. | V16 pads heads64→72 for TP3. V03 constructs a column-parallel 72×512 output projection, yielding **12288×1280** per rank. | Neither the label nor the helper geometry matches the served projection. A numerical group in that replay is not exact served query-B coverage. |
| Index query-B | D05 calls the same `rank0` helper on checkpoint4096×1280, yielding **1376×1280**. | V03 explicitly uses `ReplicatedLinear` so all32 index heads remain local: **4096×1280**. | The replay invents an output sharding that serving does not perform. Plan/grid selection can differ. |
| Old target-head replay | D05 `family_head` takes ceil(129280/3)=43094 raw rows and uses generic BF16 GEMV for capacities≤8, then `F.linear`. | V13 uses BF16 vocabulary projection with padded shard **43136×5120**: eligible one-row Triton, otherwise `F.linear`. | The old two-group result describes a proxy. Later `family_head_bf16` and patch0038 address the actual served vocabulary path and should be cited instead. |

The query-B replay helper's formula is `-(-N // 3 // 32) * 32`; evaluating it is pure integer arithmetic. The log text is not a substitute for printing actual packed matrix dimensions. Even where a replay has the right dimensions, checkpoint slicing may fail to reproduce padding, packing, rank ownership or scale layout. Prefer captured served tensors and plans.

## Evidence that remains useful

- The ratio-1 compressor replay explicitly tests a 512×5120 **BF16-output** projection. It is not interchangeable with either FP32-output ratio-2 half.
- Shared expert gate/up and down replay dimensions match the serving geometry documented in the main audit; this does not imply exhaustive numerical proof.
- The WO replay explicitly uses three groups and24 heads for TP3, unlike the FP8 query-B helper.
- The later BF16 vocabulary replay uses the appropriate wrapper and padded shard. It found the one-row problem and motivated two-row padding.
- The all-rank ref2 traces compare the running model's recorded rows and outputs. Their finite scenario coverage is stronger serving evidence than the mismatched standalone cases, while still leaving restart and longer-context questions open.

## Smallest correction to the evidence

1. Add the actual matrix shape, dtype, source revision, weight/scale fingerprint, rank ownership, prepared Q and selected plan to every replay record. Reject a label that disagrees with the tensors.
2. Replay `DeepseekCompressor._project` using its two weight halves and FP32 buffers, then compare compressor state/output as well as both intermediate projections.
3. Replay served query-B and replicated index-query-B using captured tensors and prepared plans. Do not hand-shard checkpoint tensors with a generic helper.
4. Keep the older records under their historical names and mark them as proxy results. Do not silently rewrite their labels or treat a new replay as if it had already passed.

This does not require a full serving benchmark matrix before correcting the harness. Focused exact-path replays should precede any new claim about those individual operations; use the next planned serving screen to verify the combined candidate.
