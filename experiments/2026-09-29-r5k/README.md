# r5k candidate: 256K context, display-reserve weights, and quality fixes

**Base:** r5j (`2026-09-28-karmic-kraken-r5j`). **Reference for this bundle:**
the `weights-256k` arm of `experiments/2026-09-29-display-carveout-kv` (r5j
image with patch 0021 mounted, same serving configuration).

**Changes:**

| Change | Source | Expected effect |
|---|---|---|
| Embedding and output head in the display carve-out; KV 2.2 GiB; 256K limit | vLLM 0021, configuration | capacity (measured in the display experiment) |
| DRM file closed after the carve-out import | vLLM 0021 | text console keeps drawing |
| Decode page metadata shared across CUDA graphs | vLLM 0022 (LIL `ce4be0a112`) | graph memory, bit-exact |
| BF16 sparse-attention arithmetic | vLLM 0023 (LIL `40371c7bb0`, attention half) | quality toward the reference; speed unknown at TP3 |
| Ratio-2 compressor ring written during decode | vLLM 0024 | quality: correct compressed entries for generated tokens |
| FP4 KV writer matches the reference quantizer | B12X `f8069b2c` (#435) | quality toward the reference |

The owner's bar: promote if the bundle comes out ahead, with quality fixes
allowed to cost some speed.

## Method

1. `make_arms.py` writes `cluster-candidate.json` (image
   `vllm-ds41f-kkref:04c30fa98e79-r5k`, vLLM tree `9ba14ba1`, B12X tree
   `640c8544`) and `cluster-candidate-auto.json`, the same with
   `VLLM_DS41_ATTENTION_COMPUTE=auto` to isolate 0023's speed cost.
2. `build.sh` builds the image from the lock and loads it on every node.
3. `run_arm.sh ARM LABEL [bench options]` runs the display experiment's
   protocol: quality gate, decode for prose, code and their answer-only cases
   at 1 and 8 streams (3 samples), real-text prefill, and concurrent long
   contexts. The candidate also runs prefill to 200K, four 180K contexts,
   `needle.py` at about 180K tokens, and `consistency.py`.
4. `consistency.py` generates 384 tokens greedily for ten prompts, then scores
   the same token ids as a prompt with a fresh cache salt, and reports the
   decode-versus-prefill logprob gap and argmax disagreement. On the reference
   arm (patch 0024 absent) the gap was 0.0612 mean, 0.3046 p95, 8.226 max,
   with prefill's argmax differing on 139 of 3,840 tokens (3.62%).

## Results

Build: 477 s, image `sha256:5416f8ff`, the same digest on all three nodes.
Against the reference arm (`weights-256k`, r5j image with patch 0021 mounted),
same protocol, 2026-09-29:

| | Reference | `candidate-auto` (FP8 attention) | `candidate` (BF16 attention) |
|---|---:|---:|---:|
| Quality gate | 5/5 | 5/5 | 5/5 |
| Needle at 177,654 tokens | 3/3 | 3/3 | 3/3 |
| Logprob gap mean / p95 / max | 0.0612 / 0.305 / 8.23 | 0.0506 / 0.265 / 2.54 | 0.0445 / 0.246 / 2.19 |
| Prefill argmax differs | 3.62% | 2.84% | 2.84% |
| Single-stream step, prose / code (ms) | 41.9 / 47.6 | 42.1 / 48.3 | 42.5 / 47.6 |
| 8 streams prose / code / answers (tok/s) | 166 / 184 / 172 / 236 | 165 / 185 / 170 / 232 | 159 / 183 / 167 / 237 |
| Real-text prefill 64K / 131K / 200K (tok/s) | 3.77k / 3.58k / 3.45k | 3.70k / 3.55k / 3.41k | 3.60k / 3.43k / 3.32k |
| Four 180K contexts, per stream | 11.65 tok/s | 11.92 tok/s | 11.00 tok/s |
| Lowest MemAvailable dgx1 / dgx2 / dgx3 (GiB) | 5.89 / 7.42 / 7.33 | 6.14 / 7.39 / 7.32 | 5.38 / 6.89 / 6.79 |

Patch 0024 and B12X #435 account for the consistency gain: the worst
decode/prefill disagreement fell from 8.2 to 2.5 nats and prefill's argmax
differs on 22% fewer tokens. BF16 attention (0023) cut the mean gap by a
further 12% but cost 0.5-0.76 GiB of headroom on every node (dgx1 also swapped
0.03 GiB), 8% of decode with four 180K contexts and 3% of long prefill, so the
owner chose to promote without it (`VLLM_DS41_ATTENTION_COMPUTE=auto`) until
a teacher-forced fidelity test against `=reference` quantifies its gain. The
DRM file closes after the carve-out import: 0 DRM clients on every node, and
dgx3's HDMI console shows its login prompt.

`reference.sh` ran the promotion reference on the running `candidate-auto`
arm (`manifests/benchmarks/2026-09-29-karmic-kraken-r5k.json`): decode within
noise of r5j at every point, filler prefill 2K 4.2k to 131K 4.3k, real-text
prefill 3.9k at 4K to 3.4k at 200K, prefix replay 6.71 s cold and 0.27 s warm,
four 64K contexts at 13.4 tok/s per stream and 20% of the cache, and a lowest
dgx1 MemAvailable of 6.03 GiB (the 256K limit costs about 0.4 GiB against
131K). Promoted as r5k.
