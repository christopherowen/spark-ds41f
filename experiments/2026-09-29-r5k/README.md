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

Pending.
