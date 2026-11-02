# DSpark tokens a step: why 2.3, and what could raise it (2026-10-26, CET)

Coordinator's brief: find how to raise DSpark's tokens per step substantially, toward 3.5-4.
Production (spark-ds41f-1025, TP4 ring) decodes the release suite's single stream at about
2.28 tokens a step from 4.07 verified rows. External figures are higher: an accept length of
4.26-4.29 on GSM8K-200 (5-shot, thinking off, chat format, every draft verified, c8) and
DeepSeek's own production at 3.2-3.3 (five drafts, sampled, their traffic).

Rules: temperature 0 bit-identical (drafts change only speed), no state shared between
requests, native weights and DeepSeek's reference semantics, no lossy speedups, no fallback
arms; bound before building.

Status: **interim**. The diagnosis below uses program-p1's GSM8K arms (#623, ring,
2026-10-26 22:2x-22:3x CET) and a line-by-line comparison with the checkpoint's reference
drafter. Two measurements are prepared and not yet run: [probe.json](probe.json) (ring, about
40 min) and the CPU reference parity run ([reference/drafter_probe.py](reference/drafter_probe.py)).

## Measured so far

GSM8K-200, 5-shot from the first five training examples, raw `/v1/completions` (no chat
template), temperature 0, max 512, c8; program-p1 on spark-ds41f-1025 (TP4 ring):

| Arm | Tokens a step | Verified drafts a step | Survival by position (accepted / drafts) | Step acceptance by position | Correct |
| --- | ---: | ---: | --- | --- | ---: |
| production (adaptive verification) | 3.224 | 3.21 | 0.84, 0.61, 0.39, 0.24, 0.14 | 0.84, 0.73, 0.64, 0.62, 0.56 | 180/200 |
| every draft verified | 3.563 | 5.00 | 0.83, 0.66, 0.48, 0.35, 0.24 | 0.83, 0.80, 0.73, 0.72, 0.69 | 180/200 |

- **Verify-all reaches 3.56, not 4.27.** The external figure needs a step acceptance of about
  0.86 at every position (survival 0.86, 0.74, 0.64, 0.55, 0.47). Ours starts at 0.83 and falls
  to 0.69 by position 5: the gap grows with depth.
- **The two figures differ in prompt format.** Ours is a raw continuation of the few-shot text,
  theirs one chat turn with thinking off, and they score 196-198/200 where the raw form scores
  180/200. A chat model continuing raw text is plausibly harder to draft. [probe.json](probe.json)
  runs the chat form, so the comparison will be like for like.
- **Production's budget costs 0.34 tokens a step on GSM8K** (3.22 against 3.56) and saves 1.79
  verified rows a step.
- **The arms' output-token totals differ by 31 of 17,600.** All 47 differing questions finish on a
  stop string and differ by one or two tokens, with the same correctness. That is
  `completion_tokens` past a stop string depending on where the last multi-token step ends,
  not a change in text.

## The drafter against DeepSeek's reference

Read from vLLM `next-candidate` e2b2b0b59 (`vllm/models/deepseek_v4_1/nvidia/dspark.py`,
`vllm/v1/worker/gpu/spec_decode/{dspark,dflash}/`, `adaptive_verification.py`,
`tilelang/metadata.py`, `attention.py`) against the checkpoint's `inference/model.py`
(snapshot 1bae0e10: `DSparkBlock`, `DSparkAttention`, `forward_spec`). Every item matches:

| Item | Reference | Ours |
| --- | --- | --- |
| Context states | mean over the hc copies of the input of layers 37, 38, 39 (after Engram) | `aux_hidden_state_layers` (37, 38, 39) captured at `idx + 1 == L`, `mhc.collapse` with the 0.25 mean mix |
| Context projection | `main_norm(main_proj(cat))`, then per layer `kv_norm(wkv(.))`, RoPE at the position, FP8 | `combine_hidden_states`, `precompute_and_store_context_kv`, `insert_context_kv` (RoPE, cache write) |
| Window | the last 128 context positions, anchor − 128 .. anchor − 1 | `prefix − window` .. `prefix`, `prefix` = the block's first position |
| Block | anchor + 4 noise tokens (128799) at positions anchor .. anchor + 4, attending to the window and to every block row | the same; the draft index width is window + 5 padded, every row lists all block rows |
| RoPE of the draft layers | compress ratio 0: base theta, no YaRN | `build_deepseek_v4_rope(compress_ratio=0)` |
| Draft MoE | 128 experts, top 3 | `dspark_n_routed_experts`, `dspark_num_experts_per_tok` |
| Head | `hc_pre` with the last layer's FFN pre-mix, norm, FP32 logits | `mhc.collapse(pre_mix)`, norm, `VLLM_DS41_HEAD_DTYPE=float32`, packed exact head |
| Markov chain | position i adds `head(embed(token i−1))`, token 0 = the anchor, greedy at temperature 0 | vocabulary-parallel greedy with the bias of the previous draft (`select_first_argmax` writes the next embedding) |
| Confidence | `proj(cat(pre-norm head hidden, Markov embedding of token i−1))` | `compute_confidence(sample_hidden, embeds)` in the same order |

- **CED does not truncate the drafter's context in decode.** A decode row's window
  (position − 127 .. position) never reaches before the replay start (prompt end − 128), so
  0501's bound only matters inside the prompt. What CED changes for the drafter is the target's
  states on the 128 replayed prompt rows, which saw truncated decoder windows; they are the
  drafter's context for the first ≤128 decode steps.
- **On the release suite CED is a no-op.** Its prompts are about 40 tokens, inside the 128-row
  replay, so a CED-off arm can only matter on longer prompts (GSM8K's 5-shot prompts are about
  800 tokens).
- **CED can be turned off per request.** A request with `prompt_logprobs` is planned
  `keep_full` (`ced.py`, `replay_rows`): every prompt row goes through the decoder and the
  replay floor is the prefix start. [probe.json](probe.json) uses that for an A/B on one boot.
- **The 1017 checkpoint retrained the drafter.** Its `mtp.*` tensors (main projection, Markov
  head, confidence head, attention, gate, experts) all differ from the original release's, as
  do the target's, so the drafter is not a stale one paired with a new target.
- **No defect found by reading.** What reading cannot see is numeric: the context states, the
  block's attention and the head as computed. The parity run below measures that.

## Prepared measurements

### Ring: [probe.json](probe.json) (about 40 min)

[per_request_acceptance.py](per_request_acceptance.py) sends one request at a time and reads the
per-position counters around each, so every request carries its own accepted and verified
drafts by position and its token ids. On the verify-all arm:
- GSM8K-200 in the chat format (thinking off), like for like with the external figure;
- the same with `prompt_logprobs` (CED off per request);
- 16 suite prompts (one per template) and 24 raw GSM8K prompts, as cases for the parity run.

Then GSM8K-200 in the chat format on production.

### CPU: the reference parity run

[reference/drafter_probe.py](reference/drafter_probe.py) teacher-forces the engine's own texts
through the checkpoint's reference model on CPU: one prefill over prompt + reply gives the
target's states at every position, and the reference drafter runs at every anchor at once (its
window gathered per anchor instead of read from the ring). It records the drafts, the second
candidate at every position, a second chain from the position-1 runner-up, and the
confidences, then chains anchors as a verify-all engine would.

- **Self-test (`--tiny`, a random 5-layer model).** Given the same window contents, the batched
  attention equals the reference's `DSparkAttention.forward` bit for bit at all three stages.
  51 of 59 drafts are identical with batched projections, 54 of 59 with the reference loop's
  one-row projections. The remaining flips are the reference's 140-row prefill GEMM against
  one-row GEMMs, on a random model whose top-two margins are 0.05-0.5.
- **Real run (not yet run).** It uses the loader of `2026-10-26-suffix-drafter/reference/routes_probe.py`
  (O_DIRECT, one layer resident, only the routed experts used) and json-count's CPU kernels.
  Each case takes 5-10 min and reads about 150 GB. All 2,401 `mtp.*` names resolve against
  the 1017 index.
- **What it decides.**
  - Reference tokens a step on the engine's texts at the engine's level (within noise):
    traffic, not a defect.
  - Reference clearly above the engine on the same texts: a numeric defect in our drafter path,
    which the per-anchor drafts then localize.

## Ceilings ([ceilings.py](ceilings.py), [ceilings.json](ceilings.json))

Step prices are #601's calibrated model (`2026-10-26-suffix-drafter/cost.py`): #598's c1 model
extended to TP3 and c8, with the reference route probe's prose N(r). Survival comes from
[survival.json](survival.json). GSM8K raw, against production's measured point:

| TP4 c8, GSM8K raw | Tokens a step | tok/s (model) | vs production |
| --- | ---: | ---: | ---: |
| production | 3.22 | 250 | |
| every draft verified | 3.56 | 203 | −19% |
| best static k (3) | 2.97 | 209 | −16% |
| oracle verification (rows = tokens) | 3.56 | 264 | +5.7% |
| 8 drafts, oracle verification (positions 6-8 at position 5's rate, upper bound) | 3.93 | 278 | +11% |
| tree at position 1 (q = P(runner-up right given top-1 wrong) = 0.35), every row verified | 3.75 | 156 | −38% |
| drafter at the external 4.27, every draft verified | 4.27 | 243 | −3% |
| drafter at the external 4.27, oracle verification | 4.27 | 285 | +14% |

TP3 c8 is within a point of each. The suite's c1 and c8 rows follow once its survival lands.

## Files

- [per_request_acceptance.py](per_request_acceptance.py): per-request acceptance and token ids
  (the engine side of the parity run).
- [probe.json](probe.json): the ring spec (verify-all and production arms of #623).
- [reference/drafter_probe.py](reference/drafter_probe.py): the reference drafter on CPU, with
  `--tiny` self-test.
- [ceilings.py](ceilings.py), [survival.json](survival.json), [ceilings.json](ceilings.json):
  the levers priced.
