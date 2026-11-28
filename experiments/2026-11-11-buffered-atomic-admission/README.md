# Buffered atomic admission on 1107

The buffered chat batch endpoint can preprocess several prompts in one HTTP
request, but normally sends each prompt to the engine separately. If the engine
is idle, the first prompt can start before the rest arrive. This experiment adds
the opt-in header `X-TP8-Atomic-Admission: 1`. It registers the output collectors
and submits all the prepared prompts in one engine message. Requests without
the header still use ordinary admission. The scheduler keeps its normal token,
sequence, and memory limits; admission together does not guarantee execution in
one tensor batch.

This patch is for one engine core (DP1) and the non-streaming
`/v1/chat/completions/batch` route. It accepts up to 16 greedy plain-text
requests with one output each, at most 2,048 input tokens and 256 output tokens
per request. It excludes multimodal inputs, LoRA, prompt embeddings, structured
outputs, pooling, and resumable input. Rejected requests do not silently fall
back to ordinary admission. Tests check validation and collector cleanup,
including cancellation and a submission failure.

## Paired results

There were two separate campaigns: **GPU capped at 2 GHz** and **GPU uncapped
(stock automatic clocks)**. Ordinary and atomic admission were paired on the
same image and boot within each campaign. The percentages compare admission
modes at the same GPU clock policy.

Both campaigns used switched TP8, official DeepSeek-V4.1-Flash-1017, disk
Engram, 10.5 GiB KV per rank, and the original adaptive-five DSpark profile.
Both arms used the same buffered endpoint and changed only the header. Each
cell had ten counterbalanced pairs, with cold prefixes (zero observed
prefix-cache hits), temperature 0, seed 42, and the
original [code and prose prompts](../../spark_ds41f/bench.py)
(75/68 native input tokens). All responses reached 256 native output tokens in
reasoning. The rates include prefill, queuing, and completion of the last
request. They do not measure time to first token or completed coding tasks.

| Workload | GPU capped at 2 GHz: ordinary → atomic tok/s | GPU uncapped (stock automatic): ordinary → atomic tok/s |
| --- | ---: | ---: |
| C8 code | 529.3 → 543.3 (+2.6%) | 549.3 → 569.7 (+3.7%) |
| C8 prose | 424.4 → 500.3 (+17.9%) | 450.5 → 526.1 (+16.8%) |
| C16 code | 654.2 → 744.8 (+13.9%) | 724.5 → 790.9 (+9.2%) |
| C16 prose | 628.3 → 699.7 (+11.4%) | 683.8 → 735.1 (+7.5%) |
| C16 code, 8+8 at 90 ms | 641.6 → 703.9 (+9.7%) | 721.2 → 722.5 (+0.2%) |
| C16 prose, 8+8 at 90 ms | 630.5 → 666.5 (+5.7%) | 663.1 → 688.7 (+3.9%) |

Rates are medians; percentages compare medians. Each completed campaign had
120 timed waves and another 24 ramp/warmup waves. All 1,860 lanes per campaign
matched the fresh single-request native token IDs and decoded text. The engine
trace showed the intended fresh prefill shape in all 60 timed atomic waves per
campaign. The split code case in the uncapped campaign shows little measured gain.

[Results](results.json) and the [GPU-capped](runs/capped.json) and
[GPU-uncapped](runs/stock.json) per-wave receipts preserve the observed group sizes,
native token totals, rates, and trace counts. The
[first GPU-uncapped attempt](runs/stock-incomplete.json) stopped when 11 of 16 lanes in
an ordinary 8+8 prose wave diverged from the exact reference. It retained 113
passing timed waves, the divergent wave, and six unrun waves. The subsequent
complete run does not establish the cause. Whole-window minimum available RAM
was 38.03 GiB capped and 38.09 GiB uncapped; no OOM counter increase was
observed. The capped run had a swap I/O delta.

## Reproduction and status

The [build profile](tp8-switched.json) pins the unpromoted vLLM series on the
frozen 1107 tree. A fresh replay reproduced the 641 release patches and applied
the experiment patch as patch 642. The five resulting runtime files match those
used in the measured image byte for byte. That measured image also carried a
separate parser repair; [base.json](base.json) records its identity. The
experiment profile contains only atomic admission and its trace hooks. Its
image has not been built or served, and it is disabled for launch until an
operator qualifies it. No promoted settings or release receipts change.

With a local TP8 node map, use the recipe's source preparation command:

```bash
experiment=experiments/2026-11-11-buffered-atomic-admission
bin/spark --cluster-config "$experiment/tp8-switched.json" doctor
bin/spark --cluster-config "$experiment/tp8-switched.json" build prepare --only vllm
```

Run the included five CPU tests with the absolute prepared vLLM source path
and an environment that has vLLM's Python dependencies and pytest installed:

```bash
CUDA_VISIBLE_DEVICES="" VLLM_TARGET_DEVICE=cpu HF_HUB_OFFLINE=1 \
  VLLM_ATOMIC_SOURCE="$prepared_vllm" PYTHONPATH="$prepared_vllm" \
  python -m pytest --confcutdir "$experiment" "$experiment/test_buffered_atomic.py"
```

On a separately qualified server with the experiment applied, use a body with
`model: "deepseek-v4.1-flash"`, `messages: [[{"role":"user","content": prompt}], …]`,
`temperature: 0`, `seed: 42`, `max_tokens: 256`, and `return_token_ids: true`.
Repeat the same prompt in 8 or 16 conversations. For the split cell, send two
eight-conversation requests 90 ms apart. Send the exact same bytes with and
without the opt-in header. Compare native IDs and text with a fresh ordinary
C1 response and record every failed wave. Enable
`--enable-logging-iteration-details` and `--cudagraph-metrics` in both arms to
check `TP8_ATOMIC_ADMIT`, `TP8_FRESH_PREFILL`, and `TP8_STEP` against the
timing. The original paired measurement harness SHA-256 was
`6aa92590f6cec6f7890a1793946d1d3eab540ccdd9dc3ecd7bd57aa30b8bea07`.

The streaming group coordinator and gateway are separate experiments. These
buffered measurements make no claim about staggered AgentX traffic.
