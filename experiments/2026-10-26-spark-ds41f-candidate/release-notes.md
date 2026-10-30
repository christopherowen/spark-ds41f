# spark-ds41f-1030: release notes

spark-ds41f-1030 is a release of this deployment, named after the day it is promoted. The model
is DeepSeek-V4.1-Flash-1017, DeepSeek's checkpoint, with native weights.

It is candidate 14: spark-ds41f-1025 plus vLLM 0491-0567 (candidates 7-14 in
[README.md](README.md)). Image `spark-ds41f:1030` (built as `spark-ds41f:19f2c20ed4d6-next-14`,
`sha256:7918fd6145b5`), vLLM patch head `b6de07aa`, tree `ea1281f8`. Four-node ring (tp4-ring),
and the three-node triangle (tp3-triangle) from 2026-10-30.

## Results (the ring, 2026-10-29)

The speed comparison ran on candidate 12. Candidates 13 and 14 change two things after it, each
checked for bits, memory and startup on the ring: 0566 stores the 2-bit expert scale codes in
the 4-bit codes' buffer (model loading back to 71.67 GiB a rank from candidate 12's 72.72), and
0567 is the Engram host-hash fix below. Neither changes a kernel's arithmetic or what decode
reads, and every digest is candidate 12's.

| Check | Result |
| --- | --- |
| Release tests | candidate 12's image: 6,686 passed, 163 skipped, 0 failed. Candidate 13's scale-code bundle: 117 passed; 0567's five tests pass |
| Temperature-0 outputs | every digest equals candidate 10's and candidate 12's (determinism with and without logprobs, 16K, greedy, seeded sampled, prompt logprobs, images, CED cross-shape, control spellings and controls); candidates 13 and 14 keep them (`fe2461d8`, `c9cf9dbe`, `fcd34fd2`) |
| Model behaviour against 1025 | greedy 256/256 identical; teacher-forced KL 4.2e-8 nats over 65,536 positions (0 between the candidate boots) |
| GLP dual serving | both names served; steering moves all 8 probe prompts; mixed batches and the prefix cache keep each name's text; beam search refused with 400. AdvBench (520 prompts): no refusal phrase in 2.5% stock, 51.2% steered |
| #677's load | 4 agents to 311K tokens and 8 chats resending each question, 300 s, on two boots of candidate 14: 0 failed of 150 requests each |
| Startup | 277 TileLang calls prepared; init 28-37 s; nothing compiles after |
| Memory | model loading 71.67 GiB a rank (1025's); the ring's lowest free memory in a boot 19.4 GiB or more |
| KV capacity | vLLM's startup figure: 11,554,122 tokens, 11.02 full 1M contexts (#586's count; 1025's line read 6.21 for the same pool) |

Speed against spark-ds41f-1025, one session (candidate 12, spark-ds41f-1025, candidate 12
again), verdicts against the repeat:

| Metric | spark-ds41f-1025 | spark-ds41f-1030 | Change | Verdict |
| --- | ---: | ---: | ---: | --- |
| decode grid, geometric mean (20 shapes) | | | -4.62% | better |
| one stream, decode tok/s (paired) | 75.4 | 78.3 | +3.61% | better |
| one stream, TTFT (paired) | 0.128 s | 0.122 s | -4.5% | better |
| 8 streams, decode tok/s (paired) | | | +2.02% | better |
| 8 streams, output tok/s | 194.6 | 198.4 | +1.96% | better |
| 16 streams, decode tok/s (paired) | | | +2.05% | better |
| 16 streams, output tok/s | 251.7 | 257.7 | +2.40% | better |
| prefill 32K / 131K / 262K / 524K / 1M tok/s | 6,744 / 6,798 / 6,477 / 5,870 / 4,948 | 7,313 / 7,396 / 7,120 / 6,538 / 5,549 | +8.4% to +12.2% | better |
| 131K prefill beside 15 decoding streams | 6,447 | 6,978 | +8.23% | better |
| 131K prefill beside 3 / 11 streams | | | +0.43% / -0.74% | same |
| short requests' TTFT p50 during a 131K prefill | 12.4-12.8 s | 3.3-3.6 s | 3.6-3.7x sooner | better |
| Poisson output tok/s | | | -0.01% | same |
| sampled: c16 / c8 / c1 decode (paired) | | | +1.87% / +2.77% / +2.68% | better |

The records are in [manifests/suites/2026-10-30-spark-ds41f-1030-tp4](../../manifests/suites/2026-10-30-spark-ds41f-1030-tp4/)
and its `-single-stream` and `-sampled` companions: candidate 12's first arm, now tp4-ring's
`benchmark_reference`.

## What changes for users

- **Faster.** Decode 1.9-3.6% at every concurrency, prefill 8-12% from 32K to 1M.
- **Short requests are not stuck behind a long prompt.** Prefill yield (`--prefill-yield short`,
  10 s limit): a waiting prompt that fits in one chunk, and a held first token, take the next
  step from a long prompt's prefill. Beside a 131K prefill a short request's first token comes
  in 3.3-3.6 s instead of 12.4-12.8 s; the long prompt pays about 9% of its rate while short
  requests keep arriving, and nothing when none are.
- **A second served name, `deepseek-v4.1-flash-weightless`,** steered per request with msuiche's
  published GLP vector for 1017 (#401). `deepseek-v4.1-flash` is unchanged and bit-identical
  beside it. Every node needs the vector file (docs/setup.md, section 5); `cluster start` refuses
  without it.
- **Fixed: an engine stop under mixed agent and chat traffic (#677).** A prompt with one token
  left that the scheduler padded with placeholder drafts reached the Engram host hash, whose
  check raised and stopped EngineCore. The GPU now hashes such a request's rows (0567).
  spark-ds41f-1025 has the same path; the trigger was not reproduced on the ring.
- **DeepSeek API semantics:** the effort aliases, `response_format`, `tool_choice: none` and the
  assistant prefix follow DeepSeek's API (0539-0540).
- **CED replays a prompt's last 128 rows wherever its chunks end (0526-0527).** Temperature-0
  text changes for multi-chunk prompts whose final chunk has 1-128 rows (0.79% of multi-chunk
  prompts at TP4); every other prompt is unchanged.
- **The startup KV line counts what the pool holds** (#586): 11.02 full 1M contexts.

## For operators

- TileLang is the only kernel family; `VLLM_DS41_KERNEL_BACKEND` is gone and
  `VLLM_ENABLE_ROCE_ALLREDUCE=1` is refused. B12X stays in the image for its packages only,
  loaded by nothing (`VLLM_PLUGINS` empty, `--load-format direct`).
- IKET's debug mode is in the image, off (`VLLM_DS41_IKET` unset).
- The recipe and its lock are frozen in [releases/spark-ds41f-1030](../../releases/spark-ds41f-1030/);
  the rollback is [releases/spark-ds41f-1025](../../releases/spark-ds41f-1025/) on
  `spark-ds41f:19f2c20ed4d6-next-6`.

## Results (the triangle, 2026-10-30)

The release at the triangle ([rc14/rc-tp3.json](rc14/rc-tp3.json), the 6,144-token budget),
checked in run `candidate-14-tp3` and compared in run `candidate-14-release-tp3` (the release,
spark-ds41f-1025, the release again; verdicts against the repeat).

| Check | Result |
| --- | --- |
| Temperature-0 outputs | at 8,192, every greedy digest equals spark-ds41f-1025's apart from CED's cross-shape prompt (0526); the release at 6,144 equals 8,192 in every digest |
| Model behaviour against 1025 | greedy 256/256 identical; teacher-forced KL 4.2e-8 nats |
| Functional checks and GLP | pass; AdvBench no refusal phrase in 1.9% stock |
| Memory | model loading 93.95 GiB a rank (1025: 94.12); dgx1's lowest MemAvailable 7.29 and 7.95 GiB (1025: 6.23) |
| Startup | 276 TileLang calls prepared; nothing compiles after |

| Metric | spark-ds41f-1025 | spark-ds41f-1030 | Change | Verdict |
| --- | ---: | ---: | ---: | --- |
| decode grid, geometric mean (15 shapes) | | | -5.99% | better |
| one stream, decode tok/s (paired) | 61.2 | 63.5 | +3.76% | better |
| one stream, TTFT (paired) | 0.159 s | 0.153 s | -4.1% | better |
| 4 streams, decode tok/s (paired) | | | +2.68% | better |
| 8 streams, decode tok/s (paired) | | | +2.29% | better |
| 8 streams, output tok/s | 154.7 | 158.6 | +2.51% | better |
| Poisson, output tok/s | 118.9 | 119.0 | +0.09% | same |
| prefill 32K / 131K / 262K / 500K | | | +4.4 / +4.1 / +5.3 / +7.0% | better |
| 131K prefill beside 7 streams | 5,307 | 5,465 | +2.98% | better |
| 131K prefill beside 3 streams, short requests waiting | 5,314 | 5,069 | -4.61% | worse |
| short requests' first token there, p50 | 16.6 s | 2.0 s | -87.8% | better |
| 131K prefill beside 6 streams, short requests waiting | 5,295 | 5,099 | -3.70% | worse |
| short requests' first token there, p50 | 18.3 s | 9.6 s | -47.6% | better |
| sampled c16 / c8 / c1, decode tok/s (paired) | | | +2.76 / +2.41 / +2.94% | better |

The owner accepted the two worse verdicts on 2026-10-30. The suites are
[manifests/suites/2026-10-30-spark-ds41f-1030-tp3](../../manifests/suites/2026-10-30-spark-ds41f-1030-tp3/).

## Limitations

- **On the triangle, a 131K prefill beside 3 or 6 streams with short requests waiting is
  3.7-4.6% slower** than spark-ds41f-1025's: it yields to those requests (their first token
  2-8x sooner) and runs the 6,144 budget (1025 qualified the triangle at 8,192).
- **A lone 131K prefill is 0.8% slower** than candidate 10's (in 0541-0544 or the IKET patches,
  not yet bisected); against 1025 it is still 8.8% faster.
- **Seeded sampling at 8 streams depends on the batch** under adaptive verification, as in
  spark-ds41f-1025.
- **Copy drafts are not in this release** (#619): enabled, they fail to start.
