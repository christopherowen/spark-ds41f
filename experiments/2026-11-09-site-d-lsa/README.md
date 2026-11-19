# Site-d: opt-in latent sparse attention on spark-ds41f-1107

A fourth-site check of the opt-in LSA shortlist (`VLLM_DS41_LSA=1`) described in [`experiments/2026-11-06-optional-lsa`](../../experiments/2026-11-06-optional-lsa/README.md). Four DGX Sparks in a switchless ring, TP4, 1M context, the `tp4-ring` shape. Same image and same site profile as the release; the only change is the one environment variable.

## What was measured

- Prefill, real text (`source`) and the benchmark's filler, at 32K / 128K / 1M, two boots each arm.
- Single-stream decode (300 s closed loop) and 16-stream decode.
- Quality: 12 facts planted across a 128K document, one needle at 968K tokens, and HumanEval+ (164 problems, greedy) on both arms.

All at temperature 0, thinking off, ring control plane.

## Results

### Prefill (tok/s)

| Case | Native | LSA |
| --- | ---: | ---: |
| source 32K | 8,680 | 8,728 (2 boots) |
| source 128K | 8,858 | 9,010 (2 boots) |
| source 1M | 6,342 (2 runs, 6,329–6,356) | **8,722 (2 runs, 8,686–8,758)** |
| filler 1M | 6,302 (2 runs, 6,286–6,317) | **9,042 (2 runs, 9,010–9,075)** |

The 1M gain is the headline: **+37.5% on real text** and **+43.5% on filler**, from interleaved native/LSA boots (two runs per arm, two repeats each), so the comparison shares the same hours and the same thermal envelope. 32K and 128K read +0.2 to +3.2%, which is the same size as the boot-to-boot spread; no clear gain there.

### Decode (tok/s)

| Case | Native | LSA |
| --- | ---: | ---: |
| single stream | 80.43 / 78.45 | 80.23 |
| 16 streams (interleaved A/B, 6 runs each) | 769.3 (752.3–778.6) | 766.3 (750.5–778.8) |

Single-stream is unchanged, as designed (single-row decode stays on the native indexer). 16-stream decode is unchanged too: native to LSA to native to LSA, three runs per boot, six runs per arm, 769.3 against 766.3 tok/s, a 0.4% difference inside the run-to-run spread.

### Quality

| Test | Native | LSA |
| --- | ---: | ---: |
| 12 facts at 128K | 12/12 | 12/12 |
| 12 facts at 128K, half corrected later | 12/12 | 12/12 |
| 8 facts at 968K | 8/8 | 8/8 |
| 6 facts at 968K, half corrected later | 6/6 | 6/6 |
| needle at 968K tokens | 2/2 (111.3 / 111.8 s) | 2/2 (110.5 / 110.9 s) |
| single needle at 128K, 5 trials | 5/5 | 5/5 |
| HumanEval+ base | 97.0% | 97.0% |
| HumanEval+ plus | 151/164 | 151/164 |

The 13 HumanEval+ failures are the same tasks in both arms (103, 132, 141, 145, 151, 154, 163, 32, 39, 76, 91, 97, 99), so the shortlist changed no code answer. The release record reports 152/164; the one-task difference is environmental and does not depend on LSA.

## Reading

- The gain is where the indexer dominates: 1M prefill. At 32K and 128K the indexer is a small share, so the end-to-end number barely moves.
- Quality held on every probe used here: retrieval at 128K, a near-1M needle, and HumanEval+ with identical failures. This does not confirm the paper's "sometimes better" claim; it only finds no loss in these tests.
- 16-stream decode needs an interleaved same-boot A/B before any claim; boot-to-boot spread is the dominant term.

## Limits

- Four nodes, one site. The release's eight-node shapes are not represented.
- HumanEval+ is one greedy sample per problem, so it compares pass@1, not distributions.
- The 1M needle is 968K tokens, 92% of the model's limit.
- The 16-stream case is resolved across boots (interleaved A/B), not within one boot.

## Weights

The model is DeepSeek's published checkpoint (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e10`), loaded directly. No fine-tune, no distillation, no lossy re-quantization. LSA is a runtime attention shortlist; it changes no weight.

## Commands

```sh
bin/spark --cluster-config arm-lsa.json cluster start --replace --apply
bin/spark --cluster-config arm-lsa.json bench --suites prefill --prefill-text source \
  --prefill-sizes 32768,131072,1000000 --prefill-repeats 2 --compare none --output <dir>
bin/spark --cluster-config arm-lsa.json bench --suites prefill --prefill-text filler \
  --prefill-sizes 1000000 --prefill-repeats 2 --compare none --output <dir>
bin/spark --cluster-config arm-lsa.json bench --suite single-stream --compare none --label lsa --output <dir>
```
