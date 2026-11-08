# Site-d: an independent release comparison of spark-ds41f-1107

An independent check of `spark-ds41f:1107` at a second four-node site: DGX Sparks in a switchless ring, tensor parallelism 4, 1M context, the `tp4-ring` shape.

The headline is a **same-site, same-case** comparison against this site's own 1030, because a record from another site mixes in site differences:

| Metric | This site's 1030 | 1107 |
| --- | ---: | ---: |
| conversation (prose, thinking off, 256 tokens) | 83.1 | **88.5** |
| coding (code, thinking off, 256 tokens) | 122.3 | **125.8** |
| prefill 32K | 7,990 | **8,568** |
| prefill 128K | 7,854 | **8,711** |
| prefill 256K | 7,519 | **8,302** |
| prefill 512K | 6,861 | **7,478** |
| prefill 1M | 5,828 | **6,010** |

Faster in both decode cases and at every prefill size. The documented release comparison against the recorded 1030 also passes, with no metric worse (below).

## Site

| Item | This site |
| --- | --- |
| Nodes / shape | 4× DGX Spark, switchless ring, TP4, 1M context, KV 10.5 GiB |
| OS / kernel / driver | Ubuntu 24.04.4, `7.0.0-1019-nvidia-64k`, `580.178.04` |
| Memory saver | DKMS 0.4.1, `vm.watermark_boost_factor=0` |
| Fan control | DKMS 0.1.4, the stock curve `(50,3)(55,5)(60,8)(65,10)(70,12)` |
| GPU clocks | stock; no `nvidia-smi -lgc` lock |
| Image | rebuilt locally from the pinned sources; vLLM tree `ed1bdb7a` |

The host settings follow the release notes. This site had its own GPU clock floor and fan curve; both were removed for this work so the host matches the release's assumptions. An A/B against the earlier settings came back same on every metric, so the values do not depend on that choice. One deployment detail differs from the stock advice and is not described here; it does not touch the serving path.

## Same-site baseline and cases

The 1030 column was measured on this hardware on 2026-10-31 with the ring control plane, thinking off, three runs (prose 83.0–83.4, code 122.1–122.6; tokens per step 2.500/4.339; step 28.6–28.8 ms). Prefill used the benchmark's default text, the same as the release suite's prefill. The 1107 column is this work: `prose-nothink`/`code-nothink` at concurrency 1, and the release suite's prefill (filler text, three repeats).

The 1107 numbers are two boots apart by no more than 0.5%. Both columns are the same site, same text, same extraction.

## Documented release comparison (recorded 1030)

The repository's procedure compares the candidate with the previous release's recorded run. Two candidate boots, `release` + `single-stream` + `sampled`, the recorded [`2026-10-30-spark-ds41f-1030-tp4`](../../manifests/suites/2026-10-30-spark-ds41f-1030-tp4) as the reference, `scripts/servebench_suite_table.py` for the verdict.

```sh
# dev arm: the site profile plus VLLM_SERVER_DEV_MODE=1 (the grid step needs it)
bin/spark --cluster-config arm-dev.json cluster start --replace --apply

bin/spark --cluster-config arm-dev.json bench --suite release       --compare none --label site-d --output results/private/suites/run/site-d
bin/spark --cluster-config arm-dev.json bench --suite single-stream --compare none --label site-d --output results/private/suites/run-single-stream/site-d
bin/spark --cluster-config arm-dev.json bench --suite sampled       --compare none --label site-d --output results/private/suites/run-sampled/site-d

# second boot, scoring the model step on the first boot's text
bin/spark --cluster-config arm-dev.json cluster start --replace --apply
... --suite release ... --corpus results/private/suites/run/site-d/model.json.gz --label site-d-again

python3 scripts/servebench_suite_table.py run site-d manifests/suites/2026-10-30-spark-ds41f-1030-tp4 site-d-again
```

The recorded 1030 run was used instead of re-measuring it: its suite hash is `2c1698722faf` and its request tables are present, which the document allows. The `model` step is skipped because the recorded 1030 directory does not keep `model.json.gz`; the second boot still scored its own model step on the first boot's text.

| Metric | Change vs the record | ±95% | Repeat | Verdict |
| --- | ---: | ---: | ---: | --- |
| grid: decode geometric mean (20 shapes) | −15.22% | 0.14% | −0.20% | better |
| c16: decode tok/s, paired | +3.64% | 1.14% | +0.04% | better |
| c8: decode tok/s, paired | +2.43% | 1.01% | +0.27% | better |
| c1 (single-stream): paired | +3.35% | 1.76% | +0.21% | better |
| poisson: output tok/s | −0.05% | 18.69% | −0.05% | same |
| stall-3: prefill 131K under decode | +6.39% | 1.28% | −0.27% | better |
| stall-11 | +2.76% | 0.52% | −0.23% | better |
| stall-15 | −0.01% | 0.90% | −0.77% | same |
| prefill 32,768 | +17.16% | 1.87% | −0.43% | better |
| prefill 131,072 | +17.77% | 0.34% | −0.45% | better |
| prefill 262,144 | +16.61% | 0.25% | −0.33% | better |
| prefill 524,288 | +14.38% | 0.51% | −0.17% | better |
| prefill 1,000,000 | +8.31% | 0.33% | −0.07% | better |
| sampled c16: paired | +3.67% | 1.32% | +0.03% | better |
| sampled c8: paired | +1.81% | 1.67% | −0.03% | better |
| sampled c1: paired | +4.29% | 1.70% | +0.29% | better |

The record is another site's, so its decode absolute values are 1–3% away from this site's; the verdict above is directional and the same-site table is the local check.

## Weights and outputs

The model is DeepSeek's published checkpoint (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e10`), loaded directly (`--load-format direct`, `VLLM_PLUGINS` empty). No fine-tune, no distillation, and no lossy re-quantization: the checkpoint ships in its native FP8/FP4/BF16 form. The engine does re-pack some weights losslessly at load (the output head into an exact 12-bit form, the routed experts' scale factors into a base exponent plus delta codes); the values are unchanged.

The release's determinism gate reports 256/256 greedy outputs identical to the previous release. This site did not repeat that gate; it is cited from the release record.

## Limits

- The `model` step of the release suite is skipped (the record keeps no `model.json.gz`).
- The 1030 baseline is this site's 2026-10-31 measurement, not a same-day A/B; the host settings changed after it (clock floor and fan curve), and the A/B for those changes was run on the 1107 only.
- Prefill on the 1107 is measured at the end of the release suite (about 30 minutes of load); the 1030 prefill was a standalone run. That biases against the 1107, so the same-site prefill gains are conservative.
