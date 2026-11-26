# Independent quality evaluations

Third-party benchmark and evaluation receipts run on released `spark-ds41f` builds by community operators. Each entry preserves the operator's reported configuration, deviations from the promoted recipe, and verbatim evaluation header or log receipt.

## GPQA Diamond — `spark-ds41f-1120` on `tp3-triangle-4k` (2026-11-25)

- **Operator:** `j0n` ([NVIDIA Developer Forum post #210](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438/210), Inspect log header in [post #212](https://forums.developer.nvidia.com/t/new-deepseek-4-1-flash-recipe/384438/212))
- **Receipt:** [2026-11-25-spark-ds41f-1120-tp34k-gpqa-diamond.json](2026-11-25-spark-ds41f-1120-tp34k-gpqa-diamond.json) (verbatim `header.json` from `inspect_ai 0.3.292`, `inspect_evals/gpqa/gpqa.py` commit `ba77336a`)
- **Release & shape:** `spark-ds41f-1120`, `tp3-triangle-4k` (`TP=3`, `oneshot-direct`, `262,144` context, `8` concurrent streams, `4,096` batched tokens, `2.2 GiB` KV/rank, display carve-out enabled)
- **Model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`1bae0e10`, served as `deepseek-v4.1-flash`)
- **Hardware & OS:** 3 × DGX Spark (switchless triangle, MTU 9000, ~185 Gb/s per pair), DGX OS 26.09.2+, stock `4 KiB` kernel `7.0.0-1025-nvidia` (no memory-saver kernel module)

### Command

```bash
inspect eval inspect_evals/gpqa/ --model openai-api/openai/deepseek-v4.1-flash \
  --epochs 4 --max-connections 8 --max-tokens 131072 --token-limit 262144 \
  --time-limit 10800 --temperature 1.0 --top-p 1.0 \
  --extra-body '{"chat_template_kwargs": {"thinking": true, "reasoning_effort": 100}}'
```

### Results

| Metric | Value |
| --- | --- |
| GPQA Diamond accuracy (`choice`) | **`0.9217 ± 0.0168`** (`92.2% ± 1.7%` stderr) |
| Samples completed / scored | `792 / 792` (`198` questions × `4` epochs, `0` unscored) |
| Wall time | `8h 04m 18s` (`2026-11-24T18:21:01Z` – `2026-11-25T02:25:19Z`) |
| Tokens processed | `5,798,314` total (`213,592` input, `5,580,882` output, `5,469,179` reasoning) |
| Sustained throughput (8 streams) | `~192–209 tok/s` aggregate (~`200 tok/s` steady) |
| DSpark acceptance (reasoning, `T=1.0`) | `2.4–2.7` tokens/step (`61–69%` draft acceptance) |
| KV cache utilization | `4.8–5.5%` of `2.2 GiB` per rank |
| Stability | `0` errors, `0` OOMs, `0` thermal throttling, `0` restarts |

### Recorded deviations from promoted `tp3-triangle-4k`

- Host kernel was stock `4 KiB` `7.0.0-1025-nvidia` rather than the recipe's pinned `7.0.0-1019-nvidia` (`doctor` warned; serving unaffected).
- Eval used `reasoning_effort: 100` (maximum thinking) rather than the default `75`.
- Image was built locally from the `spark-ds41f-1120` commit (`spark-ds41f:1122`, `sha256:c868148c4553...`) without host fan control installed.
