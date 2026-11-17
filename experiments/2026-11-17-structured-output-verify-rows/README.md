# Structured outputs with copy drafts: size the grammar bitmask for verify rows

Fixes [#1194](https://github.com/christopherowen/spark-ds41f/issues/1194). The diagnosis and the three-line fix are
**Zee-B33's**, from that issue; this folder packages it as **vLLM `0817`** on the candidate-60 patch head (through
`0816`, tree `afcaac35`) with a probe and receipts, so it can go into a candidate. The patch is authored as Zee-B33.

## The bug, briefly

`StructuredOutputManager.grammar_bitmask()` allocates the bitmask with
`max_num_spec_tokens = num_speculative_tokens` (5). From 1107, DSpark copy drafts verify up to
`num_verification_tokens` (16) rows per request. A burst of `response_format` requests (json_object or json_schema)
whose answers copy long prompt spans fills more rows than the bitmask holds, xgrammar's bounds check raises inside
`fill_next_token_bitmask`, and the engine exits with every in-flight request failed. Reproduced in #1194 by
Zee-B33, mjcryptocloud (four nodes, `tp4-ring`, 1107) and hxfhd (TP3, 1107).

## The change (`vllm/0817-…patch`)

```python
max_num_spec_tokens = self.vllm_config.num_speculative_tokens
spec_config = self.vllm_config.speculative_config
if spec_config is not None and getattr(spec_config, "num_verification_tokens", None):
    max_num_spec_tokens = max(max_num_spec_tokens, spec_config.num_verification_tokens)
```

Same logic as the issue, wrapped to 88 columns. The bitmask grows from 96 to 272 rows at 16 seats (about 1.6 MB to
4.4 MB); row layout and the slice sent to workers are unchanged. Not changed (as the issue notes):
`backend_outlines.py` and `backend_lm_format_enforcer.py` also read `num_speculative_tokens`.

`git apply --check` is clean on the candidate-60 tree (`afcaac35`); it also stacks with the DSML closer patch
proposed separately (`0818`).

## Evidence

4× GB10 on a switch, candidate-60 image built from this repository's c60 lock (`spark-ds41f:candidate-60`,
vLLM tree `afcaac35`), `tp4-switched` profile with site sizing (32 seats, 8,192 batched tokens), the official
1017 checkpoint. The change ran as a read-only bind mount of `vllm/v1/structured_output/__init__.py` carrying the
same three lines (variable named `_spec`, unwrapped).

`probe/so_probe.py ROOT MODEL 32 2` sends two waves of 32 concurrent chat requests, alternating `json_object` and
`json_schema`, each asking the model to copy a 40-record JSON array from the prompt (temperature 0, thinking off),
then checks HTTP 200, valid JSON, the record count, and that `/v1/models` still answers.

| boot | requests | HTTP 200 | valid JSON | exact count | engine alive after |
|---|---:|---:|---:|---:|---|
| 1 | 64 | 64 | 64 | 64 | yes |
| 2 | 64 | 64 | 64 | 64 | yes |
| 3 (with the c61 overlay) | 64 | 64 | 64 | 64 | yes |
| 4 | 32 | 32 | 32 | 32 | yes |

Receipts: `receipts/so-probe-boot{1..4}.json` (head address removed).

## What it does not prove

- **We did not reproduce the crash.** No boot without the change was probed with copy drafts on, so this shows the
  change holds under the burst, not that the burst kills the unpatched engine here. The reproductions are the ones
  in #1194.
- We did not record how many rows the copy drafts verified during the probe, so we cannot show that the burst
  actually went past 5 rows on these boots.
- The release-test bundle has not run against it. The change touches bitmask allocation only, not the forward
  pass or sampling; no throughput run was made with and without it.
