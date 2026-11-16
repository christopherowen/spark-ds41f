# 2026-11-16-c60-ring-prefill-sweep

What does candidate-60's cold prefill curve look like on the switchless TP4 ring, from 512 to
1M tokens? Candidate-60's prefill benches ran only on `tp4-switched`, at three sizes: 32K, 131K
and 262K. There it reached 10,000.1 tok/s at 131K in Step 1 and 9,924.4 in its final build. On
the ring, the last prefill sweep was candidate-58's, at the same three sizes: 9,408 / 9,755 /
9,620 tok/s.

- **Base commit**: `7db478260`.
- **Image**: `spark-ds41f:candidate-60`, built on dgx by the lab job `20261116T171622-build`.
- **Profile**: [tp4-dev.json](tp4-dev.json). It copies
  `experiments/2026-11-16-c60-prefill/tp4.json` (image, expected vLLM tree label, upstream lock)
  and adds development RPC, as `tp4-dev.json` there does. Because it requires the image's tree
  label, a rebuilt `candidate-60` tag refuses to boot instead of being measured.
- **Testbed**: `tp4-ring` (dgx1..dgx4, switchless cable ring, context 1,048,576).
- **One variable**: prompt length. Every size runs in the same boots.

## Workload

`bin/spark bench --compare none --suites prefill --prefill-text source --prefill-sizes 512,1024,2048,4096,8192,16384,32768,131072,262144,524288,1000000 --prefill-repeats 2`.

The prompts are cold, seeded Python standard library source; the rate is actual prompt tokens
divided by client-side time to first token. There are 2 boots (`c60-ring-sweep-b1`, `-b2`).

Spec: [bench-prefill-sweep-tp4.json](bench-prefill-sweep-tp4.json).

```sh
bin/spark candidate queue tp4-ring experiments/2026-11-16-c60-ring-prefill-sweep/bench-prefill-sweep-tp4.json --apply
```

## References

These are recorded runs, not measured again:

- **Candidate-58 on `tp4-ring`, source text:** 9,408 / 9,755 / 9,620 tok/s at 32K / 131K /
  262K (`experiments/2026-11-14-c58-prefill-10k`).
- **Candidate-59 on `tp4-switched`, source text, 512 to 1M:**
  `experiments/2026-11-16-c59-long-prefill`.

## Runs

Pending.

## Conclusion

Pending.
