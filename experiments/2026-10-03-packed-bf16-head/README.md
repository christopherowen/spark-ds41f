# Exact 12-bit packed BF16 vocabulary head

Base deployment commit: `3fb888b` (the [native drafter heads](../2026-10-03-native-drafter-heads/README.md)
change on `aafb3cf`).

**Variable:** the DS4.1 target vocabulary head, which the native DSpark draft
head shares, is stored in an exact 12-bit packed form instead of BF16
(`VLLM_DS41_PACKED_BF16_LM_HEAD=1`), with its own DSpark cost directory
(`ring4-collective-packedhead-20261003`). Every weight bit is kept; only the
storage and the kernel that reads it change.

Candidate: [candidate.json](candidate.json), the TP4 collective-contract
candidate with native heads plus the variable above. Sources:
[source.json](source.json) and [upstreams.lock.json](upstreams.lock.json),
extending the candidate's series with
[B12X 0012](b12x/0012-packed-bf16-vocab-projection.patch) and
[vLLM 0028](vllm/0028-deepseek-v41-packed-bf16-lm-head.patch).

**Status:** source-qualified on GB10 through kernel-lab overlays; image,
kernel tests and the TP4 window follow.

## Why

With native drafter heads every decode step reads the BF16 head shard twice:
once for target verification and once for the first draft position. At TP4
that is 2 × 331 MB per rank, about 2.6 ms of each step's tail, and these GEMMs
already stream near the GB10's bandwidth. Reading fewer bytes is the only
lever left, and the owner's rule allows it only if it is purely lossless.

## Format

A BF16 value is a sign bit, an 8-bit exponent and a 7-bit mantissa. The packed
form keeps the sign and mantissa in one byte and the exponent in a 4-bit code:
code `c` in 1–15 means exponent `base + c − 1` for a 15-exponent window chosen
per tensor shard; code 0 means exponent 0, so zeros and subnormals stay exact.
Values whose exponent falls outside the window are exceptions: their slot holds
+0 and their exact BF16 bits sit in a per-row list that the projection adds
after the main dot product. A row is `K` sign-and-mantissa bytes followed by
`K/2` code bytes, 7,680 bytes for K = 5,120.

Packing runs once after loading and refuses to start unless unpacking
reproduces every input bit. The checkpoint (`dba1be0a…`) has no zero or
subnormal values in `head.weight` and uses 31 exponents (97–127); the best
window, 112–126, leaves about 1.6 values in 10,000 as exceptions:

| Shard | Rows | Exceptions | Most in one row | BF16 bytes | Packed bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP4, each rank | 32,320 | 25,718–27,034 | 23 | 330,956,800 | 248,554,820 |
| TP3, each rank | 43,092–43,094 | 34,416–35,683 | 23 | 441,262,080 | 331,404,396 |

The packed bytes replace the BF16 tensor, so the head also frees 82 MB per
TP4 rank, and the display carve-out holds the packed form.

## Kernel

B12X's prepared `packed_bf16_vocab_projection` (Triton) decodes in registers
and accumulates in FP32 on tensor cores. Each logit is its row's 128-column
groups summed in order, then its exceptions, independent of token count and
tile configuration, so a token's logits are the same bits at any batch size.
They differ from cuBLAS BF16 logits only by FP32 accumulation order (maximum
absolute difference equal to cuBLAS's own against an FP32 reference).

Development sweep on dgx4 (real head shards, CUDA graphs, best of 432
configurations per row count; the defaults use these winners):

| Shard | Rows | cuBLAS BF16 | BF16 row kernel | Packed | Packed GB/s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4 rank 0 | 1 | 1,362 µs | 1,306 µs | 1,028 µs | 242 | 1.33× |
| TP4 rank 0 | 6 | 1,389 µs | — | 1,049 µs | 237 | 1.32× |
| TP4 rank 0 | 8 | 1,396 µs | — | 1,044 µs | 238 | 1.34× |
| TP4 rank 0 | 16 | 1,393 µs | — | 1,072 µs | 232 | 1.30× |
| TP4 rank 0 | 32 | 1,485 µs | — | 1,062 µs | 234 | 1.40× |
| TP4 rank 0 | 48 | 1,546 µs | — | 1,107 µs | 224 | 1.40× |
| TP3 rank 2 | 1 | 2,664 µs | 1,741 µs | 1,401 µs | 237 | 1.24× vs row kernel |
| TP3 rank 2 | 6 | 1,847 µs | — | 1,414 µs | 234 | 1.31× |
| TP3 rank 2 | 48 | 1,912 µs | — | 1,537 µs | 216 | 1.24× |

At TP4 that is 0.33–0.44 ms less per head read, two reads per decode step.

## Changes

- **B12X 0012**: the `gemm.packed_bf16_vocab_projection` component: `pack` /
  `unpack`, the Triton kernel, its tuning contract (with the measured
  shared-memory model that rejects configurations SM12x cannot hold) and
  tests.
- **vLLM 0028**: `PackedBf16LmHeadMethod` packs the loaded BF16 head; the
  DS4.1 target head selects it with `VLLM_DS41_PACKED_BF16_LM_HEAD=1`; the
  logits processor declares, prepares and runs B12X's packed projection per
  token capacity, for the target and for the drafter that shares the head.
  The DSpark Markov head is unchanged: it gathers individual rows.

## Bundles

- [b12x-tests](bundles/b12x-tests/candidate.json): B12X 0012's tests from the
  image's own tree.
- [vllm-tests](bundles/vllm-tests/candidate.json): vLLM 0028's logits-head
  test and the existing shared/Markov preparation test.
- [head-bench](bundles/head-bench/candidate.json): the real head shards,
  packed projection against cuBLAS, with the default configuration.
