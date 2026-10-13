# r6k: B12X's WO and context-KV projection retired for TileLang at the reference's precision

Owner, 2026-10-13: match DeepSeek's reference (quantize only where it does), and work until
TileLang is well ahead so the B12X versions can be retired.

## The candidate

r6j's vLLM series plus vLLM branch `wo-retire` ([vllm/series](vllm/series), 0120-0132):

- **WO (port B1)**: TileLangWOProjection, WO-A on BF16 activations as the reference's einsum
  (B12X quantized them to FP8), WO-B the reference's MXFP8 `linear`. Sequence-parallel steps
  project through the rank-order exchange.
  - Tiles from the 2026-10-13 sweeps: WO-A split 96/128-row tiles at 65-128 rows and 96-row
    tiles inside the kernel past 128; WO-B 64- and 96-row decode tiles.
  - Per call, level with or ahead of B12X's whole WO at every size (8192 rows: 2989 against
    4524 us).
- **DSpark context-KV (port B6)**: TileLang block-32 projection with the reference's
  activation cast (0.34-0.55x B12X at 1-64 rows).
- **Fused cast (0133)**: WO-A's last kernel rounds to BF16 and applies the reference's
  activation cast itself, so WO-B reads MXFP8 straight from WO-A (2048 rows level with
  B12X, 8192 at 0.67x; decode 1-32 rows faster).
- **Sparse MLA fold (0132)**: every route folds a segment as `fma(l, b, L * a)`. Before, the
  extend route's unrolled one-segment fold was contracted by the compiler to
  `fma(l, b, a)`, so a prompt prefilled in a forward past 144 rows (sequence-parallel
  mixed15) differed by an ulp from the prompt alone. The B12X WO's FP8 input absorbed that;
  the reference-precision WO exposed it
  ([2026-10-13-tilelang-wo](../2026-10-13-tilelang-wo/README.md)).

[r6k-release.json](r6k-release.json) is r6j's release configuration with the r6k image
(vLLM tree `69108474bcc7`, patch head `aa31eb85c656`). [image.json](image.json) builds and gates it;
[release-tp4.json](release-tp4.json) is the release comparison against r6h.

## Gate (2026-10-13)

Deterministic in every scenario, greedy digest equal across runs, quality 5/5; 16 distinct
streams 225.94 tok/s ±0.34%, 8 streams 176.86 ±0.38%.

## Release comparison against r6h (TP4, 2026-10-13)

Arms in the order r6k, r6h, r6k again on a fresh boot;
`python3 scripts/release_table.py r6k-release-tp4 r6h r6k r6k-again`.

| measure | r6h | r6k | change | verdict |
|---|---:|---:|---:|---|
| 4 distinct prompts tok/s | 133.9 | 134.5 | +0.4% ±0.9% | same |
| 8 distinct prompts tok/s | 177.2 | 178.3 | +0.6% ±0.3% | faster |
| 16 distinct prompts tok/s | 225.8 | 223.9 | -0.8% ±0.9% | same |
| acceptance sweep tok/s (paired, 16 prompts) | | | +0.6% ±2.4% | same |
| prefill 32,768 tok/s | 5835 | 5825 | -0.2% ±1.4% | same |
| prefill 262,144 tok/s | 5490 | 5519 | +0.5% ±0.6% | same |
| prefill 524,288 tok/s | 5046 | 5086 | +0.8% ±0.3% | faster |
| prefill 1,000,000 tok/s | 4378 | 4400 | +0.5% ±0.1% | faster |

- Determinism identical in every scenario for both builds, mixed15 included. The reference
  digests differ from r6h's, as expected: WO-A now reads BF16 activations, so the text changes.
- Latency (no intervals): mixed-load short-prompt TTFT -4.8%, mixed stream gap p95 +5.4%,
  the rest within 2%.
- The single-prompt bench points moved 9-14% either way, each with its draft acceptance
  (prose-c1 1.48 -> 1.11 tokens per draft, code-c1 2.07 -> 2.37). The precision change
  changed the text, and those points follow the text; they are not engine speed.
- Boot to boot (r6k against r6k again): every measure "same": distinct-prompt decode within
  0.5%, prefill within 0.3%, single-prompt points within 1.8% with identical acceptance.

r6k reaches the reference's precision without being slower on any text-independent
measure, and is faster at 8 streams and the two longest prefills. It is not yet well ahead.
Promotion is the owner's call.
