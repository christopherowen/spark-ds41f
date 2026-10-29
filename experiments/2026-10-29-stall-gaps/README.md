# What the short decode gaps during a long prefill are

On #650 the decode gaps that overlap a long cold prefill take two values: a long one, about
one step carrying a prefill chunk (about 2.25 s on the ring at a 16,384-token budget), and
a short one, at about half each. So their median flips between the two from repeat to
repeat (0.69-2.12 s for candidate 10 in `../2026-10-28-candidate-10-levers`, "yield,
depth"), while their mean and p90 hold. With yield and probes the median fell to about
0.33 s. The short gaps are either engine steps that carried no chunk (decode alone, between
chunk steps), or one step's output reaching a stream as two chunks. This run tells them
apart.

## The run (`stall-gaps.json`, run `stall-gaps`)

`suite.json`: the stall step at 131K, 7 of 16 decoding streams, four repeats, without probes
(`stall-7`) and with the yield-depth run's probes (`stall-7-probes`, 500 tokens, 2-14 s into
the prefill). The servebench stall step (#673) keeps each repeat's raw overlapping gaps and
the engine steps the prefill spanned, from `vllm:iteration_tokens_total` scraped as the
document is sent and at its first token, split at 1,024 tokens into steps with a chunk and
steps without. Arms on the ring: candidate 10 and candidate 10 with yield, one boot each.

## Reading it

Per repeat, compare each stream's overlapping gaps with the engine's steps:

- Gaps a stream about equal to the engine steps, and steps without a chunk about as many as
  those with: the short gaps are decode-only steps between chunks. Then the question for
  #650 is what schedules them, and their length against the baseline step.
- Gaps a stream about twice the engine steps, with few steps without a chunk: one step's
  output arrives as two chunks, and the stall is the chunk steps alone.

The raw gaps' start offsets place the short gaps against the chunk steps. With probes, the
steps without a chunk include the probes' own prefill steps under yield.

## Results

Run `stall-gaps`, 2026-10-29, ring, four repeats an arm and step. **Neither reading above: the
short gaps come before the prefill starts.** Each stream's gaps in order (candidate 10, no
probes) are about seven at the baseline step (70-80 ms), then the eight chunk steps of 131K at
16,384 back to back (2.08-2.27 s each), then one short gap to the first token. The short ones
are the streams decoding while the server tokenizes and admits the document, 0.42-0.52 s;
about as many fit in it as there are chunks, so the median flipped. Each stream saw about one
gap an engine step (15-17 steps, 15-17 gaps a stream), so no step's output arrives as two
chunks. The step-tokens histogram does not tell chunk steps from decode steps: vLLM counts a
prompt's tokens at its first token, so every chunk step reads as a decode step.

From the prefill's first chunk step to its first token (recomputed from the raw gaps):

| Arm, step | Admission | Gaps a stream | Mean gap | Median gap | Share held by a chunk |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate 10, `stall-7` | 0.47-0.49 s | 9.3-9.6 | 1.86-1.93 s | 2.15-2.16 s | 0.93-0.97 |
| yield, `stall-7` | 0.39-0.52 s | 8.7-10.4 | 1.73-2.05 s | 2.15-2.18 s | 0.86-1.00 |
| candidate 10, `stall-7-probes` | 0.42-0.50 s | 9.9-10.7 | 1.73-1.86 s | 2.14-2.15 s | 0.84-0.90 |
| yield, `stall-7-probes` | 0.42-0.50 s | 17.7-19.1 | 1.06-1.13 s | 0.33-0.50 s | 0.72-0.78 |

- **The stall is the chunk step.** While a long prompt prefills, every step carries a chunk and
  each decoding stream gets one step a chunk: about 2.15 s on the ring at 16,384.
- **Yield with nothing to yield to is level**: the same gaps and the same prefill rate under
  decode (+0.3%, ±0.5%) as candidate 10.
- **Yield with short requests arriving** puts their steps between chunks: about 18 gaps a
  stream instead of 10, mean gap 40% shorter (1.1 s against 1.8 s), the probes' TTFT 3.2 s
  against 10.8 s, and the long prompt's prefill under decode 9.5% slower.

The stall step now counts the gaps before the first chunk step as the admission
(`admission_s`) and leaves them out of the stall.
